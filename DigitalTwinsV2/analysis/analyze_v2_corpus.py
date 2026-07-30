#!/usr/bin/env python3
"""Analyse a provenance-tagged V2 corpus.

Differs from ``recompute_honest.py``, which audits the original corpus: this
reads the new schema (per-step ``provenance``, quarantined
``harness_observation``, ``task_success`` on the terminal step only) and
evaluates only the feature families the gate certifies as measured.

Protocol, unchanged from the audit so numbers are comparable:
scenario-grouped stratified 5-fold CV, percentile bootstrap 95% CIs, one-sided
label-permutation p-values, and a Bonferroni threshold over the comparisons made.

Usage:  python analysis/analyze_v2_corpus.py results/v2_corpus
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_V2))

from pipeline.feature_extractor import FAMILIES, FeatureExtractor  # noqa: E402
from pipeline.provenance import validate  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

SEED = 0
N_BOOT = 2000
N_PERM = 2000

# PHMForge scenario families split by whether the tool layer computes the outcome
# or fabricates it. `predict_rul` returns ground truth plus fixed-seed Gaussian
# noise tuned to pass the grader's accuracy window, and `classify_faults` invents
# both its predictions and its "ground truth" at ~75% agreement (AUDIT.md D11).
# On those families a success measures tool orchestration, not analysis. The
# other three compute deterministically from what the agent supplies, so there
# the label does reflect the agent's analytical work.
#
# Keeping the strata apart turns the defect into a contrast: if token uncertainty
# predicts failure where the analysis is genuine but not where it is fabricated,
# the signal tracks task difficulty; if it predicts equally on both, the signal is
# about orchestration. Either answer is worth reporting.
FABRICATED_PREFIXES = ("pdm_rul", "pdm_fault")
STRATUM_LABEL = {True: "fabricated outcome", False: "computed outcome"}


def stratum_of(scenario: str) -> str:
    return STRATUM_LABEL[scenario.startswith(FABRICATED_PREFIXES)]


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum():
            out += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(out)


def boot_ci(y, p, fn, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = fn(y[idx], p[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return fn(y, p), float("nan"), float("nan")
    return fn(y, p), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_p(y, p, n=N_PERM, seed=SEED) -> float:
    obs = safe_auc(y, p)
    if np.isnan(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(n):
        v = safe_auc(rng.permutation(y), p)
        if not np.isnan(v) and v >= obs:
            ge += 1
    return ge / (n + 1)


def oof(df: pd.DataFrame, feats: list[str], kind: str = "rf") -> np.ndarray:
    X = df[feats].to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["y"].to_numpy(int)
    groups = df["scenario"].to_numpy()
    out = np.full(len(y), np.nan)
    n_splits = min(5, len(np.unique(groups)), int(min(np.bincount(y))))
    if n_splits < 2:
        return np.full(len(y), y.mean())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            out[te] = y[tr].mean()
            continue
        model = (
            RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                   random_state=SEED, n_jobs=-1)
            if kind == "rf"
            else make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=5000, random_state=SEED))
        )
        out[te] = model.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


SHORT_MODEL = {
    "openai_meta-llama_Llama-3.3-70B-Instruct-Turbo": "llama-3.3-70b",
    "openai_Qwen_Qwen2.5-7B-Instruct-Turbo": "qwen2.5-7b",
    "openai_openai_gpt-oss-120b": "gpt-oss-120b",
}


def _write_latex(tabs: Path, df: pd.DataFrame, strat_rows: list,
                 per_rows: list, alpha_s: float, alpha_b: float) -> None:
    """Emits the two result tables the paper \\input{}s.

    Generated rather than transcribed so the paper cannot drift from the
    analysis: rerunning on more data updates the manuscript.
    """
    def fmt(r: dict) -> str:
        star = r"$^{\ast}$" if r.get("significant") else ""
        return (f"{r['auroc']:.3f}{star} [{r['lo']:.3f}, {r['hi']:.3f}]"
                f" & {r['p']:.4f}")

    with open(tabs / "tab_v2_stratified.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Out-of-fold failure "
                "prediction on the re-collected corpus, split by whether the "
                "benchmark's tool layer computes the outcome or fabricates it "
                "(Sec.~\\ref{sec:label}). $^{\\ast}$ marks significance at the "
                f"Bonferroni threshold $\\alpha={alpha_s:.5f}$.}}\n"
                "\\label{tab:v2strat}\n\\footnotesize\n"
                "\\begin{tabular}{llcc}\n\\toprule\n\\textbf{Stratum} & "
                "\\textbf{Signals} & \\textbf{AUROC [95\\% CI]} & $p$ \\\\\n"
                "\\midrule\n")
        for i, r in enumerate(strat_rows):
            if i and strat_rows[i - 1]["stratum"] != r["stratum"]:
                f.write("\\midrule\n")
            lab = r["stratum"] if (i == 0 or strat_rows[i - 1]["stratum"]
                                   != r["stratum"]) else ""
            f.write(f"{lab} & {r['family']} & {fmt(r)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(tabs / "tab_v2_per_backbone.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Per backbone, on the "
                "computed-outcome stratum only. Holding the model fixed removes "
                "the identity confound by construction. $^{\\ast}$ marks "
                f"$\\alpha={alpha_b:.5f}$.}}\n\\label{{tab:v2backbone}}\n"
                "\\footnotesize\n\\begin{tabular}{llcc}\n\\toprule\n"
                "\\textbf{Backbone} & \\textbf{Signals} & "
                "\\textbf{AUROC [95\\% CI]} & $p$ \\\\\n\\midrule\n")
        for i, r in enumerate(per_rows):
            if i and per_rows[i - 1]["model"] != r["model"]:
                f.write("\\midrule\n")
            lab = (f"\\texttt{{{SHORT_MODEL.get(r['model'], r['model'][:18])}}}"
                   if (i == 0 or per_rows[i - 1]["model"] != r["model"]) else "")
            f.write(f"{lab} & {r['family']} & {fmt(r)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\n  LaTeX tables -> {tabs}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    args = ap.parse_args()

    print("=" * 74)
    report = validate(args.corpus)
    print(report.render())
    if report.corpus_unusable:
        print("\nnothing measured survives; not analysing")
        return 1
    usable, unusable = report.usable_families()

    df = FeatureExtractor().extract_dataset(args.corpus)
    if df.empty:
        print("no trajectories")
        return 1

    # Post-hoc self-reported confidence, if elicit_verbalized.py has been run.
    # It is kept as its own family so it can be compared against token
    # uncertainty directly -- the comparison the literature cares about and the
    # one this corpus otherwise cannot make, since in-band elicitation perturbs
    # the agent badly enough to destroy the labels.
    vpath = _V2 / "tables" / "verbalized_posthoc.csv"
    if vpath.exists():
        vdf = pd.read_csv(vpath)
        before = len(df)
        df = df.merge(vdf, on=["model", "scenario"], how="left")
        have = df["verbalized_posthoc"].notna().sum()
        nuniq = df["verbalized_posthoc"].nunique(dropna=True)
        print(f"\n  post-hoc verbalized confidence merged on {have}/{before} "
              f"trajectories, {nuniq} distinct values")
        if nuniq >= 3 and have >= 20:
            FAMILIES["verbalized_posthoc"] = ("verbalized_posthoc",)
        else:
            print("  too few distinct values to use as a baseline; not analysed")
            df = df.drop(columns=["verbalized_posthoc"])
    y = df["y"].to_numpy(int)
    print("\n" + "=" * 74)
    print(f"corpus: {len(df)} trajectories, {df.model.nunique()} backbone(s), "
          f"{df.scenario.nunique()} scenarios, Pass@1={y.mean():.3f}")
    for fam in FAMILIES:
        gate = f"usable_{fam}"
        if gate in df.columns:
            print(f"  {fam:12s} usable on {int(df[gate].sum())}/{len(df)} trajectories")

    if len(np.unique(y)) < 2:
        print("\nlabels are single-class; no discrimination measurable")
        return 1

    # Evaluate each certified family, then their union.
    configs: list[tuple[str, list[str], str]] = []
    for fam, feats in FAMILIES.items():
        if fam in unusable:
            continue
        gate = f"usable_{fam}"
        if gate in df.columns and not df[gate].any():
            continue
        present = [f for f in feats if f in df.columns and df[f].notna().any()]
        if present:
            configs.append((fam, present, "rf"))

    all_feats = sorted({f for _, fs, _ in configs for f in fs})
    if len(configs) > 1:
        configs.append(("all measured", all_feats, "rf"))

    alpha = 0.05 / max(len(configs), 1)
    rows = []
    print("\n" + "=" * 74)
    print(f"out-of-fold results (Bonferroni alpha = 0.05/{len(configs)} = {alpha:.4f})")
    print("=" * 74)
    for name, feats, kind in configs:
        p = oof(df, feats, kind)
        auc, lo, hi = boot_ci(y, p, safe_auc)
        e, _, _ = boot_ci(y, p, ece)
        pv = perm_p(y, p)
        sig = "SIG" if pv < alpha else "   "
        rows.append(dict(family=name, n_feat=len(feats), auroc=auc, lo=lo, hi=hi,
                         ece=e, p=pv, significant=pv < alpha))
        print(f"  [{sig}] {name:16s} ({len(feats):2d} feat) "
              f"AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]  ECE {e:.3f}  p={pv:.4f}")

    out = pd.DataFrame(rows)
    tabs = _V2 / "tables"
    tabs.mkdir(exist_ok=True)
    out.to_csv(tabs / "v2_corpus_results.csv", index=False)
    df.to_csv(tabs / "v2_corpus_features.csv", index=False)

    # --- backbone confound -------------------------------------------------
    # With several models pooled, a classifier can gain AUROC without predicting
    # failure at all: if the features reveal which model produced a trajectory,
    # and the models differ in success rate, recognising the model is already
    # informative about the outcome. Both conditions have to hold, so we measure
    # both and let the reader judge.
    if df.model.nunique() > 1:
        print("\n" + "=" * 74)
        print("backbone confound check")
        print("=" * 74)
        rates = df.groupby("model")["y"].agg(["size", "mean"])
        for m, row in rates.iterrows():
            print(f"  {m[:44]:46s} n={int(row['size']):3d} Pass@1={row['mean']:.3f}")
        spread = float(rates["mean"].max() - rates["mean"].min())

        feats_all = sorted({f for _, fs, _ in configs for f in fs})
        X = np.nan_to_num(df[feats_all].to_numpy(float), nan=0.0,
                          posinf=0.0, neginf=0.0)
        codes = pd.Categorical(df.model).codes
        groups = df.scenario.to_numpy()
        ident = np.full(len(codes), np.nan)
        n_sp = min(5, int(np.min(np.bincount(codes))), df.scenario.nunique())
        if n_sp >= 2:
            cv = StratifiedGroupKFold(n_splits=n_sp, shuffle=True, random_state=SEED)
            for tr, te in cv.split(X, codes, groups):
                m = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                           random_state=SEED, n_jobs=-1).fit(X[tr], codes[tr])
                proba = m.predict_proba(X[te])
                ident[te] = proba[np.arange(len(te)), np.searchsorted(m.classes_, codes[te])]
            identifiable = float(np.nanmean(ident > 1.0 / df.model.nunique()))
            print(f"  features identify the backbone in {identifiable:.0%} of rows "
                  f"(chance {1/df.model.nunique():.0%})")
        print(f"  Pass@1 spread across backbones: {spread:.3f}")
        if spread > 0.15:
            print("  WARNING: backbones differ enough in success rate that model "
                  "identity carries outcome information; read the per-backbone "
                  "results below, where this confound cannot arise.")
        else:
            print("  spread is small: recognising the backbone says little about "
                  "the outcome, so pooled results are not inflated by it.")

    # --- stratified by whether the tool layer computes or fabricates the outcome
    df = df.assign(stratum=[stratum_of(s) for s in df.scenario])
    strat_rows = []
    # The stratified section makes its own comparisons -- one per family per
    # stratum -- so it needs its own correction. Reusing the main table's
    # threshold here would understate the multiplicity and let a borderline
    # contrast read as significant.
    n_strata = df["stratum"].nunique() if "stratum" in df else 2
    alpha_s = 0.05 / max(len(configs) * n_strata, 1)
    print("\n" + "=" * 74)
    print(f"stratified by tool-layer provenance (see AUDIT.md D11) -- "
          f"{len(configs)} families x {n_strata} strata, Bonferroni "
          f"alpha = {alpha_s:.5f}")
    print("=" * 74)
    for stratum, sub in df.groupby("stratum"):
        ys = sub["y"].to_numpy(int)
        print(f"\n  {stratum}: n={len(sub)}  Pass@1={ys.mean():.3f}  "
              f"scenarios={sub.scenario.nunique()}")
        if len(np.unique(ys)) < 2:
            print("    labels single-class in this stratum; not evaluable")
            continue
        if len(sub) < 20:
            print(f"    n={len(sub)} too small to evaluate; reporting Pass@1 only")
            continue
        for name, feats, kind in configs:
            present = [f for f in feats if f in sub.columns and sub[f].notna().any()]
            if not present:
                continue
            p = oof(sub.reset_index(drop=True), present, kind)
            auc, lo, hi = boot_ci(ys, p, safe_auc)
            pv = perm_p(ys, p)
            sig = "SIG" if pv < alpha_s else "   "
            strat_rows.append(dict(stratum=stratum, family=name, n=len(sub),
                                   auroc=auc, lo=lo, hi=hi, p=pv,
                                   significant=pv < alpha_s))
            print(f"    [{sig}] {name:16s} AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]  "
                  f"p={pv:.4f}")
    if strat_rows:
        pd.DataFrame(strat_rows).to_csv(tabs / "v2_corpus_stratified.csv", index=False)

    # --- per backbone, on the stratum where the outcome is genuinely computed
    # Holding the model fixed removes the confound above by construction: within
    # one backbone there is no model identity to recognise. These are the
    # uncontaminated estimates; the pooled numbers buy power at the cost of
    # mixing in a between-model effect.
    if df.model.nunique() > 1:
        comp = df[df.stratum == STRATUM_LABEL[False]]
        per_rows = []
        alpha_b = 0.05 / max(len(configs) * comp.model.nunique(), 1)
        print("\n" + "=" * 74)
        print(f"per backbone, computed-outcome stratum only -- Bonferroni "
              f"alpha = {alpha_b:.5f}")
        print("=" * 74)
        for model, sub in comp.groupby("model"):
            ys = sub["y"].to_numpy(int)
            print(f"\n  {model[:50]}  n={len(sub)}  Pass@1={ys.mean():.3f}")
            if len(np.unique(ys)) < 2 or len(sub) < 20:
                print("    not evaluable at this n")
                continue
            for name, feats, kind in configs:
                present = [f for f in feats
                           if f in sub.columns and sub[f].notna().any()]
                if not present:
                    continue
                p = oof(sub.reset_index(drop=True), present, kind)
                auc, lo, hi = boot_ci(ys, p, safe_auc)
                pv = perm_p(ys, p)
                per_rows.append(dict(model=model, family=name, n=len(sub),
                                     auroc=auc, lo=lo, hi=hi, p=pv,
                                     significant=pv < alpha_b))
                print(f"    [{'SIG' if pv < alpha_b else '   '}] {name:16s} "
                      f"AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]  p={pv:.4f}")
        if per_rows:
            pd.DataFrame(per_rows).to_csv(
                tabs / "v2_corpus_per_backbone.csv", index=False)

            _write_latex(tabs, df, strat_rows, per_rows, alpha_s, alpha_b)

    # Which individual features carry the signal, in the cells where there is
        # one. Reported as out-of-fold permutation importance rather than
        # impurity, which is biased toward high-cardinality continuous columns --
        # exactly the ones under test here.
        best = [r for r in per_rows if r["significant"] and r["family"] != "all measured"]
        for cell in {(r["model"], r["family"]) for r in best}:
            model, family = cell
            sub = comp[comp.model == model].reset_index(drop=True)
            feats = [f for f in FAMILIES[family]
                     if f in sub.columns and sub[f].notna().any()]
            if len(feats) < 2 or len(sub) < 20:
                continue
            X = np.nan_to_num(sub[feats].to_numpy(float), nan=0.0,
                              posinf=0.0, neginf=0.0)
            ys = sub["y"].to_numpy(int)
            groups = sub.scenario.to_numpy()
            rng = np.random.default_rng(SEED)
            n_sp = min(5, len(np.unique(groups)), int(min(np.bincount(ys))))
            if n_sp < 2:
                continue
            drops = np.zeros((n_sp, len(feats)))
            cv = StratifiedGroupKFold(n_splits=n_sp, shuffle=True, random_state=SEED)
            for k, (tr, te) in enumerate(cv.split(X, ys, groups)):
                mdl = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                             random_state=SEED, n_jobs=-1).fit(X[tr], ys[tr])
                base = safe_auc(ys[te], mdl.predict_proba(X[te])[:, 1])
                for j in range(len(feats)):
                    vals = []
                    for _ in range(20):
                        Xp = X[te].copy()
                        Xp[:, j] = rng.permutation(Xp[:, j])
                        v = safe_auc(ys[te], mdl.predict_proba(Xp)[:, 1])
                        if not np.isnan(v):
                            vals.append(base - v)
                    drops[k, j] = float(np.mean(vals)) if vals else 0.0
            mean, se = drops.mean(0), drops.std(0) / np.sqrt(n_sp)
            print(f"\n  feature attribution -- {model[:38]} / {family}")
            for j in np.argsort(-mean):
                print(f"    {feats[j]:20s} {mean[j]:+.4f} +/- {se[j]:.4f}")

    # Signal-quality evidence: real logprobs are strongly left-skewed.
    if "token" in usable and df["logprob_variance"].notna().any():
        print("\n" + "=" * 74)
        print("token-feature provenance check")
        for col in ("mean_logprob", "logprob_variance", "early_entropy"):
            if col in df.columns:
                v = df[col].dropna()
                print(f"  {col:18s} n={len(v):3d} mean={v.mean():+.4f} std={v.std():.4f}")

    print(f"\nwritten to {tabs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
