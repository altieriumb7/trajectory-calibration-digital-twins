#!/usr/bin/env python3
"""Leakage-controlled recomputation of the PHMForge trajectory-calibration study.

This script is deliberately standalone -- it does not import
``phmforge_calibration`` -- so the V2 numbers are auditable without trusting the
V1 pipeline. It differs from V1 in six ways:

1. *Feature provenance is separated.* The V1 telemetry logger fabricates
   ``thought_logprobs`` from a hard-coded confidence constant
   (``telemetry_logger.py:60`` returns 0.75; ``:62`` synthesises logprobs from
   it). Every token-level feature is therefore a draw from a fixed Gaussian.
   They are retained here only as a negative control.
2. *Grader leakage is removed.* The harness writes a terminal observation whose
   text is literally "Task finished successfully" or "Task failed". Any feature
   derived from observation strings can read the label off it. Those
   observations are excluded.
3. *Cross-validation is grouped by scenario*, because all seven backbones run
   the same 25 PHMForge scenarios; ungrouped folds leak across backbones.
4. *Every estimate carries a bootstrap 95% CI* and a label-permutation p-value.
5. *Importance is out-of-fold permutation importance*, not in-sample
   mean-decrease-in-impurity (which is biased toward the high-cardinality
   continuous features -- exactly the synthetic ones).
6. *Cost is compared against the trivial policies* (always-execute,
   always-abstain) that bound any useful selective policy.

Usage:  python analysis/recompute_honest.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
_V2 = _HERE.parent
_ROOT = _V2.parent
_TELEMETRY = _ROOT / "results" / "telemetry_runs"
_FIGS = _V2 / "figures"
_TABS = _V2 / "tables"
_FIGS.mkdir(parents=True, exist_ok=True)
_TABS.mkdir(parents=True, exist_ok=True)

SEED = 0
N_BOOT = 2000
N_PERM = 2000
N_SPLITS = 5

# Cost model (USD), stated explicitly so a reader can vary it.
C_CATASTROPHIC = 50_000.0  # acting on a recommendation from a failed trajectory
C_INSPECTION = 1_000.0     # unnecessary inspection triggered by a false alarm
C_REVIEW = 500.0           # human engineer reviews an abstained trajectory

# The V1 telemetry logger's synthetic-logprob generator, for the provenance test.
V1_DEFAULT_CONFIDENCE = 0.75
GEN_MEAN = -0.1 - (1.0 - V1_DEFAULT_CONFIDENCE) * 1.2
GEN_STD = 0.15 + (1.0 - V1_DEFAULT_CONFIDENCE) * 0.3

GRADER_STRING = re.compile(r"task\s+(finished successfully|failed)", re.I)

DISPLAY = {
    "ibm_granite-4-h-small": "granite-4-h-small",
    "meta-llama_llama-3-3-70b-instruct": "llama-3.3-70b",
    "meta-llama_llama-4-maverick-17b-128e-instruct-fp8": "llama-4-maverick",
    "mistralai_mistral-medium-2505": "mistral-medium-2505",
    "mistralai_mistral-small-3-1-24b-instruct-2503": "mistral-small-24b",
    "openai_gpt-oss-120b": "gpt-oss-120b",
    "openai_gpt-4o-mini": "gpt-4o-mini",
}

# Feature families, grouped by provenance.
EXEC = [
    "n_steps",
    "n_tool_calls",
    "mcp_error_ratio",
    "mcp_error_count",
    "loop_ratio",
    "n_unique_actions",
    "reached_finish",
]
OBS_CLEAN = ["obs_mean_clean", "obs_max_clean", "n_obs_clean"]
OBS_LEAKY = ["obs_mean_all", "obs_max_all"]
SYNTH = ["early_entropy", "logprob_variance"]
VERB = ["verbalized_score"]
V1_VECTOR = SYNTH + VERB + ["mcp_error_ratio", "loop_ratio"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_trajectories() -> pd.DataFrame:
    """One row per trajectory, with leaky and leak-controlled feature variants."""
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in _TELEMETRY.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("*.jsonl")):
            steps = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not steps:
                continue

            actions = [s.get("action", "") for s in steps if s.get("action")]
            inputs = [s.get("action_input", "") for s in steps if s.get("action_input")]
            calls = [c for s in steps for c in s.get("mcp_tool_calls", [])]
            n_err = sum(1 for c in calls if c.get("status_code", 200) != 200)
            logprobs = [lp for s in steps for lp in s.get("thought_logprobs", [])]
            early = [lp for s in steps[:2] for lp in s.get("thought_logprobs", [])]

            obs_all, obs_clean = [], []
            for i, s in enumerate(steps):
                text = str(s.get("observation", ""))
                obs_all.append(len(text))
                # The terminal record and any grader verdict are written by the
                # harness after the outcome is known: not admissible evidence.
                if i == len(steps) - 1 or GRADER_STRING.search(text):
                    continue
                obs_clean.append(len(text))

            rows.append(
                {
                    "model": model_dir.name,
                    "scenario": path.stem,
                    # --- measured execution telemetry -------------------------
                    "n_steps": len(steps),
                    "n_tool_calls": len(calls),
                    "mcp_error_ratio": n_err / len(calls) if calls else 0.0,
                    "mcp_error_count": n_err,
                    "loop_ratio": (len(actions) - len(set(zip(actions, inputs))))
                    / len(actions)
                    if actions
                    else 0.0,
                    "n_unique_actions": len(set(actions)),
                    "reached_finish": int(any(a == "Finish" for a in actions)),
                    # --- observation features, both variants ------------------
                    "obs_mean_all": float(np.mean(obs_all)) if obs_all else 0.0,
                    "obs_max_all": float(np.max(obs_all)) if obs_all else 0.0,
                    "obs_mean_clean": float(np.mean(obs_clean)) if obs_clean else 0.0,
                    "obs_max_clean": float(np.max(obs_clean)) if obs_clean else 0.0,
                    "n_obs_clean": len(obs_clean),
                    # --- synthesised token features (negative control) --------
                    "early_entropy": -float(np.mean(early)) if early else 0.0,
                    "logprob_variance": float(np.var(logprobs)) if logprobs else 0.0,
                    "verbalized_score": float(steps[-1].get("verbalized_confidence", 0.5)),
                    # --- label ------------------------------------------------
                    "y": int(bool(steps[-1].get("task_success", False))),
                }
            )
    return pd.DataFrame(rows)


def all_logprobs() -> np.ndarray:
    """Flat array of every logprob in the corpus, for the provenance test."""
    out: list[float] = []
    for path in _TELEMETRY.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.extend(json.loads(line).get("thought_logprobs", []))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum():
            ece += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC, or NaN on a degenerate label vector.

    V1 returned 0.5 here, which silently converted 'undefined' into 'chance'
    and let a constant predictor be reported as a measured null result.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def boot_ci(y, p, fn, n: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    point = fn(y, p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = fn(y[idx], p[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_pvalue(y, p, n: int = N_PERM, seed: int = SEED) -> float:
    """One-sided permutation p-value for AUROC > 0.5."""
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


# --------------------------------------------------------------------------- #
# Out-of-fold evaluation
# --------------------------------------------------------------------------- #
def make_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        )
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)
    )


def oof_predict(df: pd.DataFrame, features: Sequence[str], kind: str = "rf") -> np.ndarray:
    X = df[list(features)].to_numpy(float)
    y = df["y"].to_numpy(int)
    groups = df["scenario"].to_numpy()
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        oof[te] = make_model(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
def eoc(y: np.ndarray, p: np.ndarray, theta: float) -> float:
    """Expected operational cost per scenario under abstention threshold theta."""
    execute = p >= theta
    cost = (
        C_CATASTROPHIC * np.sum(execute & (y == 0))
        + C_REVIEW * np.sum(~execute)
        + C_INSPECTION * np.sum(~execute & (y == 1))
    )
    return float(cost / len(y))


def eoc_always_execute(y) -> float:
    return float(C_CATASTROPHIC * np.sum(y == 0) / len(y))


def eoc_always_abstain(y) -> float:
    return float((C_REVIEW * len(y) + C_INSPECTION * np.sum(y == 1)) / len(y))


def risk_coverage(y: np.ndarray, p: np.ndarray):
    order = np.argsort(-p)
    ys = y[order]
    k = np.arange(1, len(ys) + 1)
    return k / len(ys), 1.0 - np.cumsum(ys) / k


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
CONFIGS = [
    ("Verbalized confidence (V1 `raw')", VERB, None),
    ("Synthetic token features", SYNTH, "rf"),
    (r"V1 feature vector $f(\tau)$", V1_VECTOR, "rf"),
    ("Execution telemetry", EXEC, "rf"),
    ("Execution + clean observations", EXEC + OBS_CLEAN, "rf"),
    ("Execution + grader-contaminated obs.", EXEC + OBS_LEAKY, "rf"),
]


def main() -> None:
    df = load_trajectories()
    y = df["y"].to_numpy(int)
    print("=" * 78)
    print(f"trajectories={len(df)}  backbones={df.model.nunique()}  "
          f"scenarios={df.scenario.nunique()}  Pass@1={y.mean():.3f}")
    print("=" * 78)

    # --- provenance audit ---------------------------------------------------
    lp = all_logprobs()
    vb = np.unique(df.verbalized_score)
    print("\n[provenance audit]")
    print(f"  verbalized_score : {len(vb)} distinct value(s) -> {list(vb)}")
    print(f"  logprobs         : n={len(lp):,}  observed  mean={lp.mean():+.4f} std={lp.std():.4f}")
    print(f"                     V1 generator predicts mean={GEN_MEAN:+.4f} std={GEN_STD:.4f}")
    n_grader = sum(
        1
        for p_ in _TELEMETRY.rglob("*.jsonl")
        for line in p_.read_text(encoding="utf-8").splitlines()
        if line.strip() and GRADER_STRING.search(str(json.loads(line).get("observation", "")))
    )
    print(f"  grader verdict strings in observations: {n_grader}")
    print(f"  runs with no admissible observation   : {(df.n_obs_clean == 0).sum()}/{len(df)}")

    # --- headline comparison ------------------------------------------------
    records, oof_store = [], {}
    print("\n[out-of-fold, scenario-grouped 5-fold CV]")
    for name, feats, kind in CONFIGS:
        p = df[feats[0]].to_numpy(float) if kind is None else oof_predict(df, feats, kind)
        auc, lo, hi = boot_ci(y, p, safe_auc)
        ece, elo, ehi = boot_ci(y, p, expected_calibration_error)
        pv = perm_pvalue(y, p)
        oof_store[name] = p
        records.append(dict(model=name, n_feat=len(feats), auroc=auc, auroc_lo=lo,
                            auroc_hi=hi, ece=ece, ece_lo=elo, ece_hi=ehi, p=pv))
        flag = "  <-- ARTIFACT" if "contaminated" in name else ""
        print(f"  {name:38s} AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]  "
              f"ECE {ece:.3f}  p={pv:.3f}{flag}")

    main_tbl = pd.DataFrame(records)
    main_tbl.to_csv(_TABS / "main_results.csv", index=False)

    # --- per-backbone, leak-controlled --------------------------------------
    best = oof_store["Execution telemetry"]
    per = []
    for m, sub in df.groupby("model"):
        idx = sub.index.to_numpy()
        auc, lo, hi = boot_ci(y[idx], best[idx], safe_auc)
        per.append(dict(model=DISPLAY.get(m, m), n=len(sub), success=y[idx].mean(),
                        auroc=auc, lo=lo, hi=hi,
                        ece=expected_calibration_error(y[idx], best[idx])))
    per_tbl = pd.DataFrame(per).sort_values("model")
    per_tbl.to_csv(_TABS / "per_backbone.csv", index=False)
    print("\n[per-backbone, leak-controlled execution telemetry]")
    print(per_tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- cost ---------------------------------------------------------------
    thetas = np.linspace(0.0, 1.0, 101)
    costs = np.array([eoc(y, best, t) for t in thetas])
    j = int(np.argmin(costs))
    ae, aa = eoc_always_execute(y), eoc_always_abstain(y)
    leaky_costs = np.array([eoc(y, oof_store["Execution + grader-contaminated obs."], t)
                            for t in thetas])
    print("\n[expected operational cost per scenario, out-of-fold]")
    print(f"  always execute                  : ${ae:>9,.0f}")
    print(f"  always abstain                  : ${aa:>9,.0f}")
    print(f"  selective, leak-controlled @{thetas[j]:.2f} : ${costs[j]:>9,.0f}  "
          f"({(1 - costs[j]/aa):+.1%} vs always-abstain)")
    print(f"  selective, contaminated  @{thetas[int(np.argmin(leaky_costs))]:.2f} : "
          f"${leaky_costs.min():>9,.0f}  <-- ARTIFACT")
    pd.DataFrame(dict(theta=thetas, eoc_clean=costs, eoc_leaky=leaky_costs)).to_csv(
        _TABS / "eoc_sweep.csv", index=False)

    # --- figures & tables ---------------------------------------------------
    _fig_auroc(main_tbl)
    _fig_provenance(lp, df)
    _fig_risk_coverage(y, oof_store)
    _fig_cost(thetas, costs, leaky_costs, ae, aa, thetas[j], costs[j])
    _fig_importance(df, y)
    _write_latex(main_tbl, per_tbl, y, best, thetas[j], costs[j], ae, aa, len(df), n_grader)
    print(f"\nartifacts -> {_FIGS}  and  {_TABS}")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_auroc(tbl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    o = tbl.sort_values("auroc")
    ypos = np.arange(len(o))
    colors = ["#c0392b" if "contaminated" in m else
              ("#27ae60" if lo > 0.5 else "#7f8c8d")
              for m, lo in zip(o.model, o.auroc_lo)]
    ax.barh(ypos, o.auroc, color=colors, alpha=0.85)
    ax.errorbar(o.auroc, ypos,
                xerr=[o.auroc - o.auroc_lo, o.auroc_hi - o.auroc],
                fmt="none", ecolor="#2c3e50", capsize=3, lw=1.2)
    ax.axvline(0.5, color="k", ls="--", lw=1.1, label="chance")
    ax.set_yticks(ypos)
    ax.set_yticklabels(o.model, fontsize=7.5)
    ax.set_xlabel("Out-of-fold AUROC (95% bootstrap CI)")
    ax.set_xlim(0.2, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Only the grader-contaminated model separates failures", fontsize=9.5)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(_FIGS / f"fig1_auroc_comparison.{e}", dpi=300)
    plt.close(fig)


def _fig_provenance(lp: np.ndarray, df: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    a1.hist(lp, bins=80, density=True, color="#5b8ff9", alpha=0.75,
            label=f"observed ({len(lp):,} tokens)")
    xs = np.linspace(lp.min(), lp.max(), 400)
    a1.plot(xs, np.exp(-((xs - GEN_MEAN) ** 2) / (2 * GEN_STD ** 2))
            / (GEN_STD * np.sqrt(2 * np.pi)), color="#c0392b", lw=2,
            label=fr"$\mathcal{{N}}({GEN_MEAN:.2f},{GEN_STD:.3f}^2)$ generator")
    a1.set_xlabel("token logprob")
    a1.set_ylabel("density")
    a1.legend(fontsize=6.5)
    a1.set_title("Logprobs match the synthetic generator", fontsize=9)

    vals, cnts = np.unique(df.verbalized_score, return_counts=True)
    a2.bar([str(v) for v in vals], cnts, color="#c0392b", alpha=0.8, width=0.5)
    a2.set_xlabel("verbalized confidence")
    a2.set_ylabel("trajectories")
    a2.set_title("Verbalized confidence is a constant", fontsize=9)
    for v, c in zip([str(v) for v in vals], cnts):
        a2.text(v, c, f" {c}", ha="center", va="bottom", fontsize=7.5)
    a2.set_ylim(0, cnts.max() * 1.18)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(_FIGS / f"fig2_provenance.{e}", dpi=300)
    plt.close(fig)


def _fig_risk_coverage(y: np.ndarray, store: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for key, color, ls, lab in [
        ("Execution + grader-contaminated obs.", "#c0392b", "--",
         "Execution + grader-contaminated obs. (artifact)"),
        ("Execution telemetry", "#1E88E5", "-", "Execution telemetry (leak-controlled)"),
        ("Synthetic token features", "#7f8c8d", ":", "Synthetic token features"),
    ]:
        cov, risk = risk_coverage(y, store[key])
        ax.plot(cov, risk, lw=1.9, color=color, ls=ls, label=lab)
    ax.axhline(1 - y.mean(), color="k", ls=":", lw=1.2, label="no abstention")
    ax.set_xlabel("Coverage (fraction of trajectories executed)")
    ax.set_ylabel("Selective failure risk")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7)
    ax.set_title("Out-of-fold risk--coverage profile", fontsize=9.5)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(_FIGS / f"fig3_risk_coverage.{e}", dpi=300)
    plt.close(fig)


def _fig_cost(thetas, clean, leaky, ae, aa, bt, bc) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(thetas, clean, lw=2, color="#1E88E5", label="Selective (leak-controlled)")
    ax.plot(thetas, leaky, lw=1.6, ls="--", color="#c0392b",
            label="Selective (grader-contaminated, artifact)")
    ax.axhline(ae, color="#8e44ad", ls="-.", lw=1.4, label=f"Always execute (\\${ae:,.0f})")
    ax.axhline(aa, color="#f39c12", ls=":", lw=1.8, label=f"Always abstain (\\${aa:,.0f})")
    ax.plot([bt], [bc], "o", color="#27ae60", ms=7)
    ax.set_yscale("log")
    ax.set_xlabel(r"Abstention threshold $\theta$")
    ax.set_ylabel("EOC per scenario (USD, log scale)")
    ax.grid(alpha=0.3, ls="--", which="both")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Always-abstain is the baseline a policy must beat", fontsize=9.5)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(_FIGS / f"fig4_eoc.{e}", dpi=300)
    plt.close(fig)


def _fig_importance(df: pd.DataFrame, y: np.ndarray) -> None:
    """Out-of-fold permutation importance across all provenance classes."""
    feats = EXEC + OBS_CLEAN + SYNTH + VERB
    X = df[feats].to_numpy(float)
    groups = df["scenario"].to_numpy()
    rng = np.random.default_rng(SEED)
    drops = np.zeros((N_SPLITS, len(feats)))
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for f, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = make_model("rf").fit(X[tr], y[tr])
        base = safe_auc(y[te], m.predict_proba(X[te])[:, 1])
        for j in range(len(feats)):
            vals = []
            for _ in range(20):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                v = safe_auc(y[te], m.predict_proba(Xp)[:, 1])
                if not np.isnan(v):
                    vals.append(base - v)
            drops[f, j] = float(np.mean(vals)) if vals else 0.0
    mean, se = drops.mean(0), drops.std(0) / np.sqrt(N_SPLITS)
    idx = np.argsort(mean)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    colors = ["#c0392b" if feats[i] in SYNTH + VERB else "#27ae60" for i in idx]
    ax.barh(np.arange(len(idx)), mean[idx], xerr=se[idx], color=colors, alpha=0.85,
            error_kw=dict(ecolor="#2c3e50", capsize=2.5, lw=0.9))
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels([feats[i].replace("_", " ") for i in idx], fontsize=7.5)
    ax.set_xlabel("Permutation importance (AUROC drop, mean $\\pm$ s.e., 5 folds)")
    ax.set_title("Measured (green) vs. synthetic/constant (red) features", fontsize=9.5)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(_FIGS / f"fig5_permutation_importance.{e}", dpi=300)
    plt.close(fig)
    pd.DataFrame(dict(feature=feats, importance=mean, se=se)).to_csv(
        _TABS / "permutation_importance.csv", index=False)


# --------------------------------------------------------------------------- #
# LaTeX tables
# --------------------------------------------------------------------------- #
def _write_latex(main, per, y, best, bt, bc, ae, aa, n, n_grader) -> None:
    with open(_TABS / "tab_main.tex", "w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[t]\n\\centering\n"
            f"\\caption{{Out-of-fold failure prediction on PHMForge ($n={n}$ "
            "trajectories, scenario-grouped 5-fold CV). Brackets are 95\\% "
            "bootstrap CIs; $p$ is a label-permutation test against "
            "AUROC${=}0.5$. Only the grader-contaminated row is significant, "
            "and it is an artifact.}\n"
            "\\label{tab:main}\n\\footnotesize\n"
            "\\begin{tabular}{lccc}\n\\toprule\n"
            "\\textbf{Signal set} & \\textbf{AUROC [95\\% CI]} & \\textbf{ECE} "
            "& $p$ \\\\\n\\midrule\n")
        for _, r in main.iterrows():
            row = (f"{r.model} & {r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}] "
                   f"& {r.ece:.3f} & {r.p:.3f} \\\\\n")
            if "contaminated" in r.model:
                row = "\\midrule\n" + row.replace("&", "&", 1)
            f.write(row)
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(_TABS / "tab_per_backbone.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Per-backbone "
                "out-of-fold AUROC of the leak-controlled execution-telemetry "
                "model. Every interval covers chance.}\n"
                "\\label{tab:perbackbone}\n\\footnotesize\n"
                "\\begin{tabular}{lccc}\n\\toprule\n\\textbf{Backbone} & $n$ "
                "& \\textbf{Pass@1} & \\textbf{AUROC [95\\% CI]} \\\\\n\\midrule\n")
        for _, r in per.iterrows():
            f.write(f"\\texttt{{{r.model}}} & {int(r.n)} & {r.success:.2f} "
                    f"& {r.auroc:.3f} [{r.lo:.3f}, {r.hi:.3f}] \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(_TABS / "tab_eoc.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Expected operational "
                "cost per scenario. Always-abstain is the baseline any selective "
                "policy must beat; V1 reported \\$2{,}100 against an "
                "always-execute baseline only.}\n\\label{tab:eoc}\n\\footnotesize\n"
                "\\begin{tabular}{lcc}\n\\toprule\n\\textbf{Policy} & "
                "\\textbf{EOC/scenario} & \\textbf{Ratio} \\\\\n\\midrule\n")
        f.write(f"Always execute & \\${ae:,.0f} & {ae/aa:.2f}$\\times$ \\\\\n")
        f.write(f"Selective, $\\theta{{=}}0.75$ & \\${eoc(y,best,0.75):,.0f} & "
                f"{eoc(y,best,0.75)/aa:.2f}$\\times$ \\\\\n")
        f.write(f"Selective, best $\\theta{{=}}{bt:.2f}$ & \\${bc:,.0f} & "
                f"{bc/aa:.2f}$\\times$ \\\\\n")
        f.write(f"\\textbf{{Always abstain}} & \\textbf{{\\${aa:,.0f}}} & "
                "$1.00\\times$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


if __name__ == "__main__":
    main()
