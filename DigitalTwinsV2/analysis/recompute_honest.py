#!/usr/bin/env python3
"""Honest recomputation of the PHMForge trajectory-calibration results.

This script deliberately does NOT import ``phmforge_calibration`` so that the V2
results are auditable standalone. It differs from the V1 pipeline in five ways:

1. It separates *measured* execution telemetry from *synthesised* token-level
   features. The V1 telemetry logger fabricates ``thought_logprobs`` from a
   hard-coded confidence constant (``telemetry_logger.py:60,62``), so any
   feature derived from them is noise. Here they are kept only as a control.
2. Cross-validation is grouped by scenario id, because the same PHMForge
   scenario is executed by all seven backbones; ungrouped folds leak.
3. Every point estimate carries a bootstrap 95% CI, and AUROC is accompanied by
   a label-permutation p-value.
4. Feature importance is permutation importance computed out-of-fold, not
   in-sample mean-decrease-in-impurity (which is biased toward high-cardinality
   continuous features -- exactly the synthetic ones).
5. The Expected Operational Cost is computed from out-of-fold predictions and
   compared against the two trivial policies (always-execute, always-abstain),
   which bound any useful selective policy.

Usage:  python analysis/recompute_honest.py
"""
from __future__ import annotations

import json
import glob
import os
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

# Cost model (US dollars), stated explicitly so the reader can vary it.
C_CATASTROPHIC = 50_000.0  # acting on a recommendation from a failed trajectory
C_INSPECTION = 1_000.0     # unnecessary inspection triggered by a false alarm
C_REVIEW = 500.0           # human engineer reviews an abstained trajectory

DISPLAY = {
    "ibm_granite-4-h-small": "granite-4-h-small",
    "meta-llama_llama-3-3-70b-instruct": "llama-3.3-70b",
    "meta-llama_llama-4-maverick-17b-128e-instruct-fp8": "llama-4-maverick",
    "mistralai_mistral-medium-2505": "mistral-medium-2505",
    "mistralai_mistral-small-3-1-24b-instruct-2503": "mistral-small-24b",
    "openai_gpt-oss-120b": "gpt-oss-120b",
    "openai_gpt-4o-mini": "gpt-4o-mini",
}

# Feature families. "measured" comes from real ReActXen/MCP execution records;
# "synthetic" is reconstructed from the fabricated logprob stream and is
# reported only to show it carries no signal.
MEASURED = [
    "n_steps",
    "n_tool_calls",
    "mcp_error_ratio",
    "mcp_error_count",
    "loop_ratio",
    "n_unique_actions",
    "reached_finish",
    "mean_obs_len",
]
SYNTHETIC = ["early_entropy", "logprob_variance"]
VERBALIZED = ["verbalized_score"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_trajectories() -> pd.DataFrame:
    """Builds one row per trajectory from the raw JSONL telemetry."""
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in _TELEMETRY.iterdir() if p.is_dir()):
        model = model_dir.name
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
            early = [
                lp
                for s in steps[: min(2, len(steps))]
                for lp in s.get("thought_logprobs", [])
            ]
            obs_lens = [len(str(s.get("observation", ""))) for s in steps]

            rows.append(
                {
                    "model": model,
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
                    "mean_obs_len": float(np.mean(obs_lens)) if obs_lens else 0.0,
                    # --- synthesised token features (control) -----------------
                    "early_entropy": -float(np.mean(early)) if early else 0.0,
                    "logprob_variance": float(np.var(logprobs)) if logprobs else 0.0,
                    "verbalized_score": float(steps[-1].get("verbalized_confidence", 0.5)),
                    # --- label ------------------------------------------------
                    "y": int(bool(steps[-1].get("task_success", False))),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width binned ECE."""
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
    """AUROC, or NaN when the label vector is degenerate.

    V1 returned 0.5 here, which silently turned 'undefined' into 'chance' and
    made a constant predictor look like a measured null result.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def boot_ci(
    y: np.ndarray, p: np.ndarray, fn, n: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """Point estimate plus percentile bootstrap 95% CI."""
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


def perm_pvalue(y: np.ndarray, p: np.ndarray, n: int = N_PERM, seed: int = SEED) -> float:
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


def oof_predict(
    df: pd.DataFrame, features: Sequence[str], kind: str = "rf", n_splits: int = 5
) -> np.ndarray:
    """Scenario-grouped, stratified out-of-fold probability predictions."""
    X = df[list(features)].to_numpy(float)
    y = df["y"].to_numpy(int)
    groups = df["scenario"].to_numpy()
    oof = np.full(len(y), np.nan)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        m = make_model(kind)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
def eoc(y: np.ndarray, p: np.ndarray, theta: float) -> float:
    """Expected operational cost per scenario under an abstention threshold.

    execute (p >= theta) and the trajectory failed  -> C_catastrophic
    abstain (p <  theta)                            -> C_review
    abstain on a trajectory that would have succeeded -> + C_inspection
    """
    execute = p >= theta
    cost = (
        C_CATASTROPHIC * np.sum(execute & (y == 0))
        + C_REVIEW * np.sum(~execute)
        + C_INSPECTION * np.sum(~execute & (y == 1))
    )
    return float(cost / len(y))


def eoc_always_execute(y: np.ndarray) -> float:
    return float(C_CATASTROPHIC * np.sum(y == 0) / len(y))


def eoc_always_abstain(y: np.ndarray) -> float:
    return float((C_REVIEW * len(y) + C_INSPECTION * np.sum(y == 1)) / len(y))


def risk_coverage(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk as a function of coverage, sweeping the threshold."""
    order = np.argsort(-p)
    ys = y[order]
    k = np.arange(1, len(ys) + 1)
    cov = k / len(ys)
    risk = 1.0 - np.cumsum(ys) / k
    return cov, risk


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_trajectories()
    y = df["y"].to_numpy(int)
    print("=" * 78)
    print(f"trajectories={len(df)}  backbones={df.model.nunique()}  "
          f"scenarios={df.scenario.nunique()}  success_rate={y.mean():.3f}")
    print("=" * 78)

    # --- provenance audit ---------------------------------------------------
    vb = df.verbalized_score.to_numpy()
    print("\n[provenance audit]")
    print(f"  verbalized_score : {len(np.unique(vb))} distinct value(s) "
          f"-> {sorted(np.unique(vb))}")
    print(f"  logprob_variance : mean={df.logprob_variance.mean():.4f} "
          f"(generator predicts {(0.15 + 0.25 * 0.3) ** 2:.4f} at confidence=0.75)")
    print(f"  early_entropy    : mean={df.early_entropy.mean():.4f} "
          f"(generator predicts {0.1 + 0.25 * 1.2:.4f})")
    deg = df[df.n_tool_calls == 0]
    print(f"  degenerate runs (0 MCP calls): {len(deg)}/{len(df)} "
          f"({len(deg)/len(df):.1%}), success rate {deg.y.mean():.3f} "
          f"vs {df[df.n_tool_calls>0].y.mean():.3f} for the rest")

    # --- headline comparison ------------------------------------------------
    configs = [
        ("Verbalized confidence (raw)", VERBALIZED, None),
        ("Synthetic token features", SYNTHETIC, "rf"),
        ("V1 feature vector f(tau)", SYNTHETIC + VERBALIZED + ["mcp_error_ratio", "loop_ratio"], "rf"),
        ("MCP error ratio only", ["mcp_error_ratio"], "lr"),
        ("Measured execution telemetry", MEASURED, "lr"),
        ("Measured execution telemetry", MEASURED, "rf"),
    ]

    records = []
    oof_store: dict[str, np.ndarray] = {}
    for name, feats, kind in configs:
        if kind is None:
            p = df[feats[0]].to_numpy(float)
            label = name
        else:
            p = oof_predict(df, feats, kind)
            label = f"{name} [{kind.upper()}]"
        auc, lo, hi = boot_ci(y, p, safe_auc)
        ece, elo, ehi = boot_ci(y, p, expected_calibration_error)
        pv = perm_pvalue(y, p)
        oof_store[label] = p
        records.append(
            dict(model=label, n_feat=len(feats), auroc=auc, auroc_lo=lo,
                 auroc_hi=hi, ece=ece, ece_lo=elo, ece_hi=ehi, p=pv)
        )
        print(f"\n  {label}")
        print(f"    AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]  perm p={pv:.4f}")
        print(f"    ECE   {ece:.3f} [{elo:.3f}, {ehi:.3f}]")

    main_tbl = pd.DataFrame(records)
    main_tbl.to_csv(_TABS / "main_results.csv", index=False)

    # --- per-backbone -------------------------------------------------------
    best = oof_predict(df, MEASURED, "rf")
    per_model = []
    for m, sub in df.groupby("model"):
        idx = sub.index.to_numpy()
        yy, pp = y[idx], best[idx]
        auc, lo, hi = boot_ci(yy, pp, safe_auc)
        per_model.append(
            dict(model=DISPLAY.get(m, m), n=len(sub), success=yy.mean(),
                 auroc=auc, lo=lo, hi=hi, ece=expected_calibration_error(yy, pp))
        )
    per_tbl = pd.DataFrame(per_model).sort_values("model")
    per_tbl.to_csv(_TABS / "per_backbone.csv", index=False)
    print("\n[per-backbone, measured telemetry RF]")
    print(per_tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- cost analysis ------------------------------------------------------
    thetas = np.linspace(0.0, 1.0, 101)
    costs = np.array([eoc(y, best, t) for t in thetas])
    j = int(np.argmin(costs))
    ae, aa = eoc_always_execute(y), eoc_always_abstain(y)
    print("\n[expected operational cost, per scenario, out-of-fold]")
    print(f"  always execute                : ${ae:,.0f}")
    print(f"  always abstain                : ${aa:,.0f}")
    print(f"  selective @ theta={thetas[j]:.2f}      : ${costs[j]:,.0f}")
    print(f"  reduction vs always-execute   : {(1 - costs[j]/ae):.1%}")
    print(f"  reduction vs always-abstain   : {(1 - costs[j]/aa):+.1%}")
    print(f"  V1-reported threshold 0.75    : ${eoc(y, best, 0.75):,.0f}")
    pd.DataFrame(dict(theta=thetas, eoc=costs)).to_csv(
        _TABS / "eoc_sweep.csv", index=False
    )

    # --- figures ------------------------------------------------------------
    _fig_auroc(main_tbl)
    _fig_risk_coverage(y, best, oof_store)
    _fig_importance(df, y)
    _fig_cost(thetas, costs, ae, aa, thetas[j], costs[j])
    _write_latex_tables(main_tbl, per_tbl, y, best, thetas[j], costs[j], ae, aa)
    print(f"\nartifacts written to {_FIGS} and {_TABS}")


def _fig_auroc(tbl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    order = tbl.sort_values("auroc")
    ypos = np.arange(len(order))
    colors = ["#c0392b" if a < 0.6 else "#27ae60" for a in order.auroc]
    ax.barh(ypos, order.auroc, color=colors, alpha=0.85)
    ax.errorbar(
        order.auroc, ypos,
        xerr=[order.auroc - order.auroc_lo, order.auroc_hi - order.auroc],
        fmt="none", ecolor="#2c3e50", capsize=3, lw=1.2,
    )
    ax.axvline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_yticks(ypos)
    ax.set_yticklabels(order.model, fontsize=8)
    ax.set_xlabel("Out-of-fold AUROC (95% bootstrap CI)")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Failure prediction: measured telemetry vs. synthetic features", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_FIGS / f"fig1_auroc_comparison.{ext}", dpi=300)
    plt.close(fig)


def _fig_risk_coverage(y: np.ndarray, best: np.ndarray, store: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cov, risk = risk_coverage(y, best)
    ax.plot(cov, risk, lw=2, color="#1E88E5", label="Measured execution telemetry [RF]")
    syn_key = [k for k in store if k.startswith("Synthetic")][0]
    cov_s, risk_s = risk_coverage(y, store[syn_key])
    ax.plot(cov_s, risk_s, lw=1.6, ls="--", color="#c0392b", label="Synthetic token features [RF]")
    ax.axhline(1 - y.mean(), color="k", ls=":", lw=1.2, label="no abstention (full coverage)")
    ax.set_xlabel("Coverage (fraction of trajectories executed)")
    ax.set_ylabel("Selective failure risk")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8)
    ax.set_title("Out-of-fold risk--coverage profile", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_FIGS / f"fig2_risk_coverage.{ext}", dpi=300)
    plt.close(fig)


def _fig_importance(df: pd.DataFrame, y: np.ndarray) -> None:
    """Out-of-fold permutation importance over measured + synthetic features."""
    feats = MEASURED + SYNTHETIC + VERBALIZED
    X = df[feats].to_numpy(float)
    groups = df["scenario"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    drops = np.zeros((5, len(feats)))
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
            drops[f, j] = np.mean(vals) if vals else 0.0
    mean, se = drops.mean(0), drops.std(0) / np.sqrt(5)
    idx = np.argsort(mean)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = ["#c0392b" if feats[i] in SYNTHETIC + VERBALIZED else "#27ae60" for i in idx]
    ax.barh(np.arange(len(idx)), mean[idx], xerr=se[idx], color=colors,
            alpha=0.85, error_kw=dict(ecolor="#2c3e50", capsize=3, lw=1))
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels([feats[i].replace("_", " ") for i in idx], fontsize=8)
    ax.set_xlabel("Permutation importance (AUROC drop, mean $\\pm$ s.e. over 5 folds)")
    ax.set_title("Measured (green) vs. synthetic/constant (red) features", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_FIGS / f"fig3_permutation_importance.{ext}", dpi=300)
    plt.close(fig)
    pd.DataFrame(dict(feature=feats, importance=mean, se=se)).to_csv(
        _TABS / "permutation_importance.csv", index=False
    )


def _fig_cost(thetas, costs, ae, aa, best_t, best_c) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(thetas, costs, lw=2, color="#1E88E5", label="Selective abstention (OOF)")
    ax.axhline(ae, color="#c0392b", ls="--", lw=1.5, label=f"Always execute (${ae:,.0f})")
    ax.axhline(aa, color="#f39c12", ls="-.", lw=1.5, label=f"Always abstain (${aa:,.0f})")
    ax.plot([best_t], [best_c], "o", color="#27ae60", ms=8,
            label=f"Best $\\theta$={best_t:.2f} (${best_c:,.0f})")
    ax.set_xlabel(r"Abstention threshold $\theta$")
    ax.set_ylabel("Expected operational cost per scenario (USD)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8)
    ax.set_title("Cost of selective abstention against trivial policies", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_FIGS / f"fig4_eoc.{ext}", dpi=300)
    plt.close(fig)


def _write_latex_tables(main, per, y, best, best_t, best_c, ae, aa) -> None:
    with open(_TABS / "tab_main.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Out-of-fold failure "
                "prediction on PHMForge ($n=167$ trajectories, scenario-grouped "
                "5-fold CV). Brackets are 95\\% bootstrap CIs; $p$ is a "
                "label-permutation test against AUROC${=}0.5$.}\n"
                "\\label{tab:main}\n\\footnotesize\n"
                "\\begin{tabular}{lccc}\n\\toprule\n"
                "\\textbf{Signal set} & \\textbf{AUROC} & \\textbf{ECE} & $\\bm{p}$ \\\\\n\\midrule\n")
        for _, r in main.iterrows():
            f.write(f"{r.model} & {r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}] "
                    f"& {r.ece:.3f} & {r.p:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(_TABS / "tab_per_backbone.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Per-backbone "
                "out-of-fold AUROC of the measured-telemetry model. CIs are wide "
                "because each backbone contributes at most 25 trajectories.}\n"
                "\\label{tab:perbackbone}\n\\footnotesize\n"
                "\\begin{tabular}{lccc}\n\\toprule\n\\textbf{Backbone} & $\\bm{n}$ "
                "& \\textbf{Pass@1} & \\textbf{AUROC [95\\% CI]} \\\\\n\\midrule\n")
        for _, r in per.iterrows():
            f.write(f"\\texttt{{{r.model}}} & {int(r.n)} & {r.success:.2f} "
                    f"& {r.auroc:.3f} [{r.lo:.3f}, {r.hi:.3f}] \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(_TABS / "tab_eoc.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Expected operational "
                "cost per scenario. The trivial always-abstain policy is the "
                "baseline any selective policy must beat.}\n\\label{tab:eoc}\n"
                "\\footnotesize\n\\begin{tabular}{lcc}\n\\toprule\n"
                "\\textbf{Policy} & \\textbf{EOC/scenario} & \\textbf{vs.\\ best} \\\\\n\\midrule\n")
        f.write(f"Always execute & \\${ae:,.0f} & {ae/best_c:.2f}$\\times$ \\\\\n")
        f.write(f"Always abstain & \\${aa:,.0f} & {aa/best_c:.2f}$\\times$ \\\\\n")
        f.write(f"Selective, $\\theta{{=}}0.75$ & \\${eoc(y,best,0.75):,.0f} & "
                f"{eoc(y,best,0.75)/best_c:.2f}$\\times$ \\\\\n")
        f.write(f"\\textbf{{Selective, $\\theta{{=}}{best_t:.2f}$}} & "
                f"\\textbf{{\\${best_c:,.0f}}} & --- \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


if __name__ == "__main__":
    main()
