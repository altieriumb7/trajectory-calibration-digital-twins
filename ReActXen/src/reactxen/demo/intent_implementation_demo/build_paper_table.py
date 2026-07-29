"""Build the LaTeX table for Section 4 from completed Pass@1 + Pass-all-3 runs.

Reads all JSON files in results/paper_table4_runs/ and emits:
1. The framework_performance LaTeX table (copy/paste into paper)
2. A summary CSV for cross-verification (the user's "Excel" cross-check)
3. A process-metrics block (avg steps, avg tokens) for Section 3.4

Usage:
    .venv/bin/python build_paper_table.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_DEMO = Path(__file__).parent
_RESULTS_DIR = _DEMO / "results" / "paper_table4_runs"

# Display names for paper (kept short)
DISPLAY_NAMES = {
    "ibm/granite-4-h-small": "Granite 4-H Small",
    "meta-llama/llama-3-3-70b-instruct": "Llama 3.3 70B",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": "Llama 4 Maverick",
    "mistralai/mistral-medium-2505": "Mistral Medium 2505",
    "mistralai/mistral-small-3-1-24b-instruct-2503": "Mistral Small 3.1 24B",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
}

FRAMEWORK_DISPLAY = {"react": "ReAct", "reactxen": "ReActXen"}

CATEGORIES = [
    "RUL Prediction",
    "Fault Classification",
    "Engine Health Analysis",
    "Cost-Benefit Analysis",
    "Safety/Policy Evaluation",
]
CAT_TOTAL = {
    "RUL Prediction": 15,
    "Fault Classification": 15,
    "Engine Health Analysis": 30,
    "Cost-Benefit Analysis": 5,
    "Safety/Policy Evaluation": 10,
}

# Claude Code rows are evaluated manually on the FULL n=75 benchmark
# (the user runs Claude Code interactively per scenario; not API-callable).
# Pulled from results/paper_results.json. We surface them as a separate
# "Claude Code (manual, n=75)" group in the table.
CLAUDE_CODE_MANUAL = [
    {
        "framework": "Claude Code",
        "model_display": "Sonnet 4.5",
        "n_scenarios": 75,
        "overall_pass1": 0.65,
        "by_category_pass1": {
            "RUL Prediction": 0.73,
            "Fault Classification": 0.67,
            "Engine Health Analysis": 0.57,
            "Cost-Benefit Analysis": 0.80,
            "Safety/Policy Evaluation": 0.70,
        },
        "evaluation_mode": "manual",
    },
    {
        "framework": "Claude Code",
        "model_display": "Opus 4.6",
        "n_scenarios": 75,
        "overall_pass1": 0.81,
        "by_category_pass1": {
            "RUL Prediction": 0.87,
            "Fault Classification": 0.80,
            "Engine Health Analysis": 0.70,
            "Cost-Benefit Analysis": 1.00,
            "Safety/Policy Evaluation": 0.90,
        },
        "evaluation_mode": "manual",
    },
]


def load_runs() -> list[dict]:
    """Load all per-config JSONs (excludes pass-all-3 reruns)."""
    runs = []
    for f in sorted(_RESULTS_DIR.glob("*.json")):
        # Skip rerun files (_run2.json, _run3.json) — these are aggregated
        # into the run-1 JSON's summary by run_pass3.aggregate_pass3()
        if "_run2.json" in f.name or "_run3.json" in f.name:
            continue
        try:
            with open(f) as fp:
                d = json.load(fp)
            if "metadata" not in d or "summary" not in d:
                continue
            runs.append(d)
        except Exception as e:
            print(f"Skip {f.name}: {e}", file=sys.stderr)
    return runs


def fmt_pct(x: float | None, dash: str = "---") -> str:
    if x is None:
        return dash
    return f"{x*100:.1f}\\%"


def by_category(summary: dict) -> dict[str, float | None]:
    """Extract per-category pass1 from a summary."""
    out = {c: None for c in CATEGORIES}
    bc = summary.get("by_category", {})
    for c in CATEGORIES:
        if c in bc:
            out[c] = bc[c]["pass1"]
    return out


def build_latex_table(runs: list[dict], pass3_winner: str | None = None,
                      include_claude_code: bool = True) -> str:
    """Emit the LaTeX framework_performance table."""
    rows = []
    # Sort: framework asc, then overall pass1 desc
    runs.sort(key=lambda r: (
        r["metadata"].get("framework", ""),
        -float(r["summary"].get("overall_pass1") or 0),
    ))

    # Group by framework for midrules
    cur_framework = None
    for r in runs:
        meta = r["metadata"]
        summ = r["summary"]
        framework = meta["framework"]
        model_name = meta["model_name"]
        display = DISPLAY_NAMES.get(model_name, model_name)
        fw_display = FRAMEWORK_DISPLAY.get(framework, framework)

        if cur_framework is not None and framework != cur_framework:
            rows.append("\\midrule")
        cur_framework = framework

        overall = summ.get("overall_pass1")
        cats = by_category(summ)
        cat_strs = [fmt_pct(cats[c]) for c in CATEGORIES]

        # Pass-all-3 only for winner
        cfg_key = f"{framework}__{model_name}"
        if pass3_winner == cfg_key and "overall_pass_all_3" in summ:
            pass3_str = fmt_pct(summ["overall_pass_all_3"])
        else:
            pass3_str = "---"

        # Bold the winner row
        is_winner = (cfg_key == pass3_winner)
        prefix = "\\textbf{" if is_winner else ""
        suffix = "}" if is_winner else ""

        line = (f"{prefix}{fw_display} + {display}{suffix} & "
                f"{prefix}{fmt_pct(overall)}{suffix} & {pass3_str} & "
                + " & ".join(cat_strs) + " \\\\")
        if is_winner:
            line = "\\rowcolor{highlight}\n" + line
        rows.append(line)

    # Claude Code (manual, n=75) — separate section at the bottom
    if include_claude_code:
        rows.append("\\midrule")
        cc_sorted = sorted(CLAUDE_CODE_MANUAL,
                           key=lambda r: -r["overall_pass1"])
        for entry in cc_sorted:
            cat_strs = [fmt_pct(entry["by_category_pass1"].get(c)) for c in CATEGORIES]
            line = (f"{entry['framework']} + {entry['model_display']}$^{{*}}$ & "
                    f"{fmt_pct(entry['overall_pass1'])} & --- & "
                    + " & ".join(cat_strs) + " \\\\")
            rows.append(line)

    body = "\n".join(rows)
    # n_total per category may be a stratified subset — annotate this
    n_actual = max((r["summary"].get("total_scenarios", 0) for r in runs), default=0)

    table = (
        "\\begin{table*}[t]\n"
        "\\caption{Framework-and-model Pass@1 across PHM task categories on the "
        f"PHMForge benchmark. \\textbf{{Pass@1}}: mean success rate. ReAct and "
        f"ReActXen rows are evaluated automatically on a stratified subset of "
        f"n={n_actual} scenarios (5 RUL + 5 Fault + 10 Health + 2 Cost + 3 "
        f"Safety) preserving all 5 task categories. Claude Code rows ($^{{*}}$) "
        f"are evaluated \\emph{{manually}} on the full n=75 benchmark; Claude "
        f"Code is an interactive CLI tool and is not API-callable from the "
        f"automated harness. \\textbf{{Pass-all-3}}: fraction of scenarios "
        f"solved on \\emph{{all}} three runs, reported only for the strongest "
        f"automated configuration. ``---'' indicates Pass-all-3 not measured "
        f"for non-winning configurations to bound compute cost.}}\n"
        "\\label{tab:framework_performance}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lccccccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Framework + Model} & \\multicolumn{2}{c}{\\textbf{Overall}} & "
        "\\textbf{RUL} & \\textbf{Fault} & \\textbf{Health} & \\textbf{Cost} & "
        "\\textbf{Safety} \\\\\n"
        "\\cmidrule(lr){2-3}\n"
        "& Pass@1 & Pass-all-3 & "
        f"({CAT_TOTAL['RUL Prediction']}) & ({CAT_TOTAL['Fault Classification']}) & "
        f"({CAT_TOTAL['Engine Health Analysis']}) & ({CAT_TOTAL['Cost-Benefit Analysis']}) & "
        f"({CAT_TOTAL['Safety/Policy Evaluation']}) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}"
    )
    return table


def build_process_metrics_block(runs: list[dict]) -> str:
    """Emit a small per-config table with avg steps, tokens, time."""
    lines = ["\\begin{table}[t]",
             "\\caption{Process metrics for evaluated configurations: average "
             "steps taken, total tokens used, and execution time per scenario.}",
             "\\label{tab:process_metrics}",
             "\\centering\\small",
             "\\begin{tabular}{lcccc}",
             "\\toprule",
             "\\textbf{Framework + Model} & \\textbf{Avg Steps} & "
             "\\textbf{Avg Tokens} & \\textbf{Avg Time (s)} & \\textbf{Pass@1} \\\\",
             "\\midrule"]
    runs.sort(key=lambda r: (r["metadata"]["framework"],
                             -float(r["summary"].get("overall_pass1") or 0)))
    for r in runs:
        meta = r["metadata"]
        summ = r["summary"]
        display = DISPLAY_NAMES.get(meta["model_name"], meta["model_name"])
        fw = FRAMEWORK_DISPLAY.get(meta["framework"], meta["framework"])
        lines.append(
            f"{fw} + {display} & "
            f"{summ.get('avg_steps', 0):.1f} & "
            f"{summ.get('avg_total_tokens', 0):,} & "
            f"{summ.get('avg_time_s', 0):.1f} & "
            f"{fmt_pct(summ.get('overall_pass1'))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def write_csv(runs: list[dict], path: Path) -> None:
    """Write a flat CSV the user can paste into Excel for cross-verification."""
    fieldnames = [
        "framework", "model", "n_scenarios",
        "overall_pass1",
        "RUL", "Fault", "Health", "Cost", "Safety",
        "avg_steps", "avg_total_tokens", "avg_time_s",
    ]
    rows = []
    for r in runs:
        meta = r["metadata"]
        summ = r["summary"]
        cats = by_category(summ)
        rows.append({
            "framework": meta["framework"],
            "model": meta["model_name"],
            "n_scenarios": summ.get("total_scenarios", 0),
            "overall_pass1": round(summ.get("overall_pass1", 0) or 0, 4),
            "RUL": round(cats["RUL Prediction"] or 0, 4) if cats["RUL Prediction"] is not None else "",
            "Fault": round(cats["Fault Classification"] or 0, 4) if cats["Fault Classification"] is not None else "",
            "Health": round(cats["Engine Health Analysis"] or 0, 4) if cats["Engine Health Analysis"] is not None else "",
            "Cost": round(cats["Cost-Benefit Analysis"] or 0, 4) if cats["Cost-Benefit Analysis"] is not None else "",
            "Safety": round(cats["Safety/Policy Evaluation"] or 0, 4) if cats["Safety/Policy Evaluation"] is not None else "",
            "avg_steps": summ.get("avg_steps", 0),
            "avg_total_tokens": summ.get("avg_total_tokens", 0),
            "avg_time_s": summ.get("avg_time_s", 0),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    runs = load_runs()
    if not runs:
        print("No runs found in", _RESULTS_DIR)
        return

    print(f"Loaded {len(runs)} run files.\n")
    # Identify winner: highest overall_pass1
    best = max(runs, key=lambda r: float(r["summary"].get("overall_pass1") or 0))
    winner_key = f"{best['metadata']['framework']}__{best['metadata']['model_name']}"
    print(f"Winning config (by Pass@1): {winner_key}  "
          f"({best['summary']['overall_pass1']*100:.1f}%)\n")

    # Generate outputs
    out_dir = _RESULTS_DIR
    table = build_latex_table(runs, pass3_winner=winner_key)
    proc = build_process_metrics_block(runs)

    (out_dir / "table4_paper.tex").write_text(table)
    (out_dir / "process_metrics_table.tex").write_text(proc)
    write_csv(runs, out_dir / "results_summary.csv")

    print("=" * 70)
    print("MAIN RESULTS TABLE (table4_paper.tex)")
    print("=" * 70)
    print(table)
    print("\n" + "=" * 70)
    print("PROCESS METRICS (process_metrics_table.tex)")
    print("=" * 70)
    print(proc)
    print()
    print(f"All artifacts written to {out_dir}")
    print(f"  - table4_paper.tex")
    print(f"  - process_metrics_table.tex")
    print(f"  - results_summary.csv (Excel-friendly)")


if __name__ == "__main__":
    main()
