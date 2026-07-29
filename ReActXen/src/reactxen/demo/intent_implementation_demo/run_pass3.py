"""Pass-all-3 runner for the winning configuration.

Runs the SAME 25 scenarios 2 more times (rerun_2 and rerun_3) under the
same deterministic settings as Pass@1, then computes pass-all-3 = fraction
of scenarios solved correctly on ALL three runs.

Usage:
    .venv/bin/python run_pass3.py \\
        --framework react --model "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Re-use everything from benchmark_pass1
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_pass1 import (
    MODEL_NAME_TO_ID, FRAMEWORK_CONFIGS,
    run_one_scenario, load_scenarios, save_results, compute_summary,
)

_DEMO = Path(__file__).parent
_RESULTS_DIR = _DEMO / "results" / "paper_table4_runs"


def run_pass3(framework: str, model_id: int, output_path: Path,
              run_number: int, scenario_ids: list[str]) -> dict:
    """Run Pass@1 for run_number (2 or 3). Resumable."""
    scenario_file = _DEMO / "scenarios" / "phm_scenarios.json"
    all_scenarios = load_scenarios(scenario_file)
    sel = [s for s in all_scenarios if s["task_id"] in scenario_ids]

    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            prior = json.load(f)
        for r in prior.get("results", []):
            existing[r["task_id"]] = r
        print(f"Resume run{run_number}: {len(existing)} already done")

    results = list(existing.values())
    todo = [s for s in sel if s["task_id"] not in existing]

    for i, sc in enumerate(todo):
        print(f"  [run{run_number}][{i+1}/{len(todo)}] {sc['task_id']}")
        rec = run_one_scenario(sc, framework, model_id)
        rec["run_number"] = run_number
        results.append(rec)
        save_results(results, output_path, framework, model_id)

    return compute_summary(results)


def aggregate_pass3(framework: str, model_name: str) -> dict:
    """After all 3 runs are done, aggregate into pass-all-3 metric."""
    safe_name = model_name.replace("/", "_").replace(":", "_")
    paths = [
        _RESULTS_DIR / f"{framework}__{safe_name}.json",  # run 1
        _RESULTS_DIR / f"{framework}__{safe_name}_run2.json",
        _RESULTS_DIR / f"{framework}__{safe_name}_run3.json",
    ]

    runs: list[dict[str, dict]] = []  # task_id -> record
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        runs.append({r["task_id"]: r for r in d["results"]})

    common = set(runs[0]) & set(runs[1]) & set(runs[2])
    print(f"Common scenarios across all 3 runs: {len(common)}")

    pass_all_3_count = 0
    pass_all_3_by_cat: dict[str, dict] = {}

    for tid in common:
        cat = runs[0][tid]["classification_type"]
        pass_all_3_by_cat.setdefault(cat, {"all3": 0, "total": 0})
        pass_all_3_by_cat[cat]["total"] += 1
        if all(runs[r][tid].get("correct", False) for r in range(3)):
            pass_all_3_count += 1
            pass_all_3_by_cat[cat]["all3"] += 1

    n = len(common)
    summary = {
        "total_scenarios": n,
        "pass_all_3": round(pass_all_3_count / n, 4) if n else 0,
        "pass_all_3_count": pass_all_3_count,
        "by_category": {
            c: {
                "pass_all_3": round(v["all3"] / v["total"], 4),
                "all3": v["all3"], "total": v["total"]
            }
            for c, v in pass_all_3_by_cat.items()
        },
        "per_run_pass1": [
            round(sum(1 for r in runs[i].values() if r.get("correct")) / len(runs[i]), 4)
            for i in range(3)
        ],
    }

    # Persist into the run-1 JSON's summary so build_paper_table picks it up
    p1_path = paths[0]
    with open(p1_path) as f:
        d = json.load(f)
    d["summary"]["overall_pass_all_3"] = summary["pass_all_3"]
    d["summary"]["pass_all_3_by_category"] = summary["by_category"]
    d["summary"]["per_run_pass1"] = summary["per_run_pass1"]
    with open(p1_path, "w") as f:
        json.dump(d, f, indent=2)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["react", "reactxen"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--run", type=int, choices=[2, 3], required=True,
                   help="Which extra run (2 or 3)")
    p.add_argument("--scenario_ids", required=True, help="Comma-separated task IDs")
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip running, just aggregate existing run JSONs")
    args = p.parse_args()

    if args.model not in MODEL_NAME_TO_ID:
        raise SystemExit(f"Unknown model: {args.model}")
    model_id = MODEL_NAME_TO_ID[args.model]
    safe_name = args.model.replace("/", "_").replace(":", "_")

    if args.aggregate_only:
        summary = aggregate_pass3(args.framework, args.model)
        print(json.dumps(summary, indent=2))
        return

    output_path = _RESULTS_DIR / f"{args.framework}__{safe_name}_run{args.run}.json"
    sids = args.scenario_ids.split(",")

    summary = run_pass3(args.framework, model_id, output_path, args.run, sids)
    print(f"\nRun {args.run} pass1 = {summary.get('overall_pass1')}")


if __name__ == "__main__":
    main()
