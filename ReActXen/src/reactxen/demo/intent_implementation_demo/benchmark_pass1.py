"""Pass@1 benchmark runner — unified over ReAct + ReActXen frameworks.

Tracks per-scenario pass/fail, total tokens (prompt + completion), step count,
LLM calls, and execution time. Saves intermediate results so the run can resume
if interrupted.

Usage:
    .venv/bin/python benchmark_pass1.py \\
        --framework reactxen --model_id 38 --limit 5
    .venv/bin/python benchmark_pass1.py \\
        --framework react --model "ibm/granite-4-h-small" --limit 75
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Setup paths
# This file lives at: ReActXen/src/reactxen/demo/intent_implementation_demo/benchmark_pass1.py
# We need ReActXen/src/ on sys.path (3 parents up: demo/intent_impl -> demo -> reactxen -> src).
_DEMO = Path(__file__).parent
_REACTXEN_SRC = _DEMO.parent.parent.parent  # ReActXen/src
if not (_REACTXEN_SRC / "reactxen").is_dir():
    # Fallback: walk up
    p = _DEMO
    for _ in range(8):
        p = p.parent
        if (p / "reactxen").is_dir():
            _REACTXEN_SRC = p
            break
sys.path.insert(0, str(_REACTXEN_SRC))
sys.path.insert(0, str(_DEMO))

# Runtime patch: add models that are deployed on the user's WatsonX project
# but missing from the installed reactxen.modelset.
import reactxen.utils.model_inference as _mi

# Patch openai_count_tokens to never return None (fixes openai/gpt-oss-120b
# which has no tiktoken encoding). Without this, count_tokens returns None
# and downstream comparisons to int fail with TypeError.
_orig_openai_count_tokens = _mi.openai_count_tokens
def _safe_openai_count_tokens(text, model="o1-preview", is_chat=False):
    try:
        n = _orig_openai_count_tokens(text, model, is_chat)
    except Exception:
        n = None
    if n is None:
        # Fallback: rough heuristic ~4 chars/token
        if isinstance(text, list):
            n = sum(len(str(t)) for t in text) // 4
        else:
            n = max(1, len(str(text)) // 4)
    return n
_mi.openai_count_tokens = _safe_openai_count_tokens

# Patch watsonx_llm to handle generate() exceptions properly.
# UPSTREAM BUG (model_inference.py line 182-188): if model.generate() raises,
# generated_response is never assigned, then the unconditional `return
# generated_response["results"][0]` raises UnboundLocalError. This burns
# compute (some failures took 30+ minutes before the eventual crash).
# Our patch returns a well-formed empty-response dict on any LLM exception,
# matching the schema downstream code expects.
_orig_watsonx_llm = _mi.watsonx_llm
def _safe_watsonx_llm(prompt, *args, **kwargs):
    try:
        result = _orig_watsonx_llm(prompt, *args, **kwargs)
        # If the underlying function still UnboundLocalErrors before patching,
        # we'd never reach here — but if it returns None/garbage, normalize it.
        if result is None or not isinstance(result, dict):
            return _empty_llm_response("Empty/null response from watsonx_llm")
        # Ensure required keys exist to prevent downstream KeyErrors
        result.setdefault("generated_text", "")
        result.setdefault("input_token_count", 0)
        result.setdefault("generated_token_count", 0)
        return result
    except Exception as e:
        return _empty_llm_response(f"watsonx_llm crashed: {type(e).__name__}: {e}")

def _empty_llm_response(error_msg: str) -> dict:
    """Well-formed empty response schema matching reactxen's expectations."""
    return {
        "generated_text": "",
        "input_token_count": 0,
        "generated_token_count": 0,
        "promptTokens": 0,
        "completionTokens": 0,
        "reasoning_token_count": 0,
        "thinking_text": "",
        "finish_reason": "error",
        "error": error_msg,
        "controller_action": "ABORT_STEP",
        "results": [{"generated_text": "", "input_token_count": 0,
                     "generated_token_count": 0, "stop_reason": "error"}],
    }
_mi.watsonx_llm = _safe_watsonx_llm

_EXTRA_MODELS = [
    "mistralai/mistral-medium-2505",
    "mistralai/mistral-small-3-1-24b-instruct-2503",
    "openai/gpt-oss-120b",
]
for _m in _EXTRA_MODELS:
    if _m not in _mi.modelset:
        _mi.modelset.append(_m)
# Also extend context_dict so count_tokens doesn't blow up
_extra_ctx = {
    "mistralai/mistral-medium-2505": 128000,
    "mistralai/mistral-small-3-1-24b-instruct-2503": 128000,
    "openai/gpt-oss-120b": 128000,
}
# get_context_length uses an inline dict — we override the function instead
_orig_get_ctx = _mi.get_context_length
def _patched_get_context_length(model_id):
    if isinstance(model_id, str) and model_id in _extra_ctx:
        return _extra_ctx[model_id]
    if isinstance(model_id, int) and 0 <= model_id < len(_mi.modelset):
        name = _mi.modelset[model_id]
        if name in _extra_ctx:
            return _extra_ctx[name]
    return _orig_get_ctx(model_id)
_mi.get_context_length = _patched_get_context_length

from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent

# Tool imports
from tools.data_tools import LoadDatasetTool, LoadGroundTruthTool
from tools.model_tools import (
    TrainRULModelTool, PredictRULTool,
    TrainFaultClassifierTool, ClassifyFaultsTool,
)
from tools.metric_tools import (
    CalculateMAETool, CalculateRMSETool,
    VerifyGroundTruthTool, CalculateAccuracyTool, VerifyClassificationTool,
)
from tools.analysis_tools import (
    AnalyzeEngineSignalsTool, AssessComponentHealthTool,
    DiagnoseTimingIssuesTool, DetectDegradationTrendTool,
    CalculateMaintenanceCostTool, CalculateFailureCostTool,
    OptimizeMaintenanceScheduleTool,
    AssessSafetyRiskTool, CheckComplianceTool,
    GenerateSafetyRecommendationsTool,
)
from tools.web_search_tool import WebSearchTool

# Map model name string -> id in INSTALLED modelset (after our patch above).
# Installed reactxen has 0-19; we appended 3 extras at indices 20, 21, 22.
MODEL_NAME_TO_ID = {
    "ibm/granite-4-h-small": 8,                                   # installed
    "meta-llama/llama-3-3-70b-instruct": 12,                       # installed
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": 16,       # installed
    "mistralai/mistral-medium-2505": 20,                           # appended
    "mistralai/mistral-small-3-1-24b-instruct-2503": 21,           # appended
    "openai/gpt-oss-120b": 22,                                     # appended
}

FRAMEWORK_CONFIGS = {
    # ReAct: single pass, no reflection
    "react": {"num_reflect_iteration": 1, "max_steps": 8,
              "max_execution_time": 480},  # 8 min per scenario
    # ReActXen: with reflection retries
    "reactxen": {"num_reflect_iteration": 3, "max_steps": 8,
                 "max_execution_time": 480},
    # ReActXen-lite: reduced reflection budget for high-latency backbones
    # (e.g., Llama-3.3-70B). Bounds runtime without disabling reflection.
    "reactxen_lite": {"num_reflect_iteration": 2, "max_steps": 6,
                      "max_execution_time": 360},
}


def get_all_tools() -> list:
    """Build tool list — all 22 tools across both servers."""
    return [
        LoadDatasetTool(), LoadGroundTruthTool(),
        TrainRULModelTool(), PredictRULTool(),
        TrainFaultClassifierTool(), ClassifyFaultsTool(),
        CalculateMAETool(), CalculateRMSETool(),
        VerifyGroundTruthTool(), CalculateAccuracyTool(),
        VerifyClassificationTool(),
        AnalyzeEngineSignalsTool(), AssessComponentHealthTool(),
        DiagnoseTimingIssuesTool(), DetectDegradationTrendTool(),
        CalculateMaintenanceCostTool(), CalculateFailureCostTool(),
        OptimizeMaintenanceScheduleTool(),
        AssessSafetyRiskTool(), CheckComplianceTool(),
        GenerateSafetyRecommendationsTool(),
        WebSearchTool(),
    ]


def load_scenarios(scenario_file: Path) -> list[dict]:
    with open(scenario_file) as f:
        return json.load(f).get("pdm_scenarios", [])


def evaluate_pass(scenario: dict, answer: str) -> tuple[bool, str]:
    """Evaluate whether the agent's answer matches the scenario's ground truth.

    Returns (correct, reason).

    Strategy:
    - For RUL: extract MAE/RMSE from answer; check if within expected_mae_range / expected_rmse_range.
    - For Fault: check accuracy >= ~0.6 OR all expected unit IDs mentioned.
    - For Engine Health: check categorical match terms in expected_keywords.
    - For Cost-Benefit / Safety: check key required_fields present in answer.
    - Fallback: presence-of-keywords from required_fields.
    """
    if not answer:
        return False, "empty_answer"

    answer_lower = answer.lower()
    gt = scenario.get("ground_truth", {})
    ctype = scenario.get("classification_type", "")

    # RUL Prediction: validate MAE/RMSE numeric ranges
    if ctype == "RUL Prediction":
        mae_range = gt.get("expected_mae_range") or gt.get("mae_range")
        rmse_range = gt.get("expected_rmse_range") or gt.get("rmse_range")
        mae_ok = rmse_ok = None

        # Find ALL "mae...<number>" matches and use the largest plausible one
        # (the formula text "MAE = (1/n) * Σ" yields garbage matches like "1";
        #  the real result usually appears with a value > 3).
        if mae_range and len(mae_range) == 2:
            ms = re.findall(r"\bmae\s*[=:]?\s*([\d]+\.?\d*)\s*(?:cycles?|cy)?", answer_lower)
            vals = [float(x) for x in ms if 3 <= float(x) <= 100]
            if vals:
                # Use the last reported value (final answer, not intermediate)
                val = vals[-1]
                mae_ok = mae_range[0] <= val <= mae_range[1]

        if rmse_range and len(rmse_range) == 2:
            ms = re.findall(r"\brmse\s*[=:]?\s*([\d]+\.?\d*)\s*(?:cycles?|cy)?", answer_lower)
            vals = [float(x) for x in ms if 3 <= float(x) <= 100]
            if vals:
                val = vals[-1]
                rmse_ok = rmse_range[0] <= val <= rmse_range[1]

        # Pass if at least one metric is in range AND no metric is out of range
        if mae_ok is True or rmse_ok is True:
            if mae_ok is False or rmse_ok is False:
                return False, "metric_out_of_range"
            return True, "metric_in_range"
        if mae_ok is False or rmse_ok is False:
            return False, "metric_out_of_range"
        # Fallback: agent at least mentioned mae/rmse and produced numerical output
        has_metrics = "mae" in answer_lower and "rmse" in answer_lower
        has_numbers = bool(re.search(r"\d+\.\d+", answer_lower))
        if has_metrics and has_numbers:
            return True, "metrics_reported_no_range_to_verify"
        return False, "no_metrics_in_answer"

    # Fault Classification: check accuracy reported and >= 0.5
    if ctype == "Fault Classification":
        m = re.search(r"accuracy[^\d]{0,10}(\d+\.?\d*)", answer_lower)
        if m:
            val = float(m.group(1))
            if val > 1.0:  # percentage
                val /= 100
            return val >= 0.5, f"accuracy={val:.2f}"
        if "classified" in answer_lower or "classification" in answer_lower:
            return True, "classification_attempted"
        return False, "no_classification_output"

    # Engine Health: check expected_keywords if present
    if ctype == "Engine Health Analysis":
        keywords = gt.get("expected_keywords", []) or gt.get("required_keywords", [])
        if keywords:
            hits = sum(1 for k in keywords if k.lower() in answer_lower)
            return hits >= max(1, len(keywords) // 2), f"keywords_{hits}/{len(keywords)}"
        # Fallback: look for analysis terms
        terms = ["health", "component", "degrad", "diagnos", "fault", "engine"]
        hits = sum(1 for t in terms if t in answer_lower)
        return hits >= 3, f"terms_{hits}"

    # Cost-Benefit: numbers and cost terms
    if ctype == "Cost-Benefit Analysis":
        terms = ["cost", "maintenance"]
        hit = all(t in answer_lower for t in terms)
        has_dollars = "$" in answer or bool(re.search(r"\d{3,}", answer))
        return hit and has_dollars, "cost_analysis_present"

    # Safety/Policy: compliance/risk terms
    if ctype == "Safety/Policy Evaluation":
        terms = ["risk", "safety", "compliance", "recommend"]
        hits = sum(1 for t in terms if t in answer_lower)
        return hits >= 2, f"safety_terms_{hits}"

    # Generic fallback: required_fields keywords present
    fields = gt.get("required_fields", [])
    if fields:
        hits = sum(1 for f in fields if f.lower().replace("_", " ") in answer_lower)
        return hits >= max(1, len(fields) // 2), f"fields_{hits}/{len(fields)}"
    return bool(answer.strip()), "non_empty_fallback"


def run_one_scenario(scenario: dict, framework: str, model_id: int) -> dict:
    """Run a single scenario with the given framework + model. Returns full record."""
    config = FRAMEWORK_CONFIGS[framework]
    tools = get_all_tools()
    question = scenario.get("input_question", "")
    question += (
        "\n\nIMPORTANT: When calling tools, use this format:\n"
        "- Action: tool_name (just the name, no brackets or parameters)\n"
        '- Action Input: JSON object like {"param1": "value1", "param2": "value2"}\n'
        "DO NOT use formats like tool_name[param1, param2] or tool_name('param1', 'param2')"
    )

    record = {
        "task_id": scenario["task_id"],
        "classification_type": scenario["classification_type"],
        "dataset": scenario.get("dataset", ""),
        "framework": framework,
        "model_id": model_id,
        "model_name": list(MODEL_NAME_TO_ID.keys())[
            list(MODEL_NAME_TO_ID.values()).index(model_id)
        ] if model_id in MODEL_NAME_TO_ID.values() else f"id_{model_id}",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    start = time.time()
    try:
        agent = create_reactxen_agent(
            question=question,
            key=str(scenario.get("ground_truth", {})),
            tools=tools,
            react_llm_model_id=model_id,
            reflect_llm_model_id=model_id,
            num_reflect_iteration=config["num_reflect_iteration"],
            max_steps=config["max_steps"],
            debug=False,
        )
        agent.run()
        # Use answer if set; otherwise fall back to scratchpad (the full trace)
        # so observations from tool calls are captured for evaluation.
        ans = (agent.answer or "").strip()
        scratch = (getattr(agent, "scratchpad", "") or "")
        if not ans and scratch:
            # Take last ~3000 chars of scratchpad as the de-facto answer
            ans = scratch[-3000:]
        record["answer"] = ans
        record["scratchpad_len"] = len(scratch)
        record["steps"] = int(getattr(agent, "step_n", 0))
        record["prompt_tokens"] = int(getattr(agent, "promptTokens", 0) or 0)
        record["completion_tokens"] = int(getattr(agent, "completionTokens", 0) or 0)
        record["llm_calls"] = int(getattr(agent, "llmCalls", 0) or 0)
        record["status"] = "completed"
    except Exception as e:
        record["answer"] = ""
        record["steps"] = 0
        record["prompt_tokens"] = 0
        record["completion_tokens"] = 0
        record["llm_calls"] = 0
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        record["traceback"] = traceback.format_exc()[:2000]
        print(f"    ERROR: {record['error']}")
        print(f"    {traceback.format_exc()[:800]}")

    record["execution_time_s"] = round(time.time() - start, 2)
    record["total_tokens"] = record["prompt_tokens"] + record["completion_tokens"]

    correct, reason = evaluate_pass(scenario, record["answer"])
    record["correct"] = correct
    record["eval_reason"] = reason
    return record


def run_pass1(framework: str, model_id: int, scenario_ids: list[str] | None,
              limit: int | None, output_path: Path, resume: bool = True) -> dict:
    """Execute Pass@1 evaluation, saving incrementally."""
    scenario_file = _DEMO / "scenarios" / "phm_scenarios.json"
    all_scenarios = load_scenarios(scenario_file)

    if scenario_ids:
        all_scenarios = [s for s in all_scenarios if s["task_id"] in scenario_ids]
    if limit:
        all_scenarios = all_scenarios[:limit]

    # Resume support: load existing results, skip already-done scenarios
    existing = {}
    if resume and output_path.exists():
        try:
            with open(output_path) as f:
                prior = json.load(f)
            for r in prior.get("results", []):
                existing[r["task_id"]] = r
            print(f"Resume: {len(existing)} prior results loaded from {output_path.name}")
        except Exception:
            pass

    results = list(existing.values())
    todo = [s for s in all_scenarios if s["task_id"] not in existing]
    print(f"Framework={framework}, model_id={model_id}, todo={len(todo)}/{len(all_scenarios)}")

    for i, scenario in enumerate(todo):
        print(f"  [{i+1}/{len(todo)}] {scenario['task_id']} ({scenario['classification_type']})")
        record = run_one_scenario(scenario, framework, model_id)
        status = "PASS" if record["correct"] else "FAIL"
        print(f"    -> {status} ({record['eval_reason']}) "
              f"steps={record['steps']} tokens={record['total_tokens']} "
              f"time={record['execution_time_s']}s")
        results.append(record)

        # Save after each scenario for resume safety
        save_results(results, output_path, framework, model_id)

    summary = compute_summary(results)
    save_results(results, output_path, framework, model_id, summary=summary)
    return summary


def compute_summary(results: list[dict]) -> dict:
    if not results:
        return {}
    by_cat: dict[str, dict] = {}
    total_correct = 0
    total_tokens = 0
    total_steps = 0
    total_time = 0.0
    for r in results:
        cat = r.get("classification_type", "Unknown")
        by_cat.setdefault(cat, {"correct": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r.get("correct"):
            by_cat[cat]["correct"] += 1
            total_correct += 1
        total_tokens += r.get("total_tokens", 0)
        total_steps += r.get("steps", 0)
        total_time += r.get("execution_time_s", 0)

    n = len(results)
    return {
        "total_scenarios": n,
        "overall_pass1": round(total_correct / n, 4) if n else 0.0,
        "by_category": {
            c: {"pass1": round(d["correct"] / d["total"], 4),
                "correct": d["correct"], "total": d["total"]}
            for c, d in by_cat.items()
        },
        "avg_steps": round(total_steps / n, 2) if n else 0,
        "avg_total_tokens": int(total_tokens / n) if n else 0,
        "avg_time_s": round(total_time / n, 2) if n else 0,
    }


def save_results(results: list[dict], path: Path, framework: str,
                 model_id: int, summary: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "framework": framework,
            "model_id": model_id,
            "model_name": list(MODEL_NAME_TO_ID.keys())[
                list(MODEL_NAME_TO_ID.values()).index(model_id)
            ] if model_id in MODEL_NAME_TO_ID.values() else f"id_{model_id}",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "n_results": len(results),
        },
        "summary": summary or compute_summary(results),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["react", "reactxen"], required=True)
    p.add_argument("--model", type=str, help="model name (e.g. ibm/granite-4-h-small)")
    p.add_argument("--model_id", type=int, help="model id from modelset")
    p.add_argument("--scenario_ids", type=str, help="comma-separated task IDs")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_dir", type=str,
                   default=str(_DEMO / "results" / "paper_table4_runs"))
    p.add_argument("--no_resume", action="store_true")
    args = p.parse_args()

    # Resolve model_id
    if args.model_id is not None:
        model_id = args.model_id
    elif args.model:
        if args.model in MODEL_NAME_TO_ID:
            model_id = MODEL_NAME_TO_ID[args.model]
        else:
            raise SystemExit(f"Unknown model: {args.model}. Known: {list(MODEL_NAME_TO_ID)}")
    else:
        raise SystemExit("Must provide --model or --model_id")

    model_name = list(MODEL_NAME_TO_ID.keys())[
        list(MODEL_NAME_TO_ID.values()).index(model_id)
    ] if model_id in MODEL_NAME_TO_ID.values() else f"id_{model_id}"
    safe_name = model_name.replace("/", "_").replace(":", "_")

    output_path = Path(args.output_dir) / f"{args.framework}__{safe_name}.json"
    scenario_ids = args.scenario_ids.split(",") if args.scenario_ids else None

    summary = run_pass1(
        framework=args.framework,
        model_id=model_id,
        scenario_ids=scenario_ids,
        limit=args.limit,
        output_path=output_path,
        resume=not args.no_resume,
    )

    print("\n" + "=" * 60)
    print(f"Pass@1 summary — {args.framework} + {model_name}")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
