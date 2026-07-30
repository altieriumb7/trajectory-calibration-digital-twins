#!/usr/bin/env python3
"""Rerun the PHMForge sweep capturing signals instead of synthesising them.

Three subcommands:

``selftest``  exercises logger -> extractor -> validator on two fixtures, one
              honest and one built to imitate the V1 defects. The gate must
              accept the first and reject the second. Runs offline; use this to
              confirm the pipeline before spending API budget.
``capture``   runs agents live and writes provenance-tagged telemetry. Requires
              provider credentials and a working MCP tool stack.
``verify``    runs the provenance gate over an existing telemetry directory.

The important difference from ``scripts/02_run_experimental_sweep.py``: that
script did not run agents at all. It reconstructed telemetry post hoc from
ReActXen result logs, which is why no logprobs ever existed -- the original runs
never recorded any. Token-level uncertainty has to be captured at generation
time, so ``capture`` drives the agent itself.

Before running ``capture``, apply ``patches/0001-capture-real-logprobs.patch``
so the client actually requests logprobs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_V2 = _HERE.parent
_ROOT = _V2.parent
sys.path.insert(0, str(_V2))

# Agent output contains non-Latin-1 characters (arrows, dashes, superscripts).
# On Windows, stdout redirected to a file defaults to cp1252, so printing one
# raises UnicodeEncodeError, the exception propagates out of run_one_scenario,
# and the whole trajectory is lost with steps=0. Measured on this corpus that
# destroyed 44% of scenarios -- and non-randomly, since it selects against the
# more elaborate analytical answers. Reconfiguring here fixes every downstream
# print, including those inside ReActXen.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# Load .env before anything imports reactxen.utils.model_inference, which copies
# credentials out of os.environ at module import time. Nothing on that code path
# calls load_dotenv itself -- it appears only under reactxen/demo and
# reactxen/experimental -- so a .env is otherwise silently ignored.
try:
    from dotenv import load_dotenv

    _env_file = _ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass  # credentials must then come from the real environment

from pipeline.feature_extractor import FeatureExtractor  # noqa: E402
from pipeline.provenance import validate  # noqa: E402
from pipeline.telemetry_logger import TelemetryLogger  # noqa: E402

# Appended to the agent's system prompt so confidence is *elicited*, not guessed.
# V1 never asked for it, then recorded a hard-coded 0.75 when parsing failed.
CONFIDENCE_PROMPT = (
    "\n\nCONFIDENCE REPORTING: every Thought must end with its own final line of"
    " exactly this form:\n"
    "Confidence: <integer 0-100>\n"
    "expressing how likely you believe your current plan is to solve the task"
    " correctly.\n"
    "Place that line INSIDE the Thought, on its own line, immediately BEFORE the"
    " Action line.\n"
    "NEVER put it on the same line as Action or Action Input, and never after"
    " Action Input -- doing so corrupts the tool argument and the call will fail."
)

DEFAULT_MODELS = (
    "ibm/granite-4-h-small",
    "meta-llama/llama-3-3-70b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    "mistralai/mistral-medium-2505",
    "mistralai/mistral-small-3-1-24b-instruct-2503",
    "openai/gpt-oss-120b",
    "openai/gpt-4o-mini",
)


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def _honest_steps(rng: np.random.Generator, n_steps: int = 6) -> list[dict[str, Any]]:
    """A trajectory whose signals were genuinely measured.

    Logprobs follow the shape real ones have: mass piled just below zero with a
    long left tail, which is what distinguishes them from a fitted Gaussian.
    """
    steps: list[dict[str, Any]] = []
    for i in range(n_steps):
        n_tok = int(rng.integers(40, 90))
        logprobs = (-rng.exponential(0.25, n_tok)).tolist()
        steps.append(
            {
                "thought": f"Step {i}: inspecting sensor window.\nConfidence: {60 + 5 * i}",
                "action": "Finish" if i == n_steps - 1 else "load_dataset",
                "action_input": f"FD00{i % 4 + 1}",
                "observation": (
                    "Task finished successfully" if i == n_steps - 1
                    else f"Loaded 100 rows, window={i}"
                ),
                "thought_logprobs": logprobs,
                "status_code": 200 if i % 3 else 400,
            }
        )
    return steps


def _v1_style_steps(rng: np.random.Generator, n_steps: int = 6) -> list[dict[str, Any]]:
    """A trajectory reproducing the V1 defects, written directly as JSONL."""
    records = []
    for i in range(n_steps):
        n_tok = int(rng.integers(40, 90))
        records.append(
            {
                "step_index": i,
                "thought": f"Step {i}: no confidence stated here.",
                "action": "Finish" if i == n_steps - 1 else "load_dataset",
                "action_input": f"FD00{i % 4 + 1}",
                # defect 3: grader verdict left in the readable observation
                "observation": (
                    "Task finished successfully" if i == n_steps - 1
                    else f"Loaded 100 rows, window={i}"
                ),
                # defect 1: logprobs from a fixed Gaussian
                "thought_logprobs": np.minimum(
                    0.0, rng.normal(-0.40, 0.225, n_tok)
                ).tolist(),
                # defect 2: hard-coded constant
                "verbalized_confidence": 0.75,
                "mcp_tool_calls": [
                    {"tool_name": "load_dataset", "status_code": 200}
                ] if i < n_steps - 1 else [],
                "task_success": True,
            }
        )
    return records


def run_selftest() -> int:
    import shutil
    import tempfile

    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="phmforge_selftest_"))
    ok = True
    try:
        # --- fixture A: honest capture through the corrected logger ----------
        good = tmp / "good"
        logger = TelemetryLogger(good)
        for k in range(12):
            logger.log_trajectory(
                model_name="test/honest",
                scenario_id=f"scenario_{k:03d}",
                steps_log=_honest_steps(rng),
                task_success=bool(k % 2),
                final_answer="Final answer. Confidence: 72",
            )
        rep_good = validate(good)
        print(rep_good.render())
        if not rep_good.ok:
            print("\nSELFTEST FAIL: honest fixture was rejected\n")
            ok = False

        # --- fixture B: V1-style defects, written raw ------------------------
        bad = tmp / "bad"
        (bad / "test_v1style").mkdir(parents=True)
        for k in range(12):
            path = bad / "test_v1style" / f"scenario_{k:03d}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for rec in _v1_style_steps(rng):
                    handle.write(json.dumps(rec) + "\n")
        print()
        rep_bad = validate(bad)
        print(rep_bad.render())
        if rep_bad.ok:
            print("\nSELFTEST FAIL: V1-style fixture was accepted\n")
            ok = False
        else:
            checks = {f.check for f in rep_bad.fatal}
            missing = {"synthetic", "leakage"} - checks
            if missing:
                print(f"\nSELFTEST FAIL: gate missed {sorted(missing)}\n")
                ok = False

        # --- extractor must gate on provenance --------------------------------
        print()
        df = FeatureExtractor().extract_dataset(good)
        print(f"[extractor] {len(df)} trajectories from the honest fixture")
        for family in ("token", "verbalized", "execution"):
            FeatureExtractor.usable(df, family)
        if not df["usable_token"].all():
            print("SELFTEST FAIL: token family should be usable on honest fixture")
            ok = False
        if df["confidence_gradient"].std() == 0:
            print("SELFTEST FAIL: confidence gradient is degenerate")
            ok = False
        if df["obs_mean_len"].isna().all():
            print("SELFTEST FAIL: no admissible observations survived")
            ok = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
def preflight(models: Sequence[str], runner: Any) -> bool:
    """One tiny generation per backbone before any collection starts.

    Checks three things the collection silently depends on: the provider answers,
    it answers *for this model*, and it returns logprobs. Costs a few tokens and
    turns a nine-hour silent failure into a ten-second loud one.
    """
    print("preflight:")
    all_ok = True
    for model in models:
        model_id = runner.MODEL_NAME_TO_ID.get(model)
        if model_id is None:
            print(f"  [FAIL] {model}: not registered")
            all_ok = False
            continue
        try:
            from reactxen.utils.model_inference import watsonx_llm  # type: ignore

            # Budget generously. Reasoning models spend the allowance on an
            # internal trace before emitting any content, so a small probe reads
            # as "empty generation" on a perfectly healthy backbone. At 8 tokens
            # gpt-oss-120b returned nothing and this check skipped 75 runs; at
            # 64 it answers normally.
            resp = watsonx_llm("Reply with exactly: OK", model_id=model_id,
                               max_tokens=64)
            text = (resp.get("generated_text") or "").strip()
            lp = resp.get("token_logprobs")
            if not text:
                print(f"  [FAIL] {model}: empty generation")
                all_ok = False
            elif not lp:
                # Not fatal: the run can proceed without the token family, but
                # the operator should know before spending hours on it.
                print(f"  [warn] {model}: answers, but returns no logprobs")
            else:
                print(f"  [ok]   {model}: {len(lp)} logprobs")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {model}: {type(exc).__name__} "
                  f"{getattr(exc, 'status_code', '')} {str(exc)[:90]}")
            all_ok = False
    return all_ok


def run_capture(
    models: Sequence[str],
    scenarios: Sequence[str],
    out_dir: Path,
    framework: str = "reactxen",
    provider: str | None = None,
    elicit_confidence: bool = False,
) -> int:
    """Drives agents live and writes provenance-tagged telemetry.

    This delegates to ``phmforge_runner.run_one_scenario``, which already knows
    how to build a ReActXen agent for a PHMForge scenario, wire the PHM tools,
    grade the answer, and invoke the telemetry logger with
    ``log_structured_messages=True``. An earlier version of this function
    hand-rolled its own agent factory and got the constructor signature wrong;
    reusing the runner is both correct and less to maintain.

    The destination is redirected through ``PHMFORGE_TELEMETRY_DIR`` because the
    runner constructs ``TelemetryLogger()`` with no arguments.

    NOTE: needs provider credentials and the PHM tool stack. Not executed in the
    environment where it was written; ``selftest`` covers everything downstream
    of the agent boundary.
    """
    os.environ["PHMFORGE_TELEMETRY_DIR"] = str(out_dir)
    sys.path.insert(0, str(_ROOT))

    try:
        import phmforge_runner as runner  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"cannot import phmforge_runner: {exc}")
        return 1

    # Register open-weights backbones served over an OpenAI-compatible endpoint.
    # Needed because the managed gateway refuses logprobs on every open-weights
    # model it hosts, making token uncertainty unobservable there.
    if provider:
        sys.path.insert(0, str(_V2))
        from pipeline.openweights_models import register

        requested = list(models)
        registered = register(provider, extra_models=requested or None)
        # Registration stores ids under the routing prefix, so a name given on
        # the command line will not match MODEL_NAME_TO_ID as typed. Translate
        # the request rather than skipping it silently.
        if requested:
            models = [
                f"openai/{m}" if f"openai/{m}" in registered else m
                for m in requested
            ]
        else:
            models = registered
        print(f"provider {provider}: {len(registered)} registered, "
              f"running {len(models)} via {os.environ.get('OPENAI_BASE_URL')}")

    scenario_file = runner._DEMO / "scenarios" / "phm_scenarios.json"
    all_scenarios = {s["task_id"]: s for s in runner.load_scenarios(scenario_file)}

    missing = [s for s in scenarios if s not in all_scenarios]
    if missing:
        print(f"unknown scenario ids: {missing}")
        print(f"available: {sorted(all_scenarios)[:8]} ... ({len(all_scenarios)} total)")
        return 1

    # Preflight. A provider that refuses does not stop the agent loop: the call
    # fails, generated_text comes back empty, the ReAct parse yields nothing, and
    # the run burns its whole step budget writing "incorrectly formatted"
    # observations. The trajectory is then recorded as a normal failure. We lost
    # 200 of 225 that way -- fast, silent, and indistinguishable from a weak
    # model until you notice the runs finished ten times too quickly.
    if not preflight(models, runner):
        print("\nABORTED before collecting: the provider is not answering.")
        print("Nothing was written. Fix credentials/credit and rerun.")
        return 1

    n_ok = 0
    for model in models:
        model_id = runner.MODEL_NAME_TO_ID.get(model)
        if model_id is None:
            print(f"[skip] {model}: not in MODEL_NAME_TO_ID")
            continue
        for scenario_id in scenarios:
            scenario = dict(all_scenarios[scenario_id])
            # In-band confidence elicitation is off by default: asking for it
            # makes models over-generate past the action input, which corrupts
            # the tool argument and sends the agent into a single-action loop
            # until the step budget is exhausted. Measured on this corpus it
            # drove Pass@1 to 0. Enable only if you need the verbalized family
            # and have checked that trajectories still terminate.
            if elicit_confidence:
                scenario["input_question"] = (
                    scenario.get("input_question", "") + CONFIDENCE_PROMPT
                )
            record = runner.run_one_scenario(scenario, framework, model_id)
            status = record.get("status")
            print(f"[{status}] {model}/{scenario_id} "
                  f"steps={record.get('steps')} correct={record.get('correct')}")
            if status == "completed":
                n_ok += 1
            elif record.get("error"):
                print(f"        {record['error']}")

    print(f"\n{n_ok} trajectories completed; telemetry in {out_dir}")
    print("running provenance gate...\n")
    report = validate(out_dir)
    print(report.render())
    return 0 if report.ok else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="offline pipeline check (no credentials)")

    cap = sub.add_parser("capture", help="run agents live")
    cap.add_argument("--models", nargs="*", default=None,
                    help="model ids; omit with --provider to use its full set")
    cap.add_argument("--scenarios", nargs="+", required=True)
    cap.add_argument("--out", type=Path, default=_ROOT / "results" / "telemetry_runs_v2")
    cap.add_argument("--provider", default=None,
                    help="open-weights provider (together, fireworks, deepinfra, "
                         "groq, vllm, ollama). Sets OPENAI_BASE_URL and registers "
                         "its models.")
    cap.add_argument("--framework", default="reactxen", help="react | reactxen")
    cap.add_argument("--elicit-confidence", action="store_true",
                    help="ask the agent to report a confidence in-band. OFF by "
                         "default: it perturbs execution (see run_capture).")

    ver = sub.add_parser("verify", help="provenance gate on an existing directory")
    ver.add_argument("telemetry_dir", type=Path)

    args = ap.parse_args()
    if args.cmd == "selftest":
        return run_selftest()
    if args.cmd == "capture":
        models = args.models if args.models else ([] if args.provider else list(DEFAULT_MODELS))
        return run_capture(models, args.scenarios, args.out,
                           framework=args.framework, provider=args.provider,
                           elicit_confidence=args.elicit_confidence)
    report = validate(args.telemetry_dir)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
