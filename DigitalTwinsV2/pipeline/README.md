# Corrected pipeline

Drop-in replacements for `phmforge_calibration/`, plus the provenance gate.
Nothing here modifies the original package — you apply it when you're ready.

## Install

```bash
python DigitalTwinsV2/scripts/rerun_sweep.py selftest
```

That must print `SELFTEST PASSED` before you spend any API budget. It builds two
fixtures — one honestly captured, one reproducing the V1 defects — and checks
that the gate accepts the first and rejects the second.

The client patch and the corrected modules are **already applied** in this
working tree. To reapply from a clean checkout:

```bash
git apply DigitalTwinsV2/patches/0001-0002-combined-logprob-capture.patch
cp DigitalTwinsV2/pipeline/telemetry_logger.py phmforge_calibration/telemetry_logger.py
cp DigitalTwinsV2/pipeline/feature_extractor.py phmforge_calibration/feature_extractor.py
cp DigitalTwinsV2/pipeline/provenance.py      phmforge_calibration/provenance.py
```

Originals are preserved in `DigitalTwinsV2/backup_original/`.

## Credentials

Export these in the shell. **A `.env` file will not work**: nothing on the
`model_inference.py` code path calls `load_dotenv()` — it appears only under
`ReActXen/src/reactxen/demo/` and `experimental/`, neither of which is imported
here. The README at the repo root suggests writing a `.env`; that instruction is
wrong for this path.

| variable | needed for |
|---|---|
| `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID` | the five watsonx backbones (granite, llama-3.3, llama-4-maverick, both mistral). All three are mandatory. |
| `OPENAI_API_KEY` | `gpt-4o-mini` and `gpt-oss-120b` |
| `OPENAI_BASE_URL` | only if `gpt-oss-120b` is served by a gateway — it is not on `api.openai.com`, so verify how it was served originally |
| `BRAVE_API_KEY` | the `web_search` tool (142 calls in the old corpus) |

Missing watsonx credentials fail late and unhelpfully: `os.environ.get(..., "")`
yields an empty key, so you get an auth error at call time rather than a clear
"credential not set". Check them before launching.

## Smoke-test before spending budget

The logprob chain has four links, and all four must hold:

1. client sets `token_logprobs` (patched)
2. `prompt_agent` copies it to `formatted["logprobs"]` — present in `ReactAgent`
   already, added to `ReactReflectAgent` by the patch
3. the agent writes `json_log[-1]["thought_logprobs"]`
4. the logger records it with provenance `"provider"`

Link 2 only fires when `format_step` returns a dict, which depends on the
agent's `reactstyle`/`actionstyle`. So run **one** scenario first:

```bash
python DigitalTwinsV2/scripts/rerun_sweep.py capture \
  --models ibm/granite-4-h-small --scenarios pdm_rul_001 \
  --out results/smoke_test
```

If the gate prints `corpus is usable`, the logprobs arrived and the full sweep is
worth launching. If it reports `thought_logprobs: never measured`, link 2 or 3 is
still open — fix that before running 525 trajectories.

## What each file fixes

| file | defect it removes |
|---|---|
| `telemetry_logger.py` | `estimate_logprobs` (synthetic logprobs); `return 0.75` (constant confidence); confidence parsed once per trajectory (constant gradient by construction); `status_code` from regex over observation text; grader verdict left in `observation`; `execution_time_ms = 1200` fallback |
| `feature_extractor.py` | reads quarantined `harness_observation`; imputes features from unmeasured signals |
| `provenance.py` | *new* — refuses to let any of the above reach a model |

## The record schema

Every step now carries a `provenance` block. A value of `"unavailable"` means
the signal was not captured — the row gets dropped from that feature family, it
is never filled in.

```json
{
  "step_index": 0,
  "thought": "...",
  "action": "load_dataset",
  "action_input": "FD001",
  "observation": "Loaded 100 rows",          // tool output only
  "harness_observation": null,               // grader text, quarantined
  "thought_logprobs": [-0.02, -0.31, ...],   // or null
  "verbalized_confidence": 0.72,             // or null
  "mcp_tool_calls": [
    {"tool_name": "load_dataset", "status_code": 200, "source": "mcp_response"}
  ],
  "provenance": {
    "logprobs": "provider",
    "confidence": "parsed",
    "status_code": "mcp_response"
  }
}
```

## Running the sweep

```bash
python DigitalTwinsV2/scripts/rerun_sweep.py capture --scenarios pdm_rul_001 pdm_rul_002
```

Requires provider credentials and a live MCP tool stack. **This path has not
been executed** — there were no credentials in the environment where it was
written. Everything downstream of the agent boundary (logger → extractor →
gate) is covered by `selftest`. Expect to adapt `_default_agent_factory` to
your ReActXen entry point.

The run ends by invoking the gate automatically and exits non-zero if the
captured corpus is not usable.

## Verifying an existing corpus

```bash
python DigitalTwinsV2/pipeline/provenance.py results/telemetry_runs
```

On the current corpus this exits 1 with five fatal findings. That is correct
behaviour, not a bug.

## Two things to change beyond the code

1. **Use all 75 PHMForge scenarios, not 25.** The current corpus runs 25, giving
   n = 167 across seven backbones. The benchmark ships 75, so n ≈ 525 is
   available for the same engineering effort. At n = 167 nothing below roughly
   0.15 AUROC is detectable, which is why the V2 re-analysis can only report
   "uninformative" rather than a real answer.
2. **Drop backbones that cannot return logprobs.** A model that only supports
   text output cannot participate in a token-uncertainty study. Reporting it
   with an imputed column is what produced the original problem.
