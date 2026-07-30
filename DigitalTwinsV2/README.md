# DigitalTwinsV2 — audited rewrite of the IEEE SWC 2026 paper

Self-contained alternative to `IEEE_SWC_2026/`. The only change made to the
original paper folder was fixing three incorrect citations (see below); its
claims are untouched.

```
DigitalTwinsV2/
├── IEEE_SWC_2026_DigitalTwins_V2.tex   # the rewritten paper (IEEEtran)
├── IEEE_SWC_2026_DigitalTwins_V2.pdf   # compiled: 7 pp. body + 1 pp. refs
├── AUDIT.md                            # forensic findings, D0-D10
├── analysis/
│   ├── recompute_honest.py             # re-evaluates the ORIGINAL corpus
│   └── analyze_v2_corpus.py            # analyses a NEW provenance-tagged corpus
├── pipeline/                            # corrected drop-in replacements
│   ├── provenance.py                   # the gate (reusable core)
│   ├── telemetry_logger.py             # never imputes; tags provenance
│   ├── feature_extractor.py            # refuses unmeasured columns
│   └── openweights_models.py           # provider presets + logprob probe
├── scripts/rerun_sweep.py              # selftest | capture | verify
├── patches/                            # applied to ReActXen (reversible)
├── backup_original/                    # pre-change copies of every file touched
├── figures/ tables/                    # generated, do not hand-edit
```

## Start here

```bash
python DigitalTwinsV2/scripts/rerun_sweep.py selftest
```

Must print `SELFTEST PASSED` before anything else. It builds two fixtures — one
honestly captured, one reproducing the original defects — and asserts the gate
accepts the first and rejects the second.

```bash
python DigitalTwinsV2/pipeline/provenance.py results/telemetry_runs
```

Exits 1 on the original corpus with five fatal findings. That is correct.

## Why the paper changed

The original reported an 83% reduction in expected operational cost and a
feature ranking led by logprob variance (30.0%). Those numbers are
reproducible but do not measure what the paper says. `AUDIT.md` has the detail.

| # | Defect | Evidence |
|---|---|---|
| D0 | The logprob chain was broken in **four** independent places | `logprob` appears 0 times in the original client; the agent dropped the field in one of two classes, then again in the default serialisation branch; the logger fabricated a replacement |
| D1 | Token features are **synthetic** | All 203,107 values fit the logger's own `N(-0.40, 0.225²)`; observed `-0.4030 ± 0.2182` |
| D2 | Verbalized confidence **never elicited** | 162/167 trajectories carry the hard-coded `0.75`; parsed once per trajectory, so its gradient is 0 by construction |
| D3 | Tool-error rate is a **text heuristic** | `status_code` set by substring-matching the observation for "Error"/"failed"; no MCP response is consulted |
| D4 | The grader **leaks its verdict** | Terminal observation is literally `"Task finished successfully"` / `"Task failed"`; the *only* observation for the 71 single-step runs |
| D10 | Telemetry is **reconstructed from logs**, not observed | The single-use `Finish` action appears 866 times across 167 trajectories, 790 of them mid-trajectory — so every structural feature describes the log parser |
| D5–D8 | In-sample cost analysis; missing always-abstain baseline; impurity-based importance; ungrouped folds | see `AUDIT.md` |
| D9 | The best-scoring backbone mostly never terminated | `gpt-4o-mini` reached `Finish` in 29% of runs vs 100% elsewhere, and 8 of its 25 scenarios produced nothing — its 0.706 Pass@1 is survivorship-biased |

Taken together: **no feature in the original corpus measures the system it
purports to describe.**

## What V2 reports on the original corpus

Leakage-controlled, scenario-grouped CV, bootstrap CIs, permutation tests:

| Signal set | AUROC [95% CI] | p |
|---|---|---|
| Synthetic token features | 0.444 [0.348, 0.531] | 0.889 |
| Verbalized confidence | 0.532 [0.507, 0.562] | 0.022 |
| Execution + clean observations | 0.546 [0.456, 0.630] | 0.153 |
| Execution telemetry | 0.566 [0.479, 0.650] | 0.074 |
| V1 feature vector f(τ) | 0.590 [0.504, 0.674] | 0.025 |
| *Execution + grader-contaminated obs.* | *0.881 [0.829, 0.925]* | *<0.001* ⚠ artifact |

Two rows clear an uncorrected 0.05; with six comparisons the Bonferroni
threshold is 0.0083 and only the contaminated row clears it. The cost-optimal
abstention policy degenerates to always-abstain (\$1,039/scenario against the
\$2,144 the original called optimised).

## Collecting a valid corpus

The pipeline fixes are **already applied** in this working tree. From a clean
checkout:

```bash
git apply DigitalTwinsV2/patches/0001-0002-combined-logprob-capture.patch
cp DigitalTwinsV2/pipeline/*.py phmforge_calibration/
```

### Credentials

Export in the shell, or use `.env` — `rerun_sweep.py` loads it before anything
imports the inference client. One variable per provider, so adding one never
clobbers a working `OPENAI_API_KEY`.

| variable | for |
|---|---|
| `TOGETHER_API_KEY` | open-weights backbones **with logprobs** |
| `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID` | the original five backbones (no logprobs — see below) |
| `BRAVE_API_KEY` | the `web_search` tool (142 calls in the original corpus) |

### Which backbones can actually be studied

Managed gateways do not expose token logprobs. All five original open-weights
backbones answer `parameter 'logprobs' is no longer supported for this model and
is ignored`, on both the `generate` and `chat` endpoints — a *warning*, so the
call succeeds and the field is silently absent. Verified 2026-07-30.

Providers serving the same weights over an OpenAI-compatible API do expose them:

```bash
python DigitalTwinsV2/pipeline/openweights_models.py together
```

One call per model; run it before spending budget, because availability changes.
Confirmed working: `Llama-3.3-70B-Instruct-Turbo`, `Qwen2.5-7B-Instruct-Turbo`,
`openai/gpt-oss-120b`.

### Capture

```bash
python DigitalTwinsV2/scripts/rerun_sweep.py capture \
  --provider together --framework reactxen \
  --scenarios $(cat DigitalTwinsV2/v1_scenarios.txt) --out results/v2_together
```

Ends by running the gate and exits non-zero if no measured family survives.
`--elicit-confidence` is **off by default**: asking for confidence in-band makes
models over-generate past the action input, which corrupts the tool argument and
drives the agent into a single-action loop. Measured on this corpus it took
Pass@1 to 0.

### Analyse

```bash
python DigitalTwinsV2/analysis/analyze_v2_corpus.py results/v2_together
```

## Things that will bite you

- **Reconcile attempted against recorded.** Agent output containing a Unicode
  arrow raises an encoding error on a Windows console; the exception escapes the
  per-scenario handler and the trajectory is lost with `steps=0`. This destroyed
  44% of one run, selectively, and the survivors looked healthy with a *better*
  Pass@1. `rerun_sweep.py` now forces UTF-8 on stdout/stderr.
- **torch and `MAX_PATH`.** Under the Microsoft Store Python, torch's CUDA
  headers exceed 260 characters and the wheel cannot unpack, so
  `train_rul_model` silently returns `[MOCK - torch missing]`. Installed to
  `C:\dtlibs` via `pip --target` plus a `.pth`; a partial 56 MB `torch/` from the
  failed install shadowed it and was moved aside.
- **Statistical power.** At n=75 only effects ≥0.705 AUROC are detectable at the
  corrected threshold; at n=225, ≥0.62. A null result at n=75 would be
  uninformative rather than negative — the same criticism this paper makes of the
  original.

## Citation corrections carried into the original paper

- **PHMForge** — was `A. Das, M. Li, J. Smith, arXiv:2501.10982, 2025`. That ID
  resolves to a particle-physics paper on charmed-baryon decays. Correct: Feng,
  Chen, Tsai, Sun, Das, El Maghraoui, Lin, Patel, arXiv:2604.01532, 2026.
- **CalVerT** — was `T. Gupta, S. Lee, A. Ng, EMNLP 2025`. Correct: Vinod, Ding,
  Stengel-Eskin, arXiv:2606.21777, 2026; it targets knowledge-intensive QA, not
  tool telemetry, and the described behaviour was invented. The original's
  "CalVerT" baseline was an own reimplementation and is relabelled.
- **Boiko et al.** — the arXiv preprint title was spliced onto the *Nature*
  record with wrong pages. Correct: "Autonomous chemical research with large
  language models", Nature 624, 570–578, 2023.

Prose naming the wrong authors was fixed alongside the entries; keys renamed to
match. Backup in `backup_original/`.
