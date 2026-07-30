# Forensic audit of the V1 trajectory-calibration pipeline

Every claim below is reproducible with `analysis/recompute_honest.py`, which
prints a `[provenance audit]` block before any modelling.

Corpus: `../results/telemetry_runs/`, 7 backbones × 25 PHMForge scenarios,
167 non-empty trajectories (gpt-4o-mini contributes 17), Pass@1 = 0.539.

---

## D0 — The logprob chain was broken in four places

Established while repairing the pipeline. Token logprobs pass through four
hand-offs between provider and feature table, and **every one** was broken:

| # | hand-off | state before |
|---|---|---|
| 1 | client requests logprobs | never requested; `logprob` appears **0 times** in the original `model_inference.py` |
| 2 | `prompt_agent` copies them into the step dict | present in `ReactAgent`, absent in `ReactReflectAgent` |
| 3 | agent serialises them to `json_log` | only the non-default `ThoughtThenAct` branch wrote `thought_logprobs`; the default `ThoughtActTogether` branch (`agents.py:911`) omitted it |
| 4 | logger records what it received | fabricated a replacement instead |

Fixing any three would have changed nothing observable, because the fourth still
supplied plausible numbers. This is the mechanism by which a fabricated feature
survives review: the defect is distributed across individually reasonable
layers, and the fallback removes the error signal that would force someone to
look.

**Additionally: the signal was unobtainable, not merely unrequested.** After
patching the client to ask, the hosting service refuses it:

```
WatsonxAPIWarning: parameter `parameters.return_options.token_logprobs`
is no longer supported for this model and is ignored
ID: param_not_supported
```

Verified on all five open-weights backbones, on both the `generate` and `chat`
endpoints (`ChatParams.LOGPROBS` / `TOP_LOGPROBS` are equally rejected). The
rejection is a *warning*, not an error, so the call succeeds and the field is
simply absent. Of the seven backbones only `gpt-4o-mini` returns logprobs.

**Verification after repair**, on the same measure that convicted the original:

| | original corpus | repaired pipeline |
|---|---|---|
| skew | −0.20 | **−3.31** |
| mass > −0.01 | 4.3% | **76.6%** |

Real logprobs are dominated by near-certain tokens and so are strongly
left-skewed; the fabricated ones were symmetric by construction.

---

## D1 — Token-level uncertainty features are synthetic

**Where.** `phmforge_calibration/telemetry_logger.py:101-103`

```python
logprobs = step.get("thought_logprobs")
if not logprobs:
    logprobs = self.estimate_logprobs(thought, confidence)
```

`estimate_logprobs` (`:62`) is documented as generating *"realistic token-level
logprobs if not returned by LLM"* and draws from

```
mean_logprob = -0.1 - (1.0 - confidence) * 1.2
std_dev      = 0.15 + (1.0 - confidence) * 0.3
```

**Evidence.** `confidence` is a constant 0.75 (see D2), so the generator is a
fixed `N(-0.400, 0.225²)`. Across all 203,107 logged tokens the corpus shows
**-0.4030 ± 0.2182**. No provider-returned logprobs survive anywhere. The only
deviation from the Gaussian is a spike at exactly 0.0 from the generator's own
clipping.

**Consequence.** `early_entropy` and `logprob_variance` — first and third in V1's
ranking, 52.7% of attributed importance — are functions of pseudorandom noise
independent of backbone, scenario, and outcome. Given only these two features a
random forest scores **AUROC 0.444, permutation p = 0.889**.

**Fix.** Delete the fallback. Record `null` when the provider returns nothing.

---

## D2 — Verbalized confidence was never elicited

**Where.** `phmforge_calibration/telemetry_logger.py:60`

```python
# Default fallback
return 0.75  # realistic baseline default
```

**Evidence.** Of 4,727 logged steps, 4,671 carry `verbalized_confidence = 0.75`
and 56 carry `0.5` (a separate fallback in `feature_extractor.py:54,74` for
empty logs). At trajectory level: **162 of 167 are exactly 0.75**, 5 are 0.5.

**Consequence.** V1's claim that verbalized confidence is uninformative (3.98%
importance) measures a constant. The conclusion agrees with the literature, but
was not tested here. Relatedly, `confidence_gradient` — the slope of confidence
across steps — is **zero to within floating-point error on all 167 trajectories**
(71 exactly 0, the rest |slope| < 1e-16 from `polyfit` rounding), so V1's
"6-dimensional" feature vector had at most 4 live dimensions and only 2 measured
ones.

The residual AUROC 0.532 for this column is itself an artifact: all 5
trajectories with 0.5 happen to be failures.

**Fix.** Remove the fallback; record `null` and exclude from that analysis.

---

## D3 — The grading harness leaks its verdict into observations

**Where.** The ReAct loop appends a terminal observation with the outcome:

| label | terminal observation | count |
|---|---|---|
| success | `"Task finished successfully"` (26 chars) | 37 |
| failure | `"Task failed"` (11 chars) | 33 |

**Evidence.** Any feature over observation text recovers the label from the
length difference alone.

- `mean_obs_len` univariate AUROC: **0.608**
- `mean_obs_len` alone, out-of-fold random forest: **0.833**

A gap that large between a monotone ranking and a non-monotone learner is the
signature of value fingerprinting rather than a trend.

**Aggravating factor.** 71 of 167 trajectories (42.5%) are single-step, so the
grader's message is their *only* observation. For those runs the observation
features are nothing but the label.

**Magnitude.** Removing terminal and verdict-bearing observations collapses the
full model from **0.881 → 0.566**. The leakage is not a few points of optimism;
it is the entire result.

**Fix.** Segregate harness-written trace content into a field feature extraction
cannot reach.

---

## D4 — Cost analysis was in-sample

**Where.** `scripts/06_generate_eoc_figure.py:50-52`

```python
calibrator = TrajectoryCalibrator(method="HTC").fit(X, y)
probs = calibrator.predict_probability(X)
```

Fit and scored on the same rows. The 83% saving was produced by a forest that
had memorised the labels. `scripts/03_generate_paper_artifacts.py` *does* use
5-fold CV for the ECE/AUROC table, so the defect is confined to the cost figure —
but that figure carries the paper's headline economic claim.

---

## D5 — Cost analysis omitted the always-abstain baseline

**Where.** Same file, `:56`

```python
raw_cost_per_scenario = np.mean((1 - y) * C_catastrophic)
```

The comparison baseline pays \$50,000 for every failure with no recourse, so any
policy that abstains at all appears to save money.

**Evidence** (out-of-fold, cost model as stated in the paper):

| policy | EOC/scenario |
|---|---|
| always execute | \$23,054 |
| selective, θ = 0.75 | \$2,144 ← reproduces V1's \$2,100 |
| selective, best θ = 0.90 | \$1,039 |
| **always abstain** | **\$1,039** |

The optimum is attained where the policy abstains on everything. V1's "optimised"
policy is **twice as expensive as doing nothing**.

**Also.** V1's Eq. 2 states a three-term cost with
`C_inspection · N_false_alarm`, but the implementation has no inspection term at
all. V2 implements the formula as stated.

---

## D6 — Importance was impurity-based, and unresolvable either way

**Where.** `scripts/03_generate_paper_artifacts.py:258`,
`calibrator.htc_model.feature_importances_`, fit on pooled data with no CV.

Mean-decrease-in-impurity is biased toward high-cardinality continuous
predictors (Strobl et al., 2007). Cardinality in this corpus:

| feature | distinct values | V1 importance |
|---|---|---|
| logprob_variance | 166 | 30.03% |
| early_entropy | 88 | 22.71% |
| mcp_error_ratio | 56 | 28.70% |
| loop_ratio | 65 | 14.58% |
| verbalized_score | 2 | 3.98% |
| confidence_gradient | 1 (constant 0) | ~0% |

The ranking tracks cardinality, and the two features it puts first are the
synthetic ones.

**But the deeper point:** under out-of-fold permutation importance no feature
reaches 1.6 standard errors from zero, and `logprob_variance` — a synthetic
column — still ranks third at +0.013 ± 0.010. At n = 167 no feature ordering is
resolvable. V2 therefore reports none.

---

## D7 — Folds were not grouped by scenario

`KFold(n_splits=5, shuffle=True)` over pooled data places sibling runs of the
same scenario on both sides of the split. V2 uses `StratifiedGroupKFold` with
scenario as the group.

---

## D8 — Degenerate AUROC returned as chance

**Where.** `scripts/03_generate_paper_artifacts.py:57-61`

```python
def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) <= 1:
        return 0.5
```

Returning 0.5 converts "undefined" into "chance", which is how a constant
predictor came to be reported as a measured null result. V2 returns `NaN` and
excludes.

---

## D9 — The best-scoring backbone mostly never terminated

Found while attempting the corrected re-collection. Trajectory well-formedness in
the original corpus:

| backbone | n | Pass@1 | unique-action ratio | reached `Finish` |
|---|---|---|---|---|
| granite-4-h-small | 25 | 0.480 | 0.60 | **100%** |
| llama-3.3-70b | 25 | 0.360 | 0.72 | **100%** |
| llama-4-maverick | 25 | 0.640 | 0.86 | **100%** |
| mistral-medium-2505 | 25 | 0.480 | 0.62 | **100%** |
| mistral-small-24b | 25 | 0.480 | 0.61 | **100%** |
| **gpt-4o-mini** | **17** | **0.706** | 0.57 | **29%** |
| gpt-oss-120b | 25 | 0.680 | 0.72 | **100%** |

`gpt-4o-mini` does not follow this ReAct template: it emits `Final Answer: ...`
as narrative prose instead of `Action: Finish`, the parser does not recognise the
termination, and the agent loops on one action until the step budget is spent. It
reached a proper `Finish` in 29% of runs against 100% for every other backbone,
and **8 of its 25 scenarios produced no trajectory at all**.

Two consequences:

1. **Its reported Pass@1 of 0.706 — the highest in the study — is
   survivorship-biased** and not comparable with the other rows. It is computed
   over the 17 runs that produced output, having silently dropped the 8 that did
   not.
2. **The token-uncertainty study is not viable on either half of this stack.**
   The five open-weights backbones drive the agent correctly (100% termination)
   but their host refuses logprobs (D0). `gpt-4o-mini` returns logprobs but only
   completes 29% of trajectories, and the degenerate remainder carries no label
   variance — re-collection on it yielded 0/3 successes, all single-action loops.

Self-hosted open-weights inference is therefore not merely the preferable
configuration for the corrected experiment; on the evidence it is the only one in
which both the features and the labels exist.

Note for anyone re-running: asking for in-band confidence makes this worse. A
model told to report a confidence over-generates past the action input, emitting
its whole remaining plan there, so the tool receives a blob and the loop tightens
(unique-action ratio 1/8 with the prompt against 3/8 without). Structural
sanitising at the call site is required, and in-band elicitation should default
to off.

---

## D10 — The structural features are reconstruction artifacts too

`02_run_experimental_sweep.py` does not observe the agent. It re-derives
trajectories by parsing ReActXen result logs after the fact, and the
reconstruction duplicates steps. Across 167 trajectories the `Finish` action
appears **866 times**:

| where `Finish` appears | count |
|---|---|
| as an intermediate step | **790** |
| as the terminal step | 76 |
| no `Finish` at all | 12 trajectories |

796 of those steps carry an empty observation. A ReAct agent calls `Finish` once
and stops; five per trajectory is not agent behaviour.

This invalidates the remaining "measured" family. `n_steps`, `loop_ratio`,
`n_unique_actions` and `reached_finish` are all computed over a step sequence
that the reconstruction inflated, so the structural features describe the log
parser, not the agent. Together with D0-D3 this means **no feature in the V1
corpus — token, verbalized, error-rate, or structural — is a measurement of the
system it purports to describe.**

It also invalidates comparison against V1 on trace shape. We initially read a
drop in terminal-`Finish` rate (100% in V1 against 17% in a live capture) as a
regression caused by our own configuration choice, restarted a collection on that
basis, and were wrong: V1's rate is an artifact of counting duplicated steps.
Live capture is the only valid measurement here, and V1 provides no trace-shape
baseline to compare it with. Pass@1 remains comparable, since the label comes
from the grader rather than from the reconstruction.

The lesson generalises past this codebase: telemetry re-derived from logs is not
telemetry. If the instrumentation does not observe the execution as it happens,
the shape of what it reports is a property of the parser.

---

## D11 — The label does not measure prognostic competence

Found while checking whether parallel backbones would race on the model file.
They would not, and the reason is the defect: `PredictRULTool._run` never uses
the trained model. It loads it and throws the result away —
`pickle.load(f)` with no assignment, inside `except: pass`, annotated *"kept for
backwards compat"*. Predictions are synthesised:

```python
gt = _load_rul_ground_truth(fd)              # the TRUE RUL values
rng = np.random.default_rng(seed=42)
shifts = rng.normal(loc=-3.0, scale=14.0, size=n)
preds = [max(0.0, round(g + s, 2)) for g, s in zip(gt, shifts)]
```

Ground truth plus Gaussian noise at a fixed seed. The docstring states the
intent plainly: *"calibrated noise (~13-cycle MAE on FD001), so MAE/RMSE
downstream produce realistic numbers in the published-benchmark range."*

Measured against the grader's own acceptance window:

| | generated | grader accepts | |
|---|---|---|---|
| MAE | **9.01** | [7, 18] | inside |
| RMSE | **11.21** | [11, 26] | inside |

`seed=42` is fixed, so these are identical on every run, for every backbone,
forever. `train_rul_model` does train a real LSTM once torch works — the model
is simply never consulted.

**What this does to the study.** On RUL scenarios, task success does not measure
the agent's prognostic accuracy. The accuracy belongs to the noise generator.
What success measures is whether the agent orchestrated the tools in the right
order and reported the numbers they handed back. This applies to the corrected
corpus too: it is a property of the benchmark's tool layer, not of the
instrumentation we fixed.

**It does not invalidate trajectory calibration**, but it narrows the claim
sharply. "Can an agent predict whether it will complete a tool-orchestration task
correctly?" is a real question with operational value — orchestration failure is
what breaks agentic systems in production. "Can an ADT predict prognostic
failure?" is not what this benchmark measures, and any paper framing it that way
is overclaiming, ours included.

This is the first defect here that concerns the **label** rather than the
features. A feature audit would never have found it: every feature can be a
perfect measurement and the target still not be the quantity of interest.

---

## D12 — The corrected pipeline produced a false result too

Recorded because it is the most useful entry here: the defect was ours, it
appeared *after* every fix above, and the gate as designed did not catch it.

The Together account exhausted its credit partway through collection. Together
does not raise on refusal in a way the agent sees — `watsonx_llm` swallows the
exception and returns an empty generation. The ReAct parse then yields nothing,
the agent spends its full step budget writing "The generated Action and/or
Action Input are incorrectly formatted", and the trajectory is recorded as an
ordinary failure. 200 of 225 trajectories were produced this way, in 17 seconds
each against a normal 168.

The analysis then reported:

```
token        AUROC 0.921 [0.871,0.965]  p=0.0005   [SIG]
structural   AUROC 0.956 [0.915,0.988]  p=0.0005   [SIG]
observation  AUROC 0.939 [0.889,0.976]  p=0.0005   [SIG]
```

All of it spurious. Token features survived on 25 trajectories and **all 11
successes lay inside those 25**, so any model given a provenance-gated family
separated the classes by detecting which rows had been collected successfully.
The gate passed the corpus because every recorded value *was* a genuine
measurement — provenance was never the problem.

Three diagnoses given before the cause was found were wrong, all with this same
root: Qwen-7B "too weak for the task" (0/25), gpt-oss-120b "incompatible with
the ReAct parser" (0/75, empty records), and the extension "losing logprobs".
None of these were real. The tell was arithmetic, not statistical: 150
trajectories finished in 44 minutes.

### The two checks this produced

`check_degenerate` — a trajectory with no extracted action is a failed
collection, not a hard instance. Fatal above 10% of the corpus.

`check_availability_leak` — for each provenance-gated family, compare the
outcome rate on rows that have the column against rows that do not, and report
what share of the positives each group holds. Fatal when one group holds more
than 90% of them. This is leakage that provenance checking structurally cannot
see, because it is a property of *which rows were collected*, not of the values.

A third guard sits upstream: `preflight` in `rerun_sweep.py` generates once per
backbone before collecting and aborts if the response is empty or lacks the
fields the study depends on. It turns a nine-hour silent failure into a
ten-second loud one.

**The generalisable form:** a provider that fails by returning nothing is
indistinguishable from a model that performs badly, unless you check that it
answered at all. And a feature that is missing non-randomly is a label in
disguise, however honestly its present values were measured.

---

## Reporting defects (no code involved)

- **No sample size stated.** The paper never says n = 167, nor n ≤ 25 per
  backbone.
- **No intervals or significance tests.** Three-decimal point estimates only.
  Several V1 AUROC values are below 0.5 (0.153, 0.250, 0.276, 0.316, 0.404) with
  no comment.
- **`ECE = 0.000` reported as a win.** Isotonic regression on a constant input
  predicts the base rate, which is perfectly calibrated and perfectly
  non-discriminative (AUROC 0.404). Bolding it inverts the finding.
- **Section titled "SHAP / Diagnostic Utility"** reports Gini importance. No SHAP
  values were computed.
- **Bibliography of four entries, none cited in the body**; two with wrong
  authors, titles, and venues (see `README.md`).
- **Claim of "selective accuracy of 100% at >60% coverage"** appears nowhere in
  the artifacts and is not reproducible from the released code.
- **Venue status inconsistent.** The compiled V1 PDF abstract reads
  "accepted/submitted for IEEE Smart World Congress (SWC 2026)"; the current
  `.tex` says "submitted". Claiming acceptance in a submission is a desk-reject
  risk. V2 states no venue status.
- **The compiled V1 PDF was broken** (as audited at commit `bd79184`).
  `\documentclass[conference,twocolumn]{article}` — `article` has no `conference`
  option — put the title on page 3, dropped all four figures, and rendered no
  citations. A concurrent editing session has since been reworking the V1
  formatting (commits `7ca338b`..`0720601`), so this item may no longer apply;
  every other finding in this document concerns
  `phmforge_calibration/`, `scripts/`, and `results/`, which those commits did
  not touch. V2 uses `IEEEtran`.

---

## What would make this study valid

1. Capture real provider logprobs; drop backbones that cannot supply them.
   **This rules out the managed gateway entirely** — see D0. Use self-hosted
   vLLM/SGLang/llama.cpp, or a provider that exposes logprobs over an
   OpenAI-compatible API. `pipeline/openweights_models.py` has presets and a
   one-call-per-model probe; run the probe before spending budget, because
   providers change what they expose.
2. Elicit confidence with an explicit prompt and parse it; record `null` on
   failure. **Strip the reported value structurally**, not by prompt wording:
   models append it after the action input, where it corrupts the tool argument
   and inflates apparent action diversity. Both call site and loop detection need
   the strip.
3. Quarantine harness-written trace content **on evidence, not position**. A run
   that ends by exhausting its step budget has ordinary tool output in its last
   observation; discarding it by position throws away a real measurement.
4. Target substantive multi-step trajectories (the current corpus is 42.5%
   single-step, zero-tool runs).
5. Scale n well beyond 167 — PHMForge ships **75** scenarios and the corpus used
   25, so n ≈ 525 across seven backbones is available for the same effort. The
   current sample has little power below ~0.15 AUROC.
6. Report grouped CV, intervals, permutation tests, a multiple-comparison
   correction, and both trivial policies.
7. Gate per column, not per corpus. A defect in one column disqualifies the
   feature families that depend on it, not the dataset: "drop this feature" and
   "discard this corpus" are different instructions, and a gate that conflates
   them gets switched off.

## Environment notes (not defects in the study)

- **torch was broken**, so `train_rul_model` / `train_fault_model` silently
  returned `[MOCK - torch missing]`, which would have contaminated 30 of 75
  scenarios. The original corpus was *not* affected — zero MOCK observations in
  4,727 steps. Cause: this is the Microsoft Store Python, whose site-packages
  path makes torch's CUDA headers exceed the 260-character Windows `MAX_PATH`
  limit, so the wheel cannot unpack. Fixed by installing to `C:\dtlibs` with
  `pip --target` plus a `.pth`, no system settings touched. A partial 56 MB
  `torch/` left by the failed install shadowed the working copy and was moved to
  `torch.broken-failed-install`.
- **`BRAVE_API_KEY` is unset**, so `web_search` returns a placeholder. It was
  used 142 times in the original corpus, so scenarios that reach for it behave
  differently now.
