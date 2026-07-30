# Trajectory Calibration for Agentic Digital Twins — replication artifact

Supporting code and data for *"When Agentic Digital Twin Telemetry Measures
Itself: A Provenance Audit and a Gate"*, submitted to the 2026 IEEE
International Conference on Digital Twin.

The manuscript is `DigitalTwinsV2/IEEE_DT2026_ShortPaper.tex`.

Everything reported in the paper is regenerated from the raw telemetry in this
repository by the scripts below. No number in the manuscript is transcribed by
hand: the result tables are emitted as LaTeX and `\input{}` by the source.

## Quick start

```bash
python DigitalTwinsV2/scripts/rerun_sweep.py selftest
```

Must print `SELFTEST PASSED`. It builds two synthetic corpora — one honestly
captured, one reproducing the defects the paper documents — and asserts the
provenance gate accepts the first and rejects the second.

```bash
python DigitalTwinsV2/pipeline/provenance.py results/telemetry_runs   # exits 1
python DigitalTwinsV2/pipeline/provenance.py results/v2_clean         # exits 0
```

The first corpus is the one audited in Sec. III; the gate rejects it with six
fatal findings. The second is the re-collected corpus; it passes, with the
families that remain unavailable named explicitly.

```bash
python DigitalTwinsV2/analysis/analyze_v2_corpus.py results/v2_clean
```

Reproduces Table II and the per-backbone and attribution results.

## Layout

| path | contents |
|---|---|
| `DigitalTwinsV2/pipeline/provenance.py` | the gate: six defect detectors plus the degenerate-run and availability-leak checks |
| `DigitalTwinsV2/pipeline/telemetry_logger.py` | corrected logger; records `None` and a provenance tag rather than imputing |
| `DigitalTwinsV2/pipeline/feature_extractor.py` | corrected extractor; refuses unmeasured columns, never reads quarantined fields |
| `DigitalTwinsV2/pipeline/openweights_models.py` | provider presets and a one-call-per-model logprob probe |
| `DigitalTwinsV2/scripts/rerun_sweep.py` | `selftest` / `capture` / `verify` |
| `DigitalTwinsV2/analysis/` | audit of the original corpus, analysis of the re-collected one, post-hoc confidence elicitation |
| `DigitalTwinsV2/patches/` | the change to the inference client that makes logprobs observable |
| `DigitalTwinsV2/backup_original/` | pre-change copies of every file modified |
| `DigitalTwinsV2/AUDIT.md` | the findings in full, with file and line references |

## Data

| corpus | n | role |
|---|---|---|
| `results/telemetry_runs` | 167 | the audited corpus; every feature in it is an artifact |
| `results/v2_clean` | 225 | re-collected with corrected instrumentation, 75 scenarios x 3 open-weights backbones |
| `results/v2_together_CREDIT_EXHAUSTED` | 225 | retained as evidence: collected against an exhausted account, which returns empty generations rather than errors, so 200 runs recorded as ordinary failures |

The third is included deliberately. It is the corpus on which the corrected
pipeline still produced AUROC above 0.9 — spuriously, because the trajectories
that carried features were exactly the ones that succeeded. It is the reason two
of the gate's checks exist.

## Reproducing a collection

Requires an OpenAI-compatible provider that returns logprobs. Managed gateways
generally do not on open-weights models; see the paper. Put the key in `.env`
as `TOGETHER_API_KEY` (or the equivalent for another provider), then:

```bash
python DigitalTwinsV2/pipeline/openweights_models.py together        # probe first
python DigitalTwinsV2/scripts/rerun_sweep.py capture --provider together \
  --framework reactxen --scenarios $(cat DigitalTwinsV2/v1_scenarios.txt) \
  --out results/my_corpus
```

`capture` runs a preflight generation per backbone and aborts without writing
anything if a provider answers empty. Collection ends by invoking the gate.

## Notes for reviewers

The audit findings are demonstrations rather than estimates and carry no
statistical caveat: distributional identity with a fallback generator, a
constant column, a single-use action appearing 866 times. The empirical section
is illustrative — 45 trajectories per cell, three backbones, one provider — and
is labelled as such in the paper.

The original manuscript that this work audits is preserved unmodified under
`IEEE_SWC_2026/`, except for three citation corrections documented in
`DigitalTwinsV2/README.md`.
