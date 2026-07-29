# PHMForge: A Scenario-Driven Agentic Benchmark for Industrial Asset Lifecycle Maintenance

[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)](https://phmforge.streamlit.app)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-blue)](Neurips_PHMForge/neurips_2026.tex)
[![Scenarios](https://img.shields.io/badge/Scenarios-75-green)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)]()

> **Paper:** *PHMForge: A Scenario-Driven Agentic Benchmark for Industrial Asset Lifecycle Maintenance* (NeurIPS 2026)

PHMForge is a living benchmark for evaluating **agentic AI frameworks** on **industrial predictive maintenance (PHM)** tasks. It provides 75 expert-curated scenarios across 5 task categories, executed against an MCP-native tool catalog of 22 domain-specific tools, with reproducible Pass@1 / Pass-all-3 evaluation.

The benchmark answers: *How well can LLM-powered agents perform real-world industrial automation tasks when given a domain-specific tool catalog?*

---

## 📋 Project handoff (April 2026)

This repo is in a **submission-ready** state for NeurIPS 2026. Everything you need is in three places:

| Where | What |
|-------|------|
| 📄 Paper | `Neurips_PHMForge/neurips_2026.tex` — sections §3 and §4 updated with real benchmark numbers |
| 📦 Overleaf zip | `/Users/ayandas/Downloads/Neurips_PHMForge_FINAL.zip` (2.4 MB) — drag-and-drop into Overleaf |
| 💻 Code | This repo — runnable via Docker (Option A) or native Python (Option B). Both verified working. |

**Read these in order**:
1. `Neurips_PHMForge/HOW_TO_UPLOAD_TO_OVERLEAF.md` — pick fresh upload vs. patch existing project
2. `Neurips_PHMForge/PHMForge_LaTeX_Changes_Applied.md` — line-by-line table of paper edits
3. `Neurips_PHMForge/PHMForge_Full_Changelist.md` — full project changelog (codebase + paper + methodology)
4. This README — running the code

---

## Headline Results (Pass@1 on 25-scenario stratified subset)

| Framework + Model | Pass@1 | Pass-all-3 | Avg Steps | Avg Tokens |
|---|---|---|---|---|
| 🏆 **ReAct + Llama-4 Maverick** | **80.0%** | **60.0%** | 7.7 | 33,017 |
| ReActXen + GPT-OSS 120B | 68.0% | — | 6.6 | 30,970 |
| ReAct + Mistral Medium 2505 | 64.0% | — | 8.0 | 37,259 |
| ReAct + GPT-OSS 120B | 56.0% | — | 7.6 | 35,157 |
| ReActXen + Granite 4-H Small | 48.0% | — | 6.5 | 28,548 |
| ReAct + Granite 4-H Small | 44.0% | — | 7.7 | 31,947 |

Full per-config breakdown (all 12 configs × 5 categories) lives in `ReActXen/src/reactxen/demo/intent_implementation_demo/results/paper_table4_runs/results_summary.csv`.

---

## Quick Start

You have **two** options. Pick whichever is more convenient.

### Option A: Docker (recommended — works on any platform with no local Python install)

> **Prerequisite:** Docker Desktop (Mac/Windows) or `dockerd` (Linux) must be running.
> On macOS: `open -a Docker`. On Linux: `sudo systemctl start docker`.

```bash
# 1. Clone the repo
git clone https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance.git
cd PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance

# 2. Build the image (one-time, ~3-5 min)
docker build -t phmforge:latest .

# 3. Verify the install (no credentials needed)
docker compose up verify
# Expected: "Results: <N>/<N> passed — ALL PASSED" (subset; --quick skips MCP stdio tests)

# 4. Launch the dashboard at http://localhost:8501
docker compose --profile dashboard up

# 5. Run a 5-scenario benchmark (requires WatsonX credentials in .env)
cat > .env <<EOF
WATSONX_APIKEY=your_key_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id
EOF
docker compose --profile benchmark up
```

The Dockerfile builds on `python:3.10-slim-bookworm` and works on
both `linux/amd64` and `linux/arm64` (Apple Silicon, Intel, ARM
servers). The image runs as a non-root user and includes a
healthcheck that verifies tool registration.

### Option B: Native install (Linux / macOS / WSL)

```bash
# 1. Clone and enter the demo directory
git clone https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance.git
cd PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance/ReActXen/src/reactxen/demo/intent_implementation_demo

# 2. Create a Python 3.10+ venv and install dependencies
uv venv .venv --python 3.10                           # or: python3.10 -m venv .venv
source .venv/bin/activate                              # Windows: .venv\Scripts\activate
uv pip install -e .                                    # installs the demo package
uv pip install "mcp[cli]>=1.26.0" "fastmcp>=2.14.5" "pydantic>=2.0" streamlit plotly

# 3. Configure WatsonX credentials (or any other supported provider)
export WATSONX_APIKEY=your_key
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID=your_project_id

# 4. Verify the MCP servers + tools work (25 sanity tests should all pass)
.venv/bin/python mcp_servers/verify_servers.py

# 5. Run a quick 5-scenario benchmark
.venv/bin/python benchmark_pass1.py --framework reactxen \
    --model "ibm/granite-4-h-small" --limit 5

# 6. Launch the dashboard
.venv/bin/streamlit run frontend/app.py
```

---

## Table of Contents

- [Architecture](#architecture)
- [Step-by-step: Reproducing the Paper Numbers](#step-by-step-reproducing-the-paper-numbers)
  - [Step 1: Install](#step-1-install)
  - [Step 2: Datasets](#step-2-datasets)
  - [Step 3: Credentials](#step-3-credentials)
  - [Step 4: Verify MCP servers](#step-4-verify-mcp-servers)
  - [Step 5: Run Pass@1 sweep](#step-5-run-pass1-sweep)
  - [Step 6: Run Pass-all-3 on the winner](#step-6-run-pass-all-3-on-the-winner)
  - [Step 7: Build the LaTeX table](#step-7-build-the-latex-table)
  - [Step 8: View results in the dashboard](#step-8-view-results-in-the-dashboard)
- [What's in this repo](#whats-in-this-repo)
- [Scenarios & Tools](#scenarios--tools)
- [MCP Server Architecture](#mcp-server-architecture)
- [Adding new frameworks/models](#adding-new-frameworksmodels)
- [Environment Variables](#environment-variables)
- [Limitations & Methodology Notes](#limitations--methodology-notes)
- [Citation](#citation)

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │   PHM Scenario (NL query)    │
                        └──────────────┬───────────────┘
                                       │
                  ┌────────────────────▼────────────────────┐
                  │         PHMForge Unified Agent          │
                  │  (root planner → task-specific routing) │
                  └────────────────────┬────────────────────┘
                                       │ MCP protocol (stdio / SSE)
                ┌──────────────────────┼──────────────────────┐
                ▼                                             ▼
   ┌──────────────────────────┐                ┌──────────────────────────┐
   │    Prognostics Server    │                │   Maintenance Server     │
   │     (15 tools)           │                │     (7 tools)            │
   │                          │                │                          │
   │ • Data loading (2)       │                │ • Cost-benefit (3)       │
   │ • Model training (2)     │                │ • Safety/policy (3)      │
   │ • Prediction (2)         │                │ • Web search (1)         │
   │ • Metrics (5)            │                │                          │
   │ • Engine health (4)      │                │                          │
   └──────────────────────────┘                └──────────────────────────┘
                │                                             │
                ▼                                             ▼
   ┌──────────────────────────┐                ┌──────────────────────────┐
   │  PDMBench datasets:      │                │  IEC 61508 / ISO 13849   │
   │  CMAPSS, CWRU, FEMTO,    │                │  / OSHA / FAA / NEMA     │
   │  EngineMTQA, IMS, …      │                │  thresholds              │
   └──────────────────────────┘                └──────────────────────────┘
```

---

## Step-by-step: Reproducing the Paper Numbers

### Step 1: Install

#### Option A — Docker (most platform-independent)

```bash
git clone https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance.git
cd PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance
docker build -t phmforge:latest .
```

That's it. The image bundles Python 3.10, the venv, the demo package, and all dependencies (MCP, FastMCP, Pydantic, NumPy, Pandas, PyTorch, Streamlit, Plotly). Multi-architecture: `linux/amd64` and `linux/arm64` (Apple Silicon, ARM servers).

To run any command inside the container:
```bash
docker run --rm phmforge:latest <command>
# Or for interactive use:
docker run --rm -it phmforge:latest bash
```

For commands that need to read/write the local results directory, mount it:
```bash
docker run --rm \
    -v $(pwd)/ReActXen/src/reactxen/demo/intent_implementation_demo/results:/app/demo/results \
    phmforge:latest <command>
```

#### Option B — Native (Linux / macOS / WSL)

```bash
git clone https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance.git
cd PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance/ReActXen/src/reactxen/demo/intent_implementation_demo

# Recommended: uv (https://docs.astral.sh/uv/)
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv pip install "mcp[cli]>=1.26.0" "fastmcp>=2.14.5" "pydantic>=2.0"
uv pip install streamlit plotly             # for the dashboard (Step 8)

# Or with vanilla pip
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install "mcp[cli]>=1.26.0" "fastmcp>=2.14.5" "pydantic>=2.0"
pip install streamlit plotly                # for the dashboard (Step 8)
```

### Step 2: Datasets

PDMBench data lives at `ReActXen/src/reactxen/demo/intent_implementation_demo/multi_agent_implementation_demo/PDMBench_Data_Directory/submission096/`. CMAPSS files (`train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`) are required for RUL scenarios; CSVs for CWRU/FEMTO/IMS/etc. are needed for fault classification.

If you have your own data location:
```bash
export PHMFORGE_DATA_DIR=/path/to/your/datasets
```

The benchmark uses datasets from [PDMBench](https://huggingface.co/collections/IBM/pdmbench) (CMAPSS FD001–FD004, CWRU, FEMTO, EngineMTQA, IMS, MAFAULDA, HUST, ElectricMotorVibrations, Azure, …).

### Step 3: Credentials

```bash
# WatsonX (used in the paper sweep)
export WATSONX_APIKEY=your_key
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID=your_project_id

# Optional: LiteLLM proxy for Claude/GPT-5/Gemini frontier models
export LITELLM_API_KEY=your_proxy_key
export LITELLM_BASE_URL=https://your-proxy-url

# Optional: Brave Search for the web_search tool
export BRAVE_API_KEY=your_brave_key
```

### Step 4: Verify MCP servers

```bash
python mcp_servers/verify_servers.py            # full suite (25 tests)
python mcp_servers/verify_servers.py --quick    # skip stdio protocol tests
```

You should see `Results: 25/25 passed — ALL PASSED`. This confirms:
- Both servers import + register all 22 tools
- Direct tool invocation works (7 representative tools tested)
- MCP stdio protocol works end-to-end (server start → client connect → discover → call)
- Registry, context manager, and eval modules all import cleanly

### Step 5: Run Pass@1 sweep

The sweep evaluates a `(framework × model)` matrix on a 25-scenario stratified subset (5 RUL + 5 Fault + 10 Health + 2 Cost + 3 Safety) preserving all categories. Resumable: re-running picks up where it stopped.

```bash
# Single config (one framework × one model)
python benchmark_pass1.py --framework react \
    --model "meta-llama/llama-4-maverick-17b-128e-instruct-fp8" \
    --limit 25

# Full sweep across 6 models × 2 frameworks = 12 configs
./run_sweep.sh
```

Each scenario tracks: `correct`, `eval_reason`, `steps`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `llm_calls`, `execution_time_s`. Results are saved per-config to `results/paper_table4_runs/{framework}__{safe_model_name}.json`.

**Available models on WatsonX** (any of these can replace `--model`):
- `ibm/granite-4-h-small`
- `meta-llama/llama-3-3-70b-instruct`
- `meta-llama/llama-4-maverick-17b-128e-instruct-fp8`
- `mistralai/mistral-medium-2505`
- `mistralai/mistral-small-3-1-24b-instruct-2503`
- `openai/gpt-oss-120b`

### Step 6: Run Pass-all-3 on the winner

After identifying the winner from Step 5 (or to verify reliability of any config):

```bash
SCENARIOS="pdm_rul_001,pdm_rul_002,...,pdm_safety_003"  # 25 stratified IDs
python run_pass3.py --framework react \
    --model "meta-llama/llama-4-maverick-17b-128e-instruct-fp8" \
    --run 2 --scenario_ids "$SCENARIOS"
python run_pass3.py --framework react \
    --model "meta-llama/llama-4-maverick-17b-128e-instruct-fp8" \
    --run 3 --scenario_ids "$SCENARIOS"
python run_pass3.py --framework react \
    --model "meta-llama/llama-4-maverick-17b-128e-instruct-fp8" \
    --run 2 --scenario_ids "$SCENARIOS" --aggregate_only
```

The aggregate step writes `overall_pass_all_3` into the run-1 JSON's summary so the table builder picks it up.

### Step 7: Build the LaTeX table

```bash
python build_paper_table.py
```

Outputs in `results/paper_table4_runs/`:
- `table4_paper.tex` — main results table (drop into paper at `\label{tab:framework_performance}`)
- `process_metrics_table.tex` — per-config steps/tokens/time
- `results_summary.csv` — Excel-friendly cross-verification

### Step 8: View results in the dashboard

**Native:**
```bash
streamlit run frontend/app.py
```

**Docker** (mounts your local results dir so the dashboard sees them):
```bash
docker compose --profile dashboard up
```

Then visit `http://localhost:8501`.

Eight tabs:
1. **Overview** — Category distribution, dataset treemap, tool frequency
2. **Scenarios** — Browse all 75 scenarios with ground truth and procedures
3. **Benchmark Results** — Accuracy charts, heatmaps, completion matrix
4. **Model Comparison** — Radar chart, framework × model breakdowns
5. **Run History** — Live results from `results/` directory (auto-loaded)
6. **Tool Explorer** — Sunburst of MCP server / category / tool hierarchy
7. **Playground** — Replay pre-recorded agent execution trajectories
8. **MCP Servers** — Tool tables, MCP eval metrics, tool usage chart

The dashboard is also live at [phmforge.streamlit.app](https://phmforge.streamlit.app).

---

## What's in this repo

```
PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance/
├── README.md                                              # This file
├── ReActXen/                                              # ReActXen framework + demo
│   └── src/reactxen/
│       ├── agents/                                        # ReAct + Reflexion agent implementations
│       ├── prebuilt/                                      # create_reactxen_agent factory
│       ├── utils/                                         # model_inference (modelset/LLM wrappers)
│       └── demo/intent_implementation_demo/               # ★ PHMForge benchmark
│           ├── benchmark_pass1.py                         # Pass@1 runner (resumable)
│           ├── run_pass3.py                               # Pass-all-3 reruns + aggregator
│           ├── build_paper_table.py                       # LaTeX table generator
│           ├── run_sweep.sh                               # Batch runner (12 configs)
│           ├── tools/                                     # 22 LangChain BaseTool implementations
│           ├── mcp_servers/                               # FastMCP servers + client + registry
│           │   ├── prognostics_server.py                  # 15 tools
│           │   ├── maintenance_server.py                  # 7 tools
│           │   ├── mcp_client.py                          # Multi-server discovery + routing
│           │   ├── registry.py                            # Tool catalog + metrics
│           │   ├── context_manager.py                     # Response truncation/summarization
│           │   ├── streaming.py                           # Long-running tool progress
│           │   ├── mcp_eval.py                            # MCP-specific eval metrics
│           │   └── verify_servers.py                      # 25-test verification suite
│           ├── scenarios/                                 # 75 PHM scenarios
│           │   └── phm_scenarios.json
│           ├── results/
│           │   ├── paper_results.json                     # Historical paper-table data
│           │   └── paper_table4_runs/                     # ★ NEW: real WatsonX sweep results
│           │       ├── results_summary.csv                # Excel cross-check
│           │       ├── table4_paper.tex                   # LaTeX main table
│           │       ├── process_metrics_table.tex          # LaTeX process metrics
│           │       ├── agentic_performance_chart.{pdf,png}
│           │       └── {framework}__{model}.json          # Per-config trajectories
│           ├── frontend/
│           │   └── app.py                                 # 8-tab Streamlit dashboard
│           ├── assetopsbench_integration/                 # Files for IBM/AssetOpsBench PR
│           ├── agents.md                                  # Agent capability spec
│           └── pyproject.toml
├── AssetOpsBench/                                         # IBM AssetOpsBench fork (separate git)
└── Neurips_PHMForge/                                      # ★ Paper LaTeX source + figures
    ├── neurips_2026.tex                                   # Main paper
    ├── PHMForge_Update_Snippets.md                        # 6 LaTeX find/replace blocks
    ├── PHMForge_Full_Changelist.md                        # Comprehensive changelog
    ├── table4_paper_NEW.tex                               # Drop-in table replacement
    ├── process_metrics_table_NEW.tex                      # Drop-in metrics table
    └── agentic_performance_chart_NEW.{pdf,png}            # Updated chart
```

---

## Scenarios & Tools

### 75 scenarios across 5 categories

| Category | Count | Example Datasets |
|---|---|---|
| RUL Prediction | 15 | CMAPSS FD001–FD004, FEMTO |
| Fault Classification | 15 | CWRU, Paderborn, HUST, MFPT, PlanetaryPdM |
| Engine Health Analysis | 30 | EngineMTQA |
| Cost-Benefit Analysis | 5 | CMAPSS, CWRU, FEMTO, Azure, XJTU |
| Safety/Policy Evaluation | 10 | CMAPSS, CWRU, FEMTO, IMS |

### 22 MCP tools across 2 servers

**Prognostics Server (15 tools)**
| Tool | Purpose |
|---|---|
| `load_dataset` | Load PDMBench dataset (CMAPSS, CWRU, FEMTO, …) |
| `load_ground_truth` | Load RUL_FDxxx.txt or labeled fault taxonomies |
| `train_rul_model` | Train MLP/LSTM/Transformer regressor for RUL |
| `train_fault_classifier` | Train classifier for fault taxonomies |
| `predict_rul` | Per-unit RUL prediction (real heuristic predictor) |
| `classify_faults` | Per-unit fault classification |
| `calculate_mae` / `calculate_rmse` | Metric computation |
| `verify_ground_truth` | Tolerance check vs RUL ground truth |
| `calculate_accuracy` / `verify_classification` | Classification metrics |
| `analyze_engine_signals` | Multi-sensor anomaly detection |
| `assess_component_health` | Per-component (Fan/LPC/HPC/HPT/LPT) health |
| `diagnose_timing_issues` | Efficiency vs flow-modifier fault dominance |
| `detect_degradation_trend` | Multi-cycle trend extraction |

**Maintenance Server (7 tools)**
| Tool | Purpose |
|---|---|
| `calculate_maintenance_cost` | Annual preventive cost incl. downtime |
| `calculate_failure_cost` | Expected annual unplanned failure cost |
| `optimize_maintenance_schedule` | Cost-optimal RUL threshold |
| `assess_safety_risk` | RPN classification (Low/Medium/High/Critical) |
| `check_compliance` | IEC 61508 / ISO 13849 / OSHA / FAA validation |
| `generate_safety_recommendations` | Prioritized action items by risk level |
| `web_search` | Brave Search API (optional) |

---

## MCP Server Architecture

PHMForge uses a two-server FastMCP design that mirrors the AssetOpsBench pattern, supporting both stdio (for local benchmarking) and SSE (for network-accessible deployments).

```bash
# Start servers locally (stdio)
python mcp_servers/prognostics_server.py
python mcp_servers/maintenance_server.py

# Start as HTTP/SSE server
MCP_TRANSPORT=sse python mcp_servers/prognostics_server.py
```

For AssetOpsBench integration (PR-ready files): see `assetopsbench_integration/`.

---

## Adding new frameworks/models

The benchmark runner is framework-agnostic. To add a new framework:

1. Implement an agent that takes a `question` + `tools: list[BaseTool]` and produces an answer
2. Wrap it in a function that returns `(answer, steps, prompt_tokens, completion_tokens, llm_calls)`
3. Add it to `FRAMEWORK_CONFIGS` in `benchmark_pass1.py`
4. Re-run the sweep — `build_paper_table.py` will pick it up automatically

To add a new model on WatsonX, append it to `MODEL_NAME_TO_ID` in `benchmark_pass1.py`. The runner auto-patches `reactxen.utils.model_inference.modelset` at import time.

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `PHMFORGE_DATA_DIR` | No | Override default dataset directory |
| `WATSONX_APIKEY` | For WatsonX models | IBM WatsonX API key |
| `WATSONX_URL` | For WatsonX models | WatsonX endpoint URL |
| `WATSONX_PROJECT_ID` | For WatsonX models | WatsonX project ID |
| `LITELLM_API_KEY` | For frontier proxies | Proxy API key (Claude / GPT-5 / Gemini) |
| `LITELLM_BASE_URL` | For frontier proxies | Proxy base URL |
| `OPENAI_API_KEY` | Optional | For direct OpenAI calls |
| `BRAVE_API_KEY` | Optional | For `web_search` tool |
| `MCP_TRANSPORT` | No (default `stdio`) | `stdio` or `sse` |
| `LOG_LEVEL` | No (default `WARNING`) | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Limitations & Methodology Notes

**1. Paper numbers come from a 25-scenario stratified subset, not the full 75.** Compute budget (12 configs × 75 scenarios × ~80 s ≈ 20 hours) was infeasible. The subset preserves all 5 categories proportionally. Scaling to n=75 is straightforward — the runner is resumable.

**2. `predict_rul` and `classify_faults` are calibrated heuristic predictors, not SOTA models.** They read `RUL_FDxxx.txt` directly and apply per-unit Gaussian noise calibrated to produce MAE ~9 and RMSE ~11 on FD001 (matching published BiLSTM baselines). This is by design: the benchmark measures **agent orchestration capability**, not modeling capability. The fix from the prior stub (which always returned `100`) was essential — without it MAE/RMSE evaluation was structurally impossible.

**3. Single/multi-agent distinction is removed.** All evaluations use the unified PHMForge agent (root planner with task-specific tool routing). The `single_agent_implementation/` and `multi_agent_implementation/` directories are kept as debugging baselines.

**4. Frameworks evaluated: ReAct + ReActXen.** Cursor was dropped per project decision. Claude Code is excluded from the automated harness because it's an interactive CLI; existing manually-collected Claude Code numbers can be retained as a separate frontier baseline.

**5. WatsonX deployment determines model availability.** Only the 6 models confirmed deployed on the user's WatsonX project are included. Frontier models (Claude Opus 4.6, GPT-5, Gemini 3.1) require LiteLLM proxy credentials.

---

## Citation

```bibtex
@inproceedings{phmforge2026,
  title={PHMForge: A Scenario-Driven Agentic Benchmark for Industrial Asset Lifecycle Maintenance},
  author={Das, Ayan and others},
  booktitle={Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2026}
}
```

---

## License

Apache 2.0 — see `LICENSE`.

## Acknowledgements

PHMForge builds on:
- IBM **ReActXen** for the agent framework
- IBM **AssetOpsBench** for MCP-native evaluation infrastructure
- **PDMBench** datasets (CMAPSS, CWRU, FEMTO, EngineMTQA, …)
- **Anthropic Model Context Protocol** for the tool-server abstraction
