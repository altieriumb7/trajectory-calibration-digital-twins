#!/bin/bash
# Pass@1 sweep across all (framework, model) combos on 25-scenario stratified subset.
# Resumable: skips scenarios already in the per-config JSON.

set -u
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
export WATSONX_APIKEY=tQNfTcDRotQZ6rjdUDGoOmC0oq_xuAgWVKh1fCC0Gzzq
export WATSONX_API_KEY=tQNfTcDRotQZ6rjdUDGoOmC0oq_xuAgWVKh1fCC0Gzzq
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID="66834b50-6774-4319-bebb-9b987318bb53"

SCENARIOS="pdm_rul_001,pdm_rul_002,pdm_rul_003,pdm_rul_004,pdm_rul_005,pdm_fault_001,pdm_fault_002,pdm_fault_003,pdm_fault_004,pdm_fault_005,engine_health_combined_001,engine_health_combined_004,engine_health_combined_007,engine_health_combined_010,engine_health_combined_013,engine_health_combined_016,engine_health_combined_019,engine_health_combined_022,engine_health_combined_025,engine_health_combined_028,pdm_cost_benefit_001,pdm_cost_benefit_002,pdm_safety_001,pdm_safety_002,pdm_safety_003"

MODELS=(
  "ibm/granite-4-h-small"
  "mistralai/mistral-small-3-1-24b-instruct-2503"
  "openai/gpt-oss-120b"
  "mistralai/mistral-medium-2505"
  "meta-llama/llama-3-3-70b-instruct"
  "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
)

LOGDIR="results/paper_table4_runs/logs"
mkdir -p "$LOGDIR"

for framework in react reactxen; do
  for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Running $framework + $model"
    echo "=========================================="
    safe_name=$(echo "$model" | tr '/' '_' | tr ':' '_')
    log="$LOGDIR/${framework}__${safe_name}.log"
    .venv/bin/python benchmark_pass1.py \
      --framework "$framework" \
      --model "$model" \
      --scenario_ids "$SCENARIOS" \
      > "$log" 2>&1
    echo "Done. Tail of log:"
    tail -20 "$log"
    echo
  done
done
