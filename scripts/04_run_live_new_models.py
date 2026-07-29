#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=== Launching Live Model Evaluations ===")
    
    # API keys should be set in the environment or via .env file
    # e.g., export OPENAI_API_KEY="..."
    pass
    
    # Explicit list of the 25 scenarios evaluated in the baseline
    scenario_list = (
        "engine_health_combined_001,engine_health_combined_004,engine_health_combined_007,"
        "engine_health_combined_010,engine_health_combined_013,engine_health_combined_016,"
        "engine_health_combined_019,engine_health_combined_022,engine_health_combined_025,"
        "engine_health_combined_028,pdm_cost_benefit_001,pdm_cost_benefit_002,pdm_fault_001,"
        "pdm_fault_002,pdm_fault_003,pdm_fault_004,pdm_fault_005,pdm_rul_001,pdm_rul_002,"
        "pdm_rul_003,pdm_rul_004,pdm_rul_005,pdm_safety_001,pdm_safety_002,pdm_safety_003"
    )
    
    models_to_run = [
        "openai/gpt-4o-mini",
        "gemini/gemini-1.5-flash"
    ]
    
    runner_path = _PROJECT_ROOT / "phmforge_runner.py"
    
    for model in models_to_run:
        print(f"\n--- Running evaluation for {model} (Target 25 Scenarios) ---")
        cmd = [
            sys.executable,
            str(runner_path),
            "--model", model,
            "--framework", "reactxen",
            "--scenario_ids", scenario_list,
            "--no_resume" # Rerun to ensure fresh and clean files
        ]
        
        # Run subprocess and stream output
        result = subprocess.run(cmd, env=os.environ, cwd=str(_PROJECT_ROOT))
        if result.returncode == 0:
            print(f"Evaluation for {model} completed successfully.")
        else:
            print(f"Error: Evaluation for {model} failed with exit code {result.returncode}.")

if __name__ == "__main__":
    main()
