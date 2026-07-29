#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=== STARTING FULL EXPERIMENTAL PIPELINE ===")
    
    # 1. Run the live simulation sweep for the new light models
    run_live_script = _PROJECT_ROOT / "scripts" / "04_run_live_new_models.py"
    print(f"\n1. Executing live model evaluations: {run_live_script.name}")
    res1 = subprocess.run([sys.executable, str(run_live_script)], cwd=str(_PROJECT_ROOT))
    if res1.returncode != 0:
        print(f"Error: Live model evaluation failed with exit code {res1.returncode}.")
        sys.exit(res1.returncode)
        
    # 2. Run the analytics and paper artifact generator
    artifacts_script = _PROJECT_ROOT / "scripts" / "03_generate_paper_artifacts.py"
    print(f"\n2. Re-generating paper tables and figures: {artifacts_script.name}")
    res2 = subprocess.run([sys.executable, str(artifacts_script)], cwd=str(_PROJECT_ROOT))
    if res2.returncode != 0:
        print(f"Error: Artifact generation failed with exit code {res2.returncode}.")
        sys.exit(res2.returncode)
        
    print("\n=== PIPELINE RUN COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
