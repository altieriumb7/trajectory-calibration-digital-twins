#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _PROJECT_ROOT / "paper" / "figures"
_TELEMETRY_DIR = _PROJECT_ROOT / "results" / "telemetry_runs"

# Add phmforge_calibration to path
sys.path.insert(0, str(_PROJECT_ROOT))
from phmforge_calibration.feature_extractor import FeatureExtractor
from phmforge_calibration.trajectory_calibrator import TrajectoryCalibrator

# --- V2 PROVENANCE GATE ----------------------------------------------------
# This script produced figures and tables for the original submission. Those
# artifacts are invalid: the token-level features were synthesised by the old
# telemetry logger, and the grading harness leaked its verdict into the
# observation text. Background: DigitalTwinsV2/AUDIT.md
#
# The gate below refuses to run on a corpus that cannot support the analysis.
# It is not a formality: on the current telemetry it exits non-zero.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from phmforge_calibration.provenance import validate as _validate

_report = _validate(_Path(__file__).resolve().parent.parent / "results" / "telemetry_runs")
if not _report.ok:
    print(_report.render())
    print("")
    print("ABORTED: this corpus cannot support a calibration study.")
    print("Recapture first:  python DigitalTwinsV2/scripts/rerun_sweep.py capture ...")
    print("Background:       DigitalTwinsV2/AUDIT.md")
    raise SystemExit(1)
# --- END V2 PROVENANCE GATE ------------------------------------------------

def main():
    print("=== Generating Figure 4: Expected Operational Cost (EOC) Reduction Chart ===")
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    extractor = FeatureExtractor()
    models = [
        "ibm/granite-4-h-small",
        "meta-llama/llama-3-3-70b-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "mistralai/mistral-medium-2505",
        "mistralai/mistral-small-3-1-24b-instruct-2503",
        "openai/gpt-oss-120b",
        "openai/gpt-4o-mini"
    ]
    
    # Cost parameters
    C_catastrophic = 50000  # Cost of unnoticed failure ($)
    C_review = 500          # Cost of human engineer review/abstention ($)
    
    model_names_disp = []
    raw_costs = []
    htc_costs = []
    
    for model in models:
        safe_name = model.replace("/", "_").replace(":", "_")
        model_dir = _TELEMETRY_DIR / safe_name
        
        X, y, ids = extractor.extract_dataset(model_dir)
        if len(X) == 0:
            continue
            
        # Fit HTC calibrator
        calibrator = TrajectoryCalibrator(method="HTC").fit(X, y)
        probs = calibrator.predict_probability(X)
        
        # 1. Raw Cost per scenario (no abstention, raw failure cost)
        # Failure occurs when y == 0
        raw_cost_per_scenario = np.mean((1 - y) * C_catastrophic)
        
        # 2. HTC Selective Abstention Cost (threshold theta = 0.75)
        theta = 0.75
        executed = (probs >= theta)
        
        # For executed runs: if y == 0, catastrophic failure cost
        # For abstained runs: human review cost
        cost_executed = np.sum((1 - y[executed]) * C_catastrophic) if np.sum(executed) > 0 else 0
        cost_abstained = np.sum(~executed) * C_review
        
        htc_cost_per_scenario = (cost_executed + cost_abstained) / len(y)
        
        disp_name = model.split("/")[-1].replace("-instruct", "").replace("-fp8", "")
        model_names_disp.append(disp_name)
        raw_costs.append(raw_cost_per_scenario)
        htc_costs.append(htc_cost_per_scenario)
        
    # Styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 5.5))
    
    x = np.arange(len(model_names_disp))
    width = 0.35
    
    rects1 = plt.bar(x - width/2, raw_costs, width, label='Raw Execution (No Abstention)', color='#d9534f', alpha=0.9)
    rects2 = plt.bar(x + width/2, htc_costs, width, label='HTC Selective Abstention (θ ≥ 0.75)', color='#5cb85c', alpha=0.9)
    
    plt.ylabel('Expected Operational Cost per Scenario ($)', fontsize=12, fontweight='bold')
    plt.title('Industrial Financial Risk Mitigation: Expected Operational Cost (EOC)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, model_names_disp, rotation=20, ha='right', fontsize=10, fontweight='bold')
    plt.legend(fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    plt.yscale('log')  # Log scale for clear financial contrast
    
    # Value annotations on top of bars
    for rect in rects1:
        height = rect.get_height()
        plt.annotate(f'${height:,.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#a94442')
                    
    for rect in rects2:
        height = rect.get_height()
        plt.annotate(f'${height:,.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#3c763d')
                    
    plt.tight_layout()
    
    png_path = _FIGURES_DIR / "fig4_eoc_cost_reduction.png"
    pdf_path = _FIGURES_DIR / "fig4_eoc_cost_reduction.pdf"
    
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    print(f"Figure 4 saved successfully at: {png_path}")

if __name__ == "__main__":
    main()
