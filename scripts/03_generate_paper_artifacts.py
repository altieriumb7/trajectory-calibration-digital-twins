#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TELEMETRY_DIR = _PROJECT_ROOT / "results" / "telemetry_runs"
_PAPER_DIR = _PROJECT_ROOT / "paper"
_TABLES_DIR = _PAPER_DIR / "tables"
_FIGURES_DIR = _PAPER_DIR / "figures"

_TABLES_DIR.mkdir(parents=True, exist_ok=True)
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Import our custom calibration code
sys_path = str(_PROJECT_ROOT)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from phmforge_calibration.feature_extractor import FeatureExtractor
from phmforge_calibration.trajectory_calibrator import TrajectoryCalibrator
from phmforge_calibration.abstention_policy import generate_risk_coverage_curve, apply_abstention

def calculate_ece(y_true, y_prob, n_bins=5):
    """Calculates the Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    for m in range(n_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        # Handle boundaries
        if m < n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            
        prop_in_bin = np.sum(in_bin) / n_samples
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def safe_roc_auc(y_true, y_prob):
    """Calculates AUROC, returning 0.5 if target has only one class."""
    if len(np.unique(y_true)) <= 1:
        return 0.5
    return roc_auc_score(y_true, y_prob)

def main():
    print("=== Generating Paper Artifacts (Analytics, Table, and Figures) ===")
    
    extractor = FeatureExtractor()
    
    models = [
        "ibm/granite-4-h-small",
        "meta-llama/llama-3-3-70b-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "mistralai/mistral-medium-2505",
        "mistralai/mistral-small-3-1-24b-instruct-2503",
        "openai/gpt-oss-120b",
        "openai/gpt-4o-mini",
        "gemini/gemini-1.5-flash"
    ]
    
    # Store ECE and AUROC results
    calibration_results = {}
    
    # Aggregate data for global analysis
    all_X = []
    all_y = []
    
    for model_name in models:
        safe_name = model_name.replace("/", "_").replace(":", "_")
        model_dir = _TELEMETRY_DIR / safe_name
        
        X, y, ids = extractor.extract_dataset(model_dir)
        if len(X) == 0:
            print(f"Skipping {model_name} (no dataset found).")
            continue
            
        all_X.append(X)
        all_y.append(y)
        
        # 5-Fold Cross Validation Setup
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Arrays to accumulate validation fold predictions
        y_val_true = []
        y_val_raw = []
        y_val_isotonic = []
        y_val_calvert = []
        y_val_htc = []
        y_val_boosting = []
        y_val_elasticnet = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit calibrators
            calibrator_iso = TrajectoryCalibrator(method="VerbalizedIsotonic").fit(X_train, y_train)
            calibrator_cal = TrajectoryCalibrator(method="CalVerT").fit(X_train, y_train)
            calibrator_htc = TrajectoryCalibrator(method="HTC").fit(X_train, y_train)
            calibrator_boost = TrajectoryCalibrator(method="HTC-Boosting").fit(X_train, y_train)
            calibrator_enet = TrajectoryCalibrator(method="HTC-ElasticNet").fit(X_train, y_train)
            
            # Predict validation
            y_val_true.extend(y_val)
            y_val_raw.extend(X_val[:, 4])
            y_val_isotonic.extend(calibrator_iso.predict_probability(X_val))
            y_val_calvert.extend(calibrator_cal.predict_probability(X_val))
            y_val_htc.extend(calibrator_htc.predict_probability(X_val))
            y_val_boosting.extend(calibrator_boost.predict_probability(X_val))
            y_val_elasticnet.extend(calibrator_enet.predict_probability(X_val))
            
        y_val_true = np.array(y_val_true)
        y_val_raw = np.array(y_val_raw)
        y_val_isotonic = np.array(y_val_isotonic)
        y_val_calvert = np.array(y_val_calvert)
        y_val_htc = np.array(y_val_htc)
        y_val_boosting = np.array(y_val_boosting)
        y_val_elasticnet = np.array(y_val_elasticnet)
        
        # Compute ECE and AUROC
        calibration_results[model_name] = {
            "raw": {"ece": calculate_ece(y_val_true, y_val_raw), "auroc": safe_roc_auc(y_val_true, y_val_raw)},
            "isotonic": {"ece": calculate_ece(y_val_true, y_val_isotonic), "auroc": safe_roc_auc(y_val_true, y_val_isotonic)},
            "calvert": {"ece": calculate_ece(y_val_true, y_val_calvert), "auroc": safe_roc_auc(y_val_true, y_val_calvert)},
            "htc": {"ece": calculate_ece(y_val_true, y_val_htc), "auroc": safe_roc_auc(y_val_true, y_val_htc)},
            "boosting": {"ece": calculate_ece(y_val_true, y_val_boosting), "auroc": safe_roc_auc(y_val_true, y_val_boosting)},
            "elasticnet": {"ece": calculate_ece(y_val_true, y_val_elasticnet), "auroc": safe_roc_auc(y_val_true, y_val_elasticnet)},
            "y_true": y_val_true,
            "y_prob_htc": y_val_htc
        }
        
    # === 1. GENERATE LATEX RESULTS TABLE ===
    table_path = _TABLES_DIR / "table_calibration_results.tex"
    print(f"Generating LaTeX Table at: {table_path}")
    
    with open(table_path, "w") as f:
        f.write("% Auto-generated by scripts/03_generate_paper_artifacts.py\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Expected Calibration Error (ECE $\\downarrow$) and AUROC ($\\uparrow$) comparison across calibration models.}\n")
        f.write("\\label{tab:calibration_results}\n")
        f.write("\\begin{tabular}{lccccccccccc}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{2}{c}{\\textbf{Raw}} & \\multicolumn{2}{c}{\\textbf{VerbalizedIsotonic}} & \\multicolumn{2}{c}{\\textbf{CalVerT}} & \\multicolumn{2}{c}{\\textbf{HTC (RF)}} & \\multicolumn{2}{c}{\\textbf{HTC (Boosting)}} & \\multicolumn{2}{c}{\\textbf{HTC (ElasticNet)}} \\\\\n")
        f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-13}\n")
        f.write("\\textbf{LLM Backbone} & ECE & AUROC & ECE & AUROC & ECE & AUROC & ECE & AUROC & ECE & AUROC & ECE & AUROC \\\\\n")
        f.write("\\midrule\n")
        
        for model in models:
            if model not in calibration_results:
                continue
            r = calibration_results[model]
            model_disp = model.split("/")[-1].replace("-instruct", "").replace("-fp8", "").replace("_", "\\_")
            
            # Find best ECE and best AUROC
            eces = [r['raw']['ece'], r['isotonic']['ece'], r['calvert']['ece'], r['htc']['ece'], r['boosting']['ece'], r['elasticnet']['ece']]
            aurocs = [r['raw']['auroc'], r['isotonic']['auroc'], r['calvert']['auroc'], r['htc']['auroc'], r['boosting']['auroc'], r['elasticnet']['auroc']]
            
            best_ece_idx = np.argmin(eces)
            best_auroc_idx = np.argmax(aurocs)
            
            def fmt(val, idx, best_idx):
                s = f"{val:.3f}"
                if idx == best_idx:
                    return f"\\textbf{{{s}}}"
                return s
                
            f.write(f"{model_disp} & {fmt(r['raw']['ece'], 0, best_ece_idx)} & {fmt(r['raw']['auroc'], 0, best_auroc_idx)} & "
                    f"{fmt(r['isotonic']['ece'], 1, best_ece_idx)} & {fmt(r['isotonic']['auroc'], 1, best_auroc_idx)} & "
                    f"{fmt(r['calvert']['ece'], 2, best_ece_idx)} & {fmt(r['calvert']['auroc'], 2, best_auroc_idx)} & "
                    f"{fmt(r['htc']['ece'], 3, best_ece_idx)} & {fmt(r['htc']['auroc'], 3, best_auroc_idx)} & "
                    f"{fmt(r['boosting']['ece'], 4, best_ece_idx)} & {fmt(r['boosting']['auroc'], 4, best_auroc_idx)} & "
                    f"{fmt(r['elasticnet']['ece'], 5, best_ece_idx)} & {fmt(r['elasticnet']['auroc'], 5, best_auroc_idx)} \\\\\n")
            
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
        
    # === 2. FIG 1: ECE COMPARISON (BAR CHART) ===
    plt.figure(figsize=(10, 5))
    model_labels = [m.split("/")[-1].replace("-instruct", "").replace("-fp8", "") for m in models if m in calibration_results]
    raw_eces = [calibration_results[m]["raw"]["ece"] for m in models if m in calibration_results]
    htc_eces = [calibration_results[m]["htc"]["ece"] for m in models if m in calibration_results]
    
    x = np.arange(len(model_labels))
    width = 0.35
    
    plt.bar(x - width/2, raw_eces, width, label='Raw (Uncalibrated)', color='#E74C3C')
    plt.bar(x + width/2, htc_eces, width, label='HTC Calibrated', color='#2ECC71')
    
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('Expected Calibration Error Before & After HTC Calibration')
    plt.xticks(x, model_labels, rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / "fig1_ece_comparison.pdf", dpi=300)
    plt.savefig(_FIGURES_DIR / "fig1_ece_comparison.png", dpi=300)
    plt.close()
    
    # === 3. FIG 2: RISK-COVERAGE CURVE ===
    plt.figure(figsize=(8, 6))
    
    # Let's plot risk-coverage curves for Granite and Llama-4 Maverick
    target_models = ["ibm/granite-4-h-small", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"]
    colors = ['#1E88E5', '#D81B60']
    
    for model_name, color in zip(target_models, colors):
        if model_name not in calibration_results:
            continue
        r = calibration_results[model_name]
        y_true = r["y_true"]
        y_prob = r["y_prob_htc"]
        
        thresholds, selective_accuracies, coverages = generate_risk_coverage_curve(y_prob, y_true)
        
        # Risk = 1.0 - selective_accuracy
        risks = [1.0 - acc for acc in selective_accuracies]
        
        model_disp = model_name.split("/")[-1].replace("-instruct", "").replace("-fp8", "")
        plt.plot(coverages, risks, label=f"{model_disp} (HTC)", color=color, linewidth=2)
        
    plt.xlabel('Coverage (Fraction of Decisions Executed)')
    plt.ylabel('Selective Failure Risk (1 - Selective Pass@1)')
    plt.title('Risk-Coverage Profile (Selective Abstention)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / "fig2_risk_coverage_curve.pdf", dpi=300)
    plt.savefig(_FIGURES_DIR / "fig2_risk_coverage_curve.png", dpi=300)
    plt.close()
    
    # === 4. FIG 3: FEATURE IMPORTANCE ===
    # Merge all datasets to fit a global Random Forest to compute feature importances
    flat_X = np.vstack(all_X)
    flat_y = np.concatenate(all_y)
    
    calibrator = TrajectoryCalibrator(method="HTC")
    calibrator.fit(flat_X, flat_y)
    
    importances = calibrator.htc_model.feature_importances_
    features = ["Early Entropy", "Confidence Gradient", "MCP Error Ratio", "Logprob Variance", "Verbalized Score", "Action Loop Ratio"]
    
    # Sort
    indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.xlabel('Mean Decrease in Impurity (Gini Importance)')
    plt.title('HTC Diagnostic Signal Feature Importance')
    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / "fig3_feature_importance.pdf", dpi=300)
    plt.savefig(_FIGURES_DIR / "fig3_feature_importance.png", dpi=300)
    plt.close()
    
    print("=== Paper Artifacts generated successfully ===")

if __name__ == "__main__":
    main()
