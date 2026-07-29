import numpy as np
from typing import List, Dict, Any, Tuple

def apply_abstention(calibrated_probs: np.ndarray, y_true: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Applies the selective abstention policy based on a probability threshold theta.
    
    Returns:
    - selective_accuracy: Pass@1 accuracy on accepted tasks.
    - abstention_rate: Percent of tasks rejected (abstention rate).
    """
    if len(calibrated_probs) == 0:
        return 0.0, 0.0
        
    # Decision: Execute action if probability >= threshold, otherwise Abstain
    execute = calibrated_probs >= threshold
    abstain = ~execute
    
    num_total = len(calibrated_probs)
    num_abstain = int(np.sum(abstain))
    num_execute = int(np.sum(execute))
    
    abstention_rate = num_abstain / num_total
    
    if num_execute > 0:
        selective_accuracy = np.mean(y_true[execute])
    else:
        selective_accuracy = 1.0  # Perfect precision/no errors if we always abstain
        
    return float(selective_accuracy), float(abstention_rate)

def generate_risk_coverage_curve(calibrated_probs: np.ndarray, y_true: np.ndarray, 
                                  num_steps: int = 100) -> Tuple[List[float], List[float], List[float]]:
    """Generates selective accuracy (coverage) and risk curves for threshold sweep.
    
    Returns:
    - thresholds: List of threshold values swept.
    - selective_accuracies: Accuracy at each threshold.
    - coverages: Coverage (1.0 - abstention_rate) at each threshold.
    """
    thresholds = np.linspace(0.0, 1.0, num_steps)
    accuracies = []
    coverages = []
    
    for t in thresholds:
        acc, abs_rate = apply_abstention(calibrated_probs, y_true, t)
        accuracies.append(acc)
        coverages.append(1.0 - abs_rate)
        
    return thresholds.tolist(), accuracies, coverages
