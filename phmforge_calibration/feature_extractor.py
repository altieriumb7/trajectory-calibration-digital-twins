import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

def calculate_shannon_entropy(logprobs: List[float]) -> float:
    """Computes Shannon entropy (average negative log probability) of chosen tokens."""
    if not logprobs:
        return 0.0
    # Shannon entropy of chosen tokens = -1/N * sum(logprobs)
    return -np.mean(logprobs)

def calculate_linear_slope(y: List[float]) -> float:
    """Computes the linear slope of a sequence of values."""
    n = len(y)
    if n <= 1:
        return 0.0
    x = np.arange(n)
    # Fit line y = mx + c and return m (slope)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)

class FeatureExtractor:
    """Extracts trajectory features f(tau) from JSONL telemetry run records."""
    
    def extract_features_from_trajectory(self, filepath: Path) -> Tuple[Dict[str, Any], bool]:
        """Reads a JSONL file and computes the diagnostic feature vector f(tau).
        Returns a tuple of (feature_dict, task_success).
        """
        steps = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line))
        
        if not steps:
            # Empty file fallback
            return {
                "early_entropy": 0.0,
                "confidence_gradient": 0.0,
                "mcp_error_ratio": 0.0,
                "logprob_variance": 0.0,
                "verbalized_score": 0.5
            }, False
            
        # 1. Early Entropy (Shannon entropy of first 2 steps)
        early_logprobs = []
        for i in range(min(2, len(steps))):
            early_logprobs.extend(steps[i].get("thought_logprobs", []))
        early_entropy = calculate_shannon_entropy(early_logprobs)
        
        # 2. Confidence Gradient
        confidences = [step.get("verbalized_confidence", 0.5) for step in steps]
        confidence_gradient = calculate_linear_slope(confidences)
        
        # 3. MCP Error Ratio
        total_mcp_calls = 0
        failed_mcp_calls = 0
        for step in steps:
            for call in step.get("mcp_tool_calls", []):
                total_mcp_calls += 1
                if call.get("status_code", 200) != 200:
                    failed_mcp_calls += 1
        mcp_error_ratio = failed_mcp_calls / total_mcp_calls if total_mcp_calls > 0 else 0.0
        
        # 4. Logprob Variance along whole trajectory
        all_logprobs = []
        for step in steps:
            all_logprobs.extend(step.get("thought_logprobs", []))
        logprob_variance = float(np.var(all_logprobs)) if all_logprobs else 0.0
        
        # 5. Verbalized Score (final confidence)
        verbalized_score = steps[-1].get("verbalized_confidence", 0.5)
        
        # 6. Action Loop Ratio (repeated action inputs / total calls)
        actions = [step.get("action", "") for step in steps if step.get("action")]
        action_inputs = [step.get("action_input", "") for step in steps if step.get("action_input")]
        unique_calls = set(zip(actions, action_inputs))
        total_calls = len(actions)
        loop_ratio = (total_calls - len(unique_calls)) / total_calls if total_calls > 0 else 0.0
        
        # Ground Truth success
        task_success = steps[-1].get("task_success", False)
        
        features = {
            "early_entropy": early_entropy,
            "confidence_gradient": confidence_gradient,
            "mcp_error_ratio": mcp_error_ratio,
            "logprob_variance": logprob_variance,
            "verbalized_score": verbalized_score,
            "loop_ratio": loop_ratio
        }
        
        return features, task_success

    def extract_dataset(self, model_runs_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Scans a model's telemetry runs and builds feature matrix X and target y."""
        X_list = []
        y_list = []
        ids = []
        
        if not model_runs_dir.exists():
            return np.array([]), np.array([]), []
            
        for file in sorted(model_runs_dir.glob("*.jsonl")):
            features, success = self.extract_features_from_trajectory(file)
            feature_vector = [
                features["early_entropy"],
                features["confidence_gradient"],
                features["mcp_error_ratio"],
                features["logprob_variance"],
                features["verbalized_score"],
                features["loop_ratio"]
            ]
            X_list.append(feature_vector)
            y_list.append(1 if success else 0)
            ids.append(file.stem)
            
        return np.array(X_list), np.array(y_list), ids
