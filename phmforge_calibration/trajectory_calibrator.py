import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from typing import Dict, Any, Union

class TrajectoryCalibrator:
    """Trajectory calibration engine supporting five levels of diagnostic signals:
    1. VerbalizedIsotonic: Isotonic Regression on verbalized confidence.
    2. CalVerT: Logistic Regression on MCP tool telemetry.
    3. HTC: Random Forest on the full feature vector f(tau).
    4. HTC-Boosting: Gradient Boosting Classifier on f(tau).
    5. HTC-ElasticNet: ElasticNet-regularized Logistic Regression on f(tau).
    """
    def __init__(self, method: str = "HTC"):
        self.method = method
        self.verbalized_isotonic = IsotonicRegression(out_of_bounds="clip")
        self.calvert_model = LogisticRegression(class_weight="balanced")
        self.htc_model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        self.boosting_model = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42)
        self.elasticnet_model = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5, class_weight="balanced", random_state=42
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrajectoryCalibrator":
        """Fits calibration models based on feature matrix X and target y.
        X columns map to:
        0: early_entropy
        1: confidence_gradient
        2: mcp_error_ratio
        3: logprob_variance
        4: verbalized_score
        5: loop_ratio
        """
        if len(X) == 0:
            return self
            
        # 1. VerbalizedIsotonic (only use column 4: verbalized_score)
        verbalized_scores = X[:, 4]
        self.verbalized_isotonic.fit(verbalized_scores, y)
        
        # 2. CalVerT (only use column 2: mcp_error_ratio)
        mcp_features = X[:, 2:3]
        self.calvert_model.fit(mcp_features, y)
        
        # 3. HTC (use all features)
        self.htc_model.fit(X, y)
        
        # 4. HTC-Boosting
        self.boosting_model.fit(X, y)
        
        # 5. HTC-ElasticNet
        self.elasticnet_model.fit(X, y)
        
        return self
        
    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """Predicts calibrated probability of task success (Pass@1)."""
        if len(X) == 0:
            return np.array([])
            
        if self.method == "VerbalizedIsotonic":
            verbalized_scores = X[:, 4]
            return self.verbalized_isotonic.predict(verbalized_scores)
            
        elif self.method == "CalVerT":
            mcp_features = X[:, 2:3]
            return self.calvert_model.predict_proba(mcp_features)[:, 1]
            
        elif self.method == "HTC":
            return self.htc_model.predict_proba(X)[:, 1]
            
        elif self.method == "HTC-Boosting":
            return self.boosting_model.predict_proba(X)[:, 1]
            
        elif self.method == "HTC-ElasticNet":
            return self.elasticnet_model.predict_proba(X)[:, 1]
            
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
            
    def predict_raw_verbalized(self, X: np.ndarray) -> np.ndarray:
        """Returns the raw uncalibrated verbalized confidence scores."""
        return X[:, 4]
