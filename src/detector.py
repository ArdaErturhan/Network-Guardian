"""
detector.py
-----------
Wraps the trained autoencoder + scaler and applies the anomaly threshold.

Threshold strategy (computed on normal validation reconstruction errors):

    threshold = max( P99,  Mean + 4*STD,  T_max * 1.5 )

- P99            : non-parametric, distribution-agnostic tail cut-off
- Mean + 4*STD   : parametric (Gaussian) cut-off
- T_max * 1.5    : guard band above the largest observed normal error

Taking the max keeps false positives low while still catching clear
deviations. A window whose reconstruction error exceeds the threshold
is flagged as an anomaly.
"""

import numpy as np


def compute_threshold(errors):
    """errors: 1-D array of reconstruction errors on NORMAL validation data."""
    errors = np.asarray(errors, dtype=np.float64)
    p99 = float(np.percentile(errors, 99))
    mean_std = float(errors.mean() + 4.0 * errors.std())
    t_max = float(errors.max() * 1.5)
    return max(p99, mean_std, t_max)


class Detector:
    def __init__(self, model, scaler, threshold):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def score(self, feature_vector):
        """Return the reconstruction error for a single 9-D feature vector."""
        x = self.scaler.transform(feature_vector.reshape(1, -1))
        return float(self.model.reconstruction_error(x)[0])

    def is_anomaly(self, feature_vector):
        """Return (is_anomaly: bool, score: float)."""
        score = self.score(feature_vector)
        return score > self.threshold, score
