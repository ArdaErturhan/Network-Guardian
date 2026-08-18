"""
preprocessing.py
----------------
Min-max feature scaler implemented with NumPy only.

Fitted on normal training traffic so that every feature lands in [0, 1].
The same min/max are reused at inference time. Values outside the training
range are clipped to [0, 1] to keep the autoencoder input bounded.
"""

import numpy as np


class FeatureScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        X = np.atleast_2d(X).astype(np.float64)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        # avoid divide-by-zero for constant columns
        self._range = np.where((self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X):
        X = np.atleast_2d(X).astype(np.float64)
        scaled = (X - self.min_) / self._range
        return np.clip(scaled, 0.0, 1.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        np.savez(path, min_=self.min_, max_=self.max_)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        s = cls()
        s.min_ = data["min_"]
        s.max_ = data["max_"]
        s._range = np.where((s.max_ - s.min_) == 0, 1.0, s.max_ - s.min_)
        return s
