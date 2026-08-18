"""
autoencoder.py
--------------
Lightweight feedforward Autoencoder implemented from scratch using only NumPy.

Architecture: 9 -> 6 -> 9 (input -> bottleneck -> reconstruction)
- Encoder: W1 (9x6), b1  + ReLU
- Decoder: W2 (6x9), b2  (linear output)

The model is trained EXCLUSIVELY on normal traffic. Anomalies produce a
higher reconstruction error (MSE) because they deviate from the learned
normal representation.

No TensorFlow / PyTorch / scikit-learn dependency for the model itself.
"""

import numpy as np


class Autoencoder:
    def __init__(self, input_dim=9, hidden_dim=6, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        # He-style initialization for ReLU encoder
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, input_dim)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(input_dim)

    # ---- activations ---------------------------------------------------
    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0.0).astype(x.dtype)

    # ---- forward -------------------------------------------------------
    def _forward(self, X):
        z1 = X @ self.W1 + self.b1      # pre-activation, encoder
        a1 = self._relu(z1)             # latent (bottleneck)
        z2 = a1 @ self.W2 + self.b2     # reconstruction (linear)
        cache = (X, z1, a1, z2)
        return z2, cache

    def reconstruct(self, X):
        """Return the reconstructed input."""
        X = np.atleast_2d(X)
        out, _ = self._forward(X)
        return out

    def reconstruction_error(self, X):
        """Per-sample mean squared reconstruction error (the anomaly score)."""
        X = np.atleast_2d(X)
        out, _ = self._forward(X)
        return np.mean((X - out) ** 2, axis=1)

    # ---- training ------------------------------------------------------
    def train(self, X, epochs=500, lr=0.01, batch_size=32, verbose=True):
        """
        Train with mini-batch gradient descent on MSE loss.
        X must be normalized (see FeatureScaler in preprocessing.py).
        """
        X = np.atleast_2d(X).astype(np.float64)
        n = X.shape[0]
        rng = np.random.default_rng(0)

        for epoch in range(epochs):
            idx = rng.permutation(n)
            X_shuf = X[idx]
            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                m = xb.shape[0]

                out, cache = self._forward(xb)
                _, z1, a1, _ = cache

                # MSE loss gradient w.r.t. output
                dout = (2.0 / m) * (out - xb)          # (m, 9)

                # Decoder grads
                dW2 = a1.T @ dout                       # (6, 9)
                db2 = np.sum(dout, axis=0)

                # Backprop into encoder
                da1 = dout @ self.W2.T                  # (m, 6)
                dz1 = da1 * self._relu_grad(z1)         # (m, 6)
                dW1 = xb.T @ dz1                         # (9, 6)
                db1 = np.sum(dz1, axis=0)

                # SGD update
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

                epoch_loss += np.mean((out - xb) ** 2) * m

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}  loss={epoch_loss / n:.6f}")

        return self

    # ---- persistence ---------------------------------------------------
    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(input_dim=data["W1"].shape[0], hidden_dim=data["W1"].shape[1])
        model.W1, model.b1 = data["W1"], data["b1"]
        model.W2, model.b2 = data["W2"], data["b2"]
        return model
