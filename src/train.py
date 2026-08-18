"""
train.py
--------
Train the autoencoder on NORMAL traffic and persist the model, scaler and
threshold.

Input : data/normal_traffic.csv  (rows = 9-feature windows, header optional)
Output: models/autoencoder.npz
        models/scaler.npz
        models/threshold.txt

Usage:
    python3 src/train.py --data data/normal_traffic.csv --epochs 500
"""

import os
import argparse
import numpy as np

from autoencoder import Autoencoder
from preprocessing import FeatureScaler
from detector import compute_threshold
from feature_extractor import FEATURE_NAMES

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_HERE, "models")


def load_csv(path):
    # skip a header row if the first token is non-numeric
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip().split(",")[0]
    skip = 1
    try:
        float(first)
        skip = 0
    except ValueError:
        skip = 1
    data = np.loadtxt(path, delimiter=",", skiprows=skip)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape[1] == len(FEATURE_NAMES), \
        f"expected {len(FEATURE_NAMES)} columns, got {data.shape[1]}"
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "data", "normal_traffic.csv"))
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--val-split", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(_MODELS, exist_ok=True)

    X = load_csv(args.data)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X = X[idx]
    split = int(len(X) * (1.0 - args.val_split))
    X_train, X_val = X[:split], X[split:]

    scaler = FeatureScaler().fit(X_train)
    Xn_train = scaler.transform(X_train)
    Xn_val = scaler.transform(X_val)

    model = Autoencoder(input_dim=9, hidden_dim=6)
    model.train(Xn_train, epochs=args.epochs, lr=args.lr)

    val_errors = model.reconstruction_error(Xn_val)
    threshold = compute_threshold(val_errors)

    model.save(os.path.join(_MODELS, "autoencoder.npz"))
    scaler.save(os.path.join(_MODELS, "scaler.npz"))
    with open(os.path.join(_MODELS, "threshold.txt"), "w") as f:
        f.write(str(threshold))

    print(f"\nTraining complete.")
    print(f"  validation windows : {len(X_val)}")
    print(f"  threshold          : {threshold:.6f}")
    print(f"  models saved to    : {_MODELS}")


if __name__ == "__main__":
    main()
