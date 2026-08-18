#!/usr/bin/env python3
"""
main.py
-------
NetworkGuardian runtime entry point.

Loads the trained model / scaler / threshold, starts the Scapy sniffer on the
hotspot interface, and for every 10-second window:
    1. extracts the 9-feature vector
    2. scores it with the autoencoder
    3. logs the result
    4. sends an email alert if the reconstruction error exceeds the threshold

Requires root (raw packet capture):
    sudo venv/bin/python3 main.py
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from config import load_config
from autoencoder import Autoencoder
from preprocessing import FeatureScaler
from detector import Detector
from feature_extractor import extract_features, FEATURE_NAMES
from sniffer import WindowSniffer
from alerting import EmailAlerter

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "models")
_LOGS = os.path.join(_HERE, "logs")


def setup_logging():
    os.makedirs(_LOGS, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(_LOGS, "guardian.log")),
            logging.StreamHandler(),
        ],
    )


def load_detector():
    model = Autoencoder.load(os.path.join(_MODELS, "autoencoder.npz"))
    scaler = FeatureScaler.load(os.path.join(_MODELS, "scaler.npz"))
    with open(os.path.join(_MODELS, "threshold.txt")) as f:
        threshold = float(f.read().strip())
    return Detector(model, scaler, threshold)


def main():
    setup_logging()
    cfg = load_config()

    if not os.path.exists(os.path.join(_MODELS, "autoencoder.npz")):
        logging.error("No trained model found. Run: python3 src/train.py first.")
        sys.exit(1)

    detector = load_detector()
    logging.info("Loaded detector (threshold=%.6f)", detector.threshold)

    alerter = None
    a = cfg.get("alerting", {})
    if a.get("enabled"):
        alerter = EmailAlerter(
            sender=a["sender"],
            app_password=a["app_password"],
            recipient=a["recipient"],
            cooldown_seconds=a.get("cooldown_seconds", 60),
        )
        logging.info("Email alerting enabled -> %s", a["recipient"])

    net = cfg.get("network", {})
    iface = net.get("monitor_interface", "wlan1")
    window = float(net.get("window_seconds", 10.0))

    def on_window(packets):
        features = extract_features(packets, window_seconds=window)
        is_anom, score = detector.is_anomaly(features)

        summary = ", ".join(
            f"{name}={val:.2f}" for name, val in zip(FEATURE_NAMES, features)
        )

        if is_anom:
            logging.warning("ANOMALY  score=%.6f  | %s", score, summary)
            if alerter:
                alerter.send(score, detector.threshold, feature_summary=summary)
        else:
            logging.info("normal   score=%.6f  packets=%d", score, len(packets))

    sniffer = WindowSniffer(iface=iface, window_seconds=window, on_window=on_window)
    logging.info("Monitoring %s in %.0fs windows. Ctrl+C to stop.", iface, window)

    try:
        sniffer.start()
    except KeyboardInterrupt:
        sniffer.stop()
        logging.info("Stopped by user.")


if __name__ == "__main__":
    main()
