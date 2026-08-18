"""
config.py
---------
Loads config/config.yaml. Falls back to config/config.example.yaml if the
real config is missing, so a fresh checkout still runs in demo mode.
"""

import os
import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL = os.path.join(_HERE, "config", "config.yaml")
_EXAMPLE = os.path.join(_HERE, "config", "config.example.yaml")


def load_config():
    path = _REAL if os.path.exists(_REAL) else _EXAMPLE
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
