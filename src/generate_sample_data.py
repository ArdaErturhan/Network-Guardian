"""
generate_sample_data.py
-----------------------
Creates a synthetic data/normal_traffic.csv so the project can be trained and
tested without a live Raspberry Pi capture. Values loosely mimic quiet IoT
traffic across 9 features. Replace with real captured windows for real use.

Usage:
    python3 src/generate_sample_data.py --rows 2000
"""

import os
import argparse
import numpy as np

from feature_extractor import FEATURE_NAMES

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(_HERE, "data", "normal_traffic.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(7)
    n = args.rows

    packet_count   = rng.normal(120, 25, n).clip(10, None)
    avg_size       = rng.normal(340, 60, n).clip(64, 1500)
    std_size       = rng.normal(90, 20, n).clip(0, None)
    unique_dst_ips = rng.normal(4, 1.2, n).clip(1, None)
    unique_dports  = rng.normal(3, 1.0, n).clip(1, None)
    tcp_ratio      = rng.normal(0.75, 0.08, n).clip(0, 1)
    udp_ratio      = (1 - tcp_ratio) * rng.uniform(0.6, 1.0, n)
    avg_iat        = rng.normal(0.08, 0.02, n).clip(0.001, None)
    bytes_per_sec  = (packet_count * avg_size) / 10.0

    X = np.column_stack([
        packet_count, avg_size, std_size, unique_dst_ips, unique_dports,
        tcp_ratio, udp_ratio, avg_iat, bytes_per_sec,
    ])

    header = ",".join(FEATURE_NAMES)
    np.savetxt(args.out, X, delimiter=",", header=header, comments="", fmt="%.4f")
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
