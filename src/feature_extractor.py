"""
feature_extractor.py
--------------------
Aggregates the packets captured within a fixed time window (default 10s)
into a 9-dimensional feature vector describing the traffic behaviour.

The 9 features (order is fixed and must match training):
    0. packet_count      - total packets in the window
    1. avg_size          - mean packet size (bytes)
    2. std_size          - std-dev of packet size
    3. unique_dst_ips    - number of distinct destination IPs
    4. unique_dst_ports  - number of distinct dst ports < 1024
    5. tcp_ratio         - fraction of TCP packets
    6. udp_ratio         - fraction of UDP packets
    7. avg_iat           - mean inter-arrival time between packets (s)
    8. bytes_per_sec     - total bytes / window length
"""

import numpy as np

FEATURE_NAMES = [
    "packet_count",
    "avg_size",
    "std_size",
    "unique_dst_ips",
    "unique_dst_ports",
    "tcp_ratio",
    "udp_ratio",
    "avg_iat",
    "bytes_per_sec",
]


def extract_features(packets, window_seconds=10.0):
    """
    packets: list of dicts, each with keys:
        'size' (int), 'dst_ip' (str), 'dst_port' (int|None),
        'proto' ('TCP'|'UDP'|'OTHER'), 'ts' (float epoch seconds)

    Returns a numpy array of shape (9,).
    Empty windows return a zero vector.
    """
    n = len(packets)
    if n == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    sizes = np.array([p["size"] for p in packets], dtype=np.float64)
    dst_ips = {p["dst_ip"] for p in packets if p.get("dst_ip")}
    dst_ports = {
        p["dst_port"]
        for p in packets
        if p.get("dst_port") is not None and p["dst_port"] < 1024
    }

    tcp = sum(1 for p in packets if p.get("proto") == "TCP")
    udp = sum(1 for p in packets if p.get("proto") == "UDP")

    ts = sorted(p["ts"] for p in packets)
    if len(ts) > 1:
        iats = np.diff(ts)
        avg_iat = float(np.mean(iats))
    else:
        avg_iat = 0.0

    total_bytes = float(np.sum(sizes))

    features = np.array([
        float(n),                       # packet_count
        float(np.mean(sizes)),          # avg_size
        float(np.std(sizes)),           # std_size
        float(len(dst_ips)),            # unique_dst_ips
        float(len(dst_ports)),          # unique_dst_ports
        tcp / n,                        # tcp_ratio
        udp / n,                        # udp_ratio
        avg_iat,                        # avg_iat
        total_bytes / window_seconds,   # bytes_per_sec
    ], dtype=np.float64)

    return features
