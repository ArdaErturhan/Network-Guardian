"""
sniffer.py
----------
Real-time packet capture on the gateway interface using Scapy.

Collects packets for `window_seconds`, converts each into a lightweight
dict, and hands the completed window to a user-supplied callback.
"""

import time
import threading

from scapy.all import sniff, IP, TCP, UDP


def _packet_to_dict(pkt):
    """Convert a Scapy packet into the minimal dict our extractor needs."""
    if IP not in pkt:
        return None

    ip = pkt[IP]
    proto = "OTHER"
    dst_port = None

    if TCP in pkt:
        proto = "TCP"
        dst_port = int(pkt[TCP].dport)
    elif UDP in pkt:
        proto = "UDP"
        dst_port = int(pkt[UDP].dport)

    return {
        "size": len(pkt),
        "dst_ip": ip.dst,
        "dst_port": dst_port,
        "proto": proto,
        "ts": time.time(),
    }


class WindowSniffer:
    """
    Sniffs `iface` and calls `on_window(packets: list[dict])` every
    `window_seconds`. Runs the Scapy sniff loop in a background thread.
    """

    def __init__(self, iface="wlan1", window_seconds=10.0, on_window=None):
        self.iface = iface
        self.window_seconds = window_seconds
        self.on_window = on_window
        self._buffer = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _handle(self, pkt):
        d = _packet_to_dict(pkt)
        if d is not None:
            with self._lock:
                self._buffer.append(d)

    def _flush_loop(self):
        while not self._stop.is_set():
            time.sleep(self.window_seconds)
            with self._lock:
                window, self._buffer = self._buffer, []
            if self.on_window:
                self.on_window(window)

    def start(self):
        # window timer thread
        threading.Thread(target=self._flush_loop, daemon=True).start()
        # blocking sniff loop (stops when _stop is set)
        sniff(
            iface=self.iface,
            prn=self._handle,
            store=False,
            stop_filter=lambda p: self._stop.is_set(),
        )

    def stop(self):
        self._stop.set()
