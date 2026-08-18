#!/bin/bash
#
# network-guardian-startup.sh
# ---------------------------
# Brings up the IoT hotspot (wlan1) and configures NAT so hotspot clients
# reach the internet through the uplink (wlan0).
#
# Install to /etc/network-guardian-startup.sh and run at boot (see systemd unit).

set -e

UPLINK_IF="wlan0"
HOTSPOT_IF="wlan1"
HOTSPOT_IP="192.168.4.1/24"

echo "[network-guardian] configuring ${HOTSPOT_IF} ..."

# Assign a static IP to the hotspot interface
ip addr flush dev "${HOTSPOT_IF}"
ip addr add "${HOTSPOT_IP}" dev "${HOTSPOT_IF}"
ip link set "${HOTSPOT_IF}" up

# Restart the AP + DHCP services
systemctl restart hostapd
systemctl restart dnsmasq

# Enable IP forwarding (gateway behaviour)
sysctl -w net.ipv4.ip_forward=1

# --- NAT / forwarding rules -------------------------------------------------
# 1) Masquerade hotspot traffic going out via the uplink
iptables -t nat -A POSTROUTING -o "${UPLINK_IF}" -j MASQUERADE
# 2) Allow hotspot -> uplink (outbound)
iptables -A FORWARD -i "${HOTSPOT_IF}" -o "${UPLINK_IF}" -j ACCEPT
# 3) Allow uplink -> hotspot only for established/related (return traffic)
iptables -A FORWARD -i "${UPLINK_IF}" -o "${HOTSPOT_IF}" \
    -m state --state RELATED,ESTABLISHED -j ACCEPT

echo "[network-guardian] hotspot ready on ${HOTSPOT_IP} (${HOTSPOT_IF})."
