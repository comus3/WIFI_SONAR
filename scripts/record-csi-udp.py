#!/usr/bin/env python3
"""
Lightweight ESP32 CSI UDP recorder (ADR-079).

Captures raw CSI packets from ESP32 nodes over UDP and writes to JSONL.
Runs alongside collect-ground-truth.py for synchronized capture.

Usage:
    python scripts/record-csi-udp.py --duration 300 --output data/recordings
"""

import argparse
import json
import os
import socket
import struct
import time


CSI_MAGIC = 0xC5110001

def parse_csi_packet(data, src_ip=None, ip_node_map=None):
    """Parse ADR-018 CSI frame with 20-byte header.

    Filters strictly by magic 0xC5110001.  Non-CSI packets (edge vitals
    0xC5110002, features 0xC5110003, fused 0xC5110005, etc.) carry
    breathing/HR/presence — not CSI amplitudes — and are skipped.
    """
    if len(data) < 20:
        return None

    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != CSI_MAGIC:
        return None

    # 20-byte header: [magic(4), node_id, antennas, n_sub(2), freq(4), seq(4), rssi, noise, resv(2), IQ...]
    node_id = data[4]
    n_antennas = data[5]
    n_subcarriers = struct.unpack_from('<H', data, 6)[0]
    freq_mhz = struct.unpack_from('<I', data, 8)[0]
    rssi = struct.unpack_from('b', data, 16)[0]

    if ip_node_map and src_ip in ip_node_map:
        node_id = ip_node_map[src_ip]

    # IQ data starts at byte 20
    iq_data = data[20:]
    n_pairs = min(len(iq_data) - 1, n_subcarriers * n_antennas * 2)

    amplitudes = []
    for i in range(0, n_pairs, 2):
        I_val = struct.unpack('b', bytes([iq_data[i]]))[0]
        Q_val = struct.unpack('b', bytes([iq_data[i + 1]]))[0]
        amplitudes.append(round((I_val * I_val + Q_val * Q_val) ** 0.5, 2))

    return {
        "type": "raw_csi",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(time.time() * 1000) % 1000:03d}Z",
        "ts_ns": time.time_ns(),
        "node_id": node_id,
        "subcarriers": len(amplitudes),
        "rssi": rssi,
        "amplitudes": amplitudes,
    }


def main():
    parser = argparse.ArgumentParser(description="Record ESP32 CSI over UDP")
    parser.add_argument("--port", type=int, default=5005, help="UDP port (default: 5005)")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds (default: 300)")
    parser.add_argument("--output", default="data/recordings", help="Output directory")
    parser.add_argument("--ip-node-map", type=str, default=None,
                        help="Comma-separated IP:node_id overrides (e.g. '192.168.0.201:1,192.168.0.202:2')")
    args = parser.parse_args()

    # Parse IP->node_id map
    ip_node_map = {}
    if args.ip_node_map:
        for pair in args.ip_node_map.split(","):
            ip, nid = pair.strip().split(":")
            ip_node_map[ip.strip()] = int(nid.strip())
        print(f"IP->node_id map: {ip_node_map}")

    os.makedirs(args.output, exist_ok=True)
    filename = f"csi-{int(time.time())}.csi.jsonl"
    filepath = os.path.join(args.output, filename)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1)

    print(f"Recording CSI on UDP :{args.port} for {args.duration}s")
    print(f"Output: {filepath}")

    count = 0
    start = time.time()
    nodes_seen = set()

    with open(filepath, "w") as f:
        try:
            while time.time() - start < args.duration:
                try:
                    data, addr = sock.recvfrom(4096)
                    frame = parse_csi_packet(data, src_ip=addr[0], ip_node_map=ip_node_map)
                    if frame:
                        f.write(json.dumps(frame) + "\n")
                        count += 1
                        nodes_seen.add(frame["node_id"])

                        if count % 500 == 0:
                            elapsed = time.time() - start
                            rate = count / elapsed
                            print(f"  {count} frames | {rate:.0f} fps | "
                                  f"nodes: {sorted(nodes_seen)} | "
                                  f"{elapsed:.0f}s / {args.duration}s")
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("\nStopped by user")

    sock.close()
    elapsed = time.time() - start
    print(f"\n=== CSI Recording Complete ===")
    print(f"  Frames: {count}")
    print(f"  Duration: {elapsed:.0f}s")
    print(f"  Rate: {count / max(elapsed, 1):.0f} fps")
    print(f"  Nodes: {sorted(nodes_seen)}")
    print(f"  Output: {filepath}")


if __name__ == "__main__":
    main()
