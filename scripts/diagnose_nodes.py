#!/usr/bin/env python3
"""
Diagnose ESP32 node_id issue: capture raw UDP packets and dump header bytes
to verify node_id per source IP.

Run: python scripts/diagnose_nodes.py --port 5005 --duration 30
"""
import argparse, json, socket, struct, time, collections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1)

    print(f"Diagnosing UDP :{args.port} for {args.duration}s...")
    print(f"{'Source IP':>16s} | node_id | magic_LE  | hdr_hex(0..19)")
    print("-" * 80)

    seen = collections.defaultdict(lambda: {"count": 0, "mags": set(), "samples": []})
    start = time.time()

    try:
        while time.time() - start < args.duration:
            try:
                data, addr = sock.recvfrom(2048)
                src = addr[0]
                s = seen[src]
                s["count"] += 1

                node_id = data[4] if len(data) > 4 else 0
                magic = struct.unpack_from("<I", data)[0] if len(data) >= 4 else 0
                hex_20 = data[:20].hex() if len(data) >= 20 else data.hex()

                s["mags"].add(f"0x{magic:08X}")
                if len(s["samples"]) < 3:
                    s["samples"].append((node_id, magic, hex_20[:40]))
            except socket.timeout:
                continue
    except KeyboardInterrupt:
        pass
    sock.close()

    if not seen:
        print("No UDP packets received. Check ESP32 target IP:port.")
        return

    for src, info in sorted(seen.items()):
        print(f"{src:>16s} | {info['samples'][0][0]:>7d} | {', '.join(sorted(info['mags'])):>11s} | {info['samples'][0][2]}")
        if info["count"] > 1:
            fps = info["count"] / max(time.time() - start, 1)
            print(f"  -> {info['count']} pkts ({fps:.1f}/s)")

    # Check if all node_ids are the same
    all_nids = set()
    for info in seen.values():
        for nid, _, _ in info["samples"]:
            all_nids.add(nid)
    print(f"\nUnique node_ids seen: {sorted(all_nids)}")
    print(f"Source IPs sending:   {len(seen)}")

if __name__ == "__main__":
    main()
