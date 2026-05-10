# RuView Project — Full Technical Report

**Date:** 2026-05-09  
**Author:** Côme + opencode  
**Status:** Firmware fixed, data pipeline working, 2D model trained (72.5% PCK@20), 3D skeleton target not yet achieved

---

## 1. Hardware

| Pos | Role | IP | MAC | Node ID | Chip |
|-----|------|-----|-----|---------|------|
| 3 (center) | TX | 192.168.0.200 | 14:c1:9f:2b:8d:9c | 0 | ESP32-S3-N16R8 |
| 1 (left) | RX1 | 192.168.0.201 | 28:84:85:51:d3:d8 | 1 | ESP32-S3-N16R8 |
| 2 | RX2 | 192.168.0.202 | 14:c1:9f:2c:77:48 | 2 | ESP32-S3-N16R8 |
| 4 | RX3 | 192.168.0.203 | 28:84:85:50:72:58 | 3 | ESP32-S3-N16R8 |
| 5 (right) | RX4 | 192.168.0.204 | 14:c1:9f:2a:8d:00 | 4 | ESP32-S3-N16R8 |

- **Chip:** ESP32-S3-N16R8 (16 MB flash, 8 MB PSRAM)
- **Physical layout:** 30cm wood strip, 5× ESP32 spaced at 3λ/2 (9.375 cm) for 2.4 GHz
- **WiFi:** VOO-8RLHU0U (same as laptop), static DHCP leases on router
- **Laptop server:** 192.168.0.69, EndeavourOS (Arch)
- **Webcam:** HD Webcam via USB, /dev/video0
- **Orientation:** Horizontal strip at waist/chest height, facing movement area

---

## 2. Firmware

### 2.1 Source
`firmware/esp32-csi-node/main/` — ESP-IDF v5.4 project for ESP32-S3

### 2.2 Build issues fixed
1. **CMake 4.x incompatibility** — Added `cmake_policy(SET CMP0167 OLD)` to project CMakeLists.txt. CMake 4.3.2 on Arch breaks ESP-IDF v5.4's deferred target resolution.
2. **mbedtls submodule drift** — Submodule was at v4.0.0 (incompatible). Reset to v3.6.2 via `git submodule update --init components/mbedtls/mbedtls`.
3. **Other submodule drift** — Reset all submodules: `esp_wifi/lib`, `esp_phy/lib`, `esp_coex/lib` were at wrong commits causing linker errors.

### 2.3 Critical bug fixed
**Issue #390 (early-capture node_id):** WiFi init clobbers `g_nvs_config.node_id` back to Kconfig default of `1`. Fix in `csi_collector.c:261-278` — captures node_id to module-local static `s_node_id` BEFORE `wifi_init_sta()`. The pre-built binary (`release_bins/esp32-csi-node.bin`, May 5) lacked this fix.

### 2.4 Flash offsets
```
0x00000  bootloader.bin       (18,912 bytes)
0x08000  partition-table.bin  (3,072 bytes)
0x09000  NVS partition        (24,576 bytes) — node_id, WiFi creds
0x0F000  ota_data_initial.bin (8,192 bytes)
0x20000  esp32-csi-node.bin   (1,091,536 bytes, rebuilt)
```

### 2.5 CSI packet format (20-byte ADR-018 header)
```
Offset  Size  Field
0x00    4     Magic: 0xC5110001 (LE)
0x04    1     Node ID (0-255)
0x05    1     Number of antennas (typically 1)
0x06    2     Number of subcarriers (u16 LE, typically 64)
0x08    4     Frequency MHz (u32 LE)
0x0C    4     Sequence number (u32 LE)
0x10    1     RSSI (i8)
0x11    1     Noise floor (i8)
0x12    2     Reserved
0x14    N     IQ data (N = n_subcarriers × n_antennas × 2 bytes)
```

### 2.6 Build command
```bash
source /home/comus3/Cybersec/tools/esp-idf/export.sh
cd firmware/esp32-csi-node
idf.py build
```

### 2.7 Flash + provision per board
```bash
# Flash
python -m esptool --chip esp32s3 --port /dev/ttyACM0 --baud 460800 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 8MB --flash_freq 80m \
  0x0 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0xf000 build/ota_data_initial.bin \
  0x20000 build/esp32-csi-node.bin

# Provision NVS
uv run python firmware/esp32-csi-node/provision.py \
  --port /dev/ttyACM0 --baud 460800 \
  --ssid "VOO-8RLHU0U" --password "..." \
  --target-ip 192.168.0.69 --target-port 5005 \
  --node-id X
```
(Repeat for each board with `--node-id 0` through `4`)

---

## 3. Software Stack

### 3.1 Data Recording
| Script | Purpose | Fixed? |
|--------|---------|--------|
| `scripts/record-csi-udp.py` | Capture raw CSI frames over UDP | YES — fixed 20-byte header parsing, magic filtering, IP-based node_id override |
| `scripts/collect-ground-truth.py` | Webcam keypoints via MediaPipe | OK — use `--preview` for camera overlay |
| `scripts/diagnose_nodes.py` | Per-IP UDP packet dumper | NEW — created for debugging |

### 3.2 Recording procedure
```bash
# Term 1
uv run python scripts/record-csi-udp.py --duration 300

# Term 2
uv run python scripts/collect-ground-truth.py --preview --duration 300
```

### 3.3 Python environment
- **Path:** `Cybersec/wireless/RuView/.venv` (uv-managed)
- **Key packages:** esptool 5.2.0, nvs-partition-gen, esp-idf-nvs-partition-gen, torch 2.11.0, mediapipe, opencv-python, numpy
- **Clock offset:** Camera timestamps are UTC, CSI timestamps use CEST (UTC+2). Alignment needs `--clock-offset-ms 7200000`.

### 3.4 Rust Server
```
v2/crates/wifi-densepose-sensing-server/
```
- **HTTP:** port 8080 (configurable, `--http-port`)
- **WebSocket:** port 8765 (`/ws/sensing`)
- **UDP CSI receiver:** port 5005
- **Model:** loads `.rvf` containers via `--model path.rvf`
- **IP node_id override:** `--ip-node-map "ip:nid,..."` or env `SENSING_IP_NODE_MAP`

### 3.5 Pose inference bridge
```
scripts/wiflow-live-bridge.js
```
- Loads `wiflow-v1.json` (JS TCN model)
- Connects to server WebSocket at `ws://localhost:8765/ws/sensing`
- Fuses multi-node amplitudes
- Applies EMA temporal smoothing (alpha=0.35)
- Serves enriched pose data + UI on port 3002

### 3.6 Model architecture
```
Input:     64 subcarrier amplitudes × 20 time frames (300ms window)
           ↓
           TCN (4 dilated causal conv blocks, k=3, dilations=[1,2,4,8])
           channels: 64→32→32→32→32
           ↓
           Flatten [32 × 20] → Linear 640→256
           ↓
           Linear 256→34
           ↓
Output:    17 keypoints × (x, y)  [2D ONLY, no depth]
```

Scale presets:
| Scale | Channels | Hidden | Kernel | Params |
|-------|----------|--------|--------|--------|
| lite | [32,32,32,32] | 256 | 3 | 185,506 |
| small | [64,64,48,32] | 512 | 5 | ~750K |
| medium | [128,128,96,64] | 1024 | 7 | ~3M |

---

## 4. Findings & Issues

### 4.1 What works
- [x] All 5 ESP32s stream CSI with unique node_ids (0-4)
- [x] Consistent 64 subcarriers per frame
- [x] 52/64 non-zero amplitudes (normal for ESP32-S3)
- [x] Data recording pipeline reliable
- [x] Camera keypoint extraction via MediaPipe (0.85-0.93 avg confidence)
- [x] CSI↔camera alignment with clock offset correction (95-98% rate)
- [x] JS TCN model training converges
- [x] Live bridge inference works end-to-end

### 4.2 What doesn't work / limitations
- [ ] **3D pose** — Model outputs 2D (x,y) only. No depth prediction.
- [ ] **Through-wall** — Not demonstrated. Requires multistatic fusion from wifi-densepose-signal crate (ruvsense/multistatic.rs), which needs calibration + trained model.
- [ ] **Model quality** — 72.5% PCK@20 on 74 training samples is rough at inference. Need 5000+ samples.
- [ ] **TX NDP injection** — Stub in firmware (`csi_inject_ndp_frame`). Not wired as periodic task. All nodes act as RX only.
- [ ] **TDM mesh** — Not configured. All nodes operate independently.
- [ ] **RVF export** — JS training exports JSON only. No conversion to `.rvf` for Rust server.
- [ ] **Observatory UI** — Demo visualization only. Generates fake 3D skeletons from scenario presets, not real model output.
- [ ] **Multistatic fusion** — `ruvsense/multistatic.rs` exists but needs 3D CSI model + calibration.

### 4.3 Misleading elements in repo
1. **Observatory demo** (`ui/observatory.html`) — Shows 3D skeletons, WiFi waves, particle trails. These are ALL simulated. The "Live WebSocket" mode only reads presence/vitals from the server, generates fake 3D keypoints from pose presets.
2. **Through-wall claims** — The architecture supports it (multistatic + field model), but no working implementation exists.
3. **"3D pose from WiFi"** — The JS model is 2D. The Rust server's signal pipeline detects presence, not pose.

---

## 5. Training Data Collected

### Recording 2026-05-09
| Metric | Value |
|--------|-------|
| CSI frames | 1,567 |
| Subcarriers | 64 (consistent) |
| Nodes streaming | 5/5 (nodes 0-4) |
| Keypoint frames | 7,411 |
| Visible person frames | 6,966 (94%) |
| Avg keypoint confidence | 0.885 |
| Paired samples (aligned) | 74 |
| Alignment rate | 94.9% |
| Duration | ~300s |
| Model trained | WiFlow-lite, 185K params |
| Final PCK@20 | 72.5% |
| Training time | 497s |

### Movement pattern used
```
0-60s:   Walking circles
60-120s:  Arm raises, stretches
120-180s: Turn 360°, different paths
180-240s: Sit/stand/crouch cycles
240-300s: Random natural movement
```

---

## 6. Path to 3D Skeleton (Recommended Architecture)

### 6.1 Training on MM-Fi dataset (realistic)

The MM-Fi dataset provides:
- 600+ hours of CSI from 6 ESP32 receivers
- 17-keypoint 3D skeleton ground truth (motion capture)
- Through-wall scenarios
- Multi-person scenarios

```bash
# Download MM-Fi dataset (requires academic access)
# Then train:
cd v2
cargo run -p wifi-densepose-sensing-server -- \
  --train --dataset mmfi --epochs 500 --dataset-type mmfi
```

This uses the Rust transformer model (`graph_transformer.rs`) which outputs 17×3D keypoints and integrates with the multistatic fusion pipeline.

### 6.2 Multi-camera ground truth (DIY alternative)

If MM-Fi access isn't available, use 2-3 cameras for triangulated 3D ground truth:

1. Place cameras at known positions (front, side, top)
2. Calibrate with checkerboard (OpenCV `calibrateCamera`)
3. Run MediaPipe on each camera
4. Triangulate 2D keypoints → 3D using camera matrices
5. Record CSI + 3D keypoints simultaneously
6. Train a 3D model (PyTorch or Rust graph transformer)

Required toolkit: Python + OpenCV triangulation + numpy

### 6.3 Multistatic fusion (required for through-wall)

The Rust crate `wifi-densepose-signal/src/ruvsense/multistatic.rs` implements:
- Attention-weighted fusion across viewpoints
- Geometric diversity index (Cramér-Rao bounds)
- Coherence gating for through-wall signal rejection

To activate:
1. Calibrate `FieldModel` (empty room baseline)
2. Configure node positions in 3D space
3. Feed per-node CSI frames through `MultistaticFuser::fuse()`
4. Run fused features through a 3D model

### 6.4 Custom app architecture proposal

```
ESP32 Array (5 nodes)
    │ UDP CSI frames
    ▼
Rust Sensing Server (Axum)
    │ performs: signal processing, multistatic fusion
    │           field model calibration
    │ WebSocket: sensing_update + per-node features
    ▼
Model Inference (wasm or Python or ONNX)
    │ loads: trained 3D pose model (.rvf or .onnx)
    │ input:  fused CSI features + per-node amplitudes
    │ output: 17 keypoints × (x, y, z) + confidence
    ▼
Custom 3D Visualizer (Three.js + WebSocket)
    │ renders: skeleton in empty 3D space
    │         room outline / wall positions
    │         signal strength heatmap overlay
    ▼
Browser: http://localhost:3002
```

Key decisions:
1. **Model format:** ONNX (.onnx) for portability between Python training and Rust/JS inference
2. **Training:** PyTorch on MM-Fi or multi-camera setup → export ONNX
3. **Inference:** ONNX Runtime in Rust (`wifi-densepose-nn` crate) or ort-web in JS
4. **Visualization:** Standalone Three.js page, no demo fluff
5. **Through-wall:** Run multistatic fusion + field model, then same inference pipeline

---

## 7. Files Created/Modified This Session

### Created
- `scripts/diagnose_nodes.py` — UDP packet diagnostic tool
- `ui/pose3d.html` — Standalone 3D skeleton visualizer with orbit controls
- `PROJECT_REPORT.md` — This document

### Modified
- `scripts/record-csi-udp.py` — Fixed 20-byte header parsing, magic filtering, IP-based node_id override
- `scripts/wiflow-live-bridge.js` — Added multi-node amplitude fusion + EMA temporal smoothing
- `v2/crates/wifi-densepose-sensing-server/src/main.rs` — Added `--ip-node-map` CLI/env support for Rust server
- `firmware/esp32-csi-node/CMakeLists.txt` — Added CMake 4.x policy fix (CMP0167 OLD)
- `firmware/esp32-csi-node/release_bins/` — Updated with rebuilt firmware binaries

---

## 8. Quick Reference — Commands

### Firmware rebuild
```bash
source /home/comus3/Cybersec/tools/esp-idf/export.sh
cd /home/comus3/Cybersec/wireless/RuView/firmware/esp32-csi-node
idf.py build
```

### Flash + provision (per board)
```bash
python -m esptool --chip esp32s3 --port /dev/ttyACM0 --baud 460800 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 8MB --flash_freq 80m \
  0x0 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0xf000 build/ota_data_initial.bin \
  0x20000 build/esp32-csi-node.bin

uv run python firmware/esp32-csi-node/provision.py \
  --port /dev/ttyACM0 --baud 460800 \
  --ssid "VOO-8RLHU0U" --password "..." \
  --target-ip 192.168.0.69 --target-port 5005 --node-id X
```

### Record training data
```bash
# Term 1
uv run python scripts/record-csi-udp.py --duration 300
# Term 2
uv run python scripts/collect-ground-truth.py --preview --duration 300
```

### Align & train
```bash
node scripts/align-ground-truth.js \
  --gt data/ground-truth/keypoints_FILE.jsonl \
  --csi data/recordings/csi-FILE.csi.jsonl \
  --clock-offset-ms 7200000

node scripts/train-wiflow-supervised.js \
  --data data/paired/csi-FILE.paired.jsonl \
  --epochs 300 --scale small
```

### Run live inference
```bash
# Server
cd v2 && cargo run -p wifi-densepose-sensing-server -- \
  --source esp32 --bind-addr 0.0.0.0

# Bridge
node scripts/wiflow-live-bridge.js \
  models/wiflow-supervised/wiflow-v1.json \
  ws://localhost:8765/ws/sensing

# Open http://localhost:3002/pose3d.html
```

### Diagnostic
```bash
uv run python scripts/diagnose_nodes.py --port 5005 --duration 30
```

---

## 9. Next Steps (Prioritized)

1. **Acquire MM-Fi dataset** — The only realistic path to 3D through-wall pose. Request academic access at https://mm-fi.github.io/
2. **Set up multi-camera calibration** — If MM-Fi access fails, 2-3 webcams + OpenCV triangulatePoints() for DIY 3D ground truth
3. **Train 3D model** — PyTorch TCN→3D regressor, export to ONNX
4. **Wire multistatic fusion** — Configure `ruvsense/multistatic.rs` with node positions, calibrate field model
5. **Build custom visualizer** — Three.js scene with real 3D skeleton + room outline + through-wall mode
6. **Integrate ONNX inference** — Replace JS bridge with ort-web or Rust ONNX runtime
7. **Collect more 2D data in parallel** — More 300s recordings with extreme movement variety improves the 2D model as a fallback
