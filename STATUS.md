# RuView Project Status — 2026-05-08

## Hardware

| Pos | Role | IP | MAC | Node ID | Status |
|-----|------|-----|-----|---------|--------|
| 1 (left) | RX1 | 192.168.0.201 | 28:84:85:51:d3:d8 | 1 | CSI streaming (70 subc) |
| 2 | RX2 | 192.168.0.202 | 14:c1:9f:2c:77:48 | 2 | Online, CSI NOT sending |
| 3 (center) | TX | 192.168.0.200 | 14:c1:9f:2b:8d:9c | 0 | Online, CSI NOT sending |
| 4 | RX3 | 192.168.0.203 | 28:84:85:50:72:58 | 3 | Online, CSI NOT sending |
| 5 (right) | RX4 | 192.168.0.204 | 14:c1:9f:2a:8d:00 | 4 | Online, CSI NOT sending |

- **Chip**: ESP32-S3-N16R8 (16 MB flash, 8 MB PSRAM)
- **Physical layout**: 30cm wood strip, 5x ESP32 spaced at 3λ/2 (9.375 cm) for 2.4 GHz
- **MACs**: Hardware-burned, static DHCP leases on router
- **WiFi**: VOO-8RLHU0U (same as laptop)

## Firmware

- **Binary**: `firmware/esp32-csi-node/release_bins/esp32-csi-node.bin` (990 KB)
- **Source**: `firmware/esp32-csi-node/main/`
- **Flash offsets** (16 MB N16R8):
  - `0x0`: bootloader.bin
  - `0x8000`: partition-table.bin
  - `0xF000`: ota_data_initial.bin
  - `0x20000`: esp32-csi-node.bin
- **NVS partition** (provisioned by `provision.py`):
  - Offset: `0x9000`, Size: `0x6000` (24 KB)
  - Stores: `ssid`, `password`, `target_ip`, `target_port`, `node_id`
- **TX role**: NDP probe injection at 10 Hz via `csi_inject_ndp_frame()` (ADR-029)
- **All 5 boards re-flashed 2026-05-08** with same binary

## Current Problem

**Only RX1 (192.168.0.201, node_id=1) is sending CSI frames.** All 5 boards online/pingable but only one produces data. Likely causes:

1. **NVS not surviving firmware flash**: The `write-flash` erases specific regions but NVS at `0x9000` is NOT in any erase range. However, the provision.py writes after flash, so NVS should be present.
2. **Firmware Kconfig default**: `CONFIG_CSI_NODE_ID` defaults to 1. If NVS read fails, all boards report node_id=1.
3. **CSI mode not enabled**: The firmware might require CSI to be explicitly enabled in NVS (edge_tier, etc.)
4. **Single CSI source**: Maybe only one node is supposed to stream CSI (the rest are TDM listeners)

## SAURON Server (Pi 4)

- **Path**: `/opt/sauron/services/sensing-server/`
- **Docker**: `ruvnet/wifi-densepose:latest` (2026-03-04)
- **Ports**: UDP 5005 (CSI), HTTP 3000 (UI+API), WS 3001
- **Proxy**: NPM at `sensing.sauron.home` -> `sensing-server:3000`
- **Status**: Currently STOPPED (laptop testing)

## Laptop Server

- **IP**: 192.168.0.69
- **Docker**: `ruvnet/wifi-densepose:latest`
- **ESP32s provisioned to**: `192.168.0.69:5005`
- **Status**: Docker stopped during data collection, needs restart

## Scripts (all in `scripts/`)

| Script | Purpose | Source |
|--------|---------|--------|
| `provision.py` | Write WiFi + node config to NVS | Official |
| `record-csi-udp.py` | Record raw CSI amplitudes over UDP | Official |
| `collect-ground-truth.py` | Capture webcam keypoints via MediaPipe | Official |
| `align-ground-truth.js` | Align CSI timestamps with camera keypoints | Official |
| `train-wiflow-supervised.js` | Train WiFlow model (JS/SPSA) | Official |
| `eval-wiflow.js` | Evaluate trained model | Official |
| `wiflow-live-bridge.js` | Serve trained model via WebSocket | Custom (all-nighter) |
| `wiflow-model.js` | WiFlow model definitions | Custom (all-nighter) |
| `train_wiflow_torch.py` | PyTorch GPU training | Custom (today) |

## Training Data

### Latest collection (2026-05-08)
- **CSI**: `data/recordings/csi-1778250478.csi.jsonl` — 34,032 frames, 300s, 113 fps — **only node_id=1**
- **Camera**: `data/ground-truth/keypoints_20260508_162758.jsonl` — not yet checked

### Previous collection (2026-05-06)
- **CSI**: `data/recordings/csi-1778091604.csi.jsonl` — 22,585 frames (5 nodes)
- **Camera**: `data/ground-truth/keypoints_20260506_202028.jsonl` — 4,160 frames
- **Paired**: `data/paired/csi-1778091604.paired.jsonl` — 858 samples

## Python Environment

- **Path**: `~/Cybersec/wireless/RuView/.venv` (uv-managed)
- **Key packages**: esptool 5.2.0, nvs-partition-gen 0.2.0, torch 2.11.0+cu130, mediapipe, opencv-python, numpy, huggingface-hub

## Known Bugs Fixed (in train-wiflow-supervised.js)

1. **NaN bug**: `reduceSubcarriers()` OOB access when sample has fewer subcarriers than topK
2. **Div-by-zero bug**: `epochsPerStage = floor(N/4) = 0` → `Math.floor(epoch/0) = NaN`
3. **Export architecture**: Now exports actual scale channels + normalization params instead of hardcoded values

## Next Steps

1. Fix the ESP32 node_id issue (all 5 should stream CSI with their unique node_id)
2. Verify TX is injecting probe frames (check if subcarrier count varies more on TX)
3. Re-collect training data with ALL 5 nodes streaming
4. Follow user guide for camera-supervised training pipeline
5. Train model, deploy via wiflow-live-bridge or Rust RVF format
