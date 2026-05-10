#!/usr/bin/env python3
"""PyTorch GPU training for WiFlow pose estimation from paired CSI+camera data.

Architecture mirrors train-wiflow-supervised.js but uses Adam + backpropagation
instead of SPSA, enabling fast convergence on GPU (16 GB VRAM recommended).

Usage:
    pip install torch numpy
    python train_wiflow_torch.py --data data/paired/ --epochs 200 --batch 64 --scale medium
"""

import argparse
import gzip
import json
import os
import math
import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Scale presets (matches JS version)
# ---------------------------------------------------------------------------
SCALES = {
    "lite":   dict(tcn_channels=[32, 32, 32, 32], hidden_dim=256,  tcn_blocks=2, kernel=3),
    "small":  dict(tcn_channels=[64, 64, 48, 32], hidden_dim=512,  tcn_blocks=4, kernel=5),
    "medium": dict(tcn_channels=[128, 128, 96, 64], hidden_dim=1024, tcn_blocks=4, kernel=7),
    "full":   dict(tcn_channels=[256, 256, 192, 128], hidden_dim=2048, tcn_blocks=4, kernel=7),
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.has_res = in_ch != out_ch
        if self.has_res:
            self.res_conv = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.has_res:
            res = self.res_conv(res)
        return x + res


class WiFlowModel(nn.Module):
    def __init__(self, input_dim, time_steps, num_keypoints=17, scale="lite"):
        super().__init__()
        cfg = SCALES[scale]
        ch = cfg["tcn_channels"][: cfg["tcn_blocks"]]
        kernel = cfg["kernel"]
        dilations = [1, 2, 4, 8][: cfg["tcn_blocks"]]

        blocks = []
        prev_ch = input_dim
        for i, (c, d) in enumerate(zip(ch, dilations)):
            blocks.append(TCNBlock(prev_ch, c, kernel, d))
            prev_ch = c
        self.tcn = nn.Sequential(*blocks)

        flat_dim = prev_ch * time_steps
        hidden = cfg["hidden_dim"]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, num_keypoints * 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, T]
        x = self.tcn(x)           # [B, last_ch, T]
        x = x.reshape(x.size(0), -1)  # [B, last_ch * T]
        return self.fc(x)          # [B, 34]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PairedCsiDataset(Dataset):
    def __init__(self, data_dir, time_steps=20, top_k=56, augment=False):
        super().__init__()
        self.samples = []
        self.time_steps = time_steps
        self.augment = augment

        files = sorted(glob(os.path.join(data_dir, "*.jsonl*")))
        if not files:
            files = sorted(glob(os.path.join(data_dir, "*.paired.jsonl*")))
        print(f"Found {len(files)} paired file(s)")

        for fp in files:
            opener = gzip.open if fp.endswith('.gz') else open
            with opener(fp, 'rt' if fp.endswith('.gz') else 'r') as f:
                for line in f:
                    d = json.loads(line)
                    csi = np.array(d["csi"], dtype=np.float32)
                    kp = d.get("keypoints") or d.get("kp")
                    if kp is None or len(kp) == 0:
                        continue
                    csi_shape = d.get("csi_shape", [len(csi) // time_steps, time_steps])
                    self.samples.append((csi, csi_shape, kp))

        print(f"Loaded {len(self.samples)} paired samples")

        # Compute per-subcarrier variance for top-K selection (on all data)
        all_variances = []
        first_shape = self.samples[0][1]
        input_dim = first_shape[0]
        T = first_shape[1]
        for d in range(input_dim):
            values = []
            for csi, shape, _ in self.samples:
                if d < shape[0]:
                    for t in range(T):
                        values.append(float(csi[d * T + t]))
            if values:
                arr = np.array(values, dtype=np.float32)
                all_variances.append((d, float(np.var(arr))))
            else:
                all_variances.append((d, 0.0))
        all_variances.sort(key=lambda x: -x[1])
        self.top_k_indices = [idx for idx, _ in all_variances[:top_k]]

        # Normalize CSI: z-score per subcarrier
        self.means = np.zeros(top_k, dtype=np.float32)
        self.stds = np.ones(top_k, dtype=np.float32)
        all_values = {k: [] for k in range(top_k)}
        for csi, shape, _ in self.samples:
            for k_idx, d in enumerate(self.top_k_indices):
                if d < shape[0]:
                    for t in range(T):
                        all_values[k_idx].append(float(csi[d * T + t]))
        for k_idx in range(top_k):
            vals = np.array(all_values[k_idx], dtype=np.float32)
            self.means[k_idx] = float(np.mean(vals))
            self.stds[k_idx] = float(np.std(vals)) + 1e-8

        print(f"Top-{top_k} subcarriers: {self.top_k_indices[:10]}...")
        print(f"CSI mean range: [{self.means.min():.1f}, {self.means.max():.1f}]")
        print(f"CSI std range:  [{self.stds.min():.1f}, {self.stds.max():.1f}]")
        print(f"Augmentation: {augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csi, shape, kp = self.samples[idx]
        T = self.time_steps

        # Select top-K subcarriers and normalize
        features = []
        for d in self.top_k_indices:
            if d < shape[0]:
                row = np.array([float(csi[d * T + t]) for t in range(T)], dtype=np.float32)
            else:
                row = np.zeros(T, dtype=np.float32)
            features.append(row)

        csi_mat = np.stack(features)  # [topK, T]
        for k in range(len(self.top_k_indices)):
            csi_mat[k] = (csi_mat[k] - self.means[k]) / self.stds[k]

        # Augmentation
        if self.augment:
            noise = np.random.randn(*csi_mat.shape).astype(np.float32) * 0.02
            csi_mat += noise
            if np.random.random() < 0.3:
                scale = np.random.uniform(0.8, 1.2)
                csi_mat *= scale

        csi_tensor = torch.from_numpy(csi_mat)       # [K, T]

        # Keypoints: 17 x [x, y] in [0, 1] (or beyond, clip to [0, 1])
        kp_arr = np.array(kp, dtype=np.float32)       # [17, 2] or flat
        if kp_arr.ndim == 1 and len(kp_arr) == 34:
            kp_arr = kp_arr.reshape(17, 2)
        kp_arr = np.clip(kp_arr, 0.0, 1.0)
        kp_flat = kp_arr.flatten()
        kp_tensor = torch.from_numpy(kp_flat)         # [34]

        return csi_tensor, kp_tensor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def pck_metric(pred, target, threshold=0.2):
    """PCK@threshold: percentage of keypoints within threshold * bbox_size."""
    B = pred.size(0)
    pred = pred.view(B, 17, 2)
    target = target.view(B, 17, 2)

    # Bounding box size from target
    t_max = target.amax(dim=1)  # [B, 2]
    t_min = target.amin(dim=1)  # [B, 2]
    bbox = (t_max - t_min).amax(dim=1)  # [B]

    dist = torch.norm(pred - target, dim=2)  # [B, 17]
    correct = (dist < threshold * bbox.unsqueeze(1)).float()
    return correct.mean().item()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = F.smooth_l1_loss(pred, y, beta=0.05)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_pck = 0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y, beta=0.05)
            total_loss += loss.item() * x.size(0)
            total_pck += pck_metric(pred, y) * x.size(0)
            count += x.size(0)
    return total_loss / count, total_pck / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Directory with paired JSONL files")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale", choices=list(SCALES), default="medium")
    parser.add_argument("--output", default="models/wiflow-torch")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Dataset
    train_ds = PairedCsiDataset(args.data, augment=True)
    eval_ds = PairedCsiDataset(args.data, augment=False)
    # Use same top-K indices and normalization for eval
    eval_ds.top_k_indices = train_ds.top_k_indices
    eval_ds.means = train_ds.means
    eval_ds.stds = train_ds.stds

    # Split: 80/20
    n = len(train_ds)
    indices = torch.randperm(n).tolist()
    split = int(n * 0.8)
    train_ds.samples = [train_ds.samples[i] for i in indices[:split]]
    eval_ds.samples = [eval_ds.samples[i] for i in indices[split:]]
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch * 2, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Model
    top_k = len(train_ds.top_k_indices)
    T = train_ds.time_steps
    model = WiFlowModel(input_dim=top_k, time_steps=T, scale=args.scale).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.scale} scale, {n_params:,} parameters")
    print(f"Input: [{top_k}, {T}]")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_pck = 0
    best_path = None
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'Epoch':>6} {'Train Loss':>10} {'Eval Loss':>10} {'PCK@20':>8} {'LR':>10} {'Time':>8}")
    print("-" * 60)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        eval_loss, pck = evaluate(model, eval_loader, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start

        print(f"{epoch:>6} {train_loss:>10.4f} {eval_loss:>10.4f} {pck*100:>7.1f}% {lr:>10.2e} {elapsed:>7.0f}s")

        if pck > best_pck:
            best_pck = pck
            best_path = os.path.join(args.output, f"model_best_pck{pck*100:.1f}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "top_k_indices": train_ds.top_k_indices,
                "means": train_ds.means.tolist(),
                "stds": train_ds.stds.tolist(),
                "scale": args.scale,
                "pck": pck,
                "epoch": epoch,
            }, best_path)

        # Save periodic checkpoint
        if epoch % 50 == 0:
            ckpt_path = os.path.join(args.output, f"checkpoint_{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "top_k_indices": train_ds.top_k_indices,
                "means": train_ds.means.tolist(),
                "stds": train_ds.stds.tolist(),
                "scale": args.scale,
                "epoch": epoch,
            }, ckpt_path)

    print(f"\nBest PCK@20: {best_pck*100:.1f}% ({best_path})")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
