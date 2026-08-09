#!/usr/bin/env python3
"""
inference_precip_ddpm_v5b.py
-----------------------------
Inference for DDPM v5b (10× downscaling, 4-level U-Net, 1.0° → 0.1°).
Identical to v5 inference except model import and ch_mult_cap argument.

Two modes
---------
mswx  — perfect-model evaluation.
          LR generated from MSWX HR via full-field avg_pool(10) → bicubic.
          Matches training exactly.

cmip6 — real application.
          LR is the QDM-corrected CMIP6 field already interpolated to 0.1°.
          (Your pipeline: GCM → regrid to 1.0° → QDM → bilinear to 0.1°)

Key design: LR is always built on the FULL 350×350 field before patches are
extracted, matching the full-field strategy used during training. This avoids
the avg_pool boundary misalignment that would occur if done per-patch.

Usage — perfect-model
---------------------
  python inference_precip_ddpm_v5b.py \\
      --checkpoint results/precip_ddpm_v5b/best.pt \\
      --mode mswx \\
      --input_h5 /path/to/mswx_test_2011-2014.h5 \\
      --output_dir results/precip_ddpm_v5b/inference/test_set \\
      --ddim_steps 50 --gpu 0

Usage — CMIP6
-------------
  python inference_precip_ddpm_v5b.py \\
      --checkpoint results/precip_ddpm_v5b/best.pt \\
      --mode cmip6 \\
      --input_npy /path/to/qdm_cnrm_cm6_1_daily_test.npy \\
      --ref_h5 /path/to/mswx_test_2011-2014.h5 \\
      --output_dir results/precip_ddpm_v5b/inference/cmip6_cnrm_test \\
      --ddim_steps 50 --gpu 0
"""

import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import everything needed from the training script.
# compute_lr_fullfield handles full-field avg_pool — no per-patch ops.
from train_precip_ddpm_v5b import (
    Config,
    PrecipNormalizer,
    PrecipUNet,
    NoiseScheduler,
    compute_lr_fullfield,
)


# ─────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True,
                    help='Path to best.pt')
parser.add_argument('--mode', required=True, choices=['mswx', 'cmip6'],
                    help='mswx: perfect-model | cmip6: QDM field input')
parser.add_argument('--input_h5', default=None,
                    help='[mswx] MSWX HDF5 path')
parser.add_argument('--input_npy', default=None,
                    help='[cmip6] QDM daily .npy path — shape [T, 350, 350] mm/day')
parser.add_argument('--ref_h5', default=None,
                    help='[cmip6] MSWX HDF5 used only for lat / lon / dates')
parser.add_argument('--output_dir', required=True,
                    help='Directory for output .nc files')
parser.add_argument('--ddim_steps', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--max_days', type=int, default=None,
                    help='Limit number of days (debugging)')
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
device = torch.device(f'cuda:{args.gpu}')

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
config     = Config()
normalizer = PrecipNormalizer(config.log1p_max)

model = PrecipUNet(
    in_channels=2, out_channels=1,
    base_dim=config.base_dim,
    ch_mult_cap=config.ch_mult_cap,
    time_dim=config.time_dim,
).to(device)

print(f"Loading checkpoint: {args.checkpoint}")
ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
state = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
clean = {k.replace('module.', ''): v for k, v in state.items()}
model.load_state_dict(clean)
model.eval()
print(f"  Epoch:        {ckpt.get('epoch', '?')}")
print(f"  Best val:     {ckpt.get('best_val', 0):.5f}")
print(f"  Scale factor: {ckpt.get('scale_factor', 10)}×")

# ─────────────────────────────────────────────────────────────
# NOISE SCHEDULER + DDIM SEQUENCE
# ─────────────────────────────────────────────────────────────
sched = NoiseScheduler(
    config.t_steps, config.beta_start, config.beta_end, device
)
t_seq = np.linspace(config.t_steps - 1, 0, args.ddim_steps, dtype=int)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
if args.mode == 'mswx':
    assert args.input_h5, "--input_h5 required for mswx mode"
    print(f"\nMode: MSWX perfect-model")
    print(f"Loading: {args.input_h5}")
    with h5py.File(args.input_h5, 'r') as f:
        hr_data = f['precipitation'][:]   # [T, H, W] mm/day
        lat     = f['lat'][:]
        lon     = f['lon'][:]
        times   = f['time'][:]
    T, H, W = hr_data.shape
    dates   = pd.to_datetime(times)
    print(f"  HR shape: {hr_data.shape}")
    print(f"  Range:    [{hr_data.min():.3f}, {hr_data.max():.3f}] mm/day")

    # Build full-field LR for all days — same as training
    # 350 % 10 = 0 → exact, no dropped pixels
    assert H % config.scale_factor == 0, (
        f"H={H} not divisible by scale_factor={config.scale_factor}"
    )
    assert W % config.scale_factor == 0, (
        f"W={W} not divisible by scale_factor={config.scale_factor}"
    )
    print(f"\n  Pre-computing full-field LR (scale={config.scale_factor}×)...")
    print(f"  avg_pool({config.scale_factor}) on {H}×{W} → "
          f"{H//config.scale_factor}×{W//config.scale_factor} → "
          f"bicubic back to {H}×{W}")

    lr_data  = np.zeros_like(hr_data, dtype=np.float32)
    batch_sz = 50
    for t_start in tqdm(range(0, T, batch_sz), desc="  Computing LR"):
        t_end = min(t_start + batch_sz, T)
        lr_data[t_start:t_end] = compute_lr_fullfield(
            hr_data[t_start:t_end].astype(np.float32),
            config.scale_factor, H, W
        )
    print(f"  LR range: [{lr_data.min():.3f}, {lr_data.max():.3f}] mm/day")

elif args.mode == 'cmip6':
    assert args.input_npy, "--input_npy required for cmip6 mode"
    assert args.ref_h5,    "--ref_h5 required for cmip6 mode (lat/lon/dates)"
    print(f"\nMode: CMIP6 application (QDM field at 0.1°)")
    print(f"Loading QDM field: {args.input_npy}")
    lr_data = np.load(args.input_npy).astype(np.float32)   # [T, 350, 350]
    T, H, W = lr_data.shape
    print(f"  QDM shape: {lr_data.shape}")
    print(f"  QDM range: [{lr_data.min():.3f}, {lr_data.max():.3f}] mm/day")
    print(f"Loading ref HDF5 for coordinates: {args.ref_h5}")
    with h5py.File(args.ref_h5, 'r') as f:
        lat   = f['lat'][:]
        lon   = f['lon'][:]
        times = f['time'][:]
    dates   = pd.to_datetime(times)
    hr_data = None   # not used in cmip6 mode

if args.max_days:
    T = min(T, args.max_days)
    print(f"  max_days: capped at {T}")

print(f"\n  Processing {T} days → {args.output_dir}\n")

# ─────────────────────────────────────────────────────────────
# PADDING HELPERS
# ─────────────────────────────────────────────────────────────
def get_padding(H: int, W: int):
    """
    Compute reflect padding to make H, W divisible by 8.
    The U-Net has 3 stride-2 downsamples, requiring dims divisible by 8.
    350 % 8 = 6 → needs 2px padding each side (pt=1, pb=1 ... etc).
    Returns (pt, pb, pl, pr).
    """
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    pt    = pad_h // 2
    pb    = pad_h - pt
    pl    = pad_w // 2
    pr    = pad_w - pl
    return pt, pb, pl, pr


pt, pb, pl, pr = get_padding(H, W)
if pt or pb or pl or pr:
    print(f"  U-Net padding: top={pt} bottom={pb} left={pl} right={pr} "
          f"(reflect mode)")
else:
    print(f"  U-Net padding: none needed ({H}×{W} divisible by 8)")

# ─────────────────────────────────────────────────────────────
# DDIM REVERSE DIFFUSION
# ─────────────────────────────────────────────────────────────
def run_ddim(lr_norm_padded: torch.Tensor) -> torch.Tensor:
    """
    Run DDIM reverse diffusion.

    Args:
        lr_norm_padded: [1, 1, H_pad, W_pad] normalized LR on device

    Returns:
        x: [1, 1, H_pad, W_pad] denoised output in normalized space
    """
    x = torch.randn_like(lr_norm_padded)
    with torch.no_grad():
        for j, t in enumerate(t_seq):
            t_prev     = t_seq[j + 1] if j + 1 < len(t_seq) else -1
            t_tensor   = torch.tensor([t], device=device)
            noise_pred = model(torch.cat([x, lr_norm_padded], dim=1), t_tensor)
            x          = sched.ddim_step(x, noise_pred, t, t_prev)
    return x


# ─────────────────────────────────────────────────────────────
# PER-DAY INFERENCE
# ─────────────────────────────────────────────────────────────
def infer_day(lr_field: np.ndarray) -> np.ndarray:
    """
    Run inference for one day.

    Args:
        lr_field: [H, W] float32, mm/day — full-field LR for this day.
                  For mswx mode: pre-computed avg_pool(10) from HR.
                  For cmip6 mode: QDM-corrected CMIP6 field at 0.1°.

    Returns:
        pred: [H, W] float32, mm/day — DDPM downscaled output, clipped >= 0.
    """
    # To tensor and pad
    lr_t = (torch.from_numpy(lr_field)
            .unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            .float().to(device))
    lr_t = F.pad(lr_t, (pl, pr, pt, pb), mode='reflect')   # [1,1,H_pad,W_pad]

    # Normalize to [-1, 1]
    lr_norm = normalizer.normalize_torch(lr_t)

    # DDIM reverse diffusion
    x_hat = run_ddim(lr_norm)

    # Denormalize and unpad
    pred = (normalizer.denormalize_torch(x_hat.clamp(-1, 1))
            .squeeze()
            .cpu()
            .numpy())                    # [H_pad, W_pad]

    # Remove padding to recover original [H, W]
    pred_unpadded = pred[pt:pt+H, pl:pl+W]

    return pred_unpadded.clip(0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
print(f"Running DDIM inference ({args.ddim_steps} steps, mode={args.mode})...")

for i in tqdm(range(T), desc="Days"):
    pred  = infer_day(lr_data[i])
    fname = dates[i].strftime('%Y%j') + '.nc'

    ds = xr.Dataset(
        {'precipitation': (['lat', 'lon'], pred)},
        coords={'lat': lat, 'lon': lon}
    )
    ds['precipitation'].attrs = {
        'long_name':      'Daily precipitation (DDPM v5b downscaled)',
        'units':          'mm day-1',
        'model':          'DDPM v5b — 10x 4-level UNet (1.0 deg -> 0.1 deg)',
        'mode':           args.mode,
        'downscale_factor': str(config.scale_factor),
        'ddim_steps':     str(args.ddim_steps),
        'best_epoch':     str(ckpt.get('epoch', '?')),
    }
    ds.to_netcdf(os.path.join(args.output_dir, fname))

print(f"\nDone. {T} files → {args.output_dir}")