#!/usr/bin/env python3
"""
inference_precip_fm_v1.py
--------------------------
Inference for Flow Matching v1 (10× downscaling, 4-level U-Net,
1.0° → 0.1°). Drop-in counterpart to inference_precip_ddpm_v5b.py:
identical CLI, identical full-field inference design, identical
output format — only the generative core (DDIM → FM ODE) is changed.

Two modes
---------
mswx  — perfect-model evaluation.
          LR generated from MSWX HR via full-field avg_pool(10) → bicubic.
          Matches training exactly.

cmip6 — real application.
          LR is the QDM-corrected CMIP6 field already interpolated to 0.1°.
          (Your pipeline: GCM → regrid to 1.0° → QDM → bilinear to 0.1°)

Key design: LR is built on the FULL 350×350 field before being fed to the
network, matching the full-field strategy used during training. The full
field is reflect-padded to a size divisible by 16 (4 stride-2 downsamples
in the 4-level U-Net), processed in one forward pass per ODE step, then
unpadded to the original size.

Differences vs the DDPM v5b inference (everything else identical)
-----------------------------------------------------------------
  • NoiseScheduler.ddim_step  →  FlowMatching ODE solver (heun/midpoint/euler)
  • --ddim_steps              →  --fm_steps  (+ --fm_solver)
  • t ∈ {0,…,999}             →  t ∈ [0,1], scaled by t_scale=1000 internally
  • Single-member sampling    →  Optional --n_ensemble for stochastic ensembles
                                   (default 1 reproduces DDPM behavior)
  • Padding divisor 8 (v5)    →  16 (v5b: 4 stride-2 downsamples). For 350×350
                                   both yield the same pad (2 px), but 16 is
                                   the architecturally correct value.

Usage — perfect-model
---------------------
  python inference_precip_fm_v1.py \
      --checkpoint results/precip_fm_v1/best.pt \
      --mode mswx \
      --input_h5 /path/to/mswx_test_2011-2014.h5 \
      --output_dir results/precip_fm_v1/inference/test_set \
      --fm_steps 50 --fm_solver heun --gpu 0

Usage — CMIP6
-------------
  python inference_precip_fm_v1.py \
      --checkpoint results/precip_fm_v1/best.pt \
      --mode cmip6 \
      --input_npy /path/to/qdm_cnrm_cm6_1_daily_test.npy \
      --ref_h5    /path/to/mswx_test_2011-2014.h5 \
      --output_dir results/precip_fm_v1/inference/cmip6_cnrm_test \
      --fm_steps 50 --fm_solver heun --gpu 0

Usage — ensemble for UQ (stochastic over noise z)
-------------------------------------------------
  python inference_precip_fm_v1.py \
      --checkpoint results/precip_fm_v1/best.pt \
      --mode mswx \
      --input_h5  /path/to/mswx_test_2011-2014.h5 \
      --output_dir results/precip_fm_v1/inference/test_set_ens \
      --fm_steps 50 --fm_solver heun --n_ensemble 20 --seed 42 --gpu 0
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

# Import everything needed from the FM training script.
# compute_lr_fullfield handles full-field avg_pool — no per-patch ops.
from train_precip_fm_v1 import (
    Config,
    PrecipNormalizer,
    PrecipUNet,
    FlowMatching,
    compute_lr_fullfield,
)


# ─────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', 
                    default='/path/to/project/Flow_matching_downscaling/results/precip_fm_v1/best.pt',
                    help='Path to best.pt')
parser.add_argument('--mode', default='mswx', choices=['mswx', 'cmip6'],
                    help='mswx: perfect-model | cmip6: QDM field input')
parser.add_argument('--input_h5', 
                    default='/path/to/project/PhD_Precipitation/02_Data/processed/hdf5/mswx_test_2011-2014.h5',
                    help='[mswx] MSWX HDF5 path')
parser.add_argument('--input_npy', default=None,
                    help='[cmip6] QDM daily .npy path — shape [T, 350, 350] mm/day')
parser.add_argument('--ref_h5', default=None,
                    help='[cmip6] MSWX HDF5 used only for lat / lon / dates')
parser.add_argument('--output_dir', 
                    default='/path/to/project/Flow_matching_downscaling/results/precip_fm_v1/inference/test_mswx_heun50',
                    help='Directory for output .nc files')
# ── FM-specific sampling args (replace --ddim_steps) ──
parser.add_argument('--fm_steps', type=int, default=50,
                    help='Number of ODE steps. Heun/midpoint cost 2 NFE/step; '
                         'euler costs 1 NFE/step. fm_steps=50 + heun → 100 NFE '
                         '(comparable footprint to DDIM-50 single-NFE).')
parser.add_argument('--fm_solver', default='heun',
                    choices=['euler', 'midpoint', 'heun'],
                    help='ODE solver. Default heun (2nd-order, robust).')
# ── Ensemble for uncertainty quantification ──
parser.add_argument('--n_ensemble', type=int, default=1,
                    help='Number of stochastic members per day. >1 adds a '
                         '`member` dim to the output NetCDF. Each member uses '
                         'a different noise z; the LR condition is shared.')
parser.add_argument('--seed', type=int, default=42,
                    help='Base RNG seed. Per-(day, member) seeds are derived '
                         'deterministically from this.')
# ── System ──
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--max_days', type=int, default=None,
                    help='Limit number of days (debugging)')
parser.add_argument('--amp', action='store_true',
                    help='Run forward passes in autocast fp16 (saves memory '
                         'on big full-field attention). Off by default to '
                         'mirror DDPM v5b inference.')
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

# Read FM-specific fields with sensible fallbacks (for forward compatibility
# with any future trainer that omits them).
t_scale_ckpt  = float(ckpt.get('t_scale', config.t_scale))
framework_tag = ckpt.get('framework', '(none)')

print(f"  Framework:    {framework_tag}")
print(f"  Epoch:        {ckpt.get('epoch', '?')}")
print(f"  Best val:     {ckpt.get('best_val', 0):.5f}")
print(f"  Scale factor: {ckpt.get('scale_factor', 10)}×")
print(f"  Used EMA wts: {'yes' if 'ema_state_dict' in ckpt else 'no'}")
if framework_tag not in ('flow_matching_v1',):
    print(f"  ⚠  Warning: checkpoint framework tag is '{framework_tag}', "
          f"not 'flow_matching_v1'. Verify the weights were trained with "
          f"velocity-matching loss, not noise-prediction.")

# ─────────────────────────────────────────────────────────────
# FLOW MATCHING (replaces NoiseScheduler + DDIM sequence)
# ─────────────────────────────────────────────────────────────
fm = FlowMatching(t_scale=t_scale_ckpt, device=device)
print(f"  FM:           solver={args.fm_solver}, steps={args.fm_steps}, "
      f"t_scale={t_scale_ckpt}")
print(f"  NFE/sample:   {args.fm_steps * (1 if args.fm_solver == 'euler' else 2)} "
      f"(per ensemble member)")

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

print(f"\n  Processing {T} days × {args.n_ensemble} member(s) "
      f"→ {args.output_dir}\n")

# ─────────────────────────────────────────────────────────────
# PADDING HELPERS
# ─────────────────────────────────────────────────────────────
# v5b U-Net has 4 stride-2 downsamples → spatial dims must be divisible
# by 2^4 = 16. For H=W=350, this requires padding to 352 (pad 1 px on
# each side). The v5b DDPM inference script used divisor 8 (a leftover
# comment from v5 3-level), which happens to give the same result for
# 350 but is architecturally incorrect — fixed here.
UNET_DIVISOR = 16   # 2 ** num_downsample_levels (4 for v5b)


def get_padding(H: int, W: int):
    """Reflect-pad to make H, W divisible by UNET_DIVISOR. Returns (pt,pb,pl,pr)."""
    pad_h = (UNET_DIVISOR - H % UNET_DIVISOR) % UNET_DIVISOR
    pad_w = (UNET_DIVISOR - W % UNET_DIVISOR) % UNET_DIVISOR
    pt    = pad_h // 2
    pb    = pad_h - pt
    pl    = pad_w // 2
    pr    = pad_w - pl
    return pt, pb, pl, pr


pt, pb, pl, pr = get_padding(H, W)
if pt or pb or pl or pr:
    print(f"  U-Net padding: top={pt} bottom={pb} left={pl} right={pr} "
          f"(reflect mode, divisor={UNET_DIVISOR})")
else:
    print(f"  U-Net padding: none needed ({H}×{W} divisible by {UNET_DIVISOR})")

# ─────────────────────────────────────────────────────────────
# FM ODE SAMPLING (replaces DDIM reverse diffusion)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def run_fm_ode(lr_norm_padded: torch.Tensor,
               noise:          torch.Tensor) -> torch.Tensor:
    """
    Integrate the velocity ODE  dx/dt = v_theta(cat([x, c], 1), t)
    from t=0 (Gaussian noise) to t=1 (HR sample).

    Args:
        lr_norm_padded: [1, 1, H_pad, W_pad] normalized LR condition on device
        noise:          [1, 1, H_pad, W_pad] starting Gaussian on device

    Returns:
        x1: [1, 1, H_pad, W_pad] generated HR in normalized space
    """
    # FlowMatching.sample handles solver dispatch + t-scaling internally,
    # so the math here is identical to fm_loss interpretation at inference.
    if args.amp:
        with torch.amp.autocast('cuda'):
            x1 = fm.sample(
                model     = model,
                cond      = lr_norm_padded,
                shape     = lr_norm_padded.shape,
                num_steps = args.fm_steps,
                solver    = args.fm_solver,
                device    = device,
                noise     = noise,
            )
    else:
        x1 = fm.sample(
            model     = model,
            cond      = lr_norm_padded,
            shape     = lr_norm_padded.shape,
            num_steps = args.fm_steps,
            solver    = args.fm_solver,
            device    = device,
            noise     = noise,
        )
    return x1


# ─────────────────────────────────────────────────────────────
# PER-DAY INFERENCE
# ─────────────────────────────────────────────────────────────
def infer_day(lr_field: np.ndarray, day_idx: int) -> np.ndarray:
    """
    Run FM ODE inference for one day, possibly for multiple ensemble members.

    Args:
        lr_field: [H, W] float32, mm/day — full-field LR for this day.
                  For mswx mode: pre-computed avg_pool(10) from HR.
                  For cmip6 mode: QDM-corrected CMIP6 field at 0.1°.
        day_idx:  used to seed the noise for reproducibility.

    Returns:
        pred: float32 array, mm/day, clipped >= 0.
              shape [H, W]          if n_ensemble == 1  (DDPM-compatible)
              shape [M, H, W]       if n_ensemble  > 1  (M = n_ensemble)
    """
    # ── To tensor, pad, normalize once (shared across members) ──
    lr_t = (torch.from_numpy(lr_field)
            .unsqueeze(0).unsqueeze(0)               # [1,1,H,W]
            .float().to(device))
    lr_t    = F.pad(lr_t, (pl, pr, pt, pb), mode='reflect')   # [1,1,H_pad,W_pad]
    lr_norm = normalizer.normalize_torch(lr_t)

    # ── Sample n_ensemble members with deterministic per-(day,member) seeds ──
    members = []
    for m in range(args.n_ensemble):
        # Independent seeds across (day, member) → reproducible ensembles
        member_seed = args.seed * 1_000_003 + day_idx * 1009 + m
        gen   = torch.Generator(device=device).manual_seed(member_seed)
        noise = torch.randn(lr_norm.shape, generator=gen,
                            device=device, dtype=lr_norm.dtype)

        x_hat = run_fm_ode(lr_norm, noise)

        # Clamp normalized output and denormalize
        pred = (normalizer.denormalize_torch(x_hat.clamp(-1, 1))
                .squeeze()
                .cpu()
                .numpy())                                # [H_pad, W_pad]
        pred_unpadded = pred[pt:pt+H, pl:pl+W].clip(0).astype(np.float32)
        members.append(pred_unpadded)

    if args.n_ensemble == 1:
        return members[0]                                # [H, W]
    return np.stack(members, axis=0)                     # [M, H, W]


# ─────────────────────────────────────────────────────────────
# NETCDF WRITER
# ─────────────────────────────────────────────────────────────
def write_nc(pred: np.ndarray, out_path: str, ckpt: dict):
    """
    Write daily prediction to NetCDF.

    pred shape:
        [H, W]    → 2D field (single member)
        [M, H, W] → 3D field with `member` coord (ensemble)
    """
    attrs = {
        'long_name':        'Daily precipitation (Flow Matching v1 downscaled)',
        'units':            'mm day-1',
        'model':            'FM v1 — 10x 4-level UNet (1.0 deg -> 0.1 deg)',
        'mode':             args.mode,
        # NOTE: 'downscale_factor' (not 'scale_factor') is the v5b convention
        # to avoid clashes with xarray's CF decode_cf 'scale_factor' attr.
        'downscale_factor': str(config.scale_factor),
        'fm_steps':         str(args.fm_steps),
        'fm_solver':        args.fm_solver,
        'n_ensemble':       str(args.n_ensemble),
        'best_epoch':       str(ckpt.get('epoch', '?')),
        'framework':        ckpt.get('framework', '(none)'),
    }
    if pred.ndim == 2:
        ds = xr.Dataset(
            {'precipitation': (['lat', 'lon'], pred)},
            coords={'lat': lat, 'lon': lon}
        )
    else:  # ensemble
        ds = xr.Dataset(
            {'precipitation': (['member', 'lat', 'lon'], pred)},
            coords={
                'member': np.arange(pred.shape[0], dtype=np.int32),
                'lat':    lat,
                'lon':    lon,
            }
        )
    ds['precipitation'].attrs = attrs
    ds.to_netcdf(out_path)


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
print(f"Running FM ODE inference "
      f"({args.fm_solver}, {args.fm_steps} steps, "
      f"mode={args.mode}, ensemble={args.n_ensemble})...")

# Seed global RNG once for any non-Generator-routed randomness
torch.manual_seed(args.seed)
np.random.seed(args.seed)

for i in tqdm(range(T), desc="Days"):
    pred  = infer_day(lr_data[i], day_idx=i)
    fname = dates[i].strftime('%Y%j') + '.nc'
    write_nc(pred, os.path.join(args.output_dir, fname), ckpt)

print(f"\nDone. {T} files → {args.output_dir}")
if args.n_ensemble > 1:
    print(f"  Each file contains {args.n_ensemble} ensemble members "
          f"under the `member` dimension.")