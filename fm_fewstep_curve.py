#!/usr/bin/env python3
"""
fm_fewstep_curve.py
-------------------
Few-step sampling cost/quality curve for Flow Matching (FM) precipitation
downscaling, benchmarked against DDPM (DDIM) at *matched NFE*.

WHY THIS SCRIPT EXISTS
----------------------
The headline FM result is evaluated at Heun-50 = 100 NFE, i.e. ~2x the cost
of DDPM DDIM-50 (50 NFE), for a marginal quality gain. That makes FM look
expensive-for-nothing. The *point* of a rectified / straight-line flow is that
the probability path is (near-)linear, so Euler with very few steps should stay
accurate. This script measures exactly that: it sweeps the number of ODE steps
and the solver, computes a key fidelity metric at each setting, and overlays the
DDPM DDIM curve at the same NFE budget. If FM-Euler holds up at 4-16 NFE while
DDIM degrades, FM is *cheaper AND competitive*, which is the real contribution.

WHAT IT COMPUTES (per sampler configuration)
--------------------------------------------
  - Wasserstein-1 distance of pooled wet-day intensities  (primary; mirrors the
    distribution-distance metric in the analysis notebook)
  - 100-yr empirical-GEV return-level relative bias        (extremes headline)
  - P99.9 wet-day quantile ratio                           (deep-tail check)
  - RMSE of the time-mean field                            (climatology sanity)
  - Empirical wall-clock seconds / sample                  (true cost, not just NFE)

All statistics are accumulated ONLINE while sampling (running per-cell annual
maxima [4,H,W], running sum for the mean map, and a Bernoulli-subsampled pool of
wet-day intensities). Nothing of size [T,H,W] is ever held in memory or cached
to disk, so the sweep scales to the full 1461-day test set on a single GPU.

PERFECT-MODEL FRAMEWORK
-----------------------
LR is built once from the MSWX HR test field via the shared
compute_lr_fullfield (avg_pool(10) -> bicubic), identical to training and to
both inference scripts. FM and DDPM therefore see the *same* LR condition and
the *same* per-day initial noise, so the comparison isolates the generative
core (objective + solver), not the data path.

ENVIRONMENT NOTE
----------------
The pytorch_env needs the libstdc++ preload before xarray/pandas import:
    export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

USAGE
-----
  # Full sweep (default configs), full test set:
  python fm_fewstep_curve.py \
      --fm_checkpoint   results/precip_fm_v1/best.pt \
      --ddpm_checkpoint results/precip_ddpm_v5b/best.pt \
      --input_h5        .../mswx_test_2011-2014.h5 \
      --output_dir      results/precip_fm_v1/fewstep_curve --gpu 0

  # Quick smoke test on a temporal subset (RETURN LEVELS WILL BE UNRELIABLE):
  python fm_fewstep_curve.py ... --max_days 120

  # Custom step grids:
  python fm_fewstep_curve.py ... \
      --fm_euler_steps 2 4 8 16 32 \
      --fm_heun_steps  4 8 16 25 50 \
      --ddim_steps     4 8 16 25 32 50 100
"""

import os
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import genextreme, wasserstein_distance
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# h5py / pandas imported after torch to play nicely with LD_PRELOAD setups
import h5py
import pandas as pd

# ── Shared building blocks from the two trainers ──────────────────────────────
# FM side
from train_precip_fm_v1 import (
    Config        as FMConfig,
    PrecipNormalizer,
    PrecipUNet    as FMUNet,
    FlowMatching,
    compute_lr_fullfield,
)
# DDPM side (architecture identical; scheduler differs)
from train_precip_ddpm_v5b import (
    Config        as DDPMConfig,
    PrecipUNet    as DDPMUNet,
    NoiseScheduler,
)

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fm_checkpoint', required=True, help='FM best.pt')
    p.add_argument('--ddpm_checkpoint', required=True, help='DDPM v5b best.pt')
    p.add_argument('--input_h5', required=True,
                   help='MSWX HR test HDF5 (perfect-model reference)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max_days', type=int, default=None,
                   help='Cap number of days (debug). Return levels need full '
                        'annual coverage, so only use for smoke tests.')
    p.add_argument('--amp', action='store_true',
                   help='Run forward passes in fp16 autocast.')

    # Step grids (NFE = steps for euler/ddim; 2*steps for heun/midpoint)
    p.add_argument('--fm_euler_steps', type=int, nargs='+',
                   default=[2, 4, 8, 16, 32])
    p.add_argument('--fm_heun_steps', type=int, nargs='+',
                   default=[4, 8, 16, 25, 50])
    p.add_argument('--fm_midpoint_steps', type=int, nargs='+', default=[],
                   help='Optional midpoint-solver grid (2 NFE/step).')
    p.add_argument('--ddim_steps', type=int, nargs='+',
                   default=[4, 8, 16, 25, 32, 50, 100])

    # Metric / accumulation controls
    p.add_argument('--wet_threshold', type=float, default=1.0,
                   help='mm/day; wet-day definition for intensity pooling.')
    p.add_argument('--wet_keep_prob', type=float, default=0.10,
                   help='Bernoulli keep prob for pooling wet-day intensities '
                        '(memory control; applied identically to obs & models).')
    p.add_argument('--return_period_years', type=float, default=100.0,
                   help='Target return period for the RL-bias metric.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# PADDING (divisor 16 = 2**4 downsamples in the v5b 4-level U-Net)
# ──────────────────────────────────────────────────────────────────────────────
UNET_DIVISOR = 16


def get_padding(H, W):
    pad_h = (UNET_DIVISOR - H % UNET_DIVISOR) % UNET_DIVISOR
    pad_w = (UNET_DIVISOR - W % UNET_DIVISOR) % UNET_DIVISOR
    pt = pad_h // 2
    pl = pad_w // 2
    return pt, pad_h - pt, pl, pad_w - pl


# ──────────────────────────────────────────────────────────────────────────────
# ONLINE STATISTICS ACCUMULATOR
# ──────────────────────────────────────────────────────────────────────────────
class FieldStats:
    """
    Accumulates, in O(H*W) memory, everything the fidelity metrics need:
      - running sum of the field   -> time-mean map
      - per-year per-cell maximum  -> pooled annual maxima for the GEV fit
      - Bernoulli-subsampled wet-day intensity pool -> Wasserstein, P99.9
    """
    def __init__(self, H, W, year_index, n_years, wet_threshold,
                 keep_prob, rng):
        self.H, self.W = H, W
        self.year_index = year_index            # [T] -> 0..n_years-1
        self.n_years = n_years
        self.wet_threshold = wet_threshold
        self.keep_prob = keep_prob
        self.rng = rng
        self.sum_map = np.zeros((H, W), np.float64)
        self.n_days = 0
        self.ann_max = np.full((n_years, H, W), -np.inf, np.float32)
        self._wet_chunks = []

    def update(self, field2d, day_idx):
        self.sum_map += field2d
        self.n_days += 1
        y = self.year_index[day_idx]
        np.maximum(self.ann_max[y], field2d, out=self.ann_max[y])
        wet = field2d[field2d >= self.wet_threshold]
        if wet.size:
            mask = self.rng.random(wet.size) < self.keep_prob
            kept = wet[mask]
            if kept.size:
                self._wet_chunks.append(kept.astype(np.float32))

    def finalize(self):
        self.mean_map = (self.sum_map / max(self.n_days, 1)).astype(np.float32)
        self.wet_pool = (np.concatenate(self._wet_chunks)
                         if self._wet_chunks else np.array([], np.float32))
        # pooled annual maxima across all valid cells & years
        am = self.ann_max
        self.annual_maxima = am[np.isfinite(am)].astype(np.float32)
        self._wet_chunks = None
        return self


def gev_return_level(annual_maxima, period_years):
    """Fit GEV (scipy genextreme) to pooled annual maxima; return RL at T years.
    isf(1/T) gives the (1-1/T) quantile directly, independent of the xi sign
    convention. Returns (rl, shape_c, loc, scale)."""
    am = annual_maxima[np.isfinite(annual_maxima)]
    am = am[am > 0]
    if am.size < 50:
        return np.nan, np.nan, np.nan, np.nan
    c, loc, scale = genextreme.fit(am)
    rl = float(genextreme.isf(1.0 / period_years, c, loc=loc, scale=scale))
    return rl, float(c), float(loc), float(scale)


def metrics_vs_obs(stats, obs, period_years):
    """Compute the fidelity metrics for one model's FieldStats vs obs stats."""
    # Wasserstein-1 on wet-day intensities
    if stats.wet_pool.size and obs.wet_pool.size:
        w1 = float(wasserstein_distance(stats.wet_pool, obs.wet_pool))
        p999_mod = float(np.percentile(stats.wet_pool, 99.9))
        p999_obs = float(np.percentile(obs.wet_pool, 99.9))
        p999_ratio = p999_mod / p999_obs if p999_obs > 0 else np.nan
    else:
        w1 = p999_ratio = np.nan

    # 100-yr GEV return-level relative bias
    rl_mod, c_mod, _, _ = gev_return_level(stats.annual_maxima, period_years)
    rl_obs, c_obs, _, _ = gev_return_level(obs.annual_maxima, period_years)
    rl_relbias = (100.0 * (rl_mod - rl_obs) / rl_obs
                  if (rl_obs and np.isfinite(rl_obs)) else np.nan)

    # time-mean RMSE
    valid = np.isfinite(stats.mean_map) & np.isfinite(obs.mean_map)
    mean_rmse = float(np.sqrt(np.mean(
        (stats.mean_map[valid] - obs.mean_map[valid]) ** 2)))

    return dict(
        wasserstein=w1,
        rl100_relbias_pct=rl_relbias,
        rl100_value=rl_mod,
        gev_shape_c=c_mod,
        p999_ratio=p999_ratio,
        mean_rmse=mean_rmse,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SAMPLERS
# ──────────────────────────────────────────────────────────────────────────────
def make_day_noise(shape, day_idx, base_seed, device, dtype):
    """Deterministic per-day initial noise, shared across ALL configs so the
    only thing that varies between curves is the solver/step budget."""
    g = torch.Generator(device=device).manual_seed(base_seed * 7919 + day_idx)
    return torch.randn(shape, generator=g, device=device, dtype=dtype)


@torch.no_grad()
def fm_sample_day(fm, model, lr_norm, noise, steps, solver, amp, device):
    if amp:
        with torch.amp.autocast('cuda'):
            return fm.sample(model=model, cond=lr_norm, shape=lr_norm.shape,
                             num_steps=steps, solver=solver, device=device,
                             noise=noise)
    return fm.sample(model=model, cond=lr_norm, shape=lr_norm.shape,
                     num_steps=steps, solver=solver, device=device, noise=noise)


@torch.no_grad()
def ddim_sample_day(model, sched, lr_norm, noise, steps, t_steps, amp, device):
    """Replicates the DDIM loop from inference_precip_ddpm_v5b.py. NFE = steps."""
    t_seq = np.linspace(t_steps - 1, 0, steps, dtype=int)
    x = noise.clone()
    for j, t in enumerate(t_seq):
        t_prev = int(t_seq[j + 1]) if j + 1 < len(t_seq) else -1
        t_tensor = torch.tensor([int(t)], device=device)
        if amp:
            with torch.amp.autocast('cuda'):
                eps = model(torch.cat([x, lr_norm], dim=1), t_tensor)
        else:
            eps = model(torch.cat([x, lr_norm], dim=1), t_tensor)
        x = sched.ddim_step(x, eps, int(t), t_prev)
    return x


def run_config(cfg, lr_data, dates_year_index, n_years, obs_stats, args,
               models, fmcfg, ddpmcfg, normalizer, fm, sched, device,
               pt, pb, pl, pr, H, W):
    """Sample the full set under one (family, solver, steps) config and return
    a metrics row including empirical seconds/sample and NFE."""
    family, solver, steps = cfg['family'], cfg['solver'], cfg['steps']
    nfe = steps * (1 if solver in ('euler', 'ddim') else 2)
    rng = np.random.default_rng(args.seed + 13 * nfe)  # independent pool seed
    stats = FieldStats(H, W, dates_year_index, n_years, args.wet_threshold,
                       args.wet_keep_prob, rng)

    T = lr_data.shape[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    for i in tqdm(range(T), desc=f"  {family}-{solver}{steps} (NFE={nfe})",
                  leave=False):
        lr_t = (torch.from_numpy(lr_data[i]).unsqueeze(0).unsqueeze(0)
                .float().to(device))
        lr_t = F.pad(lr_t, (pl, pr, pt, pb), mode='reflect')
        lr_norm = normalizer.normalize_torch(lr_t)
        noise = make_day_noise(lr_norm.shape, i, args.seed, device, lr_norm.dtype)

        if family == 'FM':
            x = fm_sample_day(fm, models['fm'], lr_norm, noise, steps, solver,
                              args.amp, device)
        else:  # DDPM / DDIM
            x = ddim_sample_day(models['ddpm'], sched, lr_norm, noise, steps,
                                ddpmcfg.t_steps, args.amp, device)

        pred = (normalizer.denormalize_torch(x.clamp(-1, 1))
                .squeeze().cpu().numpy())
        pred = pred[pt:pt + H, pl:pl + W].clip(0).astype(np.float32)
        stats.update(pred, i)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sec_per_sample = (time.time() - t0) / max(T, 1)

    stats.finalize()
    row = dict(config=f"{family}-{solver}-{steps}", family=family,
               solver=solver, steps=steps, nfe=nfe,
               sec_per_sample=sec_per_sample)
    row.update(metrics_vs_obs(stats, obs_stats, args.return_period_years))
    return row


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────
def make_figure(df, out_png, out_pdf, period_years):
    fams = {
        ('FM', 'euler'):    dict(c='#1b7837', m='o', ls='-',  lbl='FM Euler'),
        ('FM', 'heun'):     dict(c='#2166ac', m='s', ls='-',  lbl='FM Heun'),
        ('FM', 'midpoint'): dict(c='#5aae61', m='^', ls='--', lbl='FM Midpoint'),
        ('DDPM', 'ddim'):   dict(c='#b2182b', m='D', ls='-',  lbl='DDPM DDIM'),
    }
    panels = [
        ('wasserstein',        'Wasserstein-1 of wet-day PDF (mm/day)',  'nfe', True),
        ('rl100_relbias_pct',  f'{int(period_years)}-yr return-level bias (%)', 'nfe', False),
        ('p999_ratio',         'Wet-day P99.9 ratio (model / obs)',      'nfe', False),
        ('wasserstein',        'Wasserstein-1 of wet-day PDF (mm/day)',  'sec', True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    for ax, (metric, ylab, xkind, logy) in zip(axes, panels):
        for (fam, solver), st in fams.items():
            sub = df[(df.family == fam) & (df.solver == solver)].sort_values('nfe')
            if sub.empty:
                continue
            x = sub['sec_per_sample'] if xkind == 'sec' else sub['nfe']
            ax.plot(x, sub[metric], color=st['c'], marker=st['m'],
                    ls=st['ls'], label=st['lbl'], lw=1.8, ms=6)
        if metric == 'rl100_relbias_pct':
            ax.axhline(0, color='k', lw=0.8, ls=':')
        if metric == 'p999_ratio':
            ax.axhline(1, color='k', lw=0.8, ls=':')
        ax.set_xlabel('wall-clock s / sample' if xkind == 'sec'
                      else 'NFE (network forward evals)')
        ax.set_ylabel(ylab)
        if xkind == 'nfe':
            ax.set_xscale('log', base=2)
        if logy:
            ax.set_yscale('log')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle('Few-step sampling: fidelity vs cost (FM solvers vs DDPM DDIM)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def load_model(UNetClass, ckpt_path, cfg, device, tag):
    model = UNetClass(in_channels=2, out_channels=1, base_dim=cfg.base_dim,
                      ch_mult_cap=cfg.ch_mult_cap, time_dim=cfg.time_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
    clean = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean)
    model.eval()
    print(f"  [{tag}] epoch={ckpt.get('epoch', '?')} "
          f"best_val={ckpt.get('best_val', float('nan')):.5f} "
          f"ema={'yes' if 'ema_state_dict' in ckpt else 'no'} "
          f"framework={ckpt.get('framework', '(none)')}")
    return model, ckpt


def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    fmcfg, ddpmcfg = FMConfig(), DDPMConfig()
    normalizer = PrecipNormalizer(fmcfg.log1p_max)

    # ── Load HR obs and build the shared LR once ──
    print(f"Loading MSWX HR: {args.input_h5}")
    with h5py.File(args.input_h5, 'r') as f:
        hr = f['precipitation'][:]
        times = f['time'][:]
    if args.max_days:
        hr, times = hr[:args.max_days], times[:args.max_days]
    T, H, W = hr.shape
    dates = pd.to_datetime(times)
    years = dates.year.values
    uniq_years = np.unique(years)
    year_index = np.searchsorted(uniq_years, years)
    n_years = len(uniq_years)
    print(f"  HR {hr.shape} | years {list(uniq_years)} | "
          f"range [{hr.min():.2f}, {hr.max():.2f}] mm/day")
    if n_years < 5:
        print(f"  WARNING: only {n_years} years -> {int(args.return_period_years)}-yr "
              f"return level is an extrapolation; treat RL-bias as indicative.")

    print(f"  Building shared full-field LR (avg_pool({fmcfg.scale_factor}) -> bicubic)...")
    lr = np.zeros_like(hr, dtype=np.float32)
    for s in tqdm(range(0, T, 50), desc="  LR"):
        e = min(s + 50, T)
        lr[s:e] = compute_lr_fullfield(hr[s:e].astype(np.float32),
                                       fmcfg.scale_factor, H, W)

    # ── Observed reference statistics (computed once) ──
    print("Computing observed reference statistics...")
    obs_rng = np.random.default_rng(args.seed)
    obs_stats = FieldStats(H, W, year_index, n_years, args.wet_threshold,
                           args.wet_keep_prob, obs_rng)
    for i in range(T):
        obs_stats.update(hr[i].astype(np.float32), i)
    obs_stats.finalize()
    rl_obs, c_obs, _, _ = gev_return_level(obs_stats.annual_maxima,
                                           args.return_period_years)
    print(f"  Obs {int(args.return_period_years)}-yr RL = {rl_obs:.2f} mm/day "
          f"| GEV shape c = {c_obs:.4f} | wet-pool n = {obs_stats.wet_pool.size:,}")
    del hr  # free memory; LR is all we need from here

    # ── Models ──
    print("Loading models...")
    fm_model, _ = load_model(FMUNet, args.fm_checkpoint, fmcfg, device, 'FM')
    ddpm_model, _ = load_model(DDPMUNet, args.ddpm_checkpoint, ddpmcfg, device, 'DDPM')
    models = {'fm': fm_model, 'ddpm': ddpm_model}
    fm = FlowMatching(t_scale=fmcfg.t_scale, device=device)
    sched = NoiseScheduler(ddpmcfg.t_steps, ddpmcfg.beta_start,
                           ddpmcfg.beta_end, device)

    pt, pb, pl, pr = get_padding(H, W)
    print(f"  U-Net pad: t{pt} b{pb} l{pl} r{pr} (divisor {UNET_DIVISOR})")

    # ── Build config list ──
    configs = []
    for s in args.fm_euler_steps:
        configs.append(dict(family='FM', solver='euler', steps=s))
    for s in args.fm_heun_steps:
        configs.append(dict(family='FM', solver='heun', steps=s))
    for s in args.fm_midpoint_steps:
        configs.append(dict(family='FM', solver='midpoint', steps=s))
    for s in args.ddim_steps:
        configs.append(dict(family='DDPM', solver='ddim', steps=s))
    print(f"\nSweeping {len(configs)} configurations over {T} days each...\n")

    # ── Run sweep ──
    rows = []
    for cfg in configs:
        row = run_config(cfg, lr, year_index, n_years, obs_stats, args, models,
                         fmcfg, ddpmcfg, normalizer, fm, sched, device,
                         pt, pb, pl, pr, H, W)
        rows.append(row)
        print(f"  {row['config']:>16s} | NFE {row['nfe']:>4d} | "
              f"{row['sec_per_sample']*1e3:7.1f} ms/day | "
              f"W1 {row['wasserstein']:.4f} | "
              f"RL{int(args.return_period_years)} bias {row['rl100_relbias_pct']:+6.2f}% | "
              f"P99.9 {row['p999_ratio']:.4f}")

    df = pd.DataFrame(rows)
    # attach obs reference for downstream plotting/tables
    df.attrs['obs_rl'] = rl_obs
    df.attrs['obs_gev_c'] = c_obs

    out_csv = os.path.join(args.output_dir, 'fewstep_curve_metrics.csv')
    df.to_csv(out_csv, index=False)
    with open(os.path.join(args.output_dir, 'fewstep_curve_obs_ref.json'), 'w') as fh:
        json.dump(dict(obs_return_level=rl_obs, obs_gev_shape_c=c_obs,
                       return_period_years=args.return_period_years,
                       n_years=int(n_years), n_days=int(T),
                       wet_pool_n=int(obs_stats.wet_pool.size)), fh, indent=2)

    out_png = os.path.join(args.output_dir, 'fewstep_curve.png')
    out_pdf = os.path.join(args.output_dir, 'fewstep_curve.pdf')
    make_figure(df, out_png, out_pdf, args.return_period_years)

    print(f"\nSaved:\n  {out_csv}\n  {out_png}\n  {out_pdf}")
    print("\nReviewer read: find the smallest-NFE FM config whose W1 and "
          "RL-bias match DDPM-50 (NFE 50). If FM-Euler reaches it at NFE<=16, "
          "FM is cheaper *and* competitive — that is the contribution.")


if __name__ == '__main__':
    main()
