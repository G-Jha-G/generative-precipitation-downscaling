#!/usr/bin/env python3
"""
ensemble_verification_final.py
==============================
Final, publication version of the Figure 10 ensemble verification for the
precipitation downscaling manuscript (FM vs DDPM, with deterministic baselines).

Run as a script:

    python ensemble_verification_final.py
    python ensemble_verification_final.py --day-stride 1          # full 1461 days
    python ensemble_verification_final.py --fm-solver euler --fm-steps 16
    python ensemble_verification_final.py --output-dir /path/to/out

All settings default to the Config dataclass below; any CLI flag overrides it.

CHANGES OVER ensemble_verification_notebook.py
----------------------------------------------
  A. SAMPLER CONSISTENCY (Heun-50 -> Euler-16).
     The FM ensemble is now generated with the Euler-16 sampler, matching
     Table 7 ("FM (Euler-16)") and the manuscript's few-step efficiency claim.
     Previously the figure used Heun-50, which disagreed with the table and
     inverted the spread-skill comparison against DDPM.

  B. RANK HISTOGRAM ZERO/DRIZZLE SPIKE FIXED.
     Precipitation rank histograms are dominated by dry cells. The denormalised
     ensemble carries trace drizzle (tiny positive values) on dry cells, so an
     observation of exactly 0 mm falls below every member and piles into rank 0.
     The old code tied only on exact equality (x == y), which drizzle never
     satisfies, so the promised random tie-breaking never engaged.
     Fix: values below a wet-day threshold (default 1.0 mm/day, ETCCDI standard)
     are collapsed to a common dry value so real zeros tie and receive a
     randomised rank (Hamill, 2001); cells that are dry in the observation AND
     in every member are dropped as uninformative. A genuine wet bias still
     shows up; only the numerical drizzle spike is removed.

  C. Converted from a single Jupyter cell to a runnable .py with a main()
     entry point and an argparse overlay. plt.show() calls removed (Agg backend).

  D. Bar-chart labels read cleanly ("FM (Euler-16)", "DDPM (DDIM-50)").

Retained from the previous version: shared deterministic scoring on the
identical strided day subset; per-model CRPS map with the correct title and no
political borders (GoI boundary policy); combined five-panel Figure 10; JSON and
NetCDF outputs; pandas summary table.

Author: user Kumar Jha | HIMPACT Lab, IIT Mandi
"""

# ==============================================================================
# 0 - CONFIGURATION  (edit here, or override on the command line)
# ==============================================================================

import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# Directory containing train_precip_fm_v1.py and train_precip_ddpm_v5b.py
TRAINING_SCRIPTS_DIR = (
    "/path/to/project/"
    "PhD_Precipitation/03_Code/precipitation-ddpm-india/scripts/ddpm/diffusr_climate"
)
sys.path.insert(0, TRAINING_SCRIPTS_DIR)


@dataclass
class Config:
    # -- Checkpoints --------------------------------------------------------
    fm_checkpoint: str = (
        "/path/to/project/"
        "Flow_matching_downscaling/results/precip_fm_v1/best.pt"
    )
    ddpm_checkpoint: str = (
        "/path/to/project/"
        "PhD_Precipitation/03_Code/precipitation-ddpm-india/scripts/ddpm/"
        "diffusr_climate/results/precip_ddpm_v5b/best.pt"
    )

    # -- Test data ----------------------------------------------------------
    input_h5: str = (
        "/path/to/project/"
        "PhD_Precipitation/02_Data/processed/hdf5/mswx_test_2011-2014.h5"
    )

    # -- Deterministic baselines: {label: (path, variable_name)} -----------
    det_ncs: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "SRCNN": (
            "/path/to/project/"
            "PhD_Precipitation/03_Code/results/srcnn_baseline/srcnn_pred_2011-2014.nc",
            "pr_srcnn",
        ),
        "CA": (
            "/path/to/project/"
            "PhD_Precipitation/03_Code/results/ca_baseline/ca_pred_2011-2014.nc",
            "pr_ca",
        ),
        "WT": (
            "/path/to/project/"
            "PhD_Precipitation/03_Code/results/"
            "weather_typing_baseline/wt_pred_2011-2014.nc",
            "pr_wt",
        ),
    })

    # -- Ensemble sampling --------------------------------------------------
    n_ensemble:  int = 20        # members per day
    # FIX A: FM sampler is Euler-16 (was Heun-50) to match Table 7 and the
    #        few-step efficiency claim. euler: 1 NFE/step => 16 NFE.
    fm_solver:   str = "euler"   # "euler" | "midpoint" | "heun"
    fm_steps:    int = 16        # ODE steps
    ddim_steps:  int = 50        # DDIM reverse steps (50 NFE)

    # -- Day subsampling ----------------------------------------------------
    # 1 = all 1461 test days (slow); 10 = every 10th = 147 days (fast).
    # The SAME stride is applied to both models and all deterministic baselines
    # so every metric is computed on the identical day set.
    day_stride:  int = 10
    max_days:    Optional[int] = None

    # -- Rank histogram / spread-skill --------------------------------------
    rank_keep_prob: float = 0.05   # fraction of grid cells sampled (raised for
                                   # a smoother publication histogram)
    n_spread_bins:  int   = 12
    # FIX B: wet-day threshold. Values below this (mm/day) are treated as dry
    # and tied, so the observation's rank among tied-dry members is randomised
    # instead of collapsing to rank 0. 1.0 mm/day is the ETCCDI wet-day standard;
    # set to 0.1 for a trace threshold if preferred.
    wet_threshold_mm: float = 1.0

    # -- Hardware -----------------------------------------------------------
    gpu:  int  = 0
    seed: int  = 42
    amp:  bool = False

    # -- Output -------------------------------------------------------------
    output_dir: str = (
        "/path/to/project/"
        "PhD_Precipitation/03_Code/notebooks/results/precip_ddpm_v5b/"
        "ensemble_verification_final"
    )
    dpi: int = 200


# ==============================================================================
# 1 - IMPORTS
# ==============================================================================

import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import xarray as xr
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")


# ==============================================================================
# 2 - CORE MATH
# ==============================================================================

class _nullctx:
    """No-op context manager (replaces torch.amp.autocast when amp=False)."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def crps_fair_map(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Ferro (2014) unbiased (fair) ensemble CRPS per grid cell.

        CRPS_fair = (1/M) Sum_i |X_i - y|
                    - 1/(M(M-1)) Sum_{i<j} |X_i - X_j|

    The pairwise term uses the sorted-rank identity
        Sum_{i<j} |X_i - X_j| = Sum_k X_(k) (2k - M + 1),  k = 0 .. M-1
    which is O(M log M). For M == 1 the spread term vanishes and
    CRPS_fair = |x - y| = MAE, so deterministic CRPS equals MAE.

    Args
        ens: [M, H, W]  ensemble members, mm/day
        obs: [H, W]     verifying truth, mm/day
    Returns
        [H, W] CRPS map, mm/day
    """
    M = ens.shape[0]
    term1 = np.mean(np.abs(ens - obs[None]), axis=0)
    if M == 1:
        return term1
    xs = np.sort(ens, axis=0)
    k = np.arange(M).reshape(M, 1, 1)
    pair_sum = np.sum(xs * (2 * k - M + 1), axis=0)
    return term1 - pair_sum / (M * (M - 1))


class EnsembleAccumulator:
    """
    Online accumulation of CRPS, spread-skill, and rank histogram counts
    across test days. Memory footprint O(H*W) plus the subsampled pools.
    """
    def __init__(self, H, W, M, rng, rank_keep_prob, n_spread_bins,
                 wet_threshold):
        self.H, self.W, self.M = H, W, M
        self.rng = rng
        self.rank_keep_prob = rank_keep_prob
        self.n_spread_bins = n_spread_bins
        self.wet_threshold = wet_threshold
        self.crps_sum = np.zeros((H, W), np.float64)
        self.var_sum = np.zeros((H, W), np.float64)   # ensemble variance
        self.se_sum = np.zeros((H, W), np.float64)     # (ens mean - obs)^2
        self.n_days = 0
        self.rank_counts = np.zeros(M + 1, np.int64)   # ranks 0 .. M
        self._ss_spread: list = []
        self._ss_abserr: list = []

    def update(self, ens: np.ndarray, obs: np.ndarray):
        """Process one day. ens: [M,H,W]; obs: [H,W]."""
        M = self.M
        self.n_days += 1

        self.crps_sum += crps_fair_map(ens, obs)

        emean = ens.mean(axis=0)
        evar = ens.var(axis=0, ddof=1) if M > 1 else np.zeros_like(emean)
        self.var_sum += evar
        self.se_sum += (emean - obs) ** 2

        mask = self.rng.random((self.H, self.W)) < self.rank_keep_prob
        if not mask.any():
            return

        # spread-skill pool (all sampled cells, full field)
        self._ss_spread.append(np.sqrt(evar[mask]).astype(np.float32))
        self._ss_abserr.append(np.abs(emean[mask] - obs[mask]).astype(np.float32))

        # --- rank histogram with trace-threshold tie handling (FIX B) -------
        tau = self.wet_threshold
        y = obs[mask].astype(np.float64)
        x = ens[:, mask].astype(np.float64)
        # collapse sub-threshold drizzle to a common dry value so zeros tie
        y = np.where(y < tau, 0.0, y)
        x = np.where(x < tau, 0.0, x)
        # keep only informative cells: wet in the obs or in at least one member
        info = (y > 0.0) | (x > 0.0).any(axis=0)
        if not info.any():
            return
        y = y[info]
        x = x[:, info]
        below = np.sum(x < y[None], axis=0)
        equal = np.sum(x == y[None], axis=0)          # ties, incl. dry-dry
        jitter = np.floor(self.rng.random(y.shape) * (equal + 1)).astype(int)
        ranks = (below + jitter).clip(0, M)
        self.rank_counts += np.bincount(ranks, minlength=M + 1)

    def finalize(self):
        n = max(self.n_days, 1)
        self.crps_map = (self.crps_sum / n).astype(np.float32)
        self.mean_crps = float(np.nanmean(self.crps_map))
        self.rmse_mean = float(np.sqrt(self.se_sum.sum() / (n * self.H * self.W)))
        self.mean_spread = float(np.sqrt(self.var_sum.sum() / (n * self.H * self.W)))
        self.ss_spread = (np.concatenate(self._ss_spread)
                          if self._ss_spread else np.array([], np.float32))
        self.ss_abserr = (np.concatenate(self._ss_abserr)
                          if self._ss_abserr else np.array([], np.float32))
        return self


def rank_diagnostics(counts: np.ndarray) -> dict:
    """
    Reliability index (0 = perfectly flat; larger = less flat), chi-square,
    quadratic convexity (positive -> U-shape -> under-dispersive), and
    left-right asymmetry.
    """
    p = counts / counts.sum()
    nb = len(p)
    uniform = 1.0 / nb
    ri = float(np.sum(np.abs(p - uniform)))
    expected = counts.sum() * uniform
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    xb = np.linspace(-1, 1, nb)
    quad = float(np.polyfit(xb, p, 2)[0])
    skew = float(p[-1] - p[0])
    if quad > 0.5 * uniform:
        shape = "U-shaped -> under-dispersive (ensemble too narrow)"
    elif quad < -0.5 * uniform:
        shape = "dome -> over-dispersive (ensemble too wide)"
    else:
        shape = "approximately flat -> well-calibrated"
    return dict(reliability_index=ri, chi_square=chi2,
                convexity=quad, end_skew=skew, interpretation=shape)


# ==============================================================================
# 3 - DATA GENERATORS
# ==============================================================================

def _load_hr_lr(cfg: Config) -> tuple:
    """
    Load HR test data, apply day_stride + max_days, pre-compute full-field LR.
    Returns hr, lr, lat, lon, (pt, pb, pl, pr), T, H, W.
    """
    from train_precip_fm_v1 import compute_lr_fullfield
    print("Loading MSWX test data ...")
    with h5py.File(cfg.input_h5, "r") as f:
        hr = f["precipitation"][:]   # [T, H, W] mm/day
        lat = f["lat"][:]
        lon = f["lon"][:]
    if cfg.day_stride > 1:
        hr = hr[::cfg.day_stride]
    if cfg.max_days:
        hr = hr[:cfg.max_days]
    T, H, W = hr.shape
    print(f"  {T} days retained (stride={cfg.day_stride}) | field {H}x{W}")

    # reflect-pad to nearest multiple of 16 for the 4-level U-Net
    div = 16
    ph = (div - H % div) % div
    pw = (div - W % div) % div
    pt, pl = ph // 2, pw // 2
    pb, pr = ph - pt, pw - pl

    print("  Pre-computing full-field LR ...")
    lr = np.zeros_like(hr, dtype=np.float32)
    sf = 10
    for s in tqdm(range(0, T, 50), desc="  LR batches", leave=False):
        e = min(s + 50, T)
        lr[s:e] = compute_lr_fullfield(hr[s:e].astype(np.float32), sf, H, W)
    print(f"  LR range: [{lr.min():.2f}, {lr.max():.2f}] mm/day\n")
    return hr, lr, lat, lon, (pt, pb, pl, pr), T, H, W


def _load_model_fm(cfg: Config, device):
    from train_precip_fm_v1 import (Config as FMConfig, PrecipNormalizer,
                                    PrecipUNet, FlowMatching)
    fc = FMConfig()
    nm = PrecipNormalizer(fc.log1p_max)
    m = PrecipUNet(in_channels=2, out_channels=1, base_dim=fc.base_dim,
                   ch_mult_cap=fc.ch_mult_cap, time_dim=fc.time_dim).to(device)
    ck = torch.load(cfg.fm_checkpoint, map_location=device, weights_only=False)
    st = ck.get("ema_state_dict", ck.get("model_state_dict", ck))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()})
    m.eval()
    fm = FlowMatching(t_scale=fc.t_scale, device=device)
    return m, nm, fm, fc


def _load_model_ddpm(cfg: Config, device):
    from train_precip_ddpm_v5b import (Config as DDConfig, PrecipNormalizer,
                                       PrecipUNet, NoiseScheduler)
    dc = DDConfig()
    nm = PrecipNormalizer(dc.log1p_max)
    m = PrecipUNet(in_channels=2, out_channels=1, base_dim=dc.base_dim,
                   ch_mult_cap=dc.ch_mult_cap, time_dim=dc.time_dim).to(device)
    ck = torch.load(cfg.ddpm_checkpoint, map_location=device, weights_only=False)
    st = ck.get("ema_state_dict", ck.get("model_state_dict", ck))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()})
    m.eval()
    sched = NoiseScheduler(dc.t_steps, dc.beta_start, dc.beta_end, device)
    t_seq = np.linspace(dc.t_steps - 1, 0, cfg.ddim_steps, dtype=int)
    return m, nm, sched, t_seq, dc


def run_fm_verification(cfg, hr, lr, lat, lon, pad, T, H, W, device) -> tuple:
    """Generate and verify the FM ensemble. Returns (accumulator, file_tag)."""
    print("=" * 60)
    print(f"FM  |  {cfg.fm_solver.upper()}-{cfg.fm_steps}  "
          f"({cfg.n_ensemble} members x {T} days)")
    print("=" * 60)

    model, normalizer, fm, fc = _load_model_fm(cfg, device)
    pt, pb, pl, pr = pad
    tag = f"FM-{cfg.fm_solver}{cfg.fm_steps}"

    rng = np.random.default_rng(cfg.seed)
    acc = EnsembleAccumulator(H, W, cfg.n_ensemble, rng, cfg.rank_keep_prob,
                              cfg.n_spread_bins, cfg.wet_threshold_mm)

    for i in tqdm(range(T), desc="  FM generate+verify"):
        lr_t = (torch.from_numpy(lr[i]).unsqueeze(0).unsqueeze(0)
                .float().to(device))
        lr_t = F.pad(lr_t, (pl, pr, pt, pb), mode="reflect")
        lr_n = normalizer.normalize_torch(lr_t)
        members = []
        for mem in range(cfg.n_ensemble):
            seed = cfg.seed * 1_000_003 + i * 1009 + mem
            g = torch.Generator(device=device).manual_seed(seed)
            z = torch.randn(lr_n.shape, generator=g,
                            device=device, dtype=lr_n.dtype)
            ctx = torch.amp.autocast("cuda") if cfg.amp else _nullctx()
            with ctx:
                x = fm.sample(model=model, cond=lr_n, shape=lr_n.shape,
                              num_steps=cfg.fm_steps, solver=cfg.fm_solver,
                              device=device, noise=z)
            pred = (normalizer.denormalize_torch(x.clamp(-1, 1))
                    .squeeze().cpu().numpy())
            members.append(pred[pt:pt + H, pl:pl + W].clip(0).astype(np.float32))
        acc.update(np.stack(members, 0), hr[i].astype(np.float32))

    acc.finalize()
    print(f"  CRPS  = {acc.mean_crps:.4f} mm/day")
    print(f"  Spread/RMSE = {acc.mean_spread / acc.rmse_mean:.3f}\n")
    return acc, tag


def run_ddpm_verification(cfg, hr, lr, lat, lon, pad, T, H, W, device) -> tuple:
    """Generate and verify the DDPM ensemble. Returns (accumulator, file_tag)."""
    print("=" * 60)
    print(f"DDPM  |  DDIM-{cfg.ddim_steps}  "
          f"({cfg.n_ensemble} members x {T} days)")
    print("=" * 60)

    model, normalizer, sched, t_seq, dc = _load_model_ddpm(cfg, device)
    pt, pb, pl, pr = pad
    tag = f"DDPM-ddim{cfg.ddim_steps}"

    @torch.no_grad()
    def ddim_step_loop(lr_n, noise):
        x = noise.clone()
        for j, t in enumerate(t_seq):
            t_prev = int(t_seq[j + 1]) if j + 1 < len(t_seq) else -1
            tt = torch.tensor([int(t)], device=device)
            ctx = torch.amp.autocast("cuda") if cfg.amp else _nullctx()
            with ctx:
                eps = model(torch.cat([x, lr_n], dim=1), tt)
            x = sched.ddim_step(x, eps, int(t), t_prev)
        return x

    rng = np.random.default_rng(cfg.seed)
    acc = EnsembleAccumulator(H, W, cfg.n_ensemble, rng, cfg.rank_keep_prob,
                              cfg.n_spread_bins, cfg.wet_threshold_mm)

    for i in tqdm(range(T), desc="  DDPM generate+verify"):
        lr_t = (torch.from_numpy(lr[i]).unsqueeze(0).unsqueeze(0)
                .float().to(device))
        lr_t = F.pad(lr_t, (pl, pr, pt, pb), mode="reflect")
        lr_n = normalizer.normalize_torch(lr_t)
        members = []
        for mem in range(cfg.n_ensemble):
            seed = cfg.seed * 1_000_003 + i * 1009 + mem
            g = torch.Generator(device=device).manual_seed(seed)
            z = torch.randn(lr_n.shape, generator=g,
                            device=device, dtype=lr_n.dtype)
            x = ddim_step_loop(lr_n, z)
            pred = (normalizer.denormalize_torch(x.clamp(-1, 1))
                    .squeeze().cpu().numpy())
            members.append(pred[pt:pt + H, pl:pl + W].clip(0).astype(np.float32))
        acc.update(np.stack(members, 0), hr[i].astype(np.float32))

    acc.finalize()
    print(f"  CRPS  = {acc.mean_crps:.4f} mm/day")
    print(f"  Spread/RMSE = {acc.mean_spread / acc.rmse_mean:.3f}\n")
    return acc, tag


# ==============================================================================
# 4 - DETERMINISTIC BASELINE SCORER
# ==============================================================================

def score_deterministic_baselines(cfg: Config, hr: np.ndarray) -> dict:
    """
    Score each deterministic baseline on the already-strided truth array
    (identical day subset used for ensemble generation). Returns {name: MAE}.
    MAE equals the one-member CRPS.
    """
    print("Scoring deterministic baselines ...")
    out = {}
    T = hr.shape[0]
    for name, (path, var) in cfg.det_ncs.items():
        ds = xr.open_dataset(path)
        vals = ds[var].values                       # [T_full, H, W]
        ds.close()
        if cfg.day_stride > 1:
            vals = vals[::cfg.day_stride]
        if cfg.max_days:
            vals = vals[:cfg.max_days]
        n = min(len(vals), T)
        mae = float(np.nanmean(np.abs(vals[:n] - hr[:n])))
        out[name] = mae
        print(f"  {name:<8s} MAE = {mae:.4f} mm/day")
    print()
    return out


# ==============================================================================
# 5 - PLOTTING
# ==============================================================================

C_FM = "#2166ac"     # blue  - FM
C_DDPM = "#d6604d"   # red   - DDPM
C_DET = "#888888"    # grey  - deterministic baselines


def _rank_hist_ax(ax, counts, diag, model_name, color):
    M1 = len(counts)
    p = counts / counts.sum()
    ax.bar(np.arange(M1), p, width=0.92, color=color, alpha=0.85, edgecolor="none")
    flat = 1.0 / M1
    ax.axhline(flat, color="firebrick", ls="--", lw=1.2,
               label="flat (perfect calibration)")
    ax.set_xlabel("Verification rank of observation (0 ... M)", fontsize=9)
    ax.set_ylabel("Relative frequency", fontsize=9)
    ri = diag["reliability_index"]
    ax.set_title(f"{model_name}  |  reliability index = {ri:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)


def _spread_skill_ax(ax, acc, model_name, color):
    sp, ae = acc.ss_spread, acc.ss_abserr
    if sp.size:
        qs = np.unique(np.quantile(sp, np.linspace(0, 1, acc.n_spread_bins + 1)))
        bx, by = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (sp >= lo) & (sp <= hi)
            if m.sum() > 50:
                bx.append(sp[m].mean())
                by.append(float(np.sqrt(np.mean(ae[m] ** 2))))
        bx, by = np.array(bx), np.array(by)
        if bx.size:
            hi = max(bx.max(), by.max())
            ax.plot(bx, by, "o-", color=color, lw=1.5, ms=5,
                    label="binned spread vs RMSE", zorder=3)
            ax.plot([0, hi], [0, hi], "k--", lw=1.0, label="1:1")
            tgt = float(np.sqrt((acc.M + 1) / acc.M))
            ax.plot([0, hi], [0, hi * tgt], color="gray", ls=":", lw=1.0,
                    label=f"finite-M target (x{tgt:.3f})")
    ratio = acc.mean_spread / acc.rmse_mean if acc.rmse_mean > 0 else float("nan")
    ax.set_title(f"{model_name}  |  spread/RMSE = {ratio:.3f}", fontsize=10)
    ax.set_xlabel("Ensemble spread  (std, mm day$^{-1}$)", fontsize=9)
    ax.set_ylabel("RMSE of ensemble mean  (mm day$^{-1}$)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def plot_figure10(acc_fm, diag_fm, acc_ddpm, diag_ddpm, det_scores,
                  disp_fm, disp_ddpm, cfg):
    """
    Combined Figure 10 (three rows, five panels).
        (a) full width : fair CRPS / MAE bar chart
        (b) | (c)      : FM | DDPM rank histograms
        (d) | (e)      : FM | DDPM spread-skill
    """
    fig = plt.figure(figsize=(13, 13))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32,
                           height_ratios=[1.0, 1.1, 1.1])

    # (a) fair CRPS vs MAE
    ax0 = fig.add_subplot(gs[0, :])
    labels = [f"FM\n({disp_fm})", f"DDPM\n({disp_ddpm})"] + list(det_scores.keys())
    values = [acc_fm.mean_crps, acc_ddpm.mean_crps] + list(det_scores.values())
    colours = [C_FM, C_DDPM] + [C_DET] * len(det_scores)
    hatches = ["", ""] + ["///"] * len(det_scores)
    xpos = np.arange(len(labels))
    bars = ax0.bar(xpos, values, color=colours, hatch=hatches,
                   edgecolor="white", linewidth=0.5, width=0.65)
    for bar, val in zip(bars, values):
        ax0.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax0.set_xticks(xpos)
    ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel("Fair CRPS / MAE  (mm day$^{-1}$)", fontsize=9)
    ax0.set_title(
        "Fair CRPS of generative ensembles vs MAE of deterministic methods\n"
        "(lower is better; deterministic CRPS = MAE by definition)", fontsize=10)
    ax0.grid(axis="y", alpha=0.25)
    ax0.tick_params(labelsize=8)
    legend_elems = [
        Patch(facecolor=C_FM, label=f"FM ensemble  (M={acc_fm.M})"),
        Patch(facecolor=C_DDPM, label=f"DDPM ensemble  (M={acc_ddpm.M})"),
        Patch(facecolor=C_DET, hatch="///", label="Deterministic (1 member)"),
    ]
    ax0.legend(handles=legend_elems, fontsize=8, loc="upper left")

    # (b, c) rank histograms
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    _rank_hist_ax(ax1, acc_fm.rank_counts, diag_fm, "FM rank histogram", C_FM)
    _rank_hist_ax(ax2, acc_ddpm.rank_counts, diag_ddpm, "DDPM rank histogram", C_DDPM)

    # (d, e) spread-skill
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    _spread_skill_ax(ax3, acc_fm, "FM spread-skill", C_FM)
    _spread_skill_ax(ax4, acc_ddpm, "DDPM spread-skill", C_DDPM)

    for ax, lbl in zip([ax0, ax1, ax2, ax3, ax4], ["(a)", "(b)", "(c)", "(d)", "(e)"]):
        ax.text(-0.06, 1.03, lbl, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom")

    out = os.path.join(cfg.output_dir, "figure10_ensemble_verification.png")
    fig.savefig(out, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Figure 10 -> {out}")


def plot_crps_map(crps_map, lat, lon, model_name, cfg):
    """Spatial CRPS map, correct title, no political borders (GoI policy)."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    lat_asc = lat[0] < lat[-1]
    cm = crps_map if lat_asc else crps_map[::-1]
    extent = [float(lon.min()), float(lon.max()),
              float(lat.min()), float(lat.max())]
    im = ax.imshow(cm, extent=extent, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(f"{model_name} ensemble CRPS (mm day$^{{-1}}$), lower is better",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="CRPS (mm day$^{-1}$)")
    fig.tight_layout()
    tag = model_name.lower().replace(" ", "_")
    out = os.path.join(cfg.output_dir, f"crps_map_{tag}.png")
    fig.savefig(out, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved CRPS map -> {out}")


def save_crps_nc(crps_map, lat, lon, model_name, M, cfg):
    tag = model_name.lower().replace(" ", "_")
    ds = xr.Dataset({"crps": (["lat", "lon"], crps_map)},
                    coords={"lat": lat, "lon": lon})
    ds["crps"].attrs = {
        "long_name": f"{model_name} ensemble CRPS",
        "units": "mm day-1",
        "n_members": str(M),
        "downscale_factor": "10",
    }
    out = os.path.join(cfg.output_dir, f"crps_map_{tag}.nc")
    ds.to_netcdf(out)
    print(f"  Saved CRPS NetCDF -> {out}")


def save_json(summary: dict, tag: str, cfg: Config):
    out = os.path.join(cfg.output_dir, f"ensemble_verification_{tag}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved JSON -> {out}")


# ==============================================================================
# 6 - MAIN
# ==============================================================================

def main(cfg: Config):
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print("=" * 60)
    print("ENSEMBLE VERIFICATION - FM + DDPM (final)")
    print(f"device={device}  day_stride={cfg.day_stride}  "
          f"n_ensemble={cfg.n_ensemble}")
    print(f"FM sampler={cfg.fm_solver}-{cfg.fm_steps}  DDPM=DDIM-{cfg.ddim_steps}  "
          f"wet_threshold={cfg.wet_threshold_mm} mm/day")
    print(f"output -> {cfg.output_dir}")
    print("=" * 60 + "\n")

    hr, lr, lat, lon, pad, T, H, W = _load_hr_lr(cfg)
    acc_fm, tag_fm = run_fm_verification(cfg, hr, lr, lat, lon, pad, T, H, W, device)
    acc_ddpm, tag_ddpm = run_ddpm_verification(cfg, hr, lr, lat, lon, pad, T, H, W, device)
    det_scores = score_deterministic_baselines(cfg, hr)

    diag_fm = rank_diagnostics(acc_fm.rank_counts)
    diag_ddpm = rank_diagnostics(acc_ddpm.rank_counts)
    finite_M_target = float(np.sqrt((cfg.n_ensemble + 1) / cfg.n_ensemble))

    disp_fm = f"{cfg.fm_solver.capitalize()}-{cfg.fm_steps}"   # e.g. Euler-16
    disp_ddpm = f"DDIM-{cfg.ddim_steps}"

    # console summary
    rows = []
    for label, acc, disp, diag in [("FM", acc_fm, disp_fm, diag_fm),
                                   ("DDPM", acc_ddpm, disp_ddpm, diag_ddpm)]:
        ratio = acc.mean_spread / acc.rmse_mean if acc.rmse_mean > 0 else np.nan
        rows.append({
            "Model": label, "Sampler": disp, "M": acc.M, "Days": acc.n_days,
            "CRPS (mm/d)": round(acc.mean_crps, 4),
            "RMSE ens-mean": round(acc.rmse_mean, 4),
            "Mean spread": round(acc.mean_spread, 4),
            "SpSkR": round(ratio, 3), "Target SpSkR": round(finite_M_target, 3),
            "Rel. index": round(diag["reliability_index"], 3),
            "Interpretation": diag["interpretation"],
        })
    for name, mae in det_scores.items():
        rows.append({
            "Model": name, "Sampler": "det (MAE=CRPS)", "M": 1, "Days": T,
            "CRPS (mm/d)": round(mae, 4), "RMSE ens-mean": "-",
            "Mean spread": "-", "SpSkR": "-", "Target SpSkR": "-",
            "Rel. index": "-", "Interpretation": "deterministic",
        })
    df = pd.DataFrame(rows).set_index("Model")
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(df.to_string())
    print()

    # JSONs
    for label, acc, tag, diag in [("fm", acc_fm, tag_fm, diag_fm),
                                  ("ddpm", acc_ddpm, tag_ddpm, diag_ddpm)]:
        ratio = acc.mean_spread / acc.rmse_mean if acc.rmse_mean > 0 else np.nan
        save_json(dict(
            model=label, ensemble_tag=tag, n_members=int(acc.M),
            n_days=int(acc.n_days), mean_crps_mm_day=acc.mean_crps,
            rmse_of_ensemble_mean=acc.rmse_mean, mean_spread=acc.mean_spread,
            spread_skill_ratio=float(ratio), finite_M_target_ratio=finite_M_target,
            wet_threshold_mm=cfg.wet_threshold_mm,
            rank_histogram_counts=acc.rank_counts.tolist(),
            rank_diagnostics=diag, deterministic_mae_crps=det_scores,
        ), label, cfg)

    # CRPS maps + Figure 10
    print("\nPlotting CRPS maps ...")
    plot_crps_map(acc_fm.crps_map, lat, lon, "FM", cfg)
    plot_crps_map(acc_ddpm.crps_map, lat, lon, "DDPM", cfg)
    save_crps_nc(acc_fm.crps_map, lat, lon, "FM", acc_fm.M, cfg)
    save_crps_nc(acc_ddpm.crps_map, lat, lon, "DDPM", acc_ddpm.M, cfg)

    print("\nPlotting Figure 10 ...")
    plot_figure10(acc_fm, diag_fm, acc_ddpm, diag_ddpm, det_scores,
                  disp_fm, disp_ddpm, cfg)

    print("\nAll done.")
    print(f"Outputs in: {cfg.output_dir}")


def _parse_args(cfg: Config) -> Config:
    p = argparse.ArgumentParser(description="Final FM+DDPM ensemble verification.")
    p.add_argument("--fm-solver", choices=["euler", "midpoint", "heun"],
                   default=cfg.fm_solver)
    p.add_argument("--fm-steps", type=int, default=cfg.fm_steps)
    p.add_argument("--ddim-steps", type=int, default=cfg.ddim_steps)
    p.add_argument("--n-ensemble", type=int, default=cfg.n_ensemble)
    p.add_argument("--day-stride", type=int, default=cfg.day_stride)
    p.add_argument("--max-days", type=int, default=cfg.max_days)
    p.add_argument("--wet-threshold", type=float, default=cfg.wet_threshold_mm,
                   help="mm/day; values below are treated as dry (tied).")
    p.add_argument("--rank-keep-prob", type=float, default=cfg.rank_keep_prob)
    p.add_argument("--gpu", type=int, default=cfg.gpu)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--output-dir", type=str, default=cfg.output_dir)
    a = p.parse_args()
    cfg.fm_solver = a.fm_solver
    cfg.fm_steps = a.fm_steps
    cfg.ddim_steps = a.ddim_steps
    cfg.n_ensemble = a.n_ensemble
    cfg.day_stride = a.day_stride
    cfg.max_days = a.max_days
    cfg.wet_threshold_mm = a.wet_threshold
    cfg.rank_keep_prob = a.rank_keep_prob
    cfg.gpu = a.gpu
    cfg.seed = a.seed
    cfg.output_dir = a.output_dir
    return cfg


if __name__ == "__main__":
    main(_parse_args(Config()))
