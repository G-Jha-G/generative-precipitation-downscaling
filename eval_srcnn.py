"""
eval_srcnn.py
=============
Standalone evaluation of a trained SRCNN checkpoint.

Runs inference on val (2006-2010) and test (2011-2014),
saves NetCDF predictions, and computes the full metric suite
(RMSE, MAE, bias, Pearson r, KGE, wet-day frequency,
 percentile skill P50-P99.9, seasonal metrics, FSS, PSD ratio).

All model code (architecture, normalizer, LR generation, inference)
is copied verbatim from train_srcnn.py so results are guaranteed
to match what the training script would produce.

Usage:
  python eval_srcnn.py --ckpt /path/to/srcnn_baseline/best.pt --variant original
  python eval_srcnn.py --ckpt /path/best.pt --variant original --gpu 0
"""

import os, json, logging, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from pathlib import Path
from datetime import datetime, date, timedelta
import h5py
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from tqdm import tqdm


# ============================================================
# CONFIG  — update paths for your machine
# ============================================================
class Config:
    val_h5: str = (
        r'C:\Users\IIT\OneDrive\Ph.D\4th_precip_ddpm\processed\hdf5\mswx_val_2006-2010.h5'
    )
    test_h5: str = (
        r'C:\Users\IIT\OneDrive\Ph.D\4th_precip_ddpm\processed\hdf5\mswx_test_2011-2014.h5'
    )
    results_dir: str = (
        r'C:\Users\IIT\OneDrive\Ph.D\4th_precip_ddpm\processed\results\srcnn_baseline'
    )
    ckpt: str = (
        r'C:\Users\IIT\OneDrive\Ph.D\4th_precip_ddpm\processed\results\srcnn_baseline\best.pt'
    )

    # Must match what was used during training
    variant:      str   = 'original'
    log1p_max:    float = 6.7859
    scale_factor: int   = 10
    patch_size:   int   = 128
    stride:       int   = 64
    infer_batch_size: int = 64
    gpu:          int   = 0

    # Evaluation settings
    fss_thresholds: tuple = (1.0, 5.0, 20.0)
    fss_scales:     tuple = (1, 3, 5, 10, 15, 20, 30, 50)
    percentiles:    tuple = (50, 75, 90, 95, 99, 99.5, 99.9)
    val_start_year:  int  = 2006
    test_start_year: int  = 2011


# ============================================================
# LOGGING
# ============================================================
def setup_logging(results_dir: str) -> logging.Logger:
    stamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(results_dir, f'eval_srcnn_{stamp}.log')
    logger   = logging.getLogger(f'eval_srcnn_{stamp}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                            '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, encoding='utf-8')
    sh = logging.StreamHandler()
    for h in [fh, sh]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ============================================================
# NORMALIZER  — verbatim from train_srcnn.py
# ============================================================
class PrecipNormalizer:
    def __init__(self, log1p_max: float = 6.7859):
        self.log1p_max = log1p_max

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.log1p(np.clip(x, 0, None))
        return ((x / self.log1p_max) * 2.0 - 1.0).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        x = (x + 1.0) / 2.0 * self.log1p_max
        return np.clip(np.expm1(x), 0, None).astype(np.float32)


# ============================================================
# LR GENERATION  — verbatim from train_srcnn.py
# ============================================================
def compute_lr_fullfield(hr_batch: np.ndarray,
                          scale_factor: int, H: int, W: int) -> np.ndarray:
    t    = torch.from_numpy(hr_batch.astype(np.float32)).unsqueeze(1)
    lr   = F.avg_pool2d(t, scale_factor, scale_factor)
    pred = F.interpolate(lr, size=(H, W), mode='bicubic',
                         align_corners=False).clamp(min=0.0)
    return pred.squeeze(1).numpy()


# ============================================================
# ARCHITECTURE  — verbatim from train_srcnn.py
# ============================================================
class SRCNNOriginal(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64,    32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32,     1, kernel_size=5, padding=2)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


class SRCNNDeep(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def build_model(variant: str, device: torch.device) -> nn.Module:
    if variant == 'original':
        return SRCNNOriginal(in_ch=1).to(device)
    elif variant == 'deep':
        return SRCNNDeep(in_ch=1).to(device)
    else:
        raise ValueError(f'Unknown variant: {variant}')


# ============================================================
# CHECKPOINT LOADER  — verbatim from train_srcnn.py
# ============================================================
def load_checkpoint(ckpt_path: str, model: nn.Module,
                    device: torch.device,
                    logger: logging.Logger) -> dict:
    logger.info(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f'  Variant:    {ckpt.get("variant", "unknown")}')
    logger.info(f'  Epoch:      {ckpt["epoch"]}')
    logger.info(f'  Best val:   {ckpt["best_val"]:.5f}')
    return ckpt


# ============================================================
# INFERENCE  — verbatim from train_srcnn.py
# ============================================================
def build_patch_grid(H, W, patch_size, stride):
    tops  = list(range(0, H - patch_size + 1, stride))
    lefts = list(range(0, W - patch_size + 1, stride))
    if not tops  or tops[-1]  != H - patch_size: tops.append(H - patch_size)
    if not lefts or lefts[-1] != W - patch_size: lefts.append(W - patch_size)
    return [(tp, lf) for tp in tops for lf in lefts]


def make_hanning_window(patch_size: int) -> np.ndarray:
    w = np.hanning(patch_size).astype(np.float32)
    return np.outer(w, w)


@torch.no_grad()
def infer_one_field(model, lr_field, normalizer, positions,
                     window, patch_size, batch_size, device):
    H, W     = lr_field.shape
    pred_sum = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)
    lr_norm  = normalizer.normalize(lr_field)

    model.eval()
    batch_patches, batch_pos = [], []

    def flush():
        if not batch_patches: return
        inp = torch.stack(batch_patches).to(device)
        with autocast('cuda'):
            out = model(inp).squeeze(1).cpu().numpy()
        for (tp, lf), p_norm in zip(batch_pos, out):
            p_mm = normalizer.denormalize(p_norm)
            pred_sum[tp:tp+patch_size, lf:lf+patch_size] += p_mm * window
            count   [tp:tp+patch_size, lf:lf+patch_size] += window
        batch_patches.clear(); batch_pos.clear()

    for tp, lf in positions:
        patch = lr_norm[tp:tp+patch_size, lf:lf+patch_size]
        batch_patches.append(torch.from_numpy(patch).unsqueeze(0))
        batch_pos.append((tp, lf))
        if len(batch_patches) >= batch_size:
            flush()
    flush()
    return np.divide(pred_sum, np.maximum(count, 1e-10))


def run_inference(model, normalizer, h5_path, config, logger):
    logger.info(f'Loading: {h5_path}')
    with h5py.File(h5_path, 'r') as f:
        hr_data = f['precipitation'][:]
        lat = f['lat'][:] if 'lat' in f else np.linspace(6.5,  41.0,  350)
        lon = f['lon'][:] if 'lon' in f else np.linspace(66.5, 100.0, 350)

    T, H, W   = hr_data.shape
    device    = next(model.parameters()).device
    positions = build_patch_grid(H, W, config.patch_size, config.stride)
    window    = make_hanning_window(config.patch_size)
    pred      = np.zeros((T, H, W), dtype=np.float32)

    logger.info(f'  {T}x{H}x{W} | {len(positions)} patches/timestep')
    t0 = time.time()
    for t_idx in tqdm(range(T), desc='  Inference'):
        hr_t = hr_data[t_idx].astype(np.float32)
        lr_t = compute_lr_fullfield(hr_t[None], config.scale_factor, H, W)[0]
        pred[t_idx] = infer_one_field(
            model, lr_t, normalizer, positions, window,
            config.patch_size, config.infer_batch_size, device
        )
    logger.info(f'  Done in {(time.time()-t0)/60:.1f} min  '
                f'[{pred.min():.2f}, {pred.max():.2f}] mm/day')
    return pred, hr_data.astype(np.float32), lat, lon


# ============================================================
# METRICS
# ============================================================
def rmse(o, p):      return float(np.sqrt(np.mean((p - o)**2)))
def mae(o, p):       return float(np.mean(np.abs(p - o)))
def mean_bias(o, p): return float(np.mean(p - o))
def pearson_r(o, p): return float(np.corrcoef(o.ravel(), p.ravel())[0, 1])

def kge(o, p):
    o, p  = o.ravel(), p.ravel()
    r     = float(np.corrcoef(o, p)[0, 1])
    alpha = float(p.std()  / (o.std()  + 1e-10))
    beta  = float(p.mean() / (o.mean() + 1e-10))
    return float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))

def wet_day_frequency(f, thr=1.0): return float(np.mean(f >= thr))

def percentile_values(f, pcts):
    wet = f[f >= 1.0]
    return {p: float(np.percentile(wet, p)) for p in pcts} \
           if wet.size > 0 else {p: 0.0 for p in pcts}

def percentile_skill(op, pp):
    return {p: (pp[p]/op[p] if op[p] > 0 else float('nan')) for p in op}

def fss_score(obs, pred, threshold, scale):
    if obs.ndim == 3:
        scores = [fss_score(obs[i], pred[i], threshold, scale)
                  for i in range(obs.shape[0])]
        valid  = [s for s in scores if not np.isnan(s)]
        return float(np.mean(valid)) if valid else float('nan')
    ob = (obs  >= threshold).astype(np.float32)
    pb = (pred >= threshold).astype(np.float32)
    if ob.sum() == 0 and pb.sum() == 0: return float('nan')
    of = uniform_filter(ob, size=scale, mode='constant')
    pf = uniform_filter(pb, size=scale, mode='constant')
    num = np.mean((of - pf)**2); den = np.mean(of**2 + pf**2)
    return float(1 - num / den) if den > 0 else float('nan')

def azimuthal_psd(field_2d):
    H, W   = field_2d.shape
    power  = np.abs(np.fft.fftshift(np.fft.fft2(field_2d)))**2
    kx     = np.fft.fftshift(np.fft.fftfreq(W))
    ky     = np.fft.fftshift(np.fft.fftfreq(H))
    KX, KY = np.meshgrid(kx, ky)
    K      = np.sqrt(KX**2 + KY**2)
    n_bins = min(H, W) // 2
    bins   = np.linspace(0, 0.5, n_bins + 1)
    k_mid  = 0.5 * (bins[:-1] + bins[1:])
    psd    = np.array([
        power[(K >= bins[i]) & (K < bins[i+1])].mean()
        if ((K >= bins[i]) & (K < bins[i+1])).any() else 0.0
        for i in range(n_bins)
    ])
    return k_mid, psd

def build_season_mask(T, start_year):
    sm = {12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',
           6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}
    d = date(start_year, 1, 1); labels = []
    for _ in range(T):
        labels.append(sm[d.month]); d += timedelta(days=1)
    return np.array(labels)


# ============================================================
# NETCDF
# ============================================================
def save_netcdf(pred, obs, lat, lon, out_path, config, label):
    T, H, W = pred.shape
    with nc.Dataset(out_path, 'w', format='NETCDF4') as ds:
        ds.title            = f'SRCNN ({config.variant}) — {label}'
        ds.description      = 'Dong et al. (2014) SRCNN, simple MSE loss'
        ds.downscale_factor = config.scale_factor
        ds.variant          = config.variant
        ds.created          = datetime.now().isoformat()
        ds.createDimension('time', T)
        ds.createDimension('lat',  H)
        ds.createDimension('lon',  W)
        vt = ds.createVariable('time', 'i4', ('time',))
        vt[:] = np.arange(T)
        vt.units = f'days since {label[:4]}-01-01'
        vlat = ds.createVariable('lat', 'f4', ('lat',))
        vlat[:] = lat; vlat.units = 'degrees_north'
        vlon = ds.createVariable('lon', 'f4', ('lon',))
        vlon[:] = lon; vlon.units = 'degrees_east'
        vp = ds.createVariable('pr_srcnn', 'f4', ('time','lat','lon'),
                                zlib=True, complevel=4, fill_value=-9999.)
        vp[:] = pred; vp.units = 'mm day-1'
        vp.long_name = f'SRCNN ({config.variant}) downscaled precipitation'
        vo = ds.createVariable('pr_obs', 'f4', ('time','lat','lon'),
                                zlib=True, complevel=4, fill_value=-9999.)
        vo[:] = obs; vo.units = 'mm day-1'
        vo.long_name = 'MSWX ground truth precipitation'


# ============================================================
# PLOTS
# ============================================================
PRECIP_CMAP = 'YlGnBu'

def make_plots(obs, pred, lat, lon, out_dir, label, config, logger):
    obs_mean  = obs.mean(axis=0)
    pred_mean = pred.mean(axis=0)
    extent    = [lon.min(), lon.max(), lat.min(), lat.max()]

    # Mean maps
    bias   = pred_mean - obs_mean
    vmax_p = np.percentile(obs_mean, 99)
    bmax   = np.abs(bias).max()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, data, title, cmap, vmin, vmax in [
        (axes[0], obs_mean,  f'Truth ({label})', PRECIP_CMAP, 0, vmax_p),
        (axes[1], pred_mean, f'SRCNN ({label})', PRECIP_CMAP, 0, vmax_p),
        (axes[2], bias,  'Bias (SRCNN - Truth)', 'RdBu_r',  -bmax, bmax),
    ]:
        im = ax.imshow(data, origin='lower', extent=extent,
                        cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(title); ax.set_xlabel('Lon (E)'); ax.set_ylabel('Lat (N)')
        fig.colorbar(im, ax=ax, label='mm day-1', fraction=0.03)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_mean_maps_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_mean_maps_{label}.png')

    # Daily scatter
    obs_ts  = obs.mean(axis=(1, 2))
    pred_ts = pred.mean(axis=(1, 2))
    r_ts    = float(np.corrcoef(obs_ts, pred_ts)[0, 1])
    rmse_ts = float(np.sqrt(np.mean((pred_ts - obs_ts)**2)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs_ts, pred_ts, alpha=0.25, s=8, color='darkcyan', rasterized=True)
    vmax = max(obs_ts.max(), pred_ts.max()) * 1.05
    ax.plot([0, vmax], [0, vmax], 'k--', lw=1, label='1:1')
    ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
    ax.set_xlabel('Truth (mm day-1)'); ax.set_ylabel('SRCNN (mm day-1)')
    ax.set_title(f'Domain-Mean Daily -- {label}\nr={r_ts:.3f}  RMSE={rmse_ts:.3f}')
    ax.legend(); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_scatter_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_scatter_{label}.png')

    # Percentile skill
    obs_pct  = percentile_values(obs,  config.percentiles)
    pred_pct = percentile_values(pred, config.percentiles)
    pcts = sorted(obs_pct.keys()); x = np.arange(len(pcts)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x-w/2, [obs_pct[p]  for p in pcts], w, label='Truth', color='steelblue')
    axes[0].bar(x+w/2, [pred_pct[p] for p in pcts], w, label='SRCNN', color='darkcyan')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'P{p}' for p in pcts], rotation=30)
    axes[0].set_ylabel('mm day-1')
    axes[0].set_title('Percentile Values (wet >= 1 mm day-1)')
    axes[0].legend()
    ratios = [pred_pct[p]/obs_pct[p] if obs_pct[p] > 0 else float('nan') for p in pcts]
    bar_colors = ['tomato' if r > 1.05 else ('steelblue' if r < 0.95 else 'green')
                  for r in ratios]
    axes[1].bar(x, ratios, color=bar_colors, width=0.5)
    axes[1].axhline(1.0, color='k', lw=1.5, ls='--')
    axes[1].axhspan(0.95, 1.05, alpha=0.15, color='green', label='+-5% band')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'P{p}' for p in pcts], rotation=30)
    axes[1].set_ylabel('Ratio (SRCNN / Truth)'); axes[1].legend(fontsize=9)
    plt.suptitle(f'Percentile Preservation -- SRCNN ({label})', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_percentile_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_percentile_{label}.png')

    # FSS
    step = max(1, obs.shape[0] // 200)
    obs_s, pred_s = obs[::step], pred[::step]
    fss_results = {}
    for thr in config.fss_thresholds:
        vals = [fss_score(obs_s, pred_s, thr, sc) for sc in config.fss_scales]
        fss_results[thr] = vals
    colors_fss = ['royalblue', 'tomato', 'seagreen']
    fig, ax = plt.subplots(figsize=(8, 5))
    for (thr, vals), col in zip(fss_results.items(), colors_fss):
        ax.plot(config.fss_scales, vals, '-o', color=col,
                label=f'>={thr} mm day-1', markersize=5)
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='FSS=0.5')
    ax.set_xlabel('Spatial Scale (grid cells at 0.1 deg)'); ax.set_ylabel('FSS')
    ax.set_title(f'FSS vs Scale -- SRCNN ({label})')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_fss_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_fss_{label}.png')

    # PSD
    k_obs, psd_obs   = azimuthal_psd(obs_mean)
    k_pred, psd_pred = azimuthal_psd(pred_mean)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(k_obs,  psd_obs,  color='steelblue', lw=1.5, label='Truth')
    axes[0].semilogy(k_pred, psd_pred, color='darkcyan',  lw=1.5, ls='--', label='SRCNN')
    axes[0].set_title(f'PSD -- {label}'); axes[0].legend(); axes[0].grid(alpha=0.3)
    ratio = psd_pred / (psd_obs + 1e-20)
    axes[1].plot(k_obs, ratio, color='darkcyan', lw=1.5)
    axes[1].axhline(1.0, color='k', ls='--')
    axes[1].axhspan(0.9, 1.1, alpha=0.1, color='green', label='+-10% band')
    axes[1].set_ylim(0, 2)
    axes[1].set_title(f'PSD Ratio -- {label}')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_psd_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_psd_{label}.png')

    return fss_results


# ============================================================
# PER-PERIOD EVALUATION
# ============================================================
def evaluate_period(pred, obs, lat, lon, label, start_year,
                    config, logger, out_dir):
    T = pred.shape[0]
    logger.info(f'\nMetrics for {label}...')

    overall = {
        'rmse':              round(rmse(obs, pred),       4),
        'mae':               round(mae(obs, pred),        4),
        'mean_bias':         round(mean_bias(obs, pred),  4),
        'pearson_r':         round(pearson_r(obs, pred),  4),
        'kge':               round(kge(obs, pred),        4),
        'wet_day_freq_obs':  round(wet_day_frequency(obs),  4),
        'wet_day_freq_pred': round(wet_day_frequency(pred), 4),
    }
    for k, v in overall.items():
        logger.info(f'  {k:<22}: {v}')

    obs_pct   = percentile_values(obs,  config.percentiles)
    pred_pct  = percentile_values(pred, config.percentiles)
    pct_skill = percentile_skill(obs_pct, pred_pct)
    for p in config.percentiles:
        logger.info(f'  P{p:5.1f}: obs={obs_pct[p]:.3f}  '
                    f'pred={pred_pct[p]:.3f}  ratio={pct_skill[p]:.3f}')

    season_labels  = build_season_mask(T, start_year)
    season_metrics = {}
    for s in ['DJF', 'MAM', 'JJA', 'SON']:
        mask = season_labels == s
        if not mask.any(): continue
        o, p = obs[mask], pred[mask]
        season_metrics[s] = {
            'n_days': int(mask.sum()),
            'rmse':   round(rmse(o, p),       4),
            'mae':    round(mae(o, p),         4),
            'bias':   round(mean_bias(o, p),   4),
            'r':      round(pearson_r(o, p),   4),
            'kge':    round(kge(o, p),         4),
        }
        logger.info(f'  {s}: RMSE={season_metrics[s]["rmse"]:.4f}  '
                    f'KGE={season_metrics[s]["kge"]:.4f}')

    k_obs, psd_obs   = azimuthal_psd(obs.mean(axis=0))
    k_pred, psd_pred = azimuthal_psd(pred.mean(axis=0))
    psd_ratio_hk = float(np.median(
        psd_pred[k_pred > 0.3] / (psd_obs[k_obs > 0.3] + 1e-20)
    ))
    logger.info(f'  PSD ratio (k>0.3): {psd_ratio_hk:.4f}')

    fss_results = make_plots(obs, pred, lat, lon, out_dir, label, config, logger)

    metrics = {
        'method':    'srcnn',
        'variant':   config.variant,
        'period':    label,
        'n_timesteps': T,
        'overall':   overall,
        'percentiles_obs':  {str(k): round(v, 3) for k, v in obs_pct.items()},
        'percentiles_pred': {str(k): round(v, 3) for k, v in pred_pct.items()},
        'percentile_skill': {str(k): round(v, 4) for k, v in pct_skill.items()},
        'by_season': season_metrics,
        'fss': {
            str(thr): {str(sc): round(v, 4)
                       for sc, v in zip(config.fss_scales, vals)}
            for thr, vals in fss_results.items()
        },
        'psd_ratio_high_k': round(psd_ratio_hk, 4),
    }
    json_path = os.path.join(out_dir, f'metrics_{label}.json')
    with open(json_path, 'w') as fj:
        json.dump(metrics, fj, indent=2)
    logger.info(f'  Metrics saved: {json_path}')
    return metrics


# ============================================================
# MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained SRCNN checkpoint'
    )
    parser.add_argument('--ckpt',        type=str, default=None,
                        help='Path to best.pt checkpoint')
    parser.add_argument('--variant',     type=str, default=None,
                        choices=['original', 'deep'],
                        help='Must match variant used during training')
    parser.add_argument('--val_h5',      type=str, default=None)
    parser.add_argument('--test_h5',     type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--gpu',         type=int, default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    config = Config()

    if args.ckpt:        config.ckpt        = args.ckpt
    if args.variant:     config.variant     = args.variant
    if args.val_h5:      config.val_h5      = args.val_h5
    if args.test_h5:     config.test_h5     = args.test_h5
    if args.results_dir: config.results_dir = args.results_dir
    if args.gpu is not None: config.gpu     = args.gpu

    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(config.results_dir)

    logger.info('=' * 60)
    logger.info('SRCNN Evaluation')
    logger.info('=' * 60)
    logger.info(f'Checkpoint: {config.ckpt}')
    logger.info(f'Variant:    {config.variant}')
    logger.info(f'Results:    {config.results_dir}')

    device     = torch.device(f'cuda:{config.gpu}')
    normalizer = PrecipNormalizer(config.log1p_max)
    model      = build_model(config.variant, device)
    n_params   = sum(p.numel() for p in model.parameters())

    ckpt = load_checkpoint(config.ckpt, model, device, logger)
    logger.info(f'Parameters: {n_params:,}')

    # Optionally plot loss curves if stored in checkpoint
    train_losses = ckpt.get('train_losses', [])
    val_losses   = ckpt.get('val_losses',   [])
    if train_losses and val_losses:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(train_losses, color='steelblue', label='Train MSE')
        ax.plot(val_losses,   color='tomato', ls='--', label='Val MSE')
        best_ep = int(np.argmin(val_losses))
        ax.axvline(best_ep, color='gray', ls=':', lw=1,
                   label=f'Best ep {best_ep+1}  val={min(val_losses):.5f}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title('SRCNN Training Curves')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        lc_path = os.path.join(config.results_dir, 'fig_loss_curves.png')
        fig.savefig(lc_path, dpi=150, bbox_inches='tight'); plt.close(fig)
        logger.info(f'  fig_loss_curves.png saved')

    all_metrics = {}
    t_all = time.time()

    for period_name, h5_path, label, start_year in [
        ('val',  config.val_h5,  '2006-2010', config.val_start_year),
        ('test', config.test_h5, '2011-2014', config.test_start_year),
    ]:
        logger.info(f'\n{"--"*30}')
        logger.info(f'PERIOD: {label}')
        logger.info(f'{"--"*30}')

        t0 = time.time()
        pred, obs, lat, lon = run_inference(model, normalizer, h5_path, config, logger)

        nc_path = os.path.join(config.results_dir, f'srcnn_pred_{label}.nc')
        save_netcdf(pred, obs, lat, lon, nc_path, config, label)
        logger.info(f'  NetCDF: {nc_path}')

        all_metrics[period_name] = evaluate_period(
            pred, obs, lat, lon, label, start_year,
            config, logger, config.results_dir
        )
        logger.info(f'  {period_name} done in {(time.time()-t0)/60:.1f} min')

    combined_path = os.path.join(config.results_dir, 'metrics_combined.json')
    with open(combined_path, 'w') as fj:
        json.dump(all_metrics, fj, indent=2)

    logger.info('\n' + '='*60)
    logger.info('SUMMARY')
    logger.info('='*60)
    logger.info(f'Variant: {config.variant}  ({n_params:,} params)')
    for pname, m in all_metrics.items():
        o = m['overall']
        logger.info(f"{pname} ({m['period']}): "
                    f"RMSE={o['rmse']:.4f}  KGE={o['kge']:.4f}  "
                    f"r={o['pearson_r']:.4f}  "
                    f"PSD={m['psd_ratio_high_k']:.4f}")
    logger.info(f'Total time: {(time.time()-t_all)/60:.1f} min')
    logger.info(f'Done. Combined: {combined_path}')


if __name__ == '__main__':
    main()