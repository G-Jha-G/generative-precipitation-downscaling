"""
train_srcnn.py
==============
SRCNN (Super-Resolution Convolutional Neural Network) baseline for 10×
precipitation downscaling. Implements Dong et al. (2014) adapted for
single-channel precipitation on a 128×128 patch domain.

SCIENTIFIC POSITION in the comparison:
  Bicubic → QDM → BCSD → CA → XGBoost → SRCNN → U-Net Det → SRGAN → DDPM
  SRCNN is the first deep learning SR method and the simplest CNN baseline.
  Comparing SRCNN vs U-Net Det isolates the contribution of:
    (a) skip connections / encoder-decoder structure
    (b) multi-scale feature extraction
    (c) depth (3 layers vs 4-level encoder-decoder)

ORIGINAL SRCNN (Dong et al. 2014):
  Input:  bicubic-upsampled LR image (already at HR resolution)
  Layer 1: Conv(9×9, 64 filters)  — patch extraction
  Layer 2: Conv(1×1, 32 filters)  — non-linear feature mapping
  Layer 3: Conv(5×5,  1 filter)   — reconstruction
  Output: HR prediction

PRECIPITATION ADAPTATIONS:
  ✓ Single channel (not 3ch RGB)
  ✓ Same normalization as DDPM: log1p → [-1, 1]
  ✓ Same dataset pipeline: full-field LR, 128×128 patches, stride=64
  ✓ Simple MSE loss
  ✓ AdamW + ReduceLROnPlateau + early stopping (patience=15)
  ✓ Same patch-based Hanning overlap-add inference as U-Net Det

ARCHITECTURE OPTIONS (--variant):
  'original'  — Dong et al. exactly: 9-1-5 conv, 64-32 ch  (~20K params)
  'deep'      — deeper variant: 9-3-3-5 conv, 128-64-32 ch (~800K params)
                Larger capacity while remaining far smaller than U-Net Det (~57M)
  The 'deep' variant bridges SRCNN and U-Net Det in the complexity spectrum.

Modes:
  python train_srcnn.py                        (train + eval val + test)
  python train_srcnn.py --mode eval --ckpt ... (eval only)

Outputs:
  results/srcnn_baseline/
    best.pt / latest.pt
    srcnn_pred_2006-2010.nc
    srcnn_pred_2011-2014.nc
    metrics_combined.json
    fig_mean_maps_{period}.png
    fig_scatter_{period}.png
    fig_percentile_{period}.png
    fig_fss_{period}.png
    fig_psd_{period}.png
    fig_loss_curves.png
"""

import os, json, logging, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List
import h5py
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
class Config:
    # ── Data ────────────────────────────────────────────────
    train_h5: str = (
        r'C:\Users\IIT\OneDrive\Ph.D\4th_precip_ddpm\processed\hdf5\mswx_train_1979-2005.h5'
    )
    val_h5: str = (
        r'C:\Users\IIT\OneDrive\Ph.D\4th_precip_ddpm\processed\hdf5\mswx_val_2006-2010.h5'
    )
    test_h5: str = (
        r'C:\Users\IIT\OneDrive\Ph.D\4th_precip_ddpm\processed\hdf5\mswx_test_2011-2014.h5'
    )
    results_dir: str = (
        r'C:\Users\IIT\OneDrive\Ph.D\4th_precip_ddpm\processed\results\srcnn_baseline'
    )

    # ── Architecture ─────────────────────────────────────────
    # 'original': 9-1-5 conv, 64-32 ch  (~20K params, Dong 2014 exactly)
    # 'deep':     9-3-3-5 conv, 128-64-32 ch (~800K params, extended)
    variant: str = 'deep'

    # ── Normalization (identical to DDPM v5b) ────────────────
    log1p_max: float = 6.7859

    # ── Scale / patches (identical to DDPM v5b) ──────────────
    scale_factor: int   = 10
    patch_size:   int   = 128
    stride:       int   = 64

    # ── Training ─────────────────────────────────────────────
    batch_size:      int   = 16
    epochs:          int   = 100
    patience:        int   = 15
    learning_rate:   float = 1e-4
    weight_decay:    float = 1e-5
    gradient_clip:   float = 1.0
    mixed_precision: bool  = True

    # ── System ───────────────────────────────────────────────
    num_workers: int = 0 if os.name == 'nt' else 4
    gpu:         int = 0
    seed:        int = 42

    # ── Inference ────────────────────────────────────────────
    infer_batch_size: int = 64    # patches per forward pass (SRCNN is tiny)

    # ── Evaluation ───────────────────────────────────────────
    fss_thresholds:  tuple = (1.0, 5.0, 20.0)
    fss_scales:      tuple = (1, 3, 5, 10, 15, 20, 30, 50)
    percentiles:     tuple = (50, 75, 90, 95, 99, 99.5, 99.9)
    val_start_year:  int   = 2006
    test_start_year: int   = 2011


# ============================================================
# LOGGING
# ============================================================
def setup_logging(results_dir: str, tag: str = '') -> logging.Logger:
    stamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(results_dir, f'srcnn_{tag}_{stamp}.log')
    name     = f'srcnn_{tag}_{stamp}'
    logger   = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                            '%Y-%m-%d %H:%M:%S')
    for h in [logging.FileHandler(log_file), logging.StreamHandler()]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ============================================================
# NORMALIZATION  (identical to DDPM v5b)
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
# LR GENERATION  (identical to DDPM v5b full-field pipeline)
# ============================================================
def compute_lr_fullfield(hr_batch: np.ndarray,
                          scale_factor: int, H: int, W: int) -> np.ndarray:
    """HR [B,H,W] → avg_pool(sf) → bicubic upsample → [B,H,W]. CPU-safe."""
    t    = torch.from_numpy(hr_batch.astype(np.float32)).unsqueeze(1)
    lr   = F.avg_pool2d(t, scale_factor, scale_factor)
    pred = F.interpolate(lr, size=(H, W), mode='bicubic',
                         align_corners=False).clamp(min=0.0)
    return pred.squeeze(1).numpy()


# ============================================================
# DATASET  (identical to DDPM v5b)
# ============================================================
class PrecipDataset(Dataset):
    def __init__(self, h5_path, normalizer, patch_size=128, stride=64,
                 scale_factor=10, augment=True):
        self.normalizer = normalizer
        self.patch_size = patch_size
        self.augment    = augment

        print(f'Loading {os.path.basename(h5_path)} into RAM...')
        t0 = time.time()
        with h5py.File(h5_path, 'r') as f:
            hr_data = f['precipitation'][:]

        T, H, W = hr_data.shape
        print(f'  {T}x{H}x{W}  ({hr_data.nbytes/1e9:.2f} GB) '
              f'in {time.time()-t0:.1f}s')
        assert H % scale_factor == 0 and W % scale_factor == 0

        self.hr_data = hr_data
        self.H, self.W = H, W

        # Pre-compute full-field LR for all timesteps
        print('  Pre-computing full-field LR...')
        lr_data = np.zeros_like(hr_data, dtype=np.float32)
        for t_start in tqdm(range(0, T, 50), desc='  LR gen'):
            t_end = min(t_start + 50, T)
            lr_data[t_start:t_end] = compute_lr_fullfield(
                hr_data[t_start:t_end].astype(np.float32), scale_factor, H, W
            )
        self.lr_data = lr_data

        # Build patch grid — all patches from all days, no filtering
        tops  = list(range(0, H - patch_size + 1, stride))
        lefts = list(range(0, W - patch_size + 1, stride))
        if not tops  or tops[-1]  != H - patch_size: tops.append(H - patch_size)
        if not lefts or lefts[-1] != W - patch_size: lefts.append(W - patch_size)
        all_pos    = [(tp, lf) for tp in tops for lf in lefts]
        self.index = [(t, tp, lf) for t in range(T) for tp, lf in all_pos]

        print(f'  Dataset ready: {len(self.index):,} patches')

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        t, top, left = self.index[idx]
        p  = self.patch_size
        hr = self.normalizer.normalize(
            self.hr_data[t, top:top+p, left:left+p].copy()
        )
        lr = self.normalizer.normalize(
            self.lr_data[t, top:top+p, left:left+p].copy()
        )
        if self.augment:
            if np.random.rand() > 0.5:
                hr = hr[:, ::-1].copy(); lr = lr[:, ::-1].copy()
            if np.random.rand() > 0.5:
                hr = hr[::-1].copy(); lr = lr[::-1].copy()
        return (torch.from_numpy(lr).unsqueeze(0),
                torch.from_numpy(hr).unsqueeze(0))



# ============================================================
# SRCNN ARCHITECTURE
# ============================================================
class SRCNNOriginal(nn.Module):
    """
    Dong et al. (2014) SRCNN — exactly as published.

    Input:  LR bicubic [B, 1, 128, 128] — normalized
    Output: HR pred    [B, 1, 128, 128] — normalized

    Three layers:
      f1=9, n1=64: patch extraction — large kernel captures local context
      f2=1, n2=32: non-linear mapping — 1×1 acts as per-pixel MLP
      f3=5, n3=1:  reconstruction — smooths final prediction

    Padding: SAME padding (pad = kernel//2) so spatial dims are preserved.
    Activation: ReLU (original paper).
    No batch norm, no skip connections — the key differences from U-Net.

    Parameter count: ~20K — the smallest deep learning baseline.
    """
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
    """
    Extended SRCNN with additional layers and wider channels.
    Bridges the gap between SRCNN (~20K) and U-Net Det (~57M).

    Architecture (9-3-3-5 conv, 128-64-32 ch):
      Layer 1: Conv(9×9, 1→128)   — large receptive field patch extraction
      Layer 2: Conv(3×3, 128→64)  — feature refinement
      Layer 3: Conv(3×3, 64→32)   — non-linear mapping
      Layer 4: Conv(5×5, 32→1)    — smooth reconstruction

    Still no skip connections (that's U-Net), no attention, no time conditioning.
    Uses ReLU for direct comparison with original. ~800K parameters.

    Scientific value: shows whether adding depth alone (SRCNN→SRCNNDeep)
    gives the same gains as adding skip connections (SRCNNDeep→U-Net Det).
    """
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
        model = SRCNNOriginal(in_ch=1).to(device)
    elif variant == 'deep':
        model = SRCNNDeep(in_ch=1).to(device)
    else:
        raise ValueError(f'Unknown variant: {variant}. Choose original or deep.')
    return model


# ============================================================
# TRAINER
# ============================================================
class Trainer:
    """
    Training loop — simple MSE loss, no diffusion.
    Identical structure to U-Net Det trainer for fair comparison.
    """
    def __init__(self, model, config, device, logger):
        self.model  = model
        self.config = config
        self.device = device
        self.logger = logger

        self.opt = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay, betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=5, factor=0.5
        )
        self.scaler = GradScaler('cuda') if config.mixed_precision else None

        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []
        self.best_val    = float('inf')
        self.patience_ctr = 0

    def _loss(self, pred: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Simple MSE loss."""
        return F.mse_loss(pred, hr)

    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total = 0.0
        pbar  = tqdm(loader, desc=f'Epoch {epoch:03d} [train]', ncols=90)
        for lr_b, hr_b in pbar:
            lr_b = lr_b.to(self.device, non_blocking=True)
            hr_b = hr_b.to(self.device, non_blocking=True)
            self.opt.zero_grad(set_to_none=True)
            if self.scaler:
                with autocast('cuda'):
                    pred = self.model(lr_b)
                    loss = self._loss(pred, hr_b)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                          self.config.gradient_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                pred = self.model(lr_b)
                loss = self._loss(pred, hr_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                          self.config.gradient_clip)
                self.opt.step()
            total += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.5f}'})
        return total / len(loader)

    @torch.no_grad()
    def val_epoch(self, loader) -> float:
        self.model.eval()
        total = 0.0
        for lr_b, hr_b in loader:
            lr_b = lr_b.to(self.device, non_blocking=True)
            hr_b = hr_b.to(self.device, non_blocking=True)
            with autocast('cuda'):
                pred = self.model(lr_b)
                loss = F.mse_loss(pred, hr_b)
            total += loss.item()
        return total / len(loader)

    def save(self, epoch: int, is_best: bool = False):
        m    = getattr(self.model, '_orig_mod', self.model)
        ckpt = {
            'epoch':            epoch,
            'model_state_dict': m.state_dict(),
            'opt_state_dict':   self.opt.state_dict(),
            'best_val':         self.best_val,
            'train_losses':     self.train_losses,
            'val_losses':       self.val_losses,
            'variant':          self.config.variant,
        }
        if self.scaler: ckpt['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(ckpt, os.path.join(self.config.results_dir, 'latest.pt'))
        if is_best:
            torch.save(ckpt, os.path.join(self.config.results_dir, 'best.pt'))
            self.logger.info(f'  >> Best saved (val={self.best_val:.5f})')

    def train(self, train_loader, val_loader):
        self.logger.info('Starting training...')
        for epoch in range(1, self.config.epochs + 1):
            t0  = time.time()
            tl  = self.train_epoch(train_loader, epoch)
            vl  = self.val_epoch(val_loader)
            self.train_losses.append(tl)
            self.val_losses.append(vl)
            self.scheduler.step(vl)
            elapsed = time.time() - t0
            self.logger.info(
                f'Epoch {epoch:03d}/{self.config.epochs} | '
                f'train={tl:.5f} | val={vl:.5f} | '
                f'lr={self.opt.param_groups[0]["lr"]:.2e} | '
                f'{elapsed/60:.1f}min'
            )
            is_best = vl < self.best_val
            if is_best:
                self.best_val = vl; self.patience_ctr = 0
            else:
                self.patience_ctr += 1
            self.save(epoch, is_best)
            if self.patience_ctr >= self.config.patience:
                self.logger.info(f'Early stopping at epoch {epoch}')
                break
        self.logger.info(f'Done. Best val: {self.best_val:.5f}')
        return self.train_losses, self.val_losses


# ============================================================
# INFERENCE  (Hanning overlap-add — same as U-Net Det)
# ============================================================
def build_patch_grid(H, W, patch_size, stride):
    tops  = list(range(0, H - patch_size + 1, stride))
    lefts = list(range(0, W - patch_size + 1, stride))
    if not tops  or tops[-1]  != H - patch_size: tops.append(H - patch_size)
    if not lefts or lefts[-1] != W - patch_size: lefts.append(W - patch_size)
    return [(tp, lf) for tp in tops for lf in lefts]

def make_hanning_window(patch_size):
    w = np.hanning(patch_size).astype(np.float32)
    return np.outer(w, w)

@torch.no_grad()
def infer_one_field(model, lr_field, normalizer, positions, window,
                     patch_size, batch_size, device):
    H, W     = lr_field.shape
    pred_sum = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)
    lr_norm  = normalizer.normalize(lr_field)

    model.eval()
    batch_patches, batch_pos = [], []

    def flush():
        if not batch_patches: return
        inp = torch.stack(batch_patches).to(device)   # [B, 1, P, P]
        with autocast('cuda'):
            out = model(inp).squeeze(1).cpu().numpy()  # [B, P, P]
        for (tp, lf), p_norm in zip(batch_pos, out):
            p_mm = normalizer.denormalize(p_norm)
            pred_sum[tp:tp+patch_size, lf:lf+patch_size] += p_mm  * window
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
    """Patch-based inference on a full HDF5 period. Returns (pred, obs, lat, lon)."""
    logger.info(f'Loading: {h5_path}')
    with h5py.File(h5_path, 'r') as f:
        hr_data = f['precipitation'][:]
        lat = f['lat'][:] if 'lat' in f else np.linspace(6.5, 41.0, 350)
        lon = f['lon'][:] if 'lon' in f else np.linspace(66.5, 100.0, 350)

    T, H, W   = hr_data.shape
    device    = next(model.parameters()).device
    positions = build_patch_grid(H, W, config.patch_size, config.stride)
    window    = make_hanning_window(config.patch_size)
    pred      = np.zeros((T, H, W), dtype=np.float32)

    logger.info(f'  {T}×{H}×{W} | {len(positions)} patches/timestep')
    t0 = time.time()
    for t_idx in tqdm(range(T), desc='Inference'):
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
# METRICS  (identical schema to all other baselines)
# ============================================================
def rmse(o, p):      return float(np.sqrt(np.mean((p-o)**2)))
def mae(o, p):       return float(np.mean(np.abs(p-o)))
def mean_bias(o, p): return float(np.mean(p-o))
def pearson_r(o, p): return float(np.corrcoef(o.ravel(), p.ravel())[0,1])

def kge(o, p):
    o, p  = o.ravel(), p.ravel()
    r     = float(np.corrcoef(o, p)[0,1])
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
    num = np.mean((of-pf)**2); den = np.mean(of**2+pf**2)
    return float(1-num/den) if den > 0 else float('nan')

def azimuthal_psd(field_2d):
    H, W   = field_2d.shape
    power  = np.abs(np.fft.fftshift(np.fft.fft2(field_2d)))**2
    kx     = np.fft.fftshift(np.fft.fftfreq(W))
    ky     = np.fft.fftshift(np.fft.fftfreq(H))
    KX, KY = np.meshgrid(kx, ky)
    K      = np.sqrt(KX**2 + KY**2)
    n_bins = min(H, W) // 2
    bins   = np.linspace(0, 0.5, n_bins + 1)
    k_mid  = 0.5*(bins[:-1]+bins[1:])
    psd    = np.array([power[(K>=bins[i])&(K<bins[i+1])].mean()
                       if ((K>=bins[i])&(K<bins[i+1])).any() else 0.0
                       for i in range(n_bins)])
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
        ds.title            = f'SRCNN baseline ({label})'
        ds.description      = (f'SRCNN variant={config.variant}, '
                                f'simple MSE')
        ds.downscale_factor = config.scale_factor
        ds.variant          = config.variant
        ds.created          = datetime.now().isoformat()
        ds.createDimension('time', T)
        ds.createDimension('lat',  H)
        ds.createDimension('lon',  W)
        vt = ds.createVariable('time', 'i4', ('time',))
        vt[:] = np.arange(T); vt.units = f'days since {label[:4]}-01-01'
        vlat = ds.createVariable('lat', 'f4', ('lat',))
        vlat[:] = lat; vlat.units = 'degrees_north'
        vlon = ds.createVariable('lon', 'f4', ('lon',))
        vlon[:] = lon; vlon.units = 'degrees_east'
        vp = ds.createVariable('pr_srcnn', 'f4', ('time','lat','lon'),
                                zlib=True, complevel=4, fill_value=-9999.)
        vp[:] = pred; vp.units = 'mm day-1'
        vp.long_name = f'SRCNN ({config.variant}) precipitation'
        vo = ds.createVariable('pr_obs', 'f4', ('time','lat','lon'),
                                zlib=True, complevel=4, fill_value=-9999.)
        vo[:] = obs; vo.units = 'mm day-1'
        vo.long_name = 'MSWX ground truth precipitation'


# ============================================================
# PLOTS  (no cfeature.BORDERS anywhere)
# ============================================================
PRECIP_CMAP = 'YlGnBu'

def plot_loss_curves(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, color='steelblue', label='Train (simple MSE)')
    ax.plot(val_losses,   color='tomato',    ls='--', label='Val (simple MSE)')
    best_ep = int(np.argmin(val_losses))
    ax.axvline(best_ep, color='gray', ls=':', lw=1, label=f'Best ep {best_ep+1}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('SRCNN Training Curves')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)

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
        (axes[0], obs_mean,  f'Truth ({label})',  PRECIP_CMAP, 0, vmax_p),
        (axes[1], pred_mean, f'SRCNN ({label})',  PRECIP_CMAP, 0, vmax_p),
        (axes[2], bias,      'Bias (SRCNN−Truth)','RdBu_r',  -bmax, bmax),
    ]:
        im = ax.imshow(data, origin='lower', extent=extent,
                        cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(title); ax.set_xlabel('Lon (°E)'); ax.set_ylabel('Lat (°N)')
        fig.colorbar(im, ax=ax, label='mm day⁻¹', fraction=0.03)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_mean_maps_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_mean_maps_{label}.png ✓')

    # Daily scatter
    obs_ts  = obs.mean(axis=(1,2)); pred_ts = pred.mean(axis=(1,2))
    r_ts    = float(np.corrcoef(obs_ts, pred_ts)[0,1])
    rmse_ts = float(np.sqrt(np.mean((pred_ts-obs_ts)**2)))
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(obs_ts, pred_ts, alpha=0.25, s=8, color='darkcyan', rasterized=True)
    vmax = max(obs_ts.max(), pred_ts.max()) * 1.05
    ax.plot([0,vmax],[0,vmax],'k--',lw=1,label='1:1')
    ax.set_xlim(0,vmax); ax.set_ylim(0,vmax)
    ax.set_xlabel('Truth (mm day⁻¹)'); ax.set_ylabel('SRCNN (mm day⁻¹)')
    ax.set_title(f'Domain-Mean Daily — {label}\nr={r_ts:.3f}  RMSE={rmse_ts:.3f}')
    ax.legend(); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_scatter_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_scatter_{label}.png ✓')

    # Percentile skill
    obs_pct  = percentile_values(obs,  config.percentiles)
    pred_pct = percentile_values(pred, config.percentiles)
    pcts = sorted(obs_pct.keys()); x = np.arange(len(pcts)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x-w/2, [obs_pct[p] for p in pcts],  w, label='Truth', color='steelblue')
    axes[0].bar(x+w/2, [pred_pct[p] for p in pcts], w, label='SRCNN', color='darkcyan')
    axes[0].set_xticks(x); axes[0].set_xticklabels([f'P{p}' for p in pcts], rotation=30)
    axes[0].set_ylabel('mm day⁻¹'); axes[0].set_title('Percentile Values (wet ≥1 mm/day)')
    axes[0].legend()
    ratios = [pred_pct[p]/obs_pct[p] if obs_pct[p]>0 else float('nan') for p in pcts]
    colors = ['tomato' if r>1.05 else ('steelblue' if r<0.95 else 'green') for r in ratios]
    axes[1].bar(x, ratios, color=colors, width=0.5)
    axes[1].axhline(1.0, color='k', lw=1.5, ls='--')
    axes[1].axhspan(0.95, 1.05, alpha=0.15, color='green', label='±5% band')
    axes[1].set_xticks(x); axes[1].set_xticklabels([f'P{p}' for p in pcts], rotation=30)
    axes[1].set_ylabel('Ratio (SRCNN/Truth)'); axes[1].legend(fontsize=9)
    plt.suptitle(f'Percentile Preservation — SRCNN ({label})', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_percentile_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_percentile_{label}.png ✓')

    # FSS curves
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
                label=f'≥{thr} mm day⁻¹', markersize=5)
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='FSS=0.5')
    ax.set_xlabel('Spatial Scale (grid cells at 0.1°)'); ax.set_ylabel('FSS')
    ax.set_title(f'FSS vs Scale — SRCNN ({label})\n'
                 '(MSE loss → over-smooth → FSS expected below U-Net Det)')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_fss_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_fss_{label}.png ✓')

    # PSD
    k_obs, psd_obs   = azimuthal_psd(obs_mean)
    k_pred, psd_pred = azimuthal_psd(pred_mean)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(k_obs, psd_obs,  color='steelblue', lw=1.5, label='Truth')
    axes[0].semilogy(k_pred,psd_pred, color='darkcyan',  lw=1.5, ls='--', label='SRCNN')
    axes[0].set_title(f'PSD — {label}'); axes[0].legend(); axes[0].grid(alpha=0.3)
    ratio = psd_pred / (psd_obs + 1e-20)
    axes[1].plot(k_obs, ratio, color='darkcyan', lw=1.5)
    axes[1].axhline(1.0, color='k', ls='--')
    axes[1].axhspan(0.9, 1.1, alpha=0.1, color='green', label='±10% band')
    axes[1].set_ylim(0, 2); axes[1].set_title(f'PSD Ratio — {label}')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_psd_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_psd_{label}.png ✓')

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
    for s in ['DJF','MAM','JJA','SON']:
        mask = season_labels == s
        if not mask.any(): continue
        o, p = obs[mask], pred[mask]
        season_metrics[s] = {
            'n_days': int(mask.sum()),
            'rmse': round(rmse(o,p),4), 'mae': round(mae(o,p),4),
            'bias': round(mean_bias(o,p),4), 'r': round(pearson_r(o,p),4),
            'kge':  round(kge(o,p),4),
        }
        logger.info(f'  {s}: RMSE={season_metrics[s]["rmse"]:.4f}  '
                    f'KGE={season_metrics[s]["kge"]:.4f}')

    # PSD ratio
    k_obs, psd_obs   = azimuthal_psd(obs.mean(axis=0))
    k_pred, psd_pred = azimuthal_psd(pred.mean(axis=0))
    psd_ratio_hk = float(np.median(
        psd_pred[k_pred>0.3] / (psd_obs[k_obs>0.3]+1e-20)
    ))
    logger.info(f'  PSD ratio (k>0.3): {psd_ratio_hk:.4f}')

    # Plots (also returns FSS)
    fss_results = make_plots(obs, pred, lat, lon, out_dir, label, config, logger)

    metrics = {
        'method':     'srcnn',
        'variant':    config.variant,
        'period':     label,
        'n_timesteps': T,
        'overall':    overall,
        'percentiles_obs':  {str(k): round(v,3) for k,v in obs_pct.items()},
        'percentiles_pred': {str(k): round(v,3) for k,v in pred_pct.items()},
        'percentile_skill': {str(k): round(v,4) for k,v in pct_skill.items()},
        'by_season':  season_metrics,
        'fss': {str(thr): {str(sc): round(v,4)
                            for sc,v in zip(config.fss_scales, vals)}
                for thr,vals in fss_results.items()},
        'psd_ratio_high_k': round(psd_ratio_hk, 4),
    }
    json_path = os.path.join(out_dir, f'metrics_{label}.json')
    with open(json_path, 'w') as fj:
        json.dump(metrics, fj, indent=2)
    logger.info(f'  Metrics: {json_path}')
    return metrics


# ============================================================
# CHECKPOINT LOADER
# ============================================================
def load_checkpoint(ckpt_path, model, device, logger):
    """Load model weights from checkpoint."""
    logger.info(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f'  Weights loaded (epoch {ckpt["epoch"]}  '
                f'best_val={ckpt["best_val"]:.5f})')
    return model, ckpt.get('train_losses', []), ckpt.get('val_losses', [])


# ============================================================
# MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='SRCNN baseline for precipitation downscaling'
    )
    parser.add_argument('--mode', choices=['train', 'eval', 'train_eval'],
                        default='train_eval')
    parser.add_argument('--variant', choices=['original','deep'], default=None,
                        help='original (~20K) or deep (~800K params)')
    parser.add_argument('--ckpt',        type=str, default=None)
    parser.add_argument('--train_h5',    type=str, default=None)
    parser.add_argument('--val_h5',      type=str, default=None)
    parser.add_argument('--test_h5',     type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--gpu',         type=int, default=None)
    parser.add_argument('--epochs',      type=int, default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    config = Config()

    if args.train_h5:    config.train_h5    = args.train_h5
    if args.val_h5:      config.val_h5      = args.val_h5
    if args.test_h5:     config.test_h5     = args.test_h5
    if args.results_dir: config.results_dir = args.results_dir
    if args.gpu is not None: config.gpu     = args.gpu
    if args.epochs:      config.epochs      = args.epochs
    if args.variant:     config.variant     = args.variant

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # mkdir BEFORE setup_logging
    out_dir = Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir), args.mode)

    device     = torch.device(f'cuda:{config.gpu}')
    normalizer = PrecipNormalizer(config.log1p_max)
    model      = build_model(config.variant, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info('=' * 65)
    logger.info('SRCNN Baseline — Precipitation Downscaling (Dong et al. 2014)')
    logger.info('=' * 65)
    logger.info(f'Mode:          {args.mode}')
    logger.info(f'Variant:       {config.variant}')
    logger.info(f'Parameters:    {n_params:,}  '
                f'(original ~20K, deep ~800K, U-Net Det ~57M, DDPM ~60M)')
    logger.info(f'Scale factor:  {config.scale_factor}× '
                f'(LR = {0.1*config.scale_factor:.1f}°)')
    logger.info(f'Loss:          simple MSE')
    logger.info(f'Device:        {device}')

    train_losses, val_losses = [], []

    # ── TRAINING ────────────────────────────────────────────
    if args.mode in ('train', 'train_eval'):
        logger.info('\n--- BUILDING DATASETS ---')
        train_ds = PrecipDataset(
            config.train_h5, normalizer,
            config.patch_size, config.stride, config.scale_factor,
            augment=True,
        )
        val_ds = PrecipDataset(
            config.val_h5, normalizer,
            config.patch_size, config.stride, config.scale_factor,
            augment=False,
        )
        train_loader = DataLoader(
            train_ds, config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True
        )
        logger.info(f'Train batches: {len(train_loader):,}  '
                    f'Val batches: {len(val_loader):,}')

        logger.info('\n--- TRAINING ---')
        trainer = Trainer(model, config, device, logger)
        train_losses, val_losses = trainer.train(train_loader, val_loader)

        # Load best weights for evaluation
        best_ckpt = os.path.join(str(out_dir), 'best.pt')
        model, train_losses, val_losses = load_checkpoint(
            best_ckpt, model, device, logger
        )

        # Save loss curves
        plot_loss_curves(
            train_losses, val_losses,
            os.path.join(str(out_dir), 'fig_loss_curves.png')
        )
        logger.info('  fig_loss_curves.png ✓')

    # ── EVALUATION ──────────────────────────────────────────
    if args.mode in ('eval', 'train_eval'):
        if args.mode == 'eval':
            ckpt_path = args.ckpt or os.path.join(str(out_dir), 'best.pt')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f'Checkpoint not found: {ckpt_path}. '
                    f'Run --mode train first or pass --ckpt.'
                )
            model, train_losses, val_losses = load_checkpoint(
                ckpt_path, model, device, logger
            )
            if train_losses and val_losses:
                plot_loss_curves(
                    train_losses, val_losses,
                    os.path.join(str(out_dir), 'fig_loss_curves.png')
                )

        all_metrics = {}
        t_total     = time.time()

        for period_name, h5_path, label, start_year in [
            ('val',  config.val_h5,  '2006-2010', config.val_start_year),
            ('test', config.test_h5, '2011-2014', config.test_start_year),
        ]:
            logger.info(f'\n{"─"*55}')
            logger.info(f'PERIOD: {label}')
            logger.info(f'{"─"*55}')

            t0 = time.time()
            pred, obs, lat, lon = run_inference(
                model, normalizer, h5_path, config, logger
            )

            nc_path = os.path.join(str(out_dir), f'srcnn_pred_{label}.nc')
            save_netcdf(pred, obs, lat, lon, nc_path, config, label)
            logger.info(f'  NetCDF: {nc_path}')

            all_metrics[period_name] = evaluate_period(
                pred, obs, lat, lon, label, start_year,
                config, logger, str(out_dir)
            )
            logger.info(f'  {period_name} done in {(time.time()-t0)/60:.1f} min')

        combined_path = os.path.join(str(out_dir), 'metrics_combined.json')
        with open(combined_path, 'w') as fj:
            json.dump(all_metrics, fj, indent=2)

        logger.info('\n' + '='*65)
        logger.info('SUMMARY')
        logger.info('='*65)
        logger.info(f'Variant: {config.variant}  ({n_params:,} params)')
        for pname, m in all_metrics.items():
            o = m['overall']
            logger.info(f"{pname} ({m['period']}): "
                        f"RMSE={o['rmse']:.4f}  KGE={o['kge']:.4f}  "
                        f"r={o['pearson_r']:.4f}  "
                        f"PSD={m['psd_ratio_high_k']:.4f}")
        logger.info(f'Total time: {(time.time()-t_total)/60:.1f} min')
        logger.info(f'Done. Combined: {combined_path}')


if __name__ == '__main__':
    main()