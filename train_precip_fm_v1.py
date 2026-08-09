"""
Precipitation downscaling — Flow Matching trainer (v1, derived from DDPM v5b).

Replaces the DDPM ε-prediction objective with Conditional Flow Matching
(CFM) using independent Gaussian coupling (straight-line / rectified-flow
interpolant). The U-Net, data pipeline, normalization, full-field LR
generation, LR-threshold filtering, augmentation, EMA, AMP, AdamW, and
ReduceLROnPlateau are preserved bit-for-bit from train_precip_ddpm_v5b.py.

Convention used throughout this file:
    t = 0  →  noise   (standard Gaussian)
    t = 1  →  data    (HR, log1p-normalized to [-1, 1])
    Interpolant:           x_t = (1 - t) * z + t * x_1,    z ~ N(0, I)
    Target velocity:       u_t = x_1 - z                    (constant along path)
    Training loss:         L = E[ w(x_1) * || v_theta(x_t, c, t) - (x_1 - z) ||^2 ]
    Sampling:              integrate dx/dt = v_theta(x, c, t) from t=0 to t=1

References:
    Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
    Liu et al.,    "Flow Straight and Fast: Rectified Flow", ICLR 2023.
    Tong et al.,   "Conditional Flow Matching: Simulation-Free Dynamic
                    Optimal Transport", TMLR 2024.
"""
import os, math, time, h5py, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
@dataclass
class Config:
    # ── Data paths (unchanged from v5b) ───────────────────────
    train_h5: str = '/path/to/project/PhD_Precipitation/02_Data/processed/hdf5/mswx_train_1979-2005.h5'
    val_h5:   str = '/path/to/project/PhD_Precipitation/02_Data/processed/hdf5/mswx_val_2006-2010.h5'
    results_dir: str = './results/precip_fm_v1'

    # ── Precipitation normalization (unchanged from v5b) ──────
    log1p_max: float = 6.7859

    # ── Scale factor (unchanged from v5b) ─────────────────────
    scale_factor: int = 10

    # ── Patch settings (unchanged from v5b) ───────────────────
    patch_size: int = 128
    stride:     int = 64

    # ── LR threshold filter (unchanged from v5b) ──────────────
    lr_patch_threshold: float = 1.0   # mm/day

    # ── Training (unchanged from v5b) ─────────────────────────
    batch_size:      int   = 8
    epochs:          int   = 100
    patience:        int   = 15
    learning_rate:   float = 1e-4
    weight_decay:    float = 1e-5
    gradient_clip:   float = 1.0
    mixed_precision: bool  = True

    # ── Flow Matching (replaces DDPM scheduler) ───────────────
    # Continuous t ∈ [0,1]. We pass `t_scaled = t * t_scale` to the
    # sinusoidal embedding so it sees the same numerical range as in
    # the DDPM model (preserving the time-MLP frequency design).
    t_scale: float = 1000.0

    # Sampling for validation / monitoring (not used in training loss).
    # The training loss is simulation-free; no solver is needed during training.
    fm_steps: int = 50      # number of ODE steps used at validation/inference
    fm_solver: str = 'heun' # 'euler' | 'midpoint' | 'heun'

    # ── U-Net (unchanged from v5b) ────────────────────────────
    base_dim:    int   = 64
    ch_mult_cap: int   = 8
    time_dim:    int   = 256
    use_ema:     bool  = True
    ema_decay:   float = 0.9999

    # ── System (unchanged from v5b) ───────────────────────────
    num_workers:     int   = 4
    gpu:             int   = 0
    seed:            int   = 42
    train_subsample: float = 0.30

    # ── Extreme-value weighting (unchanged from v5b) ──────────
    extreme_weight_alpha: float = 2.0

    def __post_init__(self):
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================
def setup_logging(results_dir: str) -> logging.Logger:
    log_file = os.path.join(
        results_dir,
        f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logger = logging.getLogger('precip_fm_v1')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    for handler in [logging.FileHandler(log_file), logging.StreamHandler()]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


# ============================================================
# NORMALIZATION (unchanged from v5b)
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

    def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.log1p(x.clamp(min=0)) / self.log1p_max) * 2.0 - 1.0

    def denormalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(
            (x + 1.0) / 2.0 * self.log1p_max
        ).clamp(min=0)


# ============================================================
# FULL-FIELD LR GENERATION (unchanged from v5b)
# ============================================================
def compute_lr_fullfield(hr_batch: np.ndarray,
                         scale_factor: int,
                         H: int, W: int) -> np.ndarray:
    t     = torch.from_numpy(hr_batch).unsqueeze(1).float()
    lr    = F.avg_pool2d(t, scale_factor, scale_factor)
    lr_up = F.interpolate(
        lr, size=(H, W), mode='bicubic', align_corners=False
    ).clamp(min=0)
    return lr_up.squeeze(1).numpy()


# ============================================================
# DATASET (unchanged from v5b — pipeline is generative-model-agnostic)
# ============================================================
class PrecipDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        normalizer: PrecipNormalizer,
        patch_size:   int   = 128,
        stride:       int   = 64,
        scale_factor: int   = 10,
        augment:      bool  = True,
        lr_threshold: float = 1.0,
        subsample:    float = 1.0,
    ):
        self.normalizer   = normalizer
        self.patch_size   = patch_size
        self.scale_factor = scale_factor
        self.augment      = augment
        self.lr_threshold = lr_threshold
        self._subsample   = subsample

        print(f"Loading {os.path.basename(h5_path)} into RAM...")
        t0 = time.time()
        with h5py.File(h5_path, 'r') as f:
            hr_data = f['precipitation'][:]

        T, H, W = hr_data.shape
        print(f"  Loaded:       {T} × {H} × {W}  ({hr_data.nbytes/1e9:.2f} GB) "
              f"in {time.time()-t0:.1f}s")
        print(f"  Range:        [{hr_data.min():.2f}, {hr_data.max():.2f}] mm/day")
        print(f"  Zero frac:    {(hr_data==0).mean():.1%}")

        assert H % scale_factor == 0, (
            f"H={H} not divisible by scale_factor={scale_factor}")
        assert W % scale_factor == 0, (
            f"W={W} not divisible by scale_factor={scale_factor}")

        self.hr_data = hr_data
        self.H, self.W = H, W

        print(f"\n  Pre-computing full-field LR (scale={scale_factor}×)...")
        t0 = time.time()
        lr_data  = np.zeros_like(hr_data, dtype=np.float32)
        batch_sz = 50
        for t_start in tqdm(range(0, T, batch_sz),
                            desc="  Computing full-field LR"):
            t_end = min(t_start + batch_sz, T)
            lr_data[t_start:t_end] = compute_lr_fullfield(
                hr_data[t_start:t_end].astype(np.float32),
                scale_factor, H, W
            )
        print(f"  Done in {(time.time()-t0)/60:.1f} min")
        print(f"  LR range: [{lr_data.min():.2f}, {lr_data.max():.2f}] mm/day")
        self.lr_data = lr_data

        tops  = list(range(0, H - patch_size + 1, stride))
        lefts = list(range(0, W - patch_size + 1, stride))
        if not tops  or tops[-1]  != H - patch_size:  tops.append(H - patch_size)
        if not lefts or lefts[-1] != W - patch_size: lefts.append(W - patch_size)
        all_positions  = [(top, left) for top in tops for left in lefts]
        n_pos          = len(all_positions)
        total_possible = T * n_pos
        print(f"\n  Patch grid: {len(tops)}×{len(lefts)} = "
              f"{n_pos} positions/timestep  |  total {total_possible:,}")

        if lr_threshold > 0:
            print(f"\n  Filtering: mean LR ≥ {lr_threshold} mm/day...")
            t0 = time.time()
            lr_means = np.zeros((T, n_pos), dtype=np.float32)
            for t_start in tqdm(range(0, T, batch_sz),
                                desc="  Computing LR patch means"):
                t_end    = min(t_start + batch_sz, T)
                lr_chunk = lr_data[t_start:t_end]
                for p_idx, (top, left) in enumerate(all_positions):
                    patch_lr = lr_chunk[
                        :, top:top+patch_size, left:left+patch_size
                    ]
                    lr_means[t_start:t_end, p_idx] = patch_lr.mean(axis=(1, 2))
            valid_t, valid_p = np.where(lr_means >= lr_threshold)
            self.index = [
                (int(t), *all_positions[p])
                for t, p in zip(valid_t, valid_p)
            ]
            elapsed  = time.time() - t0
            pct_kept = len(self.index) / total_possible * 100
            print(f"  Done in {elapsed/60:.1f} min")
            print(f"  Kept: {len(self.index):,} / {total_possible:,} "
                  f"({pct_kept:.1f}%)")
        else:
            self.index = [
                (t, top, left)
                for t in range(T)
                for top, left in all_positions
            ]
            print(f"  No LR filter. Total patches: {len(self.index):,}")

        if 0 < self._subsample < 1.0:
            import random
            n = max(1, int(len(self.index) * self._subsample))
            self.index = random.sample(self.index, n)
            print(f"  Subsampled to: {len(self.index):,} patches "
                  f"({self._subsample:.0%} of filtered)")
        print(f"  Dataset ready: {len(self.index):,} patches\n")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t, top, left = self.index[idx]
        p = self.patch_size
        hr = self.hr_data[t, top:top+p, left:left+p].copy()
        lr = self.lr_data[t, top:top+p, left:left+p].copy()

        hr_norm = self.normalizer.normalize(hr)
        lr_norm = self.normalizer.normalize(lr)

        if self.augment:
            if np.random.rand() > 0.5:
                hr_norm = hr_norm[:, ::-1].copy()
                lr_norm = lr_norm[:, ::-1].copy()
            if np.random.rand() > 0.5:
                hr_norm = hr_norm[::-1, :].copy()
                lr_norm = lr_norm[::-1, :].copy()

        return (
            torch.from_numpy(lr_norm).unsqueeze(0),
            torch.from_numpy(hr_norm).unsqueeze(0),
        )


# ============================================================
# FLOW MATCHING CORE — REPLACES NoiseScheduler
# ============================================================
class FlowMatching:
    """
    Conditional Flow Matching with independent Gaussian coupling
    (straight-line interpolant / rectified flow).

    Convention:  t=0 → noise,  t=1 → data.
        x_t   = (1 - t) * z + t * x_1,    z ~ N(0, I)
        u_t   = x_1 - z                   (constant along path)

    The network is trained to regress u_t given (x_t, c, t).
    Sampling integrates dx/dt = v_theta(x, c, t) from t=0 to t=1.
    """
    def __init__(self, t_scale: float = 1000.0, device='cpu'):
        self.t_scale = t_scale
        self.device  = device

    def sample_interpolant(self, x1: torch.Tensor, noise: torch.Tensor = None
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draw t ~ U(0,1), z ~ N(0,I), build x_t and target velocity u_t.
        Returns (x_t, u_t, t_scaled).

        Args:
            x1:    [B, 1, H, W]  data sample (HR, normalized to [-1, 1])
            noise: optional pre-drawn noise of same shape as x1
        """
        B = x1.shape[0]
        if noise is None:
            noise = torch.randn_like(x1)
        # t ~ U(0,1), one scalar per sample; broadcast to spatial dims
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
        t_b = t.view(B, 1, 1, 1)
        x_t = (1.0 - t_b) * noise + t_b * x1
        u_t = x1 - noise
        t_scaled = t * self.t_scale
        return x_t, u_t, t_scaled

    # -------- Deterministic ODE samplers --------
    @torch.no_grad()
    def sample(self,
               model: nn.Module,
               cond:  torch.Tensor,
               shape: Tuple[int, int, int, int],
               num_steps: int = 50,
               solver: str = 'heun',
               device=None,
               noise: torch.Tensor = None) -> torch.Tensor:
        """
        Integrate dx/dt = v_theta(cat([x, cond], dim=1), t_scaled) from t=0 to t=1.

        Args:
            model:    velocity field v_theta with signature (x_2ch, t_scaled) -> v
            cond:     [B, 1, H, W] LR condition (log1p-normalized, same coords as x_t)
            shape:    target shape of x  (B, 1, H, W)
            num_steps: integer number of ODE steps
            solver:   'euler' | 'midpoint' | 'heun'
            noise:    optional starting noise (else randn)

        Returns:
            x1_pred: [B, 1, H, W] sample at t=1 (still in normalized space)
        """
        assert solver in ('euler', 'midpoint', 'heun'), f"Unknown solver: {solver}"
        device = device or cond.device
        if noise is None:
            x = torch.randn(shape, device=device, dtype=cond.dtype)
        else:
            x = noise.to(device=device, dtype=cond.dtype)

        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=cond.dtype)
        dt = ts[1] - ts[0]   # uniform step

        def v_at(x_cur: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
            t_batch = t_scalar.expand(x_cur.shape[0]) * self.t_scale
            return model(torch.cat([x_cur, cond], dim=1), t_batch)

        for i in range(num_steps):
            t_i = ts[i]
            if solver == 'euler':
                k1 = v_at(x, t_i)
                x  = x + dt * k1
            elif solver == 'midpoint':
                k1 = v_at(x, t_i)
                x_mid = x + 0.5 * dt * k1
                k2 = v_at(x_mid, t_i + 0.5 * dt)
                x  = x + dt * k2
            elif solver == 'heun':
                # 2nd-order trapezoidal predictor-corrector
                k1 = v_at(x, t_i)
                x_pred = x + dt * k1
                # Last step: clamp t to 1.0 to avoid querying slightly > 1
                t_next = ts[i + 1]
                k2 = v_at(x_pred, t_next)
                x  = x + 0.5 * dt * (k1 + k2)

        return x


# ============================================================
# EMA (unchanged from v5b)
# ============================================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters()
                       if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(
                    p.data, alpha=1 - self.decay
                )

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict(self):           return self.shadow.copy()
    def load_state_dict(self, sd):  self.shadow = sd


# ============================================================
# U-NET (unchanged from v5b — the network is interpretation-agnostic)
# ============================================================
class SinusoidalEmbeddings(nn.Module):
    """Sinusoidal embedding. Expects t with similar magnitude to DDPM (0..1000).
       In the FM model we pre-multiply continuous t∈[0,1] by Config.t_scale
       before passing it here, so this layer is *unchanged*."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        # Cast t to float to support both long (DDPM legacy) and float (FM)
        emb  = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv1    = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1    = nn.GroupNorm(8, out_ch)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.drop     = nn.Dropout(dropout)
        self.act      = nn.SiLU()
        self.skip     = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x, t):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.act(self.time_mlp(t))[..., None, None]
        h = self.act(self.norm2(self.conv2(self.drop(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(B, 3, C, H * W).unbind(1)
        attn = (torch.einsum('bci,bcj->bij', q, k) * C**-0.5).softmax(-1)
        h = torch.einsum('bij,bcj->bci', attn, v).reshape(B, C, H, W)
        return x + self.proj(h)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, attn=False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = Attention(out_ch) if attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.attn(self.res2(self.res1(x, t), t))
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, attn=False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = Attention(out_ch) if attn else nn.Identity()

    def forward(self, x, skip, t):
        x = torch.cat([self.up(x), skip], dim=1)
        return self.attn(self.res2(self.res1(x, t), t))


class PrecipUNet(nn.Module):
    """
    4-level U-Net, channel-capped (v5b architecture, byte-identical).

    For Flow Matching, the *output* is now interpreted as the predicted
    velocity field v_theta(x_t, c, t) instead of the noise ε_theta. The
    layer shapes, channel counts, attention placement, and parameter
    count are unchanged — only the training target and loss change.
    """
    def __init__(self, in_channels=2, out_channels=1,
                 base_dim=64, ch_mult_cap=8, time_dim=256):
        super().__init__()
        d   = base_dim
        cap = ch_mult_cap

        c1 = d
        c2 = d * 2
        c3 = d * 4
        c4 = min(d * 8,  d * cap)
        c5 = min(d * 16, d * cap)

        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, c1, 3, padding=1)

        self.down1 = DownBlock(c1, c2, time_dim, attn=False)
        self.down2 = DownBlock(c2, c3, time_dim, attn=False)
        self.down3 = DownBlock(c3, c4, time_dim, attn=True)
        self.down4 = DownBlock(c4, c5, time_dim, attn=True)

        self.mid1     = ResBlock(c5, c5, time_dim)
        self.mid_attn = Attention(c5)
        self.mid2     = ResBlock(c5, c5, time_dim)

        self.up4 = UpBlock(c5, c5, c4, time_dim, attn=True)
        self.up3 = UpBlock(c4, c4, c3, time_dim, attn=True)
        self.up2 = UpBlock(c3, c3, c2, time_dim, attn=False)
        self.up1 = UpBlock(c2, c2, c1, time_dim, attn=False)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb  = self.time_mlp(t)
        x      = self.conv_in(x)
        x, s1  = self.down1(x, t_emb)
        x, s2  = self.down2(x, t_emb)
        x, s3  = self.down3(x, t_emb)
        x, s4  = self.down4(x, t_emb)
        x      = self.mid2(self.mid_attn(self.mid1(x, t_emb)), t_emb)
        x      = self.up4(x, s4, t_emb)
        x      = self.up3(x, s3, t_emb)
        x      = self.up2(x, s2, t_emb)
        x      = self.up1(x, s1, t_emb)
        return self.conv_out(x)


# ============================================================
# TRAINER — only the loss & validation are FM-specific
# ============================================================
class Trainer:
    def __init__(self, model, config, device, logger):
        self.model  = model
        self.config = config
        self.device = device
        self.logger = logger

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.scaler = GradScaler('cuda') if config.mixed_precision else None
        self.ema    = EMA(model, config.ema_decay) if config.use_ema else None

        self.train_losses = []
        self.val_losses   = []
        self.best_val     = float('inf')
        self.patience_ctr = 0

    # ---- Checkpoint resume ----
    def load_checkpoint(self, ckpt_path: str) -> int:
        """Load a saved checkpoint and return the next epoch to train from."""
        self.logger.info(f'Resuming from checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)

        m = getattr(self.model, '_orig_mod', self.model)
        m.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        if self.ema and 'ema_state_dict' in ckpt:
            self.ema.load_state_dict(ckpt['ema_state_dict'])

        self.best_val     = ckpt.get('best_val',     float('inf'))
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses   = ckpt.get('val_losses',   [])
        # Recompute patience from loss history so it is consistent
        self.patience_ctr = 0
        for v in self.val_losses:
            if v > self.best_val:
                self.patience_ctr += 1
            else:
                self.patience_ctr = 0

        resumed_epoch = ckpt['epoch']
        self.logger.info(
            f'  Loaded epoch {resumed_epoch} | best_val={self.best_val:.5f} | '
            f'patience_ctr={self.patience_ctr}'
        )
        return resumed_epoch + 1  # next epoch to run

    # ---- Core FM loss (replaces DDPM ε-MSE) ----
    def _fm_loss(self, lr_b: torch.Tensor, hr_b: torch.Tensor,
                 fm: FlowMatching) -> torch.Tensor:
        """
        L = E[ w(x_1) * || v_theta(x_t, c, t) - (x_1 - z) ||^2 ]
        with x_1 = hr_b (data), z ~ N(0,I), c = lr_b (LR condition),
              x_t = (1-t) z + t x_1.
        """
        x_t, u_t, t_scaled = fm.sample_interpolant(hr_b)
        v_pred = self.model(torch.cat([x_t, lr_b], dim=1), t_scaled)
        alpha  = self.config.extreme_weight_alpha
        # Extreme weight on the *data sample* x_1 (= hr_b). hr_b ∈ [-1,1],
        # so (hr_b + 1)/2 ∈ [0, 1] is the "extremity" measure used in v5b.
        weight = 1.0 + alpha * (hr_b + 1.0) / 2.0
        return (weight * (v_pred - u_t) ** 2).mean()

    def train_epoch(self, loader, fm: 'FlowMatching', epoch: int):
        self.model.train()
        total = 0.0
        pbar  = tqdm(loader, desc=f'Epoch {epoch:03d} [train]')

        for lr_b, hr_b in pbar:
            lr_b = lr_b.to(self.device, non_blocking=True)
            hr_b = hr_b.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler:
                with autocast('cuda'):
                    loss = self._fm_loss(lr_b, hr_b, fm)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._fm_loss(lr_b, hr_b, fm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()

            if self.ema: self.ema.update(self.model)
            total += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.5f}'})

        return total / len(loader)

    @torch.no_grad()
    def val_epoch(self, loader, fm: 'FlowMatching'):
        """
        Validation reports the same simulation-free FM loss used in
        training (no ODE integration), giving a directly comparable
        scalar to the training loss. Periodic full-sample validation
        (RMSE on denormalized output) should be run separately offline,
        as it is much more expensive.
        """
        if self.ema: self.ema.apply_shadow(self.model)
        self.model.eval()
        total = 0.0

        for lr_b, hr_b in tqdm(loader, desc='Validation     '):
            lr_b = lr_b.to(self.device, non_blocking=True)
            hr_b = hr_b.to(self.device, non_blocking=True)
            with autocast('cuda'):
                # Unweighted FM MSE for monitoring (consistent across epochs)
                x_t, u_t, t_scaled = fm.sample_interpolant(hr_b)
                v_pred = self.model(torch.cat([x_t, lr_b], dim=1), t_scaled)
                loss = F.mse_loss(v_pred, u_t)
            total += loss.item()

        if self.ema: self.ema.restore(self.model)
        return total / len(loader)

    def save(self, epoch, is_best=False):
        m    = getattr(self.model, '_orig_mod', self.model)
        ckpt = {
            'epoch':                epoch,
            'model_state_dict':     m.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val':             self.best_val,
            'train_losses':         self.train_losses,
            'val_losses':           self.val_losses,
            'scale_factor':         self.config.scale_factor,
            # FM-specific config saved for inference reproducibility
            'framework':            'flow_matching_v1',
            't_scale':              self.config.t_scale,
            'fm_steps':             self.config.fm_steps,
            'fm_solver':            self.config.fm_solver,
            'log1p_max':            self.config.log1p_max,
            'patch_size':           self.config.patch_size,
            'base_dim':             self.config.base_dim,
            'ch_mult_cap':          self.config.ch_mult_cap,
            'time_dim':             self.config.time_dim,
        }
        if self.ema:    ckpt['ema_state_dict']    = self.ema.state_dict()
        if self.scaler: ckpt['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(ckpt, os.path.join(self.config.results_dir, 'latest.pt'))
        if is_best:
            torch.save(ckpt, os.path.join(self.config.results_dir, 'best.pt'))
            self.logger.info(
                f'  → Saved best (val={self.best_val:.5f}, '
                f'scale={self.config.scale_factor}×)'
            )

    def train(self, train_loader, val_loader, fm: 'FlowMatching',
              start_epoch: int = 1):
        self.logger.info(
            f'{"Resuming" if start_epoch > 1 else "Starting"} '
            f'Flow Matching training from epoch {start_epoch}...'
        )
        for epoch in range(start_epoch, self.config.epochs + 1):
            t0         = time.time()
            train_loss = self.train_epoch(train_loader, fm, epoch)
            val_loss   = self.val_epoch(val_loader, fm)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler_lr.step(val_loss)
            elapsed = time.time() - t0

            self.logger.info(
                f'Epoch {epoch:03d}/{self.config.epochs} | '
                f'train={train_loss:.5f} | val={val_loss:.5f} | '
                f'lr={self.optimizer.param_groups[0]["lr"]:.2e} | '
                f'time={elapsed/60:.1f}min'
            )

            is_best = val_loss < self.best_val
            if is_best:
                self.best_val     = val_loss
                self.patience_ctr = 0
            else:
                self.patience_ctr += 1

            self.save(epoch, is_best)

            if self.patience_ctr >= self.config.patience:
                self.logger.info(f'Early stopping at epoch {epoch}')
                break

        self.logger.info(f'Done. Best val loss: {self.best_val:.5f}')


# ============================================================
# MAIN
# ============================================================
def main():
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device     = torch.device(f'cuda:{config.gpu}')
    logger     = setup_logging(config.results_dir)
    normalizer = PrecipNormalizer(config.log1p_max)

    logger.info('=== Precipitation Flow Matching v1 (10× downscaling) ===')
    logger.info(f'Device:               {device}')
    logger.info(f'Scale factor:         {config.scale_factor}× '
                f'(LR = {0.1 * config.scale_factor:.1f}°)')
    logger.info(f'Interpolant:          x_t = (1-t)*z + t*x_1, t∈[0,1]')
    logger.info(f'Target velocity:      u_t = x_1 - z (rectified flow)')
    logger.info(f't_scale (embed):      {config.t_scale}')
    logger.info(f'Default sampler:      {config.fm_solver} with {config.fm_steps} steps')
    logger.info(f'log1p_max:            {config.log1p_max}')
    logger.info(f'lr_patch_threshold:   {config.lr_patch_threshold} mm/day')
    logger.info(f'extreme_weight_alpha: {config.extreme_weight_alpha}')
    logger.info(f'patch_size:           {config.patch_size}')
    logger.info(f'UNet:                 4-level, base_dim={config.base_dim}, '
                f'ch_cap=d*{config.ch_mult_cap}={config.base_dim*config.ch_mult_cap}')

    train_ds = PrecipDataset(
        config.train_h5, normalizer,
        config.patch_size, config.stride, config.scale_factor,
        augment=True, lr_threshold=config.lr_patch_threshold,
        subsample=config.train_subsample,
    )
    val_ds = PrecipDataset(
        config.val_h5, normalizer,
        config.patch_size, config.stride, config.scale_factor,
        augment=False, lr_threshold=config.lr_patch_threshold,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True,
    )

    logger.info(f'Train batches/epoch: {len(train_loader):,}')
    logger.info(f'Val   batches/epoch: {len(val_loader):,}')

    fm = FlowMatching(t_scale=config.t_scale, device=device)

    model = PrecipUNet(
        in_channels=2, out_channels=1,
        base_dim=config.base_dim,
        ch_mult_cap=config.ch_mult_cap,
        time_dim=config.time_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {n_params:,}  (v5b architecture preserved)')

    trainer = Trainer(model, config, device, logger)

    # ── Auto-resume from latest checkpoint if it exists ──────────────────────
    start_epoch = 1
    latest_ckpt = os.path.join(config.results_dir, 'latest.pt')
    if os.path.isfile(latest_ckpt):
        start_epoch = trainer.load_checkpoint(latest_ckpt)
        logger.info(f'Will continue from epoch {start_epoch} / {config.epochs}')
    else:
        logger.info('No checkpoint found — training from scratch.')

    trainer.train(train_loader, val_loader, fm, start_epoch=start_epoch)


if __name__ == '__main__':
    main()