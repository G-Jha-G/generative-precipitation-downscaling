import os, math, time, h5py, logging, json
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
    # ── Data paths ──────────────────────────────────────────
    train_h5: str = '/path/to/project/PhD_Precipitation/02_Data/processed/hdf5/mswx_train_1979-2005.h5'
    val_h5:   str = '/path/to/project/PhD_Precipitation/02_Data/processed/hdf5/mswx_val_2006-2010.h5'
    results_dir: str = './results/precip_ddpm_v5b'

    # ── Precipitation normalization ──────────────────────────
    log1p_max: float = 6.7859   # unchanged from v4

    # ── Scale factor ─────────────────────────────────────────
    # v5 key change: 4× → 10×
    # 0.1° HR → avg_pool(10) → 1.0° LR → bicubic upsample → 0.1° LR input
    # This matches the CMIP6 pipeline: GCMs regridded to 1.0°, QDM-corrected,
    # then bilinear-interpolated to 0.1° as the DDPM LR channel.
    scale_factor: int = 10

    # ── Patch settings ───────────────────────────────────────
    # CRITICAL DESIGN CONSTRAINT:
    #   patch_size must be divisible by scale_factor for clean avg_pool.
    #   avg_pool2d(kernel=k, stride=k) on size N gives floor(N/k) output.
    #   If N % k != 0, the last (N % k) rows/cols are silently DROPPED,
    #   causing spatial misalignment between HR and LR at every patch edge.
    #
    #   patch_size must ALSO be divisible by 8 (U-Net has 3 stride-2
    #   downsamples → requires spatial dims divisible by 2^3 = 8).
    #
    #   Required: patch_size % LCM(scale_factor, 8) = 0
    #   LCM(10, 8) = 40 → valid sizes: 40, 80, 120, 160, 200, ...
    #
    #   HOWEVER: v5 uses full-field LR generation (see PrecipDataset),
    #   which eliminates this constraint entirely. The full 350×350 field
    #   is cleanly divisible by 10 (350/10=35), so avg_pool on the full
    #   field is exact. Patches are then extracted from the already-computed
    #   LR field — no per-patch avg_pool, no dropped pixels, no misalignment.
    #
    #   patch_size=128 is therefore valid again — but must still be
    #   divisible by 8 for the U-Net: 128 % 8 = 0 ✓
    patch_size:   int   = 128
    stride:       int   = 64

    # ── LR threshold filter ───────────────────────────────────
    lr_patch_threshold: float = 1.0   # mm/day

    # ── Training ─────────────────────────────────────────────
    batch_size:      int   = 8
    epochs:          int   = 100
    patience:        int   = 15
    learning_rate:   float = 1e-4
    weight_decay:    float = 1e-5
    gradient_clip:   float = 1.0
    mixed_precision: bool  = True

    # ── Diffusion ────────────────────────────────────────────
    t_steps:    int   = 1000
    beta_start: float = 0.0001
    beta_end:   float = 0.02
    ddim_steps: int   = 50

    # ── Model ────────────────────────────────────────────────
    # v5b changes vs v5:
    #   1. 4-level U-Net (was 3-level): adds DownBlock4 d*8→d*8 (capped)
    #      Channel progression: 64→128→256→512→512
    #      Bottleneck: 8×8 spatial (was 16×16) — larger receptive field
    #   2. ch_mult_cap=8: prevents channel explosion at level 4
    #      Without cap: level 4 would be d*16=1024 → 240M params
    #      With cap at d*8=512: ~60M params — tractable on A5000
    #   3. patch_size 128 remains valid: 128 % 16 = 0 ✓ (2^4 levels)
    #   Rationale: 10× SR must hallucinate structure from 35×35 LR blob.
    #   Deeper bottleneck (8×8) provides larger effective receptive field
    #   and more capacity to learn the SR prior from MSWX statistics.
    base_dim:    int   = 64
    ch_mult_cap: int   = 8     # max multiplier — channels cap at d*8 = 512
    time_dim:    int   = 256
    use_ema:     bool  = True
    ema_decay:   float = 0.9999

    # ── System ───────────────────────────────────────────────
    num_workers:     int   = 4
    gpu:             int   = 0
    seed:            int   = 42
    train_subsample: float = 0.30

    # ── Extreme value weighting (unchanged from v4) ───────────
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
    logger = logging.getLogger('precip_ddpm_v5')
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
# NORMALIZATION  (identical to v4)
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
# FULL-FIELD LR GENERATION UTILITY
# ============================================================
def compute_lr_fullfield(hr_batch: np.ndarray,
                         scale_factor: int,
                         H: int, W: int) -> np.ndarray:
    """
    Generate LR field by operating avg_pool on the FULL FIELD,
    then bicubic upsample back to original resolution.

    WHY FULL-FIELD (not per-patch):
      avg_pool2d(kernel=k, stride=k) on size N gives floor(N/k) output.
      If N % k != 0, the last (N % k) pixels are silently DROPPED.
        - 128 % 10 = 8 → per-patch avg_pool drops 8px per edge → misalignment
        - 350 % 10 = 0 → full-field avg_pool is exact, no dropped pixels
      Patches extracted from the pre-computed full-field LR are therefore
      perfectly spatially aligned with their HR counterparts.

    Args:
        hr_batch:     [B, H, W] float32 numpy, mm/day
        scale_factor: int — degradation factor (10 for v5)
        H, W:         full field spatial dims (must be divisible by scale_factor)

    Returns:
        lr_up: [B, H, W] float32 numpy, mm/day
    """
    t     = torch.from_numpy(hr_batch).unsqueeze(1).float()    # [B,1,H,W]
    lr    = F.avg_pool2d(t, scale_factor, scale_factor)         # [B,1,H/k,W/k]
    lr_up = F.interpolate(
        lr, size=(H, W), mode='bicubic', align_corners=False
    ).clamp(min=0)                                               # [B,1,H,W]
    return lr_up.squeeze(1).numpy()                              # [B,H,W]


# ============================================================
# DATASET
# ============================================================
class PrecipDataset(Dataset):
    """
    v5 vs v4 — two changes only:

    1. scale_factor: 10 (was 4).
       LR = avg_pool(10) on full 350×350 field → 35×35 → bicubic to 350×350.
       Represents 1.0° spatial resolution, matching GCM native resolution.

    2. Full-field LR generation (was per-patch in v4).
       v4: avg_pool(4) per 128-patch → 128%4=0 → clean, no issue.
       v5: avg_pool(10) per 128-patch → 128%10=8 → DROPS 8px edge → misalignment.
       Fix: pre-compute LR on the full 350×350 field (350%10=0 → exact),
            then extract patches from both HR and LR simultaneously.
            This guarantees pixel-perfect alignment at every patch position.

    Everything else identical to v4:
      LR threshold filtering, subsampling, augmentation, normalization,
      extreme-weighted loss (in Trainer), EMA, DDIM sampling.
    """
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

        # ── Load HR into RAM ─────────────────────────────────
        print(f"Loading {os.path.basename(h5_path)} into RAM...")
        t0 = time.time()
        with h5py.File(h5_path, 'r') as f:
            hr_data = f['precipitation'][:]   # [T, H, W] mm/day

        T, H, W = hr_data.shape
        print(f"  Loaded:       {T} × {H} × {W}  ({hr_data.nbytes/1e9:.2f} GB) "
              f"in {time.time()-t0:.1f}s")
        print(f"  Range:        [{hr_data.min():.2f}, {hr_data.max():.2f}] mm/day")
        print(f"  Zero frac:    {(hr_data==0).mean():.1%}")
        print(f"  Scale factor: {scale_factor}× → LR ≈ {0.1*scale_factor:.1f}°")

        # Verify full-field divisibility — fail fast with clear message
        assert H % scale_factor == 0, (
            f"H={H} not divisible by scale_factor={scale_factor} "
            f"(remainder {H%scale_factor}). Full-field avg_pool cannot be exact."
        )
        assert W % scale_factor == 0, (
            f"W={W} not divisible by scale_factor={scale_factor} "
            f"(remainder {W%scale_factor}). Full-field avg_pool cannot be exact."
        )
        print(f"  Divisibility: {H}%{scale_factor}={H%scale_factor} ✓  "
              f"{W}%{scale_factor}={W%scale_factor} ✓  "
              f"(full-field avg_pool is exact)")

        self.hr_data = hr_data
        self.H, self.W = H, W

        # ── Pre-compute full-field LR for ALL timesteps ──────
        # Key v5 change: LR computed on full field, not per-patch.
        # Memory: same shape as HR (upsampled back to H×W) → same RAM footprint.
        # Time: batch-wise, similar to v4's LR mean pre-computation.
        print(f"\n  Pre-computing full-field LR (scale={scale_factor}×)...")
        print(f"  avg_pool({scale_factor}) on {H}×{W} → "
              f"{H//scale_factor}×{W//scale_factor} → "
              f"bicubic back to {H}×{W}")
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

        self.lr_data = lr_data   # [T, H, W] — pixel-perfectly aligned with hr_data

        # ── Build patch position grid ────────────────────────
        tops  = list(range(0, H - patch_size + 1, stride))
        lefts = list(range(0, W - patch_size + 1, stride))
        if not tops  or tops[-1]  != H - patch_size:
            tops.append(H - patch_size)
        if not lefts or lefts[-1] != W - patch_size:
            lefts.append(W - patch_size)

        all_positions  = [(top, left) for top in tops for left in lefts]
        n_pos          = len(all_positions)
        total_possible = T * n_pos
        print(f"\n  Patch grid: {len(tops)}×{len(lefts)} = "
              f"{n_pos} positions/timestep")
        print(f"  Total patches: {total_possible:,}")

        # ── LR threshold filtering ───────────────────────────
        # Uses pre-computed full-field LR — consistent and fast.
        if lr_threshold > 0:
            print(f"\n  Filtering: mean LR ≥ {lr_threshold} mm/day...")
            t0 = time.time()

            batch_sz = 50
            lr_means = np.zeros((T, n_pos), dtype=np.float32)

            for t_start in tqdm(range(0, T, batch_sz),
                                desc="  Computing LR patch means"):
                t_end    = min(t_start + batch_sz, T)
                lr_chunk = lr_data[t_start:t_end]   # [B, H, W]
                for p_idx, (top, left) in enumerate(all_positions):
                    patch_lr = lr_chunk[
                        :, top:top+patch_size, left:left+patch_size
                    ]   # [B, P, P]
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

            kept_means = lr_means[valid_t, valid_p]
            print(f"  Kept LR stats:  mean={kept_means.mean():.3f}  "
                  f"p50={np.percentile(kept_means,50):.3f}  "
                  f"p95={np.percentile(kept_means,95):.3f} mm/day")
        else:
            self.index = [
                (t, top, left)
                for t in range(T)
                for top, left in all_positions
            ]
            print(f"  No LR filter. Total patches: {len(self.index):,}")

        # ── Random subsample ─────────────────────────────────
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

        # Extract aligned HR and LR patches from pre-computed full fields.
        # Both cover exactly the same [top:top+p, left:left+p] spatial region.
        # No dropped pixels, no extrapolation, no misalignment.
        hr = self.hr_data[t, top:top+p, left:left+p].copy()   # [P, P] mm/day
        lr = self.lr_data[t, top:top+p, left:left+p].copy()   # [P, P] mm/day

        hr_norm = self.normalizer.normalize(hr)
        lr_norm = self.normalizer.normalize(lr)

        # Augmentation: identical flips applied to both HR and LR
        if self.augment:
            if np.random.rand() > 0.5:
                hr_norm = hr_norm[:, ::-1].copy()
                lr_norm = lr_norm[:, ::-1].copy()
            if np.random.rand() > 0.5:
                hr_norm = hr_norm[::-1, :].copy()
                lr_norm = lr_norm[::-1, :].copy()

        return (
            torch.from_numpy(lr_norm).unsqueeze(0),   # [1, P, P]
            torch.from_numpy(hr_norm).unsqueeze(0),   # [1, P, P]
        )


# ============================================================
# NOISE SCHEDULER  (identical to v4)
# ============================================================
class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001,
                 beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device        = device

        b   = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        a   = 1.0 - b
        ac  = torch.cumprod(a, dim=0)
        acp = F.pad(ac[:-1], (1, 0), value=1.0)

        self.betas                         = b
        self.alphas_cumprod                = ac
        self.sqrt_alphas_cumprod           = torch.sqrt(ac)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - ac)
        self.sqrt_recip_alphas             = torch.sqrt(1.0 / a)
        self.sqrt_recipm1_alphas_cumprod   = torch.sqrt(1.0 / ac - 1)
        self.posterior_variance            = b * (1.0 - acp) / (1.0 - ac)

    def add_noise(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        return a * x0 + b * noise, noise

    def ddim_step(self, x_t, noise_pred, t, t_prev):
        a_t    = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 \
                 else torch.tensor(1.0)
        x0 = ((x_t - torch.sqrt(1 - a_t) * noise_pred)
              / torch.sqrt(a_t)).clamp(-1, 1)
        return torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev) * noise_pred


# ============================================================
# EMA  (identical to v4)
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

    def state_dict(self):  return self.shadow.copy()
    def load_state_dict(self, sd): self.shadow = sd


# ============================================================
# UNET  (identical to v4)
# ============================================================
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb  = t[:, None] * emb[None, :]
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
    4-level U-Net for 10× super-resolution (v5b).

    Input:  [noisy_hr, lr_bicubic] — 2 channels, normalized to [-1,1]
    Output: predicted noise — 1 channel

    Architecture vs v5 (3-level):
      Level   Spatial (128-patch)   Channels (d=64, cap=d*8=512)
      ──────  ───────────────────   ────────────────────────────
      Input   128×128               2
      Down1   64×64                 128      (d   → d*2)
      Down2   32×32                 256      (d*2 → d*4)
      Down3   16×16                 512      (d*4 → d*8)
      Down4    8×8                  512      (d*8 → d*8, capped)  ← NEW
      Bottleneck 8×8                512
      Up4     16×16                 512      ← NEW
      Up3     32×32                 256
      Up2     64×64                 128
      Up1    128×128                64

    Key improvement:
      Bottleneck shrinks from 16×16 (v5) to 8×8 (v5b).
      Larger effective receptive field → better global structure.
      Channel cap at d*8=512 prevents parameter explosion:
        Uncapped 4-level: ~150M params
        Capped  4-level:  ~60M params  ← this model

    Attention placement:
      down3 (16×16, 512ch): spatial attention — captures mesoscale patterns
      down4 (8×8,  512ch):  spatial attention — captures synoptic structure
      up4   (16×16, 512ch): spatial attention — symmetric with down4/down3
      up3   (32×32, 256ch): spatial attention — fine-scale reconstruction
    """
    def __init__(self, in_channels=2, out_channels=1,
                 base_dim=64, ch_mult_cap=8, time_dim=256):
        super().__init__()
        d   = base_dim
        cap = ch_mult_cap

        # Channel dims at each level — capped at d*cap
        c1 = d          #  64
        c2 = d * 2      # 128
        c3 = d * 4      # 256
        c4 = min(d * 8,  d * cap)   # 512  (would be d*8 anyway, but explicit)
        c5 = min(d * 16, d * cap)   # 512  (capped — prevents 1024)

        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, c1, 3, padding=1)

        # Encoder
        self.down1 = DownBlock(c1, c2, time_dim, attn=False)   # 128→64
        self.down2 = DownBlock(c2, c3, time_dim, attn=False)   # 64→32
        self.down3 = DownBlock(c3, c4, time_dim, attn=True)    # 32→16
        self.down4 = DownBlock(c4, c5, time_dim, attn=True)    # 16→8  ← NEW

        # Bottleneck
        self.mid1     = ResBlock(c5, c5, time_dim)
        self.mid_attn = Attention(c5)
        self.mid2     = ResBlock(c5, c5, time_dim)

        # Decoder — skip connections from matching encoder level
        self.up4 = UpBlock(c5, c5, c4, time_dim, attn=True)    # 8→16   ← NEW
        self.up3 = UpBlock(c4, c4, c3, time_dim, attn=True)    # 16→32
        self.up2 = UpBlock(c3, c3, c2, time_dim, attn=False)   # 32→64
        self.up1 = UpBlock(c2, c2, c1, time_dim, attn=False)   # 64→128

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
# TRAINER  (identical to v4)
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
            self.optimizer, patience=5, factor=0.5, verbose=False
        )
        self.scaler = GradScaler('cuda') if config.mixed_precision else None
        self.ema    = EMA(model, config.ema_decay) if config.use_ema else None

        self.train_losses = []
        self.val_losses   = []
        self.best_val     = float('inf')
        self.patience_ctr = 0

    def train_epoch(self, loader, sched, epoch):
        self.model.train()
        total = 0.0
        pbar  = tqdm(loader, desc=f'Epoch {epoch:03d} [train]')

        for lr_b, hr_b in pbar:
            lr_b = lr_b.to(self.device, non_blocking=True)
            hr_b = hr_b.to(self.device, non_blocking=True)
            t    = torch.randint(
                0, sched.num_timesteps,
                (hr_b.shape[0],), device=self.device, dtype=torch.long
            )
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler:
                with autocast('cuda'):
                    noisy, noise = sched.add_noise(hr_b, t)
                    pred   = self.model(torch.cat([noisy, lr_b], dim=1), t)
                    alpha  = self.config.extreme_weight_alpha
                    weight = 1.0 + alpha * (hr_b + 1.0) / 2.0
                    loss   = (weight * (pred - noise) ** 2).mean()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                noisy, noise = sched.add_noise(hr_b, t)
                pred   = self.model(torch.cat([noisy, lr_b], dim=1), t)
                alpha  = self.config.extreme_weight_alpha
                weight = 1.0 + alpha * (hr_b + 1.0) / 2.0
                loss   = (weight * (pred - noise) ** 2).mean()
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
    def val_epoch(self, loader, sched):
        if self.ema: self.ema.apply_shadow(self.model)
        self.model.eval()
        total = 0.0

        for lr_b, hr_b in tqdm(loader, desc='Validation     '):
            lr_b = lr_b.to(self.device, non_blocking=True)
            hr_b = hr_b.to(self.device, non_blocking=True)
            t    = torch.randint(
                0, sched.num_timesteps,
                (hr_b.shape[0],), device=self.device, dtype=torch.long
            )
            with autocast('cuda'):
                noisy, noise = sched.add_noise(hr_b, t)
                pred = self.model(torch.cat([noisy, lr_b], dim=1), t)
                loss = F.mse_loss(pred, noise)
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

    def train(self, train_loader, val_loader, noise_sched):
        self.logger.info('Starting training...')
        for epoch in range(1, self.config.epochs + 1):
            t0         = time.time()
            train_loss = self.train_epoch(train_loader, noise_sched, epoch)
            val_loss   = self.val_epoch(val_loader, noise_sched)
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

    logger.info('=== Precipitation DDPM v5b — 10x, 4-level UNet (1.0 deg -> 0.1 deg) ===')
    logger.info(f'Device:               {device}')
    logger.info(f'Scale factor:         {config.scale_factor}x '
                f'(LR = {0.1 * config.scale_factor:.1f} deg)')
    logger.info(f'log1p_max:            {config.log1p_max}')
    logger.info(f'lr_patch_threshold:   {config.lr_patch_threshold} mm/day')
    logger.info(f'extreme_weight_alpha: {config.extreme_weight_alpha}')
    logger.info(f'patch_size:           {config.patch_size}')
    logger.info(f'UNet:                 4-level, base_dim={config.base_dim}, '
                f'ch_cap=d*{config.ch_mult_cap}={config.base_dim*config.ch_mult_cap}')
    logger.info(f'LR generation:        full-field avg_pool({config.scale_factor}) '
                f'-> bicubic upsample (NOT per-patch)')

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

    noise_sched = NoiseScheduler(
        config.t_steps, config.beta_start, config.beta_end, device
    )

    model = PrecipUNet(
        in_channels=2, out_channels=1,
        base_dim=config.base_dim,
        ch_mult_cap=config.ch_mult_cap,
        time_dim=config.time_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {n_params:,}  '
                f'(v5: ~38.7M, v5b target: ~60M)')

    trainer = Trainer(model, config, device, logger)
    trainer.train(train_loader, val_loader, noise_sched)


if __name__ == '__main__':
    main()