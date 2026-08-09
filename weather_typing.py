"""
eval_weather_typing.py
======================
Weather Typing (Circulation Pattern Classification) baseline for 10×
precipitation downscaling. CPU-only.

ALGORITHM:
  Calibration (train 1979-2005):
    1. Compute LR coarse [T, Hc, Wc] for all training days.
    2. Spatially normalise each day → zero-mean, unit-variance [T, Hc*Wc].
       (Same normalisation as CA: focuses on PATTERN, not magnitude.)
    3. Run K-Means on normalised LR fields → N cluster centroids.
       Each centroid is a characteristic large-scale circulation pattern.
    4. For each cluster k: collect all training HR days assigned to k
       → compute mean HR field → cluster_hr_mean[k, H, W].
    5. Optional: compute per-cluster wet-day frequency and HR std maps
       for uncertainty estimation and diagnostics.

  Application (per validation/test day d):
    1. Compute LR coarse for day d → normalise.
    2. Find nearest cluster centroid (by Euclidean distance in
       normalised LR space — equivalent to correlation ranking).
    3. Predict: pred[d] = cluster_hr_mean[nearest_k].
    4. Optional magnitude scaling: multiply cluster mean by
       (LR_domain_mean_d / cluster_lr_domain_mean_k) to preserve
       the day's actual total precipitation magnitude while using
       the cluster's spatial pattern.

MAGNITUDE SCALING (--use_scaling, default True):
  The cluster mean is the average of many days, so it underestimates
  extremes. Multiplicative scaling:
    scale = domain_mean(LR_day) / domain_mean(LR_cluster_mean)
    pred = cluster_hr_mean * scale
  This preserves spatial pattern (from clustering) while correcting
  the overall magnitude (from the day's actual LR field).
  clip_scale_max caps the scale factor to prevent explosion on
  anomalously wet days outside the calibration range.

CLUSTER DIAGNOSTICS (WCSS elbow + silhouette):
  The elbow plot and silhouette score help choose N.
  Typical optimal N for India: 10-30 (monsoon onset, break,
  active phases, western disturbances, post-monsoon, winter).

MEMORY:
  LR normalised library [9862, 1225] float32 ~ 48 MB
  Cluster HR means      [N, 122500]  float32 ~ N × 0.5 MB  (tiny)
  No GPU needed anywhere.

INTERPRETABILITY:
  Each cluster has a physical story. The script generates a
  cluster atlas showing the LR circulation pattern and mean HR
  field for each type — scientifically the most interpretable
  figure in your comparison suite.

Usage:
  python eval_weather_typing.py
  python eval_weather_typing.py --N 20
  python eval_weather_typing.py --skip_calib
  python eval_weather_typing.py --no_scaling
  python eval_weather_typing.py --elbow_only   (cluster diagnostics only)
"""

import os, json, logging, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter
from pathlib import Path
from datetime import datetime, date, timedelta
from tqdm import tqdm

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    raise ImportError(
        'scikit-learn not installed. Run: pip install scikit-learn'
    )


# ============================================================
# CONFIG
# ============================================================
class Config:
    # ── Data ────────────────────────────────────────────────
    train_h5: str = (
        '/path/to/project/'
        'PhD_Precipitation/02_Data/processed/hdf5/mswx_train_1979-2005.h5'
    )
    val_h5: str = (
        '/path/to/project/'
        'PhD_Precipitation/02_Data/processed/hdf5/mswx_val_2006-2010.h5'
    )
    test_h5: str = (
        '/path/to/project/'
        'PhD_Precipitation/02_Data/processed/hdf5/mswx_test_2011-2014.h5'
    )
    results_dir: str = (
        '/path/to/project/'
        'PhD_Precipitation/03_Code/results/weather_typing_baseline'
    )

    # ── Domain ──────────────────────────────────────────────
    scale_factor: int = 10

    # ── Clustering ───────────────────────────────────────────
    N:            int   = 20       # number of weather types (clusters)
    # Typical optimal range for India: 10-30
    # Use --elbow_only to run WCSS/silhouette diagnostics first
    random_state: int   = 42
    max_iter:     int   = 500      # K-Means max iterations
    n_init:       int   = 10       # K-Means number of random restarts
    # MiniBatchKMeans used for speed on 10K×1225 data

    # ── Magnitude scaling ─────────────────────────────────────
    use_scaling:    bool  = True   # apply multiplicative magnitude correction
    clip_scale_max: float = 5.0    # cap scale factor (prevents explosion)
    lr_threshold:   float = 0.1    # mm/day — domain-mean below this → predict 0

    # ── Elbow diagnostics ─────────────────────────────────────
    elbow_N_range: tuple = (5, 10, 15, 20, 25, 30, 40, 50)

    # ── Processing ───────────────────────────────────────────
    lr_batch_size: int = 50

    # ── Evaluation (identical to all other baselines) ─────────
    fss_thresholds:  tuple = (1.0, 5.0, 20.0)
    fss_scales:      tuple = (1, 3, 5, 10, 15, 20, 30, 50)
    percentiles:     tuple = (50, 75, 90, 95, 99, 99.5, 99.9)
    val_start_year:  int   = 2006
    test_start_year: int   = 2011

    # ── Atlas: how many clusters to show per figure row ───────
    atlas_cols: int = 5


# ============================================================
# LOGGING
# ============================================================
def setup_logging(results_dir: str) -> logging.Logger:
    log_file = os.path.join(
        results_dir, f'eval_wt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logger = logging.getLogger('wt_eval')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                            '%Y-%m-%d %H:%M:%S')
    for h in [logging.FileHandler(log_file), logging.StreamHandler()]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ============================================================
# DATE UTILITIES
# ============================================================
def build_date_index(T: int, start_year: int):
    """Return (years, months) for T days from start_year Jan 1."""
    years, months = [], []
    d = date(start_year, 1, 1)
    for _ in range(T):
        years.append(d.year); months.append(d.month)
        d += timedelta(days=1)
    return np.array(years, np.int32), np.array(months, np.int32)


# ============================================================
# LR GENERATION  (identical full-field pipeline to all baselines)
# ============================================================
def compute_lr_coarse(hr_batch: np.ndarray, scale_factor: int) -> np.ndarray:
    """HR [B,H,W] → avg_pool(sf) → coarse [B, H//sf, W//sf]. CPU-safe."""
    t  = torch.from_numpy(hr_batch.astype(np.float32)).unsqueeze(1)
    lc = F.avg_pool2d(t, scale_factor, scale_factor)
    return lc.squeeze(1).numpy()


# ============================================================
# SPATIAL NORMALISATION
# ============================================================
def spatial_norm(fields: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalise each row to zero spatial mean and unit spatial std.

    fields: [N, P] — N days, P spatial pixels.
    Returns: [N, P] normalised (same shape, no means/stds returned).

    Dry days (std ≈ 0) → zero vector after normalisation.
    Normalisation ensures K-Means clusters on spatial PATTERN not magnitude.
    Without this, all dry days form one giant cluster and all wet days
    cluster by intensity rather than circulation type.
    """
    mu  = fields.mean(axis=1, keepdims=True)   # [N, 1]
    sig = fields.std (axis=1, keepdims=True)   # [N, 1]
    return (fields - mu) / (sig + eps)


# ============================================================
# WEATHER TYPING MODEL
# ============================================================
class WeatherTypingModel:
    """
    K-Means weather typing downscaling model.

    Calibrated arrays:
      centroids         [N, Hc*Wc]  — K-Means centroids in normalised LR space
      cluster_hr_mean   [N, H*W]    — mean HR field per weather type
      cluster_lr_dmean  [N]         — mean LR domain-mean per weather type
                                      (used for magnitude scaling)
      cluster_counts    [N]         — training days per weather type
      cluster_months    [N, 12]     — monthly frequency per weather type
                                      (interpretability: which months dominate)
    """

    def __init__(self, config: Config):
        self.config          = config
        self.centroids       = None   # [N, P_lr]  float32
        self.cluster_hr_mean = None   # [N, P_hr]  float32
        self.cluster_lr_dmean = None  # [N]        float32
        self.cluster_counts  = None   # [N]        int
        self.cluster_months  = None   # [N, 12]    int
        self.H = self.W = self.Hc = self.Wc = None
        self.N = config.N

    # ── Calibration ─────────────────────────────────────────

    def fit(self, train_h5: str, logger: logging.Logger):
        sf = self.config.scale_factor
        logger.info(f'Loading training data: {train_h5}')
        t0 = time.time()
        with h5py.File(train_h5, 'r') as f:
            hr_data = f['precipitation'][:]
        T, H, W  = hr_data.shape
        Hc, Wc   = H // sf, W // sf
        P_lr, P_hr = Hc * Wc, H * W
        self.H, self.W, self.Hc, self.Wc = H, W, Hc, Wc
        logger.info(f'  {T}×{H}×{W}  ({hr_data.nbytes/1e9:.2f} GB) '
                    f'in {time.time()-t0:.1f}s')
        assert H % sf == 0 and W % sf == 0

        _, months = build_date_index(T, 1979)

        # ── Compute LR coarse + HR for all training days ─────
        logger.info('Computing LR coarse for all training days...')
        lc_raw  = np.zeros((T, P_lr), dtype=np.float32)
        hr_flat = np.zeros((T, P_hr), dtype=np.float32)
        batch   = self.config.lr_batch_size

        for t0b in tqdm(range(0, T, batch), desc='  LR/HR build'):
            t1b    = min(t0b + batch, T)
            hr_b   = hr_data[t0b:t1b].astype(np.float32)
            lc_b   = compute_lr_coarse(hr_b, sf)   # [B, Hc, Wc]
            lc_raw [t0b:t1b] = lc_b.reshape(t1b - t0b, P_lr)
            hr_flat[t0b:t1b] = hr_b.reshape(t1b - t0b, P_hr)

        lc_domain_means = lc_raw.mean(axis=1)   # [T] — domain-mean LR per day
        logger.info(f'  LR domain mean: [{lc_domain_means.min():.2f}, '
                    f'{lc_domain_means.max():.2f}] mm/day')

        # ── Normalise for clustering ──────────────────────────
        logger.info('Normalising LR fields for K-Means...')
        lc_norm = spatial_norm(lc_raw)   # [T, P_lr]

        # ── K-Means clustering ────────────────────────────────
        logger.info(f'Running MiniBatchKMeans: N={self.N}, '
                    f'n_init={self.config.n_init}, '
                    f'max_iter={self.config.max_iter}...')
        t0 = time.time()
        km = MiniBatchKMeans(
            n_clusters    = self.N,
            random_state  = self.config.random_state,
            max_iter      = self.config.max_iter,
            n_init        = self.config.n_init,
            batch_size    = min(1024, T),
            compute_labels= True,
            verbose       = 0,
        )
        km.fit(lc_norm)
        labels = km.labels_   # [T] cluster assignment for each training day
        logger.info(f'  K-Means done in {time.time()-t0:.1f}s  '
                    f'Inertia (WCSS): {km.inertia_:.2f}')

        # ── Store centroids ───────────────────────────────────
        self.centroids = km.cluster_centers_.astype(np.float32)  # [N, P_lr]

        # ── Per-cluster statistics ────────────────────────────
        logger.info('Computing per-cluster HR mean fields...')
        self.cluster_hr_mean    = np.zeros((self.N, P_hr),  dtype=np.float32)
        self.cluster_lr_dmean   = np.zeros(self.N,           dtype=np.float32)
        self.cluster_counts     = np.zeros(self.N,           dtype=np.int32)
        self.cluster_months     = np.zeros((self.N, 12),     dtype=np.int32)

        for k in range(self.N):
            mask_k = labels == k
            n_k    = mask_k.sum()
            self.cluster_counts[k] = n_k

            if n_k == 0:
                logger.warning(f'  Cluster {k}: EMPTY — increase training data or reduce N')
                continue

            self.cluster_hr_mean [k] = hr_flat   [mask_k].mean(axis=0)
            self.cluster_lr_dmean[k] = lc_domain_means[mask_k].mean()

            # Monthly frequency (0-indexed months)
            for m in range(12):
                self.cluster_months[k, m] = ((months[mask_k] - 1) == m).sum()

            # Log dominant season
            dom_month = self.cluster_months[k].argmax() + 1
            pct       = n_k / T * 100
            hr_mean   = self.cluster_hr_mean[k].mean()
            logger.info(f'  Cluster {k:2d}: {n_k:4d} days ({pct:4.1f}%)  '
                        f'LR_mean={self.cluster_lr_dmean[k]:.2f}  '
                        f'HR_mean={hr_mean:.2f}  '
                        f'dom_month={dom_month}')

        logger.info('Calibration complete.')

    # ── Application ─────────────────────────────────────────

    def predict_period(self, h5_path: str, start_year: int,
                        logger: logging.Logger) -> tuple:
        """
        Apply weather typing to a full HDF5 period.

        Per day:
          1. Compute LR coarse → normalise.
          2. Find nearest centroid (Euclidean in normalised space).
          3. pred = cluster_hr_mean[nearest_k]
          4. Optional magnitude scaling by LR domain mean ratio.

        Returns: (pred [T,H,W], obs [T,H,W], lat, lon, cluster_seq [T])
        """
        sf = self.config.scale_factor
        logger.info(f'Loading: {h5_path}')
        with h5py.File(h5_path, 'r') as f:
            hr_data = f['precipitation'][:]
            lat = f['lat'][:] if 'lat' in f else np.linspace(6.5, 41.0, self.H)
            lon = f['lon'][:] if 'lon' in f else np.linspace(66.5, 100.0, self.W)

        T, H, W = hr_data.shape
        P_lr    = self.Hc * self.Wc
        logger.info(f'  {T}×{H}×{W}')

        # Compute LR for all days
        lc_raw = np.zeros((T, P_lr), dtype=np.float32)
        for t0 in tqdm(range(0, T, self.config.lr_batch_size), desc='  LR coarse'):
            t1 = min(t0 + self.config.lr_batch_size, T)
            lc_raw[t0:t1] = compute_lr_coarse(
                hr_data[t0:t1].astype(np.float32), sf
            ).reshape(t1 - t0, P_lr)

        lc_domain_means = lc_raw.mean(axis=1)   # [T]
        lc_norm         = spatial_norm(lc_raw)   # [T, P_lr]

        # Assign each day to nearest centroid
        # Distance matrix: [T, N] — batched for memory
        logger.info(f'Assigning {T} days to N={self.N} clusters...')
        cluster_seq = np.zeros(T, dtype=np.int32)
        batch = 500   # days per distance batch
        for t0 in range(0, T, batch):
            t1      = min(t0 + batch, T)
            # [B, P_lr] vs [N, P_lr] → squared distances [B, N]
            diff    = lc_norm[t0:t1, None, :] - self.centroids[None, :, :]
            dists   = (diff ** 2).sum(axis=-1)   # [B, N]
            cluster_seq[t0:t1] = dists.argmin(axis=1)

        # Build predictions
        logger.info('Building predictions...')
        pred = np.zeros((T, H, W), dtype=np.float32)

        dry_mask = lc_domain_means < self.config.lr_threshold
        n_dry    = dry_mask.sum()
        logger.info(f'  Dry days (LR mean < {self.config.lr_threshold}): '
                    f'{n_dry}/{T} ({n_dry/T*100:.1f}%) → predict 0')

        for t in range(T):
            if dry_mask[t]:
                pred[t] = 0.0
                continue

            k         = cluster_seq[t]
            hr_mean_k = self.cluster_hr_mean[k].reshape(H, W)
            pred[t]   = hr_mean_k

            if self.config.use_scaling:
                lr_dm_k = self.cluster_lr_dmean[k]
                lr_dm_t = lc_domain_means[t]
                if lr_dm_k > self.config.lr_threshold:
                    scale   = lr_dm_t / lr_dm_k
                    scale   = min(scale, self.config.clip_scale_max)
                    pred[t] = hr_mean_k * scale
                # else: cluster is essentially dry, no scaling

        pred = np.maximum(pred, 0.0)
        obs  = hr_data.astype(np.float32)
        logger.info(f'  Pred: [{pred.min():.2f}, {pred.max():.2f}] mm/day')
        return pred, obs, lat, lon, cluster_seq

    # ── Persistence ─────────────────────────────────────────

    def save(self, path: str, logger: logging.Logger):
        logger.info(f'Saving model: {path}')
        np.savez_compressed(
            path,
            centroids        = self.centroids,
            cluster_hr_mean  = self.cluster_hr_mean,
            cluster_lr_dmean = self.cluster_lr_dmean,
            cluster_counts   = self.cluster_counts,
            cluster_months   = self.cluster_months,
            H  = np.array(self.H),  W  = np.array(self.W),
            Hc = np.array(self.Hc), Wc = np.array(self.Wc),
            N  = np.array(self.N),
        )
        logger.info(f'  Saved {os.path.getsize(path)/1e6:.1f} MB')

    @classmethod
    def load(cls, path: str, config: Config,
             logger: logging.Logger) -> 'WeatherTypingModel':
        logger.info(f'Loading model: {path}')
        data = np.load(path)
        obj  = cls(config)
        obj.centroids        = data['centroids']
        obj.cluster_hr_mean  = data['cluster_hr_mean']
        obj.cluster_lr_dmean = data['cluster_lr_dmean']
        obj.cluster_counts   = data['cluster_counts']
        obj.cluster_months   = data['cluster_months']
        obj.H  = int(data['H']);  obj.W  = int(data['W'])
        obj.Hc = int(data['Hc']); obj.Wc = int(data['Wc'])
        obj.N  = int(data['N'])
        logger.info(f'  N={obj.N}  H={obj.H}  W={obj.W}')
        return obj


# ============================================================
# ELBOW / SILHOUETTE DIAGNOSTICS
# ============================================================
def run_elbow_diagnostics(train_h5: str, config: Config,
                           logger: logging.Logger, out_dir: str):
    """
    Compute WCSS (within-cluster sum of squares) and silhouette scores
    for a range of N values. Saves elbow plot to guide N selection.

    The elbow in WCSS and the peak in silhouette indicate the optimal N.
    For Indian precipitation, expect the elbow around N=15-25 corresponding
    to: monsoon onset, active/break phases, ENSO-driven years,
    western disturbances, post-monsoon, northeast monsoon, winter, etc.
    """
    sf = config.scale_factor
    logger.info(f'Loading training data for diagnostics: {train_h5}')
    with h5py.File(train_h5, 'r') as f:
        hr_data = f['precipitation'][:]
    T, H, W = hr_data.shape
    Hc, Wc  = H // sf, W // sf
    P_lr    = Hc * Wc

    # Build normalised LR library
    lc_raw = np.zeros((T, P_lr), dtype=np.float32)
    batch  = config.lr_batch_size
    for t0 in tqdm(range(0, T, batch), desc='  LR for diagnostics'):
        t1 = min(t0 + batch, T)
        lc_raw[t0:t1] = compute_lr_coarse(
            hr_data[t0:t1].astype(np.float32), sf
        ).reshape(t1 - t0, P_lr)
    lc_norm = spatial_norm(lc_raw)   # [T, P_lr]

    # Subsample for silhouette (O(T²) — too slow on full 10K days)
    sil_sample = min(2000, T)
    rng        = np.random.default_rng(config.random_state)
    sil_idx    = rng.choice(T, size=sil_sample, replace=False)
    lc_sil     = lc_norm[sil_idx]

    wcss_list  = []
    sil_list   = []
    N_range    = list(config.elbow_N_range)
    logger.info(f'Testing N = {N_range}...')

    for N in N_range:
        km = MiniBatchKMeans(
            n_clusters=N, random_state=config.random_state,
            max_iter=200, n_init=5, batch_size=min(1024, T)
        )
        km.fit(lc_norm)
        wcss_list.append(km.inertia_)
        sil_labels = km.predict(lc_sil)
        if len(np.unique(sil_labels)) > 1:
            sil_score = silhouette_score(lc_sil, sil_labels, sample_size=1000,
                                          random_state=config.random_state)
        else:
            sil_score = float('nan')
        sil_list.append(sil_score)
        logger.info(f'  N={N:3d}: WCSS={km.inertia_:10.1f}  '
                    f'Silhouette={sil_score:.4f}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(N_range, wcss_list, '-o', color='steelblue', markersize=6)
    axes[0].set_xlabel('Number of Weather Types (N)')
    axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    axes[0].set_title('Elbow Method — WCSS vs N\n'
                       '(Look for elbow: diminishing returns in WCSS reduction)')
    axes[0].grid(alpha=0.3)

    valid_sil = [(n, s) for n, s in zip(N_range, sil_list) if not np.isnan(s)]
    if valid_sil:
        ns, ss = zip(*valid_sil)
        axes[1].plot(ns, ss, '-o', color='tomato', markersize=6)
        best_n = ns[int(np.argmax(ss))]
        axes[1].axvline(best_n, color='gray', ls='--', lw=1.5,
                        label=f'Best N={best_n} (sil={max(ss):.4f})')
        axes[1].legend(fontsize=9)
    axes[1].set_xlabel('Number of Weather Types (N)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score vs N\n'
                       '(Higher = better-separated clusters; '
                       f'sample={sil_sample} days)')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Weather Typing — Cluster Diagnostics\n'
                 'Run first to choose N before calibration',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig_elbow_diagnostics.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_elbow_diagnostics.png saved')

    # Save results
    diag = {'N_range': N_range, 'wcss': wcss_list,
            'silhouette': [float(s) for s in sil_list]}
    with open(os.path.join(out_dir, 'cluster_diagnostics.json'), 'w') as fj:
        json.dump(diag, fj, indent=2)
    logger.info('Diagnostics complete. Inspect fig_elbow_diagnostics.png '
                'to choose N, then re-run without --elbow_only.')


# ============================================================
# CLUSTER ATLAS  — the key interpretability plot
# ============================================================
def plot_cluster_atlas(model: WeatherTypingModel,
                        lat: np.ndarray, lon: np.ndarray,
                        out_dir: str, config: Config,
                        logger: logging.Logger):
    """
    Two-panel atlas for each weather type:
      Left:  LR circulation pattern (centroid, Hc×Wc coarse grid)
      Right: Mean HR precipitation field (H×W fine grid)

    Physical interpretation guide for India:
      High LR NW India, low LR East  → Western disturbance (winter)
      High LR over Bay of Bengal      → Bay of Bengal low / monsoon onset
      High LR over Arabian Sea        → Arabian Sea branch monsoon
      Low LR everywhere               → Monsoon break
      High LR Central India           → Active monsoon phase

    Clusters with many training days (bar width) = climatologically frequent types.
    Clusters with high HR mean = wet types.

    Layout: ceil(N / atlas_cols) rows × atlas_cols columns,
    each cell = [LR centroid | HR mean] pair.
    """
    N        = model.N
    ncols    = config.atlas_cols
    nrows    = (N + ncols - 1) // ncols
    extent_lr = [lon.min(), lon.max(), lat.min(), lat.max()]

    # LR lat/lon (coarse grid)
    lat_c = np.linspace(lat.min(), lat.max(), model.Hc)
    lon_c = np.linspace(lon.min(), lon.max(), model.Wc)
    extent_c = [lon_c.min(), lon_c.max(), lat_c.min(), lat_c.max()]

    # Sort clusters by total training days (descending) for readability
    sort_idx = np.argsort(model.cluster_counts)[::-1]

    fig_lr = plt.figure(figsize=(4 * ncols, 3.5 * nrows))
    fig_hr = plt.figure(figsize=(4 * ncols, 3.5 * nrows))

    month_abbr = ['J','F','M','A','M','J','J','A','S','O','N','D']

    for rank, k in enumerate(sort_idx):
        row = rank // ncols; col = rank % ncols
        pos = row * ncols + col + 1

        # LR centroid
        centroid  = model.centroids[k].reshape(model.Hc, model.Wc)
        ax_lr = fig_lr.add_subplot(nrows, ncols, pos)
        im_lr = ax_lr.imshow(centroid, origin='lower', extent=extent_c,
                              cmap='RdBu_r', aspect='auto')
        n_k   = model.cluster_counts[k]
        dom_m = model.cluster_months[k].argmax()
        ax_lr.set_title(f'Type {k}  ({n_k}d, dom={month_abbr[dom_m]})',
                         fontsize=8)
        ax_lr.set_xlabel('Lon', fontsize=7); ax_lr.set_ylabel('Lat', fontsize=7)
        ax_lr.tick_params(labelsize=6)
        fig_lr.colorbar(im_lr, ax=ax_lr, fraction=0.046, label='norm')

        # HR mean
        hr_mean  = model.cluster_hr_mean[k].reshape(model.H, model.W)
        ax_hr = fig_hr.add_subplot(nrows, ncols, pos)
        vmax_hr = np.percentile(model.cluster_hr_mean, 99)
        im_hr = ax_hr.imshow(hr_mean, origin='lower', extent=extent_lr,
                              cmap='YlGnBu', vmin=0, vmax=vmax_hr, aspect='auto')
        ax_hr.set_title(f'Type {k}  LR={model.cluster_lr_dmean[k]:.1f} '
                         f'HR={hr_mean.mean():.1f} mm/d', fontsize=8)
        ax_hr.set_xlabel('Lon', fontsize=7); ax_hr.set_ylabel('Lat', fontsize=7)
        ax_hr.tick_params(labelsize=6)
        fig_hr.colorbar(im_hr, ax=ax_hr, fraction=0.046, label='mm/day')

    fig_lr.suptitle(f'Weather Type Atlas — LR Circulation Patterns (N={N})\n'
                     '(Normalised coarse LR field centroid per type)',
                     fontsize=11, y=1.01)
    fig_lr.tight_layout()
    lr_path = os.path.join(out_dir, 'fig_cluster_atlas_LR.png')
    fig_lr.savefig(lr_path, dpi=130, bbox_inches='tight'); plt.close(fig_lr)
    logger.info(f'  fig_cluster_atlas_LR.png ✓')

    fig_hr.suptitle(f'Weather Type Atlas — Mean HR Precipitation (N={N})\n'
                     '(Mean MSWX HR field for all training days in each type)',
                     fontsize=11, y=1.01)
    fig_hr.tight_layout()
    hr_path = os.path.join(out_dir, 'fig_cluster_atlas_HR.png')
    fig_hr.savefig(hr_path, dpi=130, bbox_inches='tight'); plt.close(fig_hr)
    logger.info(f'  fig_cluster_atlas_HR.png ✓')


def plot_cluster_frequency_calendar(model: WeatherTypingModel,
                                     cluster_seq: np.ndarray,
                                     start_year: int,
                                     out_dir: str, label: str,
                                     logger: logging.Logger):
    """
    Monthly frequency heatmap of weather type occurrence during the period.
    Rows = weather types, columns = calendar months.
    Shows whether the clustering reproduces the known seasonal cycle.
    """
    T        = len(cluster_seq)
    _, months = build_date_index(T, start_year)

    # Count occurrences [N, 12]
    freq = np.zeros((model.N, 12), dtype=np.float32)
    for k in range(model.N):
        for m in range(1, 13):
            freq[k, m-1] = ((cluster_seq == k) & (months == m)).sum()
    # Normalise by month total (fraction of that month dominated by type k)
    freq_norm = freq / (freq.sum(axis=0, keepdims=True) + 1e-10)

    # Sort by dominant training month for readability
    sort_idx = np.argsort(model.cluster_months.argmax(axis=1))
    freq_norm_sorted = freq_norm[sort_idx]

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = plt.subplots(figsize=(12, max(6, model.N * 0.35)))
    im = ax.imshow(freq_norm_sorted, aspect='auto', cmap='YlOrRd',
                    vmin=0, vmax=freq_norm_sorted.max())
    ax.set_xticks(range(12)); ax.set_xticklabels(month_names, rotation=30)
    ax.set_yticks(range(model.N))
    ax.set_yticklabels([f'Type {sort_idx[i]}' for i in range(model.N)],
                        fontsize=8)
    ax.set_xlabel('Month'); ax.set_ylabel('Weather Type')
    ax.set_title(f'Weather Type Frequency by Month — {label}\n'
                 '(Fraction of days in each month assigned to each type)\n'
                 'Seasonal coherence validates physical interpretability')
    fig.colorbar(im, ax=ax, label='Fraction of month', fraction=0.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'fig_cluster_calendar_{label}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_cluster_calendar_{label}.png ✓')


# ============================================================
# STANDARD METRICS  (identical schema to all other baselines)
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
def save_netcdf(pred, obs, lat, lon, out_path, config, label, cluster_seq):
    T, H, W = pred.shape
    with nc.Dataset(out_path, 'w', format='NETCDF4') as ds:
        ds.title            = f'Weather Typing baseline ({label})'
        ds.description      = (f'K-Means N={config.N} on normalised LR coarse, '
                                f'cluster HR mean + magnitude scaling='
                                f'{config.use_scaling}')
        ds.downscale_factor = config.scale_factor
        ds.N_clusters       = config.N
        ds.use_scaling      = str(config.use_scaling)
        ds.calibration      = '1979-2005'
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
        vp = ds.createVariable('pr_wt', 'f4', ('time','lat','lon'),
                                zlib=True, complevel=4, fill_value=-9999.)
        vp[:] = pred; vp.units = 'mm day-1'
        vp.long_name = 'Weather typing downscaled precipitation'
        vo = ds.createVariable('pr_obs', 'f4', ('time','lat','lon'),
                                zlib=True, complevel=4, fill_value=-9999.)
        vo[:] = obs; vo.units = 'mm day-1'
        vo.long_name = 'MSWX ground truth precipitation'
        vc = ds.createVariable('cluster_id', 'i2', ('time',))
        vc[:] = cluster_seq.astype(np.int16); vc.units = '1'
        vc.long_name = 'Weather type cluster assignment per day'


# ============================================================
# STANDARD PLOTS  (no cfeature.BORDERS)
# ============================================================
PRECIP_CMAP = 'YlGnBu'

def make_plots(obs, pred, lat, lon, out_dir, label, config,
               cluster_seq, start_year, model, logger):
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
        (axes[1], pred_mean, f'Weath.Typ. ({label})', PRECIP_CMAP, 0, vmax_p),
        (axes[2], bias,      'Bias (WT−Truth)',  'RdBu_r',  -bmax, bmax),
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
    ax.scatter(obs_ts, pred_ts, alpha=0.25, s=8, color='darkorange', rasterized=True)
    vmax = max(obs_ts.max(), pred_ts.max()) * 1.05
    ax.plot([0,vmax],[0,vmax],'k--',lw=1,label='1:1')
    ax.set_xlim(0,vmax); ax.set_ylim(0,vmax)
    ax.set_xlabel('Truth (mm day⁻¹)'); ax.set_ylabel('Weather Typing (mm day⁻¹)')
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
    axes[0].bar(x-w/2, [obs_pct[p] for p in pcts],  w, label='Truth',
                color='steelblue')
    axes[0].bar(x+w/2, [pred_pct[p] for p in pcts], w, label='Weath.Typ.',
                color='darkorange')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'P{p}' for p in pcts], rotation=30)
    axes[0].set_ylabel('mm day⁻¹')
    axes[0].set_title('Percentile Values (wet ≥ 1 mm day⁻¹)')
    axes[0].legend()
    ratios = [pred_pct[p]/obs_pct[p] if obs_pct[p]>0 else float('nan')
              for p in pcts]
    colors = ['tomato' if r>1.05 else ('steelblue' if r<0.95 else 'green')
              for r in ratios]
    axes[1].bar(x, ratios, color=colors, width=0.5)
    axes[1].axhline(1.0, color='k', lw=1.5, ls='--')
    axes[1].axhspan(0.95, 1.05, alpha=0.15, color='green', label='±5% band')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'P{p}' for p in pcts], rotation=30)
    axes[1].set_ylabel('Ratio (WT / Truth)'); axes[1].legend(fontsize=9)
    plt.suptitle(f'Percentile Preservation — Weather Typing ({label})', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_percentile_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_percentile_{label}.png ✓')

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
                label=f'≥{thr} mm day⁻¹', markersize=5)
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='FSS=0.5')
    ax.set_xlabel('Spatial Scale (grid cells at 0.1°)')
    ax.set_ylabel('Fractions Skill Score (FSS)')
    ax.set_title(f'FSS vs Scale — Weather Typing ({label})\n'
                 '(WT uses real observed HR days → better spatial coherence\n'
                 'than pixel-wise methods; limited by cluster representativeness)')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_fss_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_fss_{label}.png ✓')

    # PSD
    k_obs, psd_obs   = azimuthal_psd(obs_mean)
    k_pred, psd_pred = azimuthal_psd(pred_mean)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(k_obs, psd_obs,  color='steelblue',  lw=1.5, label='Truth')
    axes[0].semilogy(k_pred,psd_pred, color='darkorange', lw=1.5, ls='--',
                     label='Weath.Typ.')
    axes[0].set_title(f'PSD — {label}'); axes[0].legend(); axes[0].grid(alpha=0.3)
    ratio = psd_pred / (psd_obs + 1e-20)
    axes[1].plot(k_obs, ratio, color='darkorange', lw=1.5)
    axes[1].axhline(1.0, color='k', ls='--')
    axes[1].axhspan(0.9, 1.1, alpha=0.1, color='green', label='±10% band')
    axes[1].set_ylim(0, 2)
    axes[1].set_title(f'PSD Ratio — {label}\n'
                      '(WT uses real HR days → PSD closer to truth than\n'
                      'interpolation methods; mean across days smooths fine-scale)')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fig_psd_{label}.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    logger.info(f'  fig_psd_{label}.png ✓')

    # Cluster frequency calendar
    plot_cluster_frequency_calendar(
        model, cluster_seq, start_year, out_dir, label, logger
    )

    return fss_results


# ============================================================
# PER-PERIOD EVALUATION
# ============================================================
def run_period(h5_path, label, start_year, model, config, logger, out_dir):
    logger.info(f'\n{"─"*55}')
    logger.info(f'PERIOD: {label}')
    logger.info(f'{"─"*55}')

    t0 = time.time()
    pred, obs, lat, lon, cluster_seq = model.predict_period(
        h5_path, start_year, logger
    )
    logger.info(f'  Inference in {time.time()-t0:.1f}s')
    T = pred.shape[0]

    # NetCDF
    nc_path = os.path.join(out_dir, f'wt_pred_{label}.nc')
    save_netcdf(pred, obs, lat, lon, nc_path, config, label, cluster_seq)
    logger.info(f'  NetCDF: {nc_path}')

    # Scalar metrics
    logger.info('Computing metrics...')
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
            'rmse':   round(rmse(o, p), 4),   'mae':  round(mae(o, p), 4),
            'bias':   round(mean_bias(o, p), 4), 'r': round(pearson_r(o, p), 4),
            'kge':    round(kge(o, p), 4),
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

    # Cluster usage stats for this period
    unique, counts = np.unique(cluster_seq, return_counts=True)
    n_used  = len(unique)
    top3    = sorted(zip(counts.tolist(), unique.tolist()), reverse=True)[:3]
    logger.info(f'  Clusters used: {n_used}/{model.N}  '
                f'Top-3: {[(f"k{k}={c}d") for c, k in top3]}')

    # Plots
    fss_results = make_plots(obs, pred, lat, lon, out_dir, label, config,
                              cluster_seq, start_year, model, logger)

    # Atlas (shared, only generated once)
    atlas_path_lr = os.path.join(out_dir, 'fig_cluster_atlas_LR.png')
    if not os.path.exists(atlas_path_lr):
        plot_cluster_atlas(model, lat, lon, out_dir, config, logger)

    metrics = {
        'method':       'weather_typing',
        'N_clusters':   config.N,
        'use_scaling':  config.use_scaling,
        'clip_scale':   config.clip_scale_max,
        'calibration':  '1979-2005',
        'period':       label,
        'n_timesteps':  T,
        'overall':      overall,
        'percentiles_obs':  {str(k): round(v,3) for k,v in obs_pct.items()},
        'percentiles_pred': {str(k): round(v,3) for k,v in pred_pct.items()},
        'percentile_skill': {str(k): round(v,4) for k,v in pct_skill.items()},
        'by_season':    season_metrics,
        'fss': {str(thr): {str(sc): round(v,4)
                            for sc,v in zip(config.fss_scales, vals)}
                for thr,vals in fss_results.items()},
        'psd_ratio_high_k': round(psd_ratio_hk, 4),
        'cluster_usage': {
            'n_clusters_used': int(n_used),
            'n_clusters_total': int(model.N),
        },
    }
    json_path = os.path.join(out_dir, f'metrics_{label}.json')
    with open(json_path, 'w') as fj:
        json.dump(metrics, fj, indent=2)
    logger.info(f'  Metrics: {json_path}')
    return metrics


# ============================================================
# MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Weather Typing baseline for precipitation downscaling'
    )
    parser.add_argument('--skip_calib',  action='store_true',
                        help='Load wt_model.npz, skip calibration')
    parser.add_argument('--elbow_only',  action='store_true',
                        help='Run cluster diagnostics only (no prediction)')
    parser.add_argument('--N',           type=int, default=None,
                        help='Number of weather types (default: 20)')
    parser.add_argument('--no_scaling',  action='store_true',
                        help='Disable magnitude scaling')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--train_h5',   type=str, default=None)
    parser.add_argument('--val_h5',     type=str, default=None)
    parser.add_argument('--test_h5',    type=str, default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    config = Config()

    if args.results_dir: config.results_dir = args.results_dir
    if args.train_h5:    config.train_h5    = args.train_h5
    if args.val_h5:      config.val_h5      = args.val_h5
    if args.test_h5:     config.test_h5     = args.test_h5
    if args.N:           config.N           = args.N
    if args.no_scaling:  config.use_scaling = False

    # mkdir BEFORE setup_logging
    out_dir = Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))

    logger.info('=' * 65)
    logger.info('Weather Typing (Circulation Pattern Classification) — Val + Test')
    logger.info('=' * 65)
    logger.info(f'N weather types:  {config.N}')
    logger.info(f'Magnitude scaling:{config.use_scaling}  '
                f'(clip_max={config.clip_scale_max})')
    logger.info(f'Results dir:      {out_dir}')

    # ── ELBOW DIAGNOSTICS ONLY ──────────────────────────────
    if args.elbow_only:
        logger.info('\n--- CLUSTER DIAGNOSTICS (elbow + silhouette) ---')
        run_elbow_diagnostics(config.train_h5, config, logger, str(out_dir))
        logger.info('\nDone. Choose N from fig_elbow_diagnostics.png '
                    'and re-run without --elbow_only.')
        return

    t_all      = time.time()
    model_path = str(out_dir / 'wt_model.npz')

    # ── CALIBRATION ─────────────────────────────────────────
    if args.skip_calib and os.path.exists(model_path):
        logger.info('\n--- LOADING MODEL ---')
        model = WeatherTypingModel.load(model_path, config, logger)
    else:
        logger.info('\n--- CALIBRATION ---')
        model = WeatherTypingModel(config)
        t0 = time.time()
        model.fit(config.train_h5, logger)
        logger.info(f'  Calibration done in {(time.time()-t0)/60:.1f} min')
        model.save(model_path, logger)

    # ── EVALUATE BOTH PERIODS ───────────────────────────────
    all_metrics = {}
    for period_name, h5_path, label, start_year in [
        ('val',  config.val_h5,  '2006-2010', config.val_start_year),
        ('test', config.test_h5, '2011-2014', config.test_start_year),
    ]:
        t0 = time.time()
        all_metrics[period_name] = run_period(
            h5_path, label, start_year, model, config, logger, str(out_dir)
        )
        logger.info(f'  {period_name} done in {(time.time()-t0)/60:.1f} min')

    combined_path = str(out_dir / 'metrics_combined.json')
    with open(combined_path, 'w') as fj:
        json.dump(all_metrics, fj, indent=2)

    logger.info('\n' + '='*65)
    logger.info('SUMMARY')
    logger.info('='*65)
    logger.info(f'N={config.N}  scaling={config.use_scaling}')
    for pname, m in all_metrics.items():
        o = m['overall']
        logger.info(f"{pname} ({m['period']}): "
                    f"RMSE={o['rmse']:.4f}  KGE={o['kge']:.4f}  "
                    f"r={o['pearson_r']:.4f}  "
                    f"PSD={m['psd_ratio_high_k']:.4f}")
    logger.info(f'\nTotal time: {(time.time()-t_all)/60:.1f} min')
    logger.info(f'Done. Combined: {combined_path}')


if __name__ == '__main__':
    main()