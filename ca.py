"""
eval_ca.py - Constructed Analogues baseline (Wood 2002 / Hidalgo 2008)
Val (2006-2010) + Test (2011-2014). CPU-only.

Algorithm per day:
  1. Find top-K training days with highest spatial correlation to query LR.
  2. Solve OLS: LR_query ~ A @ w  (A = [P_lr, K] analogue LR matrix).
  3. Predict: HR_pred = w @ HR_analogues.  Clip to >= 0.

Memory: HR library [9862, 122500] float32 ~ 4.8 GB loaded once.
        Use --hr_dtype float16 for ~2.4 GB at slight precision cost.

Usage:
  python eval_ca.py
  python eval_ca.py --skip_calib
  python eval_ca.py --K 25 --hr_dtype float16
"""
import os, json, logging, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from pathlib import Path
from datetime import datetime, date, timedelta
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    train_h5 = (
        "/path/to/project/"
        "PhD_Precipitation/02_Data/processed/hdf5/mswx_train_1979-2005.h5"
    )
    val_h5 = (
        "/path/to/project/"
        "PhD_Precipitation/02_Data/processed/hdf5/mswx_val_2006-2010.h5"
    )
    test_h5 = (
        "/path/to/project/"
        "PhD_Precipitation/02_Data/processed/hdf5/mswx_test_2011-2014.h5"
    )
    results_dir = (
        "/path/to/project/"
        "PhD_Precipitation/03_Code/results/ca_baseline"
    )
    scale_factor    = 10
    K               = 30      # analogues for regression
    lr_threshold    = 0.1     # mm/day -- dry-day cutoff
    hr_dtype        = "float32"  # or float16 to halve RAM
    lr_batch_size   = 50
    infer_batch     = 100     # days per correlation search batch
    fss_thresholds  = (1.0, 5.0, 20.0)
    fss_scales      = (1, 3, 5, 10, 15, 20, 30, 50)
    percentiles     = (50, 75, 90, 95, 99, 99.5, 99.9)
    val_start_year  = 2006
    test_start_year = 2011


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(results_dir):
    log_file = os.path.join(
        results_dir, f"eval_ca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger("ca_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    for h in [logging.FileHandler(log_file), logging.StreamHandler()]:
        h.setFormatter(fmt); logger.addHandler(h)
    return logger


# ── LR generation ─────────────────────────────────────────────────────────────
def lr_coarse(hr_batch, sf):
    """HR [B,H,W] -> avg_pool -> [B, H//sf, W//sf]. CPU-safe."""
    t  = torch.from_numpy(hr_batch.astype(np.float32)).unsqueeze(1)
    lc = F.avg_pool2d(t, sf, sf)
    return lc.squeeze(1).numpy()


# ── Normalisation ─────────────────────────────────────────────────────────────
def spatial_norm(fields, eps=1e-8):
    """Zero-mean, unit-std per row. fields: [N,P] or [P]."""
    if fields.ndim == 1:
        mu = fields.mean(); sig = fields.std()
        return (fields - mu) / (sig + eps), mu, sig
    mu  = fields.mean(axis=1, keepdims=True)
    sig = fields.std (axis=1, keepdims=True)
    return (fields - mu) / (sig + eps), mu.ravel(), sig.ravel()


# ── CA Library ────────────────────────────────────────────────────────────────
class CALibrary:
    """Holds the training-period analogue library in RAM."""
    def __init__(self, config):
        self.config = config
        self.lr_norm = self.lr_raw = self.hr = self.dmeans = None
        self.T = self.H = self.W = self.Hc = self.Wc = None

    def build(self, h5_path, logger):
        sf = self.config.scale_factor
        logger.info(f"Loading training data: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            hr_data = f["precipitation"][:]
        T, H, W = hr_data.shape
        Hc, Wc  = H // sf, W // sf
        self.T, self.H, self.W, self.Hc, self.Wc = T, H, W, Hc, Wc
        assert H % sf == 0 and W % sf == 0
        logger.info(f"  {T}x{H}x{W}  coarse: {Hc}x{Wc}")

        hr_dt = np.float16 if self.config.hr_dtype == "float16" else np.float32
        self.lr_norm = np.zeros((T, Hc*Wc), dtype=np.float32)
        self.lr_raw  = np.zeros((T, Hc*Wc), dtype=np.float32)
        self.hr      = np.zeros((T, H*W),   dtype=hr_dt)
        self.dmeans  = np.zeros(T,           dtype=np.float32)
        logger.info(f"  HR library: {self.hr.nbytes/1e9:.2f} GB ({hr_dt.__name__})")

        batch = self.config.lr_batch_size
        for t0 in tqdm(range(0, T, batch), desc="  Building library"):
            t1   = min(t0 + batch, T)
            hr_b = hr_data[t0:t1].astype(np.float32)
            lc_b = lr_coarse(hr_b, sf).reshape(t1 - t0, Hc*Wc)
            self.dmeans [t0:t1] = lc_b.mean(axis=1)
            ln, _, _            = spatial_norm(lc_b)
            self.lr_norm[t0:t1] = ln.astype(np.float32)
            self.lr_raw [t0:t1] = lc_b
            self.hr     [t0:t1] = hr_b.reshape(t1 - t0, H*W).astype(hr_dt)

        n_wet = (self.dmeans >= self.config.lr_threshold).sum()
        logger.info(f"  Wet training days: {n_wet}/{T}")

    def save(self, path, logger):
        logger.info(f"Saving library: {path}")
        np.savez_compressed(path,
            lr_norm=self.lr_norm, lr_raw=self.lr_raw, hr=self.hr,
            dmeans=self.dmeans,
            T=self.T, H=self.H, W=self.W, Hc=self.Hc, Wc=self.Wc)
        logger.info(f"  {os.path.getsize(path)/1e6:.0f} MB saved")

    @classmethod
    def load(cls, path, config, logger):
        logger.info(f"Loading library: {path}")
        d = np.load(path)
        obj = cls(config)
        obj.lr_norm = d["lr_norm"]; obj.lr_raw = d["lr_raw"]
        obj.hr      = d["hr"];      obj.dmeans = d["dmeans"]
        for k in ("T","H","W","Hc","Wc"):
            setattr(obj, k, int(d[k]))
        logger.info(f"  T_train={obj.T}  HR dtype={obj.hr.dtype}")
        return obj


# ── CA Predictor ──────────────────────────────────────────────────────────────
class CAPredictor:
    """Runs CA prediction against a fitted CALibrary."""
    def __init__(self, lib, config):
        self.lib = lib; self.config = config

    def _batch(self, lc_raw_b):
        """[B, P_lr] -> [B, P_hr] predicted HR."""
        B = lc_raw_b.shape[0]; K = self.config.K
        P_hr = self.lib.H * self.lib.W

        # Normalise queries
        lc_n, _, _ = spatial_norm(lc_raw_b)

        # Pearson correlation: [B, T_train]
        q_norms   = np.linalg.norm(lc_n, axis=1, keepdims=True) + 1e-10
        lib_norms = np.linalg.norm(self.lib.lr_norm, axis=1) + 1e-10
        corr = (lc_n @ self.lib.lr_norm.T) / (q_norms * lib_norms[None,:])

        pred_b = np.zeros((B, P_hr), dtype=np.float32)
        for b in range(B):
            # Top-K by correlation
            top_k = np.argpartition(corr[b], -K)[-K:]
            top_k = top_k[np.argsort(corr[b, top_k])[::-1]]
            # OLS: lr_raw_lib[top_k].T @ w = lc_raw_b[b]
            A = self.lib.lr_raw[top_k].T   # [P_lr, K]
            w, _, _, _ = np.linalg.lstsq(A, lc_raw_b[b], rcond=None)
            # Weighted HR sum
            hr_k = self.lib.hr[top_k].astype(np.float32)  # [K, P_hr]
            pred_b[b] = w @ hr_k
        return np.maximum(pred_b, 0.0)

    def predict(self, h5_path, logger):
        """Full inference. Returns (pred [T,H,W], obs, lat, lon)."""
        sf = self.config.scale_factor
        logger.info(f"Loading: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            hr_data = f["precipitation"][:]
            lat = f["lat"][:] if "lat" in f else np.linspace(6.5,41.0,self.lib.H)
            lon = f["lon"][:] if "lon" in f else np.linspace(66.5,100.0,self.lib.W)
        T, H, W = hr_data.shape; P_lr = self.lib.Hc * self.lib.Wc
        logger.info(f"  {T}x{H}x{W}")

        # LR coarse for all days
        lc_all = np.zeros((T, self.lib.Hc, self.lib.Wc), dtype=np.float32)
        for t0 in tqdm(range(0, T, self.config.lr_batch_size), desc="  LR coarse"):
            t1 = min(t0 + self.config.lr_batch_size, T)
            lc_all[t0:t1] = lr_coarse(hr_data[t0:t1].astype(np.float32), sf)
        lc_flat = lc_all.reshape(T, P_lr)

        wet_mask = lc_flat.mean(axis=1) >= self.config.lr_threshold
        n_wet = wet_mask.sum()
        logger.info(f"  Wet: {n_wet}/{T} ({n_wet/T*100:.1f}%)")

        pred    = np.zeros((T, H*W), dtype=np.float32)
        wet_idx = np.where(wet_mask)[0]
        batch   = self.config.infer_batch
        t_inf   = time.time()
        for b0 in tqdm(range(0, len(wet_idx), batch), desc="  CA inference"):
            b1       = min(b0 + batch, len(wet_idx))
            idx_b    = wet_idx[b0:b1]
            pred[idx_b] = self._batch(lc_flat[idx_b])
        elapsed = time.time() - t_inf
        logger.info(f"  Done in {elapsed/60:.1f} min  "
                    f"({elapsed/n_wet*1000:.0f} ms/wet-day)")
        pred = pred.reshape(T, H, W)
        obs  = hr_data.astype(np.float32)
        logger.info(f"  Pred: [{pred.min():.2f}, {pred.max():.2f}] mm/day")
        return pred, obs, lat, lon


# ── Metrics ───────────────────────────────────────────────────────────────────
def rmse(o,p):      return float(np.sqrt(np.mean((p-o)**2)))
def mae(o,p):       return float(np.mean(np.abs(p-o)))
def mean_bias(o,p): return float(np.mean(p-o))
def pearson_r(o,p): return float(np.corrcoef(o.ravel(),p.ravel())[0,1])
def kge(o,p):
    o,p=o.ravel(),p.ravel()
    r=float(np.corrcoef(o,p)[0,1])
    a=float(p.std()/(o.std()+1e-10)); b=float(p.mean()/(o.mean()+1e-10))
    return float(1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2))
def wet_day_frequency(f,thr=1.0): return float(np.mean(f>=thr))
def percentile_values(f,pcts):
    wet=f[f>=1.0]
    return {p:float(np.percentile(wet,p)) for p in pcts} if wet.size>0            else {p:0.0 for p in pcts}
def percentile_skill(op,pp):
    return {p:(pp[p]/op[p] if op[p]>0 else float("nan")) for p in op}

def fss_score(obs,pred,threshold,scale):
    if obs.ndim==3:
        scores=[fss_score(obs[i],pred[i],threshold,scale)
                for i in range(obs.shape[0])]
        valid=[s for s in scores if not np.isnan(s)]
        return float(np.mean(valid)) if valid else float("nan")
    ob=(obs>=threshold).astype(np.float32)
    pb=(pred>=threshold).astype(np.float32)
    if ob.sum()==0 and pb.sum()==0: return float("nan")
    of=uniform_filter(ob,size=scale,mode="constant")
    pf=uniform_filter(pb,size=scale,mode="constant")
    num=np.mean((of-pf)**2); den=np.mean(of**2+pf**2)
    return float(1-num/den) if den>0 else float("nan")

def azimuthal_psd(f2d):
    H,W=f2d.shape
    power=np.abs(np.fft.fftshift(np.fft.fft2(f2d)))**2
    kx=np.fft.fftshift(np.fft.fftfreq(W)); ky=np.fft.fftshift(np.fft.fftfreq(H))
    KX,KY=np.meshgrid(kx,ky); K=np.sqrt(KX**2+KY**2)
    n=min(H,W)//2; bins=np.linspace(0,0.5,n+1)
    k_mid=0.5*(bins[:-1]+bins[1:])
    psd=np.array([power[(K>=bins[i])&(K<bins[i+1])].mean()
                  if ((K>=bins[i])&(K<bins[i+1])).any() else 0.0
                  for i in range(n)])
    return k_mid,psd

def build_season_mask(T,start_year):
    sm={12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",
         6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
    d=date(start_year,1,1); labels=[]
    for _ in range(T):
        labels.append(sm[d.month]); d+=timedelta(days=1)
    return np.array(labels)


# ── NetCDF ────────────────────────────────────────────────────────────────────
def save_netcdf(pred,obs,lat,lon,path,config,label):
    T,H,W=pred.shape
    with nc.Dataset(path,"w",format="NETCDF4") as ds:
        ds.title=f"Constructed Analogues ({label})"
        ds.K_analogues=config.K; ds.calibration="1979-2005"
        ds.downscale_factor=config.scale_factor
        ds.created=datetime.now().isoformat()
        ds.createDimension("time",T); ds.createDimension("lat",H)
        ds.createDimension("lon",W)
        vt=ds.createVariable("time","i4",("time",))
        vt[:]=np.arange(T); vt.units=f"days since {label[:4]}-01-01"
        vlat=ds.createVariable("lat","f4",("lat",))
        vlat[:]=lat; vlat.units="degrees_north"
        vlon=ds.createVariable("lon","f4",("lon",))
        vlon[:]=lon; vlon.units="degrees_east"
        vp=ds.createVariable("pr_ca","f4",("time","lat","lon"),
                              zlib=True,complevel=4,fill_value=-9999.)
        vp[:]=pred; vp.units="mm day-1"; vp.long_name="CA precipitation"
        vo=ds.createVariable("pr_obs","f4",("time","lat","lon"),
                              zlib=True,complevel=4,fill_value=-9999.)
        vo[:]=obs; vo.units="mm day-1"; vo.long_name="MSWX truth"


# ── Plots ─────────────────────────────────────────────────────────────────────
def make_plots(obs,pred,library,lat,lon,out_dir,label,config,logger):
    obs_mean=obs.mean(axis=0); pred_mean=pred.mean(axis=0)
    extent=[lon.min(),lon.max(),lat.min(),lat.max()]

    # Mean maps
    bias=pred_mean-obs_mean; vmax_p=np.percentile(obs_mean,99); bmax=np.abs(bias).max()
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,data,title,cmap,vmin,vmax in [
        (axes[0],obs_mean, f"Truth ({label})","YlGnBu",0,vmax_p),
        (axes[1],pred_mean,f"CA ({label})",   "YlGnBu",0,vmax_p),
        (axes[2],bias,     "Bias (CA-Truth)", "RdBu_r",-bmax,bmax),
    ]:
        im=ax.imshow(data,origin="lower",extent=extent,cmap=cmap,
                      vmin=vmin,vmax=vmax,aspect="auto")
        ax.set_title(title); ax.set_xlabel("Lon (E)"); ax.set_ylabel("Lat (N)")
        fig.colorbar(im,ax=ax,label="mm/day",fraction=0.03)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir,f"fig_mean_maps_{label}.png"),
                dpi=150,bbox_inches="tight"); plt.close(fig)
    logger.info(f"  fig_mean_maps_{label}.png")

    # Daily scatter
    obs_ts=obs.mean(axis=(1,2)); pred_ts=pred.mean(axis=(1,2))
    r_ts=float(np.corrcoef(obs_ts,pred_ts)[0,1])
    rmse_ts=float(np.sqrt(np.mean((pred_ts-obs_ts)**2)))
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(obs_ts,pred_ts,alpha=0.25,s=8,color="mediumorchid",rasterized=True)
    vmax=max(obs_ts.max(),pred_ts.max())*1.05
    ax.plot([0,vmax],[0,vmax],"k--",lw=1)
    ax.set_xlim(0,vmax); ax.set_ylim(0,vmax)
    ax.set_xlabel("Truth (mm/day)"); ax.set_ylabel("CA (mm/day)")
    ax.set_title(f"Domain-Mean Daily -- {label}\nr={r_ts:.3f}  RMSE={rmse_ts:.3f}")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir,f"fig_scatter_{label}.png"),
                dpi=150,bbox_inches="tight"); plt.close(fig)
    logger.info(f"  fig_scatter_{label}.png")

    # PSD
    k_obs,psd_obs=azimuthal_psd(obs_mean); k_p,psd_p=azimuthal_psd(pred_mean)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].semilogy(k_obs,psd_obs,color="steelblue",lw=1.5,label="Truth")
    axes[0].semilogy(k_p,psd_p,color="mediumorchid",lw=1.5,ls="--",label="CA")
    axes[0].set_title(f"PSD -- {label}"); axes[0].legend(); axes[0].grid(alpha=0.3)
    ratio=psd_p/(psd_obs+1e-20)
    axes[1].plot(k_obs,ratio,color="mediumorchid",lw=1.5)
    axes[1].axhline(1.0,color="k",ls="--"); axes[1].axhspan(0.9,1.1,alpha=0.1,color="green")
    axes[1].set_ylim(0,2); axes[1].set_title(f"PSD Ratio -- {label}"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir,f"fig_psd_{label}.png"),
                dpi=150,bbox_inches="tight"); plt.close(fig)
    logger.info(f"  fig_psd_{label}.png")

    # CA diagnostic: best-match correlation histogram
    sf=config.scale_factor; P_lr=library.Hc*library.Wc; T=obs.shape[0]
    rng=np.random.default_rng(42)
    sample=rng.choice(T,size=min(200,T),replace=False)
    lc_s=np.zeros((len(sample),P_lr),dtype=np.float32)
    for i,t in enumerate(sample):
        lc_s[i]=lr_coarse(obs[t:t+1],sf)[0].ravel()
    wet_s=lc_s.mean(axis=1)>=config.lr_threshold
    if wet_s.any():
        lc_w=lc_s[wet_s]; lc_n,_,_=spatial_norm(lc_w)
        q_norms=np.linalg.norm(lc_n,axis=1,keepdims=True)+1e-10
        lib_norms=np.linalg.norm(library.lr_norm,axis=1)+1e-10
        corr=(lc_n@library.lr_norm.T)/(q_norms*lib_norms[None,:])
        best=corr.max(axis=1)
        bias2=pred.mean(axis=0)-obs.mean(axis=0); bmax2=np.abs(bias2).max()
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        axes[0].hist(best,bins=30,color="mediumorchid",edgecolor="white",lw=0.5)
        axes[0].axvline(best.mean(),color="k",ls="--",lw=1.5,
                        label=f"Mean={best.mean():.3f}")
        axes[0].axvline(0.9,color="tomato",ls=":",lw=1.5,label="r=0.9")
        axes[0].set_xlabel("Correlation (query LR vs best analogue)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Best-Match Correlation -- {label}\n"
                          f"n={len(best)} wet days | mean={best.mean():.3f} | "
                          f"P10={np.percentile(best,10):.3f}")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
        im=axes[1].imshow(bias2,origin="lower",extent=extent,
                           cmap="RdBu_r",vmin=-bmax2,vmax=bmax2,aspect="auto")
        axes[1].set_title(f"Mean Bias (CA-Truth) -- {label}")
        axes[1].set_xlabel("Lon (E)"); axes[1].set_ylabel("Lat (N)")
        fig.colorbar(im,ax=axes[1],label="mm/day",fraction=0.03)
        plt.suptitle(f"CA Diagnostics -- {label}",y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir,f"fig_ca_diagnostics_{label}.png"),
                    dpi=150,bbox_inches="tight"); plt.close(fig)
        logger.info(f"  fig_ca_diagnostics_{label}.png")


# ── Per-period evaluation ─────────────────────────────────────────────────────
def run_period(h5_path,label,start_year,predictor,library,config,logger,out_dir):
    logger.info(f"\n{'--'*27}"); logger.info(f"PERIOD: {label}"); logger.info(f"{'--'*27}")
    pred,obs,lat,lon=predictor.predict(h5_path,logger)
    T=pred.shape[0]

    nc_path=os.path.join(out_dir,f"ca_pred_{label}.nc")
    save_netcdf(pred,obs,lat,lon,nc_path,config,label)
    logger.info(f"  NetCDF: {nc_path}")

    logger.info("Computing metrics...")
    overall={
        "rmse":round(rmse(obs,pred),4),"mae":round(mae(obs,pred),4),
        "mean_bias":round(mean_bias(obs,pred),4),
        "pearson_r":round(pearson_r(obs,pred),4),"kge":round(kge(obs,pred),4),
        "wet_day_freq_obs":round(wet_day_frequency(obs),4),
        "wet_day_freq_pred":round(wet_day_frequency(pred),4),
    }
    for k,v in overall.items(): logger.info(f"  {k:<22}: {v}")

    obs_pct=percentile_values(obs,config.percentiles)
    pred_pct=percentile_values(pred,config.percentiles)
    pct_skill=percentile_skill(obs_pct,pred_pct)
    for p in config.percentiles:
        logger.info(f"  P{p:5.1f}: obs={obs_pct[p]:.3f}  "
                    f"pred={pred_pct[p]:.3f}  ratio={pct_skill[p]:.3f}")

    slabels=build_season_mask(T,start_year); season_metrics={}
    for s in ["DJF","MAM","JJA","SON"]:
        mask=slabels==s
        if not mask.any(): continue
        o,p=obs[mask],pred[mask]
        season_metrics[s]={
            "n_days":int(mask.sum()),"rmse":round(rmse(o,p),4),
            "mae":round(mae(o,p),4),"bias":round(mean_bias(o,p),4),
            "r":round(pearson_r(o,p),4),"kge":round(kge(o,p),4),
        }
        logger.info(f"  {s}: RMSE={season_metrics[s]['rmse']:.4f}  "
                    f"KGE={season_metrics[s]['kge']:.4f}")

    logger.info("Computing FSS...")
    step=max(1,T//200); obs_s,pred_s=obs[::step],pred[::step]
    fss_results={}
    for thr in config.fss_thresholds:
        vals=[fss_score(obs_s,pred_s,thr,sc) for sc in config.fss_scales]
        fss_results[thr]=vals
        logger.info(f"  FSS thr={thr}: "+"  ".join(f"sc{sc}={v:.3f}"
                    for sc,v in zip(config.fss_scales,vals)))

    k_obs,psd_obs=azimuthal_psd(obs.mean(axis=0))
    k_p,psd_p=azimuthal_psd(pred.mean(axis=0))
    psd_ratio=float(np.median(psd_p[k_p>0.3]/(psd_obs[k_obs>0.3]+1e-20)))
    logger.info(f"  PSD ratio (k>0.3): {psd_ratio:.4f}  "
                "(CA uses real HR days -> better PSD than interpolation)")

    make_plots(obs,pred,library,lat,lon,out_dir,label,config,logger)

    metrics={
        "method":"constructed_analogues","K":config.K,
        "weight_method":"unconstrained_OLS",
        "search_metric":"spatial_correlation_normalised",
        "calibration":"1979-2005","period":label,"n_timesteps":T,
        "overall":overall,
        "percentiles_obs":{str(k):round(v,3) for k,v in obs_pct.items()},
        "percentiles_pred":{str(k):round(v,3) for k,v in pred_pct.items()},
        "percentile_skill":{str(k):round(v,4) for k,v in pct_skill.items()},
        "by_season":season_metrics,
        "fss":{str(thr):{str(sc):round(v,4)
               for sc,v in zip(config.fss_scales,vals)}
               for thr,vals in fss_results.items()},
        "psd_ratio_high_k":round(psd_ratio,4),
    }
    json_path=os.path.join(out_dir,f"metrics_{label}.json")
    with open(json_path,"w") as fj: json.dump(metrics,fj,indent=2)
    logger.info(f"  Metrics: {json_path}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--skip_calib", action="store_true")
    parser.add_argument("--K",          type=int, default=None)
    parser.add_argument("--hr_dtype",   type=str, default=None,
                        choices=["float32","float16"])
    parser.add_argument("--results_dir",type=str,default=None)
    parser.add_argument("--train_h5",   type=str,default=None)
    parser.add_argument("--val_h5",     type=str,default=None)
    parser.add_argument("--test_h5",    type=str,default=None)
    args=parser.parse_args()

    config=Config()
    if args.results_dir: config.results_dir=args.results_dir
    if args.train_h5:    config.train_h5=args.train_h5
    if args.val_h5:      config.val_h5=args.val_h5
    if args.test_h5:     config.test_h5=args.test_h5
    if args.K:           config.K=args.K
    if args.hr_dtype:    config.hr_dtype=args.hr_dtype

    # mkdir BEFORE setup_logging
    out_dir=Path(config.results_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    logger=setup_logging(str(out_dir))

    logger.info("="*60)
    logger.info("Constructed Analogues (CA) -- Val + Test")
    logger.info("Wood et al. (2002) / Hidalgo et al. (2008)")
    logger.info("="*60)
    logger.info(f"K analogues:  {config.K}")
    logger.info(f"HR dtype:     {config.hr_dtype}  (float16 = ~2.4 GB vs ~4.8 GB)")
    logger.info(f"Results:      {out_dir}")

    t_all=time.time()
    lib_path=str(out_dir/"ca_library.npz")

    if args.skip_calib and os.path.exists(lib_path):
        logger.info("\n--- LOADING LIBRARY ---")
        library=CALibrary.load(lib_path,config,logger)
    else:
        logger.info("\n--- BUILDING ANALOGUE LIBRARY ---")
        library=CALibrary(config)
        library.build(config.train_h5,logger)
        library.save(lib_path,logger)

    predictor=CAPredictor(library,config)
    all_metrics={}

    for pname,h5_path,label,start_year in [
        ("val", config.val_h5, "2006-2010", config.val_start_year),
        ("test",config.test_h5,"2011-2014", config.test_start_year),
    ]:
        t0=time.time()
        all_metrics[pname]=run_period(
            h5_path,label,start_year,predictor,library,config,logger,str(out_dir))
        logger.info(f"  {pname} done in {(time.time()-t0)/60:.1f} min")

    combined=str(out_dir/"metrics_combined.json")
    with open(combined,"w") as fj: json.dump(all_metrics,fj,indent=2)

    logger.info("\n"+"="*60+"\nSUMMARY\n"+"="*60)
    for pname,m in all_metrics.items():
        o=m["overall"]
        logger.info(f"{pname} ({m['period']}): "
                    f"RMSE={o['rmse']:.4f}  KGE={o['kge']:.4f}  "
                    f"r={o['pearson_r']:.4f}  PSD={m['psd_ratio_high_k']:.4f}")
    logger.info(f"\nTotal: {(time.time()-t_all)/60:.1f} min")
    logger.info(f"Done. {combined}")

if __name__=="__main__":
    main()