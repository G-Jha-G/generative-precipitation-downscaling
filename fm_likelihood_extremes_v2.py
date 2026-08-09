#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
fm_likelihood_extremes.py
=========================

Likelihood-based characterisation of precipitation extremes for a conditional
Flow-Matching (FM) downscaler, with an optional probability-flow comparison
against a denoising-diffusion (DDPM) model.

WHY THIS EXISTS (manuscript hook G1 / reviewer R3)
--------------------------------------------------
A Flow-Matching model trained with the rectified (optimal-transport) path learns
a velocity field v_theta(x, t, c) whose generative ODE

        dx/dt = v_theta(x, t, c),     x(0) ~ N(0, I)  ->  x(1) ~ p_theta(. | c)

is *exactly* the probability-flow ODE of a continuous normalising flow (CNF).
The same trained network that draws samples therefore also delivers the *exact*
conditional log-density via the instantaneous change-of-variables theorem
(Chen et al., 2018, Neural ODEs; Grathwohl et al., 2018, FFJORD; Lipman et al.,
2023, Flow Matching):

        d/dt log p_t(x(t)) = - tr( d v_theta / d x ).

Integrating this along the learned trajectory from the data point x1 at t = 1 to
the base point x0 at t = 0 gives

        log p_theta(x1 | c) = log N(x0; 0, I)
                              - INT_{t=1}^{0} ( - tr(dv/dx) ) dt.                (1)

This is a capability the deterministic and analogue baselines do not possess at
all (they define no density), and which diffusion provides only approximately
(via an ELBO or its own probability-flow ODE). Exposing it lets us:

  (A) report an exact density benchmark (bits/dim) for FM vs DDPM;
  (B) characterise how the model density behaves across the *extremity* spectrum;
  (C) show the likelihood encodes *physical plausibility* of extreme fields, not
      merely their intensity histogram (a typicality / shuffle test);
  (D) show the model-assigned *rarity* (an NLL percentile) is monotonically
      consistent with the extreme-value (GEV) return level already reported in
      the paper, i.e. the likelihood is a coherent rarity measure.

Together these directly answer "FM is just another generative model": FM uniquely
turns the downscaler into a tractable density model usable for extreme-event
rarity/attribution.

CORRECTNESS
-----------
The CNF likelihood engine is unit-tested against a closed-form case
(`--selftest`): a linear velocity v = lam * x transports N(0, I) to
N(0, e^{2 lam} I), whose density is analytic. The engine must reproduce it, and
the stochastic (Hutchinson) trace must agree with the exact trace.

ZERO-ATOM / QUANTISATION
------------------------
Daily precipitation has a probability atom at 0 and is quantised by the product.
A continuous density is only well defined after *dequantisation*. We add uniform
noise of one quantisation bin in physical space before the log1p transform and
correct the reported bits/dim accordingly (Theis et al., 2016; Ho et al., 2019).
The bin width is a documented parameter (`PrecipConfig.dequant_bin_mm`).

CONVENTIONS HONOURED
--------------------
- Reference product: MSWX 0.1 deg.
- Transform: log1p with a fixed maximum, affine-mapped to [-1, 1] (exactly
  invertible); inverse-transform then clip at zero.
- No cartopy national-boundary overlays are produced anywhere; the analyses here
  are statistical (no maps). If maps are added later, use the GoI-approved
  shapefile, never cfeature.BORDERS.
- NetCDF attribute naming (if ever written): use `downscale_factor`.

References
----------
Chen et al. 2018 (Neural ODEs); Grathwohl et al. 2018 (FFJORD, Hutchinson trace);
Lipman et al. 2023 / Liu et al. 2022 (Flow Matching, rectified flow);
Song et al. 2021 (score SDE / probability-flow ODE, for the DDPM adapter);
Hutchinson 1990 (stochastic trace); Theis et al. 2016, Ho et al. 2019
(dequantisation / bits-per-dim); Nalisnick et al. 2019 (typicality);
Coles 2001 (EVT return levels).

Author: (HIMPACT Lab pipeline). Plug your trained network into `FMVelocity`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence, Tuple

import h5py
import torch.nn.functional as F
try:
    from train_precip_fm_v1 import Config, PrecipNormalizer, PrecipUNet, compute_lr_fullfield
except Exception:
    # Only needed for the real --run inside your training environment.
    Config = PrecipNormalizer = PrecipUNet = compute_lr_fullfield = None

import numpy as np
import torch
from torch import Tensor

try:
    from torchdiffeq import odeint as _odeint
    _HAS_TORCHDIFFEQ = True
except Exception:  # pragma: no cover
    _HAS_TORCHDIFFEQ = False

LOG2 = math.log(2.0)
LOG_2PI = math.log(2.0 * math.pi)


# ======================================================================================
# Configuration
# ======================================================================================
@dataclass
class PrecipConfig:
    """Precipitation transform parameters. Must match the trained model's preprocessing."""
    log1p_max: float = 6.7859          # log1p of the training-set maximum (defines the scale)
    wet_threshold_mm: float = 1.0      # wet-day threshold (mm/day)
    dequant_bin_mm: float = 0.1        # quantisation bin of the product, in mm/day (for dequant)
    clip_min_mm: float = 0.0           # physical lower bound


@dataclass
class LikelihoodConfig:
    """Likelihood-engine settings."""
    method: str = "dopri5"             # "dopri5" (adaptive, recommended) | "rk4" (fixed-step)
    rtol: float = 1e-5                 # adaptive solver relative tolerance
    atol: float = 1e-5                 # adaptive solver absolute tolerance
    n_steps_fixed: int = 100           # steps if method == "rk4"
    hutchinson_probes: int = 4         # number of fixed Rademacher probes (variance reduction)
    exact_trace: bool = False          # exact divergence (only feasible for tiny dims; for tests)
    use_float64: bool = True           # integrate the ODE state in float64 for accuracy
    t0: float = 1.0                    # data time (FM convention: t=1 is data)
    t1: float = 0.0                    # base time (t=0 is the Gaussian prior)
    seed: int = 0


@dataclass
class RunConfig:
    """End-to-end run settings."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1 # <-- REDUCE BATCH SIZE TO 1
    out_dir: str = "./likelihood_outputs"
    precip: PrecipConfig = field(default_factory=PrecipConfig)
    like: LikelihoodConfig = field(default_factory=LikelihoodConfig)


# ======================================================================================
# Precipitation transform (log1p + affine to [-1, 1]) with exact log-Jacobian + dequant
# ======================================================================================
class PrecipTransform:
    r"""
    Maps physical precipitation p >= 0 (mm/day) to the model variable u in [-1, 1]:

        s = log1p(p)                       # log(1 + p)
        u = 2 * s / L - 1,    L = log1p_max

    Inverse:  p = expm1( (u + 1) * L / 2 ).

    The change-of-variables to convert a density in u-space to a density in
    p-space needs the per-element log |du/dp|:

        du/dp = (2 / L) * 1 / (1 + p),
        log|du/dp| = log(2 / L) - log(1 + p) = log(2 / L) - s.                  (2)

    so   log p_phys(p) = log p_model(u) + sum_pixels log|du/dp|.

    Dequantisation: physical data quantised to bin delta is made continuous by
    adding U[0, delta). The continuous-density estimate of the *quantised* model
    is then corrected by + D * log(delta) when converting to a per-bin
    probability, i.e. bits/dim already accounts for delta (see `bits_per_dim`).
    """

    def __init__(self, cfg: PrecipConfig):
        self.cfg = cfg
        self.L = float(cfg.log1p_max)

    # --- forward / inverse -----------------------------------------------------------
    def to_model(self, p_mm: Tensor) -> Tensor:
        s = torch.log1p(p_mm.clamp_min(self.cfg.clip_min_mm))
        return 2.0 * s / self.L - 1.0

    def to_physical(self, u: Tensor) -> Tensor:
        s = (u + 1.0) * self.L / 2.0
        return torch.expm1(s).clamp_min(self.cfg.clip_min_mm)

    # --- log-Jacobian of the u->p change of variables, summed over pixels ------------
    def log_dudp_sum(self, p_mm: Tensor) -> Tensor:
        """sum_pixels log|du/dp| for each sample; shape (B,)."""
        s = torch.log1p(p_mm.clamp_min(self.cfg.clip_min_mm))
        per_elem = math.log(2.0 / self.L) - s          # eq. (2)
        return per_elem.flatten(1).sum(dim=1)

    # --- uniform dequantisation in physical space ------------------------------------
    def dequantize(self, p_mm: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        delta = self.cfg.dequant_bin_mm
        gen_device = generator.device if generator is not None else p_mm.device
        noise = torch.rand(p_mm.shape, generator=generator, device=gen_device,
                           dtype=p_mm.dtype).to(p_mm.device) * delta
        return (p_mm + noise).clamp_min(self.cfg.clip_min_mm)


# ======================================================================================
# Velocity-field protocol and adapters
# ======================================================================================
class VelocityField(Protocol):
    """A conditional velocity field v_theta(x_t, t, cond) -> dx/dt, in MODEL space."""
    def __call__(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor: ...


class FMVelocity:
    r"""
    Adapter wrapping a trained Flow-Matching network so it satisfies `VelocityField`.

    Expected underlying network signature (typical of this pipeline):

        net(x_in, t_scaled) -> velocity

    where
        x_in     = concat([x_t, cond], dim=1)   # LR field concatenated on channels
        t_scaled = t * time_scale               # FM time in [0,1] scaled by 1000
        velocity has the same shape as x_t (HR-state channels only).

    If your network has a different calling convention, override `forward` or pass
    a `call_fn(net, x_in, t_scaled) -> velocity`.
    """

    def __init__(self, net: torch.nn.Module, time_scale: float = 1000.0,
                 concat_cond: bool = True,
                 call_fn: Optional[Callable[[torch.nn.Module, Tensor, Tensor], Tensor]] = None):
        self.net = net
        self.time_scale = float(time_scale)
        self.concat_cond = concat_cond
        self.call_fn = call_fn

    def __call__(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        # The ODE solver passes float64 (double) for precision, 
        # but the network expects float32 (float).
        orig_dtype = x_t.dtype
        net_dtype = next(self.net.parameters()).dtype

        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        
        # Scale time and cast to network's native precision
        t_scaled = (t * self.time_scale).to(net_dtype)
        
        # Concatenate and cast spatial states
        x_in = torch.cat([x_t, cond], dim=1) if self.concat_cond else x_t
        x_in = x_in.to(net_dtype)

        # Forward pass
        if self.call_fn is not None:
            v = self.call_fn(self.net, x_in, t_scaled)
        else:
            v = self.net(x_in, t_scaled)
        
        # Cast the predicted velocity back to the ODE solver's precision
        return v.to(orig_dtype)


class DDPMProbabilityFlowVelocity:
    r"""
    OPTIONAL / EXPERIMENTAL: expose a discrete VP-DDPM (epsilon-prediction) as a
    velocity field in FM time, so the SAME likelihood engine can score it.

    *** VALIDATE THIS AGAINST YOUR OWN DDPM SAMPLER BEFORE TRUSTING THE NUMBERS. ***
    The discrete-to-continuous mapping and the time-direction conventions are easy
    to get subtly wrong; the FM path above is the validated, primary route.

    Setup. A VP diffusion with continuous schedule beta(s), s in [0,1] (s=0 data,
    s=1 noise), has marginals x_s = sqrt(abar(s)) x_0 + sqrt(1-abar(s)) eps with
    abar(s) = exp(-INT_0^s beta). Its probability-flow ODE (Song et al., 2021) is

        dx/ds = f(s) x - 0.5 g(s)^2 score(x, s),
        f(s)  = -0.5 beta(s),   g(s)^2 = beta(s),
        score = - eps_theta(x, s) / sqrt(1 - abar(s)).

    We run the CNF likelihood engine in FM time t in [0,1] with t = 1 - s (so FM
    t=1 == diffusion s=0 == data). With dt = -ds, the FM-time velocity is

        v_FM(x, t) = - dx/ds |_{s = 1 - t}.

    The discrete linear-beta schedule (beta_1..beta_T) is mapped to continuous
    beta(s) = beta_min + s (beta_max - beta_min) with
    beta_min = T*beta_1, beta_max = T*beta_T (so INT_0^1 beta ds matches the
    discrete sum). Adjust if your schedule differs.
    """

    def __init__(self, eps_net: torch.nn.Module, beta_1: float, beta_T: float, T: int,
                 time_scale_steps: bool = True, concat_cond: bool = True):
        self.eps_net = eps_net
        self.T = int(T)
        self.beta_min = float(beta_1) * self.T
        self.beta_max = float(beta_T) * self.T
        self.time_scale_steps = time_scale_steps
        self.concat_cond = concat_cond

    def _beta(self, s: Tensor) -> Tensor:
        return self.beta_min + s * (self.beta_max - self.beta_min)

    def _log_abar(self, s: Tensor) -> Tensor:
        # INT_0^s beta(u) du = beta_min*s + 0.5*(beta_max-beta_min)*s^2
        integral = self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * s * s
        return -integral

    def __call__(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        s = 1.0 - t                                  # diffusion time
        beta = self._beta(s).view(-1, *([1] * (x_t.ndim - 1)))
        abar = torch.exp(self._log_abar(s)).view(-1, *([1] * (x_t.ndim - 1)))
        sigma = torch.sqrt((1.0 - abar).clamp_min(1e-12))
        # network conditioning + discrete timestep index
        if self.time_scale_steps:
            t_in = (s * (self.T - 1)).round()
        else:
            t_in = s
        x_in = torch.cat([x_t, cond], dim=1) if self.concat_cond else x_t
        eps = self.eps_net(x_in, t_in)
        score = -eps / sigma
        dxds = -0.5 * beta * x_t - 0.5 * beta * score   # f x - 0.5 g^2 score, g^2=beta
        v_fm = -dxds                                     # dt = -ds
        return v_fm


# ======================================================================================
# Divergence estimators (trace of the Jacobian of v wrt x)
# ======================================================================================
def _rademacher_like(x: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    gen_device = generator.device if generator is not None else x.device
    r = torch.randint(0, 2, x.shape, generator=generator, device=gen_device,
                      dtype=x.dtype)
    return (r.to(x.device) * 2.0 - 1.0)


def divergence_hutchinson(vel: VelocityField, x: Tensor, t: Tensor, cond: Tensor,
                          eps_list: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    r"""
    Stochastic trace estimate: tr(J) ~ mean_k eps_k^T J eps_k via vector-Jacobian
    products. The SAME eps_list is reused across the whole trajectory (FFJORD), so
    the per-trajectory likelihood estimate is unbiased.
    Returns (v_detached, divergence_estimate[shape B]).
    """
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = vel(x, t, cond)
        div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if not v.requires_grad:           # velocity independent of x => divergence 0
            return v.detach(), div
        for eps in eps_list:
            (vjp,) = torch.autograd.grad(v, x, grad_outputs=eps, retain_graph=True,
                                         create_graph=False, allow_unused=True)
            if vjp is None:
                continue
            div = div + (vjp * eps).flatten(1).sum(dim=1)
        div = div / float(len(eps_list))
    return v.detach(), div.detach()


def divergence_exact(vel: VelocityField, x: Tensor, t: Tensor, cond: Tensor
                     ) -> Tuple[Tensor, Tensor]:
    r"""
    Exact divergence by summing dv_i/dx_i over all dimensions. Cost is O(D) backward
    passes; use ONLY for tiny dimensionality (validation / toy problems).
    """
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = vel(x, t, cond)
        B = x.shape[0]
        D = x[0].numel()
        div = torch.zeros(B, device=x.device, dtype=x.dtype)
        if not v.requires_grad:           # velocity independent of x => divergence 0
            return v.detach(), div
        v_flat = v.flatten(1)
        for i in range(D):
            (grad_i,) = torch.autograd.grad(v_flat[:, i].sum(), x,
                                            retain_graph=True, create_graph=False,
                                            allow_unused=True)
            if grad_i is None:
                continue
            div = div + grad_i.flatten(1)[:, i]
    return v.detach(), div.detach()


# ======================================================================================
# Conditional CNF likelihood engine
# ======================================================================================
class ConditionalCNFLikelihood:
    r"""
    Exact conditional log-density log p_theta(x1 | cond) for a velocity field whose
    generative ODE is the probability-flow ODE (Flow Matching, or a diffusion model
    via `DDPMProbabilityFlowVelocity`).

    Integrates the augmented system  d/dt [x; a] = [v; -div(v)]  from t = t0 (=1,
    data) to t = t1 (=0, base). With  a(t0)=0,  the accumulator satisfies
        a(t1) = log p_base(x0) - log p_data(x1),
    hence
        log p_theta(x1 | cond) = log N(x0; 0, I) - a(t1).        (cf. eq. 1)
    """

    def __init__(self, velocity: VelocityField, cfg: LikelihoodConfig):
        self.velocity = velocity
        self.cfg = cfg
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(cfg.seed)

    # --- base log-density (standard normal) ------------------------------------------
    @staticmethod
    def _base_logprob(x0: Tensor) -> Tensor:
        D = x0[0].numel()
        sq = (x0 ** 2).flatten(1).sum(dim=1)
        return -0.5 * sq - 0.5 * D * LOG_2PI

    # --- augmented dynamics ----------------------------------------------------------
    def _make_dynamics(self, cond: Tensor, eps_list):
        cfg = self.cfg

        def dynamics(t: Tensor, state):
            x, _a = state
            if cfg.exact_trace:
                v, div = divergence_exact(self.velocity, x, t, cond)
            else:
                v, div = divergence_hutchinson(self.velocity, x, t, cond, eps_list)
            return v, -div

        return dynamics

    # --- fixed-step RK4 integrator (fallback, dependency-free) ------------------------
    @staticmethod
    def _rk4(dynamics, state0, t0, t1, n_steps):
        x, a = state0
        ts = torch.linspace(t0, t1, n_steps + 1, device=x.device, dtype=x.dtype)
        for i in range(n_steps):
            t = ts[i]
            h = ts[i + 1] - ts[i]

            def f(tt, st):
                return dynamics(tt, st)

            k1x, k1a = f(t, (x, a))
            k2x, k2a = f(t + 0.5 * h, (x + 0.5 * h * k1x, a + 0.5 * h * k1a))
            k3x, k3a = f(t + 0.5 * h, (x + 0.5 * h * k2x, a + 0.5 * h * k2a))
            k4x, k4a = f(t + h, (x + h * k3x, a + h * k3a))
            x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            a = a + (h / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
        return x, a

    # --- main entry ------------------------------------------------------------------
    @torch.no_grad()
    def log_prob(self, x1: Tensor, cond: Tensor) -> dict:
        """
        Returns a dict with:
          'log_prob'      : log p_theta(x1|cond) in MODEL (u) space, shape (B,)
          'base_log_prob' : log N(x0; 0, I), shape (B,)
          'delta_log_det' : the -a(t1) term (= +INT tr(dv/dx) dt along 1->0), shape (B,)
          'x0'            : the mapped base point, same shape as x1
        """
        cfg = self.cfg
        dtype = torch.float64 if cfg.use_float64 else x1.dtype
        x1 = x1.to(dtype)
        cond = cond.to(dtype)

        # fixed Hutchinson probes for this batch of trajectories
        eps_list = [
            _rademacher_like(x1, self._gen).to(x1.device)
            for _ in range(cfg.hutchinson_probes)
        ]
        a0 = torch.zeros(x1.shape[0], device=x1.device, dtype=dtype)
        dynamics = self._make_dynamics(cond, eps_list)

        if cfg.method == "dopri5" and _HAS_TORCHDIFFEQ:
            t_span = torch.tensor([cfg.t0, cfg.t1], device=x1.device, dtype=dtype)
            sol = _odeint(lambda t, st: dynamics(t, st), (x1, a0), t_span,
                          rtol=cfg.rtol, atol=cfg.atol, method="dopri5")
            xT = sol[0][-1]
            aT = sol[1][-1]
        else:
            if cfg.method == "dopri5" and not _HAS_TORCHDIFFEQ:
                print("[warn] torchdiffeq unavailable; falling back to fixed-step RK4.",
                      file=sys.stderr)
            xT, aT = self._rk4(dynamics, (x1, a0), cfg.t0, cfg.t1, cfg.n_steps_fixed)

        base_lp = self._base_logprob(xT)
        log_prob = base_lp - aT                          # eq. (1)
        return {
            "log_prob": log_prob,
            "base_log_prob": base_lp,
            "delta_log_det": -aT,
            "x0": xT,
        }


# ======================================================================================
# Physical-space bits-per-dimension
# ======================================================================================
def bits_per_dim(engine: ConditionalCNFLikelihood, transform: PrecipTransform,
                 p_hr_mm: Tensor, cond_u: Tensor,
                 dequant_generator: Optional[torch.Generator] = None) -> dict:
    r"""
    Exact conditional bits/dim of physical precipitation fields under the model.

    Pipeline:  p (mm)  --dequant-->  p~  --to_model-->  u  --CNF-->  log p_model(u)
    Convert to physical density via eq. (2), then to per-bin bits/dim:

        log p_phys(p~) = log p_model(u) + sum_pixels log|du/dp|
        bits/dim = - ( log p_phys(p~) + D*log(delta) ) / (D * log 2)

    The + D*log(delta) turns a continuous density into the log-probability of the
    quantisation cell of width delta (Theis et al., 2016), making bits/dim
    comparable across models and to the data entropy.
    """
    cfg = transform.cfg
    D = p_hr_mm[0].numel()
    p_deq = transform.dequantize(p_hr_mm, dequant_generator)
    u = transform.to_model(p_deq)
    out = engine.log_prob(u, cond_u)
    lp_model = out["log_prob"].to(torch.float64)
    log_jac = transform.log_dudp_sum(p_deq).to(torch.float64)        # u -> p
    lp_phys = lp_model + log_jac
    # per-bin log-prob -> bits/dim
    bpd = -(lp_phys + D * math.log(cfg.dequant_bin_mm)) / (D * LOG2)
    return {
        "bits_per_dim": bpd,                       # (B,)
        "log_prob_model": lp_model,
        "log_prob_phys": lp_phys,
        "nats_per_dim_phys": -lp_phys / D,
    }


# ======================================================================================
# Extremity index and the three extreme-focused analyses
# ======================================================================================
def extremity_index(p_hr_mm: Tensor, cfg: PrecipConfig, q: float = 0.999) -> Tensor:
    r"""
    A scalar 'how extreme is this field' score per sample: the high wet-day quantile
    of the field (default 99.9th percentile of wet pixels). Robust and monotone in
    the heavy-rain content. Returns shape (B,).
    """
    B = p_hr_mm.shape[0]
    flat = p_hr_mm.flatten(1)
    out = torch.empty(B, dtype=torch.float64, device=p_hr_mm.device)
    for b in range(B):
        wet = flat[b][flat[b] >= cfg.wet_threshold_mm]
        if wet.numel() < 10:
            out[b] = float(flat[b].max())
        else:
            out[b] = torch.quantile(wet.to(torch.float64), q)
    return out


def likelihood_vs_extremity(engine, transform, p_hr_mm, cond_u, batch_size=8,
                            device="cpu", dequant_gen=None):
    r"""
    ANALYSIS (B): per-event bits/dim vs an extremity index. Produces the arrays for
    the headline figure and a rank correlation. A well-behaved density model should
    assign *coherently higher* information content (bits/dim) to more extreme fields
    without pathological collapse.
    """
    n = p_hr_mm.shape[0]
    bpd_all, ext_all = [], []
    for i in range(0, n, batch_size):
        p = p_hr_mm[i:i + batch_size].to(device)
        c = cond_u[i:i + batch_size].to(device)
        bpd = bits_per_dim(engine, transform, p, c, dequant_gen)["bits_per_dim"]
        ext = extremity_index(p, transform.cfg)
        bpd_all.append(bpd.cpu())
        ext_all.append(ext.cpu())
    bpd = torch.cat(bpd_all).numpy()
    ext = torch.cat(ext_all).numpy()
    rho = _spearman(ext, bpd)
    return {"extremity": ext, "bits_per_dim": bpd, "spearman_rho": rho}


def typicality_test(engine, transform, p_hr_mm, cond_u, batch_size=8, device="cpu",
                    dequant_gen=None, shuffle_gen=None):
    r"""
    ANALYSIS (C): does the likelihood encode *physical plausibility*, or only the
    intensity histogram? For each field we build a 'spatial-shuffle' surrogate that
    permutes pixel intensities within the field. The surrogate has an identical
    one-point intensity distribution (identical histogram, identical extremes) but
    destroyed spatial structure. A physically meaningful density must assign the
    real field HIGHER likelihood (LOWER bits/dim) than its shuffle. We report both
    distributions and a separation statistic (AUC and Cohen's d).
    """
    if shuffle_gen is None:
        shuffle_gen = torch.Generator(device="cpu")
        shuffle_gen.manual_seed(1234)

    def _shuffle(p):
        B = p.shape[0]
        flat = p.flatten(1)
        out = torch.empty_like(flat)
        for b in range(B):
            perm = torch.randperm(flat.shape[1], generator=shuffle_gen).to(p.device)
            out[b] = flat[b][perm]
        return out.view_as(p)

    n = p_hr_mm.shape[0]
    bpd_real, bpd_fake = [], []
    for i in range(0, n, batch_size):
        p = p_hr_mm[i:i + batch_size].to(device)
        c = cond_u[i:i + batch_size].to(device)
        ps = _shuffle(p)
        bpd_real.append(bits_per_dim(engine, transform, p, c, dequant_gen)["bits_per_dim"].cpu())
        bpd_fake.append(bits_per_dim(engine, transform, ps, c, dequant_gen)["bits_per_dim"].cpu())
    real = torch.cat(bpd_real).numpy()
    fake = torch.cat(bpd_fake).numpy()
    auc = _auc_real_lower(real, fake)               # P(real bits/dim < shuffled)
    d = _cohens_d(fake, real)                        # >0 means shuffled is larger (worse)
    return {"bpd_real": real, "bpd_shuffled": fake, "auc_real_more_likely": auc,
            "cohens_d": d, "frac_real_more_likely": float(np.mean(real < fake))}


def _derangement(n: int, gen: torch.Generator) -> torch.Tensor:
    """A permutation of 0..n-1 with NO fixed point (every field gets a different
    day's conditioning). Random permutation, then repair any fixed points by a swap."""
    perm = torch.randperm(n, generator=gen)
    fixed = (perm == torch.arange(n)).nonzero(as_tuple=True)[0]
    for i in fixed.tolist():
        j = (i + 1) % n
        perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    # final safety: if n==1 a derangement is impossible
    return perm


def conditioning_mismatch_test(engine, transform, p_hr_mm, cond_u, batch_size=8,
                               device="cpu", dequant_gen=None, mismatch_gen=None):
    r"""
    STRICTER, downscaling-relevant typicality test (complements the spatial shuffle).
    For each high-resolution field Y_t we score its exact conditional bits/dim under
    (i) its TRUE coarse input X_t and (ii) a MISMATCHED coarse input X_{t'} taken from
    a different day (a derangement, so no field keeps its own conditioner). The HR
    field, its intensity histogram, and its spatial structure are IDENTICAL in both;
    only the conditioning differs. A conditional density that actually uses its
    predictor must assign the matched pair higher likelihood (lower bits/dim). This
    tests the specificity of p_theta(Y | X), which is the entire point of downscaling,
    and is far harder than the white-noise spatial shuffle.
    Returns matched/mismatched bits/dim distributions and the same separation stats.
    """
    if mismatch_gen is None:
        mismatch_gen = torch.Generator(device="cpu"); mismatch_gen.manual_seed(202)
    n = p_hr_mm.shape[0]
    perm = _derangement(n, mismatch_gen)
    cond_mis = cond_u[perm]
    bpd_match, bpd_mis = [], []
    for i in range(0, n, batch_size):
        p = p_hr_mm[i:i + batch_size].to(device)
        c_match = cond_u[i:i + batch_size].to(device)
        c_mis = cond_mis[i:i + batch_size].to(device)
        bpd_match.append(bits_per_dim(engine, transform, p, c_match, dequant_gen)["bits_per_dim"].cpu())
        bpd_mis.append(bits_per_dim(engine, transform, p, c_mis, dequant_gen)["bits_per_dim"].cpu())
    match = torch.cat(bpd_match).numpy()
    mis = torch.cat(bpd_mis).numpy()
    auc = _auc_real_lower(match, mis)               # P(matched bits/dim < mismatched)
    d = _cohens_d(mis, match)                         # >0 means mismatched is larger (worse)
    return {"bpd_matched": match, "bpd_mismatched": mis,
            "auc_matched_more_likely": auc, "cohens_d": d,
            "frac_matched_more_likely": float(np.mean(match < mis))}


def rarity_vs_return_level(engine, transform, p_hr_mm, cond_u, return_level_mm,
                           batch_size=8, device="cpu", dequant_gen=None):
    r"""
    ANALYSIS (D): is the model-assigned rarity consistent with extreme-value theory?
    For each (typically annual-maximum) field we compute the model NLL (bits/dim) and
    convert it to a within-sample rarity percentile; we pair it with the physical
    return level / block-maximum (`return_level_mm`, e.g. the field maximum, or the
    GEV return level already used in Table 4 / Fig 7). A monotone increasing relation
    means the likelihood is a coherent rarity measure aligned with EVT.
    """
    n = p_hr_mm.shape[0]
    bpd_all = []
    for i in range(0, n, batch_size):
        p = p_hr_mm[i:i + batch_size].to(device)
        c = cond_u[i:i + batch_size].to(device)
        bpd_all.append(bits_per_dim(engine, transform, p, c, dequant_gen)["bits_per_dim"].cpu())
    bpd = torch.cat(bpd_all).numpy()
    rl = np.asarray(return_level_mm, dtype=np.float64).reshape(-1)
    rarity_pct = _to_percentile(bpd)                 # higher bits/dim -> rarer
    rho = _spearman(rl, bpd)
    return {"bits_per_dim": bpd, "return_level_mm": rl, "rarity_percentile": rarity_pct,
            "spearman_rho": rho}


# ======================================================================================
# Small statistics helpers (no SciPy dependency)
# ======================================================================================
def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _auc_real_lower(real: np.ndarray, fake: np.ndarray) -> float:
    """P(real bits/dim < fake bits/dim) over all pairs (Mann-Whitney style)."""
    r = np.asarray(real); f = np.asarray(fake)
    wins = 0.0; tot = len(r) * len(f)
    for v in r:
        wins += np.sum(v < f) + 0.5 * np.sum(v == f)
    return float(wins / tot) if tot else float("nan")


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def _to_percentile(a: np.ndarray) -> np.ndarray:
    return 100.0 * (_rankdata(a) - 1) / max(len(a) - 1, 1)


# ======================================================================================
# Output plumbing
# ======================================================================================
def save_results(out_dir: str, name: str, arrays: dict, scalars: dict):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{name}.npz"),
             **{k: np.asarray(v) for k, v in arrays.items()})
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(scalars, fh, indent=2)


def make_figure(out_dir: str, lve: dict, typ: dict, rvr: Optional[dict],
                cmm: Optional[dict] = None):
    """Manuscript-ready figure (statistical plots only; no maps). 3 panels, or 2x2
    when the conditioning-mismatch test `cmm` is supplied."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib unavailable ({e}); skipping figure.", file=sys.stderr)
        return None

    os.makedirs(out_dir, exist_ok=True)
    if cmm is not None:
        fig, axg = plt.subplots(2, 2, figsize=(9.0, 7.2))
        ax = axg.ravel()
    else:
        ncol = 3 if rvr is not None else 2
        fig, ax = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.6))

    a0 = ax[0]
    a0.scatter(lve["extremity"], lve["bits_per_dim"], s=14, alpha=0.6,
               edgecolor="none")
    a0.set_xlabel("field 99.9th-pct intensity (mm day$^{-1}$)")
    a0.set_ylabel("bits / dim (lower = more probable)")
    a0.set_title(f"(a) likelihood vs extremity\nSpearman $\\rho$={lve['spearman_rho']:.2f}")

    a1 = ax[1]
    bins = np.linspace(min(typ["bpd_real"].min(), typ["bpd_shuffled"].min()),
                       max(typ["bpd_real"].max(), typ["bpd_shuffled"].max()), 30)
    a1.hist(typ["bpd_real"], bins=bins, alpha=0.6, label="observed", density=True)
    a1.hist(typ["bpd_shuffled"], bins=bins, alpha=0.6, label="spatial shuffle", density=True)
    a1.set_xlabel("bits / dim")
    a1.set_ylabel("density")
    a1.set_title(f"(b) typicality: spatial shuffle\nAUC={typ['auc_real_more_likely']:.2f}, "
                 f"$d$={typ['cohens_d']:.2f}")
    a1.legend(frameon=False, fontsize=8)

    if rvr is not None:
        a2 = ax[2]
        a2.scatter(rvr["return_level_mm"], rvr["bits_per_dim"], s=14, alpha=0.6,
                   edgecolor="none")
        a2.set_xlabel("block-maximum / return level (mm day$^{-1}$)")
        a2.set_ylabel("bits / dim")
        a2.set_title(f"(c) rarity vs EVT\nSpearman $\\rho$={rvr['spearman_rho']:.2f}")

    if cmm is not None:
        a3 = ax[3]
        bins2 = np.linspace(min(cmm["bpd_matched"].min(), cmm["bpd_mismatched"].min()),
                            max(cmm["bpd_matched"].max(), cmm["bpd_mismatched"].max()), 30)
        a3.hist(cmm["bpd_matched"], bins=bins2, alpha=0.6, label="matched X", density=True)
        a3.hist(cmm["bpd_mismatched"], bins=bins2, alpha=0.6, label="mismatched X", density=True)
        a3.set_xlabel("bits / dim")
        a3.set_ylabel("density")
        a3.set_title(f"(d) typicality: conditioning mismatch\n"
                     f"AUC={cmm['auc_matched_more_likely']:.2f}, $d$={cmm['cohens_d']:.2f}")
        a3.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_likelihood_extremes.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ======================================================================================
# Data loading (adapt to your store)
# ======================================================================================
def load_test_fields(cfg: RunConfig) -> Tuple[Tensor, Tensor]:
    """
    Loads test data from MSWX H5, computes LR, pads both to model space, and normalizes.
    """
    cfg_train = Config()
    normalizer = PrecipNormalizer(cfg_train.log1p_max)
    
    # Update this path to point exactly to your MSWX test file
    input_h5 = "/path/to/project/PhD_Precipitation/02_Data/processed/hdf5/mswx_test_2011-2014.h5" 

    with h5py.File(input_h5, 'r') as f:
        # Use a stride of 10 to get ~146 seasonally representative days
        # across the entire 2011-2014 test period.
        hr = f['precipitation'][::10].astype(np.float32)

    T, H, W = hr.shape
    
    # Calculate padding to ensure dimensions are divisible by 16 (for the UNet)
    div = 16
    ph = (div - H % div) % div; pw = (div - W % div) % div
    pt, pl = ph // 2, pw // 2
    pb, pr = ph - pt, pw - pl

    # Generate the Low-Res conditioning field
    lr = compute_lr_fullfield(hr, cfg_train.scale_factor, H, W)

    # Convert to PyTorch tensors and add the channel dimension: (B, C, H, W)
    p_hr_mm = torch.from_numpy(hr).unsqueeze(1) 
    lr_t = torch.from_numpy(lr).unsqueeze(1)    

    # --- THE FIX: Pad BOTH the HR field and LR field to 352x352 ---
    p_hr_mm = F.pad(p_hr_mm, (pl, pr, pt, pb), mode='reflect')
    lr_t = F.pad(lr_t, (pl, pr, pt, pb), mode='reflect')
    
    # Normalize the conditioning field
    cond_u = normalizer.normalize_torch(lr_t)

    return p_hr_mm, cond_u


# ======================================================================================
# Self-test: validate the CNF likelihood against a closed-form Gaussian
# ======================================================================================
class _LinearVelocity:
    """v(x,t) = lam * x (constant in t, ignores cond). Pushforward of N(0,I) at t=0
    to t=1 is N(0, e^{2 lam} I): an analytic target for validation."""
    def __init__(self, lam: float):
        self.lam = lam

    def __call__(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        return self.lam * x


class _ShiftVelocity:
    """v(x,t) = c (divergence 0). Pushforward N(0,I) -> N(c, I)."""
    def __init__(self, c: float):
        self.c = c

    def __call__(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        return torch.full_like(x, self.c)


def _gaussian_logprob(x: Tensor, mean: float, var: float) -> Tensor:
    D = x[0].numel()
    sq = ((x - mean) ** 2).flatten(1).sum(dim=1)
    return -0.5 * sq / var - 0.5 * D * math.log(var) - 0.5 * D * LOG_2PI


def run_selftest(device: str = "cpu") -> bool:
    torch.manual_seed(0)
    ok = True
    D, B = 6, 64
    dummy_cond = torch.zeros(B, 1, device=device)

    print("=" * 78)
    print("SELF-TEST 1: linear velocity  v=lam*x   (target N(0, e^{2lam} I))")
    for lam in (-0.6, 0.4):
        x1 = torch.randn(B, D, device=device, dtype=torch.float64) * math.exp(lam)
        analytic = _gaussian_logprob(x1, 0.0, math.exp(2 * lam))

        # exact-trace engine (reference)
        eng_exact = ConditionalCNFLikelihood(
            _LinearVelocity(lam),
            LikelihoodConfig(method="dopri5", exact_trace=True, use_float64=True,
                             rtol=1e-7, atol=1e-7))
        lp_exact = eng_exact.log_prob(x1, dummy_cond)["log_prob"]

        # Hutchinson engine (production path)
        eng_hutch = ConditionalCNFLikelihood(
            _LinearVelocity(lam),
            LikelihoodConfig(method="dopri5", exact_trace=False, hutchinson_probes=8,
                             use_float64=True, rtol=1e-6, atol=1e-6, seed=0))
        lp_hutch = eng_hutch.log_prob(x1, dummy_cond)["log_prob"]

        err_exact = (lp_exact - analytic).abs().max().item()
        err_hutch_mean = (lp_hutch.mean() - analytic.mean()).abs().item()
        rel = err_exact / analytic.abs().mean().item()
        print(f"  lam={lam:+.2f} | max|exact-analytic|={err_exact:.3e} "
              f"(rel {rel:.2e}) | |meanHutch-meanAnalytic|={err_hutch_mean:.3e}")
        ok &= (err_exact < 1e-3 * max(1.0, analytic.abs().mean().item()))
        ok &= (err_hutch_mean < 0.5)  # stochastic, mean should be close

    print("SELF-TEST 2: shift velocity   v=c       (target N(c, I), divergence 0)")
    c = 0.7
    x1 = torch.randn(B, D, device=device, dtype=torch.float64) + c
    analytic = _gaussian_logprob(x1, c, 1.0)
    eng = ConditionalCNFLikelihood(
        _ShiftVelocity(c),
        LikelihoodConfig(method="dopri5", exact_trace=True, use_float64=True,
                         rtol=1e-7, atol=1e-7))
    lp = eng.log_prob(x1, dummy_cond)["log_prob"]
    err = (lp - analytic).abs().max().item()
    print(f"  c={c:+.2f} | max|cnf-analytic|={err:.3e}")
    ok &= (err < 1e-4 * max(1.0, analytic.abs().mean().item()))

    print("SELF-TEST 3: RK4 fixed-step agrees with adaptive dopri5 (linear v)")
    lam = 0.4
    x1 = torch.randn(B, D, device=device, dtype=torch.float64) * math.exp(lam)
    eng_rk4 = ConditionalCNFLikelihood(
        _LinearVelocity(lam),
        LikelihoodConfig(method="rk4", exact_trace=True, n_steps_fixed=200,
                         use_float64=True))
    eng_dp = ConditionalCNFLikelihood(
        _LinearVelocity(lam),
        LikelihoodConfig(method="dopri5", exact_trace=True, use_float64=True,
                         rtol=1e-7, atol=1e-7))
    d = (eng_rk4.log_prob(x1, dummy_cond)["log_prob"]
         - eng_dp.log_prob(x1, dummy_cond)["log_prob"]).abs().max().item()
    print(f"  max|rk4-dopri5|={d:.3e}")
    ok &= (d < 1e-3 * max(1.0, x1.abs().mean().item()) + 1e-3)

    print("SELF-TEST 4: transform log-Jacobian via finite differences")
    cfg = PrecipConfig(log1p_max=6.7859)
    tr = PrecipTransform(cfg)
    p = torch.rand(4, 1, 3, 3, dtype=torch.float64) * 50.0
    eps = 1e-6
    u1 = tr.to_model(p); u2 = tr.to_model(p + eps)
    fd = torch.log((u2 - u1).abs() / eps).flatten(1).sum(1)
    ana = tr.log_dudp_sum(p)
    errj = (fd - ana).abs().max().item()
    print(f"  max|fd-analytic logJac|={errj:.3e}")
    ok &= (errj < 1e-4)

    print("=" * 78)
    print(f"SELF-TEST {'PASSED' if ok else 'FAILED'}")
    print("=" * 78)
    return ok


# ======================================================================================
# Orchestration
# ======================================================================================
def run_full(cfg: RunConfig):
    torch.manual_seed(cfg.like.seed)
    device = cfg.device
    transform = PrecipTransform(cfg.precip)

    print(f"Running on device: {device}")

    # Load architecture hyperparameters from your training config
    cfg_train = Config()

    # 1. Instantiate the model
    model = PrecipUNet(
        in_channels=2, 
        out_channels=1, 
        base_dim=cfg_train.base_dim,
        ch_mult_cap=cfg_train.ch_mult_cap, 
        time_dim=cfg_train.time_dim
    ).to(device)

    # 2. Load the checkpoint (Update this path to your best EMA checkpoint)
    checkpoint_path = "/path/to/project/Flow_matching_downscaling/results/precip_fm_v1/best.pt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()

    # 3. Wrap the trained network in the FMVelocity adapter
    velocity = FMVelocity(model, time_scale=cfg_train.t_scale)
    engine = ConditionalCNFLikelihood(velocity, cfg.like)

    # 4. Load the test tensors
    print("Loading test fields...")
    p_hr_mm, cond_u = load_test_fields(cfg)
    dq = torch.Generator(device="cpu"); dq.manual_seed(0)
    
    # 5. Run the Likelihood Analyses
    print("Calculating Bits per Dimension (BPD)...")
    bpd = bits_per_dim(engine, transform, p_hr_mm[:cfg.batch_size].to(device),
                       cond_u[:cfg.batch_size].to(device), dq)["bits_per_dim"]
    
    print("Running Analysis B: Likelihood vs Extremity...")
    lve = likelihood_vs_extremity(engine, transform, p_hr_mm, cond_u,
                                  cfg.batch_size, device, dq)
    
    print("Running Analysis C: Typicality Test (Spatial Shuffle)...")
    typ = typicality_test(engine, transform, p_hr_mm, cond_u,
                          cfg.batch_size, device, dq)
    
    print("Running Analysis D: Rarity vs Return Level...")
    block_max = p_hr_mm.flatten(1).max(1).values.cpu().numpy()
    rvr = rarity_vs_return_level(engine, transform, p_hr_mm, cond_u, block_max,
                                 cfg.batch_size, device, dq)

    print("Running Analysis E: Typicality Test (Conditioning Mismatch)...")
    mm = torch.Generator(device="cpu"); mm.manual_seed(cfg.like.seed + 202)
    cmm = conditioning_mismatch_test(engine, transform, p_hr_mm, cond_u,
                                     cfg.batch_size, device, dq, mismatch_gen=mm)

    print("Saving results and generating figures...")
    save_results(cfg.out_dir, "likelihood_extremes",
                 {"bpd_test_mean": bpd.cpu().numpy(),
                  "lve_extremity": lve["extremity"], "lve_bpd": lve["bits_per_dim"],
                  "typ_real": typ["bpd_real"], "typ_shuf": typ["bpd_shuffled"],
                  "cmm_matched": cmm["bpd_matched"], "cmm_mismatched": cmm["bpd_mismatched"],
                  "rvr_rl": rvr["return_level_mm"], "rvr_bpd": rvr["bits_per_dim"]},
                 {"bpd_mean": float(lve["bits_per_dim"].mean()),
                  "bpd_sem_over_fields": float(lve["bits_per_dim"].std(ddof=1)
                                               / max(len(lve["bits_per_dim"]) ** 0.5, 1.0)),
                  "n_fields": int(len(lve["bits_per_dim"])),
                  "hutchinson_probes": cfg.like.hutchinson_probes,
                  "seed": cfg.like.seed,
                  "lve_spearman": lve["spearman_rho"],
                  "typ_auc": typ["auc_real_more_likely"],
                  "typ_cohens_d": typ["cohens_d"],
                  "cmm_auc": cmm["auc_matched_more_likely"],
                  "cmm_cohens_d": cmm["cohens_d"],
                  "cmm_frac_matched_more_likely": cmm["frac_matched_more_likely"],
                  "rvr_spearman": rvr["spearman_rho"],
                  "config": dataclasses.asdict(cfg)})

    make_figure(cfg.out_dir, lve, typ, rvr, cmm)
    print("Analysis Complete!")


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="Validate the likelihood engine against closed-form densities and exit.")
    p.add_argument("--run", action="store_true",
                   help="Run the full extreme-likelihood analysis (requires wiring your model/data).")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="./likelihood_outputs")
    p.add_argument("--probes", type=int, default=8,
                   help="Hutchinson trace probes; use 8-16 for publication-quality bits/dim.")
    p.add_argument("--method", default="dopri5", choices=["dopri5", "rk4"])
    p.add_argument("--seed", type=int, default=0,
                   help="Hutchinson/derangement seed; vary it to check Monte-Carlo convergence.")
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    if args.selftest:
        ok = run_selftest(device=args.device or "cpu")
        sys.exit(0 if ok else 1)
    if args.run:
        cfg = RunConfig(out_dir=args.out_dir)
        if args.device:
            cfg.device = args.device
        cfg.like.hutchinson_probes = args.probes
        cfg.like.method = args.method
        cfg.like.seed = args.seed
        run_full(cfg)
    else:
        _build_argparser().print_help()


if __name__ == "__main__":
    main()