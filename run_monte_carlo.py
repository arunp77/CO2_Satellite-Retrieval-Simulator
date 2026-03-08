"""
run_monte_carlo.py
==================
Standalone Monte Carlo simulation script for the CO₂ Retrieval Simulator.

Runs all three simulation modes end-to-end and saves all diagnostic figures:

    Mode 1 — Noise Ensemble
        Fix atmospheric scene; repeat with N independent noise realisations.
        → Retrieval precision (random error characterisation)

    Mode 2 — XCO₂ Sweep
        Sweep true XCO₂ from 380 → 480 ppm; retrieve each.
        → Retrieval linearity, bias, true-vs-retrieved scatter

    Mode 3 — Random Scene Ensemble
        Draw all geophysical + instrument parameters randomly.
        → End-to-end mission performance across realistic scene diversity

Output figures (saved to ./mc_figures/):
    mc_01_noise_ensemble.png         — error histogram + posterior uncertainty
    mc_02_xco2_sweep.png             — true vs retrieved + error vs XCO₂
    mc_03_random_scene.png           — 6-panel diagnostic
    mc_04_snr_sensitivity.png        — precision/bias vs SNR
    mc_05_sza_sensitivity.png        — precision vs solar zenith angle
    mc_06_bias_precision_summary.png — summary comparison across all modes

Usage
-----
    python run_monte_carlo.py [--n_noise N] [--n_sweep N] [--n_random N] [--seed S]

All arguments are optional (defaults give a fast but statistically meaningful run).

Dependencies
------------
    numpy, scipy, matplotlib  (standard scientific Python stack)
    src/  directory must be in the same folder or on PYTHONPATH

Author: Arun Kumar Pandey
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — works whether run from project root or from the script's dir
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from hitran_model        import hitran_cross_section, SYNTHETIC_CO2_LINES
from radiative_transfer  import (
    forward_model, standard_atmosphere_profile,
    column_amount, air_mass_factor,
)
from retrieval           import iterative_retrieval, xco2_from_scaling

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(SCRIPT_DIR, "mc_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#d0d4e8",
    "axes.titlecolor":   "#ffffff",
    "xtick.color":       "#8a8ea8",
    "ytick.color":       "#8a8ea8",
    "text.color":        "#d0d4e8",
    "grid.color":        "#2a2d3d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#3a3d4d",
    "figure.dpi":        130,
    "font.size":         10,
})

PALETTE = {
    "blue":   "#4a9eff",
    "teal":   "#2dd4bf",
    "orange": "#fb923c",
    "pink":   "#f472b6",
    "green":  "#4ade80",
    "purple": "#a78bfa",
    "yellow": "#fbbf24",
    "red":    "#f87171",
}


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class MCResult:
    """Structured container for one Monte Carlo experiment."""
    mode: str
    xco2_true:       np.ndarray
    xco2_retrieved:  np.ndarray
    xco2_unc:        np.ndarray
    albedo_true:     np.ndarray
    sza_true:        np.ndarray
    snr_true:        np.ndarray
    n_iter:          np.ndarray
    converged:       np.ndarray

    @property
    def error(self)     -> np.ndarray: return self.xco2_retrieved - self.xco2_true
    @property
    def bias(self)      -> float:      return float(np.nanmean(self.error))
    @property
    def precision(self) -> float:      return float(np.nanstd(self.error))
    @property
    def rmse(self)      -> float:      return float(np.sqrt(np.nanmean(self.error**2)))
    @property
    def n_samples(self) -> int:        return int(np.sum(self.converged))

    def print_summary(self):
        bar = "─" * 46
        print(f"\n  ┌{bar}┐")
        print(f"  │  Mode      : {self.mode:<30}│")
        print(f"  │  N samples : {self.n_samples:<30}│")
        print(f"  │  Bias      : {self.bias:+.3f} ppm{'':<24}│")
        print(f"  │  Precision : {self.precision:.3f} ppm{'':<24}│")
        print(f"  │  RMSE      : {self.rmse:.3f} ppm{'':<24}│")
        print(f"  │  Mean σ    : {np.nanmean(self.xco2_unc):.3f} ppm{'':<24}│")
        print(f"  └{bar}┘")


# ===========================================================================
# Forward model helpers (defined once, shared everywhere)
# ===========================================================================

def _make_fwd_functions(nu: np.ndarray):
    """
    Return xsec_func and N_col_func closures for the given wavenumber grid.
    """
    def xsec_func(P_Pa: float, T_K: float) -> np.ndarray:
        return hitran_cross_section(nu, SYNTHETIC_CO2_LINES, P_Pa, T_K)

    def N_col_func(P_Pa: float, T_K: float, xco2_ppm: float) -> float:
        profile = standard_atmosphere_profile(
            P_surface_Pa=P_Pa,
            T_surface_K=T_K,
            co2_vmr_ppm=xco2_ppm,
        )
        return column_amount(profile["number_density"], profile["layer_thickness_cm"])

    return xsec_func, N_col_func


def _run_single(
    nu:              np.ndarray,
    xsec:            np.ndarray,
    N_col_true:      float,
    N_col_prior:     float,
    amf:             float,
    albedo:          float,
    sza:             float,
    snr:             float,
    xco2_prior_ppm:  float,
    sa:              float,
    max_iter:        int,
    rng:             np.random.Generator,
) -> Tuple[float, float, int, bool]:
    """
    Run one forward-model + noise + retrieval cycle.

    Returns
    -------
    xco2_ret, xco2_unc, n_iter, converged
    """
    try:
        # True (noiseless) radiance
        I_true = forward_model(nu, xsec, N_col_true, amf,
                               surface_albedo=albedo,
                               solar_zenith_deg=sza)["radiance"]

        # Instrument noise
        sigma_noise = I_true.max() / snr
        y_obs = I_true + rng.normal(0.0, sigma_noise, size=len(nu))

        # Forward function for retrieval (xi scales the prior column)
        def fwd(xi, _xs=xsec, _Np=N_col_prior, _amf=amf, _alb=albedo, _sza=sza):
            return forward_model(nu, _xs, _Np * xi, _amf,
                                 surface_albedo=_alb,
                                 solar_zenith_deg=_sza)["radiance"]

        res = iterative_retrieval(
            nu, y_obs, fwd,
            xa=1.0, sa=sa,
            sigma_noise=sigma_noise,
            max_iter=max_iter,
        )
        xco2_ret, xco2_unc = xco2_from_scaling(
            res["xi_hat"], xco2_prior_ppm, res["sigma_post"])
        return xco2_ret, xco2_unc, int(res["n_iter"]), True

    except Exception:
        return np.nan, np.nan, -1, False


# ===========================================================================
# Monte Carlo runners
# ===========================================================================

def run_noise_ensemble(
    nu:             np.ndarray,
    xsec_func:      Callable,
    N_col_func:     Callable,
    n_samples:      int   = 400,
    xco2_true_ppm:  float = 430.0,
    xco2_prior_ppm: float = 420.0,
    albedo:         float = 0.25,
    sza:            float = 30.0,
    snr:            float = 250.0,
    sa:             float = 0.10,
    max_iter:       int   = 15,
    seed:           int   = 42,
) -> MCResult:
    """
    Noise ensemble: fixed scene, N independent noise realisations.
    """
    print(f"\n{'='*60}")
    print(f"  MODE 1 — Noise Ensemble   (N={n_samples})")
    print(f"  True XCO₂={xco2_true_ppm} ppm  SNR={snr}  SZA={sza}°  A={albedo}")
    print(f"{'='*60}")

    rng  = np.random.default_rng(seed)
    xsec = xsec_func(101325.0, 288.0)

    N_col_true  = N_col_func(101325.0, 288.0, xco2_true_ppm)
    N_col_prior = N_col_func(101325.0, 288.0, xco2_prior_ppm)
    amf         = air_mass_factor(sza, 0.0)

    xco2_ret_arr  = np.full(n_samples, np.nan)
    xco2_unc_arr  = np.full(n_samples, np.nan)
    n_iter_arr    = np.full(n_samples, -1, dtype=int)
    converged_arr = np.zeros(n_samples, dtype=bool)

    t0 = time.time()
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1}/{n_samples}  ({time.time()-t0:.1f}s elapsed)")
        r = _run_single(nu, xsec, N_col_true, N_col_prior, amf,
                        albedo, sza, snr, xco2_prior_ppm, sa, max_iter, rng)
        xco2_ret_arr[i], xco2_unc_arr[i], n_iter_arr[i], converged_arr[i] = r

    valid = converged_arr & np.isfinite(xco2_ret_arr)
    print(f"  Done. {valid.sum()}/{n_samples} converged in {time.time()-t0:.1f}s")

    result = MCResult(
        mode           = "Noise Ensemble",
        xco2_true      = np.full(valid.sum(), xco2_true_ppm),
        xco2_retrieved = xco2_ret_arr[valid],
        xco2_unc       = xco2_unc_arr[valid],
        albedo_true    = np.full(valid.sum(), albedo),
        sza_true       = np.full(valid.sum(), sza),
        snr_true       = np.full(valid.sum(), snr),
        n_iter         = n_iter_arr[valid],
        converged      = converged_arr[valid],
    )
    result.print_summary()
    return result


def run_xco2_sweep(
    nu:             np.ndarray,
    xsec_func:      Callable,
    N_col_func:     Callable,
    n_samples:      int   = 100,
    xco2_min_ppm:   float = 380.0,
    xco2_max_ppm:   float = 480.0,
    xco2_prior_ppm: float = 420.0,
    albedo:         float = 0.25,
    sza:            float = 30.0,
    snr:            float = 300.0,
    sa:             float = 0.10,
    max_iter:       int   = 15,
    seed:           int   = 0,
) -> MCResult:
    """
    XCO₂ sweep: retrieve across a range of true CO₂ concentrations.
    """
    print(f"\n{'='*60}")
    print(f"  MODE 2 — XCO₂ Sweep  (N={n_samples})")
    print(f"  Range: {xco2_min_ppm}–{xco2_max_ppm} ppm  SNR={snr}")
    print(f"{'='*60}")

    rng          = np.random.default_rng(seed)
    xsec         = xsec_func(101325.0, 288.0)
    N_col_prior  = N_col_func(101325.0, 288.0, xco2_prior_ppm)
    amf          = air_mass_factor(sza, 0.0)

    xco2_true_arr  = np.linspace(xco2_min_ppm, xco2_max_ppm, n_samples)
    xco2_ret_arr   = np.full(n_samples, np.nan)
    xco2_unc_arr   = np.full(n_samples, np.nan)
    n_iter_arr     = np.full(n_samples, -1, dtype=int)
    converged_arr  = np.zeros(n_samples, dtype=bool)

    t0 = time.time()
    for i, xco2_true_i in enumerate(xco2_true_arr):
        if (i + 1) % 25 == 0:
            print(f"  Sample {i+1}/{n_samples}")
        N_col_true_i = N_col_func(101325.0, 288.0, xco2_true_i)
        r = _run_single(nu, xsec, N_col_true_i, N_col_prior, amf,
                        albedo, sza, snr, xco2_prior_ppm, sa, max_iter, rng)
        xco2_ret_arr[i], xco2_unc_arr[i], n_iter_arr[i], converged_arr[i] = r

    valid = converged_arr & np.isfinite(xco2_ret_arr)
    print(f"  Done. {valid.sum()}/{n_samples} converged in {time.time()-t0:.1f}s")

    result = MCResult(
        mode           = "XCO₂ Sweep",
        xco2_true      = xco2_true_arr[valid],
        xco2_retrieved = xco2_ret_arr[valid],
        xco2_unc       = xco2_unc_arr[valid],
        albedo_true    = np.full(valid.sum(), albedo),
        sza_true       = np.full(valid.sum(), sza),
        snr_true       = np.full(valid.sum(), snr),
        n_iter         = n_iter_arr[valid],
        converged      = converged_arr[valid],
    )
    result.print_summary()
    return result


def run_random_scenes(
    nu:             np.ndarray,
    xsec_func:      Callable,
    N_col_func:     Callable,
    n_samples:      int   = 500,
    xco2_prior_ppm: float = 420.0,
    xco2_mean_ppm:  float = 420.0,
    xco2_std_ppm:   float = 15.0,
    sa:             float = 0.10,
    max_iter:       int   = 15,
    seed:           int   = 7,
) -> MCResult:
    """
    Random scene ensemble: all geophysical and instrument parameters sampled
    from physically motivated distributions.
    """
    print(f"\n{'='*60}")
    print(f"  MODE 3 — Random Scene Ensemble  (N={n_samples})")
    print(f"  XCO₂ ~ N({xco2_mean_ppm},{xco2_std_ppm}) ppm")
    print(f"  Albedo ~ U(0.05,0.40)   SZA ~ U(10°,70°)   SNR ~ U(150,400)")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)

    # Sample all parameters
    xco2_true_arr = np.clip(rng.normal(xco2_mean_ppm, xco2_std_ppm, n_samples), 350, 550)
    albedo_arr    = rng.uniform(0.05, 0.40, n_samples)
    sza_arr       = rng.uniform(10.0, 70.0, n_samples)
    T_arr         = np.clip(rng.normal(288.0, 12.0, n_samples), 260, 310)
    P_arr         = np.clip(rng.normal(101325.0, 2000.0, n_samples), 95000, 105000)
    snr_arr       = rng.uniform(150.0, 400.0, n_samples)

    xco2_ret_arr  = np.full(n_samples, np.nan)
    xco2_unc_arr  = np.full(n_samples, np.nan)
    n_iter_arr    = np.full(n_samples, -1, dtype=int)
    converged_arr = np.zeros(n_samples, dtype=bool)

    t0 = time.time()
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1}/{n_samples}  ({time.time()-t0:.1f}s elapsed)")

        P_i, T_i = P_arr[i], T_arr[i]
        try:
            xsec_i       = xsec_func(P_i, T_i)
            N_col_true_i = N_col_func(P_i, T_i, xco2_true_arr[i])
            N_col_prior_i = N_col_func(P_i, T_i, xco2_prior_ppm)
            amf_i        = air_mass_factor(sza_arr[i], 0.0)
            r = _run_single(nu, xsec_i, N_col_true_i, N_col_prior_i, amf_i,
                            albedo_arr[i], sza_arr[i], snr_arr[i],
                            xco2_prior_ppm, sa, max_iter, rng)
            xco2_ret_arr[i], xco2_unc_arr[i], n_iter_arr[i], converged_arr[i] = r
        except Exception as e:
            pass

    valid = converged_arr & np.isfinite(xco2_ret_arr)
    print(f"  Done. {valid.sum()}/{n_samples} converged in {time.time()-t0:.1f}s")

    result = MCResult(
        mode           = "Random Scenes",
        xco2_true      = xco2_true_arr[valid],
        xco2_retrieved = xco2_ret_arr[valid],
        xco2_unc       = xco2_unc_arr[valid],
        albedo_true    = albedo_arr[valid],
        sza_true       = sza_arr[valid],
        snr_true       = snr_arr[valid],
        n_iter         = n_iter_arr[valid],
        converged      = converged_arr[valid],
    )
    result.print_summary()
    return result


def run_snr_sensitivity(
    nu:             np.ndarray,
    xsec_func:      Callable,
    N_col_func:     Callable,
    snr_values:     List[float],
    n_trials:       int   = 200,
    xco2_true_ppm:  float = 435.0,
    xco2_prior_ppm: float = 420.0,
    sa:             float = 0.10,
    seed:           int   = 99,
) -> dict:
    """
    SNR sensitivity: run noise_ensemble at multiple SNR levels.
    Returns dict of {snr: MCResult}.
    """
    print(f"\n{'='*60}")
    print(f"  SNR SENSITIVITY  (SNR={snr_values}, {n_trials} trials each)")
    print(f"{'='*60}")

    results = {}
    for snr in snr_values:
        print(f"\n  SNR = {snr:.0f} …")
        xsec = xsec_func(101325.0, 288.0)
        results[snr] = run_noise_ensemble(
            nu, xsec_func, N_col_func,
            n_samples=n_trials,
            xco2_true_ppm=xco2_true_ppm,
            xco2_prior_ppm=xco2_prior_ppm,
            snr=snr, sa=sa, seed=seed,
        )
    return results


def run_sza_sensitivity(
    nu:             np.ndarray,
    xsec_func:      Callable,
    N_col_func:     Callable,
    sza_values:     List[float],
    n_trials:       int   = 150,
    xco2_true_ppm:  float = 435.0,
    xco2_prior_ppm: float = 420.0,
    snr:            float = 250.0,
    sa:             float = 0.10,
    seed:           int   = 55,
) -> dict:
    """
    SZA sensitivity: noise ensemble at multiple solar zenith angles.
    """
    print(f"\n{'='*60}")
    print(f"  SZA SENSITIVITY  (SZA={sza_values}°, {n_trials} trials each)")
    print(f"{'='*60}")

    results = {}
    for sza in sza_values:
        print(f"\n  SZA = {sza:.0f}° …")
        results[sza] = run_noise_ensemble(
            nu, xsec_func, N_col_func,
            n_samples=n_trials,
            xco2_true_ppm=xco2_true_ppm,
            xco2_prior_ppm=xco2_prior_ppm,
            sza=sza, snr=snr, sa=sa, seed=seed,
        )
    return results


# ===========================================================================
# Plotting functions
# ===========================================================================

def _savefig(fig, filename: str):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → Saved: {path}")


def plot_noise_ensemble(result: MCResult):
    """
    Figure 1: Error histogram + posterior uncertainty scatter.
    """
    err = result.error
    mu  = result.bias
    sig = result.precision

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Monte Carlo — Noise Ensemble\n"
        f"True XCO₂ = {result.xco2_true[0]:.1f} ppm  |  SNR = {result.snr_true[0]:.0f}  |  N = {result.n_samples}",
        fontsize=13, fontweight="bold", color="white",
    )

    # --- Panel A: error histogram ---
    ax = axes[0]
    n_bins = max(25, result.n_samples // 12)
    counts, edges, patches = ax.hist(
        err, bins=n_bins, density=True, color=PALETTE["blue"],
        edgecolor="#0f1117", alpha=0.85, label="MC samples",
    )
    # Gaussian fit
    x_fit = np.linspace(err.min() - 0.5, err.max() + 0.5, 400)
    gauss = (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fit - mu) / sig)**2)
    ax.plot(x_fit, gauss, color=PALETTE["orange"], lw=2.5,
            label=f"Gaussian fit\nμ = {mu:+.3f} ppm\nσ = {sig:.3f} ppm")

    ax.axvline(0,  color="white",         lw=1.5, ls="--",  alpha=0.6, label="Zero error")
    ax.axvline(mu, color=PALETTE["pink"], lw=1.5, ls=":",   label=f"Bias")
    ax.fill_betweenx([0, gauss.max() * 1.05], mu - sig, mu + sig,
                     color=PALETTE["blue"], alpha=0.12, label="±1σ band")

    ax.set_xlabel("XCO₂ retrieval error [ppm]", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title("Error Distribution", fontsize=12, pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)

    # --- Panel B: posterior uncertainty vs |error| ---
    ax2 = axes[1]
    sc = ax2.scatter(np.abs(err), result.xco2_unc,
                     c=result.xco2_unc, cmap="plasma",
                     s=15, alpha=0.55, edgecolors="none")
    ax2.axhline(np.mean(result.xco2_unc), color=PALETTE["teal"], lw=2, ls="--",
                label=f"Mean σ_post = {np.mean(result.xco2_unc):.3f} ppm")
    ax2.axvline(sig, color=PALETTE["orange"], lw=1.5, ls=":",
                label=f"MC precision = {sig:.3f} ppm")
    plt.colorbar(sc, ax=ax2, label="σ_post [ppm]")

    ax2.set_xlabel("|Error| [ppm]", fontsize=11)
    ax2.set_ylabel("Posterior uncertainty σ_post [ppm]", fontsize=11)
    ax2.set_title("Error Magnitude vs Posterior Uncertainty", fontsize=12, pad=8)
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    _savefig(fig, "mc_01_noise_ensemble.png")
    plt.show()


def plot_xco2_sweep(result: MCResult):
    """
    Figure 2: True vs retrieved scatter + error vs true XCO₂.
    """
    true = result.xco2_true
    ret  = result.xco2_retrieved
    unc  = result.xco2_unc
    err  = result.error

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Monte Carlo — XCO₂ Sweep\n"
        f"Range: {true.min():.0f}–{true.max():.0f} ppm  |  SNR = {result.snr_true[0]:.0f}  |  N = {result.n_samples}",
        fontsize=13, fontweight="bold", color="white",
    )

    # --- Panel A: true vs retrieved ---
    ax = axes[0]
    lim = [true.min() - 3, true.max() + 3]
    ax.plot(lim, lim, color="white", lw=1.2, ls="--", alpha=0.4, label="1:1 line", zorder=1)
    ax.fill_between(true, ret - unc, ret + unc,
                    color=PALETTE["blue"], alpha=0.2, label="±1σ posterior", zorder=2)
    ax.scatter(true, ret, s=22, color=PALETTE["blue"], alpha=0.8,
               edgecolors="none", zorder=3, label="Retrieved")

    # Linear fit
    p = np.polyfit(true, ret, 1)
    x_fit = np.array(lim)
    ax.plot(x_fit, np.polyval(p, x_fit), color=PALETTE["orange"], lw=2,
            label=f"Linear fit: slope={p[0]:.4f}")

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True XCO₂ [ppm]", fontsize=11)
    ax.set_ylabel("Retrieved XCO₂ [ppm]", fontsize=11)
    ax.set_title("True vs Retrieved XCO₂", fontsize=12, pad=8)
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- Panel B: retrieval error vs true XCO₂ ---
    ax2 = axes[1]
    ax2.axhline(0, color="white", lw=1.2, ls="--", alpha=0.4)
    ax2.fill_between(true, -unc, +unc, color=PALETTE["red"], alpha=0.15,
                     label="±1σ posterior")
    ax2.scatter(true, err, s=22, color=PALETTE["red"], alpha=0.8,
                edgecolors="none", label="Error")

    # Running mean
    sort_idx = np.argsort(true)
    n_bins   = 10
    chunks   = np.array_split(sort_idx, n_bins)
    x_m = [true[c].mean()  for c in chunks if len(c) > 0]
    y_m = [err[c].mean()   for c in chunks if len(c) > 0]
    ax2.plot(x_m, y_m, "o-", color=PALETTE["yellow"], lw=2, ms=5, label="Running mean")

    ax2.set_xlabel("True XCO₂ [ppm]", fontsize=11)
    ax2.set_ylabel("Error [ppm]", fontsize=11)
    ax2.set_title(
        f"Retrieval Error vs True XCO₂\n"
        f"Bias={result.bias:+.3f} ppm   RMSE={result.rmse:.3f} ppm",
        fontsize=12, pad=8,
    )
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    _savefig(fig, "mc_02_xco2_sweep.png")
    plt.show()


def plot_random_scene(result: MCResult):
    """
    Figure 3: 6-panel random scene diagnostic.
    """
    err  = result.error
    true = result.xco2_true
    ret  = result.xco2_retrieved

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    fig.suptitle(
        f"Monte Carlo — Random Scene Ensemble   "
        f"(N={result.n_samples}  |  "
        f"Bias={result.bias:+.3f} ppm  |  "
        f"σ={result.precision:.3f} ppm  |  "
        f"RMSE={result.rmse:.3f} ppm)",
        fontsize=13, fontweight="bold", color="white", y=1.01,
    )

    def _running_mean(ax, x, y, n_bins=10, color=PALETTE["yellow"], label="Running mean"):
        si = np.argsort(x)
        xs, ys = x[si], y[si]
        chunks = np.array_split(np.arange(len(xs)), n_bins)
        xm = [xs[c].mean() for c in chunks if len(c) > 0]
        ym = [ys[c].mean() for c in chunks if len(c) > 0]
        ax.plot(xm, ym, "o-", color=color, lw=2, ms=4, label=label)

    # 1. True vs retrieved (colour = SNR)
    ax1 = fig.add_subplot(gs[0, 0])
    lim = [true.min() - 3, true.max() + 3]
    ax1.plot(lim, lim, "w--", lw=1, alpha=0.4)
    sc1 = ax1.scatter(true, ret, c=result.snr_true, cmap="viridis",
                      s=12, alpha=0.6, edgecolors="none")
    plt.colorbar(sc1, ax=ax1, label="SNR")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("True XCO₂ [ppm]"); ax1.set_ylabel("Retrieved XCO₂ [ppm]")
    ax1.set_title("True vs Retrieved\n(colour = SNR)", fontsize=10)
    ax1.grid(True)

    # 2. Error histogram + Gaussian fit
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(err, bins=40, density=True, color=PALETTE["blue"],
             edgecolor="#0f1117", alpha=0.85)
    x_fit = np.linspace(err.min(), err.max(), 300)
    g = (1 / (result.precision * np.sqrt(2*np.pi))) * \
        np.exp(-0.5 * ((x_fit - result.bias) / result.precision)**2)
    ax2.plot(x_fit, g, color=PALETTE["orange"], lw=2.5)
    ax2.axvline(0,             color="white",         lw=1.2, ls="--", alpha=0.5)
    ax2.axvline(result.bias,   color=PALETTE["pink"],  lw=1.5, ls=":",
                label=f"Bias={result.bias:+.3f} ppm")
    ax2.set_xlabel("Error [ppm]"); ax2.set_ylabel("Density")
    ax2.set_title(f"Error Histogram  σ={result.precision:.3f} ppm", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True)

    # 3. Error vs albedo
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(result.albedo_true, err, s=10, alpha=0.4,
                color=PALETTE["teal"], edgecolors="none")
    ax3.axhline(0, color="white", lw=1, ls="--", alpha=0.4)
    _running_mean(ax3, result.albedo_true, err)
    ax3.set_xlabel("Surface Albedo"); ax3.set_ylabel("Error [ppm]")
    ax3.set_title("Error vs Surface Albedo", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(True)

    # 4. Error vs SZA
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(result.sza_true, err, s=10, alpha=0.4,
                color=PALETTE["orange"], edgecolors="none")
    ax4.axhline(0, color="white", lw=1, ls="--", alpha=0.4)
    _running_mean(ax4, result.sza_true, err, color=PALETTE["red"])
    ax4.set_xlabel("Solar Zenith Angle [deg]"); ax4.set_ylabel("Error [ppm]")
    ax4.set_title("Error vs Solar Zenith Angle", fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(True)

    # 5. |Error| vs SNR
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(result.snr_true, np.abs(err), s=10, alpha=0.4,
                color=PALETTE["purple"], edgecolors="none")
    _running_mean(ax5, result.snr_true, np.abs(err),
                  color=PALETTE["pink"], label="|error| mean")
    ax5.set_xlabel("SNR"); ax5.set_ylabel("|Error| [ppm]")
    ax5.set_title("|Error| vs SNR", fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(True)

    # 6. Posterior uncertainty distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(result.xco2_unc, bins=35, color=PALETTE["green"],
             edgecolor="#0f1117", alpha=0.85)
    ax6.axvline(result.xco2_unc.mean(), color=PALETTE["yellow"], lw=2, ls="--",
                label=f"Mean = {result.xco2_unc.mean():.3f} ppm")
    ax6.set_xlabel("Posterior uncertainty [ppm]"); ax6.set_ylabel("Count")
    ax6.set_title("Retrieval Uncertainty Distribution", fontsize=10)
    ax6.legend(fontsize=8); ax6.grid(True)

    _savefig(fig, "mc_03_random_scene.png")
    plt.show()


def plot_snr_sensitivity(snr_results: dict):
    """
    Figure 4: Retrieval precision and |bias| vs SNR.
    """
    snr_vals   = np.array(sorted(snr_results.keys()))
    precisions = np.array([snr_results[s].precision for s in snr_vals])
    biases     = np.array([abs(snr_results[s].bias)  for s in snr_vals])
    mean_uncs  = np.array([snr_results[s].xco2_unc.mean() for s in snr_vals])

    # Theoretical 1/SNR envelope (fitted)
    snr_fit = np.linspace(snr_vals.min(), snr_vals.max(), 200)
    A_fit   = precisions[0] * snr_vals[0]
    theory  = A_fit / snr_fit

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SNR Sensitivity Study", fontsize=14, fontweight="bold", color="white")

    ax = axes[0]
    ax.plot(snr_fit, theory, color="white", lw=1.5, ls="--", alpha=0.4, label="∝ 1/SNR (theory)")
    ax.plot(snr_vals, precisions, "o-", color=PALETTE["blue"], lw=2.5, ms=8,
            label="MC precision (σ)")
    ax.plot(snr_vals, mean_uncs, "s--", color=PALETTE["orange"], lw=2, ms=7,
            label="Mean σ_post (OE)")
    ax.set_xlabel("SNR", fontsize=12); ax.set_ylabel("XCO₂ uncertainty [ppm]", fontsize=12)
    ax.set_title("Retrieval Precision vs SNR", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True)

    ax2 = axes[1]
    ax2.bar(snr_vals, biases, width=(snr_vals[1]-snr_vals[0])*0.6,
            color=PALETTE["teal"], alpha=0.85, edgecolor="#0f1117")
    ax2.axhline(0, color="white", lw=1, ls="--", alpha=0.4)
    ax2.set_xlabel("SNR", fontsize=12); ax2.set_ylabel("|Bias| [ppm]", fontsize=12)
    ax2.set_title("|Retrieval Bias| vs SNR", fontsize=12)
    ax2.grid(True, axis="y")

    plt.tight_layout()
    _savefig(fig, "mc_04_snr_sensitivity.png")
    plt.show()


def plot_sza_sensitivity(sza_results: dict):
    """
    Figure 5: Retrieval precision and |bias| vs solar zenith angle.
    """
    sza_vals   = np.array(sorted(sza_results.keys()))
    precisions = np.array([sza_results[s].precision for s in sza_vals])
    biases     = np.array([abs(sza_results[s].bias)  for s in sza_vals])
    mean_uncs  = np.array([sza_results[s].xco2_unc.mean() for s in sza_vals])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Solar Zenith Angle Sensitivity Study",
                 fontsize=14, fontweight="bold", color="white")

    ax = axes[0]
    ax.plot(sza_vals, precisions, "o-", color=PALETTE["purple"], lw=2.5, ms=8,
            label="MC precision (σ)")
    ax.plot(sza_vals, mean_uncs, "s--", color=PALETTE["orange"], lw=2, ms=7,
            label="Mean σ_post (OE)")
    ax.set_xlabel("Solar Zenith Angle [deg]", fontsize=12)
    ax.set_ylabel("XCO₂ uncertainty [ppm]", fontsize=12)
    ax.set_title("Precision vs SZA\n(larger SZA → longer path → more signal, but also more noise)",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(True)

    ax2 = axes[1]
    ax2.bar(sza_vals, biases,
            width=(sza_vals[1]-sza_vals[0])*0.6 if len(sza_vals) > 1 else 5,
            color=PALETTE["pink"], alpha=0.85, edgecolor="#0f1117")
    ax2.set_xlabel("Solar Zenith Angle [deg]", fontsize=12)
    ax2.set_ylabel("|Bias| [ppm]", fontsize=12)
    ax2.set_title("|Retrieval Bias| vs SZA", fontsize=12)
    ax2.grid(True, axis="y")

    plt.tight_layout()
    _savefig(fig, "mc_05_sza_sensitivity.png")
    plt.show()


def plot_summary(res_noise: MCResult, res_sweep: MCResult, res_random: MCResult):
    """
    Figure 6: Side-by-side summary comparing all three MC modes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Monte Carlo Summary — All Modes",
                 fontsize=14, fontweight="bold", color="white")

    metrics = [
        ("Bias [ppm]",      [res_noise.bias,      res_sweep.bias,      res_random.bias]),
        ("Precision [ppm]", [res_noise.precision,  res_sweep.precision,  res_random.precision]),
        ("RMSE [ppm]",      [res_noise.rmse,       res_sweep.rmse,       res_random.rmse]),
    ]
    modes  = ["Noise\nEnsemble", "XCO₂\nSweep", "Random\nScenes"]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["teal"]]

    for ax, (title, values) in zip(axes, metrics):
        bars = ax.bar(modes, np.abs(values), color=colors, edgecolor="#0f1117",
                      alpha=0.85, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:+.3f}" if "Bias" in title else f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, color="white")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("ppm", fontsize=11)
        ax.grid(True, axis="y")
        ax.set_ylim(0, max(np.abs(values)) * 1.35 + 0.01)

    plt.tight_layout()
    _savefig(fig, "mc_06_bias_precision_summary.png")
    plt.show()


# ===========================================================================
# Main entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="CO₂ Retrieval Monte Carlo Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n_noise",  type=int,   default=300,  help="Noise ensemble samples")
    p.add_argument("--n_sweep",  type=int,   default=80,   help="XCO₂ sweep samples")
    p.add_argument("--n_random", type=int,   default=400,  help="Random scene samples")
    p.add_argument("--n_snr",    type=int,   default=150,  help="Trials per SNR level")
    p.add_argument("--n_sza",    type=int,   default=100,  help="Trials per SZA level")
    p.add_argument("--seed",     type=int,   default=42,   help="Global random seed")
    p.add_argument("--nu_pts",   type=int,   default=2000, help="Spectral grid points")
    p.add_argument("--no_show",  action="store_true",      help="Don't display plots interactively")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_show:
        plt.switch_backend("Agg")

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   CO₂ Retrieval Simulator — Monte Carlo Experiment      ║")
    print("║   Author: Arun Kumar Pandey                             ║")
    print("╚" + "═"*58 + "╝")
    print(f"\n  Output figures → {OUT_DIR}/")
    print(f"  Spectral grid   : {args.nu_pts} points  (6210–6270 cm⁻¹)")
    print(f"  Random seed     : {args.seed}")

    t_total = time.time()

    # Spectral grid
    nu = np.linspace(6210, 6270, args.nu_pts)
    xsec_func, N_col_func = _make_fwd_functions(nu)

    # ------------------------------------------------------------------
    # Run all three MC modes
    # ------------------------------------------------------------------
    res_noise  = run_noise_ensemble(
        nu, xsec_func, N_col_func,
        n_samples=args.n_noise,
        xco2_true_ppm=430.0, xco2_prior_ppm=420.0,
        snr=250.0, sza=30.0, seed=args.seed,
    )

    res_sweep  = run_xco2_sweep(
        nu, xsec_func, N_col_func,
        n_samples=args.n_sweep,
        xco2_min_ppm=380.0, xco2_max_ppm=480.0,
        xco2_prior_ppm=420.0, snr=300.0, seed=args.seed,
    )

    res_random = run_random_scenes(
        nu, xsec_func, N_col_func,
        n_samples=args.n_random,
        xco2_prior_ppm=420.0, seed=args.seed + 5,
    )

    # ------------------------------------------------------------------
    # Sensitivity studies
    # ------------------------------------------------------------------
    snr_values = [75, 100, 150, 200, 300, 400]
    snr_results = run_snr_sensitivity(
        nu, xsec_func, N_col_func,
        snr_values=snr_values,
        n_trials=args.n_snr,
        seed=args.seed,
    )

    sza_values = [10, 20, 30, 45, 55, 65]
    sza_results = run_sza_sensitivity(
        nu, xsec_func, N_col_func,
        sza_values=sza_values,
        n_trials=args.n_sza,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Generate all figures
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Generating figures …")
    print(f"{'='*60}")

    plot_noise_ensemble(res_noise)
    plot_xco2_sweep(res_sweep)
    plot_random_scene(res_random)
    plot_snr_sensitivity(snr_results)
    plot_sza_sensitivity(sza_results)
    plot_summary(res_noise, res_sweep, res_random)

    print(f"\n{'='*60}")
    print(f"  All done in {time.time() - t_total:.1f}s")
    print(f"  6 figures saved to: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
