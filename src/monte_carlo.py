"""
monte_carlo.py
==============
Monte Carlo simulation engine for the CO2 Retrieval Simulator.

This module generates realistic synthetic atmospheric observations by randomly
sampling key geophysical and instrument parameters, then running the full
forward model + retrieval chain on each sample.

Three simulation modes are provided:

1. **Noise Ensemble**
   Fix the atmospheric scene; repeat with independent noise realisations.
   → Characterises the *precision* (random error) of the retrieval.

2. **XCO₂ Sweep**
   Vary true XCO₂ across a realistic range; retrieve each and compare.
   → Characterises *linearity* and *bias* of the retrieval.

3. **Random Scene Ensemble**
   Draw random P, T profiles, surface albedo, viewing geometry, and XCO₂.
   → Characterises retrieval performance across the full range of
     real-world atmospheric states encountered by a satellite.

Monte Carlo Design
------------------
Each simulation draw samples the following parameters from physically
motivated distributions:

    XCO₂       ~ N(μ_xco2, σ_xco2)          [ppm]
    Albedo      ~ Uniform(0.05, 0.40)
    SZA         ~ Uniform(10°, 70°)
    T_surface   ~ N(288, 12) K
    P_surface   ~ N(101325, 2000) Pa
    SNR         ~ Uniform(150, 400)

The retrieval is run on each synthetic observation, and results are
collected into a structured array for statistical analysis.

Author: Arun Kumar Pandey
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """Container for Monte Carlo simulation results."""

    # Inputs (truth)
    xco2_true:       np.ndarray = field(default_factory=lambda: np.array([]))
    albedo_true:     np.ndarray = field(default_factory=lambda: np.array([]))
    sza_true:        np.ndarray = field(default_factory=lambda: np.array([]))
    T_surface_true:  np.ndarray = field(default_factory=lambda: np.array([]))
    P_surface_true:  np.ndarray = field(default_factory=lambda: np.array([]))
    snr_true:        np.ndarray = field(default_factory=lambda: np.array([]))

    # Retrieval outputs
    xco2_retrieved:  np.ndarray = field(default_factory=lambda: np.array([]))
    xco2_unc:        np.ndarray = field(default_factory=lambda: np.array([]))
    n_iter:          np.ndarray = field(default_factory=lambda: np.array([]))
    converged:       np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    @property
    def error(self) -> np.ndarray:
        """Retrieval error = retrieved − true [ppm]."""
        return self.xco2_retrieved - self.xco2_true

    @property
    def bias(self) -> float:
        """Mean retrieval bias [ppm]."""
        return float(np.mean(self.error))

    @property
    def precision(self) -> float:
        """1-σ retrieval precision (std of error) [ppm]."""
        return float(np.std(self.error))

    @property
    def rmse(self) -> float:
        """Root-mean-square error [ppm]."""
        return float(np.sqrt(np.mean(self.error**2)))

    @property
    def n_samples(self) -> int:
        return len(self.xco2_true)

    def summary(self) -> str:
        return (
            f"Monte Carlo Summary  (N = {self.n_samples})\n"
            f"  Bias      : {self.bias:+.3f} ppm\n"
            f"  Precision : {self.precision:.3f} ppm\n"
            f"  RMSE      : {self.rmse:.3f} ppm\n"
            f"  Mean unc. : {np.mean(self.xco2_unc):.3f} ppm\n"
        )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    nu: np.ndarray,
    xsec_func,
    N_col_func,
    n_samples: int = 500,
    mode: str = "random_scene",
    # Fixed-scene parameters (for noise_ensemble and xco2_sweep modes)
    xco2_fixed_ppm: float = 420.0,
    xco2_prior_ppm: float = 420.0,
    albedo_fixed: float = 0.25,
    sza_fixed: float = 30.0,
    T_surface_fixed: float = 288.0,
    P_surface_fixed: float = 101325.0,
    snr_fixed: float = 250.0,
    # XCO₂ sweep parameters
    xco2_min_ppm: float = 380.0,
    xco2_max_ppm: float = 480.0,
    # Random scene distribution parameters
    xco2_mean_ppm: float = 420.0,
    xco2_std_ppm: float = 15.0,
    snr_min: float = 150.0,
    snr_max: float = 400.0,
    # Retrieval settings
    sa: float = 0.10,
    max_iter: int = 15,
    random_seed: Optional[int] = 42,
    verbose: bool = True,
) -> MCResult:
    """
    Run a Monte Carlo retrieval experiment.

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    xsec_func : callable
        xsec_func(P_Pa, T_K) → cross section array [cm²/molecule].
        Called once per sample in random_scene mode.
    N_col_func : callable
        N_col_func(P_Pa, T_K, xco2_ppm) → column amount [molecules/cm²].
    n_samples : int
        Number of Monte Carlo draws.
    mode : str
        One of ``'noise_ensemble'``, ``'xco2_sweep'``, ``'random_scene'``.
    xco2_fixed_ppm : float
        True XCO₂ for noise_ensemble mode [ppm].
    xco2_prior_ppm : float
        A priori XCO₂ used in every retrieval [ppm].
    albedo_fixed : float
        Surface albedo for noise_ensemble / xco2_sweep.
    sza_fixed : float
        Solar zenith angle [deg] for noise_ensemble / xco2_sweep.
    T_surface_fixed, P_surface_fixed : float
        Surface T [K] and P [Pa] for non-random modes.
    snr_fixed : float
        Signal-to-noise ratio for noise_ensemble / xco2_sweep.
    xco2_min_ppm, xco2_max_ppm : float
        XCO₂ range for xco2_sweep [ppm].
    xco2_mean_ppm, xco2_std_ppm : float
        Distribution of true XCO₂ in random_scene mode.
    snr_min, snr_max : float
        SNR distribution bounds for random_scene mode.
    sa : float
        A priori uncertainty (fractional std dev of ξ).
    max_iter : int
        Max iterations for Levenberg-Marquardt.
    random_seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress every 100 samples.

    Returns
    -------
    MCResult
        Structured result object with truth and retrieval arrays.
    """
    from radiative_transfer import forward_model, air_mass_factor
    from retrieval import iterative_retrieval, xco2_from_scaling

    rng = np.random.default_rng(random_seed)

    # --- Allocate output arrays ---
    xco2_true_arr  = np.zeros(n_samples)
    albedo_arr     = np.zeros(n_samples)
    sza_arr        = np.zeros(n_samples)
    T_arr          = np.zeros(n_samples)
    P_arr          = np.zeros(n_samples)
    snr_arr        = np.zeros(n_samples)
    xco2_ret_arr   = np.zeros(n_samples)
    xco2_unc_arr   = np.zeros(n_samples)
    n_iter_arr     = np.zeros(n_samples, dtype=int)
    converged_arr  = np.zeros(n_samples, dtype=bool)

    # --- Sample parameters for each mode ---
    if mode == "noise_ensemble":
        xco2_true_arr[:] = xco2_fixed_ppm
        albedo_arr[:]    = albedo_fixed
        sza_arr[:]       = sza_fixed
        T_arr[:]         = T_surface_fixed
        P_arr[:]         = P_surface_fixed
        snr_arr[:]       = snr_fixed

    elif mode == "xco2_sweep":
        xco2_true_arr    = np.linspace(xco2_min_ppm, xco2_max_ppm, n_samples)
        albedo_arr[:]    = albedo_fixed
        sza_arr[:]       = sza_fixed
        T_arr[:]         = T_surface_fixed
        P_arr[:]         = P_surface_fixed
        snr_arr[:]       = snr_fixed

    elif mode == "random_scene":
        xco2_true_arr = rng.normal(xco2_mean_ppm, xco2_std_ppm, n_samples)
        xco2_true_arr = np.clip(xco2_true_arr, 350.0, 550.0)
        albedo_arr    = rng.uniform(0.05, 0.40, n_samples)
        sza_arr       = rng.uniform(10.0, 70.0, n_samples)
        T_arr         = rng.normal(288.0, 12.0, n_samples)
        T_arr         = np.clip(T_arr, 260.0, 310.0)
        P_arr         = rng.normal(101325.0, 2000.0, n_samples)
        P_arr         = np.clip(P_arr, 95000.0, 105000.0)
        snr_arr       = rng.uniform(snr_min, snr_max, n_samples)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'noise_ensemble', 'xco2_sweep', or 'random_scene'.")

    # --- Main loop ---
    for i in range(n_samples):
        if verbose and (i % 100 == 0):
            print(f"  Sample {i+1}/{n_samples} …")

        xco2_true_i = xco2_true_arr[i]
        albedo_i    = albedo_arr[i]
        sza_i       = sza_arr[i]
        T_i         = T_arr[i]
        P_i         = P_arr[i]
        snr_i       = snr_arr[i]

        try:
            # Cross section and column at this P, T
            xsec_i  = xsec_func(P_i, T_i)
            N_col_i = N_col_func(P_i, T_i, xco2_true_i)
            N_col_prior_i = N_col_func(P_i, T_i, xco2_prior_ppm)

            amf_i = air_mass_factor(sza_i, 0.0)   # nadir satellite

            # True (noiseless) radiance
            res_true = forward_model(nu, xsec_i, N_col_i, amf_i,
                                     surface_albedo=albedo_i,
                                     solar_zenith_deg=sza_i)
            I_true = res_true["radiance"]

            # Add instrument noise
            sigma_noise_i = I_true.max() / snr_i
            noise = rng.normal(0.0, sigma_noise_i, size=len(nu))
            y_obs_i = I_true + noise

            # Prior (reference) radiance
            res_prior = forward_model(nu, xsec_i, N_col_prior_i, amf_i,
                                      surface_albedo=albedo_i,
                                      solar_zenith_deg=sza_i)

            # Forward function for retrieval (scales prior column)
            def fwd_i(xi, _xs=xsec_i, _Np=N_col_prior_i, _amf=amf_i, _alb=albedo_i, _sza=sza_i):
                N = _Np * xi
                return forward_model(nu, _xs, N, _amf,
                                     surface_albedo=_alb,
                                     solar_zenith_deg=_sza)["radiance"]

            result_i = iterative_retrieval(
                nu, y_obs_i, fwd_i,
                xa=1.0, sa=sa,
                sigma_noise=sigma_noise_i,
                max_iter=max_iter,
            )

            xi_hat_i     = result_i["xi_hat"]
            sigma_post_i = result_i["sigma_post"]
            xco2_ret_i, xco2_unc_i = xco2_from_scaling(xi_hat_i, xco2_prior_ppm, sigma_post_i)

            xco2_ret_arr[i]  = xco2_ret_i
            xco2_unc_arr[i]  = xco2_unc_i
            n_iter_arr[i]    = result_i["n_iter"]
            converged_arr[i] = True

        except Exception as exc:
            # Mark failed retrievals
            xco2_ret_arr[i]  = np.nan
            xco2_unc_arr[i]  = np.nan
            n_iter_arr[i]    = -1
            converged_arr[i] = False
            if verbose:
                print(f"    [!] Sample {i} failed: {exc}")

    # Filter NaN rows for statistics
    valid = converged_arr & np.isfinite(xco2_ret_arr)
    if verbose:
        print(f"\nCompleted. {valid.sum()}/{n_samples} samples converged successfully.")

    return MCResult(
        xco2_true      = xco2_true_arr[valid],
        albedo_true    = albedo_arr[valid],
        sza_true       = sza_arr[valid],
        T_surface_true = T_arr[valid],
        P_surface_true = P_arr[valid],
        snr_true       = snr_arr[valid],
        xco2_retrieved = xco2_ret_arr[valid],
        xco2_unc       = xco2_unc_arr[valid],
        n_iter         = n_iter_arr[valid],
        converged      = converged_arr[valid],
    )


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_noise_ensemble(result: MCResult, savefig: str = None) -> None:
    """
    Histogram and statistics for a noise ensemble simulation.
    Shows the retrieval precision for a fixed atmospheric scene.
    """
    errors = result.error
    mu     = errors.mean()
    sigma  = errors.std()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Error histogram ---
    ax = axes[0]
    n_bins = max(20, result.n_samples // 15)
    ax.hist(errors, bins=n_bins, color="steelblue", edgecolor="white",
            alpha=0.85, density=True, label="MC samples")

    # Overlay Gaussian fit
    x_fit = np.linspace(errors.min(), errors.max(), 300)
    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fit - mu) / sigma)**2)
    ax.plot(x_fit, gauss, "r-", lw=2, label=f"Gaussian fit\nμ={mu:.3f}, σ={sigma:.3f} ppm")
    ax.axvline(0,  color="black", lw=1.5, ls="--", label="Zero error")
    ax.axvline(mu, color="tomato", lw=1.5, ls=":", label=f"Bias = {mu:.3f} ppm")

    ax.set_xlabel("XCO₂ error [ppm]", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(f"Retrieval Error Distribution\n(N={result.n_samples} noise realisations)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # --- Posterior uncertainty vs error magnitude ---
    ax2 = axes[1]
    ax2.scatter(np.abs(errors), result.xco2_unc, alpha=0.4, s=12,
                color="steelblue", edgecolors="none")
    ax2.axhline(result.xco2_unc.mean(), color="tomato", lw=1.5, ls="--",
                label=f"Mean σ_post = {result.xco2_unc.mean():.3f} ppm")
    ax2.set_xlabel("|Error| [ppm]", fontsize=12)
    ax2.set_ylabel("Posterior uncertainty σ_post [ppm]", fontsize=12)
    ax2.set_title("Error Magnitude vs Posterior Uncertainty", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.suptitle("Noise Ensemble Monte Carlo", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


def plot_xco2_sweep(result: MCResult, savefig: str = None) -> None:
    """
    True vs retrieved XCO₂ scatter with error envelope.
    Shows retrieval linearity and bias across the CO₂ range.
    """
    true = result.xco2_true
    ret  = result.xco2_retrieved
    unc  = result.xco2_unc
    err  = result.error

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- True vs retrieved ---
    ax = axes[0]
    ax.plot([true.min(), true.max()], [true.min(), true.max()],
            "k--", lw=1.5, label="1:1 line")
    ax.fill_between(true, ret - unc, ret + unc, alpha=0.25, color="steelblue",
                    label="±1σ posterior")
    ax.scatter(true, ret, s=18, color="steelblue", alpha=0.7, edgecolors="none",
               label="Retrieved")
    ax.set_xlabel("True XCO₂ [ppm]", fontsize=12)
    ax.set_ylabel("Retrieved XCO₂ [ppm]", fontsize=12)
    ax.set_title("True vs Retrieved XCO₂", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # --- Retrieval error vs true XCO₂ ---
    ax2 = axes[1]
    ax2.axhline(0, color="black", lw=1.5, ls="--")
    ax2.fill_between(true, -unc, +unc, alpha=0.2, color="tomato",
                     label="±1σ posterior")
    ax2.scatter(true, err, s=18, color="tomato", alpha=0.7, edgecolors="none",
                label="Error")
    ax2.set_xlabel("True XCO₂ [ppm]", fontsize=12)
    ax2.set_ylabel("Retrieval error [ppm]", fontsize=12)
    ax2.set_title(f"Retrieval Error vs True XCO₂\n"
                  f"Bias={result.bias:+.3f} ppm  RMSE={result.rmse:.3f} ppm", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.suptitle("XCO₂ Sweep Monte Carlo", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


def plot_random_scene(result: MCResult, savefig: str = None) -> None:
    """
    Six-panel diagnostic plot for random scene Monte Carlo:
      1. True vs retrieved XCO₂
      2. Error histogram
      3. Error vs surface albedo
      4. Error vs solar zenith angle
      5. Error vs SNR
      6. Posterior uncertainty distribution
    """
    err  = result.error
    true = result.xco2_true
    ret  = result.xco2_retrieved

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. True vs retrieved
    ax1 = fig.add_subplot(gs[0, 0])
    lim = [true.min() - 5, true.max() + 5]
    ax1.plot(lim, lim, "k--", lw=1.5)
    sc = ax1.scatter(true, ret, c=result.snr_true, cmap="viridis",
                     s=12, alpha=0.6, edgecolors="none")
    plt.colorbar(sc, ax=ax1, label="SNR")
    ax1.set_xlabel("True XCO₂ [ppm]"); ax1.set_ylabel("Retrieved XCO₂ [ppm]")
    ax1.set_title("True vs Retrieved")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.grid(True, alpha=0.3)

    # 2. Error histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(err, bins=40, color="steelblue", edgecolor="white", alpha=0.85, density=True)
    x_fit = np.linspace(err.min(), err.max(), 300)
    g = (1 / (err.std() * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_fit - err.mean())/err.std())**2)
    ax2.plot(x_fit, g, "r-", lw=2)
    ax2.axvline(0, color="black", lw=1.5, ls="--")
    ax2.set_xlabel("Error [ppm]"); ax2.set_ylabel("Density")
    ax2.set_title(f"Error Distribution\nBias={result.bias:+.3f}  σ={result.precision:.3f} ppm")
    ax2.grid(True, alpha=0.3)

    # 3. Error vs albedo
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(result.albedo_true, err, s=10, alpha=0.5, color="seagreen", edgecolors="none")
    ax3.axhline(0, color="black", lw=1.2, ls="--")
    _add_running_mean(ax3, result.albedo_true, err, color="darkgreen", label="Running mean")
    ax3.set_xlabel("Surface Albedo"); ax3.set_ylabel("Error [ppm]")
    ax3.set_title("Error vs Surface Albedo")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # 4. Error vs SZA
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(result.sza_true, err, s=10, alpha=0.5, color="darkorange", edgecolors="none")
    ax4.axhline(0, color="black", lw=1.2, ls="--")
    _add_running_mean(ax4, result.sza_true, err, color="saddlebrown", label="Running mean")
    ax4.set_xlabel("Solar Zenith Angle [deg]"); ax4.set_ylabel("Error [ppm]")
    ax4.set_title("Error vs Solar Zenith Angle")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # 5. Error vs SNR
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(result.snr_true, np.abs(err), s=10, alpha=0.5,
                color="royalblue", edgecolors="none")
    _add_running_mean(ax5, result.snr_true, np.abs(err), color="navy", label="|error| mean")
    ax5.set_xlabel("SNR"); ax5.set_ylabel("|Error| [ppm]")
    ax5.set_title("|Error| vs SNR  (lower SNR → larger error)")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # 6. Posterior uncertainty distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(result.xco2_unc, bins=35, color="mediumpurple",
             edgecolor="white", alpha=0.85)
    ax6.axvline(result.xco2_unc.mean(), color="purple", lw=2, ls="--",
                label=f"Mean = {result.xco2_unc.mean():.3f} ppm")
    ax6.set_xlabel("Posterior uncertainty [ppm]"); ax6.set_ylabel("Count")
    ax6.set_title("Distribution of Retrieval Uncertainty")
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3)

    fig.suptitle(
        f"Random Scene Monte Carlo  (N={result.n_samples})\n"
        f"Bias={result.bias:+.3f} ppm   Precision={result.precision:.3f} ppm   RMSE={result.rmse:.3f} ppm",
        fontsize=13, fontweight="bold",
    )
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


def plot_snr_sensitivity(
    nu: np.ndarray,
    xsec_func,
    N_col_func,
    snr_values: list,
    xco2_true_ppm: float = 435.0,
    xco2_prior_ppm: float = 420.0,
    n_trials_per_snr: int = 200,
    sa: float = 0.10,
    random_seed: int = 0,
    savefig: str = None,
) -> None:
    """
    Show how retrieval precision and bias vary with instrument SNR.

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid.
    xsec_func : callable
        Cross section function (P_Pa, T_K) → array.
    N_col_func : callable
        Column amount function (P_Pa, T_K, xco2_ppm) → float.
    snr_values : list of float
        SNR values to test.
    xco2_true_ppm : float
        True XCO₂ for this experiment.
    xco2_prior_ppm : float
        A priori XCO₂.
    n_trials_per_snr : int
        Noise realisations per SNR level.
    sa : float
        A priori fractional uncertainty.
    random_seed : int
        Base random seed.
    savefig : str, optional
        Save path.
    """
    biases     = []
    precisions = []
    mean_uncs  = []

    for snr in snr_values:
        print(f"  SNR = {snr:.0f} …")
        res = run_monte_carlo(
            nu, xsec_func, N_col_func,
            n_samples=n_trials_per_snr,
            mode="noise_ensemble",
            xco2_fixed_ppm=xco2_true_ppm,
            xco2_prior_ppm=xco2_prior_ppm,
            snr_fixed=snr,
            sa=sa,
            random_seed=random_seed,
            verbose=False,
        )
        biases.append(res.bias)
        precisions.append(res.precision)
        mean_uncs.append(np.mean(res.xco2_unc))

    biases     = np.array(biases)
    precisions = np.array(precisions)
    mean_uncs  = np.array(mean_uncs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(snr_values, precisions, "o-", color="steelblue", lw=2,
                 label="Monte Carlo precision (σ)")
    axes[0].plot(snr_values, mean_uncs,  "s--", color="tomato", lw=2,
                 label="Mean posterior uncertainty")
    axes[0].set_xlabel("SNR", fontsize=12)
    axes[0].set_ylabel("XCO₂ uncertainty [ppm]", fontsize=12)
    axes[0].set_title("Retrieval Precision vs SNR", fontsize=12)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(snr_values, np.abs(biases), "o-", color="seagreen", lw=2)
    axes[1].axhline(0, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("SNR", fontsize=12)
    axes[1].set_ylabel("|Bias| [ppm]", fontsize=12)
    axes[1].set_title("Retrieval |Bias| vs SNR", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("SNR Sensitivity Study", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _add_running_mean(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    color: str = "red",
    label: str = "Running mean",
) -> None:
    """Overlay a running-mean line on a scatter plot."""
    sort_idx = np.argsort(x)
    x_s = x[sort_idx]; y_s = y[sort_idx]
    bins = np.array_split(np.arange(len(x_s)), n_bins)
    x_m = [x_s[b].mean() for b in bins if len(b) > 0]
    y_m = [y_s[b].mean() for b in bins if len(b) > 0]
    ax.plot(x_m, y_m, "-o", color=color, lw=2, ms=5, label=label)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from hitran_model import hitran_cross_section, SYNTHETIC_CO2_LINES
    from radiative_transfer import standard_atmosphere_profile, column_amount

    nu = np.linspace(6210, 6270, 2000)   # smaller grid for speed

    def xsec_func(P, T):
        return hitran_cross_section(nu, SYNTHETIC_CO2_LINES, P, T)

    def N_col_func(P, T, xco2_ppm):
        prof = standard_atmosphere_profile(
            P_surface_Pa=P, T_surface_K=T, co2_vmr_ppm=xco2_ppm)
        return column_amount(prof["number_density"], prof["layer_thickness_cm"])

    print("=== Noise Ensemble (N=300) ===")
    res_noise = run_monte_carlo(
        nu, xsec_func, N_col_func,
        n_samples=300, mode="noise_ensemble",
        xco2_fixed_ppm=430.0, xco2_prior_ppm=420.0, snr_fixed=250.0,
    )
    print(res_noise.summary())
    plot_noise_ensemble(res_noise)

    print("\n=== XCO₂ Sweep (N=80) ===")
    res_sweep = run_monte_carlo(
        nu, xsec_func, N_col_func,
        n_samples=80, mode="xco2_sweep",
        xco2_min_ppm=380.0, xco2_max_ppm=480.0,
    )
    print(res_sweep.summary())
    plot_xco2_sweep(res_sweep)
