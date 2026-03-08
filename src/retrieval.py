"""
retrieval.py
============
XCO₂ retrieval module for the CO2 Retrieval Simulator.

Physical Basis
--------------
Satellite missions retrieve the **column-averaged dry-air mole fraction** of CO₂:

    XCO₂ = ∫ n_CO₂(z) dz / ∫ n_dry(z) dz   [ppm or mol/mol]

This is the quantity most directly comparable between ground stations
and satellites, because it is insensitive to surface pressure variations.

Retrieval Framework: Optimal Estimation (Rodgers 2000)
------------------------------------------------------
The forward model F(x) maps a state vector x (here: XCO₂ scalar, or CO₂
column scaling factor) to a simulated spectrum.

The optimal estimate x̂ minimises the cost function:

    J(x) = (y - F(x))ᵀ Sε⁻¹ (y - F(x))  +  (x - xa)ᵀ Sa⁻¹ (x - xa)

where:
    y    : observed radiance (measured by satellite)
    F(x) : forward model spectrum
    Sε   : measurement error covariance
    xa   : a priori state
    Sa   : a priori covariance

Solution (linear approximation about xa):
    x̂ = xa + (Kᵀ Sε⁻¹ K + Sa⁻¹)⁻¹ Kᵀ Sε⁻¹ (y - F(xa))

where K = ∂F/∂x is the Jacobian (weighting function).

For a **scalar** retrieval (single CO₂ scaling factor ξ, so XCO₂ = ξ · XCO₂_a),
this reduces to a simple closed-form solution, implemented below.

Author: Arun Kumar Pandey
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Simple scalar XCO₂ retrieval
# ---------------------------------------------------------------------------

def compute_jacobian(
    nu: np.ndarray,
    radiance_ref: np.ndarray,
    cross_section: np.ndarray,
    N_col_ref: float,
    air_mass: float,
    delta_scale: float = 0.01,
) -> np.ndarray:
    """
    Compute the Jacobian K = ∂I(ν)/∂ξ  (finite-difference approximation).

    The state variable ξ scales the reference CO₂ column:
        N_col(ξ) = ξ · N_col_ref

    So K(ν) = ∂I/∂ξ ≈ [I(ξ+δ) - I(ξ)] / δ

    For the Beer-Lambert forward model:
        I(ξ) = I_ref · exp(-air_mass · σ · N_col_ref · ξ)

    The analytical Jacobian is:
        K(ν) = -air_mass · σ(ν) · N_col_ref · I(ξ)

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    radiance_ref : np.ndarray
        Radiance at ξ = 1 (reference CO₂ amount).
    cross_section : np.ndarray
        CO₂ absorption cross section [cm²/molecule].
    N_col_ref : float
        Reference CO₂ column amount [molecules/cm²].
    air_mass : float
        Two-way air-mass factor.
    delta_scale : float
        Finite-difference step in ξ (default 1%).

    Returns
    -------
    np.ndarray, shape (N_nu,)
        Jacobian K [radiance per unit ξ].
    """
    # Analytical form
    K = -air_mass * cross_section * N_col_ref * radiance_ref
    return K


def optimal_estimation_scalar(
    y_obs: np.ndarray,
    y_ref: np.ndarray,
    K: np.ndarray,
    sigma_noise: float,
    xa: float = 1.0,
    sa: float = 0.10,
) -> Tuple[float, float, float]:
    """
    Scalar optimal estimation retrieval of CO₂ scaling factor ξ.

    Minimises:
        J(ξ) = ||y - y_ref - K(ξ-1)||² / σ_noise²  +  (ξ-xa)² / sa²

    Analytical solution:
        ξ̂ = xa  +  (Kᵀ Sε⁻¹ K + Sa⁻¹)⁻¹  Kᵀ Sε⁻¹  (y - y_ref)

    Posterior uncertainty:
        σ_post = 1 / √(Kᵀ Sε⁻¹ K + Sa⁻¹)

    Parameters
    ----------
    y_obs : np.ndarray
        Observed radiance spectrum.
    y_ref : np.ndarray
        Forward model at xa (a priori spectrum).
    K : np.ndarray
        Jacobian vector K = ∂I/∂ξ.
    sigma_noise : float
        Measurement noise standard deviation (scalar, assumed uniform).
    xa : float
        A priori scaling factor (default 1.0 = prior column is correct).
    sa : float
        A priori uncertainty (standard deviation), e.g. 0.10 = 10%.

    Returns
    -------
    xi_hat : float
        Retrieved scaling factor.
    sigma_post : float
        Posterior (retrieval) uncertainty on ξ.
    degrees_of_freedom : float
        Degrees of freedom for signal (DFS = AKᵀ trace ≈ 1 for scalar).
    """
    Se_inv = 1.0 / sigma_noise**2
    Sa_inv = 1.0 / sa**2

    # Kᵀ Sε⁻¹ K  (scalar)
    KtSeK = Se_inv * np.dot(K, K)

    # A posteriori precision
    S_post_inv = KtSeK + Sa_inv

    # Optimal update
    delta_y = y_obs - y_ref
    xi_hat = xa + (Se_inv * np.dot(K, delta_y)) / S_post_inv

    sigma_post = 1.0 / np.sqrt(S_post_inv)
    dfs = KtSeK / S_post_inv      # degrees of freedom for signal

    return xi_hat, sigma_post, dfs


# ---------------------------------------------------------------------------
# Iterative Levenberg-Marquardt retrieval (non-linear extension)
# ---------------------------------------------------------------------------

def iterative_retrieval(
    nu: np.ndarray,
    y_obs: np.ndarray,
    forward_func,
    xa: float,
    sa: float,
    sigma_noise: float,
    max_iter: int = 10,
    gamma: float = 10.0,
    convergence_threshold: float = 1e-4,
) -> dict:
    """
    Iterative Gauss-Newton / Levenberg-Marquardt CO₂ scaling factor retrieval.

    At each iteration i:
        K_i   = ∂F(x_i)/∂x  (finite difference)
        Δx    = (K_i Sε⁻¹ K_i + Sa⁻¹ + γ·Sa⁻¹)⁻¹
                [K_i Sε⁻¹ (y - F(x_i)) - Sa⁻¹ (x_i - xa)]
        x_{i+1} = x_i + Δx

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid.
    y_obs : np.ndarray
        Observed radiance.
    forward_func : callable
        Function  f(xi) → simulated radiance array, where xi is the CO₂
        scaling factor.
    xa : float
        A priori CO₂ scaling factor.
    sa : float
        A priori uncertainty (std dev).
    sigma_noise : float
        Measurement noise standard deviation.
    max_iter : int
        Maximum number of iterations.
    gamma : float
        Levenberg-Marquardt damping parameter (reduces toward 0 on convergence).
    convergence_threshold : float
        Convergence criterion on |Δx|.

    Returns
    -------
    dict with keys:
        'xi_hat', 'sigma_post', 'n_iter', 'cost_history', 'xi_history'
    """
    Se_inv = 1.0 / sigma_noise**2
    Sa_inv = 1.0 / sa**2

    xi = xa
    cost_history = []
    xi_history   = [xa]

    for iteration in range(max_iter):
        F_xi    = forward_func(xi)
        residual = y_obs - F_xi

        # Finite-difference Jacobian
        dxi     = xi * 0.01 if xi != 0 else 1e-4
        F_xi_p  = forward_func(xi + dxi)
        K       = (F_xi_p - F_xi) / dxi

        # Cost function value
        cost = Se_inv * np.dot(residual, residual) + Sa_inv * (xi - xa)**2
        cost_history.append(cost)

        # Gauss-Newton step with LM damping
        KtSeK    = Se_inv * np.dot(K, K)
        KtSer    = Se_inv * np.dot(K, residual)
        prior_term = Sa_inv * (xi - xa)
        denom    = KtSeK + Sa_inv + gamma * Sa_inv
        delta_xi = (KtSer - prior_term) / denom

        xi_new = xi + delta_xi
        F_new  = forward_func(xi_new)
        cost_new = (Se_inv * np.dot((y_obs - F_new), (y_obs - F_new))
                    + Sa_inv * (xi_new - xa)**2)

        if cost_new < cost:
            xi = xi_new
            gamma = max(gamma / 3.0, 1e-6)
        else:
            gamma = min(gamma * 3.0, 1e6)

        xi_history.append(xi)

        if abs(delta_xi) < convergence_threshold:
            break

    # Posterior uncertainty at final state
    F_final    = forward_func(xi)
    dxi        = xi * 0.01 if xi != 0 else 1e-4
    K_final    = (forward_func(xi + dxi) - F_final) / dxi
    KtSeK_final = Se_inv * np.dot(K_final, K_final)
    sigma_post  = 1.0 / np.sqrt(KtSeK_final + Sa_inv)

    return {
        "xi_hat":       xi,
        "sigma_post":   sigma_post,
        "n_iter":       iteration + 1,
        "cost_history": np.array(cost_history),
        "xi_history":   np.array(xi_history),
    }


# ---------------------------------------------------------------------------
# XCO₂ from retrieved scaling factor
# ---------------------------------------------------------------------------

def xco2_from_scaling(
    xi_hat: float,
    xco2_prior_ppm: float,
    sigma_post_xi: float,
) -> Tuple[float, float]:
    """
    Convert retrieved CO₂ scaling factor to XCO₂ in ppm.

        XCO₂_retrieved = ξ̂ · XCO₂_prior

    Parameters
    ----------
    xi_hat : float
        Retrieved scaling factor.
    xco2_prior_ppm : float
        A priori XCO₂ [ppm].
    sigma_post_xi : float
        Posterior uncertainty on ξ.

    Returns
    -------
    xco2_ppm : float
        Retrieved XCO₂ [ppm].
    xco2_uncertainty_ppm : float
        1-σ retrieval uncertainty [ppm].
    """
    xco2_ppm         = xi_hat * xco2_prior_ppm
    xco2_uncertainty = sigma_post_xi * xco2_prior_ppm
    return xco2_ppm, xco2_uncertainty


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_retrieval_result(
    nu: np.ndarray,
    y_obs: np.ndarray,
    y_prior: np.ndarray,
    y_retrieved: np.ndarray,
    xco2_true: Optional[float],
    xco2_prior: float,
    xco2_ret: float,
    xco2_unc: float,
    cost_history: Optional[np.ndarray] = None,
    savefig: str = None,
) -> None:
    """
    Plot observed vs. forward model spectra and retrieval convergence.

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid.
    y_obs, y_prior, y_retrieved : np.ndarray
        Observed, a priori, and retrieved radiance spectra.
    xco2_true : float or None
        True XCO₂ (for simulation validation).
    xco2_prior, xco2_ret, xco2_unc : float
        Prior, retrieved, and uncertainty [ppm].
    cost_history : np.ndarray, optional
        Cost function per iteration.
    savefig : str, optional
        Save path.
    """
    ncols = 2 if cost_history is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(14 if ncols == 2 else 9, 5))
    if ncols == 1:
        axes = [axes]

    # --- Spectral fit ---
    ax = axes[0]
    ax.plot(nu, y_obs,       lw=0.8, color="gray",       alpha=0.7, label="Observed")
    ax.plot(nu, y_prior,     lw=1.5, color="royalblue",  ls="--",   label=f"A priori  ({xco2_prior:.1f} ppm)")
    ax.plot(nu, y_retrieved, lw=1.5, color="tomato",               label=f"Retrieved ({xco2_ret:.1f} ± {xco2_unc:.1f} ppm)")
    if xco2_true is not None:
        ax.set_title(f"Spectral Fit  |  True XCO₂ = {xco2_true:.1f} ppm", fontsize=12)
    else:
        ax.set_title("Spectral Fit", fontsize=12)
    ax.set_xlabel("Wavenumber [cm⁻¹]", fontsize=11)
    ax.set_ylabel("Radiance (normalised)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Convergence ---
    if cost_history is not None:
        axes[1].semilogy(np.arange(1, len(cost_history) + 1), cost_history,
                         "o-", color="seagreen", lw=2)
        axes[1].set_xlabel("Iteration", fontsize=11)
        axes[1].set_ylabel("Cost function J(ξ)", fontsize=11)
        axes[1].set_title("Retrieval Convergence", fontsize=12)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


def plot_xco2_sensitivity(
    nu: np.ndarray,
    forward_func,
    xco2_values_ppm: list,
    xco2_ref_ppm: float = 420.0,
    savefig: str = None,
) -> None:
    """
    Show how retrieved spectra differ for different XCO₂ concentrations.

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid.
    forward_func : callable
        Function  f(xi) → radiance, as in iterative_retrieval.
    xco2_values_ppm : list
        XCO₂ concentrations [ppm].
    xco2_ref_ppm : float
        Reference concentration (xi=1).
    savefig : str, optional
        Save path.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(xco2_values_ppm)))

    for ppm, col in zip(xco2_values_ppm, colors):
        xi = ppm / xco2_ref_ppm
        I  = forward_func(xi)
        ax.plot(nu, I, color=col, lw=1.5, label=f"{ppm:.0f} ppm")

    ax.set_xlabel("Wavenumber [cm⁻¹]", fontsize=12)
    ax.set_ylabel("Radiance (normalised)", fontsize=12)
    ax.set_title("Sensitivity of Satellite Radiance to XCO₂", fontsize=13)
    ax.legend(title="XCO₂", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from spectroscopy import absorption_cross_section, doppler_hwhm, lorentz_hwhm
    from radiative_transfer import (
        forward_model,
        solar_irradiance,
        standard_atmosphere_profile,
        column_amount,
        air_mass_factor,
    )
    from hitran_model import hitran_cross_section, SYNTHETIC_CO2_LINES

    # --- Setup ---
    nu   = np.linspace(6210, 6270, 3000)
    P    = 101325.0
    T    = 288.0
    xsec = hitran_cross_section(nu, SYNTHETIC_CO2_LINES, P, T)

    profile  = standard_atmosphere_profile(co2_vmr_ppm=420.0)
    N_col_ref = column_amount(profile["number_density"], profile["layer_thickness_cm"])
    amf      = air_mass_factor(30.0, 0.0)

    xco2_prior_ppm = 420.0
    xco2_true_ppm  = 435.0
    xi_true        = xco2_true_ppm / xco2_prior_ppm

    # --- True and prior spectra ---
    res_prior = forward_model(nu, xsec, N_col_ref, amf, surface_albedo=0.25,
                              solar_zenith_deg=30.0)
    I_prior   = res_prior["radiance"]

    # Synthetic observation: scale column by xi_true, then add noise
    N_col_true = N_col_ref * xi_true
    res_true   = forward_model(nu, xsec, N_col_true, amf, surface_albedo=0.25,
                               solar_zenith_deg=30.0, add_noise=True, snr=200.0)
    y_obs = res_true["radiance_noisy"]

    snr     = 200.0
    I_max   = I_prior.max()
    noise_σ = I_max / snr

    # --- Forward function for iterative retrieval ---
    def fwd(xi_val):
        N = N_col_ref * xi_val
        r = forward_model(nu, xsec, N, amf, surface_albedo=0.25,
                          solar_zenith_deg=30.0)
        return r["radiance"]

    # --- Iterative retrieval ---
    result = iterative_retrieval(
        nu, y_obs, fwd,
        xa=1.0, sa=0.10,
        sigma_noise=noise_σ,
        max_iter=15,
    )

    xi_hat     = result["xi_hat"]
    sigma_post = result["sigma_post"]
    xco2_ret, xco2_unc = xco2_from_scaling(xi_hat, xco2_prior_ppm, sigma_post)

    print(f"\n--- XCO₂ Retrieval Result ---")
    print(f"  True XCO₂   : {xco2_true_ppm:.1f} ppm")
    print(f"  Prior XCO₂  : {xco2_prior_ppm:.1f} ppm")
    print(f"  Retrieved   : {xco2_ret:.2f} ± {xco2_unc:.2f} ppm")
    print(f"  Converged in {result['n_iter']} iterations")
    print(f"  ξ̂ = {xi_hat:.4f}  (true ξ = {xi_true:.4f})")

    I_ret = fwd(xi_hat)

    plot_retrieval_result(
        nu, y_obs, I_prior, I_ret,
        xco2_true=xco2_true_ppm,
        xco2_prior=xco2_prior_ppm,
        xco2_ret=xco2_ret,
        xco2_unc=xco2_unc,
        cost_history=result["cost_history"],
    )
