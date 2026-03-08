"""
absorption.py
=============
Beer-Lambert absorption modeling for the CO2 Retrieval Simulator.

Physical Basis
--------------
When photons travel through an absorbing gas, their intensity decreases
exponentially with path length. This is described by the Beer-Lambert law:

    I(ν) = I₀(ν) · exp(-σ(ν) · n · L)

where:
    I₀(ν) : incident spectral radiance
    σ(ν)  : molecular absorption cross section [cm²/molecule]
    n     : molecular number density [molecules/cm³]
    L     : path length [cm]
    τ(ν)  = σ(ν) · n · L  is the optical depth (dimensionless)

Author: Arun Kumar Pandey
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union


# ---------------------------------------------------------------------------
# Core Beer-Lambert functions
# ---------------------------------------------------------------------------

def optical_depth(
    cross_section: np.ndarray,
    number_density: float,
    path_length: float,
) -> np.ndarray:
    """
    Compute spectral optical depth τ(ν).

        τ(ν) = σ(ν) · n · L

    Parameters
    ----------
    cross_section : np.ndarray
        Absorption cross section as a function of wavenumber [cm²/molecule].
    number_density : float
        Molecular number density [molecules/cm³].
    path_length : float
        Photon path length through the gas column [cm].

    Returns
    -------
    np.ndarray
        Optical depth τ(ν) [dimensionless].
    """
    return cross_section * number_density * path_length


def transmittance(tau: np.ndarray) -> np.ndarray:
    """
    Compute spectral transmittance T(ν) from optical depth.

        T(ν) = exp(-τ(ν))

    Transmittance ranges from 0 (fully opaque) to 1 (fully transparent).

    Parameters
    ----------
    tau : np.ndarray
        Optical depth τ(ν) [dimensionless].

    Returns
    -------
    np.ndarray
        Spectral transmittance [dimensionless, range 0–1].
    """
    return np.exp(-tau)


def radiance(
    incident: Union[float, np.ndarray],
    tau: np.ndarray,
) -> np.ndarray:
    """
    Compute transmitted spectral radiance via Beer-Lambert law.

        I(ν) = I₀(ν) · exp(-τ(ν))

    Parameters
    ----------
    incident : float or np.ndarray
        Incident radiance I₀(ν). Can be a scalar (flat spectrum) or
        an array matching the shape of tau.
    tau : np.ndarray
        Optical depth τ(ν).

    Returns
    -------
    np.ndarray
        Transmitted radiance I(ν).
    """
    return np.asarray(incident) * np.exp(-tau)


# ---------------------------------------------------------------------------
# Ideal gas: number density from pressure and temperature
# ---------------------------------------------------------------------------

def number_density_from_pT(
    pressure_Pa: float,
    temperature_K: float,
    mole_fraction: float = 1.0,
) -> float:
    """
    Compute molecular number density using the ideal gas law.

        n = (P · x) / (k_B · T)

    where k_B is Boltzmann's constant.

    Parameters
    ----------
    pressure_Pa : float
        Total atmospheric pressure [Pa].
    temperature_K : float
        Temperature [K].
    mole_fraction : float, optional
        Volume mixing ratio of the target gas (default 1.0 = pure gas).
        For CO₂ at ~420 ppm use 420e-6.

    Returns
    -------
    float
        Number density [molecules/cm³].
    """
    k_B = 1.380649e-23          # J/K  (SI)
    n_per_m3 = (pressure_Pa * mole_fraction) / (k_B * temperature_K)
    n_per_cm3 = n_per_m3 * 1e-6  # convert m⁻³ → cm⁻³
    return n_per_cm3


# ---------------------------------------------------------------------------
# Multi-layer optical depth (vertical atmosphere)
# ---------------------------------------------------------------------------

def multilayer_optical_depth(
    cross_section: np.ndarray,
    number_densities: np.ndarray,
    layer_thicknesses_cm: np.ndarray,
) -> np.ndarray:
    """
    Compute total optical depth through a stack of atmospheric layers.

        τ_total(ν) = Σ_i  σ(ν) · nᵢ · Lᵢ

    Each layer may have a different number density (i.e. different
    pressure, temperature, or mixing ratio).

    Parameters
    ----------
    cross_section : np.ndarray, shape (N_nu,)
        Absorption cross section [cm²/molecule] on the spectral grid.
    number_densities : np.ndarray, shape (N_layers,)
        Molecular number density for each layer [molecules/cm³].
    layer_thicknesses_cm : np.ndarray, shape (N_layers,)
        Geometric thickness of each atmospheric layer [cm].

    Returns
    -------
    np.ndarray, shape (N_nu,)
        Total column optical depth.
    """
    tau_total = np.zeros_like(cross_section)
    for n_i, L_i in zip(number_densities, layer_thicknesses_cm):
        tau_total += cross_section * n_i * L_i
    return tau_total


# ---------------------------------------------------------------------------
# Demonstration / plotting helpers
# ---------------------------------------------------------------------------

def demo_beer_lambert(
    nu_grid: np.ndarray,
    cross_section: np.ndarray,
    number_density: float,
    path_lengths_cm: list,
    savefig: str = None,
) -> None:
    """
    Plot Beer-Lambert transmittance for several path lengths to illustrate
    how absorption deepens with increasing column amount.

    Parameters
    ----------
    nu_grid : np.ndarray
        Wavenumber grid [cm⁻¹].
    cross_section : np.ndarray
        Absorption cross section on the wavenumber grid [cm²/molecule].
    number_density : float
        Fixed molecular number density [molecules/cm³].
    path_lengths_cm : list of float
        List of path lengths [cm] to compare.
    savefig : str, optional
        If given, save the figure to this path.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- Top panel: cross section ---
    axes[0].plot(nu_grid, cross_section, color="steelblue", lw=1.5)
    axes[0].set_ylabel("Cross section [cm²/molecule]", fontsize=11)
    axes[0].set_title("Absorption Cross Section and Beer-Lambert Transmittance", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # --- Bottom panel: transmittance for each path length ---
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(path_lengths_cm)))
    for L, c in zip(path_lengths_cm, colors):
        tau = optical_depth(cross_section, number_density, L)
        T   = transmittance(tau)
        axes[1].plot(nu_grid, T, color=c, lw=1.5,
                     label=f"L = {L:.1e} cm")

    axes[1].set_xlabel("Wavenumber [cm⁻¹]", fontsize=11)
    axes[1].set_ylabel("Transmittance", fontsize=11)
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Gaussian cross section centered at 6250 cm⁻¹ (1.6 µm CO₂ band)
    nu   = np.linspace(6200, 6300, 500)
    nu0  = 6250.0
    sigma_nu = 5.0                        # width [cm⁻¹]
    peak_xsec = 3e-22                     # peak cross section [cm²/molecule]
    xsec = peak_xsec * np.exp(-0.5 * ((nu - nu0) / sigma_nu)**2)

    # Typical mid-troposphere CO₂ conditions
    P   = 50000.0    # 500 hPa in Pa
    T   = 250.0      # K
    vmr = 420e-6     # 420 ppm CO₂
    n   = number_density_from_pT(P, T, vmr)
    print(f"CO₂ number density at 500 hPa, 250 K, 420 ppm: {n:.3e} molecules/cm³")

    path_lengths = [1e4, 1e5, 5e5, 1e6]  # cm  (100 m … 10 km)
    demo_beer_lambert(nu, xsec, n, path_lengths)