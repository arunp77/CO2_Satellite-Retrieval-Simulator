"""
hitran_model.py
===============
HITRAN-based CO₂ absorption spectrum simulator for the CO2 Retrieval Simulator.

Physical Basis
--------------
Real satellite retrieval algorithms use the HITRAN (HIgh-resolution TRANsmission)
molecular spectroscopic database, which tabulates for each absorption line:

    - ν₀         : line centre wavenumber [cm⁻¹]
    - S(T_ref)   : line strength at reference temperature T_ref=296 K [cm/molecule]
    - γ_air      : air-broadened HWHM [cm⁻¹/atm]
    - γ_self     : self-broadened HWHM [cm⁻¹/atm]
    - E''        : lower-state energy [cm⁻¹]
    - n_air      : temperature exponent

Line strength temperature correction:
    S(T) = S(T_ref) · [Q(T_ref)/Q(T)] · exp(-hcE''/kT) / exp(-hcE''/kT_ref)
           · [1 - exp(-hcν₀/kT)] / [1 - exp(-hcν₀/kT_ref)]

where Q(T) is the total internal partition function.

This module provides:
  1. A built-in synthetic line list for the CO₂ 1.6 µm band (for offline use)
  2. Optional HAPI (HITRAN Application Programming Interface) integration
  3. Absorption cross section computation
  4. Spectrum visualisation

Author: Arun Kumar Pandey
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# Optional HAPI import
try:
    import hapi
    HAPI_AVAILABLE = True
except ImportError:
    HAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
h   = 6.62607015e-34   # Planck constant [J·s]
k_B = 1.380649e-23     # Boltzmann constant [J/K]
c   = 2.99792458e10    # speed of light [cm/s]
T_ref = 296.0          # HITRAN reference temperature [K]


# ---------------------------------------------------------------------------
# Built-in synthetic HITRAN-like CO₂ line list  (1.6 µm band, ~6200-6280 cm⁻¹)
# ---------------------------------------------------------------------------
# These values are representative (not exact) for demonstration purposes.
# Columns: nu0 [cm⁻¹], S [cm/mol], gamma_air [cm⁻¹/atm], E_lower [cm⁻¹], n_air

SYNTHETIC_CO2_LINES = np.array([
    # ν₀        S(296K)     γ_air    E''      n_air
    [6215.78,  1.80e-25,   0.0752,  721.17,  0.76],
    [6218.23,  3.50e-25,   0.0741,  667.38,  0.76],
    [6220.11,  9.20e-25,   0.0765,  596.47,  0.76],
    [6222.45,  2.10e-24,   0.0758,  518.45,  0.76],
    [6225.90,  4.80e-24,   0.0749,  427.31,  0.76],
    [6228.35,  7.90e-24,   0.0762,  337.19,  0.76],
    [6230.01,  1.15e-23,   0.0771,  241.77,  0.76],
    [6232.78,  1.35e-23,   0.0756,  155.22,  0.76],
    [6235.44,  1.20e-23,   0.0748,  667.38,  0.76],
    [6237.81,  9.30e-24,   0.0739,  596.47,  0.76],
    [6240.55,  1.55e-23,   0.0760,  241.77,  0.76],
    [6243.12,  1.10e-23,   0.0753,  337.19,  0.76],
    [6245.60,  8.70e-24,   0.0744,  427.31,  0.76],
    [6247.88,  6.20e-24,   0.0767,  518.45,  0.76],
    [6249.50,  4.10e-24,   0.0751,  596.47,  0.76],
    [6251.90,  1.25e-23,   0.0759,  155.22,  0.76],
    [6254.23,  8.50e-24,   0.0742,  241.77,  0.76],
    [6256.77,  5.90e-24,   0.0773,  337.19,  0.76],
    [6259.10,  3.80e-24,   0.0755,  427.31,  0.76],
    [6261.45,  2.40e-24,   0.0746,  518.45,  0.76],
    [6264.01,  1.40e-24,   0.0762,  596.47,  0.76],
    [6266.78,  7.80e-25,   0.0738,  667.38,  0.76],
    [6268.50,  4.10e-25,   0.0753,  721.17,  0.76],
    [6271.20,  2.10e-25,   0.0749,  780.33,  0.76],
])


# ---------------------------------------------------------------------------
# Line strength temperature correction
# ---------------------------------------------------------------------------

def correct_line_strength(
    S_ref: np.ndarray,
    nu0: np.ndarray,
    E_lower: np.ndarray,
    temperature_K: float,
    partition_ratio: float = 1.0,
) -> np.ndarray:
    """
    Apply HITRAN temperature correction to line strengths.

        S(T) = S(T_ref) · (T_ref/T)^(partition_ratio_term)
               · exp(-hcE''/k · (1/T - 1/T_ref))
               · [1 - exp(-hcν₀/kT)] / [1 - exp(-hcν₀/kT_ref)]

    For a simplified model (ignoring partition function details):

    Parameters
    ----------
    S_ref : np.ndarray
        Line strengths at T_ref = 296 K [cm/molecule].
    nu0 : np.ndarray
        Line centre wavenumbers [cm⁻¹].
    E_lower : np.ndarray
        Lower-state energies [cm⁻¹].
    temperature_K : float
        Target temperature [K].
    partition_ratio : float
        Q(T_ref)/Q(T).  Set to 1 for demonstration (slight overestimate at
        temperatures far from 296 K).

    Returns
    -------
    np.ndarray
        Temperature-corrected line strengths [cm/molecule].
    """
    T = temperature_K
    hc_k = h * c / k_B   # hc/k  in units K·cm

    # Boltzmann population factor
    boltzmann = np.exp(-hc_k * E_lower * (1.0/T - 1.0/T_ref))

    # Stimulated emission correction
    stim_T   = 1.0 - np.exp(-hc_k * nu0 / T)
    stim_ref = 1.0 - np.exp(-hc_k * nu0 / T_ref)

    return S_ref * partition_ratio * boltzmann * (stim_T / stim_ref)


# ---------------------------------------------------------------------------
# Build absorption cross section from HITRAN line list
# ---------------------------------------------------------------------------

def hitran_cross_section(
    nu: np.ndarray,
    line_data: np.ndarray,
    pressure_Pa: float,
    temperature_K: float,
    profile: str = "voigt",
    cutoff_cm: float = 25.0,
) -> np.ndarray:
    """
    Compute CO₂ absorption cross section from HITRAN-format line list.

        k(ν) = Σ_i  S_i(T) · φ_i(ν)

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    line_data : np.ndarray, shape (N_lines, 5)
        Array with columns [nu0, S_ref, gamma_air, E_lower, n_air].
    pressure_Pa : float
        Atmospheric pressure [Pa].
    temperature_K : float
        Temperature [K].
    profile : str
        Line profile: 'voigt', 'gaussian', or 'lorentz'.
    cutoff_cm : float
        Only include lines whose centre is within this many cm⁻¹ of the
        spectral window (speeds up computation).

    Returns
    -------
    np.ndarray
        Absorption cross section [cm²/molecule] on the nu grid.
    """
    from spectroscopy import (
        voigt_profile_func,
        gaussian_profile,
        lorentz_profile,
        doppler_hwhm,
        lorentz_hwhm,
    )

    nu_min = nu.min() - cutoff_cm
    nu_max = nu.max() + cutoff_cm

    xsec = np.zeros_like(nu)

    for row in line_data:
        nu0_i, S_ref_i, gamma_air_i, E_lower_i, n_air_i = row

        # Skip lines far outside the spectral window
        if nu0_i < nu_min or nu0_i > nu_max:
            continue

        # Temperature-corrected line strength
        S_i = correct_line_strength(
            np.array([S_ref_i]),
            np.array([nu0_i]),
            np.array([E_lower_i]),
            temperature_K,
        )[0]

        # Line widths
        dnu_D = doppler_hwhm(nu0_i, temperature_K)
        dnu_L = lorentz_hwhm(gamma_air_i, pressure_Pa, temperature_K, n_air=n_air_i)

        # Profile
        if profile == "voigt":
            phi = voigt_profile_func(nu, nu0_i, dnu_D, dnu_L)
        elif profile == "gaussian":
            phi = gaussian_profile(nu, nu0_i, dnu_D)
        elif profile == "lorentz":
            phi = lorentz_profile(nu, nu0_i, dnu_L)
        else:
            raise ValueError(f"Unknown profile: {profile!r}")

        xsec += S_i * phi

    return xsec


# ---------------------------------------------------------------------------
# Optional HAPI integration
# ---------------------------------------------------------------------------

def download_hitran_co2(
    nu_min: float,
    nu_max: float,
    data_dir: str = "./data",
    table_name: str = "CO2",
) -> None:
    """
    Download CO₂ line data from HITRAN via HAPI (requires internet + hapi package).

    Parameters
    ----------
    nu_min, nu_max : float
        Wavenumber range [cm⁻¹].
    data_dir : str
        Local directory for HITRAN data files.
    table_name : str
        HAPI table name.
    """
    if not HAPI_AVAILABLE:
        print("HAPI not installed. Install with: pip install hitran-api")
        return

    import os
    os.makedirs(data_dir, exist_ok=True)
    hapi.db_begin(data_dir)
    # Molecule ID 2 = CO₂, Isotopologue 1 = 12C16O2
    hapi.fetch(table_name, 2, 1, nu_min, nu_max)
    print(f"HITRAN CO₂ data downloaded: {nu_min}–{nu_max} cm⁻¹  →  {data_dir}/")


def hapi_cross_section(
    nu: np.ndarray,
    pressure_atm: float,
    temperature_K: float,
    table_name: str = "CO2",
    data_dir: str = "./data",
) -> np.ndarray:
    """
    Compute CO₂ cross section using HAPI (if available).

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    pressure_atm : float
        Pressure [atm].
    temperature_K : float
        Temperature [K].
    table_name, data_dir : str
        HAPI table name and data directory (must match download_hitran_co2).

    Returns
    -------
    np.ndarray
        Absorption cross section [cm²/molecule] on the nu grid, or zeros
        if HAPI is unavailable.
    """
    if not HAPI_AVAILABLE:
        print("HAPI not available — returning synthetic cross section instead.")
        return hitran_cross_section(nu, SYNTHETIC_CO2_LINES, pressure_atm * 101325.0, temperature_K)

    hapi.db_begin(data_dir)
    nu_out, coeff = hapi.absorptionCoefficient_Voigt(
        SourceTables=table_name,
        HITRAN_units=False,
        GammaL="gamma_air",
        Diluent={"air": 1.0},
        WavenumberRange=[nu.min(), nu.max()],
        WavenumberStep=(nu[1] - nu[0]),
        Environment={"p": pressure_atm, "T": temperature_K},
    )
    # Interpolate onto our nu grid
    return np.interp(nu, nu_out, coeff)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_hitran_spectrum(
    nu: np.ndarray,
    xsec: np.ndarray,
    line_data: np.ndarray,
    title: str = "CO₂ Absorption Spectrum (1.6 µm band)",
    savefig: str = None,
) -> None:
    """
    Plot absorption cross section with line markers.

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    xsec : np.ndarray
        Cross section [cm²/molecule].
    line_data : np.ndarray
        HITRAN-format line array (columns: nu0, S, …).
    title : str
        Plot title.
    savefig : str, optional
        Save path.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(nu, xsec, color="steelblue", lw=1.0)
    ax1.set_ylabel("Cross section [cm²/molecule]", fontsize=11)
    ax1.set_title(title, fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Line stick spectrum in lower panel
    nu_in_range = line_data[:, 0]
    S_in_range  = line_data[:, 1]
    mask = (nu_in_range >= nu.min()) & (nu_in_range <= nu.max())
    ax2.vlines(nu_in_range[mask], 0, S_in_range[mask],
               color="tomato", lw=1.2, alpha=0.8)
    ax2.set_xlabel("Wavenumber [cm⁻¹]", fontsize=11)
    ax2.set_ylabel("S [cm/mol]", fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


def plot_spectrum_sensitivity(
    nu: np.ndarray,
    line_data: np.ndarray,
    co2_concentrations_ppm: list,
    pressure_Pa: float = 101325.0,
    temperature_K: float = 288.0,
    N_col_dry: float = 2.15e25,
    savefig: str = None,
) -> None:
    """
    Show how the absorption spectrum changes with CO₂ concentration.

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    line_data : np.ndarray
        HITRAN-format line data.
    co2_concentrations_ppm : list of float
        CO₂ concentrations to compare [ppm].
    pressure_Pa : float
        Pressure [Pa].
    temperature_K : float
        Temperature [K].
    N_col_dry : float
        Dry-air column amount [molecules/cm²].
    savefig : str, optional
        Save path.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(co2_concentrations_ppm)))

    for ppm, col in zip(co2_concentrations_ppm, colors):
        xsec  = hitran_cross_section(nu, line_data, pressure_Pa, temperature_K)
        N_col = N_col_dry * ppm * 1e-6
        tau   = xsec * N_col * 2.0     # two-way
        T_atm = np.exp(-tau)
        ax.plot(nu, T_atm, color=col, lw=1.5, label=f"{ppm:.0f} ppm")

    ax.set_xlabel("Wavenumber [cm⁻¹]", fontsize=12)
    ax.set_ylabel("Transmittance", fontsize=12)
    ax.set_title("CO₂ Spectral Sensitivity to Concentration", fontsize=13)
    ax.legend(title="[CO₂]", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nu = np.linspace(6210, 6270, 5000)

    P = 101325.0   # surface pressure [Pa]
    T = 288.0      # surface temperature [K]

    print("Computing HITRAN-based CO₂ cross section …")
    xsec = hitran_cross_section(nu, SYNTHETIC_CO2_LINES, P, T)
    print(f"  Peak cross section: {xsec.max():.3e} cm²/molecule")

    plot_hitran_spectrum(nu, xsec, SYNTHETIC_CO2_LINES)

    plot_spectrum_sensitivity(
        nu, SYNTHETIC_CO2_LINES,
        co2_concentrations_ppm=[350, 400, 420, 450, 500],
        pressure_Pa=P,
        temperature_K=T,
    )