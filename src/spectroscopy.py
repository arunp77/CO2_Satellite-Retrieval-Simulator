"""
spectroscopy.py
===============
Molecular spectral line profile modeling for the CO2 Retrieval Simulator.

Physical Basis
--------------
Real molecular absorption lines are not infinitely narrow — they are broadened
by two main physical mechanisms:

1. **Doppler broadening** (thermal motion of molecules)
   Each molecule moves at a random velocity. The Doppler shift of its
   absorbed frequency produces a Gaussian line shape:

       φ_D(ν) = (1 / (Δν_D √π)) · exp(-(ν-ν₀)²/Δν_D²)

   where the Doppler half-width is:
       Δν_D = (ν₀/c) · √(2k_B T ln2 / m)

2. **Pressure (collisional) broadening**
   Frequent molecular collisions interrupt the absorption process,
   producing a Lorentzian (Cauchy) line shape:

       φ_L(ν) = (Δν_L/π) / ((ν-ν₀)² + Δν_L²)

   The Lorentzian half-width at half maximum (HWHM) scales with pressure:
       Δν_L = γ_air · (P / P_ref)^n_air · (T_ref/T)^δ

3. **Voigt profile** — the physical line shape in the real atmosphere
   is the convolution of the Doppler and Lorentzian profiles:

       φ_V(ν) = ∫ φ_D(ν') · φ_L(ν-ν') dν'

   This is computed efficiently via the Faddeeva / Humlíček w-function.

The absorption cross section at wavenumber ν is then:

    σ(ν) = Σ_i  S_i · φ_{V,i}(ν)

where S_i is the line strength of the i-th transition.

Author: Arun Kumar Pandey
"""

import numpy as np
from scipy.special import voigt_profile      # available in SciPy ≥ 1.5
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
k_B = 1.380649e-23     # Boltzmann constant  [J/K]
c   = 2.99792458e10    # Speed of light       [cm/s]
N_A = 6.02214076e23    # Avogadro number      [mol⁻¹]
amu = 1.66053906660e-27  # atomic mass unit   [kg]

# Molar mass of CO₂  [kg/mol]
M_CO2 = 44.01e-3


# ---------------------------------------------------------------------------
# Doppler line width
# ---------------------------------------------------------------------------

def doppler_hwhm(
    nu0_cm: float,
    temperature_K: float,
    molar_mass_kg: float = M_CO2,
) -> float:
    """
    Compute the Doppler (Gaussian) half-width at half maximum (HWHM) [cm⁻¹].

        Δν_D = (ν₀/c) · √(2 k_B T ln2 / m)

    where m = M / N_A  is the mass of one molecule.

    Parameters
    ----------
    nu0_cm : float
        Line centre wavenumber [cm⁻¹].
    temperature_K : float
        Temperature [K].
    molar_mass_kg : float
        Molar mass of the molecule [kg/mol].  Default: CO₂ = 44.01 g/mol.

    Returns
    -------
    float
        Doppler HWHM [cm⁻¹].
    """
    m_kg = molar_mass_kg / N_A          # mass per molecule [kg]
    delta_nu_D = (nu0_cm / c) * np.sqrt(2.0 * k_B * temperature_K * np.log(2) / m_kg)
    return delta_nu_D


# ---------------------------------------------------------------------------
# Lorentzian (pressure) line width
# ---------------------------------------------------------------------------

def lorentz_hwhm(
    gamma_air: float,
    pressure_Pa: float,
    temperature_K: float,
    n_air: float = 0.76,
    P_ref: float = 101325.0,
    T_ref: float = 296.0,
) -> float:
    """
    Compute the Lorentzian (pressure broadening) HWHM [cm⁻¹].

        Δν_L = γ_air · (P / P_ref)^n_air · (T_ref / T)^δ

    Parameters
    ----------
    gamma_air : float
        Air-broadened half-width at reference conditions [cm⁻¹/atm].
        Typical HITRAN value for CO₂ ~ 0.07 cm⁻¹/atm.
    pressure_Pa : float
        Atmospheric pressure [Pa].
    temperature_K : float
        Temperature [K].
    n_air : float
        Temperature exponent (default 0.76, HITRAN convention).
    P_ref : float
        Reference pressure [Pa] (default 1 atm = 101 325 Pa).
    T_ref : float
        Reference temperature [K] (default 296 K).

    Returns
    -------
    float
        Lorentzian HWHM [cm⁻¹].
    """
    P_atm = pressure_Pa / 101325.0      # convert Pa → atm (γ_air is per atm)
    P_ref_atm = P_ref / 101325.0
    delta_nu_L = gamma_air * (P_atm / P_ref_atm)**n_air * (T_ref / temperature_K)**n_air
    return delta_nu_L


# ---------------------------------------------------------------------------
# Line profile functions
# ---------------------------------------------------------------------------

def gaussian_profile(
    nu: np.ndarray,
    nu0: float,
    delta_nu_D: float,
) -> np.ndarray:
    """
    Normalised Gaussian (Doppler) line profile [cm].

        φ_D(ν) = (√(ln2/π) / Δν_D) · exp(-ln2 · (ν-ν₀)²/Δν_D²)

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    nu0 : float
        Line centre [cm⁻¹].
    delta_nu_D : float
        Doppler HWHM [cm⁻¹].

    Returns
    -------
    np.ndarray
        Normalised profile [cm] (integrates to 1 over wavenumber).
    """
    x = (nu - nu0) / delta_nu_D
    return (np.sqrt(np.log(2) / np.pi) / delta_nu_D) * np.exp(-np.log(2) * x**2)


def lorentz_profile(
    nu: np.ndarray,
    nu0: float,
    delta_nu_L: float,
) -> np.ndarray:
    """
    Normalised Lorentzian (pressure) line profile [cm].

        φ_L(ν) = (Δν_L/π) / ((ν-ν₀)² + Δν_L²)

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    nu0 : float
        Line centre [cm⁻¹].
    delta_nu_L : float
        Lorentzian HWHM [cm⁻¹].

    Returns
    -------
    np.ndarray
        Normalised profile [cm].
    """
    return (delta_nu_L / np.pi) / ((nu - nu0)**2 + delta_nu_L**2)


def voigt_profile_func(
    nu: np.ndarray,
    nu0: float,
    delta_nu_D: float,
    delta_nu_L: float,
) -> np.ndarray:
    """
    Normalised Voigt profile — convolution of Gaussian and Lorentzian.

    Uses SciPy's ``voigt_profile(x, sigma, gamma)`` which takes:
        sigma = Δν_D / √(2 ln2)    (Gaussian standard deviation)
        gamma = Δν_L               (Lorentzian HWHM)

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    nu0 : float
        Line centre [cm⁻¹].
    delta_nu_D : float
        Doppler HWHM [cm⁻¹].
    delta_nu_L : float
        Lorentzian HWHM [cm⁻¹].

    Returns
    -------
    np.ndarray
        Normalised Voigt profile [cm].
    """
    sigma = delta_nu_D / np.sqrt(2.0 * np.log(2))  # Gaussian σ (std dev)
    x     = nu - nu0
    return voigt_profile(x, sigma, delta_nu_L)


# ---------------------------------------------------------------------------
# Absorption cross section from a set of spectral lines
# ---------------------------------------------------------------------------

def absorption_cross_section(
    nu: np.ndarray,
    line_positions: np.ndarray,
    line_strengths: np.ndarray,
    delta_nu_D: float,
    delta_nu_L: float,
    profile: str = "voigt",
) -> np.ndarray:
    """
    Compute absorption cross section σ(ν) as a sum of line profiles.

        σ(ν) = Σ_i  S_i · φ_i(ν)

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    line_positions : np.ndarray
        Line centre wavenumbers [cm⁻¹].
    line_strengths : np.ndarray
        Line strengths Sᵢ [cm/molecule].
    delta_nu_D : float
        Doppler HWHM [cm⁻¹] (same for all lines — simplification).
    delta_nu_L : float
        Lorentzian HWHM [cm⁻¹].
    profile : str
        One of ``'voigt'``, ``'gaussian'``, or ``'lorentz'``.

    Returns
    -------
    np.ndarray
        Absorption cross section [cm²/molecule] on the wavenumber grid.
    """
    sigma_nu = np.zeros_like(nu)

    for nu0_i, S_i in zip(line_positions, line_strengths):
        if profile == "voigt":
            phi = voigt_profile_func(nu, nu0_i, delta_nu_D, delta_nu_L)
        elif profile == "gaussian":
            phi = gaussian_profile(nu, nu0_i, delta_nu_D)
        elif profile == "lorentz":
            phi = lorentz_profile(nu, nu0_i, delta_nu_L)
        else:
            raise ValueError(f"Unknown profile type: {profile!r}")
        sigma_nu += S_i * phi

    return sigma_nu


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demo_line_profiles(
    nu: np.ndarray,
    nu0: float,
    delta_nu_D: float,
    delta_nu_L: float,
    savefig: str = None,
) -> None:
    """
    Compare Gaussian, Lorentzian, and Voigt profiles on a single plot.

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    nu0 : float
        Line centre [cm⁻¹].
    delta_nu_D, delta_nu_L : float
        Doppler and Lorentzian HWHM [cm⁻¹].
    savefig : str, optional
        Save path for the figure.
    """
    G = gaussian_profile(nu, nu0, delta_nu_D)
    L = lorentz_profile(nu, nu0, delta_nu_L)
    V = voigt_profile_func(nu, nu0, delta_nu_D, delta_nu_L)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(nu, G / G.max(), label="Gaussian (Doppler)", color="royalblue", lw=2)
    ax.plot(nu, L / L.max(), label="Lorentzian (Pressure)", color="tomato", lw=2)
    ax.plot(nu, V / V.max(), label="Voigt (combined)", color="seagreen", lw=2, ls="--")
    ax.axvline(nu0, color="gray", ls=":", lw=1, label="Line centre")
    ax.set_xlabel("Wavenumber [cm⁻¹]", fontsize=12)
    ax.set_ylabel("Normalised profile", fontsize=12)
    ax.set_title(
        f"Spectral Line Profiles at ν₀={nu0:.1f} cm⁻¹\n"
        f"Δν_D={delta_nu_D:.4f}  Δν_L={delta_nu_L:.4f} cm⁻¹",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Conditions: 500 hPa, 250 K, CO₂ line near 6250 cm⁻¹
    nu0     = 6250.0
    T       = 250.0   # K
    P       = 50000.0  # Pa (500 hPa)

    dnu_D = doppler_hwhm(nu0, T)
    dnu_L = lorentz_hwhm(gamma_air=0.07, pressure_Pa=P, temperature_K=T)

    print(f"Doppler  HWHM : {dnu_D:.5f} cm⁻¹")
    print(f"Lorentz  HWHM : {dnu_L:.5f} cm⁻¹")

    nu = np.linspace(6248, 6252, 2000)
    demo_line_profiles(nu, nu0, dnu_D, dnu_L)

    # Compute cross section for a few synthetic lines
    line_pos = np.array([6249.5, 6250.0, 6250.7])
    line_str = np.array([5e-24, 1e-23, 3e-24])  # cm/molecule
    xsec = absorption_cross_section(nu, line_pos, line_str, dnu_D, dnu_L)
    print(f"\nPeak cross section: {xsec.max():.3e} cm²/molecule")