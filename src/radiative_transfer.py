"""
radiative_transfer.py
=====================
Simplified atmospheric radiative transfer and forward radiance modeling
for the CO2 Retrieval Simulator.

Physical Basis
--------------
A satellite spectrometer looking down at the Earth measures solar photons
that have:
  1. Travelled from the Sun through the atmosphere to the surface.
  2. Been reflected by the surface.
  3. Travelled back up through the atmosphere to the satellite.

For shortwave-infrared (SWIR) measurements (1.6–2.0 µm CO₂ bands),
thermal emission from the atmosphere is negligible.  The dominant
physics is therefore pure absorption along the two-way photon path.

The simplified forward model is:

    I(ν) = I_sun(ν) · A_s · cos(θ_s) · T²(ν)

where:
    I_sun(ν)   : top-of-atmosphere solar irradiance (Planck-like source)
    A_s        : surface albedo (Lambertian)
    θ_s        : solar zenith angle
    T(ν)       : one-way atmospheric transmittance (Beer-Lambert)
    T²(ν)      : two-way transmittance (down + up)

The two-way optical depth is:

    τ_total(ν) = τ_down(ν) + τ_up(ν)
               = σ(ν) · N_col · (1/µ_sun + 1/µ_sat)

where µ = cos(θ) of the respective zenith angles, and N_col is the
vertical column amount [molecules/cm²].

Author: Arun Kumar Pandey
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Solar irradiance model (simplified Planck blackbody)
# ---------------------------------------------------------------------------

def solar_irradiance(
    nu: np.ndarray,
    T_sun: float = 5778.0,
) -> np.ndarray:
    """
    Approximate top-of-atmosphere solar irradiance as a blackbody spectrum.

        B(ν, T) = (2hc²ν³) / (exp(hcν/k_BT) - 1)

    Note: The absolute scaling is not critical for the retrieval
    demonstration; we normalise to 1 at the spectral window centre.

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    T_sun : float
        Effective solar temperature [K].  Default 5778 K.

    Returns
    -------
    np.ndarray
        Relative solar irradiance (normalised to peak = 1).
    """
    h   = 6.62607015e-34   # Planck constant [J·s]
    k_B = 1.380649e-23     # Boltzmann constant [J/K]
    c   = 2.99792458e10    # speed of light [cm/s]

    # Avoid overflow for very small ν
    nu_safe = np.where(nu > 0, nu, 1e-10)

    exponent = (h * c * nu_safe) / (k_B * T_sun)
    B = (2.0 * h * c**2 * nu_safe**3) / (np.expm1(exponent))
    return B / B.max()


# ---------------------------------------------------------------------------
# Air-mass factor
# ---------------------------------------------------------------------------

def air_mass_factor(
    solar_zenith_deg: float,
    satellite_zenith_deg: float,
) -> float:
    """
    Compute the two-way geometric air-mass factor M.

        M = 1/µ_sun + 1/µ_sat

    where µ = cos(zenith angle).

    Parameters
    ----------
    solar_zenith_deg : float
        Solar zenith angle [degrees].
    satellite_zenith_deg : float
        Satellite viewing zenith angle [degrees].

    Returns
    -------
    float
        Two-way air-mass factor [dimensionless].
    """
    mu_sun = np.cos(np.radians(solar_zenith_deg))
    mu_sat = np.cos(np.radians(satellite_zenith_deg))

    if mu_sun <= 0 or mu_sat <= 0:
        raise ValueError("Zenith angles must be < 90 degrees (daylit scene).")

    return 1.0 / mu_sun + 1.0 / mu_sat


# ---------------------------------------------------------------------------
# Vertical column amount from profile
# ---------------------------------------------------------------------------

def column_amount(
    number_densities: np.ndarray,
    layer_thicknesses_cm: np.ndarray,
) -> float:
    """
    Compute the vertical column amount N_col [molecules/cm²].

        N_col = Σ_i  nᵢ · Lᵢ

    Parameters
    ----------
    number_densities : np.ndarray
        Number density in each layer [molecules/cm³].
    layer_thicknesses_cm : np.ndarray
        Thickness of each layer [cm].

    Returns
    -------
    float
        Total column amount [molecules/cm²].
    """
    return np.sum(number_densities * layer_thicknesses_cm)


# ---------------------------------------------------------------------------
# Forward radiance model
# ---------------------------------------------------------------------------

def forward_model(
    nu: np.ndarray,
    cross_section: np.ndarray,
    N_col: float,
    air_mass: float = 2.0,
    surface_albedo: float = 0.3,
    solar_zenith_deg: float = 30.0,
    T_sun: float = 5778.0,
    add_noise: bool = False,
    snr: float = 300.0,
) -> dict:
    """
    Compute the simulated top-of-atmosphere radiance seen by a nadir satellite.

        I(ν) = I_sun(ν) · A_s · cos(θ_s) · exp(-M · σ(ν) · N_col)

    Parameters
    ----------
    nu : np.ndarray
        Wavenumber grid [cm⁻¹].
    cross_section : np.ndarray
        CO₂ absorption cross section [cm²/molecule] on the nu grid.
    N_col : float
        CO₂ vertical column amount [molecules/cm²].
    air_mass : float
        Two-way geometric air-mass factor M.
    surface_albedo : float
        Lambertian surface albedo (0–1).
    solar_zenith_deg : float
        Solar zenith angle [degrees], used to weight by cos(θ_s).
    T_sun : float
        Effective solar temperature [K].
    add_noise : bool
        If True, add Gaussian photon noise at the specified SNR.
    snr : float
        Signal-to-noise ratio for noise simulation.

    Returns
    -------
    dict with keys:
        'nu'           : wavenumber grid [cm⁻¹]
        'I_sun'        : normalised solar irradiance
        'transmittance': two-way atmospheric transmittance
        'tau'          : two-way optical depth
        'radiance'     : simulated observed radiance
        'radiance_noisy' (if add_noise=True)
    """
    mu_sun = np.cos(np.radians(solar_zenith_deg))

    I_sun = solar_irradiance(nu, T_sun)

    # Two-way optical depth
    tau = air_mass * cross_section * N_col

    # Two-way transmittance
    T_atm = np.exp(-tau)

    # Simulated radiance at satellite
    I_obs = I_sun * surface_albedo * mu_sun * T_atm

    result = {
        "nu":            nu,
        "I_sun":         I_sun,
        "transmittance": T_atm,
        "tau":           tau,
        "radiance":      I_obs,
    }

    if add_noise:
        noise_sigma = I_obs.max() / snr
        result["radiance_noisy"] = I_obs + np.random.normal(0, noise_sigma, size=I_obs.shape)

    return result


# ---------------------------------------------------------------------------
# Atmospheric profile helpers
# ---------------------------------------------------------------------------

def standard_atmosphere_profile(
    n_layers: int = 20,
    P_surface_Pa: float = 101325.0,
    T_surface_K: float = 288.0,
    co2_vmr_ppm: float = 420.0,
    scale_height_m: float = 8500.0,
) -> dict:
    """
    Create a simplified exponential atmosphere with n_layers.

    Pressure decreases exponentially with altitude:
        P(z) = P₀ · exp(-z / H)

    Temperature decreases linearly (standard tropospheric lapse rate ~6.5 K/km).

    Parameters
    ----------
    n_layers : int
        Number of vertical layers.
    P_surface_Pa : float
        Surface pressure [Pa].
    T_surface_K : float
        Surface temperature [K].
    co2_vmr_ppm : float
        CO₂ volume mixing ratio [ppm], assumed uniform with altitude.
    scale_height_m : float
        Atmospheric scale height [m].

    Returns
    -------
    dict with arrays: 'altitude_km', 'pressure_Pa', 'temperature_K',
                      'co2_vmr', 'number_density', 'layer_thickness_cm'
    """
    k_B   = 1.380649e-23
    lapse = 6.5e-3   # K/m

    # Layer mid-point altitudes (km), evenly spaced up to ~40 km
    z_km  = np.linspace(0.25, 39.75, n_layers)
    z_m   = z_km * 1e3

    P     = P_surface_Pa * np.exp(-z_m / scale_height_m)
    T     = np.maximum(T_surface_K - lapse * z_m, 200.0)  # floor at 200 K
    vmr   = np.full(n_layers, co2_vmr_ppm * 1e-6)

    # CO₂ number density  [molecules/cm³]
    n_co2 = (P * vmr) / (k_B * T) * 1e-6   # ×1e-6 converts m⁻³ → cm⁻³

    # Uniform layer thickness [cm]  (total 40 km split equally)
    dz_cm = np.full(n_layers, (40e3 / n_layers) * 100.0)

    return {
        "altitude_km":       z_km,
        "pressure_Pa":       P,
        "temperature_K":     T,
        "co2_vmr":           vmr,
        "number_density":    n_co2,
        "layer_thickness_cm": dz_cm,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_forward_model(result: dict, savefig: str = None) -> None:
    """
    Four-panel plot: solar irradiance, optical depth, transmittance, radiance.

    Parameters
    ----------
    result : dict
        Output from ``forward_model()``.
    savefig : str, optional
        Save path.
    """
    nu  = result["nu"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(nu, result["I_sun"], color="gold", lw=1.5)
    axes[0].set_title("Solar Irradiance (normalised)")
    axes[0].set_ylabel("Relative irradiance")

    axes[1].plot(nu, result["tau"], color="steelblue", lw=1.5)
    axes[1].set_title("Two-way Optical Depth τ(ν)")
    axes[1].set_ylabel("τ")

    axes[2].plot(nu, result["transmittance"], color="seagreen", lw=1.5)
    axes[2].set_title("Atmospheric Transmittance T²(ν)")
    axes[2].set_ylabel("Transmittance")
    axes[2].set_ylim(-0.02, 1.05)

    axes[3].plot(nu, result["radiance"], color="darkorange", lw=1.5, label="Clean")
    if "radiance_noisy" in result:
        axes[3].plot(nu, result["radiance_noisy"], color="gray", lw=0.6,
                     alpha=0.7, label="Noisy")
        axes[3].legend(fontsize=9)
    axes[3].set_title("Simulated Satellite Radiance")
    axes[3].set_ylabel("Radiance (rel.)")

    for ax in axes:
        ax.set_xlabel("Wavenumber [cm⁻¹]")
        ax.grid(True, alpha=0.3)

    fig.suptitle("CO₂ Retrieval Forward Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from spectroscopy import (
        absorption_cross_section,
        doppler_hwhm,
        lorentz_hwhm,
    )

    nu = np.linspace(6220, 6280, 3000)

    # Build a simple synthetic CO₂ cross section
    nu0_lines = np.array([6228.0, 6235.5, 6244.2, 6250.0, 6255.8, 6262.3, 6270.0])
    strengths  = np.array([2e-24, 8e-24, 4e-24, 1.2e-23, 6e-24, 3e-24, 2e-24])

    T   = 255.0
    P   = 60000.0   # 600 hPa
    dnu_D = doppler_hwhm(6250.0, T)
    dnu_L = lorentz_hwhm(0.07, P, T)

    xsec = absorption_cross_section(nu, nu0_lines, strengths, dnu_D, dnu_L)

    # Standard atmosphere profile
    profile = standard_atmosphere_profile(co2_vmr_ppm=420.0)
    N_col   = column_amount(profile["number_density"], profile["layer_thickness_cm"])
    print(f"CO₂ column amount: {N_col:.3e} molecules/cm²")

    amf = air_mass_factor(30.0, 0.0)   # SZA=30°, nadir satellite

    result = forward_model(
        nu, xsec, N_col,
        air_mass=amf,
        surface_albedo=0.3,
        solar_zenith_deg=30.0,
        add_noise=True,
        snr=300.0,
    )

    plot_forward_model(result)