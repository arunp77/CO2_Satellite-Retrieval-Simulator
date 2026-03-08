"""
CO2 Retrieval Simulator
=======================
Scientific Python package for simulating satellite-based atmospheric CO₂ measurements.

Modules
-------
absorption        : Beer-Lambert absorption law
spectroscopy      : Molecular spectral line profiles (Gaussian, Lorentzian, Voigt)
radiative_transfer: Atmospheric forward radiance model
hitran_model      : HITRAN-based CO₂ absorption spectrum
retrieval         : XCO₂ retrieval via optimal estimation

Author: Arun Kumar Pandey
"""

from .absorption          import optical_depth, transmittance, radiance, number_density_from_pT
from .spectroscopy        import doppler_hwhm, lorentz_hwhm, voigt_profile_func, absorption_cross_section
from .radiative_transfer  import forward_model, standard_atmosphere_profile, air_mass_factor, column_amount
from .hitran_model        import hitran_cross_section, SYNTHETIC_CO2_LINES
from .retrieval           import iterative_retrieval, xco2_from_scaling
from .monte_carlo         import run_monte_carlo,plot_noise_ensemble,plot_xco2_sweep,plot_random_scene,plot_snr_sensitivity



__all__ = [
    "optical_depth", "transmittance", "radiance", "number_density_from_pT",
    "doppler_hwhm", "lorentz_hwhm", "voigt_profile_func", "absorption_cross_section",
    "forward_model", "standard_atmosphere_profile", "air_mass_factor", "column_amount",
    "hitran_cross_section", "SYNTHETIC_CO2_LINES",
    "iterative_retrieval", "xco2_from_scaling", 
    "run_monte_carlo", "plot_noise_ensemble", "plot_xco2_sweep", "plot_random_scene", "plot_snr_sensitivity"
]
