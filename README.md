# CO₂ Satellite Retrieval Simulator

A scientific Python project demonstrating the **physical principles behind satellite-based atmospheric CO₂ measurements**. The project implements simplified models of **molecular absorption, atmospheric radiative transfer, and satellite forward modeling**, which form the foundation of greenhouse gas retrieval algorithms used in modern Earth observation missions.

This repository provides an educational and computational framework for understanding how **satellite spectrometers detect carbon dioxide (CO₂) from space** by analyzing spectral absorption features in reflected solar radiation.

The project was developed as an **independent technical study of satellite CO₂ retrieval physics** used in missions such as:

* OCO-2 (NASA)
* MicroCarb (CNES / UKSA)
* GOSAT (JAXA)
* Sentinel-5P (ESA)

---

# Motivation

Satellite missions measure atmospheric gases by observing how sunlight interacts with molecules in the atmosphere. CO₂ has strong absorption bands in the **shortwave infrared (SWIR)** region. By analyzing these absorption signatures, retrieval algorithms estimate the atmospheric CO₂ concentration.

These measurements rely on several physical processes:

* molecular spectroscopy
* photon absorption
* atmospheric radiative transfer
* spectral line broadening
* forward radiance modeling

This project reproduces these processes in a simplified computational framework.

---

# Scientific Background

## Radiative Transfer Equation

The propagation of radiation through an absorbing medium is governed by the **radiative transfer equation**:

$$
\frac{dI_\nu}{ds} = -\kappa_\nu I_\nu + j_\nu
$$

where

* $I_\nu$ : spectral radiance
* $\kappa_\nu$ : absorption coefficient
* $j_\nu$ : emission term
* $s$ : path length

For solar backscatter measurements in the shortwave infrared, thermal emission is typically negligible. The equation simplifies to pure absorption.

---

## Beer–Lambert Absorption Law

Under absorption-only conditions, the radiative transfer equation reduces to the **Beer–Lambert law**:

$$
I_\nu = I_{\nu,0} e^{-\tau_\nu}
$$

where

* $I_{\nu,0}$ : incident radiation
* $I_\nu$ : transmitted radiation
* $\tau_\nu$ : optical depth

---

## Optical Depth

Optical depth describes the cumulative absorption along the photon path:

$$
\tau_\nu = \int \sigma_\nu n(s) ds
$$

where

* $ \sigma_\nu $ : molecular absorption cross section
* $ n(s) $ : molecular number density
* $ ds $ : path element

Optical depth depends on atmospheric composition, pressure, temperature, and photon path length.

---

# Molecular Spectroscopy

Atmospheric gases absorb radiation at discrete wavelengths corresponding to molecular energy transitions.

CO₂ has strong absorption bands near:

* 1.6 µm
* 2.0 µm

These bands are used by many satellite missions to retrieve atmospheric CO₂.

---

# Spectral Line Profiles

Real molecular absorption lines are broadened due to several physical effects.

## Doppler Broadening

Thermal motion of molecules causes frequency shifts:

$$
\phi_D(\nu)
$$

## Pressure Broadening

Collisions between molecules produce Lorentzian line shapes:

$$
\phi_L(\nu)
$$

## Voigt Profile

The real spectral line shape is the convolution of Doppler and pressure broadening:

$$
\phi_V(\nu) = \int \phi_D(\nu') \phi_L(\nu - \nu') d\nu'
$$

This Voigt profile is used in atmospheric spectroscopy models.

---

# HITRAN Spectroscopic Database

Real satellite retrieval algorithms rely on the **HITRAN molecular spectroscopy database**, which provides:

* spectral line positions
* line intensities
* pressure broadening parameters
* temperature dependence

HITRAN is the standard reference for atmospheric spectroscopy modeling.

The project uses the **HITRAN API (HAPI)** to download and simulate CO₂ spectral lines.

---

# Project Objectives

The main goal of this project is to simulate the key physical components of satellite CO₂ measurements.

The project implements:

1. Beer–Lambert absorption modeling
2. CO₂ spectral line simulation
3. Voigt line profile modeling
4. HITRAN-based spectroscopy
5. Radiative transfer forward modeling
6. Synthetic satellite radiance spectra
7. Conceptual XCO₂ retrieval

---

# Project Workflow

The simulator reproduces the simplified chain of satellite measurements:

```
Sunlight
   ↓
Atmospheric absorption by CO₂
   ↓
Spectral absorption features
   ↓
Satellite spectrometer measurement
   ↓
Forward radiance model
   ↓
CO₂ concentration retrieval
```

---

# Repository Structure

```
co2_retrieval_simulator

├── notebooks
│   ├── 01_beer_lambert.ipynb
│   ├── 02_co2_spectral_line.ipynb
│   ├── 03_forward_model.ipynb
│   ├── 04_hitran_spectrum.ipynb
│   └── 05_xco2_retrieval.ipynb
│
├── src
│   ├── absorption.py
│   ├── spectroscopy.py
│   ├── radiative_transfer.py
│   ├── hitran_model.py
│   └── retrieval.py
│
├── figures
│
├── data
│   └── hitran_lines.txt
│
└── README.md
```

---

# Simulation Modules

## Beer–Lambert Absorption

Demonstrates exponential attenuation of radiation through a gas:

$$
I = I_0 e^{-\sigma n L}
$$

This simulation shows how photon intensity decreases as it passes through absorbing CO₂ molecules.

---

## CO₂ Spectral Line Simulation

Simplified Gaussian spectral lines represent molecular absorption features:

$$
\sigma(\nu) =
\sigma_0
\exp
\left(
-\frac{(\nu - \nu_0)^2}{2\sigma^2}
\right)
$$

These features correspond to transitions between molecular energy levels.

---

## HITRAN-Based Spectroscopy

Realistic CO₂ absorption spectra are generated using the **HITRAN database**.

The absorption coefficient is computed as

$$
k(\nu) = \sum_i S_i \phi_i(\nu)
$$

where

* (S_i) : line strength
* (\phi_i(\nu)) : Voigt profile

---

## Forward Radiance Model

Satellite spectrometers measure radiance after atmospheric absorption.

The forward model computes the observed spectrum:

$$
I(\nu) = I_0(\nu) e^{-\tau(\nu)}
$$

where

$$
\tau(\nu) = \sum_i \sigma_i(\nu) n_i L_i
$$

---

# Atmospheric Layer Model

The atmosphere can be modeled as multiple layers:

$$
\tau(\nu) =
\sum_{layers}
\sigma(\nu) n_i L_i
$$

Each layer contributes differently depending on:

* pressure
* temperature
* gas concentration

---

# XCO₂ Retrieval Concept

Satellite missions retrieve **column-averaged dry-air mole fraction of CO₂**:

$$
XCO_2 =
\frac{\int n_{CO_2}(z) dz}{\int n_{dry,air}(z) dz}
$$

Retrieval algorithms estimate XCO₂ by fitting simulated spectra to observed radiance.

---

# Example Outputs

The simulator generates several scientific outputs:

* Beer–Lambert absorption curves
* synthetic CO₂ spectral lines
* HITRAN-based absorption spectra
* simulated satellite radiance spectra
* conceptual CO₂ retrieval examples

These outputs illustrate how atmospheric CO₂ affects satellite observations.

---

# Tools and Libraries

The project uses the following scientific computing tools:

* Python
* NumPy
* SciPy
* Matplotlib
* Jupyter Notebook
* HITRAN API (HAPI)

---

# Future Extensions

Possible future improvements include:

* full HITRAN spectroscopy modeling
* multilayer atmospheric radiative transfer
* aerosol and cloud scattering
* optimal estimation retrieval algorithms
* comparison with real satellite spectra
* simulation of OCO-2 spectral bands

---

# Author

Arun Kumar Pandey
Remote sensing scientist and data engineer working on Earth observation systems.

GitHub
[https://github.com/arunp77](https://github.com/arunp77)

---

# License

This project is intended for educational and scientific purposes.

