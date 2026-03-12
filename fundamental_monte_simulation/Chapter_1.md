# Monte Carlo Simulation Theory for CO₂ Satellite Retrieval

A detailed, beginner-friendly guide to the theory, mathematics, and practical application of the three Monte Carlo simulation modes used in this project.

---

## Table of Contents

1. [Background: What Problem Are We Solving?](#1-background)
2. [The Forward Model: How Satellites "See" CO₂](#2-the-forward-model)
3. [The Retrieval Algorithm: Working Backwards](#3-the-retrieval-algorithm)
4. [Why Monte Carlo? Connecting Randomness to Confidence](#4-why-monte-carlo)
5. [Mode 1 — Noise Ensemble](#5-mode-1--noise-ensemble)
6. [Mode 2 — XCO₂ Sweep](#6-mode-2--xco₂-sweep)
7. [Mode 3 — Random Scene Ensemble](#7-mode-3--random-scene-ensemble)
8. [Summary and Key Takeaways](#8-summary)

---

## 1. Background

A satellite orbiting the Earth measures the **column-averaged dry-air mole fraction of CO₂** ($XCO_2$). This is essentially the average concentration of carbon dioxide in the entire air column from the ground to the top of the atmosphere, expressed in **parts per million (ppm)**.

But here is the catch: the satellite cannot directly "see" molecules of CO₂. It can only measure **sunlight** — specifically, the spectrum of shortwave-infrared light that has bounced off the Earth's surface and passed through the atmosphere twice (once coming down from the Sun, once going back up to the satellite).

So the fundamental problem is:

> **Given a noisy measurement of sunlight, can we accurately figure out how much CO₂ was in the atmosphere?**

The Monte Carlo simulation helps us rigorously answer: *How accurate is our method, and under what conditions does it fail?*

---

## 2. The Forward Model

The **Forward Model** is the set of physics equations that predicts *what the satellite should see* for a given atmospheric state. Think of it as: "If we know everything about the atmosphere and the CO₂ perfectly, what spectrum of light would the satellite record?"

### 2.1 The Radiance Equation

The radiance $I(\nu)$ measured by the satellite at wavenumber $\nu$ (a measure of light frequency, in $\text{cm}^{-1}$) is:

$$\boxed{I(\nu) = I_{\odot}(\nu) \cdot A_s \cdot \cos(\theta_s) \cdot \exp\!\big(-M \cdot \sigma(\nu) \cdot N_{\text{col}}\big)}$$

Let's go through each piece with plain English:

| Symbol | Name | Meaning | Typical Value |
|--------|------|---------|---------------|
| $I_{\odot}(\nu)$ | Solar irradiance | Brightness of the incoming sunlight at frequency $\nu$. Modelled as a blackbody at 5778 K. | Normalized to 1.0 |
| $A_s$ | Surface albedo | How reflective the ground is (0 = perfectly dark, 1 = perfect mirror). Ocean ≈ 0.06, desert ≈ 0.35. | 0.05 – 0.40 |
| $\theta_s$ | Solar zenith angle (SZA) | The angle of the Sun from directly overhead. 0° = noon overhead, 70° = Sun is low on the horizon. | 10° – 70° |
| $\cos(\theta_s)$ | Geometric factor | Accounts for the fact that when the Sun is lower, less light hits the surface per unit area. | 0.34 – 0.98 |
| $M$ | Air Mass Factor | The total path length light travels through the atmosphere. For a nadir (straight-down) satellite: $M = \frac{1}{\cos\theta_s} + \frac{1}{\cos\theta_{\text{sat}}}$ | 2.0 – 3.9 |
| $\sigma(\nu)$ | Absorption cross-section | How strongly a single CO₂ molecule absorbs light at frequency $\nu$. Depends on pressure and temperature. | ~$10^{-24}$ to $10^{-22}$ cm² |
| $N_{\text{col}}$ | Column amount | Total number of CO₂ molecules in a vertical column of air, per unit area. | ~$8 \times 10^{21}$ molecules/cm² |

### 2.2 Physical Intuition

The key term is the **exponential**:

$$T(\nu) = \exp\!\big(-M \cdot \sigma(\nu) \cdot N_{\text{col}}\big)$$

This is the **Beer-Lambert Law** — it says that light gets exponentially weaker as it passes through more absorbing material. The product $M \cdot \sigma(\nu) \cdot N_{\text{col}}$ is called the **optical depth** ($\tau$):

$$\tau(\nu) = M \cdot \sigma(\nu) \cdot N_{\text{col}}$$

- **Small $\tau$ (~0.01):** Almost no light is absorbed → the satellite sees a bright signal.
- **Large $\tau$ (~3+):** Almost all light is absorbed → the satellite sees almost nothing at that frequency.

The pattern of which frequencies are absorbed and which aren't creates a unique **spectral fingerprint** of CO₂. The retrieval algorithm reads this fingerprint to determine how much CO₂ is present.

### 2.3 Computing the Column Amount

The column amount $N_{\text{col}}$ is calculated from an atmospheric profile:

$$N_{\text{col}} = \sum_{i=1}^{L} n_i \cdot \Delta z_i$$

where $n_i$ is the CO₂ number density in layer $i$ (molecules/cm³), $\Delta z_i$ is the thickness of layer $i$ (cm), and $L$ is the number of atmospheric layers.

The number density in each layer comes from the ideal gas law:

$$n_{\text{CO}_2}(z) = \frac{P(z) \cdot \chi_{\text{CO}_2}}{k_B \cdot T(z)}$$

where $P(z)$ is pressure, $T(z)$ is temperature, $\chi_{\text{CO}_2}$ is the CO₂ volume mixing ratio (e.g., $420 \times 10^{-6}$), and $k_B$ is Boltzmann's constant.

---

## 3. The Retrieval Algorithm

The retrieval is the **inverse problem**: given a noisy observed spectrum $\mathbf{y}_{\text{obs}}$, find the best estimate of $XCO_2$.

### 3.1 The Scaling Factor Approach

Instead of retrieving $XCO_2$ directly, the algorithm retrieves a **scaling factor** $\xi$ that multiplies a reference ("prior") CO₂ column:

$$N_{\text{col}}(\xi) = \xi \cdot N_{\text{col,prior}}$$

So if $\xi = 1.0$, the atmosphere has exactly as much CO₂ as our prior guess. If $\xi = 1.035$, it has 3.5% more. The final retrieved $XCO_2$ is simply:

$$\widehat{XCO_2} = \hat{\xi} \cdot XCO_{2,\text{prior}}$$

### 3.2 The Cost Function

The algorithm finds $\hat{\xi}$ by minimising a **cost function** $J(\xi)$. This cost function has two parts — think of it as a mathematical balance:

$$\boxed{J(\xi) = \underbrace{\frac{\|\mathbf{y}_{\text{obs}} - F(\xi)\|^2}{\sigma_{\text{noise}}^2}}_{\text{How well does } \xi \text{ fit the data?}} + \underbrace{\frac{(\xi - \xi_a)^2}{\sigma_a^2}}_{\text{How far is } \xi \text{ from our prior guess?}}}$$

| Term | Name | What it does |
|------|------|--------------|
| Data term (left) | Measurement fit | Penalises guesses of $\xi$ that produce a spectrum $F(\xi)$ that doesn't match the observation. |
| Prior term (right) | Regularisation | Penalises guesses of $\xi$ that are very far from our prior belief $\xi_a$. Prevents wild guesses. |

Here $\sigma_{\text{noise}}$ is the instrument noise level, $\xi_a = 1.0$ is the prior guess, and $\sigma_a$ is our prior uncertainty. If $\sigma_a = 0.10$, we are saying: "We believe the true CO₂ is within ±10% of our prior guess."

### 3.3 The Analytical Solution (Linear Case)

If we linearise the forward model around the prior ($\xi_a$):

$$F(\xi) \approx F(\xi_a) + \mathbf{K} \cdot (\xi - \xi_a)$$

where $\mathbf{K}$ is the **Jacobian** (the sensitivity of the radiance spectrum to changes in $\xi$):

$$\mathbf{K} = \frac{\partial F}{\partial \xi} = -M \cdot \sigma(\nu) \cdot N_{\text{col,prior}} \cdot I(\nu)$$

Then the optimal $\hat{\xi}$ has a closed-form solution:

$$\hat{\xi} = \xi_a + \frac{\mathbf{K}^T S_\epsilon^{-1} (\mathbf{y}_{\text{obs}} - F(\xi_a))}{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} + S_a^{-1}}$$

And the **posterior uncertainty** (how confident we are in the answer):

$$\sigma_{\text{post}} = \frac{1}{\sqrt{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} + S_a^{-1}}}$$

### 3.4 The Iterative Solution (Levenberg-Marquardt)

Since the real forward model is slightly non-linear, the code uses an iterative method. At each iteration $k$:

1. Compute the forward model $F(\xi_k)$ and Jacobian $\mathbf{K}_k$ at the current guess.
2. Calculate the update step:

$$\Delta\xi = \frac{\mathbf{K}_k^T S_\epsilon^{-1} (\mathbf{y}_{\text{obs}} - F(\xi_k)) - S_a^{-1}(\xi_k - \xi_a)}{\mathbf{K}_k^T S_\epsilon^{-1} \mathbf{K}_k + S_a^{-1} + \gamma \cdot S_a^{-1}}$$

3. Update: $\xi_{k+1} = \xi_k + \Delta\xi$.
4. If the cost decreased, accept the step and reduce damping ($\gamma \leftarrow \gamma/3$). If it increased, reject and increase damping ($\gamma \leftarrow 3\gamma$).
5. Stop when $|\Delta\xi| < \varepsilon$ (convergence threshold).

The damping parameter $\gamma$ is the "Levenberg-Marquardt trick" — when far from the solution, it takes cautious small steps; when close, it accelerates.

### 3.5 Converting Back to ppm

Once convergence is reached, the final $XCO_2$ and its uncertainty in ppm are:

$$\widehat{XCO_2} = \hat{\xi} \cdot XCO_{2,\text{prior}}, \qquad \sigma_{XCO_2} = \sigma_{\text{post}} \cdot XCO_{2,\text{prior}}$$

---

## 4. Why Monte Carlo?

We have a forward model and a retrieval algorithm. But before trusting the retrieval to analyze real satellite data, we need to know:

- Is it **accurate** (unbiased)?
- Is it **precise** (low random error)?
- Does it work **under all conditions** (different weather, geography, instrument quality)?

We could try to answer these theoretically, but the equations are non-linear and there are many interacting variables. Calculating the exact error analytically would require solving complex multi-dimensional integrals.

Monte Carlo provides a practical alternative: **generate thousands of fake-but-realistic scenarios, run the retrieval on each one, and statistically analyze the results.**

### The Core Loop

Every Monte Carlo sample follows the same four-step structure:

```
┌─────────────────────────────────────────────────────────┐
│  Step 1:  DEFINE the true atmospheric state             │
│           (XCO₂, P, T, albedo, SZA, SNR)                │
│                          ↓                              │
│  Step 2:  RUN the forward model → true radiance         │
│           Add random noise → noisy observation y_obs    │
│                          ↓                              │
│  Step 3:  RUN the retrieval on y_obs → estimated XCO₂   │
│           (the retrieval does NOT know the true values)  │
│                          ↓                              │
│  Step 4:  COMPARE estimated vs true XCO₂                │
│           Record the error = estimated − true           │
└─────────────────────────────────────────────────────────┘
```

After thousands of samples, we compute:

| Metric | Formula | What it tells us |
|--------|---------|-----------------|
| **Bias** | $\text{Bias} = \frac{1}{N}\sum_{i=1}^{N} (\hat{x}_i - x_i^{\text{true}})$ | Systematic error: does the retrieval consistently guess too high or too low? |
| **Precision** | $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (\hat{x}_i - \bar{\hat{x}})^2}$ | Random error: how much do repeated estimates scatter around their average? |
| **RMSE** | $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (\hat{x}_i - x_i^{\text{true}})^2}$ | Overall error combining both bias and precision: $\text{RMSE}^2 = \text{Bias}^2 + \sigma^2$ |

The question is: **what to randomise?** The three modes answer this by controlling what changes between samples.

---

## 5. Mode 1 — Noise Ensemble

### 5.1 Concept (Plain English)

Imagine pointing the satellite at **the exact same spot on Earth** with **the exact same atmosphere**, but taking the measurement 500 separate times. Each time, the detector electronics generate slightly different random noise. This mode simulates exactly that scenario.

### 5.2 What is Fixed vs. What is Random

| Parameter | Value | Status |
|-----------|-------|--------|
| True $XCO_2$ | 430 ppm | **Fixed** |
| Prior $XCO_2$ | 420 ppm | **Fixed** |
| Surface albedo | 0.25 | **Fixed** |
| SZA | 30° | **Fixed** |
| $T_{\text{surface}}$ | 288 K | **Fixed** |
| $P_{\text{surface}}$ | 101325 Pa | **Fixed** |
| SNR | 250 | **Fixed** |
| Noise realisation $\epsilon_i$ | Different each time | **Random** |

### 5.3 The Mathematics

Since the atmosphere is identical every time, the true radiance $\mathbf{I}_{\text{true}}$ is the same for every sample. Only the noise varies:

$$\mathbf{y}_{\text{obs},i} = \mathbf{I}_{\text{true}} + \boldsymbol{\epsilon}_i, \qquad \boldsymbol{\epsilon}_i \sim \mathcal{N}\!\left(0, \, \sigma_{\text{noise}}^2\right)$$

where $\sigma_{\text{noise}} = I_{\text{max}} / \text{SNR}$.

Using a linear approximation of the retrieval, the retrieved $\hat{\xi}$ for each sample is:

$$\hat{\xi}_i = \xi_a + \frac{\mathbf{K}^T S_\epsilon^{-1} \big(\mathbf{I}_{\text{true}} + \boldsymbol{\epsilon}_i - F(\xi_a)\big)}{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} + S_a^{-1}}$$

Since $\mathbf{I}_{\text{true}} - F(\xi_a)$ is a constant across all samples, the only source of variation is $\boldsymbol{\epsilon}_i$. The **variance** of $\hat{\xi}$ over many samples is therefore:

$$\text{Var}(\hat{\xi}) = \frac{\mathbf{K}^T S_\epsilon^{-1} \cdot \text{Var}(\boldsymbol{\epsilon}) \cdot S_\epsilon^{-1} \mathbf{K}}{\left(\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} + S_a^{-1}\right)^2} = \frac{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K}}{\left(\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} + S_a^{-1}\right)^2}$$

This simplifies to:

$$\text{Var}(\hat{\xi}) = \frac{1}{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} + S_a^{-1}} \cdot \frac{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K}}{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} + S_a^{-1}}$$

In the limit where data dominates the prior ($\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} \gg S_a^{-1}$):

$$\text{Var}(\hat{\xi}) \approx \frac{1}{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K}} = \sigma_{\text{post}}^2$$

This is exactly the posterior uncertainty predicted by Optimal Estimation theory. Mode 1 verifies this prediction numerically.

### 5.4 What This Mode Tells You

- **Precision**: The standard deviation of the error histogram gives the retrieval's **noise floor** — the minimum random error achievable for this specific scene.
- **Noise validation**: If the Monte Carlo precision ($\sigma_{\text{MC}}$) matches the theoretical posterior uncertainty ($\sigma_{\text{post}}$), the retrieval's error propagation is mathematically correct.
- **SNR requirement**: By running this mode at different SNR values (the SNR sensitivity study), you can determine the minimum instrument quality needed to achieve a target precision (e.g., < 1 ppm).

### 5.5 Key Relationship

$$\boxed{\sigma_{\text{precision}} \propto \frac{1}{\text{SNR}}}$$

Doubling the SNR roughly halves the random error. This is verified by the SNR sensitivity plot in the code.

---

## 6. Mode 2 — XCO₂ Sweep

### 6.1 Concept (Plain English)

Now imagine the satellite measures 100 different locations on Earth, all with identical weather, terrain, and instrument quality — but each location has a **different amount of CO₂** in the atmosphere, ranging from 380 ppm to 480 ppm. The retrieval algorithm always starts from the same prior guess (420 ppm).

This mode asks: *Does the retrieval accurately track the true CO₂ as it varies, or does the prior guess "pull" the answer towards 420 ppm?*

### 6.2 What is Fixed vs. What Varies

| Parameter | Status |
|-----------|--------|
| True $XCO_2$ | **Swept** from 380 to 480 ppm (evenly spaced) |
| Prior $XCO_2$ | Fixed at 420 ppm |
| Albedo, SZA, P, T, SNR | All **fixed** |
| Noise | One random realisation per sample |

### 6.3 The Mathematics — Averaging Kernel

This mode probes the **Averaging Kernel** ($A$), which quantifies the retrieval's sensitivity to the true state. Deriving it from the Optimal Estimation solution:

Starting from the retrieved state:

$$\hat{\xi} = \xi_a + G \cdot (\mathbf{y}_{\text{obs}} - F(\xi_a))$$

where $G$ is the **Gain** (scalar gain in our 1D case):

$$G = \frac{\mathbf{K}^T S_\epsilon^{-1}}{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} + S_a^{-1}}$$

If we substitute the observation model $\mathbf{y}_{\text{obs}} = F(\xi_{\text{true}}) + \boldsymbol{\epsilon}$ and linearise $F(\xi_{\text{true}}) \approx F(\xi_a) + \mathbf{K}(\xi_{\text{true}} - \xi_a)$:

$$\hat{\xi} = \xi_a + G \cdot \mathbf{K} \cdot (\xi_{\text{true}} - \xi_a) + G \cdot \boldsymbol{\epsilon}$$

Defining the **Averaging Kernel** $A = G \cdot \mathbf{K}$:

$$\hat{\xi} = \xi_a + A(\xi_{\text{true}} - \xi_a) + G \cdot \boldsymbol{\epsilon}$$

Rearranging:

$$\hat{\xi} = (1 - A)\xi_a + A \cdot \xi_{\text{true}} + G \cdot \boldsymbol{\epsilon}$$

This is a weighted average between the prior and the truth, plus noise.

### 6.4 Understanding the Averaging Kernel

For our scalar retrieval:

$$A = \frac{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K}}{\mathbf{K}^T S_\epsilon^{-1} \mathbf{K} + S_a^{-1}}$$

- **If $A = 1$:** The retrieval perfectly follows the truth. $\hat{\xi} = \xi_{\text{true}} + \text{noise}$.
- **If $A = 0$:** The retrieval ignores the data entirely and returns the prior. $\hat{\xi} = \xi_a$.
- **If $0 < A < 1$:** The retrieval is a blend. The answer is pulled towards the prior.

### 6.5 The Smoothing Error (Systematic Bias)

Taking the expected value (averaging out the noise):

$$\boxed{E[\hat{\xi} - \xi_{\text{true}}] = (A - 1)(\xi_{\text{true}} - \xi_a)}$$

This is the **smoothing error**. It tells you the systematic bias:

- When $\xi_{\text{true}} > \xi_a$ (true CO₂ is higher than the prior guess): the error is **negative** — the retrieval underestimates.
- When $\xi_{\text{true}} < \xi_a$ (true CO₂ is lower): the error is **positive** — the retrieval overestimates.
- When $\xi_{\text{true}} = \xi_a$: the bias is **zero**.

In ppm terms:

$$E[\widehat{XCO_2} - XCO_2^{\text{true}}] = (A-1)(XCO_2^{\text{true}} - XCO_{2,\text{prior}})$$

### 6.6 What This Mode Tells You

- **Linearity**: The "True vs Retrieved" scatter plot should show a straight line close to the 1:1 diagonal. The slope of a linear fit gives $A$:
  - Slope ≈ 1.0 → excellent (data-dominated retrieval)
  - Slope < 1.0 → the prior is pulling the answer (regularised retrieval)
- **Bias structure**: The "Error vs True XCO₂" plot reveals if error grows systematically as $XCO_2$ deviates from the prior.
- **Prior independence**: A good retrieval should have $A$ very close to 1, meaning it primarily trusts the data over the prior guess.

---

## 7. Mode 3 — Random Scene Ensemble

### 7.1 Concept (Plain English)

This is the most realistic mode. It simulates the satellite flying over the entire Earth, encountering completely different conditions at every point: different CO₂ levels, different ground types (ocean, forest, desert), different Sun angles, different weather (temperature, pressure), and different instrument performance.

This mode answers: *Across the full range of real-world conditions, how well does the retrieval perform overall?*

### 7.2 What is Randomised

Every simulation draw samples **all** key parameters from physically motivated probability distributions:

| Parameter | Distribution | Range | Rationale |
|-----------|-------------|-------|-----------|
| $XCO_2$ | $\mathcal{N}(\mu{=}420, \sigma{=}15)$ ppm, clipped to [350, 550] | Gaussian centered on global mean | Real $XCO_2$ varies by ~10–20 ppm seasonally and regionally |
| Albedo ($A_s$) | $\text{Uniform}(0.05, 0.40)$ | Full range | Ocean ≈ 0.06, vegetation ≈ 0.15–0.25, desert ≈ 0.35 |
| SZA ($\theta_s$) | $\text{Uniform}(10°, 70°)$ | Daytime range | Satellites only measure in daylight; SZA > 70° is excluded |
| $T_{\text{surface}}$ | $\mathcal{N}(288, 12)$ K, clipped to [260, 310] | Gaussian | Covers polar to tropical surface temperatures |
| $P_{\text{surface}}$ | $\mathcal{N}(101325, 2000)$ Pa, clipped to [95000, 105000] | Gaussian | Accounts for high-altitude terrain and weather systems |
| SNR | $\text{Uniform}(150, 400)$ | Full instrument range | Lower SNR = more noise = harder retrieval |

### 7.3 The Mathematics — Integral Approximation

Conceptually, we want to know the overall RMSE of the retrieval across all possible operating conditions. This is a multi-dimensional integral:

$$\text{RMSE} = \sqrt{\int\!\!\int\!\!\int (\hat{x} - x_{\text{true}})^2 \, p(x_{\text{true}}) \, p(\mathbf{b}) \, p(\boldsymbol{\epsilon}) \; dx \; d\mathbf{b} \; d\boldsymbol{\epsilon}}$$

where $\mathbf{b} = (A_s, \theta_s, T, P, \text{SNR})$ represents all the nuisance parameters.

This integral is impossible to solve analytically because:
1. The forward model $F$ is non-linear (exponential Beer-Lambert law).
2. There are 6+ random variables (high-dimensional integration).
3. The retrieval itself is iterative (no closed-form expression for $\hat{x}$ as a function of all inputs).

**Monte Carlo approximates this integral via the Law of Large Numbers:**

$$\text{RMSE} \approx \sqrt{\frac{1}{N}\sum_{i=1}^{N} (\hat{x}_i - x_{\text{true},i})^2}$$

As $N \to \infty$, this discrete average converges to the true integral. The convergence rate is $O(1/\sqrt{N})$: to halve the statistical uncertainty of the RMSE estimate itself, you need 4× as many samples.

### 7.4 Error Decomposition

The total error in this mode has multiple contributing sources:

$$\hat{x}_i - x_{\text{true},i} = \underbrace{(A_i - 1)(x_{\text{true},i} - x_a)}_{\text{Smoothing error (bias from prior)}} + \underbrace{G_i \cdot \boldsymbol{\epsilon}_i}_{\text{Noise error (random)}} + \underbrace{\text{higher-order terms}}_{\text{Non-linearity error}}$$

Each source depends on the scene parameters:
- **Low albedo** → weak signal → $\epsilon$ component dominates → larger errors
- **High SZA** → longer atmospheric path ($M$ increases) → deeper absorption lines (stronger $\mathbf{K}$) but also fewer photons → competing effects
- **Low SNR** → $\epsilon$ is large → noise error increases

### 7.5 The Six-Panel Diagnostic

The random scene mode produces a diagnostic with six panels that together reveal *which conditions cause the largest errors*:

1. **True vs Retrieved scatter**: Overall correlation. Points should cluster tightly along the 1:1 line. Colour-coding by SNR reveals whether low-SNR scenes are the outliers.
2. **Error histogram**: Should be centered near zero (low bias) and narrow (high precision). A Gaussian overlay checks if errors are normally distributed.
3. **Error vs Albedo**: Reveals if dark surfaces (low albedo) cause larger errors. Expected: errors increase at low albedo because the reflected signal is weaker relative to noise.
4. **Error vs SZA**: Reveals if large Sun angles degrade performance.
5. **|Error| vs SNR**: Should show a clear trend: lower SNR → larger absolute error.
6. **Posterior uncertainty distribution**: Shows the range of uncertainties the retrieval reports across all scenes.

### 7.6 What This Mode Tells You

- **Overall mission performance**: The headline RMSE, bias, and precision numbers characterize the retrieval's expected accuracy across its entire operational domain.
- **Failure conditions**: If the error scatter shows a clear trend with any parameter (e.g., error spikes when albedo < 0.1), you know to flag those scenes as unreliable.
- **Design requirements**: If the RMSE is too large, you can diagnose which parameter is the dominant error source and target improvements (e.g., increasing SNR, restricting SZA range, using better priors).

---

## 8. Summary

| Mode | What Varies | What is Fixed | What it Tests |
|------|-------------|---------------|---------------|
| **1. Noise Ensemble** | Only the random noise $\boldsymbol{\epsilon}$ | Everything else | **Precision**: how much does the answer scatter due to instrument noise alone? |
| **2. XCO₂ Sweep** | True $XCO_2$ (380→480 ppm) | Atmosphere, geometry, noise level | **Linearity & Bias**: does the retrieval accurately track CO₂ changes, or does the prior pull the answer? |
| **3. Random Scene** | Everything ($XCO_2$, albedo, SZA, T, P, SNR, noise) | Nothing (all randomised) | **Global performance**: what is the overall expected accuracy, and which conditions cause failures? |

### The Key Formulas at a Glance

| Concept | Formula |
|---------|---------|
| Forward Model | $I(\nu) = I_\odot \cdot A_s \cdot \cos\theta_s \cdot e^{-M \sigma N_{\text{col}}}$ |
| Cost Function | $J(\xi) = \frac{\|\mathbf{y} - F(\xi)\|^2}{\sigma_\epsilon^2} + \frac{(\xi-\xi_a)^2}{\sigma_a^2}$ |
| Optimal Estimate | $\hat{\xi} = \xi_a + \frac{K^T S_\epsilon^{-1} (y - F(\xi_a))}{K^T S_\epsilon^{-1} K + S_a^{-1}}$ |
| Posterior Uncertainty | $\sigma_{\text{post}} = 1/\sqrt{K^T S_\epsilon^{-1} K + S_a^{-1}}$ |
| Averaging Kernel | $A = K^T S_\epsilon^{-1} K / (K^T S_\epsilon^{-1} K + S_a^{-1})$ |
| Smoothing Error | $E[\hat{x} - x_{\text{true}}] = (A-1)(x_{\text{true}} - x_a)$ |
| MC Bias | $\text{Bias} = \frac{1}{N}\sum (\hat{x}_i - x_i)$ |
| MC Precision | $\sigma = \text{std}(\hat{x}_i - x_i)$ |
| MC RMSE | $\text{RMSE} = \sqrt{\frac{1}{N}\sum(\hat{x}_i - x_i)^2}$ |

> [!TIP]
> A helpful way to remember the three modes:
> - **Mode 1** tests the **instrument** (noise only)
> - **Mode 2** tests the **algorithm** (prior dependence)
> - **Mode 3** tests the **mission** (real-world deployment)

### References and Source Files

- [monte_carlo.py](file:///home/arun/Documents/co2_measurements/src/monte_carlo.py) — MC simulation engine (3 modes + plotting)
- [run_monte_carlo.py](file:///home/arun/Documents/co2_measurements/run_monte_carlo.py) — Standalone runner with all diagnostics
- [retrieval.py](file:///home/arun/Documents/co2_measurements/src/retrieval.py) — Optimal Estimation and Levenberg-Marquardt retrieval
- [radiative_transfer.py](file:///home/arun/Documents/co2_measurements/src/radiative_transfer.py) — Forward model (Beer-Lambert radiance)
