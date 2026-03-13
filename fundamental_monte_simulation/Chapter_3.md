## What is Probability? (The Two Schools of Thought)

Before we get to Bayes, we need to understand that there are **two different ways** to interpret the word "probability":

### School 1: Frequentist Probability
> *"Probability is the long-run frequency of an event."*

If you flip a fair coin 10,000 times, you expect about 5,000 heads. So $P(\text{heads}) = 0.5$. This definition only works for **repeatable experiments**.

**Problem:** What is the probability that $XCO_2$ at a specific location is between 415 and 425 ppm? You can't repeat that exact measurement 10,000 times — it's a one-time observation. The frequentist framework struggles with this.

### School 2: Bayesian Probability
> *"Probability is a measure of our **degree of belief** (or confidence) about something."*

$P(XCO_2 = 420 \text{ ppm}) = 0.8$ means: "Given everything I know, I am 80% confident that $XCO_2$ is 420 ppm." It doesn't require repeating anything — it's a statement about our **state of knowledge**.

This is the foundation of Bayesian statistics: **probability represents uncertainty about unknown quantities**, and we update our beliefs as new data arrives.

---

## The Building Blocks: Basic Probability Rules

Before deriving Bayes' theorem, we need three fundamental concepts:

### 1. Marginal Probability: $P(A)$
The probability that event $A$ happens, regardless of anything else.

**Example:** $P(\text{rain}) = 0.3$ means there's a 30% chance of rain.

### 2. Joint Probability: $P(A \cap B)$ or $P(A, B)$
The probability that **both** $A$ and $B$ happen simultaneously.

**Example:** $P(\text{rain AND cold}) = 0.15$ means there's a 15% chance it's both rainy and cold.

### 3. Conditional Probability: $P(A \mid B)$
The probability of $A$ happening, **given that we already know** $B$ has happened.

**Example:** $P(\text{rain} \mid \text{cloudy}) = 0.6$ means: "If it's cloudy, there's a 60% chance of rain."

### The Fundamental Relationship

These three are connected by the **definition of conditional probability**:

$$\boxed{P(A \mid B) = \frac{P(A \cap B)}{P(B)}}$$

**Derivation / intuition:** Out of all the times $B$ happens, what fraction of those times does $A$ also happen? That fraction is $P(A \cap B) / P(B)$.

Rearranging: $P(A \cap B) = P(A \mid B) \cdot P(B)$

By symmetry (we can swap $A$ and $B$): $P(A \cap B) = P(B \mid A) \cdot P(A)$

---

## Deriving Bayes' Theorem

Since both expressions equal $P(A \cap B)$, we can set them equal:

$$P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

Divide both sides by $P(B)$:

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}$$

**That's Bayes' Theorem.** It's just algebra from the definition of conditional probability. Nothing more!

---

## Understanding Each Term

Let's rename the variables to match how Bayes' theorem is used in science. We want to learn about an unknown **parameter** $x$ (like $XCO_2$) from observed **data** $\mathbf{y}$ (like the satellite spectrum):

$$\boxed{P(x \mid \mathbf{y}) = \frac{P(\mathbf{y} \mid x) \cdot P(x)}{P(\mathbf{y})}}$$

| Term | Name | Plain English |
|------|------|---------------|
| $P(x \mid \mathbf{y})$ | **Posterior** | Our updated belief about $x$ **after** seeing the data. This is what we want! |
| $P(\mathbf{y} \mid x)$ | **Likelihood** | If $x$ were the true value, how likely is it that we'd observe this particular data $\mathbf{y}$? |
| $P(x)$ | **Prior** | Our belief about $x$ **before** seeing any data. What did we think beforehand? |
| $P(\mathbf{y})$ | **Evidence** (or marginal likelihood) | The total probability of observing this data across all possible values of $x$. Acts as a normalisation constant. |

### The Intuitive Story

$$\text{What we believe AFTER data} = \frac{\text{How well data fits each hypothesis} \times \text{What we believed BEFORE data}}{\text{Normalisation}}$$

**Simple example:** You hear a sound outside.
- **Prior** $P(x)$: Before you look, you think there's a 70% chance it's a car and 30% chance it's a motorbike.
- **Likelihood** $P(\mathbf{y} \mid x)$: The sound is very high-pitched. Cars rarely make high-pitched sounds (likelihood = 0.1), but motorbikes often do (likelihood = 0.8).
- **Posterior** $P(x \mid \mathbf{y})$: After hearing the sound, your belief shifts towards motorbike!

$$P(\text{motorbike} \mid \text{high pitch}) = \frac{0.8 \times 0.3}{0.8 \times 0.3 + 0.1 \times 0.7} = \frac{0.24}{0.31} \approx 0.77$$

Your belief jumped from 30% to 77% — the data (high-pitched sound) updated your prior belief.

---

## Continuous Variables and Probability Density Functions (PDFs)

For discrete events (car vs motorbike), probabilities are simple numbers. But $XCO_2$ is a **continuous** variable — it can be 419.37 or 420.12 or any number. For continuous variables, we use **Probability Density Functions** (PDFs).

A PDF $p(x)$ tells you the **relative likelihood** of $x$ taking different values. Key properties:

$$p(x) \geq 0 \quad \text{for all } x, \qquad \int_{-\infty}^{+\infty} p(x)\, dx = 1$$

The probability that $x$ falls between $a$ and $b$ is:

$$P(a \leq x \leq b) = \int_a^b p(x)\, dx$$

### The Gaussian (Normal) Distribution

The most important PDF in statistics. If $x$ follows a Gaussian distribution with mean $\mu$ and standard deviation $\sigma$:

$$x \sim \mathcal{N}(\mu, \sigma^2)$$

The PDF is:

$$\boxed{p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)}$$

| Parameter | Meaning |
|-----------|---------|
| $\mu$ | The **centre** (peak) of the bell curve — the most likely value. |
| $\sigma$ | The **width** — how spread out the curve is. 68% of values fall within $\mu \pm \sigma$. |
| $\sigma^2$ | The **variance** — the square of the standard deviation. |

**Why Gaussian?** The Central Limit Theorem says that when many small, independent random effects add together (like many sources of noise in a sensor), the total tends toward a Gaussian. Measurement noise is almost always Gaussian.

---

## Bayes' Theorem for Continuous Variables

For continuous $x$ and data $\mathbf{y}$, Bayes' theorem becomes:

$$p(x \mid \mathbf{y}) = \frac{p(\mathbf{y} \mid x) \cdot p(x)}{p(\mathbf{y})}$$

where $p(\mathbf{y}) = \int p(\mathbf{y} \mid x) \cdot p(x)\, dx$ ensures the posterior integrates to 1.

In practice, we often write:

$$p(x \mid \mathbf{y}) \propto p(\mathbf{y} \mid x) \cdot p(x)$$

(The symbol $\propto$ means "proportional to" — we ignore the normalisation constant because it doesn't depend on $x$.)

---

## Applying Bayes to the Satellite Retrieval

Now let's connect all of this to your CO₂ retrieval problem.

### The Prior: $p(\xi)$

Before we look at the satellite data, what do we believe about $\xi$ (the CO₂ scaling factor)? We choose a Gaussian prior:

$$p(\xi) = \frac{1}{\sigma_a\sqrt{2\pi}} \exp\!\left(-\frac{(\xi - \xi_a)^2}{2\sigma_a^2}\right)$$

With $\xi_a = 1.0$ and $\sigma_a = 0.10$, this says: "We believe $\xi$ is probably close to 1.0, and we're 68% confident it's between 0.90 and 1.10."

### The Likelihood: $p(\mathbf{y} \mid \xi)$

Given a specific $\xi$, the forward model predicts the spectrum $F(\xi)$. The observed spectrum is $\mathbf{y}_{\text{obs}} = F(\xi) + \boldsymbol{\epsilon}$, where noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$ is Gaussian. So the probability of observing $\mathbf{y}_{\text{obs}}$ given $\xi$ is:

$$p(\mathbf{y}_{\text{obs}} \mid \xi) \propto \exp\!\left(-\frac{\|\mathbf{y}_{\text{obs}} - F(\xi)\|^2}{2\sigma_{\text{noise}}^2}\right)$$

This is just the Gaussian PDF with the mean at $F(\xi)$ and width $\sigma_{\text{noise}}$, evaluated at the observed data.

### The Posterior: $p(\xi \mid \mathbf{y})$

Applying Bayes:

$$p(\xi \mid \mathbf{y}) \propto p(\mathbf{y} \mid \xi) \cdot p(\xi)$$

$$p(\xi \mid \mathbf{y}) \propto \exp\!\left(-\frac{\|\mathbf{y}_{\text{obs}} - F(\xi)\|^2}{2\sigma_{\text{noise}}^2}\right) \cdot \exp\!\left(-\frac{(\xi - \xi_a)^2}{2\sigma_a^2}\right)$$

Since $e^a \cdot e^b = e^{a+b}$:

$$p(\xi \mid \mathbf{y}) \propto \exp\!\left(-\frac{1}{2}\left[\frac{\|\mathbf{y}_{\text{obs}} - F(\xi)\|^2}{\sigma_{\text{noise}}^2} + \frac{(\xi - \xi_a)^2}{\sigma_a^2}\right]\right)$$

### The Connection to the Cost Function

Finding the $\xi$ that **maximises** the posterior $p(\xi \mid \mathbf{y})$ is the same as finding the $\xi$ that **minimises** the negative exponent. And that negative exponent is exactly:

$$J(\xi) = \frac{\|\mathbf{y}_{\text{obs}} - F(\xi)\|^2}{\sigma_{\text{noise}}^2} + \frac{(\xi - \xi_a)^2}{\sigma_a^2}$$

**This is the cost function from Step 2 of Chapter 2!** So minimising $J(\xi)$ is identical to finding the **Maximum A Posteriori (MAP) estimate** — the most probable value of $\xi$ given the data and our prior.

$$\boxed{\hat{\xi}_{\text{MAP}} = \arg\min_\xi\; J(\xi) = \arg\max_\xi\; p(\xi \mid \mathbf{y})}$$

Everything comes full circle:
- The **data term** in $J$ comes from the **likelihood** (how well the model fits the data).
- The **prior term** in $J$ comes from the **prior** (our pre-existing belief).
- The **optimal estimate** $\hat{\xi}$ is the peak of the **posterior** distribution.
- The **posterior uncertainty** $\sigma_{\text{post}}$ is the width of the **posterior** distribution.

---

## Summary: The Bayesian Recipe

```
┌─────────────────────────────────────────────────────────────┐
│  1. STATE YOUR PRIOR BELIEF                                 │
│     "Before data, I think ξ ≈ 1.0  ± 0.10"                  │
│     → p(ξ) = Gaussian(1.0, 0.10²)                           │
│                                                             │
│  2. WRITE DOWN THE LIKELIHOOD                               │
│     "If ξ were the truth, how likely is this observation?"  │
│     → p(y|ξ) = Gaussian(F(ξ), σ_noise²)                     │
│                                                             │
│  3. APPLY BAYES' THEOREM                                    │
│     posterior ∝ likelihood × prior                          │
│     → p(ξ|y) ∝ p(y|ξ) · p(ξ)                                │
│                                                             │
│  4. FIND THE PEAK (MAP estimate)                            │
│     Minimise cost function J(ξ) = −2 ln[p(ξ|y)]             │
│     → This gives ξ̂  (the best estimate)                     │
│                                                             │
│  5. MEASURE THE WIDTH (posterior uncertainty)               │
│     Curvature of J at the minimum → σ_post                  │
│     → This gives the error bar                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Formulas at a Glance

| Concept | Formula |
|---------|---------|
| Bayes' Theorem | $p(x \mid y) = \frac{p(y \mid x) \cdot p(x)}{p(y)}$ |
| Gaussian PDF | $p(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2 / (2\sigma^2)}$ |
| Prior | $p(\xi) = \mathcal{N}(\xi_a, \sigma_a^2)$ |
| Likelihood | $p(\mathbf{y} \mid \xi) \propto e^{-\|\mathbf{y} - F(\xi)\|^2 / (2\sigma_\epsilon^2)}$ |
| Posterior | $p(\xi \mid \mathbf{y}) \propto e^{-J(\xi)/2}$ |
| Cost Function | $J(\xi) = \frac{\|\mathbf{y} - F(\xi)\|^2}{\sigma_\epsilon^2} + \frac{(\xi - \xi_a)^2}{\sigma_a^2}$ |
| MAP Estimate | $\hat{\xi} = \arg\min J(\xi)$ |
| Posterior Uncertainty | $\sigma_{\text{post}} = 1/\sqrt{d^2J/d\xi^2 \big|_{\hat{\xi}} / 2}$ |

The entire retrieval algorithm is just Bayesian statistics applied to the satellite measurement problem. The "prior" is our initial guess for CO₂, and the "likelihood" comes from how well a proposed CO₂ value explains the observed spectrum through the physics of light absorption!