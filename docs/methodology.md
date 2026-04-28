# Methodology

This page summarizes the math behind each step of the pyFIES pipeline. For a
full treatment, see FAO's
[FIES Technical Paper v1.1](https://www.fao.org/fileadmin/templates/ess/voh/FIES_Technical_Paper_v1.1.pdf)
and Cafiero, Viviani, & Nord (2018).

## The Rasch model

Each FIES item *i* has a *severity* parameter $\beta_i$. Each respondent has a
latent food-insecurity severity $\theta$. The probability of an affirmative
response to item *i* given $\theta$ is

$$
P(X_i = 1 \mid \theta) = \frac{\exp(\theta - \beta_i)}{1 + \exp(\theta - \beta_i)}.
$$

This is the dichotomous Rasch model — a one-parameter item response theory
(IRT) model. A respondent's *raw score* $R = \sum_i X_i$ is a sufficient
statistic for $\theta$ under this model.

## Conditional Maximum Likelihood (CML)

CML conditions out the latent $\theta$ by working with the conditional
distribution of response patterns given the raw score. The conditional
likelihood is

$$
P(X \mid R = r, \beta) = \frac{\exp(-\sum_i x_i \beta_i)}{\gamma_r(\varepsilon)},
$$

where $\varepsilon_i = \exp(-\beta_i)$ and

$$
\gamma_r(\varepsilon) = \sum_{|J| = r} \prod_{j \in J} \varepsilon_j
$$

is the *elementary symmetric function* of order *r*. The conditional log-likelihood
across the sample is

$$
\ell(\beta) = -\sum_i T_i \beta_i - \sum_r N_r \log \gamma_r(\varepsilon),
$$

where $T_i = \sum_p w_p x_{p,i}$ is the (weighted) endorsement count of item
*i* and $N_r = \sum_{p : R_p = r} w_p$ is the (weighted) number of respondents
at raw score *r*. Sampling weights are renormalized to sum to *n* before
entering the likelihood, matching the convention used by `RM.weights`.

This likelihood is convex in $\beta$, so a quasi-Newton method (pyFIES uses
SciPy's L-BFGS-B with an analytic gradient) converges to the unique MLE up
to an additive constant. pyFIES resolves the indeterminacy with a
sum-to-zero identification constraint, the same convention used by FAO's
global standard.

The elementary symmetric functions are computed via the Andersen / Verhelst
recursion in log-space (`logaddexp`) for numerical stability. Standard errors
of $\beta$ come from inverting the analytic Hessian after projecting it onto
the sum-to-zero subspace.

### Cases dropped at fit time

* Rows with **any missing item response** are excluded from the fit.
* Cases with **extreme raw scores** ($r = 0$ or $r = k$) carry no information
  about item severities under CML, so they don't enter $\ell(\beta)$. They do
  re-enter for prevalence calculation.

## Person parameters

Once $\beta$ is estimated, the person parameter $\theta_r$ for each raw score
$r = 1, \ldots, k - 1$ is the value of $\theta$ at which the expected raw score
equals *r*:

$$
r = \sum_{i=1}^{k} \frac{1}{1 + \exp(\beta_i - \theta_r)}.
$$

The corresponding measurement error is

$$
\mathrm{se}(\theta_r) = \left( \sum_{i=1}^{k} p_i (1 - p_i) \right)^{-1/2},
\quad p_i = \frac{1}{1 + \exp(\beta_i - \theta_r)}.
$$

Extreme raw scores ($r = 0$ and $r = k$) are undefined under standard MLE —
the score equation has no finite solution. pyFIES (following `RM.weights`)
handles them by solving for *pseudo* raw scores $d_0 \in (0, 1)$ and
$d_k \in (k-1, k)$, defaulting to $0.5$ and $k - 0.5$ respectively.

## Equating to a reference standard

Direct comparison of $\beta$ across surveys is invalid because each survey's
metric is identified up to an arbitrary linear transformation. *Equating*
finds the scale and shift that make the country metric comparable to a
reference (typically the FAO 2014–2016 global standard).

The algorithm — implemented in `pyfies.core.equating.equate` — is iterative:

1. Standardize the country's $\beta$ to match the reference's mean and
   standard deviation on a candidate set of common items (initially all
   items).
2. Find the item with the largest absolute discrepancy from the reference. If
   it exceeds `tol` (default 0.35), flag it as *unique* (i.e., interpreted
   differently in this context) and recompute scale and shift from the
   remaining common items. Repeat until no item exceeds `tol` or the
   maximum number of unique items is reached.
3. After convergence, compute the final scale and shift in one shot from the
   raw $\beta$ and the converged common-items mask:

   $$
   \mathrm{scale} = \frac{\sigma(\beta_{\text{ref}}[\text{common}])}{\sigma(\beta[\text{common}])},
   \qquad
   \mathrm{shift} = \mu(\beta_{\text{ref}}[\text{common}]) - \mu(\beta[\text{common}]) \cdot \mathrm{scale}.
   $$

The country severities map onto the reference metric as
$\beta_{\text{ref-metric}} = \mathrm{shift} + \mathrm{scale} \cdot \beta$.
Equivalently, the reference's prevalence thresholds map onto the country
metric as $t_{\text{country}} = (t_{\text{ref}} - \mathrm{shift}) / \mathrm{scale}$.

## Probabilistic prevalence assignment

The final step turns person parameters into the SDG 2.1.2 prevalence rates.
For each latent threshold $t$ (typically the moderate-or-severe and severe
thresholds from the reference standard), pyFIES computes

$$
P(\text{severity} > t) = \sum_{r = 1}^{k}
    \left[ 1 - \Phi\!\left( \frac{t - \theta_r}{\mathrm{se}(\theta_r)} \right) \right] \cdot f_r,
$$

where $\Phi$ is the standard normal CDF and $f_r$ is the weighted population
proportion at raw score *r* (normalized over **all** raw scores). The sum
starts at $r = 1$ because respondents at $r = 0$ are by construction below
any non-trivial FI threshold and contribute zero.

Conceptually: each raw score's contribution is a Gaussian "smear" of severity
centered at $\theta_r$ with SD $\mathrm{se}(\theta_r)$. The marginal
prevalence is the population-weighted mixture of how much of each smear lies
above the threshold.
