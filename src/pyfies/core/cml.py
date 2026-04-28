"""Weighted Conditional Maximum Likelihood estimator for the dichotomous Rasch model.

Estimates item severity parameters :math:`\\beta_1, \\ldots, \\beta_k` by
maximizing the conditional likelihood of the response patterns given the raw
scores. The conditional likelihood is invariant under a uniform shift of all
severities; we resolve this with a sum-to-zero identification constraint
(the same convention used by FAO's RM.weights and the global standard
itself). Rows with any missing item response are dropped before fitting.

The negative conditional log-likelihood is convex in :math:`\\beta`, so a
quasi-Newton method (L-BFGS-B with analytic gradient) converges to the unique
MLE.

References:
    Cafiero, C., Viviani, S., & Nord, M. (2018). Food security measurement in a
    global context: The Food Insecurity Experience Scale. *Measurement*, 116,
    146-152.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pyfies.core.gamma import conditional_endorsement_prob, log_gamma


@dataclass
class CMLFit:
    """Result of a weighted CML fit.

    Attributes:
        beta: Item severities (sum-to-zero), shape ``(k,)``.
        se_beta: Asymptotic standard errors of ``beta``, shape ``(k,)``.
        n_complete: Number of complete cases used for estimation
            (rows with no missing responses, any raw score).
        n_complete_non_extreme: Number of complete cases with non-extreme
            raw scores (1 <= r <= k-1). Only these contribute to the CML
            log-likelihood; matches the ``n.compl`` field reported by
            RM.weights.
        n_total: Total rows in the input.
        weighted_raw_score_counts: Weighted count :math:`N_r` for each
            raw score *r* = 0, ..., *k*, shape ``(k+1,)``.
        weighted_item_totals: Weighted endorsement count :math:`T_i` for each
            item, shape ``(k,)``. Computed over complete cases only.
        loglik: Final conditional log-likelihood at the MLE.
        converged: Whether the optimizer reported convergence.
        n_iter: Number of optimizer iterations.
    """

    beta: NDArray[np.float64]
    se_beta: NDArray[np.float64]
    n_complete: int
    n_complete_non_extreme: int
    n_total: int
    weighted_raw_score_counts: NDArray[np.float64]
    weighted_item_totals: NDArray[np.float64]
    loglik: float
    converged: bool
    n_iter: int


def _normalize_weights(
    weights: NDArray[np.float64] | None, n: int
) -> NDArray[np.float64]:
    """Rescale weights to sum to *n*, matching RM.weights' convention.

    This keeps the effective sample size on the same scale as an unweighted fit
    so that asymptotic standard errors remain interpretable.
    """
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} does not match n={n}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    return w * (n / total)


def _expand_to_full(beta_free: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map :math:`(k-1)` free parameters to *k* sum-to-zero severities."""
    last = -float(beta_free.sum())
    return np.concatenate([beta_free, [last]])


def _negative_loglik_and_grad(
    beta_free: NDArray[np.float64],
    item_totals: NDArray[np.float64],
    raw_score_counts: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Negative CML log-likelihood and its gradient w.r.t. the *k-1* free params."""
    beta = _expand_to_full(beta_free)
    log_g = log_gamma(beta)  # (k+1,)
    # log L (up to constant) = -sum_i T_i β_i - sum_r N_r log γ_r
    neg_ll = float(item_totals @ beta + raw_score_counts @ log_g)

    # Gradient w.r.t. full β:
    # ∂(-L)/∂β_i = T_i - sum_r N_r * (ε_i γ_{r-1}^{(-i)} / γ_r)
    #            = T_i - E[T_i]
    pi = conditional_endorsement_prob(beta)  # (k+1, k), pi[r, i] = P(X_i=1 | R=r)
    expected_totals = raw_score_counts @ pi  # (k,)
    grad_full = item_totals - expected_totals  # (k,)

    # Sum-to-zero reparametrization: β_k = -sum(β_free), so dβ_k/dβ_free_i = -1.
    grad_free = grad_full[:-1] - grad_full[-1]
    return neg_ll, grad_free


def _hessian_full(
    beta: NDArray[np.float64],
    raw_score_counts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Hessian of the negative CML log-likelihood w.r.t. the full :math:`\\beta`.

    Diagonal:  H_ii = sum_r N_r * pi[r,i] * (1 - pi[r,i])
                      + sum_r N_r * (pi[r,i]^2 - pi2[r,i,i])
    Off-diag:  H_ij = - sum_r N_r * (pi2[r,i,j] - pi[r,i] * pi[r,j])

    where pi2[r,i,j] = P(X_i=1, X_j=1 | R=r). For the dichotomous Rasch model,
    these can be derived from elementary symmetric functions excluding two
    items at a time. For Hessian-free estimation we never call this; it's
    provided for SE computation.

    The cleanest closed form: the conditional joint endorsement probabilities
    can be obtained from γ functions excluding pairs of items. Implemented here
    via the relation
        pi2[r, i, j] = ε_i ε_j γ_{r-2}^{(-i,-j)} / γ_r,  i != j .
    """
    k = beta.shape[0]
    log_eps = -beta
    log_g = log_gamma(beta)
    pi = conditional_endorsement_prob(beta)  # (k+1, k)

    # Compute log γ^{(-i,-j)} for all pairs (i, j), i != j: shape (k, k, k+1).
    # γ^{(-i,-j)} excludes both items i and j; max degree is k-2.
    log_g_minus2 = np.full((k, k, k + 1), -np.inf, dtype=np.float64)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            g = np.full(k + 1, -np.inf, dtype=np.float64)
            g[0] = 0.0
            count = 0
            for m in range(k):
                if m in (i, j):
                    continue
                count += 1
                for r in range(count, 0, -1):
                    g[r] = np.logaddexp(g[r], log_eps[m] + g[r - 1])
            log_g_minus2[i, j, :] = g

    H = np.zeros((k, k), dtype=np.float64)
    # Diagonal entries: variance of X_i conditional on R=r, summed over r weighted by N_r.
    for i in range(k):
        var_i = pi[:, i] * (1.0 - pi[:, i])
        H[i, i] = float(raw_score_counts @ var_i)
    # Off-diagonal entries: -Cov(X_i, X_j | R=r).
    for i in range(k):
        for j in range(i + 1, k):
            pi2 = np.zeros(k + 1, dtype=np.float64)
            for r in range(2, k + 1):
                # Need γ_{r-2}^{(-i,-j)}; for r-2 > k-2 it doesn't exist (-> 0).
                if r - 2 <= k - 2:
                    log_p = (
                        log_eps[i] + log_eps[j] + log_g_minus2[i, j, r - 2] - log_g[r]
                    )
                    pi2[r] = float(np.exp(log_p))
            # When r >= k, pi2 collapses to 1 if r == k (both items endorsed by definition).
            if k >= 2:
                pi2[k] = 1.0
            cov = pi2 - pi[:, i] * pi[:, j]
            H[i, j] = -float(raw_score_counts @ cov)
            H[j, i] = H[i, j]
    return H


def fit_cml(
    data: NDArray[np.int_],
    weights: NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> CMLFit:
    """Fit the dichotomous Rasch model by weighted CML.

    Args:
        data: Response matrix of shape ``(n, k)`` with values in {0, 1} or NaN
            for missing. Affirmative responses are coded 1.
        weights: Optional sampling weights of shape ``(n,)``. If omitted, equal
            weights are used. Weights are renormalized so they sum to *n*.
        max_iter: Maximum optimizer iterations.
        tol: Optimizer convergence tolerance on the gradient infinity norm.

    Returns:
        A :class:`CMLFit` with item severities, standard errors, and summary
        statistics.

    Raises:
        ValueError: If the matrix has fewer than 2 items, no complete cases
            with non-extreme raw scores, or invalid weights.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("data must be a 2-D array")
    n, k = arr.shape
    if k < 2:
        raise ValueError("need at least 2 items to identify a Rasch model")

    w = _normalize_weights(weights, n)

    complete = ~np.isnan(arr).any(axis=1)
    n_complete = int(complete.sum())
    if n_complete == 0:
        raise ValueError("no complete cases; cannot fit Rasch model")

    X = arr[complete].astype(np.int8)
    w_complete = w[complete]
    raw = X.sum(axis=1)

    # Weighted item totals T_i and weighted raw-score counts N_r.
    item_totals = (X.T @ w_complete).astype(np.float64)
    raw_score_counts = np.zeros(k + 1, dtype=np.float64)
    for r in range(k + 1):
        raw_score_counts[r] = float(w_complete[raw == r].sum())

    # Extreme raw scores (0 and k) carry no information about item severities
    # under CML; they enter the prevalence calculation later but not the fit.
    informative = raw_score_counts.copy()
    informative[0] = 0.0
    informative[k] = 0.0

    # Initial guess: mild informative prior centered at zero.
    p_endorse = np.clip(item_totals / max(w_complete.sum(), 1.0), 1e-3, 1.0 - 1e-3)
    init_beta = -np.log(p_endorse / (1.0 - p_endorse))
    init_beta -= init_beta.mean()  # enforce sum-to-zero
    init_free = init_beta[:-1].copy()

    result = minimize(
        _negative_loglik_and_grad,
        init_free,
        args=(item_totals, informative),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "gtol": tol},
    )

    beta = _expand_to_full(result.x)

    # Standard errors: invert the Hessian on the (k-1)-dim free parametrization,
    # then map back. The Hessian on the full β is rank k-1 (singular by the
    # sum-to-zero constraint); we project it.
    H_full = _hessian_full(beta, informative)
    # Apply sum-to-zero constraint via the (k, k-1) Jacobian J where the i-th
    # free param maps to β_i and the last full param is -sum.
    J = np.vstack([np.eye(k - 1), -np.ones((1, k - 1))])  # (k, k-1)
    H_free = J.T @ H_full @ J
    cov_free = np.linalg.inv(H_free)
    cov_full = J @ cov_free @ J.T
    se_beta = np.sqrt(np.maximum(np.diag(cov_full), 0.0))

    n_non_extreme = int(((raw >= 1) & (raw <= k - 1)).sum())

    return CMLFit(
        beta=beta,
        se_beta=se_beta,
        n_complete=n_complete,
        n_complete_non_extreme=n_non_extreme,
        n_total=n,
        weighted_raw_score_counts=raw_score_counts,
        weighted_item_totals=item_totals,
        loglik=-float(result.fun),
        converged=bool(result.success),
        n_iter=int(result.nit),
    )
