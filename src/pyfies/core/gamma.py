"""Elementary symmetric functions of item easinesses, in log-space.

The Conditional Maximum Likelihood (CML) estimator for the Rasch model relies
on the elementary symmetric functions

.. math::
    \\gamma_r(\\varepsilon) = \\sum_{|J|=r} \\prod_{j \\in J} \\varepsilon_j

where :math:`\\varepsilon_i = \\exp(-\\beta_i)` is the easiness of item *i*
parametrized via its severity :math:`\\beta_i`. The CML likelihood is

.. math::
    L(\\beta) = - \\sum_i T_i \\beta_i - \\sum_r N_r \\log \\gamma_r(\\varepsilon),

where :math:`T_i` is the (weighted) number of affirmative responses to item *i*
and :math:`N_r` is the (weighted) number of respondents with raw score *r*.

For numerical stability we always compute :math:`\\log \\gamma_r` via the
Andersen / Verhelst recursion combined with ``logaddexp``.

References:
    Andersen, E. B. (1972). The numerical solution of a set of conditional
    estimation equations. *J. Roy. Statist. Soc. B*, 34, 42-54.

    Verhelst, N. D., Glas, C. A. W., & van der Sluis, A. (1984). Estimation
    problems in the Rasch model: The basic symmetric functions.
    *Computational Statistics Quarterly*, 1, 245-262.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def log_gamma(beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute :math:`\\log \\gamma_r` for *r* = 0, ..., *k*.

    Args:
        beta: Item severity parameters, shape ``(k,)``.

    Returns:
        Array of shape ``(k+1,)`` whose *r*-th entry is
        :math:`\\log \\gamma_r(\\exp(-\\beta))`.
    """
    k = beta.shape[0]
    log_eps = -beta.astype(np.float64, copy=False)
    g = np.full(k + 1, -np.inf, dtype=np.float64)
    g[0] = 0.0
    for j in range(k):
        # Update in reverse so that g[r-1] is still the j-1 value when used.
        for r in range(j + 1, 0, -1):
            g[r] = np.logaddexp(g[r], log_eps[j] + g[r - 1])
    return g


def log_gamma_minus_one(beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute :math:`\\log \\gamma_r^{(-i)}` for every item *i* and order *r*.

    Returns the elementary symmetric functions of the easinesses with item *i*
    excluded. Used to evaluate the CML score and the conditional probability
    that a respondent with raw score *r* endorses item *i*:

    .. math::
        P(X_i = 1 \\mid R = r) = \\varepsilon_i \\,
            \\gamma_{r-1}^{(-i)} / \\gamma_r .

    Args:
        beta: Item severity parameters, shape ``(k,)``.

    Returns:
        Array of shape ``(k, k+1)``. Entry ``[i, r]`` is
        :math:`\\log \\gamma_r^{(-i)}`. Entry ``[i, k]`` is :math:`-\\infty`
        (cannot form a subset of size *k* from *k-1* items).
    """
    k = beta.shape[0]
    log_eps = -beta.astype(np.float64, copy=False)
    out = np.full((k, k + 1), -np.inf, dtype=np.float64)
    for i in range(k):
        g = np.full(k + 1, -np.inf, dtype=np.float64)
        g[0] = 0.0
        count = 0  # how many items have been folded in so far
        for j in range(k):
            if j == i:
                continue
            count += 1
            for r in range(count, 0, -1):
                g[r] = np.logaddexp(g[r], log_eps[j] + g[r - 1])
        out[i, :] = g
    return out


def conditional_endorsement_prob(beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Probability of endorsing each item conditional on each raw score.

    Returns ``pi[r, i] = P(X_i = 1 | R = r)`` for raw scores
    *r* = 0, ..., *k* and items *i* = 0, ..., *k - 1*.

    By construction ``pi[0, :] = 0`` and ``pi[k, :] = 1``: a respondent with
    raw score 0 endorses no items, one with raw score *k* endorses all of them.
    """
    k = beta.shape[0]
    log_eps = -beta.astype(np.float64, copy=False)
    log_g = log_gamma(beta)  # (k+1,)
    log_g_minus = log_gamma_minus_one(beta)  # (k, k+1)
    pi = np.zeros((k + 1, k), dtype=np.float64)
    # Raw scores 1..k-1: pi[r, i] = exp(log_eps[i] + log_g_minus[i, r-1] - log_g[r])
    # log_g[r] is finite for 0 <= r <= k.
    for r in range(1, k):
        for i in range(k):
            log_pi = log_eps[i] + log_g_minus[i, r - 1] - log_g[r]
            pi[r, i] = float(np.exp(log_pi))
    pi[k, :] = 1.0
    return pi
