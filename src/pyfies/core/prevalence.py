"""Probabilistic prevalence assignment along the latent FI trait.

Implements the Gaussian-mixture prevalence formula used by
``RM.weights::prob.assign``: for each latent threshold ``t`` and raw score
*r*, assume the posterior severity for respondents at raw score *r* is
Gaussian with mean :math:`\\theta_r` and standard deviation
:math:`\\mathrm{se}(\\theta_r)`. The marginal prevalence beyond ``t`` is

.. math::
    P(\\text{severity} > t) = \\sum_{r=1}^{k} \\big[ 1 - \\Phi
        \\big( (t - \\theta_r) / \\mathrm{se}(\\theta_r) \\big) \\big] \\cdot f_r,

where :math:`f_r` is the *weighted* proportion of respondents at raw score
*r*, normalized over **all** raw scores 0, ..., *k*. Raw score 0 is
excluded from the sum (respondents with no affirmative responses can't be
food insecure beyond a threshold above the lowest item).

Defaults to :math:`f_r` computed from the model's own raw-score distribution,
matching RM.weights' default behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


@dataclass
class PrevalenceTable:
    """Prevalence rates beyond each latent threshold.

    Attributes:
        thresholds: Latent-trait thresholds at which prevalence was evaluated,
            shape ``(t,)``.
        prevalence: Prevalence (in [0, 1]) beyond each threshold,
            shape ``(t,)``. Population proportion.
        prob_per_raw_score: Conditional probability of being beyond each
            threshold at each raw score, shape ``(k+1, t)``. Row 0 is forced
            to 0 to match the RM.weights convention.
        raw_score_freq: Weighted frequency :math:`f_r` of each raw score,
            normalized over all raw scores including 0, shape ``(k+1,)``.
    """

    thresholds: NDArray[np.float64]
    prevalence: NDArray[np.float64]
    prob_per_raw_score: NDArray[np.float64]
    raw_score_freq: NDArray[np.float64]


def assign_prevalence(
    theta: NDArray[np.float64],
    se_theta: NDArray[np.float64],
    raw_score_freq: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> PrevalenceTable:
    """Compute population prevalence beyond each latent-trait threshold.

    Args:
        theta: Person parameter for each raw score *r* = 0, ..., *k*,
            shape ``(k+1,)``.
        se_theta: Measurement error for each ``theta``, shape ``(k+1,)``.
        raw_score_freq: Weighted relative frequency of each raw score
            *r* = 0, ..., *k*, shape ``(k+1,)``. Should sum to (approximately)
            1; will be used as-is. The entry for raw score 0 is treated as
            0 contribution (population fraction at r = 0 is still subtracted
            from the denominator implicitly because they cannot exceed any
            threshold above the lowest possible severity).
        thresholds: Latent-trait thresholds, shape ``(t,)``.

    Returns:
        :class:`PrevalenceTable`.
    """
    theta = np.asarray(theta, dtype=np.float64)
    se = np.asarray(se_theta, dtype=np.float64)
    f = np.asarray(raw_score_freq, dtype=np.float64).copy()
    thr = np.asarray(thresholds, dtype=np.float64)

    if theta.shape != se.shape:
        raise ValueError("theta and se_theta must have the same shape")
    if theta.shape != f.shape:
        raise ValueError("raw_score_freq must align with theta (length k+1)")
    if np.any(se <= 0):
        raise ValueError("se_theta must be positive")

    # Per-raw-score conditional probability of being beyond each threshold.
    # Shape: (k+1, t)
    z = (thr[None, :] - theta[:, None]) / se[:, None]
    cond_prob = 1.0 - norm.cdf(z)
    # Raw score 0 always contributes 0 (matches RM.weights f_j[1] = 0).
    cond_prob[0, :] = 0.0

    f[0] = 0.0
    prevalence = (cond_prob * f[:, None]).sum(axis=0)

    return PrevalenceTable(
        thresholds=thr,
        prevalence=prevalence,
        prob_per_raw_score=cond_prob,
        raw_score_freq=np.asarray(raw_score_freq, dtype=np.float64),
    )
