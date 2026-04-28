"""Post-hoc maximum-likelihood estimation of person parameters.

Once item severities :math:`\\beta` are estimated by CML, person parameters
:math:`\\theta_r` for each raw score *r* = 1, ..., *k - 1* are obtained by
solving the marginal score equation

.. math::
    r = \\sum_{i=1}^{k} \\frac{1}{1 + \\exp(\\beta_i - \\theta_r)} ,

i.e. the value of :math:`\\theta` at which the expected raw score under the
Rasch model equals the observed raw score *r*. The corresponding measurement
error is

.. math::
    \\mathrm{se}(\\theta_r) = \\Big(\\sum_{i=1}^{k} p_i (1 - p_i)\\Big)^{-1/2},
    \\quad p_i = \\frac{1}{1 + \\exp(\\beta_i - \\theta_r)} .

Extreme raw scores (*r* = 0 and *r* = *k*) are undefined under standard MLE.
Following RM.weights we estimate them by solving for pseudo-raw-scores
:math:`d_0 \\in (0, 1)` and :math:`d_k \\in (k - 1, k)` (defaults: 0.5 and
*k* - 0.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


@dataclass
class PersonParameters:
    """Person severity per raw score.

    Attributes:
        theta: Estimated person severity for each raw score *r* = 0, ..., *k*,
            shape ``(k+1,)``. Entries 0 and *k* use the pseudo-raw-score
            assumptions in ``pseudo_extreme``.
        se_theta: Measurement errors for ``theta``, shape ``(k+1,)``.
        pseudo_extreme: The pseudo raw scores ``(d0, dk)`` used for the two
            extreme entries.
    """

    theta: NDArray[np.float64]
    se_theta: NDArray[np.float64]
    pseudo_extreme: tuple[float, float]


def _expected_raw_score(theta: float, beta: NDArray[np.float64]) -> float:
    # P(X_i = 1 | theta) for the Rasch model.
    return float(np.sum(1.0 / (1.0 + np.exp(beta - theta))))


def _se_theta(theta: float, beta: NDArray[np.float64]) -> float:
    p = 1.0 / (1.0 + np.exp(beta - theta))
    info = float(np.sum(p * (1.0 - p)))
    return float(np.sqrt(1.0 / info)) if info > 0 else float("inf")


def fit_person_parameters(
    beta: NDArray[np.float64],
    pseudo_extreme: tuple[float, float] | None = None,
    bracket: tuple[float, float] = (-20.0, 20.0),
    xtol: float = 1e-10,
) -> PersonParameters:
    """Estimate :math:`\\theta_r` for every raw score given item severities.

    Args:
        beta: Item severities, shape ``(k,)``.
        pseudo_extreme: Pseudo raw scores ``(d0, dk)`` for the two extreme
            scores. Defaults to ``(0.5, k - 0.5)``.
        bracket: Bracket passed to ``scipy.optimize.brentq`` when inverting
            the score equation. Wide enough for any plausible FIES fit.
        xtol: Absolute tolerance on theta in the root finder.

    Returns:
        :class:`PersonParameters` with ``theta`` and ``se_theta`` of shape
        ``(k + 1,)``.
    """
    beta = np.asarray(beta, dtype=np.float64)
    k = beta.shape[0]
    if pseudo_extreme is None:
        d0, dk = 0.5, k - 0.5
    else:
        d0, dk = pseudo_extreme
        if not (0.0 < d0 < 1.0):
            raise ValueError("pseudo extreme d0 must be in (0, 1)")
        if not (k - 1.0 < dk < k):
            raise ValueError(f"pseudo extreme dk must be in ({k - 1}, {k})")

    theta = np.zeros(k + 1, dtype=np.float64)
    se = np.zeros(k + 1, dtype=np.float64)

    # Interior raw scores: solve E[R | theta] = r exactly.
    for r in range(1, k):
        theta_r = brentq(
            lambda t, target=float(r): _expected_raw_score(t, beta) - target,
            bracket[0],
            bracket[1],
            xtol=xtol,
        )
        theta[r] = theta_r
        se[r] = _se_theta(theta_r, beta)

    # Extreme raw scores via pseudo-raw-score assumption.
    for r, target in ((0, d0), (k, dk)):
        theta_r = brentq(
            lambda t, t0=target: _expected_raw_score(t, beta) - t0,
            bracket[0],
            bracket[1],
            xtol=xtol,
        )
        theta[r] = theta_r
        se[r] = _se_theta(theta_r, beta)

    return PersonParameters(
        theta=theta, se_theta=se, pseudo_extreme=(float(d0), float(dk))
    )
