"""Equating a country FIES scale to a reference standard.

Implements the iterative scale-and-shift procedure used by RM.weights to
calibrate item severities across measurement contexts. The algorithm:

1. Standardize the country's item severities to match the reference's mean
   and standard deviation on a candidate set of common items (initially
   all items).
2. Compute the absolute discrepancy of each candidate common item from the
   reference. If the largest exceeds ``tol``, flag that item as *unique*
   (i.e. measuring something different in this context) and re-estimate
   scale and shift from the remaining common items. Repeat.
3. Stop when no further items would be flagged, or when the number of
   uniques reaches ``max_unique``.

Final ``scale`` and ``shift`` are recomputed in one shot from the *raw*
country severities and the converged common-items mask:

.. math::
    \\text{scale} = \\sigma(\\beta_{\\text{ref}}[\\text{common}])
                    / \\sigma(\\beta[\\text{common}]),
    \\quad
    \\text{shift} = \\mu(\\beta_{\\text{ref}}[\\text{common}])
                    - \\mu(\\beta[\\text{common}]) \\cdot \\text{scale}.

The country severities map onto the reference metric as
``beta_on_reference = shift + scale * beta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EquatingResult:
    """Output of :func:`equate`.

    Attributes:
        scale: Multiplicative factor mapping country β to the reference metric.
        shift: Additive offset mapping country β to the reference metric.
        common: Boolean mask of items judged common (True) vs unique (False).
        adj_thresholds: Reference thresholds mapped back onto the country
            metric, so prevalence can be computed without rescaling β.
        correlation: Pearson correlation of common items between the country
            (after equating) and the reference scale.
        equated_beta: Country severities transformed onto the reference metric,
            shape ``(k,)``.
        n_iter: Number of iterations actually performed.
    """

    scale: float
    shift: float
    common: NDArray[np.bool_]
    adj_thresholds: NDArray[np.float64]
    correlation: float
    equated_beta: NDArray[np.float64]
    n_iter: int


def equate(
    beta: NDArray[np.float64],
    reference_beta: NDArray[np.float64],
    reference_thresholds: NDArray[np.float64],
    tol: float = 0.35,
    max_unique: int = 3,
) -> EquatingResult:
    """Equate ``beta`` to the metric defined by ``reference_beta``.

    Args:
        beta: Country item severity parameters, shape ``(k,)``.
        reference_beta: Reference item severities (same item ordering),
            shape ``(k,)``.
        reference_thresholds: Latent-trait thresholds on the *reference*
            metric (e.g. severities of items 5 and 8 of the FAO global
            standard), shape ``(t,)``.
        tol: Absolute discrepancy above which an item is flagged as unique.
        max_unique: Maximum number of items that may be flagged as unique
            (also a hard cap on iterations).

    Returns:
        :class:`EquatingResult`.
    """
    b1 = np.asarray(beta, dtype=np.float64)
    b_ref = np.asarray(reference_beta, dtype=np.float64)
    thres_ref = np.asarray(reference_thresholds, dtype=np.float64)
    if b1.shape != b_ref.shape:
        raise ValueError(
            f"beta has {b1.shape} but reference_beta has {b_ref.shape}"
        )
    k = b1.shape[0]
    if k < 3:
        raise ValueError("equating requires at least 3 items")

    # All items start as candidate common.
    common = np.ones(k, dtype=bool)

    # Initial standardization to common items' moments on the reference scale.
    b_st = _standardize(b1, b_ref, common)

    n_iter = 0
    for _ in range(max_unique + 1):
        n_iter += 1
        diff = np.where(common, np.abs(b_st - b_ref), -np.inf)
        worst = int(np.argmax(diff))
        if not np.isfinite(diff[worst]) or diff[worst] < tol:
            break
        if int(common.sum()) - 1 < 2:
            # Need at least 2 common items to estimate scale/shift.
            break
        common[worst] = False
        if int((~common).sum()) > max_unique:
            common[worst] = True  # revert: would exceed cap
            break
        # Re-equate using only remaining common items.
        scale1 = float(b_ref[common].std(ddof=1) / b_st[common].std(ddof=1))
        shift1 = float(b_ref[common].mean() - b_st[common].mean() * scale1)
        b_st = shift1 + scale1 * b_st

    # Final scale and shift recomputed from RAW β and the converged mask.
    scale = float(b_ref[common].std(ddof=1) / b1[common].std(ddof=1))
    shift = float(b_ref[common].mean() - b1[common].mean() * scale)
    equated_beta = shift + scale * b1
    adj_thresholds = (thres_ref - shift) / scale

    if int(common.sum()) >= 2:
        correlation = float(
            np.corrcoef(equated_beta[common], b_ref[common])[0, 1]
        )
    else:
        correlation = float("nan")

    return EquatingResult(
        scale=scale,
        shift=shift,
        common=common,
        adj_thresholds=adj_thresholds,
        correlation=correlation,
        equated_beta=equated_beta,
        n_iter=n_iter,
    )


def _standardize(
    b1: NDArray[np.float64],
    b_ref: NDArray[np.float64],
    common: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Linearly map ``b1`` so that its common-items mean and SD match ``b_ref``."""
    mu1 = float(b1[common].mean())
    sd1 = float(b1[common].std(ddof=1))
    mu2 = float(b_ref[common].mean())
    sd2 = float(b_ref[common].std(ddof=1))
    return (b1 - mu1) / sd1 * sd2 + mu2
