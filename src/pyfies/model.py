"""High-level scikit-learn-style API for the FIES Rasch workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyfies.core.cml import CMLFit, fit_cml
from pyfies.core.equating import EquatingResult
from pyfies.core.equating import equate as _equate
from pyfies.core.person import PersonParameters, fit_person_parameters
from pyfies.core.prevalence import PrevalenceTable, assign_prevalence
from pyfies.standards import FAO_2014_2016, ReferenceStandard

logger = logging.getLogger(__name__)


@dataclass
class PrevalenceResult:
    """Headline output: SDG 2.1.2 prevalence rates.

    Attributes:
        moderate_or_severe: Estimated prevalence (0-1) of moderate-or-severe
            food insecurity at the equated moderate-or-severe threshold.
        severe: Estimated prevalence (0-1) of severe food insecurity.
        thresholds_country_metric: The two thresholds used, expressed on the
            *country* metric (i.e., reference thresholds back-transformed
            via the equating).
        on_reference_metric: Reference standard against which prevalence
            was equated, or None if no equating was performed.
        table: Underlying :class:`PrevalenceTable` with per-raw-score
            conditional probabilities.
    """

    moderate_or_severe: float
    severe: float
    thresholds_country_metric: tuple[float, float]
    on_reference_metric: ReferenceStandard | None
    table: PrevalenceTable


class RaschModel:
    """Weighted dichotomous Rasch model for FIES data.

    Example:
        >>> from pyfies import RaschModel, FAO_2014_2016
        >>> model = RaschModel().fit(X, sample_weight=w)  # doctest: +SKIP
        >>> model.equate(FAO_2014_2016)                   # doctest: +SKIP
        >>> result = model.prevalence()                   # doctest: +SKIP
        >>> print(result.moderate_or_severe, result.severe)  # doctest: +SKIP
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self._fit: CMLFit | None = None
        self._person: PersonParameters | None = None
        self._equating: EquatingResult | None = None
        self._reference: ReferenceStandard | None = None

    @property
    def beta(self) -> NDArray[np.float64]:
        """Item severities on the country metric (sum-to-zero)."""
        return self._require_fit().beta

    @property
    def se_beta(self) -> NDArray[np.float64]:
        """Asymptotic standard errors of item severities."""
        return self._require_fit().se_beta

    @property
    def theta(self) -> NDArray[np.float64]:
        """Person parameter for each raw score *r* = 0, ..., *k*."""
        return self._require_person().theta

    @property
    def se_theta(self) -> NDArray[np.float64]:
        """Measurement error for each person parameter."""
        return self._require_person().se_theta

    @property
    def equating(self) -> EquatingResult:
        """Equating result. Raises if :meth:`equate` has not been called."""
        if self._equating is None:
            raise RuntimeError("call equate() first")
        return self._equating

    @property
    def equated_beta(self) -> NDArray[np.float64]:
        """Item severities on the reference metric (after equating)."""
        return self.equating.equated_beta

    def fit(
        self,
        X: NDArray[np.int_] | pd.DataFrame,
        sample_weight: NDArray[np.float64] | None = None,
        pseudo_extreme: tuple[float, float] | None = None,
    ) -> RaschModel:
        """Estimate item severities and person parameters.

        Args:
            X: 2-D matrix of dichotomized responses (1 = affirmative,
                0 = negative, NaN = missing). Rows are respondents, columns
                are items in a fixed order.
            sample_weight: Optional sampling weights (one per row). If None,
                all respondents are weighted equally.
            pseudo_extreme: Pseudo raw scores ``(d0, dk)`` used to anchor
                person parameters at the two extreme scores. Defaults to
                ``(0.5, k - 0.5)``.

        Returns:
            Self, to support fluent chaining.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self._fit = fit_cml(
            X, weights=sample_weight, max_iter=self.max_iter, tol=self.tol
        )
        logger.info(
            "CML fit: n_complete=%d / %d, converged=%s, iter=%d",
            self._fit.n_complete,
            self._fit.n_total,
            self._fit.converged,
            self._fit.n_iter,
        )
        self._person = fit_person_parameters(
            self._fit.beta, pseudo_extreme=pseudo_extreme
        )
        # Invalidate any prior equating since β changed.
        self._equating = None
        self._reference = None
        return self

    def equate(
        self,
        reference: ReferenceStandard = FAO_2014_2016,
        tol: float = 0.35,
        max_unique: int = 3,
    ) -> RaschModel:
        """Calibrate the country metric to a reference standard.

        Args:
            reference: Reference scale (default: FAO 2014-2016 global standard).
            tol: Tolerance for flagging an item as unique.
            max_unique: Maximum number of items that may be flagged unique.

        Returns:
            Self, to support fluent chaining.
        """
        thresholds_ref = np.array(
            [
                reference.moderate_or_severe_threshold,
                reference.severe_threshold,
            ],
            dtype=np.float64,
        )
        self._equating = _equate(
            self.beta,
            reference.severities,
            thresholds_ref,
            tol=tol,
            max_unique=max_unique,
        )
        self._reference = reference
        logger.info(
            "Equated to %s: scale=%.4f, shift=%.4f, %d/%d items common",
            reference.name,
            self._equating.scale,
            self._equating.shift,
            int(self._equating.common.sum()),
            len(self._equating.common),
        )
        return self

    def prevalence(
        self,
        thresholds_country_metric: tuple[float, float] | None = None,
    ) -> PrevalenceResult:
        """Compute the SDG 2.1.2 prevalence rates.

        Defaults to using the equated thresholds (the reference standard's
        moderate-or-severe and severe thresholds back-transformed onto the
        country metric). If :meth:`equate` has not been called, falls back to
        the country's own item-5 and item-8 severities.

        Args:
            thresholds_country_metric: Optional override for the two
                thresholds, expressed on the country metric.

        Returns:
            :class:`PrevalenceResult`.
        """
        fit = self._require_fit()
        person = self._require_person()
        k = fit.beta.shape[0]

        if thresholds_country_metric is None:
            if self._equating is not None:
                t_mod = float(self._equating.adj_thresholds[0])
                t_sev = float(self._equating.adj_thresholds[1])
            else:
                # Fall back to the country's items 5 and 8 (FAO convention).
                t_mod = float(fit.beta[4])
                t_sev = float(fit.beta[k - 1])
        else:
            t_mod, t_sev = thresholds_country_metric

        # Raw score frequencies normalized over all raw scores.
        n_total_weight = float(fit.weighted_raw_score_counts.sum())
        f = fit.weighted_raw_score_counts / n_total_weight

        table = assign_prevalence(
            theta=person.theta,
            se_theta=person.se_theta,
            raw_score_freq=f,
            thresholds=np.array([t_mod, t_sev], dtype=np.float64),
        )

        return PrevalenceResult(
            moderate_or_severe=float(table.prevalence[0]),
            severe=float(table.prevalence[1]),
            thresholds_country_metric=(t_mod, t_sev),
            on_reference_metric=self._reference,
            table=table,
        )

    def _require_fit(self) -> CMLFit:
        if self._fit is None:
            raise RuntimeError("call fit() before accessing fitted attributes")
        return self._fit

    def _require_person(self) -> PersonParameters:
        if self._person is None:
            raise RuntimeError("call fit() before accessing fitted attributes")
        return self._person
