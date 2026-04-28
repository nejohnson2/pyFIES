"""Reference standards for equating country FIES scales to a common metric.

The FAO 2014-2016 global standard is the canonical reference scale for SDG
indicator 2.1.2. Item severities below are taken from the values hardcoded in
``RM.weights::equating.fun`` (Cafiero, Viviani, Nord, 2018), which were
estimated from pooled Gallup World Poll data over 2014-2016.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyfies.items import DEFAULT_FIES_ITEMS


@dataclass(frozen=True)
class ReferenceStandard:
    """A reference scale of item severities used as the target for equating.

    Attributes:
        name: Human-readable identifier for the standard.
        items: Names of the items, in the order matching ``severities``.
        severities: Item severity parameters on the standard's latent metric.
        moderate_or_severe_threshold: Latent-trait threshold (item severity)
            above which respondents are classified as moderately-or-severely
            food insecure. Defaults to the severity of item 5.
        severe_threshold: Latent-trait threshold for severe food insecurity.
            Defaults to the severity of item 8.
    """

    name: str
    items: tuple[str, ...]
    severities: np.ndarray
    moderate_or_severe_threshold: float
    severe_threshold: float


_FAO_2014_2016_SEVERITIES = np.array(
    [
        -1.2230564,  # WORRIED
        -0.8471210,  # HEALTHY
        -1.1056616,  # FEWFOOD
        0.3509848,  # SKIPPED
        -0.3117999,  # ATELESS
        0.5065051,  # RUNOUT
        0.7546138,  # HUNGRY
        1.8755353,  # WHLDAY
    ],
    dtype=np.float64,
)


FAO_2014_2016 = ReferenceStandard(
    name="FAO 2014-2016 global standard",
    items=DEFAULT_FIES_ITEMS,
    severities=_FAO_2014_2016_SEVERITIES,
    moderate_or_severe_threshold=float(_FAO_2014_2016_SEVERITIES[4]),
    severe_threshold=float(_FAO_2014_2016_SEVERITIES[7]),
)
"""FAO's 2014-2016 global FIES standard (the SDG 2.1.2 reference metric)."""
