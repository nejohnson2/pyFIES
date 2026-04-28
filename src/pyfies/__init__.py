"""pyFIES — Python implementation of FAO's Food Insecurity Experience Scale.

Computes the SDG 2.1.2 indicator (prevalence of moderate-or-severe and severe
food insecurity) via a weighted Conditional Maximum Likelihood Rasch model,
equating to FAO's 2014-2016 global standard, and probabilistic prevalence
assignment along the latent food-insecurity trait.

The reference implementation is the R package RM.weights (Cafiero, Viviani,
Nord). pyFIES is a clean-room reimplementation under Apache-2.0; numerical
parity is validated against R fixtures.
"""

from pyfies.items import DEFAULT_FIES_ITEMS
from pyfies.model import RaschModel
from pyfies.standards import FAO_2014_2016

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_FIES_ITEMS",
    "FAO_2014_2016",
    "RaschModel",
    "__version__",
]
