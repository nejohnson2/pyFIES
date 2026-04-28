# API reference

The pyFIES public API is intentionally small. Most users only need:

* [`RaschModel`](model.md) — the sklearn-style class wrapping the full pipeline.
* [`FAO_2014_2016`](standards.md) — the global reference standard.
* `DEFAULT_FIES_ITEMS` — canonical item names.

For users who want direct access to the numerical primitives (e.g., to build
custom workflows or to validate intermediate quantities), the
[numerical core](core.md) is also documented.

## Module layout

```
pyfies
├── RaschModel         — sklearn-style fit/equate/prevalence pipeline
├── FAO_2014_2016      — reference standard constant
├── DEFAULT_FIES_ITEMS — the 8 standard FIES item names
└── core
    ├── cml            — weighted Conditional Maximum Likelihood Rasch fit
    ├── gamma          — log-stable elementary symmetric functions
    ├── person         — post-hoc person parameter MLE
    ├── equating       — iterative scale-and-shift equating
    └── prevalence     — Gaussian-mixture prevalence assignment
```
