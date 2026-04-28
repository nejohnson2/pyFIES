# Changelog

All notable changes to pyFIES are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-27

### Fixed
- `Documentation` project URL on PyPI now points to the published docs site
  (https://nejohnson2.github.io/pyFIES/) instead of the GitHub README anchor
  it was originally seeded with.

## [0.1.0] - 2026-04-27

First public release.

### Added
- Initial scaffold: package layout, Apache-2.0 license, CI config, Makefile.
- Numerical core: log-stable elementary symmetric functions for CML.
- Weighted Conditional Maximum Likelihood estimator for the dichotomous Rasch model.
- Post-hoc maximum-likelihood estimation of person parameters per raw score.
- Equating to a reference standard via iterative scale-and-shift
  (`pyfies.core.equating.equate`).
- Probabilistic prevalence assignment along the latent FI trait
  (`pyfies.core.prevalence.assign_prevalence`).
- `RaschModel.fit() / .equate() / .prevalence()` end-to-end pipeline for the
  SDG 2.1.2 indicator.
- FAO 2014–2016 global FIES standard as a versioned constant.
- R fixture-generation script and parity tests verifying numerical agreement
  with `RM.weights` on all four FAO sample countries:
  - β within 2e-4
  - θ within 1e-2
  - equating scale & shift within 1e-3
  - common-items mask exact match
  - prevalence rates within 0.5 percentage points
