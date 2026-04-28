# pyFIES

[![PyPI](https://img.shields.io/pypi/v/pyfies.svg)](https://pypi.org/project/pyfies/)
[![Python](https://img.shields.io/pypi/pyversions/pyfies.svg)](https://pypi.org/project/pyfies/)
[![CI](https://github.com/nejohnson2/pyFIES/actions/workflows/ci.yml/badge.svg)](https://github.com/nejohnson2/pyFIES/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://nejohnson2.github.io/pyFIES/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19838795.svg)](https://doi.org/10.5281/zenodo.19838795)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Python implementation of FAO's **Food Insecurity Experience Scale (FIES)** — the
methodology behind UN SDG indicator **2.1.2** (prevalence of moderate-or-severe
and severe food insecurity in the population).

`pyfies` is a from-scratch port of the R package
[`RM.weights`](https://cran.r-project.org/package=RM.weights) (Cafiero, Viviani, Nord),
implementing the weighted Conditional Maximum Likelihood (CML) Rasch estimator,
equating to FAO's 2014–2016 global standard, and probabilistic prevalence
assignment along the latent food-insecurity trait.

> **Status:** alpha. v0.1 covers the dichotomous Rasch model, equating, and
> prevalence estimation. The polytomous (partial credit) extension and full
> diagnostics suite are planned for v0.2.

## Installation

```bash
pip install pyfies          # from PyPI
pip install -e ".[dev]"     # editable install with dev tooling
```

Requires Python 3.11+.

## Quickstart

```python
import pandas as pd
from pyfies import RaschModel, FAO_2014_2016, DEFAULT_FIES_ITEMS

# df has 8 columns named WORRIED, HEALTHY, FEWFOOD, SKIPPED,
# ATELESS, RUNOUT, HUNGRY, WHLDAY (1=affirmative, 0=negative, NA=missing)
X = df[DEFAULT_FIES_ITEMS].to_numpy()
w = df["sampling_weight"].to_numpy()

model = RaschModel().fit(X, sample_weight=w)
model.equate(reference=FAO_2014_2016)
result = model.prevalence()

print(f"Moderate or severe food insecurity: {result.moderate_or_severe:.1%}")
print(f"Severe food insecurity:             {result.severe:.1%}")
```

## What it computes

1. **Item severity parameters** for each FIES question via weighted CML.
2. **Person parameters** per raw score (post-hoc MLE conditional on item parameters).
3. **Equating** to the FAO 2014–2016 global metric so prevalence rates are
   comparable across countries and survey rounds.
4. **Prevalence rates** at any latent-trait threshold via Gaussian-mixture
   probabilistic assignment.

## Methodology references

- FAO. *The Food Insecurity Experience Scale — Development of a Global Standard
  for Monitoring Hunger Worldwide.* Technical Paper v1.1, 2016.
  [link](https://www.fao.org/fileadmin/templates/ess/voh/FIES_Technical_Paper_v1.1.pdf)
- Cafiero, C., Viviani, S., Nord, M. (2018). *Food security measurement in a
  global context: The Food Insecurity Experience Scale.* Measurement, 116, 146–152.
  [doi:10.1016/j.measurement.2017.10.065](https://doi.org/10.1016/j.measurement.2017.10.065)
- FAO. [Voices of the Hungry / FIES landing page](https://www.fao.org/measuring-hunger/access-to-food/about-the-food-insecurity-experience-scale-(fies)/en).

## Numerical parity with `RM.weights`

pyFIES is validated against the reference R package on the `data.FAO_country1..4`
sample datasets shipped with `RM.weights`. To regenerate parity fixtures (R
required, one-time):

```bash
Rscript scripts/generate_r_fixtures.R
```

Tolerances: `atol=2e-4` on item severities, `atol=5e-3` on prevalence rates
(typically achieved within 0.3 percentage points). See
[Parity](https://nejohnson2.github.io/pyFIES/parity/) for details.

## Citing

If you use pyFIES in published research, please cite both the package and the
underlying FAO methodology. The package's "Cite this repository" button on
GitHub (powered by [`CITATION.cff`](CITATION.cff)) will give you the
package citation in BibTeX, APA, and other formats. The Zenodo Concept DOI

> [10.5281/zenodo.19838795](https://doi.org/10.5281/zenodo.19838795)

always resolves to the latest release. To pin to a specific version, use the
version-specific DOI shown on the [Zenodo record page](https://doi.org/10.5281/zenodo.19838795).

For the underlying methodology, cite Cafiero, C., Viviani, S., & Nord, M.
(2018). *Food security measurement in a global context: The Food Insecurity
Experience Scale.* Measurement, 116, 146–152.
[doi:10.1016/j.measurement.2017.10.065](https://doi.org/10.1016/j.measurement.2017.10.065).

## License

[Apache 2.0](LICENSE).
