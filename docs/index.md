# pyFIES

**Python implementation of FAO's Food Insecurity Experience Scale.**

`pyfies` computes UN SDG indicator **2.1.2** — the prevalence of moderate-or-severe
and severe food insecurity in a population — directly from raw survey responses.
It is a from-scratch port of the R package
[`RM.weights`](https://cran.r-project.org/package=RM.weights) developed by
FAO's Voices of the Hungry team (Cafiero, Viviani, Nord), with numerical
results validated against the reference implementation.

## What it does

Given a matrix of responses to the eight FIES questions and optional sampling
weights, pyFIES estimates:

1. **Item severity parameters** for each FIES question via weighted Conditional
   Maximum Likelihood (CML) on a single-parameter Rasch model.
2. **Person parameters** per raw score, by post-hoc maximum likelihood.
3. **Equating** of the country scale to FAO's 2014–2016 global standard, so
   prevalence rates are comparable across countries and survey rounds.
4. **Prevalence rates** at any latent-trait threshold via Gaussian-mixture
   probabilistic assignment.

## At a glance

```python
from pyfies import RaschModel, FAO_2014_2016, DEFAULT_FIES_ITEMS

X = df[DEFAULT_FIES_ITEMS].to_numpy()        # (n, 8) matrix of 0/1/NaN
w = df["sampling_weight"].to_numpy()         # (n,) sampling weights

model = RaschModel().fit(X, sample_weight=w).equate(FAO_2014_2016)
result = model.prevalence()

print(f"Moderate or severe food insecurity: {result.moderate_or_severe:.1%}")
print(f"Severe food insecurity:             {result.severe:.1%}")
```

## Project status

**v0.1 (alpha)** — covers the dichotomous Rasch model, equating, and prevalence
estimation. Polytomous (partial credit) responses and full diagnostics suites
(item infit/outfit, residual correlations, ICC plots) are planned for v0.2.

Numerical agreement with `RM.weights` is verified on the four FAO sample
countries shipped with the R package: see [Parity](parity.md).

## Citing

If you use pyFIES in published research, please cite both the package and the
underlying FAO methodology (Cafiero, Viviani, & Nord, 2018):

> Cafiero, C., Viviani, S., & Nord, M. (2018). Food security measurement in a
> global context: The Food Insecurity Experience Scale. *Measurement*, 116,
> 146–152. [doi:10.1016/j.measurement.2017.10.065](https://doi.org/10.1016/j.measurement.2017.10.065)

## License

[Apache License 2.0](https://github.com/nejohnson2/pyFIES/blob/main/LICENSE).
