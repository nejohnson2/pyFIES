# Parity with RM.weights

pyFIES is a clean-room reimplementation of FAO's R package
[`RM.weights`](https://cran.r-project.org/package=RM.weights). Numerical
agreement is verified against the four FAO sample country datasets shipped
with `RM.weights` (`data.FAO_country1` through `data.FAO_country4`).

## How parity is enforced

The reference outputs come from a single-shot R run:

```bash
make fixtures   # runs scripts/generate_r_fixtures.R
```

This installs `RM.weights` if needed, runs `RM.w`, `prob.assign`, and
`equating.fun` on each sample country, and dumps the numerical outputs as
JSON under `tests/fixtures/r_reference/`. The Python parity tests
(`tests/test_parity_r.py`) reload the input data from those fixtures, re-fit
in pyFIES, and assert agreement to fixed tolerances.

R is not required to run pyFIES, only to regenerate fixtures.

## Tolerances and what to expect

| Quantity | Tolerance | Notes |
|---|---|---|
| Item severities $\beta$ | $2 \times 10^{-4}$ | Limited by finite-precision $\gamma_r$ summation; worst case is country 4 (β spread ≈ 6 units). |
| Person parameters $\theta$ | $10^{-2}$ | $\beta$ noise propagates through the score-equation inversion. |
| Equating scale & shift | $10^{-3}$ | |
| Common-items mask | exact match | All 4 countries flag the same items as unique. |
| Adjusted thresholds | $5 \times 10^{-4}$ | |
| Prevalence rates | $5 \times 10^{-3}$ (= 0.5 pp) | Empirically ≤ 0.3 pp on all 4 countries. |
| `n.compl` (R) ↔ `n_complete_non_extreme` (pyFIES) | exact | |

## Identification convention

`RM.weights` does not strictly enforce a sum-to-zero identification on
$\beta$ — the iterative algorithm leaves a residual offset of order
$10^{-5}$. pyFIES enforces sum-to-zero exactly. The conditional likelihood
is invariant under a uniform $\beta$ shift, so both answers are the same
MLE up to the identification constant. The parity tests subtract `mean(R β)`
from R's reported $\beta$ before comparison.

## When parity could break

* Item severities with very wide spread (say > 6 units between most and
  least severe items) push the elementary symmetric function recursion
  closer to floating-point precision limits. The 2e-4 tolerance accommodates
  this; if a real dataset shows wider spread, expect somewhat larger
  pyFIES-vs-R differences (still well within research reporting precision).
* Optimizer convergence: pyFIES uses SciPy's L-BFGS-B with `gtol=1e-8`.
  If you tighten or loosen `RaschModel(tol=...)` you may move modestly
  closer to or further from R.

## Sample dataset summary

| Country | n | Complete non-extreme | Unique items (vs. global standard) |
|---|---|---|---|
| 1 | 1000 | 423 | WORRIED, HEALTHY |
| 2 | 1000 | 505 | FEWFOOD, SKIPPED |
| 3 | 1008 | 734 | HUNGRY |
| 4 | 1000 | 597 | WORRIED |

These are anonymized Gallup World Poll datasets from the Voices of the
Hungry project, distributed inside `RM.weights` and used here only as
reference numerical fixtures.
