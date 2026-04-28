# Quickstart

## Installation

pyFIES requires Python 3.11 or newer.

```bash
pip install pyfies
```

For development (tests, linting, docs):

```bash
git clone https://github.com/nejohnson2/pyFIES
cd pyFIES
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Five-minute tutorial

### 1. Prepare your data

The FIES has eight questions, asked in a fixed order. Responses are
*dichotomized* before analysis:

* `Never` → 0
* `Rarely`, `Sometimes`, `Often` → 1
* Missing or "Don't know" → `NaN`

```python
import pandas as pd
import numpy as np
from pyfies import DEFAULT_FIES_ITEMS

# DEFAULT_FIES_ITEMS = ('WORRIED', 'HEALTHY', 'FEWFOOD', 'SKIPPED',
#                       'ATELESS', 'RUNOUT', 'HUNGRY', 'WHLDAY')

df = pd.read_csv("my_survey.csv")
X = df[list(DEFAULT_FIES_ITEMS)].to_numpy(dtype=np.float64)
w = df["weight"].to_numpy(dtype=np.float64)  # optional sampling weights
```

### 2. Fit the Rasch model

```python
from pyfies import RaschModel

model = RaschModel().fit(X, sample_weight=w)
```

After `fit()` you have access to:

* `model.beta` — item severities on the country metric (sum-to-zero).
* `model.se_beta` — asymptotic standard errors of `model.beta`.
* `model.theta` — person severity for each raw score (length `k+1`).
* `model.se_theta` — measurement error per person parameter.

### 3. Equate to the global metric

```python
from pyfies import FAO_2014_2016

model.equate(FAO_2014_2016)

print("scale:", model.equating.scale)
print("shift:", model.equating.shift)
print("unique items:",
      [name for name, common in zip(DEFAULT_FIES_ITEMS, model.equating.common)
       if not common])
print("correlation of common items:", model.equating.correlation)
```

The `equate()` step calibrates the country's metric onto the FAO 2014–2016
global standard. Items whose post-equating discrepancy from the standard
exceeds `tol` (default 0.35) are flagged as *unique* and removed from the
calibration. See [Methodology](methodology.md) for the algorithm.

### 4. Compute prevalence — the SDG 2.1.2 indicator

```python
result = model.prevalence()

print(f"Moderate or severe food insecurity: {result.moderate_or_severe:.1%}")
print(f"Severe food insecurity:             {result.severe:.1%}")
```

`prevalence()` uses the equated thresholds — that is, the FAO global standard's
moderate-or-severe and severe FI thresholds, back-transformed to the country
metric so they can be applied to the country's own person parameters. If you
skip `equate()`, prevalence falls back to the country's own item-5 and item-8
severities (which is rarely what you want for cross-country comparisons).

### 5. End-to-end, one chain

```python
from pyfies import RaschModel, FAO_2014_2016

result = (
    RaschModel()
    .fit(X, sample_weight=w)
    .equate(FAO_2014_2016)
    .prevalence()
)
print(result.moderate_or_severe, result.severe)
```

## What's next

* Read the [Methodology](methodology.md) page for the math behind each step.
* See the [API reference](api/index.md) for full signatures and options.
* If you previously used R's `RM.weights`, the [Parity](parity.md) page
  documents what numerical agreement to expect.
