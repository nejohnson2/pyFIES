# Contributing to pyFIES

Thanks for your interest in contributing. pyFIES aims to be the canonical Python
implementation of FAO's Food Insecurity Experience Scale methodology, with
numerical results that match the reference R package `RM.weights` to within
floating-point precision.

## Development setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running the test suite

```bash
make test           # fast unit tests
make test-parity    # bit-for-bit parity against R fixtures (requires fixtures)
make lint           # ruff + mypy
```

## Numerical correctness

Any change that touches `src/pyfies/core/` must be validated against the
reference R fixtures in `tests/fixtures/r_reference/`. To regenerate fixtures,
install R and run:

```bash
Rscript scripts/generate_r_fixtures.R
```

Tolerances default to `atol=1e-6` for item severities and prevalence rates.
If a test reveals a tolerance that needs loosening, document the source of the
discrepancy in the test and in the PR description.

## Coding standards

- Python 3.11+. Type hints on public APIs.
- `ruff format` and `ruff check` are enforced in CI.
- Public functions need a docstring with a `References:` section pointing to the
  FIES Technical Paper or relevant Rasch literature when implementing
  documented methodology.
- Never copy code verbatim from `RM.weights` (GPL-3). Reimplement from the
  published methodology and Rasch literature.

## Reporting bugs

Open an issue with: input data shape, sampling weights summary, expected vs.
observed output, and the version of pyFIES (`pip show pyfies`).
