"""Smoke tests for the weighted CML estimator."""

from __future__ import annotations

import numpy as np
import pytest

from pyfies import RaschModel
from pyfies.core.cml import fit_cml


def _simulate_rasch(
    beta_true: np.ndarray,
    n: int,
    rng: np.random.Generator,
    theta_scale: float = 1.5,
) -> np.ndarray:
    """Generate Rasch responses from a normal latent distribution."""
    theta = rng.normal(0.0, theta_scale, size=n)
    p = 1.0 / (1.0 + np.exp(beta_true[None, :] - theta[:, None]))
    return (rng.uniform(size=p.shape) < p).astype(np.int8)


def test_fit_recovers_severities_8_items() -> None:
    """With ample data, MLE should land near the data-generating severities."""
    rng = np.random.default_rng(20260427)
    beta_true = np.array([-1.22, -0.85, -1.11, 0.35, -0.31, 0.51, 0.75, 1.88])
    beta_true = beta_true - beta_true.mean()  # match sum-to-zero convention
    X = _simulate_rasch(beta_true, n=8000, rng=rng)
    fit = fit_cml(X)
    assert fit.converged
    assert fit.beta.sum() == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(fit.beta, beta_true, atol=0.1)


def test_fit_handles_sample_weights() -> None:
    """Weights should not crash and should produce a finite, sum-to-zero β."""
    rng = np.random.default_rng(0)
    beta_true = np.array([-1.0, 0.0, 1.0, 0.5, -0.5])
    beta_true -= beta_true.mean()
    X = _simulate_rasch(beta_true, n=2000, rng=rng)
    w = rng.uniform(0.1, 5.0, size=X.shape[0])
    fit = fit_cml(X, weights=w)
    assert fit.converged
    assert fit.beta.sum() == pytest.approx(0.0, abs=1e-10)
    assert np.isfinite(fit.se_beta).all()
    assert (fit.se_beta > 0).all()


def test_rasch_model_api_smoke() -> None:
    """The sklearn-style facade should fit and expose β, θ, SEs."""
    rng = np.random.default_rng(42)
    beta_true = np.array([-0.8, 0.0, 0.8])
    X = _simulate_rasch(beta_true, n=1000, rng=rng)
    model = RaschModel().fit(X)
    assert model.beta.shape == (3,)
    assert model.se_beta.shape == (3,)
    # theta has k+1 entries (one per raw score 0..k)
    assert model.theta.shape == (4,)
    assert model.se_theta.shape == (4,)
    # Person parameters should be monotone increasing in raw score.
    assert np.all(np.diff(model.theta) > 0)


def test_missing_rows_dropped() -> None:
    rng = np.random.default_rng(1)
    beta_true = np.zeros(4)
    X = _simulate_rasch(beta_true, n=200, rng=rng).astype(np.float64)
    X[10:20, 0] = np.nan  # 10 incomplete rows
    fit = fit_cml(X)
    assert fit.n_complete == 190
    assert fit.n_total == 200
