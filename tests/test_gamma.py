"""Tests for the elementary symmetric function recursion."""

from __future__ import annotations

import numpy as np
import pytest

from pyfies.core.gamma import (
    conditional_endorsement_prob,
    log_gamma,
    log_gamma_minus_one,
)


def _gamma_brute(beta: np.ndarray) -> np.ndarray:
    """Brute-force elementary symmetric functions via subset enumeration."""
    from itertools import combinations

    eps = np.exp(-beta)
    k = len(beta)
    g = np.zeros(k + 1)
    g[0] = 1.0
    for r in range(1, k + 1):
        s = 0.0
        for subset in combinations(range(k), r):
            s += float(np.prod([eps[i] for i in subset]))
        g[r] = s
    return g


@pytest.mark.parametrize(
    "beta",
    [
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([-1.22, -0.85, -1.11, 0.35, -0.31, 0.51, 0.75, 1.88]),
    ],
)
def test_log_gamma_matches_brute_force(beta: np.ndarray) -> None:
    expected = _gamma_brute(beta)
    got = np.exp(log_gamma(beta))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_log_gamma_endpoints() -> None:
    beta = np.array([-1.0, 0.5, 0.0, 0.7])
    log_g = log_gamma(beta)
    # γ_0 = 1 always.
    assert log_g[0] == pytest.approx(0.0)
    # γ_k = ∏ exp(-β_i) = exp(-Σ β_i)
    assert log_g[-1] == pytest.approx(-beta.sum(), rel=0, abs=1e-12)


def test_log_gamma_minus_one_consistency() -> None:
    """γ_r = γ_r^{(-i)} + ε_i · γ_{r-1}^{(-i)} for all i, r."""
    beta = np.array([-0.7, 0.1, 0.4, -0.2, 0.9])
    eps = np.exp(-beta)
    g = np.exp(log_gamma(beta))
    g_minus = np.exp(log_gamma_minus_one(beta))
    k = len(beta)
    for i in range(k):
        for r in range(1, k + 1):
            g_minus_r_minus_one = g_minus[i, r - 1] if r - 1 <= k - 1 else 0.0
            g_minus_r = g_minus[i, r] if r <= k - 1 else 0.0
            reconstruction = g_minus_r + eps[i] * g_minus_r_minus_one
            assert g[r] == pytest.approx(reconstruction, rel=1e-12, abs=1e-12)


def test_conditional_endorsement_prob_marginalizes_to_raw_score() -> None:
    """Σ_i P(X_i=1 | R=r) must equal r."""
    beta = np.array([-1.22, -0.85, -1.11, 0.35, -0.31, 0.51, 0.75, 1.88])
    pi = conditional_endorsement_prob(beta)
    k = len(beta)
    for r in range(k + 1):
        assert pi[r, :].sum() == pytest.approx(r, abs=1e-10)


def test_conditional_endorsement_prob_endpoints() -> None:
    beta = np.array([-0.5, 0.0, 0.5])
    pi = conditional_endorsement_prob(beta)
    np.testing.assert_array_equal(pi[0, :], np.zeros(3))
    np.testing.assert_array_equal(pi[3, :], np.ones(3))
