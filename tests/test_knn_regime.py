"""Unit tests for src/knn_regime.py (paper Sec. 6, Algorithm 2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.knn_regime import knn_conditional_moments


def test_knn_conditional_moments_shapes() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 6))
    returns_train = rng.normal(scale=0.01, size=(200, 2))
    x_t = rng.normal(size=6)

    mu, sigma = knn_conditional_moments(x_t, X_train, returns_train, n_neighbors=20)
    assert mu.shape == (2,)
    assert sigma.shape == (2, 2)
    # Covariance must be symmetric positive semi-definite.
    assert np.allclose(sigma, sigma.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.all(eigvals >= -1e-8)


def test_knn_mean_matches_plain_average_of_neighbor_returns() -> None:
    """mu_t = (1/K) sum_{s in N_t} r_s (Algorithm 2, step 9)."""
    X_train = np.arange(10).reshape(-1, 1).astype(float)
    returns_train = np.arange(10).reshape(-1, 1).astype(float) * 0.01
    x_t = np.array([4.4])  # nearest 3 points by construction: indices 4, 5, 3

    mu, _ = knn_conditional_moments(x_t, X_train, returns_train, n_neighbors=3)
    expected = returns_train[[3, 4, 5]].mean()
    assert mu[0] == pytest.approx(expected, rel=1e-8)


def test_knn_requires_at_least_two_neighbors() -> None:
    X_train = np.zeros((5, 2))
    returns_train = np.zeros((5, 2))
    with pytest.raises(ValueError):
        knn_conditional_moments(np.zeros(2), X_train, returns_train, n_neighbors=1)


def test_knn_rejects_mismatched_row_counts() -> None:
    X_train = np.zeros((5, 2))
    returns_train = np.zeros((4, 2))
    with pytest.raises(ValueError):
        knn_conditional_moments(np.zeros(2), X_train, returns_train, n_neighbors=3)
