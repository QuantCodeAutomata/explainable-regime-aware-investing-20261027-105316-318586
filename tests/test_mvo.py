"""Unit tests for src/mvo.py (paper Sec. 5.6, transaction-cost-aware MVO)."""

from __future__ import annotations

import numpy as np
import pytest

from src.mvo import effective_n, solve_mvo, turnover


def test_solve_mvo_satisfies_budget_and_bounds() -> None:
    mu = np.array([0.001, 0.0005, 0.0003, -0.0002, 0.0001])
    sigma = np.diag([0.02, 0.01, 0.015, 0.03, 0.005]) ** 2
    w_prev = np.full(5, 0.2)
    w = solve_mvo(mu, sigma, w_prev, gamma=5.0, tau=0.05, w_max=0.6)

    assert w.shape == (5,)
    assert np.all(w >= -1e-8)
    assert np.all(w <= 0.6 + 1e-6)
    assert w.sum() == pytest.approx(1.0, abs=1e-5)


def test_solve_mvo_defaults_to_equal_weight_anchor_when_no_w_prev() -> None:
    mu = np.zeros(4)
    sigma = np.eye(4) * 0.01
    w = solve_mvo(mu, sigma, None, gamma=1.0, tau=0.0, w_max=1.0)
    # With zero expected returns and no transaction cost, minimizing
    # variance with equal covariances returns the equal-weight portfolio.
    assert np.allclose(w, np.full(4, 0.25), atol=1e-3)


def test_solve_mvo_higher_tau_reduces_turnover() -> None:
    mu = np.array([0.01, -0.01, 0.0, 0.0, 0.0])
    sigma = np.eye(5) * 0.01
    w_prev = np.full(5, 0.2)

    w_low_tau = solve_mvo(mu, sigma, w_prev, gamma=1.0, tau=0.001, w_max=1.0)
    w_high_tau = solve_mvo(mu, sigma, w_prev, gamma=1.0, tau=5.0, w_max=1.0)

    to_low = turnover(w_low_tau, w_prev)
    to_high = turnover(w_high_tau, w_prev)
    assert to_high <= to_low + 1e-6


def test_solve_mvo_rejects_mismatched_shapes() -> None:
    mu = np.zeros(3)
    sigma = np.eye(4)
    with pytest.raises(ValueError):
        solve_mvo(mu, sigma, None)


def test_turnover_formula() -> None:
    w_t = np.array([0.5, 0.5])
    w_prev = np.array([0.2, 0.8])
    assert turnover(w_t, w_prev) == pytest.approx(0.5 * (0.3 + 0.3))


def test_effective_n_bounds() -> None:
    w_equal = np.full(5, 0.2)
    w_concentrated = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert effective_n(w_equal) == pytest.approx(5.0)
    assert effective_n(w_concentrated) == pytest.approx(1.0)
    assert effective_n(w_equal) > effective_n(w_concentrated)
