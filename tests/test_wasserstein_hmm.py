"""Unit tests for src/wasserstein_hmm.py (paper Sec. 5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.wasserstein_hmm import (
    TemplateTracker,
    complexity_penalty,
    extract_return_submoments,
    filtered_state_probs_last,
    fit_gaussian_hmm,
    forward_log_predictive,
    select_hmm_order,
    wasserstein2_gaussian,
)


def test_wasserstein_distance_is_zero_for_identical_gaussians() -> None:
    rng = np.random.default_rng(0)
    mu = rng.normal(size=3)
    A = rng.normal(size=(3, 3))
    sigma = A @ A.T + np.eye(3) * 0.1
    d = wasserstein2_gaussian(mu, sigma, mu, sigma)
    assert d == pytest.approx(0.0, abs=1e-6)


def test_wasserstein_distance_is_symmetric_and_nonnegative() -> None:
    rng = np.random.default_rng(1)
    mu1, mu2 = rng.normal(size=3), rng.normal(size=3)
    A1, A2 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    s1 = A1 @ A1.T + np.eye(3) * 0.1
    s2 = A2 @ A2.T + np.eye(3) * 0.1
    d12 = wasserstein2_gaussian(mu1, s1, mu2, s2)
    d21 = wasserstein2_gaussian(mu2, s2, mu1, s1)
    assert d12 >= 0.0
    assert d12 == pytest.approx(d21, rel=1e-6)


def test_wasserstein_distance_matches_univariate_closed_form() -> None:
    """For scalar Gaussians, W2^2 = (mu1-mu2)^2 + (sigma1-sigma2)^2."""
    mu1, mu2 = np.array([0.0]), np.array([2.0])
    s1, s2 = np.array([[1.0]]), np.array([[4.0]])
    d = wasserstein2_gaussian(mu1, s1, mu2, s2)
    expected = np.sqrt((0.0 - 2.0) ** 2 + (1.0 - 2.0) ** 2)
    assert d == pytest.approx(expected, rel=1e-4)


def test_complexity_penalty_increases_with_states_and_features() -> None:
    assert complexity_penalty(2, 3) < complexity_penalty(3, 3)
    assert complexity_penalty(2, 3) < complexity_penalty(2, 6)


def _synthetic_regime_features(n_per_regime: int = 150, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    regime_a = rng.normal(loc=[0.5, 0.1, 0.0], scale=0.05, size=(n_per_regime, 3))
    regime_b = rng.normal(loc=[-0.5, 0.3, -0.2], scale=0.05, size=(n_per_regime, 3))
    return np.vstack([regime_a, regime_b])


def test_fit_gaussian_hmm_recovers_two_well_separated_clusters() -> None:
    X = _synthetic_regime_features()
    model = fit_gaussian_hmm(X, n_states=2, n_iter=50, tol=1e-4)
    means_sorted = np.sort(model.means_[:, 0])
    assert means_sorted[0] < -0.1
    assert means_sorted[1] > 0.1


def test_forward_log_predictive_sums_to_total_loglik() -> None:
    X = _synthetic_regime_features(n_per_regime=50)
    model = fit_gaussian_hmm(X, n_states=2, n_iter=30, tol=1e-4)
    log_c = forward_log_predictive(model, X)
    total_from_recursion = log_c.sum()
    total_from_hmmlearn = model.score(X)
    assert total_from_recursion == pytest.approx(total_from_hmmlearn, rel=1e-3)


def test_filtered_state_probs_sum_to_one() -> None:
    X = _synthetic_regime_features(n_per_regime=50)
    model = fit_gaussian_hmm(X, n_states=2, n_iter=30, tol=1e-4)
    p = filtered_state_probs_last(model, X)
    assert p.shape == (2,)
    assert np.all(p >= 0.0)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


def test_select_hmm_order_prefers_two_states_for_two_regime_data() -> None:
    X = _synthetic_regime_features(n_per_regime=120)
    best_k, scores = select_hmm_order(
        X, k_min=1, k_max=3, validation_len=20, complexity_lambda=0.01, n_iter=30
    )
    assert best_k in scores
    assert best_k >= 2  # a single Gaussian cannot explain two separated clusters


def test_template_tracker_maps_and_aggregates_consistently() -> None:
    X_calib = _synthetic_regime_features(n_per_regime=100)
    tracker = TemplateTracker.initialize(X_calib, n_templates=2, random_state=0)

    means = np.array([[0.5, 0.1, 0.0], [-0.5, 0.3, -0.2]])
    covars = np.stack([np.eye(3) * 0.05, np.eye(3) * 0.05])
    p_state = np.array([0.7, 0.3])

    mapping = tracker.map_components(means, covars)
    assert mapping.shape == (2,)
    p_template = tracker.aggregate(p_state, mapping)
    assert p_template.sum() == pytest.approx(p_state.sum(), abs=1e-8)

    tracker.update(means, covars, p_state, mapping)
    mu_t, sigma_t = tracker.mixture_moments(p_template)
    assert mu_t.shape == (3,)
    assert sigma_t.shape == (3, 3)


def test_extract_return_submoments_picks_first_coordinate_per_asset() -> None:
    n_assets = 2
    mean_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cov_mat = np.eye(6) * np.arange(1, 7)
    mu_r, sigma_r = extract_return_submoments(mean_vec, cov_mat, n_assets)
    assert np.allclose(mu_r, [1.0, 4.0])
    assert sigma_r.shape == (2, 2)
    assert np.allclose(np.diag(sigma_r), [1.0, 4.0])
