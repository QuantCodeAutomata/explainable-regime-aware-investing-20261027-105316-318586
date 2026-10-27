"""Integration tests for src/backtest.py (paper Sec. 7, Algorithms 1 and 2).

These tests run the shared expanding-window backtest engines on a small
synthetic price panel (not used to produce the paper's headline results)
to check structural properties: output shapes/layout, budget and bound
constraints on the daily weights, and absence of look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_knn_backtest, run_wasserstein_hmm_backtest
from src.mvo import turnover


def _short_oos_start(panel: pd.DataFrame, n_oos_days: int) -> str:
    return str(panel.index[-n_oos_days].date())


def test_knn_backtest_weights_satisfy_mvo_constraints(synthetic_price_panel: pd.DataFrame) -> None:
    oos_start = _short_oos_start(synthetic_price_panel, 30)
    result = run_knn_backtest(
        synthetic_price_panel, oos_start, n_neighbors=10, min_lookback=50, w_max=0.6
    )
    weights = result["weights"]
    assert set(weights.columns) == set(synthetic_price_panel.columns)
    assert np.allclose(weights.sum(axis=1).values, 1.0, atol=1e-4)
    assert (weights.values >= -1e-6).all()
    assert (weights.values <= 0.6 + 1e-4).all()
    assert len(result["returns"]) == len(weights)
    assert len(result["turnover"]) == len(weights)
    assert len(result["n_eff"]) == len(weights)


def test_knn_backtest_realized_return_uses_same_day_asset_return(
    synthetic_price_panel: pd.DataFrame,
) -> None:
    """Portfolio return at day t must equal w_t . r_t exactly (no shift)."""
    oos_start = _short_oos_start(synthetic_price_panel, 25)
    result = run_knn_backtest(
        synthetic_price_panel, oos_start, n_neighbors=10, min_lookback=50, w_max=0.6
    )
    recomputed = (result["weights"].values * result["asset_returns"].values).sum(axis=1)
    assert np.allclose(recomputed, result["returns"].values, atol=1e-8)


def test_knn_backtest_turnover_matches_weight_changes(synthetic_price_panel: pd.DataFrame) -> None:
    oos_start = _short_oos_start(synthetic_price_panel, 25)
    result = run_knn_backtest(
        synthetic_price_panel, oos_start, n_neighbors=10, min_lookback=50, w_max=0.6
    )
    w = result["weights"].values
    to = result["turnover"].values
    # First day's anchor is the equal-weight portfolio (Algorithm 1, line 4).
    n_assets = w.shape[1]
    w_prev = np.full(n_assets, 1.0 / n_assets)
    for i in range(len(w)):
        expected_to = turnover(w[i], w_prev)
        assert to[i] == pytest.approx(expected_to, abs=1e-6)
        w_prev = w[i]


def test_knn_backtest_rejects_insufficient_history(synthetic_price_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        run_knn_backtest(synthetic_price_panel, "2020-01-10", min_lookback=5000)


def test_wasserstein_hmm_backtest_weights_satisfy_mvo_constraints(
    synthetic_price_panel: pd.DataFrame,
) -> None:
    oos_start = _short_oos_start(synthetic_price_panel, 15)
    result = run_wasserstein_hmm_backtest(
        synthetic_price_panel,
        oos_start,
        k_min=1,
        k_max=2,
        order_selection_freq=5,
        validation_slice=15,
        min_train_obs=200,
        n_templates=2,
        order_select_iter=10,
        daily_refit_iter=3,
        w_max=0.6,
    )
    weights = result["weights"]
    assert np.allclose(weights.sum(axis=1).values, 1.0, atol=1e-4)
    assert (weights.values >= -1e-6).all()
    assert (weights.values <= 0.6 + 1e-4).all()
    assert "template_probs" in result and "selected_k" in result
    assert result["template_probs"].shape[1] == 2
    # Template probabilities aggregate to 1 each day (Sec. 5.4).
    assert np.allclose(result["template_probs"].sum(axis=1).values, 1.0, atol=1e-4)


def test_wasserstein_hmm_backtest_rejects_insufficient_history(
    synthetic_price_panel: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        run_wasserstein_hmm_backtest(synthetic_price_panel, "2020-01-10", min_train_obs=5000)
