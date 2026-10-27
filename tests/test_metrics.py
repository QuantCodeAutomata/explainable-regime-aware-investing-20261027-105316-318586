"""Unit tests for src/metrics.py (paper Sec. 8, "Out-of-Sample Performance")."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import effective_n_stats, log_to_simple, performance_summary, turnover_stats


def test_log_to_simple_matches_definition() -> None:
    log_r = pd.Series([0.0, np.log(1.1), np.log(0.9)])
    simple = log_to_simple(log_r)
    assert np.allclose(simple.values, [0.0, 0.1, -0.1], atol=1e-10)


def test_performance_summary_positive_drift_gives_positive_sharpe() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=500)
    returns = pd.Series(rng.normal(loc=0.0008, scale=0.01, size=500), index=dates)
    perf = performance_summary(returns)
    assert perf["sharpe"] > 0
    assert perf["sortino"] > 0
    assert perf["max_drawdown"] <= 0
    assert perf["n_days"] == 500


def test_performance_summary_negative_drift_gives_negative_sharpe() -> None:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2023-01-01", periods=500)
    returns = pd.Series(rng.normal(loc=-0.0008, scale=0.01, size=500), index=dates)
    perf = performance_summary(returns)
    assert perf["sharpe"] < 0


def test_performance_summary_requires_min_observations() -> None:
    with pytest.raises(ValueError):
        performance_summary(pd.Series([0.01]))


def test_max_drawdown_is_zero_for_monotonically_increasing_returns() -> None:
    dates = pd.bdate_range("2023-01-01", periods=50)
    returns = pd.Series(np.full(50, 0.001), index=dates)
    perf = performance_summary(returns)
    assert perf["max_drawdown"] == pytest.approx(0.0, abs=1e-6)


def test_turnover_stats_threshold_share() -> None:
    turnover = pd.Series([0.0, 0.02, 0.005, 0.5, np.nan])
    stats = turnover_stats(turnover, threshold=0.01)
    assert stats["pct_days_above_threshold"] == pytest.approx(50.0)  # 2 of 4 non-NaN days


def test_effective_n_stats_matches_direct_computation() -> None:
    n_eff = pd.Series([1.0, 2.0, 3.0, np.nan])
    stats = effective_n_stats(n_eff)
    assert stats["avg_n_eff"] == pytest.approx(2.0)
    assert stats["min_n_eff"] == pytest.approx(1.0)
    assert stats["max_n_eff"] == pytest.approx(3.0)
