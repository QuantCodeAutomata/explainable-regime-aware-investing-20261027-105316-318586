"""Unit tests for src/features.py (paper Sec. 4, "Feature Representation")."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MEAN_WINDOW, VOL_WINDOW
from src.features import build_feature_matrix, feature_columns, log_returns


def test_log_returns_matches_definition(synthetic_price_panel: pd.DataFrame) -> None:
    rets = log_returns(synthetic_price_panel)
    expected = np.log(synthetic_price_panel).diff().iloc[1:]
    pd.testing.assert_frame_equal(rets, expected)
    assert len(rets) == len(synthetic_price_panel) - 1


def test_log_returns_rejects_non_positive_prices() -> None:
    bad_panel = pd.DataFrame({"A": [1.0, -2.0, 3.0]})
    with pytest.raises(ValueError):
        log_returns(bad_panel)


def test_feature_matrix_is_strictly_causal(synthetic_price_panel: pd.DataFrame) -> None:
    """x_t must only use information up to t-1 (Sec. 4): the feature row at
    day t, recomputed with all data from t onward blanked out, must be
    identical to the feature row computed from the full series."""
    rets = log_returns(synthetic_price_panel)
    features_full = build_feature_matrix(rets)

    t = features_full.index[200]
    truncated_rets = rets.loc[rets.index <= rets.index[rets.index.get_loc(t) - 1]]
    features_truncated = build_feature_matrix(pd.concat([truncated_rets, rets.loc[[t]]]))

    pd.testing.assert_series_equal(
        features_full.loc[t], features_truncated.loc[t], check_names=False
    )


def test_feature_matrix_shape_and_columns(synthetic_price_panel: pd.DataFrame) -> None:
    rets = log_returns(synthetic_price_panel)
    features = build_feature_matrix(rets)
    expected_cols = feature_columns(list(rets.columns))
    assert list(features.columns) == expected_cols
    assert features.shape[1] == 3 * rets.shape[1]
    # The first valid row requires VOL_WINDOW (the larger of the two
    # rolling windows) prior lagged-return observations plus the one-day
    # shift (Sec. 4).
    min_window = max(VOL_WINDOW, MEAN_WINDOW)
    assert len(features) <= len(rets) - min_window
    assert not features.isna().any().any()


def test_feature_matrix_requires_enough_history() -> None:
    short_rets = pd.DataFrame(
        np.random.default_rng(0).normal(size=(10, 2)) * 0.01,
        columns=["A", "B"],
        index=pd.bdate_range("2021-01-01", periods=10),
    )
    with pytest.raises(ValueError):
        build_feature_matrix(short_rets)
