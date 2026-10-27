"""Feature representation (paper Sec. 4, "Feature Representation").

For each asset i and day t the paper defines a feature vector
``x_t = [r_t; sigma_t; m_t]`` where

* ``r_t``     is the daily log return,
* ``sigma_t`` is a rolling realized volatility of the log returns over a
  ``VOL_WINDOW``-day window, and
* ``m_t``     is a rolling mean of the log returns over a ``MEAN_WINDOW``-day
  window,

and every quantity used at day t is computed strictly from information
available up to t-1 ("strict causality", Sec. 4). This module builds the
full, asset-stacked daily feature matrix used by both the Wasserstein-HMM
(src/wasserstein_hmm.py) and the KNN (src/knn_regime.py) regime models.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.config import MEAN_WINDOW, VOL_WINDOW


def log_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ``r_t = ln(P_t) - ln(P_{t-1})`` for every asset column."""
    if (price_panel <= 0).any().any():
        raise ValueError("Non-positive prices found; cannot take log returns.")
    returns = np.log(price_panel).diff()
    return returns.iloc[1:]


def build_feature_matrix(
    returns: pd.DataFrame,
    vol_window: int = VOL_WINDOW,
    mean_window: int = MEAN_WINDOW,
) -> pd.DataFrame:
    """Build the strictly causal feature matrix ``X`` with columns
    ``[r_<asset>, sigma_<asset>, m_<asset>]`` for every asset, stacked in the
    asset order of ``returns.columns`` (Sec. 4).

    Causality (Sec. 4): the feature row at day t must only use returns up to
    and including day t-1. We therefore compute the rolling statistics on the
    *lagged* return series (``returns.shift(1)``) so that ``sigma_t`` and
    ``m_t`` are the window statistics ending at t-1, and the realized return
    component ``r_t`` of the feature vector is itself the lagged return
    r_{t-1} (the most recently observed realization at the time the feature
    is used to predict/allocate for day t). This matches Algorithm 1/2's use
    of ``x_t`` to form the expanding history ``H_t = {x_s : s <= t}`` that
    never contains information from day t itself.
    """
    assets: List[str] = list(returns.columns)
    lagged_returns = returns.shift(1)  # r_{t-1}, available at t (strict causality)

    feature_blocks = []
    for asset in assets:
        r = lagged_returns[asset].rename(f"r_{asset}")
        sigma = (
            lagged_returns[asset].rolling(vol_window, min_periods=vol_window).std(ddof=1)
        ).rename(f"sigma_{asset}")
        m = (
            lagged_returns[asset].rolling(mean_window, min_periods=mean_window).mean()
        ).rename(f"m_{asset}")
        feature_blocks.extend([r, sigma, m])

    features = pd.concat(feature_blocks, axis=1)
    features = features.dropna(how="any")
    if features.empty:
        raise ValueError("Feature matrix is empty after applying rolling windows.")
    return features


def feature_columns(assets: List[str]) -> List[str]:
    """Column names in the fixed ``[r, sigma, m]``-per-asset order used throughout."""
    cols = []
    for asset in assets:
        cols.extend([f"r_{asset}", f"sigma_{asset}", f"m_{asset}"])
    return cols
