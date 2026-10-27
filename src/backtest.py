"""Daily out-of-sample backtest engines (paper Sec. 7, Algorithms 1 and 2).

Both engines share the same expanding-window loop structure, the same daily
decision timing, and the same transaction-cost-aware MVO layer
(``src/mvo.py:solve_mvo``):

1. At day t, form the conditional moments (mu_t, Sigma_t) from information
   available up to t-1 only (the feature matrix row x_t in
   ``src/features.py`` is itself built from returns up to t-1).
2. Solve the MVO problem for w_t using w_{t-1} as the transaction-cost
   anchor.
3. Realize the portfolio's day-t log return as w_t . r_t, where r_t is the
   day's *actual* asset return (only revealed after w_t has been chosen),
   so the weights never use information unavailable at decision time.
4. Record turnover TO_t and concentration N_eff,t (Sec. 7.1 diagnostics),
   then roll w_{t-1} <- w_t for the next day.

``run_wasserstein_hmm_backtest`` implements Algorithm 1 (Sec. 7.1):
periodic predictive model-order selection, a daily (warm-started) Gaussian
HMM refit on the expanding history, nearest-template mapping by 2-Wasserstein
distance, exponential-smoothing template updates, and mixture moments.

``run_knn_backtest`` implements Algorithm 2 (Sec. 7.2): K-nearest-neighbor
conditional moments on the expanding history of strictly prior days.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    HMM_COMPLEXITY_LAMBDA,
    HMM_DAILY_REFIT_ITER,
    HMM_K_MAX,
    HMM_K_MIN,
    HMM_MIN_TRAIN_OBS,
    HMM_ORDER_SELECT_ITER,
    HMM_ORDER_SELECTION_FREQ,
    HMM_VALIDATION_SLICE,
    KNN_LOOKBACK,
    KNN_N_NEIGHBORS,
    MVO_GAMMA,
    MVO_KNN_GAMMA,
    MVO_KNN_TAU,
    MVO_TAU,
    MVO_W_MAX,
    N_TEMPLATES,
    TEMPLATE_CALIBRATION_FRACTION,
)
from src.features import build_feature_matrix, log_returns
from src.knn_regime import knn_conditional_moments
from src.mvo import effective_n, solve_mvo, turnover
from src.wasserstein_hmm import (
    TemplateTracker,
    extract_return_submoments,
    filtered_state_probs_last,
    fit_gaussian_hmm,
    select_hmm_order,
)


def _assemble_result(
    assets: List[str],
    dates: List[pd.Timestamp],
    weights: List[np.ndarray],
    port_returns: List[float],
    turnovers: List[float],
    n_effs: List[float],
    extra: Optional[Dict[str, list]] = None,
) -> Dict[str, object]:
    weights_df = pd.DataFrame(weights, index=dates, columns=assets)
    result = {
        "weights": weights_df,
        "returns": pd.Series(port_returns, index=dates, name="portfolio_log_return"),
        "turnover": pd.Series(turnovers, index=dates, name="turnover"),
        "n_eff": pd.Series(n_effs, index=dates, name="n_eff"),
    }
    if extra:
        for key, values in extra.items():
            result[key] = pd.DataFrame(values, index=dates) if isinstance(values[0], (list, np.ndarray)) else pd.Series(values, index=dates, name=key)
    return result


def run_wasserstein_hmm_backtest(
    price_panel: pd.DataFrame,
    oos_start: str,
    k_min: int = HMM_K_MIN,
    k_max: int = HMM_K_MAX,
    order_selection_freq: int = HMM_ORDER_SELECTION_FREQ,
    validation_slice: int = HMM_VALIDATION_SLICE,
    complexity_lambda: float = HMM_COMPLEXITY_LAMBDA,
    min_train_obs: int = HMM_MIN_TRAIN_OBS,
    n_templates: int = N_TEMPLATES,
    calibration_fraction: float = TEMPLATE_CALIBRATION_FRACTION,
    gamma: float = MVO_GAMMA,
    tau: float = MVO_TAU,
    w_max: float = MVO_W_MAX,
    order_select_iter: int = HMM_ORDER_SELECT_ITER,
    daily_refit_iter: int = HMM_DAILY_REFIT_ITER,
) -> Dict[str, object]:
    """Run the full Wasserstein-HMM + MVO backtest (Algorithm 1, Sec. 7.1).

    Returns a dict with keys ``weights`` (DataFrame, date x asset),
    ``returns`` (Series of daily portfolio log returns), ``turnover``,
    ``n_eff``, ``selected_k`` (Series of the HMM order chosen each day),
    and ``template_probs`` (DataFrame, date x template index) of the
    aggregated template probabilities p_{t,g}.
    """
    assets = list(price_panel.columns)
    n_assets = len(assets)
    returns = log_returns(price_panel)
    features = build_feature_matrix(returns)

    oos_start_ts = pd.Timestamp(oos_start, tz=features.index.tz)
    pre_oos_idx = features.index[features.index < oos_start_ts]
    test_dates = features.index[features.index >= oos_start_ts]

    if len(pre_oos_idx) < min_train_obs:
        raise ValueError(
            f"Pre-OOS history ({len(pre_oos_idx)} rows) is shorter than "
            f"HMM_MIN_TRAIN_OBS={min_train_obs}; cannot start the expanding-window fit."
        )
    if len(test_dates) == 0:
        raise ValueError(f"No feature rows on or after oos_start={oos_start}.")

    calib_len = max(n_templates + 1, int(len(pre_oos_idx) * calibration_fraction))
    calib_len = min(calib_len, len(pre_oos_idx))
    X_calib = features.loc[pre_oos_idx[-calib_len:]].values
    tracker = TemplateTracker.initialize(X_calib, n_templates=n_templates)

    w_prev = np.full(n_assets, 1.0 / n_assets)
    current_model = None
    current_k: Optional[int] = None
    # Per-order warm-start cache (K -> last fitted GaussianHMM), mutated by
    # select_hmm_order: each candidate order is re-fit from its own previous
    # converged parameters on every re-selection day instead of a cold EM
    # restart (see wasserstein_hmm.select_hmm_order docstring).
    k_warm_models: Dict[int, object] = {}

    dates, weights, port_returns, turnovers, n_effs = [], [], [], [], []
    selected_k_hist, template_prob_rows = [], []

    for day_i, t in enumerate(test_dates):
        X_hist = features.loc[features.index <= t].values

        if current_model is None or day_i % order_selection_freq == 0:
            best_k, _ = select_hmm_order(
                X_hist,
                k_min=k_min,
                k_max=k_max,
                validation_len=validation_slice,
                complexity_lambda=complexity_lambda,
                n_iter=order_select_iter,
                warm_models=k_warm_models,
                warm_iter=daily_refit_iter,
            )
            current_k = best_k

        warm = current_model if (current_model is not None and current_model.n_components == current_k) else k_warm_models.get(current_k)
        n_iter_fit = daily_refit_iter if warm is not None else order_select_iter
        model = fit_gaussian_hmm(X_hist, current_k, n_iter=n_iter_fit, tol=1e-3, init_model=warm)
        current_model = model
        k_warm_models[current_k] = model

        p_state = filtered_state_probs_last(model, X_hist)
        mapping = tracker.map_components(model.means_, model.covars_)
        p_template = tracker.aggregate(p_state, mapping)
        tracker.update(model.means_, model.covars_, p_state, mapping)
        mu_vec, sigma_mat = tracker.mixture_moments(p_template)
        mu_r, sigma_r = extract_return_submoments(mu_vec, sigma_mat, n_assets)

        w_t = solve_mvo(mu_r, sigma_r, w_prev, gamma=gamma, tau=tau, w_max=w_max)

        r_t = returns.loc[t, assets].values
        port_ret = float(np.dot(w_t, r_t))
        to = turnover(w_t, w_prev)
        neff = effective_n(w_t)

        dates.append(t)
        weights.append(w_t)
        port_returns.append(port_ret)
        turnovers.append(to)
        n_effs.append(neff)
        selected_k_hist.append(current_k)
        template_prob_rows.append(p_template)

        w_prev = w_t

    result = _assemble_result(assets, dates, weights, port_returns, turnovers, n_effs)
    result["selected_k"] = pd.Series(selected_k_hist, index=dates, name="selected_k")
    result["template_probs"] = pd.DataFrame(
        template_prob_rows, index=dates, columns=[f"template_{g}" for g in range(n_templates)]
    )
    result["asset_returns"] = returns.loc[dates, assets]
    return result


def run_knn_backtest(
    price_panel: pd.DataFrame,
    oos_start: str,
    n_neighbors: int = KNN_N_NEIGHBORS,
    min_lookback: int = KNN_LOOKBACK,
    gamma: float = MVO_KNN_GAMMA,
    tau: float = MVO_KNN_TAU,
    w_max: float = MVO_W_MAX,
) -> Dict[str, object]:
    """Run the full KNN + MVO backtest (Algorithm 2, Sec. 7.2).

    Returns a dict with keys ``weights``, ``returns``, ``turnover``, and
    ``n_eff``, with the same layout as `run_wasserstein_hmm_backtest`.
    """
    assets = list(price_panel.columns)
    n_assets = len(assets)
    returns = log_returns(price_panel)
    features = build_feature_matrix(returns)

    oos_start_ts = pd.Timestamp(oos_start, tz=features.index.tz)
    pre_oos_idx = features.index[features.index < oos_start_ts]
    test_dates = features.index[features.index >= oos_start_ts]

    if len(pre_oos_idx) < min_lookback:
        raise ValueError(
            f"Pre-OOS history ({len(pre_oos_idx)} rows) is shorter than "
            f"KNN_LOOKBACK={min_lookback}; cannot start the expanding-window KNN fit."
        )
    if len(test_dates) == 0:
        raise ValueError(f"No feature rows on or after oos_start={oos_start}.")

    w_prev = np.full(n_assets, 1.0 / n_assets)
    dates, weights, port_returns, turnovers, n_effs = [], [], [], [], []

    for t in test_dates:
        train_idx = features.index[features.index < t]
        X_train = features.loc[train_idx].values
        returns_train = returns.loc[train_idx, assets].values
        x_t = features.loc[t].values

        mu_t, sigma_t = knn_conditional_moments(x_t, X_train, returns_train, n_neighbors)
        w_t = solve_mvo(mu_t, sigma_t, w_prev, gamma=gamma, tau=tau, w_max=w_max)

        r_t = returns.loc[t, assets].values
        port_ret = float(np.dot(w_t, r_t))
        to = turnover(w_t, w_prev)
        neff = effective_n(w_t)

        dates.append(t)
        weights.append(w_t)
        port_returns.append(port_ret)
        turnovers.append(to)
        n_effs.append(neff)

        w_prev = w_t

    result = _assemble_result(assets, dates, weights, port_returns, turnovers, n_effs)
    result["asset_returns"] = returns.loc[dates, assets]
    return result
