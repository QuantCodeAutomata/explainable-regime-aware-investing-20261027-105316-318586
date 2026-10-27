"""Performance metrics for regime-aware portfolios (paper Sec. 8, "Out-of-Sample
Performance").

Daily log returns are converted to simple returns (``exp(r) - 1``) before being
passed to `quantstats.stats` (Context7: QuantStats `sharpe`/`sortino`/
`max_drawdown`), because QuantStats compounds returns multiplicatively
(``(1 + r).cumprod()``) when building the equity curve used for drawdown.
Sharpe and Sortino are annualized with ``periods=252`` trading days, the
standard annualization factor for daily data and the one used by QuantStats
by default.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import quantstats as qs

TRADING_DAYS_PER_YEAR = 252


def log_to_simple(log_returns: pd.Series) -> pd.Series:
    """Convert daily log returns to simple returns for compounding."""
    return np.expm1(log_returns)


def performance_summary(log_returns: pd.Series) -> Dict[str, float]:
    """Compute Sharpe, Sortino, cumulative log return, and max drawdown.

    Parameters
    ----------
    log_returns : daily portfolio (or asset) log returns, strictly out-of-sample.
    """
    log_returns = log_returns.dropna()
    if len(log_returns) < 2:
        raise ValueError("Need at least 2 return observations to compute performance metrics.")

    simple_returns = log_to_simple(log_returns)

    sharpe = float(qs.stats.sharpe(simple_returns, rf=0.0, periods=TRADING_DAYS_PER_YEAR))
    sortino = float(qs.stats.sortino(simple_returns, rf=0.0, periods=TRADING_DAYS_PER_YEAR))
    max_dd = float(qs.stats.max_drawdown(simple_returns))
    cumulative_log_return = float(log_returns.sum())
    annualized_vol = float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "cumulative_log_return": cumulative_log_return,
        "annualized_vol": annualized_vol,
        "n_days": int(len(log_returns)),
    }


def turnover_stats(turnover: pd.Series, threshold: float = 0.01) -> Dict[str, float]:
    """Average daily turnover and the share of days exceeding ``threshold``
    (paper Sec. 9, diagnostics: :math:`TO_t = \\tfrac12\\|w_t-w_{t-1}\\|_1`)."""
    turnover = turnover.dropna()
    return {
        "avg_turnover": float(turnover.mean()),
        "pct_days_above_threshold": float((turnover > threshold).mean() * 100.0),
    }


def effective_n_stats(n_eff: pd.Series) -> Dict[str, float]:
    """Summary statistics of portfolio concentration :math:`N_{eff,t}`."""
    n_eff = n_eff.dropna()
    return {
        "avg_n_eff": float(n_eff.mean()),
        "min_n_eff": float(n_eff.min()),
        "max_n_eff": float(n_eff.max()),
    }
