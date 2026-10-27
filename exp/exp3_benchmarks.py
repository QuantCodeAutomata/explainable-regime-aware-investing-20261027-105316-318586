"""Experiment 3: Passive Benchmarks (SPX Buy & Hold and Equal-Weight).

Reproduces the "Benchmarking Against Passive Allocations" section of
"Explainable Regime-Aware Investing" (Boukardagha, Columbia University):
out-of-sample risk-adjusted performance of (i) a buy-and-hold position in
the S&P 500 proxy and (ii) a static equal-weight (20% each) portfolio
across the five-asset universe, over the same out-of-sample window used
by the active regime-aware strategies.

Run as:

    python -m exp.exp3_benchmarks          # executed configuration
    python -m exp.exp3_benchmarks --full    # paper's full sample period
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_price_panel
from src.config import (
    ASSET_LABELS,
    EQUAL_WEIGHT,
    OOS_START,
    SAMPLE_END_PAPER,
    SAMPLE_END_RUN,
    SAMPLE_START,
)
from src.features import log_returns
from src.metrics import performance_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Reported for the passive benchmarks ("Risk-Adjusted Performance"), used
# only for the comparison printed/saved alongside the computed metrics.
PAPER_EXPECTED = {
    "spx_buy_hold": {"sharpe": 1.18, "sortino": 1.50, "max_drawdown": -0.1462},
    "equal_weight": {"sharpe": 1.59, "sortino": 2.27, "max_drawdown": -0.0987},
}


def main(full: bool = False) -> dict:
    end_date = SAMPLE_END_PAPER if full else SAMPLE_END_RUN
    logger.info("Loading price panel %s..%s for assets %s", SAMPLE_START, end_date, list(ASSET_LABELS.values()))
    panel = load_price_panel(SAMPLE_START, end_date)
    rets = log_returns(panel)

    oos_mask = rets.index >= pd.Timestamp(OOS_START, tz=rets.index.tz)
    oos_rets = rets.loc[oos_mask]
    logger.info("OOS window: %s..%s, n_days=%d", oos_rets.index.min(), oos_rets.index.max(), len(oos_rets))

    # SPX Buy & Hold: raw daily log returns of the S&P 500 proxy.
    spx_returns = oos_rets["SPX"]

    # Equal-Weight portfolio: a static 20%-per-asset allocation. Daily
    # portfolio log return is the weight-averaged sum of asset daily log
    # returns, the same linear approximation used throughout this
    # repository for the active strategies (see src/backtest.py), applied
    # here with fixed weights instead of regime-conditional ones.
    weights = np.full(oos_rets.shape[1], EQUAL_WEIGHT)
    assert np.isclose(weights.sum(), 1.0)
    ew_returns = pd.Series(oos_rets.values @ weights, index=oos_rets.index, name="equal_weight")

    spx_perf = performance_summary(spx_returns)
    ew_perf = performance_summary(ew_returns)

    metrics = {
        "oos_start": OOS_START,
        "sample_start_used": str(panel.index.min().date()),
        "sample_end_used": str(panel.index.max().date()),
        "n_oos_days": int(len(oos_rets)),
        "spx_buy_hold": spx_perf,
        "equal_weight": ew_perf,
        "paper_expected": PAPER_EXPECTED,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp3_benchmarks_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    returns_df = pd.DataFrame({"spx_buy_hold": spx_returns, "equal_weight": ew_returns})
    returns_df.to_csv(RESULTS_DIR / "exp3_benchmarks_returns.csv")

    _plot_cumulative_returns(returns_df, RESULTS_DIR / "exp3_benchmarks_cumulative_returns.png")

    logger.info("exp3 metrics: %s", {k: v for k, v in metrics.items() if k != "paper_expected"})
    return metrics


def _plot_cumulative_returns(returns_df: pd.DataFrame, path: Path) -> None:
    cum = returns_df.cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in cum.columns:
        ax.plot(cum.index, cum[col], label=col)
    ax.set_title("Passive Benchmarks: Out-of-Sample Cumulative Log Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative log return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the paper's full sample period (SAMPLE_END_PAPER).")
    args = parser.parse_args()
    main(full=args.full)
