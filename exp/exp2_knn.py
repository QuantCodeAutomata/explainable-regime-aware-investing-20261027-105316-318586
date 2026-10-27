"""Experiment 2: Non-Parametric KNN Regime Inference + MVO.

Reproduces Sec. 7.2 ("Out-of-Sample Performance") of "Explainable
Regime-Aware Investing" (Boukardagha, Columbia University): local
conditional moments estimated by K-nearest-neighbors on an expanding
history of causal feature vectors (Sec. 6, Algorithm 2), with Ledoit-Wolf
covariance shrinkage on the neighbor return sample, combined with the same
transaction-cost-aware mean-variance optimizer used by the parametric
strategy (Sec. 5.6), serving as a non-parametric benchmark.

Run as:

    python -m exp.exp2_knn            # executed configuration
    python -m exp.exp2_knn --full      # paper's full sample period
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.data_loader import load_price_panel
from src.backtest import run_knn_backtest
from src.config import (
    ASSET_LABELS,
    KNN_LOOKBACK,
    KNN_N_NEIGHBORS,
    OOS_START,
    SAMPLE_END_PAPER,
    SAMPLE_END_RUN,
    SAMPLE_START,
)
from src.metrics import effective_n_stats, performance_summary, turnover_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Reported in Sec. 9 ("Out-of-Sample Performance") for the KNN benchmark,
# used only for the comparison printed/saved alongside computed metrics.
PAPER_EXPECTED = {
    "sharpe": 1.81,
    "max_drawdown": -0.1252,
    "avg_turnover": 0.5665,
    "pct_days_above_threshold": 94.13,
}


def main(full: bool = False) -> dict:
    end_date = SAMPLE_END_PAPER if full else SAMPLE_END_RUN
    logger.info("Loading price panel %s..%s for assets %s", SAMPLE_START, end_date, list(ASSET_LABELS.values()))
    panel = load_price_panel(SAMPLE_START, end_date)
    logger.info(
        "Loaded panel: rows=%d, range=%s..%s, assets=%s, K=%d, lookback=%d",
        len(panel), panel.index.min(), panel.index.max(), list(panel.columns),
        KNN_N_NEIGHBORS, KNN_LOOKBACK,
    )

    t0 = time.time()
    result = run_knn_backtest(panel, OOS_START)
    elapsed = time.time() - t0
    logger.info("KNN backtest completed in %.1f s over %d OOS days", elapsed, len(result["returns"]))

    perf = performance_summary(result["returns"])
    to_stats = turnover_stats(result["turnover"])
    neff_stats = effective_n_stats(result["n_eff"])

    metrics = {
        "strategy": "knn_mvo",
        "oos_start": OOS_START,
        "sample_start_used": str(panel.index.min().date()),
        "sample_end_used": str(panel.index.max().date()),
        "n_oos_days": int(len(result["returns"])),
        "runtime_seconds": elapsed,
        **perf,
        **to_stats,
        **neff_stats,
        "paper_expected": PAPER_EXPECTED,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp2_knn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    result["weights"].to_csv(RESULTS_DIR / "exp2_knn_weights.csv")
    result["turnover"].to_csv(RESULTS_DIR / "exp2_knn_turnover.csv")

    _plot_cumulative_return(result["returns"], RESULTS_DIR / "exp2_knn_cumulative_returns.png")
    _plot_turnover(result["turnover"], RESULTS_DIR / "exp2_knn_turnover.png")

    logger.info("exp2 metrics: %s", {k: v for k, v in metrics.items() if k != "paper_expected"})
    return metrics


def _plot_cumulative_return(log_returns: pd.Series, path: Path) -> None:
    cum = log_returns.cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cum.index, cum.values, label="KNN + MVO", color="tab:orange")
    ax.set_title("Non-Parametric KNN Strategy: Out-of-Sample Cumulative Log Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative log return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_turnover(turnover: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(turnover.index, turnover.values, linewidth=0.7, color="tab:red")
    ax.axhline(0.01, color="black", linestyle="--", linewidth=1, label="1% threshold")
    ax.set_title("Non-Parametric KNN Strategy: Daily Turnover $TO_t$")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the paper's full sample period (SAMPLE_END_PAPER).")
    args = parser.parse_args()
    main(full=args.full)
