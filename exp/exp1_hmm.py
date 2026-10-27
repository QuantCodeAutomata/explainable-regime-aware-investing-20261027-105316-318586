"""Experiment 1: Parametric Regime Investing with Wasserstein HMM + MVO.

Reproduces Sec. 7.1 ("Out-of-Sample Performance") and Sec. 9 ("Economic
Interpretation of Market Regimes") of "Explainable Regime-Aware Investing"
(Boukardagha, Columbia University): a causal, expanding-window Gaussian HMM
with periodic predictive model-order selection (Sec. 5.2), persistent
regime templates tracked by the closed-form 2-Wasserstein distance
(Sec. 5.4), and a transaction-cost-aware mean-variance optimizer
(Sec. 5.6) run day by day over the out-of-sample window.

Run as:

    python -m exp.exp1_hmm            # executed configuration
    python -m exp.exp1_hmm --full      # paper's full sample period
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
import numpy as np
import pandas as pd

from data.data_loader import load_price_panel
from src.backtest import run_wasserstein_hmm_backtest
from src.config import (
    ASSET_LABELS,
    OOS_START,
    SAMPLE_END_PAPER,
    SAMPLE_END_RUN,
    SAMPLE_START,
)
from src.metrics import effective_n_stats, performance_summary, turnover_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Reported in Sec. 8, Table 1 (paper's headline figures for the parametric
# Wasserstein-HMM strategy), used only for the comparison printed/saved
# alongside the computed metrics -- never fed back into the model.
PAPER_EXPECTED = {
    "sharpe": 2.18,
    "sortino": 2.82,
    "max_drawdown": -0.0543,
    "avg_turnover": 0.0079,
}


def main(full: bool = False) -> dict:
    end_date = SAMPLE_END_PAPER if full else SAMPLE_END_RUN
    logger.info("Loading price panel %s..%s for assets %s", SAMPLE_START, end_date, list(ASSET_LABELS.values()))
    panel = load_price_panel(SAMPLE_START, end_date)
    logger.info(
        "Loaded panel: rows=%d, range=%s..%s, assets=%s",
        len(panel), panel.index.min(), panel.index.max(), list(panel.columns),
    )

    t0 = time.time()
    result = run_wasserstein_hmm_backtest(panel, OOS_START)
    elapsed = time.time() - t0
    logger.info("Wasserstein-HMM backtest completed in %.1f s over %d OOS days", elapsed, len(result["returns"]))

    perf = performance_summary(result["returns"])
    to_stats = turnover_stats(result["turnover"])
    neff_stats = effective_n_stats(result["n_eff"])

    metrics = {
        "strategy": "wasserstein_hmm_mvo",
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
    with open(RESULTS_DIR / "exp1_hmm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    result["weights"].to_csv(RESULTS_DIR / "exp1_hmm_weights.csv")
    result["template_probs"].to_csv(RESULTS_DIR / "exp1_hmm_template_probs.csv")
    result["selected_k"].to_csv(RESULTS_DIR / "exp1_hmm_selected_k.csv")
    result["turnover"].to_csv(RESULTS_DIR / "exp1_hmm_turnover.csv")

    _plot_cumulative_return(result["returns"], RESULTS_DIR / "exp1_hmm_cumulative_returns.png")
    _plot_weights(result["weights"], RESULTS_DIR / "exp1_hmm_weights.png")
    _plot_template_probs(result["template_probs"], RESULTS_DIR / "exp1_hmm_template_probs.png")

    regime_table = _regime_conditional_asset_performance(result)
    regime_table.to_csv(RESULTS_DIR / "exp1_hmm_regime_conditional_returns.csv")

    logger.info("exp1 metrics: %s", {k: v for k, v in metrics.items() if k != "paper_expected"})
    return metrics


def _plot_cumulative_return(log_returns: pd.Series, path: Path) -> None:
    cum = log_returns.cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cum.index, cum.values, label="Wasserstein-HMM + MVO")
    ax.set_title("Wasserstein-HMM Regime Strategy: Out-of-Sample Cumulative Log Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative log return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_weights(weights: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(weights.index, [weights[c] for c in weights.columns], labels=list(weights.columns))
    ax.set_title("Wasserstein-HMM Strategy: Daily Portfolio Weights")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_template_probs(template_probs: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(
        template_probs.index,
        [template_probs[c] for c in template_probs.columns],
        labels=list(template_probs.columns),
    )
    ax.set_title("Aggregated Template (Regime) Probabilities p_{t,g}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _regime_conditional_asset_performance(result: dict) -> pd.DataFrame:
    """Economic interpretation of market regimes (Sec. 9): average asset and
    portfolio daily log return conditional on the day's dominant regime
    template (argmax_g p_{t,g}), plus the number of days each template
    dominates and the strategy's own daily log return in that regime.
    """
    template_probs = result["template_probs"]
    dominant = template_probs.idxmax(axis=1)
    asset_returns = result["asset_returns"]
    port_returns = result["returns"]

    rows = []
    for template in sorted(template_probs.columns, key=lambda c: int(c.split("_")[1])):
        mask = dominant == template
        n_days = int(mask.sum())
        row = {"template": template, "n_days": n_days, "share_of_days": n_days / len(dominant) if len(dominant) else np.nan}
        if n_days > 0:
            for asset in asset_returns.columns:
                row[f"mean_ret_{asset}"] = float(asset_returns.loc[mask, asset].mean())
            row["mean_portfolio_ret"] = float(port_returns.loc[mask].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("template")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the paper's full sample period (SAMPLE_END_PAPER).")
    args = parser.parse_args()
    main(full=args.full)
