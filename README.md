# Explainable Regime-Aware Investing — Replication

Replication of selected experiments from **"Explainable Regime-Aware Investing"**
(Amine Boukardagha, Columbia University).

This repository reproduces three experiments from the paper:

1. **Parametric regime investing with a Wasserstein HMM** (Sec. 5, 7.1, 9): a
   strictly causal, expanding-window Gaussian HMM with periodic predictive
   model-order selection (Sec. 5.2, Algorithm 1), persistent regime templates
   tracked with the closed-form 2-Wasserstein distance between Gaussians
   (Sec. 5.4-5.5), and a transaction-cost-aware mean-variance optimizer
   (Sec. 5.6) run day by day over the out-of-sample window.
2. **Non-parametric KNN regime inference** (Sec. 6, 7.2): local conditional
   moments estimated with K-nearest-neighbors on an expanding history of
   causal feature vectors, Ledoit-Wolf shrinkage on the neighbor return
   sample, and the same transaction-cost-aware mean-variance optimizer,
   serving as a non-parametric benchmark to the HMM strategy.
3. **Passive benchmarks** ("Benchmarking Against Passive Allocations"): an
   SPX buy-and-hold position and a static equal-weight (20% each) portfolio
   over the same five-asset universe and out-of-sample window.

## Repository layout

```
src/            Shared pipeline: config, features, Wasserstein HMM, KNN,
                mean-variance optimizer, performance metrics, backtest loop
exp/            Experiment entry points (exp1_hmm, exp2_knn, exp3_benchmarks)
data/           data_loader.py - OHLCV loading through the QuantX Dataloader
tests/          pytest unit and market-data tests
results/        Metrics (JSON/CSV), plots, and RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY (and MASSIVE_API_KEY if available)
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                         # full suite, needs DATALOADER_API_KEY
```

## Running the experiments

Each experiment script loads the shared five-asset daily price panel through
`data/data_loader.py`, runs its strategy, and writes metrics, CSV artifacts,
and plots directly to `results/`.

```bash
python -m exp.exp1_hmm            # Wasserstein-HMM + MVO (executed configuration)
python -m exp.exp2_knn            # KNN regime inference + MVO
python -m exp.exp3_benchmarks      # SPX buy & hold and equal-weight benchmarks

python -m exp.exp1_hmm --full      # paper's full sample period (2005-01-01..2026-01-31)
python -m exp.exp2_knn --full
python -m exp.exp3_benchmarks --full
```

`reproduce.sh` runs the full test suite followed by all three experiments in
their `--full` (paper) configuration.

## Data

Daily adjusted close prices for five ETF proxies (S&P 500, broad bond, gold,
oil, U.S. dollar) are loaded through the QuantX Dataloader
(`POST /fetch-ohlcvs`), which is built on Massive market data and adds a
Yahoo Finance backstop. See `results/RESULTS.md`, "Deviations from the
paper", for the exact tickers used, the data source actually returned for
each ticker, and the realized sample dates. No downloaded market data is
committed to this repository; `data/data_loader.py` regenerates it on every
run.

## Parameters

Every parameter used by the shared pipeline (feature windows, HMM order
bounds, template-tracking settings, transaction-cost-aware MVO coefficients,
KNN settings, annualization convention) lives in `src/config.py` as a
module-level constant, each annotated with the paper section/equation it
comes from, or marked "not specified in the paper" when the paper leaves it
open.

## Results

See `results/RESULTS.md` for the full comparison against the paper's
reported figures (Table 1 / Sec. 8-9 for the HMM strategy, Sec. 7.2 for the
KNN benchmark, "Risk-Adjusted Performance" for the passive benchmarks),
deviations from the paper, and what was not reproduced.
