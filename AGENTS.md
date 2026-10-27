# AGENTS.md

Guide for contributors and coding agents working in this repository.

## What this repository implements

A replication of three experiments from **"Explainable Regime-Aware
Investing"** (Amine Boukardagha, Columbia University):

1. Parametric regime investing with a causal, expanding-window Gaussian HMM,
   periodic predictive model-order selection, Wasserstein-distance regime
   template tracking, and a transaction-cost-aware mean-variance optimizer
   (paper Sec. 5, 7.1, 9).
2. Non-parametric K-nearest-neighbors regime inference with Ledoit-Wolf
   covariance shrinkage and the same mean-variance optimizer (Sec. 6, 7.2).
3. Passive benchmarks: SPX buy-and-hold and a static equal-weight portfolio
   over the same five-asset universe and out-of-sample window
   ("Benchmarking Against Passive Allocations").

See `README.md` for the paper-level description and `results/RESULTS.md` for
the numbers produced by the latest run, compared against the paper, with
verdicts, deviations and what was not reproduced.

## Layout

```
src/            Flat library modules: config, data schema helpers, feature
                construction, the Wasserstein HMM, the KNN moment estimator,
                the mean-variance optimizer, performance metrics, and the
                shared daily backtest loop used by both active strategies.
exp/            Experiment entry points: exp1_hmm, exp2_knn, exp3_benchmarks.
                Each runs as `python -m exp.<name>` and calls only the
                functions in src/ with their documented signatures.
data/           data_loader.py loads daily OHLCV through the QuantX
                Dataloader. No other loader exists anywhere in the repository.
tests/          pytest unit tests (synthetic/fixture-based) and
                `@pytest.mark.market_data` tests that use a cached reduced
                slice of real data.
results/        Metrics (JSON/CSV), plots and RESULTS.md. Rerunning the
                experiments overwrites these files.
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY (and MASSIVE_API_KEY if available)
```

## Tests

```bash
python -m pytest -m "not market_data"   # fast, no network/API key required
python -m pytest                        # includes market_data tests; needs DATALOADER_API_KEY
```

## Experiments

```bash
python -m exp.exp1_hmm
python -m exp.exp2_knn
python -m exp.exp3_benchmarks
```

Each experiment's reduced, time-bounded configuration is the one documented
in `src/config.py` and used for the results checked into `results/`; none of
the three experiments has a `--full` flag because none of their parameters
were reduced for runtime — the full paper-specified universe, sample period
and daily frequency are what every run above executes.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` variables, each with a one-line comment giving its meaning,
  unit and source (a paper section, or "not specified in the paper"). No
  parameter is hardcoded anywhere else.
- Market data is loaded only through `data/data_loader.py`
  (`POST /fetch-ohlcvs` on the QuantX Dataloader). No module under `src/` or
  `exp/` downloads data directly.
- `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from the environment
  (via an uncommitted `.env`) and are never written to code, configuration,
  logs or results. `.env.example` holds only empty placeholders.
- After rerunning an experiment, the files it writes in `results/` are
  overwritten. The numbers and verdicts written by hand in
  `results/RESULTS.md` must then be checked against the new files and
  updated to match — they are not regenerated automatically.
- Run the full test suite after changing any shared `src/` component or the
  data path, and before pushing. After a small, localized edit, it is enough
  to run the tests for the module touched.
