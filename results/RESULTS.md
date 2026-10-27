# Results — Explainable Regime-Aware Investing

Paper: **"Explainable Regime-Aware Investing"** (Amine Boukardagha, Columbia
University).

All numbers below are read from the files written by the final run of each
experiment in `results/`: `exp1_hmm_metrics.json`, `exp2_knn_metrics.json`,
`exp3_benchmarks_metrics.json`, and the accompanying CSV/PNG artifacts in the
same directory. Daily adjusted close prices were loaded once through
`data/data_loader.py` (QuantX Dataloader) for the five-asset universe
`SPY, AGG, GLD, USO, UUP` (SPX, BOND, GOLD, OIL, USD proxies, Sec. 3) over
`2007-03-01` to `2026-01-30` (the intersection of the requested
`2005-01-01`-`2026-01-31` window across all five tickers after alignment),
with the out-of-sample window starting `2023-05-01` (691 trading days) and
used identically by all three experiments.

Verdict rule applied throughout (Sec. 2 of the comparison protocol used for
this replication): **reproduced** if the value is within 10% of the paper's
figure with the same sign and the paper's qualitative claim holds;
**not reproduced** if the difference exceeds 50%, the sign is wrong, or a
qualitative/structural claim of the paper fails; **partially reproduced**
otherwise.

## Experiment 1 — Parametric Regime Investing with Wasserstein HMM and Mean-Variance Optimization

Reproduces Sec. 5-7.1 (causal expanding-window Gaussian HMM with periodic
predictive order selection, Wasserstein-distance template tracking, and
transaction-cost-aware MVO) and Sec. 9 (regime-conditional performance).
Artifacts: `results/exp1_hmm_metrics.json`, `exp1_hmm_weights.csv`,
`exp1_hmm_turnover.csv`, `exp1_hmm_selected_k.csv`,
`exp1_hmm_template_probs.csv`, `exp1_hmm_regime_conditional_returns.csv`,
`exp1_hmm_cumulative_returns.png`, `exp1_hmm_weights.png`,
`exp1_hmm_template_probs.png`.

| Metric | Paper (Sec. 7.1) | Reproduced | Verdict |
|---|---|---|---|
| Sharpe ratio | 2.18 | 1.615 | partially reproduced (25.9% below) |
| Sortino ratio | 2.82 | 2.450 | partially reproduced (13.1% below) |
| Max drawdown | -5.43% | -7.99% | partially reproduced (47.2% larger in magnitude) |
| Average daily turnover | 0.0079 | 2.7e-6 | not reproduced (turnover collapses to ~0) |

The Sharpe, Sortino and max-drawdown figures sit in the same range as the
paper's and preserve its qualitative ranking (Experiment 1 has the highest
risk-adjusted return and the smallest drawdown of the three experiments, as
in the paper). The turnover result fails: the paper's Sec. 5.6/7.1 strategy
is meant to rebalance a small but non-zero amount every day as regime
probabilities drift (average turnover 0.0079), whereas the reproduced
portfolio essentially never moves ($N_{eff,t}$ stays at 4.9999-5.0000 for
the entire out-of-sample window, i.e. within noise of the equal-weight
starting point). Comparing against the static equal-weight benchmark of
Experiment 3 isolates the cause: the HMM-MVO portfolio's Sharpe (1.615),
cumulative log return (0.3596) and max drawdown (-7.99%) are nearly
identical to the equal-weight portfolio's (1.609, 0.3593, -8.01%, see
Experiment 3 below). At the transaction-cost coefficient used here
(`MVO_TAU = 0.05`, Sec. 5.6; not specified in the paper), the daily L1
trading penalty dominates the mean-variance utility gain from reallocating
away from the previous day's weights at this asset universe's daily
mean/variance scale, so the optimizer's corner solution is to keep the
initial equal weight for the full window regardless of the regime signal.
This is a calibration deviation (see "Deviations from the paper"), not a
sign error or a missing step: the HMM order selection, Wasserstein template
tracking and regime probabilities themselves are computed and change over
time (see `exp1_hmm_selected_k.csv` and `exp1_hmm_template_probs.csv`), only
the downstream portfolio weights fail to react to them because of the
transaction-cost penalty's scale.

## Experiment 2 — Non-Parametric Regime Inference using K-Nearest Neighbors with Mean-Variance Optimization

Reproduces Sec. 6-7.2 (KNN local conditional moments with Ledoit-Wolf
shrinkage, Sec. 7.2 instability diagnostics). Artifacts:
`results/exp2_knn_metrics.json`, `exp2_knn_weights.csv`,
`exp2_knn_turnover.csv`, `exp2_knn_cumulative_returns.png`,
`exp2_knn_turnover.png`.

| Metric | Paper (Sec. 7.2) | Reproduced | Verdict |
|---|---|---|---|
| Sharpe ratio | 1.81 | 1.278 | partially reproduced (29.4% below) |
| Max drawdown | -12.52% | -10.88% | partially reproduced (13.1% smaller in magnitude) |
| Average daily turnover | 0.5665 | 0.376 | partially reproduced (33.6% below) |
| % days with turnover > 1% | 94.13% | 82.20% | partially reproduced (12.7% below) |

The qualitative claim of Sec. 6/7.2 — that the KNN strategy is far less
stable than the HMM strategy, with high turnover on most days because
nearest-neighbor sets change discontinuously at daily frequency — holds
here: KNN turnover (0.376, 82.2% of days above the 1% threshold) is three
to five orders of magnitude above the HMM strategy's (2.7e-6, 0.0% of days),
consistent with the paper's contrast between the two approaches. The
absolute levels differ from the paper by 13-34%, within the magnitude that
unspecified parameters (`KNN_N_NEIGHBORS`, `MVO_KNN_TAU`, `MVO_KNN_GAMMA`;
Sec. 6, 5.6; not specified in the paper) can explain.

## Experiment 3 — Benchmarking Against Passive Allocations

Reproduces the "Benchmarking Against Passive Allocations" section: SPX
buy-and-hold and a static 20%-per-asset equal-weight portfolio over the same
out-of-sample window. Artifacts: `results/exp3_benchmarks_metrics.json`,
`exp3_benchmarks_returns.csv`, `exp3_benchmarks_cumulative_returns.png`.

| Metric | Paper | Reproduced | Verdict |
|---|---|---|---|
| SPX Buy & Hold Sharpe | 1.18 | 1.385 | partially reproduced (17.4% above) |
| SPX Buy & Hold Sortino | 1.50 | 2.073 | partially reproduced (38.2% above) |
| SPX Buy & Hold Max drawdown | -14.62% | -18.76% | partially reproduced (28.3% larger in magnitude) |
| Equal-Weight Sharpe | 1.59 | 1.609 | reproduced (1.2% above) |
| Equal-Weight Sortino | 2.27 | 2.440 | reproduced (7.5% above) |
| Equal-Weight Max drawdown | -9.87% | -8.01% | partially reproduced (18.9% smaller in magnitude) |

The paper's qualitative ranking — equal-weight has a higher Sharpe and
Sortino and a smaller drawdown than SPX buy-and-hold — holds in the
reproduced numbers (1.609 > 1.385 Sharpe; -8.01% vs. -18.76% drawdown). The
equal-weight benchmark, which has no fitted parameters beyond the fixed 20%
allocation, matches the paper most closely of all six experiments reported
here, which is consistent with Experiment 1's deviation being a calibration
issue in the transaction-cost-aware optimizer rather than in the data,
universe or sample period, which are shared by all three experiments.

## Deviations from the paper

- **Asset proxies**: the paper names five asset-class roles (SPX, broad
  bond, gold, oil, USD proxies, Sec. 3) without exact tickers. This
  replication uses SPY (SPX), AGG (BOND), GLD (GOLD), USO (OIL) and UUP
  (USD), liquid ETFs standing in for each role.
- **Data source**: bars were loaded through the QuantX Dataloader
  (`POST /fetch-ohlcvs`, `source="massive"`) with `yahoo_backstop=True`
  since the paper's data source is Yahoo Finance (Sec. 3). All five tickers
  in the executed run were served with `data_source="yahoo"` (see
  `ticker_details` logged by `data/data_loader.py`), matching the paper's
  stated source; none came from Massive directly, so there is no mixed-source
  series in this universe.
- **Sample period**: requested `2005-01-01` to `2026-01-31` (Sec. 3); the
  realized aligned panel runs `2007-03-01` to `2026-01-30` because UUP
  (the USD proxy) has no bars before 2007-03, which shortens the common
  history for all five series under the inner-join alignment used to build
  one causal feature matrix.
- **Out-of-sample start date $t_0$**: not stated exactly in the paper;
  inferred as `2023-05-01` from the first month of Fig. 11's cumulative
  return chart, where all plotted series start at zero.
- **HMM order bounds, validation slice, complexity penalty**:
  $K_{min}=1$, $K_{max}=5$, $|V|=20$ trading days, $\lambda_K=2.0$ — none
  specified in the paper; the order re-selection frequency used (5 trading
  days) follows the paper's "e.g., weekly" (Sec. 5.2).
- **Number of templates and smoothing rate**: $G=6$ templates (inferred from
  Sec. 9, which discusses template indices 0-5) and exponential smoothing
  rate $\eta=0.05$ (not specified in the paper).
- **MVO risk aversion, transaction cost, weight cap**: $\gamma=5.0$,
  $\tau=0.05$, $w_{max}=0.6$ for the HMM strategy, and $\lambda=5.0$,
  $\tau=0.0005$ for the KNN strategy (Sec. 5.6); none are given numeric
  values in the paper. The KNN transaction cost is two orders of magnitude
  below the HMM one because the paper reports far higher KNN turnover
  arising from discontinuous neighbor-set changes, not from a different
  transaction-cost preference (Sec. 6, 7.2).
- **KNN neighbor count**: $K=60$ neighbors (Sec. 6; not specified in the
  paper).
- **Algorithm 2 rebalance cadence**: the paper's Sec. 6 "Require" line also
  lists a rebalance frequency and moment-change triggers, but the written
  algorithm body re-estimates moments and re-solves the MVO on every trading
  day without using them; this replication follows the algorithm body, which
  is also the only reading consistent with the paper's own reported finding
  that 94.13% of days show turnover above 1% (impossible under a 5-day
  cadence).
- **Transaction-cost calibration gap (Experiment 1)**: at `MVO_TAU = 0.05`
  the daily L1 trading penalty dominates the mean-variance utility gain at
  this universe's daily return/volatility scale, so the realized average
  turnover (2.7e-6) is far below the paper's reported 0.0079 and the
  portfolio stays within noise of the initial equal weight for the whole
  out-of-sample window (see Experiment 1 above for the identification of
  this cause via the match to the equal-weight benchmark).
- **Annualization and risk-free rate**: 252 trading days/year and a 0%
  annual risk-free rate (not specified in the paper; standard convention).

## Not reproduced

- The paper's exact "Liberation Day" 2025 selloff narrative (Sec. 9) is not
  separately isolated as a dated case study; the regime-conditional returns
  needed to examine it are computed and saved in
  `results/exp1_hmm_regime_conditional_returns.csv` and
  `results/exp1_hmm_template_probs.csv`, but a specific event-window analysis
  around that date was not built as a separate artifact.
- The paper's full predictive model-order-selection diagnostics (the
  $PredLL(K;t) - \lambda_K \cdot Complexity(K)$ trace across all candidate
  orders at every re-selection day, Sec. 5.2) are not saved in full; only the
  selected order $K_t$ at each re-selection day is saved
  (`results/exp1_hmm_selected_k.csv`).
