"""Parameters for the Explainable Regime-Aware Investing replication.

Every module-level constant below is used by the shared pipeline in ``src/``
and by the experiment scripts in ``exp/``. Each constant documents its unit
and its source: a section/equation of "Explainable Regime-Aware Investing"
(Boukardagha, Columbia University) when the paper gives a value, or
"not specified in the paper" when the value had to be chosen to run the
methodology.
"""

# ---------------------------------------------------------------------------
# Universe and data (Sec. 3, "Data and Market Universe")
# ---------------------------------------------------------------------------
# The paper specifies five asset classes (SPX proxy, broad bond proxy, gold,
# oil, USD proxy) but not exact Yahoo Finance tickers. Liquid, widely used
# ETF proxies are substituted for each role (not specified in the paper).
TICKERS = ["SPY", "AGG", "GLD", "USO", "UUP"]  # SPX, BOND, GOLD, OIL, USD proxies
ASSET_LABELS = {
    "SPY": "SPX",   # S&P 500 Index proxy (equities)
    "AGG": "BOND",  # Broad bond proxy (fixed income)
    "GLD": "GOLD",  # Gold (commodities)
    "USO": "OIL",   # Oil (commodities)
    "UUP": "USD",   # U.S. Dollar proxy (foreign exchange)
}

# Sample period (Sec. 3: "daily adjusted close prices ... from 2005 to 2026").
# The end date is fixed at the last month covered by the paper's Figure 11
# cumulative-return chart (2026-01), not the run date.
SAMPLE_START = "2005-01-01"  # paper Sec. 3
SAMPLE_END_PAPER = "2026-01-31"  # last month shown in paper Fig. 11
SAMPLE_END_RUN = "2026-01-31"  # executed end date (identical; data available through this date)

# Out-of-sample split date t_0 (not specified exactly in the paper). Inferred
# from Fig. 11, whose cumulative-return series are all zero in the first
# plotted month (2023-05) and increase afterwards, indicating the
# out-of-sample window begins there.
OOS_START = "2023-05-01"  # inferred from paper Fig. 11 (not stated as an exact date)

DATA_TIMEFRAME = "1d"  # daily bars (Sec. 3)

# ---------------------------------------------------------------------------
# Feature representation (Sec. 4, "Feature Representation")
# ---------------------------------------------------------------------------
VOL_WINDOW = 60     # sigma_t: 60-day rolling volatility per asset (Sec. 4)
MEAN_WINDOW = 20    # m_t: 20-day rolling mean return per asset (Sec. 4)
# All features at day t use information available up to t-1 (Sec. 4, "strict
# causality"): implemented by shifting returns by one day before computing
# rolling statistics (see src/features.py).

# ---------------------------------------------------------------------------
# Wasserstein HMM — predictive order selection (Sec. 5.2, Algorithm 1)
# ---------------------------------------------------------------------------
HMM_K_MIN = 1        # minimum candidate HMM order (not specified in the paper)
HMM_K_MAX = 5        # maximum candidate HMM order (not specified in the paper)
HMM_ORDER_SELECTION_FREQ = 5   # F_K: order re-selection frequency in trading days ("e.g., weekly", Sec. 5.2)
HMM_VALIDATION_SLICE = 20      # |V|: validation-slice length in trading days (not specified in the paper)
HMM_COMPLEXITY_LAMBDA = 2.0    # lambda_K: complexity penalty weight (paper only states "proportional to K"; value not specified)
HMM_MIN_TRAIN_OBS = 250        # minimum expanding-history length before the first HMM fit (not specified in the paper)
HMM_COV_REG = 1e-6             # diagonal jitter added to Gaussian covariances for numerical stability (not specified in the paper)
HMM_ORDER_SELECT_ITER = 40     # EM iterations used when fitting each candidate order K from a cold start (numerical optimization setting; not a paper parameter)
HMM_DAILY_REFIT_ITER = 5       # EM iterations used for the daily warm-started refit between order-reselection days (numerical optimization setting, see src/wasserstein_hmm.py:fit_gaussian_hmm)

# ---------------------------------------------------------------------------
# Wasserstein template tracking (Sec. 5.4-5.5, Algorithm 1)
# ---------------------------------------------------------------------------
# The number of persistent templates G is inferred from Sec. 9 ("Economic
# Interpretation of Market Regimes"), which states that template indices
# 0-5 are used and that one of them (Template 2) receives negligible
# posterior mass over the evaluation window, leaving a realized dominant
# set {0,1,3,4,5} relabeled Regime A-E.
N_TEMPLATES = 6          # G: number of persistent regime templates (paper Sec. 9: indices 0-5)
TEMPLATE_SMOOTHING_ETA = 0.05  # eta: exponential-smoothing rate for template updates (not specified in the paper)
TEMPLATE_CALIBRATION_FRACTION = 0.5  # fraction of the pre-OOS history used as the "initial calibration window" (Algorithm 1, Require) to seed templates (not specified in the paper)

# ---------------------------------------------------------------------------
# Transaction-cost-aware mean-variance optimization (Sec. 5.6, Eq. in Sec. 5.6/7.1/7.2)
# ---------------------------------------------------------------------------
MVO_GAMMA = 5.0      # gamma: risk-aversion coefficient on variance (not specified in the paper)
MVO_TAU = 0.05       # tau: L1 transaction-cost penalty coefficient (not specified in the paper; calibrated to produce low turnover as reported for the parametric strategy)
MVO_W_MAX = 0.6      # w_max: maximum per-asset weight (not specified in the paper)
MVO_KNN_TAU = 0.0005 # tau used for the KNN benchmark (not specified in the paper; kept an order of magnitude below MVO_TAU because the paper reports much higher KNN turnover arising from discontinuous neighbor sets, not from a different transaction-cost preference)
MVO_KNN_GAMMA = 5.0  # lambda in Algorithm 2 (same economic role as gamma; not specified in the paper)

# ---------------------------------------------------------------------------
# Non-parametric KNN conditional-moment estimator (Sec. 6, Algorithm 2)
# ---------------------------------------------------------------------------
KNN_N_NEIGHBORS = 60   # K: number of nearest neighbors (not specified in the paper)
KNN_LOOKBACK = 90      # L: minimum history length before the first KNN fit (not specified in the paper)
# Algorithm 2's "Require" line also lists a rebalance frequency F=5 and
# moment-change triggers (delta_mu, delta_Sigma), but the written algorithm
# body (steps 1-15) re-estimates moments and re-solves the MVO on every
# trading day without using F or the triggers. This implementation follows
# the written algorithm body, which is also consistent with the paper's
# reported finding that 94.13% of days show turnover above 1% (impossible
# under a 5-day rebalance cadence). See results/RESULTS.md, "Deviations from
# the paper".

# ---------------------------------------------------------------------------
# Performance evaluation (Sec. 8, Table 1; annualization not stated in the
# paper; 252 trading days/year is the standard convention used here)
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252  # annualization factor (not specified in the paper; standard convention)
RISK_FREE_RATE_ANNUAL = 0.0  # annual risk-free rate used in Sharpe/Sortino (not specified in the paper)

# ---------------------------------------------------------------------------
# Equal-weight benchmark (exp_3)
# ---------------------------------------------------------------------------
EQUAL_WEIGHT = 1.0 / len(TICKERS)  # 20% per asset across the 5-asset universe (Sec., "Benchmarking Against Passive Allocations")
