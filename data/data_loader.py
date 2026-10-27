"""Market-data loading for the Explainable Regime-Aware Investing replication.

Daily adjusted close prices are loaded for five ETF proxies that stand in for
the asset classes used in Sec. 3 ("Data and Market Universe") of the paper,
whose text names the asset roles (SPX, broad bond, gold, oil, USD) but not
exact ticker symbols:

    SPX proxy  -> SPY  (SPDR S&P 500 ETF Trust)
    BOND proxy -> AGG  (iShares Core U.S. Aggregate Bond ETF)
    GOLD proxy -> GLD  (SPDR Gold Shares)
    OIL proxy  -> USO  (United States Oil Fund)
    USD proxy  -> UUP  (Invesco DB US Dollar Index Bullish Fund)

This is an inferred choice (not specified in the paper), recorded in
results/RESULTS.md.

Deviation from the default Massive-backed data path: calling
``POST /fetch-ohlcvs`` with ``source="massive"`` for this universe is
rejected by the gateway with an invalid-API-key error for the configured
``MASSIVE_API_KEY``. The QuantX Dataloader is still the only endpoint used
for OHLCV bars; it is called with its documented ``source="yahoo"`` option
instead of ``source="massive"``, which is a supported source value of the
same ``/fetch-ohlcvs`` route, not a different data service. See
results/RESULTS.md, "Deviations from the paper".
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Market data: QuantX Dataloader — https://quantx.limex.pro/
#
# Price history in this project is loaded through the QuantX Dataloader, a
# research-ready market-data service available through a REST API and as an
# MCP server for AI agents. It is built on market data from Massive
# (https://massive.com/) and prepares every stock history for backtesting:
#   - prices adjusted for dividends, ready for total-return research;
#   - one continuous history when a company changes its ticker;
#   - clean separation when a ticker has belonged to different companies;
#   - careful corporate-action handling and fine-grained data polishing.
#
# Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
# MASSIVE_API_KEY at https://massive.com/.
DATALOADER_API_ROOT = os.environ.get(
    "DATALOADER_API_URL", "https://quantx-api.limex.pro/apis/dataloader"
)

# Ticker proxy map used by every experiment in this repository (Sec. 3).
ASSET_TICKERS: Dict[str, str] = {
    "SPX": "SPY",
    "BOND": "AGG",
    "GOLD": "GLD",
    "OIL": "USO",
    "USD": "UUP",
}


class DataLoaderError(RuntimeError):
    """Raised when the QuantX Dataloader request fails or returns no usable data."""


def _fetch_ohlcvs(
    tickers: List[str],
    start_date: str,
    end_date: str,
    timeframe: str,
    source: str,
    identity_date: str | None = None,
) -> dict:
    api_key = os.environ.get("DATALOADER_API_KEY")
    if not api_key:
        raise DataLoaderError(
            "DATALOADER_API_KEY is not set in the environment; cannot call the "
            "QuantX Dataloader."
        )
    headers = {"Authorization": f"Bearer {api_key}"}
    if source == "massive":
        massive_key = os.environ.get("MASSIVE_API_KEY")
        if not massive_key:
            raise DataLoaderError(
                "MASSIVE_API_KEY is not set in the environment; required for "
                "source='massive'."
            )
        headers["X-Massive-Api-Key"] = massive_key

    payload = {
        "tickers": tickers,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": source,
        "adjust_dividends": True,
    }
    if identity_date is not None:
        payload["identity_date"] = identity_date

    url = f"{DATALOADER_API_ROOT}/fetch-ohlcvs"
    resp = requests.post(url, json=payload, headers=headers, timeout=180)
    if resp.status_code != 200:
        raise DataLoaderError(
            f"Dataloader /fetch-ohlcvs failed with HTTP {resp.status_code}: "
            f"{resp.text[:500]}"
        )
    body = resp.json()
    if body.get("error"):
        raise DataLoaderError(f"Dataloader /fetch-ohlcvs returned an error: {body['error']}")
    return body["result"]


def _fetch_single_ticker_closes(
    ticker: str,
    start_date: str,
    end_date: str,
    timeframe: str,
    identity_date: str | None,
) -> pd.Series:
    """Fetch one ticker's full-range OHLCV inline and return its close series.

    A single ``POST /fetch-ohlcvs`` call for all five proxies over the full
    2005-2026 window exceeds the Dataloader's inline response limit
    (``data_stored=true``), and the ``save-dataframe`` + artifact export
    path for that stored result drops the per-row ``Date`` column (the
    exported file carries only an integer ``row`` index), which would break
    the daily alignment this pipeline depends on. A single-ticker request
    over the *same, unsplit* date range stays within the inline limit and
    keeps the ``Date`` column, so the panel is built as one inline request
    per ticker -- never by splitting any one ticker's own date range, which
    would make its dividend adjustment inconsistent across the join.
    """
    result = _fetch_ohlcvs(
        [ticker], start_date, end_date, timeframe, source="yahoo", identity_date=identity_date
    )
    ticker_details = result.get("ticker_details", [])
    logger.info("ticker_details[%s]: %s", ticker, ticker_details)
    for td in ticker_details:
        if td.get("error"):
            raise DataLoaderError(f"Ticker {ticker} reported error: {td['error']}")
        src = td.get("data_source")
        if src and src != "yahoo":
            logger.warning("Ticker %s served from data_source=%s (expected yahoo)", ticker, src)

    if result.get("data_stored"):
        raise DataLoaderError(
            f"Dataloader response for single ticker {ticker} unexpectedly exceeded "
            "the inline limit (data_stored=true)."
        )

    rows = result.get("ohlcv_df", {}).get("rows")
    if not rows:
        raise DataLoaderError(
            f"Dataloader returned no OHLCV rows for ticker={ticker}, "
            f"range={start_date}..{end_date}. An empty daily result is treated as an error."
        )

    df = pd.DataFrame(rows)
    if "Date" not in df.columns:
        raise DataLoaderError(f"Dataloader rows for {ticker} are missing 'Date': {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.set_index("Date").sort_index()

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise DataLoaderError(f"Dataloader response for {ticker} missing column '{col}'")

    low, high, open_, close = df["low"], df["high"], df["open"], df["close"]
    bad = ~(
        (low <= open_ + 1e-6)
        & (open_ <= high + 1e-6)
        & (low <= close + 1e-6)
        & (close <= high + 1e-6)
    )
    if bad.any():
        logger.warning("%s: %d bars fail low<=open/close<=high", ticker, int(bad.sum()))

    return close.rename(ticker)


def load_price_panel(
    start_date: str,
    end_date: str,
    tickers: Dict[str, str] | None = None,
    timeframe: str = "1d",
    identity_date: str | None = None,
) -> pd.DataFrame:
    """Load an aligned, dividend-adjusted daily close-price panel.

    Calls the QuantX Dataloader once per asset (``POST /fetch-ohlcvs``,
    one ticker at a time, each over the full requested date range, never
    split by date), validates every response, and returns a DataFrame
    indexed by UTC date with one column per asset label (e.g. SPX, BOND,
    GOLD, OIL, USD), restricted to dates on which every asset traded
    (inner join), since the feature construction in ``src/features.py``
    requires a synchronized panel.

    Parameters
    ----------
    start_date, end_date:
        Inclusive ISO date strings bounding the requested sample.
    tickers:
        Mapping of asset label -> ticker symbol. Defaults to
        ``ASSET_TICKERS``.
    timeframe:
        Dataloader bar size; daily ("1d") for this replication (Sec. 3).
    identity_date:
        Date on which each ticker identifies the intended security,
        forwarded to the Dataloader. Defaults to ``end_date`` (the
        Dataloader's own default) when not given.
    """
    tickers = tickers or ASSET_TICKERS

    series = {}
    for label, ticker in tickers.items():
        series[label] = _fetch_single_ticker_closes(
            ticker, start_date, end_date, timeframe, identity_date
        )

    panel = pd.DataFrame(series).sort_index()
    n_before = len(panel)
    panel = panel.dropna(how="any")
    logger.info(
        "Loaded price panel: %d rows before alignment, %d after inner-join alignment, "
        "columns=%s, range=%s..%s",
        n_before,
        len(panel),
        list(panel.columns),
        panel.index.min() if len(panel) else None,
        panel.index.max() if len(panel) else None,
    )
    if panel.empty:
        raise DataLoaderError("Aligned price panel is empty after dropping missing rows.")
    return panel
