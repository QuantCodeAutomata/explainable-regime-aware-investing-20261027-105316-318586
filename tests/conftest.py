"""Shared pytest fixtures for the Explainable Regime-Aware Investing tests."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: tests that load real market data through data/data_loader.py"
    )


@pytest.fixture(scope="session")
def synthetic_price_panel() -> pd.DataFrame:
    """A small deterministic 5-asset daily price panel for unit tests.

    Not used to produce the paper's headline results; it is a synthetic
    fixture for testing the shared pipeline's mathematical properties
    (causality, constraint satisfaction, shapes) on data with a known
    structure, independent of any live market-data call.
    """
    rng = np.random.default_rng(42)
    n_days = 420
    assets = ["SPX", "BOND", "GOLD", "OIL", "USD"]
    dates = pd.bdate_range("2020-01-01", periods=n_days, tz="UTC")
    drifts = np.array([0.0004, 0.0001, 0.0002, 0.0003, -0.00005])
    vols = np.array([0.012, 0.004, 0.009, 0.02, 0.005])
    log_rets = rng.normal(loc=drifts, scale=vols, size=(n_days, len(assets)))
    log_prices = np.cumsum(log_rets, axis=0)
    prices = 100.0 * np.exp(log_prices)
    return pd.DataFrame(prices, index=dates, columns=assets)


@pytest.fixture(scope="session")
def market_data_panel(request: pytest.FixtureRequest) -> pd.DataFrame:
    """A reduced real-data slice loaded once per session via the QuantX
    Dataloader and cached on disk between test runs (market-data skill: tests
    must not depend on a live call by default, so this fixture is skipped
    whenever DATALOADER_API_KEY is absent).
    """
    if not os.environ.get("DATALOADER_API_KEY"):
        pytest.skip("DATALOADER_API_KEY is not set; skipping market-data test.")

    cache_dir = request.config.cache.mkdir("market_data_panel")
    cache_file = cache_dir / "panel.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    from data.data_loader import load_price_panel

    # Reduced slice: ~2 years, full 5-asset universe (small enough to stay
    # well within the data smoke-test budget).
    panel = load_price_panel("2022-01-01", "2023-12-31")
    panel.to_parquet(cache_file)
    return panel
