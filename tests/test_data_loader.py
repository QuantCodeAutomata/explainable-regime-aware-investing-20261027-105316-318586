"""Tests for data/data_loader.py (paper Sec. 3, "Data and Market Universe").

Marked `market_data`: they load a reduced real slice through the QuantX
Dataloader (cached across runs) and are skipped when DATALOADER_API_KEY is
absent, per the market-data skill's rule that tests do not depend on a live
call by default.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import ASSET_LABELS


@pytest.mark.market_data
def test_market_data_panel_has_expected_columns_and_no_gaps(market_data_panel: pd.DataFrame) -> None:
    expected_labels = set(ASSET_LABELS.values())
    assert set(market_data_panel.columns) == expected_labels
    assert not market_data_panel.isna().any().any()
    assert (market_data_panel > 0).all().all()
    assert market_data_panel.index.is_monotonic_increasing
    assert market_data_panel.index.tz is not None


@pytest.mark.market_data
def test_market_data_panel_covers_requested_range(market_data_panel: pd.DataFrame) -> None:
    assert market_data_panel.index.min() >= pd.Timestamp("2022-01-01", tz=market_data_panel.index.tz)
    assert market_data_panel.index.max() <= pd.Timestamp("2023-12-31", tz=market_data_panel.index.tz)
    assert len(market_data_panel) > 250  # at least one trading year of bars
