# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from finagent import etf, markets
from tests import fake_sources

TODAY = date(2026, 11, 7)


def test_yield_adjusts_for_later_splits_and_sums_same_day_rows() -> None:
    dividends = etf.by_ticker(
        pd.DataFrame(markets.parse_cash_dividends(fake_sources.ALPACA_DIVIDENDS)),
        "rate",
    )
    splits = etf.by_ticker(
        pd.DataFrame(markets.parse_splits(fake_sources.ALPACA_SPLITS)), "ratio"
    )

    vgt = etf.distribution_yield(dividends["VGT"], splits["VGT"], 107.06, TODAY)
    ewt = etf.distribution_yield(dividends["EWT"], [], 85.88, TODAY)

    assert vgt == pytest.approx(0.4725 / 107.06 * 100)
    assert ewt == pytest.approx(2.81615 / 85.88 * 100)
