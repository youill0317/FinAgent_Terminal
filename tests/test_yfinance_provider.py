from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pandas as pd  # noqa: PANDAS_OK

from finagent_terminal.data.models import FreshnessState, MarketScalar
from finagent_terminal.data.yfinance_provider import YFinanceProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

SPY_FAST_INFO: Final[dict[str, MarketScalar]] = {
    "lastPrice": 501.25,
    "previousClose": 500.0,
    "currency": "USD",
    "marketCap": 450_000_000_000,
    "volume": 70_000_000,
    "dayHigh": 502.0,
    "dayLow": 499.5,
}
SPY_INFO: Final[dict[str, MarketScalar]] = {
    "longName": "SPDR S&P 500 ETF Trust",
    "sector": "ETF",
    "industry": "Exchange Traded Fund",
    "trailingPE": 24.5,
    "forwardPE": 21.0,
    "dividendYield": 0.012,
}


@dataclass(frozen=True, slots=True)
class FakeTicker:
    fast_info: Mapping[str, MarketScalar]
    info: Mapping[str, MarketScalar]
    frame: pd.DataFrame

    def history(
        self,
        *,
        period: str,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        assert period == "5d"
        assert interval == "1d"
        assert auto_adjust is False
        return self.frame


def spy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [499.0, 500.0],
            "High": [501.0, 502.0],
            "Low": [498.0, 499.5],
            "Close": [500.0, 501.25],
            "Volume": [65_000_000, 70_000_000],
        },
        index=pd.date_range("2024-01-02", periods=2, freq="D"),
    )


def ticker_factory(ticker: FakeTicker) -> Callable[[str], FakeTicker]:
    def factory(symbol: str) -> FakeTicker:
        assert symbol == "SPY"
        return ticker

    return factory


def test_spy_fixture_normalizes_quote_history_and_profile() -> None:
    ticker = FakeTicker(SPY_FAST_INFO, SPY_INFO, spy_frame())
    provider = YFinanceProvider(ticker_factory=ticker_factory(ticker))

    snapshot = provider.fetch("spy", period="5d", interval="1d")

    assert snapshot.symbol == "SPY"
    assert snapshot.freshness == FreshnessState.FRESH
    assert snapshot.error is None
    assert snapshot.quote is not None
    assert snapshot.quote.last_price == 501.25
    assert snapshot.quote.change == 1.25
    assert snapshot.quote.change_percent == 0.0025
    assert snapshot.history is not None
    assert snapshot.history.rows == 2
    assert snapshot.history.frame["Close"].tolist() == [500.0, 501.25]
    assert snapshot.profile is not None
    assert snapshot.profile.name == "SPDR S&P 500 ETF Trust"


def test_empty_history_returns_metadata_without_rows() -> None:
    ticker = FakeTicker(SPY_FAST_INFO, SPY_INFO, pd.DataFrame())
    provider = YFinanceProvider(ticker_factory=ticker_factory(ticker))

    snapshot = provider.fetch("SPY", period="5d", interval="1d")

    assert snapshot.history is not None
    assert snapshot.history.rows == 0
    assert snapshot.history.frame.empty
