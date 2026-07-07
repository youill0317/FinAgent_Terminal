from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, override

import pandas as pd  # noqa: PANDAS_OK

from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    MarketScalar,
)
from finagent_terminal.data.session import DataSession
from finagent_terminal.data.yfinance_provider import YFinanceProvider

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pytest

SPY_FAST_INFO: Final[dict[str, MarketScalar]] = {
    "lastPrice": 101.0,
    "previousClose": 100.0,
}
SPY_INFO: Final[dict[str, MarketScalar]] = {"longName": "SPY Trust"}


@dataclass(frozen=True, slots=True)
class FakeTicker:
    fast_info: Mapping[str, MarketScalar]
    info: Mapping[str, MarketScalar]
    frame: pd.DataFrame
    expected_period: str | None = "5d"
    expected_interval: str | None = "1d"

    def history(
        self,
        *,
        period: str,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        if self.expected_period is not None:
            assert period == self.expected_period
        if self.expected_interval is not None:
            assert interval == self.expected_interval
        assert auto_adjust is False
        return self.frame


@dataclass(slots=True)  # noqa: MUTABLE_OK
class MutableClock:
    current: datetime

    def now(self) -> datetime:
        return self.current


@dataclass(slots=True)  # noqa: MUTABLE_OK
class CountingFactory:
    ticker: FakeTicker
    calls: int = 0

    def __call__(self, symbol: str) -> FakeTicker:
        assert symbol == "SPY"
        self.calls += 1
        return self.ticker


@dataclass(frozen=True, slots=True)
class ProviderOfflineError(RuntimeError):
    symbol: str
    period: str
    interval: str

    @override
    def __str__(self) -> str:
        return f"{self.symbol} {self.period} {self.interval} unavailable"


class ExplodingProvider:
    def fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        raise ProviderOfflineError(symbol, period, interval)


def spy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0, 101.0], "Volume": [1_000, 2_000]},
        index=pd.date_range("2024-01-02", periods=2, freq="D"),
    )


def test_spy_fixture_and_cache_freshness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FINAGENT_HOME", str(tmp_path))
    clock = MutableClock(datetime(2024, 1, 3, tzinfo=UTC))
    ticker = FakeTicker(SPY_FAST_INFO, SPY_INFO, spy_frame())
    factory = CountingFactory(ticker)
    session = DataSession(
        YFinanceProvider(ticker_factory=factory, clock=clock.now),
        clock=clock.now,
        stale_after=timedelta(minutes=5),
    )

    first = session.get_or_fetch("SPY", period="5d", interval="1d")
    second = session.get_or_fetch("spy", period="5d", interval="1d")
    clock.current += timedelta(minutes=6)
    cached = session.get_cached("SPY", period="5d", interval="1d")

    assert factory.calls == 1
    assert first is second
    assert first.freshness == FreshnessState.FRESH
    assert cached is not None
    assert cached.freshness == FreshnessState.STALE
    assert list(tmp_path.rglob("*")) == []


def test_provider_exception_sets_failed_state() -> None:
    clock = MutableClock(datetime(2024, 1, 3, tzinfo=UTC))
    session = DataSession(ExplodingProvider(), clock=clock.now)

    snapshot = session.get_or_fetch("BAD", period="5d", interval="1d")

    assert snapshot.freshness == FreshnessState.FAILED
    assert snapshot.error is not None
    assert snapshot.error.symbol == "BAD"
    assert "unavailable" in snapshot.error.message
    assert session.freshness("BAD", period="5d", interval="1d") == FreshnessState.FAILED


def test_cache_key_separates_period_and_interval() -> None:
    clock = MutableClock(datetime(2024, 1, 3, tzinfo=UTC))
    ticker = FakeTicker(
        SPY_FAST_INFO,
        SPY_INFO,
        spy_frame(),
        expected_period=None,
        expected_interval=None,
    )
    factory = CountingFactory(ticker)
    session = DataSession(
        YFinanceProvider(ticker_factory=factory, clock=clock.now),
        clock=clock.now,
    )

    _ = session.get_or_fetch("SPY", period="5d", interval="1d")
    _ = session.get_or_fetch("SPY", period="1mo", interval="1d")
    _ = session.get_or_fetch("SPY", period="5d", interval="1wk")
    _ = session.get_or_fetch("SPY", period="5d", interval="1d")

    assert factory.calls == 3


def test_not_loaded_state_before_fetch() -> None:
    clock = MutableClock(datetime(2024, 1, 3, tzinfo=UTC))
    ticker = FakeTicker(SPY_FAST_INFO, SPY_INFO, spy_frame())
    session = DataSession(
        YFinanceProvider(ticker_factory=CountingFactory(ticker), clock=clock.now),
        clock=clock.now,
    )

    assert (
        session.freshness("SPY", period="5d", interval="1d")
        == FreshnessState.NOT_LOADED
    )
