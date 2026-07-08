from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import final, override

from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    OhlcvFrame,
    Profile,
    Quote,
)
from finagent_terminal.data.session import DataSession
from finagent_terminal.ui.screens.ticker import TickerScreen
from finagent_terminal.ui.state import create_app_state
from tests.test_compare import fixture_frame

FETCHED_AT = datetime(2024, 1, 5, 12, tzinfo=UTC)
PRICES = {
    "AAPL": [500.0, 501.25, 505.0, 502.0, 501.25],
    "SPY": [100.0, 110.0, 111.0, 112.0, 113.0],
    "QQQ": [50.0, 60.0],
    "DIA": [200.0, 210.0],
    "MSFT": [200.0, 220.0],
    "NVDA": [300.0, 330.0],
    "AMD": [100.0, 120.0],
    "FLAT": [100.0, 100.0, 100.0, 100.0],
    "SHORT": [100.0],
}


@dataclass(frozen=True, slots=True)
class SymbolUnavailableError(Exception):
    symbol: str

    @override
    def __str__(self) -> str:
        return f"{self.symbol} unavailable"


@dataclass(frozen=True, slots=True)
class FakeProvider:
    failed_symbol: str | None = None
    missing_close_symbol: str | None = None

    def fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        if symbol == self.failed_symbol:
            raise SymbolUnavailableError(symbol)
        prices = PRICES[symbol]
        frame = fixture_frame(prices)
        if symbol == self.missing_close_symbol:
            frame = frame.rename(columns={"Close": "Open"})
        last_price = prices[-1]
        previous_close = prices[-2] if len(prices) > 1 else prices[-1]
        return MarketDataSnapshot(
            symbol=symbol,
            period=period,
            interval=interval,
            fetched_at=FETCHED_AT,
            freshness=FreshnessState.FRESH,
            quote=Quote(
                symbol=symbol,
                last_price=last_price,
                previous_close=previous_close,
                change=last_price - previous_close,
                change_percent=(last_price - previous_close) / previous_close,
                currency="USD",
                market_cap=3_000_000_000,
                volume=13,
                day_high=502.0,
                day_low=499.5,
                fetched_at=FETCHED_AT,
            ),
            history=OhlcvFrame(
                symbol=symbol,
                period=period,
                interval=interval,
                frame=frame,
                rows=len(frame),
                fetched_at=FETCHED_AT,
            ),
            profile=Profile(
                symbol=symbol,
                name="Apple Inc.",
                sector="Technology",
                industry="Consumer Electronics",
                website=None,
                market_cap=3_000_000_000,
                trailing_pe=28.5,
                forward_pe=24.0,
                dividend_yield=0.005,
                beta=1.2,
                fetched_at=FETCHED_AT,
            ),
        )


@final
class CountingProvider:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider
        self.calls: list[str] = []

    def fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        self.calls.append(symbol)
        return self.provider.fetch(symbol, period, interval)


def test_ticker_fetches_symbol_and_renders_metrics_indicators() -> None:
    screen = TickerScreen(create_app_state(), DataSession(FakeProvider()))

    screen.load_symbol("aapl")

    rendered = "\n".join(screen.lines())
    assert "Symbol: AAPL" in rendered
    assert "Freshness: fresh" in rendered
    assert "Last: 501.25 USD" in rendered
    assert "Price chart:" in rendered
    assert "Indicators:" in rendered
    assert "rsi_14" in rendered
    assert "Fetched: 2024-01-05 12:00 UTC" in rendered


def test_ticker_risk_metrics_success_with_spy_benchmark() -> None:
    # Given
    screen = TickerScreen(create_app_state(), DataSession(FakeProvider()))

    # When
    screen.load_symbol("aapl")

    # Then
    rendered = "\n".join(screen.lines())
    assert "Risk Metrics:" in rendered
    assert "Volatility:" in rendered
    assert "Max Drawdown:" in rendered
    assert "Sharpe:" in rendered
    assert "Sortino:" in rendered
    assert "Historical VaR 95:" in rendered
    assert "Beta vs SPY:" in rendered
    assert "Alpha vs SPY:" in rendered


def test_ticker_risk_missing_benchmark_renders_na() -> None:
    # Given
    screen = TickerScreen(create_app_state(), DataSession(FakeProvider("SPY")))

    # When
    screen.load_symbol("aapl")

    # Then
    rendered = "\n".join(screen.lines())
    assert "Last: 501.25 USD" in rendered
    assert "Price chart:" in rendered
    assert "Benchmark: SPY unavailable" in rendered
    assert "Beta vs SPY: n/a" in rendered
    assert "Alpha vs SPY: n/a" in rendered


def test_ticker_risk_missing_close_no_spy_fetch_with_default_compare() -> None:
    # Given
    state = create_app_state()
    provider = CountingProvider(FakeProvider(missing_close_symbol="AAPL"))
    screen = TickerScreen(
        state,
        DataSession(provider),
    )

    # When
    screen.load_symbol("aapl")

    # Then
    rendered = "\n".join(screen.lines())
    assert "Last: 501.25 USD" in rendered
    assert "Price chart:" in rendered
    assert "(missing close data)" in rendered
    assert provider.calls == ["AAPL"]


def test_ticker_risk_empty_returns_no_spy_fetch_with_default_compare() -> None:
    # Given
    state = create_app_state()
    provider = CountingProvider(FakeProvider())
    screen = TickerScreen(state, DataSession(provider))

    # When
    screen.load_symbol("short")

    # Then
    rendered = "\n".join(screen.lines())
    assert "Volatility: n/a" in rendered
    assert "Sharpe: n/a" in rendered
    assert "Sortino: n/a" in rendered
    assert "Historical VaR 95: n/a" in rendered
    assert "Beta vs SPY: n/a" in rendered
    assert provider.calls == ["SHORT"]


def test_ticker_risk_zero_variance_renders_na() -> None:
    # Given
    screen = TickerScreen(create_app_state(), DataSession(FakeProvider()))

    # When
    screen.load_symbol("flat")

    # Then
    rendered = "\n".join(screen.lines())
    assert "Volatility: 0.00%" in rendered
    assert "Sharpe: n/a" in rendered
    assert "Sortino: n/a" in rendered
    assert "Historical VaR 95: 0.00%" in rendered
    assert "Beta vs SPY: 0.00" in rendered


def test_ticker_risk_refresh_recomputes_for_current_symbol() -> None:
    # Given
    screen = TickerScreen(create_app_state(), DataSession(FakeProvider()))

    # When
    screen.load_symbol("aapl")
    screen.load_symbol("flat")

    # Then
    rendered = "\n".join(screen.lines())
    assert "Symbol: FLAT" in rendered
    assert "Last: 100.00 USD" in rendered
    assert "Last: 501.25 USD" not in rendered
    assert "Volatility: 0.00%" in rendered


def test_ticker_compare_renders_focused_symbol_vs_default_peers() -> None:
    screen = TickerScreen(create_app_state(), DataSession(FakeProvider()))

    screen.load_symbol("aapl")

    rendered = "\n".join(screen.lines())
    assert "Compare: AAPL vs SPY, QQQ, DIA" in rendered
    assert "Selected: AAPL, SPY, QQQ, DIA" in rendered
    assert "Skipped: none" in rendered
    assert "Normalized returns:" in rendered
    assert "Relative performance:" in rendered
    assert "SPY" in rendered
    assert "10.00%" in rendered


def test_ticker_compare_keeps_output_when_peer_is_skipped() -> None:
    screen = TickerScreen(create_app_state(), DataSession(FakeProvider("QQQ")))

    screen.load_symbol("aapl")

    rendered = "\n".join(screen.lines())
    assert "Selected: AAPL, SPY, QQQ, DIA" in rendered
    assert "Skipped: QQQ" in rendered
    assert "Normalized returns:" in rendered
    assert "AAPL" in rendered
    assert "SPY" in rendered


def test_ticker_compare_shows_clear_message_for_one_symbol() -> None:
    state = replace(create_app_state(), selected_symbols=())
    screen = TickerScreen(state, DataSession(FakeProvider()))

    screen.load_symbol("aapl")

    rendered = "\n".join(screen.lines())
    assert "Compare: AAPL vs peers" in rendered
    assert "compare mode requires 2 to 6 symbols, got 1" in rendered
    assert "Normalized returns:" not in rendered


def test_ticker_compare_shows_clear_message_for_more_than_six_symbols() -> None:
    state = replace(
        create_app_state(),
        selected_symbols=("SPY", "QQQ", "DIA", "MSFT", "NVDA", "AMD"),
    )
    screen = TickerScreen(state, DataSession(FakeProvider()))

    screen.load_symbol("aapl")

    rendered = "\n".join(screen.lines())
    assert "Compare: AAPL vs SPY, QQQ, DIA, MSFT, NVDA, AMD" in rendered
    assert "compare mode requires 2 to 6 symbols, got 7" in rendered
    assert "Normalized returns:" not in rendered
