from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

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
from tests.test_yfinance_provider import spy_frame

FETCHED_AT = datetime(2024, 1, 5, 12, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class FakeProvider:
    def fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        frame = spy_frame()
        return MarketDataSnapshot(
            symbol=symbol,
            period=period,
            interval=interval,
            fetched_at=FETCHED_AT,
            freshness=FreshnessState.FRESH,
            quote=Quote(
                symbol=symbol,
                last_price=501.25,
                previous_close=500.0,
                change=1.25,
                change_percent=0.0025,
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
