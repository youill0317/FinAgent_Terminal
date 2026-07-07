from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import override

from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    OhlcvFrame,
)
from finagent_terminal.data.session import DataSession
from finagent_terminal.ui.screens.tools import ToolsScreen
from finagent_terminal.ui.state import create_app_state
from tests.test_compare import fixture_frame

FETCHED_AT = datetime(2024, 1, 5, 12, tzinfo=UTC)
PRICES = {
    "AAPL": [100.0, 110.0, 121.0],
    "MSFT": [200.0, 210.0, 220.0],
    "QQQ": [50.0, 55.0, 60.0],
}


@dataclass(frozen=True, slots=True)
class SymbolUnavailableError(Exception):
    symbol: str

    @override
    def __str__(self) -> str:
        return f"{self.symbol} unavailable"


@dataclass(frozen=True, slots=True)
class CompareProvider:
    failed_symbol: str | None = None

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
        return MarketDataSnapshot(
            symbol=symbol,
            period=period,
            interval=interval,
            fetched_at=FETCHED_AT,
            freshness=FreshnessState.FRESH,
            history=OhlcvFrame(
                symbol=symbol,
                period=period,
                interval=interval,
                frame=frame,
                rows=len(frame),
                fetched_at=FETCHED_AT,
            ),
        )


def test_tools_compare_renders_normalized_returns_from_manual_symbols() -> None:
    screen = ToolsScreen(create_app_state(), DataSession(CompareProvider()))

    screen.compare_manual("aapl, msft qqq")

    rendered = "\n".join(screen.lines())
    assert "Selected: AAPL, MSFT, QQQ" in rendered
    assert "Normalized returns:" in rendered
    assert "AAPL" in rendered
    assert "121.00" in rendered
    assert "QQQ" in rendered
    assert "20.00%" in rendered
    assert "Analytics:" in rendered
    assert "moving_average_20" in rendered
    assert "moving_average_50" in rendered
    assert "volatility_20" in rendered
    assert "drawdown" in rendered
    assert "rsi_14" in rendered
    assert "macd" in rendered
    assert "macd_signal" in rendered


def test_tools_compare_shows_clear_message_for_more_than_six_symbols() -> None:
    screen = ToolsScreen(create_app_state(), DataSession(CompareProvider()))

    screen.compare_manual("AAPL MSFT QQQ DIA SPY NVDA AMD")

    rendered = "\n".join(screen.lines())
    assert "compare mode requires 2 to 6 symbols" in rendered
    assert "Normalized returns:" not in rendered


def test_tools_compare_keeps_output_when_one_symbol_fails() -> None:
    screen = ToolsScreen(create_app_state(), DataSession(CompareProvider("QQQ")))

    screen.compare_manual("AAPL MSFT QQQ")

    rendered = "\n".join(screen.lines())
    assert "Selected: AAPL, MSFT, QQQ" in rendered
    assert "Skipped: QQQ" in rendered
    assert "AAPL" in rendered
    assert "MSFT" in rendered
    assert "QQQ: moving_average_20" not in rendered
