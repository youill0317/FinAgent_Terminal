from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd  # noqa: PANDAS_OK
import pytest
from textual.app import App
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Static

from finagent_terminal.data.models import FreshnessState, MarketDataSnapshot, OhlcvFrame
from finagent_terminal.data.session import DataSession
from finagent_terminal.ui.screens.tools import ToolsScreen
from finagent_terminal.ui.state import create_app_state

FETCHED_AT = datetime(2024, 1, 5, 12, tzinfo=UTC)


class ToolsHarness(App[None]):
    def __init__(self, screen: ToolsScreen) -> None:
        super().__init__()
        self.tools_screen: ToolsScreen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.tools_screen)


@dataclass(frozen=True, slots=True)
class ToolsProvider:
    frame: pd.DataFrame

    def fetch(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
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
                frame=self.frame,
                rows=len(self.frame),
                fetched_at=FETCHED_AT,
            ),
        )


@pytest.mark.anyio
async def test_tools_no_compare_ui() -> None:
    screen = ToolsScreen(create_app_state())

    async with ToolsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rendered = str(screen.query_one("#tools_body", Static).render())

        assert "Compare: enter 2-6 symbols" not in rendered
        assert "normalized returns" not in rendered.lower()
        with pytest.raises(NoMatches):
            _ = screen.query_one("#compare_symbols", Input)


@pytest.mark.anyio
async def test_tools_monte_carlo_success_renders_hub_inputs_and_result() -> None:
    frame = _frame({"Close": [100.0, 102.0, 99.0, 105.0, 108.0, 106.0]})
    screen = ToolsScreen(create_app_state(), DataSession(ToolsProvider(frame)))

    async with ToolsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen.query_one("#monte_carlo_symbol", Input).value = "aapl"
        screen.query_one("#monte_carlo_days", Input).value = "5"
        screen.query_one("#monte_carlo_paths", Input).value = "20"
        screen.query_one("#monte_carlo_seed", Input).value = "7"
        _ = screen.query_one("#monte_carlo_run", Button).press()
        await pilot.pause()

        rendered = str(screen.query_one("#tools_body", Static).render())
        assert "Analysis hub" in rendered
        assert "Monte Carlo" in rendered
        assert "Symbol: AAPL" in rendered
        assert "P10: 98.15" in rendered
        assert "P50: 104.40" in rendered
        assert "P90: 116.80" in rendered
        assert "Histogram: 3 | 8 | 6 | 3" in rendered


@pytest.mark.anyio
async def test_tools_monte_carlo_missing_close_error_has_no_traceback() -> None:
    screen = ToolsScreen(
        create_app_state(),
        DataSession(ToolsProvider(_frame({"Open": [100.0, 101.0, 102.0]}))),
    )

    async with ToolsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen.query_one("#monte_carlo_symbol", Input).value = "AAPL"
        screen.query_one("#monte_carlo_days", Input).value = "5"
        screen.query_one("#monte_carlo_paths", Input).value = "20"
        screen.query_one("#monte_carlo_seed", Input).value = "7"
        _ = screen.query_one("#monte_carlo_run", Button).press()
        await pilot.pause()

        rendered = str(screen.query_one("#tools_body", Static).render())
        assert "Monte Carlo error: missing close column: Close" in rendered
        assert "Traceback" not in rendered


@pytest.mark.anyio
async def test_tools_monte_carlo_too_few_returns_error_has_no_traceback() -> None:
    screen = ToolsScreen(
        create_app_state(),
        DataSession(ToolsProvider(_frame({"Close": [100.0, None, 100.0]}))),
    )

    async with ToolsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen.query_one("#monte_carlo_symbol", Input).value = "AAPL"
        screen.query_one("#monte_carlo_days", Input).value = "5"
        screen.query_one("#monte_carlo_paths", Input).value = "20"
        screen.query_one("#monte_carlo_seed", Input).value = "7"
        _ = screen.query_one("#monte_carlo_run", Button).press()
        await pilot.pause()

        rendered = str(screen.query_one("#tools_body", Static).render())
        assert "Monte Carlo error: need at least 2 returns; got 1" in rendered
        assert "Traceback" not in rendered


def _frame(columns: dict[str, list[float | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        columns,
        index=pd.date_range("2024-01-01", periods=len(next(iter(columns.values())))),
    )
