from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from textual.app import App
from textual.widgets import Static

from finagent_terminal.config import FinAgentConfig
from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    ProviderError,
)
from finagent_terminal.symbols import SymbolGroup
from finagent_terminal.ui.screens.markets import MarketsScreen
from finagent_terminal.ui.state import create_app_state
from finagent_terminal.ui.widgets.freshness import freshness_badge

if TYPE_CHECKING:
    from pathlib import Path


class MarketsHarness(App[None]):
    def __init__(self, screen: MarketsScreen) -> None:
        super().__init__()
        self.markets_screen: MarketsScreen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.markets_screen)


@dataclass(frozen=True, slots=True)
class FailingSession:
    def get_or_fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        assert symbol == "BAD"
        return MarketDataSnapshot(
            symbol=symbol,
            period=period,
            interval=interval,
            fetched_at=datetime(2024, 1, 3, tzinfo=UTC),
            freshness=FreshnessState.FAILED,
            error=ProviderError(
                symbol=symbol,
                message="provider unavailable",
                error_type="ProviderOfflineError",
            ),
        )

    def freshness(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> FreshnessState:
        assert period == "1mo"
        assert interval == "1d"
        if symbol == "BAD":
            return FreshnessState.FAILED
        return FreshnessState.NOT_LOADED


def test_freshness_badges_cover_all_states() -> None:
    # Given: every session freshness state.
    states = tuple(FreshnessState)

    # When: badges are rendered.
    badges = tuple(freshness_badge(state) for state in states)

    # Then: each visible badge is stable.
    assert badges == ("[not_loaded]", "[fresh]", "[stale]", "[failed]")


@pytest.mark.anyio
async def test_failed_fetch_visible(tmp_path: Path) -> None:
    # Given: a Markets row whose provider fetch fails.
    screen = MarketsScreen(
        create_app_state(),
        config=FinAgentConfig(
            groups=(
                SymbolGroup.model_validate({"name": "Broken", "symbols": ["BAD"]}),
            ),
        ),
        config_path=tmp_path / "config.json",
        session=FailingSession(),
    )

    # When: the user refreshes data.
    async with MarketsHarness(screen).run_test(size=(120, 40)) as pilot:
        _ = await pilot.click("#market_refresh")
        await pilot.pause()
        rendered = str(screen.query_one("#markets_body", Static).render())

    # Then: the row remains visible with failed state and error text.
    assert "BAD" in rendered
    assert "[failed]" in rendered
    assert "provider unavailable" in rendered
    assert list_files(tmp_path) == ()


def list_files(path: Path) -> tuple[Path, ...]:
    return tuple(sorted(child for child in path.rglob("*") if child.is_file()))
