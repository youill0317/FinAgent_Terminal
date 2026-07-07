from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

import pandas as pd  # noqa: RUF100  # noqa: PANDAS_OK
import pytest
from pydantic import TypeAdapter
from textual.app import App
from textual.widgets import Input, Static

from finagent_terminal.config import FinAgentConfig, load_config
from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    OhlcvFrame,
    Quote,
)
from finagent_terminal.symbols import SymbolGroup
from finagent_terminal.ui.screens.markets import MarketsScreen
from finagent_terminal.ui.state import create_app_state

if TYPE_CHECKING:
    from pathlib import Path


class SavedGroup(TypedDict):
    name: str


class SavedConfig(TypedDict):
    groups: list[SavedGroup]


SAVED_CONFIG_ADAPTER: TypeAdapter[SavedConfig] = TypeAdapter(SavedConfig)


class MarketsHarness(App[None]):
    def __init__(self, screen: MarketsScreen) -> None:
        super().__init__()
        self.markets_screen: MarketsScreen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.markets_screen)


@dataclass(frozen=True, slots=True)
class FakeSession:
    snapshots: dict[str, MarketDataSnapshot]
    fetched: list[str]

    def get_or_fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        assert period == "1mo"
        assert interval == "1d"
        self.fetched.append(symbol)
        return self.snapshots[symbol]

    def freshness(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> FreshnessState:
        assert period == "1mo"
        assert interval == "1d"
        snapshot = self.snapshots.get(symbol)
        if snapshot is None:
            return FreshnessState.NOT_LOADED
        return snapshot.freshness


@pytest.mark.anyio
async def test_default_presets_render_without_fetching(tmp_path: Path) -> None:
    # Given: a first-run Markets screen with no saved config.
    session = FakeSession(snapshots={}, fetched=[])
    screen = MarketsScreen(
        create_app_state(),
        config=FinAgentConfig(),
        config_path=tmp_path / "config.json",
        session=session,
    )

    # When: Markets lines are rendered.
    async with MarketsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rendered = str(screen.query_one("#markets_body", Static).render())

    # Then: default preset groups are visible and no data was fetched.
    assert "Major Indices" in rendered
    assert "Mega Cap Tech" in rendered
    assert "SPY" in rendered
    assert "[not_loaded]" in rendered
    assert session.fetched == []


@pytest.mark.anyio
async def test_custom_watchlist_persists_only_config(tmp_path: Path) -> None:
    # Given: an editable Markets screen mounted in Textual.
    config_path = tmp_path / "config.json"
    session = FakeSession(snapshots={}, fetched=[])
    screen = MarketsScreen(
        create_app_state(),
        config=FinAgentConfig(),
        config_path=config_path,
        session=session,
    )

    # When: a user creates and edits a watchlist.
    async with MarketsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        set_market_inputs(screen, "Watchlist", "AAPL")
        _ = await pilot.click("#market_add_group")
        await pilot.pause()
        set_market_inputs(screen, "Watchlist", "nvda")
        _ = await pilot.click("#market_add_symbol")
        await pilot.pause()
        _ = await pilot.click("#market_symbol_up")
        await pilot.pause()
        set_market_inputs(screen, "Watchlist", "AAPL")
        _ = await pilot.click("#market_remove_symbol")
        await pilot.pause()
        rendered = str(screen.query_one("#markets_body", Static).render())

    # Then: only config JSON is written, with normalized symbols.
    saved = load_config(config_path)
    watchlist = next(group for group in saved.groups if group.name == "Watchlist")
    assert tuple(entry.symbol for entry in watchlist.symbols) == ("NVDA",)
    assert "Watchlist" in rendered
    assert list_files(tmp_path) == (config_path,)


@pytest.mark.anyio
async def test_group_reorder_and_remove_persist_config(tmp_path: Path) -> None:
    # Given: custom groups in the Markets editor.
    config_path = tmp_path / "config.json"
    config = FinAgentConfig(
        groups=(
            SymbolGroup.model_validate({"name": "One", "symbols": ["AAPL"]}),
            SymbolGroup.model_validate({"name": "Two", "symbols": ["MSFT"]}),
        ),
    )
    screen = MarketsScreen(
        create_app_state(),
        config=config,
        config_path=config_path,
        session=FakeSession(snapshots={}, fetched=[]),
    )

    # When: a group is moved and another is removed.
    async with MarketsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        set_market_inputs(screen, "Two")
        _ = await pilot.click("#market_group_up")
        await pilot.pause()
        set_market_inputs(screen, "One")
        _ = await pilot.click("#market_remove_group")
        await pilot.pause()

    # Then: persisted config reflects only symbol/group edits.
    raw_config = SAVED_CONFIG_ADAPTER.validate_python(
        json.loads(config_path.read_text(encoding="utf-8")),
    )
    assert tuple(group["name"] for group in raw_config["groups"]) == ("Two",)
    assert list_files(tmp_path) == (config_path,)


@pytest.mark.anyio
async def test_refresh_updates_quotes_charts_and_freshness(tmp_path: Path) -> None:
    # Given: Markets has one symbol with fake market data.
    snapshot = make_snapshot("NVDA", FreshnessState.FRESH)
    session = FakeSession(snapshots={"NVDA": snapshot}, fetched=[])
    config = FinAgentConfig(
        groups=(
            SymbolGroup.model_validate(
                {"name": "Semiconductors", "symbols": ["NVDA"]},
            ),
        ),
    )
    screen = MarketsScreen(
        create_app_state(),
        config=config,
        config_path=tmp_path / "config.json",
        session=session,
    )

    # When: the user refreshes market data.
    async with MarketsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.press("r")
        await pilot.pause()
        rendered = str(screen.query_one("#markets_body", Static).render())

    # Then: the row shows quote, chart, and freshness.
    assert session.fetched == ["NVDA"]
    assert "NVDA" in rendered
    assert "125.00" in rendered
    assert "+5.00" in rendered
    assert "[fresh]" in rendered
    assert "chart: .@" in rendered
    assert "chart: --" not in rendered


def make_snapshot(symbol: str, freshness: FreshnessState) -> MarketDataSnapshot:
    fetched_at = datetime(2024, 1, 3, tzinfo=UTC)
    frame = pd.DataFrame(
        {"Close": [100.0, 110.0, 125.0], "Volume": [1, 2, 3]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    return MarketDataSnapshot(
        symbol=symbol,
        period="1mo",
        interval="1d",
        fetched_at=fetched_at,
        freshness=freshness,
        quote=Quote(
            symbol=symbol,
            last_price=125.0,
            previous_close=120.0,
            change=5.0,
            change_percent=4.1667,
            currency="USD",
            market_cap=None,
            volume=3,
            day_high=None,
            day_low=None,
            fetched_at=fetched_at,
        ),
        history=OhlcvFrame(
            symbol=symbol,
            period="1mo",
            interval="1d",
            frame=frame,
            rows=len(frame),
            fetched_at=fetched_at,
        ),
    )


def list_files(path: Path) -> tuple[Path, ...]:
    return tuple(sorted(child for child in path.rglob("*") if child.is_file()))


def set_market_inputs(screen: MarketsScreen, group: str, symbol: str = "") -> None:
    screen.query_one("#market_group_input", Input).value = group
    screen.query_one("#market_symbol_input", Input).value = symbol
