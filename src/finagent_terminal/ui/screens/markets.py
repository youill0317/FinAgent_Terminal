from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, Protocol, override

from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Input, Static, Tab, Tabs

from finagent_terminal.config import (
    FinAgentConfig,
    load_config,
    save_config,
)
from finagent_terminal.config import config_path as default_config_path
from finagent_terminal.data.models import FreshnessState, MarketDataSnapshot
from finagent_terminal.data.session import DataSession
from finagent_terminal.ui.screens import WorkspaceScreen
from finagent_terminal.ui.state import WORKSPACE_LABELS, WORKSPACES
from finagent_terminal.ui.widgets import market_editor
from finagent_terminal.ui.widgets.market_rows import market_row

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from textual.app import ComposeResult
    from textual.binding import BindingType

    from finagent_terminal.symbols import SymbolEntry
    from finagent_terminal.ui.state import AppState, WorkspaceId

DEFAULT_GROUP_NAME: Final = "Watchlist"
DEFAULT_SYMBOL: Final = "SPY"


class MarketSession(Protocol):
    def get_or_fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot: ...

    def freshness(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> FreshnessState: ...


class MarketsScreen(WorkspaceScreen):
    workspace: ClassVar[WorkspaceId] = "markets"
    label: ClassVar[str] = "Markets"
    summary: ClassVar[str] = "Preset groups and watchlists"
    BINDINGS: ClassVar[list[BindingType]] = [
        ("r", "refresh", "Refresh"),
        ("a", "add_symbol_from_inputs", "Add Symbol"),
        ("delete", "remove_symbol_from_inputs", "Remove Symbol"),
    ]

    def __init__(
        self,
        state: AppState,
        *,
        config: FinAgentConfig | None = None,
        config_path: Path | None = None,
        session: MarketSession | None = None,
    ) -> None:
        """Create the Markets workspace with optional test doubles."""
        super().__init__(state)
        self._config_path: Path = (
            default_config_path() if config_path is None else config_path
        )
        self._config: FinAgentConfig = (
            load_config(self._config_path) if config is None else config
        )
        self._session: MarketSession | None = session
        self._snapshots: dict[str, MarketDataSnapshot] = {}

    @override
    def compose(self) -> ComposeResult:
        yield Static("FinAgent Terminal", classes="app-title")
        yield Tabs(
            *(
                Tab(label, id=workspace)
                for workspace, label in zip(WORKSPACES, WORKSPACE_LABELS, strict=True)
            ),
            active=self.workspace,
            id="workspace-tabs",
        )
        yield Static(self.label, classes="workspace-title")
        with Horizontal():
            yield Input(placeholder="Group", id="market_group_input")
            yield Input(placeholder="Symbol", id="market_symbol_input")
        with Horizontal():
            yield Button("Refresh", id="market_refresh")
            yield Button("Add G", id="market_add_group")
            yield Button("Del G", id="market_remove_group")
            yield Button("G Up", id="market_group_up")
            yield Button("G Down", id="market_group_down")
        with Horizontal():
            yield Button("Add S", id="market_add_symbol")
            yield Button("Del S", id="market_remove_symbol")
            yield Button("S Up", id="market_symbol_up")
            yield Button("S Down", id="market_symbol_down")
        yield Static("\n".join(self.lines()), id="markets_body", markup=False)
        yield Footer()

    @override
    def lines(self) -> tuple[str, ...]:
        data_line = (
            "Data: not loaded"
            if len(self._snapshots) == 0
            else f"Data: {len(self._snapshots)} symbols loaded"
        )
        lines = [
            self.summary,
            f"Selected: {', '.join(self.state.selected_symbols)}",
            data_line,
        ]
        for group in self._config.groups:
            lines.append(group.name)
            lines.extend(self._row_for_symbol(entry) for entry in group.symbols)
        return tuple(lines)

    async def action_refresh(self) -> None:
        self.refresh_market_data()

    def action_add_symbol_from_inputs(self) -> None:
        self.add_symbol(self._group_input(), self._symbol_input())

    def action_remove_symbol_from_inputs(self) -> None:
        self.remove_symbol(self._group_input(), self._symbol_input())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        _ = event.stop()
        button_id = event.button.id
        if button_id is None:
            return
        action = self._button_actions().get(button_id)
        if action is not None:
            action()

    def _button_actions(self) -> Mapping[str, Callable[[], None]]:
        return {
            "market_refresh": self.refresh_market_data,
            "market_add_group": lambda: self.add_group(
                self._group_input(),
                self._symbol_input(default=DEFAULT_SYMBOL),
            ),
            "market_remove_group": lambda: self.remove_group(self._group_input()),
            "market_group_up": lambda: self.move_group(self._group_input(), -1),
            "market_group_down": lambda: self.move_group(self._group_input(), 1),
            "market_add_symbol": self.action_add_symbol_from_inputs,
            "market_remove_symbol": self.action_remove_symbol_from_inputs,
            "market_symbol_up": lambda: self.move_symbol(
                self._group_input(),
                self._symbol_input(),
                -1,
            ),
            "market_symbol_down": lambda: self.move_symbol(
                self._group_input(),
                self._symbol_input(),
                1,
            ),
        }

    def refresh_market_data(self) -> None:
        session = self._market_session()
        for symbol in self._symbols():
            self._snapshots[symbol] = session.get_or_fetch(symbol)
        self._update_body()

    def add_group(self, name: str, symbol: str = "SPY") -> None:
        self._replace_config(market_editor.add_group(self._config, name, symbol))

    def remove_group(self, name: str) -> None:
        self._replace_config(market_editor.remove_group(self._config, name))

    def move_group(self, name: str, direction: int) -> None:
        self._replace_config(market_editor.move_group(self._config, name, direction))

    def add_symbol(self, group_name: str, symbol: str) -> None:
        self._replace_config(market_editor.add_symbol(self._config, group_name, symbol))

    def remove_symbol(self, group_name: str, symbol: str) -> None:
        self._replace_config(
            market_editor.remove_symbol(self._config, group_name, symbol),
        )

    def move_symbol(self, group_name: str, symbol: str, direction: int) -> None:
        self._replace_config(
            market_editor.move_symbol(self._config, group_name, symbol, direction),
        )

    def _replace_config(self, config: FinAgentConfig) -> None:
        self._config = config
        _ = save_config(self._config, self._config_path)
        self._update_body()

    def _update_body(self) -> None:
        try:
            body = self.query_one("#markets_body", Static)
        except NoMatches:
            return
        body.update("\n".join(self.lines()))

    def _market_session(self) -> MarketSession:
        if self._session is None:
            from finagent_terminal.data.yfinance_provider import YFinanceProvider  # noqa: I001, PLC0415

            self._session = DataSession(YFinanceProvider())
        return self._session

    def _symbols(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                entry.symbol for group in self._config.groups for entry in group.symbols
            ),
        )

    def _row_for_symbol(self, entry: SymbolEntry) -> str:
        snapshot = self._snapshots.get(entry.symbol)
        return market_row(
            entry.symbol,
            snapshot,
            self._freshness(entry.symbol, snapshot),
        )

    def _freshness(
        self,
        symbol: str,
        snapshot: MarketDataSnapshot | None,
    ) -> FreshnessState:
        if snapshot is not None:
            return snapshot.freshness
        if self._session is None:
            return FreshnessState.NOT_LOADED
        return self._session.freshness(symbol)

    def _group_input(self) -> str:
        value = self.query_one("#market_group_input", Input).value.strip()
        if value:
            return value
        if self._config.groups:
            return self._config.groups[0].name
        return DEFAULT_GROUP_NAME

    def _symbol_input(self, default: str | None = None) -> str:
        value = self.query_one("#market_symbol_input", Input).value.strip()
        if value:
            return value
        return DEFAULT_SYMBOL if default is None else default
