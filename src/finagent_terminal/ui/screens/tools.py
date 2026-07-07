from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, Final, override

from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Input, Static, Tab, Tabs

from finagent_terminal.analytics import MissingCloseColumnError, compute_indicators
from finagent_terminal.compare import (
    MAX_COMPARE_SYMBOLS,
    MIN_COMPARE_SYMBOLS,
    CompareResult,
    CompareSymbolLimitError,
)
from finagent_terminal.compare import compare_symbols as build_compare
from finagent_terminal.data.session import DataSession
from finagent_terminal.symbols import normalize_symbol
from finagent_terminal.ui.screens import WorkspaceScreen
from finagent_terminal.ui.state import WORKSPACE_LABELS, WORKSPACES
from finagent_terminal.ui.widgets.charts import render_frame_chart
from finagent_terminal.ui.widgets.tables import (
    render_performance_table,
    render_tail_table,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd  # noqa: PANDAS_OK
    from textual.app import ComposeResult

    from finagent_terminal.ui.state import AppState, WorkspaceId


SYMBOL_SPLIT: Final = re.compile(r"[\s,]+")


class ToolsScreen(WorkspaceScreen):
    workspace: ClassVar[WorkspaceId] = "tools"
    label: ClassVar[str] = "Tools"
    summary: ClassVar[str] = "Compare 2-6 symbols with normalized returns."

    def __init__(self, state: AppState, session: DataSession | None = None) -> None:
        """Create the tools screen with an optional fake data session."""
        super().__init__(state)
        self._session: DataSession | None = session
        self.selected: tuple[str, ...] = ()
        self.result: CompareResult | None = None
        self.analytics: tuple[str, ...] = ()
        self.skipped: tuple[str, ...] = ()
        self.message: str | None = None

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
        yield Input(placeholder="AAPL MSFT QQQ", id="compare_symbols")
        yield Button("Compare", id="compare_load", variant="primary")
        yield Static("\n".join(self.lines()), id="tools_body")
        yield Footer()

    def compare_manual(self, raw_symbols: str) -> None:
        try:
            symbols = _parse_symbols(raw_symbols, fallback=self.state.selected_symbols)
        except ValueError:
            self._set_error(f"Invalid compare symbols: {raw_symbols.strip()}")
            return
        self.compare_selected(symbols)

    def compare_selected(self, symbols: tuple[str, ...]) -> None:
        self.selected = symbols
        self.result = None
        self.analytics = ()
        self.skipped = ()
        try:
            _require_symbol_limit(symbols)
        except CompareSymbolLimitError as error:
            self._set_error(str(error))
            return

        snapshots = tuple(
            (symbol, self._data_session().get_or_fetch(symbol)) for symbol in symbols
        )
        skipped = [
            symbol
            for symbol, snapshot in snapshots
            if snapshot.error is not None or snapshot.history is None
        ]
        frames = {
            symbol: snapshot.history.frame
            for symbol, snapshot in snapshots
            if snapshot.error is None and snapshot.history is not None
        }

        if len(frames) < MIN_COMPARE_SYMBOLS:
            self.skipped = tuple(skipped)
            self._set_error("compare mode needs at least 2 loaded symbols")
            return

        result = build_compare(frames)
        self.result = result
        self.analytics = _analytics_lines(frames)
        self.skipped = (*skipped, *result.skipped_symbols)
        self.message = None
        self._refresh_body()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "compare_symbols":
            self.compare_manual(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "compare_load":
            return
        self.compare_manual(self.query_one("#compare_symbols", Input).value)

    @override
    def lines(self) -> tuple[str, ...]:
        if self.result is None:
            return (
                self.summary,
                "Compare: enter 2-6 symbols",
                self.message or "Output: normalized returns pending",
            )

        return (
            f"Selected: {', '.join(self.selected)}",
            f"Skipped: {', '.join(self.skipped) if self.skipped else 'none'}",
            "Normalized returns:",
            *render_tail_table(self.result.normalized_returns),
            "Relative performance:",
            *render_performance_table(self.result.relative_performance),
            "Chart:",
            *render_frame_chart(self.result.normalized_returns),
            "Analytics:",
            *self.analytics,
        )

    def _data_session(self) -> DataSession:
        if self._session is None:
            import finagent_terminal.data.yfinance_provider as provider  # noqa: PLC0415

            self._session = DataSession(provider.YFinanceProvider())
        return self._session

    def _set_error(self, message: str) -> None:
        self.result = None
        self.message = message
        self._refresh_body()

    def _refresh_body(self) -> None:
        try:
            self.query_one("#tools_body", Static).update("\n".join(self.lines()))
        except NoMatches:
            return


def _parse_symbols(raw_symbols: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    raw_parts = tuple(part for part in SYMBOL_SPLIT.split(raw_symbols.strip()) if part)
    parts = fallback if len(raw_parts) == 0 else raw_parts
    return tuple(dict.fromkeys(normalize_symbol(part) for part in parts))


def _require_symbol_limit(symbols: tuple[str, ...]) -> None:
    symbol_count = len(symbols)
    if symbol_count < MIN_COMPARE_SYMBOLS or symbol_count > MAX_COMPARE_SYMBOLS:
        raise CompareSymbolLimitError(count=symbol_count)


def _analytics_lines(frames: Mapping[str, pd.DataFrame]) -> tuple[str, ...]:
    rows: list[str] = []
    for symbol, frame in frames.items():
        try:
            indicators = compute_indicators(frame)
        except MissingCloseColumnError:
            continue
        if indicators.empty:
            continue
        latest = indicators.iloc[-1]
        rows.append(
            "; ".join(
                (
                    f"{symbol}: moving_average_20 {latest['moving_average_20']:.2f}",
                    f"moving_average_50 {latest['moving_average_50']:.2f}",
                    f"volatility_20 {latest['volatility_20']:.2f}",
                    f"drawdown {latest['drawdown']:.2%}",
                    f"rsi_14 {latest['rsi_14']:.2f}",
                    f"macd {latest['macd']:.2f}",
                    f"macd_signal {latest['macd_signal']:.2f}",
                ),
            ),
        )
    return tuple(rows) if rows else ("(no data)",)
