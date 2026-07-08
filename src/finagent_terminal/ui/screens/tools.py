from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, override

from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Input, Static, Tab, Tabs

from finagent_terminal.analytics import (
    MissingCloseColumnError,
    MonteCarloConfig,
    TooFewReturnsError,
    monte_carlo_simulation,
)
from finagent_terminal.data.session import DataSession
from finagent_terminal.data.yfinance_provider import YFinanceProvider
from finagent_terminal.ui.screens import WorkspaceScreen
from finagent_terminal.ui.state import WORKSPACE_LABELS, WORKSPACES

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from finagent_terminal.ui.state import AppState, WorkspaceId


@dataclass(frozen=True, slots=True)
class InvalidMonteCarloInputError(Exception):
    field: str
    value: str

    @override
    def __str__(self) -> str:
        return f"{self.field} must be a positive integer: {self.value}"


class ToolsScreen(WorkspaceScreen):
    workspace: ClassVar[WorkspaceId] = "tools"
    label: ClassVar[str] = "Tools"
    summary: ClassVar[str] = "Analysis hub"

    def __init__(self, state: AppState, session: DataSession | None = None) -> None:
        """Create the Tools workspace with an optional test data session."""
        super().__init__(state)
        self._session: DataSession | None = session
        self._monte_carlo_lines: tuple[str, ...] = ()
        self._message: str | None = None

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
            yield Input(placeholder="Symbol", id="monte_carlo_symbol", value="AAPL")
            yield Input(placeholder="Days", id="monte_carlo_days", value="30")
            yield Input(placeholder="Paths", id="monte_carlo_paths", value="1000")
        with Horizontal():
            yield Input(placeholder="Seed", id="monte_carlo_seed", value="7")
            yield Input(placeholder="Bins", id="monte_carlo_bins", value="4")
            yield Button("Run", id="monte_carlo_run", variant="primary")
        yield Static("\n".join(self.lines()), id="tools_body", markup=False)
        yield Footer()

    @override
    def lines(self) -> tuple[str, ...]:
        lines = [
            self.summary,
            "Monte Carlo: simulate future close-return paths from ticker history.",
            "Inputs: symbol, days, paths, seed, histogram bins.",
        ]
        if self._message is not None:
            lines.append(self._message)
        if len(self._monte_carlo_lines) > 0:
            lines.extend(self._monte_carlo_lines)
        else:
            lines.append("Monte Carlo: waiting for a run")
        return tuple(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "monte_carlo_run":
            return
        _ = event.stop()
        self.run_monte_carlo_from_inputs()

    def run_monte_carlo_from_inputs(self) -> None:
        symbol = self.query_one("#monte_carlo_symbol", Input).value.strip()
        if not symbol:
            self._show_error("symbol is required")
            return

        try:
            config = MonteCarloConfig(
                days=self._positive_int("#monte_carlo_days", "days"),
                paths=self._positive_int("#monte_carlo_paths", "paths"),
                seed=self._optional_int("#monte_carlo_seed", "seed"),
                bins=self._positive_int("#monte_carlo_bins", "bins"),
            )
        except InvalidMonteCarloInputError as error:
            self._show_error(str(error))
            return

        self.run_monte_carlo(symbol, config)

    def run_monte_carlo(self, symbol: str, config: MonteCarloConfig) -> None:
        snapshot = self._data_session().get_or_fetch(symbol, period="1y")
        if snapshot.error is not None:
            self._show_error(snapshot.error.message)
            return
        if snapshot.history is None:
            self._show_error(f"no close history for {snapshot.symbol}")
            return

        try:
            result = monte_carlo_simulation(snapshot.history.frame, config)
        except (MissingCloseColumnError, TooFewReturnsError) as error:
            self._show_error(str(error))
            return

        self._message = None
        self._monte_carlo_lines = (
            "Monte Carlo result:",
            f"Symbol: {snapshot.symbol}",
            result.summary_line,
            f"P10: {result.percentile_10:.2f}",
            f"P50: {result.percentile_50:.2f}",
            f"P90: {result.percentile_90:.2f}",
            "Histogram: " + " | ".join(str(count) for count in result.histogram_counts),
        )
        self._refresh_body()

    def _data_session(self) -> DataSession:
        if self._session is None:
            self._session = DataSession(YFinanceProvider())
        return self._session

    def _positive_int(self, selector: str, field: str) -> int:
        value = self.query_one(selector, Input).value.strip()
        try:
            parsed = int(value)
        except ValueError:
            raise InvalidMonteCarloInputError(field=field, value=value) from None
        if parsed <= 0:
            raise InvalidMonteCarloInputError(field=field, value=value)
        return parsed

    def _optional_int(self, selector: str, field: str) -> int | None:
        value = self.query_one(selector, Input).value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            raise InvalidMonteCarloInputError(field=field, value=value) from None

    def _show_error(self, message: str) -> None:
        self._message = f"Monte Carlo error: {message}"
        self._monte_carlo_lines = ()
        self._refresh_body()

    def _refresh_body(self) -> None:
        try:
            self.query_one("#tools_body", Static).update("\n".join(self.lines()))
        except NoMatches:
            return
