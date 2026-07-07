from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Input, Static, Tab, Tabs

from finagent_terminal.analytics import MissingCloseColumnError, compute_indicators
from finagent_terminal.data.session import DataSession
from finagent_terminal.ui.screens import WorkspaceScreen
from finagent_terminal.ui.state import WORKSPACE_LABELS, WORKSPACES
from finagent_terminal.ui.widgets.charts import render_series_chart
from finagent_terminal.ui.widgets.tables import render_key_value_rows, render_tail_table

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from finagent_terminal.data.models import MarketDataSnapshot
    from finagent_terminal.ui.state import AppState, WorkspaceId


class TickerScreen(WorkspaceScreen):
    workspace: ClassVar[WorkspaceId] = "ticker"
    label: ClassVar[str] = "Ticker"
    summary: ClassVar[str] = "Symbol detail workspace is ready."

    def __init__(self, state: AppState, session: DataSession | None = None) -> None:
        """Create the ticker screen with an optional fake data session."""
        super().__init__(state)
        self._session: DataSession | None = session
        self.snapshot: MarketDataSnapshot | None = None
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
        yield Input(placeholder="Symbol", id="ticker_symbol")
        yield Button("Load", id="ticker_load", variant="primary")
        yield Static("\n".join(self.lines()), id="ticker_body")
        yield Footer()

    def load_symbol(self, raw_symbol: str) -> None:
        try:
            snapshot = self._data_session().get_or_fetch(raw_symbol)
        except ValueError:
            self.snapshot = None
            self.message = f"Invalid symbol: {raw_symbol.strip()}"
            self._refresh_body()
            return

        self.snapshot = snapshot
        self.message = None
        self._refresh_body()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "ticker_symbol":
            self.load_symbol(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "ticker_load":
            return
        self.load_symbol(self.query_one("#ticker_symbol", Input).value)

    @override
    def lines(self) -> tuple[str, ...]:
        if self.message is not None:
            return (self.summary, self.message, "Metrics: not loaded")
        if self.snapshot is None:
            return (
                self.summary,
                "Chart: waiting for a symbol request",
                "Metrics: not loaded",
            )
        if self.snapshot.error is not None:
            return (
                f"Symbol: {self.snapshot.symbol}",
                f"Freshness: {self.snapshot.freshness.value}",
                f"Error: {self.snapshot.error.message}",
                "Metrics: not loaded",
            )

        return (
            f"Symbol: {self.snapshot.symbol}",
            f"Freshness: {self.snapshot.freshness.value}",
            f"Fetched: {self.snapshot.fetched_at:%Y-%m-%d %H:%M %Z}",
            "Price chart:",
            self._price_chart(),
            "Metrics:",
            *render_key_value_rows(self._metric_rows()),
            "Indicators:",
            *self._indicator_lines(),
        )

    def _data_session(self) -> DataSession:
        if self._session is None:
            import finagent_terminal.data.yfinance_provider as provider  # noqa: PLC0415

            self._session = DataSession(provider.YFinanceProvider())
        return self._session

    def _refresh_body(self) -> None:
        try:
            self.query_one("#ticker_body", Static).update("\n".join(self.lines()))
        except NoMatches:
            return

    def _price_chart(self) -> str:
        if self.snapshot is None or self.snapshot.history is None:
            return "(no data)"
        return render_series_chart(self.snapshot.history.frame["Close"])

    def _metric_rows(self) -> tuple[tuple[str, str], ...]:
        if self.snapshot is None or self.snapshot.quote is None:
            return (("Quote", "not loaded"),)
        quote = self.snapshot.quote
        profile = self.snapshot.profile
        return (
            ("Last", _format_money(quote.last_price, quote.currency)),
            ("Change", _format_percent(quote.change_percent)),
            ("Volume", _format_int(quote.volume)),
            ("Market Cap", _format_int(quote.market_cap)),
            ("Day Range", _format_range(quote.day_low, quote.day_high)),
            ("Name", _format_text(profile.name if profile else None)),
            ("Sector", _format_text(profile.sector if profile else None)),
        )

    def _indicator_lines(self) -> tuple[str, ...]:
        if self.snapshot is None or self.snapshot.history is None:
            return ("(no data)",)
        try:
            indicators = compute_indicators(self.snapshot.history.frame)
        except MissingCloseColumnError:
            return ("(missing close data)",)
        if indicators.empty:
            return ("(no data)",)
        latest = indicators.tail(1).T.rename(columns={indicators.index[-1]: "latest"})
        return render_tail_table(latest, rows=len(latest))


def _format_money(value: float | None, currency: str | None) -> str:
    if value is None:
        return "n/a"
    suffix = f" {currency}" if currency else ""
    return f"{value:.2f}{suffix}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _format_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def _format_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "n/a"
    return f"{low:.2f} - {high:.2f}"


def _format_text(value: str | None) -> str:
    return value or "n/a"
