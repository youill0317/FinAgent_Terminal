from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, override

from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Input, Static, Tab, Tabs

from finagent_terminal.data.source_routing import (
    SourceRoute,
    SourceRouteBundle,
    route_source_bundle,
)
from finagent_terminal.data.sources import SourceSession
from finagent_terminal.symbols import InvalidSymbolError, normalize_symbol
from finagent_terminal.ui.screens import WorkspaceScreen
from finagent_terminal.ui.state import WORKSPACE_LABELS, WORKSPACES

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from finagent_terminal.data.source_models import SourceItem
    from finagent_terminal.ui.state import AppState, WorkspaceId

NEWS_ROUTE_SECTIONS: Final[tuple[tuple[str, SourceRoute], ...]] = (
    ("Market news", SourceRoute.MARKET_NEWS),
    ("Ticker news", SourceRoute.TICKER_NEWS),
    ("Macro/events", SourceRoute.MACRO_EVENTS),
    ("Filings", SourceRoute.FILINGS),
)
DEFAULT_NEWS_TICKER: Final = "AAPL"


class NewsScreen(WorkspaceScreen):
    workspace: ClassVar[WorkspaceId] = "news"
    label: ClassVar[str] = "News"
    summary: ClassVar[str] = "Browse market, ticker, macro/event, and filing sources."

    def __init__(
        self,
        state: AppState,
        source_session: SourceSession | None = None,
    ) -> None:
        """Create the News workspace with an optional fake source session."""
        super().__init__(state)
        self._source_session: SourceSession | None = source_session
        self._owns_source_session: bool = source_session is None
        self._ticker: str = DEFAULT_NEWS_TICKER
        self._routes: tuple[SourceRouteBundle, ...] = ()
        self._warning: str | None = None

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
        yield Input(value=self._ticker, placeholder="Ticker", id="news_ticker")
        yield Button("Refresh", id="news_refresh", variant="primary")
        yield Static("\n".join(self.lines()), id="news_body")
        yield Footer()

    def on_unmount(self) -> None:
        if self._owns_source_session and self._source_session is not None:
            self._source_session.close()
            self._source_session = None

    def refresh_news(self, raw_ticker: str | None = None) -> None:
        ticker = self._input_ticker(raw_ticker)
        self._ticker = ticker
        try:
            ticker_key = normalize_symbol(ticker)
        except InvalidSymbolError:
            self._routes = ()
            self._warning = f"Invalid ticker: {self._display_ticker()}"
            self._refresh_body()
            return
        session = self._session()
        session.clear()
        self._warning = None
        self._routes = (
            route_source_bundle(session, SourceRoute.MARKET_NEWS),
            route_source_bundle(session, SourceRoute.TICKER_NEWS, ticker_key),
            route_source_bundle(session, SourceRoute.MACRO_EVENTS, ticker_key),
            route_source_bundle(session, SourceRoute.FILINGS, ticker_key),
        )
        self._refresh_body()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "news_ticker":
            self.refresh_news(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "news_refresh":
            self.refresh_news()

    @override
    def lines(self) -> tuple[str, ...]:
        if len(self._routes) == 0:
            return (
                self.summary,
                f"Ticker/query: {self._display_ticker()}",
                *(() if self._warning is None else (self._warning,)),
                "Press Refresh to load source-backed news.",
            )
        lines = [self.summary, f"Ticker/query: {self._display_ticker()}"]
        for (title, _route), bundle in zip(
            NEWS_ROUTE_SECTIONS,
            self._routes,
            strict=True,
        ):
            lines.extend(_section_lines(title, bundle))
        return tuple(lines)

    def _session(self) -> SourceSession:
        if self._source_session is None:
            self._source_session = SourceSession()
        return self._source_session

    def _input_ticker(self, raw_ticker: str | None) -> str:
        if raw_ticker is not None:
            return raw_ticker.strip().upper()
        try:
            return self.query_one("#news_ticker", Input).value.strip().upper()
        except NoMatches:
            return self._ticker

    def _display_ticker(self) -> str:
        return self._ticker or "(empty)"

    def _refresh_body(self) -> None:
        try:
            self.query_one("#news_body", Static).update("\n".join(self.lines()))
        except NoMatches:
            return


def _section_lines(title: str, route: SourceRouteBundle) -> tuple[str, ...]:
    lines = [title, f"Sources: {_sources_text(route.source_labels)}"]
    if len(route.warnings) > 0:
        lines.append(f"Warnings: {'; '.join(route.warnings)}")
    if len(route.citations) == 0:
        lines.append("Items: none")
    else:
        lines.extend(_item_line(item) for item in route.citations)
    return tuple(lines)


def _sources_text(labels: tuple[str, ...]) -> str:
    return ", ".join(labels) if len(labels) > 0 else "none"


def _item_line(item: SourceItem) -> str:
    details: list[str] = []
    if item.published_at is not None:
        details.append(item.published_at.strftime("%Y-%m-%d %H:%M %Z"))
    if item.url is not None:
        details.append(item.url)
    metadata = f" ({'; '.join(details)})" if len(details) > 0 else ""
    summary = f": {item.summary}" if item.summary is not None else ""
    return f"- {_source_label(item.source)}: {item.title}{metadata}{summary}"


def _source_label(source: str) -> str:
    if source == "SEC EDGAR":
        return "SEC"
    return source
