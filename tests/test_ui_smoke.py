from __future__ import annotations

import sys

import httpx
import pytest
from textual.widgets import Static, Tab

from finagent_terminal.ui.app import FinAgentApp
from finagent_terminal.ui.screens.markets import MarketsScreen
from finagent_terminal.ui.screens.news import NewsScreen
from finagent_terminal.ui.screens.reports import ReportsScreen
from finagent_terminal.ui.screens.settings import SettingsScreen
from finagent_terminal.ui.screens.ticker import TickerScreen
from finagent_terminal.ui.screens.tools import ToolsScreen

WORKSPACE_SCREEN_TYPES = (
    MarketsScreen,
    TickerScreen,
    NewsScreen,
    ToolsScreen,
    ReportsScreen,
    SettingsScreen,
)
EXPECTED_WORKSPACE_TAB_IDS = (
    "markets",
    "ticker",
    "news",
    "tools",
    "reports",
    "settings",
)


@pytest.mark.anyio
async def test_workspace_navigation() -> None:
    # Given: the Textual shell starts with fake integrations enabled by the test env.
    app = FinAgentApp()

    # When: a user navigates across every workspace with key bindings.
    async with app.run_test() as pilot:
        await pilot.press("1")
        await pilot.pause()
        assert isinstance(app.screen, MarketsScreen)
        markets_body = str(app.screen.query_one("#markets_body", Static).render())
        assert "Data: not loaded" in markets_body

        await pilot.press("2")
        await pilot.pause()
        assert isinstance(app.screen, TickerScreen)
        ticker_body = str(app.screen.query_one("#ticker_body", Static).render())
        assert "Metrics: not loaded" in ticker_body

        await pilot.press("3")
        await pilot.pause()
        assert isinstance(app.screen, NewsScreen)
        news_body = str(app.screen.query_one("#news_body", Static).render())
        assert "Browse market, ticker, macro/event, and filing sources." in news_body

        await pilot.press("4")
        await pilot.pause()
        assert isinstance(app.screen, ToolsScreen)
        tools_body = str(app.screen.query_one("#tools_body", Static).render())
        assert "Compare: enter 2-6 symbols" not in tools_body
        assert "Normalized returns:" not in tools_body

        await pilot.press("5")
        await pilot.pause()
        assert isinstance(app.screen, ReportsScreen)
        reports_body = str(app.screen.query_one("#reports_body", Static).render())
        assert "Progress: 0/0 (not_started)" in reports_body

        await pilot.press("6")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        settings_body = str(app.screen.query_one("#settings_body", Static).render())
        assert "Status: defaults" in settings_body
        assert len(workspace_screen_names(app)) == 1


@pytest.mark.anyio
async def test_news_workspace_starts_without_report_state() -> None:
    # Given: the standalone News workspace is part of the shell.
    app = FinAgentApp()

    # When: News is opened by key binding.
    async with app.run_test() as pilot:
        await pilot.press("3")
        await pilot.pause()
        assert isinstance(app.screen, NewsScreen)
        news_body = str(app.screen.query_one("#news_body", Static).render())

        # Then: News renders its own shell body, not report progress.
        assert "Press Refresh to load source-backed news." in news_body
        assert "Progress:" not in news_body
        assert "report" not in news_body.lower()


@pytest.mark.anyio
async def test_repeated_workspace_navigation_replaces_active_screen() -> None:
    # Given: the workspace shell has one active workspace.
    app = FinAgentApp()

    # When: navigation repeats across workspaces.
    async with app.run_test() as pilot:
        for key in ("2", "3", "4", "5", "6", "1", "6", "3"):
            await pilot.press(key)
            await pilot.pause()

        # Then: only the current workspace screen remains mounted.
        assert isinstance(app.screen, NewsScreen)
        assert app.screen.query_one("#news_body", Static)
        assert workspace_screen_names(app) == ("NewsScreen",)


@pytest.mark.anyio
async def test_navigation_closes_prior_news_source_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: News creates default-owned source clients during refresh.
    client_factory = httpx.Client
    clients: list[httpx.Client] = []

    def make_client(
        *,
        timeout: httpx.Timeout,
        limits: httpx.Limits,
        follow_redirects: bool,
    ) -> httpx.Client:
        assert timeout is not None
        assert limits is not None
        assert follow_redirects is True
        client = client_factory(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={}, request=request),
            ),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", make_client)
    app = FinAgentApp()

    # When: News is refreshed, replaced, then opened and refreshed again.
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("3")
        await pilot.pause()
        _ = await pilot.click("#news_refresh")
        await pilot.pause()
        assert len(clients) == 1
        assert clients[0].is_closed is False

        await pilot.press("4")
        await pilot.pause()
        assert clients[0].is_closed is True

        await pilot.press("3")
        await pilot.pause()
        _ = await pilot.click("#news_refresh")
        await pilot.pause()

        # Then: the prior News client stays closed and the new screen owns a new one.
        assert len(clients) == 2
        assert clients[0].is_closed is True
        assert clients[1].is_closed is False


@pytest.mark.anyio
async def test_visible_tabs_route_to_rendered_workspaces() -> None:
    # Given: the visible workspace tabs are rendered in the Textual shell.
    app = FinAgentApp()

    # When: a user clicks the tab labels instead of pressing key bindings.
    async with app.run_test() as pilot:
        _ = await pilot.click("#ticker")
        await pilot.pause()
        assert isinstance(app.screen, TickerScreen)
        ticker_body = str(app.screen.query_one("#ticker_body", Static).render())
        assert "Metrics: not loaded" in ticker_body

        _ = await pilot.click("#news")
        await pilot.pause()
        assert isinstance(app.screen, NewsScreen)
        news_body = str(app.screen.query_one("#news_body", Static).render())
        assert "Browse market, ticker, macro/event, and filing sources." in news_body

        _ = await pilot.click("#reports")
        await pilot.pause()
        assert isinstance(app.screen, ReportsScreen)
        reports_body = str(app.screen.query_one("#reports_body", Static).render())
        assert "Progress: 0/0 (not_started)" in reports_body

        _ = await pilot.click("#settings")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        settings_body = str(app.screen.query_one("#settings_body", Static).render())
        assert "Status: defaults" in settings_body

        # Then: clicks use the same replacement route as keyboard navigation.
        assert workspace_screen_names(app) == ("SettingsScreen",)


@pytest.mark.anyio
async def test_visible_tabs_follow_workspace_order() -> None:
    # Given: the shell renders workspace tabs from the active workspace list.
    app = FinAgentApp()

    # When: the initial screen is mounted.
    async with app.run_test() as pilot:
        await pilot.pause()

        # Then: rendered tab order matches the keyboard navigation order.
        tab_ids = tuple(tab.id for tab in app.screen.query(Tab))
        assert tab_ids == EXPECTED_WORKSPACE_TAB_IDS


@pytest.mark.anyio
async def test_missing_config_defaults() -> None:
    # Given: no Todo 2 config module is required for UI startup.
    app = FinAgentApp()

    # When: Settings is opened.
    async with app.run_test() as pilot:
        await pilot.press("6")
        await pilot.pause()

    # Then: the UI exposes fallback defaults and a concrete config path.
        assert isinstance(app.screen, SettingsScreen)
        rendered_settings = str(app.screen.query_one("#settings_body", Static).render())
        assert "Status: defaults" in rendered_settings
        assert "config.json" in rendered_settings


@pytest.mark.anyio
async def test_startup_does_not_import_yfinance() -> None:
    # Given: market data must not be fetched or even wired at app startup.
    _ = sys.modules.pop("yfinance", None)

    # When: the shell starts and renders its first screen.
    app = FinAgentApp()
    async with app.run_test() as pilot:
        await pilot.pause()

    # Then: Todo 6 remains a shell and does not import yfinance.
    assert "yfinance" not in sys.modules
    assert app.state.session_cache.market_data_keys == ()


def workspace_screen_names(app: FinAgentApp) -> tuple[str, ...]:
    return tuple(
        type(screen).__name__
        for screen in app.screen_stack
        if isinstance(screen, WORKSPACE_SCREEN_TYPES)
    )
