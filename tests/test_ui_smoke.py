from __future__ import annotations

import sys

import pytest
from textual.widgets import Static

from finagent_terminal.ui.app import FinAgentApp
from finagent_terminal.ui.screens.markets import MarketsScreen
from finagent_terminal.ui.screens.reports import ReportsScreen
from finagent_terminal.ui.screens.settings import SettingsScreen
from finagent_terminal.ui.screens.ticker import TickerScreen
from finagent_terminal.ui.screens.tools import ToolsScreen

WORKSPACE_SCREEN_TYPES = (
    MarketsScreen,
    TickerScreen,
    ToolsScreen,
    ReportsScreen,
    SettingsScreen,
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
        assert isinstance(app.screen, ToolsScreen)
        tools_body = str(app.screen.query_one("#tools_body", Static).render())
        assert "Compare: enter 2-6 symbols" in tools_body

        await pilot.press("4")
        await pilot.pause()
        assert isinstance(app.screen, ReportsScreen)
        reports_body = str(app.screen.query_one("#reports_body", Static).render())
        assert "Progress: 0/0 (not_started)" in reports_body

        await pilot.press("5")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        settings_body = str(app.screen.query_one("#settings_body", Static).render())
        assert "Status: defaults" in settings_body
        assert len(workspace_screen_names(app)) == 1


@pytest.mark.anyio
async def test_repeated_workspace_navigation_replaces_active_screen() -> None:
    # Given: the workspace shell has one active workspace.
    app = FinAgentApp()

    # When: navigation repeats across workspaces.
    async with app.run_test() as pilot:
        for key in ("2", "3", "4", "5", "1", "5", "3"):
            await pilot.press(key)
            await pilot.pause()

        # Then: only the current workspace screen remains mounted.
        assert isinstance(app.screen, ToolsScreen)
        assert workspace_screen_names(app) == ("ToolsScreen",)


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
async def test_missing_config_defaults() -> None:
    # Given: no Todo 2 config module is required for UI startup.
    app = FinAgentApp()

    # When: Settings is opened.
    async with app.run_test() as pilot:
        await pilot.press("5")
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
