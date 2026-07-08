# pyright: reportUnnecessaryComparison=false
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, assert_never

from textual.app import App

from finagent_terminal.ui.screens.markets import MarketsScreen
from finagent_terminal.ui.screens.news import NewsScreen
from finagent_terminal.ui.screens.reports import ReportsScreen
from finagent_terminal.ui.screens.settings import SettingsScreen
from finagent_terminal.ui.screens.ticker import TickerScreen
from finagent_terminal.ui.screens.tools import ToolsScreen
from finagent_terminal.ui.state import AppState, WorkspaceId, create_app_state

if TYPE_CHECKING:
    from textual.binding import BindingType

    from finagent_terminal.ui.screens import WorkspaceRequested


class FinAgentApp(App[None]):
    CSS: ClassVar[str] = (
        "Screen { layout: vertical; }\n"
        "Tabs { dock: top; }\n"
        ".workspace-title { text-style: bold; padding: 1 2 0 2; }\n"
        "Static { padding: 0 2; }\n"
    )
    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Quit"),
        ("1", "workspace('markets')", "Markets"),
        ("2", "workspace('ticker')", "Ticker"),
        ("3", "workspace('news')", "News"),
        ("4", "workspace('tools')", "Tools"),
        ("5", "workspace('reports')", "Reports"),
        ("6", "workspace('settings')", "Settings"),
    ]

    def __init__(self, state: AppState | None = None) -> None:
        """Create the offline workspace shell state."""
        super().__init__()
        self.state: AppState = create_app_state() if state is None else state
        self.workspace: WorkspaceId = "markets"

    async def on_mount(self) -> None:
        await self.push_screen(MarketsScreen(self.state))

    async def on_workspace_requested(self, event: WorkspaceRequested) -> None:
        _ = event.stop()
        await self.action_workspace(event.workspace)

    async def action_workspace(self, workspace: WorkspaceId) -> None:
        self.workspace = workspace
        if self.screen_stack:
            await self.pop_screen()
        match workspace:
            case "markets":
                await self.push_screen(MarketsScreen(self.state))
                return
            case "ticker":
                await self.push_screen(TickerScreen(self.state))
                return
            case "news":
                await self.push_screen(NewsScreen(self.state))
                return
            case "tools":
                await self.push_screen(ToolsScreen(self.state))
                return
            case "reports":
                await self.push_screen(ReportsScreen(self.state))
                return
            case "settings":
                await self.push_screen(SettingsScreen(self.state))
                return
            case unreachable:
                assert_never(unreachable)
