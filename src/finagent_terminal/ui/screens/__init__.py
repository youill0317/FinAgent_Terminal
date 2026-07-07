from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Static, Tab, Tabs

from finagent_terminal.ui.state import WORKSPACE_BY_TAB_ID, WORKSPACE_LABELS, WORKSPACES

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from finagent_terminal.ui.state import AppState, WorkspaceId


class WorkspaceRequested(Message):
    def __init__(self, workspace: WorkspaceId) -> None:
        """Create a workspace tab navigation message."""
        super().__init__()
        self.workspace: WorkspaceId = workspace


class WorkspaceScreen(Screen[None]):
    workspace: ClassVar[WorkspaceId]
    label: ClassVar[str]
    summary: ClassVar[str]

    def __init__(self, state: AppState) -> None:
        """Store shared shell state for workspace rendering."""
        super().__init__()
        self.state: AppState = state

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
        yield Static("\n".join(self.lines()), id=f"{self.workspace}_body")
        yield Footer()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        tab_id = event.tab.id
        if tab_id is None:
            return
        workspace = WORKSPACE_BY_TAB_ID.get(tab_id)
        if workspace is None:
            return
        _ = event.stop()
        if workspace == self.workspace:
            return
        _ = self.post_message(WorkspaceRequested(workspace))

    def lines(self) -> tuple[str, ...]:
        return (self.summary,)

__all__ = ("WorkspaceRequested", "WorkspaceScreen")
