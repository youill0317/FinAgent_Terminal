from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING, ClassVar, override

from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Input, Static, Tab, Tabs

from finagent_terminal.config import app_home
from finagent_terminal.llm import (
    LLMSelection,
    ProviderName,
    create_llm_client,
    resolve_llm_config,
)
from finagent_terminal.reports.export import (
    ReportExport,
    render_markdown_report,
    write_markdown_report,
)
from finagent_terminal.reports.models import DepthMode
from finagent_terminal.reports.progress import ReportProgress, new_report_progress
from finagent_terminal.reports.state import (
    ReportClientFactory,
    ReportDataSession,
    ReportJob,
    ReportRequest,
    ReportScreenState,
    ScreenReportResult,
    report_state_for,
    run_screen_report,
    set_report_state_for,
)
from finagent_terminal.ui.screens import WorkspaceScreen
from finagent_terminal.ui.state import WORKSPACE_LABELS, WORKSPACES

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from textual.app import ComposeResult

    from finagent_terminal.ui.state import AppState, WorkspaceId


class ReportsScreen(WorkspaceScreen):
    workspace: ClassVar[WorkspaceId] = "reports"
    label: ClassVar[str] = "Reports"
    summary: ClassVar[str] = "Generate structured research reports."

    def __init__(
        self,
        state: AppState,
        *,
        data_session: ReportDataSession | None = None,
        client_factory: ReportClientFactory = create_llm_client,
        export_dir: Path | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Create the Reports workspace."""
        super().__init__(state)
        self._data_session: ReportDataSession | None = data_session
        self._client_factory: ReportClientFactory = client_factory
        self._export_dir: Path = (
            app_home() / "reports" if export_dir is None else export_dir
        )
        self._clock: Callable[[], float] = clock
        self._report_state: ReportScreenState = report_state_for(state)

    @override
    def compose(self) -> ComposeResult:
        progress = self._report_state.progress
        job_text = f"Job: {progress.provider}/{progress.model}"
        body_text = f"{self.summary}\n{job_text}\nProgress: 0/0 (not_started)"
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
        yield Input(value="AAPL", placeholder="Ticker", id="report_ticker")
        yield Input(value="openai", placeholder="Provider", id="report_provider")
        yield Input(
            value=progress.model,
            placeholder="Model",
            id="report_model",
        )
        yield Input(value="Standard", placeholder="Depth", id="report_depth")
        yield Button("Start", id="start_report", variant="primary")
        yield Button("Export", id="export_report")
        yield Static(body_text, id="reports_body")
        yield Static(self._progress_text(), id="report_progress")
        yield Static(self._preview_text(), id="report_preview")
        yield Static(self._report_state.export_status, id="report_export")
        yield Footer()

    def on_mount(self) -> None:
        _ = self.set_interval(0.1, self._refresh_widgets)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "start_report":
            self.start_report()
        elif button_id == "export_report":
            _ = self.export_report()

    def start_report(self) -> None:
        request = self._read_request()
        if request is None:
            return
        self._set_report_state(
            ReportScreenState(
                progress=_started_progress(request, self._clock()),
                export_status="No report exported.",
                running=True,
            ),
        )
        self._refresh_widgets()
        _ = self.run_worker(
            lambda: self._run_report_worker(request),
            exclusive=True,
            group="reports",
            thread=True,
        )

    def _run_report_worker(self, request: ReportRequest) -> None:
        screen_result = run_screen_report(
            ReportJob(
                request=request,
                session=self._session(),
                client_factory=self._client_factory,
                started_at=self._report_state.progress.started_at,
                on_update=self._queue_report_update,
            ),
        )
        self.call_next(self._finish_report, screen_result)

    def _queue_report_update(self, screen_result: ScreenReportResult) -> None:
        self._store_report_result(screen_result)
        self.call_next(self._refresh_widgets)

    def _finish_report(self, screen_result: ScreenReportResult) -> None:
        self._store_report_result(screen_result)
        self._refresh_widgets()

    def export_report(self) -> Path | None:
        if self._report_state.latest is None or self._report_state.running:
            self._set_report_state(
                replace(
                    self._report_state,
                    export_status="No completed report to export.",
                ),
            )
            self._refresh_widgets()
            return None
        path = write_markdown_report(
            ReportExport(
                context=self._report_state.latest.context,
                result=self._report_state.latest.result,
                created_at=datetime.now(UTC),
            ),
            self._export_dir,
        )
        self._set_report_state(
            replace(self._report_state, export_status=f"Exported: {path}"),
        )
        self._refresh_widgets()
        return path

    def _read_request(self) -> ReportRequest | None:
        symbol = self.query_one("#report_ticker", Input).value.strip().upper()
        provider_text = self.query_one("#report_provider", Input).value.strip().lower()
        model = self.query_one("#report_model", Input).value.strip()
        depth_text = self.query_one("#report_depth", Input).value.strip().title()
        try:
            provider = ProviderName(provider_text)
            depth = DepthMode(depth_text)
        except ValueError:
            self._set_report_state(
                replace(self._report_state, export_status="Invalid provider or depth."),
            )
            self._refresh_widgets()
            return None
        return ReportRequest(
            symbol=symbol or "AAPL",
            provider=provider,
            model=model or resolve_llm_config(
                selection=LLMSelection(provider=provider),
            ).model,
            depth=depth,
        )

    def _session(self) -> ReportDataSession:
        if self._data_session is not None:
            return self._data_session
        from finagent_terminal.data.session import DataSession  # noqa: PLC0415
        from finagent_terminal.data.yfinance_provider import (  # noqa: PLC0415
            YFinanceProvider,
        )

        self._data_session = DataSession(YFinanceProvider())
        return self._data_session

    def _refresh_widgets(self) -> None:
        if not self.is_mounted:
            return
        try:
            self.query_one("#report_progress", Static).update(self._progress_text())
            self.query_one("#report_preview", Static).update(self._preview_text())
            self.query_one("#report_export", Static).update(
                self._report_state.export_status,
            )
        except NoMatches:
            return

    def _progress_text(self) -> str:
        progress = self._report_state.progress
        active = progress.active_section or "none"
        now = self._clock() if progress.finished_at is None else progress.finished_at
        lines = [
            self.summary,
            f"Provider/model: {progress.provider}/{progress.model}",
            f"Depth: {progress.depth.value}",
            f"Active: {active}",
            f"Elapsed: {progress.elapsed_seconds(now=now):.1f}s",
            f"Completed: {progress.completed_count}/{progress.total_count}",
            f"Latest error: {progress.latest_error or 'none'}",
        ]
        lines.extend(
            f"{section.title}: {section.state.value}"
            + (f" - {section.error}" if section.error else "")
            for section in progress.sections
        )
        return "\n".join(lines)

    def _preview_text(self) -> str:
        if self._report_state.latest is None:
            return "No report generated."
        return render_markdown_report(
            ReportExport(
                context=self._report_state.latest.context,
                result=self._report_state.latest.result,
                created_at=datetime.now(UTC),
            ),
        )
    def _store_report_result(self, screen_result: ScreenReportResult) -> None:
        self._set_report_state(
            ReportScreenState(
                progress=screen_result.result.progress,
                latest=screen_result,
                export_status=self._report_state.export_status,
                running=screen_result.result.progress.finished_at is None,
            ),
        )

    def _set_report_state(self, report_state: ReportScreenState) -> None:
        self._report_state = set_report_state_for(self.state, report_state)


def _started_progress(request: ReportRequest, started_at: float) -> ReportProgress:
    progress = new_report_progress(
        provider=request.provider.value,
        model=request.model,
        depth=request.depth,
        started_at=started_at,
    )
    return progress.mark_running(progress.sections[0].id, now=started_at)
