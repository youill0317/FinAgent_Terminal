from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual.widgets import Input, Static

from finagent_terminal.ui.screens.reports import ReportsScreen
from finagent_terminal.ui.state import create_app_state
from tests.report_screen_fixtures import (
    FakeSession,
    MutableClock,
    ReportsTestApp,
    StepwiseLLMClient,
    aapl_snapshot,
    client_factory,
    failing_client_factory,
)

if TYPE_CHECKING:
    from pathlib import Path

    from finagent_terminal.llm import LLMSelection
    from finagent_terminal.llm.base import StructuredLLMProvider


@pytest.mark.anyio
async def test_reports_screen_generates_preview_and_exports(tmp_path: Path) -> None:
    screen = ReportsScreen(
        create_app_state(),
        data_session=FakeSession(aapl_snapshot()),
        client_factory=client_factory,
        export_dir=tmp_path,
    )
    app = ReportsTestApp(screen)

    async with app.run_test() as pilot:
        app.screen.query_one("#report_ticker", Input).value = "AAPL"
        app.screen.query_one("#report_model", Input).value = "fake-model"
        _ = await pilot.click("#start_report")
        for _ in range(10):
            await pilot.pause()
            progress_text = str(
                app.screen.query_one("#report_progress", Static).render(),
            )
            if "Completed: 5/5" in progress_text:
                break

        progress_text = str(app.screen.query_one("#report_progress", Static).render())
        preview_text = str(app.screen.query_one("#report_preview", Static).render())
        assert "Provider/model: openai/fake-model" in progress_text
        assert "Completed: 5/5" in progress_text
        assert "Latest error: none" in progress_text
        assert "## Overview" in preview_text
        assert "## Bull Case" in preview_text

        _ = await pilot.click("#export_report")
        await pilot.pause()

        export_status = str(app.screen.query_one("#report_export", Static).render())
        assert "aapl-standard-" in export_status


@pytest.mark.anyio
async def test_reports_screen_updates_partial_preview_while_running(
    tmp_path: Path,
) -> None:
    client = StepwiseLLMClient()
    clock = MutableClock()

    def stepwise_client_factory(
        selection: LLMSelection | None = None,
    ) -> StructuredLLMProvider:
        _ = selection
        return client

    screen = ReportsScreen(
        create_app_state(),
        data_session=FakeSession(aapl_snapshot()),
        client_factory=stepwise_client_factory,
        export_dir=tmp_path,
        clock=clock.now,
    )
    app = ReportsTestApp(screen)

    async with app.run_test() as pilot:
        _ = await pilot.click("#start_report")
        clock.current = 1.2
        for _ in range(20):
            await pilot.pause()
            if client.has_started("overview"):
                break
        clock.current = 1.2
        for _ in range(20):
            await pilot.pause()
            progress_text = str(
                app.screen.query_one("#report_progress", Static).render(),
            )
            if "Elapsed: 1.2s" in progress_text:
                break

        progress_text = str(app.screen.query_one("#report_progress", Static).render())
        preview_text = str(app.screen.query_one("#report_preview", Static).render())
        assert "Active: overview" in progress_text
        assert "Overview: running" in progress_text
        assert "Completed: 0/5" in progress_text
        assert "Elapsed: 1.2s" in progress_text
        assert "## Overview" not in preview_text

        client.release("overview")
        for _ in range(20):
            await pilot.pause()
            preview_text = str(app.screen.query_one("#report_preview", Static).render())
            if "## Overview" in preview_text and client.has_started("risk_notes"):
                break

        progress_text = str(app.screen.query_one("#report_progress", Static).render())
        preview_text = str(app.screen.query_one("#report_preview", Static).render())
        assert "Active: risk_notes" in progress_text
        assert "Risk Notes: running" in progress_text
        assert "Completed: 1/5" in progress_text
        assert "## Overview" in preview_text
        assert "## Risk Notes" not in preview_text
        client.release("risk_notes")
        client.release("technical_indicators")
        client.release("bull_case")
        client.release("bear_case")
        for _ in range(20):
            await pilot.pause()
            progress_text = str(
                app.screen.query_one("#report_progress", Static).render(),
            )
            if "Completed: 5/5" in progress_text:
                break


@pytest.mark.anyio
async def test_reports_screen_preserves_progress_after_navigation(
    tmp_path: Path,
) -> None:
    client = StepwiseLLMClient()
    shared_state = create_app_state()

    def stepwise_client_factory(
        selection: LLMSelection | None = None,
    ) -> StructuredLLMProvider:
        _ = selection
        return client

    first = ReportsScreen(
        shared_state,
        data_session=FakeSession(aapl_snapshot()),
        client_factory=stepwise_client_factory,
        export_dir=tmp_path,
    )
    app = ReportsTestApp(first)

    async with app.run_test() as pilot:
        _ = await pilot.click("#start_report")
        for _ in range(20):
            await pilot.pause()
            if client.has_started("overview"):
                break
        await app.pop_screen()
        second = ReportsScreen(
            shared_state,
            data_session=FakeSession(aapl_snapshot()),
            client_factory=stepwise_client_factory,
            export_dir=tmp_path,
        )
        await app.push_screen(second)
        await pilot.pause()

        progress_text = str(app.screen.query_one("#report_progress", Static).render())
        assert "Active: overview" in progress_text
        assert "Overview: running" in progress_text

        client.release("overview")
        client.release("risk_notes")
        client.release("technical_indicators")
        client.release("bull_case")
        client.release("bear_case")
        for _ in range(20):
            await pilot.pause()
            progress_text = str(
                app.screen.query_one("#report_progress", Static).render(),
            )
            if "Completed: 5/5" in progress_text:
                break


@pytest.mark.anyio
async def test_reports_screen_failed_section_visible(tmp_path: Path) -> None:
    screen = ReportsScreen(
        create_app_state(),
        data_session=FakeSession(aapl_snapshot()),
        client_factory=failing_client_factory,
        export_dir=tmp_path,
    )
    app = ReportsTestApp(screen)

    async with app.run_test() as pilot:
        _ = await pilot.click("#start_report")
        for _ in range(10):
            await pilot.pause()
            progress_text = str(
                app.screen.query_one("#report_progress", Static).render(),
            )
            if "Bull Case: failed" in progress_text:
                break

        progress_text = str(app.screen.query_one("#report_progress", Static).render())
        preview_text = str(app.screen.query_one("#report_preview", Static).render())
        assert "Bull Case: failed" in progress_text
        assert "Latest error: fake provider failed: fake failure for Bull Case" in (
            progress_text
        )
        assert "## Bull Case" not in preview_text
        assert "## Bear Case" in preview_text
