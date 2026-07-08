from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual.widgets import Input, Static

from finagent_terminal.ui.screens.reports import ReportScreenSessions, ReportsScreen
from finagent_terminal.ui.state import create_app_state
from tests import test_report_source_context as source_context_tests
from tests.report_screen_fixtures import (
    FakeSession,
    MutableClock,
    ReportsTestApp,
    StepwiseLLMClient,
    aapl_snapshot,
    client_factory,
    failing_client_factory,
    report_source_session,
)

if TYPE_CHECKING:
    from pathlib import Path

    from finagent_terminal.llm import LLMSelection
    from finagent_terminal.llm.base import StructuredLLMProvider


@pytest.mark.anyio
async def test_reports_screen_generates_preview_and_exports(tmp_path: Path) -> None:
    screen = ReportsScreen(
        create_app_state(),
        sessions=report_sessions(),
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
            if "Completed: 7/7" in progress_text:
                break

        progress_text = str(app.screen.query_one("#report_progress", Static).render())
        preview_text = str(app.screen.query_one("#report_preview", Static).render())
        assert "Provider/model: openai/fake-model" in progress_text
        assert "Completed: 7/7" in progress_text
        assert "Latest error: none" in progress_text
        assert "## Reports Editor" in preview_text
        assert "AAPL Reports Editor generated from supplied context." in preview_text
        assert "## Technical Analyst" not in preview_text

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
        sessions=report_sessions(),
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
            if client.has_started("technical_analyst"):
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
        assert "Active: technical_analyst" in progress_text
        assert "Technical Analyst: running" in progress_text
        assert "Completed: 0/7" in progress_text
        assert "Elapsed: 1.2s" in progress_text
        assert "## Technical Analyst" not in preview_text

        client.release("technical_analyst")
        for _ in range(20):
            await pilot.pause()
            preview_text = str(app.screen.query_one("#report_preview", Static).render())
            if (
                "## Technical Analyst" in preview_text
                and client.has_started("fundamentals_analyst")
            ):
                break

        progress_text = str(app.screen.query_one("#report_progress", Static).render())
        preview_text = str(app.screen.query_one("#report_preview", Static).render())
        assert "Active: fundamentals_analyst" in progress_text
        assert "Fundamentals Analyst: running" in progress_text
        assert "Completed: 1/7" in progress_text
        assert "## Technical Analyst" in preview_text
        assert "## Fundamentals Analyst" not in preview_text
        client.release("fundamentals_analyst")
        client.release("news_sentiment_analyst")
        client.release("bullish_researcher")
        client.release("bearish_researcher")
        client.release("debate_refinement")
        client.release("reports_editor")
        for _ in range(20):
            await pilot.pause()
            progress_text = str(
                app.screen.query_one("#report_progress", Static).render(),
            )
            if "Completed: 7/7" in progress_text:
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
        sessions=report_sessions(),
        client_factory=stepwise_client_factory,
        export_dir=tmp_path,
    )
    app = ReportsTestApp(first)

    async with app.run_test() as pilot:
        _ = await pilot.click("#start_report")
        for _ in range(20):
            await pilot.pause()
            if client.has_started("technical_analyst"):
                break
        await app.pop_screen()
        second = ReportsScreen(
            shared_state,
            sessions=report_sessions(),
            client_factory=stepwise_client_factory,
            export_dir=tmp_path,
        )
        await app.push_screen(second)
        await pilot.pause()

        progress_text = str(app.screen.query_one("#report_progress", Static).render())
        assert "Active: technical_analyst" in progress_text
        assert "Technical Analyst: running" in progress_text

        client.release("technical_analyst")
        client.release("fundamentals_analyst")
        client.release("news_sentiment_analyst")
        client.release("bullish_researcher")
        client.release("bearish_researcher")
        client.release("debate_refinement")
        client.release("reports_editor")
        for _ in range(20):
            await pilot.pause()
            progress_text = str(
                app.screen.query_one("#report_progress", Static).render(),
            )
            if "Completed: 7/7" in progress_text:
                break


@pytest.mark.anyio
async def test_reports_screen_failed_role_visible(tmp_path: Path) -> None:
    screen = ReportsScreen(
        create_app_state(),
        sessions=report_sessions(),
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
            if "Bullish Researcher: failed" in progress_text:
                break

        progress_text = str(app.screen.query_one("#report_progress", Static).render())
        preview_text = str(app.screen.query_one("#report_preview", Static).render())
        assert "Bullish Researcher: failed" in progress_text
        assert "Bearish Researcher: complete" in progress_text
        assert "Reports Editor: complete" in progress_text
        assert (
            "Latest error: fake provider failed: fake failure for Bullish Researcher"
            in progress_text
        )
        assert "## Reports Editor" in preview_text
        assert "## Bullish Researcher" not in preview_text


def report_sessions() -> ReportScreenSessions:
    return ReportScreenSessions(
        data=FakeSession(aapl_snapshot()),
        source=report_source_session(),
    )


def test_report_source_context() -> None:
    source_context_tests.test_report_source_context()


def test_reports_independent_from_news() -> None:
    source_context_tests.test_reports_independent_from_news()


def test_all_sources_unavailable() -> None:
    source_context_tests.test_all_sources_unavailable()
