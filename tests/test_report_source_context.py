from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

import finagent_terminal.reports.state as report_state
from finagent_terminal.data.sources import SourceSession
from finagent_terminal.llm import ProviderName
from finagent_terminal.reports.models import (
    DepthMode,
    ReportSectionPrompt,
    ReportSectionResult,
)
from finagent_terminal.reports.state import (
    ReportClientFactory,
    ReportJob,
    ReportRequest,
    run_screen_report,
)
from finagent_terminal.ui.screens.news import NewsScreen
from finagent_terminal.ui.state import create_app_state
from tests.report_screen_fixtures import (
    FakeSession,
    aapl_snapshot,
    report_source_response,
)
from tests.source_routing_fixtures import SEC_SAMPLE, source_session
from tests.vendor_source_fixtures import FINNHUB_SAMPLE, FMP_SAMPLE

if TYPE_CHECKING:
    import pytest

    from finagent_terminal.llm import LLMSelection
    from finagent_terminal.llm.base import StructuredLLMProvider


def _record_source_session_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[SourceSession]:
    close_calls: list[SourceSession] = []
    original_close = SourceSession.close

    def close(source_session: SourceSession) -> None:
        close_calls.append(source_session)
        original_close(source_session)

    monkeypatch.setattr(SourceSession, "close", close)
    return close_calls


def _report_job(
    client: RecordingPromptClient,
    source_session: SourceSession | None = None,
) -> ReportJob:
    return ReportJob(
        request=ReportRequest(
            symbol="AAPL",
            provider=ProviderName.OPENAI,
            model="fake-model",
            depth=DepthMode.STANDARD,
        ),
        session=FakeSession(aapl_snapshot()),
        source_session=source_session,
        client_factory=recording_client_factory(client),
    )


def test_run_screen_report_closes_internal_source_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls = _record_source_session_closes(monkeypatch)
    captured: list[SourceSession] = []

    def fetch_context(source_session: SourceSession, symbol: str) -> None:
        assert symbol == "AAPL"
        captured.append(source_session)
        assert close_calls == []

    monkeypatch.setattr(report_state, "fetch_report_source_context", fetch_context)

    _ = run_screen_report(_report_job(RecordingPromptClient()))

    assert close_calls == captured


def test_run_screen_report_keeps_injected_source_session_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls = _record_source_session_closes(monkeypatch)
    source = SourceSession()

    def fetch_context(source_session: SourceSession, symbol: str) -> None:
        assert symbol == "AAPL"
        assert source_session is source
        assert close_calls == []

    monkeypatch.setattr(report_state, "fetch_report_source_context", fetch_context)

    try:
        _ = run_screen_report(_report_job(RecordingPromptClient(), source))

        assert close_calls == []
    finally:
        source.close()


def test_report_source_context() -> None:
    requests: list[httpx.Request] = []
    client = RecordingPromptClient()

    result = run_screen_report(
        _report_job(
            client,
            SourceSession(
                env={
                    "FINAGENT_SEC_USER_AGENT": SEC_SAMPLE,
                    "FMP_API_KEY": FMP_SAMPLE,
                    "FINNHUB_API_KEY": FINNHUB_SAMPLE,
                    "FRED_API_KEY": "fred-test-secret",
                },
                http_client=httpx.Client(
                    transport=httpx.MockTransport(
                        lambda request: report_source_response(request, requests),
                    ),
                ),
            ),
        ),
    )

    rendered = client.rendered_prompts[0]
    assert result.result.final_report is not None
    assert '<source_context trust="untrusted">' in rendered
    assert _source_labels(client) >= {
        "SEC",
        "FMP",
        "Finnhub",
        "FRED",
        "Treasury Fiscal Data",
        "GDELT",
    }
    assert "Apple Inc. company facts" in rendered
    assert "News: Apple supplier update" in rendered
    assert "FEDFUNDS observations" in rendered
    assert SEC_SAMPLE not in rendered
    assert FMP_SAMPLE not in rendered
    assert FINNHUB_SAMPLE not in rendered
    assert any(request.url.host == "api.stlouisfed.org" for request in requests)
    assert any(
        request.url.host == "api.fiscaldata.treasury.gov" for request in requests
    )


def test_reports_independent_from_news() -> None:
    state = create_app_state()
    news_requests: list[httpx.Request] = []
    report_requests: list[httpx.Request] = []
    news_screen = NewsScreen(
        state,
        source_session(
            lambda request: labeled_source_response(
                request,
                news_requests,
                "News-only cached source",
            ),
        ),
    )
    news_screen.refresh_news("AAPL")
    client = RecordingPromptClient()

    _ = run_screen_report(
        _report_job(
            client,
            source_session(
                lambda request: labeled_source_response(
                    request,
                    report_requests,
                    "Report-only source",
                ),
            ),
        ),
    )

    rendered = "\n".join(client.rendered_prompts)
    assert "Report-only source" in rendered
    assert "News-only cached source" not in rendered
    assert len(news_requests) > 0
    assert len(report_requests) > 0


def test_all_sources_unavailable() -> None:
    def failing_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"}, request=request)

    client = RecordingPromptClient()

    result = run_screen_report(
        _report_job(
            client,
            SourceSession(
                env={
                    "FINAGENT_SEC_USER_AGENT": "sec-test-agent",
                    "FMP_API_KEY": "fmp-test-key",
                    "FINNHUB_API_KEY": "finnhub-test-key",
                    "FRED_API_KEY": "fred-test-key",
                },
                http_client=httpx.Client(
                    transport=httpx.MockTransport(failing_response),
                ),
            ),
        ),
    )

    rendered = "\n".join(client.rendered_prompts)
    assert result.result.final_report is not None
    assert "HTTP 500" in rendered
    assert _source_labels(client) >= {
        "SEC",
        "FMP",
        "Finnhub",
        "FRED",
        "Treasury Fiscal Data",
        "GDELT",
    }


class RecordingPromptClient:
    def __init__(self) -> None:
        self.prompts: list[ReportSectionPrompt] = []
        self.rendered_prompts: list[str] = []

    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        self.prompts.append(prompt)
        self.rendered_prompts.append(prompt.render())
        summary = f"{prompt.context.symbol} {prompt.section.title} generated."
        if prompt.section.id == "reports_editor":
            summary = (
                f"## {prompt.section.title}\n"
                f"{prompt.context.symbol} final editor report."
            )
        return ReportSectionResult(
            section_id=prompt.section.id,
            title=prompt.section.title,
            summary=summary,
            key_points=(prompt.context.market_summary,),
            risks=prompt.context.warnings
            or ("No additional supplied-context risks.",),
        )


def recording_client_factory(client: RecordingPromptClient) -> ReportClientFactory:
    def factory(selection: LLMSelection | None = None) -> StructuredLLMProvider:
        _ = selection
        return client

    return factory


def labeled_source_response(
    request: httpx.Request,
    requests: list[httpx.Request],
    label: str,
) -> httpx.Response:
    if request.url.host == "api.gdeltproject.org":
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "articles": [
                    {
                        "title": label,
                        "url": "https://example.test/gdelt",
                        "seendate": "20240101120000",
                        "domain": "example.test",
                    },
                ],
            },
            request=request,
        )
    return report_source_response(request, requests)


def _source_labels(client: RecordingPromptClient) -> set[str]:
    return {
        label
        for note in client.prompts[0].context.source_notes
        for label in note.detail.replace(";", ",").replace(":", ",").split(", ")
    } | {note.label for note in client.prompts[0].context.source_notes}
