from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from finagent_terminal.llm.base import (
    FakeLLMClient,
    LLMProviderConfig,
    ProviderCallError,
    ProviderName,
    ProviderUnavailableError,
)
from finagent_terminal.llm.gemini_provider import GeminiProvider
from finagent_terminal.reports.export import ReportExport, render_markdown_report
from finagent_terminal.reports.models import (
    DepthMode,
    ReportContext,
    ReportSectionPrompt,
    ReportSectionResult,
)
from finagent_terminal.reports.pipeline import ReportPipeline, run_report_pipeline
from finagent_terminal.reports.progress import (
    ReportProgress,
    SectionState,
)

if TYPE_CHECKING:
    import pytest


def test_standard_report_progress_completes() -> None:
    context = ReportContext(
        symbol="AAPL",
        market_summary="price up",
        indicator_summary="rsi neutral",
    )

    result = run_report_pipeline(
        client=FakeLLMClient(mode="1"),
        context=context,
        depth=DepthMode.STANDARD,
    )

    assert result.progress.completed_count == result.progress.total_count
    assert result.progress.latest_error is None
    assert {section.state for section in result.progress.sections} == {
        SectionState.COMPLETE,
    }
    assert len(result.sections) == result.progress.total_count


def test_section_failure_sets_latest_error() -> None:
    context = ReportContext(
        symbol="AAPL",
        market_summary="price up",
        indicator_summary="rsi neutral",
    )

    result = run_report_pipeline(
        client=FailingBullishClient(),
        context=context,
        depth=DepthMode.STANDARD,
    )

    failed = result.progress.section("bullish_researcher")
    assert failed.state is SectionState.FAILED
    assert (
        result.progress.latest_error
        == "fake provider failed: fake failure for Bullish Researcher"
    )
    assert result.progress.completed_count == result.progress.total_count - 1


def test_failed_role_visible() -> None:
    context = ReportContext(
        symbol="AAPL",
        market_summary="price up",
        indicator_summary="rsi neutral",
    )

    result = run_report_pipeline(
        client=FailingBullishClient(),
        context=context,
        depth=DepthMode.STANDARD,
    )

    failed = result.progress.section("bullish_researcher")
    assert failed.title == "Bullish Researcher"
    assert failed.state is SectionState.FAILED
    assert failed.error == "fake provider failed: fake failure for Bullish Researcher"
    assert "Bullish Researcher" not in {section.title for section in result.sections}


def test_fake_fail_section_targets_bullish_role_and_later_roles_continue() -> None:
    context = ReportContext(
        symbol="AAPL",
        market_summary="price up",
        indicator_summary="rsi neutral",
    )

    result = run_report_pipeline(
        client=FakeLLMClient(mode="fail_section"),
        context=context,
        depth=DepthMode.STANDARD,
    )

    failed = result.progress.section("bullish_researcher")
    assert failed.title == "Bullish Researcher"
    assert failed.state is SectionState.FAILED
    assert failed.error == "fake provider failed: fake failure for Bullish Researcher"
    assert result.progress.latest_error == failed.error
    assert result.progress.section("bearish_researcher").state is SectionState.COMPLETE
    assert result.progress.section("debate_refinement").state is SectionState.COMPLETE
    assert result.progress.section("reports_editor").state is SectionState.COMPLETE
    assert result.progress.completed_count == result.progress.total_count - 1


def test_report_pipeline_records_provider_errors_without_autofallback() -> None:
    pipeline = ReportPipeline(client=RaisingClient())
    context = ReportContext(
        symbol="IWM",
        market_summary="risk on",
        indicator_summary="trend down",
    )

    result = pipeline.run(context=context, depth=DepthMode.QUICK)

    assert isinstance(result.progress, ReportProgress)
    assert result.progress.completed_count == 0
    assert result.progress.latest_error == "fake provider failed: Reports Editor"
    assert result.sections == ()


def test_report_pipeline_records_provider_unavailable_without_abort() -> None:
    pipeline = ReportPipeline(client=UnavailableClient(), provider="openai")
    context = ReportContext(
        symbol="IWM",
        market_summary="risk on",
        indicator_summary="trend down",
    )

    result = pipeline.run(context=context, depth=DepthMode.QUICK)

    assert result.progress.completed_count == 0
    assert result.progress.latest_error == "openai provider unavailable: Reports Editor"
    assert result.sections == ()


def test_gemini_http_error_redacts_key_from_progress_and_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_key = "GEMINI-SENTINEL-12345"
    observed_key_bearing_url = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_key_bearing_url
        url = str(request.url)
        observed_key_bearing_url = "key=" in url and sentinel_key in url
        return httpx.Response(503, request=request, text="temporarily unavailable")

    def client_factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(
        "finagent_terminal.llm.gemini_provider.create_provider_http_client",
        client_factory,
    )
    context = ReportContext(
        symbol="IWM",
        market_summary="risk on",
        indicator_summary="trend down",
    )
    pipeline = ReportPipeline(
        client=GeminiProvider(
            config=LLMProviderConfig(
                provider=ProviderName.GEMINI,
                model="gemini-test",
                api_key=sentinel_key,
            ),
        ),
        provider=ProviderName.GEMINI.value,
        model="gemini-test",
    )

    result = pipeline.run(context=context, depth=DepthMode.QUICK)
    progress_text = "\n".join(
        error
        for snapshot in result.timeline
        for error in (
            snapshot.latest_error,
            *(section.error for section in snapshot.sections),
        )
        if error is not None
    )
    markdown = render_markdown_report(
        ReportExport(
            context=context,
            result=result,
            created_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        ),
    )

    progress_leaked = sentinel_key in progress_text or "key=" in progress_text
    markdown_leaked = sentinel_key in markdown or "key=" in markdown
    assert observed_key_bearing_url is True
    assert progress_leaked is False
    assert markdown_leaked is False


@dataclass(frozen=True, slots=True)
class FailingBullishClient:
    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        if prompt.section.id == "bullish_researcher":
            raise ProviderCallError(
                provider="fake",
                message=f"fake failure for {prompt.section.title}",
            )
        return ReportSectionResult(
            section_id=prompt.section.id,
            title=prompt.section.title,
            summary=f"{prompt.section.title} generated from supplied context.",
            key_points=(prompt.context.market_summary,),
        )


@dataclass(frozen=True, slots=True)
class RaisingClient:
    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        raise ProviderCallError(provider="fake", message=prompt.section.title)


@dataclass(frozen=True, slots=True)
class UnavailableClient:
    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        raise ProviderUnavailableError(
            provider=ProviderName.OPENAI,
            message=prompt.section.title,
        )
