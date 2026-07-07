from __future__ import annotations

from dataclasses import dataclass

from finagent_terminal.llm.base import (
    FakeLLMClient,
    ProviderCallError,
    ProviderName,
    ProviderUnavailableError,
)
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
    new_report_progress,
)


def test_section_states_are_exact_user_labels() -> None:
    assert [state.value for state in SectionState] == [
        "not_started",
        "running",
        "complete",
        "failed",
        "skipped",
    ]


def test_progress_tracks_running_complete_counts_and_elapsed_time() -> None:
    progress = new_report_progress(
        provider="fake",
        model="fake-model",
        depth=DepthMode.QUICK,
        started_at=10.0,
    )
    first = progress.sections[0].id

    running = progress.mark_running(first, now=10.0)
    complete = running.mark_complete(first, now=13.5)

    assert running.active_section == first
    assert complete.active_section is None
    assert complete.completed_count == 1
    assert complete.total_count == len(progress.sections)
    assert complete.elapsed_seconds(now=14.0) == 4.0


def test_progress_captures_failure_and_latest_error() -> None:
    progress = new_report_progress(
        provider="fake",
        model="fake-model",
        depth=DepthMode.STANDARD,
    )
    first = progress.sections[0].id

    failed = progress.mark_failed(first, "provider failed", now=5.0)

    assert failed.sections[0].state is SectionState.FAILED
    assert failed.latest_error == "provider failed"


def test_progress_tracks_skipped_sections() -> None:
    progress = new_report_progress(
        provider="fake",
        model="fake-model",
        depth=DepthMode.STANDARD,
    )

    skipped = progress.mark_skipped("technical_indicators", "missing indicators")

    section = skipped.section("technical_indicators")
    assert section.state is SectionState.SKIPPED
    assert section.error == "missing indicators"


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
        client=FakeLLMClient(mode="fail_section"),
        context=context,
        depth=DepthMode.STANDARD,
    )

    failed = result.progress.section("bull_case")
    assert failed.state is SectionState.FAILED
    assert (
        result.progress.latest_error
        == "fake provider failed: fake failure for Bull Case"
    )
    assert result.progress.completed_count == result.progress.total_count - 1


def test_failed_section_visible() -> None:
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

    failed = result.progress.section("bull_case")
    assert failed.title == "Bull Case"
    assert failed.state is SectionState.FAILED
    assert failed.error == "fake provider failed: fake failure for Bull Case"
    assert "Bull Case" not in {section.title for section in result.sections}


def test_pipeline_skips_indicator_section_when_indicator_context_missing() -> None:
    context = ReportContext(
        symbol="SPY",
        market_summary="price flat",
        indicator_summary="",
    )

    result = run_report_pipeline(
        client=FakeLLMClient(mode="1"),
        context=context,
        depth=DepthMode.STANDARD,
    )

    assert result.progress.section("technical_indicators").state is SectionState.SKIPPED
    section_ids = {section.section_id for section in result.sections}
    assert "technical_indicators" not in section_ids


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
    assert result.progress.latest_error == "fake provider failed: Risk Notes"
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
    assert result.progress.latest_error == "openai provider unavailable: Risk Notes"
    assert result.sections == ()
