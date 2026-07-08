from __future__ import annotations

from finagent_terminal.reports.models import (
    DepthMode,
    ReportRoleArtifact,
    ReportRoleId,
    ReportRoleStatus,
)
from finagent_terminal.reports.progress import (
    SectionState,
    new_report_progress,
)

EXPECTED_ROLE_TITLES = [
    "Technical Analyst",
    "Fundamentals Analyst",
    "News/Sentiment Analyst",
    "Bullish Researcher",
    "Bearish Researcher",
    "Debate/Refinement",
    "Reports Editor",
]


def test_section_states_are_exact_user_labels() -> None:
    assert [state.value for state in SectionState] == [
        "not_started",
        "running",
        "complete",
        "failed",
        "skipped",
    ]


def test_report_progress_role_order_matches_sequential_agents() -> None:
    progress = new_report_progress(
        provider="fake",
        model="fake-model",
        depth=DepthMode.QUICK,
        started_at=10.0,
    )

    assert [role.title for role in progress.roles] == EXPECTED_ROLE_TITLES


def test_progress_artifact_valid_snapshots_are_isolated() -> None:
    first = new_report_progress(
        provider="fake",
        model="fake-model",
        depth=DepthMode.QUICK,
        started_at=10.0,
    )
    second = new_report_progress(
        provider="fake",
        model="fake-model",
        depth=DepthMode.QUICK,
        started_at=10.0,
    )
    artifact = ReportRoleArtifact(
        role_id=ReportRoleId.TECHNICAL_ANALYST,
        title="Technical Analyst",
        status=ReportRoleStatus.COMPLETE,
        body="Technical view.",
        source_notes=(),
        warnings=("stale indicator",),
    )

    complete = first.mark_complete("technical_analyst", now=11.0, artifact=artifact)

    assert first.roles[0].artifact is None
    assert second.roles[0].artifact is None
    assert complete.roles[0].artifact == artifact
    complete_artifact = complete.roles[0].artifact
    assert complete_artifact is not None
    assert complete_artifact.warnings == ("stale indicator",)


def test_progress_accepts_legacy_section_ids_for_export_compatibility() -> None:
    progress = new_report_progress(
        provider="fake",
        model="fake-model",
        depth=DepthMode.STANDARD,
    )

    failed = progress.mark_failed("bull_case", "provider failed", now=5.0)

    section = failed.section("bull_case")
    assert section.title == "Bull Case"
    assert section.state is SectionState.FAILED
    assert section.error == "provider failed"


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

    skipped = progress.mark_skipped("technical_analyst", "missing indicators")

    section = skipped.section("technical_analyst")
    assert section.state is SectionState.SKIPPED
    assert section.error == "missing indicators"
