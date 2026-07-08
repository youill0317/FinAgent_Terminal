from __future__ import annotations

import pytest
from pydantic import ValidationError

from finagent_terminal.reports.models import (
    REPORT_ROLE_ORDER,
    DepthMode,
    ReportContext,
    ReportFinalEditorOutput,
    ReportRoleArtifact,
    ReportRoleId,
    ReportRoleStatus,
    ReportSectionPrompt,
    ReportSectionResult,
    ReportSourceNote,
    section_specs_for_depth,
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


def test_depth_modes_are_exact_user_labels() -> None:
    assert [mode.value for mode in DepthMode] == ["Quick", "Standard", "Deep"]


def test_report_role_order_matches_sequential_agent_titles() -> None:
    assert [role.title for role in REPORT_ROLE_ORDER] == EXPECTED_ROLE_TITLES


def test_report_depth_uses_same_sequential_role_order() -> None:
    quick = section_specs_for_depth(DepthMode.QUICK)
    standard = section_specs_for_depth(DepthMode.STANDARD)
    deep = section_specs_for_depth(DepthMode.DEEP)

    assert [section.title for section in quick] == EXPECTED_ROLE_TITLES
    assert [section.title for section in standard] == EXPECTED_ROLE_TITLES
    assert [section.title for section in deep] == EXPECTED_ROLE_TITLES


def test_report_role_artifact_valid_accepts_source_notes_and_warnings() -> None:
    artifact = ReportRoleArtifact(
        role_id=ReportRoleId.TECHNICAL_ANALYST,
        title="Technical Analyst",
        status=ReportRoleStatus.COMPLETE,
        body="Trend is constructive above the moving average.",
        source_notes=(
            ReportSourceNote(
                label="market snapshot",
                detail="Close and RSI were supplied by the report context.",
            ),
        ),
        warnings=("Indicators may be stale.",),
    )

    assert artifact.role_id is ReportRoleId.TECHNICAL_ANALYST
    assert artifact.status is ReportRoleStatus.COMPLETE
    assert artifact.source_notes[0].label == "market snapshot"


def test_report_role_artifact_invalid_role_rejects_unknown_ids_and_statuses() -> None:
    with pytest.raises(ValidationError):
        _ = ReportRoleArtifact.model_validate(
            {
                "role_id": "portfolio_manager",
                "title": "Portfolio Manager",
                "status": "complete",
                "body": "Not a configured role.",
            },
        )

    with pytest.raises(ValidationError):
        _ = ReportRoleArtifact.model_validate(
            {
                "role_id": "technical_analyst",
                "title": "Technical Analyst",
                "status": "done",
                "body": "Invalid status.",
            },
        )


def test_final_editor_output_missing_disclaimer_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = ReportFinalEditorOutput(
            title="AAPL Research Report",
            body="A final editor report body.",
            source_notes=(),
            warnings=(),
            disclaimer="Investment advice.",
        )


def test_report_section_result_requires_non_advice_disclaimer() -> None:
    with pytest.raises(ValidationError):
        _ = ReportSectionResult(
            section_id="overview",
            title="Overview",
            summary="Looks good",
            key_points=("revenue",),
            risks=(),
            disclaimer="Buy now.",
        )


def test_report_prompt_preserves_supplied_market_and_indicator_context() -> None:
    context = ReportContext(
        symbol="NVDA",
        market_summary="latest quote 100",
        indicator_summary="rsi neutral",
        warnings=("stale quote",),
    )
    prompt = ReportSectionPrompt(
        context=context,
        depth=DepthMode.STANDARD,
        section=section_specs_for_depth(DepthMode.STANDARD)[0],
    )

    rendered = prompt.render()

    assert "NVDA" in rendered
    assert "latest quote 100" in rendered
    assert "rsi neutral" in rendered
    assert "stale quote" in rendered
