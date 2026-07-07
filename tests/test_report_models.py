from __future__ import annotations

import pytest
from pydantic import ValidationError

from finagent_terminal.reports.models import (
    DepthMode,
    ReportContext,
    ReportSectionPrompt,
    ReportSectionResult,
    section_specs_for_depth,
)


def test_depth_modes_are_exact_user_labels() -> None:
    assert [mode.value for mode in DepthMode] == ["Quick", "Standard", "Deep"]


def test_report_depth_selects_more_sections_as_depth_increases() -> None:
    quick = section_specs_for_depth(DepthMode.QUICK)
    standard = section_specs_for_depth(DepthMode.STANDARD)
    deep = section_specs_for_depth(DepthMode.DEEP)

    assert len(quick) < len(standard) < len(deep)
    assert "bull_case" in {section.id for section in standard}
    assert "scenario_watch" in {section.id for section in deep}


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
