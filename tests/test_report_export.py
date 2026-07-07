from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from finagent_terminal.reports.export import (
    ReportExport,
    render_markdown_report,
    write_markdown_report,
)
from finagent_terminal.reports.models import (
    DepthMode,
    ReportContext,
    ReportSectionResult,
)
from finagent_terminal.reports.pipeline import ReportRunResult
from finagent_terminal.reports.progress import new_report_progress

if TYPE_CHECKING:
    from pathlib import Path


def test_export_writes_one_consolidated_markdown_report(tmp_path: Path) -> None:
    context = ReportContext(
        symbol="AAPL",
        market_summary="last price 101.00 USD",
        indicator_summary="rsi_14 55.0",
        warnings=("missing profile beta",),
    )
    progress = new_report_progress(
        provider="openai",
        model="fake-model",
        depth=DepthMode.STANDARD,
    )
    result = ReportRunResult(
        sections=(
            ReportSectionResult(
                section_id="overview",
                title="Overview",
                summary="AAPL generated from supplied context.",
                key_points=("last price 101.00 USD",),
                risks=("missing profile beta",),
            ),
        ),
        progress=progress,
        timeline=(progress,),
    )

    path = write_markdown_report(
        ReportExport(
            context=context,
            result=result,
            created_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        ),
        tmp_path,
    )

    assert path.name == "aapl-standard-20240102T030405Z.md"
    assert path.is_file()
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]
    body = path.read_text(encoding="utf-8")
    assert "# AAPL Research Report" in body
    assert "## Disclaimer" in body
    assert "Research assistance only; not investment advice." in body
    assert "## Provenance" in body
    assert "- Provider: openai" in body
    assert "- Model: fake-model" in body
    assert "## Missing Data Warnings" in body
    assert "- missing profile beta" in body
    assert "## Overview" in body


def test_render_includes_failed_section_warning_without_section_tree() -> None:
    context = ReportContext(
        symbol="MSFT",
        market_summary="quote unavailable",
        warnings=("quote fetch failed",),
    )
    progress = new_report_progress(
        provider="anthropic",
        model="fake-model",
        depth=DepthMode.STANDARD,
    ).mark_failed("bull_case", "fake provider failed", now=1.0)
    result = ReportRunResult(sections=(), progress=progress, timeline=(progress,))

    body = render_markdown_report(
        ReportExport(
            context=context,
            result=result,
            created_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        ),
    )

    assert "- quote fetch failed" in body
    assert "- Bull Case: fake provider failed" in body
    assert "sections/" not in body
