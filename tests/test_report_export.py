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
    ReportFinalEditorOutput,
    ReportSectionResult,
    ReportSourceCitation,
    ReportSourceNote,
)
from finagent_terminal.reports.pipeline import ReportRunResult
from finagent_terminal.reports.progress import new_report_progress
from tests.vendor_source_fixtures import FMP_SAMPLE

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


def test_render_uses_final_editor_output_when_available() -> None:
    context = ReportContext(
        symbol="AAPL",
        market_summary="last price 101.00 USD",
    )
    progress = new_report_progress(
        provider="openai",
        model="fake-model",
        depth=DepthMode.STANDARD,
    )
    result = ReportRunResult(
        sections=(
            ReportSectionResult(
                section_id="technical_analyst",
                title="Technical Analyst",
                summary="Technical draft.",
                key_points=("draft only",),
            ),
        ),
        progress=progress,
        timeline=(progress,),
        final_report=ReportFinalEditorOutput(
            title="Reports Editor",
            body="## Reports Editor\nFinal editor markdown.",
        ),
    )

    body = render_markdown_report(
        ReportExport(
            context=context,
            result=result,
            created_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        ),
    )

    assert "## Reports Editor" in body
    assert "Final editor markdown." in body
    assert "## Technical Analyst" not in body


def test_export_source_citations() -> None:
    context = ReportContext(
        symbol="AAPL",
        market_summary="last price 101.00 USD",
        source_notes=(
            ReportSourceNote(
                label="Ticker news",
                detail="Sources: FMP, Finnhub, GDELT; Citations: 3",
            ),
        ),
        source_citations=(
            ReportSourceCitation(
                label="Finnhub",
                title="News: Apple supplier update",
                url="https://example.test/finnhub?token=secret",
                published_at=datetime(2024, 2, 2, 13, 0, tzinfo=UTC),
            ),
            ReportSourceCitation(
                label="SEC",
                title="AAPL submissions",
                url="https://www.sec.gov/submissions/CIK0000320193.json",
                published_at=None,
            ),
        ),
        warnings=("FRED_API_KEY missing; source unavailable",),
    )
    progress = new_report_progress(
        provider="openai",
        model="fake-model",
        depth=DepthMode.STANDARD,
    )
    result = ReportRunResult(sections=(), progress=progress, timeline=(progress,))

    body = render_markdown_report(
        ReportExport(
            context=context,
            result=result,
            created_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        ),
    )

    assert "## Source Context" in body
    assert "Ticker news: Sources: FMP, Finnhub, GDELT; Citations: 3" in body
    assert "## Source Citations" in body
    assert "Finnhub: News: Apple supplier update" in body
    assert "2024-02-02T13:00:00Z" in body
    assert "https://example.test/finnhub" in body
    assert "token=secret" not in body
    assert "SEC: AAPL submissions" in body
    assert "FRED_API_KEY missing; source unavailable" in body


def test_export_excludes_raw_vendor_payloads() -> None:
    sentinel_key = "GEMINI-SENTINEL-12345"
    context = ReportContext(
        symbol="AAPL",
        market_summary="last price 101.00 USD",
        warnings=(
            f"Gemini failed: https://example.test/model?key={sentinel_key}",
        ),
        source_notes=(
            ReportSourceNote(
                label="FMP",
                detail=(
                    "full article body and vendor JSON should not persist "
                    f"{FMP_SAMPLE}"
                ),
            ),
        ),
        source_citations=(
            ReportSourceCitation(
                label="FMP",
                title="FMP profile citation",
                url=f"https://financialmodelingprep.com/stable/profile?apikey={FMP_SAMPLE}",
                published_at=None,
            ),
        ),
    )
    progress = new_report_progress(
        provider="openai",
        model="fake-model",
        depth=DepthMode.STANDARD,
    )
    result = ReportRunResult(
        sections=(),
        progress=progress,
        timeline=(progress,),
        final_report=ReportFinalEditorOutput(
            title="Reports Editor",
            body="## Reports Editor\nFinal editor markdown.",
        ),
    )

    body = render_markdown_report(
        ReportExport(
            context=context,
            result=result,
            created_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        ),
    )

    assert "Final editor markdown." in body
    assert FMP_SAMPLE not in body
    generic_key_leaked = sentinel_key in body or "?key=" in body or "&key=" in body
    assert generic_key_leaked is False
    assert "apikey=" not in body
    assert "[redacted warning]" in body
    assert "full article body" not in body
    assert "vendor JSON" not in body
