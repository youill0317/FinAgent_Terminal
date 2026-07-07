from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from re import sub
from typing import TYPE_CHECKING, Final

from finagent_terminal.reports.models import REPORT_DISCLAIMER, ReportContext
from finagent_terminal.reports.progress import SectionState

if TYPE_CHECKING:
    from pathlib import Path

    from finagent_terminal.reports.pipeline import ReportRunResult


@dataclass(frozen=True, slots=True)
class ReportExport:
    context: ReportContext
    result: ReportRunResult
    created_at: datetime


def render_markdown_report(export: ReportExport) -> str:
    progress = export.result.progress
    lines = [
        f"# {export.context.symbol} Research Report",
        "",
        "## Disclaimer",
        REPORT_DISCLAIMER,
        "",
        "## Provenance",
        f"- Symbol: {export.context.symbol}",
        f"- Provider: {progress.provider}",
        f"- Model: {progress.model}",
        f"- Depth: {progress.depth.value}",
        f"- Created: {_utc_stamp(export.created_at)}",
        "- Sources: supplied market data snapshot and computed indicators",
        "",
        "## Missing Data Warnings",
        *_warning_lines(export),
        "",
    ]
    for section in export.result.sections:
        lines.extend(
            [
                f"## {section.title}",
                section.summary,
                "",
                "Key points:",
                *(f"- {point}" for point in section.key_points),
                "",
                "Risks:",
                *(f"- {risk}" for risk in section.risks),
                "",
            ],
        )
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(export: ReportExport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _filename(export)
    _ = path.write_text(render_markdown_report(export), encoding="utf-8")
    return path


def _warning_lines(export: ReportExport) -> list[str]:
    warnings = list(export.context.warnings)
    for section in export.result.progress.sections:
        if section.state not in {SectionState.FAILED, SectionState.SKIPPED}:
            continue
        warnings.append(f"{section.title}: {section.error or section.state.value}")
    if not warnings:
        return ["- none supplied"]
    return [f"- {warning}" for warning in warnings]


def _filename(export: ReportExport) -> str:
    symbol = sub(r"[^a-z0-9_-]+", "-", export.context.symbol.lower()).strip("-")
    depth = export.result.progress.depth.value.lower()
    return f"{symbol}-{depth}-{_utc_stamp(export.created_at)}.md"


def _utc_stamp(value: datetime) -> str:
    created_at: Final = value.astimezone(UTC)
    return created_at.strftime("%Y%m%dT%H%M%SZ")
