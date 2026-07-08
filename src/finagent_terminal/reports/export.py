from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from re import sub
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit, urlunsplit

from finagent_terminal.reports.models import (
    REPORT_DISCLAIMER,
    ReportContext,
    ReportFinalEditorOutput,
    ReportSourceCitation,
    ReportSourceNote,
)
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
        f"- Sources: {_sources_text(export.context)}",
        "",
        "## Source Context",
        *_source_context_lines(export.context.source_notes),
        "",
        "## Source Citations",
        *_citation_lines(export.context.source_citations),
        "",
        "## Missing Data Warnings",
        *_warning_lines(export),
        "",
    ]
    if export.result.final_report is not None:
        lines.extend(_final_report_lines(export.result.final_report))
        return "\n".join(lines).rstrip() + "\n"
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
    return [f"- {_safe_warning(warning)}" for warning in warnings]


def _sources_text(context: ReportContext) -> str:
    labels: list[str] = []
    for item in (*context.source_notes, *context.source_citations):
        if item.label not in labels:
            labels.append(item.label)
    if not labels:
        return "supplied market data snapshot and computed indicators"
    return (
        "supplied market data snapshot, computed indicators, "
        + ", ".join(labels)
    )


def _source_context_lines(notes: tuple[ReportSourceNote, ...]) -> list[str]:
    if not notes:
        return ["- none supplied"]
    lines: list[str] = []
    for note in notes:
        detail = _safe_note_detail(note.detail)
        if detail is None:
            lines.append(f"- {note.label}")
        else:
            lines.append(f"- {note.label}: {detail}")
    return lines


def _citation_lines(citations: tuple[ReportSourceCitation, ...]) -> list[str]:
    if not citations:
        return ["- none supplied"]
    return [f"- {_citation_line(citation)}" for citation in citations]


def _citation_line(citation: ReportSourceCitation) -> str:
    details = [
        detail
        for detail in (_safe_url(citation.url), _citation_timestamp(citation))
        if detail is not None
    ]
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"{citation.label}: {citation.title}{suffix}"


def _final_report_lines(final_report: ReportFinalEditorOutput) -> list[str]:
    if final_report.body.lstrip().startswith("#"):
        return [final_report.body, ""]
    return [f"## {final_report.title}", final_report.body, ""]


def _filename(export: ReportExport) -> str:
    symbol = sub(r"[^a-z0-9_-]+", "-", export.context.symbol.lower()).strip("-")
    depth = export.result.progress.depth.value.lower()
    return f"{symbol}-{depth}-{_utc_stamp(export.created_at)}.md"


def _utc_stamp(value: datetime) -> str:
    created_at: Final = value.astimezone(UTC)
    return created_at.strftime("%Y%m%dT%H%M%SZ")


def _citation_timestamp(citation: ReportSourceCitation) -> str | None:
    if citation.published_at is None:
        return None
    return citation.published_at.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _safe_url(raw_url: str | None) -> str | None:
    if raw_url is None:
        return None
    parts = urlsplit(raw_url)
    if parts.scheme == "" or parts.netloc == "":
        return None
    sanitized = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    if _looks_sensitive(sanitized):
        return None
    return sanitized


def _safe_note_detail(detail: str) -> str | None:
    if "\n" in detail or _looks_sensitive(detail):
        return None
    return detail


def _safe_warning(warning: str) -> str:
    if _looks_sensitive(warning):
        return "[redacted warning]"
    return warning


def _looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "apikey=",
            "api_key=",
            "?key=",
            "&key=",
            "access_token",
            "token=",
            "secret",
            "vendor json",
            "full article body",
            "{",
            "}",
        )
    )
