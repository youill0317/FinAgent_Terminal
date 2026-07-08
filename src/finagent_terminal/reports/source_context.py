from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit, urlunsplit

from finagent_terminal.data.source_routing import (
    SourceRoute,
    SourceRouteBundle,
    route_source_bundle,
)
from finagent_terminal.reports.models import ReportSourceCitation, ReportSourceNote

if TYPE_CHECKING:
    from finagent_terminal.data.source_models import SourceItem
    from finagent_terminal.data.sources import SourceSession

REPORT_SOURCE_ROUTES: Final = (
    SourceRoute.TICKER_NEWS,
    SourceRoute.FILINGS,
    SourceRoute.FUNDAMENTALS,
    SourceRoute.ESTIMATES,
    SourceRoute.SENTIMENT,
    SourceRoute.MACRO_EVENTS,
)
_SENSITIVE_URL_MARKERS: Final = (
    "apikey",
    "api_key",
    "access_token",
    "token=",
    "secret",
)


@dataclass(frozen=True, slots=True)
class ReportSourceContext:
    notes: tuple[ReportSourceNote, ...]
    citations: tuple[ReportSourceCitation, ...]
    warnings: tuple[str, ...]


def fetch_report_source_context(
    session: SourceSession,
    symbol: str,
) -> ReportSourceContext:
    session.clear()
    routes = tuple(
        route_source_bundle(session, route, symbol) for route in REPORT_SOURCE_ROUTES
    )
    return ReportSourceContext(
        notes=tuple(_source_note(route) for route in routes),
        citations=_source_citations(routes),
        warnings=_source_warnings(routes),
    )


def _source_note(route: SourceRouteBundle) -> ReportSourceNote:
    labels = ", ".join(route.source_labels) if route.source_labels else "none"
    detail = f"Sources: {labels}; Citations: {len(route.citations)}"
    titles = "; ".join(item.title for item in route.citations[:3])
    if titles:
        detail = f"{detail}; Items: {titles}"
    return ReportSourceNote(label=_route_title(route), detail=detail)


def _source_citations(
    routes: tuple[SourceRouteBundle, ...],
) -> tuple[ReportSourceCitation, ...]:
    citations: list[ReportSourceCitation] = []
    seen: set[tuple[str, str, str | None]] = set()
    for route in routes:
        for item in route.citations:
            citation = _citation(item)
            key = (citation.label, citation.title, citation.url)
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
    return tuple(citations)


def _citation(item: SourceItem) -> ReportSourceCitation:
    return ReportSourceCitation(
        label=_source_label(item.source),
        title=item.title,
        url=_safe_url(item.url),
        published_at=item.published_at,
    )


def _source_warnings(routes: tuple[SourceRouteBundle, ...]) -> tuple[str, ...]:
    warnings: list[str] = []
    for route in routes:
        for warning in route.warnings:
            if warning not in warnings:
                warnings.append(warning)
    return tuple(warnings)


def _route_title(route: SourceRouteBundle) -> str:
    if route.route is None:
        return "Unknown source route"
    text = route.route.value.replace("_", " ")
    return text[:1].upper() + text[1:]


def _source_label(source: str) -> str:
    if source == "SEC EDGAR":
        return "SEC"
    return source


def _safe_url(raw_url: str | None) -> str | None:
    if raw_url is None:
        return None
    parts = urlsplit(raw_url)
    if parts.scheme == "" or parts.netloc == "":
        return None
    sanitized = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    lowered = sanitized.lower()
    if any(marker in lowered for marker in _SENSITIVE_URL_MARKERS):
        return None
    return sanitized
