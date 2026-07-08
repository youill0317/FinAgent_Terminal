from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Final, assert_never

from finagent_terminal.data.source_models import (
    SourceBundle,
    SourceItem,
    SourceStatus,
    SourceStatusValue,
)
from finagent_terminal.symbols import InvalidSymbolError, normalize_symbol

if TYPE_CHECKING:
    from finagent_terminal.data.sources import SourceSession

FRED_MACRO_SERIES: Final = "FEDFUNDS"
TREASURY_RATES_ENDPOINT: Final = "v1/accounting/od/rates_of_exchange"


@unique
class SourceRoute(StrEnum):
    MARKET_NEWS = "market_news"
    TICKER_NEWS = "ticker_news"
    MACRO_EVENTS = "macro_events"
    FILINGS = "filings"
    FUNDAMENTALS = "fundamentals"
    ESTIMATES = "estimates"
    SENTIMENT = "sentiment"


@dataclass(frozen=True, slots=True)
class SourceRouteBundle:
    route: SourceRoute | None
    bundle: SourceBundle
    source_labels: tuple[str, ...]
    warnings: tuple[str, ...]
    citations: tuple[SourceItem, ...]
    summary_lines: tuple[str, ...]


TICKER_SOURCE_ROUTES: Final = (
    SourceRoute.TICKER_NEWS,
    SourceRoute.FILINGS,
    SourceRoute.FUNDAMENTALS,
    SourceRoute.ESTIMATES,
    SourceRoute.SENTIMENT,
)


def route_ticker_sources(
    session: SourceSession,
    ticker: str,
) -> tuple[SourceRouteBundle, ...]:
    return tuple(
        route_source_bundle(session, route, ticker) for route in TICKER_SOURCE_ROUTES
    )


def route_source_bundle(
    session: SourceSession,
    route: SourceRoute | str,
    subject: str = "",
) -> SourceRouteBundle:
    parsed_route = _parse_route(route)
    if parsed_route is None:
        return _summarize(
            None,
            _warning_bundle("unknown_route", f"unknown source route {route}"),
        )
    cache_subject = _cache_subject(parsed_route, subject)
    if cache_subject is None:
        return _summarize(
            parsed_route,
            _warning_bundle(
                parsed_route.value,
                f"missing ticker for {parsed_route.value}",
            ),
        )
    bundle = session.get_or_fetch(
        ("source-route", parsed_route.value, cache_subject),
        lambda: _fetch_route_bundle(session, parsed_route, cache_subject),
    )
    return _summarize(parsed_route, bundle)


def _fetch_route_bundle(
    session: SourceSession,
    route: SourceRoute,
    subject: str,
) -> SourceBundle:
    match route:
        case SourceRoute.MARKET_NEWS:
            bundles = (session.gdelt_articles(subject),)
        case SourceRoute.TICKER_NEWS:
            bundles = (
                session.fmp_news(subject),
                session.finnhub_news(subject),
                session.gdelt_articles(subject),
            )
        case SourceRoute.MACRO_EVENTS:
            bundles = (
                session.fred_series_observations(FRED_MACRO_SERIES),
                session.treasury_fiscal_data(TREASURY_RATES_ENDPOINT, page_size=5),
                session.gdelt_events(subject),
            )
        case SourceRoute.FILINGS:
            bundles = (session.sec_submissions(subject),)
        case SourceRoute.FUNDAMENTALS:
            bundles = (
                session.sec_companyfacts(subject),
                session.fmp_fundamentals(subject),
            )
        case SourceRoute.ESTIMATES:
            bundles = (
                session.fmp_estimates(subject),
                session.finnhub_estimates(subject),
            )
        case SourceRoute.SENTIMENT:
            bundles = (
                session.finnhub_sentiment(subject),
                session.gdelt_events(subject),
            )
        case _:
            assert_never(route)
    return _combine(route, bundles)


def _parse_route(route: SourceRoute | str) -> SourceRoute | None:
    match route:
        case SourceRoute():
            return route
        case str() as raw:
            try:
                return SourceRoute(raw)
            except ValueError:
                return None
        case _:
            assert_never(route)


def _cache_subject(route: SourceRoute, subject: str) -> str | None:
    match route:
        case SourceRoute.MARKET_NEWS:
            return _query_or_default(subject, "market news stocks")
        case SourceRoute.MACRO_EVENTS:
            return _query_or_default(subject, "federal reserve inflation rates")
        case (
            SourceRoute.TICKER_NEWS
            | SourceRoute.FILINGS
            | SourceRoute.FUNDAMENTALS
            | SourceRoute.ESTIMATES
            | SourceRoute.SENTIMENT
        ):
            return _ticker_key(subject)
        case _:
            assert_never(route)


def _ticker_key(ticker: str) -> str | None:
    if ticker.strip() == "":
        return None
    try:
        return normalize_symbol(ticker)
    except InvalidSymbolError:
        return None


def _query_or_default(query: str, default: str) -> str:
    stripped = query.strip()
    return stripped or default


def _combine(route: SourceRoute, bundles: tuple[SourceBundle, ...]) -> SourceBundle:
    statuses = tuple(status for bundle in bundles for status in bundle.statuses)
    warnings = _warning_strings(bundles)
    return SourceBundle(
        source=route.value,
        items=tuple(item for bundle in bundles for item in bundle.items),
        statuses=statuses,
        warnings=warnings,
        requests=tuple(request for bundle in bundles for request in bundle.requests),
    )


def _summarize(route: SourceRoute | None, bundle: SourceBundle) -> SourceRouteBundle:
    warnings = _warning_strings((bundle,))
    summary_bundle = (
        bundle
        if bundle.warnings == warnings
        else bundle.model_copy(update={"warnings": warnings})
    )
    return SourceRouteBundle(
        route=route,
        bundle=summary_bundle,
        source_labels=_source_labels(route, bundle),
        warnings=warnings,
        citations=bundle.items,
        summary_lines=tuple(_summary_line(item) for item in bundle.items),
    )


def _source_labels(route: SourceRoute | None, bundle: SourceBundle) -> tuple[str, ...]:
    labels: list[str] = []
    for label in (
        *(item.source for item in bundle.items),
        *(status.source for status in bundle.statuses),
        *(request.source for request in bundle.requests),
    ):
        source_label = _source_label(label)
        if _is_route_label(route, source_label):
            continue
        if source_label not in labels:
            labels.append(source_label)
    return tuple(labels)


def _warning_strings(bundles: tuple[SourceBundle, ...]) -> tuple[str, ...]:
    warnings: list[str] = []
    for bundle in bundles:
        status_warnings = tuple(
            _status_warning(status) for status in bundle.statuses
        )
        for warning in (*bundle.warnings, *status_warnings):
            if warning is not None and warning not in warnings:
                warnings.append(warning)
    return tuple(warnings)


def _status_warning(status: SourceStatus) -> str | None:
    match status.status:
        case SourceStatusValue.AVAILABLE:
            return None
        case SourceStatusValue.UNAVAILABLE | SourceStatusValue.WARNING:
            return status.message
        case _:
            assert_never(status.status)


def _summary_line(item: SourceItem) -> str:
    if item.summary is None:
        return f"[{_source_label(item.source)}] {item.title}"
    return f"[{_source_label(item.source)}] {item.title}: {item.summary}"


def _source_label(source: str) -> str:
    if source == "SEC EDGAR":
        return "SEC"
    return source


def _is_route_label(route: SourceRoute | None, label: str) -> bool:
    if route is None:
        return label == "unknown_route"
    return label == route.value


def _warning_bundle(source: str, message: str) -> SourceBundle:
    return SourceBundle(
        source=source,
        statuses=(
            SourceStatus(
                source=source,
                status=SourceStatusValue.WARNING,
                message=message,
            ),
        ),
        warnings=(message,),
    )
