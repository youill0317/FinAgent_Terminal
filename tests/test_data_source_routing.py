from __future__ import annotations

from collections import Counter

import httpx

from finagent_terminal.data.source_routing import (
    SourceRoute,
    route_source_bundle,
    route_ticker_sources,
)
from finagent_terminal.data.sources import SourceSession, SourceStatusValue
from tests.source_routing_fixtures import (
    SEC_SAMPLE,
    macro_response,
    macro_source_session,
    response_with_unrelated_failures,
    source_session,
    success_response,
)
from tests.vendor_source_fixtures import FINNHUB_SAMPLE, FMP_SAMPLE


def test_aapl_route_sources() -> None:
    # Given: a source session with successful SEC, FMP, Finnhub, and GDELT fakes.
    requests: list[httpx.Request] = []
    session = source_session(
        lambda request: success_response(request, requests, sec_has_aapl=True),
    )

    # When: ticker-specific source routes are built for AAPL.
    routes = route_ticker_sources(session, " aapl ")

    # Then: each role has compact labeled summaries without raw vendor payloads.
    labels = {label for route in routes for label in route.source_labels}
    summaries = "\n".join(line for route in routes for line in route.summary_lines)
    citations = tuple(citation for route in routes for citation in route.citations)

    assert {route.route for route in routes} == {
        SourceRoute.TICKER_NEWS,
        SourceRoute.FILINGS,
        SourceRoute.FUNDAMENTALS,
        SourceRoute.ESTIMATES,
        SourceRoute.SENTIMENT,
    }
    assert {"SEC", "FMP", "Finnhub", "GDELT"} <= labels
    assert "Apple Inc. company facts" in summaries
    assert "News: Apple supplier update" in summaries
    assert any(citation.url == "https://example.test/finnhub" for citation in citations)
    assert FMP_SAMPLE not in summaries
    assert FINNHUB_SAMPLE not in summaries
    assert SEC_SAMPLE not in summaries


def test_missing_vendor_keys() -> None:
    # Given: a session without FMP or Finnhub API keys.
    session = SourceSession(
        env={"FINAGENT_SEC_USER_AGENT": SEC_SAMPLE},
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: success_response(request, [], sec_has_aapl=True),
            ),
        ),
    )

    # When: the ticker-news route asks vendor-backed sources for AAPL context.
    route = route_source_bundle(session, SourceRoute.TICKER_NEWS, "AAPL")

    # Then: missing keys appear as warning strings and labeled unavailable statuses.
    assert {"FMP", "Finnhub", "GDELT"} <= set(route.source_labels)
    assert "FMP_API_KEY missing; source unavailable" in route.warnings
    assert "FINNHUB_API_KEY missing; source unavailable" in route.warnings
    assert any(
        status.status is SourceStatusValue.UNAVAILABLE
        and status.source in {"FMP", "Finnhub"}
        for status in route.bundle.statuses
    )


def test_duplicate_route_cache() -> None:
    # Given: a route-capable source session and request counter.
    requests: list[httpx.Request] = []
    session = source_session(
        lambda request: success_response(request, requests, sec_has_aapl=True),
    )

    # When: the same normalized ticker route is requested twice.
    first = route_source_bundle(session, SourceRoute.FUNDAMENTALS, "aapl")
    second = route_source_bundle(session, SourceRoute.FUNDAMENTALS, " AAPL ")

    # Then: SourceSession returns the cached route bundle and no endpoint repeats.
    path_counts = Counter(request.url.path for request in requests)
    assert first.bundle is second.bundle
    assert path_counts == Counter(
        {
            "/files/company_tickers.json": 1,
            "/api/xbrl/companyfacts/CIK0000320193.json": 1,
            "/stable/profile": 1,
            "/stable/income-statement": 1,
            "/stable/balance-sheet-statement": 1,
            "/stable/cash-flow-statement": 1,
        },
    )


def test_ticker_news_route_excludes_unrelated_vendor_endpoints() -> None:
    requests: list[httpx.Request] = []
    session = source_session(
        lambda request: success_response(request, requests, sec_has_aapl=True),
    )

    route = route_source_bundle(session, SourceRoute.TICKER_NEWS, "AAPL")

    assert Counter(request.url.path for request in requests) == Counter(
        {
            "/stable/news/stock": 1,
            "/api/v1/company-news": 1,
            "/api/v2/doc/doc": 1,
        },
    )
    assert all("HTTP 500" not in warning for warning in route.warnings)


def test_fundamentals_route_excludes_news_warning_leak() -> None:
    requests: list[httpx.Request] = []
    session = source_session(
        lambda request: response_with_unrelated_failures(request, requests),
    )

    route = route_source_bundle(session, SourceRoute.FUNDAMENTALS, "AAPL")

    assert Counter(request.url.path for request in requests) == Counter(
        {
            "/files/company_tickers.json": 1,
            "/api/xbrl/companyfacts/CIK0000320193.json": 1,
            "/stable/profile": 1,
            "/stable/income-statement": 1,
            "/stable/balance-sheet-statement": 1,
            "/stable/cash-flow-statement": 1,
        },
    )
    assert "FMP HTTP 500" not in route.warnings
    assert not any(item.title.startswith("News:") for item in route.bundle.items)


def test_estimates_route_uses_estimate_endpoints_only() -> None:
    requests: list[httpx.Request] = []
    session = source_session(
        lambda request: success_response(request, requests, sec_has_aapl=True),
    )

    route = route_source_bundle(session, SourceRoute.ESTIMATES, "AAPL")

    assert Counter(request.url.path for request in requests) == Counter(
        {
            "/stable/earnings-calendar": 1,
            "/api/v1/stock/recommendation": 1,
            "/api/v1/stock/eps-estimate": 1,
        },
    )
    assert {"Earnings calendar 2024-02-01: AAPL", "EPS estimate 2024-03: AAPL"} <= {
        item.title for item in route.bundle.items
    }


def test_ticker_news_route_excludes_estimate_warning_leak() -> None:
    requests: list[httpx.Request] = []
    session = source_session(
        lambda request: response_with_unrelated_failures(request, requests),
    )

    route = route_source_bundle(session, SourceRoute.TICKER_NEWS, "AAPL")

    assert Counter(request.url.path for request in requests) == Counter(
        {
            "/stable/news/stock": 1,
            "/api/v1/company-news": 1,
            "/api/v2/doc/doc": 1,
        },
    )
    assert "Finnhub HTTP 500" not in route.warnings
    assert not any(item.title.startswith("EPS estimate") for item in route.bundle.items)


def test_market_news_and_macro_events_have_explicit_routes() -> None:
    requests: list[httpx.Request] = []
    session = macro_source_session(lambda request: macro_response(request, requests))

    market = route_source_bundle(session, SourceRoute.MARKET_NEWS)
    macro = route_source_bundle(session, SourceRoute.MACRO_EVENTS)

    assert market.source_labels == ("GDELT",)
    assert {"FRED", "Treasury Fiscal Data", "GDELT"} <= set(macro.source_labels)
    assert Counter(request.url.path for request in requests) == Counter(
        {
            "/api/v2/doc/doc": 2,
            "/fred/series/observations": 1,
            "/services/api/fiscal_service/v1/accounting/od/rates_of_exchange": 1,
        },
    )


def test_unknown_route_returns_warning_without_requests() -> None:
    session = SourceSession(
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(AssertionError(request.url)),
            ),
        ),
    )

    route = route_source_bundle(session, "not_a_route", "AAPL")

    assert route.route is None
    assert route.bundle.requests == ()
    assert route.warnings == ("unknown source route not_a_route",)


def test_missing_cik() -> None:
    # Given: SEC ticker lookup resolves a different company only.
    session = source_session(
        lambda request: success_response(request, [], sec_has_aapl=False),
    )

    # When: a SEC-backed filing route is requested for AAPL.
    route = route_source_bundle(session, SourceRoute.FILINGS, "AAPL")

    # Then: the missing CIK is visible as a route warning, not a crash.
    assert route.source_labels == ("SEC",)
    assert route.bundle.items == ()
    assert route.warnings == ("missing SEC CIK for AAPL",)
    assert route.bundle.statuses[0].status is SourceStatusValue.WARNING


def test_malformed_input() -> None:
    # Given: a session that should not be reached for invalid input.
    session = SourceSession(
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(AssertionError(request.url)),
            ),
        ),
    )

    # When: a ticker-only route receives an empty ticker.
    route = route_source_bundle(session, SourceRoute.FILINGS, " ")

    # Then: the route returns a compact warning bundle without network requests.
    assert route.source_labels == ()
    assert route.bundle.items == ()
    assert route.bundle.requests == ()
    assert route.warnings == ("missing ticker for filings",)
