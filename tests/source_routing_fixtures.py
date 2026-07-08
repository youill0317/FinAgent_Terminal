from __future__ import annotations

from collections.abc import Callable
from typing import Final

import httpx

from finagent_terminal.data.sources import SourceSession
from tests.vendor_source_fixtures import (
    FINNHUB_SAMPLE,
    FMP_SAMPLE,
    finnhub_success_response,
    fmp_success_response,
)

SEC_SAMPLE: Final = "sec-agent-secret"

type SourceHandler = Callable[[httpx.Request], httpx.Response]


def source_session(handler: SourceHandler) -> SourceSession:
    return SourceSession(
        env={
            "FINAGENT_SEC_USER_AGENT": SEC_SAMPLE,
            "FMP_API_KEY": FMP_SAMPLE,
            "FINNHUB_API_KEY": FINNHUB_SAMPLE,
        },
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def macro_source_session(handler: SourceHandler) -> SourceSession:
    return SourceSession(
        env={"FRED_API_KEY": "fred-test-secret"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def success_response(
    request: httpx.Request,
    requests: list[httpx.Request],
    *,
    sec_has_aapl: bool,
) -> httpx.Response:
    requests.append(request)
    return success_response_without_tracking(request, sec_has_aapl=sec_has_aapl)


def response_with_unrelated_failures(
    request: httpx.Request,
    requests: list[httpx.Request],
) -> httpx.Response:
    requests.append(request)
    if request.url.path in {
        "/stable/news/stock",
        "/api/v1/stock/recommendation",
        "/api/v1/stock/eps-estimate",
    }:
        return httpx.Response(500, json={"error": "unrelated failure"}, request=request)
    return success_response_without_tracking(request, sec_has_aapl=True)


def success_response_without_tracking(
    request: httpx.Request,
    *,
    sec_has_aapl: bool,
) -> httpx.Response:
    if request.url.host in {"www.sec.gov", "data.sec.gov"}:
        return sec_response(request, sec_has_aapl=sec_has_aapl)
    if request.url.host == "financialmodelingprep.com":
        return fmp_success_response(request, [])
    if request.url.host == "finnhub.io":
        return finnhub_success_response(request, [])
    if request.url.host == "api.gdeltproject.org":
        return gdelt_response(request)
    message = f"unexpected source host {request.url.host}"
    raise AssertionError(message)


def sec_response(request: httpx.Request, *, sec_has_aapl: bool) -> httpx.Response:
    if request.url.path == "/files/company_tickers.json":
        ticker = "AAPL" if sec_has_aapl else "MSFT"
        return httpx.Response(
            200,
            json={"0": {"ticker": ticker, "cik_str": 320193, "title": "Apple Inc."}},
            request=request,
        )
    if request.url.path == "/api/xbrl/companyfacts/CIK0000320193.json":
        return httpx.Response(
            200,
            json={"entityName": "Apple Inc.", "facts": {"us-gaap": {}}},
            request=request,
        )
    if request.url.path == "/submissions/CIK0000320193.json":
        return httpx.Response(200, json={"name": "Apple Inc."}, request=request)
    message = f"unexpected SEC path {request.url.path}"
    raise AssertionError(message)


def macro_response(
    request: httpx.Request,
    requests: list[httpx.Request],
) -> httpx.Response:
    requests.append(request)
    if request.url.host == "api.gdeltproject.org":
        return gdelt_response(request)
    if request.url.host == "api.stlouisfed.org":
        return httpx.Response(
            200,
            json={"observations": [{"date": "2024-01-01", "value": "5.33"}]},
            request=request,
        )
    if request.url.host == "api.fiscaldata.treasury.gov":
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "record_date": "2024-01-01",
                        "country": "Canada",
                        "exchange_rate": "1.3",
                    },
                ],
            },
            request=request,
        )
    message = f"unexpected macro host {request.url.host}"
    raise AssertionError(message)


def gdelt_response(request: httpx.Request) -> httpx.Response:
    query = request.url.params["query"]
    return httpx.Response(
        200,
        json={
            "articles": [
                {
                    "title": f"{query} context",
                    "url": "https://example.test/gdelt",
                    "seendate": "20240101120000",
                    "domain": "example.test",
                },
            ],
        },
        request=request,
    )
