from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

import httpx

from finagent_terminal.data.vendor_clients import FinnhubClient, FmpClient

if TYPE_CHECKING:
    from finagent_terminal.data.sources import SourceBundle

FMP_RESERVED_SAMPLE: Final = "fmp+/=?:&token"
FMP_SAMPLE: Final = "fmp-test-secret"
FINNHUB_RESERVED_SAMPLE: Final = "finnhub+/=?:&token"
FINNHUB_SAMPLE: Final = "finnhub-test-secret"

type VendorHandler = Callable[[httpx.Request], httpx.Response]


def fetch_fmp_bundle(handler: VendorHandler, symbol: str = "AAPL") -> SourceBundle:
    return FmpClient.from_env(
        {"FMP_API_KEY": FMP_SAMPLE},
        http_client=httpx.Client(
            base_url="https://financialmodelingprep.com",
            transport=httpx.MockTransport(handler),
        ),
    ).fetch_symbol(symbol)


def fetch_finnhub_bundle(
    handler: VendorHandler,
    symbol: str = "AAPL",
) -> SourceBundle:
    return FinnhubClient.from_env(
        {"FINNHUB_API_KEY": FINNHUB_SAMPLE},
        http_client=httpx.Client(
            base_url="https://finnhub.io",
            transport=httpx.MockTransport(handler),
        ),
    ).fetch_symbol(symbol)


def fmp_success_response(
    request: httpx.Request,
    requested_paths: list[str],
) -> httpx.Response:
    assert request.headers.get("apikey") == FMP_SAMPLE
    requested_paths.append(request.url.path)
    payloads = {
        "/stable/profile": [
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "mktCap": 3000000000000,
            },
        ],
        "/stable/income-statement": [
            {
                "symbol": "AAPL",
                "calendarYear": "2024",
                "revenue": 383000000000,
                "netIncome": 97000000000,
            },
        ],
        "/stable/balance-sheet-statement": [
            {
                "symbol": "AAPL",
                "calendarYear": "2024",
                "totalAssets": 352000000000,
                "totalLiabilities": 290000000000,
            },
        ],
        "/stable/cash-flow-statement": [
            {
                "symbol": "AAPL",
                "calendarYear": "2024",
                "operatingCashFlow": 110000000000,
                "freeCashFlow": 99000000000,
            },
        ],
        "/stable/earnings-calendar": [
            {
                "symbol": "AAPL",
                "date": "2024-02-01",
                "eps": 2.18,
                "epsEstimated": 2.1,
            },
        ],
        "/stable/news/stock": [
            {
                "symbol": "AAPL",
                "title": "Apple launches product",
                "text": "Launch summary",
                "url": "https://example.test/aapl",
                "publishedDate": "2024-02-02T13:00:00Z",
            },
        ],
    }
    payload = payloads.get(request.url.path)
    if payload is None:
        message = f"unexpected FMP path {request.url.path}"
        raise AssertionError(message)
    return httpx.Response(200, json=payload, request=request)


def finnhub_success_response(
    request: httpx.Request,
    requested_paths: list[str],
) -> httpx.Response:
    assert request.headers.get("X-Finnhub-Token") == FINNHUB_SAMPLE
    requested_paths.append(request.url.path)
    payloads = {
        "/api/v1/quote": {
            "c": 191.5,
            "d": 1.5,
            "dp": 0.79,
            "pc": 190.0,
            "t": 1700000000,
        },
        "/api/v1/company-news": [
            {
                "headline": "Apple supplier update",
                "summary": "Supplier summary",
                "url": "https://example.test/finnhub",
                "datetime": 1700000100,
            },
        ],
        "/api/v1/stock/social-sentiment": {
            "symbol": "AAPL",
            "reddit": [{"mention": 4, "positiveScore": 0.75, "negativeScore": 0.1}],
        },
        "/api/v1/stock/recommendation": [
            {
                "symbol": "AAPL",
                "period": "2024-01",
                "strongBuy": 10,
                "buy": 20,
                "hold": 6,
                "sell": 1,
                "strongSell": 0,
            },
        ],
        "/api/v1/stock/eps-estimate": {
            "symbol": "AAPL",
            "data": [{"period": "2024-03", "epsAvg": 2.31, "numberAnalysts": 23}],
        },
    }
    payload = payloads.get(request.url.path)
    if payload is None:
        message = f"unexpected Finnhub path {request.url.path}"
        raise AssertionError(message)
    return httpx.Response(200, json=payload, request=request)


def unexpected_request(request: httpx.Request) -> httpx.Response:
    message = f"unexpected request to {request.url}"
    raise AssertionError(message)
