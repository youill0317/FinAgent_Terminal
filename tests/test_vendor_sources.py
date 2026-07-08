from __future__ import annotations

from urllib.parse import quote

import httpx

from finagent_terminal.data.sources import SourceStatusValue, source_availability_lines
from finagent_terminal.data.vendor_clients import FinnhubClient, FmpClient
from tests.vendor_source_fixtures import (
    FINNHUB_RESERVED_SAMPLE,
    FINNHUB_SAMPLE,
    FMP_RESERVED_SAMPLE,
    FMP_SAMPLE,
    fetch_finnhub_bundle,
    fetch_fmp_bundle,
    finnhub_success_response,
    fmp_success_response,
    unexpected_request,
)


def test_fmp_success() -> None:
    requested_paths: list[str] = []
    bundle = fetch_fmp_bundle(
        lambda request: fmp_success_response(request, requested_paths),
        "aapl",
    )

    assert bundle.source == "FMP"
    assert {item.title for item in bundle.items} == {
        "Profile: Apple Inc.",
        "Income statement 2024: AAPL",
        "Balance sheet 2024: AAPL",
        "Cash flow 2024: AAPL",
        "Earnings calendar 2024-02-01: AAPL",
        "News: Apple launches product",
    }
    assert requested_paths == [
        "/stable/profile",
        "/stable/income-statement",
        "/stable/balance-sheet-statement",
        "/stable/cash-flow-statement",
        "/stable/earnings-calendar",
        "/stable/news/stock",
    ]
    assert bundle.statuses[0].status is SourceStatusValue.AVAILABLE
    assert FMP_SAMPLE not in bundle.model_dump_json()


def test_finnhub_success() -> None:
    requested_paths: list[str] = []
    bundle = fetch_finnhub_bundle(
        lambda request: finnhub_success_response(request, requested_paths),
        "aapl",
    )

    assert bundle.source == "Finnhub"
    assert {item.title for item in bundle.items} == {
        "Quote: AAPL",
        "News: Apple supplier update",
        "Sentiment: AAPL",
        "Recommendations 2024-01: AAPL",
        "EPS estimate 2024-03: AAPL",
    }
    assert requested_paths == [
        "/api/v1/quote",
        "/api/v1/company-news",
        "/api/v1/stock/social-sentiment",
        "/api/v1/stock/recommendation",
        "/api/v1/stock/eps-estimate",
    ]
    assert bundle.statuses[0].status is SourceStatusValue.AVAILABLE
    assert FINNHUB_SAMPLE not in bundle.model_dump_json()


def test_vendor_source_redacts_credentials_fmp_finnhub() -> None:
    seen_credentials: list[str] = []

    def fmp_handler(request: httpx.Request) -> httpx.Response:
        seen_credentials.append(request.headers["apikey"])
        return fmp_success_response(request, [])

    def finnhub_handler(request: httpx.Request) -> httpx.Response:
        seen_credentials.append(request.headers["X-Finnhub-Token"])
        return finnhub_success_response(request, [])

    fmp_bundle = FmpClient.from_env(
        {"FMP_API_KEY": FMP_SAMPLE},
        http_client=httpx.Client(
            base_url="https://financialmodelingprep.com",
            transport=httpx.MockTransport(fmp_handler),
        ),
    ).fetch_symbol("AAPL")
    finnhub_bundle = FinnhubClient.from_env(
        {"FINNHUB_API_KEY": FINNHUB_SAMPLE},
        http_client=httpx.Client(
            base_url="https://finnhub.io",
            transport=httpx.MockTransport(finnhub_handler),
        ),
    ).fetch_symbol("AAPL")

    visible_text = "\n".join(
        (
            fmp_bundle.model_dump_json(),
            finnhub_bundle.model_dump_json(),
            *source_availability_lines(
                {"FMP_API_KEY": FMP_SAMPLE, "FINNHUB_API_KEY": FINNHUB_SAMPLE},
            ),
        ),
    )
    assert FMP_SAMPLE in seen_credentials
    assert FINNHUB_SAMPLE in seen_credentials
    assert FMP_SAMPLE not in visible_text
    assert FINNHUB_SAMPLE not in visible_text

def test_vendor_source_redacts_credentials_url_encoded_fmp_finnhub() -> None:
    seen_credentials: list[str] = []
    fmp_encoded = quote(FMP_RESERVED_SAMPLE, safe="")
    finnhub_encoded = quote(FINNHUB_RESERVED_SAMPLE, safe="")

    def fmp_handler(request: httpx.Request) -> httpx.Response:
        seen_credentials.append(request.headers["apikey"])
        message = f"raw {FMP_RESERVED_SAMPLE} encoded {fmp_encoded}"
        raise httpx.ConnectError(message, request=request)

    def finnhub_handler(request: httpx.Request) -> httpx.Response:
        seen_credentials.append(request.headers["X-Finnhub-Token"])
        message = f"raw {FINNHUB_RESERVED_SAMPLE} encoded {finnhub_encoded}"
        raise httpx.ConnectError(message, request=request)

    fmp_bundle = FmpClient.from_env(
        {"FMP_API_KEY": FMP_RESERVED_SAMPLE},
        http_client=httpx.Client(
            base_url="https://financialmodelingprep.com",
            transport=httpx.MockTransport(fmp_handler),
        ),
    ).fetch_symbol("AAPL")
    finnhub_bundle = FinnhubClient.from_env(
        {"FINNHUB_API_KEY": FINNHUB_RESERVED_SAMPLE},
        http_client=httpx.Client(
            base_url="https://finnhub.io",
            transport=httpx.MockTransport(finnhub_handler),
        ),
    ).fetch_symbol("AAPL")

    visible_text = "\n".join(
        (
            fmp_bundle.model_dump_json(),
            finnhub_bundle.model_dump_json(),
            *source_availability_lines(
                {
                    "FMP_API_KEY": FMP_RESERVED_SAMPLE,
                    "FINNHUB_API_KEY": FINNHUB_RESERVED_SAMPLE,
                },
            ),
        ),
    )
    assert FMP_RESERVED_SAMPLE in seen_credentials
    assert FINNHUB_RESERVED_SAMPLE in seen_credentials
    assert FMP_RESERVED_SAMPLE not in visible_text
    assert fmp_encoded not in visible_text
    assert FINNHUB_RESERVED_SAMPLE not in visible_text
    assert finnhub_encoded not in visible_text
    assert "<redacted>" in visible_text


def test_fmp_missing_key() -> None:
    client = FmpClient.from_env(
        {},
        http_client=httpx.Client(transport=httpx.MockTransport(unexpected_request)),
    )

    bundle = client.fetch_symbol("AAPL")

    assert bundle.items == ()
    assert bundle.statuses[0].status is SourceStatusValue.UNAVAILABLE
    assert bundle.statuses[0].message == "FMP_API_KEY missing; source unavailable"


def test_finnhub_missing_key() -> None:
    client = FinnhubClient.from_env(
        {},
        http_client=httpx.Client(transport=httpx.MockTransport(unexpected_request)),
    )

    bundle = client.fetch_symbol("AAPL")

    assert bundle.items == ()
    assert bundle.statuses[0].status is SourceStatusValue.UNAVAILABLE
    assert bundle.statuses[0].message == "FINNHUB_API_KEY missing; source unavailable"


def test_vendor_unauthorized_fmp() -> None:
    bundle = fetch_fmp_bundle(
        lambda request: httpx.Response(
            401,
            json={"message": f"bad key {FMP_SAMPLE}"},
            request=request,
        ),
    )

    assert bundle.items == ()
    assert bundle.statuses[0].status is SourceStatusValue.UNAVAILABLE
    assert "401" in (bundle.statuses[0].message or "")
    assert FMP_SAMPLE not in bundle.model_dump_json()


def test_malformed_vendor_payload_fmp() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/stable/profile":
            return httpx.Response(200, json=[{"bad": "shape"}], request=request)
        malformed_payloads = {
            "/stable/income-statement": {"symbol": "AAPL", "totalAssets": 1},
            "/stable/balance-sheet-statement": {"symbol": "AAPL", "revenue": 1},
            "/stable/cash-flow-statement": {"symbol": "AAPL", "revenue": 1},
            "/stable/earnings-calendar": {"symbol": "AAPL", "date": "2024-02-01"},
        }
        payload = malformed_payloads.get(request.url.path)
        if payload is not None:
            return httpx.Response(200, json=[payload], request=request)
        return fmp_success_response(request, [])

    bundle = fetch_fmp_bundle(handler)
    item_titles = {item.title for item in bundle.items}

    assert any(status.status is SourceStatusValue.WARNING for status in bundle.statuses)
    warning_count = sum(
        status.status is SourceStatusValue.WARNING for status in bundle.statuses
    )
    assert warning_count >= 5
    assert "Profile: Apple Inc." not in item_titles
    assert not any(title.startswith("Income statement") for title in item_titles)
    assert not any(title.startswith("Balance sheet") for title in item_titles)
    assert not any(title.startswith("Cash flow") for title in item_titles)
    assert not any(title.startswith("Earnings calendar") for title in item_titles)
    assert "News: Apple launches product" in item_titles

def test_malformed_vendor_payload_finnhub() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/quote":
            return httpx.Response(200, json={"bad": "shape"}, request=request)
        if request.url.path == "/api/v1/stock/social-sentiment":
            return httpx.Response(
                200,
                json={"symbol": "AAPL", "reddit": [{"bad": "shape"}]},
                request=request,
            )
        if request.url.path == "/api/v1/stock/eps-estimate":
            return httpx.Response(
                200,
                json={"symbol": "AAPL", "data": [{"period": "2024-03"}]},
                request=request,
            )
        return finnhub_success_response(request, [])

    bundle = fetch_finnhub_bundle(handler)
    item_titles = {item.title for item in bundle.items}

    assert any(status.status is SourceStatusValue.WARNING for status in bundle.statuses)
    warning_count = sum(
        status.status is SourceStatusValue.WARNING for status in bundle.statuses
    )
    assert warning_count >= 3
    assert "Quote: AAPL" not in item_titles
    assert "Sentiment: AAPL" not in item_titles
    assert not any(title.startswith("EPS estimate") for title in item_titles)
    assert "News: Apple supplier update" in item_titles


def test_vendor_error_redacts_credentials_fmp() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = f"boom {FMP_SAMPLE}"
        raise httpx.ConnectError(message, request=request)

    bundle = fetch_fmp_bundle(handler)

    assert bundle.items == ()
    assert bundle.statuses[0].status is SourceStatusValue.WARNING
    assert "<redacted>" in (bundle.statuses[0].message or "")
    assert FMP_SAMPLE not in bundle.model_dump_json()
