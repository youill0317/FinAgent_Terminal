from __future__ import annotations

from typing import Final

import httpx

from finagent_terminal.data.sources import (
    SourceBundle,
    SourceSession,
    SourceStatusValue,
)

FRED_SAMPLE: Final = "fred-secret"
SEC_SAMPLE: Final = "sec-agent-secret"


def test_sec_success_uses_user_agent_and_padded_cik() -> None:
    requests: list[httpx.Request] = []
    env = {"FINAGENT_SEC_USER_AGENT": SEC_SAMPLE}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["User-Agent"] == SEC_SAMPLE
        if request.url.path == "/files/company_tickers.json":
            return httpx.Response(
                200,
                json={
                    "0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}
                },
            )
        assert request.url.path == "/api/xbrl/companyfacts/CIK0000320193.json"
        return httpx.Response(
            200,
            json={
                "entityName": "Apple Inc.",
                "facts": {
                    "us-gaap": {
                        "Revenues": {"units": {"USD": [{"fy": 2023, "val": 1}]}}
                    },
                },
            },
        )

    session = SourceSession(
        env=env,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    bundle = session.sec_companyfacts(" aapl ")

    assert [request.url.path for request in requests] == [
        "/files/company_tickers.json",
        "/api/xbrl/companyfacts/CIK0000320193.json",
    ]
    assert bundle.items[0].title == "Apple Inc. company facts"
    assert bundle.items[0].summary == "facts: us-gaap"
    assert bundle.statuses[0].status is SourceStatusValue.AVAILABLE
    _assert_public_metadata(bundle)
    assert SEC_SAMPLE not in bundle.model_dump_json()


def test_sec_cik_lookup_uses_normalized_session_cache() -> None:
    ticker_lookup_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal ticker_lookup_count
        if request.url.path == "/files/company_tickers.json":
            ticker_lookup_count += 1
            return httpx.Response(
                200,
                json={
                    "0": {"ticker": "AAPL", "cik_str": "320193", "title": "Apple Inc."}
                },
            )
        return httpx.Response(200, json={"entityName": "Apple Inc.", "facts": {}})

    session = SourceSession(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    first = session.sec_companyfacts("aapl")
    second = session.sec_companyfacts(" AAPL ")

    assert ticker_lookup_count == 1
    assert first.items[0].title == second.items[0].title


def test_fred_success_sends_key_and_redacts_metadata() -> None:
    env = {"FRED_API_KEY": FRED_SAMPLE}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.params["api_key"] == FRED_SAMPLE
        return httpx.Response(
            200,
            json={
                "observations": [
                    {"date": "2024-01-01", "value": "123.4"},
                    {"date": "2024-02-01", "value": "."},
                ],
            },
        )

    session = SourceSession(
        env=env,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    bundle = session.fred_series_observations(" gdp ")

    assert requests[0].url.path == "/fred/series/observations"
    assert bundle.items[0].title == "GDP observations"
    assert bundle.items[0].summary == "2024-01-01: 123.4"
    assert FRED_SAMPLE not in bundle.model_dump_json()
    assert "api_key=<redacted>" in bundle.requests[0].url
    _assert_public_metadata(bundle)


def test_treasury_success_returns_items_and_request_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            request.url.path
            == "/services/api/fiscal_service/v1/accounting/od/rates_of_exchange"
        )
        assert request.url.params["page[size]"] == "2"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "record_date": "2024-01-01",
                        "country": "Canada",
                        "exchange_rate": "1.3",
                    }
                ],
            },
        )

    session = SourceSession(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    bundle = session.treasury_fiscal_data(
        "v1/accounting/od/rates_of_exchange",
        page_size=2,
    )

    assert bundle.items[0].title == "Canada"
    assert bundle.items[0].summary == "record_date=2024-01-01; exchange_rate=1.3"
    assert bundle.requests[0].status_code == 200
    _assert_public_metadata(bundle)


def test_gdelt_success_returns_articles_and_doc_event_context() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/api/v2/doc/doc"
        query = request.url.params["query"]
        title = "Fed update" if query == "federal reserve" else "Rate decision"
        return httpx.Response(
            200,
            json={
                "articles": [
                    {
                        "title": title,
                        "url": "https://example.test/gdelt",
                        "seendate": "20240101120000",
                        "domain": "example.test",
                    }
                ]
            },
        )

    session = SourceSession(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    articles = session.gdelt_articles("federal reserve")
    events = session.gdelt_events("rate decision")

    assert [request.url.path for request in requests] == [
        "/api/v2/doc/doc",
        "/api/v2/doc/doc",
    ]
    assert articles.items[0].title == "Fed update"
    assert articles.items[0].url == "https://example.test/gdelt"
    assert events.items[0].title == "Event context: Rate decision"
    _assert_public_metadata(articles)
    _assert_public_metadata(events)


def _assert_public_metadata(bundle: SourceBundle) -> None:
    assert bundle.requests != ()
    for request in bundle.requests:
        assert request.throttle is not None
    for status in bundle.statuses:
        assert status.throttle is not None
