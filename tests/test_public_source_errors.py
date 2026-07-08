from __future__ import annotations

from typing import Final
from urllib.parse import quote, quote_plus

import httpx

from finagent_terminal.data.sources import (
    SourceBundle,
    SourceSession,
    SourceStatusValue,
)

FRED_SAMPLE: Final = "fred-secret"
FRED_RESERVED_SAMPLE: Final = "fred+/=&secret"
FRED_RESERVED_VARIANTS: Final = (
    FRED_RESERVED_SAMPLE,
    quote(FRED_RESERVED_SAMPLE, safe=""),
    quote_plus(FRED_RESERVED_SAMPLE, safe=""),
)
SEC_SAMPLE: Final = "sec-agent-secret"


def test_missing_fred_key_returns_unavailable_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = f"unexpected request: {request.url}"
        raise AssertionError(message)

    session = SourceSession(
        env={},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    bundle = session.fred_series_observations("GDP")

    assert bundle.items == ()
    assert bundle.requests == ()
    assert bundle.statuses[0].status is SourceStatusValue.UNAVAILABLE
    assert bundle.statuses[0].message == "missing FRED_API_KEY"
    assert bundle.statuses[0].throttle is not None


def test_sec_missing_cik_returns_warning_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"0": {"ticker": "MSFT", "cik_str": 789019, "title": "Microsoft"}},
        )

    session = SourceSession(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    bundle = session.sec_companyfacts("AAPL")

    assert [request.url.path for request in requests] == ["/files/company_tickers.json"]
    assert bundle.items == ()
    assert bundle.statuses[0].status is SourceStatusValue.WARNING
    assert bundle.statuses[0].throttle is not None
    assert bundle.requests[0].throttle is not None
    assert bundle.warnings == ("missing SEC CIK for AAPL",)


def test_source_http_error_returns_warning_status_and_throttle_metadata() -> None:
    env = {"FRED_API_KEY": FRED_SAMPLE}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == FRED_SAMPLE
        return httpx.Response(429, headers={"Retry-After": "60"}, json={})

    session = SourceSession(
        env=env,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    bundle = session.fred_series_observations("GDP")

    assert bundle.items == ()
    assert bundle.statuses[0].status is SourceStatusValue.WARNING
    assert bundle.statuses[0].message == "FRED HTTP 429"
    assert bundle.statuses[0].throttle is not None
    assert bundle.statuses[0].retry_after == "60"
    assert bundle.requests[0].throttle is not None
    assert bundle.requests[0].retry_after == "60"


def test_public_source_error_redacts_credentials() -> None:
    env = {
        "FRED_API_KEY": FRED_RESERVED_SAMPLE,
        "FINAGENT_SEC_USER_AGENT": SEC_SAMPLE,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.stlouisfed.org":
            assert request.url.params["api_key"] == FRED_RESERVED_SAMPLE
        if request.url.host == "www.sec.gov":
            assert request.headers["User-Agent"] == SEC_SAMPLE
        return httpx.Response(500, json={"error": "upstream failed"})

    session = SourceSession(
        env=env,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    fred = session.fred_series_observations("GDP")
    sec = session.sec_companyfacts("AAPL")
    visible_text = _visible_source_text(fred, sec)

    assert "api_key=<redacted>" in fred.requests[0].url
    for variant in (*FRED_RESERVED_VARIANTS, SEC_SAMPLE):
        assert variant not in fred.requests[0].url
        assert variant not in visible_text
    assert all(
        "500" in (status.message or "") for status in (*fred.statuses, *sec.statuses)
    )


def test_malformed_public_source_inputs_return_warnings_without_requests() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = f"unexpected request: {request.url}"
        raise AssertionError(message)

    session = SourceSession(
        env={"FRED_API_KEY": FRED_SAMPLE},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    sec = session.sec_companyfacts(" ")
    fred = session.fred_series_observations("")
    gdelt = session.gdelt_articles("\t")

    assert sec.statuses[0].status is SourceStatusValue.WARNING
    assert fred.statuses[0].status is SourceStatusValue.WARNING
    assert gdelt.statuses[0].status is SourceStatusValue.WARNING
    assert sec.requests == ()
    assert fred.requests == ()
    assert gdelt.requests == ()


def _visible_source_text(*bundles: SourceBundle) -> str:
    return "\n".join(bundle.model_dump_json() for bundle in bundles)
