from __future__ import annotations

from dataclasses import replace

import httpx
import pytest
from textual.app import App
from textual.widgets import Input, Static

from finagent_terminal.data.sources import SourceSession
from finagent_terminal.ui.screens.news import NewsScreen
from finagent_terminal.ui.state import ReportProgress, create_app_state
from tests.source_routing_fixtures import (
    macro_response,
    source_session,
    success_response,
)
from tests.vendor_source_fixtures import FINNHUB_SAMPLE, FMP_SAMPLE


class NewsHarness(App[None]):
    def __init__(self, screen: NewsScreen) -> None:
        super().__init__()
        self.news_screen: NewsScreen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.news_screen)


def news_response(
    request: httpx.Request,
    requests: list[httpx.Request],
) -> httpx.Response:
    if request.url.host in {
        "api.fiscaldata.treasury.gov",
        "api.stlouisfed.org",
    }:
        return macro_response(request, requests)
    return success_response(request, requests, sec_has_aapl=True)


def _owned_source_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[httpx.Client, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: news_response(request, requests),
        ),
    )

    def make_client(
        *,
        timeout: httpx.Timeout,
        limits: httpx.Limits,
        follow_redirects: bool,
    ) -> httpx.Client:
        assert timeout is not None
        assert limits is not None
        assert follow_redirects is True
        return client

    monkeypatch.setattr(httpx, "Client", make_client)
    return client, requests


@pytest.mark.anyio
async def test_renders_fake_aapl_news() -> None:
    requests: list[httpx.Request] = []
    screen = NewsScreen(
        create_app_state(),
        source_session(lambda request: news_response(request, requests)),
    )

    async with NewsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen.query_one("#news_ticker", Input).value = "aapl"
        _ = await pilot.click("#news_refresh")
        await pilot.pause()
        rendered = str(screen.query_one("#news_body", Static).render())

    assert "Ticker/query: AAPL" in rendered
    assert "Market news" in rendered
    assert "Ticker news" in rendered
    assert "Macro/events" in rendered
    assert "Filings" in rendered
    assert "Sources: FMP, Finnhub, GDELT" in rendered
    assert "Sources: SEC" in rendered
    assert "News: Apple supplier update" in rendered
    assert "News: Apple launches product" in rendered
    assert "Finnhub: News: Apple supplier update" in rendered
    assert "AAPL context" in rendered
    assert "AAPL submissions" in rendered
    assert "https://example.test/finnhub" in rendered
    assert "2024-02-02 13:00 UTC" in rendered
    assert FMP_SAMPLE not in rendered
    assert FINNHUB_SAMPLE not in rendered


@pytest.mark.anyio
async def test_missing_keys_render_unavailable_source_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FMP_API_KEY", FMP_SAMPLE)
    monkeypatch.setenv("FINNHUB_API_KEY", FINNHUB_SAMPLE)
    monkeypatch.setenv("FRED_API_KEY", "fred-test-secret")
    screen = NewsScreen(
        create_app_state(),
        SourceSession(
            env={},
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: news_response(request, []),
                ),
            ),
        ),
    )

    async with NewsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _ = await pilot.click("#news_refresh")
        await pilot.pause()
        rendered = str(screen.query_one("#news_body", Static).render())

    assert "FMP_API_KEY missing; source unavailable" in rendered
    assert "FINNHUB_API_KEY missing; source unavailable" in rendered
    assert "missing FRED_API_KEY" in rendered
    assert "Market news" in rendered
    assert "Macro/events" in rendered


@pytest.mark.anyio
async def test_source_failure_stays_refreshable_without_stale_duplicates() -> None:
    calls: list[str] = []

    def source_failure(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500, json={"error": "down"}, request=request)

    screen = NewsScreen(
        create_app_state(),
        SourceSession(
            env={
                "FINAGENT_SEC_USER_AGENT": "sec-test-agent",
                "FMP_API_KEY": FMP_SAMPLE,
                "FINNHUB_API_KEY": FINNHUB_SAMPLE,
                "FRED_API_KEY": "fred-test-secret",
            },
            http_client=httpx.Client(transport=httpx.MockTransport(source_failure)),
        ),
    )

    async with NewsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _ = await pilot.click("#news_refresh")
        await pilot.pause()
        first = str(screen.query_one("#news_body", Static).render())
        _ = await pilot.click("#news_refresh")
        await pilot.pause()
        second = str(screen.query_one("#news_body", Static).render())

    assert "FMP HTTP 500" in second
    assert "Finnhub HTTP 500" in second
    assert "GDELT HTTP 500" in second
    assert "SEC EDGAR HTTP 500" in second
    assert second.count("FMP HTTP 500") == first.count("FMP HTTP 500")
    assert len(calls) > 0


@pytest.mark.anyio
async def test_malformed_ticker_renders_warning_without_requests() -> None:
    calls: list[str] = []

    def track_request(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return news_response(request, [])

    screen = NewsScreen(
        create_app_state(),
        SourceSession(
            env={},
            http_client=httpx.Client(
                transport=httpx.MockTransport(track_request),
            ),
        ),
    )

    async with NewsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen.query_one("#news_ticker", Input).value = "bad ticker"
        _ = await pilot.click("#news_refresh")
        await pilot.pause()
        rendered = str(screen.query_one("#news_body", Static).render())

    assert "Ticker/query: BAD TICKER" in rendered
    assert "Invalid ticker: BAD TICKER" in rendered
    assert calls == []


@pytest.mark.anyio
async def test_news_lifecycle_closes_default_source_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, requests = _owned_source_client(monkeypatch)
    screen = NewsScreen(create_app_state())

    async with NewsHarness(screen).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _ = await pilot.click("#news_refresh")
        await pilot.pause()
        assert requests != []
        assert client.is_closed is False
        await pilot.app.pop_screen()
        await pilot.pause()

    assert client.is_closed is True


@pytest.mark.anyio
async def test_news_lifecycle_keeps_injected_source_session_open() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: news_response(request, [])),
    ) as injected_client:
        screen = NewsScreen(
            create_app_state(),
            SourceSession(env={}, http_client=injected_client),
        )
        async with NewsHarness(screen).run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            _ = await pilot.click("#news_refresh")
            await pilot.pause()
            await pilot.app.pop_screen()
            await pilot.pause()

        assert injected_client.is_closed is False


def test_news_does_not_render_report_state() -> None:
    state = replace(
        create_app_state(),
        active_report=ReportProgress(
            status="running",
            label="Quarterly report should stay out of News",
            completed=3,
            total=7,
        ),
    )
    screen = NewsScreen(state)

    rendered = "\n".join(screen.lines())

    assert "Quarterly report" not in rendered
    assert "Progress:" not in rendered
    assert "report" not in rendered.lower()
