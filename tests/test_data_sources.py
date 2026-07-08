from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

import httpx

from finagent_terminal.data.source_http import SourceHttpClient
from finagent_terminal.data.sources import (
    SourceBundle,
    SourceItem,
    SourceSession,
    source_availability,
    source_availability_lines,
)

if TYPE_CHECKING:
    import pytest

SOURCE_SECRETS: Final = ("fred-secret", "finn-secret", "fmp-secret", "sec-secret")


@final
class CountingSourceFetcher:
    __slots__ = ("calls",)

    def __init__(self) -> None:
        self.calls = 0

    calls: int

    def __call__(self) -> SourceBundle:
        self.calls += 1
        return SourceBundle(
            source="fred",
            items=(
                SourceItem(
                    source="fred",
                    title="GDP",
                    summary="Gross domestic product",
                ),
            ),
        )


def test_source_availability_redacts_secret_values() -> None:
    env = {
        "FRED_API_KEY": "fred-secret",
        "FINNHUB_API_KEY": "finn-secret",
        "FMP_API_KEY": "fmp-secret",
        "FINAGENT_SEC_USER_AGENT": "sec-secret",
    }

    lines = source_availability_lines(env)

    text = "\n".join(lines)
    assert "FRED_API_KEY" in text
    assert "FINNHUB_API_KEY" in text
    assert "FMP_API_KEY" in text
    assert "FINAGENT_SEC_USER_AGENT" in text
    assert "available" in text
    assert all(secret not in text for secret in SOURCE_SECRETS)


def test_blank_source_availability_values_are_unavailable() -> None:
    env = {
        "FRED_API_KEY": " ",
        "FINNHUB_API_KEY": "\t",
        "FMP_API_KEY": "",
        "FINAGENT_SEC_USER_AGENT": "sec-agent",
    }

    by_env = {item.env_var: item for item in source_availability(env)}

    assert by_env["FRED_API_KEY"].configured is False
    assert by_env["FINNHUB_API_KEY"].configured is False
    assert by_env["FMP_API_KEY"].configured is False
    assert by_env["FINAGENT_SEC_USER_AGENT"].configured is True


def test_source_session_cache_hit_avoids_second_fake_fetch() -> None:
    fetcher = CountingSourceFetcher()

    with SourceSession() as session:
        first = session.get_or_fetch((" fred ", " GDP "), fetcher)
        second = session.get_or_fetch(("fred", "GDP"), fetcher)

    assert fetcher.calls == 1
    assert first is second
    assert first.items[0].title == "GDP"


def test_source_session_clear_prevents_stale_state() -> None:
    fetcher = CountingSourceFetcher()

    with SourceSession() as session:
        _ = session.get_or_fetch("fred:gdp", fetcher)
        session.clear()
        _ = session.get_or_fetch("fred:gdp", fetcher)
    with SourceSession() as new_session:
        new_session_result = new_session.get_or_fetch("fred:gdp", fetcher)

    assert fetcher.calls == 3
    assert new_session_result.items[0].source == "fred"


def test_source_http_client_closes_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_client = httpx.Client(transport=httpx.MockTransport(_empty_response))

    def make_client(
        *,
        timeout: httpx.Timeout,
        limits: httpx.Limits,
        follow_redirects: bool,
    ) -> httpx.Client:
        assert timeout is not None
        assert limits is not None
        assert follow_redirects is True
        return owned_client

    monkeypatch.setattr(httpx, "Client", make_client)

    source_client = SourceHttpClient(env={}, http_client=None)
    source_client.close()

    assert owned_client.is_closed is True


def test_source_http_client_does_not_close_injected_client() -> None:
    injected_client = httpx.Client(transport=httpx.MockTransport(_empty_response))
    source_client = SourceHttpClient(env={}, http_client=injected_client)

    source_client.close()

    assert injected_client.is_closed is False
    injected_client.close()


def _empty_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={}, request=request)
