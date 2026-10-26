from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import date
from typing import TYPE_CHECKING

import httpx
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import pytest

from finagent.fetch import FetchContext, SourceError
from finagent.sources import (
    FredSeries,
    YahooCloses,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

TODAY = date(2026, 10, 12)
KEYS = {"FRED_API_KEY": "fred-secret-key", "ECOS_API_KEY": "ecos-secret-key"}


def run[T](
    handler: Callable[[httpx.Request], httpx.Response],
    call: Callable[[FetchContext], Coroutine[None, None, T]],
    env: dict[str, str] | None = None,
) -> T:
    async def main() -> T:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            ctx = FetchContext(client, KEYS if env is None else env, retry_delay=0)
            return await call(ctx)

    return asyncio.run(main())


def json_response(payload: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode())


def test_fred_sends_key_and_skips_missing_dots() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response(
            {
                "observations": [
                    {"date": "2026-10-08", "value": "4.10"},
                    {"date": "2026-10-09", "value": "."},
                ]
            }
        )

    series = run(handler, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))

    assert series == ((date(2026, 10, 8), 4.1),)
    params = seen[0].url.params
    assert params["series_id"] == "DGS10"
    assert params["api_key"] == "fred-secret-key"
    assert params["file_type"] == "json"
    assert params["observation_start"] == "2022-10-12"


def test_http_errors_never_leak_the_key() -> None:
    def server_error(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    def network_error(request: httpx.Request) -> httpx.Response:
        message = f"cannot reach {request.url}"
        raise httpx.ConnectError(message, request=request)

    for handler in (server_error, network_error):
        with pytest.raises(SourceError) as caught:
            _ = run(handler, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))
        assert "fred-secret-key" not in str(caught.value)


def test_server_errors_are_retried_once_only() -> None:
    calls: list[int] = []

    def flaky(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(502)
        return json_response({"observations": [{"date": "2026-10-08", "value": "4"}]})

    series = run(flaky, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))

    assert series == ((date(2026, 10, 8), 4.0),)
    assert len(calls) == 2

    calls.clear()

    def down(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(SourceError, match="FRED HTTP 503"):
        _ = run(down, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))
    assert len(calls) == 2  # gives up after one retry


# ---------------------------------------------------------------- Yahoo


def fake_yfinance(
    monkeypatch: pytest.MonkeyPatch, answer: pd.DataFrame | Exception
) -> list[dict[str, object]]:
    """Stand in for the yfinance module; returns the ``history`` calls."""
    calls: list[dict[str, object]] = []

    class Ticker:
        def __init__(self, ticker: str) -> None:
            self.ticker: str = ticker

        def history(self, **kwargs: object) -> pd.DataFrame:
            calls.append({"ticker": self.ticker, **kwargs})
            if isinstance(answer, Exception):
                raise answer
            return answer

    def point(_: str) -> None:
        pass

    fake = types.ModuleType("yfinance")
    fake.set_tz_cache_location = point  # pyright: ignore[reportAttributeAccessIssue]
    fake.Ticker = Ticker  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return calls


def test_yahoo_failures_are_source_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = fake_yfinance(monkeypatch, ConnectionError("down"))
    with pytest.raises(SourceError, match="YAHOO GC=F: ConnectionError"):
        _ = YahooCloses(tmp_path)("GC=F", TODAY)

    _ = fake_yfinance(monkeypatch, pd.DataFrame({"Close": []}))
    with pytest.raises(SourceError, match="YAHOO GC=F: 관측값 없음"):
        _ = YahooCloses(tmp_path)("GC=F", TODAY)
