from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from typing import TYPE_CHECKING

import httpx

from finagent.fetch import FetchContext
from finagent.sources import (
    EcosSeries,
    FredSeries,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


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
    assert params["observation_start"] == "2015-10-01"


def test_ecos_pages_past_its_row_limit() -> None:
    days = [date(1990, 1, 1) + timedelta(days=i) for i in range(12_000)]
    asked: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        first, last = (int(part) for part in request.url.path.split("/")[6:8])
        asked.append((first, last))
        rows = [
            {"TIME": day.strftime("%Y%m%d"), "DATA_VALUE": str(i)}
            for i, day in enumerate(days[first - 1 : last])
        ]
        body = {"list_total_count": len(days), "row": rows}
        return json_response({"StatisticSearch": body})

    series = run(
        handler,
        lambda ctx: EcosSeries("802Y001", "D", "0001000", years=40).fetch(ctx, TODAY),
    )

    assert asked == [(1, 10_000), (10_001, 20_000)]
    assert len(series) == 12_000
    assert series[-1] == (days[-1], 1_999.0)


FRED = FredSeries("CPIAUCSL", years=4)
