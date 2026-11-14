# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import httpx

from finagent import markets

if TYPE_CHECKING:
    from collections.abc import Callable


KEYS = markets.Keys("alpaca-id", "alpaca-secret")
NOW = datetime(2026, 10, 30, 21, 0, tzinfo=UTC)


def client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def bar(day: str, close: float) -> dict[str, object]:
    return {"t": f"{day}T04:00:00Z", "o": close, "h": close + 1, "l": close - 1,
            "c": close, "v": 100, "n": 1, "vw": close}  # fmt: skip


def test_alpaca_pages_through_every_symbol() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "page_token" not in request.url.params:
            body = {
                "bars": {"BRK.B": [bar("2026-10-28", 500.0)]},
                "next_page_token": "p2",
            }
        else:
            body = {"bars": {"BRK.B": [bar("2026-10-29", 501.0)],
                             "SPY": [bar("2026-10-29", 600.0)]},
                    "next_page_token": None}  # fmt: skip
        return httpx.Response(200, json=body)

    got = markets.Alpaca(KEYS, client(handler)).bars(
        ["BRK-B", "SPY"], date(2026, 10, 1), date(2026, 10, 29), "all", now=NOW
    )

    assert len(seen) == 2
    first = seen[0]
    assert first.headers["APCA-API-KEY-ID"] == "alpaca-id"
    assert first.headers["APCA-API-SECRET-KEY"] == "alpaca-secret"
    assert first.url.params["symbols"] == "BRK.B,SPY"
    assert first.url.params["adjustment"] == "all"
    assert first.url.params["feed"] == "sip"
    assert first.url.params["end"] == "2026-10-30T03:59:59Z"
    assert seen[1].url.params["page_token"] == "p2"  # noqa: S105
    assert sorted(got["ticker"].unique()) == ["BRK-B", "SPY"]
    assert list(got["date"].dt.date.unique()) == [
        date(2026, 10, 28),
        date(2026, 10, 29),
    ]
