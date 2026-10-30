"""Alpaca's bars and Massive's overviews without the network."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import httpx
import pandas as pd
import pytest

from finagent import markets, portfolio, screen

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

KEYS = markets.Keys("alpaca-id", "alpaca-secret", "massive-key")
NOW = datetime(2026, 10, 30, 21, 0, tzinfo=UTC)  # 17:00 in New York


def client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def bar(day: str, close: float) -> dict[str, object]:
    # Alpaca stamps a daily bar at midnight New York.
    return {"t": f"{day}T04:00:00Z", "o": close, "h": close + 1, "l": close - 1,
            "c": close, "v": 100, "n": 1, "vw": close}  # fmt: skip


def test_keys_need_both_alpaca_halves() -> None:
    assert markets.keys({"ALPACA_API_KEY_ID": " id "}).alpaca is False
    found = markets.keys(
        {
            "ALPACA_API_KEY_ID": "id",
            "ALPACA_API_SECRET_KEY": "s",
            "MASSIVE_API_KEY": "m",
        }
    )
    assert found.alpaca
    assert found.massive == "m"


def test_us_listings_go_to_the_providers() -> None:
    assert markets.us_listed("BRK-B")
    assert not markets.us_listed("^GSPC")
    assert not markets.us_listed("KRW=X")
    assert not markets.us_listed("005930.KS")
    assert markets.provider_symbol("BRK-B") == "BRK.B"
    assert markets.app_symbol("BRK.B") == "BRK-B"


def test_bar_end_keeps_the_last_day_and_the_sip_delay() -> None:
    # A past day: its whole New York day.
    assert markets.bar_end(date(2026, 10, 29), NOW) == "2026-10-30T03:59:59Z"
    # Today: 16 minutes before now.
    assert markets.bar_end(date(2026, 10, 30), NOW) == "2026-10-30T20:44:00Z"


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

    got = markets.Alpaca(KEYS, client(handler), sleep=lambda _: None).bars(
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
    assert seen[1].url.params["page_token"] == "p2"  # noqa: S105 - a page, not a secret
    assert sorted(got["ticker"].unique()) == ["BRK-B", "SPY"]
    assert list(got["date"].dt.date.unique()) == [
        date(2026, 10, 28),
        date(2026, 10, 29),
    ]


def test_a_429_waits_and_retries() -> None:
    answers = iter([
        httpx.Response(429, headers={"retry-after": "7"}),
        httpx.Response(200, json={"bars": {}, "next_page_token": None}),
    ])  # fmt: skip
    waits: list[float] = []
    alpaca = markets.Alpaca(KEYS, client(lambda _: next(answers)), sleep=waits.append)

    got = alpaca.bars(["SPY"], date(2026, 10, 1), date(2026, 10, 29), "split", NOW)

    assert 7.0 in waits
    assert got.empty


def test_429s_past_the_retries_raise() -> None:
    alpaca = markets.Alpaca(
        KEYS, client(lambda _: httpx.Response(429)), sleep=lambda _: None
    )
    with pytest.raises(markets.RateLimitedError):
        _ = alpaca.bars(["SPY"], date(2026, 10, 1), date(2026, 10, 29), "split", NOW)


def test_massive_overview_sends_the_key_in_a_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("NOPE"):
            return httpx.Response(404, json={"status": "NOT_FOUND"})
        results = {"market_cap": 1.0e12, "currency_name": "usd", "type": "CS"}
        return httpx.Response(200, json={"status": "OK", "results": results})

    massive = markets.Massive(KEYS, client(handler), sleep=lambda _: None)
    got = massive.overview("BRK-B")

    assert got == markets.Overview(1.0e12, "USD", "CS")
    assert seen[0].url.path == "/v3/reference/tickers/BRK.B"
    assert seen[0].headers["Authorization"] == "Bearer massive-key"
    assert "massive-key" not in str(seen[0].url)
    assert massive.overview("NOPE") is None


def test_alpaca_bars_become_prices_and_screen_rows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        scale = 1.0 if request.url.params["adjustment"] == "split" else 0.9
        bars = {
            "SPY": [bar("2026-10-28", 100.0 * scale), bar("2026-10-29", 101.0 * scale)]
        }
        return httpx.Response(200, json={"bars": bars, "next_page_token": None})

    alpaca = markets.Alpaca(KEYS, client(handler), sleep=lambda _: None)
    prices = portfolio.alpaca_prices(alpaca, ["SPY"], date(2026, 10, 1))

    assert list(prices.close["SPY"]) == [100.0, 101.0]
    assert list(prices.adjusted["SPY"]) == pytest.approx([90.0, 90.9])
    assert list(prices.high["SPY"]) == [101.0, 102.0]

    split = alpaca.bars(["SPY"], date(2026, 10, 1), date(2026, 10, 29), "split", NOW)
    whole = alpaca.bars(["SPY"], date(2026, 10, 1), date(2026, 10, 29), "all", NOW)
    rows = screen.alpaca_long(split, whole)
    assert list(rows.columns) == list(screen.PRICE_COLUMNS)
    assert list(rows["adj_close"]) == pytest.approx([90.0, 90.9])


def test_official_sends_only_us_listings_to_alpaca(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asked: dict[str, list[str]] = {}

    def frame(tickers: list[str]) -> pd.DataFrame:
        index = pd.to_datetime(["2026-10-29"])
        return pd.DataFrame({t: [1.0] for t in tickers}, index=index)

    class FakeYahoo:
        def __init__(self, root: Path) -> None:
            del root

        def history(self, tickers: list[str], start: date) -> portfolio.Prices:
            del start
            asked["yahoo"] = list(tickers)
            return portfolio.Prices(frame(tickers), frame(tickers))

    def fake_alpaca(
        alpaca: markets.Alpaca, tickers: list[str], start: date
    ) -> portfolio.Prices:
        del alpaca, start
        asked["alpaca"] = list(tickers)
        return portfolio.Prices(frame(tickers), frame(tickers))

    monkeypatch.setattr(portfolio, "Yahoo", FakeYahoo)
    monkeypatch.setattr(portfolio, "alpaca_prices", fake_alpaca)
    tickers = ["SPY", "BRK-B", "005930.KS", "^GSPC", "KRW=X"]

    got = portfolio.Official(tmp_path, KEYS).history(tickers, date(2016, 1, 1))
    assert asked == {
        "yahoo": ["005930.KS", "^GSPC", "KRW=X"],
        "alpaca": ["SPY", "BRK-B"],
    }
    assert got.tickers == frozenset(tickers)

    asked.clear()
    _ = portfolio.Official(tmp_path, markets.Keys()).history(tickers, date(2016, 1, 1))
    assert asked == {"yahoo": tickers}  # no keys: Yahoo for all

    asked.clear()

    def official(ticker: str, start: date) -> pd.Series | None:
        del start
        if ticker != portfolio.FX:
            return None  # no key or no source: Yahoo's
        return pd.Series([1400.0], index=pd.to_datetime(["2026-10-29"]))

    got = portfolio.Official(tmp_path, markets.Keys(), official).history(
        tickers, date(2016, 1, 1)
    )
    assert asked == {"yahoo": ["SPY", "BRK-B", "005930.KS", "^GSPC"]}
    assert list(got.close["KRW=X"]) == [1400.0]
    assert list(got.adjusted["KRW=X"]) == [1400.0]
