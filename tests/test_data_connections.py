# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import httpx
import pytest

from finagent import (
    catalog,
    etf,
    etf_universe,
    fund_filings,
    fundamentals,
    investors,
    memory,
    prices,
    sec,
)
from finagent.models import Fmt, IndicatorRow
from finagent.sources import EtfSeries
from finagent.ui import etf_page, guides, schedule
from tests import fake_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

TODAY = date(2026, 11, 10)
AGENT = "Test User test@example.com"


def test_company_filings_parses_list_and_str_items_and_caches_same_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "4", "10-Q", "10-K"],
                "accessionNumber": [
                    "0000320193-26-000071",
                    "0000320193-26-000070",
                    "0000320193-26-000060",
                    "0000320193-25-000050",
                ],
                "filingDate": [
                    "2026-08-01",
                    "2026-07-15",
                    "2026-05-02",
                    "2025-11-01",
                ],
                "items": [["2.02", "9.01"], "", "2.02", []],
                "primaryDocument": ["a8k.htm", "f4.xml", "a10q.htm", ""],
            }
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json=tickers)
        return httpx.Response(200, json=submissions)

    def make_client(self: sec.Client, agent: str) -> None:
        del agent
        self._client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(sec.Client, "__init__", make_client)

    first = fundamentals.company_filings("AAPL", AGENT, TODAY)
    second = fundamentals.company_filings(
        "AAPL", AGENT, TODAY, forms=("10-K", "8-K"), limit=2
    )

    assert len(calls) == 2
    assert [row.form for row in first] == ["8-K", "10-Q", "10-K"]
    assert first[0].items == "2.02, 9.01"
    assert first[0].url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000071/a8k.htm"
    )
    assert first[1].items == "2.02"
    assert first[2].url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000050/"
    )
    assert [row.form for row in second] == ["8-K", "10-K"]


def test_annual_rows_reuses_same_day_memory_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_sec(symbol: str, agent: str, today: date) -> dict[str, object]:
        nonlocal calls
        del symbol, agent, today
        calls += 1
        return {"source": "SEC companyfacts", "currency": "USD", "fy_end": "2026-09-30"}

    monkeypatch.setattr(fundamentals, "_sec_annual", fake_sec)

    first = fundamentals.annual_rows("AAPL", AGENT, TODAY)
    second = fundamentals.annual_rows("AAPL", AGENT, TODAY)

    assert first == second
    assert calls == 1


def test_etf_maturity_and_countries_handle_valid_invalid_and_overflow() -> None:
    tlt_parsed = fund_filings.parse_nport(fake_sources.NPORT_TLT)
    tlt = etf.Composition(
        "TLT", tlt_parsed.as_of, etf.NPORT, "acc", full=True, rows=tlt_parsed.rows
    )
    maturity = etf_page.etf_maturity(tlt)
    assert maturity == pytest.approx((49.0 * 28.96 + 49.15 * 18.96) / 98.15, abs=0.02)

    bad_date = etf.Composition(
        "TLT", "invalid-date", etf.NPORT, "acc", full=True, rows=tlt_parsed.rows
    )
    assert etf_page.etf_maturity(bad_date) is None

    rows = tuple(
        fund_filings.Row(
            name=f"C{i}",
            cusip=None,
            isin=None,
            ticker=None,
            weight=float(20 - i),
            asset="stock",
            country=code,
            maturity=None,
        )
        for i, code in enumerate(
            ("US", "JP", "GB", "CA", "FR", "DE", "CH", "KR", "TW", "AU", "XX", "YY")
        )
    )
    global_etf = etf.Composition(
        "VT", "2026-08-31", etf.NPORT, "acc", full=True, rows=rows
    )
    lands = etf_page.etf_countries(global_etf)
    assert len(lands) == etf_page.COUNTRY_LIMIT + 1
    assert "미국" in lands
    assert "기타" in lands
    assert sum(lands.values()) == pytest.approx(100.0)


def test_catalog_etf_universe_and_guides_consistency() -> None:
    for indicator_id in ("gl_gold_ov", "gl_gold", "gl_copper"):
        ind = catalog.BY_ID[indicator_id]
        assert ind.freq == "D"
        assert ind.fetch.source == "YAHOO"
        assert isinstance(ind.fetch, EtfSeries)

    monetary_ids = {
        ind.id
        for spec in catalog.PANELS
        if spec.page == "monetary"
        for ind in spec.indicators
    }
    assert {"jp_boj_assets", "kr_bok_base"} <= monetary_ids

    macro_alpaca = {
        ind.fetch.ticker
        for spec in catalog.PANELS
        for ind in spec.indicators
        if isinstance(ind.fetch, EtfSeries) and ind.fetch.source == "ALPACA"
    }
    universe_tickers = {product.ticker for product in etf_universe.DEFAULTS}
    assert macro_alpaca <= universe_tickers

    assert set(catalog.BY_ID) <= set(guides.GUIDES)


def test_schedule_display_code_and_extra_cards() -> None:
    _ = prices.current().ensure(["XLK", "GC=F"], date(2026, 10, 1), TODAY)
    resolved = schedule.price_sources()
    xlk_row = IndicatorRow(
        id="us_xlk",
        label="기술 (XLK)",
        code="ALPACA:XLK",
        source="ALPACA",
        area="US",
        freq="D",
        fmt=Fmt(),
        status="ok",
    )
    gold_row = IndicatorRow(
        id="gl_gold",
        label="금",
        code="YAHOO:GC=F",
        source="YAHOO",
        area="GL",
        freq="D",
        fmt=Fmt(),
        status="ok",
    )

    assert schedule.display_code(xlk_row, resolved) == "Yahoo:XLK"
    assert schedule.display_code(gold_row, resolved) == "YAHOO:GC=F"
    assert schedule.display_code(xlk_row, {}) == "ALPACA:XLK"

    demo_cards = schedule.extra_cards(demo=True)
    assert [(c.source, c.value, c.delta, c.color) for c in demo_cards] == [
        ("ALPACA", "데모", "데모", "gray"),
        ("YAHOO", "데모", "데모", "gray"),
        ("SEC", "데모", "데모", "gray"),
        ("OPENFIGI", "데모", "데모", "gray"),
    ]

    store = memory.current()
    store.yahoo_info.put("AAPL", (TODAY, {"quoteType": "EQUITY"}))
    store.cusips.put("037833100", investors.Link(ticker="AAPL"))
    store.filers.put(
        "0001067983",
        investors.Filer(
            cik="0001067983", name="BH", fetched=datetime.now().astimezone()
        ),
    )

    env: Mapping[str, str] = {
        "SEC_USER_AGENT": AGENT,
        "OPENFIGI_API_KEY": "figi-key",
    }
    cards = {card.source: card for card in schedule.extra_cards(demo=False, env=env)}
    assert cards["ALPACA"].value == "미설정"
    assert cards["ALPACA"].delta == "Yahoo 대체"
    assert cards["ALPACA"].color == "gray"
    assert cards["YAHOO"].value == "시세 2 · 정보 1"
    assert cards["YAHOO"].color == "green"
    assert cards["SEC"].value == "캐시 1건"
    assert cards["SEC"].delta == "준비됨"
    assert cards["SEC"].color == "green"
    assert cards["OPENFIGI"].value == "1건"
    assert cards["OPENFIGI"].delta == "키 확장"
    assert cards["OPENFIGI"].color == "green"
