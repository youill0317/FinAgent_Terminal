from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, NoReturn, cast

import httpx
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from finagent import etf, fundamentals, memory, prices, sec
from finagent.analysis import context, documents
from finagent.analysis.briefing import Briefing
from finagent.analysis.company_tools import Company
from finagent.analysis.service import AnalysisService, RunInput
from finagent.analysis.tools import Toolbox
from finagent.demo import dashboard as demo_dashboard
from finagent.fetch import FetchContext, SharedWork
from finagent.models import Fmt, IndicatorRow, Panel
from finagent.service import DashboardService, SeriesStore
from finagent.sources import FredSeries
from tests.fake_sources import DEMO_KEYS, demo_transport

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pytest

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
TODAY = NOW.date()


def test_period_end_and_stale_days() -> None:
    assert context.period_end("2026-08", "M") == date(2026, 8, 31)
    assert context.period_end("2026Q2", "Q") == date(2026, 6, 30)
    assert context.period_end("2025", "A") == date(2025, 12, 31)
    assert context.period_end("invalid", "M") is None

    assert context.stale_days(date(2026, 10, 8), "D", TODAY) is None
    assert context.stale_days(date(2026, 10, 7), "D", TODAY) == 5
    assert context.stale_days(date(2026, 9, 29), "W", TODAY) is None
    assert context.stale_days(date(2026, 9, 28), "W", TODAY) == 14
    assert context.stale_days("2026-08", "M", date(2026, 10, 9)) is None
    assert context.stale_days("2026-08", "M", date(2026, 10, 10)) == 40
    assert context.stale_days("2026Q2", "Q", date(2026, 10, 22)) is None
    assert context.stale_days("2026Q2", "Q", date(2026, 10, 23)) == 115
    assert context.stale_days("2025", "A", date(2027, 2, 3)) is None
    assert context.stale_days("2025", "A", date(2027, 2, 4)) == 400


def test_build_document_marks_stale_and_includes_recent_releases() -> None:
    base = demo_dashboard.dashboard(NOW)
    panel = Panel(
        id="p_test",
        page="economy",
        title="테스트",
        note="",
        rows=(
            IndicatorRow(
                id="us_cpi",
                label="CPI",
                code="FRED:CPIAUCSL",
                source="FRED",
                area="US",
                freq="M",
                fmt=Fmt(),
                status="ok",
                value=3.0,
                date="2026-08",
            ),
            IndicatorRow(
                id="us_10y",
                label="10Y",
                code="FRED:DGS10",
                source="FRED",
                area="US",
                freq="D",
                fmt=Fmt(),
                status="ok",
                value=4.1,
                date="2026-10-09",
            ),
        ),
    )
    board = base.model_copy(update={"panels": (panel,)})
    doc = context.build_document(board)
    cpi_line = next(line for line in doc.text.splitlines() if 'id="us_cpi"' in line)
    y10_line = next(line for line in doc.text.splitlines() if 'id="us_10y"' in line)

    assert 'stale="true"' in cpi_line
    assert 'lag_days="42"' in cpi_line
    assert "stale=" not in y10_line
    assert "recent_releases" in doc.ids
    assert 'id="recent_releases"' in doc.text
    assert "recent_releases" in documents.page_document(board, "schedule").ids
    assert "recent_releases" not in documents.page_document(board, "economy").ids


def test_build_dashboard_populates_recent_releases_distinct_from_calendar() -> None:
    transport = demo_transport(TODAY)
    svc = DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: NOW,
        retry_delay=0,
    )
    board = asyncio.run(svc.get())

    recent = {entry.name for entry in board.recent_releases}
    upcoming = {entry.name for entry in board.calendar}
    assert {"고용보고서", "개인소득·지출 (PCE)"} <= recent
    assert "생산자물가 PPI" in upcoming
    assert "고용보고서" not in upcoming


def test_series_store_ttl_expiration_priority_and_fresh_bypass() -> None:
    now = [NOW]
    store = SeriesStore(clock=lambda: now[0])
    fetch = FredSeries("CPIAUCSL", years=2)
    old_points = ((date(2026, 8, 1), 300.0),)
    new_points = ((date(2026, 9, 1), 305.0),)
    memory.current().macro[fetch] = old_points
    memory.current().macro_at = now[0]

    assert store.covering("FRED", "CPIAUCSL", date(2025, 1, 1)) == old_points

    now[0] = NOW + timedelta(minutes=30)
    assert store.covering("FRED", "CPIAUCSL", date(2025, 1, 1)) == old_points

    store.put_extra("FRED", "CPIAUCSL", date(2025, 1, 1), new_points)
    assert store.covering("FRED", "CPIAUCSL", date(2025, 1, 1)) == new_points

    now[0] = NOW + timedelta(hours=2)
    assert store.covering("FRED", "CPIAUCSL", date(2025, 1, 1)) is None

    now[0] = NOW + timedelta(days=1)
    assert store.covering("FRED", "CPIAUCSL", date(2025, 1, 1)) is None

    now[0] = NOW + timedelta(minutes=35)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"observations": [{"date": "2026-10-01", "value": "310.0"}]},
        )

    async def run() -> tuple[tuple[date, float], ...]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ctx = FetchContext(client, DEMO_KEYS, retry_delay=0, shared=SharedWork())
            return await store.through(fetch, ctx, TODAY, fresh=True)

    refreshed = asyncio.run(run())
    assert calls == 1
    assert refreshed == ((date(2026, 10, 1), 310.0),)


def test_tools_get_series_fresh_stale_and_search_stale_filter() -> None:
    obs_calls = 0
    include_september = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal obs_calls
        if request.url.path.endswith("/series/search"):
            return httpx.Response(
                200,
                json={
                    "seriess": [
                        {
                            "id": "OLD_SERIES",
                            "title": "Ended CPI",
                            "frequency_short": "M",
                            "units_short": "Index",
                            "observation_start": "2000-01-01",
                            "observation_end": "2025-01-01",
                            "notes": "",
                        },
                        {
                            "id": "LIVE_SERIES",
                            "title": "Active CPI",
                            "frequency_short": "M",
                            "units_short": "Index",
                            "observation_start": "2000-01-01",
                            "observation_end": "2026-10-01",
                            "notes": "",
                        },
                    ]
                },
            )
        obs_calls += 1
        months = (
            [f"2024-{month:02d}-01" for month in range(8, 13)]
            + [f"2025-{month:02d}-01" for month in range(1, 13)]
            + [f"2026-{month:02d}-01" for month in range(1, 9)]
        )
        if include_september:
            months.append("2026-09-01")
        observations = [
            {"date": day, "value": f"{300 + idx}"} for idx, day in enumerate(months)
        ]
        return httpx.Response(200, json={"observations": observations})

    transport = httpx.MockTransport(handler)
    svc = DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: NOW,
        retry_delay=0,
    )

    async def run(name: str, args: dict[str, object]) -> tuple[bool, dict[str, object]]:
        async with svc.fetch_context() as ctx:
            box = Toolbox(ctx, svc.series_store, TODAY, Company({}, TODAY))
            res = await box.run(name, args)
            return res.ok, cast("dict[str, object]", json.loads(res.text))

    ok, first = asyncio.run(run("get_series", {"id": "us_cpi"}))
    assert ok
    assert first["as_of"] == "2026-10-12"
    assert first["stale"] is True
    assert first["lag_days"] == 42
    assert obs_calls == 1

    ok_hist, hist = asyncio.run(run("get_series", {"id": "us_cpi", "end": "2026-07"}))
    assert ok_hist
    assert "stale" not in hist

    include_september = True
    ok_cached, cached = asyncio.run(run("get_series", {"id": "us_cpi"}))
    assert ok_cached
    assert cached["points"] == first["points"]
    assert obs_calls == 1

    ok_fresh, fresh = asyncio.run(run("get_series", {"id": "us_cpi", "fresh": True}))
    assert ok_fresh
    assert obs_calls == 2
    assert "stale" not in fresh
    points = cast("list[list[object]]", fresh["points"])
    assert points[-1][0] == "2026-09"

    ok_bad, _ = asyncio.run(run("get_series", {"id": "us_cpi", "fresh": "yes"}))
    assert not ok_bad

    ok_search, search = asyncio.run(
        run("search_series", {"source": "FRED", "query": "CPI"})
    )
    assert ok_search
    results = cast("list[dict[str, object]]", search["results"])
    assert [row["code"] for row in results] == ["LIVE_SERIES", "OLD_SERIES"]
    assert "stale" not in results[0]
    assert results[1]["stale"] is True


def test_dashboard_service_snapshot_ttl_and_date_rollover() -> None:
    now = [NOW]
    transport = demo_transport(TODAY)
    svc = DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: now[0],
        retry_delay=0,
    )

    first = asyncio.run(svc.get())
    assert svc.refreshes == 0

    now[0] = NOW + timedelta(minutes=30)
    second = asyncio.run(svc.get())
    assert second.generated_at == first.generated_at
    assert svc.refreshes == 0

    now[0] = NOW + timedelta(minutes=70)
    third = asyncio.run(svc.get())
    assert third.generated_at == now[0].isoformat(timespec="seconds")
    assert svc.refreshes == 1

    now[0] = NOW + timedelta(days=1)
    fourth = asyncio.run(svc.get())
    assert fourth.generated_at == now[0].isoformat(timespec="seconds")
    assert svc.refreshes == 2


def test_last_session_market_timezones_and_price_store_tail_refetch() -> None:
    at_17_kst = datetime(2026, 10, 12, 8, 0, tzinfo=UTC)
    assert prices.last_session(at_17_kst, "005930.KS") == date(2026, 10, 12)
    assert prices.last_session(at_17_kst, "^KS11") == date(2026, 10, 12)
    assert prices.last_session(at_17_kst, "7203.T") == date(2026, 10, 12)
    assert prices.last_session(at_17_kst, "GC=F") == date(2026, 10, 12)
    assert prices.last_session(at_17_kst, "KRW=X") == date(2026, 10, 12)
    assert prices.last_session(at_17_kst, "AAPL") == date(2026, 10, 11)

    at_15_kst = datetime(2026, 10, 12, 6, 0, tzinfo=UTC)
    assert prices.last_session(at_15_kst, "005930.KS") == date(2026, 10, 11)

    clock_ref = [datetime(2026, 10, 11, 22, 0, tzinfo=UTC)]
    store = prices.PriceStore(clock=lambda: clock_ref[0])
    calls: list[tuple[tuple[str, ...], date, date]] = []

    def fake_fetch(tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        calls.append((tuple(tickers), start, end))
        days = [start + timedelta(days=offset) for offset in range((end - start).days)]
        return pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "date": day,
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 1.0,
                    "source": "Yahoo",
                }
                for ticker in tickers
                for day in days
            ]
        )

    _ = store.ensure(
        ["005930.KS", "AAPL"],
        date(2026, 10, 1),
        date(2026, 10, 11),
        fetch=fake_fetch,
    )
    calls.clear()
    clock_ref[0] = at_17_kst
    _ = store.ensure(
        ["005930.KS", "AAPL"],
        date(2026, 10, 1),
        date(2026, 10, 12),
        fetch=fake_fetch,
    )
    assert calls == [(("005930.KS",), date(2026, 10, 4), date(2026, 10, 13))]


def test_get_prices_fresh_and_briefing_etf_points_ensure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen_force: list[bool] = []
    orig_ensure = prices.PriceStore.ensure

    def spy_ensure(  # noqa: PLR0913
        self: prices.PriceStore,
        tickers: Sequence[str],
        start: date,
        end: date,
        *,
        force: bool = False,
        max_wait: float = 0.0,
    ) -> prices.PriceResult:
        seen_force.append(force)
        return orig_ensure(self, tickers, start, end, force=force, max_wait=max_wait)

    monkeypatch.setattr(prices.PriceStore, "ensure", spy_ensure)
    svc = DashboardService(DEMO_KEYS, clock=lambda: NOW, demo=True)

    async def call_prices(args: dict[str, object]) -> tuple[bool, dict[str, object]]:
        async with svc.fetch_context() as ctx:
            box = Toolbox(ctx, svc.series_store, TODAY, Company({}, TODAY))
            res = await box.run("get_prices", args)
            return res.ok, cast("dict[str, object]", json.loads(res.text))

    ok, body = asyncio.run(call_prices({"tickers": ["AAPL"]}))
    assert ok
    assert body["as_of"] == "2026-10-12"
    assert seen_force[-1] is False

    ok_fresh, _ = asyncio.run(call_prices({"tickers": ["AAPL"], "fresh": True}))
    assert ok_fresh
    assert seen_force[-1] is True

    ok_bad, _ = asyncio.run(call_prices({"tickers": ["AAPL"], "fresh": "bad"}))
    assert not ok_bad

    def unused() -> NoReturn:
        raise AssertionError

    def unused_look(_weights: pd.Series) -> NoReturn:
        raise AssertionError

    live_svc = DashboardService(DEMO_KEYS, clock=lambda: NOW)
    brief = Briefing(
        root=tmp_path,
        settings_path=tmp_path / "etf.json",
        demo=False,
        env={},
        dashboard=live_svc,
        held=list,
        confirm=lambda _t: "확인 실패",
        options=list,
        prices=unused,
        fetcher=unused,
        updater=unused,
        look=unused_look,
    )
    points = brief.etf_points(["XLK"])
    assert "XLK" in points
    assert len(points["XLK"]) > 0


def test_company_submissions_ttl_facts_staleness_and_prepare_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub_calls = 0
    latest_end = "2026-04-01"

    def fake_cik(_self: sec.Archive, _ticker: str) -> str:
        return "0000320193"

    def fake_facts(_self: sec.Archive, _cik: str) -> dict[str, object]:
        return {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "end": latest_end,
                                    "val": 100.0,
                                    "fy": 2026,
                                    "fp": "Q2",
                                    "form": "10-Q",
                                    "filed": "2026-05-01",
                                    "frame": "CY2026Q1",
                                }
                            ]
                        }
                    }
                }
            }
        }

    def fake_submissions(
        _self: sec.Client, _cik: str, _max_wait: float
    ) -> httpx.Response:
        nonlocal sub_calls
        sub_calls += 1
        return httpx.Response(
            200,
            request=httpx.Request(
                "GET",
                "https://data.sec.gov/submissions/CIK0000320193.json",
            ),
            json={
                "filings": {
                    "recent": {
                        "form": ["8-K"],
                        "accessionNumber": ["0000320193-26-000071"],
                        "filingDate": ["2026-08-01"],
                        "items": ["2.02"],
                        "primaryDocument": ["a8k.htm"],
                    }
                }
            },
        )

    def fake_info(
        _ticker: str, _keys: object, _agent: str, day: date
    ) -> fundamentals.Company:
        cell = etf.Cell("Apple Inc.", "Yahoo", day.isoformat())
        return fundamentals.Company(
            ticker="AAPL",
            symbol="AAPL",
            fetched_on=day.isoformat(),
            cells={"name": cell},
        )

    monkeypatch.setattr(sec.Archive, "cik", fake_cik)
    monkeypatch.setattr(sec.Archive, "facts", fake_facts)
    monkeypatch.setattr(sec.Client, "submissions", fake_submissions)
    monkeypatch.setattr(fundamentals, "company_info", fake_info)

    mono = [1000.0]
    comp = Company(
        {"SEC_USER_AGENT": "Test test@example.com"}, TODAY, clock=lambda: mono[0]
    )

    stale_facts = comp.run("company_facts", {"ticker": "AAPL", "metrics": ["revenue"]})
    assert stale_facts.body["as_of"] == "2026-10-12"
    assert stale_facts.body["stale"] is True
    assert stale_facts.body["lag_days"] == 194

    latest_end = "2026-09-26"
    fresh_facts = comp.run("company_facts", {"ticker": "AAPL", "metrics": ["revenue"]})
    assert "stale" not in fresh_facts.body

    profile = comp.run("company_profile", {"ticker": "AAPL"})
    assert profile.body["as_of"] == "2026-10-12"

    first_filings = comp.run("company_filings", {"ticker": "AAPL"})
    assert first_filings.body["as_of"] == "2026-10-12"
    assert sub_calls == 1

    mono[0] = 2000.0
    _ = comp.run("company_filings", {"ticker": "AAPL"})
    assert sub_calls == 1

    mono[0] = 4700.0
    _ = comp.run("company_filings", {"ticker": "AAPL"})
    assert sub_calls == 2

    now = [NOW]
    svc = DashboardService(DEMO_KEYS, clock=lambda: now[0], demo=True)
    analysis_svc = AnalysisService(svc, {}, prepare=lambda _brief: RunInput("요청"))
    synced = analysis_svc.company()
    assert synced.today == date(2026, 10, 12)
    now[0] = NOW + timedelta(days=1)
    _ = asyncio.run(analysis_svc.prepare("macro", "", "", synced))
    assert synced.today == date(2026, 10, 13)
