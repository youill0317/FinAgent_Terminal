from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from finagent import catalog
from finagent.cli import check_report
from finagent.demo import DEMO_KEYS, demo_transport
from finagent.models import Dashboard
from finagent.server import create_app
from finagent.service import DashboardService

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def service(
    env: dict[str, str] | None = None,
    wrap: Callable[[httpx.Request], httpx.Response | None] | None = None,
) -> DashboardService:
    demo = demo_transport(NOW.date())

    def handler(request: httpx.Request) -> httpx.Response:
        if wrap is not None and (override := wrap(request)) is not None:
            return override
        return demo.handle_request(request)

    transport = httpx.MockTransport(handler)
    return DashboardService(
        DEMO_KEYS if env is None else env,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: NOW,
        retry_delay=0,
    )


def build(svc: DashboardService) -> Dashboard:
    return asyncio.run(svc.get())


def test_every_catalog_indicator_resolves_with_demo_data() -> None:
    dashboard = build(service())

    rows = [row for panel in dashboard.panels for row in panel.rows]
    assert [row.id for row in rows if row.status != "ok"] == []
    assert len(rows) == sum(len(spec.indicators) for spec in catalog.PANELS)
    assert all(source.state == "ok" for source in dashboard.sources)
    assert all(row.history for row in rows)


def test_tape_ids_exist_in_panels() -> None:
    ids = {indicator.id for spec in catalog.PANELS for indicator in spec.indicators}

    assert set(catalog.TAPE) <= ids
    assert {"kr_ktb10", "us_sp500", "kr_kospi"} <= set(catalog.TAPE)


def test_charts_regime_and_calendar() -> None:
    dashboard = build(service())

    assert dashboard.curve is not None
    assert dashboard.curve.tenors[0] == "1M"
    assert [line.label for line in dashboard.curve.lines] == [
        "최신",
        "1개월 전",
        "1년 전",
    ]
    assert dashboard.spread is not None
    assert dashboard.spread.recessions == (("2020-02", "2020-04"),)
    assert [cell.label for cell in dashboard.regime] == [
        "물가",
        "고용",
        "금리곡선",
        "금융여건",
    ]
    assert all(cell.tag != "데이터 없음" for cell in dashboard.regime)
    names = [entry.name for entry in dashboard.calendar]
    assert "소비자물가 CPI" in names
    assert all("Kansas" not in name for name in names)


def test_fomc_projections_compare_with_the_previous_release() -> None:
    dashboard = build(service())

    sep = dashboard.projections
    assert sep is not None
    assert sep.message is None
    assert sep.published == "2026-09-15"
    assert sep.prior_published == "2026-06-16"
    assert sep.years == ("2026", "2027", "2028", "2029")
    rows = {row.label: row for row in sep.rows}
    funds = rows["연방기금금리"]
    assert [(cell.value, cell.prev) for cell in funds.years] == [
        (3.6, 3.9),
        (3.4, 3.6),
        (3.1, 3.4),
        (3.1, None),
    ]
    assert funds.longer_run is not None
    assert funds.longer_run.value == 3.0
    assert funds.current == 3.625
    assert rows["근원 PCE 물가"].longer_run is None
    assert all(row.current is not None for row in sep.rows)
    assert "FOMC SEP 2026-09-15 발표" in check_report(dashboard)


def test_one_missing_projection_series_keeps_the_rest() -> None:
    def no_unrate(request: httpx.Request) -> httpx.Response | None:
        if request.url.params.get("series_id") == "UNRATEMD":
            return httpx.Response(400)
        return None

    sep = build(service(wrap=no_unrate)).projections

    assert sep is not None
    assert sep.message == "FRED HTTP 400"
    rows = {row.label: row for row in sep.rows}
    assert all(cell.value is None for cell in rows["실업률"].years)
    assert rows["연방기금금리"].years[0].value == 3.6


def test_missing_keys_mark_only_keyed_sources() -> None:
    dashboard = build(service(env={}))

    states = {source.id: source.state for source in dashboard.sources}
    assert states["FRED"] == "missing_key"
    assert states["ECOS"] == "missing_key"
    assert states["TREASURY"] == "ok"
    assert states["ECB"] == "ok"
    assert dashboard.curve is None
    assert dashboard.projections is None
    assert dashboard.calendar == ()
    assert dashboard.regime[0].tag == "데이터 없음"


def test_one_failing_source_does_not_hide_the_others() -> None:
    def bis_down(request: httpx.Request) -> httpx.Response | None:
        return httpx.Response(503) if request.url.host == "stats.bis.org" else None

    dashboard = build(service(wrap=bis_down))

    states = {source.id: source.state for source in dashboard.sources}
    assert states["BIS"] == "error"
    assert states["FRED"] == "ok"
    rows = {row.id: row for panel in dashboard.panels for row in panel.rows}
    assert rows["boj_rate"].status == "error"
    assert rows["boj_rate"].message == "BIS HTTP 503"
    assert rows["us_unrate"].status == "ok"
    assert "BIS:WS_CBPOL/JP" in check_report(dashboard)


def test_snapshot_is_reused_until_refresh() -> None:
    requests: list[str] = []

    def count(request: httpx.Request) -> None:
        requests.append(str(request.url))

    svc = service(wrap=count)

    async def scenario() -> tuple[int, int, int]:
        _ = await svc.get()
        first = len(requests)
        _ = await svc.get()
        second = len(requests)
        _ = await svc.get(refresh=True)
        return first, second, len(requests)

    first, second, third = asyncio.run(scenario())
    assert first > 0
    assert second == first
    assert third == first * 2


def test_concurrent_callers_share_one_pass() -> None:
    requests: list[str] = []

    def count(request: httpx.Request) -> None:
        requests.append(str(request.url))

    svc = service(wrap=count)

    async def scenario() -> None:
        _ = await asyncio.gather(svc.get(), svc.get(refresh=True), svc.get())

    asyncio.run(scenario())
    single_pass = len(requests)
    requests.clear()
    _ = asyncio.run(svc.get(refresh=True))
    assert single_pass == len(requests)


def test_server_routes() -> None:
    app = create_app(service(), warm=False)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://local"
        ) as client:
            page = await client.get("/")
            assert page.status_code == 200
            assert "<title>FinAgent</title>" in page.text
            script = await client.get("/static/app.js")
            assert script.headers["content-type"].startswith("text/javascript")
            assert (await client.get("/static/app.css")).status_code == 200
            assert (await client.get("/static/http.py")).status_code == 404

            response = await client.get("/api/dashboard")
            payload = Dashboard.model_validate_json(response.text)
            assert payload.generated_at == NOW.isoformat()
            assert {source.id for source in payload.sources} >= {
                "FRED",
                "ECOS",
                "OECD",
            }
            refreshed = await client.get("/api/dashboard", params={"refresh": "true"})
            assert refreshed.status_code == 200
            assert refreshed.headers["cache-control"] == "no-store"

    asyncio.run(scenario())


def test_a_pass_writes_nothing_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    _ = build(service())

    assert list(Path(tmp_path).iterdir()) == []
