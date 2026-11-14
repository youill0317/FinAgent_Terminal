from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest

from finagent import catalog
from finagent.dashboard import (
    BuildProgress,
    all_fetches,
    build_dashboard,
)
from finagent.service import DashboardService
from tests.fake_sources import DEMO_KEYS, demo_transport

if TYPE_CHECKING:
    from collections.abc import Callable

    from finagent.models import Dashboard

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
OECD_REQUESTS = 4


def service(
    wrap: Callable[[httpx.Request], httpx.Response | None] | None = None,
) -> DashboardService:
    demo = demo_transport(NOW.date())

    def handler(request: httpx.Request) -> httpx.Response:
        if wrap is not None and (override := wrap(request)) is not None:
            return override
        return demo.handle_request(request)

    transport = httpx.MockTransport(handler)
    return DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: NOW,
        retry_delay=0,
    )


def build(svc: DashboardService) -> Dashboard:
    return asyncio.run(svc.get())


def test_a_cold_build_resolves_every_indicator_and_counts_every_fetch() -> None:
    oecd: set[httpx.URL] = set()
    progress = BuildProgress()

    def record(request: httpx.Request) -> httpx.Response | None:
        if request.url.host == "sdmx.oecd.org":
            oecd.add(request.url)
        return None

    async def run() -> Dashboard:
        async with service(wrap=record).fetch_context() as ctx:
            return await build_dashboard(
                ctx, NOW, cache={}, refresh=False, stale={}, progress=progress
            )

    dashboard = asyncio.run(run())

    rows = [row for panel in dashboard.panels for row in panel.rows]
    assert [row.id for row in rows if row.status != "ok"] == []
    assert len(rows) == sum(len(spec.indicators) for spec in catalog.PANELS)
    assert all(source.state == "ok" for source in dashboard.sources)
    assert all(row.history for row in rows)
    millions = {row.id: row.value for row in rows}["us_fed_assets_mp"]
    assert millions == pytest.approx(6.55, abs=0.01)
    assert len(oecd) == OECD_REQUESTS
    assert progress.series + progress.etfs == progress.count == len(all_fetches())
    assert progress.done == progress.total
    assert {page for _, page in progress.total} == {s.page for s in catalog.PANELS}
    assert progress.extra == "done"


def test_a_failed_refresh_keeps_old_values_and_a_retry_fetches_only_the_failed() -> (
    None
):
    limited = False
    asked: list[httpx.URL] = []

    def fred_429(request: httpx.Request) -> httpx.Response | None:
        asked.append(request.url)
        if limited and request.url.host == "api.stlouisfed.org":
            return httpx.Response(429, headers={"retry-after": "0"})
        return None

    svc = service(wrap=fred_429)
    before = build(svc)
    limited = True
    after = asyncio.run(svc.get(refresh=True))

    old = {row.id: row for panel in before.panels for row in panel.rows}
    new = {row.id: row for panel in after.panels for row in panel.rows}
    assert new["us_unrate"].status == "ok"
    assert new["us_unrate"].value == old["us_unrate"].value
    assert new["us_unrate"].message == "갱신 실패, 이전 값: FRED 요청 한도 초과"
    assert old["us_unrate"].message is None
    states = {source.id: source.state for source in after.sources}
    assert states["FRED"] == "error"
    assert svc.full_series("us_unrate") != ()

    limited = False
    asked.clear()
    fixed = asyncio.run(svc.retry_failed())

    assert {source.id: source.state for source in fixed.sources}["FRED"] == "ok"
    rows = [row for panel in fixed.panels for row in panel.rows]
    assert [row.message for row in rows if row.message] == []
    assert "ecos.bok.or.kr" not in {url.host for url in asked}
    assert not any(is_extra(url) for url in asked)


def is_extra(url: httpx.URL) -> bool:
    return "release" in url.path or "realtime_start" in url.params
