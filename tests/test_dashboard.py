from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from finagent import catalog
from finagent.cli import check_report
from finagent.service import DashboardService
from tests.fake_sources import DEMO_KEYS, demo_closes, demo_transport

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

    from finagent.fetch import Closes
    from finagent.models import Dashboard

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def service(
    env: dict[str, str] | None = None,
    wrap: Callable[[httpx.Request], httpx.Response | None] | None = None,
    yahoo: Closes | None = None,
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
        yahoo=demo_closes(NOW.date()) if yahoo is None else yahoo,
    )


def build(svc: DashboardService, deep: frozenset[str] = catalog.ALL_VIEWS) -> Dashboard:
    return asyncio.run(svc.get(deep=deep))


def test_every_catalog_indicator_resolves_with_demo_data() -> None:
    dashboard = build(service())

    rows = [row for panel in dashboard.panels for row in panel.rows]
    assert [row.id for row in rows if row.status != "ok"] == []
    assert len(rows) == sum(len(spec.indicators) for spec in catalog.PANELS)
    assert all(source.state == "ok" for source in dashboard.sources)
    assert all(row.history for row in rows)


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


def test_a_pass_writes_nothing_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    _ = build(service())

    assert list(Path(tmp_path).iterdir()) == []
