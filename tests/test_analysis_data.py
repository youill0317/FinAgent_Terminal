from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import httpx

from finagent.analysis import prompt, tools
from finagent.analysis.company_tools import Company
from finagent.analysis.tools import Toolbox
from finagent.service import DashboardService
from tests.fake_sources import DEMO_KEYS, demo_transport

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent.analysis.tools import ToolResult

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def service(env: dict[str, str] | None = None) -> DashboardService:
    transport = demo_transport(NOW.date())
    return DashboardService(
        DEMO_KEYS if env is None else env,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: NOW,
        retry_delay=0,
    )


def call(name: str, args: Mapping[str, object], svc: DashboardService) -> ToolResult:
    async def run() -> ToolResult:
        async with svc.fetch_context() as ctx:
            box = Toolbox(ctx, svc.series_store, svc.today(), Company({}, svc.today()))
            return await box.run(name, args)

    return asyncio.run(run())


def body(result: ToolResult) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(result.text))


def test_tool_errors_are_results_not_exceptions() -> None:
    svc = service()

    unknown = call("get_series", {"id": "nope"}, svc)
    bad_code = call("get_series", {"source": "ECOS", "code": "bad"}, svc)
    no_key = call("get_series", {"id": "kr_usdkrw"}, service({"FRED_API_KEY": "demo"}))

    assert not unknown.ok
    assert "search_series" in cast("str", body(unknown)["error"])
    assert not bad_code.ok
    assert "ECOS_API_KEY" in cast("str", body(no_key)["error"])


def test_data_map_only_names_tools_that_exist_and_answer() -> None:
    map_text = prompt.text("data_map.txt")
    names = {name for name, _, _ in tools.TOOL_SPECS}
    named = set(cast("list[str]", re.findall(r"[a-z]+_[a-z_]+", map_text)))
    assert named
    assert named <= names
    assert names - {"source_guide"} <= named

    svc = service()
    samples = data_map_samples()

    async def run(name: str) -> ToolResult:
        dashboard = await svc.get()
        async with svc.fetch_context() as ctx:
            box = Toolbox(
                ctx,
                svc.series_store,
                svc.today(),
                Company({}, svc.today(), demo=True),
                dashboard=dashboard,
                portfolios=lambda _: [{"name": "기본", "cite_as": "pf_a"}],
            )
            return await box.run(name, samples[name])

    for name in named:
        assert asyncio.run(run(name)).ok, name


def data_map_samples() -> dict[str, dict[str, object]]:
    return {
        "search_series": {"query": "CPI"},
        "get_series": {"id": "us_unrate"},
        "get_prices": {"tickers": ["AAPL"]},
        "company_facts": {"ticker": "AAPL"},
        "company_filings": {"ticker": "AAPL"},
        "read_filing": {"ticker": "AAPL", "accession": "0000320193-26-000071"},
        "etf_lookup": {"limit": 1},
        "etf_holdings": {"ticker": "XLK"},
        "company_profile": {"ticker": "AAPL"},
        "read_dashboard": {"page": "overview"},
        "read_portfolio": {},
    }
