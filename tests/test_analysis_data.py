from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

import httpx

from finagent.analysis.context import build_document
from finagent.analysis.evidence import Evidence
from finagent.analysis.tools import Toolbox
from finagent.analysis.verify import verify
from finagent.demo import DEMO_KEYS, demo_closes, demo_transport
from finagent.service import DashboardService

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent.analysis.context import Document
    from finagent.analysis.tools import ToolResult

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def service(env: dict[str, str] | None = None) -> DashboardService:
    transport = demo_transport(NOW.date())
    return DashboardService(
        DEMO_KEYS if env is None else env,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: NOW,
        retry_delay=0,
        yahoo=demo_closes(NOW.date()),
    )


def document(svc: DashboardService) -> Document:
    return build_document(asyncio.run(svc.get()), view="핵심")


def call(name: str, args: Mapping[str, object], svc: DashboardService) -> ToolResult:
    async def run() -> ToolResult:
        async with svc.fetch_context() as ctx:
            box = Toolbox(ctx, svc.series_cache, svc.today())
            return await box.run(name, args)

    return asyncio.run(run())


def body(result: ToolResult) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(result.text))


# ---------------------------------------------------------------- verify


EVIDENCE: Final[dict[str, Evidence]] = {
    "us_unrate": Evidence("us_unrate", "실업률", (4.1, 4.2, 4.3)),
    "us_10y2y": Evidence("us_10y2y", "10년 − 2년", (0.54, 0.57)),
}


def counts(text: str) -> dict[str, int]:
    return {str(key): value for key, value in verify(text, EVIDENCE).counts.items()}


def test_cited_numbers_are_checked_at_the_precision_written() -> None:
    report = verify("실업률은 9월 4.3%로 직전 4.2%에서 올랐다. [[us_unrate]]", EVIDENCE)

    assert report.counts["ok"] == 2
    assert report.counts["mismatch"] == 0
    assert report.cited == {"us_unrate"}


def test_wrong_numbers_report_the_closest_real_value() -> None:
    report = verify("실업률은 4.5%까지 올랐다. [[us_unrate]]", EVIDENCE)

    lines = cast("list[list[dict[str, object]]]", report.to_json()["lines"])
    numbers = [piece for line in lines for piece in line if "n" in piece]
    assert numbers == [{"n": "4.5%", "s": "mismatch", "a": "4.3"}]


# ---------------------------------------------------------------- tools


def test_tool_errors_are_results_not_exceptions() -> None:
    svc = service()

    unknown = call("get_series", {"id": "nope"}, svc)
    bad_code = call("get_series", {"source": "ECOS", "code": "bad"}, svc)
    no_key = call("get_series", {"id": "kr_base"}, service({"FRED_API_KEY": "demo"}))

    assert not unknown.ok
    assert "search_series" in cast("str", body(unknown)["error"])
    assert not bad_code.ok
    assert "ECOS_API_KEY" in cast("str", body(no_key)["error"])
