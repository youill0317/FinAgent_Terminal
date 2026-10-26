from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast, override

import httpx

from finagent.analysis.providers.base import (
    ProviderError,
    ProviderStatus,
    TextDelta,
    Usage,
)
from finagent.analysis.providers.demo import DemoProvider
from finagent.analysis.service import AnalysisService, RunInput
from finagent.demo import DEMO_KEYS, demo_closes, demo_transport
from finagent.service import DashboardService
from finagent.settings import (
    AnalysisChoice,
    load_analysis,
    save_analysis,
    save_view,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from finagent.analysis.providers.base import (
        ProviderEvent,
        ProviderId,
        RunRequest,
    )

type Event = dict[str, object]

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def dashboard(clock: list[datetime] | None = None) -> DashboardService:
    transport = demo_transport(NOW.date())
    times = clock if clock is not None else [NOW]
    return DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: times[-1],
        retry_delay=0,
        yahoo=demo_closes(NOW.date()),
    )


class Recorder:
    """Answers with fixed text and keeps every request it was given."""

    id: ProviderId = "demo"
    title: str = "기록"
    default_model: str = "fixed"

    def __init__(self, text: str = "실업률은 4.1%다. [[us_unrate]]") -> None:
        self.text: str = text
        self.seen: list[RunRequest] = []

    async def status(self) -> ProviderStatus:
        return ProviderStatus("ready")

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        self.seen.append(request)
        yield TextDelta(self.text)
        yield Usage(10, 5)


class Failing(Recorder):
    @override
    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        self.seen.append(request)
        yield TextDelta("앞부분")
        msg = "한도 초과"
        raise ProviderError(msg)


def analysis(
    provider: Recorder | DemoProvider,
    svc: DashboardService | None = None,
    settings_path: Path | None = None,
) -> AnalysisService:
    return AnalysisService(
        svc or dashboard(),
        {},
        settings_path=settings_path,
        providers={"demo": provider},
    )


async def collect(service: AnalysisService, given: RunInput) -> list[Event]:
    return [event async for event in service.run(given)]


def of_type(events: list[Event], kind: str) -> list[Event]:
    return [event for event in events if event["type"] == kind]


def test_demo_run_streams_tools_text_and_a_verified_answer() -> None:
    service = analysis(DemoProvider(delay=0))

    events = asyncio.run(collect(service, RunInput("지금 국면은?")))

    kinds = [str(event["type"]) for event in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "tool" in kinds
    assert "text" in kinds
    start = events[0]
    assert start["stale"] is False
    assert start["view"] == "지금 보기"
    done = events[-1]
    report = cast("dict[str, object]", done["verify"])
    counts = cast("dict[str, int]", report["counts"])
    assert counts["ok"] >= 1
    assert counts["mismatch"] >= 1
    assert counts["unsupported"] >= 1
    assert done["tool_calls"] == 1
    assert done["received"] != []


def test_a_failed_run_reports_the_error_and_keeps_no_turn() -> None:
    provider = Failing()
    service = analysis(provider)

    async def scenario() -> tuple[list[Event], list[Event]]:
        failed = await collect(service, RunInput("질문"))
        thread = str(failed[0]["conversation"])
        again = await collect(service, RunInput("다시", conversation=thread))
        return failed, again

    failed, _ = asyncio.run(scenario())

    assert of_type(failed, "done") == []
    assert of_type(failed, "error") == [{"type": "error", "message": "한도 초과"}]
    assert provider.seen[-1].conversation.turns == ()


def test_settings_keep_each_others_keys(tmp_path: Path) -> None:
    path = tmp_path / "finagent.settings.json"
    save_analysis(path, AnalysisChoice("gemini_api", {"gemini_api": "m"}))
    save_view(path, frozenset({"curve"}))

    body = cast("dict[str, object]", json.loads(path.read_text("utf-8")))
    assert body == {
        "version": 1,
        "deep": ["curve"],
        "analysis": {
            "provider": "gemini_api",
            "models": {"gemini_api": "m"},
            "scope": "view",
        },
    }
    assert load_analysis(tmp_path / "missing.json") is None
    _ = path.write_text('{"analysis": {"provider": 3}}', "utf-8")
    assert load_analysis(path) is None
