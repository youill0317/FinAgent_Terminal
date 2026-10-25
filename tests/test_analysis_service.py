from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast, override

import httpx

from finagent.analysis.providers import DEFAULT_PROVIDER, registry
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
    load_view,
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


class Waiting(Recorder):
    def __init__(self) -> None:
        super().__init__()
        self.release: asyncio.Event = asyncio.Event()

    @override
    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        self.seen.append(request)
        _ = await self.release.wait()
        yield TextDelta("끝")


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


def test_follow_ups_stay_on_the_same_snapshot() -> None:
    clock = [NOW]
    svc = dashboard(clock)
    provider = Recorder()
    service = analysis(provider, svc)

    async def scenario() -> tuple[list[Event], list[Event]]:
        first = await collect(service, RunInput("실업률은?"))
        clock.append(NOW + timedelta(hours=1))
        _ = await svc.get(refresh=True)
        thread = str(first[0]["conversation"])
        second = await collect(service, RunInput("더 자세히", conversation=thread))
        return first, second

    first, second = asyncio.run(scenario())

    assert second[0]["conversation"] == first[0]["conversation"]
    assert second[0]["snapshot"] == first[0]["snapshot"] == NOW.isoformat()
    assert second[0]["stale"] is True
    later = provider.seen[-1].conversation
    assert [turn.request for turn in later.turns] == ["실업률은?"]
    assert later.request == "더 자세히"
    assert later.document == provider.seen[0].conversation.document


def test_preset_and_scope_widen_the_document_of_a_new_thread() -> None:
    provider = Recorder()
    service = analysis(provider)

    async def scenario() -> tuple[list[Event], list[Event]]:
        preset = await collect(service, RunInput("유동성", preset="macro"))
        whole = await collect(service, RunInput("전부", scope="all"))
        return preset, whole

    preset, whole = asyncio.run(scenario())

    assert str(preset[0]["view"]).startswith("지금 보기 + ")
    # Building a wider document is not a refresh.
    assert preset[0]["stale"] is False
    assert whole[0]["stale"] is False
    assert whole[0]["view"] == "전체 매크로"
    assert preset[0]["conversation"] != whole[0]["conversation"]
    sizes = [len(request.conversation.document) for request in provider.seen]
    assert sizes[0] < sizes[1]


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


def test_only_one_run_at_a_time() -> None:
    provider = Waiting()
    service = analysis(provider)

    async def scenario() -> tuple[list[Event], list[Event]]:
        first = asyncio.create_task(collect(service, RunInput("하나")))
        for _ in range(200):
            await asyncio.sleep(0)
            if service.busy:
                break
        second = await collect(service, RunInput("둘"))
        provider.release.set()
        return await first, second

    first, second = asyncio.run(scenario())

    assert second == [{"type": "error", "message": "이미 분석이 실행 중입니다"}]
    assert first[-1]["type"] == "done"


def test_empty_request_and_unknown_provider_are_refused() -> None:
    service = analysis(Recorder())

    async def scenario() -> tuple[list[Event], list[Event]]:
        empty = await collect(service, RunInput("  "))
        unknown = await collect(service, RunInput("질문", provider="nope"))
        return empty, unknown

    empty, unknown = asyncio.run(scenario())

    assert empty == [{"type": "error", "message": "요청을 입력해 주세요"}]
    assert unknown[0]["type"] == "error"


def test_the_choice_is_saved_next_to_the_view(tmp_path: Path) -> None:
    path = tmp_path / "finagent.settings.json"
    save_view(path, frozenset({"housing"}))
    service = analysis(Recorder(), settings_path=path)

    picked = service.choose("demo", "custom-model")

    assert picked == AnalysisChoice("demo", {"demo": "custom-model"})
    assert load_analysis(path) == picked
    assert load_view(path) == (frozenset({"housing"}), None)
    assert service.model_for("demo") == "custom-model"
    again = analysis(Recorder(), settings_path=path)
    assert again.choice == picked
    try:
        _ = service.choose("nope")
    except ProviderError:
        pass
    else:
        raise AssertionError


def test_the_scope_is_saved_and_loaded(tmp_path: Path) -> None:
    path = tmp_path / "finagent.settings.json"
    service = analysis(Recorder(), settings_path=path)
    assert service.choice.scope == "view"

    picked = service.choose("demo", scope="all")
    kept = service.choose("demo", "m")  # no scope given keeps the saved one

    assert picked.scope == "all"
    assert kept.scope == "all"
    assert load_analysis(path) == kept
    assert analysis(Recorder(), settings_path=path).choice.scope == "all"


def test_a_saved_provider_missing_from_this_run_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "finagent.settings.json"
    save_analysis(path, AnalysisChoice("gemini_api", {"gemini_api": "m"}, "all"))

    service = analysis(Recorder(), settings_path=path)

    assert service.choice == AnalysisChoice("demo", {"gemini_api": "m"}, "all")


def test_a_file_without_a_valid_scope_loads_with_the_default(tmp_path: Path) -> None:
    path = tmp_path / "finagent.settings.json"
    old = '{"analysis": {"provider": "demo", "models": {"demo": "m"}}}'
    _ = path.write_text(old, "utf-8")
    assert load_analysis(path) == AnalysisChoice("demo", {"demo": "m"}, "view")

    for bad in ('"nope"', "3", "null"):
        _ = path.write_text(
            '{"analysis": {"provider": "demo", "scope": ' + bad + "}}", "utf-8"
        )
        loaded = load_analysis(path)
        assert loaded is not None
        assert loaded.scope == "view"


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


def test_registry_and_api_key_status() -> None:
    providers = registry({}, demo=False)

    assert DEFAULT_PROVIDER in providers
    assert list(registry({}, demo=True)) == ["demo"]

    async def keyed() -> list[ProviderStatus]:
        ids = ("claude_api", "openai_api", "gemini_api")
        return [await providers[i].status() for i in ids]

    assert [status.kind for status in asyncio.run(keyed())] == ["missing_key"] * 3
    with_key = registry({"GEMINI_API_KEY": "x"}, demo=False)["gemini_api"]
    assert asyncio.run(with_key.status()).ready
