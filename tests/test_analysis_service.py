from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import httpx

from finagent import catalog
from finagent.analysis import mcp
from finagent.analysis.providers import DEFAULT_PROVIDER, registry
from finagent.analysis.providers.base import (
    ProviderError,
    ProviderStatus,
    TextDelta,
    Usage,
)
from finagent.analysis.providers.demo import DemoProvider
from finagent.analysis.service import AnalysisService, RunInput
from finagent.cli import provider_report
from finagent.demo import DEMO_KEYS, demo_transport
from finagent.server import create_app
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

    from finagent.analysis.providers.base import (
        Kind,
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
    )


class Recorder:
    """Answers with fixed text and keeps every request it was given."""

    id: ProviderId = "demo"
    title: str = "기록"
    kind: Kind = "demo"
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
    mcp_url: str | None = None,
) -> AnalysisService:
    return AnalysisService(
        svc or dashboard(),
        {},
        workspace=Path.cwd(),
        settings_path=settings_path,
        mcp_url=mcp_url,
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
        preset = await collect(service, RunInput("유동성", preset="liquidity"))
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
            if (await service.meta())["busy"]:
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


def test_settings_keep_each_others_keys(tmp_path: Path) -> None:
    path = tmp_path / "finagent.settings.json"
    save_analysis(path, AnalysisChoice("gemini_api", {"gemini_api": "m"}))
    save_view(path, frozenset({"curve"}))

    body = cast("dict[str, object]", json.loads(path.read_text("utf-8")))
    assert body == {
        "version": 1,
        "deep": ["curve"],
        "analysis": {"provider": "gemini_api", "models": {"gemini_api": "m"}},
    }
    assert load_analysis(tmp_path / "missing.json") is None
    _ = path.write_text('{"analysis": {"provider": 3}}', "utf-8")
    assert load_analysis(path) is None


def test_registry_and_api_key_status() -> None:
    providers = registry({}, demo=False)

    assert list(providers) == [
        "claude_sub",
        "claude_api",
        "codex_sub",
        "openai_api",
        "gemini_api",
    ]
    assert DEFAULT_PROVIDER == "claude_sub"
    assert providers["claude_sub"].default_model == "claude-opus-5-5"
    assert providers["openai_api"].default_model == "gpt-6-astra"
    assert providers["gemini_api"].default_model == "gemini-4-argon"
    assert list(registry({}, demo=True)) == ["demo"]

    async def keyed() -> list[ProviderStatus]:
        ids = ("claude_api", "openai_api", "gemini_api")
        return [await providers[i].status() for i in ids]

    assert [status.kind for status in asyncio.run(keyed())] == ["missing_key"] * 3
    with_key = registry({"GEMINI_API_KEY": "x"}, demo=False)["gemini_api"]
    assert asyncio.run(with_key.status()).ready


def test_provider_report_lists_every_provider() -> None:
    report = provider_report(
        {
            "claude_sub": ProviderStatus("ready", "로그인됨"),
            "openai_api": ProviderStatus("missing_key", "OPENAI_API_KEY가 없습니다"),
        }
    )

    lines = report.splitlines()
    assert lines[0] == "AI 공급자 (선택 기능):"
    assert "claude_sub" in lines[1]
    assert "ready" in lines[1]
    assert "OPENAI_API_KEY" in lines[2]


def sse(text: str) -> list[Event]:
    return [
        cast("Event", json.loads(line.removeprefix("data: ")))
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def test_analysis_routes() -> None:
    svc = dashboard()
    service = analysis(DemoProvider(delay=0), svc)
    app = create_app(svc, warm=False, analysis=service)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://local"
        ) as client:
            meta = cast(
                "dict[str, object]", (await client.get("/api/analysis/meta")).json()
            )
            presets = cast("list[dict[str, object]]", meta["presets"])
            assert len(presets) == 8
            assert meta["tools"] == ["search_series", "get_series", "source_guide"]
            providers = cast("list[dict[str, object]]", meta["providers"])
            assert providers[0]["status"] == "ready"
            assert meta["busy"] is False

            bad = await client.put("/api/analysis/choice", json={"provider": "x"})
            assert bad.status_code == 400
            good = await client.put(
                "/api/analysis/choice", json={"provider": "demo", "model": "m"}
            )
            assert good.json() == {"provider": "demo", "models": {"demo": "m"}}

            response = await client.post("/api/analysis", json={"request": "요약"})
            assert response.headers["content-type"].startswith("text/event-stream")
            events = sse(response.text)
            assert events[0]["type"] == "start"
            assert events[0]["model"] == "m"
            assert events[-1]["type"] == "done"
            assert (await client.get("/static/analysis.js")).status_code == 200

    asyncio.run(scenario())


class McpProbe(Recorder):
    """Calls FinAgent's /mcp endpoint the way the Codex runtime would."""

    def __init__(self) -> None:
        super().__init__()
        self.client: httpx.AsyncClient | None = None
        self.answers: list[httpx.Response] = []

    @override
    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        assert request.mcp is not None
        assert self.client is not None
        good = {"authorization": f"Bearer {request.mcp.token}"}
        bad = {"authorization": "Bearer wrong"}
        calls: list[tuple[dict[str, str], object]] = [
            (bad, {"jsonrpc": "2.0", "id": 0, "method": "ping"}),
            (good, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            (good, {"jsonrpc": "2.0", "method": "notifications/initialized"}),
            (good, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            (
                good,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "get_series", "arguments": {"id": "us_unrate"}},
                },
            ),
            (good, [{"jsonrpc": "2.0", "id": 4, "method": "nope"}]),
        ]
        for headers, body in calls:
            self.answers.append(
                await self.client.post("/mcp", json=body, headers=headers)
            )
        yield TextDelta("완료")


def test_mcp_endpoint_serves_tools_only_during_a_run() -> None:
    svc = dashboard()
    probe = McpProbe()
    service = analysis(probe, svc, mcp_url="http://local/mcp")
    transport = httpx.ASGITransport(app=create_app(svc, warm=False, analysis=service))

    async def scenario() -> list[Event]:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://local"
        ) as client:
            probe.client = client
            outside = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 9})
            assert outside.status_code == 401
            assert (await client.get("/mcp")).status_code == 405
            return await collect(service, RunInput("도구 확인"))

    events = asyncio.run(scenario())

    assert events[-1]["type"] == "done"
    assert [event["summary"] for event in of_type(events, "tool")] != []

    statuses = [answer.status_code for answer in probe.answers]
    assert statuses == [401, 200, 202, 200, 200, 200]
    bodies = [
        cast("object", answer.json()) for answer in probe.answers if answer.content
    ]
    init, listed, called, batch = (
        json.dumps(body, ensure_ascii=False) for body in bodies[1:]
    )
    assert '"name": "finagent"' in init
    assert all(
        f'"name": "{name}"' in listed
        for name in ("search_series", "get_series", "source_guide")
    )
    assert '"isError": false' in called
    assert "us_unrate" in called
    assert '"code": -32601' in batch
    assert not service.mcp_token_ok("")


def test_mcp_handle_without_a_run() -> None:
    service = analysis(Recorder())

    async def scenario() -> list[object]:
        return [
            await mcp.handle(service, "not an object"),
            await mcp.handle(service, {"jsonrpc": "2.0", "method": "x"}),
            await mcp.handle(
                service,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "get_series", "arguments": {}},
                },
            ),
        ]

    invalid, notification, call = asyncio.run(scenario())

    assert cast("dict[str, dict[str, object]]", invalid)["error"]["code"] == -32600
    assert notification is None
    result = cast("dict[str, dict[str, object]]", call)["result"]
    assert result["isError"] is True


def test_default_view_is_the_core_view() -> None:
    assert dashboard().view == catalog.CORE_VIEW
