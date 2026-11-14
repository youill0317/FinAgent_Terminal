from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast, override

import httpx

from finagent.analysis.context import Document
from finagent.analysis.providers.base import (
    ProviderError,
    ProviderStatus,
    TextDelta,
    Usage,
)
from finagent.analysis.providers.demo import DemoProvider
from finagent.analysis.service import (
    MAIN,
    AnalysisService,
    RunInput,
)
from finagent.display import Board
from finagent.service import DashboardService
from tests.fake_sources import DEMO_KEYS, demo_transport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from finagent.analysis.providers.base import (
        ProviderEvent,
        ProviderId,
        RunRequest,
    )
    from finagent.analysis.service import Brief, Prepare, Workspace
    from finagent.analysis.tools import ToolResult

type Event = dict[str, object]

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
WAIT: Final = 20.0


@dataclass(slots=True)
class Tape(Board):
    seen: list[Event] = field(default_factory=list)

    @override
    def apply(self, event: Event) -> None:
        self.seen.append(event)
        Board.apply(self, event)


def dashboard(*, fetched: bool = False) -> DashboardService:
    if not fetched:
        return DashboardService({}, clock=lambda: NOW, demo=True)
    transport = demo_transport(NOW.date())
    return DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: NOW,
        retry_delay=0,
    )


class Recorder:
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


class Delegating(Recorder):
    def __init__(self, calls: list[dict[str, object]]) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = calls
        self.gate: asyncio.Event = asyncio.Event()
        self.results: list[ToolResult] = []
        self.working: int = 0
        self.peak: int = 0

    @override
    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        self.seen.append(request)
        ask = request.conversation.request
        if request.kind != "main":
            self.working += 1
            self.peak = max(self.peak, self.working)
            try:
                _ = await self.gate.wait()
            finally:
                self.working -= 1
            yield TextDelta(f"결과: {ask} [[us_unrate]]")
        elif ask == "맡겨":
            for call in self.calls:
                self.results.append(await request.run_tool("start_thread", call))
            yield TextDelta("맡겼다")
        else:
            yield TextDelta("보고 받음" if not ask else "직접 답")


class Asking(Recorder):
    def __init__(self, *, relay: bool, held: bool = False) -> None:
        super().__init__()
        self.relay: bool = relay
        self.gate: asyncio.Event = asyncio.Event()
        self.slow: asyncio.Event = asyncio.Event()
        if not held:
            self.gate.set()

    @override
    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        self.seen.append(request)
        ask = request.conversation.request
        notes = request.conversation.request_document
        if request.kind != "main":
            if ask == "느린 조사":
                _ = await self.slow.wait()
                yield TextDelta("느린 결과")
                return
            if ask != "종목 조사":
                _ = await self.gate.wait()
                yield TextDelta("매크로는 중립")
                return
            reply = await request.run_tool("ask_main", {"message": "매크로 판단"})
            yield TextDelta(f"종목 결과 / {reply.text}")
        elif ask == "맡겨":
            _ = await request.run_tool("start_thread", research("종목"))
            yield TextDelta("맡겼다")
        elif ask == "하나 더":
            _ = await request.run_tool("start_thread", research("느린"))
            yield TextDelta("맡겼다")
        elif "<thread_request" in notes and self.relay:
            _ = await request.run_tool("start_thread", research("매크로"))
            yield TextDelta("매크로를 맡겼다")
        elif '<thread_report id="t2"' in notes:
            sent = {"thread": "t1", "message": "매크로는 중립"}
            _ = await request.run_tool("message_thread", sent)
            yield TextDelta("전했다")
        else:
            yield TextDelta("정리")


def analysis(
    provider: Recorder | DemoProvider,
    prepare: Prepare | None = None,
    *,
    fetched: bool = False,
) -> AnalysisService:
    return AnalysisService(
        dashboard(fetched=fetched), {}, providers={"demo": provider}, prepare=prepare
    )


async def until(
    ws: Workspace, events: list[Event], reached: Callable[[], bool] | None = None
) -> None:
    if not isinstance(ws.board, Tape):
        ws.board = Tape()
    tape = ws.board
    async with asyncio.timeout(WAIT):
        while True:
            await asyncio.sleep(0.001)
            idle = not ws.busy
            ws.drain()
            events.extend(tape.seen)
            tape.seen.clear()
            if reached() if reached is not None else idle:
                return


async def asked(service: AnalysisService, text: str) -> list[Event]:
    ws = service.open()
    events: list[Event] = []
    await ws.ask(text)
    await until(ws, events)
    return events


def of_thread(events: list[Event], thread: str) -> list[Event]:
    return [event for event in events if event["thread"] == thread]


def ended(events: list[Event], thread: str) -> list[object]:
    return [e["state"] for e in of_type(of_thread(events, thread), "end")]


def research(title: str) -> dict[str, object]:
    return {"title": title, "kind": "research", "task": f"{title} 조사"}


def of_type(events: list[Event], kind: str) -> list[Event]:
    return [event for event in events if event["type"] == kind]


def marks(done: Event) -> list[tuple[str, bool]]:
    lines = cast("list[list[dict[str, list[object]]]]", done["lines"])
    return [
        (str(cite), bool(known))
        for line in lines
        for piece in line
        if "c" in piece
        for cite, known in zip(piece["c"], piece["k"], strict=True)
    ]


def test_demo_run_streams_tools_text_and_a_cited_answer() -> None:
    service = analysis(DemoProvider(delay=0), fetched=True)

    events = asyncio.run(asked(service, "지금 국면은?"))

    kinds = [str(event["type"]) for event in events]
    assert kinds[:2] == ["turn", "start"]
    assert kinds[-2:] == ["done", "end"]
    assert "tool" in kinds
    assert "text" in kinds
    assert {event["thread"] for event in events} == {MAIN}
    done = events[-2]
    assert marks(done) == [("us_unrate", True)]
    assert done["data_calls"] == 1
    assert done["web_steps"] == 0
    assert done["received"] != []
    assert events[-1]["state"] == "done"


def test_a_failed_run_reports_the_error_and_keeps_no_turn() -> None:
    provider = Failing()
    service = analysis(provider)

    async def scenario() -> list[Event]:
        ws = service.open()
        events: list[Event] = []
        await ws.ask("질문")
        await until(ws, events)
        await ws.ask("다시")
        await until(ws, [])
        return events

    failed = asyncio.run(scenario())

    assert of_type(failed, "done") == []
    assert [e["message"] for e in of_type(failed, "error")] == ["한도 초과"]
    assert ended(failed, MAIN) == ["failed"]
    assert provider.seen[-1].conversation.turns == ()


def document(text: str, cite: str = "pv_1") -> Document:
    return Document(text, frozenset({cite}), "2026-10-12T09:00:00")


def prepared(brief: Brief) -> RunInput:
    return RunInput("지침", document("문서"), brief.kind)


def test_the_coordinator_keeps_talking_while_its_threads_work_in_parallel() -> None:
    provider = Delegating([research("금리"), research("환율")])
    service = analysis(provider)

    async def scenario() -> tuple[list[Event], list[Event], list[Event]]:
        ws = service.open()
        started: list[Event] = []
        talked: list[Event] = []
        reported: list[Event] = []
        await ws.ask("맡겨")
        await until(
            ws,
            started,
            lambda: (
                provider.working == len(provider.calls) and bool(ended(started, MAIN))
            ),
        )
        await ws.ask("다른 질문")
        await until(ws, talked, lambda: bool(ended(talked, MAIN)))
        assert provider.working == len(provider.calls)
        provider.gate.set()
        await until(ws, reported)
        return started, talked, reported

    started, talked, reported = asyncio.run(scenario())

    assert all(result.ok for result in provider.results)
    assert [e["thread"] for e in of_type(started, "thread")] == ["t1", "t2"]
    assert provider.peak == len(provider.calls)
    assert "직접 답" in "".join(str(e["text"]) for e in of_type(talked, "text"))
    assert ended(reported, "t1") == ["done"]
    assert ended(reported, "t2") == ["done"]
    workers = [seen for seen in provider.seen if seen.kind != "main"]
    assert {seen.conversation.request for seen in workers} == {"금리 조사", "환율 조사"}
    assert all(seen.conversation.turns == () for seen in workers)
    assert all(seen.web for seen in workers)
    woken = provider.seen[-1]
    assert woken.kind == "main"
    assert woken.conversation.request == ""
    assert woken.conversation.request_document.count("<thread_report") == len(
        provider.calls
    )
    assert "결과: 금리 조사" in woken.conversation.request_document
    assert 'state="done">금리</thread>' in woken.conversation.request_document
    assert [turn.request for turn in woken.conversation.turns] == ["맡겨", "다른 질문"]
    assert [e["question"] for e in of_type(of_thread(reported, MAIN), "turn")] == [""]
    assert marks(of_type(of_thread(reported, MAIN), "done")[0]) == []
    assert ended(reported, MAIN) == ["done"]


def test_stopping_one_thread_leaves_the_other_and_the_report_intact() -> None:
    provider = Delegating([research("금리"), research("환율")])
    service = analysis(provider)

    async def scenario() -> list[Event]:
        ws = service.open()
        events: list[Event] = []
        await ws.ask("맡겨")
        await until(ws, events, lambda: provider.working == len(provider.calls))
        await ws.stop("t1")
        provider.gate.set()
        await until(ws, events)
        return events

    events = asyncio.run(scenario())

    assert ended(events, "t1") == ["stopped"]
    assert ended(events, "t2") == ["done"]
    woken = provider.seen[-1].conversation.request_document
    assert woken.count("<thread_report") == len(provider.calls)
    assert '<thread_report id="t1" title="금리" state="stopped">' in woken
    assert 'state="stopped">금리</thread>' in woken


def test_a_thread_asks_the_coordinator_and_goes_on_with_the_relayed_answer() -> None:
    provider = Asking(relay=True)
    service = analysis(provider)

    events = asyncio.run(asked(service, "맡겨"))

    assert [e["thread"] for e in of_type(events, "thread")] == ["t1", "t2"]
    assert len(of_type(of_thread(events, "t1"), "turn")) == 1
    assert ended(events, "t1") == ["done"]
    assert ended(events, "t2") == ["done"]
    stock = "".join(str(e["text"]) for e in of_type(of_thread(events, "t1"), "text"))
    assert "매크로는 중립" in stock
    mains = [seen.conversation for seen in provider.seen if seen.kind == "main"]
    assert [talk.request for talk in mains] == ["맡겨", "", "", ""]
    assert '<thread_request id="t1" title="종목">\n매크로 판단' in (
        mains[1].request_document
    )
    assert 'state="waiting">종목</thread>' in mains[1].request_document
    assert '<thread_report id="t1"' in mains[3].request_document
    assert ended(events, MAIN) == ["done"] * 4
