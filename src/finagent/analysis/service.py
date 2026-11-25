"""AI 분석 서비스·작업 공간(Workspace)·분석 에이전트 팀 워크플로우 오케스트레이션."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from queue import Empty, SimpleQueue
from threading import Lock
from typing import TYPE_CHECKING, Final, Literal, cast, final
from xml.sax.saxutils import escape, quoteattr

from finagent.analysis import agents, cites, prompt, web
from finagent.analysis.agents import BEAR, BULL, POOL, SYNTH, WORKFLOWS, Output
from finagent.analysis.company_tools import Company, text_arg
from finagent.analysis.prompt import Conversation, ToolRecord, Turn, system_prompt
from finagent.analysis.providers import DEFAULT_PROVIDER, registry
from finagent.analysis.providers.base import (
    WEB_STEP_LIMIT,
    ProviderError,
    RunRequest,
    TextDelta,
    ToolRound,
    ToolSpec,
    Usage,
    WebFound,
)
from finagent.analysis.tools import (
    MAX_TOOL_CALLS,
    TOOL_SPECS,
    PortfolioReader,
    Toolbox,
    ToolResult,
)
from finagent.display import STATE_TEXT, Board
from finagent.limits import INTERACTIVE_WAIT
from finagent.settings import AnalysisChoice, load_analysis, save_analysis

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Callable,
        Collection,
        Coroutine,
        Mapping,
        Sequence,
    )
    from pathlib import Path

    from finagent.analysis.agents import Agent, Seat, Workflow
    from finagent.analysis.context import Document
    from finagent.analysis.providers.base import Provider, ProviderStatus, StepKind
    from finagent.analysis.web import WebSource
    from finagent.models import Dashboard
    from finagent.service import DashboardService

type Event = dict[str, object]
type ThreadState = Literal["working", "done", "failed", "stopped"]
type Prepare = Callable[[Brief], RunInput]
type Work = Coroutine[object, object, str]
MAIN: Final = ""
ERROR_LIMIT: Final = 400
PRESET_TOOL_CALLS: Final = 40
KINDS: Final = ("macro", "stock", "etf", "research")
STOPPED: Final = "사용자 중지"
UNANSWERED: Final = "조정자 응답 없음. 보유 자료로 진행할 것"
NO_TOOLS: Final = "페이지 챗봇은 도구 미사용"
UNAVAILABLE: Final = "이 에이전트에 없는 도구"
NO_REPORTS: Final = "수집 에이전트 전원 실패"
SYNTH_FAILED: Final = "종합 에이전트 실패: {failure}"
SYNTH_QUESTION: Final = "결론·시나리오·최종 판단 작성"
ROUND_QUESTION: Final = "{number}라운드"
STAGE_STEP: Final = "{stage} 시작 · {names}"
ATTACH_SHAPE: Final = "attach는 {thread, sections}의 목록"
ATTACH: Final = {
    "type": "array",
    "description": "함께 넘길 다른 스레드의 마지막 답변",
    "items": {
        "type": "object",
        "properties": {
            "thread": {"type": "string", "description": "답변을 가져올 스레드 id"},
            "sections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "가져올 ## 제목 · 비우면 전체",
            },
        },
        "required": ["thread"],
        "additionalProperties": False,
    },
}
AGENT_CHOICES: Final = ", ".join(f"{agent.id} {agent.title}" for agent in POOL.values())
SPECS: Final = tuple(ToolSpec(name, text, schema) for name, text, schema in TOOL_SPECS)
THREAD_SPECS: Final = (
    ToolSpec(
        "start_thread",
        prompt.text("tool_start_thread.txt"),
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "스레드 이름 · 20자 안쪽"},
                "kind": {"type": "string", "enum": list(KINDS)},
                "ticker": {"type": "string", "description": "stock, etf의 티커"},
                "task": {"type": "string", "description": "스레드에 줄 작업 지시"},
                "agent": {
                    "type": "string",
                    "enum": list(POOL),
                    "description": f"research에 부여할 에이전트 역할 · {AGENT_CHOICES}",
                },
                "attach": ATTACH,
            },
            "required": ["title", "kind"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "message_thread",
        prompt.text("tool_message_thread.txt"),
        {
            "type": "object",
            "properties": {
                "thread": {"type": "string", "description": "스레드 id"},
                "message": {"type": "string"},
                "attach": ATTACH,
            },
            "required": ["thread", "message"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "read_thread",
        prompt.text("tool_read_thread.txt"),
        {
            "type": "object",
            "properties": {
                "thread": {"type": "string", "description": "스레드 id"},
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "읽을 ## 제목 · 비우면 전체",
                },
            },
            "required": ["thread"],
            "additionalProperties": False,
        },
    ),
)
ASK_SPECS: Final = (
    ToolSpec(
        "ask_main",
        prompt.text("tool_ask_main.txt"),
        {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "조정자에게 보낼 요청"},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    ),
)
ASKING: Final = frozenset(spec.name for spec in ASK_SPECS)
DELEGATED: Final = frozenset(spec.name for spec in (*THREAD_SPECS, *ASK_SPECS))


@dataclass(slots=True)
class RunTrace:
    data_calls: int = 0
    web_steps: int = 0

    def step(self, summary: str, *, ok: bool, kind: StepKind) -> Event:
        self.data_calls += kind == "data"
        self.web_steps += kind == "web" and ok
        return {"type": "tool", "summary": summary, "ok": ok, "kind": kind}


@dataclass(slots=True)
class Thread:
    id: str
    title: str = ""
    kind: str = "main"
    state: ThreadState = "working"
    documents: list[Document] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    received: dict[str, str] = field(default_factory=dict)
    web: dict[str, WebSource] = field(default_factory=dict)
    parent: str = ""
    agent: Agent | None = None
    team: list[Thread] = field(default_factory=list)
    calls: int = 0
    searches: int = 0

    @property
    def answer(self) -> str:
        return self.turns[-1].answer if self.turns else ""


@dataclass(frozen=True, slots=True)
class RunInput:
    request: str
    document: Document | None = None
    kind: str = ""
    guide: str = ""


@dataclass(frozen=True, slots=True)
class Brief:
    kind: str
    ticker: str
    portfolio: str
    dashboard: Dashboard
    company: Company


@dataclass(frozen=True, slots=True)
class _Lent:
    source: Thread
    names: tuple[str, ...]
    body: str


@dataclass(slots=True)
class _Ask:
    text: str
    reply: asyncio.Future[str]
    sent: bool = False
    helpers: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _Order:
    flow: Workflow
    given: RunInput
    task: str
    notes: str

    def kind(self, agent: Agent) -> str:
        return f"{self.flow.kind}:{agent.id}"

    def collect(self, seat: Seat) -> RunInput:
        request = agents.collect_request(self.flow, seat, self.given.guide, self.task)
        return RunInput(request, self.given.document, self.kind(seat.agent))

    def round(self, agent: Agent, number: int) -> RunInput:
        if number > 1:
            return RunInput(agents.round_request(agent, number), kind=self.kind(agent))
        request = agents.debate_request(self.flow, agent, self.task)
        return RunInput(request, self.given.document, self.kind(agent))

    def synth(self) -> RunInput:
        request = agents.synth_request(self.flow, self.given.guide, self.task)
        return RunInput(request, self.given.document, self.kind(SYNTH))


@dataclass(slots=True)
class _Job:
    given: RunInput
    thread: Thread
    provider: Provider
    model: str
    added: Document | None
    notes: str = ""
    answer: str = ""
    completed: bool = False
    records: list[ToolRecord] = field(default_factory=list)
    found: dict[str, WebSource] = field(default_factory=dict)
    ids: set[str] = field(default_factory=set)
    trace: RunTrace = field(default_factory=RunTrace)

    def __post_init__(self) -> None:
        self.found = dict(self.thread.web)
        added = self.added.ids if self.added is not None else ()
        self.ids = {*self.thread.ids, *added}

    @property
    def searching(self) -> bool:
        return self.thread.id != MAIN

    @property
    def web(self) -> bool:
        agent = self.thread.agent
        return self.searching and (agent is None or agent.web)

    @property
    def limit(self) -> int:
        return PRESET_TOOL_CALLS if self.searching else MAX_TOOL_CALLS

    @property
    def kind(self) -> str:
        return self.given.kind or ("page" if self.searching else "main")

    @property
    def added_text(self) -> str:
        added = self.added.text if self.added is not None else ""
        return "\n\n".join(filter(None, (added, self.notes)))

    def turn(self) -> Turn:
        return Turn(
            self.given.request.strip(),
            self.answer,
            tuple(self.records),
            self.added_text,
        )

    def conversation(self, system: str) -> Conversation:
        first = self.thread.documents[:1]
        return Conversation(
            system,
            tuple(filter(None, (part for d in first for part in (d.base, d.text)))),
            tuple(self.thread.turns),
            self.given.request.strip(),
            self.added_text,
        )

    def commit(self, ids: set[str], received: dict[str, str]) -> None:
        thread = self.thread
        thread.turns.append(self.turn())
        if self.added is not None:
            thread.documents.append(self.added)
        thread.ids = ids
        thread.received.update(received)
        thread.web = self.found


@final
class AnalysisService:
    __slots__ = (
        "_choice",
        "_dashboard",
        "_env",
        "_portfolios",
        "_prepare",
        "_providers",
        "_settings_path",
    )

    def __init__(  # noqa: PLR0913  # 대시보드·공급자·브리핑·포트폴리오 의존성 주입
        self,
        dashboard: DashboardService,
        env: Mapping[str, str],
        *,
        settings_path: Path | None = None,
        providers: Mapping[str, Provider] | None = None,
        prepare: Prepare | None = None,
        portfolios: PortfolioReader | None = None,
    ) -> None:
        self._dashboard = dashboard
        self._env = env
        self._providers = (
            dict(providers)
            if providers is not None
            else registry(env, demo=dashboard.demo)
        )
        self._settings_path = settings_path
        self._prepare = prepare
        self._portfolios = portfolios
        saved = load_analysis(settings_path) if settings_path is not None else None
        first = DEFAULT_PROVIDER if DEFAULT_PROVIDER in self._providers else ""
        fallback = first or next(iter(self._providers))
        if saved is not None and saved.provider not in self._providers:
            saved = replace(saved, provider=fallback)
        self._choice = saved or AnalysisChoice(fallback)

    @property
    def choice(self) -> AnalysisChoice:
        return self._choice

    @property
    def providers(self) -> Mapping[str, Provider]:
        return self._providers

    def model_for(self, provider_id: str, *, thread: bool = False) -> str:
        provider = self._providers[provider_id]
        own = self._choice.thread_models.get(provider_id) if thread else None
        return own or self._choice.models.get(provider_id) or provider.default_model

    def choose(
        self,
        provider_id: str,
        model: str = "",
        instructions: str | None = None,
        thread_model: str = "",
    ) -> AnalysisChoice:
        if provider_id not in self._providers:
            msg = f"알 수 없는 공급자: {provider_id}"
            raise ProviderError(msg)
        models = dict(self._choice.models)
        if model.strip():
            models[provider_id] = model.strip()
        else:
            _ = models.pop(provider_id, None)
        thread_models = dict(self._choice.thread_models)
        if thread_model.strip():
            thread_models[provider_id] = thread_model.strip()
        else:
            _ = thread_models.pop(provider_id, None)
        text = (
            self._choice.instructions if instructions is None else instructions.strip()
        )
        self._choice = AnalysisChoice(provider_id, models, text, thread_models)
        if self._settings_path is not None:
            save_analysis(self._settings_path, self._choice)
        return self._choice

    async def statuses(self) -> dict[str, ProviderStatus]:
        ids = list(self._providers)
        found = await asyncio.gather(*(self._providers[i].status() for i in ids))
        return dict(zip(ids, found, strict=True))

    async def unready(self) -> str:
        status = await self._providers[self._choice.provider].status()
        return "" if status.ready else status.message or status.kind

    def opening(self) -> Event:
        provider_id = self._choice.provider
        return {
            "type": "start",
            "provider_title": self._providers[provider_id].title,
            "model": self.model_for(provider_id, thread=True),
        }

    def open(self, *, chat: bool = False) -> Workspace:
        return Workspace(self, chat=chat)

    def company(self) -> Company:
        return Company(
            self._env,
            self._dashboard.today(),
            demo=self._dashboard.demo,
            settings=self._settings_path,
        )

    async def prepare(
        self, kind: str, ticker: str, portfolio: str, company: Company
    ) -> RunInput:
        if self._prepare is None:
            msg = "현재 실행 환경 내 보고서 분석 준비 미지원"
            raise ValueError(msg)
        brief = Brief(kind, ticker, portfolio, await self._dashboard.get(), company)
        return await asyncio.to_thread(self._prepare, brief)

    async def turn(
        self,
        thread: Thread,
        given: RunInput,
        desk: Workspace,
        *,
        notes: str = "",
    ) -> AsyncGenerator[Event]:
        provider_id = self._choice.provider
        provider = self._providers[provider_id]
        status = await provider.status()
        if not status.ready:
            yield {"type": "error", "message": status.message or status.kind}
            return
        model = self.model_for(provider_id, thread=thread.id != MAIN)
        dashboard = await self._dashboard.get()
        if not thread.documents and given.document is not None:
            thread.documents.append(given.document)
            thread.ids |= given.document.ids
        job = _Job(given, thread, provider, model, _new_document(thread, given), notes)
        desk.company.today = self._dashboard.today()
        yield {
            "type": "start",
            "provider_title": provider.title,
            "model": model,
        }
        specs = () if desk.chat else _specs(thread)
        received: dict[str, str] = {}
        if not specs:
            async for event in self._answer(job, specs, None, desk):
                yield event
        else:
            async with self._dashboard.fetch_context(max_wait=INTERACTIVE_WAIT) as ctx:
                box = Toolbox(
                    ctx,
                    self._dashboard.series_store,
                    self._dashboard.today(),
                    desk.company,
                    in_document=frozenset(job.ids),
                    demo=self._dashboard.demo,
                    limit=job.limit,
                    dashboard=dashboard,
                    portfolios=self._portfolios,
                )
                async for event in self._answer(job, specs, box, desk):
                    yield event
            received = box.received
        if not job.completed:
            return
        ids = {*job.ids, *received}
        job.commit(ids, received)
        yield _done(
            job.answer,
            job.thread,
            ids,
            job.found,
            counts=(job.trace.data_calls, job.trace.web_steps),
            report=bool(job.given.kind) and not job.thread.parent,
        )

    async def _answer(
        self,
        job: _Job,
        specs: tuple[ToolSpec, ...],
        box: Toolbox | None,
        desk: Workspace,
    ) -> AsyncGenerator[Event]:
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        main = job.thread.id == MAIN
        names = {spec.name for spec in specs}

        async def run_tool(name: str, arguments: Mapping[str, object]) -> ToolResult:
            delegated = name in DELEGATED
            if box is None or name not in names:
                result = _refused(name, NO_TOOLS if desk.chat else UNAVAILABLE)
            elif delegated:
                result = await desk.use(job.thread, name, arguments)
            else:
                result = await box.run(name, arguments)
            if delegated:
                job.ids |= job.thread.ids
                job.found.update(job.thread.web)
            if not (delegated and main):
                job.records.append(
                    ToolRecord(
                        name,
                        json.dumps(dict(arguments), ensure_ascii=False),
                        result.text,
                        result.ok,
                    )
                )
            kind: StepKind = "thread" if delegated else "data"
            await queue.put(job.trace.step(result.summary, ok=result.ok, kind=kind))
            return result

        system = system_prompt(
            self._choice.instructions,
            web_limit=WEB_STEP_LIMIT if job.web else 0,
            coordinator=main and not desk.chat,
            asking=not main and bool(names & ASKING),
            team=bool(job.thread.parent),
            chat=desk.chat,
            tools=names - DELEGATED,
        )
        request = RunRequest(
            conversation=job.conversation(system),
            model=job.model,
            tools=specs,
            run_tool=run_tool,
            web=job.web,
            max_rounds=job.limit if specs else 0,
            kind="chat" if desk.chat else job.kind,
        )
        segments = [""]
        task = asyncio.create_task(self._pump(job, request, queue, segments))
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            if not task.done():
                _ = task.cancel()
            _ = await asyncio.wait([task])
        if task.cancelled() or not task.result():
            return
        answer = next((text for text in reversed(segments) if text.strip()), "")
        if not answer:
            yield {"type": "error", "message": "모델 응답 없음"}
            return
        job.answer = answer
        job.completed = True

    async def _pump(
        self,
        job: _Job,
        request: RunRequest,
        queue: asyncio.Queue[Event | None],
        segments: list[str],
    ) -> bool:
        try:
            async for event in job.provider.stream(request):
                if isinstance(event, TextDelta):
                    segments[-1] += event.text
                    await queue.put({"type": "text", "text": event.text})
                elif isinstance(event, ToolRound):
                    segments.append("")
                elif isinstance(event, WebFound):
                    web.collect(job.found, event.sources)
                elif isinstance(event, Usage):
                    await queue.put(
                        {
                            "type": "usage",
                            "input": event.input_tokens,
                            "output": event.output_tokens,
                            "note": event.note,
                        }
                    )
                else:
                    await queue.put(
                        job.trace.step(event.summary, ok=event.ok, kind=event.kind)
                    )
        except Exception as error:  # noqa: BLE001  # 스트림 펌프 최상위 예외 포착
            await queue.put({"type": "error", "message": _message(error)})
            await queue.put(None)
            return False
        await queue.put(None)
        return True


@final
class Workspace:
    __slots__ = (
        "_active",
        "_asked",
        "_events",
        "_inbox",
        "_lock",
        "_main",
        "_service",
        "_tasks",
        "_threads",
        "_waiting",
        "_wake",
        "board",
        "chat",
        "company",
        "portfolio",
    )

    def __init__(self, service: AnalysisService, *, chat: bool = False) -> None:
        self._service = service
        self.chat = chat
        self._events: SimpleQueue[Event] = SimpleQueue()
        self._lock = Lock()
        self._main = Thread(MAIN)
        self._threads: dict[str, Thread] = {}
        self._tasks: dict[str, asyncio.Task[str]] = {}
        self._inbox: list[tuple[Thread, str]] = []
        self._asked: list[RunInput] = []
        self._waiting: dict[str, _Ask] = {}
        self._wake = False
        self._active = 0
        self.board = Board()
        self.company = service.company()
        self.portfolio = ""

    @property
    def busy(self) -> bool:
        return bool(self._active)

    def drain(self) -> None:
        with self._lock:
            while True:
                try:
                    event = self._events.get_nowait()
                except Empty:
                    return
                self.board.apply(event)

    async def ask(self, text: str, document: Document | None = None) -> None:
        self._asked.append(RunInput(text, document))
        self._advance()

    async def stop(self, thread: str = MAIN) -> None:
        task = self._tasks.get(thread)
        if task is not None:
            _ = task.cancel()

    async def use(
        self, caller: Thread, name: str, arguments: Mapping[str, object]
    ) -> ToolResult:
        try:
            if name == "ask_main":
                reply = await self._request(caller, text_arg(arguments, "message"))
                body = json.dumps({"reply": reply}, ensure_ascii=False)
                return ToolResult(body, ok=True, summary="조정자 응답 수신")
            if name == "read_thread":
                read = self._lent([dict(arguments)])[0]
                summary = " · ".join(("스레드 읽기", read.source.title, *read.names))
                found = {
                    "thread": read.source.id,
                    "sections": list(read.names),
                    "text": read.body,
                }
                body = json.dumps(found, ensure_ascii=False)
                return ToolResult(body, ok=True, summary=summary)
            lent = self._lent(arguments.get("attach"))
            if name == "start_thread":
                thread = self._start(arguments, lent)
                summary = f"스레드 시작 · {thread.title}"
            else:
                thread = self._follow(
                    text_arg(arguments, "thread"),
                    text_arg(arguments, "message"),
                    lent,
                )
                summary = f"스레드에 전달 · {thread.title}"
        except (ValueError, TypeError) as error:
            body = json.dumps({"error": str(error)}, ensure_ascii=False)
            return ToolResult(body, ok=False, summary=f"{name} · 실패: {error}")
        body = json.dumps({"thread": thread.id, "state": thread.state})
        return ToolResult(body, ok=True, summary=summary)

    async def _request(self, thread: Thread, text: str) -> str:
        if not text:
            raise ValueError("message 필수 입력")
        if thread.id in self._waiting:
            raise ValueError("조정자 응답 대기 중")
        ask = _Ask(text, asyncio.get_running_loop().create_future())
        self._waiting[thread.id] = ask
        step = {"type": "tool", "summary": "조정자에게 요청", "ok": True}
        self._emit(thread.id, {**step, "kind": "thread"})
        self._advance()
        try:
            return await ask.reply
        finally:
            _ = self._waiting.pop(thread.id, None)

    def _lent(self, raw: object) -> tuple[_Lent, ...]:
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise TypeError(ATTACH_SHAPE)
        out: list[_Lent] = []
        for item in cast("list[object]", raw):
            if not isinstance(item, dict):
                raise TypeError(ATTACH_SHAPE)
            entry = cast("dict[str, object]", item)
            key = str(entry.get("thread", ""))
            source = self._threads.get(key)
            if source is None or not source.turns:
                raise ValueError(f"첨부할 답변이 없는 스레드: {key}")
            names = _names(entry.get("sections"))
            out.append(_Lent(source, names, _excerpt(source.answer, names)))
        return tuple(out)

    def _hand(self, thread: Thread, lent: tuple[_Lent, ...]) -> str:
        for item in lent:
            thread.ids |= item.source.ids
            thread.web.update(item.source.web)
        return "\n\n".join(
            _tagged(
                "attached_report",
                item.source,
                item.body,
                sections=", ".join(item.names) or "전체",
            )
            for item in lent
        )

    def _refer(self, thread: Thread, lent: tuple[_Lent, ...]) -> None:
        for item in lent:
            label = " · ".join((item.source.title, *item.names))
            self._note(thread.id, f"참조 · {label}")

    def _note(self, thread: str, summary: str, *, ok: bool = True) -> None:
        step = {"type": "tool", "summary": summary, "ok": ok}
        self._emit(thread, {**step, "kind": "thread"})

    def _start(
        self, arguments: Mapping[str, object], lent: tuple[_Lent, ...]
    ) -> Thread:
        kind = text_arg(arguments, "kind")
        ticker = text_arg(arguments, "ticker")
        task = text_arg(arguments, "task")
        if kind not in KINDS:
            raise ValueError(f"kind는 {', '.join(KINDS)} 중 하나")
        if kind in {"stock", "etf"} and not ticker:
            raise ValueError("ticker 필수 입력")
        if kind == "research" and not task:
            raise ValueError("task 필수 입력")
        agent = _role(kind, text_arg(arguments, "agent"))
        name = text_arg(arguments, "title") or ticker or kind
        thread = Thread(f"t{len(self._threads) + 1}", name, kind, agent=agent)
        self._threads[thread.id] = thread
        self._emit(thread.id, {"type": "thread", "title": thread.title})
        notes = self._hand(thread, lent)
        flow = WORKFLOWS.get(kind)
        work = (
            self._first(thread, task, notes)
            if flow is None
            else self._workflow(thread, flow, ticker.upper(), task, notes)
        )
        self._begin(thread, task or thread.title, work)
        self._refer(thread, lent)
        return thread

    def _follow(self, key: str, text: str, lent: tuple[_Lent, ...]) -> Thread:
        thread = self._threads.get(key)
        if thread is None:
            raise ValueError(f"없는 스레드: {key}")
        if not text:
            raise ValueError("message 필수 입력")
        ask = self._waiting.pop(key, None)
        if ask is not None and not ask.reply.done():
            ask.reply.set_result(
                "\n\n".join(filter(None, (text, self._hand(thread, lent))))
            )
            self._refer(thread, lent)
            return thread
        if key in self._tasks:
            raise ValueError(f"{thread.title} 스레드 작업 진행 중")
        if not thread.documents:
            raise ValueError(f"{thread.title} 스레드 미시작 상태")
        notes = self._hand(thread, lent)
        self._begin(thread, text, self._run(thread, RunInput(text), notes))
        self._refer(thread, lent)
        return thread

    def _begin(self, thread: Thread, question: str, work: Work) -> None:
        thread.state = "working"
        self._emit(thread.id, {"type": "turn", "question": question})
        self._active += 1
        if thread.id != MAIN:
            for ask in self._waiting.values():
                if ask.sent:
                    ask.helpers.add(thread.id)
        task = asyncio.create_task(self._guard(thread, work))
        self._tasks[thread.id] = task
        task.add_done_callback(lambda done: self._settle(thread, done))

    def _spawn(self, thread: Thread, question: str, work: Work) -> asyncio.Task[str]:
        thread.state = "working"
        self._emit(thread.id, {"type": "turn", "question": question})
        self._active += 1
        task = asyncio.create_task(self._guard(thread, work))
        self._tasks[thread.id] = task
        task.add_done_callback(lambda done: self._close(thread, question, done))
        return task

    async def _guard(self, thread: Thread, work: Work) -> str:
        try:
            return await work
        except Exception as error:  # noqa: BLE001  # 스레드 작업 최상위 예외 포착
            failure = _message(error)
            self._emit(thread.id, {"type": "error", "message": failure})
            return failure

    def _ended(self, thread: Thread, task: asyncio.Task[str]) -> str:
        del self._tasks[thread.id]
        failure = STOPPED if task.cancelled() else task.result()
        if task.cancelled():
            thread.state = "stopped"
        else:
            thread.state = "failed" if failure else "done"
        self._emit(thread.id, {"type": "end", "state": thread.state})
        return failure

    def _settle(self, thread: Thread, task: asyncio.Task[str]) -> None:
        failure = self._ended(thread, task)
        if thread.id != MAIN:
            body = failure or _digest(thread.answer)
            report = _tagged("thread_report", thread, body, state=thread.state)
            self._inbox.append((thread, report))
            self._wake = self._wake or not task.cancelled()
        self._advance()
        self._active -= 1

    def _close(self, thread: Thread, question: str, task: asyncio.Task[str]) -> None:
        failure = self._ended(thread, task)
        self._active -= 1
        summary = f"{thread.title} · {question} · {STATE_TEXT[thread.state]}"
        if thread.state == "failed":
            summary = f"{summary}: {failure}"
        self._note(thread.parent, summary, ok=thread.state == "done")

    def _advance(self) -> None:
        if MAIN in self._tasks:
            return
        if self._asked:
            given = self._asked.pop(0)
            self._begin(self._main, given.request, self._coordinate(given))
        elif self._due() or (self._wake and not self._busy()):
            self._begin(self._main, "", self._coordinate(RunInput("")))

    def _busy(self) -> set[str]:
        return self._tasks.keys() - self._waiting.keys() - {MAIN}

    def _reporting(self) -> set[str]:
        return {thread.id for thread, _ in self._inbox}

    def _due(self) -> bool:
        reporting = self._reporting()
        return any(
            not ask.sent
            or bool(ask.helpers & reporting)
            or not ask.helpers & self._tasks.keys()
            for ask in self._waiting.values()
        )

    def _helped(self, ask: _Ask) -> bool:
        return bool(ask.helpers & (self._tasks.keys() | self._reporting()))

    async def _coordinate(self, given: RunInput) -> str:
        arrived, self._inbox, self._wake = self._inbox, [], False
        for thread, _ in arrived:
            self._main.ids |= thread.ids
            self._main.web.update(thread.web)
        asked = [
            (self._threads[key], ask)
            for key, ask in self._waiting.items()
            if not ask.sent
        ]
        for _, ask in asked:
            ask.sent = True
        notes = "\n\n".join(
            filter(
                None,
                (
                    *(text for _, text in arrived),
                    *(_tagged("thread_request", t, a.text) for t, a in asked),
                    self._status(),
                ),
            )
        )
        answered = False
        try:
            failure = await self._run(self._main, given, notes)
            answered = not failure
            return failure
        finally:
            if not answered:
                self._inbox[:0] = arrived
            for key, ask in list(self._waiting.items()):
                if ask.sent and not (answered and self._helped(ask)):
                    del self._waiting[key]
                    if not ask.reply.done():
                        ask.reply.set_result(UNANSWERED)

    def _status(self) -> str:
        if not self._threads:
            return ""
        rows = (_row(t, waiting=t.id in self._waiting) for t in self._threads.values())
        return "\n".join(("<threads>", *rows, "</threads>"))

    async def _first(self, thread: Thread, task: str, notes: str) -> str:
        agent = thread.agent
        request = task if agent is None else agents.research_request(agent, task)
        return await self._run(thread, RunInput(request), notes)

    async def _workflow(
        self, parent: Thread, flow: Workflow, ticker: str, task: str, notes: str
    ) -> str:
        if failure := await self._service.unready():
            raise ProviderError(failure)
        try:
            given = await self._service.prepare(
                flow.kind, ticker, self.portfolio, self.company
            )
        except Exception as error:
            raise ProviderError(f"{parent.title} 준비 실패: {error}") from error
        if given.document is not None:
            parent.documents.append(given.document)
            parent.ids |= given.document.ids
        self._emit(parent.id, self._service.opening())
        crew: list[asyncio.Task[str]] = []
        try:
            return await self._staged(parent, _Order(flow, given, task, notes), crew)
        finally:
            running = [job for job in crew if not job.done()]
            for job in running:
                _ = job.cancel()
            if running:
                _ = await asyncio.wait(running)

    async def _staged(
        self, parent: Thread, order: _Order, crew: list[asyncio.Task[str]]
    ) -> str:
        flow = order.flow
        names = ", ".join(seat.name for seat in flow.collect)
        self._note(parent.id, STAGE_STEP.format(stage=agents.COLLECT, names=names))
        seats = [
            (seat, self._child(parent, seat.agent, seat.name, agents.COLLECT))
            for seat in flow.collect
        ]
        jobs = [
            self._spawn(
                child,
                agents.label(seat.parts),
                self._run(child, order.collect(seat), order.notes),
            )
            for seat, child in seats
        ]
        crew.extend(jobs)
        outputs = [
            Output(seat.name, seat.parts, "" if failure else child.answer, failure)
            for (seat, child), failure in zip(seats, await _outcomes(jobs), strict=True)
        ]
        attached = "\n\n".join(
            _tagged("attached_report", child, output.answer)
            for (_, child), output in zip(seats, outputs, strict=True)
            if output.answer
        )
        if not attached:
            return NO_REPORTS
        transcript = await self._debate(parent, order, attached, crew)
        self._note(
            parent.id, STAGE_STEP.format(stage=agents.SUMMARY, names=SYNTH.title)
        )
        synth = self._child(parent, SYNTH, SYNTH.title, agents.SUMMARY)
        job = self._spawn(
            synth,
            SYNTH_QUESTION,
            self._run(synth, order.synth(), _joined(attached, transcript)),
        )
        crew.append(job)
        (failure,) = await _outcomes([job])
        if failure:
            return SYNTH_FAILED.format(failure=failure)
        final = Output(SYNTH.title, flow.synth, synth.answer)
        self._deliver(parent, order.task or parent.title, flow, outputs, final)
        return ""

    async def _debate(
        self,
        parent: Thread,
        order: _Order,
        attached: str,
        crew: list[asyncio.Task[str]],
    ) -> str:
        bull = self._child(parent, BULL, BULL.title, agents.DEBATE)
        bear = self._child(parent, BEAR, BEAR.title, agents.DEBATE)
        sides = ((bull, BULL, bear), (bear, BEAR, bull))
        said: list[str] = []
        for number in range(1, agents.DEBATE_ROUNDS + 1):
            stage = f"{agents.DEBATE} {ROUND_QUESTION.format(number=number)}"
            names = f"{BULL.title}, {BEAR.title}"
            self._note(parent.id, STAGE_STEP.format(stage=stage, names=names))
            heard = {
                side.id: attached if number == 1 else _said(other, number - 1)
                for side, _, other in sides
            }
            jobs = [
                self._spawn(
                    side,
                    ROUND_QUESTION.format(number=number),
                    self._run(side, order.round(agent, number), heard[side.id]),
                )
                for side, agent, _ in sides
            ]
            crew.extend(jobs)
            failures = await _outcomes(jobs)
            said.extend(
                _said(side, number)
                for (side, _, _), failure in zip(sides, failures, strict=True)
                if not failure
            )
            if any(failures):
                break
        return "\n\n".join(said)

    def _deliver(
        self,
        parent: Thread,
        question: str,
        flow: Workflow,
        outputs: Sequence[Output],
        final: Output,
    ) -> None:
        answer = agents.assemble(flow, outputs, final)
        for child in parent.team:
            parent.ids |= child.ids
            web.collect(parent.web, child.web.values())
            parent.received.update(child.received)
        parent.turns.append(Turn(question, answer))
        counts = (
            sum(child.calls for child in parent.team),
            sum(child.searches for child in parent.team),
        )
        self._emit(parent.id, {"type": "text", "text": answer})
        self._emit(
            parent.id,
            _done(answer, parent, parent.ids, parent.web, counts=counts, report=True),
        )

    def _child(self, parent: Thread, agent: Agent, title: str, stage: str) -> Thread:
        thread = Thread(
            f"{parent.id}-{len(parent.team) + 1}",
            title,
            parent.kind,
            ids=set(parent.ids),
            web=dict(parent.web),
            parent=parent.id,
            agent=agent,
        )
        parent.team.append(thread)
        event: Event = {"type": "thread", "title": title, "parent": parent.id}
        self._emit(thread.id, {**event, "stage": stage})
        return thread

    async def _run(self, thread: Thread, given: RunInput, notes: str = "") -> str:
        failure = ""
        async for event in self._service.turn(thread, given, self, notes=notes):
            self._emit(thread.id, event)
            if event["type"] == "error":
                failure = str(event["message"])
            elif event["type"] == "done":
                thread.calls += cast("int", event["data_calls"])
                thread.searches += cast("int", event["web_steps"])
            elif event["type"] == "usage" and thread.parent:
                self._emit(thread.parent, event)
        return failure

    def _emit(self, thread: str, event: Event) -> None:
        self._events.put({**event, "thread": thread})


async def _outcomes(jobs: Sequence[asyncio.Task[str]]) -> list[str]:
    _ = await asyncio.wait(jobs)
    return [STOPPED if job.cancelled() else job.result() for job in jobs]


def _specs(thread: Thread) -> tuple[ToolSpec, ...]:
    agent = thread.agent
    data = (
        SPECS
        if agent is None
        else tuple(spec for spec in SPECS if spec.name in agent.tools)
    )
    if thread.id == MAIN:
        return (*data, *THREAD_SPECS)
    return data if thread.parent else (*data, *ASK_SPECS)


def _role(kind: str, name: str) -> Agent | None:
    if not name:
        return None
    if kind != "research":
        raise ValueError("agent는 research에만 지정")
    agent = POOL.get(name)
    if agent is None:
        raise ValueError(f"agent는 {', '.join(POOL)} 중 하나")
    return agent


def _refused(name: str, reason: str) -> ToolResult:
    body = json.dumps({"error": reason}, ensure_ascii=False)
    return ToolResult(body, ok=False, summary=f"{name} · 실패: {reason}")


def _done(  # noqa: PLR0913  # 답변·스레드·인용 범위·출처·집계·보고서 여부로 완료 이벤트 조립
    answer: str,
    thread: Thread,
    ids: Collection[str],
    found: Mapping[str, WebSource],
    *,
    counts: tuple[int, int],
    report: bool,
) -> Event:
    return {
        "type": "done",
        "lines": cites.parse(answer, ids, found),
        "received": [
            {"cite": cite, "info": info} for cite, info in thread.received.items()
        ],
        "data_calls": counts[0],
        "web_steps": counts[1],
        "report": report,
        "sources": [
            {"url": source.url, "title": source.title}
            for source in cites.cited_sources(answer, found)
        ],
    }


def _row(thread: Thread, *, waiting: bool) -> str:
    state = "waiting" if waiting else thread.state
    pairs = (("id", thread.id), ("kind", thread.kind), ("state", state))
    attrs = " ".join(f"{name}={quoteattr(value)}" for name, value in pairs)
    return f"<thread {attrs}>{escape(thread.title)}</thread>"


def _tagged(tag: str, thread: Thread, body: str, **more: str) -> str:
    pairs = {"id": thread.id, "title": thread.title, **more}
    attrs = " ".join(f"{name}={quoteattr(value)}" for name, value in pairs.items())
    return f"<{tag} {attrs}>\n{body}\n</{tag}>"


def _said(thread: Thread, number: int) -> str:
    side, round_ = quoteattr(thread.title), quoteattr(str(number))
    return (
        f"<debate_round side={side} round={round_}>\n{thread.answer}\n</debate_round>"
    )


def _joined(*blocks: str) -> str:
    return "\n\n".join(block for block in blocks if block)


def _digest(answer: str) -> str:
    head = answer.strip().split("\n\n", 1)[0]
    names = [line[3:].strip() for line in answer.splitlines() if line.startswith("## ")]
    return f"{head}\n제목: {', '.join(names)}" if names else head


def _names(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError("sections는 ## 제목의 목록")
    return tuple(str(name) for name in cast("list[object]", raw))


def _excerpt(answer: str, names: tuple[str, ...]) -> str:
    if not names:
        return answer
    blocks: dict[str, list[str]] = {"": []}
    name = ""
    for line in answer.splitlines():
        if line.startswith("## "):
            name = line[3:].strip()
        blocks.setdefault(name, []).append(line)
    missing = [wanted for wanted in names if wanted not in blocks]
    if missing:
        found = ", ".join(filter(None, blocks))
        raise ValueError(f"없는 제목: {', '.join(missing)} · 있는 제목: {found}")
    return "\n".join(line for key in ("", *names) for line in blocks[key]).strip()


def _new_document(thread: Thread, given: RunInput) -> Document | None:
    document = given.document
    if document is None or thread.documents[-1].text == document.text:
        return None
    return document


def _message(error: BaseException) -> str:
    text = str(error).strip() or type(error).__name__
    if not isinstance(error, ProviderError):
        text = f"{type(error).__name__}: {text}"
    return text[:ERROR_LIMIT]
