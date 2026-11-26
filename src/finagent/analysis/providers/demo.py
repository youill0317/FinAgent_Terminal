"""외부 LLM 호출 없이 도구 실행 및 예시 스트림을 재현하는 데모 공급자."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Final, cast

from finagent.analysis import agents, prompt
from finagent.analysis.providers.base import (
    ProviderStatus,
    TextDelta,
    Usage,
    WebFound,
    WebStep,
)
from finagent.analysis.web import WebSource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from finagent.analysis.providers.base import (
        ProviderEvent,
        ProviderId,
        RunRequest,
    )

CHUNK: Final = 14
REPORT_CHUNK: Final = 120
DELAY_SECONDS: Final = 0.01
STEP_PAUSES: Final = 40
DATA_STEPS: Final = 12
WEB_ROUNDS: Final = 10

TITLE_LIMIT: Final = 20
TICKER: Final = re.compile(r"[A-Z][A-Z0-9.]{1,9}")
REPORTED: Final = re.compile(r'<thread_report id="[^"]*" title="([^"]*)"')
STARTED: Final = "{titles} 스레드 작업 위임 및 완료 시 보고"
ARRIVED: Final = "**결론**: {titles} 스레드 보고 완료 (전문 스레드 참조)"

INTRO: Final = "데모 공급자 예시 답변 (모델 호출 없는 도구 결과 기반)"
CHAT: Final = "데모 페이지 챗봇 예시 답변 (모델 호출 없는 화면 데이터 기반)"
INDICATOR: Final = re.compile(r'<indicator id="([^"]+)"')
SUMMARY: Final = "**결론**: 도구 기반 장기 이력 수신 완료"
ASSIGNED: Final = re.compile(r"<assignment>\n(.*?)\n</assignment>", re.DOTALL)
ASSIGNED_PART: Final = re.compile(r"^- PART ([IVX]+)\. ", re.MULTILINE)
SIDES: Final = {agents.BULL.id: agents.BULL.title, agents.BEAR.id: agents.BEAR.title}
DEBATE: Final = "**{side} {number}라운드**: 데모 {side} 논거 (모델 호출 없는 형식 예시)"
DEMO_CALL: Final[tuple[str, dict[str, object]]] = ("get_series", {"id": "us_unrate"})
FALLBACK_CALL: Final[tuple[str, dict[str, object]]] = (
    "read_dashboard",
    {"page": "overview"},
)


class DemoProvider:
    id: ProviderId = "demo"
    title: str = "데모 (모델 없음)"
    default_model: str = "demo"

    def __init__(self, delay: float = DELAY_SECONDS) -> None:
        self._delay: float = delay

    async def status(self) -> ProviderStatus:
        return ProviderStatus("ready", "예시 답변 생성 데모 공급자")

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        if request.kind == "chat":
            events = self._chat(request)
        elif request.kind == "main":
            events = self._main(request)
        elif ":" in request.kind:
            events = self._agent(request)
        else:
            events = self._page(request)
        async for event in events:
            yield event

    async def _main(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        ask = request.conversation.request
        calls = _delegations(ask)
        if ask and not calls:
            async for event in self._page(request):
                yield event
            return
        for arguments in calls:
            _ = await request.run_tool("start_thread", arguments)
        titles = [str(call["title"]) for call in calls] or REPORTED.findall(
            request.conversation.request_document
        )
        text = (STARTED if calls else ARRIVED).format(titles=", ".join(titles))
        async for event in self._type(text, CHUNK):
            yield event

    async def _chat(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        talk = request.conversation
        pages = (talk.request_document, *(turn.document for turn in talk.turns[::-1]))
        found = (INDICATOR.search(page) for page in (*pages, *talk.documents))
        cite = next((match.group(1) for match in found if match), "")
        text = f"{CHAT}\n\n화면 첫 항목 확인 [[{cite}]]" if cite else CHAT
        async for event in self._type(text, CHUNK):
            yield event

    async def _agent(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        names = {spec.name for spec in request.tools}
        call = DEMO_CALL if DEMO_CALL[0] in names else FALLBACK_CALL
        for _ in range(DATA_STEPS if names else 0):
            await self._pause()
            _ = await request.run_tool(*call)
        if request.web:
            sources = _sample_sources()
            yield WebFound(sources)
            for round_ in range(1, WEB_ROUNDS + 1):
                await self._pause()
                yield WebStep(f"웹 검색: 데모 이벤트 {round_}")
                await self._pause()
                yield WebStep(f"원문 읽기: {sources[0].url}")
        async for event in self._type(_agent_text(request), REPORT_CHUNK):
            yield event

    async def _pause(self) -> None:
        if self._delay:
            await asyncio.sleep(self._delay * STEP_PAUSES)

    async def _type(self, text: str, size: int) -> AsyncIterator[ProviderEvent]:
        for start in range(0, len(text), size):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield TextDelta(text[start : start + size])
        yield Usage(note="데모 · 모델 호출 없음")

    async def _page(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        result = await request.run_tool(
            "get_series", {"id": "us_unrate", "start": "2016-01"}
        )
        long_run = ""
        if result.ok:
            body = cast("dict[str, object]", json.loads(result.text))
            points = cast("list[list[float]]", body["points"])
            values = [float(point[1]) for point in points]
            low, high = _num(min(values)), _num(max(values))
            since = points[0][0]
            long_run = f"도구 조회 실업률 범위 ({since} 이후): {low}%~{high}%"
            long_run += " [[us_unrate]]"
        text = f"{INTRO}\n\n{SUMMARY}\n\n{long_run}"
        async for event in self._type(text, CHUNK):
            yield event


def _delegations(ask: str) -> list[dict[str, object]]:
    if "분석" not in ask and "조사" not in ask:
        return []
    calls: list[dict[str, object]] = []
    if "매크로" in ask:
        calls.append({"title": "매크로 분석", "kind": "macro"})
    kind, label = ("etf", "ETF") if "ETF" in ask else ("stock", "종목")
    calls.extend(
        {"title": f"{ticker} {label} 분석", "kind": kind, "ticker": ticker}
        for ticker in dict.fromkeys(
            cast("list[str]", TICKER.findall(ask.replace("ETF", "")))
        )
    )
    if not calls:
        calls.append({"title": ask[:TITLE_LIMIT], "kind": "research", "task": ask})
    return calls


def _agent_text(request: RunRequest) -> str:
    report, _, role = request.kind.partition(":")
    talk = request.conversation
    if role in SIDES:
        return DEBATE.format(side=SIDES[role], number=len(talk.turns) + 1)
    head, sections = agents.split(prompt.text(f"demo_{report}.txt"))
    found = ASSIGNED.search(talk.request)
    listed = cast("list[str]", ASSIGNED_PART.findall(found.group(1))) if found else []
    wanted = set(listed)
    synth = role == agents.SYNTH.id
    kept = [
        section.body
        for section in sections
        if (part := agents.numeral(section.title)) in wanted or (synth and not part)
    ]
    return "\n\n".join((head, *kept) if synth else kept)


def _sample_sources() -> tuple[WebSource, ...]:
    lines = prompt.text("demo_web_sources.txt").splitlines()
    return tuple(WebSource(*line.split("\t")) for line in lines if line.strip())


def _num(value: float) -> str:
    return f"{value:,.1f}" if abs(value) < 1000 else f"{value:,.0f}"  # noqa: PLR2004  # 소수점 표기 전환 자릿수(1,000) 기준
