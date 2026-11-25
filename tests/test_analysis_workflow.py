from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Final, cast, override

import pytest

from finagent.analysis import agents, prompt
from finagent.analysis.providers.base import ProviderError, TextDelta
from finagent.analysis.providers.demo import DemoProvider
from finagent.analysis.service import RunInput
from tests.test_analysis_service import (
    Recorder,
    analysis,
    document,
    ended,
    marks,
    of_thread,
    of_type,
    until,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from finagent.analysis.providers.base import ProviderEvent, RunRequest
    from finagent.analysis.service import Brief
    from finagent.analysis.tools import ToolResult

type Event = dict[str, object]

MACRO_CALL: Final[dict[str, object]] = {"title": "매크로 분석", "kind": "macro"}
ASSIGNED: Final = re.compile(r"^- PART ([IVX]+) ", re.MULTILINE)
GUIDE: Final = re.compile(r"<macro_guide>(.*?)</macro_guide>", re.DOTALL)
ROLES: Final = {
    "bull": "debate",
    "bear": "debate",
    "synth": "synth",
    "page": "research",
}


def prepared(brief: Brief) -> RunInput:
    guides = {"macro": "macro_guide.txt", "etf": "etf_guide.txt"}
    name = guides.get(brief.kind)
    guide = prompt.text(name) if name else ""
    return RunInput(
        "", document("<macro_changes>문서</macro_changes>"), brief.kind, guide
    )


def role_of(request: RunRequest) -> str:
    report, _, role = request.kind.partition(":")
    return role or report


def assigned(request: RunRequest) -> list[str]:
    rest = request.conversation.request.partition("<assignment>")[2]
    return ASSIGNED.findall(rest.partition("</assignment>")[0])


class Team(Recorder):
    def __init__(
        self,
        call: dict[str, object],
        *,
        failing: tuple[str, ...] = (),
        barriers: dict[str, int] | None = None,
        hold: bool = False,
    ) -> None:
        super().__init__()
        self.call: dict[str, object] = call
        self.failing: tuple[str, ...] = failing
        self.barriers: dict[str, asyncio.Barrier] = {
            group: asyncio.Barrier(count) for group, count in (barriers or {}).items()
        }
        self.hold: asyncio.Event = asyncio.Event()
        if not hold:
            self.hold.set()
        self.live: dict[str, int] = {}
        self.results: list[ToolResult] = []

    @override
    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        self.seen.append(request)
        if request.kind == "main":
            if request.conversation.request == "맡겨":
                self.results.append(await request.run_tool("start_thread", self.call))
                yield TextDelta("맡겼다")
            else:
                yield TextDelta("보고 받음")
            return
        role = role_of(request)
        group = ROLES.get(role, "collect")
        self.live[group] = self.live.get(group, 0) + 1
        try:
            if (barrier := self.barriers.get(group)) is not None:
                _ = await barrier.wait()
            _ = await self.hold.wait()
        finally:
            self.live[group] -= 1
        if role in self.failing:
            msg = f"{role} 실패"
            raise ProviderError(msg)
        if role == "technical":
            _ = await request.run_tool("get_series", {"id": "us_unrate"})
        yield TextDelta(self.answer(role, request))

    @staticmethod
    def answer(role: str, request: RunRequest) -> str:
        number = len(request.conversation.turns) + 1
        if role in {"bull", "bear"}:
            return f"{role} {number} 발언 [[pv_1]]"
        parts = [
            f"## PART {part}\n\n| 기록 | 수치·내용 |\n|---|---|\n| {part} | {role} 값"
            + (" [[us_unrate]]" if role == "technical" else " [[pv_1]]")
            + " |"
            for part in assigned(request)
        ]
        if role == "synth":
            tail = "## 종합\n\n종합 본문\n\n## 시나리오\n\n시나리오 본문"
            return "\n\n".join(("**결론**: 종합 결론 [[pv_1]]", *parts, tail))
        return "\n\n".join((f"{role} 서두", *parts, "## 메모\n\n버릴 내용"))


def requests(provider: Recorder, role: str) -> list[RunRequest]:
    return [seen for seen in provider.seen if role_of(seen) == role]


def final(events: list[Event], thread: str = "t1") -> str:
    return "".join(str(e["text"]) for e in of_type(of_thread(events, thread), "text"))


async def run(provider: Recorder, *, fetched: bool = False) -> list[Event]:
    service = analysis(provider, prepared, fetched=fetched)
    ws = service.open()
    events: list[Event] = []
    await ws.ask("맡겨")
    await until(ws, events)
    return events


def test_the_macro_team_collects_debates_three_rounds_and_reports_once() -> None:
    provider = Team(MACRO_CALL, barriers={"collect": 3, "debate": 2})

    events = asyncio.run(run(provider, fetched=True))

    made = of_type(events, "thread")
    assert [(e["thread"], e.get("parent", ""), e.get("stage", "")) for e in made] == [
        ("t1", "", ""),
        ("t1-1", "t1", "수집"),
        ("t1-2", "t1", "수집"),
        ("t1-3", "t1", "수집"),
        ("t1-4", "t1", "토론"),
        ("t1-5", "t1", "토론"),
        ("t1-6", "t1", "종합"),
    ]
    assert [e["title"] for e in made[1:]] == [
        "매크로",
        "기술",
        "센티먼트",
        "강세",
        "약세",
        "종합",
    ]
    assert ended(events, "t1") == ["done"]
    assert ended(events, "t1-4") == ["done"] * agents.DEBATE_ROUNDS
    assert ended(events, "t1-5") == ["done"] * agents.DEBATE_ROUNDS

    collectors = {role: requests(provider, role)[0] for role in ("macro", "technical")}
    assert assigned(collectors["macro"]) == ["I", "II", "III", "IV"]
    assert assigned(collectors["technical"]) == ["V"]
    macro_guide = cast(
        "list[str]", GUIDE.findall(collectors["macro"].conversation.request)
    )[0]
    assert "## PART IV" in macro_guide
    assert "## PART V" not in macro_guide
    assert {spec.name for spec in collectors["macro"].tools} == set(agents.MACRO.tools)
    assert all(seen.web for seen in collectors.values())
    assert "근원" not in collectors["technical"].conversation.request_document

    bulls = requests(provider, "bull")
    assert len(bulls) == agents.DEBATE_ROUNDS
    assert all(seen.tools == () and not seen.web for seen in bulls)
    assert bulls[0].conversation.request_document.count("<attached_report") == 3
    second, third = (seen.conversation.request_document for seen in bulls[1:])
    assert second.startswith('<debate_round side="약세" round="1">\nbear 1 발언')
    assert third.startswith('<debate_round side="약세" round="2">\nbear 2 발언')
    assert [turn.answer for turn in bulls[2].conversation.turns] == [
        "bull 1 발언 [[pv_1]]",
        "bull 2 발언 [[pv_1]]",
    ]

    synth = requests(provider, "synth")[0]
    notes = synth.conversation.request_document
    assert synth.tools == ()
    assert notes.count("<attached_report") == 3
    assert notes.count("<debate_round") == 2 * agents.DEBATE_ROUNDS
    assert '<debate_round side="강세" round="3">\nbull 3 발언' in notes

    answer = final(events)
    assert answer.startswith("**결론**: 종합 결론")
    order = [
        answer.index(f"## PART {n}\n") for n in ("I", "II", "III", "IV", "V", "VI")
    ]
    assert order == sorted(order)
    assert (
        answer.index("## PART VI")
        < answer.index("## 종합")
        < answer.index("## 시나리오")
    )
    assert "## 메모" not in answer
    assert "서두" not in answer
    done = of_type(of_thread(events, "t1"), "done")[0]
    assert done["report"] is True
    assert ("us_unrate", True) in marks(done)
    assert all(known for _, known in marks(done))
    assert done["data_calls"] == 1

    woken = [seen for seen in provider.seen if seen.kind == "main"][-1]
    talk = woken.conversation.request_document
    assert talk.count("<thread_report") == 1
    assert '<thread_report id="t1" title="매크로 분석" state="done">' in talk
    assert "t1-1" not in talk


def test_the_report_announces_its_plan_with_its_turn_and_no_stage_notes() -> None:
    provider = Team(MACRO_CALL)

    events = asyncio.run(run(provider))

    own = of_thread(events, "t1")
    assert [e["type"] for e in own[:3]] == ["thread", "turn", "plan"]
    plan = {k: v for k, v in own[2].items() if k not in {"type", "thread"}}
    assert plan == {
        "collect": [
            {"name": "매크로", "parts": ["I", "II", "III", "IV"]},
            {"name": "기술", "parts": ["V"]},
            {"name": "센티먼트", "parts": ["VI"]},
        ],
        "sides": ["강세", "약세"],
        "rounds": agents.DEBATE_ROUNDS,
        "synth": {"name": "종합", "parts": []},
    }
    assert of_type(own, "tool") == []


def test_all_collectors_failing_is_reported_on_the_report_thread() -> None:
    provider = Team(MACRO_CALL, failing=("macro", "technical", "sentiment"))

    events = asyncio.run(run(provider))

    assert ended(events, "t1") == ["failed"]
    errors = [e["message"] for e in of_type(of_thread(events, "t1"), "error")]
    assert errors == ["수집 에이전트 전원 실패"]
    stages = {e.get("stage") for e in of_type(events, "thread")}
    assert stages == {None, "수집"}


def test_a_failed_collector_leaves_a_marked_gap_and_the_report_still_lands() -> None:
    provider = Team(MACRO_CALL, failing=("technical",))

    events = asyncio.run(run(provider))

    assert ended(events, "t1-2") == ["failed"]
    assert ended(events, "t1") == ["done"]
    answer = final(events)
    assert "## PART V\n\n자료 부족 · 기술 에이전트 실패: technical 실패" in answer
    assert (
        answer.index("## PART IV")
        < answer.index("## PART V\n")
        < answer.index("## PART VI")
    )
    notes = requests(provider, "synth")[0].conversation.request_document
    assert notes.count("<attached_report") == 2


def test_a_failed_debater_ends_the_debate_and_the_synthesis_gets_what_was_said() -> (
    None
):
    provider = Team(MACRO_CALL, failing=("bull",), barriers={"debate": 2})

    events = asyncio.run(run(provider))

    assert ended(events, "t1-4") == ["failed"]
    assert ended(events, "t1-5") == ["done"]
    assert ended(events, "t1") == ["done"]
    notes = requests(provider, "synth")[0].conversation.request_document
    assert notes.count("<debate_round") == 1
    assert '<debate_round side="약세" round="1">' in notes


def test_a_failed_synthesis_fails_the_report() -> None:
    provider = Team(MACRO_CALL, failing=("synth",))

    events = asyncio.run(run(provider))

    assert ended(events, "t1-6") == ["failed"]
    assert ended(events, "t1") == ["failed"]
    errors = [e["message"] for e in of_type(of_thread(events, "t1"), "error")]
    assert errors == ["종합 에이전트 실패: synth 실패"]
    assert final(events) == ""
    woken = [seen for seen in provider.seen if seen.kind == "main"][-1]
    assert "종합 에이전트 실패: synth 실패" in woken.conversation.request_document


def test_stopping_the_report_stops_every_agent_it_started() -> None:
    provider = Team(MACRO_CALL, hold=True)
    service = analysis(provider, prepared)

    async def scenario() -> list[Event]:
        ws = service.open()
        events: list[Event] = []
        await ws.ask("맡겨")
        await until(ws, events, lambda: provider.live.get("collect") == 3)
        await ws.stop("t1")
        await until(ws, events)
        return events

    events = asyncio.run(scenario())

    assert ended(events, "t1") == ["stopped"]
    assert [ended(events, f"t1-{n}") for n in (1, 2, 3)] == [["stopped"]] * 3
    assert [e["thread"] for e in of_type(events, "thread")] == [
        "t1",
        "t1-1",
        "t1-2",
        "t1-3",
    ]


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (
            {"title": "x", "kind": "macro", "agent": "macro"},
            "agent는 research에만 지정",
        ),
        ({"title": "x", "kind": "research", "task": "x", "agent": "nope"}, "agent는"),
    ],
)
def test_agent_roles_are_only_for_research(call: dict[str, object], error: str) -> None:
    provider = Team(call)

    _ = asyncio.run(run(provider))

    assert not provider.results[0].ok
    assert error in provider.results[0].text


def test_a_research_thread_takes_the_chosen_agents_role_and_tools() -> None:
    call: dict[str, object] = {
        "title": "금리",
        "kind": "research",
        "task": "금리 경로 조사",
        "agent": "macro",
    }
    provider = Team(call)

    events = asyncio.run(run(provider))

    assert provider.results[0].ok
    seen = requests(provider, "page")[0]
    names = {spec.name for spec in seen.tools}
    assert names == {*agents.MACRO.tools, "ask_main"}
    assert seen.conversation.request.startswith(agents.MACRO.role)
    assert seen.conversation.request.endswith("금리 경로 조사")
    assert seen.web
    assert "company_facts" not in seen.conversation.system
    assert ended(events, "t1") == ["done"]


@pytest.mark.parametrize(
    ("ask", "kind"),
    [
        ("매크로 분석 해줘", "macro"),
        ("AAPL 종목 분석", "stock"),
        ("QQQ ETF 분석", "etf"),
    ],
)
def test_the_demo_team_assembles_each_report_in_part_order(ask: str, kind: str) -> None:
    service = analysis(DemoProvider(delay=0), prepared, fetched=True)

    async def scenario() -> list[Event]:
        ws = service.open()
        events: list[Event] = []
        await ws.ask(ask)
        await until(ws, events)
        return events

    events = asyncio.run(scenario())

    assert ended(events, "t1") == ["done"]
    assert final(events) == prompt.text(f"demo_{kind}.txt")
    flow = agents.WORKFLOWS[kind]
    children = [e for e in of_type(events, "thread") if e.get("parent") == "t1"]
    assert len(children) == len(flow.collect) + 3
    reported = cast("str", of_type(of_thread(events, ""), "text")[-1]["text"])
    assert reported


@pytest.mark.parametrize(
    ("ask", "kind"),
    [
        ("매크로 분석 해줘", "macro"),
        ("AAPL 종목 분석", "stock"),
        ("QQQ ETF 분석", "etf"),
    ],
)
def test_each_report_starts_exactly_the_agents_its_plan_names(
    ask: str, kind: str
) -> None:
    service = analysis(DemoProvider(delay=0), prepared, fetched=True)

    async def scenario() -> list[Event]:
        ws = service.open()
        events: list[Event] = []
        await ws.ask(ask)
        await until(ws, events)
        return events

    events = asyncio.run(scenario())

    plan = of_type(of_thread(events, "t1"), "plan")[0]
    collect = cast("list[dict[str, object]]", plan["collect"])
    synth = cast("dict[str, object]", plan["synth"])
    slots = {("수집", str(seat["name"])) for seat in collect}
    slots |= {("토론", str(side)) for side in cast("list[object]", plan["sides"])}
    slots.add(("종합", str(synth["name"])))
    started = {
        (str(e["stage"]), str(e["title"]))
        for e in of_type(events, "thread")
        if e.get("parent") == "t1"
    }
    assert started == slots
    assert synth["parts"] == list(agents.WORKFLOWS[kind].synth)
    debaters = [
        str(e["thread"]) for e in of_type(events, "thread") if e.get("stage") == "토론"
    ]
    for key in debaters:
        assert len(of_type(of_thread(events, key), "turn")) == agents.DEBATE_ROUNDS
