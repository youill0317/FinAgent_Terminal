from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final

import pytest

from finagent import display
from finagent.analysis import agents
from finagent.analysis.providers.demo import DemoProvider
from finagent.display import Board, FlowView
from tests.test_analysis_service import analysis, until
from tests.test_analysis_workflow import MACRO_CALL, Team, prepared, run

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finagent.analysis.service import Brief, RunInput

type Event = dict[str, object]

WAITING: Final = [
    "토론 1라운드 · 대기",
    "토론 2라운드 · 대기",
    "토론 3라운드 · 대기",
    "종합 · 대기",
]


def board_of(events: Sequence[Event]) -> Board:
    board = Board()
    for event in events:
        board.apply(event)
    return board


def upto(events: Sequence[Event], thread: str, kind: str, nth: int = 1) -> list[Event]:
    seen = 0
    for index, event in enumerate(events):
        if event["thread"] == thread and event["type"] == kind:
            seen += 1
            if seen == nth:
                return list(events[: index + 1])
    raise AssertionError((thread, kind, nth))


def view(events: Sequence[Event], key: str = "t1") -> FlowView:
    found = display.flow_view(board_of(events), key)
    assert found is not None
    return found


def labels(flow: FlowView) -> list[str]:
    return [step.label for step in flow.steps]


def test_the_view_follows_the_macro_team_from_plan_to_report() -> None:
    events = asyncio.run(run(Team(MACRO_CALL)))

    planned = view(upto(events, "t1", "plan"))
    assert labels(planned) == ["준비 · 작업 중", "수집 · 대기", *WAITING]
    assert planned.end_label == "보고서 · 대기"
    assert planned.bar_text == "준비 · 입력 자료 준비 중 · 0/11"
    assert planned.short == "준비 · 0/11"
    assert all(a.thread == "" and a.turn == -1 for s in planned.steps for a in s.agents)

    collecting = view(upto(events, "t1-3", "turn"))
    assert labels(collecting) == ["준비 · 완료", "수집 · 작업 중 0/3", *WAITING]
    assert collecting.bar_text == "수집 · 에이전트 3개 작업 중 · 1/11"
    team = collecting.steps[1].agents
    assert [(a.thread, a.turn) for a in team] == [("t1-1", 0), ("t1-2", 0), ("t1-3", 0)]
    assert team[1].task == "PART V"

    debating = view(upto(events, "t1-5", "turn", 2))
    assert debating.bar_text == "토론 2라운드 · 강세, 약세 작업 중 · 6/11"
    second = debating.steps[3].agents
    assert [(a.name, a.thread, a.turn) for a in second] == [
        ("강세", "t1-4", 1),
        ("약세", "t1-5", 1),
    ]

    assert view(upto(events, "t1-6", "turn")).bar_text == "종합 · 결론 작성 중 · 10/11"

    final = view(events)
    assert (final.state, final.done, final.total, final.fraction) == ("done", 11, 11, 1)
    assert all(label.endswith("· 완료") for label in labels(final))
    assert final.end_label == "보고서 · 완료"
    assert final.activity == final.short == "완료"
    assert final.team == 6


def test_progress_never_goes_back_while_the_team_works() -> None:
    events = asyncio.run(run(Team(MACRO_CALL)))
    board = Board()
    last = 0

    for event in events:
        board.apply(event)
        flow = display.flow_view(board, "t1")
        if flow is not None and flow.working:
            assert last <= flow.done <= flow.total
            last = flow.done


def test_a_failed_collector_is_counted_and_the_run_still_finishes() -> None:
    events = asyncio.run(run(Team(MACRO_CALL, failing=("technical",))))

    flow = view(events)

    assert flow.steps[1].label == "수집 · 완료 · :red[실패 1]"
    assert (flow.state, flow.done, flow.total) == ("done", 11, 11)
    assert flow.fold_label("데이터 1회 · 웹 0회") == (
        "작업 과정 · 에이전트 6개 · 데이터 1회 · 웹 0회 · :red[실패 1]"
    )


def test_a_failed_debater_skips_the_rounds_it_cut() -> None:
    team = Team(MACRO_CALL, failing=("bull",), barriers={"debate": 2})
    events = asyncio.run(run(team))

    flow = view(events)

    assert labels(flow)[2:5] == [
        "토론 1라운드 · :red[실패]",
        "토론 2라운드 · 미실행",
        "토론 3라운드 · 미실행",
    ]
    assert (flow.state, flow.done, flow.total) == ("done", 7, 7)


def test_a_failed_synthesis_marks_the_last_step_and_the_report() -> None:
    events = asyncio.run(run(Team(MACRO_CALL, failing=("synth",))))
    board = board_of(events)

    flow = display.flow_view(board, "t1")

    assert flow is not None
    assert flow.state == "failed"
    assert flow.steps[-1].label == "종합 · :red[실패]"
    assert flow.end_label == "보고서 · :red[실패]"
    assert flow.activity == "실패 · 종합"
    assert ("종합 에이전트 실패: synth 실패", True) in board.threads["t1"].turns[
        0
    ].notices


def test_all_collectors_failing_skips_the_rest() -> None:
    failing = ("macro", "technical", "sentiment")
    flow = view(asyncio.run(run(Team(MACRO_CALL, failing=failing))))

    assert labels(flow)[1:] == [
        "수집 · :red[실패]",
        "토론 1라운드 · 미실행",
        "토론 2라운드 · 미실행",
        "토론 3라운드 · 미실행",
        "종합 · 미실행",
    ]
    assert flow.activity == "실패 · 수집"


def test_stopping_during_collection_shows_where_it_stopped() -> None:
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

    flow = view(asyncio.run(scenario()))

    assert labels(flow)[1:3] == ["수집 · 중지", "토론 1라운드 · 미실행"]
    assert flow.activity == "중지 · 수집"
    assert flow.end_label == "보고서 · 중지"
    assert flow.short == "중지"


def test_a_failed_preparation_fails_the_first_step() -> None:
    def broken(brief: Brief) -> RunInput:
        raise ValueError(brief.kind)

    provider = Team(MACRO_CALL)
    service = analysis(provider, broken)

    async def scenario() -> list[Event]:
        ws = service.open()
        events: list[Event] = []
        await ws.ask("맡겨")
        await until(ws, events)
        return events

    flow = view(asyncio.run(scenario()))

    assert flow.steps[0].label == "준비 · :red[실패]"
    assert flow.activity == "실패 · 준비"
    assert labels(flow)[-1] == "종합 · 미실행"


def test_a_follow_up_turn_does_not_revive_the_finished_flow() -> None:
    events = asyncio.run(run(Team(MACRO_CALL)))
    board = board_of(events)
    board.apply({"type": "turn", "thread": "t1", "question": "추가"})

    flow = display.flow_view(board, "t1")

    assert board.threads["t1"].working
    assert flow is not None
    assert flow.state == "done"
    assert not flow.working


def test_only_report_threads_have_a_flow() -> None:
    call: dict[str, object] = {"title": "금리", "kind": "research", "task": "금리"}
    board = board_of(asyncio.run(run(Team(call))))

    assert display.flow_view(board, "t1") is None
    assert display.flow_view(board, "t9") is None


def test_the_plan_is_read_and_preparation_is_marked_done_on_start() -> None:
    events = asyncio.run(run(Team(MACRO_CALL)))

    before = board_of(upto(events, "t1", "plan")).threads["t1"].turns[0]
    after = board_of(upto(events, "t1", "start")).threads["t1"].turns[0]

    assert not before.ready
    assert after.ready
    plan = before.plan
    assert plan is not None
    flow = agents.MACRO_REPORT
    assert [seat.name for seat in plan.collect] == [s.name for s in flow.collect]
    assert [seat.parts for seat in plan.collect] == [s.parts for s in flow.collect]
    assert plan.sides == (agents.BULL.title, agents.BEAR.title)
    assert plan.rounds == agents.DEBATE_ROUNDS


def test_a_partly_started_stage_counts_as_working() -> None:
    events = asyncio.run(run(Team(MACRO_CALL)))

    flow = view(upto(events, "t1-1", "turn"))

    assert flow.steps[1].label == "수집 · 작업 중 0/3"


@pytest.mark.parametrize(
    ("ask", "units", "seats", "task"),
    [("AAPL 종목 분석", 14, 6, "PART XVI"), ("QQQ ETF 분석", 13, 5, "PART XIII")],
)
def test_stock_and_etf_flows_count_their_own_teams(
    ask: str, units: int, seats: int, task: str
) -> None:
    service = analysis(DemoProvider(delay=0), prepared, fetched=True)

    async def scenario() -> list[Event]:
        ws = service.open()
        events: list[Event] = []
        await ws.ask(ask)
        await until(ws, events)
        return events

    flow = view(asyncio.run(scenario()))

    assert flow.total == units
    assert len(flow.steps[1].agents) == seats
    assert flow.steps[-1].agents[0].task == task
