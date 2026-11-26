from __future__ import annotations

from typing import Final

import pytest

from finagent.analysis import agents, prompt
from finagent.analysis.agents import Output

SYNTH_BLOCKS: Final = (
    "**결론**: 결론",
    "## 시나리오",
    "표",
    "## PART XIV. 포트폴리오 적합성",
    "적합성",
    "## PART XVI. 종합 분석 및 투자 매력도 평가",
    "매력도",
    "## PART V. 성과와 위험",
    "남의 PART",
)


@pytest.mark.parametrize(
    ("kind", "records"), [("macro", 16), ("stock", 62), ("etf", 53)]
)
def test_every_part_has_exactly_one_writer(kind: str, records: int) -> None:
    flow = agents.WORKFLOWS[kind]
    written = [part for seat in flow.collect for part in seat.parts]
    written.extend(flow.synth)

    assert sorted(written, key=agents.NUMERALS.index) == [
        part.numeral for part in flow.parts
    ]
    assert sum(part.count for part in flow.parts) == records


def test_reports_leave_out_agents_that_do_not_fit_them() -> None:
    macro = {seat.agent.id for seat in agents.MACRO_REPORT.collect}

    assert macro == {"macro", "technical", "sentiment"}
    assert {seat.name for seat in agents.STOCK_REPORT.collect} >= {
        "사업 · 업계·제품",
        "사업 · 경영진·해자",
    }
    assert agents.ETF_REPORT.synth == ("XIV", "XVI")
    assert agents.STOCK_REPORT.synth == ("XVI",)


def test_record_ids_follow_the_part_number() -> None:
    part = agents.STOCK_REPORT.part("XII")

    assert part.records == ("12.1", "12.2")
    assert part.line == "- PART XII. 경제 지표 분석: 기록 12.1, 12.2"


def test_the_guide_excerpt_keeps_the_head_and_only_the_chosen_parts() -> None:
    etf = agents.excerpt(prompt.text("etf_guide.txt"), ("IV", "XV"))
    macro = agents.excerpt(prompt.text("macro_guide.txt"), ("II",))

    assert etf.startswith("<etf_guide>\nETF 대상 16개 PART")
    assert etf.endswith("</etf_guide>")
    assert "## PART IV. 구성과 집중도" in etf
    assert "## PART XV. 대체 상품 비교" in etf
    assert "## PART I. 상품 구조" not in etf
    assert "기록 2.2" in macro
    assert "기록 1.1" not in macro


def test_the_stock_team_reads_the_whole_guide() -> None:
    flow = agents.STOCK_REPORT
    seat = flow.collect[0]

    request = agents.collect_request(
        flow, seat, "<analysis_guide>전체</analysis_guide>", ""
    )

    assert "<analysis_guide>전체</analysis_guide>" in request
    assert "- PART I. 재무 보고서 분석: 기록 1.1, 1.2, 1.3, 1.4, 1.5" in request
    assert "추가 요청:\n" not in request


def test_assembly_orders_parts_and_marks_gaps() -> None:
    flow = agents.ETF_REPORT
    outputs = [
        Output(
            seat.name,
            seat.parts,
            "\n\n".join(
                f"## {flow.part(part).name}\n\n{seat.name} {part}"
                for part in seat.parts
            )
            + "\n\n## PART I. 상품 구조 및 추종 전략\n\n남의 PART",
        )
        for seat in flow.collect
        if seat.agent is not agents.SENTIMENT
    ]
    outputs.append(Output("센티먼트", ("IX",), failure="시간 초과"))
    synth = Output(
        "종합",
        ("XIV", "XVI"),
        "\n\n".join(SYNTH_BLOCKS),
    )

    report = agents.assemble(flow, outputs, synth)

    heads = [line for line in report.splitlines() if line.startswith("## ")]
    assert heads == [*(f"## {part.name}" for part in flow.parts), "## 시나리오"]
    assert report.startswith("**결론**: 결론")
    gap = "## PART IX. 뉴스·공시 및 시장 심리\n\n자료 부족 · 센티먼트 에이전트 실패"
    assert f"{gap}: 시간 초과" in report
    assert "## PART I. 상품 구조 및 추종 전략\n\n펀더멘털 I" in report
    assert "남의 PART" not in report
    assert "## PART XIV. 포트폴리오 적합성\n\n적합성" in report
    assert "## PART XVI. 종합 분석 및 투자 매력도 평가\n\n매력도" in report


def test_assembly_notes_a_missing_part() -> None:
    flow = agents.MACRO_REPORT
    outputs = [Output(seat.name, seat.parts, "") for seat in flow.collect]

    report = agents.assemble(flow, outputs, Output("종합", (), "**결론**: x"))

    assert "## PART V. 시장 가격 분석\n\n자료 부족 · 기술 에이전트 출력 누락" in report


def test_debate_requests_name_the_side_its_focus_and_each_round() -> None:
    flow = agents.MACRO_REPORT

    first = agents.debate_request(flow, agents.BEAR, "금리 경로 위주")

    assert "약세 측을 맡은 에이전트다. 상대는 강세 측이다." in first
    assert flow.bear in first
    assert first.endswith("추가 요청:\n금리 경로 위주")
    assert agents.round_request(agents.BULL, 2).startswith("2라운드")
    assert "강세 측 최종 입장" in agents.round_request(agents.BULL, 3)


def test_seat_names_are_unique_within_each_workflow() -> None:
    for flow in agents.WORKFLOWS.values():
        names = [seat.name for seat in flow.collect]
        assert len(names) == len(set(names)), flow.kind


@pytest.mark.parametrize("name", ["synth_stock.txt", "synth_etf.txt"])
def test_synth_verdict_uses_market_relative_choices(name: str) -> None:
    verdict_lines = [
        line
        for line in prompt.text(name).splitlines()
        if line.startswith("- 최종 상태 판정은")
    ]

    assert len(verdict_lines) == 1
    verdict = verdict_lines[0]
    assert "'시장 지수 대비 우세'" in verdict
    assert "'시장 지수 대비 열세'" in verdict
    for legacy in (
        "검토 우선",
        "가격 조건부 관찰",
        "가설 확인 대기",
        "위험 우세",
        "판단 보류",
    ):
        assert legacy not in verdict


@pytest.mark.parametrize(
    ("coordinator", "asking", "team", "chat"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_system_prompt_bans_disclaimers_in_every_mode(
    coordinator: bool, asking: bool, team: bool, chat: bool
) -> None:
    generated = prompt.system_prompt(
        coordinator=coordinator, asking=asking, team=team, chat=chat
    )

    assert "면책 문구" in generated
    assert "방어적 표현" in generated
    assert "당부성 표현" in generated
