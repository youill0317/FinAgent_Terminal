from __future__ import annotations

from typing import Final

import pytest

from finagent.analysis import agents, prompt
from finagent.analysis.agents import Output

SYNTH_BLOCKS: Final = (
    "**결론**: 결론",
    "## 시나리오",
    "표",
    "## PART XIII",
    "적합성",
    "## PART V",
    "남의 PART",
)


@pytest.mark.parametrize(
    ("kind", "records"), [("macro", 17), ("stock", 62), ("etf", 49)]
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
    assert agents.ETF_REPORT.synth == ("XIII",)
    assert agents.STOCK_REPORT.synth == ("XVI",)


def test_record_ids_follow_the_part_number() -> None:
    part = agents.STOCK_REPORT.part("XII")

    assert part.records == ("12.1", "12.2")
    assert part.line == "- PART XII 경제 지표: 기록 12.1, 12.2"


def test_the_guide_excerpt_keeps_the_head_and_only_the_chosen_parts() -> None:
    etf = agents.excerpt(prompt.text("etf_guide.txt"), ("IV", "XIV"))
    macro = agents.excerpt(prompt.text("macro_guide.txt"), ("II",))

    assert etf.startswith("<etf_guide>\nETF 대상 14개 PART")
    assert etf.endswith("</etf_guide>")
    assert "## PART IV 구성과 집중도" in etf
    assert "## PART XIV 대체 상품" in etf
    assert "## PART I 상품 구조" not in etf
    assert "기록 2.3" in macro
    assert "기록 1.1" not in macro


def test_the_stock_team_reads_the_whole_guide() -> None:
    flow = agents.STOCK_REPORT
    seat = flow.collect[0]

    request = agents.collect_request(
        flow, seat, "<analysis_guide>전체</analysis_guide>", ""
    )

    assert "<analysis_guide>전체</analysis_guide>" in request
    assert "- PART I 재무 보고서: 기록 1.1, 1.2, 1.3, 1.4, 1.5" in request
    assert "추가 요청" not in request


def test_assembly_orders_parts_and_marks_gaps() -> None:
    flow = agents.ETF_REPORT
    outputs = [
        Output(
            seat.name,
            seat.parts,
            "\n\n".join(f"## PART {part}\n\n{seat.name} {part}" for part in seat.parts)
            + "\n\n## PART I\n\n남의 PART",
        )
        for seat in flow.collect
        if seat.agent is not agents.SENTIMENT
    ]
    outputs.append(Output("센티먼트", ("VIII",), failure="시간 초과"))
    synth = Output(
        "종합",
        ("XIII",),
        "\n\n".join(SYNTH_BLOCKS),
    )

    report = agents.assemble(flow, outputs, synth)

    heads = [line for line in report.splitlines() if line.startswith("## ")]
    assert heads == [*(f"## PART {part.numeral}" for part in flow.parts), "## 시나리오"]
    assert report.startswith("**결론**: 결론")
    assert "## PART VIII\n\n자료 부족 · 센티먼트 에이전트 실패: 시간 초과" in report
    assert "## PART I\n\n펀더멘털 I" in report
    assert "남의 PART" not in report
    assert "## PART XIII\n\n적합성" in report


def test_assembly_notes_a_missing_part() -> None:
    flow = agents.MACRO_REPORT
    outputs = [Output(seat.name, seat.parts, "") for seat in flow.collect]

    report = agents.assemble(flow, outputs, Output("종합", (), "**결론**: x"))

    assert "## PART V\n\n자료 부족 · 기술 에이전트 출력 누락" in report


def test_debate_requests_name_the_side_its_focus_and_each_round() -> None:
    flow = agents.MACRO_REPORT

    first = agents.debate_request(flow, agents.BEAR, "금리 경로 위주")

    assert "약세 측을 맡은 에이전트다. 상대는 강세 측." in first
    assert flow.bear in first
    assert first.endswith("추가 요청:\n금리 경로 위주")
    assert agents.round_request(agents.BULL, 2).startswith("2라운드")
    assert "강세 측 최종 입장" in agents.round_request(agents.BULL, 3)
