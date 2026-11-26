"""분석 에이전트 역할 정의, 매크로·종목·ETF 보고서 PART 배정표 및 보고서 조립."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from finagent.analysis import prompt

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

DEBATE_ROUNDS: Final = 3
COLLECT: Final = "수집"
DEBATE: Final = "토론"
SUMMARY: Final = "종합"
NUMERALS: Final = (
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
)
COMMON_TOOLS: Final = ("read_dashboard", "read_portfolio", "source_guide")
PART_TITLE: Final = re.compile(r"PART\s+([IVX]+)\b", re.IGNORECASE)
GUIDE_TAG: Final = re.compile(r"<([a-z_]+)>\s")
EXTRA: Final = "추가 요청:\n{task}"


@dataclass(frozen=True, slots=True)
class Agent:
    id: str
    title: str
    tools: tuple[str, ...] = ()
    web: bool = False

    @property
    def role(self) -> str:
        return prompt.text(f"agent_{self.id}.txt")


FUNDAMENTAL: Final = Agent(
    "fundamental",
    "펀더멘털",
    (
        *COMMON_TOOLS,
        "company_facts",
        "company_filings",
        "read_filing",
        "company_profile",
        "etf_lookup",
        "etf_holdings",
        "get_prices",
    ),
    web=True,
)
BUSINESS: Final = Agent(
    "business",
    "사업",
    (
        *COMMON_TOOLS,
        "company_profile",
        "company_facts",
        "company_filings",
        "read_filing",
        "etf_holdings",
        "etf_lookup",
        "get_prices",
    ),
    web=True,
)
TECHNICAL: Final = Agent(
    "technical",
    "기술",
    (
        *COMMON_TOOLS,
        "get_prices",
        "get_series",
        "search_series",
        "etf_lookup",
        "etf_holdings",
        "company_profile",
    ),
    web=True,
)
SENTIMENT: Final = Agent(
    "sentiment",
    "센티먼트",
    (
        *COMMON_TOOLS,
        "get_prices",
        "get_series",
        "search_series",
        "company_profile",
        "etf_holdings",
        "etf_lookup",
    ),
    web=True,
)
MACRO: Final = Agent(
    "macro",
    "매크로",
    (*COMMON_TOOLS, "search_series", "get_series", "get_prices"),
    web=True,
)
BULL: Final = Agent("bull", "강세")
BEAR: Final = Agent("bear", "약세")
SYNTH: Final = Agent("synth", "종합")
POOL: Final = {
    agent.id: agent for agent in (FUNDAMENTAL, BUSINESS, TECHNICAL, SENTIMENT, MACRO)
}


@dataclass(frozen=True, slots=True)
class Part:
    numeral: str
    title: str
    count: int

    @property
    def records(self) -> tuple[str, ...]:
        number = NUMERALS.index(self.numeral) + 1
        return tuple(f"{number}.{n}" for n in range(1, self.count + 1))

    @property
    def name(self) -> str:
        return f"PART {self.numeral}. {self.title}"

    @property
    def line(self) -> str:
        return f"- {self.name}: 기록 {', '.join(self.records)}"


@dataclass(frozen=True, slots=True)
class Seat:
    agent: Agent
    parts: tuple[str, ...]
    title: str = ""

    @property
    def name(self) -> str:
        return self.title or self.agent.title


@dataclass(frozen=True, slots=True)
class Workflow:
    kind: str
    title: str
    parts: tuple[Part, ...]
    collect: tuple[Seat, ...]
    synth: tuple[str, ...]
    bull: str
    bear: str
    whole_guide: bool = False

    def part(self, numeral: str) -> Part:
        return next(part for part in self.parts if part.numeral == numeral)


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    body: str


@dataclass(frozen=True, slots=True)
class Output:
    name: str
    parts: tuple[str, ...]
    answer: str = ""
    failure: str = ""


MACRO_REPORT: Final = Workflow(
    "macro",
    "매크로 분석",
    parts=(
        Part("I", "경제 지표 분석", 2),
        Part("II", "정책 환경 분석", 2),
        Part("III", "글로벌 경제 환경 분석", 2),
        Part("IV", "경제 주기 분석", 2),
        Part("V", "시장 가격 분석", 4),
        Part("VI", "뉴스·발표 및 시장 심리", 4),
    ),
    collect=(
        Seat(MACRO, ("I", "II", "III", "IV")),
        Seat(TECHNICAL, ("V",)),
        Seat(SENTIMENT, ("VI",)),
    ),
    synth=(),
    bull="리스크온: 확장·회복 신호, 완화 정책, 가격 강세",
    bear="리스크오프: 둔화·침체 신호, 긴축, 신용 스프레드 확대, 심리 과열",
)
STOCK_REPORT: Final = Workflow(
    "stock",
    "종목 분석",
    parts=(
        Part("I", "재무 보고서 분석", 5),
        Part("II", "업계 상태 분석", 5),
        Part("III", "경영진 및 기업 거버넌스", 5),
        Part("IV", "경제적 해자 분석", 6),
        Part("V", "제품 및 서비스 분석", 6),
        Part("VI", "차트 패턴", 4),
        Part("VII", "기술적 지표 및 오실레이터", 3),
        Part("VIII", "거래량 분석", 4),
        Part("IX", "뉴스 및 공식 발표 영향", 5),
        Part("X", "소셜 미디어 및 포럼 센티먼트", 3),
        Part("XI", "시장 센티먼트 종합", 2),
        Part("XII", "경제 지표 분석", 2),
        Part("XIII", "정책 환경 분석", 3),
        Part("XIV", "글로벌 경제 환경 분석", 2),
        Part("XV", "경제 주기 분석", 2),
        Part("XVI", "종합 분석 및 투자 매력도 평가", 5),
    ),
    collect=(
        Seat(FUNDAMENTAL, ("I",)),
        Seat(BUSINESS, ("II", "V"), "사업 · 업계·제품"),
        Seat(BUSINESS, ("III", "IV"), "사업 · 경영진·해자"),
        Seat(TECHNICAL, ("VI", "VII", "VIII")),
        Seat(SENTIMENT, ("IX", "X", "XI")),
        Seat(MACRO, ("XII", "XIII", "XIV", "XV")),
    ),
    synth=("XVI",),
    bull="사업의 질, 해자, 성장성, 가격 매력",
    bear="재무 위험, 밸류에이션 부담, 경쟁·경영 리스크",
    whole_guide=True,
)
ETF_REPORT: Final = Workflow(
    "etf",
    "ETF 분석",
    parts=(
        Part("I", "상품 구조 및 추종 전략", 5),
        Part("II", "비용과 운용 규모", 3),
        Part("III", "유동성과 추종 품질", 3),
        Part("IV", "구성과 집중도", 4),
        Part("V", "성과와 위험", 4),
        Part("VI", "차트 패턴", 4),
        Part("VII", "기술적 지표 및 오실레이터", 3),
        Part("VIII", "거래량 분석", 4),
        Part("IX", "뉴스·공시 및 시장 심리", 5),
        Part("X", "경제 지표 분석", 2),
        Part("XI", "정책 환경 분석", 2),
        Part("XII", "글로벌 경제 환경 분석", 2),
        Part("XIII", "경제 주기 분석", 2),
        Part("XIV", "포트폴리오 적합성", 3),
        Part("XV", "대체 상품 비교", 2),
        Part("XVI", "종합 분석 및 투자 매력도 평가", 5),
    ),
    collect=(
        Seat(FUNDAMENTAL, ("I", "II", "III")),
        Seat(BUSINESS, ("IV", "XV")),
        Seat(TECHNICAL, ("V", "VI", "VII", "VIII")),
        Seat(SENTIMENT, ("IX",)),
        Seat(MACRO, ("X", "XI", "XII", "XIII")),
    ),
    synth=("XIV", "XVI"),
    bull="비용·규모 우위, 구성의 질, 성과, 거시 순풍",
    bear=(
        "구조 리스크(레버리지 일일 재조정, 선물 롤오버, ETN 발행사 신용), "
        "집중도, 추종 오차, 거시 역풍"
    ),
)
WORKFLOWS: Final = {
    flow.kind: flow for flow in (MACRO_REPORT, STOCK_REPORT, ETF_REPORT)
}


def split(text: str) -> tuple[str, list[Section]]:
    blocks: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        if line.startswith("## "):
            blocks.append((line[3:].strip(), [line]))
        else:
            blocks[-1][1].append(line)
    (_, head), *rest = blocks
    sections = [Section(title, "\n".join(lines).strip()) for title, lines in rest]
    return "\n".join(head).strip(), sections


def numeral(title: str) -> str:
    found = PART_TITLE.match(title)
    value = found.group(1).upper() if found else ""
    return value if value in NUMERALS else ""


def excerpt(guide: str, numerals: Collection[str]) -> str:
    text = guide.strip()
    opened = GUIDE_TAG.match(text)
    tag = opened.group(1) if opened else ""
    if tag:
        text = text.removeprefix(f"<{tag}>").removesuffix(f"</{tag}>").strip()
    head, sections = split(text)
    kept = [section.body for section in sections if numeral(section.title) in numerals]
    body = "\n\n".join(filter(None, (head, *kept)))
    return f"<{tag}>\n{body}\n</{tag}>" if tag else body


def assignment(parts: Sequence[Part]) -> str:
    return "\n".join(("<assignment>", *(part.line for part in parts), "</assignment>"))


def label(parts: Sequence[str]) -> str:
    return f"PART {', '.join(parts)} 작성"


def collect_request(flow: Workflow, seat: Seat, guide: str, task: str) -> str:
    head = prompt.text("agent_collect.txt").format(report=flow.title, agent=seat.name)
    return _joined(
        head,
        seat.agent.role,
        assignment([flow.part(name) for name in seat.parts]),
        prompt.text(f"report_{flow.kind}.txt"),
        _guide(flow, guide, seat.parts),
        _extra(task),
    )


def debate_request(flow: Workflow, side: Agent, task: str) -> str:
    bull = side.id == BULL.id
    head = prompt.text("agent_debate.txt").format(
        report=flow.title,
        side=side.title,
        other=(BEAR if bull else BULL).title,
        focus=flow.bull if bull else flow.bear,
        rounds=DEBATE_ROUNDS,
    )
    return _joined(head, round_request(side, 1), _extra(task))


def round_request(side: Agent, number: int) -> str:
    rounds = prompt.text("debate_rounds.txt").splitlines()
    return rounds[number - 1].format(side=side.title)


def synth_request(flow: Workflow, guide: str, task: str) -> str:
    head = prompt.text("agent_synth.txt").format(
        report=flow.title, rounds=DEBATE_ROUNDS
    )
    parts = [flow.part(name) for name in flow.synth]
    return _joined(
        head,
        prompt.text(f"synth_{flow.kind}.txt"),
        assignment(parts) if parts else "",
        prompt.text(f"report_{flow.kind}.txt"),
        _guide(flow, guide, flow.synth),
        _extra(task),
    )


def research_request(agent: Agent, task: str) -> str:
    return _joined(agent.role, task)


def assemble(flow: Workflow, outputs: Sequence[Output], synth: Output) -> str:
    owners: dict[str, Output] = {}
    written: dict[str, str] = {}
    for output in (*outputs, synth):
        owners.update(dict.fromkeys(output.parts, output))
        for section in split(output.answer)[1]:
            found = numeral(section.title)
            if found in output.parts:
                _ = written.setdefault(found, section.body)
    head, sections = split(synth.answer)
    blocks = [head]
    blocks.extend(
        written.get(part.numeral) or _gap(part, owners[part.numeral])
        for part in flow.parts
    )
    blocks.extend(section.body for section in sections if not numeral(section.title))
    return "\n\n".join(block for block in blocks if block)


def _gap(part: Part, output: Output) -> str:
    reason = (
        f"{output.name} 에이전트 실패: {output.failure}"
        if output.failure
        else f"{output.name} 에이전트 출력 누락"
    )
    return f"## {part.name}\n\n자료 부족 · {reason}"


def _guide(flow: Workflow, guide: str, parts: Collection[str]) -> str:
    if not guide or not parts:
        return ""
    return guide if flow.whole_guide else excerpt(guide, parts)


def _extra(task: str) -> str:
    return EXTRA.format(task=task) if task else ""


def _joined(*blocks: str) -> str:
    return "\n\n".join(block.strip() for block in blocks if block.strip())
