"""대시보드 패널·지표 행·수익률 곡선·경제전망(SEP)·발표 일정 불변 데이터 모델."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

SourceId = Literal[
    "FRED",
    "ECOS",
    "ECB",
    "BIS",
    "OECD",
    "ALPACA",
    "YAHOO",
    "SEC",
    "OPENFIGI",
]
Freq = Literal["D", "W", "M", "Q"]
RowStatus = Literal["ok", "missing_key", "error"]
SourceState = Literal["ok", "partial", "missing_key", "error"]
Tone = Literal["good", "warn", "crit", "neutral"]
Area = Literal["US", "KR", "EA", "JP", "CN", "GL"]

AREA_LABEL: dict[Area, str] = {
    "US": "미국",
    "KR": "한국",
    "EA": "유로존",
    "JP": "일본",
    "CN": "중국",
    "GL": "글로벌",
}


class Frozen(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class Fmt(Frozen):
    pre: str = ""
    suf: str = ""
    dec: int = 2
    signed: bool = False


class IndicatorRow(Frozen):
    id: str
    label: str
    code: str
    source: SourceId
    area: Area
    copyright: str | None = None
    freq: Freq
    fmt: Fmt
    status: RowStatus
    message: str | None = None
    value: float | None = None
    prev: float | None = None
    date: str | None = None
    history: tuple[tuple[str, float], ...] = ()


class Panel(Frozen):
    id: str
    page: str
    title: str
    note: str
    rows: tuple[IndicatorRow, ...]


class SourceSummary(Frozen):
    id: SourceId
    name: str
    state: SourceState
    total: int
    ok: int
    message: str | None = None


class RegimeCell(Frozen):
    source: str
    label: str
    measure: str
    value: str
    tone: Tone
    tag: str
    rule: str


class CurveLine(Frozen):
    label: str
    date: str
    values: tuple[float | None, ...]


class Curve(Frozen):
    area: Area
    label: str
    tenors: tuple[str, ...]
    lines: tuple[CurveLine, ...]


class Spread(Frozen):
    points: tuple[tuple[str, float], ...]
    recessions: tuple[tuple[str, str], ...]


class CalendarEntry(Frozen):
    date: str
    name: str
    source: str


class PolicyChange(Frozen):
    area: Area
    bank: str
    date: str
    rate: float
    prev: float
    count_12m: int


class ProjectionCell(Frozen):
    value: float | None = None
    prev: float | None = None


class ProjectionRow(Frozen):
    label: str
    code: str
    current: float | None
    current_note: str
    current_dec: int
    years: tuple[ProjectionCell, ...]
    longer_run: ProjectionCell | None


class Projections(Frozen):
    published: str
    prior_published: str | None
    years: tuple[str, ...]
    rows: tuple[ProjectionRow, ...]
    message: str | None = None


class Dashboard(Frozen):
    generated_at: str
    elapsed_seconds: float
    demo: bool
    sources: tuple[SourceSummary, ...]
    regime: tuple[RegimeCell, ...]
    curves: tuple[Curve, ...]
    spread: Spread | None
    projections: Projections | None
    panels: tuple[Panel, ...]
    calendar: tuple[CalendarEntry, ...]
    policy_changes: tuple[PolicyChange, ...]
    fomc_dates: tuple[str, ...]
    recent_releases: tuple[CalendarEntry, ...] = ()
    calendar_message: str | None = None
    fomc_message: str | None = None
