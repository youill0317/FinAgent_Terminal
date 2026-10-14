from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

SourceId = Literal["FRED", "TREASURY", "NYFED", "ECOS", "ECB", "BIS", "OECD"]
Freq = Literal["D", "W", "M", "Q"]
RowStatus = Literal["ok", "missing_key", "error"]
SourceState = Literal["ok", "partial", "missing_key", "error"]
Tone = Literal["good", "warn", "crit", "neutral"]


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
    key: str
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
    label: str
    value: str
    unit: str
    tone: Tone
    tag: str
    rule: str


class CurveLine(Frozen):
    label: str
    date: str
    values: tuple[float | None, ...]


class Curve(Frozen):
    tenors: tuple[str, ...]
    lines: tuple[CurveLine, ...]


class Spread(Frozen):
    points: tuple[tuple[str, float], ...]
    recessions: tuple[tuple[str, str], ...]


class CalendarEntry(Frozen):
    date: str
    name: str
    source: str


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


class ViewOption(Frozen):
    id: str
    key: str
    title: str
    core: int
    extra: int


class Dashboard(Frozen):
    version: str
    generated_at: str
    elapsed_seconds: float
    demo: bool
    sources: tuple[SourceSummary, ...]
    tape: tuple[str, ...]
    regime: tuple[RegimeCell, ...]
    curve: Curve | None
    spread: Spread | None
    projections: Projections | None
    panels: tuple[Panel, ...]
    calendar: tuple[CalendarEntry, ...]
    deep: tuple[str, ...]
    views: tuple[ViewOption, ...]
    settings_file: str | None = None
    settings_error: str | None = None
