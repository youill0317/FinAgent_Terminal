"""Text for the page: number formats, table rows and the analysis log.

Nothing here touches Streamlit, so it is tested with plain pytest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Final, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from finagent.analysis.service import Event
    from finagent.models import (
        Dashboard,
        Fmt,
        IndicatorRow,
        Panel,
        ProjectionCell,
        Projections,
    )

type Color = Literal["red", "orange", "yellow", "blue", "green", "violet", "gray"]

KEY_HELP: Final = {
    "FRED": ("FRED_API_KEY", "https://fredaccount.stlouisfed.org/apikeys"),
    "ECOS": ("ECOS_API_KEY", "https://ecos.bok.or.kr/api/#/AuthKeyApply"),
}
TAPE_LABELS: Final = {
    "us_fftu": "연준 상단",
    "us_10y": "UST 10Y",
    "us_10y3m": "10Y−3M",
    "us_sp500": "S&P 500",
    "us_dollar": "달러 광의",
    "wti": "WTI",
    "us_vix": "VIX",
    "kr_usdkrw": "USD/KRW",
    "kr_ktb10": "KTB 10Y",
    "kr_kospi": "KOSPI",
}
SOURCE_COLOR: Final[dict[str, Color]] = {
    "ok": "green",
    "partial": "orange",
    "missing_key": "gray",
    "error": "red",
}
TONE: Final[dict[str, tuple[Color, str]]] = {
    "good": ("green", "●"),
    "warn": ("orange", "▲"),
    "crit": ("red", "■"),
    "neutral": ("gray", "◆"),
}
STATUS_TEXT: Final = {
    "ready": "준비됨",
    "missing_key": "키 없음",
    "not_installed": "설치 안 됨",
}
NUMBER_COLOR: Final = {"ok": "green", "mismatch": "red", "unsupported": "orange"}
WEEKDAYS: Final = "월화수목금토일"
IDLE_NOTE: Final = "수동 실행 · 대화는 메모리에만"
STOPPED: Final = "멈췄습니다. 이 턴은 대화에 남지 않습니다."
CUT_SHORT: Final = "응답이 끝나기 전에 실행이 끝났습니다."
# Citation marks ``[[id, id]]`` are hidden while the answer streams in.
LIVE_MARK: Final = re.compile(r"\s?\[\[[^\]]*\]\]")
OPEN_MARK: Final = re.compile(r"\s?\[\[[^\]]*$")


# ---------------------------------------------------------------- numbers


def number(value: float, dec: int) -> str:
    return f"{value:,.{dec}f}"


def fmt(value: float | None, f: Fmt) -> str:
    if value is None:
        return "—"
    sign = "−" if value < 0 else ("+" if f.signed and value > 0 else "")
    return f"{sign}{f.pre}{number(abs(value), f.dec)}{f.suf}"


def change(row: IndicatorRow) -> float | None:
    """Latest minus previous, ``0.0`` when it rounds away, ``None`` if unknown."""
    if row.value is None or row.prev is None:
        return None
    diff = row.value - row.prev
    return 0.0 if abs(diff) < 10**-row.fmt.dec / 2 else diff


def change_text(row: IndicatorRow) -> str:
    diff = change(row)
    if diff is None:
        return ""
    if diff == 0:
        return "— 0"
    return f"{'▲' if diff > 0 else '▼'} {number(abs(diff), row.fmt.dec)}"


def metric_delta(row: IndicatorRow) -> str | None:
    """A signed delta for ``st.metric``, which colors by the leading sign."""
    diff = change(row)
    return None if diff is None else f"{diff:+,.{row.fmt.dec}f}"


def short_date(row: IndicatorRow) -> str:
    text = row.date or ""
    if row.freq == "Q":
        return text[2:]
    if row.freq == "M":
        return text[2:].replace("-", ".", 1)
    return text[5:].replace("-", "/", 1)


def short_time(iso: str) -> str:
    return iso.replace("T", " ")[:16]


def stamp(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M:%S")


def weekday(day: str) -> str:
    return f"{day[5:].replace('-', '/')} {WEEKDAYS[date.fromisoformat(day).weekday()]}"


# ---------------------------------------------------------------- view


def view_label(d: Dashboard) -> str:
    if not d.deep:
        return "핵심"
    if len(d.deep) == len(d.views):
        return "전체"
    return f"핵심 + 심화 {len(d.deep)}"


def view_note(core: int, extra: int, *, curve: bool) -> str:
    if curve:
        return f"차트 · 시리즈 {extra}개"
    if not extra:
        return f"모두 핵심 {core}개"
    return f"핵심 {core} · 심화 +{extra}" if core else f"기본 숨김 · 켜면 {extra}개"


def view_summary(d: Dashboard, on: frozenset[str]) -> str:
    panels = [view for view in d.views if view.id != "curve"]
    total = sum(view.core + view.extra for view in panels)
    shown = sum(view.core + (view.extra if view.id in on else 0) for view in panels)
    curve = " · 수익률 곡선 차트" if "curve" in on else ""
    return f"표시 지표 {shown} / {total}개{curve}"


def row_count(d: Dashboard) -> int:
    return sum(len(panel.rows) for panel in d.panels)


def rows_by_id(d: Dashboard) -> dict[str, IndicatorRow]:
    return {row.id: row for panel in d.panels for row in panel.rows}


def panel_of(d: Dashboard) -> dict[str, str]:
    """Indicator id to the panel it is shown in, for citation links."""
    return {row.id: panel.id for panel in d.panels for row in panel.rows}


# ---------------------------------------------------------------- tables


def panel_table(panel: Panel) -> dict[str, list[object]]:
    table: dict[str, list[object]] = {
        "지표": [],
        "코드": [],
        "최신": [],
        "변화": [],
        "추이": [],
        "기준": [],
    }
    for row in panel.rows:
        code = row.code + (f" · © {row.copyright}" if row.copyright else "")
        ok = row.status == "ok"
        missing = "키 없음" if row.status == "missing_key" else "받지 못함"
        table["지표"].append(row.label)
        table["코드"].append(code)
        table["최신"].append(fmt(row.value, row.fmt) if ok else missing)
        table["변화"].append(change_text(row) if ok else "")
        table["추이"].append([value for _, value in row.history] if ok else [])
        table["기준"].append(short_date(row) if ok else "")
    return table


def panel_problems(panel: Panel) -> list[str]:
    return [
        f"{row.code}: {row.message}"
        for row in panel.rows
        if row.status != "ok" and row.message
    ]


def sep_cell(cell: ProjectionCell | None) -> str:
    if cell is None or cell.value is None:
        return "—"
    value = number(cell.value, 1)
    if cell.prev is None:
        return f"{value} 신규"
    diff = cell.value - cell.prev
    if abs(diff) < 0.05:  # noqa: PLR2004 - half of the one decimal shown
        return f"{value} — 0"
    return f"{value} {'▲' if diff > 0 else '▼'} {number(abs(diff), 1)}"


def sep_table(p: Projections) -> dict[str, list[str]]:
    longer = any(row.longer_run for row in p.rows)
    table: dict[str, list[str]] = {"지표": [], "코드": [], "현재": []}
    for year in p.years:
        table[year] = []
    if longer:
        table["장기"] = []
    for row in p.rows:
        table["지표"].append(row.label)
        table["코드"].append(row.code)
        current = "—" if row.current is None else number(row.current, row.current_dec)
        table["현재"].append(f"{current} {row.current_note}".strip())
        for year, cell in zip(p.years, row.years, strict=False):
            table[year].append(sep_cell(cell))
        if longer:
            table["장기"].append(sep_cell(row.longer_run))
    return table


# ---------------------------------------------------------------- analysis text


def md(text: str) -> str:
    """Escape what ``st.markdown`` would read as LaTeX."""
    return text.replace("$", "\\$")


def live_text(raw: str) -> str:
    return md(OPEN_MARK.sub("", LIVE_MARK.sub("", raw)))


def _piece(piece: Mapping[str, object], links: Mapping[str, str]) -> str:
    if "t" in piece:
        return md(str(piece["t"]))
    if "n" in piece:
        shown = md(str(piece["n"]))
        status = str(piece.get("s"))
        actual = piece.get("a")
        color = NUMBER_COLOR.get(status, "gray")
        if status == "mismatch" and actual:
            return f":{color}[{shown}] :gray[(실제 {md(str(actual))})]"
        return f":{color}[{shown}]"
    ids = cast("list[str]", piece.get("c", []))
    known = cast("list[bool]", piece.get("k", []))
    out: list[str] = []
    for cite, ok in zip(ids, known, strict=False):
        if not ok:
            out.append(f":red[`{cite}?`]")
        elif cite in links:
            out.append(f"[`{cite}`](#{links[cite]})")
        else:
            out.append(f"`{cite}`")
    return " " + " ".join(out) if out else ""


def answer_markdown(lines: Sequence[object], links: Mapping[str, str]) -> str:
    """The verified answer: numbers colored by check, citations as links."""
    return "\n".join(
        "".join(
            _piece(cast("Mapping[str, object]", piece), links)
            for piece in cast("list[object]", line)
        )
        for line in lines
    )


def verify_summary(counts: Mapping[str, int], tool_calls: int) -> str:
    mismatch = counts.get("mismatch", 0)
    unsupported = counts.get("unsupported", 0)
    bits = [
        f":green[일치 **{counts.get('ok', 0)}**]",
        f":red[다름 **{mismatch}**]" if mismatch else "다름 **0**",
        f":orange[근거 없음 **{unsupported}**]" if unsupported else "근거 없음 **0**",
    ]
    if unknown := counts.get("unknown_id", 0):
        bits.append(f":red[모르는 근거 **{unknown}**]")
    bits.append(f"도구 {tool_calls}회")
    return "숫자 검증 · " + " · ".join(bits)


@dataclass(slots=True)
class TurnView:
    """One question and what came back for it."""

    question: str
    run: str = "준비 중…"
    tools: list[tuple[str, bool]] = field(default_factory=list)
    raw: str = ""
    lines: list[object] | None = None
    summary: str = ""
    notices: list[tuple[str, bool]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    usage_note: str = ""

    @property
    def run_line(self) -> str:
        if not (self.tokens_in or self.tokens_out):
            return self.run
        note = f" · {self.usage_note}" if self.usage_note else ""
        used = f"토큰 입력 {self.tokens_in:,} · 출력 {self.tokens_out:,}{note}"
        return f"{self.run} · {used}"


@dataclass(slots=True)
class AnalysisLog:
    """The page's copy of one conversation; the service keeps the real one."""

    conversation: str = ""
    snapshot: str = ""
    stale: bool = False
    stopped: bool = False
    received: list[tuple[str, str]] = field(default_factory=list)
    turns: list[TurnView] = field(default_factory=list)

    @property
    def header(self) -> str:
        if not self.snapshot:
            return IDLE_NOTE
        return f"데이터 {short_time(self.snapshot)}에 고정 · 메모리에만"

    def ask(self, question: str) -> None:
        self.stopped = False
        self.turns.append(TurnView(question))

    def apply(self, event: Event) -> bool:
        """Apply one event to the last turn; ``True`` once the turn is done."""
        turn = self.turns[-1]
        kind = event.get("type")
        if kind == "start":
            self.conversation = str(event["conversation"])
            self.snapshot = str(event["snapshot"])
            self.stale = self.stale or bool(event.get("stale"))
            turn.run = (
                f"{event['provider_title']} · {event['model']} · {event['view']}"
                f" · 데이터 {short_time(self.snapshot)}"
            )
        elif kind == "text":
            turn.raw += str(event["text"])
        elif kind == "tool":
            turn.tools.append((str(event["summary"]), bool(event["ok"])))
        elif kind == "usage":
            turn.tokens_in += cast("int", event.get("input") or 0)
            turn.tokens_out += cast("int", event.get("output") or 0)
            turn.usage_note = str(event.get("note") or "")
        elif kind == "error":
            turn.notices.append((str(event["message"]), True))
        elif kind == "done":
            report = cast("Mapping[str, object]", event["verify"])
            turn.lines = cast("list[object]", report["lines"])
            turn.summary = verify_summary(
                cast("Mapping[str, int]", report["counts"]),
                cast("int", event["tool_calls"]),
            )
            received = cast("list[Mapping[str, str]]", event["received"])
            self.received = [(item["cite"], item["info"]) for item in received]
            return True
        return False

    def finish(self, *, done: bool) -> None:
        """The run ended; say why when it ended without an answer."""
        turn = self.turns[-1]
        if self.stopped:
            turn.notices.append((STOPPED, False))
        elif not done and not turn.notices:
            turn.notices.append((CUT_SHORT, True))
        self.stopped = False
