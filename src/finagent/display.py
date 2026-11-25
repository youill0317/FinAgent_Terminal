from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Final, Literal, cast

from finagent import catalog
from finagent.analysis import web
from finagent.fetch import ECOS_KEY_ENV, FRED_KEY_ENV
from finagent.investors import FIGI_KEY
from finagent.markets import ALPACA_KEY_ENV, ALPACA_SECRET_ENV
from finagent.models import AREA_LABEL

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from finagent.analysis.service import Event
    from finagent.limits import LimitState
    from finagent.models import (
        Dashboard,
        Fmt,
        IndicatorRow,
        Panel,
        ProjectionCell,
        Projections,
        RegimeCell,
    )

type Color = Literal["red", "orange", "yellow", "blue", "green", "violet", "gray"]

KEY_HELP: Final = {
    "FRED": (FRED_KEY_ENV, "https://fredaccount.stlouisfed.org/apikeys"),
    "ECOS": (ECOS_KEY_ENV, "https://ecos.bok.or.kr/api/#/AuthKeyApply"),
    "ALPACA": (
        f"{ALPACA_KEY_ENV}` · `{ALPACA_SECRET_ENV}",
        "https://app.alpaca.markets/signup",
    ),
    "OPENFIGI": (FIGI_KEY, "https://www.openfigi.com/user/profile/api-key"),
}
SOURCE_COLOR: Final[dict[str, Color]] = {
    "ok": "green",
    "partial": "orange",
    "missing_key": "gray",
    "error": "red",
}
TONE: Final[dict[str, tuple[Color, str]]] = {
    "good": ("green", "●"),
    "warn": ("orange", "◆"),
    "crit": ("red", "■"),
    "neutral": ("gray", "○"),
}
STATUS_TEXT: Final = {
    "ready": "준비됨",
    "missing_key": "키 없음",
    "not_installed": "설치 안 됨",
}
WEEKDAYS: Final = "월화수목금토일"
STATE_TEXT: Final = {
    "working": "작업 중",
    "done": "완료",
    "failed": "실패",
    "stopped": "중지",
}
STOPPED: Final = "중지 완료"
STOPPED_HELP: Final = (
    "이 턴은 대화 기록에 미저장  \n다음 질문은 이 턴이 없던 것처럼 이어짐"
)
SUMMARY_HELP: Final = (
    "이 턴의 데이터 도구 호출 횟수 및 웹 검색·원문 조회 횟수\n\n"
    "빨간색 `?` 표시 인용은 수신 데이터에 없는 지표 ID 또는 미수신 웹 출처"
)
LIVE_MARK: Final = re.compile(r"\s?\[\[[^\]]*\]\]")
OPEN_MARK: Final = re.compile(r"\s?\[\[[^\]]*$")
LIVE_LIMIT: Final = 6000
PERCENT_DEC: Final = 2


def number(value: float, dec: int) -> str:
    return f"{value:,.{dec}f}"


def fmt(value: float | None, f: Fmt) -> str:
    if value is None:
        return "—"
    text = number(abs(value), f.dec)
    sign = "-" if value < 0 else ("+" if f.signed and value > 0 else "")
    if not any(digit in text for digit in "123456789"):
        sign = ""
    return f"{sign}{f.pre}{text}{f.suf}"


def share(value: float) -> str:
    text = f"{value:.0%}"
    if text == "0%" and value > 0:
        return "<1%"
    if text == "100%" and value < 1:
        return ">99%"
    return text


def percent(row: IndicatorRow) -> bool:
    return catalog.BY_ID[row.id].change == "pct"


def change(row: IndicatorRow) -> float | None:
    if row.value is None or row.prev is None:
        return None
    if percent(row):
        diff, dec = (row.value / row.prev - 1) * 100, PERCENT_DEC
    else:
        diff, dec = row.value - row.prev, row.fmt.dec
    return 0.0 if abs(diff) < 10**-dec / 2 else diff


def change_text(row: IndicatorRow) -> str:
    diff = change(row)
    if diff is None:
        return ""
    if diff == 0:
        return "—"
    arrow = "▲" if diff > 0 else "▼"
    if percent(row):
        return f"{arrow} {number(abs(diff), PERCENT_DEC)}%"
    return f"{arrow} {number(abs(diff), row.fmt.dec)}"


def arrow_dir(text: str) -> Literal["up", "down"] | None:
    if "▲" in text:
        return "up"
    if "▼" in text:
        return "down"
    return None


def metric_color(row: IndicatorRow) -> Literal["red", "blue", "off"]:
    diff = change(row)
    if not diff:
        return "off"
    return "red" if diff > 0 else "blue"


def unit(f: Fmt) -> str:
    return f"{f.pre}{f.suf}"


def metric_delta(row: IndicatorRow) -> str | None:
    diff = change(row)
    if diff is None:
        return None
    if percent(row):
        pre, dec, suf = "", PERCENT_DEC, "%"
    else:
        pre, dec = row.fmt.pre, row.fmt.dec
        suf = "%p" if row.fmt.suf == "%" else row.fmt.suf
    text = f"{diff:+,.{dec}f}"
    sign = "" if diff == 0 else text[0]
    return f"{sign}{pre}{text[1:]}{suf}"


def short_date(row: IndicatorRow) -> str:
    text = row.date or ""
    if row.freq == "Q":
        return text[2:]
    if row.freq == "M":
        return text[2:].replace("-", ".", 1)
    return text[5:].replace("-", "/", 1)


def weekday(day: str) -> str:
    return f"{day[5:].replace('-', '/')} {WEEKDAYS[date.fromisoformat(day).weekday()]}"


def rows_by_id(d: Dashboard) -> dict[str, IndicatorRow]:
    return {row.id: row for panel in d.panels for row in panel.rows}


def area_name(row: IndicatorRow) -> str:
    return AREA_LABEL[row.area]


def trend(row: IndicatorRow | None) -> list[float] | None:
    if row is None:
        return None
    return [value for _, value in row.history] or None


def regime_card(cell: RegimeCell) -> tuple[str, str]:
    return f"{cell.label} · {cell.measure}", cell.value


def panel_of(d: Dashboard) -> dict[str, str]:
    return {row.id: panel.id for panel in d.panels for row in panel.rows}


def panel_table(rows: Sequence[IndicatorRow]) -> dict[str, list[object]]:
    table: dict[str, list[object]] = {
        "지표": [],
        "국가": [],
        "최신": [],
        "변화": [],
        "추이": [],
        "기준": [],
    }
    for row in rows:
        label = row.label
        ok = row.status == "ok"
        missing = "키 없음" if row.status == "missing_key" else "수신 실패"
        table["지표"].append(label)
        table["국가"].append(area_name(row))
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


def panel_stale(panel: Panel) -> list[str]:
    return [
        f"{row.code}: {row.message}"
        for row in panel.rows
        if row.status == "ok" and row.message
    ]


CHART_RANGES: Final = {"2Y": 24, "5Y": 60, "10Y": 120}


def label_day(label: str) -> date:
    if "Q" in label:
        year, quarter = label.split("Q")
        return date(int(year), (int(quarter) - 1) * 3 + 1, 1)
    year, month = label.split("-")
    return date(int(year), int(month), 1)


def history_series(row: IndicatorRow) -> tuple[tuple[date, float], ...]:
    return tuple((label_day(label), value) for label, value in row.history)


def clip(series: Sequence[tuple[date, float]], months: int) -> list[tuple[date, float]]:
    if not series:
        return list(series)
    last = series[-1][0]
    count = last.year * 12 + last.month - months
    cutoff = date(count // 12, count % 12 + 1, 1)
    return [point for point in series if point[0] >= cutoff]


def sep_value(cell: ProjectionCell | None) -> str:
    if cell is None or cell.value is None:
        return "—"
    return number(cell.value, 1)


def sep_change(cell: ProjectionCell | None) -> str:
    if cell is None or cell.value is None:
        return ""
    if cell.prev is None:
        return "신규"
    diff = cell.value - cell.prev
    if abs(diff) < 0.05:  # noqa: PLR2004
        return "—"
    return f"{'▲' if diff > 0 else '▼'} {number(abs(diff), 1)}"


def sep_change_key(name: str) -> str:
    return f"{name}Δ"


def sep_table(p: Projections) -> dict[str, list[str]]:
    longer = any(row.longer_run for row in p.rows)
    names = [*p.years, *(["장기"] if longer else [])]
    table: dict[str, list[str]] = {"지표": [], "현재": []}
    for name in names:
        table[name] = []
        table[sep_change_key(name)] = []
    for row in p.rows:
        table["지표"].append(row.label)
        current = "—" if row.current is None else number(row.current, row.current_dec)
        table["현재"].append(current)
        cells: dict[str, ProjectionCell | None] = dict(
            zip(p.years, row.years, strict=False)
        )
        if longer:
            cells["장기"] = row.longer_run
        for name in names:
            table[name].append(sep_value(cells.get(name)))
            table[sep_change_key(name)].append(sep_change(cells.get(name)))
    return table


def md(text: str) -> str:
    return text.replace("$", "\\$").replace("~", "\\~")


def live_text(raw: str) -> str:
    return md(OPEN_MARK.sub("", LIVE_MARK.sub("", raw)))


def live_tail(raw: str, limit: int = LIVE_LIMIT) -> str:
    if len(raw) <= limit:
        return live_text(raw)
    _, _, whole_lines = raw[-limit:].partition("\n")
    return live_text(whole_lines)


def _piece(piece: Mapping[str, object], links: Mapping[str, str]) -> str:
    if "t" in piece:
        return md(str(piece["t"]))
    ids = cast("list[str]", piece.get("c", []))
    known = cast("list[bool]", piece.get("k", []))
    urls = cast("list[str]", piece.get("u", []))
    out: list[str] = []
    for index, (cite, ok) in enumerate(zip(ids, known, strict=False)):
        url = urls[index] if index < len(urls) else ""
        if web.is_web(cite):
            name = md(web.label(cite))
            out.append(f"([{name}]({url}))" if ok and url else f":red[`{name}?`]")
        elif not ok:
            out.append(f":red[`{cite}?`]")
        elif cite in links:
            out.append(f"[`{cite}`](#{links[cite]})")
        else:
            out.append(f"`{cite}`")
    return " " + " ".join(out) if out else ""


def answer_markdown(lines: Sequence[object], links: Mapping[str, str]) -> str:
    return "\n".join(
        "".join(
            _piece(cast("Mapping[str, object]", piece), links)
            for piece in cast("list[object]", line)
        )
        for line in lines
    )


def conclusion_line(lines: Sequence[object]) -> str:
    for line in lines:
        text = answer_markdown([line], {})
        if text.strip():
            return text.strip()
    return ""


def limit_line(state: LimitState, now: float) -> str:
    at = f"{datetime.fromtimestamp(state.next_at).astimezone():%H:%M}"
    if state.status == "waiting":
        minutes, seconds = divmod(max(round(state.next_at - now), 0), 60)
        wait = f"{minutes}분 {seconds}초" if minutes else f"{seconds}초"
        return f"{state.source} 대기 {wait}"
    if state.status == "reached":
        return f"{state.source} 다음 가능 {at}"
    if state.status == "blocked":
        return f"{state.source} 제한 응답 · 다음 가능 {at}"
    return ""


def limit_lines(states: Iterable[LimitState], now: float) -> str:
    return " · ".join(filter(None, (limit_line(state, now) for state in states)))


def run_summary(data_calls: int, web_steps: int) -> str:
    return f"데이터 {data_calls}회 · 웹 {web_steps}회"


@dataclass(slots=True)
class TurnView:
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
    threads: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.raw or self.tools or self.notices or self.lines is not None)

    @property
    def run_line(self) -> str:
        if not (self.tokens_in or self.tokens_out):
            return self.run
        note = f" · {self.usage_note}" if self.usage_note else ""
        used = f"토큰 입력 {self.tokens_in:,} · 출력 {self.tokens_out:,}{note}"
        return f"{self.run} · {used}"


@dataclass(slots=True)
class AnalysisLog:
    title: str = ""
    state: str = ""
    received: list[tuple[str, str]] = field(default_factory=list)
    sources: list[tuple[str, str]] = field(default_factory=list)
    turns: list[TurnView] = field(default_factory=list)
    report: bool = False
    conclusion: str = ""

    @property
    def working(self) -> bool:
        return self.state == "working"

    def apply(self, event: Event) -> None:
        kind = event.get("type")
        if kind == "turn":
            self.state = "working"
            self.turns.append(TurnView(str(event["question"])))
            return
        turn = self.turns[-1]
        if kind == "start":
            turn.run = f"{event['provider_title']} · {event['model']}"
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
            self._done(turn, event)
        elif kind == "end":
            self._end(turn, str(event["state"]))

    def _end(self, turn: TurnView, state: str) -> None:
        self.state = state
        if state == "stopped":
            turn.notices.append((STOPPED, False))

    def _done(self, turn: TurnView, event: Event) -> None:
        turn.lines = cast("list[object]", event["lines"])
        turn.summary = run_summary(
            cast("int", event["data_calls"]), cast("int", event["web_steps"])
        )
        received = cast("list[Mapping[str, str]]", event["received"])
        self.received = [(item["cite"], item["info"]) for item in received]
        cited = cast("list[Mapping[str, str]]", event.get("sources") or [])
        self.sources.extend(
            (item["title"] or web.host(item["url"]), item["url"])
            for item in cited
            if item["url"] not in {url for _, url in self.sources}
        )
        if event.get("report"):
            self.report = True
            self.conclusion = conclusion_line(turn.lines)


@dataclass(slots=True)
class Board:
    main: AnalysisLog = field(default_factory=AnalysisLog)
    threads: dict[str, AnalysisLog] = field(default_factory=dict)

    def apply(self, event: Event) -> None:
        key = str(event["thread"])
        if event["type"] == "thread":
            title = str(event["title"])
            self.threads[key] = AnalysisLog(title, "working")
            self.main.turns[-1].threads.append(key)
            return
        (self.threads[key] if key else self.main).apply(event)
