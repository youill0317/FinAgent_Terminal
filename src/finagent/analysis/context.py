"""대시보드 지표·국면·수익률 곡선·경제전망(SEP) AI 컨텍스트 XML 문서 변환기."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final
from xml.sax.saxutils import escape, quoteattr

from finagent import display
from finagent.sources import parse_period

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent.models import Dashboard, IndicatorRow, Panel, Projections

MISSING: Final = "—"
STALE_LAG_DAYS: Final[Mapping[str, int]] = {
    "D": 5,
    "W": 14,
    "M": 40,
    "Q": 115,
    "A": 400,
}


def period_end(value: date | str, freq: str) -> date | None:
    """관측 시점(일·월·분기·연)의 해당 기간 종료일을 반환한다."""
    day = value if isinstance(value, date) else parse_period(value)
    if day is None:
        return None
    if freq in {"D", "W"}:
        return day
    if freq == "M":
        return date(day.year, day.month, monthrange(day.year, day.month)[1])
    if freq == "Q":
        end_month = ((day.month - 1) // 3 + 1) * 3
        return date(day.year, end_month, monthrange(day.year, end_month)[1])
    if freq == "A":
        return date(day.year, 12, 31)
    return day


def stale_days(value: date | str, freq: str, today: date) -> int | None:
    """관측 종료일이 기준일 대비 지연 임계치 이상이면 경과 일수를 반환한다."""
    end = period_end(value, freq)
    limit = STALE_LAG_DAYS.get(freq)
    if end is None or limit is None:
        return None
    lag = (today - end).days
    return lag if lag >= limit else None


@dataclass(frozen=True, slots=True)
class Document:
    text: str
    ids: frozenset[str]
    snapshot: str
    base: str = ""


def build_document(dashboard: Dashboard) -> Document:
    snapshot = quoteattr(dashboard.generated_at)
    as_of = date.fromisoformat(dashboard.generated_at[:10])
    lines = [f"<indicator_document snapshot={snapshot}>"]
    ids: set[str] = set()
    for panel in dashboard.panels:
        for row in panel.rows:
            text, known = _indicator(panel, row, as_of)
            lines.append(text)
            if known:
                ids.add(row.id)
    for cell in dashboard.regime:
        if cell.value == MISSING:
            continue
        cite = f"regime_{cell.source}"
        body = f"{cell.value} ({cell.measure}) · 규칙: {cell.rule} · 판정: {cell.tag}"
        lines.append(element(cite, f"국면 · {cell.label}", "FinAgent 규칙", body))
        ids.add(cite)
    for curve in dashboard.curves:
        cite = f"curve_{curve.area.lower()}"
        curve_lines = [
            f"{line.label} {line.date}: "
            + " · ".join(
                f"{tenor} {display.number(value, 2)}"
                for tenor, value in zip(curve.tenors, line.values, strict=True)
                if value is not None
            )
            for line in curve.lines
        ]
        lines.append(
            element(
                cite,
                f"{curve.label} 수익률 곡선",
                "수익률 곡선",
                " | ".join(curve_lines),
            )
        )
        ids.add(cite)
    if dashboard.spread is not None and dashboard.spread.points:
        points = dashboard.spread.points
        bands = ", ".join(
            f"{start}~{end}" for start, end in dashboard.spread.recessions
        )
        body = (
            f"월평균 {len(points)}개 {points[0][0]}~{points[-1][0]} · "
            f"최근 {points[-1][0]} {display.number(points[-1][1], 2)}%p · "
            f"최저 {display.number(min(v for _, v in points), 2)} · "
            f"최고 {display.number(max(v for _, v in points), 2)} · "
            f"NBER 침체 구간 {bands or '없음'}"
        )
        lines.append(
            element(
                "spread", "10년물 − 2년물 장기 추이", "FRED:T10Y2Y, FRED:USREC", body
            )
        )
        ids.add("spread")
    if dashboard.projections is not None:
        sep_lines, sep_ids = _projections(dashboard.projections)
        lines.extend(sep_lines)
        ids.update(sep_ids)
    for text, cite in _release_elements(dashboard):
        lines.append(text)
        ids.add(cite)
    lines.append("</indicator_document>")
    return Document(
        text="\n".join(lines),
        ids=frozenset(ids),
        snapshot=dashboard.generated_at,
    )


def _release_elements(dashboard: Dashboard) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for cite, title, entries in (
        ("calendar", "발표 일정", dashboard.calendar),
        ("recent_releases", "최근 발표 완료", dashboard.recent_releases),
    ):
        if entries:
            body = " · ".join(
                f"{entry.date} {entry.name} ({entry.source})" for entry in entries
            )
            out.append((element(cite, title, "FRED releases", body), cite))
    return out


def _indicator(panel: Panel, row: IndicatorRow, as_of: date) -> tuple[str, bool]:
    attrs = {
        "id": row.id,
        "name": row.label,
        "panel": panel.title,
        "source": row.code,
        "freq": row.freq,
        "status": row.status,
    }
    if row.copyright:
        attrs["copyright"] = row.copyright
    if row.status == "ok" and row.date:
        lag = stale_days(row.date, row.freq, as_of)
        if lag is not None:
            attrs["stale"] = "true"
            attrs["lag_days"] = str(lag)
    head = (
        "<indicator " + " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items()) + ">"
    )
    if row.status != "ok" or row.value is None:
        return f"{head}\n{escape(row.message or '데이터 없음')}\n</indicator>", False
    parts = [f"최신 {row.date} {display.fmt(row.value, row.fmt)}"]
    if row.prev is not None:
        parts.append(f"직전 {display.fmt(row.prev, row.fmt)}")
    if row.history:
        label = "이력(월평균)" if row.freq in {"D", "W"} else "이력"
        parts.append(
            f"{label} "
            + ", ".join(
                f"{day} {display.number(value, row.fmt.dec)}"
                for day, value in row.history
            )
        )
    body = escape(" · ".join(parts))
    return f"{head}\n{body}\n</indicator>", True


def _projections(sep: Projections) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    ids: list[str] = []
    published = f"발표 {sep.published}" + (
        f" (직전 발표 {sep.prior_published})" if sep.prior_published else ""
    )
    for row in sep.rows:
        cite = f"sep_{row.code.rpartition(':')[2].lower()}"
        cells: list[str] = [published]
        if row.current is not None:
            current = display.number(row.current, row.current_dec)
            cells.append(f"현재 {current} ({row.current_note})")
        for year, cell in zip(sep.years, row.years, strict=True):
            if cell.value is None:
                continue
            prior = (
                f" (직전 발표 {display.number(cell.prev, 1)})"
                if cell.prev is not None
                else ""
            )
            cells.append(f"{year} {display.number(cell.value, 1)}{prior}")
        if row.longer_run is not None and row.longer_run.value is not None:
            prior = (
                f" (직전 발표 {display.number(row.longer_run.prev, 1)})"
                if row.longer_run.prev is not None
                else ""
            )
            cells.append(f"장기 {display.number(row.longer_run.value, 1)}{prior}")
        name = f"FOMC 경제전망 · {row.label}"
        lines.append(element(cite, name, row.code, " · ".join(cells)))
        ids.append(cite)
    return lines, ids


def element(cite: str, name: str, source: str, body: str) -> str:
    return (
        f"<indicator id={quoteattr(cite)} name={quoteattr(name)} "
        f'source={quoteattr(source)} status="ok">\n{escape(body)}\n</indicator>'
    )
