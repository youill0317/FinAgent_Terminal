"""The dashboard snapshot as one text document the model reads.

Each indicator becomes one ``<indicator id=...>`` element. Values are written
the way the page shows them (same prefix, suffix, decimals and sign), so the
numbers in an answer can be checked against what the user sees. The same
snapshot always gives the same text, which lets provider prompt caches hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from xml.sax.saxutils import escape, quoteattr

from finagent.analysis.evidence import Evidence, numbers_in

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent.models import Dashboard, Fmt, IndicatorRow, Panel, Projections

REGIME_IDS: Final[Mapping[str, str]] = {
    "물가": "regime_inflation",
    "고용": "regime_labor",
    "금리곡선": "regime_curve",
    "금융여건": "regime_fincon",
}
SEP_IDS: Final[Mapping[str, str]] = {
    "FRED:FEDTARMD": "sep_fedfunds",
    "FRED:GDPC1MD": "sep_gdp",
    "FRED:UNRATEMD": "sep_unrate",
    "FRED:PCECTPIMD": "sep_pce",
    "FRED:JCXFEMD": "sep_core_pce",
}
MINUS: Final = "−"


@dataclass(frozen=True, slots=True)
class Document:
    text: str
    evidence: dict[str, Evidence]
    view: str
    snapshot: str


def fmt_value(value: float, fmt: Fmt) -> str:
    """Format like the page: thousands separators, typographic minus."""
    sign = MINUS if value < 0 else ("+" if fmt.signed and value > 0 else "")
    return f"{sign}{fmt.pre}{abs(value):,.{fmt.dec}f}{fmt.suf}"


def plain(value: float, dec: int) -> str:
    text = f"{value:,.{dec}f}"
    return text.replace("-", MINUS)


def build_document(dashboard: Dashboard, *, view: str) -> Document:
    snapshot, view_attr = quoteattr(dashboard.generated_at), quoteattr(view)
    lines = [f"<indicator_document snapshot={snapshot} view={view_attr}>"]
    evidence: list[Evidence] = []
    for panel in dashboard.panels:
        for row in panel.rows:
            text, item = _indicator(panel, row)
            lines.append(text)
            if item is not None:
                evidence.append(item)
    for cell in dashboard.regime:
        cite = REGIME_IDS.get(cell.label, f"regime_{cell.label}")
        body = (
            f"{cell.value} ({cell.unit}) · 규칙: {cell.rule} · 판정: {cell.tag}"
            if cell.value
            else "데이터 없음"
        )
        lines.append(_element(cite, f"국면 · {cell.label}", "FinAgent 규칙", body))
        evidence.append(
            Evidence(
                cite,
                f"국면 · {cell.label}",
                numbers_in(cell.value),
                numbers_in(cell.rule),
            )
        )
    if dashboard.curve is not None:
        curve_lines = [
            f"{line.label} {line.date}: "
            + " · ".join(
                f"{tenor} {plain(value, 2)}"
                for tenor, value in zip(
                    dashboard.curve.tenors, line.values, strict=True
                )
                if value is not None
            )
            for line in dashboard.curve.lines
        ]
        lines.append(
            _element(
                "curve",
                "미 국채 수익률 곡선",
                "FRED:DGS1MO~DGS30",
                " | ".join(curve_lines),
            )
        )
        evidence.append(
            Evidence(
                "curve",
                "미 국채 수익률 곡선",
                extra=tuple(
                    value
                    for line in dashboard.curve.lines
                    for value in line.values
                    if value is not None
                ),
            )
        )
    if dashboard.spread is not None and dashboard.spread.points:
        points = dashboard.spread.points
        bands = ", ".join(
            f"{start}~{end}" for start, end in dashboard.spread.recessions
        )
        body = (
            f"월평균 {len(points)}개 {points[0][0]}~{points[-1][0]} · "
            f"최근 {points[-1][0]} {plain(points[-1][1], 2)}%p · "
            f"최저 {plain(min(v for _, v in points), 2)} · "
            f"최고 {plain(max(v for _, v in points), 2)} · "
            f"NBER 침체 구간 {bands or '없음'}"
        )
        lines.append(
            _element(
                "spread", "10년물 − 2년물 장기 추이", "FRED:T10Y2Y, FRED:USREC", body
            )
        )
        evidence.append(
            Evidence("spread", "10년물 − 2년물 장기 추이", tuple(v for _, v in points))
        )
    if dashboard.projections is not None:
        sep_lines, sep_evidence = _projections(dashboard.projections)
        lines.extend(sep_lines)
        evidence.extend(sep_evidence)
    if dashboard.calendar:
        body = " · ".join(
            f"{entry.date} {entry.name} ({entry.source})"
            for entry in dashboard.calendar
        )
        lines.append(_element("calendar", "발표 일정", "FRED releases", body))
    lines.append("</indicator_document>")
    return Document(
        text="\n".join(lines),
        evidence={item.cite: item for item in evidence},
        view=view,
        snapshot=dashboard.generated_at,
    )


def _indicator(panel: Panel, row: IndicatorRow) -> tuple[str, Evidence | None]:
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
    head = (
        "<indicator " + " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items()) + ">"
    )
    if row.status != "ok" or row.value is None:
        return f"{head}\n{escape(row.message or '데이터 없음')}\n</indicator>", None
    parts = [f"최신 {row.date} {fmt_value(row.value, row.fmt)}"]
    if row.prev is not None:
        parts.append(f"직전 {fmt_value(row.prev, row.fmt)}")
    if row.history:
        label = "이력(월평균)" if row.freq in {"D", "W"} else "이력"
        parts.append(
            f"{label} "
            + ", ".join(
                f"{day} {plain(value, row.fmt.dec)}" for day, value in row.history
            )
        )
    points = [value for _, value in row.history]
    if row.prev is not None:
        points.append(row.prev)
    points.append(row.value)
    body = escape(" · ".join(parts))
    return (
        f"{head}\n{body}\n</indicator>",
        Evidence(row.id, row.label, tuple(points)),
    )


def _projections(sep: Projections) -> tuple[list[str], list[Evidence]]:
    lines: list[str] = []
    evidence: list[Evidence] = []
    published = f"발표 {sep.published}" + (
        f" (직전 발표 {sep.prior_published})" if sep.prior_published else ""
    )
    for row in sep.rows:
        cite = SEP_IDS.get(row.code, f"sep_{row.label}")
        cells: list[str] = [published]
        values: list[float] = []
        if row.current is not None:
            cells.append(
                f"현재 {plain(row.current, row.current_dec)} ({row.current_note})"
            )
            values.append(row.current)
        for year, cell in zip(sep.years, row.years, strict=True):
            if cell.value is None:
                continue
            prior = (
                f" (직전 발표 {plain(cell.prev, 1)})" if cell.prev is not None else ""
            )
            cells.append(f"{year} {plain(cell.value, 1)}{prior}")
            values.append(cell.value)
            if cell.prev is not None:
                values.extend((cell.prev, cell.value - cell.prev))
        if row.longer_run is not None and row.longer_run.value is not None:
            prior = (
                f" (직전 발표 {plain(row.longer_run.prev, 1)})"
                if row.longer_run.prev is not None
                else ""
            )
            cells.append(f"장기 {plain(row.longer_run.value, 1)}{prior}")
            values.append(row.longer_run.value)
        name = f"FOMC 경제전망 · {row.label}"
        lines.append(_element(cite, name, row.code, " · ".join(cells)))
        evidence.append(Evidence(cite, name, extra=tuple(values)))
    return lines, evidence


def _element(cite: str, name: str, source: str, body: str) -> str:
    return (
        f"<indicator id={quoteattr(cite)} name={quoteattr(name)} "
        f'source={quoteattr(source)} status="ok">\n{escape(body)}\n</indicator>'
    )
