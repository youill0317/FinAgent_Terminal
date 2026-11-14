from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
from statistics import median
from typing import TYPE_CHECKING, Final

import altair  # noqa: ICN001
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent import display
from finagent.models import Fmt
from finagent.ui import palette
from finagent.ui.guides import GUIDES

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from finagent.models import (
        Area,
        Curve,
        Dashboard,
        IndicatorRow,
        Panel,
        PolicyChange,
    )
    from finagent.ui.context import UiContext

type Line = Sequence[tuple[date, float]]
type Point = dict[str, str | float]


@altair.theme.register("finagent", enable=True)
def chart_theme() -> altair.theme.ThemeConfig:
    return {
        "config": {
            "locale": {
                "number": {
                    "decimal": ".",
                    "thousands": ",",
                    "grouping": [3],
                    "currency": ["$", ""],
                    "minus": "-",
                }
            }
        }
    }


AREA_ORDER: Final[tuple[Area, ...]] = ("US", "KR", "EA", "JP", "CN", "GL")
CHART_HEIGHT: Final = 260
SMALL_CHART_HEIGHT: Final = 180
RANGES: Final[dict[str, int]] = display.CHART_RANGES
MIN_METRIC_COLUMNS: Final = 3
METRICS_PER_ROW: Final = 5
POINT_LIMIT: Final = 60
MAX_POINTS: Final = 400
BAR_ROOM: Final = 0.3
YEARLY_TICKS_AFTER: Final = 3 * 365
WIDE_AREAS: Final = 4
PERCENT_FLOOR: Final = 10.0
PERCENT_GAP: Final = 4.0
LEVEL_GAP: Final = 2.5
PERCENT_UNITS: Final = frozenset({"%", "%p"})
AREA_WORDS: Final = re.compile(r"^(미국|한국|유로존|유럽|일본|중국|글로벌|국제)\s*")
PARENS: Final = re.compile(r"\s*\([^)]*\)")
NO_DATA: Final = "데이터 없음"
REBASED_FMT: Final = Fmt(dec=1)
REBASED_TITLE: Final = "선택 기간 첫날 = 100"
TABLE_HELP: Final = "지표 셀 선택 시 상세 추이 차트 표시"
CHANGE_HELP: Final = "직전 관측치 대비 변동폭  \n주가지수·환율·원자재·ETF는 등락률(%)"
SECTOR_LABELS: Final = {
    "technology": "기술",
    "financialservices": "금융",
    "healthcare": "헬스케어",
    "communicationservices": "커뮤니케이션",
    "industrials": "산업재",
    "consumercyclical": "임의소비재",
    "consumerdefensive": "필수소비재",
    "energy": "에너지",
    "realestate": "부동산",
    "utilities": "유틸리티",
    "basicmaterials": "소재",
}
LATEST_HELP: Final = "국가별 최신 관측값  \n기준 시점은 국가별 상이"
PREV_HELP: Final = "직전 관측값  \n최신 변동폭 산출 기준"


def panels_of(d: Dashboard, page: str) -> list[Panel]:
    return [p for p in d.panels if p.page == page]


def panel(d: Dashboard, panel_id: str) -> Panel | None:
    return next((p for p in d.panels if p.id == panel_id), None)


def rows_by_area(p: Panel) -> dict[Area, list[IndicatorRow]]:
    found: dict[Area, list[IndicatorRow]] = {}
    for row in p.rows:
        found.setdefault(row.area, []).append(row)
    return {area: found[area] for area in AREA_ORDER if area in found}


def headline(rows: Sequence[IndicatorRow]) -> list[IndicatorRow]:
    if not rows:
        return []
    name = base_label(rows[0])
    first: dict[Area, IndicatorRow] = {}
    for row in rows:
        if base_label(row) == name:
            _ = first.setdefault(row.area, row)
    return [first[area] for area in AREA_ORDER if area in first]


def notice(text: str) -> None:
    _ = st.badge("", icon=":material/priority_high:", color="orange", help=text)


def section(
    title: str,
    *,
    help: str | None = None,  # noqa: A002
    anchor: str | None = None,
) -> None:
    _ = st.subheader(title, anchor=anchor, help=help or None)


def metric(
    row: IndicatorRow,
    *,
    label: str | None = None,
    trend: bool = True,
) -> None:
    ok = row.status == "ok"
    missing = "키 없음" if row.status == "missing_key" else "수신 실패"
    delta = display.metric_delta(row) if ok else None
    _ = st.metric(
        label or row.label,
        display.fmt(row.value, row.fmt) if ok else missing,
        delta,
        delta_color=display.metric_color(row),
        delta_arrow="off" if display.change(row) == 0 else "auto",
        delta_description=display.short_date(row) if delta else None,
        help=GUIDES.get(row.id),
        border=True,
        height="stretch",
        chart_data=display.trend(row) if trend and ok else None,
    )


def metric_label(row: IndicatorRow, *, area_prefix: bool) -> str:
    name = display.area_name(row)
    if not area_prefix or row.label.startswith(name):
        return row.label
    return f"{name} · {row.label}"


def short_labels(rows: Sequence[IndicatorRow]) -> tuple[str, list[str]]:
    heads = [base_label(row) for row in rows]
    common = max(heads, key=heads.count)
    labels = [
        display.area_name(row) if head == common else head
        for row, head in zip(rows, heads, strict=True)
    ]
    return common, labels


def area_metrics(rows: Sequence[IndicatorRow], *, key: str) -> None:
    if not rows:
        return
    common, labels = short_labels(rows)
    unit = unit_text(rows[0].fmt)
    _ = st.caption(f"{common} · 단위 {unit}" if unit else common)
    with st.container(key=f"{key}_metrics", gap="small"):
        cols = st.columns(len(rows), gap="small")
        for col, row, label in zip(cols, rows, labels, strict=True):
            with col:
                metric(row, label=label)


def metrics(
    rows: Sequence[IndicatorRow], *, key: str, area_prefix: bool = False
) -> None:
    if not rows:
        return
    with st.container(key=f"{key}_metrics", gap="small"):
        lines = -(-len(rows) // METRICS_PER_ROW)
        per_line = -(-len(rows) // lines)
        slots = max(per_line, MIN_METRIC_COLUMNS)
        for start in range(0, len(rows), per_line):
            cols = st.columns(slots, gap="small")
            for col, row in zip(cols, rows[start : start + per_line], strict=False):
                with col:
                    metric(row, label=metric_label(row, area_prefix=area_prefix))


def policy_cards(changes: Sequence[PolicyChange], rows: Sequence[IndicatorRow]) -> None:
    by_area = {row.area: row for row in rows}
    for col, change in zip(st.columns(len(changes), gap="small"), changes, strict=True):
        diff = round(change.rate - change.prev, 2)
        row = by_area.get(change.area)
        _ = col.metric(
            change.bank,
            f"{change.rate:.2f}%",
            f"{diff:+.2f}%p" if diff else 0,
            delta_color="red" if diff > 0 else "blue" if diff < 0 else "off",
            delta_description=f"{change.date[2:]} · 12개월 {change.count_12m}회",
            help=GUIDES.get(row.id) if row else None,
            border=True,
            height="stretch",
            chart_data=display.trend(row),
        )


def range_control(key: str, default: str = "5Y") -> int:
    picked = st.segmented_control(
        "기간",
        list(RANGES),
        default=default,
        required=True,
        key=key,
        persist_state="session",
        label_visibility="collapsed",
        width="content",
    )
    return RANGES[picked or default]


def page_range(page: str) -> int:
    with st.container(horizontal=True, horizontal_alignment="right"):
        return range_control(f"{page}_range")


def chart_header(title: str, *, key: str, default: str) -> int:
    with st.container(
        horizontal=True,
        vertical_alignment="center",
        horizontal_alignment="distribute",
    ):
        _ = st.caption(title)
        return range_control(key, default)


def series_of(
    row: IndicatorRow, ctx: UiContext, months: int
) -> list[tuple[date, float]]:
    if row.status != "ok":
        return []
    series = ctx.full_series(row.id) or display.history_series(row)
    return display.clip(series, months)


def names(rows: Sequence[IndicatorRow]) -> list[str]:
    counts: dict[Area, int] = {}
    for row in rows:
        counts[row.area] = counts.get(row.area, 0) + 1
    out: list[str] = []
    for row in rows:
        name = display.area_name(row)
        text = (
            name
            if counts[row.area] == 1
            else metric_label(row, area_prefix=len(counts) > 1)
        )
        while text in out:
            text = f"{text} ({row.code})"
        out.append(text)
    return out


def colors_of(rows: Sequence[IndicatorRow], labels: Sequence[str]) -> dict[str, str]:
    seen: dict[Area, int] = {}
    out: dict[str, str] = {}
    area = palette.current().area
    for row, name in zip(rows, labels, strict=True):
        step = seen.get(row.area, 0)
        seen[row.area] = step + 1
        out[name] = palette.tint(area[row.area], step)
    return out


def sector_label(name: str) -> str:
    return SECTOR_LABELS.get(re.sub(r"[^a-z]", "", name.lower()), name)


DATE_AXIS: Final = altair.Axis(
    labelExpr=(
        "date(datum.value) != 1 ? timeFormat(datum.value, '%m/%d')"
        " : month(datum.value) != 0 ? timeFormat(datum.value, '%Y.%m')"
        " : timeFormat(datum.value, '%Y')"
    ),
    tickMinStep=86_400_000,
    labelOverlap=True,
    grid=False,
)


def time_axis(points: Sequence[Point]) -> altair.Axis:
    days = [str(point["day"]) for point in points]
    span = (
        date.fromisoformat(max(days)) - date.fromisoformat(min(days))
        if days
        else timedelta()
    )
    if span.days > YEARLY_TICKS_AFTER:
        return altair.Axis(format="%Y", tickCount="year", labelOverlap=True, grid=False)
    return altair.Axis(
        format="%Y.%m",
        tickCount=altair.TimeIntervalStep(interval="month", step=3),
        labelOverlap=True,
        grid=False,
    )


def axis_format(f: Fmt) -> str:
    return f",.{min(f.dec, 2)}~f"


def build_line_chart(  # noqa: PLR0913
    points: Sequence[Point],
    order: Sequence[str],
    *,
    fmt: Fmt,
    colors: Mapping[str, str] | None,
    step: bool,
    zero_line: bool,
    bands: Sequence[tuple[str, str]],
) -> altair.LayerChart:
    data = pd.DataFrame(list(points))
    many = len(order) > 1
    focus = altair.selection_point(fields=["series"], bind="legend")
    hover = altair.selection_point(
        fields=["day"], nearest=True, on="pointerover", clear="pointerout", empty=False
    )
    legend = (
        altair.Legend(orient="bottom", title=None, symbolType="stroke")
        if many
        else None
    )
    scale = (
        altair.Scale(domain=list(order), range=[colors[name] for name in order])
        if colors
        else altair.Scale(domain=list(order))
    )
    color = altair.Color("series:N", sort=list(order), scale=scale, legend=legend)
    x = altair.X(
        "day:T",
        title=None,
        axis=time_axis(points),
    )
    y = altair.Y(
        "value:Q",
        title=None,
        scale=altair.Scale(zero=False),
        axis=altair.Axis(format=axis_format(fmt), tickCount=5),
    )
    tooltip = [
        altair.Tooltip("day:T", title="날짜", format="%Y-%m-%d"),
        altair.Tooltip("series:N", title="시리즈"),
        altair.Tooltip("text:N", title="값"),
    ]
    longest = max(
        (sum(1 for p in points if p["series"] == n) for n in order), default=0
    )
    base = altair.Chart(data).encode(x=x, y=y, color=color)
    lines = base.mark_line(
        interpolate="step-after" if step else "linear",
        point=altair.OverlayMarkDef(size=18, filled=True)
        if longest <= POINT_LIMIT
        else False,
        strokeWidth=1.8,
    )
    if many:
        lines = lines.encode(
            opacity=altair.when(focus)
            .then(altair.value(1))
            .otherwise(altair.value(0.15))
        ).add_params(focus)
    dots = (
        base.mark_point(filled=True, size=70)
        .encode(
            opacity=altair.when(hover).then(altair.value(1)).otherwise(altair.value(0)),
            tooltip=tooltip,
        )
        .add_params(hover)
    )
    guide = palette.current().guide
    rule = (
        altair.Chart(data)
        .mark_rule(color=guide, strokeDash=[3, 3])
        .encode(x="day:T")
        .transform_filter(hover)
    )
    layers: list[altair.Chart] = []
    if bands and points:
        days = [str(point["day"]) for point in points]
        first, last = min(days), max(days)
        shaded = [
            {"start": max(start, first), "end": min(end, last)}
            for start, end in bands
            if end >= first and start <= last
        ]
        if shaded:
            layers.append(
                altair.Chart(altair.Data(values=shaded))
                .mark_rect(opacity=0.15, color=guide)
                .encode(x="start:T", x2="end:T")
            )
    if zero_line:
        layers.append(
            altair.Chart(altair.Data(values=[{"zero": 0}]))
            .mark_rule(strokeDash=[2, 3], color=guide)
            .encode(y="zero:Q")
        )
    return altair.LayerChart(layer=[*layers, lines, rule, dots])


def line_points(
    lines: Mapping[str, Line], formats: Mapping[str, Fmt], fallback: Fmt
) -> list[Point]:
    return [
        {
            "day": day.isoformat(),
            "value": value,
            "series": name,
            "text": display.fmt(value, formats.get(name, fallback)),
        }
        for name, points in lines.items()
        for day, value in thin(points)
    ]


def thin(points: Line) -> Line:
    step = -(-len(points) // MAX_POINTS)
    if step <= 1:
        return points
    kept = list(points[::-step][::-1])
    return kept if kept[0] == points[0] else [points[0], *kept]


def line_chart(  # noqa: PLR0913
    lines: Mapping[str, Line],
    *,
    fmt: Fmt,
    colors: Mapping[str, str] | None = None,
    height: int = CHART_HEIGHT,
    alt: str,
    step: bool = False,
    zero_line: bool = False,
    bands: Sequence[tuple[str, str]] = (),
    formats: Mapping[str, Fmt] | None = None,
) -> None:
    kept = {name: points for name, points in lines.items() if points}
    if not any(len(points) > 1 for points in kept.values()):
        empty()
        return
    chart = build_line_chart(
        line_points(kept, formats or {}, fmt),
        list(kept),
        fmt=fmt,
        colors=colors,
        step=step,
        zero_line=zero_line,
        bands=bands,
    )
    _ = st.altair_chart(chart, height=height, width="stretch", alt=alt)


def rebased(points: Line) -> Line:
    base = points[0][1]
    return [(day, value / base * 100) for day, value in points]


def rows_chart(  # noqa: PLR0913
    rows: Sequence[IndicatorRow],
    ctx: UiContext,
    months: int,
    *,
    alt: str,
    step: bool = False,
    height: int = CHART_HEIGHT,
    rebase: bool = False,
) -> None:
    shown = [row for row in rows if row.status == "ok"]
    if not shown:
        empty()
        return
    labels = names(shown)
    lines = {
        name: series_of(row, ctx, months)
        for name, row in zip(labels, shown, strict=True)
    }
    formats = {name: row.fmt for name, row in zip(labels, shown, strict=True)}
    if rebase:
        lines = {name: rebased(points) for name, points in lines.items()}
        formats = dict.fromkeys(labels, REBASED_FMT)
    line_chart(
        lines,
        fmt=formats[labels[0]],
        colors=colors_of(shown, labels),
        height=height,
        alt=alt,
        step=step,
        formats=formats,
    )


def compare_chart(
    rows: Sequence[IndicatorRow],
    ctx: UiContext,
    months: int,
    *,
    alt: str,
    step: bool = False,
) -> None:
    _ = st.caption(chart_title(rows, group_name(rows) or alt))
    rows_chart(rows, ctx, months, alt=alt, step=step)


@st.fragment
def single_chart(
    row: IndicatorRow, ctx: UiContext, *, key: str, default_range: str
) -> None:
    months = chart_header(chart_title([row]), key=f"{key}_range", default=default_range)
    if row.status != "ok" and row.message:
        _ = st.caption(f":red[{row.message}]")
        return
    rows_chart([row], ctx, months, alt=f"{row.label} 추이")


def bar_points(rows: Sequence[IndicatorRow]) -> tuple[list[Point], dict[str, str]]:
    shown = [row for row in rows if row.status == "ok" and row.value is not None]
    labels = names(shown)
    points: list[Point] = [
        {
            "name": name,
            "value": row.value or 0.0,
            "text": display.fmt(row.value, row.fmt),
            "date": display.short_date(row),
        }
        for name, row in zip(labels, shown, strict=True)
    ]
    return points, colors_of(shown, labels)


def bar_chart(
    rows: Sequence[IndicatorRow],
    *,
    key: str,
    alt: str,
    height: int = SMALL_CHART_HEIGHT,
) -> None:
    points, colors = bar_points(rows)
    if not points:
        empty()
        return
    order = list(colors)
    base = alt_chart(points).encode(
        y=altair.Y("name:N", sort=order, title=None, axis=altair.Axis(labelLimit=140)),
        x=altair.X(
            "value:Q",
            title=None,
            axis=altair.Axis(tickCount=4),
            scale=altair.Scale(domain=bar_domain(points), nice=False),
        ),
        tooltip=[
            altair.Tooltip("name:N", title="국가"),
            altair.Tooltip("text:N", title="값"),
            altair.Tooltip("date:N", title="기준"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, opacity=0.8).encode(
        color=altair.Color(
            "name:N",
            sort=order,
            scale=altair.Scale(domain=order, range=list(colors.values())),
            legend=None,
        )
    )
    labels = base.mark_text(
        align=altair.expr("datum.value < 0 ? 'right' : 'left'"),
        dx=altair.expr("datum.value < 0 ? -4 : 4"),
        fontSize=11,
        color=palette.current().muted,
    ).encode(text="text:N")
    _ = st.altair_chart(
        altair.LayerChart(layer=[bars, labels]),
        height=max(height, 30 * len(points)),
        width="stretch",
        alt=alt,
        key=key,
    )


def bar_domain(points: Sequence[Point]) -> list[float]:
    values = [float(point["value"]) for point in points]
    low, high = min(0.0, *values), max(0.0, *values)
    room = (high - low) * BAR_ROOM
    return [low - room if low < 0 else 0.0, high + room if high > 0 else 0.0]


def alt_chart(points: Sequence[Point]) -> altair.Chart:
    return altair.Chart(altair.Data(values=list(points)))


def curve_chart(curve: Curve, *, height: int = CHART_HEIGHT) -> None:
    labels = [f"{line.label} {line.date[2:]}" for line in curve.lines]
    points: list[Point] = [
        {"tenor": tenor, "value": value, "line": name, "text": f"{value:.2f}%"}
        for line, name in zip(curve.lines, labels, strict=True)
        for tenor, value in zip(curve.tenors, line.values, strict=False)
        if value is not None
    ]
    if not points:
        empty()
        return
    color = palette.current().area[curve.area]
    chart = (
        alt_chart(points)
        .mark_line(point=altair.OverlayMarkDef(size=36, filled=True), strokeWidth=1.8)
        .encode(
            x=altair.X(
                "tenor:O",
                sort=list(curve.tenors),
                title=None,
                axis=altair.Axis(labelAngle=0),
            ),
            y=altair.Y(
                "value:Q",
                title=None,
                scale=altair.Scale(zero=False),
                axis=altair.Axis(format=",.2~f"),
            ),
            color=altair.Color(
                "line:N",
                sort=labels,
                scale=altair.Scale(
                    domain=labels,
                    range=[palette.tint(color, i) for i in range(len(labels))],
                ),
                legend=altair.Legend(orient="bottom", title=None, symbolType="stroke"),
            ),
            strokeDash=altair.StrokeDash(
                "line:N",
                sort=labels,
                scale=altair.Scale(
                    domain=labels,
                    range=[[1, 0], [6, 3], [2, 2], [1, 3]][: len(labels)],
                ),
                legend=None,
            ),
            tooltip=[
                altair.Tooltip("line:N", title="곡선"),
                altair.Tooltip("tenor:O", title="만기"),
                altair.Tooltip("text:N", title="수익률"),
            ],
        )
    )
    _ = st.altair_chart(
        chart, height=height, width="stretch", alt=f"{curve.label} 수익률 곡선"
    )


def arrow_style(text: str) -> str:
    arrow = display.arrow_dir(text)
    return f"color: {palette.current().signed(arrow == 'up')}" if arrow else ""


def clear_selection(key: str) -> None:
    st.session_state[key] = {"selection": {"rows": [], "columns": [], "cells": []}}


def open_detail(row: IndicatorRow, ctx: UiContext, key: str) -> None:
    title = metric_label(row, area_prefix=True)
    shown = st.dialog(title, width="large", on_dismiss=partial(clear_selection, key))(
        detail
    )
    shown(row, ctx)


@st.fragment
def table(rows: Sequence[IndicatorRow], ctx: UiContext, *, key: str) -> None:
    if not rows:
        empty()
        return
    name = f"table_{key}"
    frame = pd.DataFrame(display.panel_table(rows))
    event = st.dataframe(  # pyright: ignore[reportUnknownMemberType]
        frame.style.map(arrow_style, subset=["변화"]),  # pyright: ignore[reportUnknownMemberType]
        hide_index=True,
        height="content",
        width="stretch",
        column_config={
            "지표": st.column_config.TextColumn(width="medium", help=TABLE_HELP),
            "국가": st.column_config.TextColumn(width="small"),
            "변화": st.column_config.TextColumn(help=CHANGE_HELP),
            "추이": st.column_config.LineChartColumn(width="small", color="gray"),
        },
        alt="지표 목록 및 상세 보기",
        on_select="rerun",
        selection_mode="single-cell",
        key=name,
    )
    for index, _ in event.selection.cells:
        if 0 <= index < len(rows):
            open_detail(rows[index], ctx, name)
            return


def detail(row: IndicatorRow, ctx: UiContext) -> None:
    latest, prev, when = st.columns(3, gap="small")
    with latest:
        metric(row, label="최신", trend=False)
    with prev:
        _ = st.metric(
            "이전",
            display.fmt(row.prev, row.fmt),
            help=PREV_HELP,
            border=True,
            height="stretch",
        )
    with when:
        _ = st.metric(
            "기준", display.short_date(row) or "—", border=True, height="stretch"
        )
    with st.container(border=True):
        single_chart(row, ctx, key="detail", default_range="10Y")
        if row.status == "ok" and row.message:
            _ = st.caption(f":orange[{row.message}]")


def problems(p: Panel) -> None:
    for text in display.panel_problems(p):
        _ = st.caption(f":red[{text}]")
    for text in display.panel_stale(p):
        _ = st.caption(f":orange[{text}]")


def empty(text: str = NO_DATA) -> None:
    _ = st.caption(f":material/info: {text}")


def unit_text(f: Fmt) -> str:
    return display.unit(f).strip()


def base_label(row: IndicatorRow) -> str:
    text = PARENS.sub("", row.label).strip()
    return AREA_WORDS.sub("", text).strip() or row.label


def group_name(rows: Sequence[IndicatorRow]) -> str:
    if len(rows) == 1:
        return rows[0].label
    heads = {base_label(row) for row in rows}
    return heads.pop() if len(heads) == 1 else ""


def chart_title(rows: Sequence[IndicatorRow], name: str | None = None) -> str:
    if not rows:
        return name or ""
    unit = unit_text(rows[0].fmt)
    title = name or group_name(rows)
    return " · ".join(part for part in (title, f"단위 {unit}" if unit else "") if part)


def magnitude(row: IndicatorRow, floor: float) -> float:
    values = [abs(value) for _, value in row.history]
    if not values:
        values = [abs(row.value) if row.value is not None else 0.0]
    return max(median(values), floor)


def scale_split(
    rows: Sequence[IndicatorRow], *, floor: float, gap: float
) -> list[list[IndicatorRow]]:
    groups: list[list[IndicatorRow]] = []
    for row in sorted(rows, key=lambda row: magnitude(row, floor)):
        last = groups[-1] if groups else None
        if last is not None and magnitude(row, floor) <= gap * magnitude(
            last[0], floor
        ):
            last.append(row)
        else:
            groups.append([row])
    return groups


def spans(rows: Sequence[IndicatorRow]) -> int:
    return len({row.area for row in rows})


def charted(rows: Sequence[IndicatorRow]) -> list[IndicatorRow]:
    return [row for row in rows if row.status == "ok" and row.history]


def capped(group: Sequence[IndicatorRow]) -> list[list[IndicatorRow]]:
    parts: list[list[IndicatorRow]] = []
    for row in group:
        same = sum(1 for other in parts[-1] if other.area == row.area) if parts else 0
        if not parts or same == len(palette.TINTS):
            parts.append([])
        parts[-1].append(row)
    return parts


def chart_groups(
    rows: Sequence[IndicatorRow], lead: Sequence[IndicatorRow], *, rebase: bool
) -> list[list[IndicatorRow]]:
    if rebase:
        return capped(charted(rows))
    first = charted(lead)
    if len(first) < 2 or len({unit_text(row.fmt) for row in first}) > 1:  # noqa: PLR2004
        first = []
    taken = {row.id for row in first}
    rest = rest_groups([row for row in charted(rows) if row.id not in taken])
    return [first, *rest] if first else rest


def rest_groups(shown: Sequence[IndicatorRow]) -> list[list[IndicatorRow]]:
    by_label: dict[str, list[IndicatorRow]] = {}
    for row in shown:
        by_label.setdefault(base_label(row), []).append(row)
    same_unit = [
        same
        for same in by_label.values()
        if len({unit_text(row.fmt) for row in same}) == 1
    ]
    groups = [same for same in same_unit if spans(same) >= WIDE_AREAS]
    percent: dict[str, list[IndicatorRow]] = {}
    levels: dict[tuple[Area, str], list[IndicatorRow]] = {}
    groups.extend(
        same
        for same in same_unit
        if 1 < spans(same) < WIDE_AREAS and unit_text(same[0].fmt) not in PERCENT_UNITS
    )
    grouped = {row.id for group in groups for row in group}
    for row in shown:
        if row.id in grouped:
            continue
        unit = unit_text(row.fmt)
        if unit in PERCENT_UNITS:
            percent.setdefault(unit, []).append(row)
        else:
            levels.setdefault((row.area, unit), []).append(row)
    for members in percent.values():
        groups.extend(scale_split(members, floor=PERCENT_FLOOR, gap=PERCENT_GAP))
    for members in levels.values():
        groups.extend(scale_split(members, floor=0.0, gap=LEVEL_GAP))
    position = {row.id: i for i, row in enumerate(shown)}
    for group in groups:
        group.sort(key=lambda row: position[row.id])
    groups.sort(key=lambda group: position[group[0].id])
    return [part for group in groups for part in capped(group)]


def is_wide(group: Sequence[IndicatorRow]) -> bool:
    return spans(group) >= WIDE_AREAS


@dataclass(frozen=True, slots=True)
class ChartSpec:
    ctx: UiContext
    months: int
    key: str
    name: str
    rebase: bool


def group_chart(group: Sequence[IndicatorRow], spec: ChartSpec, height: int) -> None:
    title = (
        (group[0].label if len(group) == 1 else "")
        if spec.rebase
        else chart_title(group, group_name(group) or spec.name)
    )
    _ = st.caption(title or " ")
    rows_chart(
        group,
        spec.ctx,
        spec.months,
        alt=title or "추이 비교",
        height=height,
        rebase=spec.rebase,
    )


def wide_chart(group: Sequence[IndicatorRow], spec: ChartSpec) -> None:
    chart_col, bar_col = st.columns([3, 1], gap="medium")
    with chart_col:
        group_chart(group, spec, CHART_HEIGHT)
    with bar_col:
        _ = st.caption("최신값", help=LATEST_HELP)
        bar_chart(
            group,
            key=f"{spec.key}_{group[0].id}_latest",
            alt=f"{chart_title(group, spec.name)} 최신값 비교",
            height=CHART_HEIGHT,
        )


def grid(groups: Sequence[Sequence[IndicatorRow]], spec: ChartSpec) -> None:
    if len(groups) == 1:
        group_chart(groups[0], spec, CHART_HEIGHT)
        return
    for start in range(0, len(groups), 2):
        cols = st.columns(2, gap="medium")
        for col, group in zip(cols, groups[start : start + 2], strict=False):
            with col:
                group_chart(group, spec, SMALL_CHART_HEIGHT)


def charts(
    rows: Sequence[IndicatorRow],
    spec: ChartSpec,
    lead: Sequence[IndicatorRow],
) -> None:
    groups = chart_groups(rows, lead, rebase=spec.rebase)
    if not groups:
        empty()
        return
    if spec.rebase:
        _ = st.caption(REBASED_TITLE)
    pending: list[list[IndicatorRow]] = []
    for group in groups:
        if is_wide(group):
            if pending:
                grid(pending, spec)
                pending = []
            wide_chart(group, spec)
        else:
            pending.append(group)
    if pending:
        grid(pending, spec)


@st.fragment
def table_expander(rows: Sequence[IndicatorRow], ctx: UiContext, *, key: str) -> None:
    if len(rows) <= 2:  # noqa: PLR2004
        return
    exp = st.expander(
        "표",
        expanded=False,
        icon=":material/table_rows:",
        key=f"exp_{key}",
        on_change="rerun",
    )
    if exp.open:
        with exp:
            table(rows, ctx, key=key)


@contextmanager
def card(
    p: Panel | None, *, title: str | None = None, stretch: bool = False
) -> Generator[None]:
    section(
        title or (p.title if p else ""),
        help=p.note if p else None,
        anchor=p.id if p else None,
    )
    with st.container(border=True, height="stretch" if stretch else "content"):
        if p is None:
            empty()
        yield
        if p is not None:
            problems(p)


def standard(  # noqa: PLR0913
    p: Panel,
    ctx: UiContext,
    months: int,
    rows: Sequence[IndicatorRow] | None = None,
    *,
    key: str | None = None,
    lead: Sequence[IndicatorRow] = (),
    rebase: bool = False,
) -> None:
    shown = list(p.rows if rows is None else rows)
    name = key or p.id
    if not shown:
        empty()
        return
    metrics(shown, key=name, area_prefix=len({row.area for row in shown}) > 1)
    charts(shown, ChartSpec(ctx, months, name, p.title, rebase), lead)
    table_expander(shown, ctx, key=name)
