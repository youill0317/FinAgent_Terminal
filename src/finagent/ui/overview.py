"""거시 국면 요약, 삼의 법칙, 수익률 곡선 및 FOMC 경제전망(SEP) 종합 화면."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import altair as alt
import streamlit as st

from finagent import catalog, display
from finagent.dashboard import SAHM_TRIGGER
from finagent.models import AREA_LABEL, Fmt
from finagent.ui import components, palette
from finagent.ui.guides import GUIDES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finagent.models import (
        Dashboard,
        IndicatorRow,
        Panel,
        Projections,
    )
    from finagent.ui.context import UiContext

PAGE: Final = "overview"
SPREAD_LABEL: Final = "10Y−2Y"
SPREAD_FMT: Final = Fmt(suf="%p", dec=2, signed=True)
MIN_POINTS: Final = 2
REGIME_HELP_TEXT: Final = (
    "[정의] 근원 PCE, 삼의 법칙 지표, 장단기 금리차, VIX 기반 거시 국면 요약"
    "\n\n[해석] ● 양호, ◆ 주의, ■ 위험, ○ 중립"
    "\n\n[산출] 최신 발표치 및 국면 임계치 비교"
)
TAPE_HELP_TEXT: Final = (
    "[정의] 주요 주가지수·변동성·환율 최신 시세 요약"
    "\n\n[해석] 상승 빨강, 하락 파랑"
    "\n\n[산출] 직전 관측치 대비 변화(주가지수·환율 등락률 %, 변동성 변동폭)"
)
SPREAD_HELP: Final = (
    "[정의] 미 국채 10년물·2년물 수익률 스프레드 및 NBER 경기침체 기간(음영)"
    "\n\n[해석] 0%p 하회 시 장단기 금리 역전"
    "\n\n[산출] 10년 만기 국채 금리 − 2년 만기 국채 금리 (%p)"
)
SAHM_HELP: Final = (
    "[정의] 미국 실업률 기반 경기침체 조기 경보 지표"
    f"\n\n[해석] 침체 신호 임계치 {SAHM_TRIGGER:.2f}%p (점선)"
    "\n\n[산출] 최근 3개월 평균 실업률 − 직전 12개월 최저 3개월 평균 실업률"
)
CURVE_HELP: Final = (
    "[정의] 만기별 국채 수익률 곡선 (유로존: AAA 등급 국채 기준)"
    "\n\n[해석] 우상향 정상, 단기 금리 상회 시 장단기 역전"
    "\n\n[산출] 최신·1개월 전·1년 전 만기별 수익률 비교"
)
SEP_HELP: Final = (
    "[정의] 연준 FOMC 위원 경제전망(SEP) 중앙값 (연 4회 발표)"
    "\n\n[해석] 연도 열 당해 말 전망치, 장기 열 균형 수준, 화살표 직전 전망 대비 변동폭"
    "\n\n[산출] 현재 열 최근 실적치(성장률 최근 분기, 물가 최근 월 전년비)"
)
POLICY_CHANGE_HELP: Final = (
    "[정의] 주요국 중앙은행 최근 정책금리 조정 내역"
    "\n\n[해석] 인상(빨강)·인하(파랑) 방향 및 통화정책 기조"
    "\n\n[산출] 최근 금리 변경일, 조정폭(bp), 최근 12개월 변경 횟수"
)


@st.fragment
def render(d: Dashboard, ctx: UiContext) -> None:
    _regime(d)
    _tape(d)
    months = components.page_range(PAGE)
    left, right = st.columns(2)
    with left:
        _growth(d, ctx, months)
        _inflation(d, ctx, months)
        _unemployment(d, ctx, months)
    with right:
        _projections(d.projections)
        _rates(d, months)
        _commodities(d, ctx, months)
    _policy(d, ctx, months)


def _ok(rows: Sequence[IndicatorRow]) -> list[IndicatorRow]:
    return [row for row in rows if row.status == "ok"]


def _title(p: Panel | None, fallback: str) -> str:
    return p.title if p is not None else fallback


def _regime(d: Dashboard) -> None:
    components.section("국면 요약", help=REGIME_HELP_TEXT, anchor="regime")
    if not d.regime:
        components.empty("국면 판정 데이터 없음")
        return
    rows = display.rows_by_id(d)
    for col, cell in zip(st.columns(len(d.regime)), d.regime, strict=True):
        color, icon = display.TONE[cell.tone]
        label, value = display.regime_card(cell)
        _ = col.metric(
            label,
            value,
            f"{icon} {cell.tag}",
            delta_color=color,
            delta_arrow="off",
            help=GUIDES[cell.source],
            border=True,
            height="stretch",
            chart_data=display.trend(rows.get(cell.source)),
        )


def _tape(d: Dashboard) -> None:
    components.section("시장 요약", help=TAPE_HELP_TEXT, anchor="tape")
    rows = display.rows_by_id(d)
    groups = [
        (title, [rows[i] for i in ids if i in rows])
        for title, ids in catalog.TAPE_GROUPS
    ]
    groups = [(title, found) for title, found in groups if found]
    if not groups:
        components.empty("시장 요약 데이터 없음")
        return
    head, *rest = groups
    _tape_group(*head)
    if rest:
        for col, group in zip(st.columns(len(rest)), rest, strict=True):
            with col:
                _tape_group(*group)


def _tape_group(title: str, rows: Sequence[IndicatorRow]) -> None:
    with st.container(gap="small"):
        _ = st.caption(f"**{title}**")
        for col, row in zip(st.columns(len(rows), gap="small"), rows, strict=True):
            with col:
                components.metric(row)


def _growth(d: Dashboard, ctx: UiContext, months: int) -> None:
    p = components.panel(d, "ov_growth")
    with components.card(p, title=_title(p, "경제 성장"), stretch=True):
        if p is None:
            return
        components.area_metrics(
            [row for row in p.rows if row.freq == "Q"], key="ov_growth"
        )
        cli = _ok([row for row in p.rows if row.freq == "M"])
        if cli:
            components.compare_chart(
                cli, ctx, months, alt="미국·한국·일본·중국 OECD 경기선행지수 추이"
            )
        else:
            components.empty("경기선행지수 데이터 없음")


def _spread(d: Dashboard, months: int) -> None:
    spread = d.spread
    if spread is None or len(spread.points) < MIN_POINTS:
        components.empty("10년물 − 2년물 금리차 데이터 없음")
        return
    _ = st.caption("미 국채 10년물 − 2년물 · 단위 %p", help=SPREAD_HELP)
    points = spread.points[-months:]
    components.line_chart(
        {SPREAD_LABEL: [(display.label_day(month), value) for month, value in points]},
        fmt=SPREAD_FMT,
        colors={SPREAD_LABEL: palette.current().area["US"]},
        alt="미 국채 10년물 − 2년물 금리차와 NBER 경기침체 기간",
        zero_line=True,
        bands=[(f"{start}-01", f"{end}-01") for start, end in spread.recessions],
        height=components.SMALL_CHART_HEIGHT,
    )


def _inflation(d: Dashboard, ctx: UiContext, months: int) -> None:
    p = components.panel(d, "ov_inflation")
    with components.card(p, title=_title(p, "인플레이션"), stretch=True):
        if p is None:
            return
        cpi = [row for row in p.rows if row.id != "us_core_pce"]
        components.area_metrics(cpi, key="ov_inflation")
        prices = _ok(cpi)
        if prices:
            components.compare_chart(
                prices,
                ctx,
                months,
                alt="미국·한국·유로존·일본·중국 CPI 전년 대비 상승률 추이",
            )


def _unemployment(d: Dashboard, ctx: UiContext, months: int) -> None:
    p = components.panel(d, "ov_unemployment")
    with components.card(p, title=_title(p, "실업률"), stretch=True):
        if p is None:
            return
        components.area_metrics(
            [row for row in p.rows if row.id != "us_sahm"], key="ov_unemployment"
        )
        sahm = next((row for row in _ok(p.rows) if row.id == "us_sahm"), None)
        if sahm is not None:
            _sahm_chart(sahm, ctx, months)


def _sahm_chart(row: IndicatorRow, ctx: UiContext, months: int) -> None:
    _ = st.caption("삼의 법칙 지표 · 단위 %p", help=SAHM_HELP)
    points = components.series_of(row, ctx, months)
    if len(points) < MIN_POINTS:
        components.empty("삼의 법칙 지표 시계열 데이터 없음")
        return
    values = [{"day": day.isoformat(), "value": value} for day, value in points]
    line = (
        alt.Chart(alt.Data(values=values))
        .mark_line(color=palette.current().area["US"], strokeWidth=1.8)
        .encode(
            x=alt.X("day:T", title=None, axis=components.time_axis(values)),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format=",.2~f")),
            tooltip=[
                alt.Tooltip("day:T", title="날짜", format="%Y-%m"),
                alt.Tooltip("value:Q", title="삼의 법칙", format=".2f"),
            ],
        )
    )
    threshold = (
        alt.Chart(alt.Data(values=[{"y": SAHM_TRIGGER}]))
        .mark_rule(strokeDash=[4, 3], color=palette.current().up)
        .encode(y="y:Q")
    )
    _ = st.altair_chart(
        alt.layer(line, threshold),
        height=components.SMALL_CHART_HEIGHT,
        width="stretch",
        alt=f"삼의 법칙 지표 추이와 {SAHM_TRIGGER:.2f}%p 침체 신호 기준선",
    )


def _rates(d: Dashboard, months: int) -> None:
    p = components.panel(d, "ov_rates")
    with components.card(p, title=_title(p, "금리"), stretch=True):
        if p is not None:
            components.metrics(p.rows, key="ov_rates")
        _spread(d, months)
        _curve(d)


def _curve(d: Dashboard) -> None:
    curves = {AREA_LABEL[curve.area]: curve for curve in d.curves if curve.lines}
    if not curves:
        components.empty("수익률 곡선 데이터 없음")
        return
    with st.container(
        horizontal=True,
        vertical_alignment="center",
        horizontal_alignment="distribute",
    ):
        _ = st.caption("국채 수익률 곡선 · 단위 %", help=CURVE_HELP)
        picked = st.segmented_control(
            "곡선 국가",
            list(curves),
            default=next(iter(curves)),
            required=True,
            key="ov_curve_area",
            label_visibility="collapsed",
        )
    components.curve_chart(curves[picked], height=components.SMALL_CHART_HEIGHT)


def _projections(p: Projections | None) -> None:
    components.section("FOMC 경제전망 · 위원 중앙값", help=SEP_HELP, anchor="sep")
    with st.container(border=True, height="stretch"):
        if p is None:
            components.empty("FOMC 경제전망 데이터 없음")
            return
        if not p.rows:
            msg = (
                f"FOMC 경제전망 수신 실패 · {p.message}"
                if p.message
                else "FOMC 경제전망 수신 실패"
            )
            components.empty(msg)
            return
        _ = st.table(  # pyright: ignore[reportUnknownMemberType]
            _sep_cells(p),
            border="horizontal",
            hide_index=True,
            alt="FOMC 경제전망 위원 중앙값과 직전 발표 대비 변화",
        )
        prior = f" · 직전 {p.prior_published}" if p.prior_published else ""
        _ = st.caption(f"발표 {p.published}{prior}")
        if p.message:
            _ = st.caption(f":orange[일부 항목 수신 실패 · {p.message}]")


def _sep_cells(p: Projections) -> dict[str, list[str]]:
    data = display.sep_table(p)
    change_keys = {display.sep_change_key(name) for name in data}
    cells = {name: values for name, values in data.items() if name not in change_keys}
    for name, values in cells.items():
        changes = data.get(display.sep_change_key(name))
        if changes is not None:
            cells[name] = [
                _with_change(value, change)
                for value, change in zip(values, changes, strict=True)
            ]
    return cells


def _with_change(value: str, change: str) -> str:
    if not change:
        return value
    color = "red" if "▲" in change else "blue" if "▼" in change else "gray"
    return f"{value} :{color}[{change}]"


def _commodities(d: Dashboard, ctx: UiContext, months: int) -> None:
    p = components.panel(d, "ov_commodities")
    with components.card(p, title=_title(p, "원자재"), stretch=True):
        if p is None:
            return
        for row in p.rows:
            _ = st.caption(components.chart_title([row]))
            components.rows_chart([row], ctx, months, alt=f"{row.label} 추이")


def _policy(d: Dashboard, ctx: UiContext, months: int) -> None:
    p = components.panel(d, "ov_policy")
    with components.card(p, title=_title(p, "정책 변화")):
        _ = st.caption("최근 정책금리 변경", help=POLICY_CHANGE_HELP)
        rates = _ok([row for row in p.rows if row.source == "BIS"]) if p else []
        if d.policy_changes:
            components.policy_cards(d.policy_changes, rates)
        else:
            components.empty("정책금리 변경 데이터 없음")
        if rates:
            components.compare_chart(
                rates,
                ctx,
                months,
                alt="미국·유로존·일본·중국·한국 정책금리 추이",
                step=True,
            )
        else:
            components.empty("정책금리 데이터 없음")
