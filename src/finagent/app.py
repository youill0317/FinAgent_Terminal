"""The dashboard app. ``macro`` starts it with Streamlit on 127.0.0.1.

Script arguments (after ``--``) are the CLI's ``--demo``, ``--env-file`` and
``--settings``. The services live in one shared runtime, so every browser tab
sees the same view, snapshot and analysis lock.
"""

# Session state is untyped, element calls return a DeltaGenerator the page
# never uses, and page prose is split across lines.
# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false, reportUnknownMemberType=false

from __future__ import annotations

import dataclasses
import math
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent import catalog, display, investors, perf, portfolio, quant, screen
from finagent.analysis.presets import PRESETS
from finagent.analysis.providers.base import ProviderError
from finagent.analysis.service import AnalysisService, RunInput
from finagent.cli import build_service, parse_args, resolve_env, resolve_settings
from finagent.fetch import SourceError
from finagent.guides import GUIDES
from finagent.runtime import Run, Runtime
from finagent.sources import EcosSeries, FredSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import date

    import numpy as np
    from numpy.typing import NDArray

    from finagent.models import Curve, Dashboard, IndicatorRow, Panel, Spread
    from finagent.sources import Fetch
    from finagent.sources import Series as Points

CLOCKS = (("NY", "America/New_York"), ("SEOUL", "Asia/Seoul"), ("UTC", "UTC"))
SPREAD_RANGES = {"2Y": 24, "5Y": 60, "전체": 0}
CHART_HEIGHT = 240
DETAIL_POINT_LIMIT = 60
SCOPES = {"view": "지금 보기", "all": "전체 매크로"}
# The floating AI button and the panel above it; {bg} is the page background.
AI_CSS = """<style>
.st-key-ai_fab {{position: fixed; right: 1.5rem; bottom: 1.5rem; width: auto;
  z-index: 1000}}
.st-key-ai_fab button {{width: 3.25rem; height: 3.25rem; padding: 0;
  border-radius: 50%}}
.st-key-ai_panel {{position: fixed; right: 1.5rem; bottom: 5.5rem; z-index: 1000;
  width: min(440px, 100vw - 3rem); max-height: 70vh; overflow: hidden;
  display: flex; flex-direction: column;
  background: {bg}; box-shadow: 0 0.5rem 1.5rem rgb(0 0 0 / 0.25)}}
/* The conversation scrolls alone. column-reverse keeps it at the newest
   message; its single child holds the turns in their normal order. */
.st-key-ai_panel > div:has(.st-key-ai_turns), .st-key-ai_panel > .st-key-ai_turns
  {{flex: 1 1 auto; min-height: 0; overflow-y: auto; display: flex;
  flex-direction: column-reverse}}
</style>"""


@st.cache_resource(show_spinner=False)
def runtime(demo: bool, env_file: str, settings: str) -> Runtime:
    env = resolve_env(Path(env_file) if env_file else None, os.environ)
    settings_path = resolve_settings(Path(settings) if settings else None)
    service = build_service(demo=demo, env=env, settings_path=settings_path)
    return Runtime(service, AnalysisService(service, env, settings_path=settings_path))


st.set_page_config(
    page_title="FinAgent", page_icon=":material/monitoring:", layout="wide"
)
args = parse_args(sys.argv[1:])
rt = runtime(args.demo, str(args.env_file or ""), str(args.settings or ""))
state = st.session_state
state.setdefault("toast", "")
state.setdefault("ai_open", False)
state.setdefault("ai_log", display.AnalysisLog())
state.setdefault("ai_run", None)
state.setdefault("ai_error", "")


def conversation() -> display.AnalysisLog:
    """The log in session state; read it each time, a new chat replaces it."""
    return state.ai_log


# ---------------------------------------------------------------- actions


def refresh() -> None:
    d = rt.call(rt.dashboard.get(refresh=True))
    ok = sum(source.ok for source in d.sources)
    total = sum(source.total for source in d.sources)
    state.toast = f"{ok}/{total}개 시리즈 갱신 · {d.elapsed_seconds:.1f}s"
    log = conversation()
    if log.conversation:
        log.stale = True


def set_deep(deep: frozenset[str]) -> None:
    if deep == rt.dashboard.view:
        return
    d = rt.call(rt.dashboard.set_view(deep))
    state.toast = f"보기 {display.view_label(d)} · 지표 {display.row_count(d)}개 표시"


def toggle_deep(view: str) -> None:
    set_deep(rt.dashboard.view ^ {view})


def toggle_ai() -> None:
    state.ai_open = not state.ai_open


# ---------------------------------------------------------------- settings


def check_all(d: Dashboard, *, on: bool) -> None:
    for view in d.views:
        if view.extra:
            state[f"view_{view.id}"] = on


def ai_settings() -> None:
    """Provider, model and scope for the AI analysis; saved by 적용."""
    providers = rt.analysis.providers
    found = rt.call(rt.analysis.statuses())
    with st.expander("AI 분석", icon=":material/smart_toy:"):
        provider_id = st.selectbox(
            "공급자",
            list(providers),
            index=list(providers).index(rt.analysis.choice.provider),
            format_func=lambda pid: (
                f"{providers[pid].title} · "
                f"{display.STATUS_TEXT.get(found[pid].kind, found[pid].kind)}"
            ),
            key="set_provider",
        )
        _ = st.text_input(
            "모델",
            value=rt.analysis.model_for(provider_id),
            placeholder=providers[provider_id].default_model,
            key=f"set_model_{provider_id}",
        )
        _ = st.segmented_control(
            "새 대화의 범위",
            list(SCOPES),
            format_func=SCOPES.__getitem__,
            default=rt.analysis.choice.scope,
            key="set_scope",
            help="진행 중인 대화는 시작할 때의 범위를 유지합니다",
        )


def apply_ai() -> None:
    provider_id: str = state.set_provider
    model: str = state.get(f"set_model_{provider_id}", "")
    try:
        _ = rt.analysis.choose(
            provider_id,
            model or rt.analysis.model_for(provider_id),
            state.get("set_scope") or "",
        )
    except (ProviderError, OSError) as error:
        state.ai_error = f"선택을 저장하지 못했습니다: {error}"
    else:
        state.ai_error = ""


@st.dialog("보기 설정", width="large")
def settings_dialog(d: Dashboard) -> None:
    st.caption("깊게 볼 패널만 켜세요. 켠 패널의 나머지 지표만 새로 받아옵니다.")
    with st.container(horizontal=True):
        st.button("핵심만", on_click=check_all, args=(d,), kwargs={"on": False})
        st.button("전체 매크로", on_click=check_all, args=(d,), kwargs={"on": True})
    sections = [("개요", ":material/dashboard:", ("curve",))] + [
        (title, icon, ids) for _, title, icon, ids in display.PANEL_PAGES
    ]
    for title, icon, ids in sections:
        views = [view for view in d.views if view.id in ids]
        switchable = [view for view in views if view.extra]
        count = sum(bool(state.get(f"view_{view.id}")) for view in switchable)
        with st.expander(f"{title} · {count}/{len(switchable)}", icon=icon):
            for view in views:
                note = display.view_note(
                    view.core, view.extra, curve=view.id == "curve"
                )
                st.checkbox(
                    f"{view.title} :gray[{note}]",
                    key=f"view_{view.id}",
                    disabled=not view.extra,
                )
    ai_settings()
    on = frozenset(view.id for view in d.views if state.get(f"view_{view.id}"))
    st.caption(display.view_summary(d, on))
    st.caption(
        f"설정은 {d.settings_file} 파일에 저장됩니다."
        if d.settings_file
        else "설정은 앱이 켜져 있는 동안만 유지됩니다."
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("취소"):
            st.rerun()
        if st.button("적용", type="primary"):
            set_deep(on)
            apply_ai()
            st.rerun()


# ---------------------------------------------------------------- charts


def curve_chart(curve: Curve) -> alt.LayerChart:
    names = [f"{line.label} {line.date[2:]}" for line in curve.lines]
    rows = [
        {"tenor": tenor, "value": value, "line": name}
        for line, name in zip(curve.lines, names, strict=True)
        for tenor, value in zip(curve.tenors, line.values, strict=True)
        if value is not None
    ]
    lines = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("tenor:O", sort=list(curve.tenors), title=None),
            y=alt.Y("value:Q", title="%", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "line:N", sort=names, title=None, legend=alt.Legend(orient="bottom")
            ),
            tooltip=[
                alt.Tooltip("line:N", title="곡선"),
                alt.Tooltip("tenor:O", title="만기"),
                alt.Tooltip("value:Q", title="%", format=".2f"),
            ],
        )
    )
    return alt.LayerChart(layer=[lines])


def spread_chart(spread: Spread, months: int) -> alt.LayerChart:
    points = spread.points[-months:] if months else spread.points
    first, last = points[0][0], points[-1][0]
    rows = [{"month": f"{month}-01", "value": value} for month, value in points]
    bands = [
        {"start": f"{max(start, first)}-01", "end": f"{min(end, last)}-01"}
        for start, end in spread.recessions
        if end >= first and start <= last
    ]
    area = (
        alt.Chart(alt.Data(values=rows))
        .mark_area(line=True, opacity=0.25)
        .encode(
            x=alt.X("month:T", title=None),
            y=alt.Y("value:Q", title="%p"),
            tooltip=[
                alt.Tooltip("month:T", title="월", format="%Y-%m"),
                alt.Tooltip("value:Q", title="10Y−2Y %p", format="+.2f"),
            ],
        )
    )
    zero = (
        alt.Chart(alt.Data(values=[{"y": 0}]))
        .mark_rule(strokeDash=[2, 3], color="gray")
        .encode(y="y:Q")
    )
    if not bands:
        return alt.LayerChart(layer=[area, zero])
    recessions = (
        alt.Chart(alt.Data(values=bands))
        .mark_rect(opacity=0.15, color="gray")
        .encode(x="start:T", x2="end:T")
    )
    return alt.LayerChart(layer=[recessions, area, zero])


# ---------------------------------------------------------------- sections


@st.fragment(run_every=30)
def clocks(generated_at: str) -> None:
    now = datetime.now().astimezone()
    updated = datetime.fromisoformat(generated_at).astimezone()
    with st.container(horizontal=True, gap="xsmall", vertical_alignment="center"):
        for i, (label, zone) in enumerate(CLOCKS):
            st.badge(
                f"{label} {now.astimezone(ZoneInfo(zone)):%H:%M}",
                icon=":material/schedule:" if i == 0 else None,
                color="gray",
            )
        st.badge(f"갱신 {updated:%H:%M}", color="gray", help="데이터를 받은 시각")


def status(d: Dashboard) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        for source in d.sources:
            st.badge(
                f"{source.name} {source.ok}/{source.total}",
                color=display.SOURCE_COLOR[source.state],
                help=source.message,
            )


def notices(d: Dashboard) -> None:
    if d.settings_error:
        st.error(d.settings_error)
    if d.demo:
        st.info(
            "데모 모드: 수치는 모두 예시 값입니다. 실제 데이터는 `macro`를 키와"
            " 함께 실행하세요."
        )
    for source in d.sources:
        help_ = display.KEY_HELP.get(source.id)
        if source.state == "missing_key" and help_:
            env, url = help_
            st.warning(
                f"{source.name} 키가 없어 {source.total}개 시리즈를 받지 못했습니다."
                f" 무료 키를 [여기서]({url}) 발급받아 `{env}` 환경변수나 `.env`"
                " 파일에 넣은 뒤 다시 실행하세요."
            )
        elif source.state in {"error", "partial"}:
            st.error(
                f"{source.name}: {source.total - source.ok}개 시리즈를 받지"
                f" 못했습니다. {source.message or ''}"
            )


def tape(d: Dashboard) -> None:
    st.header("시장", anchor="market")
    groups = display.tape_groups(d.tape, display.rows_by_id(d))
    width = max((len(rows) for _, rows in groups), default=0)
    for title, rows in groups:
        st.markdown(f"**{title}**")
        for col, row in zip(st.columns(width), rows, strict=False):
            flat = display.change(row) == 0
            col.metric(
                display.TAPE_LABELS.get(row.id, row.label),
                display.fmt(row.value, row.fmt),
                display.metric_delta(row),
                delta_color=display.metric_color(row),
                delta_arrow="off" if flat else "auto",
                help=GUIDES.get(row.id, row.code),
                border=True,
                height="stretch",
                chart_data=display.trend(row),
            )


def regime(d: Dashboard) -> None:
    st.header("국면 요약", anchor="overview")
    rows = display.rows_by_id(d)
    for col, cell in zip(st.columns(len(d.regime)), d.regime, strict=True):
        color, icon = display.TONE[cell.tone]
        source = rows.get(display.REGIME_SOURCE.get(cell.label, ""))
        col.metric(
            cell.label,
            f"{cell.value} :small[:gray[{cell.unit}]]",
            f"{icon} {cell.tag}",
            delta_color=color,
            delta_arrow="off",
            help=cell.rule,
            border=True,
            height="stretch",
            chart_data=display.trend(source),
        )


def curves(d: Dashboard) -> None:
    st.header("금리곡선", anchor="curve")
    with st.container(border=True, height="stretch"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**2s10s** 10년물 − 2년물")
            picked = st.segmented_control(
                "기간",
                list(SPREAD_RANGES),
                default="전체",
                key="spread_range",
                label_visibility="collapsed",
            )
        spread = d.spread
        if spread is None or len(spread.points) < 2:  # noqa: PLR2004
            st.caption("FRED 키가 있어야 표시됩니다")
        else:
            months = SPREAD_RANGES[picked or "전체"]
            st.altair_chart(
                spread_chart(spread, months),
                height=CHART_HEIGHT,
                alt="10년물 − 2년물 금리차",
            )
    if "curve" in d.deep:
        with st.container(border=True):
            st.markdown("**미 국채 수익률 곡선**")
            if d.curve is None:
                st.caption("FRED 키가 있어야 표시됩니다")
            else:
                st.altair_chart(
                    curve_chart(d.curve), height=CHART_HEIGHT, alt="미 국채 수익률 곡선"
                )


def projections(d: Dashboard) -> None:
    st.header("FOMC 경제전망 · 위원 중앙값", anchor="sep")
    p = d.projections
    with st.container(border=True, height="stretch"):
        if p is None:
            st.caption("FRED 키가 있어야 표시됩니다.")
            return
        if not p.rows:
            st.caption(f"받지 못했습니다. {p.message or ''}")
            return
        data = display.sep_table(p)
        table = pd.DataFrame(data)
        changes = [key for key in data if key.endswith("Δ")]
        st.dataframe(
            table.style.map(arrow_style, subset=changes),
            column_config={
                c: st.column_config.TextColumn(
                    "Δ", width="small", help="직전 발표 대비"
                )
                for c in changes
            },
            hide_index=True,
            alt="FOMC 경제전망 요약",
        )
        if p.message:
            st.caption(f"일부를 받지 못했습니다. {p.message}")


def arrow_style(text: str) -> str:
    color = display.arrow_color(text)
    return f"color: {color}" if color else ""


def detail_chart(points: list[dict[str, str | float]]) -> alt.Chart:
    return (
        alt.Chart(alt.Data(values=points))
        .mark_line(point=len(points) <= DETAIL_POINT_LIMIT)
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("day:T", title="날짜", format="%Y-%m-%d"),
                alt.Tooltip("value:Q", title="값", format=",.2f"),
            ],
        )
        .properties(height=360)
    )


def detail_body(row: IndicatorRow) -> None:
    with st.container(horizontal=True):
        st.metric(
            "최신",
            display.fmt(row.value, row.fmt),
            display.metric_delta(row),
            delta_color=display.metric_color(row),
            delta_arrow="off" if display.change(row) == 0 else "auto",
        )
        st.metric("이전", display.fmt(row.prev, row.fmt))
        st.metric("기준", display.short_date(row) or "—")
    picked = st.segmented_control(
        "기간", list(display.DETAIL_RANGES), default="4Y", key="detail_range"
    )
    series = rt.dashboard.full_series(row.id) or display.history_series(row)
    points = display.detail_points(series, display.DETAIL_RANGES[picked or "전체"])
    if len(points) > 1:
        st.altair_chart(detail_chart(points), alt=f"{row.label} 추이")
    elif row.message:
        st.caption(f":red[{row.message}]")
    st.markdown(GUIDES.get(row.id, ""))
    st.caption(row.code + (f" · © {row.copyright}" if row.copyright else ""))


def open_detail(row: IndicatorRow) -> None:
    """Show the detail dialog, then give the tables fresh keys.

    A table keeps its selected row until it is rebuilt, so clicking that row
    again would only deselect it. New keys on the dismiss rerun (and on any
    later one) clear the selection, so the next click selects, and reopens.
    """
    state["table_gen"] = state.get("table_gen", 0) + 1
    shown = st.dialog(row.label, width="large", on_dismiss="rerun")(detail_body)
    shown(row)


def panel(d: Dashboard, p: Panel) -> None:
    view = next((v for v in d.views if v.id == p.id), None)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(p.title, anchor=p.id)
            with st.popover(":material/help:", help="지표 해석", type="tertiary"):
                for row in p.rows:
                    st.markdown(f"**{row.label}** {GUIDES.get(row.id, row.code)}")
        event = st.dataframe(
            pd.DataFrame(display.panel_table(p)).style.map(
                arrow_style, subset=["변화"]
            ),
            hide_index=True,
            height="content",
            column_config={
                "지표": st.column_config.TextColumn(width="medium"),
                "추이": st.column_config.LineChartColumn(width="small"),
            },
            alt=p.title,
            on_select="rerun",
            selection_mode="single-cell",
            key=f"table_{p.id}_{state.get('table_gen', 0)}",
        )
        for index, column in event.selection.cells:
            if column == "지표":
                open_detail(p.rows[index])
        if view is not None and view.extra:
            on = p.id in d.deep
            st.button(
                "핵심만 보기" if on else f"＋ 지표 {view.extra}개 더 보기",
                key=f"more_{p.id}",
                help="핵심 지표만 보기" if on else "나머지 지표도 보기",
                type="tertiary",
                on_click=toggle_deep,
                args=(p.id,),
            )
        for problem in display.panel_problems(p):
            st.caption(f":red[{problem}]")


def calendar(d: Dashboard) -> None:
    with st.container(border=True):
        st.subheader("앞으로 7일 미국 발표", anchor="calendar")
        if d.calendar:
            st.dataframe(
                {
                    "날짜": [display.weekday(entry.date) for entry in d.calendar],
                    "발표": [entry.name for entry in d.calendar],
                    "출처": [entry.source for entry in d.calendar],
                },
                hide_index=True,
                alt="앞으로 7일 미국 발표",
            )
        else:
            fred = any(s.id == "FRED" and s.state != "missing_key" for s in d.sources)
            st.caption(
                "앞으로 7일 안에 주요 발표가 없습니다."
                if fred
                else "FRED 키가 있으면 이번 주 발표 일정이 표시됩니다."
            )


def panels(d: Dashboard, ids: tuple[str, ...]) -> None:
    for card in d.panels:
        if card.id in ids:
            panel(d, card)


def provenance(d: Dashboard) -> None:
    st.header("출처", anchor="sources")
    st.subheader("데이터 호출 점검", anchor=False)
    status(d)
    st.caption(
        "This product uses the FRED® API but is not endorsed or certified by the"
        " Federal Reserve Bank of St. Louis. · 출처: FRED, 미 재무부, 뉴욕 연은,"
        " 한국은행 ECOS, ECB, BIS, OECD, Yahoo Finance(금 선물, yfinance)"
    )


# ---------------------------------------------------------------- analysis


def new_conversation() -> None:
    if state.ai_run is None:
        state.ai_log = display.AnalysisLog()


def start_run(text: str, preset: str = "") -> bool:
    text = text.strip()
    if not text:
        return False
    provider_id = rt.analysis.choice.provider
    log = conversation()
    log.ask(text)
    state.ai_run = rt.start(
        RunInput(
            request=text,
            provider=provider_id,
            model=rt.analysis.model_for(provider_id),
            conversation=log.conversation,
            preset="" if log.conversation else preset,
            scope="all" if rt.analysis.choice.scope == "all" else "view",
        )
    )
    return True


def drain(run: Run) -> None:
    """Move the run's new events into the log; rerun the page once it ends."""
    events, ended = run.drain()
    log = conversation()
    done = False
    for event in events:
        done = log.apply(event) or done
    if ended:
        log.finish(done=done)
        state.ai_run = None
        st.rerun()


def choice_line(statuses: dict[str, str], started: bool) -> str:
    """The provider, model and scope the next run uses; they are set in 설정."""
    pid = rt.analysis.choice.provider
    parts = [
        rt.analysis.providers[pid].title,
        rt.analysis.model_for(pid),
        statuses[pid],
    ]
    if not started:
        parts.append(SCOPES[rt.analysis.choice.scope])
    line = " · ".join(parts)
    if statuses[pid] != display.STATUS_TEXT["ready"]:
        line += " — 설정에서 공급자를 고르세요"
    return line


def controls() -> None:
    running = state.ai_run is not None
    log = conversation()
    with st.container(horizontal=True, vertical_alignment="center"):
        # A preset opens a conversation; once one exists the input takes over.
        for preset in PRESETS if not log.conversation else ():
            if st.button(
                preset.title,
                key=f"ai_preset_{preset.id}",
                disabled=running,
                help="새 대화로 바로 실행합니다",
            ) and start_run(preset.text, preset.id):
                st.rerun()
        if running and st.button("멈춤") and state.ai_run is not None:
            log.stopped = True
            state.ai_run.stop()
        st.button("새 대화", on_click=new_conversation, disabled=running)
        if log.conversation:
            st.caption(f"대화 {log.conversation}")
    request = st.chat_input(
        "무엇을 분석할까요? 화면의 지표와 도구로 받은 공개 데이터만 근거로 답합니다."
        " (Enter 실행 · Shift+Enter 줄바꿈)",
        key="ai_input",
        disabled=running,
    )
    if request and start_run(request):
        st.rerun()


def turns(d: Dashboard) -> None:
    # Citations link only to panels on this page; another page would need a
    # page load, which starts a new session and loses the conversation.
    links = {
        row: panel for row, panel in display.panel_of(d).items() if panel in page_panels
    }
    for turn in conversation().turns:
        with st.chat_message("user"):
            st.markdown(display.md(turn.question))
        with st.chat_message("assistant"):
            st.caption(turn.run_line)
            for summary, ok in turn.tools:
                st.caption(summary if ok else f":red[{summary}]")
            if turn.lines is not None:
                st.markdown(display.answer_markdown(turn.lines, links))
                st.caption(turn.summary)
            elif turn.raw:
                st.markdown(display.live_text(turn.raw))
            for message, error in turn.notices:
                if error:
                    st.error(message)
                else:
                    st.caption(message)


def analysis(d: Dashboard) -> None:
    run: Run | None = state.ai_run

    @st.fragment(run_every=0.5 if run is not None else None)
    def body() -> None:
        current: Run | None = state.ai_run
        if current is not None:
            drain(current)
        log = conversation()
        statuses = {
            pid: display.STATUS_TEXT.get(status.kind, status.kind)
            for pid, status in rt.call(rt.analysis.statuses()).items()
        }
        # Header on top, the scrolling conversation in the middle and the input
        # at the bottom; the CSS in AI_CSS makes the panel a column.
        st.subheader("AI 매크로 분석 · 선택 기능", anchor="analysis")
        st.caption(log.header)
        if state.ai_error:
            st.error(state.ai_error)
        else:
            st.caption(choice_line(statuses, bool(log.conversation)))
        if log.stale and log.conversation:
            st.warning(
                f"이 대화는 {display.short_time(log.snapshot)} 데이터에 고정돼"
                " 있습니다. 새로 받은 데이터로 보려면 새 대화를 시작하세요."
            )
        with st.container(key="ai_turns"), st.container(key="ai_turns_inner"):
            turns(d)
            if log.received:
                with st.expander("받은 데이터"):
                    st.markdown(
                        "\n".join(
                            f"- `{cite}` {display.md(info)}"
                            for cite, info in log.received
                        )
                    )
        controls()

    body()


# ---------------------------------------------------------------- screening


@st.cache_resource(show_spinner=False)
def screen_updater(root: str) -> screen.Updater:
    """One updater per data folder, shared by every tab."""
    return screen.Updater(Path(root))


@st.cache_data(show_spinner="스크리닝 계산 중…", max_entries=2)
def screen_table(root: str, stamp: tuple[tuple[str, int], ...]) -> pd.DataFrame | None:
    """The evaluated table; ``stamp`` changes whenever a stored file does."""
    inputs = screen.load(Path(root)) if stamp else None
    return None if inputs is None else screen.evaluate(inputs)


def parts_line(parts: screen.Parts) -> str:
    return (
        f"시세 {parts.prices or '—'} 종가 · 재무 {parts.fundamentals or '—'}"
        f" · 추정치 {parts.estimates or '—'} · 유니버스 {parts.universe or '—'}"
        " · USD · yfinance (Yahoo Finance 비공식 데이터)"
    )


def screen_bar(updater: screen.Updater, root: Path) -> None:
    """Data dates, the 갱신 button and, while an update runs, its progress."""
    polling = updater.progress.running
    missing = screen.missing_packages()

    @st.fragment(run_every=1.0 if polling else None)
    def body() -> None:
        progress = updater.progress
        if polling and not progress.running:
            st.rerun()
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(parts_line(screen.parts(root)))
            st.space("stretch")
            if st.button(
                "갱신",
                icon=":material/refresh:",
                type="primary",
                key="screen_refresh",
                disabled=progress.running or bool(missing) or args.demo,
                help=(
                    "시세는 새 날짜만, 추정치는 하루 1회, 재무는 분기 1회,"
                    " 구성 종목은 월 1회 받습니다. 처음에는 오래 걸립니다."
                ),
            ):
                # Rerun the page even if another tab started first, so it polls.
                updater.start()
                st.rerun()
        if progress.running:
            done = progress.done / progress.total if progress.total else 0.0
            count = f" {progress.done}/{progress.total}" if progress.total else ""
            st.progress(done, text=f"{progress.stage}{count}")
        elif progress.message:
            if progress.failed:
                st.error(progress.message)
            else:
                st.success(progress.message)
        if missing:
            st.info(
                f"갱신하려면 추가 패키지({', '.join(missing)})가 필요합니다:"
                f" `{screen.INSTALL_HINT}`"
            )
        elif args.demo:
            st.caption("데모 모드에서는 갱신하지 않습니다.")

    body()


def return_style(value: object) -> str:
    """Red up, blue down, as on the dashboard."""
    if not isinstance(value, int | float) or value == 0 or value != value:  # noqa: PLR0124
        return ""
    return f"color: {display.UP_COLOR if value > 0 else display.DOWN_COLOR}"


def stock_chart(points: list[dict[str, str | float]]) -> alt.Chart:
    lines = ["종가", "20일선", "60일선", "200일선"]
    return (
        alt.Chart(alt.Data(values=points))
        .mark_line()
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("value:Q", title="$", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "line:N", sort=lines, title=None, legend=alt.Legend(orient="bottom")
            ),
            strokeWidth=alt.condition(
                alt.datum.line == "종가", alt.value(2), alt.value(1)
            ),
            tooltip=[
                alt.Tooltip("day:T", title="날짜", format="%Y-%m-%d"),
                alt.Tooltip("line:N", title="선"),
                alt.Tooltip("value:Q", title="$", format=",.2f"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )


def check_lines(lines: list[tuple[bool, str, str]]) -> str:
    return "  \n".join(
        display.md(f"{'✓' if ok else '✗'} {label} · :gray[{detail}]")
        for ok, label, detail in lines
    )


def status_lines(lines: list[tuple[str, str, str]]) -> str:
    return "  \n".join(
        display.md(f"**{result or '—'}** {test} · :gray[{detail}]")
        for result, test, detail in lines
    )


def stock_body(root: Path, row: dict[str, object]) -> None:
    fmt = screen.fmt
    st.caption(
        f"{row['sector']} · {row['industry']} · 비교 업종"
        f" {row['peer']} ({screen.peer_level(row)}, {fmt('int', row['peer_n'])}종목,"
        " S&P 500 내) · 재무 통화"
        f" {row['currency'] or '—'}"
    )
    groups = [
        name for key, name in screen.GROUP_NAMES.items() if row.get(f"in_{key}") is True
    ]
    with st.container(horizontal=True):
        st.metric("주가", fmt("usd", row["price"]))
        st.metric("시가총액", fmt("big", row["mcap"]))
        st.metric("기술 점수", f"{fmt('int', row['score'])}/10 {row['chart']}")
        st.metric("52주 고점 대비", fmt("signed", row["from_high"]))
        st.metric("그룹", ", ".join(groups) or "—")
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        st.markdown("**기술 점수**")
        st.markdown(check_lines(screen.tech_lines(row)))
        st.markdown("**과열 표시**")
        st.markdown(check_lines(screen.overheat_lines(row)))
        st.caption(display.md(screen.TECH_BASIS))
    with right, st.container(border=True, height="stretch"):
        st.markdown("**결격 점검**")
        st.markdown(status_lines(screen.dq_lines(row)))
        st.markdown("**라벨** :gray[업종 중앙값은 S&P 500 내 업종 중앙값]")
        st.markdown(status_lines(screen.label_lines(row)))
        st.markdown(f"**EPS 추정치** :gray[{row['est_currency'] or '—'}]")
        st.dataframe(screen.estimate_table(row), hide_index=True, alt="EPS 추정치")
    points = screen.chart_rows(screen.history(root, str(row["ticker"])))
    if len(points) > 1:
        st.altair_chart(stock_chart(points), alt=f"{row['ticker']} 1년 주가")
    if row["error"]:
        st.caption(f":red[{display.md(str(row['error']))}]")


def open_stock(root: Path, row: dict[str, object]) -> None:
    """Show the stock dialog; a fresh table key clears the clicked cell.

    Same pattern as ``open_detail``: clicking that cell again reopens the
    dialog, and unrelated reruns do not.
    """
    state["screen_gen"] = state.get("screen_gen", 0) + 1
    title = f"{row['ticker']} · {row['name']}"
    st.dialog(title, width="large", on_dismiss="rerun")(stock_body)(root, row)


def pick_group(key: str) -> None:
    state["screen_group"] = key


def group_button(key: str, counts: dict[str, int], current: str) -> None:
    st.button(
        f"{screen.GROUP_NAMES[key]} · {counts[key]}",
        key=f"screen_cell_{key}",
        type="primary" if key == current else "secondary",
        width="stretch",
        help=screen.GROUPS[key],
        on_click=pick_group,
        args=(key,),
    )


def group_map(counts: dict[str, int], current: str) -> None:
    """Chart strength by valuation; 저평가 개선 sits beside it on its own axis."""
    widths = [1, 2, 2]
    with st.container(horizontal=True, vertical_alignment="top"):
        with st.container(border=True):
            for cell, text in zip(
                st.columns(widths), ("", *screen.MAP_COLUMNS), strict=True
            ):
                cell.caption(text)
            for label, keys in screen.MAP_ROWS:
                cells = st.columns(widths, vertical_alignment="center")
                cells[0].caption(label)
                for cell, key in zip(cells[1:], keys, strict=True):
                    with cell:
                        if key is None:
                            st.button(
                                "—",
                                key="screen_cell_none",
                                disabled=True,
                                width="stretch",
                                help="차트 약이면서 비쌈인 종목은 어느 그룹에도"
                                " 넣지 않습니다",
                            )
                        else:
                            group_button(key, counts, current)
        with st.container(border=True, width=220):
            st.caption("추정치 개선")
            group_button("D", counts, current)


def screen_results(root: Path, table: pd.DataFrame) -> None:
    liquid = screen.liquid(table)
    chosen: list[str] = state.get("screen_sectors") or []
    filtered = screen.pick(table, "all", chosen)
    counts = screen.counts(filtered)
    state.setdefault("screen_group", "all")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        picked = st.segmented_control(
            "그룹",
            ["all", *screen.GROUPS],
            format_func=lambda key: (
                f"전체 {len(filtered)}"
                if key == "all"
                else f"{screen.GROUP_NAMES[key]} {counts[key]}"
            ),
            key="screen_group",
            help=screen.group_help(),
        )
        sectors = st.multiselect(
            "섹터",
            screen.sectors(liquid),
            key="screen_sectors",
            placeholder="전체 섹터",
            width=320,
        )
    key = picked or "all"
    group_map(counts, key)
    rows = screen.pick(table, key, sectors)
    st.caption(
        f"유동성 통과 {len(rows)}종목 · 시가총액순 · 티커나 종목 이름을 누르면"
        " 점검 내용을 봅니다"
        if key == "all"
        else f"**{screen.GROUP_NAMES[key]}** 상위 {len(rows)}종목 ·"
        f" {screen.GROUP_SORT[key]}"
    )
    shown = screen.view(rows)
    if key != "D":
        shown = shown.drop(columns=["signals"])
    columns = [c for c, _, _ in screen.TABLE if c in shown.columns]
    formats = {c: partial(screen.fmt, kind) for c, _, kind in screen.TABLE if kind}
    styled = shown.style.map(return_style, subset=list(screen.RETURN_COLUMNS)).format(
        {c: formats[c] for c in columns if c in formats}
    )
    labels = {c: label for c, label, _ in screen.TABLE}
    event = st.dataframe(
        styled,
        hide_index=True,
        height=560,
        on_select="rerun",
        selection_mode="single-cell",
        key=f"screen_table_{state.get('screen_gen', 0)}",
        column_config={
            column: st.column_config.Column(
                labels[column],
                pinned=column == "ticker",
                width="medium" if column == "name" else None,
            )
            for column in columns
        },
        alt="스크리닝 결과",
    )
    st.caption(
        "커트라인(상위 20%·하위 40%·결격 3개·은행 TCE/TA 4%)은 백테스트 없이 정한"
        " 값입니다 ·"
        " 재무는 Yahoo 단일 출처라 일회성 항목에 약합니다 · 공매도 비율은 약 2주"
        " 늦습니다 · 기술 점수와 재무는 합산하지 않습니다"
    )
    failed = screen.failures(table)
    if len(failed):
        with st.expander(f"제외되었거나 받지 못한 값이 있는 종목 {len(failed)}"):
            st.dataframe(failed, hide_index=True, alt="제외된 종목")
    records = screen.records(rows)
    for index, column in event.selection.cells:
        if column in {"ticker", "name"}:
            open_stock(root, records[index])


# ---------------------------------------------------------------- portfolio

PF_SECTIONS = {
    "holdings": "보유·거래",
    "returns": "수익률",
    "analysis": "위험지표",
    "correlation": "상관·분산",
    "risk": "시뮬레이션·VaR",
    "stress": "스트레스",
}
PF_MIN_QUANTITY = 0.0001
PF_SIDE_WIDTH = 72  # 매수 and 매도 the same width
PF_CURRENCY_HELP = "종목의 거래 통화 기준 (AAPL은 USD, 005930.KS는 KRW)"
PF_DIALOGS = {
    "new": "새 포트폴리오",
    "edit": "포트폴리오 편집",
    "delete": "포트폴리오 삭제",
}


@st.cache_resource(show_spinner=False)
def portfolio_fetcher(root: str) -> portfolio.Fetcher:
    """One price fetcher per data folder, shared by every tab."""
    return portfolio.Fetcher(Path(root))


def pf_root() -> Path:
    return resolve_settings(args.settings).parent / portfolio.DATA_DIR


def pf_store(change: Callable[[portfolio.Book], portfolio.Book]) -> str:
    """Apply ``change`` to the saved book and save it; the error text, if any.

    The file is read again first, so a change made in another tab is kept.
    """
    root = pf_root()
    try:
        portfolio.save_book(root, change(portfolio.load_book(root)))
    except (portfolio.BookError, OSError) as error:
        return str(error)
    return ""


def pf_style(text: str) -> str:
    if text.startswith("+") or text == "익절":
        return f"color: {display.UP_COLOR}"
    if text.startswith("−") or text == "손절":
        return f"color: {display.DOWN_COLOR}"
    return ""


def pf_color(value: float) -> Literal["red", "blue", "off"]:
    if not value or math.isnan(value):
        return "off"
    return "red" if value > 0 else "blue"


def pf_table(data: dict[str, list[str]], alt_text: str) -> None:
    st.dataframe(pd.DataFrame(data).style.map(pf_style), hide_index=True, alt=alt_text)


def pf_ratio(value: float) -> str:
    if not math.isfinite(value):
        return portfolio.UNKNOWN
    return f"{'−' if value < 0 else ''}{abs(value):.2f}"


def pf_axis(base: portfolio.Currency) -> str:
    return ",.0f" if portfolio.DECIMALS[base] == 0 else ",.2f"


# ---- caches: in memory only, keyed by everything the inputs depend on
# The leading-underscore arguments are not hashed; the key stands for them:
# the saved prices' stamp, the portfolio or book JSON (trades and base), the
# window and other choices, and for investors the 13F files' stamp.

type Key = tuple[object, ...]


def pf_files_stamp(root: Path) -> tuple[tuple[str, int, int], ...]:
    out: list[tuple[str, int, int]] = []
    for name in (portfolio.CLOSE_FILE, portfolio.ADJUSTED_FILE):
        path = root / name
        if path.is_file():
            info = path.stat()
            out.append((name, info.st_mtime_ns, info.st_size))
    return tuple(out)


@st.cache_resource(show_spinner=False, max_entries=2)
def pf_prices(root: str, stamp: tuple[tuple[str, int, int], ...]) -> portfolio.Prices:
    """The saved prices, read once per ``stamp`` (``pf_files_stamp``).

    Shared, not copied: nothing writes into the frames (pandas copy-on-write
    keeps every derived frame separate).
    """
    del stamp
    return portfolio.load_prices(Path(root))


def pf_key(*parts: object) -> Key:
    """A cache key: the saved prices' stamp (taken before loading) and ``parts``."""
    return (state.get("pf_prices_stamp", ()), *parts)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_twr(key: Key, _p: portfolio.Portfolio, _close: pd.DataFrame) -> pd.Series:
    del key
    return portfolio.twr(_p, _close)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_history(
    key: Key, _p: portfolio.Portfolio, _close: pd.DataFrame
) -> pd.DataFrame:
    del key
    return portfolio.history(_p, _close)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_held(
    key: Key, _p: portfolio.Portfolio, _adjusted: pd.DataFrame
) -> pd.Series:
    del key
    return portfolio.held_returns(_p, _adjusted)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_lineup(
    key: Key,
    base: portfolio.Currency,
    months: int,
    _book: portfolio.Book,
    _prices: portfolio.Prices,
) -> tuple[pd.DataFrame, list[bool]]:
    del key
    return portfolio.lineup(_book, _prices, base, months)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_lagged(
    key: Key, korean: tuple[bool, ...], _frame: pd.DataFrame
) -> portfolio.Correlation | None:
    del key
    return portfolio.lagged_correlation(_frame, korean)


# ---- new analytics: memory only, keyed like the caches above


@st.cache_data(show_spinner=False, max_entries=16)
def cached_benchmarks(
    key: Key, _p: portfolio.Portfolio, _close: pd.DataFrame, _days: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    del key
    return portfolio.benchmark_lines(_p, _close, _days)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_flows(
    key: Key, _p: portfolio.Portfolio, _close: pd.DataFrame
) -> tuple[perf.FxSplit | None, portfolio.Trading | None]:
    """The currency split and the do-nothing comparison."""
    del key
    return portfolio.currency_split(_p, _close), portfolio.trading(_p, _close)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_ratios(
    key: Key,
    _rets: pd.Series,
    _rf: pd.Series,
    _bench: Callable[[], pd.Series],
) -> portfolio.Ratios:
    """``key`` stands for the series, prices, window and rates.

    ``_bench`` gives the daily benchmark returns, built only on a miss.
    """
    del key
    return portfolio.ratios(_rets, _rf, _bench())


@st.cache_data(show_spinner=False, max_entries=16)
def cached_joint(
    key: Key,
    base: portfolio.Currency,
    months: int,
    _adjusted: pd.DataFrame,
    _w: pd.Series,
) -> portfolio.Joint:
    del key
    return portfolio.joint(_adjusted, _w, base, months)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_covariance(
    key: Key, _j: portfolio.Joint
) -> tuple[NDArray[np.float64], float, float]:
    del key
    return portfolio.covariance(_j)


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_draws(stamp: object, n: int, horizon: int, block: int) -> NDArray[np.int64]:
    """Stationary-bootstrap day indices, drawn once per window length and horizon.

    Any weight vector reuses them; read-only, so the cached array is shared.
    """
    del stamp
    picks = quant.stationary_indices(n, portfolio.PATHS, horizon, block)
    picks.flags.writeable = False
    return picks


@st.cache_data(show_spinner=False, max_entries=16)
def cached_paths(
    key: Key,
    fhs: bool,
    value: float,
    standing: tuple[float, float],
    _j: portfolio.Joint,
    _picks: NDArray[np.int64],
    _w: NDArray[np.float64] | None = None,
) -> portfolio.Simulation:
    """The summary, fan and histogram; the paths themselves are not kept.

    ``_w`` replaces ``_j``'s weights on the same draws; ``key`` stands for it.
    """
    del key
    growth = portfolio.growth(_j, _picks, fhs=fhs, w=_w)
    return portfolio.simulation(growth, value, *standing)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_tails(key: Key, _j: portfolio.Joint) -> portfolio.Tails:
    del key
    return portfolio.tails(_j)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_replays(
    key: Key,
    base: portfolio.Currency,
    _values: dict[str, float],
    _adjusted: pd.DataFrame,
    _fx: pd.Series,
) -> list[portfolio.Replayed]:
    del key
    return portfolio.scenario_replays(_values, _adjusted, base, _fx)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_betas(
    key: Key, months: int, names: tuple[str, ...], _adjusted: pd.DataFrame
) -> NDArray[np.float64]:
    del key
    return portfolio.local_betas(_adjusted, names, months)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_bad_prints(
    key: Key, names: tuple[str, ...], _adjusted: pd.DataFrame
) -> dict[str, list[date]]:
    del key
    return portfolio.bad_print_days(_adjusted, names)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_turbulence(
    key: Key, base: portfolio.Currency, names: tuple[str, ...], _adjusted: pd.DataFrame
) -> portfolio.Turbulence | None:
    del key
    return portfolio.turbulence(_adjusted, names, base)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_style(
    key: Key, base: portfolio.Currency, _adjusted: pd.DataFrame, _w: pd.Series
) -> portfolio.Style | None:
    del key
    return portfolio.style(_adjusted, _w, base)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_trimmed(
    key: Key,
    base: portfolio.Currency,
    months: int,
    _adjusted: pd.DataFrame,
    _w: pd.Series,
) -> tuple[pd.Series, list[str]]:
    """``portfolio.covered`` over the window; ``key`` stands for ``_w``."""
    del key
    names = portfolio.labels(_w)
    rets = portfolio.window(portfolio.asset_returns(_adjusted, names, base), months)
    return portfolio.covered(rets, _w)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_peers(
    key: Key, _series: dict[str, pd.Series], _rf: pd.Series
) -> portfolio.Peers:
    del key
    return portfolio.peers(_series, _rf)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_added(
    key: Key,
    fraction: float,
    ticker: str,
    _j: portfolio.Joint,
    _cov: NDArray[np.float64],
) -> portfolio.Added:
    """``key`` stands for ``_j`` and ``_cov``."""
    del key
    return portfolio.added(_j, _cov, ticker, fraction)


# ---- what the shared sections analyse

PF_HEAT = 15  # holdings in a correlation heatmap or beta table
PF_LONG_MONTHS = 40  # turbulence and style: ``portfolio.WEEKS`` weeks and a margin


@dataclasses.dataclass(frozen=True, slots=True)
class Subject:
    """A daily return series and current weights, for the shared sections.

    A portfolio or a 13F clone, later a single stock or a what-if mix.
    ``key`` stands for everything else here in the caches.
    """

    key: Key
    label: str  # what ``rets`` are, for captions
    base: portfolio.Currency
    rets: pd.Series  # realized daily returns in base, whole history
    w: pd.Series  # current weights by ticker, sum 1
    value: float  # base, for amounts
    bench: Callable[[pd.DatetimeIndex], pd.Series]  # daily benchmark returns on days
    bench_label: str
    trim: bool = False  # leave out holdings whose returns start or stop in a window
    notes: tuple[str, ...] = ()
    holding: str = "지금 보유분"  # what the stress amounts are, for headings


def pf_subject(p: portfolio.Portfolio, prices: portfolio.Prices) -> Subject:
    pid = p.model_dump_json()
    items = portfolio.holdings(p, prices.close)
    w = portfolio.weights(items)
    unknown = portfolio.unpriced(items)
    return Subject(
        key=pf_key(pid),
        label="보유 기록",
        base=p.base,
        rets=cached_held(pf_key(pid), p, prices.adjusted),
        w=w,
        value=sum(h.value for h in items if h.ticker in w.index),
        bench=partial(portfolio.blended, p, prices.adjusted),
        bench_label="혼합 벤치마크",
        notes=(f"제외 (평가액 미확인): {', '.join(unknown)}",) if unknown else (),
    )


def pf_weights(
    s: Subject, prices: portfolio.Prices, months: int
) -> tuple[pd.Series, float, list[str]]:
    """The weights a window uses, their share of ``s.w`` and the names left out."""
    if not s.trim or s.w.empty:
        return s.w, 1.0, []
    key = (*s.key, months, "trim")
    w, dropped = cached_trimmed(key, s.base, months, prices.adjusted, s.w)
    kept = cast("float", s.w.reindex(portfolio.labels(w)).sum())
    return w, float(kept), dropped


# ---- rates from the dashboard's sources (memory only)

PF_CD91 = "kr_cd91"
PF_RATE_YEARS = 20  # CD 91-day back to about HISTORY_START
PF_FX_YEARS = 40  # FRED DEXKOUS back before 1987
PF_DEXKOUS = "DEXKOUS"


def pf_empty() -> pd.Series:
    return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))


def pf_series(points: Points) -> pd.Series:
    return pd.Series(
        [value for _, value in points],
        index=pd.DatetimeIndex([pd.Timestamp(day) for day, _ in points]),
        dtype="float64",
    )


def pf_cd91() -> Fetch | None:
    found = next(
        (i for spec in catalog.PANELS for i in spec.indicators if i.id == PF_CD91), None
    )
    if found is None or not isinstance(found.fetch, EcosSeries):
        return None
    return dataclasses.replace(found.fetch, years=PF_RATE_YEARS)


@st.cache_data(show_spinner=False, max_entries=4)
def pf_fetched(name: str, day: str, refreshes: int) -> pd.Series:
    """CD 91-day (ECOS) or USD/KRW (FRED DEXKOUS) through the dashboard's sources.

    Empty without a key or on any source error. ``day`` and ``refreshes``
    fetch again on a new day or after the dashboard's 새로고침.
    """
    del day, refreshes
    fetch = pf_cd91() if name == PF_CD91 else FredSeries(PF_DEXKOUS, years=PF_FX_YEARS)
    if fetch is None:
        return pf_empty()

    async def get() -> Points:
        async with rt.dashboard.fetch_context() as ctx:
            return await fetch.fetch(ctx, rt.dashboard.today())

    try:
        return pf_series(rt.call(get()))
    except (SourceError, OSError, ValueError):
        return pf_empty()


def pf_dashboard_series(name: str) -> pd.Series:
    """``name`` from the dashboard's memory, else fetched; empty in demo mode."""
    if args.demo:
        return pf_empty()
    if name == PF_CD91:
        known = rt.dashboard.full_series(PF_CD91)
        if known:
            return pf_series(known)
    today = rt.dashboard.today().isoformat()
    return pf_fetched(name, today, rt.dashboard.refreshes)


def pf_rates(base: portfolio.Currency, close: pd.DataFrame) -> tuple[pd.Series, str]:
    """Annual percent risk-free rates for ``base`` and their name; empty when none."""
    if base == "KRW":
        return pf_dashboard_series(PF_CD91), "CD 91일"
    if portfolio.RF_USD in close.columns:
        return cast(
            "pd.Series", close[portfolio.RF_USD]
        ).dropna(), "미국 3개월 국채(^IRX)"
    return pf_empty(), "^IRX"


def pf_rates_key(rates: pd.Series) -> tuple[int, str]:
    last = cast("pd.Timestamp", rates.index[-1]) if len(rates) else None
    return len(rates), str(last)


# ---- dialogs


def pf_open(kind: str, p: portfolio.Portfolio | None = None) -> None:
    state.pf_dialog = kind
    state.pf_dialog_error = ""
    if kind == "new":
        state.pf_new_name = ""
        state.pf_new_base = "KRW"
    elif p is not None:
        state.pf_edit_name = p.name
        state.pf_edit_base = p.base


def pf_close() -> None:
    state.pf_dialog = ""


def pf_create() -> None:
    made: list[str] = []

    def change(book: portfolio.Book) -> portfolio.Book:
        new, pid = portfolio.add_portfolio(
            book, state.pf_new_name, state.pf_new_base or "KRW"
        )
        made.append(pid)
        return new

    state.pf_dialog_error = pf_store(change)
    if not state.pf_dialog_error:
        state.pf_selected = made[0]
        pf_close()


def pf_edit(pid: str) -> None:
    state.pf_dialog_error = pf_store(
        lambda book: portfolio.edit_portfolio(
            book, pid, state.pf_edit_name, state.pf_edit_base or "KRW"
        )
    )
    if not state.pf_dialog_error:
        pf_close()


def pf_delete(pid: str) -> None:
    state.pf_dialog_error = pf_store(lambda book: portfolio.remove_portfolio(book, pid))
    if not state.pf_dialog_error:
        state.pf_selected = None
        pf_close()


def pf_dialog_body(kind: str, p: portfolio.Portfolio | None) -> None:
    if state.get("pf_dialog") != kind:
        # A button here closed it; only the dialog reran, so rerun the page.
        st.rerun()
    if kind == "delete" and p is not None:
        st.markdown(f"**{p.name}**과 거래 {len(p.trades)}건을 지웁니다.")
    else:
        prefix = "pf_new" if kind == "new" else "pf_edit"
        st.text_input("이름", key=f"{prefix}_name", max_chars=40)
        st.segmented_control(
            "기준통화", ["KRW", "USD"], required=True, key=f"{prefix}_base"
        )
    if state.pf_dialog_error:
        st.error(state.pf_dialog_error)
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.button("취소", key="pf_cancel", on_click=pf_close)
        pid = p.id if p is not None else ""
        if kind == "new":
            st.button("만들기", type="primary", key="pf_new_save", on_click=pf_create)
        elif kind == "edit":
            st.button(
                "저장",
                type="primary",
                key="pf_edit_save",
                on_click=pf_edit,
                args=(pid,),
            )
        else:
            st.button(
                "삭제",
                type="primary",
                key="pf_delete_ok",
                on_click=pf_delete,
                args=(pid,),
            )


def pf_dialogs(p: portfolio.Portfolio | None) -> None:
    """The open dialog, if any; it stays open until saved, cancelled or dismissed."""
    kind: str = state.get("pf_dialog", "")
    if kind in PF_DIALOGS and (kind == "new" or p is not None):
        st.dialog(PF_DIALOGS[kind], on_dismiss=pf_close)(pf_dialog_body)(kind, p)


def pf_selector(book: portfolio.Book) -> portfolio.Portfolio | None:
    names = {p.id: p.name for p in book.portfolios}
    if state.get("pf_selected") not in names:
        state.pf_selected = next(iter(names), None)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        if names:
            st.selectbox(
                "포트폴리오",
                list(names),
                format_func=names.__getitem__,
                key="pf_selected",
                width=280,
            )
        st.button(
            "새로 만들기",
            icon=":material/add:",
            key="pf_new",
            on_click=pf_open,
            args=("new",),
        )
        chosen = book.get(state.pf_selected or "")
        if chosen is not None:
            st.button(
                "편집",
                icon=":material/edit:",
                key="pf_edit",
                on_click=pf_open,
                args=("edit", chosen),
            )
            st.button(
                "삭제",
                icon=":material/delete:",
                key="pf_delete",
                on_click=pf_open,
                args=("delete", chosen),
            )
    return chosen


# ---- prices


def pf_bar(
    fetcher: portfolio.Fetcher,
    root: Path,
    book: portfolio.Book,
    missing: list[str],
    extra: Sequence[str] = (),
) -> None:
    """When prices were saved, the 가격 갱신 button and a running fetch.

    ``extra``: tickers outside the book that 가격 갱신 also fetches.
    """
    polling = fetcher.progress.running
    tickers = [*portfolio.book_tickers(book), *extra]

    @st.fragment(run_every=1.0 if polling else None)
    def body() -> None:
        progress = fetcher.progress
        if polling and not progress.running:
            st.rerun()
        saved = portfolio.prices_saved_at(root)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(
                f"가격 {saved:%Y-%m-%d %H:%M} 저장 · yfinance (Yahoo Finance 비공식"
                " 데이터)"
                if saved
                else "저장된 가격이 없습니다."
            )
            st.space("stretch")
            if st.button(
                "가격 갱신",
                icon=":material/refresh:",
                key="pf_refresh",
                disabled=progress.running or bool(missing) or args.demo or not tickers,
                help="모든 종목의 가격 이력을 다시 받습니다",
            ):
                fetcher.start(list(dict.fromkeys([*portfolio.wanted(book), *extra])))
                st.rerun()
        if progress.running:
            st.caption(
                f":material/progress_activity: 가격 받는 중 · {progress.message}"
            )
        elif progress.message:
            if progress.failed:
                st.warning(progress.message)
            else:
                st.success(progress.message)
        if missing:
            st.info(
                f"가격을 받으려면 추가 패키지({', '.join(missing)})가 필요합니다:"
                f" `{portfolio.INSTALL_HINT}`"
            )
        elif args.demo:
            st.caption("데모 모드에서는 가격을 받지 않습니다.")

    body()


def pf_no_prices() -> None:
    st.info(
        "가격이 없습니다." + ("" if args.demo else " 위의 **가격 갱신**으로 받습니다.")
    )


# ---- holdings and trades


def pf_add_trade(pid: str, side: portfolio.Side) -> None:
    state.pf_trade_error = ""
    quantity: float | None = state.pf_quantity
    price: float | None = state.pf_price
    if not quantity or not price:
        state.pf_trade_error = "수량과 가격을 입력하세요."
        return
    try:
        trade = portfolio.Trade(
            id=portfolio.new_id(),
            day=state.pf_day,
            ticker=portfolio.clean_ticker(state.pf_ticker),
            side=side,
            quantity=quantity,
            price=price,
            fee=state.pf_fee or 0.0,
            memo=state.get("pf_memo", "").strip(),
        )
    except portfolio.BookError as error:
        state.pf_trade_error = str(error)
        return
    state.pf_trade_error = pf_store(lambda book: portfolio.add_trade(book, pid, trade))
    if not state.pf_trade_error:
        state.toast = f"{trade.ticker} {portfolio.SIDES[side]} 거래를 추가했습니다"
        state.pf_quantity = 1.0
        state.pf_price = None
        state.pf_fee = 0.0
        state.pf_price_note = ""
        state.pf_memo = ""


def pf_remove_trades(pid: str, ids: list[str], key: str) -> None:
    rows: list[int] = state[key]["selection"]["rows"] if key in state else []
    chosen = {ids[row] for row in rows}
    if not chosen:
        return
    state.pf_trade_error = pf_store(
        lambda book: portfolio.remove_trades(book, pid, chosen)
    )
    if not state.pf_trade_error:
        state.pf_gen = state.get("pf_gen", 0) + 1
        state.toast = f"거래 {len(chosen)}건을 지웠습니다"


def pf_prefill_later() -> None:
    state.pf_prefill = True


def pf_prefill() -> None:
    """Fill 가격 with the close on the trade date, or the last one before it.

    Saved prices first; a ticker without any is fetched once (not in demo).
    Runs only after the ticker or date changes, so a typed price stays.
    """
    state.pf_price_note = ""
    try:
        ticker = portfolio.clean_ticker(state.get("pf_ticker", ""))
    except portfolio.BookError:
        return
    root = pf_root()
    prices = pf_prices(str(root), pf_files_stamp(root))
    if ticker not in prices.tickers:
        if args.demo or portfolio.missing_packages():
            state.pf_price = None
            state.pf_price_note = "저장된 가격 없음"
            return
        try:
            with st.spinner(f"{ticker} 가격 받는 중…"):
                prices = portfolio_fetcher(str(root)).fetch_now([ticker])
        except Exception:  # noqa: BLE001 - the field just stays empty
            prices = portfolio.no_prices()
    price, day = portfolio.close_on(prices.close, ticker, state.pf_day)
    if day is None:
        state.pf_price = None
        state.pf_price_note = "가격을 찾지 못했습니다"
        return
    state.pf_price = round(price, portfolio.DECIMALS[portfolio.currency(ticker)])
    state.pf_price_note = f"{day:%m/%d} 종가 기준 · 지연"


def pf_trade_form(p: portfolio.Portfolio) -> None:
    today = datetime.now().astimezone().date()
    state.setdefault("pf_quantity", 1.0)
    state.setdefault("pf_fee", 0.0)
    if state.get("pf_prefill"):
        state.pf_prefill = False
        pf_prefill()
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.date_input(
                "날짜",
                max_value=today,
                key="pf_day",
                width=150,
                on_change=pf_prefill_later,
            )
            st.text_input(
                "티커",
                key="pf_ticker",
                width=170,
                on_change=pf_prefill_later,
            )
            st.number_input(
                "수량",
                min_value=PF_MIN_QUANTITY,
                step=1.0,
                format="%.4g",  # whole shares step by 1; fractions can be typed
                key="pf_quantity",
                width=130,
            )
            st.number_input(
                "가격",
                min_value=0.0,
                value=None,
                key="pf_price",
                width=150,
                help=PF_CURRENCY_HELP,
            )
            st.number_input(
                "수수료", min_value=0.0, key="pf_fee", width=120, help=PF_CURRENCY_HELP
            )
            st.text_input(
                "메모",
                key="pf_memo",
                max_chars=portfolio.MEMO_LENGTH,
                width="stretch",
            )
            for side, label in portfolio.SIDES.items():
                st.button(
                    label,
                    type="primary" if side == "buy" else "secondary",
                    key=f"pf_{side}",
                    width=PF_SIDE_WIDTH,
                    on_click=pf_add_trade,
                    args=(p.id, side),
                )
        if state.get("pf_price_note"):
            st.caption(state.pf_price_note)
        if state.get("pf_trade_error"):
            st.error(state.pf_trade_error)


def pf_trade_log(p: portfolio.Portfolio) -> None:
    table, ids = portfolio.trades_table(p)
    key = f"pf_trades_{state.get('pf_gen', 0)}"
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f"**거래 {len(ids)}건**")
        st.space("stretch")
        st.button(
            "선택 삭제",
            icon=":material/delete:",
            key="pf_remove",
            type="tertiary",
            on_click=pf_remove_trades,
            args=(p.id, ids, key),
        )
    st.dataframe(
        table.style.map(pf_style),
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=key,
        column_config={"메모": st.column_config.TextColumn(width="large")},
        alt="거래 내역",
    )


def pf_holdings(p: portfolio.Portfolio, prices: portfolio.Prices) -> None:
    items = portfolio.holdings(p, prices.close)
    held = [h for h in items if h.is_open]
    if held:
        totals = portfolio.totals(items)
        with st.container(horizontal=True):
            st.metric("평가액", portfolio.money(totals.value, p.base), border=True)
            st.metric(
                "평가손익",
                portfolio.money(totals.unrealized, p.base, signed=True),
                portfolio.percent(totals.open_return, signed=True),
                delta_color=pf_color(totals.unrealized),
                delta_arrow="off",
                border=True,
            )
        pf_table(portfolio.holdings_table(items, p.base), "보유 종목")
        unknown = portfolio.unpriced(items)
        if unknown:
            st.caption(f"평가액 미확인: {', '.join(unknown)} (종가 또는 환율 없음)")
    elif not p.trades:
        st.info("거래를 추가하세요. 날짜, 티커, 수량, 가격을 넣습니다.")
    pf_trade_form(p)
    if p.trades:
        pf_trade_log(p)


# ---- performance


def pf_line_chart(
    points: list[dict[str, str | float]], number: str, height: int = CHART_HEIGHT
) -> alt.Chart:
    return (
        alt.Chart(pf_rows(points))
        .mark_line()
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format=number)),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=[
                alt.Tooltip("day:T", title="날짜", format="%Y-%m-%d"),
                alt.Tooltip("series:N", title="구분"),
                alt.Tooltip("value:Q", title="값", format=number),
            ],
        )
        .properties(height=height)
    )


def pf_rows(points: list[dict[str, str | float]]) -> pd.DataFrame:
    """Chart rows as a frame, sent as a dataset.

    Inline values are schema-checked row by row: seconds for a few thousand days.
    """
    return pd.DataFrame(points, columns=["day", "series", "value"])


def pf_contribution_chart(
    points: list[dict[str, str | float]], base: portfolio.Currency
) -> alt.Chart:
    up = alt.datum.pnl > 0
    return (
        alt.Chart(alt.Data(values=points))
        .mark_bar()
        .encode(
            x=alt.X("pnl:Q", title=base, axis=alt.Axis(format=pf_axis(base))),
            y=alt.Y("ticker:N", sort="-x", title=None),
            color=alt.when(up)
            .then(alt.value(display.UP_COLOR))
            .otherwise(alt.value(display.DOWN_COLOR)),
            tooltip=[
                alt.Tooltip("ticker:N", title="종목"),
                alt.Tooltip("pnl:Q", title="손익", format=pf_axis(base)),
            ],
        )
        .properties(height=max(CHART_HEIGHT // 2, 28 * len(points)))
    )


PF_DASH = "—"
PF_SHORT = "기간 부족"
PF_LUCK = (
    "운과 매매 시점이 함께 담긴 값이며 실력을 뜻하지 않습니다 · 무매매: 첫 거래일의"
    " 보유 수량을 그대로 두고, 이후 매수·매도 금액만큼 그 구성을 같은 비율로"
    " 사고판 경우"
)


def pf_annual(value: perf.Annualised | None) -> str:
    """The annual rate from a year on, else the cumulative return."""
    if value is None:
        return PF_DASH
    if value.annual is not None:
        return f"연 {portfolio.percent(value.annual, signed=True)}"
    return portfolio.percent(value.cumulative, signed=True)


def pf_gap(value: perf.Gap | None) -> str:
    if value is None:
        return PF_DASH
    return f"{'연 ' if value.annualised else ''}{portfolio.points(value.value)}"


def pf_benchmark_table(
    time: perf.Annualised | None, lines: dict[str, pd.Series]
) -> None:
    rows: list[tuple[str, perf.Annualised | None]] = [
        (name, portfolio.time_weighted(line)) for name, line in lines.items()
    ]
    pf_table(
        {
            "구분": ["포트폴리오 TWR", *(name for name, _ in rows)],
            "누적": [
                portfolio.percent(a.cumulative, signed=True) if a else PF_DASH
                for a in (time, *(a for _, a in rows))
            ],
            "연환산": [
                portfolio.percent(a.annual, signed=True)
                if a and a.annual is not None
                else PF_DASH
                for a in (time, *(a for _, a in rows))
            ],
            "TWR − 지수": [
                PF_DASH,
                *(
                    pf_gap(portfolio.gap(time, a)) if time and a else PF_DASH
                    for _, a in rows
                ),
            ],
        },
        "벤치마크 비교",
    )


def pf_split(split: perf.FxSplit) -> None:
    with st.container(horizontal=True):
        for label, value in (
            ("현지 수익률", split.local),
            ("환율", split.fx),
            ("교차항", split.cross),
            ("합계", split.base),
        ):
            st.metric(label, portfolio.percent(value, signed=True), border=True)
    st.caption(
        "보유분의 종가 기준 연결 수익률 · (1+합계) = (1+현지)(1+환율),"
        " 교차항 = 현지 × 환율"
    )


def pf_trading(
    money: perf.Annualised | None,
    time: perf.Annualised | None,
    result: portfolio.Trading | None,
    base: portfolio.Currency,
) -> None:
    timing = perf.timing_gap(money, time) if money and time else None
    with st.container(horizontal=True):
        st.metric(
            "타이밍 차이",
            pf_gap(timing),
            help="MWR − TWR: 넣고 뺀 금액의 시점이 수익률에 준 차이",
            border=True,
        )
        if result is not None:
            st.metric(
                "무매매 평가액",
                portfolio.money(result.held, base),
                portfolio.money(result.actual - result.held, base, signed=True),
                delta_color=pf_color(result.actual - result.held),
                delta_arrow="off",
                help="첫 거래일 구성 그대로였다면의 평가액 · 아래 숫자는 실제 − 무매매",
                border=True,
            )
            st.metric(
                "매매 효과",
                pf_gap(result.effect),
                help="실제 MWR − 무매매 MWR",
                border=True,
            )
    st.caption(PF_LUCK)


def pf_return_metrics(
    totals: portfolio.Totals,
    time: perf.Annualised | None,
    money: perf.Annualised | None,
    base: portfolio.Currency,
) -> None:
    with st.container(horizontal=True):
        cells = (
            (
                "평가손익",
                totals.unrealized,
                totals.open_return,
                "보유 종목의 원가 대비",
            ),
            (
                "실현손익",
                totals.realized,
                totals.realized_return,
                "매도한 주식의 원가 대비",
            ),
            (
                "총손익",
                totals.pnl,
                totals.total_return,
                "누적 매수금액(수수료 포함) 대비",
            ),
        )
        for label, amount, ratio, note in cells:
            st.metric(
                label,
                portfolio.money(amount, base, signed=True),
                portfolio.percent(ratio, signed=True),
                delta_color=pf_color(amount),
                delta_arrow="off",
                help=note,
                border=True,
            )
        st.metric(
            "TWR 누적",
            portfolio.percent(time.cumulative, signed=True) if time else PF_DASH,
            pf_annual(time) if time and time.annual is not None else None,
            delta_color="off",
            delta_arrow="off",
            help="시간가중수익률: 매수·매도 금액의 영향을 뺀 수익률 · 1년 이상이면"
            " 연환산도 표시",
            border=True,
        )
        st.metric(
            "MWR",
            pf_annual(money),
            help="금액가중수익률(XIRR) · 거래 기간이 1년 미만이면 연환산하지 않고"
            " 기간 수익률",
            border=True,
        )


def pf_realized(
    p: portfolio.Portfolio, prices: portfolio.Prices, items: list[portfolio.Holding]
) -> None:
    base = p.base
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown(f"**종목별 손익** · 실현 + 평가, {base}")
        points = portfolio.contribution_points(items)
        if points:
            st.altair_chart(pf_contribution_chart(points, base), alt="종목별 손익")
    with right, st.container(border=True):
        st.markdown(f"**연도별 실현손익** · {base}")
        sales = portfolio.ledger(p, prices.close).sales
        pf_period_table(sales, base, "연도", monthly=False)
        with st.expander("월별"):
            pf_period_table(sales, base, "월", monthly=True)
    sales = portfolio.ledger(p, prices.close).sales
    if sales:
        st.markdown(f"**실현손익 내역** · 매도 {len(sales)}건")
        pf_table(portfolio.realized_table(sales, base), "실현손익 내역")
    closed = portfolio.closed_table(items, base)
    if closed["종목"]:
        st.markdown("**전량 매도한 종목**")
        pf_table(closed, "전량 매도한 종목")


def pf_returns(p: portfolio.Portfolio, prices: portfolio.Prices) -> None:
    base = p.base
    pid = p.model_dump_json()
    items = portfolio.holdings(p, prices.close)
    totals = portfolio.totals(items)
    curve = cached_twr(pf_key(pid), p, prices.close)
    time = portfolio.time_weighted(curve)
    money = portfolio.money_weighted(p, prices.close, totals.value)
    pf_return_metrics(totals, time, money, base)
    days = pd.DatetimeIndex(curve.index)
    lines = cached_benchmarks(pf_key(pid, "bench"), p, prices.close, days)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown(f"**누적 수익률** · TWR와 가격지수, {base}")
        if len(curve) > 1:
            shown = {"포트폴리오": curve, **lines}
            st.altair_chart(
                pf_line_chart(portfolio.line_points(shown), "+.1%"), alt="누적 수익률"
            )
            pf_benchmark_table(time, lines)
            st.caption(
                "지수는 배당 제외 가격지수를 기준통화로 환산 · 혼합 벤치마크는 전날"
                " 종가의 한국·미국 비중으로 KOSPI와 S&P 500을 섞음"
            )
        else:
            st.caption(portfolio.UNKNOWN)
    with right, st.container(border=True):
        st.markdown(f"**누적 손익** · {base}")
        frame = cached_history(pf_key(pid), p, prices.close)
        if len(frame) > 1:
            points = portfolio.line_points(portfolio.pnl_lines(frame))
            st.altair_chart(pf_line_chart(points, pf_axis(base)), alt="누적 손익")
        else:
            st.caption(portfolio.UNKNOWN)
    split, result = cached_flows(pf_key(pid, "flows"), p, prices.close)
    if split is not None:
        st.markdown("**환율 효과**")
        pf_split(split)
    st.markdown("**매매 시점**")
    pf_trading(money, time, result, base)
    pf_realized(p, prices, items)


def pf_period_table(
    sales: tuple[portfolio.Sale, ...],
    base: portfolio.Currency,
    label: str,
    *,
    monthly: bool,
) -> None:
    by = portfolio.realized_by(sales, monthly=monthly)
    if not by:
        st.caption("매도 거래가 없습니다.")
        return
    pf_table(
        {
            label: [period for period, _ in by],
            "실현손익": [portfolio.money(v, base, signed=True) for _, v in by],
        },
        f"{label}별 실현손익",
    )


# ---- analysis


def pf_weight_chart(w: pd.Series, label: str = "종목") -> alt.Chart:
    points = [{"ticker": str(t), "weight": float(v)} for t, v in w.items()]
    return (
        alt.Chart(alt.Data(values=points))
        .mark_bar()
        .encode(
            x=alt.X("weight:Q", title=None, axis=alt.Axis(format=".0%")),
            y=alt.Y("ticker:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("ticker:N", title=label),
                alt.Tooltip("weight:Q", title="비중", format=".1%"),
            ],
        )
        .properties(height=max(CHART_HEIGHT // 2, 28 * len(points)))
    )


def pf_ratio_or(value: float | None, text: str = PF_DASH) -> str:
    return text if value is None else pf_ratio(value)


def pf_drawdown(stats: portfolio.Ratios) -> None:
    under = stats.under
    st.metric(
        "최대 낙폭",
        portfolio.percent(under.max_drawdown) if under else PF_DASH,
        f"{under.peak:%Y-%m-%d} → {under.trough:%Y-%m-%d}"
        if under and under.max_drawdown < 0
        else None,
        delta_color="off",
        delta_arrow="off",
        help="고점 → 저점 날짜"
        + (
            f" · 회복 {under.recovery:%Y-%m-%d} ({under.recovery_days}일)"
            if under and under.recovery is not None and under.max_drawdown < 0
            else " · 아직 회복 전"
            if under and under.max_drawdown < 0
            else ""
        ),
        border=True,
    )
    st.metric(
        "고점 아래 기간",
        f"{under.current_days:,}일" if under else PF_DASH,
        help=(
            f"마지막 고점 이후 달력일 (지금 고점이면 0) · 기간 중 가장 길었던"
            f" 수중 기간 {under.longest_days:,}일"
            if under
            else None
        ),
        border=True,
    )


def pf_sharpe(stats: portfolio.Ratios) -> None:
    ci = stats.ci
    st.metric(
        "샤프 지수",
        pf_ratio_or(stats.sharpe),
        f"95% {pf_ratio(ci.low)} ~ {pf_ratio(ci.high)}" if ci else PF_SHORT,
        delta_color="off",
        delta_arrow="off",
        help=f"연환산 · 신뢰구간은 {perf.MIN_OBS}일 이상에서 (Mertens 표준오차)",
        border=True,
    )
    enough = stats.days >= perf.MIN_OBS
    st.metric(
        "PSR",
        portfolio.percent(stats.psr) if stats.psr is not None else PF_SHORT,
        help="참 샤프 지수가 0보다 클 확률의 추정치 (왜도·첨도 반영)",
        border=True,
    )
    trl = stats.min_trl
    st.metric(
        "MinTRL",
        PF_SHORT if not enough else (f"{trl:,.0f}거래일" if trl else PF_DASH),
        help="PSR 95%에 필요한 최소 기록 길이 · 샤프 지수가 0 이하면 —",
        border=True,
    )


def pf_advanced(stats: portfolio.Ratios, bench: str) -> None:
    year = "1년 미만"
    with st.container(horizontal=True):
        st.metric("소르티노", pf_ratio_or(stats.sortino), help="연환산", border=True)
        st.metric(
            "칼마", pf_ratio_or(stats.calmar, year), help="1년 이상만", border=True
        )
        st.metric(
            "얼서 지수", f"{stats.ulcer:.2f}", help="낙폭의 제곱평균(%)", border=True
        )
        st.metric("UPI", pf_ratio_or(stats.upi, year), help="1년 이상만", border=True)
        st.metric("오메가", pf_ratio_or(stats.omega), border=True)
        st.metric("CDaR 95%", portfolio.percent(stats.cdar), border=True)
        st.metric(
            "추적오차",
            portfolio.percent(stats.te) if stats.te is not None else PF_SHORT,
            help=f"{bench} 대비, 연환산",
            border=True,
        )
        st.metric(
            "정보비율",
            pf_ratio_or(stats.ir, PF_SHORT),
            help=f"{bench} 대비",
            border=True,
        )


def pf_stats(s: Subject, prices: portfolio.Prices, months: int) -> None:
    """Risk-adjusted statistics of ``s.rets`` over the window."""
    rets = portfolio.window(s.rets, months)
    rates, source = pf_rates(s.base, prices.close)
    rf = portfolio.risk_free(rates, pd.DatetimeIndex(rets.index))
    key = (*s.key, months, source, pf_rates_key(rates))
    bench = partial(s.bench, pd.DatetimeIndex(s.rets.index))
    stats = cached_ratios(key, rets, rf, bench)
    with st.container(horizontal=True):
        st.metric(
            "연 변동성",
            portfolio.percent(stats.volatility)
            if stats.volatility is not None
            else PF_DASH,
            border=True,
        )
        pf_drawdown(stats)
        pf_sharpe(stats)
        if not portfolio.korean_weight(s.w):  # Seoul closes: see the Dimson beta
            beta = portfolio.market_beta(rets, prices.adjusted, s.base)
            st.metric(
                "베타",
                pf_ratio(beta),
                help=f"{portfolio.BENCHMARK_LABEL} 대비 Welch 베타, {s.base} 일간",
                border=True,
            )
    rate_note = (
        f"무위험수익률 {source}"
        if not rates.empty
        else "무위험수익률 자료가 없어 0으로 계산"
    )
    st.caption(
        f"{s.label}의 일간 수익률 {stats.days}일 (수정 종가, {s.base}) · {rate_note}"
        f" · {perf.MIN_OBS}일 미만은 {PF_SHORT}"
    )
    more = st.expander("고급", key="pf_advanced", on_change="rerun")
    if more.open:
        with more:
            pf_advanced(stats, s.bench_label)


def pf_analysis(p: portfolio.Portfolio, prices: portfolio.Prices, months: int) -> None:
    s = pf_subject(p, prices)
    pf_stats(s, prices, months)
    left, right = st.columns([1, 2])
    with left, st.container(border=True):
        st.markdown("**비중**")
        if s.w.empty:
            st.caption("보유 중인 종목이 없습니다.")
        else:
            st.altair_chart(pf_weight_chart(s.w), alt="비중")
    with right, st.container(border=True):
        st.markdown(f"**평가액과 순투입** · {p.base}")
        frame = cached_history(pf_key(p.model_dump_json()), p, prices.close)
        lines = portfolio.value_lines(frame)
        if len(frame) > 1:
            st.altair_chart(
                pf_line_chart(portfolio.line_points(lines), pf_axis(p.base)),
                alt="평가액과 순투입",
            )
        else:
            st.caption(portfolio.UNKNOWN)
    pf_style_box(s, prices, "pf_style")


def pf_style_box(s: Subject, prices: portfolio.Prices, key: str) -> None:
    box = st.expander("스타일 분석", key=key, on_change="rerun")
    if box.open and not s.w.empty:
        with box:
            pf_style_fit(s, prices)


PF_STYLE_NOTE = "과거 움직임이 비슷했던 조합이며 실제 보유나 추천이 아닙니다"


def pf_style_fit(s: Subject, prices: portfolio.Prices) -> None:
    """Fetches the proxies only when asked; computes on what is saved."""
    absent = [t for t in portfolio.STYLE_PROXIES if t not in prices.tickers]
    fetcher = portfolio_fetcher(str(pf_root()))
    can = bool(absent) and not args.demo and not portfolio.missing_packages()
    if can and st.button(
        "비교 지수 받기",
        icon=":material/download:",
        key="pf_style_fetch",
        disabled=fetcher.progress.running,
        help=", ".join(absent),
    ):
        fetcher.start(absent)
        st.rerun()
    w, _, _ = pf_weights(s, prices, PF_LONG_MONTHS)
    fit = cached_style((*s.key, "style"), s.base, prices.adjusted, w)
    if fit is None:
        st.caption(
            f"지금 비중의 주간 수익률이 {portfolio.WEEKS}주 이상이고 비교 지수가"
            " 있어야 합니다."
        )
        return
    shown = pd.Series(fit.weights, index=list(fit.names), dtype="float64")
    shown = cast("pd.Series", shown[shown > 0.0005])  # noqa: PLR2004 - rounding dust
    st.altair_chart(pf_weight_chart(shown, "지수"), alt="스타일 비중")
    r2 = f"{fit.r2:.0%}" if fit.r2 is not None else PF_DASH
    notes = [
        f"{PF_STYLE_NOTE} · 최근 {fit.weeks}주 {s.base} 주간 수익률 · 설명력 R² {r2}"
    ]
    if fit.selection is not None:
        notes.append(
            f"조합 대비 차이(표본 외, {fit.selection_weeks}주 평균) 연"
            f" {portfolio.percent(fit.selection, signed=True)}"
        )
    if fit.dropped:
        notes.append(f"기간을 채우지 못해 뺀 지수: {', '.join(fit.dropped)}")
    st.caption(" · ".join(notes))


# ---- correlation


def pf_heatmap(matrix: pd.DataFrame) -> alt.LayerChart:
    order = portfolio.labels(matrix)
    base = alt.Chart(alt.Data(values=portfolio.matrix_points(matrix))).encode(
        x=alt.X("a:N", sort=order, title=None),
        y=alt.Y("b:N", sort=order, title=None),
    )
    cells = base.mark_rect().encode(
        color=alt.Color(
            "value:Q",
            title=None,
            scale=alt.Scale(domain=[-1, 1], scheme="redblue", reverse=True),
        ),
        tooltip=[
            alt.Tooltip("a:N", title="가로"),
            alt.Tooltip("b:N", title="세로"),
            alt.Tooltip("value:Q", title="상관계수", format=".2f"),
        ],
    )
    labels = base.mark_text().encode(text=alt.Text("value:Q", format=".2f"))
    size = max(CHART_HEIGHT, 44 * len(order))
    return alt.LayerChart(layer=[cells, labels]).properties(height=size)


def pf_joint(
    s: Subject, prices: portfolio.Prices, months: int
) -> tuple[portfolio.Joint | None, float, list[str]]:
    """``s``'s holdings' joint window returns (cached), their weight, the dropped."""
    w, share, dropped = pf_weights(s, prices, months)
    if w.empty:
        return None, share, dropped
    key = (*s.key, months, "joint")
    return cached_joint(key, s.base, months, prices.adjusted, w), share, dropped


def pf_dropped(dropped: list[str]) -> None:
    if dropped:
        more = " 외" if len(dropped) > PF_HEAT else ""
        st.caption(
            f"기간 중 자료가 시작하거나 끝나 뺀 {len(dropped)}종목:"
            f" {', '.join(dropped[:PF_HEAT])}{more}"
        )


def pf_correlation(
    s: Subject,
    prices: portfolio.Prices,
    months: int,
    compare: Callable[[], None],
) -> None:
    """Holdings' correlation, betas and spread; ``compare`` fills the right panel."""
    j, _, dropped = pf_joint(s, prices, months)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown(f"**보유 종목** · {s.base} 일간 수익률")
        if j is None or len(j.names) < 2 or len(j.window) < portfolio.MIN_DAYS:  # noqa: PLR2004
            st.caption("수익률이 있는 보유 종목이 둘 이상이어야 합니다.")
        else:
            cov, f, shrink = cached_covariance((*s.key, months, "cov"), j)
            shown = pf_largest(j)
            full = portfolio.correlation_frame(cov, j.names)
            matrix = cast("pd.DataFrame", full.loc[shown, shown])
            st.altair_chart(pf_heatmap(matrix), alt="종목 상관")
            notes = [f"{len(j.window)}일 · 모든 종목 값이 있는 날"]
            if len(shown) < len(j.names):
                notes.append(f"비중 상위 {len(shown)}종목")
            if j.mixed:
                notes.append(f"한국·미국 종가 시차 보정 (전날 공분산 × {f:.2f})")
            if len(j.names) > 2:  # noqa: PLR2004 - two names: the target is the sample
                notes.append(f"Ledoit–Wolf 축소 {shrink:.0%}")
            if j.late:
                notes.append(f"자료 시작: {j.late}")
            st.caption(" · ".join(notes))
            pf_dropped(dropped)
    with right, st.container(border=True):
        compare()
    if j is None or len(j.window) < portfolio.MIN_DAYS:
        return
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown(f"**베타** · {s.base} 일간 수익률")
        shown = set(pf_largest(j))
        rows = portfolio.betas(j)
        pf_betas(
            [b for b in rows if b.name in shown or b.name == portfolio.PORTFOLIO_LABEL]
        )
    with right, st.container(border=True):
        st.markdown("**분산** · 지금 비중")
        cov, _, _ = cached_covariance((*s.key, months, "cov"), j)
        pf_spread(j, portfolio.diversification(j, cov))


def pf_largest(j: portfolio.Joint) -> list[str]:
    """The ``PF_HEAT`` largest holdings, in ``j``'s order."""
    if len(j.names) <= PF_HEAT:
        return list(j.names)
    cut = sorted(float(v) for v in j.w)[-PF_HEAT]
    return [n for n, v in zip(j.names, j.w, strict=True) if v >= cut][:PF_HEAT]


def pf_book_heatmap(
    book: portfolio.Book,
    prices: portfolio.Prices,
    base: portfolio.Currency,
    months: int,
) -> None:
    st.markdown(f"**포트폴리오와 {portfolio.BENCHMARK_LABEL}** · {base}")
    key = pf_key(book.model_dump_json())
    frame, korean = cached_lineup(key, base, months, book, prices)
    pf_compare_heatmap((*key, base, months), frame, korean)
    st.caption("포트폴리오마다 지금 비중으로 계산 · 보유 종목이 없으면 빠짐")


def pf_betas(rows: list[portfolio.Beta]) -> None:
    pf_table(
        {
            "구분": [b.name for b in rows],
            f"{portfolio.BENCHMARK_LABEL} (Welch)": [pf_ratio(b.welch) for b in rows],
            "Dimson": [
                pf_ratio(b.dimson) if math.isfinite(b.dimson) else PF_DASH for b in rows
            ],
            "KOSPI": [
                pf_ratio(b.kospi) if math.isfinite(b.kospi) else PF_DASH for b in rows
            ],
        },
        "베타",
    )
    st.caption(
        "Welch: 하루 수익률을 지수의 −2~4배로 제한하고 최근 날에 더 무게 ·"
        " Dimson: 한국 종목은 같은 날과 전날 S&P 500에 대한 기울기의 합 · KOSPI"
        " 베타는 한국 종목만"
    )


def pf_spread(j: portfolio.Joint, spread: portfolio.Spread) -> None:
    with st.container(horizontal=True):
        st.metric(
            "분산 비율",
            pf_ratio(spread.ratio),
            help="종목 변동성의 가중합 ÷ 포트폴리오 변동성 (1이면 분산 효과 없음)",
            border=True,
        )
        st.metric(
            "유효 종목 수",
            f"{spread.effective:.1f}",
            help="1 ÷ 비중 제곱합 (같은 비중 N종목이면 N)",
            border=True,
        )
    pf_table(
        {
            "종목": list(j.names),
            "비중": [portfolio.percent(float(v)) for v in j.w],
            "위험 기여": [portfolio.percent(float(v)) for v in spread.risk],
        },
        "위험 기여",
    )
    st.caption("위험 기여: 포트폴리오 분산 중 각 종목의 몫 (오일러 분해, 합 100%)")


def pf_compare_heatmap(key: Key, frame: pd.DataFrame, korean: list[bool]) -> None:
    """Whole portfolios' daily correlation, lag-adjusted and shrunk.

    ``key`` stands for ``frame``.
    """
    found = cached_lagged((*key, "compare"), tuple(korean), frame)
    if found is None:
        st.caption(
            f"모두 값이 있는 날이 {portfolio.MIN_DAYS}일 이상인 수익률이 둘 이상이어야"
            " 합니다."
        )
        return
    st.altair_chart(pf_heatmap(found.matrix), alt="포트폴리오 상관")
    notes = [f"{found.days}일 · 모두 값이 있는 날, 일간 수익률"]
    if any(korean):
        notes.append(
            f"한국 비중이 절반을 넘는 포트폴리오는 한국 종가로 보고 시차 보정"
            f" (전날 공분산 × {found.lag:.2f})"
        )
    if len(found.matrix) > 2:  # noqa: PLR2004 - two series: the target is the sample
        notes.append(f"Ledoit–Wolf 축소 {found.shrink:.0%}")
    st.caption(" · ".join(notes))


# ---- risk


def pf_fan_chart(
    rows: list[dict[str, float]], base: portfolio.Currency
) -> alt.LayerChart:
    number = pf_axis(base)
    chart = alt.Chart(alt.Data(values=rows)).encode(x=alt.X("day:Q", title="거래일"))
    wide = chart.mark_area(opacity=0.18).encode(
        y=alt.Y("p5:Q", title=base, axis=alt.Axis(format=number)), y2="p95:Q"
    )
    mid = chart.mark_area(opacity=0.35).encode(y="p25:Q", y2="p75:Q")
    median = chart.mark_line().encode(
        y="p50:Q",
        tooltip=[
            alt.Tooltip("day:Q", title="거래일"),
            *(
                alt.Tooltip(f"p{q}:Q", title=f"{q}%", format=number)
                for q in portfolio.PERCENTILES
            ),
        ],
    )
    return alt.LayerChart(layer=[wide, mid, median]).properties(height=CHART_HEIGHT)


def pf_end_chart(
    rows: list[dict[str, float]], value: float, base: portfolio.Currency
) -> alt.LayerChart:
    number = pf_axis(base)
    bars = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("start:Q", title=base, axis=alt.Axis(format=number)),
            x2="end:Q",
            y=alt.Y("count:Q", title="경로 수"),
            tooltip=[
                alt.Tooltip("start:Q", title="부터", format=number),
                alt.Tooltip("end:Q", title="까지", format=number),
                alt.Tooltip("count:Q", title="경로 수"),
            ],
        )
    )
    now = (
        alt.Chart(alt.Data(values=[{"x": value}]))
        .mark_rule(strokeDash=[3, 3], color="gray")
        .encode(x="x:Q")
    )
    return alt.LayerChart(layer=[bars, now]).properties(height=CHART_HEIGHT)


PF_THIN = "표본 부족"


def pf_paths(
    key: Key,
    j: portfolio.Joint,
    value: float,
    base: portfolio.Currency,
    held: pd.Series,
) -> None:
    """Monte Carlo of today's weights; the draws are shared across weights.

    ``key`` stands for ``j`` and ``held``.
    """
    with st.container(horizontal=True, vertical_alignment="bottom"):
        picked = st.segmented_control(
            "기간",
            list(portfolio.HORIZONS),
            default="1M",
            required=True,
            key="pf_horizon",
        )
        fhs = st.toggle("현재 변동성 반영 (FHS)", key="pf_fhs")
    label = picked or "1M"
    days = portfolio.HORIZONS[label]
    block = portfolio.mean_block(j)
    n = len(j.window)
    picks = cached_draws(state.get("pf_prices_stamp", ()), n, days, block)
    standing = portfolio.standing(held)
    sim = cached_paths((*key, days, block), fhs, value, standing, j, picks)
    ten, twenty = (sim.losses[level] for level in portfolio.LOSS_LEVELS)
    st.markdown(
        f"**{label} 뒤** · 모의 경로 100개 중 약 {round(ten * 100)}개가 10% 이상,"
        f" {round(twenty * 100)}개가 20% 이상 손실"
    )
    with st.container(horizontal=True):
        st.metric("10% 이상 손실 경로 비율", portfolio.percent(ten), border=True)
        st.metric("20% 이상 손실 경로 비율", portfolio.percent(twenty), border=True)
        for level, odds in sim.paths.drawdown_odds.items():
            st.metric(
                f"기간 중 −{level:.0%} 낙폭",
                portfolio.percent(odds),
                help="기간 안에 지금 또는 그 뒤 고점에서 이만큼 넘게 떨어진 경로 비율",
                border=True,
            )
        pf_recovery(sim.paths)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**시뮬레이션 범위** · 5–95%, 25–75%, 중앙값")
        st.altair_chart(pf_fan_chart(sim.fan, base), alt="범위")
    with right, st.container(border=True):
        st.markdown(f"**{label} 뒤 평가액 분포** · 점선은 지금")
        st.altair_chart(pf_end_chart(sim.ends, value, base), alt="분포")
    first = cast("pd.Timestamp", j.window.index[0]).date()
    last = cast("pd.Timestamp", j.window.index[-1]).date()
    st.caption(
        f"평균 수익 0 가정 · 과거 {n}거래일({first}–{last})의 같은 날 수익률을 평균"
        f" {block}일 묶음으로 재표본(경로 {portfolio.PATHS:,}개) · 지금 비중으로 매일"
        " 재조정"
        + (" · FHS: 지금 변동성에서 시작해 장기 수준으로 돌아감" if fhs else "")
        + " · 예측이 아닙니다"
    )


def pf_recovery(paths: perf.PathRisk) -> None:
    if paths.never == 0 and paths.recovery_median is None:
        st.metric("전고점 회복", "지금이 고점", border=True)
        return
    middle = paths.recovery_median
    low, high = paths.recovery_low, paths.recovery_high
    st.metric(
        "전고점 회복 (중앙값)",
        f"{middle}거래일" if middle is not None else "기간 내 미회복",
        f"10–90%: {low if low is not None else '—'}–"
        f"{high if high is not None else '기간 밖'}",
        delta_color="off",
        delta_arrow="off",
        help=f"기간 안에 회복하지 못한 경로 {portfolio.percent(paths.never)}",
        border=True,
    )


def pf_loss(value: float | None) -> float | None:
    """A loss for display: a gain shows as 0."""
    return None if value is None else max(value, 0.0)


def pf_tail_table(t: portfolio.Tails, value: float, base: portfolio.Currency) -> None:
    def pct(x: float | None) -> str:
        loss = pf_loss(x)
        return PF_THIN if loss is None else portfolio.percent(loss, dec=2)

    def amount(x: float | None) -> str:
        loss = pf_loss(x)
        return PF_THIN if loss is None else portfolio.money(loss * value, base)

    pf_table(
        {
            "방법": [r.method for r in t.rows],
            "신뢰수준": [f"{r.level:.1%}".replace(".0%", "%") for r in t.rows],
            "VaR": [pct(r.var) for r in t.rows],
            f"VaR ({base})": [amount(r.var) for r in t.rows],
            "ES": [pct(r.es) for r in t.rows],
            f"ES ({base})": [amount(r.es) for r in t.rows],
        },
        "VaR·ES",
    )
    st.caption(
        "1일 손실 · 최소 표본: 95% 250일, 97.5% 400일, 99% 1,000일 · FHS는 종목마다"
        " 과거 수익률을 그날 변동성으로 나눈 뒤 지금 변동성을 곱함 · 이익이면 0"
    )


def pf_tails(
    key: Key, j: portfolio.Joint, value: float, base: portfolio.Currency
) -> None:
    """``key`` stands for ``j``."""
    t = cached_tails((*key, "tails"), j)
    head = portfolio.tail_of(t, "역사적", quant.TAIL_LEVEL)
    es = pf_loss(head.es)
    with st.container(horizontal=True):
        st.metric(
            "ES 97.5% (1일)",
            portfolio.money(es * value, base) if es is not None else PF_THIN,
            portfolio.percent(es, dec=2) if es is not None else None,
            delta_color="off",
            delta_arrow="off",
            help=(
                f"90% 범위 {portfolio.percent(max(t.band[0], 0), dec=2)}–"
                f"{portfolio.percent(max(t.band[1], 0), dec=2)} (재표본)"
                if t.band
                else "역사적 기준, 최악 2.5% 날의 평균 손실"
            ),
            border=True,
        )
        month = t.month
        for label, part in (("겹치는 21일", 0), ("√21 환산", 1)):
            loss = pf_loss(month[part]) if month else None
            st.metric(
                f"1개월 VaR 95% · {label}",
                portfolio.percent(loss) if loss is not None else PF_THIN,
                border=True,
            )
    if month and min(month) > 0 and max(month) / min(month) > portfolio.WIDE_MONTH:
        st.caption(
            "1개월 VaR의 두 계산이 크게 다릅니다 · 겹치는 21일은 독립 표본이"
            f" {t.days // portfolio.MONTH_DAYS}개 남짓이고, √21 환산은 날마다"
            " 독립이라고 가정합니다"
        )
    pf_tail_table(t, value, base)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**사후검정** · 95% VaR, 최근 250일")
        pf_backtest(t)
    if len(j.names) == 1:  # one holding: its ES share is the whole ES
        return
    with right, st.container(border=True):
        st.markdown("**종목별 ES 기여** · 97.5%")
        if t.component is None:
            st.caption(PF_THIN)
        else:
            pf_table(
                {
                    "종목": list(j.names),
                    "ES 기여": [
                        portfolio.percent(float(c), dec=2) for c in t.component
                    ],
                    f"({base})": [
                        portfolio.money(float(c) * value, base) for c in t.component
                    ],
                },
                "종목별 ES",
            )
            st.caption("포트폴리오가 가장 나빴던 2.5% 날에 각 종목이 낸 평균 손실")


PF_LIGHTS: dict[str, tuple[str, Literal["green", "orange", "red"]]] = {
    "green": ("녹색", "green"),
    "yellow": ("황색", "orange"),
    "red": ("적색", "red"),
}


def pf_backtest(t: portfolio.Tails) -> None:
    bt = t.backtest
    if bt is None:
        st.caption(f"{PF_THIN} (전체 기간 수익률 {t.history}일, 500일 이상 필요)")
        return
    name, color = PF_LIGHTS[bt.light]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.metric("초과 횟수", f"{bt.exceptions}회", f"기대 {bt.expected:.1f}회",
                  delta_color="off", delta_arrow="off", border=True)  # fmt: skip
        st.metric("Kupiec p", f"{bt.kupiec:.2f}", border=True)
        st.metric("Christoffersen p", f"{bt.christoffersen:.2f}", border=True)
        st.badge(f"신호등 {name}", color=color)
    st.caption(
        "최근 250일 중 직전 250일로 구한 VaR보다 손실이 컸던 날 · p가 작으면 초과 빈도"
        "(Kupiec)나 몰림(Christoffersen)이 95% 가정과 맞지 않는다는 뜻"
    )


def pf_risk(s: Subject, prices: portfolio.Prices, months: int) -> None:
    j, share, dropped = pf_joint(s, prices, months)
    if j is None:
        st.info("가격이 확인된 보유 종목이 없습니다.")
        return
    value = s.value * share
    st.caption(
        f"과거 일간 수익률을 바탕으로 한 모의 결과이며 예측이 아닙니다 · 지금 비중,"
        f" {len(j.window)}일"
    )
    for note in s.notes:
        st.caption(note)
    pf_dropped(dropped)
    if j.late:
        st.caption(f"자료 시작이 늦어 모든 종목에 값이 있는 날만 씀: {j.late}")
    if len(j.window) < portfolio.MIN_DAYS:
        st.info(f"모든 보유 종목의 수익률이 있는 날이 {len(j.window)}일뿐입니다.")
        return
    key = (*s.key, months)
    pf_paths(key, j, value, s.base, s.rets)
    st.markdown("**VaR·ES** · 지금 비중, 1일")
    pf_tails(key, j, value, s.base)


# ---- stress


def pf_scenario_table(rows: list[portfolio.Replayed], base: portfolio.Currency) -> None:
    def note(r: portfolio.Replayed) -> str:
        parts: list[str] = []
        if r.window is None:
            parts.append("지수 자료 없음")
        if r.scenario.us_only:
            parts.append("미국 지수만")
        if r.no_fx:
            parts.append("환율 미반영")
        if r.proxied:
            more = len(r.proxied) - 3
            named = [
                f"{t} β{b:.2f}" for t, b in zip(r.proxied, r.proxy_betas, strict=False)
            ] or list(r.proxied)
            shown = ", ".join(named[:3]) + (f" 외 {more}종목" if more > 0 else "")
            parts.append(f"지수×베타로 대체: {shown}")
        return " · ".join(parts) or PF_DASH

    pf_table(
        {
            "시나리오": [r.scenario.name for r in rows],
            "구간": [
                f"{r.window[0]} → {r.window[1]}" if r.window else PF_DASH for r in rows
            ],
            "구간 끝": [portfolio.percent(r.change, signed=True) for r in rows],
            "최악": [portfolio.percent(r.worst, signed=True) for r in rows],
            "환율 몫": [portfolio.points(r.fx) for r in rows],
            f"평가액 변화 ({base})": [
                portfolio.money(r.pnl, base, signed=True) for r in rows
            ],
            portfolio.BENCHMARK_LABEL: [
                portfolio.percent(r.sp500, signed=True) for r in rows
            ],
            "KOSPI": [portfolio.percent(r.kospi, signed=True) for r in rows],
            "비고": [note(r) for r in rows],
        },
        "과거 구간 재현",
    )


def pf_shocks(
    values: pd.Series,
    betas: NDArray[np.float64],
    base: portfolio.Currency,
    holding: str,
) -> None:
    total = float(cast("float", values.sum()))
    rows = portfolio.shocked(values, betas, base)

    def moves(s: portfolio.Shock) -> str:
        parts = [
            f"{name} {portfolio.percent(move, signed=True, dec=0)}"
            for name, move in (
                ("KOSPI", s.kospi),
                (portfolio.BENCHMARK_LABEL, s.sp500),
                ("원/달러", s.usdkrw),
            )
            if move
        ]
        return ", ".join(parts)

    pf_table(
        {
            "충격": [s.name for s, _ in rows],
            "가정": [moves(s) for s, _ in rows],
            holding: [portfolio.percent(v, signed=True) for _, v in rows],
            f"평가액 변화 ({base})": [
                portfolio.money(v * total, base, signed=True) for _, v in rows
            ],
        },
        "가상 충격",
    )
    fall = portfolio.reverse(values, betas)
    st.metric(
        f"−{portfolio.REVERSE_LOSS:.0%} 손실에 필요한 지수 하락폭",
        portfolio.percent(fall) if fall is not None else "−100%에서도 미도달",
        help="KOSPI와 S&P 500이 같은 폭으로 떨어지고 환율은 그대로라고 볼 때",
        border=True,
    )
    st.caption(
        "종목마다 자기 지수(한국 KOSPI, 그 밖 S&P 500)에 대한 현지 통화 베타(Welch)를"
        " 1 쪽으로 n/(n+250)만큼 당겨 씀 · 지수 등락 × 베타 뒤 환율 반영"
    )


def pf_turbulence(s: Subject, prices: portfolio.Prices, names: tuple[str, ...]) -> None:
    now = cached_turbulence((*s.key, "turbulence"), s.base, names, prices.adjusted)
    if now is None:
        st.caption(
            f"보유 종목이 둘 이상이고 모두 값이 있는 주가 {portfolio.WEEKS}주를 넘어야"
            " 합니다."
        )
        return
    with st.container(horizontal=True):
        st.metric(
            "이번 주 이례 정도",
            f"{now.level:.2f}",
            f"과거 {now.weeks:,}주 중 {portfolio.percent(now.rank, dec=0)} 수준 이하",
            delta_color="off",
            delta_arrow="off",
            help="1이면 평소 수준",
            border=True,
        )
        st.metric(
            "상관 이례도",
            f"{now.surprise:.2f}",
            f"백분위 {portfolio.percent(now.surprise_rank, dec=0)}"
            if math.isfinite(now.surprise_rank)
            else None,
            delta_color="off",
            delta_arrow="off",
            help="종목끼리 함께 움직이는 방식이 평소와 다른 정도 (1이면 평소)",
            border=True,
        )
    st.caption(
        f"{now.week} 주 · 보유 종목 주간 수익률을 직전 {portfolio.WEEKS}주의 평균·"
        "공분산과 견준 거리 (Kritzman–Li) · 과거 기준의 설명이며 앞날을 말하지 않습니다"
    )


def pf_stress_section(s: Subject, prices: portfolio.Prices, months: int) -> None:
    values = {str(t): float(v) * s.value for t, v in s.w.items()}
    if not values:
        st.info("가격이 확인된 보유 종목이 없습니다.")
        return
    older = pf_dashboard_series(PF_DEXKOUS)
    fx = portfolio.long_fx(prices.adjusted, older)
    st.markdown(f"**과거 구간 재현** · {s.holding} · 각 구간 첫날부터 그대로 보유")
    for note in s.notes:
        st.caption(note)
    key = (*s.key, "replay", len(older))
    rows = cached_replays(key, s.base, values, prices.adjusted, fx)
    pf_scenario_table(rows, s.base)
    notes = [
        "구간: 지수마다 적힌 고점·저점 ±90일에서 찾은 고점 중 이른 날 → 저점 중"
        " 늦은 날 · 최악: 구간 안 고점 대비 가장 깊은 하락 · 환율 몫:"
        " (1+구간 끝)/(1+현지 통화 기준) − 1"
    ]
    if any(r.window is None for r in rows):
        notes.append(
            "지수 자료 없음: 가격 갱신 때 1987년부터의 지수·환율 이력을 한 번 받음"
        )
    if any(r.no_fx for r in rows):
        notes.append("환율 미반영: 2003년 이전 원/달러는 FRED DEXKOUS(키 필요)")
    st.caption(" · ".join(notes))
    names = tuple(portfolio.labels(pd.Series(values)))
    bad = cached_bad_prints((*s.key, "bad"), names, prices.adjusted)
    if bad:
        listed = ", ".join(
            f"{t} ({', '.join(str(d) for d in days[:3])})" for t, days in bad.items()
        )
        st.caption(f"튀었다 바로 돌아온 가격 (계산에 그대로 포함): {listed}")
    st.markdown(f"**가상 충격** · {s.holding}")
    betas = cached_betas((*s.key, "betas"), months, names, prices.adjusted)
    pf_shocks(
        pd.Series(values, dtype="float64").reindex(list(names)),
        betas,
        s.base,
        s.holding,
    )
    if len(names) < 2:  # noqa: PLR2004 - one holding: a squared z-score
        return
    box = st.expander("주간 난기류", key="pf_turbulence", on_change="rerun")
    if box.open:
        with box:
            long, _, _ = pf_weights(s, prices, PF_LONG_MONTHS)
            pf_turbulence(s, prices, tuple(portfolio.labels(long)))


# ---- method


def pf_method(p: portfolio.Portfolio, prices: portfolio.Prices, window: str) -> None:
    rate, day = portfolio.last_close(prices.close, portfolio.FX)
    fx = f"₩{rate:,.2f} ({day})" if day else portfolio.UNKNOWN
    base, fill = p.base, portfolio.FILL_LIMIT
    paths, seed = portfolio.PATHS, portfolio.SEED
    lines = (
        f"기준통화 {base} · 환율 {portfolio.FX} 최근 {fx} · 외화 거래는 거래일 환율",
        "원가: 이동평균법. 매수 수수료는 원가에 더하고 매도 수수료는 매도금액에서 뺌",
        "평가·손익: 입력한 거래가격과 종가(분할 반영, 배당 미반영). 종가일이 오늘이면"
        " 장중 시세일 수 있음",
        "총수익률 = 총손익 ÷ 누적 매수금액(수수료 포함) · TWR·MWR(XIRR)은 기간이"
        " 1년 미만이면 연환산하지 않고 기간 수익률",
        "TWR: 같은 날 매수·매도는 상계합니다. 순유입은 그날 시작부터, 순유출은"
        " 그날 끝에 반영합니다. 일간 수익률을 이어 붙임 (배당 제외)",
        "벤치마크: 배당 제외 가격지수(S&P 500, KOSPI)를 기준통화로 환산. 혼합"
        " 벤치마크는 전날 종가의 한국·미국 비중으로 두 지수를 섞음",
        "환율 효과: 보유분의 종가 수익률을 현지·환율·교차항으로 나눔"
        " ((1+기준) = (1+현지)(1+환율), 교차항 = 현지 × 환율)",
        "타이밍 차이 = MWR − TWR · 무매매: 첫 거래일 보유 수량을 그대로 두고 이후"
        " 매수·매도 금액만큼 그 구성을 사고판 경우 · 매매 효과 = 실제 MWR − 무매매"
        " MWR. 운과 시점이 섞인 값",
        "보유일: 수량이 0에서 다시 산 첫 매수일부터",
        f"위험 지표·상관·시뮬레이션: 수정 종가(배당·분할 반영) 일간 수익률, {window},"
        f" 연 {portfolio.TRADING_DAYS}거래일",
        "무위험수익률: 원화는 CD 91일(한국은행 ECOS), 달러는 미국 3개월 국채(^IRX)."
        " 자료가 없으면(데모, 키 없음) 0 · 첫 금리 전의 날은 뺌",
        f"샤프 지수 95% 신뢰구간(Mertens 2002 표준오차), PSR·MinTRL(Bailey–López de"
        f" Prado 2012, 왜도·첨도 반영): {perf.MIN_OBS}일 미만은 {PF_SHORT} ·"
        " 소르티노는 모든 날의 하방편차 · 칼마·UPI는 1년 이상 · CDaR 95%: 가장 깊은"
        " 5% 낙폭의 평균 · 추적오차·정보비율: 혼합 벤치마크 대비",
        f"휴장일 정렬에만 직전 종가를 최대 {fill}행 이어 씀. 더 긴 공백은 미확인",
        "상관: 모든 보유 종목 값이 있는 날. 한국 종가는 같은 날 미국 개장 전이라"
        " 한국 종목과 전날 미국 종목의 공분산을 더하고(행렬이 성립하는 만큼),"
        " Ledoit–Wolf 2004 상수상관 축소 · 포트폴리오끼리는 지금 비중의 일간"
        " 수익률(기간 중 자료가 시작하거나 끝난 종목은 빼고 재조정)로 같은 방법,"
        " 한국 비중이 절반을 넘는 포트폴리오는 한국 종가로 봄",
        "베타: Welch(하루 수익률을 지수의 −2~4배로 제한, 최근 날 가중), 한국 종목은"
        " Dimson 1979(같은 날 + 전날 S&P 500)와 KOSPI 베타도 · 분산: 오일러 위험"
        " 기여, 분산 비율, 유효 종목 수(1/HHI)",
        f"몬테카를로: Politis–Romano 1994 정상 부트스트랩. 같은 날의 종목 수익률을"
        f" 함께, 평균 n^(1/3)일 묶음(5~20일, 한국·미국 혼합은 10일 이상, n/10 이하)."
        f" 경로 {paths:,}개, 시드 {seed}, 평균 수익 0, 지금 비중으로 매일 재조정 ·"
        " FHS: EWMA(λ 0.94)로 나눈 잔차에 지금 변동성을 곱하고 하루 0.985씩 장기"
        " 수준으로 돌아감 · 낙폭·회복은 기간 안에서",
        "VaR·ES: 역사적·FHS, 95%·97.5%·99%, 최소 250·400·1,000일(아니면 표본"
        " 부족) · ES 97.5% 범위는 부트스트랩 1,000회의 5–95% · 1개월 VaR은 겹치는"
        " 21일과 √21 환산 둘 다 · 종목별 ES 기여(Tasche) · 이익이면 0",
        "사후검정: 최근 250일, 날마다 직전 250일의 역사적 95% VaR과 비교."
        " Kupiec 1995, Christoffersen 1998, Basel 신호등(이항 누적확률 95%·99.99%"
        " 기준) · 전체 수익률 500일 미만은 표본 부족",
        "스트레스: 지수마다 적힌 고점·저점 ±90일에서 찾아 이른 고점 → 늦은 저점."
        " 지금 보유분을 첫날부터 보유해 구간 끝과 최악(구간 안 최대 낙폭), 환율 몫을"
        " 계산. 구간 자료가 없는 종목은 KOSPI(한국) 또는 S&P 500(그 밖, 1987은"
        " 모두)의 일간 등락 × 베타로 대신함. 베타는 그 종목의 저장된 전체 기간에서"
        " 같은 시장 지수엔 Welch, 한국 종목을 S&P 500으로 대신할 땐 Dimson으로 구해"
        " n/(n+250)만큼 1 쪽으로 당김,"
        f" 겹치는 날이 {portfolio.PROXY_MIN_DAYS}일 미만이면 1 · 지수·환율은"
        " 1987년부터(KOSPI는 Yahoo 자료가 있는 때부터) 한 번 받아 둠 · 2003년 12월"
        " 이전 원/달러는 FRED DEXKOUS, 키가 없으면 환율 미반영 · 튀었다 바로"
        " 돌아온 가격은 표시만 하고 그대로 씀",
        "가상 충격: 종목마다 자기 지수에 대한 현지 통화 Welch 베타를 n/(n+250)만큼"
        " 1 쪽으로 당겨 지수 충격에 곱하고 환율 반영 · 역스트레스: 모든 지수가 같은"
        " 폭으로 떨어질 때 −30%가 되는 폭",
        f"난기류: 주간 수익률, 직전 {portfolio.WEEKS}주 평균·공분산(Ledoit–Wolf,"
        f" {portfolio.TURBULENCE_REFIT}주마다 재추정)과의 마할라노비스 거리"
        " (Kritzman–Li 2010) · 상관 이례도 = 거리 ÷ 상관을 무시한 거리",
        f"스타일: 수익률 기반 스타일 분석(Sharpe 1992), 최근 {portfolio.WEEKS}주"
        f" {base} 주간 수익률, 비교 지수 {', '.join(portfolio.STYLE_PROXIES)}와 현금,"
        " 비중 0 이상·합 1 · 기간을 못 채운 지수는 뺌 · 비교 지수는 버튼을 눌러야"
        " 받음",
    )
    with st.expander("방법·가정", icon=":material/info:"):
        st.markdown("\n".join(f"- {line}" for line in lines))


# ---- one stock, and adding it to a portfolio

STOCK_SECTIONS = {
    "analysis": "위험지표",
    "risk": "시뮬레이션·VaR",
    "stress": "스트레스",
    "whatif": "넣으면 어떻게 되나",
}
STOCK_INDICES = (portfolio.FX, portfolio.BENCHMARK, portfolio.KOSPI, portfolio.RF_USD)
WHATIF_PERCENT = (1, 30, 5)  # slider: least, most, first shown


def stock_choose(ticker: str) -> None:
    state.stock_error = ""
    if ticker != state.get("stock_ticker"):
        state.stock_ticker = ticker
        state.stock_base = portfolio.currency(ticker)


def stock_enter() -> None:
    """The typed ticker, on Enter or 분석."""
    text: str = state.get("stock_input", "")
    if not text.strip():
        return
    try:
        ticker = portfolio.clean_ticker(text)
    except portfolio.BookError as error:
        state.stock_error = str(error)
        return
    state.stock_held = None
    stock_choose(ticker)


def stock_held() -> None:
    picked: str | None = state.get("stock_held")
    if picked:
        state.stock_input = picked
        stock_choose(picked)


def stock_owned(book: portfolio.Book, prices: portfolio.Prices) -> list[str]:
    """Tickers open in any portfolio."""
    return sorted(
        {
            h.ticker
            for p in book.portfolios
            for h in portfolio.holdings(p, prices.close)
            if h.is_open
        }
    )


def stock_fetch(
    fetcher: portfolio.Fetcher, ticker: str, prices: portfolio.Prices
) -> None:
    """A ticker without saved prices, with the indices, fetched once; then rerun."""
    todo = fetcher.untried([ticker, *STOCK_INDICES], prices.tickers)
    if ticker not in todo:
        return
    with st.spinner(f"{ticker} 가격 받는 중…"):
        try:
            fetcher.fetch_now(todo)
        except Exception as error:  # noqa: BLE001 - shown on the page
            state.stock_error = f"가격을 받지 못했습니다: {type(error).__name__}"
    st.rerun()


def stock_share(
    ticker: str, base: portfolio.Currency, prices: portfolio.Prices
) -> float:
    """One share at the last close, in ``base`` at the last USD/KRW."""
    price, _ = portfolio.last_close(prices.close, ticker)
    rate, _ = portfolio.last_close(prices.close, portfolio.FX)
    return portfolio.convert(price, portfolio.currency(ticker), base, rate)


def stock_subject(
    ticker: str, base: portfolio.Currency, prices: portfolio.Prices
) -> Subject:
    """The stock alone: weight 1, amounts for one share."""
    rets = portfolio.asset_returns(prices.adjusted, [ticker], base)
    value = stock_share(ticker, base, prices)
    index = portfolio.stand_in(ticker)
    return Subject(
        key=pf_key("stock", ticker, base),
        label=ticker,
        base=base,
        rets=cast("pd.Series", rets[ticker]).dropna(),
        w=pd.Series({ticker: 1.0}, dtype="float64"),
        value=value,
        bench=partial(portfolio.market_returns, prices.adjusted, base, index=index),
        bench_label="KOSPI" if index == portfolio.KOSPI else portfolio.BENCHMARK_LABEL,
        notes=(f"금액은 1주 기준 (최근 종가 {portfolio.money(value, base)})",),
        holding="이 종목 1주",
    )


def stock_betas(s: Subject, prices: portfolio.Prices, months: int) -> None:
    """A Korean stock's Welch, Dimson and KOSPI betas (``pf_stats`` leaves them out)."""
    j, _, _ = pf_joint(s, prices, months)
    if j is None or len(j.window) < portfolio.MIN_DAYS:
        return
    with st.container(border=True):
        st.markdown(f"**베타** · {s.base} 일간 수익률")
        pf_betas(portfolio.betas(j)[:1])


def stock_change(before: float, after: float, text: Callable[[float], str]) -> str:
    return f"{text(before)} → {text(after)}"


def stock_signed(value: float) -> str:
    return ("+" if value > 0 else "") + pf_ratio(value)


def stock_es(value: float) -> str:
    return portfolio.percent(max(value, 0.0), dec=2)


def whatif_metric(
    label: str,
    pair: tuple[float, float],
    text: Callable[[float], str],
    delta: str,
    tip: str | None = None,
) -> None:
    """``before → after`` with the change, in no color."""
    st.metric(
        label,
        stock_change(*pair, text),
        delta,
        delta_color="off",
        delta_arrow="off",
        help=tip,
        border=True,
    )


def whatif_risk(a: portfolio.Added, ticker: str) -> None:
    vol, beta, ratio, es = a.vol, a.beta, a.ratio, a.es
    with st.container(horizontal=True):
        whatif_metric(
            "연 변동성", vol, portfolio.percent, portfolio.points(vol[1] - vol[0])
        )
        whatif_metric(
            "베타",
            beta,
            pf_ratio,
            stock_signed(beta[1] - beta[0]),
            f"{portfolio.BENCHMARK_LABEL} 대비, 한국 종목은 Dimson",
        )
        whatif_metric(
            "분산 비율",
            ratio,
            pf_ratio,
            stock_signed(ratio[1] - ratio[0]),
            "종목 변동성의 가중합 ÷ 포트폴리오 변동성",
        )
        if es is None:
            st.metric(
                "ES 97.5% (1일)", PF_THIN, help="400일 이상 (기간 3Y)", border=True
            )
        else:
            whatif_metric(
                "ES 97.5% (1일)",
                es,
                stock_es,
                portfolio.points(es[1] - es[0]),
                "가장 나쁜 2.5% 날의 평균 손실 · 변화는 증분 ES",
            )
        st.metric(
            f"추가 후 ES 중 {ticker} 몫",
            portfolio.percent(a.es_share) if a.es_share is not None else PF_THIN,
            help="포트폴리오가 가장 나빴던 2.5% 날의 손실 중 이 종목 몫 (Tasche)",
            border=True,
        )
    st.markdown(
        f"이 종목과 포트폴리오 상관이 **{a.threshold:.2f}**보다 낮으면 변동성이"
        f" 줄어듭니다 · 지금 상관 {a.correlation:.2f} (조금 더할 때 기준)"
    )


def whatif_paths(
    key: Key, j: portfolio.Joint, ticker: str, fraction: float, label: str
) -> None:
    """Monte Carlo before and after on the same cached draws."""
    days = portfolio.HORIZONS[label]
    block = portfolio.mean_block(j)
    picks = cached_draws(state.get("pf_prices_stamp", ()), len(j.window), days, block)
    flat = (1.0, 1.0)  # only the horizon's own drawdowns are shown
    before = cached_paths(
        (*key, days, block), fhs=False, value=1.0, standing=flat, _j=j, _picks=picks
    )
    after = cached_paths(
        (*key, days, block, ticker, fraction),
        fhs=False,
        value=1.0,
        standing=flat,
        _j=j,
        _picks=picks,
        _w=portfolio.mixed_in(j, ticker, fraction),
    )
    ten = portfolio.LOSS_LEVELS[0]
    pairs = [
        (f"{label} 뒤 10% 이상 손실 경로 비율", before.losses[ten], after.losses[ten]),
        *(
            (f"기간 중 −{level:.0%} 낙폭", odds, after.paths.drawdown_odds[level])
            for level, odds in before.paths.drawdown_odds.items()
        ),
    ]
    mid = (before.fan[-1]["p50"] - 1, after.fan[-1]["p50"] - 1)
    with st.container(horizontal=True):
        for name, old, new in pairs:
            whatif_metric(
                name, (old, new), portfolio.percent, portfolio.points(new - old)
            )
        whatif_metric(
            f"{label} 뒤 중앙값",
            mid,
            partial(portfolio.percent, signed=True),
            portfolio.points(mid[1] - mid[0]),
        )
    st.caption(
        f"평균 수익 0 가정 · 추가 전후 같은 재표본 경로 {portfolio.PATHS:,}개, 평균"
        f" {block}일 묶음 · 매일 같은 비중으로 재조정 · 예측이 아닙니다"
    )


def whatif_stress(
    s: Subject,
    prices: portfolio.Prices,
    j: portfolio.Joint,
    ticker: str,
    fraction: float,
) -> None:
    older = pf_dashboard_series(PF_DEXKOUS)
    fx = portfolio.long_fx(prices.adjusted, older)
    now = {str(t): float(v) * s.value for t, v in s.w.items()}
    mix = portfolio.mixed_in(j, ticker, fraction)
    later = {t: float(v) * s.value for t, v in zip(j.names, mix, strict=True)}
    key = (*s.key, "replay", len(older))  # the stress section's, for ``now``
    before = cached_replays(key, s.base, now, prices.adjusted, fx)
    after = cached_replays((*key, ticker, fraction), s.base, later, prices.adjusted, fx)

    def note(r: portfolio.Replayed) -> str:
        if r.window is None:
            return "지수 자료 없음"
        if ticker not in r.proxied:
            return PF_DASH
        at = r.proxied.index(ticker)
        if at < len(r.proxy_betas):
            return f"{ticker} 지수×베타 {r.proxy_betas[at]:.2f}로 대체"
        return f"{ticker} 지수로 대체"

    def ends(rows: list[portfolio.Replayed]) -> list[str]:
        return [portfolio.percent(r.change, signed=True) for r in rows]

    def lows(rows: list[portfolio.Replayed]) -> list[str]:
        return [portfolio.percent(r.worst, signed=True) for r in rows]

    pf_table(
        {
            "시나리오": [r.scenario.name for r in before],
            "구간 끝": ends(before),
            "구간 끝 · 추가 후": ends(after),
            "최악": lows(before),
            "최악 · 추가 후": lows(after),
            "비고": [note(r) for r in after],
        },
        "스트레스 추가 전후",
    )
    st.caption(
        "스트레스 섹션과 같은 재현 · 자료가 없는 구간은 KOSPI(한국) 또는 S&P 500의"
        " 일간 등락 × 1 쪽으로 당긴 베타로"
    )


def stock_whatif(
    ticker: str, book: portfolio.Book, prices: portfolio.Prices, months: int
) -> None:
    names = {p.id: p.name for p in book.portfolios if p.trades}
    if not names:
        st.info("거래가 있는 포트폴리오가 없습니다.")
        return
    if state.get("whatif_pf") not in names:
        mine = state.get("pf_selected")
        state.whatif_pf = mine if mine in names else next(iter(names))
    least, most, first = WHATIF_PERCENT
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.selectbox(
            "포트폴리오",
            list(names),
            format_func=names.__getitem__,
            key="whatif_pf",
            width=280,
        )
        st.slider(
            "추가 비중",
            least,
            most,
            first,
            format="%d%%",
            key="whatif_percent",
            width=280,
        )
        picked = st.segmented_control(
            "시뮬레이션 기간",
            list(portfolio.HORIZONS),
            default="1M",
            required=True,
            key="whatif_horizon",
        )
    p = cast("portfolio.Portfolio", book.get(state.whatif_pf))
    s = pf_subject(p, prices)
    if s.w.empty:
        st.info("가격이 확인된 보유 종목이 없습니다.")
        return
    fraction = int(state.whatif_percent) / 100
    key = (*s.key, months, "whatif", ticker)
    w = portfolio.with_candidate(s.w, ticker)
    j = cached_joint((*key, "joint"), p.base, months, prices.adjusted, w)
    if len(j.window) < portfolio.MIN_DAYS:
        st.info(
            f"보유 종목과 {ticker}의 수익률이 모두 있는 날이 {len(j.window)}일뿐입니다."
        )
        return
    amount = fraction * s.value
    notes = [
        f"평가액의 {fraction:.0%} ({portfolio.money(amount, p.base)}, 약"
        f" {amount / stock_share(ticker, p.base, prices):,.1f}주)를 {ticker}에 넣고"
        " 기존 보유분을 비중대로 줄인 경우"
    ]
    if ticker in s.w.index:
        now = portfolio.percent(float(cast("float", s.w[ticker])))
        notes.append(f"이미 보유 중: 지금 비중 {now}에 더함")
    notes.append(f"{len(j.window)}일 · 모든 종목 값이 있는 날, {p.base} 일간 수익률")
    if j.late:
        notes.append(f"자료 시작이 늦어 기간이 짧아짐: {j.late}")
    st.caption(" · ".join(notes))
    for note in s.notes:
        st.caption(note)
    cov, _, _ = cached_covariance((*key, "cov"), j)
    whatif_risk(cached_added((*key, "added"), fraction, ticker, j, cov), ticker)
    st.markdown(f"**몬테카를로** · {picked or '1M'}")
    whatif_paths(key, j, ticker, fraction, picked or "1M")
    st.markdown("**과거 구간 재현** · 지금 보유분과 추가 후")
    whatif_stress(s, prices, j, ticker, fraction)


def stock_section(
    section: str,
    ticker: str,
    book: portfolio.Book,
    prices: portfolio.Prices,
    months: int,
) -> None:
    if section == "whatif":
        stock_whatif(ticker, book, prices, months)
        return
    s = stock_subject(ticker, state.stock_base or portfolio.currency(ticker), prices)
    if len(s.rets) < 2:  # noqa: PLR2004
        pf_no_prices()
    elif section == "analysis":
        pf_stats(s, prices, months)
        if portfolio.korean(ticker):
            stock_betas(s, prices, months)
        pf_style_box(s, prices, "stock_style")
    elif section == "risk":
        pf_risk(s, prices, months)
    else:
        pf_stress_section(s, prices, months)


def stock_method() -> None:
    lines = (
        "가격: 수정 종가(배당·분할 반영). 저장된 가격이 없는 티커는 처음 고를 때 지수·"
        f"환율과 함께 한 번 받아 `{portfolio.DATA_DIR}/`에 저장",
        "기준통화: 종목의 거래 통화가 기본, 일별 USD/KRW로 환산 · 금액은 1주 기준",
        "벤치마크: 한국 종목은 KOSPI, 그 밖은 S&P 500 (배당 제외 가격지수)",
        "위험지표·시뮬레이션·VaR·스트레스: 내 포트폴리오와 같은 방법, 이 종목 100%."
        " 상관·분산 비율·유효 종목 수·종목별 ES 기여·난기류는 한 종목에서 뜻이 없어"
        " 뺌",
        "넣으면 어떻게 되나: 평가액의 일정 비율을 이 종목에 넣고 기존 보유분을 비중대로"
        " 줄여 마련(이미 보유하면 지금 비중에 더함) · 보유 종목과 이 종목이 모두 값이"
        " 있는 날의 일간 수익률을 포트폴리오 기준통화로, 시차 보정과 Ledoit–Wolf 축소"
        " 공분산 · 변동성 기준: 상관 ρ < σp/σc이면 조금 더할 때 변동성이 줄어듦"
        " (σp 포트폴리오, σc 이 종목) · 베타는 종목 베타의 가중합(한국 종목은 Dimson)"
        " · ES 97.5% 변화는 같은 날들로 추가 전후 포트폴리오를 다시 계산한 증분 ES"
        " (완전 재평가), 400일 이상 · 몬테카를로는 추가 전후 같은 재표본 날짜 ·"
        " 과거 구간 재현은 스트레스 섹션과 같은 방법",
    )
    with st.expander("방법·가정", icon=":material/info:"):
        st.markdown("\n".join(f"- {line}" for line in lines))


def stock_view(
    book: portfolio.Book,
    prices: portfolio.Prices,
    fetcher: portfolio.Fetcher,
    missing: list[str],
) -> None:
    owned = stock_owned(book, prices)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.text_input("티커", key="stock_input", width=170, on_change=stock_enter)
        st.button("분석", key="stock_go", on_click=stock_enter)
        if owned:
            st.pills("보유 종목", owned, key="stock_held", on_change=stock_held)
    if state.get("stock_error"):
        st.error(state.stock_error)
    ticker: str | None = state.get("stock_ticker")
    blocked = args.demo or bool(missing)
    if ticker and ticker not in prices.tickers and not blocked:
        stock_fetch(fetcher, ticker, prices)
    pf_bar(fetcher, pf_root(), book, missing, [ticker] if ticker else [])
    if not ticker:
        st.info("티커를 넣거나 보유 종목을 고릅니다.")
        return
    if ticker not in prices.tickers:
        st.info(f"{ticker}의 저장된 가격이 없습니다.")
        return
    if not blocked:
        fetcher.deepen(prices)  # stress windows back to 1987, once
    section = st.segmented_control(
        "보기",
        list(STOCK_SECTIONS),
        format_func=STOCK_SECTIONS.__getitem__,
        default="analysis",
        required=True,
        key="stock_section",
        label_visibility="collapsed",
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        state.setdefault("stock_base", portfolio.currency(ticker))
        st.segmented_control(
            "기준통화",
            ["KRW", "USD"],
            required=True,
            key="stock_base",
            disabled=section == "whatif",  # the portfolio's own base there
        )
        window = st.segmented_control(
            "기간",
            list(portfolio.WINDOWS),
            default="3Y",
            required=True,
            key="pf_window",
        )
    months = portfolio.WINDOWS[window or "3Y"]
    stock_section(section or "analysis", ticker, book, prices, months)
    stock_method()


# ---- famous investors (SEC 13F)

PF_MODES = {"mine": "내 포트폴리오", "stock": "종목 분석", "investors": "유명 투자자"}
INV_SECTIONS = {
    "holdings": "보유",
    "returns": "수익률",
    "analysis": "위험지표",
    "correlation": "상관·분산",
    "risk": "시뮬레이션·VaR",
    "stress": "스트레스",
    "compare": "비교",
}
INV_VARIANTS = {
    "filed": "제출일 복제",
    "top": f"상위 {investors.CONVICTION}종목 복제",
}
INV_THEORY = "이론값 (분기말)"
INV_THEORY_NOTE = "공개 전 정보 사용, 실제로 따라 할 수 없음"
INV_LIMITS = (
    "13F는 미국 상장 주식 롱 포지션만 보여줍니다. 공매도·옵션·채권·현금·분기 중"
    " 매매는 빠져 있어 실제 펀드 수익률과 다릅니다."
)
INV_CHANGE_STYLE = {"신규": display.UP_COLOR, "증가": display.UP_COLOR}
INV_CHANGE_STYLE["감소"] = display.DOWN_COLOR

type InvData = tuple[
    investors.Filer | None,
    list[investors.Quarter],
    dict[str, investors.Link],
    dict[str, investors.Sector],
]


@st.cache_resource(show_spinner=False)
def investor_fetcher(root: str) -> investors.Fetcher:
    """One 13F fetcher per data folder, shared by every tab."""
    return investors.Fetcher(Path(root))


@st.cache_data(show_spinner=False, max_entries=16)
def inv_load(root: str, cik: str, stamp: tuple[tuple[str, int], ...]) -> InvData:
    """The saved filings, quarters and caches; ``stamp`` changes with the files."""
    del stamp
    path = Path(root)
    filer = investors.load_filer(path, cik)
    tables = investors.load_tables(path, filer) if filer is not None else []
    return (
        filer,
        investors.quarters(tables),
        investors.load_cusips(path),
        investors.load_sectors(path),
    )


def inv_root() -> Path:
    return resolve_settings(args.settings).parent / investors.DATA_DIR


def sec_agent() -> str:
    """``SEC_USER_AGENT`` from the same env path as the API keys; never shown."""
    env = resolve_env(args.env_file, os.environ)
    return env.get(investors.AGENT_KEY, "").strip()


def inv_blocked(agent: str, missing: list[str]) -> str:
    """Why nothing can be fetched now, or empty."""
    if args.demo:
        return "데모 모드에서는 받지 않습니다."
    if missing:
        packages = ", ".join(missing)
        return f"추가 패키지({packages})가 필요합니다: `{portfolio.INSTALL_HINT}`"
    if not agent:
        return (
            f"SEC에서 받으려면 `.env`에 `{investors.AGENT_KEY}`(이름과 이메일)를"
            " 넣으세요."
        )
    return ""


def inv_style(text: str) -> str:
    color = INV_CHANGE_STYLE.get(text)
    return f"color: {color}" if color else pf_style(text)


def inv_priced(
    q: investors.Quarter, cusips: dict[str, investors.Link], prices: portfolio.Prices
) -> pd.Series:
    """Reported weights of the covered holdings with a ticker and saved prices."""
    w = investors.analysis_weights(q.rows, cusips)
    return cast("pd.Series", w[w.index.isin(list(prices.tickers))])


# ---- add a CIK


def inv_open_add() -> None:
    state.inv_dialog = True
    state.inv_dialog_error = ""
    state.inv_new_cik = ""


def inv_close() -> None:
    state.inv_dialog = False


def inv_add(root: Path, agent: str) -> None:
    try:
        cik = investors.clean_cik(state.inv_new_cik)
        name = investor_fetcher(str(root)).check(cik, agent)
        investors.add_investor(root, cik, name)
    except (investors.InvestorError, OSError) as error:
        state.inv_dialog_error = str(error)
        return
    state.inv_selected = cik
    inv_close()


def inv_dialog_body(root: Path, agent: str) -> None:
    if not state.get("inv_dialog"):
        st.rerun()
    st.text_input("CIK", key="inv_new_cik", max_chars=13)
    if state.inv_dialog_error:
        st.error(state.inv_dialog_error)
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.button("취소", key="inv_cancel", on_click=inv_close)
        st.button(
            "추가",
            type="primary",
            key="inv_add_ok",
            on_click=inv_add,
            args=(root, agent),
        )


def inv_selector(
    people: list[investors.Investor], root: Path, blocked: str
) -> investors.Investor:
    labels: dict[str, str] = {}
    today = datetime.now().astimezone().date()
    for person in people:
        filer = investors.load_filer(root, person.cik)
        last = investors.latest(filer)
        stopped = " · 보고 중단" if investors.stale(filer, today) else ""
        labels[person.cik] = (
            f"{person.label} · {last.period} 기준 · 제출 {last.filed}{stopped}"
            if last
            else f"{person.label} · 미수신"
        )
    if state.get("inv_selected") not in labels:
        state.inv_selected = people[0].cik
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.selectbox(
            "투자자",
            list(labels),
            format_func=labels.__getitem__,
            key="inv_selected",
            width=560,
        )
        st.button(
            "CIK 추가",
            icon=":material/add:",
            key="inv_add",
            on_click=inv_open_add,
            disabled=bool(blocked),
        )
    return next(p for p in people if p.cik == state.inv_selected)


# ---- fetching


def inv_bar(  # noqa: PLR0913 - the bar shows both fetchers
    fetcher: investors.Fetcher,
    pfetch: portfolio.Fetcher,
    root: Path,
    cik: str,
    agent: str,
    blocked: str,
) -> None:
    """When the 13F were checked, the 갱신 button and running fetches."""
    polling = fetcher.progress.running or pfetch.progress.running

    @st.fragment(run_every=1.0 if polling else None)
    def body() -> None:
        progress = fetcher.progress
        pricing = pfetch.progress.running
        if polling and not progress.running and not pricing:
            st.rerun()
        filer = investors.load_filer(root, cik)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(
                f"SEC EDGAR 13F {filer.fetched:%Y-%m-%d %H:%M} 확인 · 티커 OpenFIGI"
                " · 섹터·가격 yfinance (Yahoo Finance 비공식 데이터)"
                if filer
                else "저장된 13F가 없습니다."
            )
            st.space("stretch")
            if st.button(
                "갱신",
                icon=":material/refresh:",
                type="primary",
                key="inv_refresh",
                disabled=progress.running or bool(blocked),
                help="새 13F 제출, 티커·섹터, 보유 종목 가격을 받습니다",
            ):
                fetcher.start(cik, agent)
                state.inv_reprice = cik
                st.rerun()
        if progress.running:
            done = progress.done / progress.total if progress.total else 0.0
            count = f" {progress.done}/{progress.total}" if progress.total else ""
            st.progress(done, text=f"{progress.stage}{count}")
        elif progress.message and progress.cik == cik:
            if progress.failed:
                st.warning(progress.message)
            else:
                st.success(progress.message)
        if pricing:
            st.caption(":material/progress_activity: 가격 받는 중")
        if blocked:
            st.caption(blocked)

    body()


def inv_prices(
    pfetch: portfolio.Fetcher,
    blocked: bool,
    cik: str,
    names: list[str],
    prices: portfolio.Prices,
) -> None:
    """Prices go to the portfolio page's store: untried tickers once, all on 갱신.

    ``blocked``: demo, missing packages, no User-Agent, or a 13F fetch running.
    """
    if blocked:
        return
    wanted = [*names, portfolio.BENCHMARK, portfolio.FX, portfolio.RF_USD]
    if state.get("inv_reprice") == cik:
        if pfetch.start(wanted):
            state.inv_reprice = ""
    else:
        pfetch.start(pfetch.untried(wanted, prices.tickers))


# ---- holdings


def inv_weight_history(points: list[tuple[object, float]]) -> alt.Chart:
    rows = [
        {"period": str(day), "weight": value}
        for day, value in points
        if math.isfinite(value)
    ]
    return (
        alt.Chart(alt.Data(values=rows))
        .mark_bar()
        .encode(
            x=alt.X("period:N", title=None),
            y=alt.Y("weight:Q", title=None, axis=alt.Axis(format=".1%")),
            tooltip=[
                alt.Tooltip("period:N", title="분기 말"),
                alt.Tooltip("weight:Q", title="비중", format=".2%"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )


def inv_stock_body(
    found: list[investors.Quarter], row: dict[str, object], prices: portfolio.Prices
) -> None:
    st.caption(f"{row['sector']} · CUSIP {row['cusip']}")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**분기별 비중** · 보고 가치 기준")
        history = investors.weight_history(found, str(row["cusip"]))
        st.altair_chart(inv_weight_history(list(history)), alt="분기별 비중")
    with right, st.container(border=True):
        st.markdown("**1년 주가** · 종가, USD")
        ticker = str(row["ticker"])
        if ticker in prices.close.columns:
            series = portfolio.window(
                cast("pd.Series", prices.close[ticker]).dropna(), 12
            )
            points = portfolio.line_points({"종가": series})
            st.altair_chart(pf_line_chart(points, ",.2f"), alt=f"{ticker} 1년 주가")
        else:
            st.caption(portfolio.UNKNOWN)


def inv_open_stock(
    found: list[investors.Quarter], row: dict[str, object], prices: portfolio.Prices
) -> None:
    """Same pattern as ``open_stock``: a fresh table key clears the clicked cell."""
    state.inv_gen = state.get("inv_gen", 0) + 1
    title = f"{row['ticker'] or '—'} · {row['name']}"
    st.dialog(title, width="large", on_dismiss="rerun")(inv_stock_body)(
        found, row, prices
    )


def inv_holdings(
    found: list[investors.Quarter],
    cusips: dict[str, investors.Link],
    sectors: dict[str, investors.Sector],
    prices: portfolio.Prices,
) -> None:
    periods = [q.period for q in found]
    labels = {q.period: investors.quarter_label(q) for q in found}
    if state.get("inv_quarter") not in periods:
        state.inv_quarter = periods[0]
    st.selectbox(
        "분기", periods, format_func=labels.__getitem__, key="inv_quarter", width=360
    )
    at = periods.index(state.inv_quarter)
    q = found[at]
    before = found[at + 1] if at + 1 < len(found) else None
    frame = investors.holdings_frame(q, before, cusips, sectors)
    priced = float(cast("float", inv_priced(q, cusips, prices).sum()))
    excluded = investors.excluded_share(q.rows)
    with st.container(horizontal=True):
        st.metric("보고 가치", investors.dollars(q.value), border=True)
        st.metric("종목 수", f"{len(q.rows):,}", border=True)
        st.metric(
            "상위 10 비중", portfolio.percent(investors.top_share(q.rows)), border=True
        )
    left_out: list[str] = []
    if q.options:
        left_out.append(f"옵션 {q.options}건 ({investors.dollars(q.options_value)})")
    if q.principal:
        left_out.append(
            f"원금 단위(PRN) {q.principal}건 ({investors.dollars(q.principal_value)})"
        )
    notes = [
        f"분기 말 {q.period} · 제출 {q.filed}",
        *(f"정정 {day}" for day in q.amended),
        *([f"제외: {', '.join(left_out)}"] if left_out else []),
        *([f"반영 안 한 정정 {q.ignored}건"] if q.ignored else []),
        *([f"미포함 비중 {portfolio.percent(excluded)}"] if excluded > 0 else []),
        f"가격 미연결 비중 {portfolio.percent(1 - excluded - priced)}",
    ]
    st.caption(" · ".join(notes))
    records = cast("list[dict[str, object]]", frame.to_dict("records"))
    shown = pd.DataFrame(
        {
            "티커": [str(r["ticker"]) or "—" for r in records],
            "종목": [str(r["name"]) for r in records],
            "섹터": [str(r["sector"]) for r in records],
            "주식 수": [portfolio.shares(cast("float", r["shares"])) for r in records],
            "가치": [investors.dollars(cast("float", r["value"])) for r in records],
            "비중": [portfolio.percent(cast("float", r["weight"])) for r in records],
            "직전 분기 대비": [str(r["change"]) for r in records],
        }
    )
    event = st.dataframe(
        shown.style.map(inv_style, subset=["직전 분기 대비"]),
        hide_index=True,
        height=420,
        on_select="rerun",
        selection_mode="single-cell",
        key=f"inv_table_{state.get('inv_gen', 0)}",
        column_config={"종목": st.column_config.TextColumn(width="medium")},
        alt="보유 종목",
    )
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**섹터 비중** · 보고 가치 기준")
        st.altair_chart(
            pf_weight_chart(investors.sector_weights(frame), "섹터"), alt="섹터 비중"
        )
    with right, st.container(border=True):
        gone = investors.sold_out(q, before)
        st.markdown(f"**청산** · 직전 분기 대비 {len(gone)}종목")
        if gone:
            pf_table(
                {
                    "종목": [r.name for r in gone],
                    "직전 주식 수": [portfolio.shares(r.shares) for r in gone],
                    "직전 가치": [investors.dollars(r.value) for r in gone],
                },
                "청산 종목",
            )
    for index, column in event.selection.cells:
        if column in {"티커", "종목"}:
            inv_open_stock(found, records[index], prices)


# ---- clones


@st.cache_data(show_spinner=False, max_entries=32)
def inv_clone(
    key: Key,
    top: bool,
    equal: bool,
    _filer: investors.Filer | None,
    _found: list[investors.Quarter],
    _cusips: dict[str, investors.Link],
    _prices: portfolio.Prices,
) -> tuple[portfolio.Clone, pd.Series]:
    """The filing-date clone, and the quarter-end one from its first day.

    ``top``: each set cut to its ``investors.CONVICTION`` largest weights,
    ``equal`` weighted alike. ``key`` (prices and 13F stamps) stands for the
    13F files and the prices.
    """
    del key
    adjusted = _prices.adjusted
    filed = investors.filed_schedule(_found, _cusips, investors.buy_days(_filer))
    ends = investors.period_schedule(_found, _cusips)
    if top:
        filed = investors.top_schedule(filed, equal=equal)
        ends = investors.top_schedule(ends, equal=equal)
    made = portfolio.clone(adjusted, filed)
    theory = portfolio.rebalanced(adjusted, ends)
    if made.cumulative.empty:
        return made, theory
    start = cast("pd.Timestamp", made.cumulative.index[0])
    return made, portfolio.since(theory, start)


def inv_variant() -> tuple[bool, bool]:
    """The chosen clone: (top weights only, equal weights)."""
    top = state.get("inv_variant") == "top"
    return top, top and bool(state.get("inv_equal"))


def inv_label(top: bool, equal: bool) -> str:
    label = INV_VARIANTS["top" if top else "filed"]
    return f"{label} · 같은 비중" if equal else label


def inv_controls(section: str) -> int:
    """기간 (not on 수익률) and the clone; returns the window in months."""
    with st.container(horizontal=True, vertical_alignment="bottom"):
        window = (
            st.segmented_control(
                "기간",
                list(portfolio.WINDOWS),
                default="3Y",
                required=True,
                key="pf_window",
            )
            if section != "returns"
            else None
        )
        picked = st.segmented_control(
            "복제",
            list(INV_VARIANTS),
            format_func=INV_VARIANTS.__getitem__,
            default="filed",
            required=True,
            key="inv_variant",
            help="제출일 복제: 제출이 공개된 날 종가에 보고 비중으로 삼",
        )
        if picked == "top":
            st.toggle("같은 비중", key="inv_equal")
    return portfolio.WINDOWS[window or "3Y"]


def inv_sp500(adjusted: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    return portfolio.market_returns(adjusted, "USD", days)


def inv_subject(
    key: Key, data: InvData, prices: portfolio.Prices, *, top: bool, equal: bool
) -> Subject | None:
    """The clone's daily returns and today's drifted weights, in dollars."""
    filer, found, cusips, _ = data
    made, _ = inv_clone(key, top, equal, filer, found, cusips, prices)
    if len(made.cumulative) < 2 or made.weights.empty:  # noqa: PLR2004
        return None
    q = found[0]
    reported = inv_priced(q, cusips, prices)
    held = reported.reindex(portfolio.labels(made.weights))
    value = q.value * float(cast("float", held.sum()))
    return Subject(
        key=(*key, top, equal),
        label=inv_label(top, equal),
        base="USD",
        rets=portfolio.daily(made.cumulative),
        w=made.weights,
        value=value,
        bench=partial(inv_sp500, prices.adjusted),
        bench_label=portfolio.BENCHMARK_LABEL,
        trim=True,
        notes=(
            f"금액은 {q.period} 보고 가치 중 이 종목들의 몫"
            f" {investors.dollars(value)} 기준",
        ),
    )


def inv_usd(p: portfolio.Portfolio) -> portfolio.Portfolio:
    """The portfolio valued in dollars, at daily USD/KRW like the page converts."""
    return p.model_copy(update={"base": "USD"})


# ---- clone returns


def inv_chart(lines: dict[str, pd.Series], theory: pd.Series) -> alt.LayerChart:
    """Cumulative returns, with the quarter-end clone as a grey dashed line."""
    grey = (
        alt.Chart(pf_rows(portfolio.line_points({INV_THEORY: theory})))
        .mark_line(color="gray", strokeDash=[4, 3], opacity=0.6)
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("value:Q", title=None),
            tooltip=[
                alt.Tooltip("day:T", title="날짜", format="%Y-%m-%d"),
                alt.Tooltip("series:N", title="구분"),
                alt.Tooltip("value:Q", title="값", format="+.1%"),
            ],
        )
    )
    shown = pf_line_chart(portfolio.line_points(lines), "+.1%")
    return alt.LayerChart(layer=[grey, shown])


def inv_steps(steps: tuple[portfolio.Step, ...]) -> None:
    pf_table(
        {
            "매수일": [str(s.day) for s in reversed(steps)],
            "종목 수": [f"{s.names:,}" for s in reversed(steps)],
            "회전율": [
                portfolio.percent(s.turnover) if s.turnover is not None else PF_DASH
                for s in reversed(steps)
            ],
        },
        "리밸런싱",
    )
    st.caption(
        "회전율: 직전 보유가 가격대로 변한 비중과 새 비중의 차이 합의 절반(편도) ·"
        " 첫 매수는 —"
    )


def inv_returns(  # noqa: PLR0913 - the view's inputs
    key: Key,
    data: InvData,
    book: portfolio.Book,
    prices: portfolio.Prices,
    *,
    top: bool,
    equal: bool,
) -> None:
    filer, found, cusips, _ = data
    made, theory = inv_clone(key, top, equal, filer, found, cusips, prices)
    filed = made.cumulative
    if len(filed) < 2:  # noqa: PLR2004
        pf_no_prices()
        return
    label = inv_label(top, equal)
    q = found[0]
    today = datetime.now().astimezone().date()
    turns = [s.turnover for s in made.steps if s.turnover is not None]
    start = cast("pd.Timestamp", filed.index[0])
    with st.container(horizontal=True):
        st.metric(
            f"누적 · {label}",
            portfolio.percent(portfolio.last(filed), signed=True),
            help=f"{start.date()}부터, USD",
            border=True,
        )
        st.metric(
            "종목 수",
            f"{len(made.weights):,}",
            help="지금 들고 있는 종목 (가격이 이어지는 것)",
            border=True,
        )
        st.metric(
            "가격 연결 비중",
            portfolio.percent(
                float(cast("float", inv_priced(q, cusips, prices).sum()))
            ),
            help=f"{q.period} 보고 가치 중 가격으로 계산한 몫",
            border=True,
        )
        st.metric(
            "분기말 이후 경과일",
            f"{(today - q.period).days:,}일",
            help=f"최근 보고 {q.period} 기준 · 분기 말에서 제출까지"
            f" {(q.filed - q.period).days}일",
            border=True,
        )
        st.metric(
            "평균 회전율",
            portfolio.percent(sum(turns) / len(turns)) if turns else PF_DASH,
            help="리밸런싱마다 바꾼 비중, 편도",
            border=True,
        )
    gap = portfolio.points(portfolio.last(filed) - portfolio.last(theory))
    st.markdown(
        f":gray[{INV_THEORY} 누적"
        f" {portfolio.percent(portfolio.last(theory), signed=True)} · 차이 {gap}"
        f" · {INV_THEORY_NOTE}]"
    )
    days = pd.DatetimeIndex(filed.index)
    lines = {
        label: filed,
        portfolio.BENCHMARK_LABEL: portfolio.cumulative(
            inv_sp500(prices.adjusted, days)
        ),
    }
    for p in book.portfolios:
        if p.trades:
            usd = inv_usd(p)
            twr = cached_twr(pf_key(usd.model_dump_json()), usd, prices.close)
            lines[f"{p.name} TWR"] = portfolio.since(twr, start)
    with st.container(border=True):
        st.markdown(f"**누적 수익률** · USD, {start.date()}부터")
        st.altair_chart(inv_chart(lines, theory), alt="복제 수익률")
    st.caption(
        f"복제: 수정 종가 · 회색 점선은 {INV_THEORY}: 분기 말 종가에 샀다고 본 값 ·"
        " 내 포트폴리오는 TWR(종가, 배당 제외)을 일별 USD/KRW로 환산 · 늦게 시작한"
        " 선은 첫 날이 0%"
    )
    missing = investors.untimed(filer)
    if missing:
        st.caption(f"접수 시각이 없는 제출 {missing}건은 다음 거래일 종가에 삽니다")
    st.markdown(f"**리밸런싱** · {len(made.steps)}회")
    inv_steps(made.steps)


# ---- correlation and the investors side by side


@st.cache_data(show_spinner=False, max_entries=32)
def inv_fixed(
    key: Key,
    months: int,
    _q: investors.Quarter,
    _cusips: dict[str, investors.Link],
    _prices: portfolio.Prices,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Window returns, the reported weights of full-window holdings, the dropped.

    ``key`` (prices and 13F stamps) stands for the quarter, the CUSIPs and the prices.
    """
    del key
    priced = inv_priced(_q, _cusips, _prices)
    rets = portfolio.window(
        portfolio.asset_returns(_prices.adjusted, portfolio.labels(priced), "USD"),
        months,
    )
    w, dropped = portfolio.covered(rets, priced)
    return rets, w, dropped


def inv_people(
    people: list[investors.Investor],
) -> list[tuple[investors.Investor, Key, InvData]]:
    """Investors with saved 13F, each with its cache key and data."""
    root = inv_root()
    out: list[tuple[investors.Investor, Key, InvData]] = []
    for person in people:
        stamp = investors.stamp(root, person.cik)
        data = inv_load(str(root), person.cik, stamp)
        if data[1]:
            out.append((person, pf_key(person.cik, stamp), data))
    return out


def inv_lineup(
    book: portfolio.Book,
    people: list[investors.Investor],
    prices: portfolio.Prices,
    months: int,
) -> None:
    st.markdown(f"**투자자·내 포트폴리오·{portfolio.BENCHMARK_LABEL}** · USD")
    book_key = pf_key(book.model_dump_json())
    frame, flags = cached_lineup(book_key, "USD", months, book, prices)
    frame, korean = frame.copy(), [*flags]
    keys: list[object] = [book_key]
    for person, their, data in inv_people(people):
        their_rets, w, _ = inv_fixed(their, months, data[1][0], data[2], prices)
        if not w.empty:
            frame[person.name] = portfolio.weighted(their_rets, w)
            korean.append(False)
            keys.append(their)
    pf_compare_heatmap((*keys, "USD", months, "investors"), frame, korean)
    st.caption(
        "투자자는 최근 분기 보고 비중, 내 포트폴리오는 지금 비중으로 계산 · 받은"
        " 13F가 없는 투자자는 빠짐"
    )


def inv_league(
    people: list[investors.Investor],
    prices: portfolio.Prices,
    months: int,
    *,
    top: bool,
    equal: bool,
) -> None:
    """The clones' Sharpe ratios side by side, shrunk and deflated from two on."""
    series: dict[str, pd.Series] = {}
    keys: list[object] = []
    for person, their, data in inv_people(people):
        made, _ = inv_clone(their, top, equal, data[0], data[1], data[2], prices)
        if len(made.cumulative) > 1:
            daily = portfolio.daily(made.cumulative)
            series[person.label] = portfolio.window(daily, months)
            keys.append(their)
    label = inv_label(top, equal)
    st.markdown(f"**투자자 비교** · {label}, USD, {state.get('pf_window') or '3Y'}")
    if not series:
        st.caption("가격이 있는 복제가 없습니다.")
        return
    days = (
        pd.DatetimeIndex(pd.concat(list(series.values())).index).unique().sort_values()
    )
    rates, source = pf_rates("USD", prices.close)
    rf = portfolio.risk_free(rates, days)
    key = (*keys, top, equal, months, source, pf_rates_key(rates))
    found = cached_peers(key, series, rf)
    pooled = found.count > 1
    table = {
        "투자자": [r.name for r in found.rows],
        "기간 수익률": [portfolio.percent(r.total, signed=True) for r in found.rows],
        "샤프 (연)": [pf_ratio_or(r.sharpe) for r in found.rows],
        **(
            {"축소 샤프": [pf_ratio_or(r.shrunk) for r in found.rows]} if pooled else {}
        ),
        "일수": [f"{r.days:,}" for r in found.rows],
    }
    pf_table(table, "투자자 비교")
    notes = ["샤프 순 · 무위험수익률 " + (source if not rates.empty else "0")]
    if pooled:
        dsr = (
            portfolio.percent(found.deflated)
            if found.deflated is not None
            else f"{PF_SHORT} ({perf.MIN_OBS}일 미만)"
        )
        notes.append(
            "축소 샤프: 각 값을 표준오차만큼 여럿의 평균 쪽으로 당긴 값(경험적 베이즈)"
        )
        notes.append(
            f"{found.count}명 중 최고값이 운으로 나올 수준을 넘는지 (DSR): {dsr}"
        )
    st.caption(" · ".join(notes))


def inv_correlation(  # noqa: PLR0913 - the view's inputs
    s: Subject,
    book: portfolio.Book,
    people: list[investors.Investor],
    prices: portfolio.Prices,
    months: int,
    *,
    top: bool,
    equal: bool,
) -> None:
    compare = partial(inv_lineup, book, people, prices, months)
    pf_correlation(s, prices, months, compare)
    inv_league(people, prices, months, top=top, equal=equal)


# ---- compare


def inv_ratios(
    key: Key, rets: pd.Series, prices: portfolio.Prices, months: int
) -> portfolio.Ratios:
    """``key`` stands for ``rets``; over the S&P 500 in dollars."""
    window = portfolio.window(rets, months)
    rates, source = pf_rates("USD", prices.close)
    rf = portfolio.risk_free(rates, pd.DatetimeIndex(window.index))
    bench = partial(inv_sp500, prices.adjusted, pd.DatetimeIndex(window.index))
    tag = (*key, months, "sp500", source, pf_rates_key(rates))  # not the blended one
    return cached_ratios(tag, window, rf, bench)


def inv_overlap(theirs: pd.Series, mine: pd.Series, period: object) -> None:
    us = portfolio.us_sleeve(mine)
    total = float(cast("float", theirs.sum()))
    their = theirs / total if total > 0 else theirs
    share = 1 - perf.active_share(their, us) if not us.empty else None
    st.metric(
        "보유 겹침",
        portfolio.percent(share) if share is not None else PF_DASH,
        help=(
            f"액티브 셰어 {portfolio.percent(1 - share)} (= 1 − 겹침)"
            if share is not None
            else "내 포트폴리오에 미국 주식이 없습니다"
        ),
        border=True,
    )
    shared = investors.overlap(their, us)
    if shared:
        pf_table(
            {
                "티커": [t for t, _, _ in shared],
                "투자자 비중": [portfolio.percent(a) for _, a, _ in shared],
                "내 미국 주식 중": [portfolio.percent(b) for _, _, b in shared],
            },
            "겹치는 종목",
        )
    st.caption(
        f"내 포트폴리오의 미국 주식만(지금 평가액, 합 100%)과 투자자 {period} 보고"
        " 비중(티커가 연결된 종목, 합 100%) · 겹침 = Σ 종목마다 작은 쪽 비중 ="
        " 1 − 액티브 셰어(½Σ|비중 차이|) · 실력이나 성과를 재는 지표가 아닙니다"
    )


def inv_compare(
    s: Subject,
    data: InvData,
    book: portfolio.Book,
    prices: portfolio.Prices,
    months: int,
) -> None:
    _, found, cusips, _ = data
    mine = {p.id: p.name for p in book.portfolios if p.trades}
    if not mine:
        st.info("비교할 내 포트폴리오가 없습니다. **내 포트폴리오**에서 만듭니다.")
        return
    if state.get("inv_mine") not in mine:
        state.inv_mine = next(iter(mine))
    st.selectbox(
        "내 포트폴리오",
        list(mine),
        format_func=mine.__getitem__,
        key="inv_mine",
        width=280,
    )
    p = book.get(state.inv_mine)
    if p is None:
        return
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**보유 겹침** · 미국 주식")
        theirs = investors.weights(found[0].rows, cusips)
        own = portfolio.weights(portfolio.holdings(p, prices.close))
        inv_overlap(theirs, own, found[0].period)
    with right, st.container(border=True):
        st.markdown(f"**위험 지표** · USD, {state.get('pf_window') or '3Y'}")
        usd = inv_usd(p)
        held = cached_held(pf_key(usd.model_dump_json()), usd, prices.adjusted)
        sp_days = portfolio.trading_days(prices.adjusted, [portfolio.BENCHMARK])
        rows = {
            s.label: (s.key, s.rets),
            p.name: (pf_key(usd.model_dump_json()), held),
            portfolio.BENCHMARK_LABEL: (
                pf_key("benchmark"),
                inv_sp500(prices.adjusted, sp_days).dropna(),
            ),
        }
        stats = {
            name: inv_ratios(key, rets, prices, months)
            for name, (key, rets) in rows.items()
        }
        windows = {
            name: portfolio.window(rets, months).dropna().to_numpy(dtype="float64")
            for name, (_, rets) in rows.items()
        }
        korean: set[str] = {p.name} if portfolio.korean_weight(own) else set()
        betas = {
            name: portfolio.market_beta(
                portfolio.window(rets, months), prices.adjusted, "USD"
            )
            if name not in korean
            else math.nan
            for name, (_, rets) in rows.items()
        }
        pf_table(
            {
                "구분": list(rows),
                "기간 수익률": [
                    portfolio.percent(float((1 + w).prod()) - 1, signed=True)
                    if len(w)
                    else portfolio.UNKNOWN
                    for w in windows.values()
                ],
                "연 변동성": [
                    portfolio.percent(r.volatility)
                    if r.volatility is not None
                    else PF_DASH
                    for r in stats.values()
                ],
                "최대 낙폭": [
                    portfolio.percent(r.under.max_drawdown) if r.under else PF_DASH
                    for r in stats.values()
                ],
                "샤프 (연)": [pf_ratio_or(r.sharpe) for r in stats.values()],
                "베타": [
                    pf_ratio(b) if math.isfinite(b) else PF_DASH for b in betas.values()
                ],
                "일수": [f"{r.days:,}" for r in stats.values()],
            },
            "위험 지표 비교",
        )
        st.caption(
            f"투자자는 {s.label}, 내 포트폴리오는 보유 기록의 수정 종가 수익률 ·"
            f" 샤프는 무위험수익률 차감 · 베타는 {portfolio.BENCHMARK_LABEL} 대비"
            " Welch"
            + (" · 한국 종목이 있으면 베타 — (종가 시각 차이)" if korean else "")
        )


# ---- method


def inv_method() -> None:
    lines = (
        f"출처: SEC EDGAR 13F-HR·13F-HR/A 정보표, 최근 {investors.QUARTERS}개 분기 ·"
        " 티커는 OpenFIGI(CUSIP → 미국 종합 티커) · 섹터·가격은 yfinance",
        f"가치: {investors.DOLLARS_FROM} 이후 제출분은 달러, 이전 제출분은 천 달러로"
        " 보고되어 달러로 환산",
        "같은 CUSIP의 여러 행(운용역별)은 주식 수와 가치를 합산 · 풋·콜 옵션과 원금"
        " 단위(PRN) 행은 보유에서 제외",
        "정정: RESTATEMENT는 그 분기를 바꾸고 NEW HOLDINGS는 더함. 유형을 알 수 없거나"
        " 정보표가 없는 정정은 반영하지 않음 · 기밀 처리된 종목은 빠져 있다가 나중에"
        " 정정으로 공개될 수 있음",
        "직전 분기 대비: 주식 수 비교(분할 미조정) · 비중: 보고 가치 기준",
        f"가격·섹터는 분기마다 가치가 큰 종목부터 보고 가치의"
        f" {investors.COVER_SHARE:.0%}까지(최대 {investors.COVER_MAX}종목)만 받음."
        " 나머지는 미포함 비중으로 표시하고 표에는 모두 나옴",
        "미포함이거나 티커를 찾지 못했거나 가격이 없는 종목은 수익률·위험 계산에서"
        " 빼고 나머지를 100%로 재조정 (가격 미연결 비중으로 표시)",
        f"{INV_VARIANTS['filed']}: EDGAR 접수 시각(뉴욕)이 16:00 전이면 그날, 16:00"
        " 이후면 다음날 종가에 보고 비중으로 사서 다음 제출까지 보유(비중은 가격대로"
        " 변함). 휴장이면 다음 거래일, 접수 시각이 없으면 제출일 다음 거래일. 정정도"
        " 그 접수 시각에 반영하고, 더 최근 분기가 공개된 뒤의 옛 분기 정정은 무시",
        f"{INV_VARIANTS['top']}: 같은 날, 보고 비중이 큰 {investors.CONVICTION}종목만"
        " 보고 가치 비중대로(또는 같은 비중으로) 삼 · 회전율: 직전 보유가 가격대로 변한"
        " 비중과 새 비중의 차이 합의 절반",
        f"보고 중단: 마감(분기 말 + {investors.DEADLINE_DAYS}일)이 지난 최근 분기보다"
        f" {investors.STALE_QUARTERS}개 분기 이상 늦은 마지막 보고 · 13F-NT는 다른"
        " 운용사가 대신 보고했다는 신고",
        f"{INV_THEORY}: 분기 말(휴장이면 다음 거래일) 종가에 산 값. {INV_THEORY_NOTE}",
        "수정 종가(배당·분할 반영), USD. 기간 중 가격이 끊긴 종목(인수·상장폐지)은"
        " 마지막 가격에 현금으로 둠. 가격 기록이 없는 종목은 빠져 생존 편향이 있음",
        "내 포트폴리오는 일별 USD/KRW로 USD 환산 (포트폴리오 페이지와 같은 방식)",
        "위험지표·상관·시뮬레이션·VaR·스트레스: 내 포트폴리오와 같은 방법. 복제의"
        " 일간 수익률로 샤프·PSR·낙폭을, 지금 비중(마지막 매수 뒤 가격대로 변한 비중)"
        " 으로 상관·몬테카를로·VaR·스트레스를 계산. 기간 중 자료가 시작하거나 끝난"
        " 종목은 빼고 재조정 · 무위험수익률은 미국 3개월 국채(^IRX), 벤치마크는"
        " S&P 500",
        "포트폴리오 비교 상관: 투자자는 최근 분기 보고 비중, 내 포트폴리오는 지금"
        " 비중의 일간 수익률을 모두 값이 있는 날에 맞춤. 한국 비중이 절반을 넘는"
        " 포트폴리오는 한국 종가로 보고 시차 보정, Ledoit–Wolf 축소",
        "투자자 비교: 복제의 샤프를 표준오차(Mertens)만큼 여럿의 평균 쪽으로 당긴"
        " 경험적 베이즈 값을 함께 표시 · DSR(Bailey–López de Prado 2014): 비교한"
        f" 수만큼의 운에서 기대되는 최고 샤프를 넘을 확률, {perf.MIN_OBS}일 이상",
        "보유 겹침: 미국 주식 부분만 비교, 1 − 액티브 셰어(Cremers–Petajisto 2009)",
        f"티커 조회는 OpenFIGI 무키 한도(요청당 {investors.FIGI_JOBS}건, 분당 25회)를"
        f" 지키고, 찾지 못한 CUSIP은 {investors.MISS_RETRY_DAYS}일 뒤에 다시 조회",
    )
    with st.expander("방법·가정", icon=":material/info:"):
        st.markdown("\n".join(f"- {line}" for line in lines))


# ---- the view


def inv_flags(filer: investors.Filer | None) -> None:
    """보고 중단 and 13F-NT, when they apply."""
    stopped = investors.stale(filer, datetime.now().astimezone().date())
    notice = investors.notice(filer)
    if not stopped and not notice:
        return
    with st.container(horizontal=True, vertical_alignment="center"):
        if stopped:
            st.badge(f"보고 중단 · 마지막 {stopped}", color="orange")
        if notice:
            st.caption(f"최근 분기 보고 없음 ({notice.form}, {notice.filed})")


def inv_section(  # noqa: PLR0913 - the view's inputs
    section: str,
    key: Key,
    book: portfolio.Book,
    people: list[investors.Investor],
    data: InvData,
    prices: portfolio.Prices,
) -> None:
    _, found, cusips, sectors = data
    if section == "holdings":
        inv_holdings(found, cusips, sectors, prices)
        return
    months = inv_controls(section)
    top, equal = inv_variant()
    if section == "returns":
        inv_returns(key, data, book, prices, top=top, equal=equal)
        return
    s = inv_subject(key, data, prices, top=top, equal=equal)
    if s is None:
        pf_no_prices()
    elif section == "analysis":
        pf_stats(s, prices, months)
        pf_style_box(s, prices, "inv_style")
    elif section == "correlation":
        inv_correlation(s, book, people, prices, months, top=top, equal=equal)
    elif section == "risk":
        pf_risk(s, prices, months)
    elif section == "stress":
        pf_stress_section(s, prices, months)
    else:
        inv_compare(s, data, book, prices, months)


def inv_view(
    book: portfolio.Book,
    prices: portfolio.Prices,
    pfetch: portfolio.Fetcher,
    missing: list[str],
) -> None:
    root = inv_root()
    agent = sec_agent()
    fetcher = investor_fetcher(str(root))
    blocked = inv_blocked(agent, missing)
    people = investors.roster(root)
    person = inv_selector(people, root, blocked)
    if state.get("inv_dialog"):
        st.dialog("CIK 추가", on_dismiss=inv_close)(inv_dialog_body)(root, agent)
    cik = person.cik
    if (
        not blocked
        and investors.load_filer(root, cik) is None
        and not fetcher.tried(cik)
    ):
        fetcher.start(cik, agent)  # first opened without data
    inv_bar(fetcher, pfetch, root, cik, agent, blocked)
    stamp = investors.stamp(root, cik)  # before loading, like the prices
    data = inv_load(str(root), cik, stamp)
    inv_flags(data[0])
    found = data[1]
    if not found:
        if not fetcher.progress.running:
            st.info(
                "저장된 13F 자료가 없습니다."
                + ("" if blocked else " **갱신**으로 받습니다.")
            )
        return
    hold = bool(blocked) or fetcher.progress.running
    inv_prices(pfetch, hold, cik, investors.tickers(found, data[2]), prices)
    st.info(INV_LIMITS, icon=":material/info:")
    section = st.segmented_control(
        "보기",
        list(INV_SECTIONS),
        format_func=INV_SECTIONS.__getitem__,
        default="holdings",
        required=True,
        key="inv_section",
        label_visibility="collapsed",
    )
    key = pf_key(cik, stamp)
    inv_section(section or "holdings", key, book, people, data, prices)
    inv_method()


# ---- page sections


def pf_section(
    section: str,
    book: portfolio.Book,
    p: portfolio.Portfolio,
    prices: portfolio.Prices,
) -> None:
    if not p.trades:
        st.info("거래를 추가하세요. **보유·거래**에서 넣습니다.")
        return
    if not prices.tickers & {t.ticker for t in p.trades}:
        pf_no_prices()
        return
    if section == "returns":
        pf_returns(p, prices)
        return
    window = st.segmented_control(
        "기간", list(portfolio.WINDOWS), default="3Y", required=True, key="pf_window"
    )
    months = portfolio.WINDOWS[window or "3Y"]
    if section == "analysis":
        pf_analysis(p, prices, months)
        return
    s = pf_subject(p, prices)
    if section == "correlation":
        compare = partial(pf_book_heatmap, book, prices, p.base, months)
        pf_correlation(s, prices, months, compare)
    elif section == "stress":
        pf_stress_section(s, prices, months)
    else:
        pf_risk(s, prices, months)


# ---------------------------------------------------------------- pages


def overview() -> None:
    regime(data)
    tape(data)
    left, right = st.columns(2)
    with left:
        curves(data)
    with right:
        projections(data)


def panel_page(ids: tuple[str, ...]) -> Callable[[], None]:
    def show() -> None:
        panels(data, ids)

    return show


def schedule() -> None:
    calendar(data)
    provenance(data)


def screening() -> None:
    st.header("S&P 500 스크리닝", anchor="screen")
    root = resolve_settings(args.settings).parent / screen.SNAPSHOT_DIR
    screen_bar(screen_updater(str(root)), root)
    try:
        table = screen_table(str(root), screen.stamp(root))
    except (OSError, ValueError, KeyError) as error:
        st.error(f"`{screen.SNAPSHOT_DIR}/`의 파일을 읽지 못했습니다: {error}")
        return
    if table is None:
        st.info(
            "아직 받은 데이터가 없습니다. **갱신**을 누르면 S&P 500 종목을 받아"
            f" `{screen.SNAPSHOT_DIR}/`에 저장합니다. 처음에는 시세와 재무를 모두"
            " 받아 오래 걸립니다."
        )
        return
    screen_results(root, table)


def portfolio_page() -> None:
    st.header("포트폴리오", anchor="portfolio")
    root = pf_root()
    try:
        book = portfolio.load_book(root)
    except portfolio.BookError as error:
        st.error(str(error))
        return
    missing = portfolio.missing_packages()
    fetcher = portfolio_fetcher(str(root))
    state.pf_prices_stamp = pf_files_stamp(root)  # before loading: see pf_key
    prices = pf_prices(str(root), state.pf_prices_stamp)
    if portfolio.book_tickers(book) and not args.demo and not missing:
        # Tickers without saved prices, e.g. just traded, are fetched once.
        fetcher.start(fetcher.untried(portfolio.wanted(book), prices.tickers))
        fetcher.deepen(prices)  # the indices back to 1987, once
    mode = st.segmented_control(
        "구분",
        list(PF_MODES),
        format_func=PF_MODES.__getitem__,
        default="mine",
        required=True,
        key="pf_mode",
        label_visibility="collapsed",
    )
    if mode == "investors":
        inv_view(book, prices, fetcher, missing)
        return
    if mode == "stock":
        stock_view(book, prices, fetcher, missing)
        return
    p = pf_selector(book)
    pf_dialogs(p)
    pf_bar(fetcher, root, book, missing)
    if p is None:
        st.info("포트폴리오가 없습니다. **새로 만들기**로 시작합니다.")
        return
    section = st.segmented_control(
        "보기",
        list(PF_SECTIONS),
        format_func=PF_SECTIONS.__getitem__,
        default="holdings",
        required=True,
        key="pf_section",
        label_visibility="collapsed",
    )
    if section in {None, "holdings"}:
        pf_holdings(p, prices)
    else:
        pf_section(section, book, p, prices)
    pf_method(p, prices, state.get("pf_window") or "3Y")


# ---------------------------------------------------------------- app

with st.container(horizontal=True, vertical_alignment="center"):
    st.title("FinAgent", anchor=False)
    st.space("stretch")
    clock_slot = st.empty()
    open_settings = st.button("설정", icon=":material/tune:", shortcut="S")
    st.button("새로고침", icon=":material/refresh:", type="primary", on_click=refresh)

if state.toast:
    st.toast(state.toast)
    state.toast = ""

with st.spinner("공개 API에서 받는 중…"):
    data = rt.call(rt.dashboard.get())

with clock_slot:
    clocks(data.generated_at)

if open_settings:
    for v in data.views:
        state[f"view_{v.id}"] = v.id in data.deep
    settings_dialog(data)

page = st.navigation(
    [
        st.Page(overview, title="개요", icon=":material/dashboard:", default=True),
        *(
            st.Page(panel_page(ids), title=title, icon=icon, url_path=path)
            for path, title, icon, ids in display.PANEL_PAGES
        ),
        st.Page(schedule, title="일정·출처", icon=":material/event:"),
        st.Page(screening, title="스크리닝", icon=":material/filter_alt:"),
        st.Page(
            portfolio_page,
            title="포트폴리오",
            icon=":material/pie_chart:",
            url_path="portfolio",
        ),
    ],
    position="top",
)
page_panels = next(
    (ids for path, _, _, ids in display.PANEL_PAGES if path == page.url_path), ()
)
notices(data)
page.run()

st.html(AI_CSS.format(bg="#0e1117" if st.context.theme.type == "dark" else "#fff"))
with st.container(key="ai_fab"):
    st.button(
        "",
        icon=":material/close:" if state.ai_open else ":material/smart_toy:",
        key="ai_toggle",
        help="AI 분석",
        type="primary",
        shortcut="A",
        on_click=toggle_ai,
    )
if state.ai_open:
    with st.container(key="ai_panel", border=True):
        analysis(data)
