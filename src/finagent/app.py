"""The dashboard page. ``finagent`` starts it with Streamlit on 127.0.0.1.

Script arguments (after ``--``) are the CLI's ``--demo``, ``--env-file`` and
``--settings``. The services live in one shared runtime, so every browser tab
sees the same view, snapshot and analysis lock.
"""

# Session state is untyped, element calls return a DeltaGenerator the page
# never uses, and page prose is split across lines.
# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false, reportUnknownMemberType=false

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import altair as alt
import streamlit as st

from finagent import display
from finagent.analysis.presets import PLACEHOLDER, PRESET_BY_ID, PRESETS
from finagent.analysis.providers.base import ProviderError
from finagent.analysis.service import AnalysisService, RunInput
from finagent.cli import build_service, parse_args, resolve_env, resolve_settings
from finagent.runtime import Run, Runtime

if TYPE_CHECKING:
    from finagent.models import Curve, Dashboard, Panel, Spread

CLOCKS = (("NY", "America/New_York"), ("SEOUL", "Asia/Seoul"), ("UTC", "UTC"))
SPREAD_RANGES = {"2Y": 24, "5Y": 60, "전체": 0}
SCOPES = {"view": "지금 보기", "all": "전체 매크로"}


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


@st.dialog("보기 설정", width="large")
def settings_dialog(d: Dashboard) -> None:
    st.markdown(
        "기본 화면은 기업 분석에 쓰는 거시 지표만 보여줍니다. 경제 지표, 주요국 "
        "통화정책, 경기 주기(부채·통화량), 시장 심리 지표입니다. 매크로를 더 깊게 "
        "볼 패널만 켜세요. 켠 패널의 나머지 지표만 새로 받아옵니다."
    )
    with st.container(horizontal=True):
        st.button("핵심만", on_click=check_all, args=(d,), kwargs={"on": False})
        st.button("전체 매크로", on_click=check_all, args=(d,), kwargs={"on": True})
    for view in d.views:
        note = display.view_note(view.core, view.extra, curve=view.id == "curve")
        st.checkbox(
            f"**{view.key}** {view.title} :gray[{note}]",
            key=f"view_{view.id}",
            disabled=not view.extra,
        )
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
def clocks() -> None:
    now = datetime.now().astimezone()
    st.caption(
        " · ".join(
            f"**{label}** {now.astimezone(ZoneInfo(zone)):%H:%M}"
            for label, zone in CLOCKS
        )
    )


def status(d: Dashboard) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        for source in d.sources:
            st.badge(
                f"{source.name} {source.ok}/{source.total}",
                color=display.SOURCE_COLOR[source.state],
                help=source.message,
            )
        total = sum(source.ok for source in d.sources)
        st.caption(
            f"보기 {display.view_label(d)} · v{d.version} · 메모리 {total}개 시리즈"
            f" · 디스크 저장 0 · {d.elapsed_seconds:.1f}s"
            f" · 갱신 {display.stamp(d.generated_at)}"
        )


def notices(d: Dashboard) -> None:
    if d.settings_error:
        st.error(d.settings_error)
    if d.demo:
        st.info(
            "데모 모드: 수치는 모두 예시 값입니다. 실제 데이터는 `finagent`를 키와"
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
    rows = display.rows_by_id(d)
    with st.container(horizontal=True):
        for tape_id in d.tape:
            row = rows.get(tape_id)
            if row is None:
                continue
            delta = display.metric_delta(row)
            flat = display.change(row) == 0
            st.metric(
                display.TAPE_LABELS.get(tape_id, row.label),
                display.fmt(row.value, row.fmt),
                delta,
                delta_color="off" if flat else "normal",
                delta_arrow="off" if flat else "auto",
                help=row.code,
                border=True,
            )


def regime(d: Dashboard) -> None:
    st.header("국면 요약 · 공개 규칙 기준", anchor="overview")
    with st.container(horizontal=True):
        for cell in d.regime:
            color, icon = display.TONE[cell.tone]
            with st.container(border=True):
                st.caption(cell.label)
                st.badge(f"{icon} {cell.tag}", color=color)
                st.markdown(f"### {cell.value} :gray[{cell.unit}]")
                st.caption(cell.rule)


def curves(d: Dashboard) -> None:
    st.header("금리곡선", anchor="curve")
    show_curve = "curve" in d.deep
    left, right = st.columns(2) if show_curve else (None, st.container())
    if left is not None:
        with left, st.container(border=True):
            st.markdown("**YC** 미 국채 수익률 곡선 :gray[FRED H.15 · DGS1MO~DGS30]")
            if d.curve is None:
                st.caption("FRED 키가 있어야 표시됩니다")
            else:
                st.altair_chart(curve_chart(d.curve), alt="미 국채 수익률 곡선")
            st.caption("만기는 동일 간격 표시 · 단위 %")
    with right, st.container(border=True):
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
            st.altair_chart(spread_chart(spread, months), alt="10년물 − 2년물 금리차")
        st.caption("T10Y2Y 월평균 %p · 음영은 NBER 경기침체(USREC) · 0 아래는 역전")


def projections(d: Dashboard) -> None:
    st.header("FOMC 경제전망 · 위원 중앙값", anchor="sep")
    p = d.projections
    with st.container(border=True):
        if p is None:
            st.caption("FRED 키가 있어야 표시됩니다.")
            return
        if not p.rows:
            st.caption(f"받지 못했습니다. {p.message or ''}")
            return
        since = (
            f" · 작은 화살표는 {p.prior_published} 발표 대비"
            if p.prior_published
            else ""
        )
        st.markdown(f"**SEP** FOMC 경제전망 요약 :gray[{p.published} 발표{since}]")
        st.dataframe(display.sep_table(p), hide_index=True, alt="FOMC 경제전망 요약")
        if p.message:
            st.caption(f"일부를 받지 못했습니다. {p.message}")


def panel(d: Dashboard, p: Panel) -> None:
    view = next((v for v in d.views if v.id == p.id), None)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(f"{p.key} · {p.title}", anchor=p.id)
            if view is not None and view.extra:
                on = p.id in d.deep
                st.button(
                    "핵심만" if on else f"+{view.extra} 심화",
                    key=f"more_{p.id}",
                    help="핵심 지표만 보기"
                    if on
                    else f"나머지 {view.extra}개 지표도 보기",
                    type="tertiary",
                    on_click=toggle_deep,
                    args=(p.id,),
                )
        st.caption(p.note)
        st.dataframe(
            display.panel_table(p),
            hide_index=True,
            height="content",
            column_config={
                "지표": st.column_config.TextColumn(width="medium"),
                "추이": st.column_config.LineChartColumn(width="small"),
            },
            alt=p.title,
        )
        for problem in display.panel_problems(p):
            st.caption(f":red[{problem}]")


def calendar(d: Dashboard) -> None:
    with st.container(border=True):
        st.subheader("CAL · 앞으로 7일 미국 발표", anchor="calendar")
        st.caption("FRED releases/dates")
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


def panels(d: Dashboard) -> None:
    st.header("지표 · 최신 · 직전 대비 · 추이", anchor="panels")
    cards = [*d.panels, None]
    for start in range(0, len(cards), 3):
        for column, card in zip(st.columns(3), cards[start : start + 3], strict=False):
            with column:
                if card is None:
                    calendar(d)
                else:
                    panel(d, card)


def sources() -> None:
    st.header("데이터", anchor="data")
    st.markdown(
        "- **키 2개면 충분.** FRED와 한국은행 ECOS만 무료 개인 키가 필요합니다."
        " 미 재무부, 뉴욕 연은, BIS, ECB, OECD는 키 없이 열려 있습니다.\n"
        "- **저장하지 않음.** 받은 시계열은 앱이 켜져 있는 동안 메모리에만 있습니다."
        " 디스크에는 보기 설정(패널 목록)만 저장합니다. 갱신은 앱을 시작할 때 한 번,"
        " 그리고 새로고침을 누를 때뿐입니다.\n"
        "- **출처 표기.** 각 지표의 코드가 원 출처입니다. 예: `FRED:UNRATE`,"
        " `ECOS:722Y001/0101000`, `BIS:WS_CBPOL/JP`."
    )


def navigation(d: Dashboard) -> None:
    links = [("개요", "overview"), ("금리곡선", "curve"), ("FOMC 전망", "sep")]
    links += [(f"{p.key} · {p.title}", p.id) for p in d.panels]
    links += [("앞으로 7일 발표", "calendar"), ("데이터", "data")]
    with st.sidebar:
        st.markdown("\n".join(f"- [{label}](#{anchor})" for label, anchor in links))


# ---------------------------------------------------------------- analysis


def choose_provider() -> None:
    provider_id: str = state.ai_provider
    model: str = state.get(f"ai_model_{provider_id}", "")
    try:
        _ = rt.analysis.choose(provider_id, model or rt.analysis.model_for(provider_id))
    except (ProviderError, OSError) as error:
        state.ai_error = f"선택을 저장하지 못했습니다: {error}"
    else:
        state.ai_error = ""


def fill_preset() -> None:
    preset = PRESET_BY_ID.get(state.ai_preset or "")
    if preset is not None:
        state.ai_input = preset.text


def input_changed() -> None:
    if not state.ai_input.strip():
        state.ai_preset = None


def new_conversation() -> None:
    if state.ai_run is None:
        state.ai_log = display.AnalysisLog()


def request_text(d: Dashboard) -> str:
    text: str = state.ai_input.strip()
    if PLACEHOLDER in text:
        row = display.rows_by_id(d).get(state.get("ai_indicator") or "")
        if row is not None:
            text = text.replace(PLACEHOLDER, f"{row.label} ({row.id})")
    return text


def start_run(d: Dashboard) -> bool:
    text = request_text(d)
    if not text:
        return False
    provider_id: str = state.ai_provider
    log = conversation()
    log.ask(text)
    state.ai_run = rt.start(
        RunInput(
            request=text,
            provider=provider_id,
            model=state.get(f"ai_model_{provider_id}", "").strip(),
            conversation=log.conversation,
            preset="" if log.conversation else state.ai_preset or "",
            scope="all" if state.get("ai_scope") == "all" else "view",
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
        if done:
            state.ai_input = ""
            state.ai_preset = None
        st.rerun()


def controls(d: Dashboard, statuses: dict[str, str]) -> None:
    providers = rt.analysis.providers
    running = state.ai_run is not None
    log = conversation()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        provider_id = st.selectbox(
            "공급자",
            list(providers),
            index=list(providers).index(rt.analysis.choice.provider),
            format_func=lambda pid: f"{providers[pid].title} · {statuses[pid]}",
            key="ai_provider",
            on_change=choose_provider,
        )
        st.text_input(
            "모델",
            value=rt.analysis.model_for(provider_id),
            placeholder=providers[provider_id].default_model,
            key=f"ai_model_{provider_id}",
            on_change=choose_provider,
        )
        st.segmented_control(
            "새 대화의 범위",
            list(SCOPES),
            format_func=SCOPES.__getitem__,
            default="view",
            key="ai_scope",
            disabled=bool(log.conversation),
            help="범위는 새 대화에서 바꿀 수 있습니다" if log.conversation else None,
        )
    st.pills(
        "프리셋",
        [preset.id for preset in PRESETS],
        format_func=lambda pid: PRESET_BY_ID[pid].title,
        key="ai_preset",
        on_change=fill_preset,
        label_visibility="collapsed",
    )
    if PLACEHOLDER in state.get("ai_input", ""):
        names = {
            row.id: f"{p.title} · {row.label}"
            for p in d.panels
            for row in p.rows
            if row.status == "ok"
        }
        st.selectbox(
            "지표", list(names), format_func=names.__getitem__, key="ai_indicator"
        )
    st.text_area(
        "요청",
        key="ai_input",
        placeholder=(
            "무엇을 분석할까요? 화면의 지표와 도구로 받은 공개 데이터만"
            " 근거로 답합니다."
        ),
        label_visibility="collapsed",
        on_change=input_changed,
    )
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button(
            "실행", type="primary", shortcut="Ctrl+Enter", disabled=running
        ) and start_run(d):
            st.rerun()
        if running and st.button("멈춤") and state.ai_run is not None:
            log.stopped = True
            state.ai_run.stop()
        st.button("새 대화", on_click=new_conversation, disabled=running)
        if log.conversation:
            st.caption(f"대화 {log.conversation}")


def turns(d: Dashboard) -> None:
    links = display.panel_of(d)
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
        with st.container(border=True):
            st.subheader("AI 매크로 분석 · 선택 기능", anchor="analysis")
            st.caption(log.header)
            controls(d, statuses)
            if state.ai_error:
                st.error(state.ai_error)
            else:
                pid: str = state.ai_provider
                st.caption(statuses[pid])
            if log.stale and log.conversation:
                st.warning(
                    f"이 대화는 {display.short_time(log.snapshot)} 데이터에 고정돼"
                    " 있습니다. 새로 받은 데이터로 보려면 새 대화를 시작하세요."
                )
            turns(d)
            if log.received:
                with st.expander("받은 데이터"):
                    st.markdown(
                        "\n".join(
                            f"- `{cite}` {display.md(info)}"
                            for cite, info in log.received
                        )
                    )

    body()


# ---------------------------------------------------------------- page

with st.container(horizontal=True, vertical_alignment="center"):
    st.title("FinAgent", anchor=False)
    clocks()
    st.button("AI 분석", icon=":material/smart_toy:", shortcut="A", on_click=toggle_ai)
    open_settings = st.button("설정", icon=":material/tune:", shortcut="S")
    st.button("새로고침", icon=":material/refresh:", type="primary", on_click=refresh)

if state.toast:
    st.toast(state.toast)
    state.toast = ""

with st.spinner("공개 API에서 받는 중…"):
    data = rt.call(rt.dashboard.get())

if open_settings:
    for v in data.views:
        state[f"view_{v.id}"] = v.id in data.deep
    settings_dialog(data)

navigation(data)
status(data)
notices(data)
tape(data)
if state.ai_open:
    analysis(data)
regime(data)
curves(data)
projections(data)
panels(data)
sources()
