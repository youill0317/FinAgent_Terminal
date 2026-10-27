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

import contextlib
import dataclasses
import itertools
import math
import os
import sys
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent import (
    catalog,
    colab_jobs,
    demo_results,
    demo_workspace,
    display,
    generative,
    investors,
    perf,
    portfolio,
    quant,
    screen,
)
from finagent.analysis.presets import PRESETS
from finagent.analysis.providers.base import ProviderError
from finagent.analysis.service import AnalysisService, RunInput
from finagent.cli import build_service, parse_args, resolve_env, resolve_settings
from finagent.dashboard import REGIME_HELP
from finagent.fetch import SourceError
from finagent.guides import GUIDES
from finagent.runtime import Run, Runtime
from finagent.settings import (
    ScreenChoice,
    TrainingChoice,
    load_screen,
    load_training,
    save_screen,
    save_training,
)
from finagent.sources import EcosSeries, FredSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from numpy.typing import NDArray
    from streamlit.delta_generator import DeltaGenerator

    from finagent.models import Curve, Dashboard, IndicatorRow, Panel, Spread
    from finagent.sources import Fetch
    from finagent.sources import Series as Points

CLOCKS = (("NY", "America/New_York"), ("SEOUL", "Asia/Seoul"), ("UTC", "UTC"))
CHART_HEIGHT = 240
LIST_HEIGHT = 560  # long tables: screening results, investor holdings
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
# Equal cards on a grid: a horizontal container sizes each card by its
# content. auto-fill keeps a card the same width however many share the row,
# so one or two cards do not stretch across it, and 1fr rows keep a wrapped
# card as tall as the row above. 188px fits the 최대 낙폭 dates; 160px keeps
# the eight 고급 cards on one line at 1600px wide; 260px fits amounts and the
# 난기류 pill; 280px the what-if "before → after" values.
CARDS_CSS = """<style>
.st-key-pf_risk_cards, .st-key-pf_spread_cards, .st-key-detail_cards,
.st-key-screen_stock_cards
  {display: grid; grid-auto-rows: 1fr;
  grid-template-columns: repeat(auto-fill, minmax(188px, 1fr))}
.st-key-pf_more_cards {display: grid; grid-auto-rows: 1fr;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr))}
.st-key-pf_now_cards, .st-key-pf_return_cards, .st-key-pf_trade_cards,
.st-key-pf_split_cards, .st-key-pf_turbulence_cards, .st-key-pf_reverse_cards,
.st-key-inv_now_cards, .st-key-inv_return_cards, .st-key-inv_overlap_cards,
.st-key-inv_stock_cards
  {display: grid; grid-auto-rows: 1fr;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr))}
.st-key-pf_whatif_cards, .st-key-pf_whatif_path_cards {display: grid;
  grid-auto-rows: 1fr; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr))}
</style>"""
FRED_NOTICE = "**조건** FRED API 키 · 환경변수 FRED_API_KEY 또는 .env"


def notice(text: str) -> None:
    """A "!" whose hover shows ``text``, like a heading's "?"."""
    st.badge("", icon=":material/priority_high:", color="orange", help=text)


def headed(text: str, help_: str | None = None, warns: Sequence[str] = ()) -> None:
    """A bold ``st.markdown`` heading, with a "!" beside it for ``warns``."""
    if not warns:
        st.markdown(text, help=help_)
        return
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(text, help=help_)
        notice("\n\n".join(warns))


@st.cache_resource(show_spinner=False)
def demo_settings() -> Path:
    """The demo's own settings and data, made once per process."""
    return demo_workspace.workspace()


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
if args.demo and args.settings is None:
    args.settings = demo_settings()
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
    provider_col, model_col, scope_col = st.columns(3)
    provider_id = provider_col.selectbox(
        "공급자",
        list(providers),
        index=list(providers).index(rt.analysis.choice.provider),
        format_func=lambda pid: (
            f"{providers[pid].title} · "
            f"{display.STATUS_TEXT.get(found[pid].kind, found[pid].kind)}"
        ),
        key="set_provider",
        help="**정의** AI 분석을 실행할 공급자 · 옆은 상태(준비됨 · 키 없음 ·"
        " 설치 안 됨)",
    )
    _ = model_col.text_input(
        "모델",
        value=rt.analysis.model_for(provider_id),
        placeholder=providers[provider_id].default_model,
        key=f"set_model_{provider_id}",
        help="**정의** 공급자에 보낼 모델 이름 · 비우면 기본 모델",
    )
    _ = scope_col.segmented_control(
        "새 대화의 범위",
        list(SCOPES),
        format_func=SCOPES.__getitem__,
        default=rt.analysis.choice.scope,
        key="set_scope",
        help="**정의** 새 대화가 읽는 지표 범위 · 지금 보기: 보기 설정에서 켠"
        " 지표 · 전체 매크로: 모든 지표\n\n**기준** 진행 중인 대화는 시작 시점의"
        " 범위 유지",
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


# ---- 생성 모델 학습 (Colab)

GEN_LABELS: dict[str, str] = {
    "tailgan": "Tail-GAN",
    "sigcwgan": "SigCWGAN",
    "nsde": "신경망 SDE",
    "kronos": "Kronos",
}


@st.cache_resource(show_spinner=False)
def trainer(root: str) -> colab_jobs.Trainer:
    """One training job runner per models folder; picks up an unfinished job."""
    jobs = colab_jobs.Trainer(Path(root))
    jobs.reconcile()
    return jobs


type Jobs = colab_jobs.Trainer | colab_jobs.DemoTrainer


def demo_trainer() -> colab_jobs.DemoTrainer:
    """The demo's canned stand-in, one per session so the dialog keeps it."""
    return cast(
        "colab_jobs.DemoTrainer",
        state.setdefault("train_demo", colab_jobs.DemoTrainer()),
    )


def gen_root() -> Path:
    return resolve_settings(args.settings).parent / generative.MODELS_DIR


def train_request() -> colab_jobs.Request | None:
    """The dialog's training choices; None while the dates are half picked."""
    span = cast("tuple[date, ...]", state.get("train_span") or ())
    if len(span) != 2:  # noqa: PLR2004 - a start and an end
        return None
    return colab_jobs.Request(
        tickers=(*(state.get("train_tickers") or ()), portfolio.FX),
        start=span[0].isoformat(),
        end=span[1].isoformat(),
        gpu=state.get("train_gpu") or colab_jobs.DEFAULT_GPU,
        models=tuple(state.get("train_models") or ()),
    )


def train_fetch_later() -> None:
    state.train_fetch = True


def priced_tickers(adjusted: pd.DataFrame) -> list[str]:
    priced = cast("list[object]", adjusted.dropna(axis=1, how="all").columns.tolist())
    return sorted(str(c) for c in priced)


def trainable(ticker: str) -> bool:
    """Offered in 종목: not an index or rate (^…), not USD/KRW (always trained)."""
    return not ticker.startswith("^") and ticker != portfolio.FX


def train_fetch(root: Path) -> None:
    """Fetch the typed tickers without saved prices; drop the ones that fail.

    Runs before the 종목 field is drawn, after it changed, so its value can
    still be set.
    """
    priced = priced_tickers(pf_prices(str(root), pf_files_stamp(root)).adjusted)
    kept: list[str] = []
    errors: list[str] = []
    for text in cast("list[str]", state.get("train_tickers") or []):
        try:
            ticker = portfolio.clean_ticker(text)
        except portfolio.BookError as error:
            errors.append(f"{text}: {error}")
            continue
        if ticker.startswith("^"):
            errors.append(f"{ticker}: 지수·금리는 학습하지 않습니다.")
        elif ticker != portfolio.FX:  # always trained; nothing to add
            kept.append(ticker)
    todo = [t for t in dict.fromkeys(kept) if t not in priced]
    if todo:
        with st.spinner(f"{', '.join(todo)} 가격 받는 중…"):
            try:
                adjusted = portfolio_fetcher(str(root)).fetch_now(todo).adjusted
            except Exception:  # noqa: BLE001 - shown in the dialog
                adjusted = portfolio.no_prices().adjusted
        got = priced_tickers(adjusted)
        failed = [t for t in todo if t not in got]
        if failed:
            errors.append(f"{', '.join(failed)} 가격을 받지 못했습니다.")
        kept = [t for t in kept if t not in failed]
    state.train_tickers = list(dict.fromkeys(kept))
    state.train_error = " · ".join(errors)


def apply_training() -> None:
    r = train_request()
    if args.demo or r is None or "train_tickers" not in state:
        return
    choice = TrainingChoice(r.tickers, r.start, r.end, r.gpu, r.models)
    if load_training(resolve_settings(args.settings)) == choice:
        return
    try:
        save_training(resolve_settings(args.settings), choice)
    except OSError as error:
        state.toast = f"학습 설정을 저장하지 못했습니다: {error}"


def start_training(jobs: Jobs, adjusted: pd.DataFrame) -> None:
    r = train_request()
    if r is None:
        return
    apply_training()
    _ = jobs.start(r, adjusted)


def hours(span: timedelta) -> str:
    seconds = int(span.total_seconds())
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}"


def busy(jobs: Jobs) -> str | None:
    """A disabled button's help while a job runs; None if none."""
    return "학습 중입니다" if jobs.running else None


def training_status(jobs: Jobs) -> None:
    job, last = jobs.job, jobs.last
    if job is not None:
        stage = colab_jobs.STAGES.get(job.stage, job.stage)
        if job.message:
            stage += f" ({job.message})"
        results = " · ".join(
            f"{GEN_LABELS.get(m, m)} {outcome}" for m, outcome in job.results.items()
        )
        st.markdown(
            f"**단계** {stage} · **경과** {hours(jobs.elapsed(job))}"
            + (f" · {results}" if results else "")
        )
    if last is not None:
        when = datetime.fromisoformat(last.finished).astimezone()
        r = last.request
        st.markdown(
            f"**마지막 학습** {when:%Y-%m-%d %H:%M} · {', '.join(r.tickers)}"
            f" · {r.start}–{r.end}"
        )
    canned = colab_jobs.DEMO_REQUEST
    current = (
        [
            (m, f"{canned.start}–{canned.end} 학습", canned.tickers)
            for m in canned.models
        ]
        if args.demo
        else [
            (s.model, s.note, s.tickers)
            for s in generative.status(gen_root())
            if s.model != "kronos"
        ]
    )
    st.markdown(
        "**현재 모델**\n"
        + "".join(
            f"\n- **{GEN_LABELS[m]}** {note}"
            + (f" · {', '.join(tickers)}" if tickers else "")
            for m, note, tickers in current
        )
    )


def training_settings() -> None:
    """Tickers, dates, GPU and models for retraining on Colab; 학습 시작 runs it.

    The demo shows the same controls on canned choices and jobs.
    """
    root = pf_root()
    jobs: Jobs
    typed = not args.demo and not portfolio.missing_packages()
    if args.demo:
        adjusted, jobs = colab_jobs.demo_prices(), demo_trainer()
        r = colab_jobs.DEMO_REQUEST
        saved = TrainingChoice(r.tickers, r.start, r.end, r.gpu, r.models)
    else:
        if typed and state.pop("train_fetch", False):
            train_fetch(root)
        adjusted = pf_prices(str(root), pf_files_stamp(root)).adjusted
        jobs = trainer(str(gen_root()))
        saved = load_training(resolve_settings(args.settings)) or TrainingChoice()
    if adjusted.empty:
        st.info("저장된 가격이 없습니다.")
        return
    options = [t for t in priced_tickers(adjusted) if trainable(t)]
    wanted = saved.tickers or portfolio.book_tickers(portfolio.load_book(root))
    tickers_col, span_col = st.columns(2)
    _ = tickers_col.multiselect(
        "종목",
        options,
        # Only on the first draw: train_fetch may have set the value.
        default=None
        if "train_tickers" in state
        else [t for t in dict.fromkeys(wanted) if t in options],
        key="train_tickers",
        on_change=train_fetch_later,
        accept_new_options=typed,
        placeholder="티커 입력 (예: NVDA, 005930.KS)" if typed else None,
        help="**정의** 생성 모델 학습 종목 · 포트폴리오 페이지에 수정 종가가 저장된"
        " 종목"
        + (" · 목록에 없는 티커는 입력하면 가격을 받아 추가" if typed else "")
        + f" · 환율 {portfolio.FX}는 항상 포함 · 지수·금리 제외"
        "\n\n**기준** 꼬리위험·전망 탭 사용 조건: 보유 종목을 모두 담은 모델 ·"
        f" 기준 통화와 다른 종목은 {portfolio.FX}로 환산",
    )
    if error := state.pop("train_error", ""):
        st.error(error)
    first = cast("pd.Timestamp", adjusted.index[0]).date()
    last = cast("pd.Timestamp", adjusted.index[-1]).date()
    start = min(max(date.fromisoformat(saved.start or first.isoformat()), first), last)
    end = min(max(date.fromisoformat(saved.end or last.isoformat()), first), last)
    if start > end:
        start, end = first, last
    _ = span_col.date_input(
        "학습 기간",
        value=(start, end),
        min_value=first,
        max_value=last,
        key="train_span",
        help="**정의** 학습 자료 기간 · 고른 모든 종목에 종가가 있는 날\n\n**기준**"
        " 시간순 앞 80% 학습 · 뒤 20% 평가",
    )
    request = train_request()
    if request is not None and request.tickers:
        rows = colab_jobs.returns_table(
            adjusted,
            request.tickers,
            date.fromisoformat(request.start),
            date.fromisoformat(request.end),
        )
        span = (
            f"{cast('pd.Timestamp', rows.index[0]):%Y-%m-%d}–"
            f"{cast('pd.Timestamp', rows.index[-1]):%Y-%m-%d}, {len(rows):,}일"
            if len(rows)
            else "없음"
        )
        span_col.caption(f"겹치는 기간 {span}")
    models_col, gpu_col = st.columns(2)
    _ = gpu_col.segmented_control(
        "Colab GPU",
        colab_jobs.GPUS,
        default=saved.gpu if saved.gpu in colab_jobs.GPUS else "T4",
        required=True,
        key="train_gpu",
        help="**정의** 학습에 쓸 Colab 런타임 GPU",
    )
    _ = models_col.multiselect(
        "모델",
        generative.METHODS,
        default=[m for m in generative.METHODS if m in saved.models]
        or list(generative.METHODS),
        format_func=GEN_LABELS.__getitem__,
        key="train_models",
        help="**정의** 학습할 생성 모델 · Tail-GAN(Cont 외) · SigCWGAN(Liao 외"
        " 2020) · 신경망 SDE(Issa 외 2023)",
    )
    training_status(jobs)
    request = train_request()
    blocked = (
        (None if args.demo else colab_jobs.blocker())
        or busy(jobs)
        or (None if state.get("train_tickers") else "종목을 고르세요")
        or (
            colab_jobs.problem(request, adjusted)
            if request is not None
            else "학습 기간을 고르세요"
        )
    )
    with st.container(horizontal=True):
        st.button(
            "학습 시작",
            icon=":material/play_arrow:",
            key="train_start",
            disabled=blocked is not None,
            help=blocked
            or "**정의** Colab VM 설치 뒤 고른 모델 순차 학습 · 받은 모델 검증 뒤"
            " 교체(이전 모델은 data/models/<모델>.prev)\n\n**기준** 상태 확인"
            f" {colab_jobs.POLL_SECONDS / 60:.0f}분 간격 · 종료 시 Colab 세션 중지",
            on_click=start_training,
            args=(jobs, adjusted),
        )
        if jobs.running:
            st.button(
                "중지",
                icon=":material/stop:",
                key="train_cancel",
                on_click=jobs.cancel,
            )


def view_settings(d: Dashboard, on: frozenset[str]) -> None:
    """The 보기 tab: a card of switches per page, three to a row."""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.button("핵심만", on_click=check_all, args=(d,), kwargs={"on": False})
        st.button("전체 매크로", on_click=check_all, args=(d,), kwargs={"on": True})
        st.caption(display.view_summary(d, on))
    sections = [("개요", ":material/dashboard:", ("curve",))] + [
        (title, icon, ids) for _, title, icon, ids in display.PANEL_PAGES
    ]
    for row in itertools.batched(sections, 3):
        for col, (title, icon, ids) in zip(
            st.columns(3, border=True), row, strict=False
        ):
            views = [view for view in d.views if view.id in ids]
            switchable = [view for view in views if view.extra]
            count = sum(view.id in on for view in switchable)
            col.markdown(f"{icon} **{title}** :gray[{count}/{len(switchable)}]")
            for view in views:
                note = display.view_note(
                    view.core, view.extra, curve=view.id == "curve"
                )
                col.checkbox(
                    f"{view.title} :gray[{note}]",
                    key=f"view_{view.id}",
                    disabled=not view.extra,
                )


@st.dialog("보기 설정", width="large")
def settings_dialog(d: Dashboard) -> None:
    on = frozenset(view.id for view in d.views if state.get(f"view_{view.id}"))
    # Fixed labels: a changed one would reopen the first tab.
    views, ai, training, criteria = st.tabs(
        [
            ":material/visibility: 보기",
            ":material/smart_toy: AI 분석",
            ":material/model_training: 생성 모델 학습",
            ":material/filter_alt: 스크리닝 기준",
        ]
    )
    with views:
        view_settings(d, on)
    with ai:
        ai_settings()
    with training:
        training_settings()
    with criteria:
        criteria_panel()
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("취소"):
            st.rerun()
        if st.button("적용", type="primary"):
            set_deep(on)
            apply_ai()
            apply_training()
            save_criteria()
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
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
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
            y=alt.Y("value:Q", title=None),
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
        st.badge(f"갱신 {updated:%H:%M}", color="gray")


def status(d: Dashboard) -> None:
    with st.container(horizontal=True, gap="xsmall", vertical_alignment="center"):
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
        st.info("데모 모드: 예시 값입니다.")
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
        headed(f"**{title}**", display.TAPE_HELP.get(title))
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
        label, value = display.regime_card(cell)
        col.metric(
            label,
            value,
            f"{icon} {cell.tag}",
            delta_color=color,
            delta_arrow="off",
            help=REGIME_HELP.get(cell.label, cell.rule),
            border=True,
            height="stretch",
            chart_data=display.trend(source),
        )


def curves(d: Dashboard) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.header("금리곡선", anchor="curve", width="content")
    spread = d.spread
    missing = spread is None or len(spread.points) < 2  # noqa: PLR2004
    with st.container(border=True, height="stretch"):
        with st.container(horizontal=True, vertical_alignment="center"):
            headed(
                "**2s10s** · 10년물 − 2년물, %p",
                "**정의** 미 국채 10년물 − 2년물 수익률 차(T10Y2Y)의 월평균(%p) ·"
                " 회색 음영 NBER 경기침체 기간(USREC)\n\n**해석** 0 아래 장단기 역전 ·"
                " 경기침체 선행 신호로 쓰이는 구간",
                [FRED_NOTICE] if missing else [],
            )
            picked = st.segmented_control(
                "기간",
                list(display.CHART_RANGES),
                default="5Y",
                key="spread_range",
                label_visibility="collapsed",
            )
        if spread is not None and not missing:
            months = display.CHART_RANGES[picked or "전체"]
            st.altair_chart(
                spread_chart(spread, months),
                height=CHART_HEIGHT,
                alt="10년물 − 2년물 금리차",
            )
        else:
            st.caption("2s10s 금리차 자료가 없습니다.")
    if "curve" in d.deep:
        with st.container(border=True):
            headed(
                "**미 국채 수익률 곡선** · %",
                "**정의** 만기별 미 국채 수익률(FRED DGS1MO~DGS30, %) · 최신 ·"
                " 1개월 전 · 1년 전 곡선\n\n**해석** 우상향은 정상 · 평탄화·역전은"
                " 단기 금리의"
                " 상대적 상승 · 곡선 간 간격은 금리 수준 변화",
                [FRED_NOTICE] if d.curve is None else [],
            )
            if d.curve is not None:
                st.altair_chart(
                    curve_chart(d.curve), height=CHART_HEIGHT, alt="미 국채 수익률 곡선"
                )
            else:
                st.caption("수익률 곡선 자료가 없습니다.")


def projections(d: Dashboard) -> None:
    p = d.projections
    with st.container(horizontal=True, vertical_alignment="center"):
        st.header(
            "FOMC 경제전망 · 위원 중앙값",
            anchor="sep",
            help="**정의** FOMC 경제전망(SEP) 위원 전망 중앙값 · 연도 말 기준 · 장기"
            " 전망 · 현재: 최근 실적\n\n**기준** Δ: 직전 SEP 대비 변화",
            width="content",
        )
        if p is None:
            notice(FRED_NOTICE)
    with st.container(border=True, height="stretch"):
        if p is None:
            st.caption("FOMC 경제전망 자료가 없습니다.")
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
                    "Δ", width="small", help="**정의** 직전 SEP 발표 대비 변화"
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


def detail_chart(points: list[dict[str, str | float]], dec: int) -> alt.Chart:
    return (
        alt.Chart(alt.Data(values=points))
        .mark_line(point=len(points) <= DETAIL_POINT_LIMIT)
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("day:T", title="날짜", format="%Y-%m-%d"),
                alt.Tooltip("value:Q", title="값", format=f",.{dec}f"),
            ],
        )
    )


def detail_body(row: IndicatorRow) -> None:
    source = row.code + (f" · © {row.copyright}" if row.copyright else "")
    st.caption(f"출처 {source}")
    with st.container(horizontal=True, key="detail_cards"):
        st.metric(
            "최신",
            display.fmt(row.value, row.fmt),
            display.metric_delta(row),
            delta_color=display.metric_color(row),
            delta_arrow="off" if display.change(row) == 0 else "auto",
            help=GUIDES.get(row.id),
            border=True,
            height="stretch",
        )
        st.metric("이전", display.fmt(row.prev, row.fmt), border=True, height="stretch")
        st.metric("기준", display.short_date(row) or "—", border=True, height="stretch")
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            unit = display.unit(row.fmt)
            headed(f"**추이** · {unit}" if unit else "**추이**")
            picked = st.segmented_control(
                "기간",
                list(display.CHART_RANGES),
                default="5Y",
                key="detail_range",
                label_visibility="collapsed",
            )
        series = rt.dashboard.full_series(row.id) or display.history_series(row)
        points = display.detail_points(series, display.CHART_RANGES[picked or "전체"])
        if len(points) > 1:
            st.altair_chart(
                detail_chart(points, row.fmt.dec),
                height=CHART_HEIGHT,
                alt=f"{row.label} 추이",
            )
        elif row.message:
            st.caption(f":red[{row.message}]")
        else:
            st.caption("추이 자료가 없습니다.")


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
            st.subheader(p.title, anchor=p.id, width="content")
            with st.popover(":material/help:", help="지표 해석", type="tertiary"):
                for row in p.rows:
                    st.markdown(f"**{row.label}**\n\n{GUIDES.get(row.id, row.code)}")
        event = st.dataframe(
            pd.DataFrame(display.panel_table(p)).style.map(
                arrow_style, subset=["변화"]
            ),
            hide_index=True,
            height="content",
            column_config={
                "지표": st.column_config.TextColumn(width="medium"),
                "추이": st.column_config.LineChartColumn(width="small", color="gray"),
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
                "핵심만 보기" if on else f"지표 {view.extra}개 더 보기",
                key=f"more_{p.id}",
                type="tertiary",
                icon=None if on else ":material/add:",
                on_click=toggle_deep,
                args=(p.id,),
            )
        for problem in display.panel_problems(p):
            st.caption(f":red[{problem}]")


def calendar(d: Dashboard) -> None:
    fred = any(s.id == "FRED" and s.state != "missing_key" for s in d.sources)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.header(
            "앞으로 7일 미국 발표",
            anchor="calendar",
            help="**정의** FRED 발표 일정 중 미국 주요 지표 발표 · 오늘부터 7일 ·"
            " 날짜는 발표 예정일\n\n**기준** 물가·고용·성장·소비·주택·연준·금융여건"
            " 발표만 추림",
            width="content",
        )
        if not d.calendar and not fred:
            notice(FRED_NOTICE)
    with st.container(border=True):
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
        elif fred:
            st.caption("앞으로 7일 안에 주요 발표가 없습니다.")
        else:
            st.caption("FRED API 키가 없어 발표 일정을 받지 못했습니다.")


def panels(d: Dashboard, ids: tuple[str, ...]) -> None:
    for card in d.panels:
        if card.id in ids:
            panel(d, card)


def provenance(d: Dashboard) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.header("출처", anchor="sources", width="content")
    with st.container(border=True):
        headed("**데이터 호출 점검**")
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
            ) and start_run(preset.text, preset.id):
                st.rerun()
        if (
            running
            and st.button("중지", icon=":material/stop:", key="ai_stop")
            and state.ai_run is not None
        ):
            log.stopped = True
            state.ai_run.stop()
        st.button("새 대화", on_click=new_conversation, disabled=running)
    request = st.chat_input(
        "무엇을 분석할까요?",
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
        st.subheader("AI 분석", anchor="analysis")
        # The pinned data time shows once: as a warning once newer data came.
        if log.stale and log.conversation:
            st.warning(
                f"이 대화는 {display.short_time(log.snapshot)} 데이터 기준입니다."
            )
        elif log.header:
            st.caption(log.header)
        if state.ai_error:
            st.error(state.ai_error)
        else:
            st.caption(choice_line(statuses, bool(log.conversation)))
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
def screen_updater(root: str, universe: str) -> screen.Updater:
    """One updater per universe's data folder, shared by every tab."""
    return screen.Updater(Path(root), source=partial(screen.Yahoo, universe=universe))


@st.cache_data(show_spinner="스크리닝 계산 중…", max_entries=2)
def screen_prepared(
    root: str, stamp: tuple[tuple[str, int], ...]
) -> pd.DataFrame | None:
    """The stored files with indicators; ``stamp`` changes whenever one does."""
    inputs = screen.load(Path(root)) if stamp else None
    return None if inputs is None else screen.prepare(inputs)


@st.cache_data(show_spinner="스크리닝 계산 중…", max_entries=8)
def screen_table(
    root: str, stamp: tuple[tuple[str, int], ...], criteria: screen.Criteria
) -> pd.DataFrame | None:
    """The table screened by ``criteria``; only a changed rule recomputes it."""
    prepared = screen_prepared(root, stamp)
    return None if prepared is None else screen.judge(prepared, criteria)


def crit_values(c: screen.Criteria) -> dict[str, bool]:
    """The 기준 widgets' values for ``c``: a switch per rule."""
    return {f"crit_on_{rule.key}": c.on(rule.key) for rule in screen.RULES}


def saved_criteria() -> screen.Criteria:
    """The criteria in the settings file: the ones the screening page uses."""
    return screen.criteria(load_screen(resolve_settings(args.settings)).off)


def seed_criteria() -> None:
    """Set the 스크리닝 기준 widgets to the saved criteria as 설정 opens."""
    for key, value in crit_values(saved_criteria()).items():
        state[key] = value


def crit_draft() -> screen.Criteria:
    """The criteria the 스크리닝 기준 widgets hold, applied by 적용."""
    return screen.criteria(
        r.key
        for r in screen.RULES
        if state[f"crit_on_{r.key}"] != screen.DEFAULT.on(r.key)
    )


def save_criteria() -> None:
    choice = ScreenChoice(tuple(screen.changed(crit_draft())))
    path = resolve_settings(args.settings)
    if load_screen(path) == choice:
        return
    try:
        save_screen(path, choice)
    except OSError as error:
        state.toast = f"스크리닝 기준을 저장하지 못했습니다: {error}"


def reset_criteria() -> None:
    for key, value in crit_values(screen.DEFAULT).items():
        state[key] = value


def crit_switch(rule: screen.Rule, *, changed: bool) -> None:
    with st.container(horizontal=True, vertical_alignment="center", width="content"):
        st.checkbox(
            rule.name,
            key=f"crit_on_{rule.key}",
            help=rule.help,
        )
        if changed:
            st.badge("변경", color="blue")


def criteria_panel() -> None:
    """The 스크리닝 기준 tab of 설정: every switch by section, with a reset."""
    changed = set(screen.changed(crit_draft()))
    with st.container(horizontal=True, vertical_alignment="center"):
        st.button(
            "기본값으로",
            icon=":material/restart_alt:",
            key="crit_reset",
            disabled=not changed,
            on_click=reset_criteria,
        )
        if changed:
            st.caption(f"기본값에서 {len(changed)}개 변경 · 적용을 눌러야 반영")
    sections: dict[str, list[screen.Rule]] = {}
    for rule in screen.RULES:
        sections.setdefault(rule.section, []).append(rule)
    # Sections in order down two columns, split where the heights come closest.
    lines = [1 + len(rules) for rules in sections.values()]
    split = min(
        range(len(lines) + 1),
        key=lambda k: abs(sum(lines[:k]) - sum(lines[k:])),
    )
    columns = st.columns(2)
    for n, (title, rules) in enumerate(sections.items()):
        with columns[n >= split], st.container(border=True):
            st.markdown(f"**{title}**")
            for rule in rules:
                crit_switch(rule, changed=rule.key in changed)


def parts_line(parts: screen.Parts) -> str:
    return (
        f"시세 {parts.prices or '—'} 종가 · 재무 {parts.fundamentals or '—'}"
        f" · 추정치 {parts.estimates or '—'} · 유니버스 {parts.universe or '—'}"
        f" · SEC {parts.margins or '—'}"
    )


def screen_bar(
    updater: screen.Updater, root: Path, help_: str, criteria: screen.Criteria
) -> None:
    """Data dates with the page's ``help_``, the 갱신 button and any progress.

    A finished update saves a snapshot screened by ``criteria``.
    """
    polling = updater.progress.running
    missing = screen.missing_packages()

    @st.fragment(run_every=1.0 if polling else None)
    def body() -> None:
        progress = updater.progress
        if polling and not progress.running:
            st.rerun()
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(parts_line(screen.parts(root)), help=help_)
            st.space("stretch")
            if st.button(
                "갱신",
                icon=":material/refresh:",
                type="primary",
                key="screen_refresh",
                disabled=progress.running or bool(missing) or args.demo,
            ):
                # Rerun the page even if another tab started first, so it polls.
                updater.start(sec_agent(), criteria)
                st.rerun()
        if progress.running:
            done = progress.done / progress.total if progress.total else 0.0
            count = f" {progress.done}/{progress.total}" if progress.total else ""
            st.progress(done, text=f"{progress.stage}{count}")
        elif progress.message:
            if progress.failed:
                st.warning(progress.message)
            else:
                st.success(progress.message)
        if missing:
            st.info(
                f"갱신하려면 추가 패키지({', '.join(missing)})가 필요합니다:"
                f" `{screen.INSTALL_HINT}`"
            )

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
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
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


def status_lines(lines: list[tuple[str, str, str]]) -> str:
    return "  \n".join(
        display.md(f"**{result or '—'}** {test} · :gray[{detail}]")
        for result, test, detail in lines
    )


def stock_body(root: Path, row: dict[str, object], c: screen.Criteria) -> None:
    fmt = screen.fmt
    st.caption(
        f"{row['sector']} · {row['industry']} · 재무 통화 {row['currency'] or '—'}"
    )
    with st.container(horizontal=True, key="screen_stock_cards"):
        st.metric("주가", fmt("usd", row["price"]), border=True, height="stretch")
        st.metric("시가총액", fmt("big", row["mcap"]), border=True, height="stretch")
        st.metric(
            "퀄리티",
            fmt("z", row.get("quality")),
            help=screen.quality_help(c),
            border=True,
            height="stretch",
        )
        st.metric(
            "밸류에이션",
            fmt("z", row.get("value")),
            help=screen.value_help(c),
            border=True,
            height="stretch",
        )
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed("**가격 지표**", "퀄리티·밸류에이션 계산에 쓰지 않음")
        st.markdown(status_lines(screen.tech_lines(row)))
        headed("**영업이익률**", "퀄리티·밸류에이션 계산에 쓰지 않음")
        st.caption(display.md(screen.margin_line(row)))
    with right, st.container(border=True, height="stretch"):
        universe = screen.UNIVERSES[state.get("screen_universe") or "sp500"]
        headed(
            "**퀄리티·밸류에이션**",
            f"{screen.basis_note()} · 유니버스 {universe.name} 구성 종목",
        )
        st.markdown(status_lines(screen.measure_lines(row, c)))
        headed(
            f"**EPS 추정치** :gray[{row['est_currency'] or '—'}]",
            screen.ESTIMATE_HELP,
        )
        st.caption(display.md(screen.estimate_note(row)))
        st.dataframe(screen.estimate_table(row), hide_index=True, alt="EPS 추정치")
    points = screen.chart_rows(screen.history(root, str(row["ticker"])))
    if len(points) > 1:
        with st.container(border=True):
            headed("**1년 주가** · 종가·이동평균, USD")
            st.altair_chart(stock_chart(points), alt=f"{row['ticker']} 1년 주가")
    if row["error"]:
        st.caption(f":red[{display.md(str(row['error']))}]")


def open_stock(root: Path, row: dict[str, object], c: screen.Criteria) -> None:
    """Show the stock dialog; a fresh table key clears the clicked cell.

    Same pattern as ``open_detail``: clicking that cell again reopens the
    dialog, and unrelated reruns do not.
    """
    state["screen_gen"] = state.get("screen_gen", 0) + 1
    title = f"{row['ticker']} · {row['name']}"
    st.dialog(title, width="large", on_dismiss="rerun")(stock_body)(root, row, c)


JITTER = 0.02  # in Z-score units: where points sit, not their values


def map_points(points: list[dict[str, object]]) -> list[dict[str, object]]:
    """The map's points, a fixed hash-driven nudge off their values.

    Overlapping points part without changing them.
    """
    shown: list[dict[str, object]] = []
    for point in points:
        seed = sum(ord(ch) for ch in str(point["ticker"]))
        nudge = ((seed % 7) - 3) / 3 * JITTER, ((seed % 5) - 2) / 2 * JITTER
        shown.append(
            point
            | {
                "x": float(cast("float", point["value"])) + nudge[0],
                "y": float(cast("float", point["quality"])) + nudge[1],
            }
        )
    return shown


def map_chart(points: list[dict[str, object]], *, guides: bool) -> alt.LayerChart:
    """밸류에이션 against 퀄리티, both Z-scores; the chart takes no clicks."""
    limit = [-screen.Z_LIMIT, screen.Z_LIMIT]
    dots = (
        alt.Chart(alt.Data(values=map_points(points)))
        .mark_circle(size=40, opacity=0.6)
        .encode(
            x=alt.X("x:Q", title="밸류에이션", scale=alt.Scale(domain=limit)),
            y=alt.Y(
                "y:Q",
                title="퀄리티",
                scale=alt.Scale(domain=limit),
                # Level, above the axis at its top left, not turned on its side.
                axis=alt.Axis(
                    titleAngle=0,
                    titleAlign="left",
                    titleAnchor="end",
                    titleBaseline="bottom",
                    titleX=-30,
                    titleY=-8,
                ),
            ),
            tooltip=[
                alt.Tooltip("ticker:N", title="티커"),
                alt.Tooltip("name:N", title="종목"),
                alt.Tooltip("sector:N", title="섹터"),
                alt.Tooltip("quality:Q", title="퀄리티", format="+.2f"),
                alt.Tooltip("value:Q", title="밸류에이션", format="+.2f"),
                alt.Tooltip("counted:N", title="산출 항목 수"),
            ],
        )
    )
    zero = alt.Chart(alt.Data(values=[{"v": 0.0}])).mark_rule(
        strokeDash=[2, 3], color="gray"
    )
    layers: list[alt.Chart] = [
        *([zero.encode(x="v:Q"), zero.encode(y="v:Q")] if guides else []),
        dots,
    ]
    return alt.LayerChart(layer=layers).properties(height=2 * CHART_HEIGHT)


def value_map(rows: pd.DataFrame, c: screen.Criteria) -> None:
    """The stocks as 퀄리티 by 밸류에이션 points with the guide toggle, boxed."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            headed(
                "**퀄리티·밸류에이션 분포** · z 점수",
                f"{screen.quality_help(c)}\n\n{screen.value_help(c)}",
            )
            st.space("stretch")
            guides = st.toggle("평균선", value=True, key="map_guides")
        st.altair_chart(
            map_chart(screen.map_rows(rows), guides=guides),
            alt="퀄리티와 밸류에이션 분포",
        )


def screen_results(root: Path, table: pd.DataFrame, c: screen.Criteria) -> None:
    liquid = screen.liquid(table)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.space("stretch")
        sectors = st.multiselect(
            "섹터",
            screen.sectors(liquid),
            key="screen_sectors",
            placeholder="전체 섹터",
            width=320,
        )
    rows = screen.pick(table, sectors)
    value_map(rows, c)
    shown = screen.view(rows)
    columns = [c for c, _, _ in screen.TABLE if c in shown.columns]
    formats = {c: partial(screen.fmt, kind) for c, _, kind in screen.TABLE if kind}
    signed = [c for c, _, kind in screen.TABLE if kind == "signed"]
    styled = shown.style.map(return_style, subset=signed).format(
        {c: formats[c] for c in columns if c in formats}
    )
    labels = {column: label for column, label, _ in screen.TABLE}
    tips = screen.column_help(c)
    event = st.dataframe(
        styled,
        hide_index=True,
        height=LIST_HEIGHT,
        placeholder="—",  # NaN cells; the grid ignores the Styler's dash for them
        on_select="rerun",
        selection_mode="single-cell",
        key=f"screen_table_{state.get('screen_gen', 0)}",
        column_config={
            column: st.column_config.Column(
                labels[column],
                pinned=column == "ticker",
                width="medium" if column == "name" else None,
                help=tips.get(column),
            )
            for column in columns
        },
        alt="스크리닝 결과",
    )
    failed = screen.failures(table)
    if len(failed):
        with st.expander(f"제외되었거나 받지 못한 값이 있는 종목 {len(failed)}"):
            st.dataframe(failed, hide_index=True, alt="제외된 종목")
    records = screen.records(rows)
    for index, column in event.selection.cells:
        if column in {"ticker", "name"}:
            open_stock(root, records[index], c)


# ---------------------------------------------------------------- portfolio

PF_SECTIONS = {
    "returns": "수익률",
    "analysis": "위험지표",
    "correlation": "상관·분산",
    "tail": "꼬리위험",
    "outlook": "전망",
    "stress": "스트레스",
}
PF_WINDOW_HELP = (
    "**정의** 지표·분포 추정에 쓰는 최근 일간 수익률 기간 · "
    + " · ".join(f"{k} {v}개월" for k, v in portfolio.WINDOWS.items())
    + "\n\n**기준** 전망·편입 효과: 재표본 대상 과거 수익률 · 시뮬레이션 길이:"
    " 전망 기간"
)
PF_HORIZON_HELP = (
    "**정의** 시뮬레이션 경로 길이 · "
    + " · ".join(f"{k} {v}거래일" for k, v in portfolio.HORIZONS.items())
    + "\n\n**기준** 지금 평가액에서 출발 · 재표본 대상 과거 기간: 추정 기간"
)
PF_MIN_QUANTITY = 0.0001
PF_SIDE_WIDTH = 72  # 매수 and 매도 the same width
PF_CURRENCY_HELP = "**정의** 종목의 거래 통화 기준 금액 · AAPL은 USD · 005930.KS는 KRW"
PF_VALUE_HELP = (
    "**정의** Σ 보유 수량 × 최근 종가 · 다른 통화 종목은 최근 USD/KRW로 기준 통화 환산"
)
PF_UNREALIZED_HELP = (
    "**정의** 평가액 − 보유분 원가 · 원가: 이동평균, 매수 수수료 포함, 매수일"
    " 환율\n\n**해석** 아래 비율: 보유분 원가 대비 수익률"
)
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


def pf_periods(horizon: str | None = None) -> int:
    """추정 기간, and 전망 기간 under the key ``horizon`` on a tab that simulates.

    Drawn in the caller's row; returns the window in months. The simulation
    reads its horizon from ``state``.
    """
    window = st.segmented_control(
        "추정 기간",
        list(portfolio.WINDOWS),
        default="3Y",
        required=True,
        key="pf_window",
        help=PF_WINDOW_HELP,
    )
    if horizon is not None:
        st.segmented_control(
            "전망 기간",
            list(portfolio.HORIZONS),
            default="1M",
            required=True,
            key=horizon,
            help=PF_HORIZON_HELP,
        )
    return portfolio.WINDOWS[window or "3Y"]


# ---- caches: in memory only, keyed by everything the inputs depend on
# The leading-underscore arguments are not hashed; the key stands for them:
# the saved prices' stamp, the portfolio or book JSON (trades and base), the
# window and other choices, and for investors the 13F files' stamp.

type Key = tuple[object, ...]


def pf_files_stamp(root: Path) -> tuple[tuple[str, int, int], ...]:
    out: list[tuple[str, int, int]] = []
    for name in (portfolio.CLOSE_FILE, portfolio.ADJUSTED_FILE, *portfolio.RANGE_FILES):
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
    key: Key, _p: portfolio.Portfolio, _close: pd.DataFrame, _adjusted: pd.DataFrame
) -> pd.Series:
    del key
    return portfolio.held_returns(_p, _close, _adjusted)


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
def cached_draws(
    stamp: object, n: int, horizon: int, block: float
) -> NDArray[np.int64]:
    """Stationary-bootstrap day indices, drawn once per window length and horizon.

    Any weight vector reuses them; read-only, so the cached array is shared.
    """
    del stamp
    if args.demo:  # the canned paths read only the horizon
        return np.zeros((1, horizon), dtype=np.int64)
    picks = quant.stationary_indices(n, portfolio.PATHS, horizon, block)
    picks.flags.writeable = False
    return picks


@st.cache_data(show_spinner=False, max_entries=16)
def cached_filter(
    key: Key, _j: portfolio.Joint, _v: portfolio.Vol | None = None
) -> quant.Devolatilised:
    """``key`` stands for ``_j`` and the filter's options ``_v``."""
    del key
    if args.demo:
        days = min(len(_j.full), quant.FILTER_DAYS)
        return demo_results.volatility(len(_j.names), days)
    return portfolio.volatility(_j, _v)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_paths(
    key: Key,
    fhs: bool,
    value: float,
    standing: tuple[float, float],
    _j: portfolio.Joint,
    _picks: NDArray[np.int64],
    _w: NDArray[np.float64] | None = None,
    _f: quant.Devolatilised | None = None,
    drift: float = 0.0,
) -> portfolio.Simulation:
    """The summary, fan and histogram; the paths themselves are not kept.

    ``_w`` replaces ``_j``'s weights on the same draws; ``key`` stands for it.
    ``_f`` is the volatility filter when ``fhs``; ``drift`` the daily
    expected return (``portfolio.simulation``).
    """
    del key
    if args.demo:
        return demo_results.simulation(value, _picks.shape[1], drift)
    growth = portfolio.growth(_j, _picks, _f if fhs else None, _w)
    return portfolio.simulation(growth, value, *standing, drift)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_tails(
    key: Key,
    _j: portfolio.Joint,
    _f: quant.Devolatilised,
    _v: portfolio.Vol | None = None,
) -> portfolio.Tails:
    del key
    if args.demo:
        return demo_results.tails(len(_j.names), len(_j.window))
    return portfolio.tails(_j, _f, _v)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_density(
    key: Key,
    _j: portfolio.Joint,
    _f: quant.Devolatilised | None,
    _drift: NDArray[np.float64] | None = None,
) -> portfolio.Density:
    """``key`` stands for ``_j``, whether ``_f`` is given and which, and ``_drift``."""
    del key
    if args.demo:
        return demo_results.density()
    return portfolio.density(_j, _f, drift=_drift)


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
        rets=cached_held(pf_key(pid), p, prices.close, prices.adjusted),
        w=w,
        value=sum(h.value for h in items if h.ticker in w.index),
        bench=partial(portfolio.blended, p, prices.close),
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


def pf_rates(
    base: portfolio.Currency, close: pd.DataFrame
) -> tuple[pd.Series, str, perf.Quote]:
    """Annual percent risk-free rates for ``base``, their name and their quote.

    Empty when there are none.
    """
    if base == "KRW":
        return pf_dashboard_series(PF_CD91), "CD 91일", "simple"
    if portfolio.RF_USD in close.columns:
        rates = cast("pd.Series", close[portfolio.RF_USD]).dropna()
        return rates, "미국 3개월 국채(^IRX)", "discount"
    return pf_empty(), "^IRX", "discount"


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
                type="secondary",
                on_click=pf_open,
                args=("delete", chosen),
            )
    return chosen


# ---- prices


def pf_bar(  # noqa: PLR0913 - the bar's inputs
    fetcher: portfolio.Fetcher,
    root: Path,
    book: portfolio.Book,
    missing: list[str],
    extra: Sequence[str] = (),
    have: Collection[str] = (),
) -> None:
    """When prices were saved, the 갱신 button and a running fetch.

    ``extra``: tickers outside the book that 갱신 also fetches;
    ``have``: the saved tickers (``portfolio.wanted``).
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
                f"가격 {saved:%Y-%m-%d %H:%M} 저장"
                if saved
                else "저장된 가격이 없습니다."
            )
            st.space("stretch")
            if st.button(
                "갱신",
                icon=":material/refresh:",
                type="primary",
                key="pf_refresh",
                disabled=progress.running or bool(missing) or args.demo or not tickers,
            ):
                fetcher.start(
                    list(dict.fromkeys([*portfolio.wanted(book, have), *extra]))
                )
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

    body()


def pf_no_prices() -> None:
    st.info("가격이 없습니다.")


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
        state.pf_trades_gen = state.get("pf_trades_gen", 0) + 1
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
    state.pf_price_note = f"{day:%m/%d} 종가 기준"


def pf_trade_form(p: portfolio.Portfolio) -> None:
    today = datetime.now().astimezone().date()
    state.setdefault("pf_quantity", 1.0)
    state.setdefault("pf_fee", 0.0)
    if state.get("pf_prefill"):
        state.pf_prefill = False
        pf_prefill()
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
    key = f"pf_trades_{state.get('pf_trades_gen', 0)}"
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"{len(ids)}건")
        st.space("stretch")
        st.button(
            "선택 삭제",
            icon=":material/delete:",
            key="pf_remove",
            type="secondary",
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


def pf_trade_open() -> None:
    state.pf_trade_dialog = True


def pf_trade_close() -> None:
    state.pf_trade_dialog = False


def pf_trade_body(p: portfolio.Portfolio) -> None:
    if state.toast:
        # A trade was added: only the dialog reran, so rerun the page behind it.
        st.rerun()
    pf_trade_form(p)


PF_DONUT_SLICES = 8  # more names than this: the largest 7 and 기타


def pf_donut(
    rows: list[tuple[str, float, float]],
    text: Callable[[float], str],
    amount: str = "평가액",
) -> alt.Chart:
    """Weights as a donut, from (name, weight, ``amount``) rows; ``text`` shows it."""
    merged: dict[str, tuple[float, float]] = {}
    for name, weight, value in rows:
        w, v = merged.get(name, (0.0, 0.0))
        merged[name] = (w + weight, v + value)
    ranked = sorted(merged.items(), key=lambda item: item[1][0], reverse=True)
    if len(ranked) > PF_DONUT_SLICES:
        rest = ranked[PF_DONUT_SLICES - 1 :]
        other = (sum(w for _, (w, _) in rest), sum(v for _, (_, v) in rest))
        ranked = [*ranked[: PF_DONUT_SLICES - 1], ("기타", other)]
    points = [
        {"name": name, "weight": w, "value": text(v), "rank": i}
        for i, (name, (w, v)) in enumerate(ranked)
    ]
    return (
        alt.Chart(alt.Data(values=points))
        .mark_arc(innerRadius=50, padAngle=0.01)
        .encode(
            theta=alt.Theta("weight:Q"),
            order=alt.Order("rank:Q"),
            color=alt.Color(
                "name:N",
                sort=[name for name, _ in ranked],
                title=None,
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("name:N", title="종목"),
                alt.Tooltip("weight:Q", title="비중", format=".1%"),
                alt.Tooltip("value:N", title=amount),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )


def pf_holdings(p: portfolio.Portfolio, prices: portfolio.Prices) -> None:
    """현황: always shown above the analysis sections."""
    items = portfolio.holdings(p, prices.close)
    held = [h for h in items if h.is_open]
    with st.container(horizontal=True, vertical_alignment="center"):
        headed(
            "**현황**",
            f"**정의** 보유 종목의 평가액·평가손익·비중과 거래 내역 · {p.base}"
            " 환산 · 원가: 이동평균",
        )
        st.space("stretch")
        st.button(
            "거래 추가",
            icon=":material/add:",
            key="pf_trade_open",
            on_click=pf_trade_open,
        )
    if not p.trades:
        st.info("거래를 추가하세요.")
    if held:
        totals = portfolio.totals(items)
        with st.container(horizontal=True, key="pf_now_cards"):
            st.metric(
                "평가액",
                portfolio.money(totals.value, p.base),
                help=PF_VALUE_HELP,
                border=True,
                height="stretch",
            )
            st.metric(
                "평가손익",
                portfolio.money(totals.unrealized, p.base, signed=True),
                portfolio.percent(totals.open_return, signed=True),
                delta_color=pf_color(totals.unrealized),
                delta_arrow="off",
                help=PF_UNREALIZED_HELP,
                border=True,
                height="stretch",
            )
    if prices.tickers & {t.ticker for t in p.trades}:
        left, right = st.columns([1, 2])
        with left, st.container(border=True, height="stretch"):
            headed(
                "**비중**",
                f"**정의** 보유 종목 평가액 비중 · {p.base} 환산\n\n**기준**"
                f" {PF_DONUT_SLICES}종목 초과 시 상위 {PF_DONUT_SLICES - 1}종목과"
                " 기타",
            )
            w = portfolio.weights(items)
            if w.empty:
                st.caption("보유 중인 종목이 없습니다.")
            else:
                rows = [
                    (h.ticker, cast("float", w[h.ticker]), h.value)
                    for h in held
                    if h.ticker in w
                ]
                st.altair_chart(
                    pf_donut(rows, partial(portfolio.money, unit=p.base)),
                    alt="비중",
                )
        with right, st.container(border=True, height="stretch"):
            headed(
                f"**평가액과 순투입** · {p.base}",
                "**정의** 평가액: 보유 수량 × 종가 · 순투입: 매수 금액(수수료"
                " 포함) − 매도 금액(수수료 차감)의 누적\n\n**해석** 평가액 −"
                " 순투입 = 총손익",
            )
            frame = cached_history(pf_key(p.model_dump_json()), p, prices.close)
            lines = portfolio.value_lines(frame)
            if len(frame) > 1:
                st.altair_chart(
                    pf_line_chart(portfolio.line_points(lines), pf_axis(p.base)),
                    alt="평가액과 순투입",
                )
            else:
                st.caption(PF_NO_LINE)
    if held:
        with st.container(border=True):
            headed(f"**보유 종목** · {p.base}")
            pf_table(portfolio.holdings_table(items, p.base), "보유 종목")
            unknown = portfolio.unpriced(items)
            if unknown:
                st.caption(
                    f"평가액 미확인: {', '.join(unknown)}",
                    help="**기준** 종가 또는 환율 없음",
                )
    if p.trades:
        with st.expander("거래 내역"):
            pf_trade_log(p)
    if state.get("pf_trade_dialog"):
        st.dialog("거래 추가", width="large", on_dismiss=pf_trade_close)(pf_trade_body)(
            p
        )


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
            x=alt.X("pnl:Q", title=None, axis=alt.Axis(format=pf_axis(base))),
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
PF_NO_LINE = "이틀 이상의 가격 기록이 없습니다."  # a line chart's empty state


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
    with st.container(horizontal=True, key="pf_split_cards"):
        for label, value, tip in (
            (
                "현지 수익률",
                split.local,
                "**정의** 보유분의 현지 통화 기준 연결 수익률",
            ),
            (
                "환율",
                split.fx,
                "**정의** 보유 통화의 기준 통화 대비 가치 변화의 연결 기여",
            ),
            ("교차항", split.cross, "**정의** 현지 수익률 × 환율"),
            (
                "합계",
                split.base,
                "**정의** (1 + 현지)(1 + 환율) − 1 · 기준 통화 연결 수익률",
            ),
        ):
            st.metric(
                label,
                portfolio.percent(value, signed=True),
                help=tip,
                border=True,
                height="stretch",
            )


def pf_trading(
    money: perf.Annualised | None,
    time: perf.Annualised | None,
    result: portfolio.Trading | None,
    base: portfolio.Currency,
) -> None:
    timing = perf.timing_gap(money, time) if money and time else None
    with st.container(horizontal=True, key="pf_trade_cards"):
        st.metric(
            "타이밍 차이",
            pf_gap(timing),
            help="**정의** MWR − TWR\n\n**해석** 양수면 넣고 뺀 시점이 수익률에 유리 ·"
            " 음수면 불리\n\n**기준** 1년 이상 연율 차 · 1년 미만 기간 수익률 차",
            border=True,
            height="stretch",
        )
        if result is not None:
            st.metric(
                "무매매 평가액",
                portfolio.money(result.held, base),
                portfolio.money(result.actual - result.held, base, signed=True),
                delta_color=pf_color(result.actual - result.held),
                delta_arrow="off",
                help="**정의** 첫 거래일 보유 수량 유지 · 이후 순유입·순유출만큼 같은"
                " 구성을 비례 매수·매도한 경우의 평가액\n\n**해석** 아래 값: 실제"
                " 평가액 − 무매매 평가액 · 양수면 실제 매매가 유리",
                border=True,
                height="stretch",
            )
            st.metric(
                "매매 효과",
                pf_gap(result.effect),
                help="**정의** 실제 MWR − 무매매 MWR\n\n**해석** 양수면 종목 교체·비중"
                " 조절 매매의 기여 · 음수면 손실\n\n**기준** 1년 이상 연율 차 · 1년"
                " 미만 기간 수익률 차",
                border=True,
                height="stretch",
            )


def pf_return_metrics(
    totals: portfolio.Totals,
    time: perf.Annualised | None,
    money: perf.Annualised | None,
    base: portfolio.Currency,
) -> None:
    with st.container(horizontal=True, key="pf_return_cards"):
        cells = (
            (
                "평가손익",
                totals.unrealized,
                totals.open_return,
                PF_UNREALIZED_HELP,
            ),
            (
                "실현손익",
                totals.realized,
                totals.realized_return,
                "**정의** Σ(매도 순대금 − 매도 주식 원가) · 순대금: 수수료 차감 ·"
                " 원가: 이동평균\n\n**해석** 아래 비율: 매도 주식 원가 대비 수익률",
            ),
            (
                "총손익",
                totals.pnl,
                totals.total_return,
                "**정의** 실현손익 + 평가손익\n\n**해석** 아래 비율: 누적"
                " 매수금액(수수료 포함) 대비 수익률",
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
                height="stretch",
            )
        st.metric(
            "TWR 누적",
            portfolio.percent(time.cumulative, signed=True) if time else PF_DASH,
            pf_annual(time) if time and time.annual is not None else None,
            delta_color="off",
            delta_arrow="off",
            help="**정의** 시간가중수익률 · 일별 (평가액 − 전일 평가액 − 순유입) ÷"
            " (전일 평가액 + 당일 순유입)의 연결 · 순유출은 장 마감 시점 반영\n\n"
            "**해석** 넣고 뺀 금액의 영향을 뺀 운용 수익률 · 아래 값: 1년 이상이면"
            " 연환산\n\n**기준** 배당 제외 종가 · 기준 통화",
            border=True,
            height="stretch",
        )
        st.metric(
            "MWR",
            pf_annual(money),
            help="**정의** 금액가중수익률 · 매수·매도 현금흐름과 현재 평가액의"
            " 내부수익률(XIRR, 실제 일수/365)\n\n**해석** 넣고 뺀 시점까지 반영한"
            " 투자자 수익률\n\n**기준** 거래 기간 1년 미만이면 연환산 없는 기간"
            " 수익률",
            border=True,
            height="stretch",
        )


def pf_realized(
    p: portfolio.Portfolio, prices: portfolio.Prices, items: list[portfolio.Holding]
) -> None:
    base = p.base
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(
            f"**종목별 손익** · 실현 + 평가, {base}",
            "**정의** 종목별 실현손익 + 평가손익 · 전량 매도 종목 포함",
        )
        points = portfolio.contribution_points(items)
        if points:
            st.altair_chart(pf_contribution_chart(points, base), alt="종목별 손익")
        else:
            st.caption("손익을 계산할 수 있는 종목이 없습니다.")
    with right, st.container(border=True, height="stretch"):
        headed(
            f"**연도별 실현손익** · {base}",
            "**정의** 매도일 기준 연도별 실현손익 합계 · 월별은 매도 월 기준",
        )
        sales = portfolio.ledger(p, prices.close).sales
        pf_period_table(sales, base, "연도", monthly=False)
        with st.expander("월별"):
            pf_period_table(sales, base, "월", monthly=True)
    sales = portfolio.ledger(p, prices.close).sales
    if sales:
        with st.container(border=True):
            headed(
                f"**실현손익 내역** · 매도 {len(sales)}건",
                "**정의** 매도 건별 실현손익 · 평균단가: 이동평균 원가(매수 수수료"
                " 포함) · 수익률: 매도 주식 원가 대비 · 보유일: 해당 보유분 첫"
                " 매수일부터 매도일까지 달력일 · 구분: 익절·손절·본전",
            )
            pf_table(portfolio.realized_table(sales, base), "실현손익 내역")
    closed = portfolio.closed_table(items, base)
    if closed["종목"]:
        with st.container(border=True):
            headed(
                "**전량 매도한 종목**",
                "**정의** 보유 수량 0인 종목의 누적 실현손익 · 원가 대비: 실현손익 ÷"
                " 매도 주식 원가",
            )
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
    with left, st.container(border=True, height="stretch"):
        headed(
            f"**누적 수익률** · TWR와 지수, {base}",
            "**정의** 포트폴리오 TWR과 지수의 누적 수익률 · 기준 통화 환산 · 표:"
            " 누적 · 연환산(1년 이상) · TWR − 지수\n\n**기준** "
            + PF_INDEX_NOTE
            + "(Yahoo 무료 총수익지수 없음) · 혼합 벤치마크: 전일 종가 기준 한국·미국"
            " 비중으로 두 지수의 일별 혼합",
        )
        if len(curve) > 1:
            shown = {"포트폴리오": curve, **lines}
            st.altair_chart(
                pf_line_chart(portfolio.line_points(shown), "+.1%"), alt="누적 수익률"
            )
            pf_benchmark_table(time, lines)
        else:
            st.caption(PF_NO_LINE)
    with right, st.container(border=True, height="stretch"):
        headed(
            f"**누적 손익** · {base}",
            "**정의** 총손익 = 평가액 − 순투입 · 실현손익: 매도 손익 누적 ·"
            " 평가손익 = 총손익 − 실현손익",
        )
        frame = cached_history(pf_key(pid), p, prices.close)
        if len(frame) > 1:
            points = portfolio.line_points(portfolio.pnl_lines(frame))
            st.altair_chart(pf_line_chart(points, pf_axis(base)), alt="누적 손익")
        else:
            st.caption(PF_NO_LINE)
    split, result = cached_flows(pf_key(pid, "flows"), p, prices.close)
    if split is not None:
        headed(
            "**환율 효과**",
            "**정의** 보유분 연결 수익률의 현지·환율 분해 · (1 + 합계) = (1 +"
            " 현지)(1 + 환율) · 교차항 = 현지 × 환율\n\n**해석** 환율 양수: 보유 통화"
            " 강세의 기여 · 음수: 약세의 손실\n\n**기준** 전일 종가 비중 · 배당 제외"
            " 종가",
        )
        pf_split(split)
    headed(
        "**매매 시점**",
        "**정의** 넣고 뺀 시점과 매매가 수익률에 준 효과 · 타이밍 차이 · 무매매"
        " 평가액 · 매매 효과",
    )
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
        help="**정의** 기간 중 고점 대비 가장 깊은 하락 · 1에서 시작한 누적 자산"
        " 기준\n\n**해석** 0에 가까울수록 얕은 손실 구간 · 아래 날짜: 고점 → 저점"
        + (
            f"\n\n**기준** 회복 {under.recovery:%Y-%m-%d}(저점부터"
            f" {under.recovery_days}일)"
            if under and under.recovery is not None and under.max_drawdown < 0
            else "\n\n**기준** 미회복"
            if under and under.max_drawdown < 0
            else ""
        ),
        border=True,
        height="stretch",
    )
    st.metric(
        "고점 아래 기간",
        f"{under.current_days:,}일" if under else PF_DASH,
        help=(
            "**정의** 마지막 고점 이후 달력일 · 지금 고점이면 0\n\n**기준** 기간 중"
            f" 최장 수중 기간(고점 → 회복 또는 오늘) {under.longest_days:,}일"
            if under
            else None
        ),
        border=True,
        height="stretch",
    )


def pf_sharpe(stats: portfolio.Ratios) -> None:
    ci = stats.ci
    st.metric(
        "샤프 지수",
        pf_ratio_or(stats.sharpe),
        f"95% {pf_ratio(ci.low)} ~ {pf_ratio(ci.high)}" if ci else PF_SHORT,
        delta_color="off",
        delta_arrow="off",
        help="**정의** 일간 초과수익률 평균 ÷ 표준편차(ddof 1) × √(연간 관측일"
        " 수)\n\n**해석** 높을수록 위험 대비 초과수익 우위 · 연 1 이상 양호 수준으로"
        " 통용 · 아래 값: 95% 신뢰구간 · 구간 전체가 0 위면 양의 성과 유의\n\n"
        f"**기준** 신뢰구간: Mertens(2002) 표준오차(왜도·첨도 반영), {perf.MIN_OBS}일"
        " 이상 · " + PF_YEAR_NOTE,
        border=True,
        height="stretch",
    )
    enough = stats.days >= perf.MIN_OBS
    st.metric(
        "PSR",
        portfolio.percent(stats.psr) if stats.psr is not None else PF_DASH,
        None if stats.psr is not None else PF_SHORT,
        delta_color="off",
        delta_arrow="off",
        help="**정의** 참 샤프 지수 > 0 확률(Bailey–López de Prado 2012) · Φ(SR ×"
        " √(n − 1) ÷ √(1 − 왜도 × SR + (첨도 − 1)/4 × SR²)) · SR: 일간 샤프 지수\n\n"
        "**해석** 높을수록 양의 샤프 지수 확신 · 95% 이상 통상 유의 수준\n\n"
        f"**기준** {perf.MIN_OBS}일 이상",
        border=True,
        height="stretch",
    )
    trl = stats.min_trl
    st.metric(
        "MinTRL",
        f"{trl:,.0f}" if enough and trl else PF_DASH,
        None if enough else PF_SHORT,
        delta_color="off",
        delta_arrow="off",
        help="**정의** PSR 95% 도달에 필요한 최소 기록 길이(거래일) · 1 + (1 − 왜도 ×"
        " SR + (첨도 − 1)/4 × SR²) × (z₀.₉₅ ÷ SR)² · SR: 일간 샤프 지수\n\n**해석**"
        " 실제 일수보다 짧으면 기록 길이 충분 · 길면 부족\n\n**기준**"
        f" {perf.MIN_OBS}일 이상 · 샤프 지수 0 이하면 —",
        border=True,
        height="stretch",
    )


def pf_advanced(stats: portfolio.Ratios, bench: str) -> None:
    with st.container(horizontal=True, key="pf_more_cards"):
        st.metric(
            "소르티노",
            pf_ratio_or(stats.sortino),
            help="**정의** 일간 초과수익률 평균 ÷ 하방편차 × √(연간 관측일 수) ·"
            " 하방편차: √(전체 일수의 min(초과수익률, 0)² 평균)\n\n**해석** 높을수록"
            " 하락 위험 대비 초과수익 우위 · 샤프보다 크면 상승 쪽 변동 우세\n\n"
            "**기준** 목표수익률: 무위험수익률 · " + PF_YEAR_NOTE,
            border=True,
            height="stretch",
        )
        st.metric(
            "칼마",
            pf_ratio_or(stats.calmar),
            None if stats.calmar is not None else PF_SHORT,
            delta_color="off",
            delta_arrow="off",
            help="**정의** 연복리 수익률(CAGR) ÷ |기간 최대 낙폭|\n\n**해석** 높을수록"
            " 최대 낙폭 대비 수익 우위\n\n**기준** 기간 1년(365일) 이상 · 무위험수익률"
            " 미차감",
            border=True,
            height="stretch",
        )
        st.metric(
            "얼서 지수",
            pf_ratio(stats.ulcer) if math.isfinite(stats.ulcer) else PF_DASH,
            help="**정의** 일별 낙폭(%)의 제곱평균제곱근 · 전체 일수 기준(Martin)\n\n"
            "**해석** 낮을수록 얕고 짧은 낙폭 · 0이면 낙폭 없음",
            border=True,
            height="stretch",
        )
        st.metric(
            "UPI",
            pf_ratio_or(stats.upi),
            None if stats.upi is not None else PF_SHORT,
            delta_color="off",
            delta_arrow="off",
            help="**정의** (연복리 수익률 − 연 무위험수익률) × 100 ÷ 얼서 지수(Martin"
            " 비율)\n\n**해석** 높을수록 낙폭 부담 대비 초과수익 우위\n\n**기준** 기간"
            " 1년(365일) 이상 · 연 무위험수익률: 일간 평균의 연복리",
            border=True,
            height="stretch",
        )
        st.metric(
            "오메가",
            pf_ratio_or(stats.omega),
            help="**정의** Σ 무위험수익률 초과분 ÷ Σ 무위험수익률 미달분 · 일간"
            " 기준(Keating–Shadwick 2002)\n\n**해석** 1 초과면 초과분 합 우위 · 1"
            " 미만이면 미달분 합 우위",
            border=True,
            height="stretch",
        )
        st.metric(
            "CDaR 95%",
            portfolio.percent(stats.cdar),
            help="**정의** 일별 낙폭 중 가장 깊은"
            f" {1 - perf.CDAR_LEVEL:.0%}의 평균(조건부 낙폭 위험, 경계값 비례 반영)\n\n"
            "**해석** 0에 가까울수록 꼬리 낙폭 완화 · 최대 낙폭과 평균 낙폭 사이 값",
            border=True,
            height="stretch",
        )
        st.metric(
            "추적오차",
            portfolio.percent(stats.te) if stats.te is not None else PF_DASH,
            None if stats.te is not None else PF_SHORT,
            delta_color="off",
            delta_arrow="off",
            help="**정의** 일간 (수익률 − 벤치마크 수익률) 표준편차 × √(연간 관측일"
            " 수)\n\n**해석** 낮을수록 벤치마크와 비슷한 움직임\n\n**기준**"
            f" {bench} · {PF_INDEX_NOTE} · {perf.MIN_OBS}일 이상",
            border=True,
            height="stretch",
        )
        st.metric(
            "정보비율",
            pf_ratio_or(stats.ir),
            None if stats.ir is not None else PF_SHORT,
            delta_color="off",
            delta_arrow="off",
            help="**정의** 일간 (수익률 − 벤치마크 수익률) 평균 × 연간 관측일 수 ÷"
            " 추적오차\n\n**해석** 높을수록 꾸준한 초과수익 · 0.5 이상 양호, 1 이상"
            " 우수 수준으로 통용(Grinold–Kahn)\n\n**기준**"
            f" {bench} · {PF_INDEX_NOTE} · {perf.MIN_OBS}일 이상",
            border=True,
            height="stretch",
        )


def pf_stats(s: Subject, prices: portfolio.Prices, months: int) -> None:
    """Risk-adjusted statistics of ``s.rets`` over the window."""
    rets = portfolio.window(s.rets, months)
    rates, source, quote = pf_rates(s.base, prices.close)
    rf = portfolio.risk_free(rates, pd.DatetimeIndex(rets.index), quote)
    key = (*s.key, months, source, pf_rates_key(rates))
    bench = partial(s.bench, pd.DatetimeIndex(s.rets.index))
    stats = cached_ratios(key, rets, rf, bench)
    rate_note = f"무위험수익률 {source}" if not rates.empty else "무위험수익률 0"
    with st.container(horizontal=True, key="pf_risk_cards"):
        st.metric(
            "연 변동성",
            portfolio.percent(stats.volatility)
            if stats.volatility is not None
            else PF_DASH,
            help="**정의** 일간 초과수익률(수익률 − 무위험수익률) 표준편차 × √(연간"
            " 관측일 수)\n\n**해석** 높을수록 수익률 변동 확대\n\n**기준**"
            f" {s.label} 일간 수익률 {stats.days}일 · {rate_note} · " + PF_YEAR_NOTE,
            border=True,
            height="stretch",
        )
        pf_drawdown(stats)
        pf_sharpe(stats)
        if not portfolio.korean_weight(s.w):  # Seoul closes: see the Dimson beta
            beta = portfolio.market_beta(rets, prices.adjusted, s.base)
            st.metric(
                "베타",
                pf_ratio(beta),
                help=f"**정의** {portfolio.BENCHMARK_LABEL} 가격지수"
                f"({portfolio.BENCHMARK}) 대비 Welch 베타 · {s.base} 일간 수익률\n\n"
                "**해석** 1이면 지수와 같은 폭 · 1 초과 지수보다 큰 변동 · 0 근처"
                " 지수와 무관\n\n**기준** "
                + PF_DECAY_NOTE
                + " · 한국 종목 보유 시 미표시(종가"
                " 시각 차이)",
                border=True,
                height="stretch",
            )
    more = st.expander("고급", key="pf_advanced", on_change="rerun")
    if more.open:
        with more:
            pf_advanced(stats, s.bench_label)


def pf_analysis(p: portfolio.Portfolio, prices: portfolio.Prices, months: int) -> None:
    s = pf_subject(p, prices)
    pf_stats(s, prices, months)
    pf_style_box(s, prices, "pf_style")


def pf_style_box(s: Subject, prices: portfolio.Prices, key: str) -> None:
    box = st.expander("스타일 분석", key=key, on_change="rerun")
    if box.open and not s.w.empty:
        with box:
            pf_style_fit(s, prices)


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
    about = (
        "**정의** 주간 수익률을 가장 가깝게 따르는 비교 지수·현금 조합(Sharpe"
        " 1992) · 비중 0 이상, 합 100%\n\n**해석** 설명력 R² 높을수록 조합과 비슷한"
        " 움직임"
    )
    if fit is None:
        headed(
            "**스타일 비중**",
            about,
            [f"**조건** 지금 비중 주간 수익률 {portfolio.WEEKS}주 이상 · 비교 지수"],
        )
        return
    shown = pd.Series(fit.weights, index=list(fit.names), dtype="float64")
    shown = cast("pd.Series", shown[shown > 0.0005])  # noqa: PLR2004 - rounding dust
    r2 = f"{fit.r2:.0%}" if fit.r2 is not None else PF_DASH
    notes = [f"최근 {fit.weeks}주 {s.base} 주간 수익률 · 설명력 R² {r2}"]
    if fit.selection is not None:
        notes.append(
            f"조합 대비 차이(표본 외, {fit.selection_weeks}주 평균) 연"
            f" {portfolio.percent(fit.selection, signed=True)}"
        )
    if fit.dropped:
        notes.append(f"기간 미달로 뺀 지수 {', '.join(fit.dropped)}")
    headed("**스타일 비중**", about + "\n\n**기준** " + " · ".join(notes))
    st.altair_chart(pf_weight_chart(shown, "지수"), alt="스타일 비중")


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
    labels = base.mark_text(color=pf_ink()).encode(
        text=alt.Text("value:Q", format=".2f")
    )
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


def pf_gaps(dropped: list[str], late: str) -> list[str]:
    """The "!" lines for holdings left out of a window or starting inside it."""
    warns: list[str] = []
    if dropped:
        more = " 외" if len(dropped) > PF_HEAT else ""
        warns.append(
            f"**자료** 기간 중 자료 시작·종료로 뺀 {len(dropped)}종목 ·"
            f" {', '.join(dropped[:PF_HEAT])}{more}"
        )
    if late:
        warns.append(f"**자료** 기간 중 자료 시작 · {late}")
    return warns


def pf_correlation(
    s: Subject,
    prices: portfolio.Prices,
    months: int,
    compare: Callable[[], None],
) -> None:
    """Holdings' correlation, betas and spread; ``compare`` fills the right panel."""
    j, _, dropped = pf_joint(s, prices, months)
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        title = f"**보유 종목** · {s.base} 일간 수익률"
        about = (
            "**정의** 보유 종목 일간 수익률의 상관계수\n\n**해석** 1에 가까울수록"
            " 같은 방향 · 0 근처 무관 · 음수 반대 방향\n\n**기준** 비중 상위"
            f" {PF_HEAT}종목 · 모든 종목 값이 있는 날 · " + PF_COV_NOTE
        )
        if j is None or len(j.names) < 2 or len(j.window) < portfolio.MIN_DAYS:  # noqa: PLR2004
            names, days = (len(j.names), len(j.window)) if j is not None else (0, 0)
            headed(
                title,
                about,
                [
                    "**조건** 수익률 있는 보유 종목 2개 이상 · 모든 종목 값이 있는 날"
                    f" {portfolio.MIN_DAYS}일 이상 · 현재 {names}종목, {days}일"
                ],
            )
        else:
            cov, f, shrink = cached_covariance((*s.key, months, "cov"), j)
            shown = pf_largest(j)
            full = portfolio.correlation_frame(cov, j.names)
            matrix = cast("pd.DataFrame", full.loc[shown, shown])
            notes = [f"{len(j.window)}일"]
            if j.mixed:
                notes.append(f"시차 보정 f {f:.2f}")
            if len(j.names) > 2:  # noqa: PLR2004 - two names: the target is the sample
                notes.append(f"Ledoit–Wolf 축소 {shrink:.0%}")
            headed(title, about + " · " + " · ".join(notes), pf_gaps(dropped, j.late))
            st.altair_chart(pf_heatmap(matrix), alt="종목 상관")
    with right, st.container(border=True, height="stretch"):
        compare()
    if j is None or len(j.window) < portfolio.MIN_DAYS:
        return
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(f"**베타** · {s.base} 일간 수익률", PF_BETA_HELP)
        shown = set(pf_largest(j))
        rows = portfolio.betas(j)
        pf_betas(
            [b for b in rows if b.name in shown or b.name == portfolio.PORTFOLIO_LABEL]
        )
    with right:
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
    key = pf_key(book.model_dump_json())
    frame, korean = cached_lineup(key, base, months, book, prices)
    pf_compare_heatmap(
        (*key, base, months),
        frame,
        korean,
        f"**포트폴리오와 {portfolio.BENCHMARK_LABEL}** · {base}",
        "**정의** 포트폴리오별 지금 비중 일간 수익률과 지수의 상관계수\n\n"
        "**해석** 1에 가까울수록 같은 방향 · 0 근처 무관\n\n**기준** 보유 종목 없는"
        " 포트폴리오 제외 · " + PF_INDEX_NOTE + " · " + PF_COV_NOTE,
    )


# Fragments of the tooltips below: each goes in a **기준** paragraph.
PF_INDEX_NOTE = (
    f"S&P 500: 배당 재투자 총수익지수({portfolio.TOTAL_RETURN}) · KOSPI: 배당 제외"
    " 가격지수"
)
PF_DECAY_NOTE = (
    "Welch 가중치: 하루 지날 때마다 e^(−2/256)배 감소, 기간 선택과 무관하게 최근"
    " 약 1년 중심"
)
PF_PROXY_NOTE = (
    "대체 베타: 저장된 모든 날 같은 가중치(감쇠 없음), 한국 종목의 미국 지수 대체는"
    f" Dimson · 1 쪽으로 {quant.BETA_PRIOR_DAYS}/(n + {quant.BETA_PRIOR_DAYS}) 축소(n:"
    f" 사용 일수) · {portfolio.PROXY_MIN_DAYS}일 미만이면 1"
)
PF_YEAR_NOTE = (
    "연환산: 자료의 연간 관측일 수 기준(한국·미국 혼합 약 258일, 미국 약 251일)"
)
PF_COV_NOTE = (
    "공분산: 한국·미국 종가 시차 보정(전날 교차 공분산 × f, 양의 준정부호 유지"
    " 최대값) · Ledoit–Wolf(2004) 상수상관 축소"
)
PF_BETA_HELP = (
    f"**정의** 일간 수익률의 시장 베타 · {portfolio.BENCHMARK_LABEL}(Welch): 하루"
    " 수익률을 지수의 −2~4배로 제한한 가중 회귀 기울기 · Dimson: 한국 종목의 같은 날"
    f" + 전날 {portfolio.BENCHMARK_LABEL} 기울기 합 · KOSPI: 한국 종목의 KOSPI 대비"
    " Welch 베타\n\n**해석** 1이면 지수와 같은 폭 · 1 초과 지수보다 큰 변동 · 0 근처"
    f" 지수와 무관\n\n**기준** {portfolio.BENCHMARK_LABEL} 가격지수"
    f"({portfolio.BENCHMARK}) · 기준 통화 · " + PF_DECAY_NOTE
)


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


def pf_spread(j: portfolio.Joint, spread: portfolio.Spread) -> None:
    with st.container(horizontal=True, key="pf_spread_cards"):
        st.metric(
            "분산 비율",
            pf_ratio(spread.ratio),
            help="**정의** Σ 비중 × 종목 변동성 ÷ 포트폴리오 변동성(Choueifaty–Coignard"
            " 2008)\n\n**해석** 1이면 분산 효과 없음 · 높을수록 분산 효과 확대\n\n"
            "**기준** " + PF_COV_NOTE,
            border=True,
            height="stretch",
        )
        st.metric(
            "유효 종목 수",
            f"{spread.effective:.1f}",
            help="**정의** 1 ÷ Σ 비중²(허핀달 지수의 역수)\n\n**해석** 같은 비중"
            " N종목이면 N · 낮을수록 소수 종목 집중",
            border=True,
            height="stretch",
        )
    with st.container(border=True, height="stretch"):  # the cards stay outside
        headed(
            "**분산** · 지금 비중",
            "**정의** 위험 기여: 포트폴리오 분산 중 종목 몫 · w_i(Σw)_i ÷"
            " w′Σw(오일러 분해, 합 100%)\n\n**해석** 비중보다 위험 기여가 크면"
            " 변동성 집중 종목\n\n"
            "**기준** " + PF_COV_NOTE,
        )
        pf_table(
            {
                "종목": list(j.names),
                "비중": [portfolio.percent(float(v)) for v in j.w],
                "위험 기여": [portfolio.percent(float(v)) for v in spread.risk],
            },
            "위험 기여",
        )


def pf_compare_heatmap(
    key: Key, frame: pd.DataFrame, korean: list[bool], title: str, about: str
) -> None:
    """Whole portfolios' daily correlation, lag-adjusted and shrunk.

    ``key`` stands for ``frame``; ``about`` is the heading's help, ending in
    its **기준** paragraph.
    """
    found = cached_lagged((*key, "compare"), tuple(korean), frame)
    if found is None:
        headed(
            title,
            about,
            [
                f"**조건** 모두 값이 있는 날 {portfolio.MIN_DAYS}일 이상인 수익률 2개"
                " 이상"
            ],
        )
        return
    notes = [f"{found.days}일 · 모두 값이 있는 날"]
    if any(korean):
        notes.append(
            "한국 비중 절반 초과 포트폴리오: 한국 종가 기준 시차 보정 f"
            f" {found.lag:.2f}"
        )
    if len(found.matrix) > 2:  # noqa: PLR2004 - two series: the target is the sample
        notes.append(f"Ledoit–Wolf 축소 {found.shrink:.0%}")
    headed(title, about + " · " + " · ".join(notes))
    st.altair_chart(pf_heatmap(found.matrix), alt="포트폴리오 상관")


# ---- risk


PF_FIT_PAD = 0.05  # a fitted value axis: this share of the span either side
PF_BREAK = "≈"  # at the start of a value axis that does not start at 0
# Each side's bar, darkest first: Reach field, opacity, then per side (falls,
# rises) the column header and what the share counts ({} the level).
PF_SHADES = (
    (
        "end",
        1.0,
        ("기간 말", "기간 말"),
        ("끝에 지금 대비 −{} 이하", "끝에 지금 대비 +{} 이상"),
    ),
    (
        "touched",
        0.6,
        ("기간 중 도달", "기간 중 도달"),
        ("지금 대비 한 번이라도 −{} 이하", "지금 대비 한 번이라도 +{} 이상"),
    ),
    (
        "swing",
        0.3,
        ("고점 대비", "저점 대비"),
        ("기간 중 고점에서 {} 이상 하락", "기간 중 저점에서 {} 이상 반등"),
    ),
)
PF_REACH_MIDDLE = 48  # px, half the centre labels' room
PF_REACH_HEAD = " "  # the row naming the sides and the value columns
PF_REACH_ROW = 44  # px per row
PF_REACH_COLUMN = 54  # px per value column beside the bars
PF_REACH_PAD = 6  # px between the bars and the first value column


def pf_fit(low: float, high: float) -> list[float]:
    """A value axis around ``low``..``high``, ``PF_FIT_PAD`` of the span either side."""
    pad = PF_FIT_PAD * ((high - low) or abs(high) or 1.0)
    return [low - pad, high + pad]


def pf_break(x: float, y: float, *, angle: int) -> alt.Chart:
    """``PF_BREAK`` at ``(x, y)``, the start of a value axis that skips 0.

    ``angle`` 0 crosses a vertical axis line, 90 a horizontal one. Nudged
    along the line off the corner (Vega offsets after rotating: up, or right
    at 90), clear of the tick labels beside it.
    """
    return (
        alt.Chart(alt.Data(values=[{}]))
        .mark_text(text=PF_BREAK, angle=angle, fontSize=14, dy=-10)
        .encode(x=alt.datum(x), y=alt.datum(y))
    )


# Dollars from which amounts show in B/M like investors.dollars: a 13F filer
# manages at least $100M, a personal portfolio stays below.
PF_COMPACT = 1e8


def pf_compact(base: portfolio.Currency, total: float) -> bool:
    """Whether amounts up to ``total`` show in B/M (a 13F clone's billions)."""
    return base == "USD" and abs(total) >= PF_COMPACT


def pf_money(
    value: float, base: portfolio.Currency, *, compact: bool, signed: bool = False
) -> str:
    """``portfolio.money``, or ``investors.dollars`` with a sign when ``compact``."""
    if not compact or not math.isfinite(value):
        return portfolio.money(value, base, signed=signed)
    sign = "−" if value < 0 else ("+" if signed and value > 0 else "")
    return sign + investors.dollars(abs(value))


def pf_value_axis(base: portfolio.Currency, high: float) -> alt.Axis:
    """Amount labels; in B/M when ``pf_compact``, as Vega's SI "G" is not "B"."""
    if not pf_compact(base, high):
        return alt.Axis(format=pf_axis(base))
    return alt.Axis(
        labelExpr="abs(datum.value) >= 1e9"
        " ? format(datum.value / 1e9, ',.1~f') + 'B'"
        " : format(datum.value / 1e6, ',.1~f') + 'M'"
    )


def pf_fan_chart(
    rows: list[dict[str, float]], base: portfolio.Currency
) -> alt.LayerChart:
    number = pf_axis(base)
    low, high = pf_fit(min(r["p5"] for r in rows), max(r["p95"] for r in rows))
    chart = alt.Chart(alt.Data(values=rows)).encode(x=alt.X("day:Q", title="거래일"))
    wide = chart.mark_area(opacity=0.18).encode(
        y=alt.Y(
            "p5:Q",
            title=None,
            axis=pf_value_axis(base, high),
            scale=alt.Scale(domain=[low, high], zero=False, nice=False),
        ),
        y2="p95:Q",
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
    return alt.LayerChart(
        layer=[wide, mid, median, pf_break(0, low, angle=0)]
    ).properties(height=CHART_HEIGHT)


def pf_end_chart(
    rows: list[dict[str, float]], value: float, base: portfolio.Currency
) -> alt.LayerChart:
    number = pf_axis(base)
    low, high = pf_fit(
        min(value, *(r["start"] for r in rows)), max(value, *(r["end"] for r in rows))
    )
    bars = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(
                "start:Q",
                title=None,
                axis=pf_value_axis(base, high),
                scale=alt.Scale(domain=[low, high], zero=False, nice=False),
            ),
            x2="end:Q",
            y=alt.Y("count:Q", title="경로 수"),
            y2=alt.datum(0),  # without it Vega draws a thin bar at the count
            tooltip=[
                alt.Tooltip("start:Q", title="부터", format=number),
                alt.Tooltip("end:Q", title="까지", format=number),
                alt.Tooltip("count:Q", title="경로 수"),
            ],
        )
    )
    now = (
        alt.Chart(alt.Data(values=[{"x": value}]))
        .mark_rule(strokeDash=[3, 3], color=pf_ink())  # gray is lost on the bars
        .encode(x="x:Q")
    )
    return alt.LayerChart(layer=[bars, now, pf_break(low, 0, angle=90)]).properties(
        height=CHART_HEIGHT
    )


def pf_ink() -> str:
    """Text marks' colour for the theme; Vega draws text black on either."""
    return "#fafafa" if st.context.theme.type == "dark" else "#31333f"


def pf_reach_chart(paths: perf.PathRisk) -> alt.LayerChart:
    """Falls left, rises right: per level one bar in three nested shades.

    The shades are adjoining segments, so each reads as its own share from
    the centre without overlap. Past each side's bars, one column of shares
    per shade (the darkest nearest), headed by its name and a swatch. Each
    side has its own x scale, so the centre labels keep their room in px.
    """
    ink = pf_ink()
    longest = max(
        0.05, *(r.swing for d in (paths.falls, paths.rises) for r in d.values())
    )
    middle = PF_REACH_MIDDLE
    sides = (
        ("하락", -1, paths.falls, display.DOWN_COLOR, [0, f"width / 2 - {middle}"]),
        ("상승", 1, paths.rises, display.UP_COLOR, [f"width / 2 + {middle}", "width"]),
    )
    rows = [f"{level:.0%}" for level in perf.PATH_LEVELS]
    y = alt.Y(
        "row:N",
        sort=[PF_REACH_HEAD, *rows],
        axis=None,
        scale=alt.Scale(paddingInner=0.3),
    )
    head = alt.Data(values=[{"row": PF_REACH_HEAD, "x": 0.0}])
    layers: list[alt.Chart] = []
    for side, sign, reach, color, span in sides:
        part = int(sign > 0)
        x = partial(
            alt.X,
            axis=None,
            scale=alt.Scale(
                domain=sorted([0.0, sign * longest]),
                range=[{"expr": str(end)} for end in span],
            ),
        )
        layers.append(
            alt.Chart(head)
            .mark_text(
                text=side,
                align="left" if sign > 0 else "right",
                baseline="bottom",
                dy=22,
                fontSize=14,
                fontWeight="bold",
                color=ink,
            )
            .encode(x=x("x:Q"), y=y)
        )
        outside = alt.value(0 if sign < 0 else "width")  # the bars' end, in px
        bars: list[dict[str, object]] = []
        start = [0.0] * len(rows)
        for k, (field, alpha, names, means) in enumerate(PF_SHADES):
            name, mean = names[part], means[part]
            dx = sign * (PF_REACH_PAD + (k + 0.5) * PF_REACH_COLUMN)
            values: list[dict[str, object]] = []
            for i, (row, (level, r)) in enumerate(
                zip(rows, reach.items(), strict=True)
            ):
                stop = float(getattr(r, field))
                share = display.share(stop)
                bars.append(
                    {
                        "row": row,
                        "shade": field,
                        "x": sign * start[i],
                        "x2": sign * stop,
                        "name": name,
                        "mean": mean.format(f"{level:.0%}"),
                        "share": share,
                    }
                )
                values.append({"row": row, "text": share})
                start[i] = stop
            layers += [
                alt.Chart(alt.Data(values=values))
                .mark_text(dx=dx, fontSize=13, color=ink)
                .encode(x=outside, y=y, text="text:N"),
                alt.Chart(head)
                .mark_text(
                    text=name.rsplit(" ", 1),
                    dx=dx,
                    dy=-1,
                    baseline="bottom",
                    fontSize=12,
                    lineHeight=14,
                    color=ink,
                )
                .encode(x=outside, y=y),
                alt.Chart(head)
                .mark_text(
                    text="■",
                    dx=dx,
                    dy=24,
                    baseline="bottom",
                    fontSize=12,
                    color=color,
                    opacity=alpha,
                )
                .encode(x=outside, y=y),
            ]
        layers.append(
            alt.Chart(alt.Data(values=bars))
            .mark_bar(color=color)
            .encode(
                x=x("x:Q"),
                x2="x2:Q",
                y=y,
                opacity=alt.Opacity(
                    "shade:N",
                    scale=alt.Scale(
                        domain=[field for field, _, _, _ in PF_SHADES],
                        range=[alpha for _, alpha, _, _ in PF_SHADES],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("name:N", title="측정"),
                    alt.Tooltip("mean:N", title="정의"),
                    alt.Tooltip("share:N", title="경로 비율"),
                ],
            )
        )
        if sign < 0:  # the centre labels, middle px right of the falls' zero
            layers.append(
                alt.Chart(
                    alt.Data(
                        values=[
                            {"row": row, "x": 0.0, "text": f"−{row} | +{row}"}
                            for row in rows
                        ]
                    )
                )
                .mark_text(dx=middle, fontSize=13, color=ink)
                .encode(x=x("x:Q"), y=y, text="text:N")
            )
    return (
        alt.LayerChart(layer=layers)
        .resolve_scale(x="independent")
        .properties(height=PF_REACH_ROW * (len(rows) + 1))
    )


PF_ZERO_DRIFT = "평균 수익 0 가정"


def pf_reach_help(paths: perf.PathRisk, drift: str = PF_ZERO_DRIFT) -> str:
    """The 하락·상승 chart's help, with the recovery figures under 기준."""
    low, mid, high = paths.recovery_low, paths.recovery_median, paths.recovery_high
    if paths.never == 0 and mid is None:
        recovery = "전고점 회복: 지금이 고점"
    else:
        if low is None:
            spread = "기간 밖"
        elif high is None:
            spread = f"{low}거래일–기간 밖"
        else:
            spread = f"{low}–{high}거래일"
        middle = "기간 밖" if mid is None else f"{mid}거래일"
        recovery = (
            f"전고점 회복 중앙값 {middle} · 10–90% {spread} · 기간 내 미회복"
            f" {display.share(paths.never)}"
        )
    levels = "·".join(f"{level * 100:.0f}" for level in perf.PATH_LEVELS)
    return (
        f"**정의** {levels}% 하락·상승에 닿은 모의 경로 비율 · 왼쪽 하락 · 오른쪽 상승"
        " · 막대 하나에 세 겹, 짙을수록 좁은 기준 · 기간 말: 기간 끝 평가액의 지금 대비"
        " 변화 · 기간 중 도달: 기간 중 한때 지금 대비 도달 · 고점 대비: 그때까지의"
        " 고점(지금 포함) 대비 하락 · 저점 대비: 그때까지의 저점(지금 포함) 대비 상승 ·"
        " 숫자: 막대 바깥 열 · 열 머리 색 견본: 해당 겹\n\n**해석** 겹 사이 간격:"
        " 기간 중 겪었지만 기간 끝에 남지 않은 손실(하락 쪽) · 기회(상승 쪽)\n\n"
        f"**기준** {drift} · 경로 {portfolio.PATHS:,}개 · {recovery}"
    )


PF_THIN = "표본 부족"
PF_MODELS: dict[str, str] = {"gjr": "GJR-GARCH", "har": "HAR", "pdv": "경로의존(PDV)"}


@dataclasses.dataclass(frozen=True, slots=True)
class PfVol:
    """The risk section's volatility filter: its options, fit and label.

    ``key`` (model, ranges used, VIX used) stands for ``v`` in the caches.
    """

    v: portfolio.Vol
    key: Key
    f: quant.Devolatilised
    label: str


def pf_vol(
    key: Key,
    s: Subject,
    prices: portfolio.Prices,
    j: portfolio.Joint,
    *,
    inline: bool = False,
) -> PfVol:
    """The 변동성 모델 controls and the filter they choose (cached).

    The opens, highs and lows or VIX are fetched only when asked. ``inline``:
    in the caller's row, not a row of their own.
    """
    fetcher = portfolio_fetcher(str(pf_root()))
    can = not args.demo and not portfolio.missing_packages()
    row = (
        contextlib.nullcontext()
        if inline
        else st.container(horizontal=True, vertical_alignment="bottom")
    )
    with row:
        picked = st.segmented_control(
            "변동성 모델",
            list(PF_MODELS),
            format_func=PF_MODELS.__getitem__,
            default="gjr",
            required=True,
            key="pf_vol_model",
            persist_state="page",  # 전망 shows it only under FHS
            help="**정의** FHS·EVT VaR·ES, FHS 시뮬레이션, FHS·EVT 사후검정의"
            " 변동성 필터 · GJR-GARCH(1,1): 음의 충격에 더 큰 분산 반응 · HAR: 어제·"
            "지난주·지난달 평균 분산의 회귀(Corsi 2009) · 경로의존(PDV): 과거 수익률과"
            " 그 제곱의 지수가중 평균(Guyon–Lekeufack 2023)\n\n**기준** 종목별 최근"
            f" {quant.FILTER_DAYS:,}거래일 추정 · GJR-GARCH는 {quant.MIN_GARCH}일 이상,"
            f" 적합 실패 종목은 EWMA(λ={quant.EWMA_LAMBDA})",
        )
        model = cast("quant.Filter", picked or "gjr")
        ranges = vix = False
        if model != "gjr":
            with st.container(width="content"):  # the toggles stacked: a narrow row
                ranges = st.toggle(
                    "고가·저가 사용",
                    key="pf_ranges",
                    persist_state="page",
                    help="**정의** 필터의 일간 분산 입력을 Garman–Klass(1980) 범위"
                    " 분산으로 대체 · 시가·고가·저가·종가와 전일 종가(갭 포함)\n\n"
                    "**해석** 수익률 제곱보다 잡음이 적은 분산 추정\n\n**기준** HAR·PDV"
                    " 전용 · 자료 없는 날·종목은 수익률 제곱",
                )
                vix = model == "har" and st.toggle(
                    "VIX 포함",
                    key="pf_vix",
                    persist_state="page",
                    help="**정의** 전일 VIX의 일간 분산 환산값((VIX/100)²/252)을 HAR"
                    " 회귀변수로 추가\n\n**기준** HAR 전용 · 한국 종목은 그 전 미국"
                    " 종가의 VIX",
                )
        absent = portfolio.no_ranges(prices, j.names) if ranges else []
        running = fetcher.progress.running
        if (
            can
            and absent
            and st.button(
                "고가·저가 받기",
                icon=":material/download:",
                key="pf_ranges_fetch",
                disabled=running,
                help=", ".join(absent),
            )
        ):
            fetcher.start(absent, ranges=True)
            st.rerun()
        index = cast("pd.Index", j.full.index[-portfolio.FIT_ROWS :])  # fits read
        x = portfolio.vix_regressor(prices.adjusted, j.names, index) if vix else None
        if (
            can
            and vix
            and x is None
            and st.button(
                "VIX 받기",
                icon=":material/download:",
                key="pf_vix_fetch",
                disabled=running,
                help=portfolio.VIX,
            )
        ):
            fetcher.start([portfolio.VIX])
            st.rerun()
    proxy = portfolio.range_proxy(prices, j.names, s.base, index) if ranges else None
    v = portfolio.Vol(model, proxy, x)
    chosen = (model, proxy is not None, x is not None)
    f = cached_filter((*key, "filter", *chosen), j, v)
    label = portfolio.filter_label(j, f)
    inputs = [n for n, a in (("고가·저가", proxy), ("VIX", x)) if a is not None]
    if f.model is not None and inputs:
        label += f" ({', '.join(inputs)})"
    return PfVol(v, chosen, f, label)


def pf_density(d: portfolio.Density, *, fhs: bool, capm: bool = False) -> None:
    drawn = (
        f"필터 추정 기간의 겹치지 않는 {quant.MONTH_DAYS}일 구간마다 첫날 변동성"
        " 기준 모의"
        if fhs
        else f"전체 기간의 겹치지 않는 {quant.MONTH_DAYS}일 구간마다 직전 선택 기간"
        " 수익률의 묶음 재표본(Politis–White 2004, 묶음 길이도 그 기간"
        " 추정) · 앞에 선택 기간만큼 자료가 있는 구간"
    ) + (
        " · 기대수익: 구간 첫날 무위험수익률 + 지금 베타 × 위험프리미엄"
        if capm
        else f" · {PF_ZERO_DRIFT}"
    )
    headed(
        f"**분포 검증** · 1개월, {d.blocks}구간",
        "**정의** 1개월 수익률 예측 분포의 사후 검증 · PIT: 예측 1개월 수익률"
        f" {portfolio.DENSITY_PATHS:,}개 중 실제 수익률의 위치 · LR: Berkowitz(2001)"
        " 우도비, Φ⁻¹(PIT)의 AR(1)이 평균 0·분산 1·자기상관 0인지 검정(χ²(3))\n\n"
        "**해석** p가 작을수록(통상 0.05 미만) 예측 분포의 위치·폭·자기상관"
        f" 불일치\n\n**기준** 예측: {drawn} · 구간 {quant.MIN_MONTHS}개 이상",
        []
        if d.test is not None
        else [f"**조건** 1개월 구간 {quant.MIN_MONTHS}개 이상 · 현재 {d.blocks}구간"],
    )
    if d.test is None:
        return
    pf_table(
        {
            "구간": [f"{d.blocks}개"],
            "LR": [f"{d.test.lr:.2f}"],
            "p": [f"{d.test.p:.2f}"],
            "평균": [f"{d.test.mean:+.2f}"],
            "표준편차": [f"{d.test.sd:.2f}"],
            "자기상관": [f"{d.test.rho:+.2f}"],
        },
        "분포 검증",
    )


PF_RESAMPLE = "resample"
PF_GEN_CHUNK = 2000  # paths per generator call: (2000, 252, 6) is 24 MB
PF_GEN_HISTORY = 20  # recent days handed to SigCWGAN (it reads its last p)
PF_RETRAIN = "설정 > 생성 모델 학습에서 재학습"


@dataclasses.dataclass(frozen=True, slots=True)
class PfGens:
    """Trained generators the subject can use, and why the others cannot."""

    usable: dict[str, generative.Generator]
    blocked: dict[str, str]  # label: reason
    base: portfolio.Currency
    adjusted: pd.DataFrame  # SigCWGAN's recent returns come from here


def pf_gen_stamp(folder: Path) -> tuple[int, ...]:
    files = (folder / generative.WEIGHTS, folder / generative.META)
    return tuple(n for f in files for n in (f.stat().st_mtime_ns, f.stat().st_size))


@st.cache_resource(show_spinner=False, max_entries=6)
def cached_generator(
    folder: str, stamp: tuple[int, ...]
) -> generative.Generator | None:
    """A generator's weights, read once per file stamp; read-only."""
    del stamp
    return generative.load(Path(folder))


def pf_gens(
    j: portfolio.Joint, base: portfolio.Currency, adjusted: pd.DataFrame
) -> PfGens:
    """Ready generators whose tickers hold every name, and USD/KRW if needed."""
    if args.demo:  # every model to choose, none loaded
        tickers = (*j.names, portfolio.FX)
        usable = {m: demo_results.generator(m, tickers) for m in generative.METHODS}
        return PfGens(usable, {"Kronos": generative.IN_PROGRESS}, base, adjusted)
    root = gen_root()
    usable: dict[str, generative.Generator] = {}
    blocked: dict[str, str] = {}
    foreign = any(portfolio.currency(n) != base for n in j.names)
    for s in generative.status(root):
        label = GEN_LABELS[s.model]
        if s.model == "kronos":
            blocked[label] = generative.IN_PROGRESS if s.ready else s.note
            continue
        folder = root / s.model
        gen = cached_generator(str(folder), pf_gen_stamp(folder)) if s.ready else None
        if gen is None:
            blocked[label] = f"{s.note} · {PF_RETRAIN}"
            continue
        missing = [n for n in j.names if n not in gen.tickers]
        if foreign and portfolio.FX not in gen.tickers:
            missing.append(portfolio.FX)
        if missing:
            blocked[label] = f"{', '.join(missing)} 없음 · {PF_RETRAIN}"
        else:
            usable[s.model] = gen
    return PfGens(usable, blocked, base, adjusted)


def pf_generator_controls(gens: PfGens) -> generative.Generator | None:
    """경로 생성; the picked generator, None for resampling."""
    if state.get("pf_gen", PF_RESAMPLE) not in (PF_RESAMPLE, *gens.usable):
        del state["pf_gen"]  # the subject changed: back to resampling
    picked = st.segmented_control(
        "경로 생성",
        [PF_RESAMPLE, *gens.usable],
        format_func=lambda m: "재표본" if m == PF_RESAMPLE else GEN_LABELS[m],
        default=PF_RESAMPLE,
        required=True,
        key="pf_gen",
        help="**정의** 시뮬레이션 경로 생성 방식 · 재표본: 과거 수익률"
        " 재표본(FHS 선택 가능) · Tail-GAN(Cont 외) · SigCWGAN(Liao 외 2020) ·"
        " 신경망 SDE(Issa 외 2023): data/models의 학습된 생성 모델 경로\n\n"
        "**기준** 보유 종목을 모두 담은 모델 · 학습: 설정 > 생성 모델 학습",
    )
    return gens.usable.get(picked or PF_RESAMPLE)


def pf_blocked_generators(gens: PfGens) -> None:
    """The generators this subject cannot use, shown disabled."""
    if gens.blocked:
        st.segmented_control(
            "학습 필요",
            list(gens.blocked),
            disabled=True,
            key="pf_gen_off",
            help="**정의** 이 구성에 쓸 수 없는 생성 모델\n\n**기준** "
            + " · ".join(f"{label}: {why}" for label, why in gens.blocked.items()),
        )


def pf_history(adjusted: pd.DataFrame, gen: generative.Generator) -> pd.DataFrame:
    """SigCWGAN's condition: its tickers' latest simple returns, days all traded."""
    if gen.method != "sigcwgan":
        return pd.DataFrame()
    prices = adjusted.reindex(columns=list(gen.tickers)).dropna(how="any")
    return cast("pd.DataFrame", (prices / prices.shift() - 1).iloc[1:]).tail(
        PF_GEN_HISTORY
    )


def pf_gen_key(gen: generative.Generator) -> Key:
    if args.demo:
        return (gen.method,)
    return (gen.method, *pf_gen_stamp(gen_root() / gen.method))


def pf_gen_note(
    gen: generative.Generator,
    base: portfolio.Currency,
    j: portfolio.Joint,
    recent: bool,
    drift: str,
) -> str:
    """How the generator's paths are drawn, for the method notes."""
    span = f"{gen.meta.get('train_start', '?')}–{gen.meta.get('train_end', '?')}"
    how = (
        f"{gen.window}일 단위 독립 생성 뒤 연결"
        if gen.window is not None
        else (
            "최근 수익률 조건의 일별 생성"
            if recent
            else "학습 자료 마지막 날들 조건의 일별 생성"
        )
    )
    fx = (
        f" · 다른 통화 종목은 같은 경로의 {portfolio.FX}로 {base} 환산"
        if any(portfolio.currency(n) != base for n in j.names)
        else ""
    )
    return (
        f"{GEN_LABELS[gen.method]}({span} 학습) 생성 경로 {portfolio.PATHS:,}개,"
        f" {how}{fx} · {drift} · 지금 비중 매일 재조정"
    )


def pf_generated(
    gen: generative.Generator,
    j: portfolio.Joint,
    base: portfolio.Currency,
    days: int,
    history: pd.DataFrame | None,
) -> NDArray[np.float64]:
    """``portfolio.PATHS`` generated paths as daily portfolio returns in ``base``.

    Drawn ``PF_GEN_CHUNK`` at a time from seed ``quant.SEED``, so only the
    (paths, days) portfolio returns are kept.
    """
    rng = np.random.default_rng(quant.SEED)
    columns = [gen.tickers.index(n) for n in j.names]
    fx = gen.tickers.index(portfolio.FX) if portfolio.FX in gen.tickers else None
    parts: list[NDArray[np.float64]] = []
    for done in range(0, portfolio.PATHS, PF_GEN_CHUNK):
        drawn = gen.paths(min(PF_GEN_CHUNK, portfolio.PATHS - done), days, rng, history)
        rates = None if fx is None else drawn[..., fx]
        parts.append(portfolio.generated_returns(j, drawn[..., columns], rates, base))
    return np.concatenate(parts)


@st.cache_data(show_spinner="생성 모델 경로 계산 중…", max_entries=8)
def cached_generated(  # noqa: PLR0913 - every input of the cached result
    key: Key,
    base: portfolio.Currency,
    days: int,
    value: float,
    standing: tuple[float, float],
    _gen: generative.Generator,
    _j: portfolio.Joint,
    _history: pd.DataFrame | None,
    drift: float = 0.0,
) -> portfolio.Simulation:
    """``cached_paths`` for a generator; ``key`` stands for it, ``_j`` and history."""
    del key
    if args.demo:
        return demo_results.simulation(value, days, drift)
    daily = pf_generated(_gen, _j, base, days, _history)
    return portfolio.simulation(portfolio.path_growth(daily), value, *standing, drift)


def pf_paths(  # noqa: PLR0913 - the section's inputs
    key: Key,
    j: portfolio.Joint,
    value: float,
    s: Subject,
    vol: PfVol | None,
    gen: generative.Generator | None,
    adjusted: pd.DataFrame,
    warns: list[str],
    drift: PfDrift | None = None,
) -> None:
    """Monte Carlo of today's weights; the draws are shared across weights.

    ``key`` stands for ``j`` and ``s.rets``; ``vol`` is the filter under FHS,
    None for plain resampling; ``gen`` the generator drawing the paths
    instead, its recent returns from ``adjusted``; ``warns`` the "!" lines
    for the first heading; ``drift`` the expected returns, None for zero.
    """
    base, held = s.base, s.rets
    today = 0.0 if drift is None else float(drift.daily[-1])
    assumed = PF_ZERO_DRIFT if drift is None else drift.note
    label = state.get("pf_horizon") or "1M"  # 전망 기간, in the top row
    days = portfolio.HORIZONS[label]
    standing = portfolio.standing(held)
    if gen is not None:
        history = pf_history(adjusted, gen)
        sim = cached_generated(
            (
                *key,
                *pf_gen_key(gen),
                str(cast("object", history.index.max())) if len(history) else "",
            ),
            base,
            days,
            value,
            standing,
            gen,
            j,
            history if len(history) else None,
            today,
        )
        method = pf_gen_note(gen, base, j, len(history) > 0, assumed)
    else:
        f = None if vol is None else vol.f
        block = 1 if vol else portfolio.mean_block(j)  # residuals: independent days
        n = len(j.window)
        picks = cached_draws(state.get("pf_prices_stamp", ()), n, days, block)
        chosen = () if vol is None else vol.key
        sim = cached_paths(
            (*key, days, block, *chosen),
            f is not None,
            value,
            standing,
            j,
            picks,
            _f=f,
            drift=today,
        )
        first = cast("pd.Timestamp", j.window.index[0]).date()
        last = cast("pd.Timestamp", j.window.index[-1]).date()
        drawn = (
            f"같은 날 수익률의 평균 {block:.1f}일 묶음(Politis–White 2004)"
            if f is None
            else "같은 날 잔차의 하루 단위"
        )
        method = (
            f"{assumed} · 과거 {n}거래일({first}–{last}) {drawn} 재표본"
            f"(경로 {portfolio.PATHS:,}개) · 지금 비중 매일 재조정"
            + (f" · {pf_fhs_note(vol)}" if vol else "")
        )
    with st.container(border=True):
        headed(
            f"**하락·상승 경로 비율** · {label}",
            pf_reach_help(sim.paths, assumed),
            warns,
        )
        st.altair_chart(pf_reach_chart(sim.paths), alt="하락·상승")
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(
            f"**시뮬레이션 범위** · 5–95%, 25–75%, 중앙값, {base}",
            "**정의** 날짜별 모의 평가액의 5·25·50·75·95 백분위\n\n**기준** "
            + method
            + "".join(f" · {note}" for note in s.notes),
        )
        st.altair_chart(pf_fan_chart(sim.fan, base), alt="범위")
    with right, st.container(border=True, height="stretch"):
        headed(
            f"**{label} 뒤 평가액 분포** · {base}",
            f"**정의** 기간 끝 모의 평가액의 히스토그램({portfolio.BINS}구간) ·"
            " 점선: 지금 평가액",
        )
        st.altair_chart(pf_end_chart(sim.ends, value, base), alt="분포")


def pf_fhs_note(vol: PfVol) -> str:
    """How simulated volatility moves under the filter, for the method notes."""
    f = vol.f
    head = f"FHS {vol.label}: 지금 변동성 출발"
    if f.model is not None:
        return (
            f"{head} · 처음 {quant.FEEDBACK_DAYS}일 경로별"
            f" {portfolio.MODEL_NAMES[f.model.kind]} 분산식 갱신 · 그 뒤 기댓값 경로"
        )
    if not f.garch.any():
        return f"{head} · 장기 수준 회귀"
    rest = "" if f.garch.all() else " · EWMA 종목은 장기 수준 회귀"
    return (
        f"{head} · 처음 {quant.FEEDBACK_DAYS}일 경로별 GARCH 분산식 갱신 · 그 뒤"
        f" 기댓값 경로로 장기 수준 수렴{rest}"
    )


def pf_loss(value: float | None) -> float | None:
    """A loss for display: a gain shows as 0."""
    return None if value is None else max(value, 0.0)


def pf_tail_table(t: portfolio.Tails, value: float, base: portfolio.Currency) -> None:
    def pct(x: float | None) -> str:
        loss = pf_loss(x)
        return PF_THIN if loss is None else portfolio.percent(loss, dec=2)

    compact = pf_compact(base, value)

    def amount(x: float | None) -> str:
        loss = pf_loss(x)
        return (
            PF_THIN if loss is None else pf_money(loss * value, base, compact=compact)
        )

    def span(ci: tuple[float, float] | None) -> str:
        if ci is None:
            return PF_DASH
        high = "∞" if math.isinf(ci[1]) else pct(ci[1])
        return f"{pct(ci[0])}–{high}"

    pf_table(
        {
            "방법": [r.method for r in t.rows],
            "신뢰수준": [f"{r.level:.1%}".replace(".0%", "%") for r in t.rows],
            "VaR": [pct(r.var) for r in t.rows],
            f"VaR ({base})": [amount(r.var) for r in t.rows],
            "VaR 신뢰구간": [span(r.var_ci) for r in t.rows],
            "ES": [pct(r.es) for r in t.rows],
            f"ES ({base})": [amount(r.es) for r in t.rows],
            "ES 신뢰구간": [span(r.es_ci) for r in t.rows],
        },
        "VaR·ES",
    )


def pf_tails(
    t: portfolio.Tails,
    j: portfolio.Joint,
    value: float,
    base: portfolio.Currency,
) -> None:
    """``t``: ``j``'s tails, below the VaR·ES box."""
    with st.container(border=True):
        headed(
            f"**사후검정** · 95% VaR·ES, 최근 {quant.BACKTEST_DAYS}일"
            + (f", 직전 {t.lookback:,}일 기준" if t.lookback else ""),
            f"**정의** 최근 {quant.BACKTEST_DAYS}일 각각의 95% VaR·ES 예측과"
            " 실제 손실의 비교 · 초과 횟수: 손실 > 그날 VaR인 날 · 기대: 일수 × 5% ·"
            " Kupiec p: 초과 빈도의 우도비 검정(χ²(1)) · Christoffersen p: 초과의 연속"
            " 몰림 검정(χ²(1)) · ES Z2: 1 − Σ(초과일 손실 ÷ 그날 ES) ÷ (일수 × 5%),"
            " Acerbi–Székely(2014) · Z2 p: 단측 · U p: 누적 초과 크기, Du–Escanciano"
            f"(2017) 양측 · C p: 누적 초과의 1–{quant.DE_LAGS}일 시차 자기상관(같은"
            " 논문) · 신호등: 초과 횟수의 이항 누적확률"
            f" {quant.GREEN:.0%} 미만 녹색, {quant.YELLOW:.2%} 미만 황색, 그 이상"
            " 적색\n\n**해석** p가 작을수록(통상 0.05 미만) 95% 가정과 불일치 · Z2 0"
            " 미만이면 ES 과소 추정 · 신호등 기준: Basel(250일 99%면 0–4·5–9·10회"
            " 이상)\n\n**기준** 예측: 직전 선택 기간과 같은 일수(FHS는 그 잔차 × 그날"
            f" 변동성, EVT는 FHS 예측의 손실 상위 {quant.GPD_SHARE:.0%}를 일반화"
            " 파레토 분포로 대체), 기간"
            f" {quant.min_samples(quant.BACKTEST_LEVEL)}일 미만이거나"
            f" 그만큼 앞선 자료가 없으면 직전 {quant.BACKTEST_DAYS}일 · FHS 필터: 검정"
            " 기간 전 자료로 추정 · Z2·U·C p: 예측 분포 모의"
            f" {quant.ES_SIMS:,}회",
            pf_backtest_notes(t),
        )
        pf_backtest(t)
    if len(j.names) == 1:  # one holding: its ES share is the whole ES
        return
    with st.container(border=True):
        headed(
            "**종목별 ES 기여** · 97.5%",
            "**정의** Tasche 성분 ES · 포트폴리오가 가장 나빴던"
            f" {1 - quant.TAIL_LEVEL:.1%} 날의 종목별 평균 손실 × 비중 · 합계 ="
            f" 포트폴리오 ES {quant.TAIL_LEVEL:.1%}\n\n**해석** 클수록 꼬리 손실 기여"
            " 확대 · 음수면 꼬리 구간 손실 상쇄",
            []
            if t.component is not None
            else [
                f"**조건** 기간 수익률 {quant.min_samples(quant.TAIL_LEVEL):,}일 이상 ·"
                f" 현재 {t.days:,}일"
            ],
        )
        if t.component is not None:
            pf_table(
                {
                    "종목": list(j.names),
                    "ES 기여": [
                        portfolio.percent(float(c), dec=2) for c in t.component
                    ],
                    f"ES 기여 ({base})": [
                        pf_money(
                            float(c) * value, base, compact=pf_compact(base, value)
                        )
                        for c in t.component
                    ],
                },
                "종목별 ES",
            )


PF_LIGHTS: dict[str, str] = {"green": "녹색", "yellow": "황색", "red": "적색"}


def pf_backtest_notes(t: portfolio.Tails) -> list[str]:
    if not t.backtests:
        return [f"**조건** 전체 기간 수익률 500일 이상 · 현재 {t.history:,}일"]
    if t.backtests.get("FHS") and not t.backtests.get("EVT"):
        days = quant.MIN_EXCESSES / quant.GPD_SHARE
        return [
            f"**조건** EVT: 예측 기준 일수 {days:,.0f}일 이상 · 현재 {t.lookback:,}일"
        ]
    return []


def pf_backtest(t: portfolio.Tails) -> None:
    if not t.backtests:
        return
    rows = list(t.backtests.items())

    def cells(show: Callable[[quant.Backtest], str]) -> list[str]:
        return [PF_THIN if bt is None else show(bt) for _, bt in rows]

    pf_table(
        {
            "방법": [method for method, _ in rows],
            "초과 횟수": cells(lambda bt: f"{bt.exceptions}회"),
            "기대": cells(lambda bt: f"{bt.expected:.1f}회"),
            "Kupiec p": cells(lambda bt: f"{bt.kupiec:.2f}"),
            "Christoffersen p": cells(lambda bt: f"{bt.christoffersen:.2f}"),
            "ES Z2": cells(lambda bt: pf_ratio(bt.z2)),
            "Z2 p": cells(lambda bt: f"{bt.es:.2f}"),
            "U p": cells(lambda bt: f"{bt.de_u:.2f}"),
            "C p": cells(lambda bt: f"{bt.de_c:.2f}"),
            "신호등": cells(lambda bt: PF_LIGHTS[bt.light]),
        },
        "사후검정",
    )


def pf_risk_inputs(
    s: Subject, prices: portfolio.Prices, months: int
) -> tuple[Key, portfolio.Joint, float, list[str]] | None:
    """The tail and outlook sections' joint returns (cached).

    Last, the "!" lines for the section's first heading.
    """
    j, share, dropped = pf_joint(s, prices, months)
    if j is None:
        st.info("가격이 확인된 보유 종목이 없습니다.")
        return None
    value = s.value * share
    if len(j.window) < portfolio.MIN_DAYS:
        st.info(f"모든 보유 종목의 수익률이 있는 날이 {len(j.window)}일뿐입니다.")
        return None
    key = (*s.key, months)
    return key, j, value, pf_gaps(dropped, j.late)


def pf_tail(
    s: Subject, prices: portfolio.Prices, months: int, row: DeltaGenerator
) -> None:
    """``row``: the 추정 기간 row, which the 변동성 모델 control joins."""
    found = pf_risk_inputs(s, prices, months)
    if found is None:
        return
    key, j, value, warns = found
    with row:
        vol = pf_vol(key, s, prices, j, inline=True)
    f = vol.f
    t = cached_tails((*key, "tails", *vol.key), j, vol.f, vol.v)
    excesses = quant.MIN_EXCESSES / quant.GPD_SHARE
    with st.container(border=True):
        headed(
            "**VaR·ES** · 지금 비중, 1일",
            "**정의** 지금 비중 포트폴리오의 1일 손실 · VaR: 손실 상위 n × (1 −"
            " 신뢰수준)번째 값 · ES: 그 상위 손실의 평균(경계값 비례 반영) · 역사적:"
            " 기간 수익률 그대로 · FHS: 종목마다 과거 수익률 ÷ 그날 변동성 × 지금"
            f" 변동성 · EVT: FHS 손실 상위 {quant.GPD_SHARE:.0%}에 일반화 파레토 분포"
            " 적합 뒤 그 너머 외삽(McNeil–Frey 2000)\n\n**해석** VaR: 신뢰수준만큼의"
            " 날에 넘지 않는 손실 · ES: VaR 너머 손실의 평균 · 신뢰구간: 추정의"
            f" {quant.CI_LEVEL:.0%} 범위, 위 끝 ∞는 자료로 상한 미정(EVT) · 이익이면"
            f" 0\n\n**기준** 최소 표본: VaR 너머 날 {quant.MIN_TAIL}일 이상("
            + ", ".join(
                f"{level:.1%}".replace(".0%", "%") + f" {quant.min_samples(level):,}일"
                for level in portfolio.VAR_LEVELS
            )
            + f"), EVT는 상위 {quant.GPD_SHARE:.0%} 손실 {quant.MIN_EXCESSES}일 이상"
            f"({excesses:,.0f}일) · 변동성 필터 {vol.label}(최근"
            f" {len(f.z):,}거래일로 추정) · 신뢰구간: 역사적·FHS는 정상 블록"
            " 부트스트랩(Politis–Romano 1994, 평균 블록 Politis–White 2004)"
            f" {quant.BOOT_PATHS:,}회의 백분위, EVT는 프로파일 우도"
            "(McNeil–Frey–Embrechts 2015), 임계값 초과 손실 비율의 이항 불확실성"
            " 포함, 지금 변동성은 주어진 값"
            + "".join(f" · {note}" for note in s.notes),
            warns,
        )
        pf_tail_table(t, value, s.base)
    pf_tails(t, j, value, s.base)


PF_DRIFTS = {"zero": "0", "capm": "시장 기대"}


@dataclasses.dataclass(frozen=True, slots=True)
class PfDrift:
    """CAPM daily expected returns on the joint's days, their note and key."""

    daily: NDArray[np.float64]
    note: str
    key: tuple[int, str]  # the risk-free rates' (``pf_rates_key``)


def pf_drift(
    j: portfolio.Joint, base: portfolio.Currency, prices: portfolio.Prices
) -> PfDrift:
    rates, source, quote = pf_rates(base, prices.close)
    rf = portfolio.risk_free(rates, pd.DatetimeIndex(j.full.index), quote)
    daily = portfolio.expected_returns(j, rf)
    annual = (1 + daily[-1]) ** portfolio.per_year(j.full.index) - 1
    rate = f"무위험수익률 {source}" if not rates.empty else "무위험수익률 0"
    return PfDrift(
        daily,
        f"기대수익 연 {annual:.1%}(CAPM: {rate} + 베타 × 위험프리미엄 연"
        f" {portfolio.ERP:.0%})",
        pf_rates_key(rates),
    )


def pf_outlook(s: Subject, prices: portfolio.Prices, months: int) -> None:
    found = pf_risk_inputs(s, prices, months)
    if found is None:
        return
    key, j, value, warns = found
    gens = pf_gens(j, s.base, prices.adjusted)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        gen = pf_generator_controls(gens)
        fhs = gen is None and st.toggle(
            "현재 변동성 반영 (FHS)",
            key="pf_fhs",
            persist_state="page",  # kept while a generator hides it
            help="**정의** 필터 잔차 재표본(Filtered Historical Simulation) ·"
            " 오늘 변동성에서 시작하는 경로\n\n**해석** 끄면 과거 수익률 묶음"
            " 재표본 · 켜면 지금 변동성 수준 반영\n\n**기준** 필터: 옆 변동성 모델"
            " · 재표본 전용",
        )
        # the filter only for FHS, its controls next to the toggle
        vol = pf_vol(key, s, prices, j, inline=True) if fhs else None
        expected = st.segmented_control(
            "기대수익",
            list(PF_DRIFTS),
            format_func=PF_DRIFTS.__getitem__,
            default="zero",
            required=True,
            key="pf_drift",
            help="**정의** 경로에 더하는 기대수익 · 0: 평균 수익 0 · 시장 기대:"
            " 무위험수익률 + 포트폴리오 베타(S&P 500, 추정 기간) × 위험프리미엄"
            f" 연 {portfolio.ERP:.0%}(CAPM)",
        )
        pf_blocked_generators(gens)
    drift = pf_drift(j, s.base, prices) if expected == "capm" else None
    pf_paths(key, j, value, s, vol, gen, gens.adjusted, warns, drift)
    if gen is not None:  # a generator drew the paths: no density check
        return
    f = None if vol is None else vol.f
    chosen = ("resample",) if vol is None else ("fhs", *vol.key)
    if drift is not None:
        chosen = (*chosen, "capm", *drift.key)
    with st.container(border=True):
        pf_density(
            cached_density(
                (*key, "density", *chosen), j, f, None if drift is None else drift.daily
            ),
            fhs=fhs,
            capm=drift is not None,
        )


# ---- stress


def pf_scenario_table(
    rows: list[portfolio.Replayed], base: portfolio.Currency, value: float
) -> None:
    """``value``: the holdings' worth, which decides B/M amounts."""
    compact = pf_compact(base, value)

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
                f"{t} β{pf_ratio(b)}"
                for t, b in zip(r.proxied, r.proxy_betas, strict=False)
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
            "최대 하락": [portfolio.percent(r.worst, signed=True) for r in rows],
            f"최대 하락 ({base})": [
                pf_money(r.pnl, base, compact=compact, signed=True) for r in rows
            ],
            "구간 끝": [portfolio.percent(r.change, signed=True) for r in rows],
            "환율 배율": [portfolio.percent(r.fx, signed=True) for r in rows],
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

    compact = pf_compact(base, total)
    with st.container(border=True):
        headed(
            f"**가상 충격** · {holding}",
            "**정의** 지수·환율 가상 충격 시 보유분 평가액 변화 · 종목 등락 = 자기"
            " 지수(한국 KOSPI, 그 밖 S&P 500) 등락 × 현지 통화 베타, 그 뒤 환율"
            " 반영\n\n**기준** 베타: Welch, 1 쪽으로"
            f" {quant.BETA_PRIOR_DAYS}/(n + {quant.BETA_PRIOR_DAYS}) 축소(n: Welch"
            " 가중치의 유효 표본 수 (Σw)²/Σw², 6M 약 117·1Y 약 193·3Y 약 255일) · "
            + PF_DECAY_NOTE,
        )
        pf_table(
            {
                "충격": [s.name for s, _ in rows],
                "가정": [moves(s) for s, _ in rows],
                holding: [portfolio.percent(v, signed=True) for _, v in rows],
                f"평가액 변화 ({base})": [
                    pf_money(v * total, base, compact=compact, signed=True)
                    for _, v in rows
                ],
            },
            "가상 충격",
        )
    fall = portfolio.reverse(values, betas)
    with st.container(horizontal=True, key="pf_reverse_cards"):  # outside the box
        st.metric(
            f"−{portfolio.REVERSE_LOSS:.0%} 손실에 필요한 지수 하락폭",
            portfolio.percent(fall) if fall is not None else "−100%에서도 미도달",
            help="**정의** KOSPI·S&P 500 동일 폭 하락 시 보유분"
            f" −{portfolio.REVERSE_LOSS:.0%} 손실에 필요한 지수 하락폭 · 종목 등락 ="
            " 지수 등락 × 베타\n\n**해석** 절댓값이 작을수록 지수 하락에 민감한"
            " 보유분\n\n**기준** 환율 고정 · 베타: 가상 충격과 같은 값",
            border=True,
            height="stretch",
        )


def pf_turbulence(s: Subject, prices: portfolio.Prices, names: tuple[str, ...]) -> None:
    now = cached_turbulence((*s.key, "turbulence"), s.base, names, prices.adjusted)
    if now is None:
        headed(
            "**이번 주 이례 정도** · 보유 종목 주간 수익률",
            "**정의** 보유 종목 주간 수익률의 Mahalanobis 거리 ÷ 종목"
            " 수(Kritzman–Li 2010)",
            [
                f"**조건** 보유 종목 2개 이상 · 모두 값이 있는 주 {portfolio.WEEKS}주"
                f" 초과 · 현재 {len(names)}종목"
            ],
        )
        return
    with st.container(horizontal=True, key="pf_turbulence_cards"):
        st.metric(
            "이번 주 이례 정도",
            f"{now.level:.2f}",
            f"과거 {now.weeks:,}주 중 {portfolio.percent(now.rank, dec=0)} 수준 이하",
            delta_color="off",
            delta_arrow="off",
            help="**정의** 보유 종목 주간 수익률의 Mahalanobis 거리 ÷ 종목"
            " 수(Kritzman–Li 2010) · 평균·공분산: 직전"
            f" {portfolio.WEEKS}주(Ledoit–Wolf 축소),"
            f" {portfolio.TURBULENCE_REFIT}주마다 재추정\n\n**해석** 1이면 평소"
            " 수준 · 클수록 이례적인 주간 움직임 · 아래"
            f" 값: 과거 주 중 이하 비율\n\n**기준** {now.week} 주 · 진행 중인 주 포함",
            border=True,
            height="stretch",
        )
        st.metric(
            "상관 이례도",
            f"{now.surprise:.2f}",
            f"백분위 {portfolio.percent(now.surprise_rank, dec=0)}"
            if math.isfinite(now.surprise_rank)
            else None,
            delta_color="off",
            delta_arrow="off",
            help="**정의** 이례 정도 ÷ 상관 무시(대각 공분산) 부분 · 종목 간 상관"
            " 구조만의 이례 정도\n\n**해석** 1이면 평소 상관 · 1 초과면 평소와 다른"
            " 동조 방식 · 아래 값: 과거 주 중 백분위",
            border=True,
            height="stretch",
        )


def pf_stress_section(s: Subject, prices: portfolio.Prices, months: int) -> None:
    values = {str(t): float(v) * s.value for t, v in s.w.items()}
    if not values:
        st.info("가격이 확인된 보유 종목이 없습니다.")
        return
    older = pf_dashboard_series(PF_DEXKOUS)
    fx = portfolio.long_fx(prices.adjusted, older)
    key = (*s.key, "replay", len(older))
    rows = cached_replays(key, s.base, values, prices.adjusted, fx)
    notes = [
        "지수×베타: 첫날 가격 없는 종목(구간 안 상장 포함)은 구간 내내 지수 등락 ×"
        " 베타, 환율은 종목 자신의 통화 · " + PF_PROXY_NOTE
    ]
    if any(r.window is None for r in rows):
        notes.append(
            f"지수 자료 없음: 가격 갱신 때 {portfolio.LONG_START.year}년부터의"
            " 지수·환율 이력 1회 수신"
        )
    if any(r.no_fx for r in rows):
        notes.append("환율 미반영: 2003년 이전 원/달러는 FRED DEXKOUS(키 필요)")
    notes.extend(s.notes)
    names = tuple(portfolio.labels(pd.Series(values)))
    bad = cached_bad_prints((*s.key, "bad"), names, prices.adjusted)
    listed = ", ".join(
        f"{t} ({', '.join(str(d) for d in days[:3])})" for t, days in bad.items()
    )
    with st.container(border=True):
        headed(
            f"**과거 구간 재현** · {s.holding}",
            "**정의** 지금 보유분을 과거 위기 구간 첫날부터 그대로 보유한 결과 ·"
            f" 구간: 지수별 기록 고점·저점 ±{quant.SEARCH_DAYS}일 안에서 찾은 고점 중"
            " 이른 날 → 저점 중 늦은 날\n\n**해석** 최대 하락: 구간 안 고점 대비 가장"
            " 깊은 하락, 금액은 지금 평가액 × 그 비율 · 구간 끝: 첫날 대비 마지막"
            " 날(반등 시 최대 하락보다 작은 값) · 환율 배율: (1 + 구간 끝)/(1 + 현지"
            " 통화 기준) − 1, 곱해지는 몫 · S&P 500·KOSPI: 같은 구간 지수 자체"
            " 등락(현지 통화)\n\n**기준** " + " · ".join(notes),
            [f"**자료** 튄 뒤 바로 돌아온 가격 · {listed}"] if bad else [],
        )
        pf_scenario_table(rows, s.base, s.value)
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


# ---- one stock, and adding it to a portfolio

STOCK_SECTIONS = {
    "analysis": "위험지표",
    "tail": "꼬리위험",
    "outlook": "전망",
    "stress": "스트레스",
    "whatif": "편입 효과",
}
STOCK_INDICES = (
    portfolio.FX,
    portfolio.BENCHMARK,
    portfolio.TOTAL_RETURN,
    portfolio.KOSPI,
    portfolio.RF_USD,
)
WHATIF_PERCENT = (1, 30, 5)  # slider: least, most, first shown


def stock_choose(ticker: str) -> None:
    state.stock_error = ""
    if ticker != state.get("stock_ticker"):
        state.stock_ticker = ticker


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
    state.stock_retry = True
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
    fetcher: portfolio.Fetcher,
    ticker: str,
    prices: portfolio.Prices,
    *,
    retry: bool,
) -> None:
    """A ticker without saved prices, with the indices, fetched once; then rerun.

    ``retry``: 분석 was pressed, so a ticker an earlier fetch missed is tried again.
    """
    todo = fetcher.untried([ticker, *STOCK_INDICES], prices.tickers)
    if retry and ticker not in todo:
        todo = [ticker, *todo]
    if ticker not in todo:
        return
    with st.spinner(f"{ticker} 가격 받는 중…"):
        try:
            saved = fetcher.fetch_now(todo)
        except Exception as error:  # noqa: BLE001 - shown on the page
            state.stock_error = f"가격을 받지 못했습니다: {type(error).__name__}"
        else:
            if ticker not in saved.tickers:
                state.stock_error = f"{ticker} 가격을 받지 못했습니다."
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
    index = portfolio.benchmark_of(ticker)
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
    b = portfolio.betas(j)[0]
    left, _ = st.columns(2)
    with left, st.container(border=True):
        headed(f"**베타** · {s.base} 일간 수익률", PF_BETA_HELP)
        pf_table(
            {
                f"{portfolio.BENCHMARK_LABEL} (Welch)": [pf_ratio(b.welch)],
                "Dimson": [pf_ratio(b.dimson) if math.isfinite(b.dimson) else PF_DASH],
                "KOSPI": [pf_ratio(b.kospi) if math.isfinite(b.kospi) else PF_DASH],
            },
            "베타",
        )


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
        height="stretch",
    )


def whatif_risk(a: portfolio.Added, ticker: str) -> None:
    vol, beta, ratio, es = a.vol, a.beta, a.ratio, a.es
    with st.container(horizontal=True, key="pf_whatif_cards"):
        whatif_metric(
            "연 변동성",
            vol,
            portfolio.percent,
            portfolio.points(vol[1] - vol[0]),
            "**정의** 지금 비중 → 추가 후 비중의 연 변동성 · √(w′Σw × 연간 관측일"
            " 수)\n\n**기준** " + PF_COV_NOTE,
        )
        whatif_metric(
            "베타",
            beta,
            pf_ratio,
            stock_signed(beta[1] - beta[0]),
            "**정의** 비중 가중 베타 Σ w × β · β:"
            f" {portfolio.BENCHMARK_LABEL} 가격지수 대비, 한국 종목은 Dimson, 그 밖"
            " Welch\n\n**해석** 1 초과 지수보다 큰 변동\n\n**기준** " + PF_DECAY_NOTE,
        )
        whatif_metric(
            "분산 비율",
            ratio,
            pf_ratio,
            stock_signed(ratio[1] - ratio[0]),
            "**정의** Σ 비중 × 종목 변동성 ÷ 포트폴리오 변동성\n\n**해석** 높을수록"
            " 분산 효과 확대 · 1이면 분산 효과 없음\n\n**기준** " + PF_COV_NOTE,
        )
        if es is None:
            st.metric(
                "ES 97.5% (1일)",
                PF_THIN,
                help=f"**정의** 1일 ES {quant.TAIL_LEVEL:.1%}\n\n**기준**"
                f" {quant.min_samples(quant.TAIL_LEVEL):,}일 이상(기간 3Y)",
                border=True,
                height="stretch",
            )
        else:
            whatif_metric(
                "ES 97.5% (1일)",
                es,
                stock_es,
                portfolio.points(es[1] - es[0]),
                f"**정의** 1일 ES {quant.TAIL_LEVEL:.1%}: 기간 중 가장 나쁜"
                f" {1 - quant.TAIL_LEVEL:.1%} 날의 평균 손실 · 아래 값: 증분 ES(같은 날"
                " 수익률로 전체 재평가)\n\n**해석** 증분 ES 음수면 꼬리 손실 완화",
            )
        st.metric(
            f"추가 후 ES 중 {ticker} 몫",
            portfolio.percent(a.es_share) if a.es_share is not None else PF_THIN,
            help="**정의** 추가 후 포트폴리오가 가장 나빴던"
            f" {1 - quant.TAIL_LEVEL:.1%} 날의 손실 중 이 종목 몫 · Tasche 성분 ES ÷"
            " ES\n\n**해석** 추가 비중보다 크면 꼬리 손실 집중",
            border=True,
            height="stretch",
        )
        st.metric(
            "포트폴리오와 상관",
            pf_ratio(a.correlation),
            f"기준 {pf_ratio(a.threshold)}",
            delta_color="off",
            delta_arrow="off",
            help="**정의** 후보 종목과 지금 포트폴리오 일간 수익률의 상관계수 · 기준:"
            " 포트폴리오 변동성 ÷ 후보 변동성\n\n**해석** 기준보다 낮으면 소량 추가 시"
            " 변동성 감소 · 높으면 증가\n\n**기준** " + PF_COV_NOTE,
            border=True,
            height="stretch",
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
    ten = perf.PATH_LEVELS[0]
    falls = (before.paths.falls, after.paths.falls)
    pairs = [
        (f"{label} 뒤 10% 이상 손실 경로 비율", falls[0][ten].end, falls[1][ten].end),
        *(
            (f"기간 중 −{level:.0%} 낙폭", reach.swing, falls[1][level].swing)
            for level, reach in falls[0].items()
        ),
    ]
    mid = (before.fan[-1]["p50"] - 1, after.fan[-1]["p50"] - 1)
    headed(
        f"**몬테카를로** · {label}",
        "**정의** 추가 전 → 후 모의 경로 지표 · 손실 경로 비율 · 기간 중 낙폭"
        " 경로 비율 · 기간 끝 중앙값 수익률\n\n**기준** 평균 수익 0 가정 · 추가 전후"
        f" 같은 재표본 경로 {portfolio.PATHS:,}개, 평균 {block:.1f}일 묶음 · 매일 같은"
        " 비중 재조정",
    )
    with st.container(horizontal=True, key="pf_whatif_path_cards"):
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
            return f"지수×베타로 대체: {ticker} β{pf_ratio(r.proxy_betas[at])}"
        return f"지수×베타로 대체: {ticker}"

    def ends(rows: list[portfolio.Replayed]) -> list[str]:
        return [portfolio.percent(r.change, signed=True) for r in rows]

    def lows(rows: list[portfolio.Replayed]) -> list[str]:
        return [portfolio.percent(r.worst, signed=True) for r in rows]

    pf_table(
        {
            "시나리오": [r.scenario.name for r in before],
            "구간": [
                f"{r.window[0]} → {r.window[1]}" if r.window else PF_DASH
                for r in before
            ],
            "최대 하락": lows(before),
            "최대 하락 · 추가 후": lows(after),
            "구간 끝": ends(before),
            "구간 끝 · 추가 후": ends(after),
            "비고": [note(r) for r in after],
        },
        "스트레스 추가 전후",
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
            help="**정의** 지금 평가액 대비 후보 종목 매수 비중 · 기존 보유분 비례"
            " 축소",
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
        f"{ticker}에 평가액의 {fraction:.0%} ({portfolio.money(amount, p.base)}, 약"
        f" {amount / stock_share(ticker, p.base, prices):,.1f}주) 추가"
    ]
    if ticker in s.w.index:
        held = float(cast("float", s.w[ticker]))
        mixed = (1 - fraction) * held + fraction
        notes.append(
            f"이미 보유 중: 비중 {portfolio.percent(held)} → {portfolio.percent(mixed)}"
        )
    basis = [
        f"{len(j.window)}일 · 모든 종목 값이 있는 날, {p.base} 일간 수익률",
        *s.notes,
    ]
    headed(
        "**위험 지표** · " + " · ".join(notes),
        "**정의** 추가 후 비중 · 기존 보유분: 지금 비중 × (1 − 추가 비중) · 이미"
        " 보유한 종목: (1 − 추가 비중) × 지금 비중 + 추가 비중\n\n**기준** "
        + " · ".join(basis),
        [f"**자료** 기간 중 자료 시작 · {j.late}"] if j.late else [],
    )
    cov, _, _ = cached_covariance((*key, "cov"), j)
    whatif_risk(cached_added((*key, "added"), fraction, ticker, j, cov), ticker)
    whatif_paths(key, j, ticker, fraction, state.get("whatif_horizon") or "1M")
    with st.container(border=True):
        headed(
            "**과거 구간 재현** · 지금 보유분과 추가 후",
            "**정의** 지금 보유분과 추가 후 보유분의 과거 위기 구간 재현 · 스트레스"
            " 탭과 같은 방식\n\n**기준** 자료 없는 구간: KOSPI(한국) 또는 S&P 500 일간"
            " 등락 × 베타 · " + PF_PROXY_NOTE,
        )
        whatif_stress(s, prices, j, ticker, fraction)


def stock_section(  # noqa: PLR0913 - the view's inputs
    section: str,
    ticker: str,
    book: portfolio.Book,
    prices: portfolio.Prices,
    months: int,
    row: DeltaGenerator,
) -> None:
    if section == "whatif":
        stock_whatif(ticker, book, prices, months)
        return
    s = stock_subject(ticker, portfolio.currency(ticker), prices)
    if len(s.rets) < 2:  # noqa: PLR2004
        pf_no_prices()
    elif section == "analysis":
        pf_stats(s, prices, months)
        if portfolio.korean(ticker):
            stock_betas(s, prices, months)
        pf_style_box(s, prices, "stock_style")
    elif section == "tail":
        pf_tail(s, prices, months, row)
    elif section == "outlook":
        pf_outlook(s, prices, months)
    else:
        pf_stress_section(s, prices, months)


def stock_view(
    book: portfolio.Book,
    prices: portfolio.Prices,
    fetcher: portfolio.Fetcher,
    missing: list[str],
) -> None:
    owned = stock_owned(book, prices)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.text_input("티커", key="stock_input", width=170, on_change=stock_enter)
        st.button("분석", type="primary", key="stock_go", on_click=stock_enter)
        if owned:
            st.pills("보유 종목", owned, key="stock_held", on_change=stock_held)
    if state.get("stock_error"):
        st.error(state.stock_error)
    ticker: str | None = state.get("stock_ticker")
    blocked = args.demo or bool(missing)
    retry = bool(state.pop("stock_retry", False))
    if ticker and ticker not in prices.tickers and not blocked:
        stock_fetch(fetcher, ticker, prices, retry=retry)
    pf_bar(
        fetcher, pf_root(), book, missing, [ticker] if ticker else [], prices.tickers
    )
    if not ticker:
        if not state.get("stock_error"):
            st.info("분석할 종목이 없습니다. 티커를 입력하세요.")
        return
    if ticker not in prices.tickers:
        if not state.get("stock_error"):  # the error above already says why
            st.info(f"{ticker}의 저장된 가격이 없습니다.")
        return
    if not blocked:
        fetcher.deepen(prices)  # stress windows back to 1987, once
    headed(f"**{ticker}**")  # no name here: the book and prices carry tickers only
    section = st.segmented_control(
        "보기",
        list(STOCK_SECTIONS),
        format_func=STOCK_SECTIONS.__getitem__,
        default="analysis",
        required=True,
        key="stock_section",
        label_visibility="collapsed",
    )
    horizons = {"outlook": "pf_horizon", "whatif": "whatif_horizon"}
    row = st.container(horizontal=True, vertical_alignment="bottom")
    with row:
        months = pf_periods(horizons.get(section or "analysis"))
    stock_section(section or "analysis", ticker, book, prices, months, row)


# ---- famous investors (SEC 13F)

PF_MODES = {"mine": "내 포트폴리오", "stock": "종목 분석", "investors": "유명 투자자"}
INV_SECTIONS = {
    "returns": "수익률",
    "analysis": "위험지표",
    "correlation": "상관·분산",
    "tail": "꼬리위험",
    "outlook": "전망",
    "stress": "스트레스",
    "compare": "비교",
}
INV_VARIANTS = {
    "filed": "제출일 복제",
    "top": f"상위 {investors.CONVICTION}종목 복제",
}
INV_THEORY = "이론값 (분기말)"
INV_COVER_NOTE = (
    f"편입 범위: 보고 가치 상위 {investors.COVER_SHARE:.0%} 또는"
    f" {investors.COVER_MAX}종목까지 중 티커·가격 확인 종목"
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
            help="**정의** SEC 13F 보고 기관투자자 · 미국 상장 주식 롱 포지션\n\n"
            "**기준** 분기 말 보유 · 분기 종료 후"
            f" {investors.DEADLINE_DAYS}일 안 제출 · 옵션·원금 단위(PRN) 행 제외",
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
                f"13F {filer.fetched:%Y-%m-%d %H:%M} 확인"
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
        if blocked and not args.demo:
            st.info(blocked)

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
    wanted = [
        *names,
        portfolio.BENCHMARK,
        portfolio.TOTAL_RETURN,
        portfolio.FX,
        portfolio.RF_USD,
    ]
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
            y=alt.Y("weight:Q", title=None, axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("period:N", title="분기 말"),
                alt.Tooltip("weight:Q", title="비중", format=".1%"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )


def inv_stock_body(
    found: list[investors.Quarter], row: dict[str, object], prices: portfolio.Prices
) -> None:
    st.caption(f"{row['sector']} · CUSIP {row['cusip']}")
    with st.container(horizontal=True, key="inv_stock_cards"):
        weight = cast("float", row["weight"])
        for label, value in (
            ("가치", investors.dollars(cast("float", row["value"]))),
            ("비중", portfolio.percent(weight) if math.isfinite(weight) else PF_DASH),
            ("주식 수", portfolio.shares(cast("float", row["shares"]))),
            ("직전 분기 대비", str(row["change"])),
        ):
            st.metric(label, value, border=True, height="stretch")
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(
            "**분기별 비중** · 보고 가치 기준",
            "**정의** 분기별 보고 가치 중 이 종목 비중 · 미보유 분기 0",
        )
        history = investors.weight_history(found, str(row["cusip"]))
        st.altair_chart(inv_weight_history(list(history)), alt="분기별 비중")
    with right, st.container(border=True, height="stretch"):
        headed("**1년 주가** · 종가, USD")
        ticker = str(row["ticker"])
        if ticker in prices.close.columns:
            series = portfolio.window(
                cast("pd.Series", prices.close[ticker]).dropna(), 12
            )
            points = portfolio.line_points({"종가": series})
            st.altair_chart(pf_line_chart(points, ",.2f"), alt=f"{ticker} 1년 주가")
        else:
            st.caption("저장된 가격이 없습니다.")


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
    warns: list[str],
) -> None:
    """현황: the reported quarter, always shown above the analysis sections.

    ``warns``: ``inv_flags``, the "!" beside the heading.
    """
    periods = [q.period for q in found]
    labels = {q.period: investors.quarter_label(q) for q in found}
    if state.get("inv_quarter") not in periods:
        state.inv_quarter = periods[0]
    at = periods.index(state.inv_quarter)
    q = found[at]
    before = found[at + 1] if at + 1 < len(found) else None
    frame = investors.holdings_frame(q, before, cusips, sectors)
    priced = float(cast("float", inv_priced(q, cusips, prices).sum()))
    excluded = investors.excluded_share(q.rows)
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
    with st.container(horizontal=True, vertical_alignment="center"):
        headed(
            "**현황**",
            "**정의** 고른 분기 13F 보고 종목의 가치·비중·직전 분기 대비 변화 · USD"
            "\n\n**기준** " + " · ".join(notes),
            warns,
        )
        st.space("stretch")
        st.selectbox(
            "분기",
            periods,
            format_func=labels.__getitem__,
            key="inv_quarter",
            width=360,
            label_visibility="collapsed",
        )
    with st.container(horizontal=True, key="inv_now_cards"):
        st.metric(
            "보고 가치",
            investors.dollars(q.value),
            help="**정의** 해당 분기 13F 보고 가치 합계(USD) · 옵션·PRN 제외",
            border=True,
            height="stretch",
        )
        st.metric(
            "종목 수",
            f"{len(q.rows):,}",
            help="**정의** 보고 종목(CUSIP) 수 · 옵션·PRN 제외",
            border=True,
            height="stretch",
        )
        st.metric(
            "상위 10 비중",
            portfolio.percent(investors.top_share(q.rows)),
            help="**정의** 보고 가치 상위 10종목의 비중 합\n\n**해석** 높을수록 소수"
            " 종목 집중",
            border=True,
            height="stretch",
        )
    records = cast("list[dict[str, object]]", frame.to_dict("records"))
    left, right = st.columns([1, 2])
    with left, st.container(border=True, height="stretch"):
        headed(
            "**비중** · 보고 가치 기준",
            "**정의** 보고 가치 기준 종목 비중\n\n**기준**"
            f" {PF_DONUT_SLICES}종목 초과 시 상위 {PF_DONUT_SLICES - 1}종목과 기타",
        )
        rows = [
            (
                str(r["ticker"]) or str(r["name"]),
                cast("float", r["weight"]),
                cast("float", r["value"]),
            )
            for r in records
            if math.isfinite(cast("float", r["weight"]))
        ]
        st.altair_chart(pf_donut(rows, investors.dollars, "가치"), alt="비중")
    with right, st.container(border=True, height="stretch"):
        headed(
            "**섹터 비중** · 보고 가치 기준",
            "**정의** 보고 가치 기준 섹터별 비중 합 · 섹터: yfinance\n\n**기준**"
            f" {investors.EXCLUDED}: 상위 보유 범위(가치 {investors.COVER_SHARE:.0%}"
            f" 또는 {investors.COVER_MAX}종목) 밖 · {investors.UNKNOWN}: 섹터 미수신",
        )
        st.altair_chart(
            pf_weight_chart(investors.sector_weights(frame), "섹터"), alt="섹터 비중"
        )
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
    with st.container(border=True):
        headed("**보유 종목** · USD")
        event = st.dataframe(
            shown.style.map(inv_style, subset=["직전 분기 대비"]),
            hide_index=True,
            height=LIST_HEIGHT,
            on_select="rerun",
            selection_mode="single-cell",
            key=f"inv_table_{state.get('inv_gen', 0)}",
            column_config={
                "종목": st.column_config.TextColumn(width="medium"),
                "직전 분기 대비": st.column_config.TextColumn(
                    help="**정의** 직전 분기 대비 주식 수 변화 · "
                    + " · ".join(investors.CHANGES)
                ),
            },
            alt="보유 종목",
        )
    gone = investors.sold_out(q, before)
    with st.container(border=True):
        headed(
            f"**청산** · 직전 분기 대비 {len(gone)}종목",
            "**정의** 직전 분기 보유 중 이번 분기 보고에 없는 종목",
        )
        if gone:
            pf_table(
                {
                    "종목": [r.name for r in gone],
                    "직전 주식 수": [portfolio.shares(r.shares) for r in gone],
                    "직전 가치": [investors.dollars(r.value) for r in gone],
                },
                "청산 종목",
            )
        else:
            st.caption("직전 분기 대비 청산한 종목이 없습니다.")
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


def inv_controls(section: str) -> tuple[int, DeltaGenerator]:
    """추정·전망 기간 (not on 수익률) and the clone: the window in months, the row."""
    row = st.container(horizontal=True, vertical_alignment="bottom")
    with row:
        months = (
            pf_periods("pf_horizon" if section == "outlook" else None)
            if section != "returns"
            else portfolio.WINDOWS["3Y"]
        )
        picked = st.segmented_control(
            "복제",
            list(INV_VARIANTS),
            format_func=INV_VARIANTS.__getitem__,
            default="filed",
            required=True,
            key="inv_variant",
            help="**정의** 13F 보고 비중을 따라 산 가상 포트폴리오 · 제출일 복제: 제출"
            " 공개 후 첫 종가(뉴욕 16시 이후·접수 시각 없는 제출은 다음 거래일)에 보고"
            " 비중 매수 · 상위"
            f" {investors.CONVICTION}종목 복제: 보고 비중 상위"
            f" {investors.CONVICTION}종목\n\n**기준** 다음 제출까지 보유 · "
            + INV_COVER_NOTE
            + " · 편입 종목 비중 합 100%로 재조정",
        )
        if picked == "top":
            st.toggle(
                "같은 비중",
                key="inv_equal",
                help=f"**정의** 상위 {investors.CONVICTION}종목을 보고 비중 대신 같은"
                " 비중으로 매수",
            )
    return months, row


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
    time = portfolio.time_weighted(filed)
    with st.container(horizontal=True, key="inv_return_cards"):
        st.metric(
            f"누적 · {label}",
            portfolio.percent(portfolio.last(filed), signed=True),
            pf_annual(time) if time and time.annual is not None else None,
            delta_color="off",
            delta_arrow="off",
            help=f"**정의** {label} 누적 수익률 · {start.date()}부터 · USD\n\n"
            "**해석** 아래 값: 1년 이상이면 연환산\n\n"
            "**기준** 수정 종가 · 제출마다 새 비중 매수 뒤 다음 제출까지 보유",
            border=True,
            height="stretch",
        )
        st.metric(
            "종목 수",
            f"{len(made.weights):,}",
            help="**정의** 복제가 지금 보유 중인 종목 수 · 가격이 이어지는 종목",
            border=True,
            height="stretch",
        )
        st.metric(
            "가격 연결 비중",
            portfolio.percent(
                float(cast("float", inv_priced(q, cusips, prices).sum()))
            ),
            help=f"**정의** {q.period} 보고 가치 중 가격으로 계산한 종목의 몫\n\n"
            "**기준** " + INV_COVER_NOTE + " · 보고 가치 전체 대비",
            border=True,
            height="stretch",
        )
        st.metric(
            "분기말 이후 경과일",
            f"{(today - q.period).days:,}일",
            help=f"**정의** 최근 보고 분기 말({q.period})부터 오늘까지 달력일\n\n"
            "**해석** 길수록 오래된 보유 정보\n\n**기준** 분기 말에서 제출까지"
            f" {(q.filed - q.period).days}일",
            border=True,
            height="stretch",
        )
        st.metric(
            "평균 회전율",
            portfolio.percent(sum(turns) / len(turns)) if turns else PF_DASH,
            help="**정의** 리밸런싱별 편도 회전율의 평균 · ½Σ|새 비중 − 직전 보유의"
            " 가격 변동 반영 비중|\n\n**해석** 높을수록 분기마다 큰 비중 교체",
            border=True,
            height="stretch",
        )
    gap = portfolio.points(portfolio.last(filed) - portfolio.last(theory))
    st.caption(
        f"{INV_THEORY} 누적"
        f" {portfolio.percent(portfolio.last(theory), signed=True)} · 차이 {gap}",
        help=f"**정의** {INV_THEORY}: 분기 말 종가에 보고 비중으로 매수한 경우의 누적"
        " 수익률 · 차이: 복제 − 이론값\n\n**해석** 차이 음수: 제출 지연 동안의 손실",
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
    missing = investors.untimed(filer)
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(
            f"**누적 수익률** · USD, {start.date()}부터",
            f"**정의** 복제·{portfolio.BENCHMARK_LABEL}·내 포트폴리오 TWR의 누적"
            f" 수익률 · USD · 회색 점선: {INV_THEORY}\n\n**기준**"
            f" {portfolio.BENCHMARK_LABEL}: 총수익지수({portfolio.TOTAL_RETURN}) · 내"
            " 포트폴리오: TWR의 일별 USD/KRW 환산"
            + (
                f" · 접수 시각 없는 제출 {missing}건: 다음 거래일 종가 매수"
                if missing
                else ""
            ),
        )
        st.altair_chart(inv_chart(lines, theory), alt="복제 수익률")
    with right, st.container(border=True, height="stretch"):
        headed(
            f"**리밸런싱** · {len(made.steps)}회",
            "**정의** 제출별 매수일 · 종목 수 · 회전율\n\n**기준** 회전율: ½Σ|새"
            " 비중 − 직전 보유의 가격 변동 반영 비중|(편도) · 첫 매수·현금 보유 뒤"
            " 매수 —",
        )
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
    pf_compare_heatmap(
        (*keys, "USD", months, "investors"),
        frame,
        korean,
        f"**투자자·내 포트폴리오·{portfolio.BENCHMARK_LABEL}** · USD",
        "**정의** 투자자 복제·내 포트폴리오·지수의 일간 수익률 상관계수 · USD\n\n"
        "**해석** 1에 가까울수록 같은 방향 · 0 근처 무관\n\n**기준** 투자자: 최근 분기"
        " 보고 비중 고정 · 내 포트폴리오: 지금 비중 · " + PF_INDEX_NOTE + " · 한국"
        " 비중 절반 초과 포트폴리오: 한국 종가 기준 시차 보정 · 셋 이상:"
        " Ledoit–Wolf 축소",
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
    title = f"**투자자 비교** · {label}, USD, {state.get('pf_window') or '3Y'}"
    about = (
        "**정의** 투자자별 복제의 기간 수익률·연 샤프 지수 · 축소 샤프: 경험적"
        " 베이즈 축소, 평균 + τ²/(τ² + SE²) × (값 − 평균) · τ² = max(0, 샤프 분산 − SE²"
        " 평균) · SE: Mertens 표준오차\n\n**해석** 표준오차가 클수록 평균 쪽으로 큰"
        " 축소 · 축소 샤프: 운의 영향을 줄인 비교\n\n**기준** 일간 초과수익률로"
        " 계산 후 연환산 · 일수: 무위험수익률이 있는 날"
    )
    if not series:
        headed(title, about, ["**조건** 가격이 있는 투자자 복제 1개 이상"])
        return
    days = (
        pd.DatetimeIndex(pd.concat(list(series.values())).index).unique().sort_values()
    )
    rates, source, quote = pf_rates("USD", prices.close)
    rf = portfolio.risk_free(rates, days, quote)
    key = (*keys, top, equal, months, source, pf_rates_key(rates))
    found = cached_peers(key, series, rf)
    pooled = found.count > 1
    notes = ["무위험수익률 " + (source if not rates.empty else "0")]
    if pooled:
        dsr = (
            portfolio.percent(found.deflated)
            if found.deflated is not None
            else f"{PF_SHORT} ({perf.MIN_OBS}일 미만)"
        )
        notes.append(f"{found.count}명 중 최고값의 운 이상 확률(DSR) {dsr}")
    st.markdown(title, help=about + " · " + " · ".join(notes))
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
    with st.container(border=True):
        inv_league(people, prices, months, top=top, equal=equal)


# ---- compare


def inv_ratios(
    key: Key, rets: pd.Series, prices: portfolio.Prices, months: int
) -> portfolio.Ratios:
    """``key`` stands for ``rets``; over the S&P 500 in dollars."""
    window = portfolio.window(rets, months)
    rates, source, quote = pf_rates("USD", prices.close)
    rf = portfolio.risk_free(rates, pd.DatetimeIndex(window.index), quote)
    bench = partial(inv_sp500, prices.adjusted, pd.DatetimeIndex(window.index))
    tag = (*key, months, "sp500", source, pf_rates_key(rates))  # not the blended one
    return cached_ratios(tag, window, rf, bench)


def inv_overlap(theirs: pd.Series, mine: pd.Series) -> None:
    us = portfolio.us_sleeve(mine)
    total = float(cast("float", theirs.sum()))
    their = theirs / total if total > 0 else theirs
    share = 1 - perf.active_share(their, us) if not us.empty else None
    with st.container(horizontal=True, key="inv_overlap_cards"):
        st.metric(
            "보유 겹침",
            portfolio.percent(share) if share is not None else PF_DASH,
            help=(
                "**정의** Σ 종목별 작은 쪽 비중 = 1 − 액티브 셰어\n\n**해석** 1에"
                " 가까울수록 같은 구성 · 0이면 공통 종목 없음\n\n**기준** 액티브 셰어"
                f" {portfolio.percent(1 - share)}"
                if share is not None
                else "**정의** Σ 종목별 작은 쪽 비중\n\n**기준** 내 포트폴리오 미국"
                " 주식 없음"
            ),
            border=True,
            height="stretch",
        )
    shared = investors.overlap(their, us)
    with st.container(border=True):
        headed(
            f"**겹치는 종목** · {len(shared)}종목",
            "**정의** 투자자 보고 비중과 내 미국 주식에 함께 있는 종목 · 각각 합 100%",
        )
        if shared:
            pf_table(
                {
                    "티커": [t for t, _, _ in shared],
                    "투자자 비중": [portfolio.percent(a) for _, a, _ in shared],
                    "내 미국 주식 중": [portfolio.percent(b) for _, _, b in shared],
                },
                "겹치는 종목",
            )
        else:
            st.caption("겹치는 종목이 없습니다.")


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
        st.info("비교할 내 포트폴리오가 없습니다.")
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
    headed(
        "**보유 겹침** · 미국 주식",
        "**정의** 내 미국 주식(지금 평가액)과 투자자 보고 비중의 겹침 · 각각"
        " 합 100% · 겹침 = Σ 종목별 작은 쪽 비중 = 1 − 액티브 셰어\n\n**해석** 1에"
        " 가까울수록 같은 구성 · 0이면 공통 종목 없음\n\n**기준** 투자자"
        f" {found[0].period} 보고 · 티커 확인 종목",
    )
    theirs = investors.weights(found[0].rows, cusips)
    own = portfolio.weights(portfolio.holdings(p, prices.close))
    inv_overlap(theirs, own)
    korean: set[str] = {p.name} if portfolio.korean_weight(own) else set()
    with st.container(border=True):
        headed(
            f"**위험 지표** · USD, {state.get('pf_window') or '3Y'}",
            "**정의** 투자자 복제·내 포트폴리오·지수의 기간 수익률·연 변동성·최대"
            " 낙폭·샤프 지수·베타 · USD\n\n**기준** 투자자: "
            f"{s.label} · 내 포트폴리오: 보유 기록 수익률(비중은 종가, 등락은 수정"
            " 종가)"
            f" · {portfolio.BENCHMARK_LABEL} 행: 총수익지수({portfolio.TOTAL_RETURN}) ·"
            f" 베타: {portfolio.BENCHMARK_LABEL} 가격지수 대비 Welch, "
            + PF_DECAY_NOTE
            + (" · 한국 종목 보유 시 베타 —(종가 시각 차이)" if korean else ""),
        )
        usd = inv_usd(p)
        held = cached_held(
            pf_key(usd.model_dump_json()), usd, prices.close, prices.adjusted
        )
        sp_days = portfolio.trading_days(prices.adjusted, [portfolio.TOTAL_RETURN])
        sp500 = inv_sp500(prices.adjusted, sp_days).dropna()
        rows = {
            s.label: (s.key, s.rets),
            p.name: (pf_key(usd.model_dump_json()), held),
        }
        if not sp500.empty:  # none until ^SP500TR is fetched
            rows[portfolio.BENCHMARK_LABEL] = (pf_key("benchmark"), sp500)
        stats = {
            name: inv_ratios(key, rets, prices, months)
            for name, (key, rets) in rows.items()
        }
        windows = {
            name: portfolio.window(rets, months).dropna().to_numpy(dtype="float64")
            for name, (_, rets) in rows.items()
        }
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
                    else PF_DASH
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


# ---- the view


def inv_flags(filer: investors.Filer | None) -> list[str]:
    """보고 중단 and 13F-NT, when they apply: the 현황 heading's "!"."""
    stopped = investors.stale(filer, datetime.now().astimezone().date())
    late = investors.notice(filer)
    return [
        *([f"**보고 중단** 마지막 보고 분기 {stopped}"] if stopped else []),
        *([f"**최근 분기 보고 없음** {late.form}, {late.filed}"] if late else []),
    ]


def inv_section(  # noqa: PLR0913 - the view's inputs
    section: str,
    key: Key,
    book: portfolio.Book,
    people: list[investors.Investor],
    data: InvData,
    prices: portfolio.Prices,
) -> None:
    months, row = inv_controls(section)
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
    elif section == "tail":
        pf_tail(s, prices, months, row)
    elif section == "outlook":
        pf_outlook(s, prices, months)
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
    found = data[1]
    if not found:
        if not fetcher.progress.running:
            st.info("저장된 13F 자료가 없습니다.")
        return
    hold = bool(blocked) or fetcher.progress.running
    inv_prices(pfetch, hold, cik, investors.tickers(found, data[2]), prices)
    inv_holdings(found, data[2], data[3], prices, inv_flags(data[0]))
    section = st.segmented_control(
        "보기",
        list(INV_SECTIONS),
        format_func=INV_SECTIONS.__getitem__,
        default="returns",
        required=True,
        key="inv_section",
        label_visibility="collapsed",
    )
    key = pf_key(cik, stamp)
    inv_section(section or "returns", key, book, people, data, prices)


# ---- page sections


def pf_section(
    section: str,
    book: portfolio.Book,
    p: portfolio.Portfolio,
    prices: portfolio.Prices,
) -> None:
    if not prices.tickers & {t.ticker for t in p.trades}:
        pf_no_prices()
        return
    if section == "returns":
        pf_returns(p, prices)
        return
    row = st.container(horizontal=True, vertical_alignment="bottom")
    with row:
        months = pf_periods("pf_horizon" if section == "outlook" else None)
    if section == "analysis":
        pf_analysis(p, prices, months)
        return
    s = pf_subject(p, prices)
    if section == "correlation":
        compare = partial(pf_book_heatmap, book, prices, p.base, months)
        pf_correlation(s, prices, months, compare)
    elif section == "stress":
        pf_stress_section(s, prices, months)
    elif section == "tail":
        pf_tail(s, prices, months, row)
    else:
        pf_outlook(s, prices, months)


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


def screen_help(name: str, c: screen.Criteria) -> str:
    """The data line's tooltip, as the criteria in force set it."""
    gate = (
        f"관문: 주가 ${screen.MIN_PRICE:g} 미만 제외"
        if c.on("min_price")
        else "관문 없음"
    )
    rules = [
        gate,
        screen.basis_note(),
        f"정의 기준일 {screen.DEFINITIONS} · 자세한 근거는 docs/screening.md",
    ]
    return (
        f"**정의** {name} 구성 종목의 퀄리티·밸류에이션 z 점수({screen.FTSE})\n\n"
        "**기준** " + " · ".join(rules)
    )


def screening() -> None:
    key = state.setdefault("screen_universe", "sp500") or "sp500"
    universe = screen.UNIVERSES[key]
    criteria = saved_criteria()
    st.segmented_control(
        "유니버스",
        list(screen.UNIVERSES),
        format_func=lambda k: screen.UNIVERSES[k].name,
        required=True,
        key="screen_universe",
        label_visibility="collapsed",
    )
    root = (
        resolve_settings(args.settings).parent / screen.SNAPSHOT_DIR / universe.folder
    )
    screen_bar(
        screen_updater(str(root), key),
        root,
        screen_help(universe.name, criteria),
        criteria,
    )
    if changed := screen.changed(criteria):
        st.caption(
            f"스크리닝 기준 {len(changed)}개를 기본값에서 바꿔 적용 중입니다"
            " · 상단 설정에서 바꿀 수 있습니다"
        )
    try:
        stamp = screen.stamp(root)
        table = screen_table(str(root), stamp, criteria)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"`{screen.SNAPSHOT_DIR}/`의 파일을 읽지 못했습니다: {error}")
        return
    if table is None:
        st.info("아직 받은 데이터가 없습니다.")
        return
    screen_results(root, table, criteria)


def portfolio_page() -> None:
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
    pf_bar(fetcher, root, book, missing, have=prices.tickers)
    if p is None:
        st.info("포트폴리오가 없습니다.")
        return
    pf_holdings(p, prices)
    if not p.trades:
        return
    section = st.segmented_control(
        "보기",
        list(PF_SECTIONS),
        format_func=PF_SECTIONS.__getitem__,
        default="returns",
        required=True,
        key="pf_section",
        label_visibility="collapsed",
    )
    pf_section(section or "returns", book, p, prices)


# ---------------------------------------------------------------- app

with st.container(horizontal=True, vertical_alignment="center"):
    st.title("FinAgent", anchor=False)
    st.space("stretch")
    clock_slot = st.empty()
    open_settings = st.button("설정", icon=":material/tune:", shortcut="S")
    st.button("갱신", icon=":material/refresh:", type="primary", on_click=refresh)

if state.toast:
    st.toast(state.toast)
    state.toast = ""

with st.spinner("공개 API에서 받는 중…"):
    data = rt.call(rt.dashboard.get())

with clock_slot:
    clocks(data.generated_at)

if not args.demo:
    _ = trainer(str(gen_root()))  # resumes or stops a job a restart left behind

if open_settings:
    for v in data.views:
        state[f"view_{v.id}"] = v.id in data.deep
    seed_criteria()
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

st.html(CARDS_CSS)
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
