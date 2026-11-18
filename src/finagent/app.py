# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false

from __future__ import annotations

import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import streamlit as st

from finagent import etf, fundamentals, portfolio, settings
from finagent.analysis.briefing import Briefing
from finagent.analysis.service import AnalysisService
from finagent.cli import build_service, configure_sources, parse_args, resolve_settings
from finagent.dashboard import SOURCE_NAMES
from finagent.demo import book as demo_book
from finagent.demo import etf as demo_etf
from finagent.prices import current as price_store
from finagent.runtime import Runtime
from finagent.ui import economy as ui_economy
from finagent.ui import etf_page, palette, shared
from finagent.ui import finance as ui_finance
from finagent.ui import monetary as ui_monetary
from finagent.ui import overview as ui_overview
from finagent.ui import schedule as ui_schedule
from finagent.ui.ai_panel import (
    View,
    agent_page,
    ai_settings,
    apply_ai,
    etf_options,
    page_chat,
)
from finagent.ui.context import UiContext
from finagent.ui.pf_common import (
    pf_current,
    pf_look,
    pf_update,
    portfolio_fetcher,
)
from finagent.ui.portfolio_page import portfolio_page
from finagent.ui.shared import (
    env_values,
    etf_confirm,
    etf_held,
    etf_updater,
    limit_note,
    market_keys,
    sec_agent,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from finagent.models import Dashboard

CLOCKS = (("NY", "America/New_York"), ("SEOUL", "Asia/Seoul"), ("UTC", "UTC"))
EXTRA_STATES = {"wait": "대기", "run": "수신 중", "done": "완료"}
DEMO_HELP = "화면 검증 목적 전 지표값 999 고정 데모 모드"
REFRESH_HELP = (
    "현재 화면 데이터 단독 재수신  \nETF 및 포트폴리오 화면은 해당 화면 갱신 재시작"
)
AI_CSS = """<style>
.st-key-ai_fab {position: fixed; right: 1.5rem; bottom: 1.5rem; width: auto;
  z-index: 1000}
.st-key-ai_fab button {width: 3.25rem; height: 3.25rem; padding: 0;
  border-radius: 50%}
.st-key-ai_panel {position: fixed; right: 1.5rem; top: 4rem; bottom: 5.5rem;
  z-index: 1000; width: min(800px, 100vw - 3rem); overflow: hidden;
  display: flex; flex-direction: column; gap: 0.5rem;
  background: Canvas; background: light-dark(#FFFFFF, #0F172A);
  box-shadow: 0 0.5rem 1.5rem rgb(0 0 0 / 0.25);
  transform-origin: bottom right;
  animation: ai-panel-in 240ms cubic-bezier(0.16, 1, 0.3, 1)}
@keyframes ai-panel-in {from {opacity: 0; transform: translateY(0.75rem) scale(0.97)}}
@media (prefers-reduced-motion: reduce) {
  .st-key-ai_panel {animation: ai-panel-fade 120ms ease-out}
  @keyframes ai-panel-fade {from {opacity: 0}}}
.st-key-ai_panel:has(.st-key-ai_wide_on) {width: calc(100vw - 3rem)}
.st-key-ai_page {flex: none; height: calc(100dvh - 12rem); min-height: 28rem;
  width: 100%; max-width: 56rem; margin-inline: auto}
.st-key-ai_panel > div {flex: 1 1 auto; min-height: 0}
.st-key-ai_panel > div > [data-testid="stVerticalBlock"] {flex: 1 1 auto;
  min-height: 0; height: 100%; gap: 0.5rem}
:is(.st-key-ai_panel, .st-key-ai_page) div:has(> .st-key-ai_turns),
:is(.st-key-ai_panel, .st-key-ai_page) div:has(> .st-key-ai_report)
  {flex: 1 1 0; min-height: 0}
:is(.st-key-ai_panel, .st-key-ai_page) .st-key-ai_turns {height: 100%;
  overflow-y: auto; flex-direction: column-reverse; justify-content: flex-start}
.st-key-ai_turns > * {flex-shrink: 0}
.st-key-ai_report {height: 100%; overflow-y: auto; overflow-anchor: none}
.st-key-ai_threads {flex: none; height: calc(100dvh - 12rem); min-height: 28rem;
  overflow-y: auto; overflow-x: hidden}
.st-key-ai_threads > * {flex-shrink: 0}
.st-key-ai_turns, .st-key-ai_report, .st-key-ai_threads {scrollbar-width: thin;
  scrollbar-color: light-dark(#CBD5E1, #334155) transparent}
.st-key-ai_conclusion {padding-bottom: 0.5rem;
  border-bottom: 1px solid light-dark(#CBD5E1, #334155)}
.st-key-ai_conclusion p {font-size: 14px; font-weight: 600}
.st-key-ai_panel h2 {font-size: 18px; font-weight: 600;
  padding: 1.25rem 0 0.25rem}
:is(.st-key-ai_panel, .st-key-ai_page) [data-testid="stChatMessage"] h3
  {font-size: 14px; font-weight: 600; padding: 1rem 0 0.25rem}
:is(.st-key-ai_panel, .st-key-ai_page) [data-testid="stChatMessage"] p
  {max-width: 75ch}
:is(.st-key-ai_panel, .st-key-ai_page) table {width: 100%; font-size: 13px;
  font-variant-numeric: tabular-nums; word-break: keep-all;
  overflow-wrap: anywhere}
:is(.st-key-ai_panel, .st-key-ai_page) th {white-space: nowrap}
.st-key-ai_input_slash {display: none}
.fa-slash {position: fixed; z-index: 1100; padding: 0.25rem;
  border: 1px solid light-dark(#CBD5E1, #334155); border-radius: 0.5rem;
  background: light-dark(#FFFFFF, #1E293B);
  box-shadow: 0 0.5rem 1.5rem rgb(0 0 0 / 0.2)}
.fa-slash-item {display: flex; gap: 0.75rem; align-items: baseline;
  padding: 0.5rem 0.75rem; border-radius: 0.375rem; cursor: pointer;
  font-size: 14px}
.fa-slash-item[aria-selected="true"], .fa-slash-item:hover
  {background: light-dark(#F1F5F9, #334155)}
.fa-slash-item b {font-weight: 600}
.fa-slash-item span, .fa-slash-item small {color: light-dark(#475569, #94A3B8)}
.fa-slash-item small {margin-left: auto}
[data-testid="stChatInputSubmitButton"]:disabled {color: light-dark(#64748B, #94A3B8)}
</style>"""
CARDS_CSS = """<style>
div:has(> [class*="st-key-"][class*="_cards"]) {container-type: inline-size}
[class*="st-key-"][class*="_cards"] {display: grid; grid-auto-rows: 1fr;
  grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr)}
.st-key-etf_stock_cards {grid-auto-flow: row;
  grid-template-columns: repeat(4, minmax(0, 1fr))}
@container (max-width: 1000px) {
  [class*="st-key-"][class*="_cards"] {grid-auto-flow: row;
    grid-template-columns: repeat(2, minmax(0, 1fr))}
  [class*="st-key-"][class*="_cards"]:has(> :nth-child(3):last-child),
  [class*="st-key-"][class*="_cards"]:has(> :nth-child(5))
    {grid-template-columns: repeat(3, minmax(0, 1fr))}
}
h1, h2, h3 {word-break: keep-all; overflow-wrap: break-word}
h3:not(:is(.st-key-ai_panel, .st-key-ai_page) h3) {padding-top: 1.75rem}
div:has(> .st-key-page_heading) {position: absolute; width: 1px; height: 1px;
  overflow: hidden; clip-path: inset(50%)}
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"] * {color: light-dark(#fff, #0F172A)}
button[data-testid="stBaseButton-primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled *
  {color: light-dark(#64748B, #94A3B8)}
[data-testid="stSelectbox"] [role="group"]:not(:focus-within),
[data-testid="stMultiSelect"] [role="group"]:not(:focus-within),
[data-testid="stTextInputRootElement"]:not(:focus-within),
[data-testid="stNumberInputContainer"]:not(:focus-within),
[data-testid="stDateInputField"]:not(:focus-within),
[data-testid="stTextAreaRootElement"]:not(:focus-within)
  {border-color: light-dark(#CBD5E1, #64748B)}
</style>"""


@st.fragment(run_every=3.0)
def theme_watch() -> None:
    theme = palette.kind()
    if state.setdefault("theme_seen", theme) != theme:
        state.theme_seen = theme
        st.rerun(scope="app")


@st.cache_resource(show_spinner=False)
def demo_settings() -> Path:
    return demo_book.settings()


@st.cache_resource(show_spinner=False)
def runtime(demo: bool, env_file: str, settings: str) -> Runtime:
    env = env_values(env_file)
    settings_path = resolve_settings(Path(settings) if settings else None)
    configure_sources(env)
    service = build_service(demo=demo, env=env)
    briefing = Briefing(
        root=settings_path.parent,
        settings_path=settings_path,
        demo=demo,
        env=env,
        dashboard=service,
        held=etf_held,
        confirm=etf_confirm,
        options=etf_options,
        prices=pf_current,
        fetcher=portfolio_fetcher,
        updater=etf_updater,
        look=pf_look,
    )
    return Runtime(
        service,
        AnalysisService(
            service,
            env,
            settings_path=settings_path,
            prepare=briefing.prepare,
            portfolios=briefing.portfolios,
        ),
    )


st.set_page_config(
    page_title="FinAgent", page_icon=":material/monitoring:", layout="wide"
)
args = parse_args(sys.argv[1:])
if args.demo and args.settings is None:
    args.settings = demo_settings()
settings_path = resolve_settings(args.settings)
try:
    settings.check(settings_path)
except settings.SettingsError as error:
    st.error(str(error))
    st.stop()
rt = runtime(args.demo, str(args.env_file or ""), str(args.settings or ""))
root = settings_path.parent
shared.configure(shared.App(args.demo, str(args.env_file or ""), rt, settings_path))
state = st.session_state
state.setdefault("toast", "")
state.setdefault("ai_open", False)
state.setdefault("ai_thread", "")
state.setdefault("ai_wide", False)


def progress_table() -> tuple[str, float, dict[str, list[object]], str]:
    p = rt.dashboard.progress
    if p is None or not p.total:
        return "데이터 수신 중", 0.0, {}, ""
    total, done = p.total, dict(p.done)
    pages = [
        (key, title)
        for _, key, title, _, _ in DASHBOARD_PAGES
        if any(page == key for _, page in total)
    ]
    table: dict[str, list[object]] = {"소스": [], "완료": []}
    table |= {title: [] for _, title in pages}
    for source, name in SOURCE_NAMES:
        cells = [(source, key) for key, _ in pages if (source, key) in total]
        if not cells:
            continue
        got = sum(done.get(cell, 0) for cell in cells)
        table["소스"].append(name)
        table["완료"].append(f"{got}/{sum(total[cell] for cell in cells)}")
        for key, title in pages:
            cell = (source, key)
            table[title].append(
                100 * done.get(cell, 0) / total[cell] if cell in total else None
            )
    elapsed = time.perf_counter() - p.started
    finished = p.series + p.etfs
    label = f"데이터 수신 중 · {finished}/{p.count} · {elapsed:.1f}s"
    extra = f"일정·FOMC·전망: {EXTRA_STATES[p.extra]}"
    return label, finished / p.count if p.count else 0.0, table, extra


def background_lines() -> list[str]:
    if args.demo:
        return []
    updater = etf_updater()
    fetcher = portfolio_fetcher()
    lines = [
        f"- ETF 유니버스: {item.stage}"
        + (f" {item.done}/{item.total}" if item.total else "")
        for item in (updater.progress, updater.slow)
        if item.running
    ]
    lines += [
        f"- 포트폴리오 {name}: {item.message}"
        for name, item in (
            ("가격", fetcher.progress),
            ("ETF 구성", fetcher.look_progress),
            ("규모", fetcher.facts_progress),
        )
        if item.running
    ]
    return ["**백그라운드**", *lines] if lines else []


def show_progress(*, full: bool) -> None:
    label, share, table, extra = progress_table()
    if not full:
        progress_slot.progress(min(share, 1.0), text=label)
        return
    with progress_slot.container(border=True):
        st.progress(min(share, 1.0), text=label)
        main, side = st.columns([4, 1])
        if table:
            config = {
                column: st.column_config.ProgressColumn(
                    column, format="%d%%", min_value=0, max_value=100
                )
                for column in table
                if column not in {"소스", "완료"}
            }
            main.dataframe(  # pyright: ignore[reportUnknownMemberType]
                table, hide_index=True, column_config=config
            )
        side.markdown("\n".join([extra, "", *background_lines()]))


def fetched(
    coro: Coroutine[object, object, Dashboard], *, full: bool = False
) -> Dashboard:
    job = rt.submit(coro)
    try:
        return job.result(timeout=0.1)
    except TimeoutError:
        pass
    while not job.done():
        show_progress(full=full)
        time.sleep(0.25)
    progress_slot.empty()
    return job.result()


def retry() -> None:
    d = fetched(rt.dashboard.retry_failed())
    left = ui_schedule.failed_count(d)
    done = f"{left}개 시리즈 미수신" if left else "실패 시리즈 전체 재수신 완료"
    state.toast = f"{done} · {d.elapsed_seconds:.1f}s"


def refresh(page: str | None) -> None:
    d = fetched(rt.dashboard.get(refresh=True, page=page), full=page is None)
    if page is None:
        ok = sum(source.ok for source in d.sources)
        total = sum(source.total for source in d.sources)
        state.toast = f"{ok}/{total}개 시리즈 갱신 · {d.elapsed_seconds:.1f}s"
    elif page == "schedule":
        failed = [
            name
            for name, message in (
                ("발표 일정", d.calendar_message),
                ("FOMC 일정", d.fomc_message),
            )
            if message
        ]
        done = f"{'·'.join(failed)} 수신 실패" if failed else "발표 일정·FOMC 일정 갱신"
        state.toast = f"{done} · {d.elapsed_seconds:.1f}s"
    else:
        rows = [r for p in d.panels if p.page == page for r in p.rows]
        ok = sum(r.status == "ok" and r.message is None for r in rows)
        state.toast = f"{ok}/{len(rows)}개 지표 갱신 · {d.elapsed_seconds:.1f}s"


def toggle_ai() -> None:
    state.ai_open = not state.ai_open


@st.dialog("설정", width="large", on_dismiss="rerun")
def settings_dialog() -> None:
    ai, universe = st.tabs(
        [
            ":material/smart_toy: AI 분석",
            ":material/stacked_line_chart: ETF 유니버스",
        ],
        key="settings_tabs",
        on_change="rerun",
    )
    if universe.open:
        with universe:
            etf_page.universe_settings(etf_context())
    with ai:
        ai_settings()
        with st.container(horizontal=True, horizontal_alignment="right"):
            if st.button("취소"):
                st.rerun()
            if st.button("적용", type="primary"):
                apply_ai()
                st.rerun()


@st.fragment
def settings_button() -> None:
    if st.button("설정", icon=":material/tune:", shortcut="S"):
        state.settings_open = True
        settings_dialog()
    elif state.pop("settings_open", False) and state.pop("etf_universe_changed", False):
        st.rerun()


@st.fragment
def ai_dock(view: View) -> None:
    with st.container(key="ai_fab"):
        st.button(
            "",
            icon=":material/close:" if state.ai_open else ":material/smart_toy:",
            key="ai_toggle",
            help="페이지 챗봇",
            type="primary",
            shortcut="A",
            on_click=toggle_ai,
        )
    if state.ai_open:
        with st.container(key="ai_panel", border=True):
            page_chat(view)


@st.fragment(run_every=60.0)
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


def etf_stamp(updater: etf.Updater) -> object:
    return price_store().stamp(), updater.store.version


def etf_context() -> etf_page.EtfContext:
    if args.demo:
        return etf_page.EtfContext(
            settings=settings_path,
            updater=None,
            held=etf_held,
            confirm=etf_confirm,
            note=limit_note,
            rows=demo_etf.rows,
            funds=demo_etf.funds,
            stamp=lambda: demo_etf.STAMP,
            detail=demo_etf.detail,
            company=demo_etf.company,
            stock_rows=demo_etf.stock_rows,
        )
    updater = etf_updater()
    keys, agent = market_keys(), sec_agent()
    return etf_page.EtfContext(
        settings=settings_path,
        updater=updater,
        held=etf_held,
        confirm=etf_confirm,
        note=limit_note,
        rows=lambda tickers: price_store().rows(list(tickers)),
        funds=updater.store.funds,
        stamp=partial(etf_stamp, updater),
        detail=lambda ticker: etf.open_detail(
            ticker, agent, updater.store, datetime.now().astimezone().date()
        ),
        company=lambda ticker: fundamentals.company_info(
            ticker, keys, agent, datetime.now().astimezone().date()
        ),
        stock_rows=lambda symbol: price_store().rows([symbol]),
    )


def show_etf() -> None:
    etf_page.page(etf_context())


def show_agent() -> None:
    agent_page(View(data, "agent", set()))


def update_all() -> None:
    etf_page.update(etf_context())
    try:
        book = portfolio.load_book(root)
    except portfolio.BookError:
        return
    pf_update(portfolio_fetcher(), book, portfolio.missing_packages())


def dashboard_page(
    render: Callable[[Dashboard, UiContext], None],
) -> Callable[[], None]:
    def show() -> None:
        render(data, UiContext(rt.dashboard.full_series))

    return show


state.pop("etf_universe_changed", None)
with st.container(horizontal=True, vertical_alignment="center"):
    st.title("FinAgent", anchor=False, width="content")
    if args.demo:
        st.badge("데모", color="gray", help=DEMO_HELP)
    st.space("stretch")
    with st.container(width="content"):
        clock_slot = st.empty()
    with st.container(width="content"):
        settings_button()
    refreshing = st.button(
        "갱신",
        icon=":material/refresh:",
        type="primary",
        shortcut="U",
        help=REFRESH_HELP,
    )
progress_slot = st.empty()

DASHBOARD_PAGES = (
    ("", "overview", "개요", ":material/dashboard:", ui_overview.render),
    ("economy", "economy", "경제 지표", ":material/query_stats:", ui_economy.render),
    (
        "monetary",
        "monetary",
        "통화정책·금리",
        ":material/account_balance:",
        ui_monetary.render,
    ),
    (
        "finance",
        "finance",
        "금융시장·심리",
        ":material/candlestick_chart:",
        ui_finance.render,
    ),
    ("schedule", "schedule", "일정·출처", ":material/event:", ui_schedule.render),
)
page = st.navigation(
    [
        *(
            st.Page(
                dashboard_page(render),
                title=title,
                icon=icon,
                url_path=path or None,
                default=not path,
            )
            for path, _, title, icon, render in DASHBOARD_PAGES
        ),
        st.Page(
            show_etf,
            title="ETF",
            icon=":material/stacked_line_chart:",
            url_path="etf",
        ),
        st.Page(
            show_agent,
            title="에이전트",
            icon=":material/smart_toy:",
            url_path="agent",
        ),
        st.Page(
            portfolio_page,
            title="포트폴리오",
            icon=":material/pie_chart:",
            url_path="portfolio",
        ),
    ],
    position="top",
)
page_key = next(
    (key for path, key, _, _, _ in DASHBOARD_PAGES if path == page.url_path),
    page.url_path,
)
if rt.dashboard.refreshes == 0:
    update_all()
    refresh(None)
elif state.pop("retry_failed", False):
    retry()
elif refreshing and page_key in {key for _, key, _, _, _ in DASHBOARD_PAGES}:
    refresh(page_key)
elif refreshing:
    state.refresh_page = True
data = fetched(rt.dashboard.get())

if state.toast:
    st.toast(state.toast)
    state.toast = ""

with clock_slot:
    clocks(data.generated_at)

page_panels = {p.id for p in data.panels if p.page == page_key}
with st.container(key="page_heading"):
    st.header(page.title, anchor=False)
page.run()
state.pop("refresh_page", None)

theme_watch()
st.html(palette.scheme_css(palette.kind()))
st.html(CARDS_CSS)
st.html(AI_CSS)
if page_key not in {"agent", "portfolio"}:
    ai_dock(View(data, page_key, page_panels))
