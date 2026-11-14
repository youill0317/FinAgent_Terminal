# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import TYPE_CHECKING

import streamlit as st

from finagent import display, portfolio
from finagent.analysis import documents, skill
from finagent.analysis.briefing import Briefing
from finagent.analysis.company_tools import universe
from finagent.analysis.presets import ETF, MACRO, STOCK, Preset
from finagent.analysis.providers.base import ProviderError
from finagent.demo import etf as demo_etf
from finagent.ui import shared
from finagent.ui.shared import etf_confirm, etf_held, etf_updater

if TYPE_CHECKING:
    from finagent.analysis.context import Document
    from finagent.analysis.providers.base import ProviderStatus
    from finagent.analysis.service import Workspace
    from finagent.models import Dashboard

state = st.session_state
STOCK_POPOVER = "ai_stock_popover"
ETF_POPOVER = "ai_etf_popover"
STATE_ICON = {
    "working": ":material/progress_activity:",
    "done": ":material/check_circle:",
    "failed": ":material/error:",
    "stopped": ":material/stop_circle:",
}


@dataclasses.dataclass(frozen=True, slots=True)
class View:
    data: Dashboard
    key: str
    panels: set[str]


@dataclasses.dataclass(frozen=True, slots=True)
class Ready:
    statuses: dict[str, ProviderStatus]
    held: list[str]
    refused: str
    etfs: list[str]
    skill: bool


def workspace() -> Workspace:
    return shared.app.rt.workspace


def conversation() -> display.AnalysisLog:
    found = workspace().board
    return found.threads.get(state.ai_thread, found.main)


def ai_settings() -> None:
    providers = shared.app.rt.analysis.providers
    found = shared.app.rt.call(shared.app.rt.analysis.statuses())
    provider_col, model_col, thread_col = st.columns(3)
    provider_id = provider_col.selectbox(
        "공급자",
        list(providers),
        index=list(providers).index(shared.app.rt.analysis.choice.provider),
        format_func=lambda pid: (
            f"{providers[pid].title} · "
            f"{display.STATUS_TEXT.get(found[pid].kind, found[pid].kind)}"
        ),
        key="set_provider",
        help=(
            "AI 분석 엔진 공급자 선택  \n이름 옆 상태가 준비됨일 때만 실행  \n"
            "키 없음은 환경변수 API 키 부재, 설치 안 됨은 공급자 패키지 미설치"
        ),
    )
    _ = model_col.text_input(
        "조정자 모델",
        value=shared.app.rt.analysis.choice.models.get(provider_id, ""),
        placeholder=providers[provider_id].default_model,
        key=f"set_model_{provider_id}",
        help=(
            "사용자 대화 및 스레드 작업 위임 전담 조정자 모델  \n"
            "공란 시 입력란에 흐리게 표시된 기본 모델 적용"
        ),
    )
    _ = thread_col.text_input(
        "스레드 모델",
        value=shared.app.rt.analysis.choice.thread_models.get(provider_id, ""),
        key=f"set_thread_model_{provider_id}",
        help=(
            "개별 심층 분석 및 보고서 작성 전담 스레드 모델  \n"
            "공란 시 조정자와 동일 모델 적용"
        ),
    )
    _ = st.text_area(
        "시스템 프롬프트",
        value=shared.app.rt.analysis.choice.instructions,
        height=160,
        key="set_instructions",
        help=(
            "모든 AI 분석에서 모델에 시스템 프롬프트로 보내는 지침  \n"
            "공란 시 미전송  \n"
            "다음 실행부터 적용, 앱 인용 규칙·도구 안내는 항상 앞에 첨부"
        ),
    )


def apply_ai() -> None:
    provider_id: str = state.set_provider
    model: str = state.get(f"set_model_{provider_id}", "")
    thread_model: str = state.get(f"set_thread_model_{provider_id}", "")
    try:
        _ = shared.app.rt.analysis.choose(
            provider_id,
            model.strip(),
            state.get("set_instructions"),
            thread_model,
        )
    except (ProviderError, OSError) as error:
        state.toast = f"AI 분석 설정 저장 실패: {error}"


def new_conversation() -> None:
    shared.app.rt.renew()
    state.ai_thread = ""


def open_document(view: View) -> Document:
    snapshot = view.data.generated_at
    if view.key == "portfolio":
        values = dict(state.ai_page_values)
        for part in state.ai_page_parts.values():
            values.update(part)
        return documents.portfolio_document(values, snapshot)
    if view.key == "etf":
        found: documents.EtfView | None = state.get("ai_etf_view")
        if found is None:
            return documents.portfolio_document({}, snapshot)
        return documents.etf_document(found, snapshot)
    return documents.page_document(view.data, view.key)


def start_run(text: str, view: View) -> bool:
    text = text.strip()
    if not text:
        return False
    shared.app.rt.call(workspace().ask(text, open_document(view)))
    return True


def start_thread(preset: Preset, view: View, ticker: str = "") -> None:
    target = f"{ticker} {preset.title}" if ticker else preset.title
    text = f"{target} 스레드 시작 요청"
    document = documents.request_document(view.data.generated_at)
    shared.app.rt.call(workspace().ask(text, document, open_document(view)))


def etf_options() -> list[str]:
    listed = [p.ticker for p in universe(shared.app.settings_path)]
    funds = demo_etf.funds() if shared.app.demo else etf_updater().store.funds()
    held = [
        t for t in etf_held() if t not in listed and portfolio.is_fund(funds.get(t))
    ]
    return [*listed, *held]


def start_etf(view: View, ticker: str) -> None:
    state.ai_last_etf = ticker
    state[ETF_POPOVER] = False
    start_thread(ETF, view, ticker)


def etf_picker(view: View, options: list[str]) -> None:
    last = state.ai_last_etf
    ticker = st.selectbox(
        "ETF",
        options,
        index=options.index(last) if last in options else None,
        placeholder="ETF 티커 선택",
        help=(
            "유니버스 등록 및 보유 ETF 목록  \n"
            "마지막으로 상세를 열었거나 분석한 ETF 미리 선택"
        ),
    )
    if st.button(
        "분석 시작",
        type="primary",
        disabled=not ticker,
        key="ai_etf_go",
        on_click=start_etf,
        args=(view, ticker),
    ):
        st.rerun()


def choice_line(status: ProviderStatus) -> str:
    pid = shared.app.rt.analysis.choice.provider
    model = shared.app.rt.analysis.model_for(pid)
    worker = shared.app.rt.analysis.model_for(pid, thread=True)
    parts = [
        shared.app.rt.analysis.providers[pid].title,
        model if worker == model else f"{model} · 스레드 {worker}",
        display.STATUS_TEXT.get(status.kind, status.kind),
    ]
    return " · ".join(parts)


def ready() -> Ready:
    held: list[str] = []
    refused = ""
    try:
        held = portfolio.book_tickers(portfolio.load_book(shared.app.root))
    except portfolio.BookError as error:
        refused = str(error)
    return Ready(
        shared.app.rt.call(shared.app.rt.analysis.statuses()),
        held,
        refused,
        etf_options(),
        shared.app.demo or skill.available(shared.app.root / skill.DATA_DIR),
    )


def start_stock(view: View, ticker: str, held: list[str]) -> None:
    state.ai_stock_refused = ""
    if ticker not in held:
        try:
            Briefing.refuse(ticker, etf_confirm(ticker), {"찾을 수 없음"})
        except ValueError as error:
            state.ai_stock_refused = str(error)
            return
    state.ai_last_stock = ticker
    state[STOCK_POPOVER] = False
    start_thread(STOCK, view, ticker)


def stock_picker(view: View, found: Ready) -> None:
    if found.refused:
        st.error(found.refused)
        return
    held = found.held
    last = state.ai_last_stock
    picked = st.selectbox(
        "보유 종목",
        held,
        index=held.index(last) if last in held else None,
    )
    typed = (
        st.text_input(
            "티커",
            help=(
                "미보유 종목 직접 입력  \n"
                "입력 시 위에서 고른 종목 대신 이 티커 분석  \n"
                "미국 밖 종목은 거래소 접미사 포함(예: 005930.KS)"
            ),
        )
        .strip()
        .upper()
    )
    ticker = typed or picked
    if refused := state.get("ai_stock_refused"):
        st.error(refused)
    started = st.button(
        "분석 시작",
        type="primary",
        disabled=not ticker,
        on_click=start_stock,
        args=(view, ticker, held),
    )
    if started and not state.ai_stock_refused:
        st.rerun()


def stop() -> None:
    shared.app.rt.call(workspace().stop(state.ai_thread))


def open_thread(key: str) -> None:
    state.ai_thread = key


def thread_card(key: str, place: str) -> None:
    log = workspace().board.threads[key]
    st.button(
        f"{log.title} · {display.STATE_TEXT[log.state]}",
        icon=STATE_ICON[log.state],
        key=f"ai_card_{place}_{key}",
        on_click=open_thread,
        args=(key,),
        width="stretch",
    )
    if log.conclusion:
        st.caption(log.conclusion)


def overview() -> None:
    threads = workspace().board.threads
    if not threads:
        return
    working = sum(log.working for log in threads.values())
    label = f"스레드 {len(threads)}개" + (f" · 작업 중 {working}" if working else "")
    with st.expander(label, type="compact"):
        for found, title in display.STATE_TEXT.items():
            keys = [key for key, log in threads.items() if log.state == found]
            if keys:
                st.caption(title)
            for key in keys:
                thread_card(key, "overview")


def presets(view: View, found: Ready) -> None:
    folder = shared.app.root / skill.DATA_DIR
    if st.button(MACRO.title, key="ai_preset_macro"):
        start_thread(MACRO, view)
        st.rerun()
    with st.popover(
        STOCK.title,
        disabled=not found.skill,
        help=None if found.skill else f"`{folder}` 디렉터리 내 참고 자료 필요",
        key=STOCK_POPOVER,
        on_change="rerun",
    ):
        if found.skill:
            stock_picker(view, found)
    with st.popover(ETF.title, key=ETF_POPOVER, on_change="rerun"):
        etf_picker(view, found.etfs)


def controls(view: View, found: Ready) -> None:
    log = conversation()
    key: str = state.ai_thread
    with st.container(horizontal=True, vertical_alignment="center"):
        if not key:
            presets(view, found)
        if log.working:
            st.button("중지", icon=":material/stop:", key="ai_stop", on_click=stop)
        if not key:
            st.button("새 대화", on_click=new_conversation, disabled=workspace().busy)
    if key:
        return
    request = st.chat_input("분석 요청 사항 입력", key="ai_input", disabled=log.working)
    if request and start_run(request, view):
        st.rerun()


def turn_steps(tools: list[tuple[str, bool]]) -> None:
    for summary, ok in tools:
        st.caption(summary if ok else f":red[{summary}]")


def finished_steps(tools: list[tuple[str, bool]]) -> None:
    failed = Counter(summary for summary, ok in tools if not ok)
    for summary, count in failed.items():
        st.caption(f":red[{summary} · {count}회]" if count > 1 else f":red[{summary}]")
    if tools:
        with st.expander(f"수행 단계 {len(tools)}개", type="compact"):
            turn_steps(tools)


def answer(turn: display.TurnView, links: dict[str, str], *, live: bool) -> None:
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        st.caption(turn.run_line)
        if turn.lines is not None:
            finished_steps(turn.tools)
            st.markdown(display.answer_markdown(turn.lines, links))
            st.caption(turn.summary, help=display.SUMMARY_HELP)
        elif live:
            turn_steps(turn.tools)
            st.markdown(display.live_tail(turn.raw))
        else:
            finished_steps(turn.tools)
            st.markdown(display.live_text(turn.raw))
        for message, error in turn.notices:
            if error:
                st.error(message)
            else:
                st.caption(message, help=display.STOPPED_HELP)


def turns(view: View, log: display.AnalysisLog) -> None:
    links = {
        row: panel
        for row, panel in display.panel_of(view.data).items()
        if panel in view.panels
    }
    for index, turn in enumerate(log.turns):
        live = log.working and index == len(log.turns) - 1
        if turn.question:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(display.md(turn.question))
        if live or not turn.empty:
            answer(turn, links, live=live)
        for key in turn.threads:
            thread_card(key, "turn")


def received_data(log: display.AnalysisLog) -> None:
    if not log.received and not log.sources:
        return
    with st.expander("수신 데이터", type="compact"):
        lines = [f"- `{cite}` {display.md(info)}" for cite, info in log.received]
        lines.extend(f"- [{display.md(name)}]({url})" for name, url in log.sources)
        st.markdown("\n".join(lines))


def toggle_wide() -> None:
    state.ai_wide = not state.ai_wide


def panel_head(d: Dashboard, log: display.AnalysisLog, status: ProviderStatus) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        if state.ai_thread:
            st.button(
                "",
                icon=":material/arrow_back:",
                key="ai_back",
                help="메인 대화 목록 복귀",
                type="tertiary",
                on_click=open_thread,
                args=("",),
            )
        st.subheader(log.title or "AI 분석", anchor="analysis")
        st.space("stretch")
        st.button(
            "",
            icon=":material/close_fullscreen:"
            if state.ai_wide
            else ":material/open_in_full:",
            key="ai_wide_on" if state.ai_wide else "ai_wide_off",
            help="기본 너비 복귀" if state.ai_wide else "패널 너비 확장",
            type="tertiary",
            on_click=toggle_wide,
        )
    basis = log.header or f"데이터 기준 시점: {display.short_time(d.generated_at)}"
    st.caption(
        f"{choice_line(status)} · {basis}",
        help=display.STATUS_HELP.get(status.kind, "").format(message=status.message)
        or None,
    )
    if log.conclusion:
        with st.container(key="ai_conclusion"):
            st.markdown(log.conclusion)
    if not state.ai_thread:
        overview()


def analysis(view: View) -> None:
    busy = workspace().busy
    found = ready()

    @st.fragment(run_every=0.5 if busy else None)
    def body() -> None:
        ws = workspace()
        ws.portfolio = state.get("pf_selected") or ""
        ws.drain()
        if state.ai_thread not in ws.board.threads:
            state.ai_thread = ""
        if busy and not ws.busy and not state.get("settings_open"):
            st.rerun()
        log = conversation()
        status = found.statuses[shared.app.rt.analysis.choice.provider]
        panel_head(view.data, log, status)
        reading = log.report and len(log.turns) == 1 and not log.working
        outer = "ai_report" if reading else "ai_turns"
        with st.container(key=outer), st.container(key=f"{outer}_inner"):
            turns(view, log)
            received_data(log)
        controls(view, found)

    body()
