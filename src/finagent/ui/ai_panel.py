# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import TYPE_CHECKING, Final

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
from finagent.ui.slash import Command, slash_menu

if TYPE_CHECKING:
    from finagent.analysis.context import Document
    from finagent.analysis.service import Workspace
    from finagent.models import Dashboard

state = st.session_state
COMMANDS: Final[dict[str, Preset]] = {
    "/매크로": MACRO,
    "/macro": MACRO,
    "/etf": ETF,
    "/종목": STOCK,
    "/stock": STOCK,
}
COMMAND_TEXT: Final = "/매크로, /etf 티커, /종목 티커"
AGENT_HINT: Final = "요청 입력 · / 입력 시 프리셋 목록"
SLASH: Final = (
    Command("/매크로", MACRO.title, "", ("매크로", "macro")),
    Command("/etf", ETF.title, "티커", ("etf",)),
    Command("/종목", STOCK.title, "티커", ("종목", "stock")),
)
STATE_ICON: Final[dict[str, str]] = {
    "working": ":material/progress_activity:",
    "done": ":material/check_circle:",
    "failed": ":material/error:",
    "stopped": ":material/stop_circle:",
}
STEP_ICON: Final[dict[str, str | None]] = {
    "pending": None,
    "working": "spinner",
    "done": ":material/check_circle:",
    "failed": ":material/error:",
    "stopped": ":material/stop_circle:",
    "skipped": ":material/block:",
}
AGENT_ICON: Final[dict[str, str]] = {
    **STATE_ICON,
    "pending": ":material/schedule:",
    "skipped": ":material/block:",
}
FLOW_AVATAR: Final = ":material/account_tree:"


@dataclasses.dataclass(frozen=True, slots=True)
class View:
    data: Dashboard
    key: str
    panels: set[str]


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
            "페이지 챗봇도 같은 모델 사용  \n"
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


def start_thread(preset: Preset, ticker: str = "") -> None:
    target = f"{ticker} {preset.title}" if ticker else preset.title
    shared.app.rt.call(workspace().ask(f"{target} 스레드 시작 요청"))


def etf_options() -> list[str]:
    listed = [p.ticker for p in universe(shared.app.settings_path)]
    funds = demo_etf.funds() if shared.app.demo else etf_updater().store.funds()
    held = [
        t for t in etf_held() if t not in listed and portfolio.is_fund(funds.get(t))
    ]
    return [*listed, *held]


def check_stock(ticker: str) -> None:
    folder = shared.app.root / skill.DATA_DIR
    if not (shared.app.demo or skill.available(folder)):
        raise ValueError(f"종목 분석에는 `{folder}` 디렉터리 내 참고 자료 필요")
    held = portfolio.book_tickers(portfolio.load_book(shared.app.root))
    if ticker not in held:
        Briefing.refuse(ticker, etf_confirm(ticker), {"찾을 수 없음"})


def check_etf(ticker: str) -> None:
    if ticker not in etf_options():
        Briefing.refuse(ticker, etf_confirm(ticker), {"찾을 수 없음", "개별 주식"})


def run_command(text: str) -> None:
    name, _, rest = text.partition(" ")
    preset = COMMANDS.get(name.lower())
    if preset is None:
        raise ValueError(f"알 수 없는 명령: {name} · 사용 가능: {COMMAND_TEXT}")
    ticker = rest.strip().upper()
    if preset is MACRO:
        start_thread(MACRO)
        return
    if not ticker:
        raise ValueError(f"{name} 뒤에 티커 입력 (예: {name} AAPL)")
    if preset is STOCK:
        check_stock(ticker)
    else:
        check_etf(ticker)
    start_thread(preset, ticker)


def submit_agent() -> None:
    text: str = (state.get("ai_input") or "").strip()
    state.ai_command_error = ""
    if not text:
        return
    if not text.startswith("/"):
        shared.app.rt.call(workspace().ask(text))
        return
    try:
        run_command(text)
    except (ValueError, portfolio.BookError) as error:
        state.ai_command_error = str(error)


def stop() -> None:
    shared.app.rt.call(workspace().stop(state.ai_thread))


def open_thread(key: str) -> None:
    state.ai_thread = key


def card_label(log: display.AnalysisLog, flow: display.FlowView | None) -> str:
    working = flow is not None and flow.working
    status = flow.short if flow is not None and working else None
    return f"{log.title} · {status or display.STATE_TEXT[log.state]}"


def thread_card(key: str, place: str) -> display.FlowView | None:
    board = workspace().board
    log = board.threads[key]
    flow = display.flow_view(board, key)
    st.button(
        card_label(log, flow),
        icon=STATE_ICON[log.state],
        key=f"ai_card_{place}_{key}",
        on_click=open_thread,
        args=(key,),
        type="tertiary" if place == "turn" else "secondary",
        width="content" if place == "turn" else "stretch",
    )
    return flow


def list_card(key: str) -> None:
    log = workspace().board.threads[key]
    with st.container(gap="xsmall"):
        flow = thread_card(key, "list")
        if flow is not None and flow.working:
            st.progress(flow.fraction)
        elif flow is not None and flow.state in display.HALTED:
            st.caption(flow.activity)
        if log.conclusion:
            st.caption(log.conclusion)


def thread_list() -> None:
    threads = workspace().board.top()
    working = sum(log.working for log in threads.values())
    label = f"스레드 {len(threads)}개" + (f" · 작업 중 {working}" if working else "")
    st.subheader(label, anchor=False)
    for found, title in display.STATE_TEXT.items():
        keys = [key for key, log in threads.items() if log.state == found]
        if keys:
            st.caption(title)
        for key in keys:
            list_card(key)


def icon_button(icon: str, key: str, label: str, **options: object) -> None:
    st.button(
        "",
        icon=icon,
        key=key,
        help=label,
        type="tertiary",
        **options,  # pyright: ignore[reportArgumentType]
    )


def controls(log: display.AnalysisLog) -> None:
    if state.ai_thread:
        if log.working:
            icon_button(":material/stop_circle:", "ai_stop", "중지", on_click=stop)
        return
    if error := state.get("ai_command_error"):
        st.error(error)
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        if log.working:
            icon_button(":material/stop_circle:", "ai_stop", "중지", on_click=stop)
        else:
            icon_button(
                ":material/add_comment:",
                "ai_new",
                "새 대화",
                on_click=new_conversation,
                disabled=workspace().busy,
            )
        st.chat_input(
            AGENT_HINT,
            key="ai_input",
            disabled=log.working,
            on_submit=submit_agent,
        )
    slash_menu("ai_input", SLASH)


def steps_label(turn: display.TurnView) -> str:
    failed = sum(not ok for _, ok in turn.tools)
    parts = [turn.summary or f"단계 {len(turn.tools)}개"]
    if failed:
        parts.append(f":red[실패 {failed}]")
    return " · ".join(parts)


def turn_meta(turn: display.TurnView) -> None:
    failed = Counter(summary for summary, ok in turn.tools if not ok)
    with st.expander(steps_label(turn), type="compact"):
        st.caption(turn.run_line, help=display.SUMMARY_HELP)
        for summary, ok in turn.tools:
            if ok:
                st.caption(summary)
        for summary, count in failed.items():
            st.caption(f":red[{summary}" + (f" · {count}회]" if count > 1 else "]"))


def reply(turn: display.TurnView, links: dict[str, str], *, live: bool) -> None:
    if live:
        step = turn.tools[-1][0] if turn.tools else "작업 중"
        st.caption(f":shimmer[{step}]")
        st.markdown(display.live_tail(turn.raw))
        return
    if turn.tools:
        turn_meta(turn)
    if turn.lines is not None:
        st.markdown(display.answer_markdown(turn.lines, links))
    else:
        st.markdown(display.live_text(turn.raw))


def notices(turn: display.TurnView) -> None:
    for message, error in turn.notices:
        if error:
            st.error(message)
        else:
            st.caption(message, help=display.STOPPED_HELP)


def answer(turn: display.TurnView, links: dict[str, str], *, live: bool) -> None:
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        reply(turn, links, live=live)
        notices(turn)


def agent_label(agent: display.FlowAgent) -> str:
    text = f"{AGENT_ICON[agent.state]} {agent.name}"
    return f":red[{text}]" if agent.state == "failed" else text


def agent_pick(
    flow: display.FlowView, step: display.FlowStep
) -> display.FlowAgent | None:
    names = {agent.name: agent for agent in step.agents}
    picked = st.pills(
        "에이전트",
        list(names),
        format_func=lambda name: agent_label(names[name]),
        key=f"ai_pick_{flow.key}_{step.key}",
        label_visibility="collapsed",
    )
    return names.get(picked) if picked else None


def agent_pane(agent: display.FlowAgent, links: dict[str, str]) -> None:
    log = workspace().board.threads.get(agent.thread)
    if log is None or not 0 <= agent.turn < len(log.turns):
        st.caption(agent.line)
        return
    turn = log.turns[agent.turn]
    with st.container(border=True):
        if agent.task:
            st.caption(agent.task)
        reply(turn, links, live=agent.state == "working")
        notices(turn)


def step_body(
    flow: display.FlowView,
    step: display.FlowStep,
    turn: display.TurnView,
    links: dict[str, str],
) -> None:
    if step.key == "prepare":
        note = display.PREPARE_NOTE.get(step.state, "")
        st.caption(f"{note} · {turn.run}" if step.state == "done" else note)
        for summary, _ in turn.tools:
            st.caption(summary)
    elif step.key == "synth":
        if turn.lines is not None:
            st.caption(display.DELIVERED)
        else:
            agent_pane(step.agents[0], links)
    elif (picked := agent_pick(flow, step)) is not None:
        agent_pane(picked, links)


def flow_steps(
    flow: display.FlowView, turn: display.TurnView, links: dict[str, str]
) -> None:
    for step in flow.steps:
        with st.expander(
            step.label,
            key=f"ai_step_{flow.key}_{step.key}",
            type="step",
            icon=STEP_ICON[step.state],
        ):
            step_body(flow, step, turn, links)
    st.expander(
        flow.end_label,
        key=f"ai_step_{flow.key}_end",
        type="step",
        icon=STEP_ICON[flow.end_state],
    )


def flow_turn(
    flow: display.FlowView, turn: display.TurnView, links: dict[str, str]
) -> None:
    with st.chat_message("assistant", avatar=FLOW_AVATAR):
        if turn.lines is None:
            flow_steps(flow, turn, links)
        else:
            with st.expander(
                flow.fold_label(turn.summary), key=f"ai_flow_{flow.key}", type="compact"
            ):
                st.caption(turn.run_line, help=display.SUMMARY_HELP)
                flow_steps(flow, turn, links)
            st.markdown(display.answer_markdown(turn.lines, links))
        notices(turn)


def flow_head(flow: display.FlowView | None) -> None:
    if flow is None:
        return
    if flow.working:
        st.progress(flow.fraction, text=flow.bar_text)
    elif flow.state in display.HALTED:
        st.caption(flow.activity)


def turns(
    view: View, log: display.AnalysisLog, flow: display.FlowView | None = None
) -> None:
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
        if flow is not None and turn.plan is not None:
            flow_turn(flow, turn, links)
        elif live or not turn.empty:
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


def thread_head(log: display.AnalysisLog) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        icon_button(
            ":material/arrow_back:",
            "ai_back",
            "메인 대화 복귀",
            on_click=open_thread,
            args=("",),
        )
        st.subheader(log.title, anchor="analysis")
    if log.conclusion:
        with st.container(key="ai_conclusion"):
            st.markdown(log.conclusion)


def conversation_body(view: View, log: display.AnalysisLog) -> None:
    board = workspace().board
    flow = display.flow_view(board, state.ai_thread) if state.ai_thread else None
    with st.container(key="ai_page"):
        if state.ai_thread:
            thread_head(log)
            flow_head(flow)
        single = len(log.turns) == 1
        reading = single and (flow is not None or (log.report and not log.working))
        outer = "ai_report" if reading else "ai_turns"
        with st.container(key=outer), st.container(key=f"{outer}_inner"):
            turns(view, log, flow)
            received_data(log)
        controls(log)


def agent_page(view: View) -> None:
    busy = workspace().busy

    @st.fragment(run_every=0.5 if busy else None)
    def body() -> None:
        ws = workspace()
        ws.portfolio = state.get("pf_selected") or ""
        ws.drain()
        opened = ws.board.threads.get(state.ai_thread)
        if opened is None:
            state.ai_thread = ""
        elif opened.parent:
            state.ai_thread = opened.parent
        if busy != ws.busy and not state.get("settings_open"):
            st.rerun()
        log = conversation()
        if not ws.board.threads:
            conversation_body(view, log)
            return
        talk, side = st.columns([3, 1], gap="large")
        with talk:
            conversation_body(view, log)
        with side, st.container(key="ai_threads"):
            thread_list()

    body()


def chat() -> Workspace:
    return shared.app.rt.chat


def open_document(view: View) -> Document:
    if view.key == "etf":
        found: documents.EtfView | None = state.get("ai_etf_view")
        if found is None:
            return documents.page_document(view.data, view.key)
        return documents.etf_document(found, view.data.generated_at)
    return documents.page_document(view.data, view.key)


def ask_page(view: View) -> None:
    text: str = (state.get("chat_input") or "").strip()
    if text:
        shared.app.rt.call(chat().ask(text, open_document(view)))


def stop_chat() -> None:
    shared.app.rt.call(chat().stop())


def chat_head(log: display.AnalysisLog) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("페이지 챗봇", anchor=False)
        st.space("stretch")
        if log.working:
            icon_button(
                ":material/stop_circle:", "chat_stop", "중지", on_click=stop_chat
            )
        else:
            icon_button(
                ":material/add_comment:",
                "chat_new",
                "새 대화",
                on_click=shared.app.rt.renew_chat,
                disabled=chat().busy,
            )
        icon_button(
            ":material/close_fullscreen:"
            if state.ai_wide
            else ":material/open_in_full:",
            "ai_wide_on" if state.ai_wide else "ai_wide_off",
            "기본 너비 복귀" if state.ai_wide else "패널 너비 확장",
            on_click=toggle_wide,
        )


def page_chat(view: View) -> None:
    busy = chat().busy

    @st.fragment(run_every=0.5 if busy else None)
    def body() -> None:
        ws = chat()
        ws.drain()
        if busy != ws.busy and not state.get("settings_open"):
            st.rerun()
        log = ws.board.main
        chat_head(log)
        with st.container(key="ai_turns"), st.container(key="ai_turns_inner"):
            turns(view, log)
        st.chat_input(
            "현재 화면에 대해 질문",
            key="chat_input",
            disabled=log.working,
            on_submit=ask_page,
            args=(view,),
        )

    body()
