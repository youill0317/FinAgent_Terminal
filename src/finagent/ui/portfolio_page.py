# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false, reportUnknownMemberType=false
"""포트폴리오 보유 현황, 수익률(TWR·MWR), 실질 보유(Look-through) 및 관리 화면."""

from __future__ import annotations

import math
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, cast

import altair as alt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent import demo, etf, investors, lookthrough, perf, portfolio, quant
from finagent.ui import palette, shared
from finagent.ui.components import clear_selection, sector_label
from finagent.ui.investors_view import inv_view
from finagent.ui.pf_common import (
    CHART_HEIGHT,
    PF_AXIS,
    PF_DASH,
    PF_DONUT_SLICES,
    PF_MIN_QUANTITY,
    Key,
    Subject,
    cached,
    headed,
    pf_annual,
    pf_ask,
    pf_bar,
    pf_choose,
    pf_color,
    pf_compact,
    pf_current,
    pf_donut,
    pf_funds,
    pf_line_chart,
    pf_look,
    pf_metric,
    pf_money,
    pf_no_prices,
    pf_num,
    pf_part,
    pf_percent,
    pf_periods,
    pf_picked,
    pf_points,
    pf_pp,
    pf_prices,
    pf_shares,
    pf_store,
    pf_subject,
    pf_table,
    pf_trades_key,
    portfolio_fetcher,
)
from finagent.ui.pf_simulation import pf_simulation
from finagent.ui.pf_stats import (
    pf_book_heatmap,
    pf_component,
    pf_correlation,
    pf_stats,
    pf_style_box,
)
from finagent.ui.stock_view import stock_view

if TYPE_CHECKING:
    from collections.abc import Callable

state = st.session_state

PF_SECTIONS = {
    "returns": "수익률",
    "lookthrough": "실질 보유",
    "stats": "통계",
    "simulation": "시뮬레이션",
}

PF_SIDE_WIDTH = 72
PF_VALUE_HELP = (
    "[정의] 보유 종목 현재 시장가치 총합\n\n"
    "[해석] 포트폴리오 전체 자산 규모\n\n"
    "[산출] 종목별 (보유 수량 × 최근 종가) 합산(종가 미확인 종목 제외)"
)
PF_UNREALIZED_HELP = (
    "[정의] 현재 보유 수량의 미실현 평가손익 및 원가 대비 수익률\n\n"
    "[해석] 양(+)은 보유분 평가이익, 음(-)은 평가손실\n\n"
    "[산출] 현재 평가액 − 보유분 이동평균 원가(매수 수수료 포함)"
)
PF_DIALOGS = {
    "new": "새 포트폴리오",
    "edit": "포트폴리오 편집",
    "delete": "포트폴리오 삭제",
}


@st.cache_resource(show_spinner=False, max_entries=2)
def pf_look_prices(names: tuple[str, ...], stamp: int) -> portfolio.Prices:
    del stamp
    return shared.app.calc.load_prices(shared.app.root, names)


PF_LOOK_MONTHS = 12


def pf_open(kind: str, p: portfolio.Portfolio | None = None) -> None:
    state.pf_dialog = kind
    state.pf_dialog_error = ""
    if kind == "new":
        state.pf_new_name = ""
    elif p is not None:
        state.pf_edit_name = p.name


def pf_close() -> None:
    state.pf_dialog = ""


def pf_create() -> None:
    made: list[str] = []

    def change(book: portfolio.Book) -> portfolio.Book:
        new, pid = portfolio.add_portfolio(book, state.pf_new_name)
        made.append(pid)
        return new

    state.pf_dialog_error = pf_store(change)
    if not state.pf_dialog_error:
        state.pf_selected = made[0]
        pf_close()


def pf_edit(pid: str) -> None:
    state.pf_dialog_error = pf_store(
        lambda book: portfolio.edit_portfolio(book, pid, state.pf_edit_name)
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
        st.rerun()
    pid = p.id if p is not None else ""
    if kind == "delete" and p is not None:
        st.markdown(f"**{p.name}** 및 거래 {len(p.trades)}건 삭제 확인")
        if state.pf_dialog_error:
            st.error(state.pf_dialog_error)
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button(
                "삭제",
                type="primary",
                key="pf_delete_ok",
                on_click=pf_delete,
                args=(pid,),
            )
            st.button("취소", key="pf_cancel", on_click=pf_close)
        return
    with st.form(f"pf_{kind}_form", border=False):
        st.text_input("이름", key=f"pf_{kind}_name", max_chars=40)
        if state.pf_dialog_error:
            st.error(state.pf_dialog_error)
        with st.container(horizontal=True, horizontal_alignment="right"):
            if kind == "new":
                st.form_submit_button(
                    "만들기", type="primary", key="pf_new_save", on_click=pf_create
                )
            else:
                st.form_submit_button(
                    "저장",
                    type="primary",
                    key="pf_edit_save",
                    on_click=pf_edit,
                    args=(pid,),
                )
            st.form_submit_button("취소", key="pf_cancel", on_click=pf_close)


def pf_dialogs(p: portfolio.Portfolio | None) -> None:
    kind: str = state.get("pf_dialog", "")
    if kind in PF_DIALOGS and (kind == "new" or p is not None):
        st.dialog(PF_DIALOGS[kind], on_dismiss=pf_close)(pf_dialog_body)(kind, p)


def pf_selector(book: portfolio.Book) -> portfolio.Portfolio | None:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        pf_choose("포트폴리오", {p.id: p.name for p in book.portfolios}, "pf_selected")
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


def pf_splits() -> portfolio.Splits:
    return {} if shared.app.demo else portfolio.load_splits()


def pf_add_trade(pid: str, side: portfolio.Side) -> None:
    state.pf_trade_error = ""
    quantity: float | None = state.pf_quantity
    price: float | None = state.pf_price
    if not quantity or not price:
        state.pf_trade_error = "수량·가격 필수 입력"
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
    if (
        trade.ticker not in pf_current().tickers
        and shared.etf_confirm(trade.ticker) == "찾을 수 없음"
    ):
        state.pf_trade_error = f"{trade.ticker}: 미확인 티커"
        return
    splits = pf_splits()
    state.pf_trade_error = pf_store(
        lambda book: portfolio.add_trade(book, pid, trade, splits)
    )
    if not state.pf_trade_error:
        state.toast = f"{trade.ticker} {portfolio.SIDES[side]} 거래 추가 완료"
        state.pf_quantity = 1.0
        state.pf_price = None
        state.pf_fee = 0.0
        state.pf_price_note = ""
        state.pf_memo = ""


def pf_remove_trades(pid: str, ids: list[str]) -> None:
    key = f"pf_trades_{pid}"
    rows: list[int] = state[key]["selection"]["rows"] if key in state else []
    chosen = {ids[row] for row in rows}
    if not chosen:
        return
    splits = pf_splits()
    state.pf_log_error = pf_store(
        lambda book: portfolio.remove_trades(book, pid, chosen, splits)
    )
    if not state.pf_log_error:
        clear_selection(key)
        state.toast = f"거래 {len(chosen)}건 삭제 완료"


def pf_prefill_later() -> None:
    state.pf_prefill = True


def pf_prefill() -> None:
    state.pf_price_note = ""
    try:
        ticker = portfolio.clean_ticker(state.get("pf_ticker", ""))
    except portfolio.BookError:
        return
    prices = pf_current()
    if ticker not in prices.tickers:
        if shared.app.demo or portfolio.missing_packages():
            state.pf_price = None
            state.pf_price_note = "수신된 가격 없음"
            return
        try:
            with st.spinner(f"{ticker} 가격 수신 중…"):
                prices = portfolio_fetcher().fetch_now([ticker])
        except Exception:  # noqa: BLE001  # 네트워크·공급자 조회 실패 시 빈 가격 객체 폴백
            prices = portfolio.no_prices()
    price, day = shared.app.calc.close_on(prices.close, ticker, state.pf_day)
    if day is None:
        state.pf_price = None
        state.pf_price_note = "가격 미확인"
        return
    state.pf_price = round(price, portfolio.DECIMALS)
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
            min_value=portfolio.HISTORY_START,
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
            format="%g",
            key="pf_quantity",
            width=130,
        )
        st.number_input(
            "가격",
            min_value=0.0,
            value=None,
            format="%g",
            key="pf_price",
            width=150,
        )
        st.number_input("수수료", min_value=0.0, key="pf_fee", width=120)
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


def pf_trades_table(p: portfolio.Portfolio) -> tuple[pd.DataFrame, list[str]]:
    trades = list(reversed(portfolio.ordered(p.trades)))
    table = pd.DataFrame(
        {
            "날짜": [t.day.isoformat() for t in trades],
            "종목": [t.ticker for t in trades],
            "구분": [portfolio.SIDES[t.side] for t in trades],
            "수량": [pf_shares(t.quantity) for t in trades],
            "가격": [pf_money(t.price) for t in trades],
            "수수료": [pf_money(t.fee) for t in trades],
            "메모": [t.memo for t in trades],
        }
    )
    return table, [t.id for t in trades]


def pf_trade_log(p: portfolio.Portfolio) -> None:
    table, ids = pf_trades_table(p)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(
            f"{len(ids)}건",
            help="[정의] 포트폴리오 누적 거래 건수\n\n"
            "[산출] 등록된 매수·매도 거래 레코드 합계(동일 일자 복수 거래는 입력"
            " 순서대로 원가·수량 정산)",
            width="content",
        )
        st.button(
            "선택 삭제",
            icon=":material/delete:",
            key="pf_remove",
            type="secondary",
            on_click=pf_remove_trades,
            args=(p.id, ids),
        )
    if error := state.pop("pf_log_error", ""):
        st.error(error)
    st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"pf_trades_{p.id}",
        column_config={"메모": st.column_config.TextColumn(width="large")},
        alt="거래 내역",
    )


@st.fragment
def pf_trade_box(p: portfolio.Portfolio) -> None:
    if state.toast:
        st.rerun()
    box = st.expander("거래 내역", key="pf_trades_exp", on_change="rerun")
    if box.open:
        with box:
            pf_trade_log(p)


def pf_trade_open() -> None:
    state.pf_trade_dialog = True
    state.pf_trade_error = ""
    state.pf_price_note = ""


def pf_trade_close() -> None:
    state.pf_trade_dialog = False


def pf_trade_body(p: portfolio.Portfolio) -> None:
    if state.toast:
        st.rerun()
    pf_trade_form(p)


def pf_trade_dialog(p: portfolio.Portfolio) -> None:
    if state.get("pf_trade_dialog"):
        st.dialog("거래 추가", width="large", on_dismiss=pf_trade_close)(pf_trade_body)(
            p
        )


def pf_holdings(
    p: portfolio.Portfolio, prices: portfolio.Prices, entered: portfolio.Portfolio
) -> None:
    pf_trade_dialog(entered)
    items = shared.app.calc.holdings(p, prices.close)
    held = [h for h in items if h.is_open]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("**현황**")
        st.space("stretch")
        st.button(
            "거래 추가",
            icon=":material/add:",
            key="pf_trade_open",
            on_click=pf_trade_open,
        )
    if held:
        totals = portfolio.totals(items)
        with st.container(horizontal=True, key="pf_now_cards"):
            pf_metric(
                "평가액",
                pf_money(totals.value),
                help=PF_VALUE_HELP,
            )
            pf_metric(
                "평가손익",
                pf_money(totals.unrealized, signed=True),
                pf_percent(totals.open_return, signed=True),
                delta_color=pf_color(totals.unrealized),
                delta_arrow="off",
                help=PF_UNREALIZED_HELP,
            )
    if prices.tickers & {t.ticker for t in p.trades}:
        left, right = st.columns([1, 2])
        with left, st.container(border=True, height="stretch"):
            headed(
                "**비중**",
                "[정의] 보유 종목별 현재 평가액 구성 비중"
                f"({PF_DONUT_SLICES}종목 초과 시 상위 {PF_DONUT_SLICES - 1}종목 외 기타"
                " 항목 병합 표기)\n\n"
                "[산출] 개별 종목 평가액 ÷ 포트폴리오 총평가액",
            )
            w = portfolio.weights(items)
            if w.empty:
                st.caption("평가액 확인 종목 없음" if held else "보유 종목 없음")
            else:
                rows = [
                    (h.ticker, cast("float", w[h.ticker]), h.value)
                    for h in held
                    if h.ticker in w
                ]
                st.altair_chart(
                    pf_donut(rows, pf_money),
                    alt="비중",
                )
        with right, st.container(border=True, height="stretch"):
            headed(
                "**평가액과 순투입**",
                "[정의] 일별 포트폴리오 평가액 및 누적 순투입 원금 추이\n\n"
                "[해석] 두 시계열 간 수직 간격이 해당 시점 누적 총손익\n\n"
                "[산출] 평가액 = Σ(보유 수량 × 당일 종가), 순투입 = 누적 매수액(수수료"
                " 포함) − 누적 매도 회수액(수수료 차감)",
            )
            frame = cached(
                "history",
                pf_trades_key(p),
                partial(shared.app.calc.history, p, prices.close),
            )
            lines = portfolio.value_lines(frame)
            if len(frame) > 1:
                st.altair_chart(
                    pf_line_chart(portfolio.line_points(lines), PF_AXIS),
                    alt="평가액과 순투입",
                )
            else:
                st.caption(PF_NO_LINE)
    if held:
        pf_held_table(items, held)
    if p.trades:
        pf_trade_box(entered)


def pf_holdings_table(items: list[portfolio.Holding]) -> dict[str, list[str]]:
    held = [h for h in items if h.is_open]
    w = portfolio.weights(held)
    return {
        "종목": [h.ticker for h in held],
        "수량": [pf_shares(h.position.quantity) for h in held],
        "평균단가": [pf_money(h.position.average) for h in held],
        "종가": [pf_money(h.price) for h in held],
        "종가일": [h.day.isoformat() if h.day else portfolio.UNKNOWN for h in held],
        "평가액": [pf_money(h.value) for h in held],
        "비중": [
            pf_percent(float(cast("float", w[h.ticker])))
            if h.ticker in w.index
            else portfolio.UNKNOWN
            for h in held
        ],
        "평가손익": [pf_money(h.unrealized, signed=True) for h in held],
        "수익률": [pf_percent(h.return_rate, signed=True) for h in held],
    }


def pf_jump(tickers: list[str]) -> None:
    picked = pf_picked("pf_held_table")
    if picked is not None:
        state.pf_jump_stock = tickers[picked]


def pf_held_table(
    items: list[portfolio.Holding], held: list[portfolio.Holding]
) -> None:
    with st.container(border=True):
        headed("**보유 종목**", "셀 선택 시 해당 종목 상세 분석 화면 이동")
        pf_table(
            pf_holdings_table(items),
            "보유 종목",
            signed=["평가손익", "수익률"],
            key="pf_held_table",
            on_select=partial(pf_jump, [h.ticker for h in held]),
            selection_mode="single-cell",
        )
        unknown = portfolio.unpriced(items)
        if unknown:
            st.caption(f"평가액 미확인: {', '.join(unknown)}")


def pf_contribution_chart(points: list[dict[str, str | float]]) -> alt.Chart:
    if shared.app.demo:
        points = [point | {"pnl": demo.VALUE} for point in points]
    up = alt.datum.pnl > 0
    p = palette.current()
    return (
        alt.Chart(alt.Data(values=points))
        .mark_bar()
        .encode(
            x=alt.X("pnl:Q", title=None, axis=alt.Axis(format=PF_AXIS)),
            y=alt.Y("ticker:N", sort="-x", title=None),
            color=alt.when(up).then(alt.value(p.up)).otherwise(alt.value(p.down)),
            tooltip=[
                alt.Tooltip("ticker:N", title="종목"),
                alt.Tooltip("pnl:Q", title="손익", format=PF_AXIS),
            ],
        )
        .properties(height=max(CHART_HEIGHT // 2, 28 * len(points)))
    )


PF_NO_LINE = "2거래일 이상 가격 기록 없음"


def pf_gap(value: perf.Gap | None) -> str:
    if value is None:
        return PF_DASH
    return f"{'연 ' if value.annualised else ''}{pf_pp(value.value)}"


def pf_benchmark_table(
    time: perf.Annualised | None, lines: dict[str, pd.Series]
) -> None:
    rows: list[tuple[str, perf.Annualised | None]] = [
        (name, shared.app.calc.time_weighted(line)) for name, line in lines.items()
    ]
    pf_table(
        {
            "구분": ["포트폴리오 TWR", *(name for name, _ in rows)],
            "누적": [
                pf_percent(a.cumulative, signed=True) if a else PF_DASH
                for a in (time, *(a for _, a in rows))
            ],
            "연환산": [
                pf_percent(a.annual, signed=True)
                if a and a.annual is not None
                else PF_DASH
                for a in (time, *(a for _, a in rows))
            ],
            "TWR − 지수": [
                PF_DASH,
                *(
                    pf_gap(shared.app.calc.gap(time, a)) if time and a else PF_DASH
                    for _, a in rows
                ),
            ],
        },
        "벤치마크 비교",
        signed=["누적", "연환산", "TWR − 지수"],
        helps={
            "TWR − 지수": "[정의] 포트폴리오 시간가중수익률(TWR)과 벤치마크 지수 수익률"
            " 격차\n\n"
            "[해석] 양(+)은 시장 지수 대비 초과성과 달성\n\n"
            "[산출] 포트폴리오 TWR − 지수 수익률(둘 다 1년 이상이면 연환산 기준, 그"
            " 외 누적 기준)"
        },
    )


def pf_trading(result: portfolio.Trading) -> None:
    pf_metric(
        "매매 효과",
        pf_gap(result.effect),
        f"평가액 {pf_money(result.actual - result.held, signed=True)}",
        delta_color=pf_color(result.actual - result.held),
        delta_arrow="off",
        help="[정의] 초기 비중 유지(Buy & Hold) 가정 대비 실제 매매·리밸런싱의 수익률"
        " 기여도\n\n"
        "[해석] 양(+)은 종목 교체·비중 조절을 통한 가치 증대\n\n"
        "[산출] 실제 금액가중수익률(MWR) − 첫 거래일 구성 고정(입출금 비례 매매·배당"
        " 반영) 가상 MWR(하단 금액은 실제 평가액 − 무매매 평가액)",
    )


def pf_return_metrics(
    totals: portfolio.Totals,
    time: perf.Annualised | None,
    money: perf.Annualised | None,
    trading: portfolio.Trading | None,
) -> None:
    with st.container(horizontal=True, key="pf_return_cards"):
        cells = (
            (
                "실현손익",
                totals.realized,
                totals.realized_return,
                "[정의] 매도 체결로 확정된 누적 손익 금액 및 매도 원가 대비 수익률\n\n"
                "[해석] 양(+)은 확정 매도 이익, 음(-)은 확정 매도 손실\n\n"
                "[산출] Σ(매도 대금 − 매도 수수료 − 매도 수량 × 이동평균 단가)",
            ),
            (
                "총손익",
                totals.pnl,
                totals.total_return,
                "[정의] 실현손익과 미실현 평가손익의 총합 및 누적 매수 원금(수수료"
                " 포함) 대비 수익률\n\n"
                "[해석] 포트폴리오 전체 거래에서 발생한 순자산 증감 총액\n\n"
                "[산출] 실현손익 + 평가손익(하단 비율은 수수료 포함 누적 총매수액"
                " 대비)",
            ),
        )
        for label, amount, ratio, note in cells:
            pf_metric(
                label,
                pf_money(amount, signed=True),
                pf_percent(ratio, signed=True),
                delta_color=pf_color(amount),
                delta_arrow="off",
                help=note,
            )
        pf_metric(
            "TWR 누적",
            pf_percent(time.cumulative, signed=True) if time else PF_DASH,
            pf_annual(time) if time and time.annual is not None else None,
            delta_color="off",
            delta_arrow="off",
            help="[정의] 현금 유출입 왜곡을 제거한 시간가중수익률(Time-Weighted"
            " Return)\n\n"
            "[해석] 순수 종목 선택 및 자산배분 운용 성과 평가\n\n"
            "[산출] 입출금 영향 제거·배당 포함 일별 수익률의 복리 연쇄 곱(1년 이상 시"
            " 하단 연환산 표기, 전량 매도 시 최종 거래일 종료)",
        )
        timing = perf.gap(money, time) if money and time else None
        pf_metric(
            "MWR",
            pf_annual(money),
            f"TWR 대비 {pf_gap(timing)}" if timing else None,
            delta_color="off",
            delta_arrow="off",
            help="[정의] 자금 유출입 규모·시점을 반영한 금액가중수익률(내부수익률,"
            " IRR, 수익률 미정 시 —)\n\n"
            "[해석] 투자자 체감 수익률, 하단(MWR − TWR) 양(+)은 자금 투입·회수"
            " 타이밍 우위\n\n"
            "[산출] 매수·매도·배당 현금흐름의 순현재가치(NPV)를 현재 평가액과"
            " 일치시키는 할인율(1년 미만 비연환산, 하단은 둘 다 1년 이상이면 연환산"
            " 수익률 차이)",
        )
        if trading is not None:
            pf_trading(trading)


def pf_realized_table(sales: tuple[portfolio.Sale, ...]) -> dict[str, list[str]]:
    rows = sorted(sales, key=lambda s: s.trade.day, reverse=True)
    return {
        "매도일": [s.trade.day.isoformat() for s in rows],
        "종목": [s.trade.ticker for s in rows],
        "수량": [pf_shares(s.trade.quantity) for s in rows],
        "평균단가": [pf_money(s.average) for s in rows],
        "매도가": [pf_money(s.trade.price) for s in rows],
        "실현손익": [pf_money(s.pnl, signed=True) for s in rows],
        "수익률": [pf_percent(s.return_rate, signed=True) for s in rows],
        "보유일": [f"{s.held_days:,}" for s in rows],
        "구분": [pf_tag(s.pnl) for s in rows],
        "메모": [s.trade.memo for s in rows],
    }


def pf_tag(pnl: float) -> str:
    if not math.isfinite(pnl):
        return portfolio.UNKNOWN
    return "익절" if pnl > 0 else ("손절" if pnl < 0 else "본전")


def pf_closed_table(items: list[portfolio.Holding]) -> dict[str, list[str]]:
    done = [h.position for h in items if not h.is_open]
    return {
        "종목": [p.ticker for p in done],
        "실현손익": [pf_money(p.realized, signed=True) for p in done],
        "원가 대비": [
            pf_percent(p.realized / p.sold if p.sold else math.nan, signed=True)
            for p in done
        ],
        "첫 매수": [p.first.isoformat() if p.first else "—" for p in done],
        "마지막 매도": [p.last_sell.isoformat() if p.last_sell else "—" for p in done],
    }


def pf_realized(p: portfolio.Portfolio, items: list[portfolio.Holding]) -> None:
    sales = shared.app.calc.ledger(p).sales
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(
            "**종목별 손익** · 실현 + 평가",
            "[정의] 개별 종목의 누적 실현손익 및 현재 평가손익 합산 금액\n\n"
            "[해석] 전량 매도 청산 종목을 포함한 종목별 총손익 기여 규모 비교\n\n"
            "[산출] 종목별 누적 실현손익 + 미실현 평가손익",
        )
        points = portfolio.contribution_points(items)
        if points:
            st.altair_chart(pf_contribution_chart(points), alt="종목별 손익")
        else:
            st.caption("손익 산출 가능 종목 없음")
    with right, st.container(border=True, height="stretch"):
        headed(
            "**연도별 실현손익**",
            "[정의] 매도 체결일 기준 연도·월별 확정 실현손익 집계\n\n"
            "[해석] 기간별 매도 차익 실현 규모 추이 파악\n\n"
            "[산출] 매도 거래일이 속한 연도(하단 펼침 시 월)별 실현손익 합산",
        )
        pf_period_table(sales, "연도", monthly=False)
        monthly_box = st.expander("월별", key="pf_monthly_exp", on_change="rerun")
        if monthly_box.open:
            with monthly_box:
                pf_period_table(sales, "월", monthly=True)
    if sales:
        with st.container(border=True):
            st.markdown(f"**실현손익 내역** · 매도 {len(sales)}건")
            pf_table(
                pf_realized_table(sales),
                "실현손익 내역",
                signed=["실현손익", "수익률", "구분"],
                helps={
                    "평균단가": "[정의] 매도 시점 해당 종목의 이동평균 매수 원가(매수"
                    " 수수료 포함)",
                    "수익률": "[정의] 매도 주식 이동평균 원가 대비 실현 수익률",
                    "보유일": "[정의] 해당 포지션 최초 매수일로부터 매도일까지의 경과"
                    " 달력일수",
                },
            )
    closed = pf_closed_table(items)
    if closed["종목"]:
        with st.container(border=True):
            headed(
                "**전량 매도한 종목**",
                "[정의] 현재 잔여 보유 수량이 0인 완전 청산 종목의 누적 실현성과\n\n"
                "[해석] 과거 청산 포지션의 최종 확정 손익 확인\n\n"
                "[산출] 청산 종목별 누적 실현손익 합계 및 누적 매도 원가 대비 수익률",
            )
            pf_table(
                closed,
                "전량 매도한 종목",
                signed=["실현손익", "원가 대비"],
                helps={
                    "원가 대비": "[정의] 누적 실현손익 ÷ 누적 매도 주식 이동평균 원가"
                },
            )


def pf_returns(p: portfolio.Portfolio, prices: portfolio.Prices) -> None:
    key = pf_trades_key(p)
    items = shared.app.calc.holdings(p, prices.close)
    totals = portfolio.totals(items)
    curve = cached(
        "twr",
        key,
        partial(shared.app.calc.twr, p, prices.close, prices.adjusted),
    )
    time = shared.app.calc.time_weighted(curve)
    frame = cached("history", key, partial(shared.app.calc.history, p, prices.close))
    money = cached(
        "mwr",
        key,
        partial(
            shared.app.calc.money_weighted,
            p,
            prices.close,
            prices.adjusted,
            totals.value,
            pd.DatetimeIndex(frame.index),
        ),
    )
    result = cached(
        "trading",
        key,
        partial(shared.app.calc.trading, p, prices.close, prices.adjusted),
    )
    pf_return_metrics(totals, time, money, result)
    days = pd.DatetimeIndex(curve.index)
    lines = cached(
        "bench",
        key,
        partial(shared.app.calc.benchmark_lines, prices.close, days),
    )
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(
            "**누적 수익률** · TWR와 지수",
            "[정의] 포트폴리오 시간가중수익률(TWR) 및 주요 벤치마크 지수 누적 수익률"
            " 비교(S&P 500은 배당 재투자 총수익지수)\n\n"
            "[해석] 현금 유출입 영향 배제 기준 시장 대비 상대 성과 추이\n\n"
            "[산출] 시작일 기준 일별 수익률 누적 복리 연결(1년 이상 시 표 내 연환산"
            " 병기)",
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
            "**누적 손익**",
            "[정의] 일별 총손익·실현손익·평가손익 누적 추이\n\n"
            "[해석] 매도 확정 이익(실현손익)과 보유분 미실현 손익(평가손익)의 구성"
            " 변화\n\n"
            "[산출] 총손익 = 평가액 − 순투입, 실현손익 = 누적 매도 손익, 평가손익 ="
            " 총손익 − 실현손익",
        )
        if len(frame) > 1:
            points = portfolio.line_points(portfolio.pnl_lines(frame))
            st.altair_chart(
                pf_line_chart(points, PF_AXIS), height="stretch", alt="누적 손익"
            )
        else:
            st.caption(PF_NO_LINE)
    pf_realized(p, items)


def pf_period_table(
    sales: tuple[portfolio.Sale, ...],
    label: str,
    *,
    monthly: bool,
) -> None:
    by = portfolio.realized_by(sales, monthly=monthly)
    if not by:
        st.caption("매도 거래 없음")
        return
    pf_table(
        {
            label: [period for period, _ in by],
            "실현손익": [pf_money(v, signed=True) for _, v in by],
        },
        f"{label}별 실현손익",
        signed=["실현손익"],
    )


PF_MODES = {"mine": "내 포트폴리오", "stock": "종목 분석", "investors": "유명 투자자"}


def pf_look_table(look: lookthrough.Lookthrough) -> None:
    rows = look.rows
    pf_table(
        {
            "종목": [r.ticker for r in rows],
            "직접": [pf_percent(r.direct) if r.direct else PF_DASH for r in rows],
            "경유": [
                pf_percent(sum(r.via.values())) if r.via else PF_DASH for r in rows
            ],
            "합계": [pf_percent(r.total) for r in rows],
            "경유 ETF": [
                " · ".join(f"{t} {pf_percent(v, dec=2)}" for t, v in r.via.items())
                or PF_DASH
                for r in rows
            ],
        },
        "실질 보유",
    )


def pf_look_funds(
    w: pd.Series,
    look: lookthrough.Lookthrough,
    comps: dict[str, etf.Composition],
) -> None:
    funds = pf_funds()
    names = [
        n
        for n in portfolio.labels(w)
        if n in comps or n in look.residual or portfolio.is_fund(funds.get(n))
    ]
    if not names:
        return

    def picked(name: str, show: Callable[[etf.Composition], str]) -> str:
        return show(comps[name]) if name in comps else PF_DASH

    with st.container(border=True, height="stretch"):
        st.markdown("**ETF별 잔여**")
        pf_table(
            {
                "ETF": names,
                "보유 비중": [pf_percent(float(cast("float", w[n]))) for n in names],
                "잔여 비중": [
                    pf_percent(look.residual[n]) if n in look.residual else PF_DASH
                    for n in names
                ],
                "구성 기준일": [picked(n, lambda c: c.as_of[:10]) for n in names],
                "계산": ["펼침" if n in look.residual else "한 자산" for n in names],
            },
            "ETF별 잔여",
            helps={
                "잔여 비중": "[정의] ETF 보유 비중 중 개별 주식으로 분해되지 않은"
                " 미분해 몫(채권·현금·비상위 종목 등)\n\n"
                "[산출] ETF 보유 비중 − 분해된 상위 구성 주식 비중 합계",
                "구성 기준일": "[정의] ETF 구성 종목 공시 데이터의 포트폴리오 기준"
                " 일자",
                "계산": "[정의] 실질 보유 분해 반영 방식(펼침: 구성 주식으로 분해"
                " 반영, 한 자산: 구성 데이터 부재 또는 주식 비중 합 불일치로 단일 자산"
                " 유지)",
            },
        )


def pf_look_risk(s: Subject, totals: dict[str, float], stamp: Key) -> None:
    extra = pf_look_prices(tuple(totals), state.get("pf_prices_stamp", 0))

    def risk() -> tuple[dict[str, float], float]:
        days = pd.DatetimeIndex(s.rets.index)
        names = list(totals)
        stocks = shared.app.calc.look_returns(extra.adjusted, names, days)
        held = portfolio.window(s.rets.dropna(), PF_LOOK_MONTHS)
        return lookthrough.risk_contributions(totals, stocks, held)

    found, left = cached("risk", (*s.key, *stamp), risk)
    order = sorted(found, key=lambda t: -found[t])
    headed(
        f"**위험 기여** · 최근 {PF_LOOK_MONTHS}개월",
        "[정의] 실질 보유(직접+ETF 경유) 기준 종목별 포트폴리오 변동성 기여 비중\n\n"
        "[해석] 합계 비중 초과 시 해당 종목에 실질 위험 집중, 음(-)은 변동성 상쇄\n\n"
        f"[산출] 최근 {PF_LOOK_MONTHS}개월 일간 수익률 공분산 기반 한계 위험 기여도"
        " 합산(시계열 부족 종목 및 ETF 잔여분은 잔여 행 통합)",
    )
    pf_table(
        {
            "종목": [*order, lookthrough.RESIDUAL],
            "합계 비중": [
                *(pf_percent(totals[t]) for t in order),
                pf_percent(1 - sum(totals[t] for t in order)),
            ],
            "위험 기여": [
                *(pf_percent(found[t], signed=True) for t in order),
                pf_percent(left, signed=True),
            ],
        },
        "위험 기여",
    )


def pf_look_shock(s: Subject, totals: dict[str, float]) -> None:
    compact = pf_compact(s.value)
    headed(
        "**종목 충격**",
        "[정의] 단일 실질 보유 종목의 가격 충격 시 포트폴리오 전체 손익 민감도\n\n"
        "[해석] 직접 보유 및 ETF 내재 보유분을 합산한 실질 노출 위험 측정\n\n"
        "[산출] 타 종목 가격 불변 가정 하 (대상 종목 합계 비중 × 가정 변동 폭)",
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        ticker = st.selectbox("종목", list(totals), key="pf_look_ticker", width=200)
        move = st.number_input(
            "변동 폭 (%)",
            value=-20.0,
            step=5.0,
            format="%.1f",
            key="pf_look_move",
            width=160,
        )
        change = lookthrough.stock_shock(totals, ticker or "", move / 100)
        pf_metric(
            "평가액 변화",
            pf_money(change * s.value, compact=compact, signed=True),
            pf_percent(change, signed=True),
            delta_color=pf_color(change),
            delta_arrow="off",
            help="[정의] 선택 종목 변동 폭 적용 시 예상 포트폴리오 평가액 증감액 및"
            " 수익률\n\n"
            "[해석] 단일 종목 급락·급등 시 실질 계좌 충격 규모\n\n"
            "[산출] 현재 포트폴리오 총평가액 × (종목 합계 비중 × 변동 폭)",
        )


def pf_lookthrough(s: Subject) -> None:
    if s.w.empty:
        st.info("가격 확인 보유 종목 없음")
        return
    if portfolio_fetcher().look_progress.running:
        st.info("ETF 구성 데이터 수신 중")
        return
    look, comps = pf_look(s.w)
    totals = look.totals()
    stamp = tuple((t, c.as_of, c.accession) for t, c in sorted(comps.items()))
    left = sum(look.residual.values())
    with st.container(horizontal=True, key="pf_look_cards"):
        pf_metric(
            "유효 종목 수",
            pf_num(quant.effective_holdings(np.array(list(totals.values()))), ".1f")
            if totals
            else PF_DASH,
            help="[정의] 실질 보유 비중의 역 허핀달-허쉬만 지수(Effective Number of"
            " Holdings), 동일 비중 포트폴리오 환산 등가 종목 수\n\n"
            "[해석] 낮을수록 소수 종목 집중\n\n"
            "[산출] 1 ÷ Σ(잔여 제외 정규화 실질 보유 비중²)",
        )
        pf_metric(
            "잔여 비중",
            pf_percent(left),
            help="[정의] ETF 보유분 중 개별 주식으로 분해되지 않은 미분해 비중 총합\n\n"
            "[해석] 클수록 하단 실질 보유 지표의 전체 포트폴리오 설명 범위 축소\n\n"
            "[산출] 보유 ETF별 미분해 잔여 비중 합산",
        )
    if not totals:
        pf_look_funds(s.w, look, comps)
        return
    with st.container(border=True):
        headed(
            f"**실질 보유** · {len(totals):,}종목",
            "[정의] 직접 보유 주식과 ETF 내재 구성 주식을 통합한 종목별 실질 노출"
            " 비중\n\n"
            "[해석] 복수 ETF 중복 편입에 따른 숨은 특정 종목 집중도 파악\n\n"
            "[산출] 합계 비중 = 직접 비중 + Σ(ETF 비중 × ETF 내 종목 비중), ETF별"
            f" 상위 누적 {investors.COVER_SHARE:.0%} 또는 최대"
            f" {investors.COVER_MAX}종목까지 분해",
        )
        pf_look_table(look)
    left_col, right_col = st.columns(2)
    with left_col, st.container(border=True, height="stretch"):
        headed(
            "**섹터 비중**",
            "[정의] 실질 보유 합계 비중의 섹터별 집계\n\n"
            "[해석] 직접 보유 및 ETF 경유분을 통합한 포트폴리오 산업 섹터 노출도\n\n"
            "[산출] 섹터별 실질 보유 합계 비중 합산(미분해 몫은 잔여 표기)",
        )
        sectors = lookthrough.sector_weights(look, shared.app.calc.look_sectors())
        pf_table(
            {
                "섹터": [sector_label(name) for name in sectors],
                "비중": [pf_percent(v) for v in sectors.values()],
            },
            "섹터 비중",
        )
    with right_col:
        pf_look_funds(s.w, look, comps)
    with st.container(border=True):
        pf_look_risk(s, totals, stamp)
    with st.container(border=True):
        pf_look_shock(s, totals)


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
    if section == "lookthrough":
        s = pf_subject(p, prices)
        pf_ask(portfolio_fetcher().start_holdings, portfolio.labels(s.w))
        pf_lookthrough(s)
        return
    top = st.container(horizontal=True, vertical_alignment="bottom")
    row = st.container(horizontal=True, vertical_alignment="bottom")
    with row:
        months = pf_periods("pf_horizon" if section == "simulation" else None)
    s = pf_subject(p, prices)
    if section == "stats":
        pf_stats(s, months)
        compare = partial(pf_book_heatmap, book, prices, months)
        pf_correlation(s, prices, months, compare)
        pf_component(s, prices, months)
        pf_style_box(s, prices, "pf_style")
    else:
        sized = portfolio.sizes(shared.app.calc.load_facts(), pf_funds())
        pf_ask(
            portfolio_fetcher().start_facts,
            [t for t in portfolio.labels(s.w) if t not in sized],
        )
        pf_simulation(s, prices, months, top, p)


def pf_view(
    book: portfolio.Book, p: portfolio.Portfolio, prices: portfolio.Prices
) -> None:
    section = st.segmented_control(
        "보기",
        list(PF_SECTIONS),
        format_func=PF_SECTIONS.__getitem__,
        default="returns",
        required=True,
        key="pf_section",
        label_visibility="collapsed",
        bind="query-params",
    )
    pf_section(section or "returns", book, p, prices)


def pf_unreplayed(book: portfolio.Book, splits: portfolio.Splits) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in book.portfolios:
        try:
            _ = portfolio.replay(portfolio.split_adjusted(p, splits).trades)
        except portfolio.BookError as error:
            out[p.id] = str(error)
    return out


def pf_broken(
    p: portfolio.Portfolio, reason: str, splits: portfolio.Splits, fetching: bool
) -> None:
    unsplit = sorted({t.ticker for t in p.trades} - splits.keys())
    st.error(reason)
    if unsplit and not shared.app.demo:
        if fetching:
            st.info("액면분할 내역 수신 중")
        else:
            st.warning(f"액면분할 내역 수신 실패: {', '.join(unsplit)}")
    st.button(
        "거래 추가",
        icon=":material/add:",
        key="pf_trade_open",
        on_click=pf_trade_open,
    )
    with st.expander("거래 내역", expanded=True):
        pf_trade_log(p)
    pf_trade_dialog(p)


def portfolio_page() -> None:
    try:
        book = portfolio.load_book(shared.app.root)
    except portfolio.BookError as error:
        st.error(str(error))
        return
    missing = portfolio.missing_packages()
    fetcher = portfolio_fetcher()
    state.pf_prices_stamp = fetcher.stamp()
    prices = pf_prices(str(shared.app.root), state.pf_prices_stamp)
    if state.get("refresh_page"):
        pf_points.clear()
    if portfolio.book_tickers(book) and not shared.app.demo and not missing:
        fetcher.start(fetcher.untried(portfolio.wanted(book)))
    splits = pf_splits()
    broken = pf_unreplayed(book, splits)
    usable = portfolio.adjusted_book(
        book.model_copy(
            update={
                "portfolios": tuple(q for q in book.portfolios if q.id not in broken)
            }
        ),
        splits,
    )
    if jump := state.pop("pf_jump_stock", ""):
        state.pf_mode = "stock"
        state.stock_error = ""
        state.stock_ticker = jump
        state.stock_input = jump
        state.stock_held = jump
        st.query_params["stock_input"] = jump
    mode = st.segmented_control(
        "구분",
        list(PF_MODES),
        format_func=PF_MODES.__getitem__,
        default=None if "pf_mode" in state else "mine",
        required=True,
        key="pf_mode",
        label_visibility="collapsed",
        bind="query-params",
    )
    if mode == "investors":
        inv_view(usable, prices, fetcher, missing)
        return
    if mode == "stock":
        stock_view(usable, prices, fetcher, missing)
        return
    p = pf_selector(book)
    pf_dialogs(p)
    pf_bar(fetcher, book, missing)
    if p is None:
        st.info("등록된 포트폴리오 없음")
        return
    if p.id in broken:
        pf_broken(p, broken[p.id], splits, fetcher.progress.running)
        return
    shown = portfolio.split_adjusted(p, splits)
    pf_holdings(shown, prices, p)
    if not p.trades:
        return
    pf_part(partial(pf_view, usable, shown, prices))
