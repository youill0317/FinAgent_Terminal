# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false, reportUnknownMemberType=false
"""SEC Form 13F 기관투자자 보유 공시, 복제 성과 및 포트폴리오 비교 화면."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, cast

import altair as alt
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent import demo, investors, perf, portfolio
from finagent.demo import portfolio as demo_portfolio
from finagent.ui import palette, shared
from finagent.ui.components import DATE_AXIS, clear_selection, sector_label
from finagent.ui.pf_common import (
    CHART_HEIGHT,
    PF_DASH,
    PF_DONUT_SLICES,
    PF_SHORT,
    Key,
    Subject,
    cached,
    headed,
    pf_annual,
    pf_ask,
    pf_choose,
    pf_donut,
    pf_jobs,
    pf_key,
    pf_line_chart,
    pf_look,
    pf_metric,
    pf_no_prices,
    pf_part,
    pf_percent,
    pf_periods,
    pf_pp,
    pf_rates,
    pf_rates_key,
    pf_ratio,
    pf_ratio_or,
    pf_rows,
    pf_shares,
    pf_style,
    pf_table,
    pf_trades_key,
    pf_weight_chart,
    portfolio_fetcher,
)
from finagent.ui.pf_simulation import pf_simulation
from finagent.ui.pf_stats import (
    pf_compare_heatmap,
    pf_component,
    pf_correlation,
    pf_stats,
    pf_style_box,
)
from finagent.ui.shared import limit_note, sec_agent

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import date

    from streamlit.delta_generator import DeltaGenerator

    from finagent import portfolio_fetch

state = st.session_state

LIST_HEIGHT = 560
TABLE_KEY = "inv_table"


def list_height(rows: int) -> int:
    return min(LIST_HEIGHT, (rows + 1) * 35 + 3)


INV_SECTIONS = {
    "returns": "수익률",
    "stats": "통계",
    "simulation": "시뮬레이션",
    "compare": "비교",
}
INV_MONTHS = {
    "returns": 0,
    "stats": max(portfolio.WINDOWS.values()),
    "compare": max(portfolio.WINDOWS.values()),
}
INV_VARIANTS = {
    "filed": "제출일 복제",
    "top": f"상위 {investors.CONVICTION}종목 복제",
}
INV_THEORY = "이론값 (분기말)"
INV_CHANGE_UP = {"신규": True, "증가": True, "감소": False}

type InvData = tuple[
    investors.Filer | None,
    list[investors.Quarter],
    dict[str, investors.Link],
    dict[str, investors.Sector],
]


@st.cache_resource(show_spinner=False)
def investor_fetcher() -> investors.Fetcher:
    return investors.Fetcher()


@st.cache_resource(show_spinner=False, max_entries=16)
def inv_load(cik: str, stamp: tuple[int, ...]) -> InvData:
    del stamp
    if shared.app.demo:
        return demo_portfolio.investor()
    filer = investors.load_filer(cik)
    tables = investors.load_tables(filer) if filer is not None else []
    return (
        filer,
        investors.quarters(tables),
        investors.load_cusips(),
        investors.load_sectors(),
    )


def inv_filer(cik: str) -> investors.Filer | None:
    if shared.app.demo:
        return demo_portfolio.investor()[0]
    return investors.load_filer(cik)


def inv_roster() -> list[investors.Investor]:
    if shared.app.demo:
        return [demo_portfolio.INVESTOR]
    return investors.roster(shared.app.root)


def inv_blocked(agent: str, missing: list[str]) -> str:
    if shared.app.demo:
        return ""
    if missing:
        packages = ", ".join(missing)
        return f"필수 패키지 누락({packages}): `{portfolio.INSTALL_HINT}`"
    if not agent:
        return (
            f"SEC 데이터 수신 설정 필요: `.env` 내 `{investors.AGENT_KEY}`(이름·이메일)"
        )
    if not agent.isascii():
        return f"`{investors.AGENT_KEY}` 영문(ASCII) 이름·이메일 입력 필요"
    return ""


def inv_style(text: str) -> str:
    if text in INV_CHANGE_UP:
        return f"color: {palette.current().signed(INV_CHANGE_UP[text])}"
    return pf_style(text)


def inv_priced(
    q: investors.Quarter, cusips: dict[str, investors.Link], prices: portfolio.Prices
) -> pd.Series:
    w = investors.analysis_weights(q.rows, cusips)
    return cast("pd.Series", w[w.index.isin(list(prices.tickers))])


def inv_open_add() -> None:
    state.inv_dialog = True
    state.inv_new_cik = ""


def inv_close() -> None:
    state.inv_dialog = False


def inv_add(agent: str) -> str:
    try:
        cik = investors.clean_cik(state.inv_new_cik)
        name = investor_fetcher().check(cik, agent)
        investors.add_investor(shared.app.root, cik, name)
    except (investors.InvestorError, OSError) as error:
        return str(error)
    state.inv_picked = cik
    inv_close()
    return ""


def inv_dialog_body(agent: str) -> None:
    if not state.get("inv_dialog"):
        st.rerun()
    with st.form("inv_add_form", border=False):
        st.text_input(
            "CIK",
            key="inv_new_cik",
            max_chars=13,
            help="[정의] SEC 공시 제출 기관 고유 식별 번호(Central Index Key), EDGAR"
            " 기업·기관 검색을 통해 확인 가능\n\n"
            "[산출] SEC 부여 최대 10자리 숫자 식별자",
        )
        with st.container(horizontal=True, horizontal_alignment="right"):
            asked = st.form_submit_button("추가", type="primary", key="inv_add_ok")
            st.form_submit_button("취소", key="inv_cancel", on_click=inv_close)
    if asked:
        with st.spinner("EDGAR 공시 확인 중…"):
            error = inv_add(agent)
        if not error:
            st.rerun()
        st.error(error)


def inv_unpick() -> None:
    if TABLE_KEY in state:
        clear_selection(TABLE_KEY)


def inv_latest_quarter() -> None:
    state.pop("inv_quarter", None)
    inv_unpick()


def inv_selector(people: list[investors.Investor], frozen: bool) -> investors.Investor:
    labels: dict[str, str] = {}
    today = datetime.now().astimezone().date()
    for person in people:
        filer = inv_filer(person.cik)
        last = investors.latest(filer)
        stopped = " · 보고 중단" if investors.stale(filer, today) else ""
        labels[person.cik] = (
            f"{person.label} · {last.period} 기준 · 제출 {last.filed}{stopped}"
            if last
            else f"{person.label} · 미수신"
        )
    if "inv_picked" in state:
        state.inv_selected = state.pop("inv_picked")
        inv_latest_quarter()
    if state.get("inv_selected") not in labels:
        state.inv_selected = people[0].cik
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.selectbox(
            "투자자",
            list(labels),
            format_func=labels.__getitem__,
            key="inv_selected",
            on_change=inv_latest_quarter,
            persist_state="session",
            width=560,
            help="[정의] SEC Form 13F 분기별 보유 현황 공시 대상 기관투자자(분기 말"
            " 미국 상장 주식 매수(Long) 포지션 대상, 분기 종료 후"
            f" {investors.DEADLINE_DAYS}일 이내 제출)\n\n"
            "[산출] 분기 말 기준 보유분 집계(옵션 및 원금 단위 행 제외)",
        )
        person = next(p for p in people if p.cik == state.inv_selected)
        st.button(
            "CIK 추가",
            icon=":material/add:",
            key="inv_add",
            on_click=inv_open_add,
            disabled=frozen,
        )
        if not shared.app.demo and person not in investors.PRESETS:
            st.button(
                "삭제",
                icon=":material/delete:",
                key="inv_remove",
                type="secondary",
                on_click=investors.remove_investor,
                args=(shared.app.root, person.cik),
            )
    return person


def inv_bar(
    fetcher: investors.Fetcher,
    pfetch: portfolio_fetch.Fetcher,
    people: list[investors.Investor],
    cik: str,
    blocked: str,
) -> None:
    def jobs() -> list[tuple[str, portfolio_fetch.Progress]]:
        return [("가격", pfetch.progress), ("ETF 구성", pfetch.look_progress)]

    polling = fetcher.progress.running or any(job.running for _, job in jobs())

    @st.fragment(run_every=1.0 if polling else None)
    def body() -> None:
        progress = fetcher.progress
        now = jobs()
        ended = polling and not progress.running and not any(j.running for _, j in now)
        if ended and not state.get("settings_open"):
            st.rerun()
        filer = inv_filer(cik)
        st.caption(
            f"13F {filer.fetched:%Y-%m-%d %H:%M} 확인"
            if filer
            else "수신된 13F 공시 없음"
        )
        if progress.running:
            names = {p.cik: p.label for p in people}
            done = progress.done / progress.total if progress.total else 0.0
            count = f" {progress.done}/{progress.total}" if progress.total else ""
            st.progress(
                done,
                text=f"{names.get(progress.cik, progress.cik)} · {progress.stage}"
                f"{count}{limit_note()}",
            )
        elif progress.failed and progress.cik == cik:
            st.warning(progress.message)
        pf_jobs(now)
        if blocked:
            st.info(blocked)

    body()


def inv_since(found: list[investors.Quarter]) -> date:
    section: str = state.get("inv_section") or st.query_params.get("inv_section", "")
    if section == "simulation":
        return portfolio.HISTORY_START
    today = cast("pd.Timestamp", pd.Timestamp(datetime.now().astimezone().date()))
    back = today - pd.DateOffset(months=INV_MONTHS.get(section, 0))
    first = min(q.period for q in found)
    return min(back.date(), first) - timedelta(days=portfolio.GAP_DAYS)


def inv_prices(
    pfetch: portfolio_fetch.Fetcher,
    blocked: bool,
    cik: str,
    found: list[investors.Quarter],
    cusips: dict[str, investors.Link],
) -> bool:
    if blocked:
        return False
    wanted = [
        *investors.tickers(found, cusips),
        portfolio.BENCHMARK,
        portfolio.TOTAL_RETURN,
    ]
    since = inv_since(found)
    if state.get("inv_reprice") != cik:
        return pfetch.start(pfetch.untried(wanted, since), since)
    started = pfetch.start(wanted, since, force=True)
    if started:
        state.inv_reprice = ""
    return started


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
    st.caption(f"{sector_label(str(row['sector']))} · CUSIP {row['cusip']}")
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(
            "**분기별 비중** · 보고 가치 기준",
            "[정의] 분기별 13F 보고 가치 총합 내 해당 종목 편입 비중 추이\n\n"
            "[해석] 투자자의 분기별 포지션 확대·축소 이력 확인\n\n"
            "[산출] 분기별 종목 보고 가치 ÷ 분기 전체 13F 보고 가치 합계(미보유 분기는"
            " 0%)",
        )
        history = investors.weight_history(found, str(row["cusip"]))
        st.altair_chart(inv_weight_history(list(history)), alt="분기별 비중")
    with right, st.container(border=True, height="stretch"):
        st.markdown("**1년 주가** · 종가")
        ticker = str(row["ticker"])
        series = (
            portfolio.window(cast("pd.Series", prices.close[ticker]).dropna(), 12)
            if ticker in prices.close.columns
            else pd.Series(dtype="float64")
        )
        if series.empty:
            st.caption("수신된 가격 데이터 없음")
        else:
            points = portfolio.line_points({"종가": series})
            st.altair_chart(pf_line_chart(points, ",.2f"), alt=f"{ticker} 1년 주가")


def inv_holdings(
    found: list[investors.Quarter],
    cusips: dict[str, investors.Link],
    sectors: dict[str, investors.Sector],
    prices: portfolio.Prices,
    warns: list[str],
) -> None:
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
        *(f"정정 {day}" for day in q.amended),
        *([f"제외: {', '.join(left_out)}"] if left_out else []),
        *([f"미반영 정정 {q.ignored}건"] if q.ignored else []),
        *([f"미포함 비중 {pf_percent(excluded)}"] if excluded > 0 else []),
        f"가격 미연결 비중 {pf_percent(1 - excluded - priced)}",
    ]
    with st.container(horizontal=True, vertical_alignment="center"):
        headed("**현황**", None, warns)
        st.space("stretch")
        st.selectbox(
            "분기",
            periods,
            format_func=labels.__getitem__,
            key="inv_quarter",
            on_change=inv_unpick,
            width=360,
            label_visibility="collapsed",
        )
    st.caption(
        " · ".join(notes),
        help="[정의] 분석 제외(미포함) 및 시세 미매칭(가격 미연결) 비중\n\n"
        "[해석] 복제 포트폴리오에서 누락되는 공시 자산 비율\n\n"
        f"[산출] 미포함 = 보고 가치 상위 {investors.COVER_SHARE:.0%}·"
        f"{investors.COVER_MAX}종목 외 하위 비중, 가격 미연결 = 분석 대상 내 티커·가격"
        " 미확인 비중",
    )
    with st.container(horizontal=True, key="inv_now_cards"):
        pf_metric(
            "보고 가치",
            investors.dollars(q.value),
            help="[정의] 선택 분기 Form 13F 공시 보유 자산 총가치\n\n"
            "[해석] 투자자의 미국 주식 공시 운용 규모\n\n"
            "[산출] 옵션 및 원금 단위(PRN) 행을 제외한 보고 가치 합산",
        )
        pf_metric(
            "보고 종목 수",
            f"{len(q.rows):,}",
            help="[정의] 선택 분기 Form 13F 공시 보유 종목 수\n\n"
            "[해석] 포트폴리오 종목 분산 범위\n\n"
            "[산출] 옵션 및 원금 단위(PRN) 행을 제외한 공시 종목 수",
        )
        pf_metric(
            "상위 10 비중",
            pf_percent(investors.top_share(q.rows)),
            help="[정의] 보고 가치 상위 10개 종목의 합산 비중\n\n"
            "[해석] 높을수록 핵심 소수 종목 집중 투자 성향\n\n"
            "[산출] 상위 10개 종목 보고 가치 합계 ÷ 전체 보고 가치",
        )
    records = cast("list[dict[str, object]]", frame.to_dict("records"))
    left, right = st.columns([1, 2])
    with left, st.container(border=True, height="stretch"):
        headed(
            "**비중** · 보고 가치 기준",
            "[정의] 분기 말 보고 가치 기준 종목별 구성 비중"
            f"({PF_DONUT_SLICES}종목 초과 시 상위 {PF_DONUT_SLICES - 1}종목 외 기타"
            " 병합 표기)\n\n"
            "[산출] 개별 종목 보고 가치 ÷ 전체 보고 가치",
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
            "[정의] 분기 말 보고 가치의 섹터별 비중 집계("
            f"{investors.EXCLUDED}은 상위 {investors.COVER_SHARE:.0%}·"
            f"{investors.COVER_MAX}종목 외 하위 종목, {investors.UNKNOWN}은 섹터 미확인"
            " 종목)\n\n"
            "[산출] 섹터별 종목 보고 가치 합계 ÷ 전체 보고 가치",
        )
        st.altair_chart(
            pf_weight_chart(
                investors.sector_weights(frame).rename(sector_label), "섹터"
            ),
            alt="섹터 비중",
        )
    shown = pd.DataFrame(
        {
            "티커": [str(r["ticker"]) or "—" for r in records],
            "종목": [str(r["name"]) for r in records],
            "섹터": [sector_label(str(r["sector"])) for r in records],
            "주식 수": [pf_shares(cast("float", r["shares"])) for r in records],
            "가치": [investors.dollars(cast("float", r["value"])) for r in records],
            "비중": [pf_percent(cast("float", r["weight"])) for r in records],
            "직전 분기 대비": [str(r["change"]) for r in records],
        }
    )
    with st.container(border=True):
        headed("**보유 종목**", "셀 선택 시 해당 종목 분기별 비중·주가 상세 표시")
        event = st.dataframe(
            shown.style.map(inv_style, subset=["직전 분기 대비"]),
            hide_index=True,
            height=list_height(len(shown)),
            on_select="rerun",
            selection_mode="single-cell",
            key=TABLE_KEY,
            column_config={
                "종목": st.column_config.TextColumn(width="medium"),
                "직전 분기 대비": st.column_config.TextColumn(
                    help="[정의] 직전 분기 보고 대비 보유 주식 수 변화(신규·증가·감소·"
                    "유지)"
                ),
            },
            alt="보유 종목, 셀을 누르면 상세 보기",
        )
    gone = investors.sold_out(q, before)
    with st.container(border=True):
        headed(
            f"**청산** · 직전 분기 대비 {len(gone)}종목",
            "[정의] 직전 분기 보유 후 당기 13F 공시에서 전량 제외된 종목\n\n"
            "[해석] 분기 중 전량 매도 또는 공시 대상 제외 포지션\n\n"
            "[산출] 직전 분기 보유 목록 − 당기 분기 보유 목록 차집합",
        )
        if gone:
            pf_table(
                {
                    "종목": [r.name for r in gone],
                    "직전 주식 수": [pf_shares(r.shares) for r in gone],
                    "직전 가치": [investors.dollars(r.value) for r in gone],
                },
                "청산 종목",
            )
    cells = event.selection.cells
    if cells and cells[0][0] < len(records):
        row = records[cells[0][0]]
        title = f"{row['ticker'] or '—'} · {row['name']}"
        st.dialog(title, width="large", on_dismiss=partial(clear_selection, TABLE_KEY))(
            inv_stock_body
        )(found, row, prices)


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
    del key
    adjusted = _prices.adjusted
    filed = investors.filed_schedule(_found, _cusips, investors.buy_days(_filer))
    ends = investors.period_schedule(_found, _cusips)
    if top:
        filed = investors.top_schedule(filed, equal=equal)
        ends = investors.top_schedule(ends, equal=equal)
    made = shared.app.calc.clone(adjusted, filed)
    theory = shared.app.calc.rebalanced(adjusted, ends)
    if made.cumulative.empty:
        return made, theory
    start = cast("pd.Timestamp", made.cumulative.index[0])
    return made, portfolio.since(theory, start)


def inv_variant() -> tuple[bool, bool]:
    top = state.get("inv_variant") == "top"
    return top, top and bool(state.get("inv_equal"))


def inv_label(top: bool, equal: bool) -> str:
    label = INV_VARIANTS["top" if top else "filed"]
    return f"{label} · 같은 비중" if equal else label


def inv_controls(section: str) -> tuple[int, DeltaGenerator]:
    top = st.container(horizontal=True, vertical_alignment="bottom")
    row = st.container(horizontal=True, vertical_alignment="bottom")
    with row:
        months = (
            pf_periods("pf_horizon" if section == "simulation" else None)
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
            help="[정의] 13F 공시 공개 후 첫 종가 매수·차기 제출까지 보유 기준 복제"
            " 포트폴리오\n\n"
            f"[해석] 상위 {investors.CONVICTION}종목 복제는 핵심 고확신 종목 집중"
            " 추종\n\n"
            f"[산출] 보고 가치 상위 {investors.COVER_SHARE:.0%}·"
            f"{investors.COVER_MAX}종목 내 가격 확인 종목 비중을 100%로 정규화하여"
            " 산출",
        )
        if picked == "top":
            st.toggle(
                "같은 비중",
                key="inv_equal",
                help=f"[정의] 상위 {investors.CONVICTION}종목 동일"
                f" 비중(1/{investors.CONVICTION}) 배분 적용",
            )
    return months, top


def inv_subject(
    key: Key, data: InvData, prices: portfolio.Prices, *, top: bool, equal: bool
) -> Subject | None:
    filer, found, cusips, _ = data
    made, _ = inv_clone(key, top, equal, filer, found, cusips, prices)
    if len(made.cumulative) < 2 or made.weights.empty:  # noqa: PLR2004  # 수익률 산출 최소 2거래일 기준
        return None
    q = found[0]
    reported = inv_priced(q, cusips, prices)
    held = reported.reindex(portfolio.labels(made.weights))
    value = q.value * float(cast("float", held.sum()))
    return Subject(
        key=(*key, top, equal),
        label=inv_label(top, equal),
        rets=portfolio.daily(made.cumulative),
        w=made.weights,
        value=value,
        bench=partial(shared.app.calc.market_returns, prices.adjusted),
        trim=True,
        notes=(
            f"금액은 {q.period} 보고 가치 중 이 종목들의 몫"
            f" {investors.dollars(value)} 기준",
        ),
    )


def inv_chart(lines: dict[str, pd.Series], theory: pd.Series) -> alt.LayerChart:
    points = portfolio.line_points({INV_THEORY: theory})
    if shared.app.demo:
        points = [point | {"value": demo.SHARE} for point in points]
    grey = (
        alt.Chart(pf_rows(points))
        .mark_line(color=palette.current().muted, strokeDash=[4, 3])
        .encode(
            x=alt.X("day:T", title=None, axis=DATE_AXIS),
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
                pf_percent(s.turnover) if s.turnover is not None else PF_DASH
                for s in reversed(steps)
            ],
        },
        "리밸런싱",
        helps={
            "회전율": "[정의] 리밸런싱 시점 포트폴리오 종목 교체 비율(One-Way"
            " Turnover)\n\n"
            "[해석] 높을수록 분기별 편입 종목·비중 교체 폭 확대\n\n"
            "[산출] Σ|신규 목표 비중 − 가격 변동 반영 직전 보유 비중| ÷ 2(최초 매수·"
            "재진입 시 — 표기)"
        },
    )


def inv_returns(  # noqa: PLR0913  # 복제 수익률 화면 렌더링 컨텍스트·옵션 일괄 전달
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
    if len(filed) < 2:  # noqa: PLR2004  # 수익률 시계열 최소 2거래일 기준
        pf_no_prices()
        return
    label = inv_label(top, equal)
    q = found[0]
    today = datetime.now().astimezone().date()
    turns = [s.turnover for s in made.steps if s.turnover is not None]
    start = cast("pd.Timestamp", filed.index[0])
    time = shared.app.calc.time_weighted(filed)
    with st.container(horizontal=True, key="inv_return_cards"):
        pf_metric(
            f"누적 · {label}",
            pf_percent(portfolio.last(filed), signed=True),
            pf_annual(time) if time and time.annual is not None else None,
            delta_color="off",
            delta_arrow="off",
            help="[정의] 복제 개시일 이후 배당 재투자 반영 누적 총수익률\n\n"
            "[해석] 13F 공시 추종 전략의 장기 누적 성과\n\n"
            "[산출] 수정 종가 기준 리밸런싱 구간별 일별 수익률 복리 연쇄 곱(1년 이상"
            " 시 하단 연환산 병기)",
        )
        pf_metric(
            "복제 보유 종목 수",
            f"{len(made.weights):,}",
            help="[정의] 현재 복제 포트폴리오에 편입된 시세 확인 종목 수\n\n"
            "[해석] 실제 복제 수익률 산출에 반영 중인 종목 수\n\n"
            "[산출] 최근 분기 분석 대상 중 가격 시계열 유효 종목 집계",
        )
        pf_metric(
            "가격 연결 비중",
            pf_percent(float(cast("float", inv_priced(q, cusips, prices).sum()))),
            help="[정의] 최근 분기 전체 보고 가치 중 가격 시계열이 연결된 종목 비중"
            " 합계\n\n"
            "[해석] 낮을수록 원 공시 포트폴리오 대비 복제 대표성 저하\n\n"
            "[산출] 시세 확인 편입 종목 보고 가치 합계 ÷ 최근 분기 전체 보고 가치",
        )
        pf_metric(
            "분기말 이후 경과일",
            f"{(today - q.period).days:,}일",
            f"제출까지 {(q.filed - q.period).days}일",
            delta_color="off",
            delta_arrow="off",
            help="[정의] 최근 공시 기준 분기 말로부터 현재까지의 경과 달력일수\n\n"
            "[해석] 길수록 공시 시차에 따른 실제 현재 포지션과의 괴리 가능성 확대\n\n"
            "[산출] 현재일 − 최근 분기 기준일(하단은 제출일 − 분기 기준일)",
        )
        pf_metric(
            "평균 회전율",
            pf_percent(sum(turns) / len(turns)) if turns else PF_DASH,
            help="[정의] 분기별 13F 공시 리밸런싱 회전율의 산술평균\n\n"
            "[해석] 높을수록 분기별 포트폴리오 종목 교체 빈도 상승\n\n"
            "[산출] 유효 리밸런싱 시점별 편도 회전율(Σ|Δw| ÷ 2)의 기간 평균",
        )
    gap = pf_pp(portfolio.last(filed) - portfolio.last(theory))
    st.caption(
        f"{INV_THEORY} 누적"
        f" {pf_percent(portfolio.last(theory), signed=True)} · 차이 {gap}",
        help="[정의] 분기 말 즉시 리밸런싱 가정 이론 수익률 및 제출일 복제 수익률 간"
        " 격차\n\n"
        "[해석] 음(-)의 차이는 공시 제출 지연(최대"
        f" {investors.DEADLINE_DAYS}일)에 따른 추종 시차 비용\n\n"
        "[산출] 제출일 복제 누적 수익률 − 분기 말 종가 즉시 매수 가정 누적 수익률",
    )
    days = pd.DatetimeIndex(filed.index)
    lines = {
        label: filed,
        portfolio.BENCHMARK_LABEL: shared.app.calc.cumulative(
            shared.app.calc.market_returns(prices.adjusted, days)
        ),
    }
    for p in book.portfolios:
        if p.trades:
            twr = cached(
                "twr",
                pf_trades_key(p),
                partial(shared.app.calc.twr, p, prices.close, prices.adjusted),
            )
            lines[f"{p.name} TWR"] = portfolio.since(twr, start)
    missing = investors.untimed(filer)
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(
            f"**누적 수익률** · {start.date()}부터",
            f"[정의] 투자자 복제·{portfolio.BENCHMARK_LABEL} 총수익지수·내 포트폴리오"
            f" TWR 누적 수익률 추이(회색 점선: {INV_THEORY})\n\n"
            "[해석] 시장 및 내 포트폴리오 대비 투자자 추종 전략 상대 성과 비교\n\n"
            "[산출] 복제 개시일 기준 일별 누적 복리 수익률 시계열",
        )
        st.altair_chart(inv_chart(lines, theory), alt="복제 수익률")
    with right, st.container(border=True, height="stretch"):
        headed(
            f"**리밸런싱** · {len(made.steps)}회"
            + (f", 접수 시각 미상 제출 {missing}건" if missing else ""),
            "[정의] 분기 공시 제출별 리밸런싱 매수일·편입 종목 수·회전율 이력\n\n"
            "[산출] 제출일(장 마감 후 접수 또는 접수 시각 미상 시 익거래일) 종가 기준"
            " 목표 비중 재조정 내역",
        )
        inv_steps(made.steps)


@st.cache_data(show_spinner=False, max_entries=32)
def inv_fixed(
    key: Key,
    months: int,
    _q: investors.Quarter,
    _cusips: dict[str, investors.Link],
    _prices: portfolio.Prices,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    del key
    priced = inv_priced(_q, _cusips, _prices)
    rets = portfolio.window(
        shared.app.calc.asset_returns(_prices.adjusted, portfolio.labels(priced)),
        months,
    )
    w, dropped = shared.app.calc.covered(rets, priced)
    return rets, w, dropped


def inv_people(
    people: list[investors.Investor],
) -> list[tuple[investors.Investor, Key, InvData]]:
    out: list[tuple[investors.Investor, Key, InvData]] = []
    for person in people:
        stamp = investors.stamp()
        data = inv_load(person.cik, stamp)
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
    frame = cached(
        "lineup",
        (*book_key, months),
        partial(shared.app.calc.lineup, book, prices, months),
    ).copy()
    keys: list[object] = [book_key]
    for person, their, data in inv_people(people):
        their_rets, w, _ = inv_fixed(their, months, data[1][0], data[2], prices)
        if not w.empty:
            frame[person.name] = portfolio.weighted(their_rets, w)
            keys.append(their)
    pf_compare_heatmap(
        (*keys, months, "investors"),
        frame,
        f"**투자자·내 포트폴리오·{portfolio.BENCHMARK_LABEL}**",
        "[정의] 투자자 복제·내 포트폴리오·벤치마크 지수 간 일간 수익률 상관계수\n\n"
        "[해석] +1 근접 시 동조 움직임, 0 근접 시 독립 변동\n\n"
        "[산출] 투자자 최근 분기 비중 및 내 포트폴리오 현재 비중 고정 기준 축소"
        " 상관계수",
    )


def inv_league(
    people: list[investors.Investor],
    prices: portfolio.Prices,
    months: int,
    *,
    top: bool,
    equal: bool,
) -> None:
    series: dict[str, pd.Series] = {}
    keys: list[object] = []
    for person, their, data in inv_people(people):
        made, _ = inv_clone(their, top, equal, data[0], data[1], data[2], prices)
        if len(made.cumulative) > 1:
            daily = portfolio.daily(made.cumulative)
            series[person.label] = portfolio.window(daily, months)
            keys.append(their)
    label = inv_label(top, equal)
    headed(
        f"**투자자 비교** · {label}, {state.get('pf_window') or '3Y'}",
        "[정의] 등록 투자자별 복제 포트폴리오의 기간 수익률 및 샤프 지수 비교\n\n"
        "[해석] 동일 복제 규칙 하 투자자 간 위험 조정 성과 순위 확인\n\n"
        "[산출] 선택 기간 내 복제 일간 수익률 기반 누적 수익률·연환산 샤프·축소 샤프"
        " 산출",
    )
    if not series:
        st.caption("가격 확인 투자자 복제 데이터 없음")
        return
    days = (
        pd.DatetimeIndex(pd.concat(list(series.values())).index).unique().sort_values()
    )
    rates = pf_rates()
    rf = shared.app.calc.risk_free(rates, days)
    key = (*keys, top, equal, months, pf_rates_key(rates))
    found = cached("peers", key, partial(shared.app.calc.peers, series, rf))
    pooled = found.count > 1
    table = {
        "투자자": [r.name for r in found.rows],
        "기간 수익률": [pf_percent(r.total, signed=True) for r in found.rows],
        "샤프 (연)": [pf_ratio_or(r.sharpe) for r in found.rows],
        **(
            {"축소 샤프": [pf_ratio_or(r.shrunk) for r in found.rows]} if pooled else {}
        ),
        "일수": [f"{r.days:,}" for r in found.rows],
    }
    pf_table(
        table,
        "투자자 비교",
        signed=["기간 수익률"],
        helps={
            "축소 샤프": "[정의] 경험적 베이즈(Empirical Bayes) 축소 추정 연환산 샤프"
            " 지수\n\n"
            "[해석] 표본 오차·운의 영향을 완화한 보수적 성과 비교 지표\n\n"
            "[산출] 개별 샤프 지수를 추정 불확실성에 비례하여 집단 평균 방향으로 수축",
            "일수": "[정의] 무위험수익률 대조 유효 거래일수",
        },
    )
    if pooled:
        dsr = pf_percent(found.deflated) if found.deflated is not None else PF_SHORT
        st.caption(
            f"최고 샤프가 운 이상일 확률(DSR) {dsr}",
            help="[정의] 다중 비교 편향을 보정한 디플레이티드 샤프 지수(Deflated Sharpe"
            " Ratio) 확률\n\n"
            f"[해석] {found.count}명 중 최고 샤프 지수가 단순 우연(다중 검정)을"
            " 넘어설 통계적 신뢰도\n\n"
            "[산출] 왜도·첨도 및 비교 대상 수를 반영한 최고 샤프 통계량 누적확률"
            f"({perf.MIN_OBS}일 이상 시 산출)",
        )


def inv_correlation(  # noqa: PLR0913  # 상관행렬·리그 테이블 렌더링 인자 일괄 전달
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


def inv_ratios(
    key: Key, rets: pd.Series, prices: portfolio.Prices, months: int
) -> portfolio.Ratios:
    window = portfolio.window(rets, months)
    rates = pf_rates()
    rf = shared.app.calc.risk_free(rates, pd.DatetimeIndex(window.index))
    bench = partial(
        shared.app.calc.market_returns, prices.adjusted, pd.DatetimeIndex(window.index)
    )
    tag = (*key, months, "sp500", pf_rates_key(rates))
    cut = len(rets) - len(window)
    start = cast("pd.Timestamp", rets.index[cut - 1]) if cut > 0 else None
    return cached(
        "ratios", tag, lambda: shared.app.calc.ratios(window, rf, bench(), start)
    )


def inv_overlap(theirs: pd.Series, mine: pd.Series) -> None:
    mine_total = float(cast("float", mine.sum()))
    us = cast("pd.Series", mine / mine_total if mine_total > 0 else mine.iloc[:0])
    total = float(cast("float", theirs.sum()))
    their = theirs / total if total > 0 else theirs
    share = 1 - perf.active_share(their, us) if not us.empty else None
    overlap = investors.overlap(their, us)
    with st.container(border=True):
        headed(
            f"**겹치는 종목** · {len(overlap)}종목 · 겹침"
            f" {pf_percent(share) if share is not None else PF_DASH}",
            "[정의] 투자자 공시 종목과 내 실질 보유 간 공통 편입 종목 및 비중"
            " 중첩도(1 − Active Share)\n\n"
            "[해석] 100%는 완전 동일 구성, 0%는 공통 종목 전무\n\n"
            "[산출] 양측 비중 합계 100% 정규화 후 종목별 min(투자자 비중, 내 비중)"
            " 합산",
        )
        if overlap:
            pf_table(
                {
                    "티커": [t for t, _, _ in overlap],
                    "투자자 비중": [pf_percent(a) for _, a, _ in overlap],
                    "내 실질 보유 중": [pf_percent(b) for _, _, b in overlap],
                },
                "겹치는 종목",
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
        st.info("비교 대상 내 포트폴리오 없음")
        return
    pf_choose("내 포트폴리오", mine, "inv_mine")
    p = book.get(state.inv_mine)
    if p is None:
        return
    headed(
        "**보유 겹침**",
        "[정의] ETF 분해 반영 내 실질 보유 비중과 투자자 최근 분기 공시 비중 대조\n\n"
        "[해석] 직·간접 보유를 망라한 투자자 포트폴리오와의 포지션 동조성 확인\n\n"
        "[산출] 티커 확인된 투자자 13F 종목 비중과 내 포트폴리오 Look-through 합산 비중"
        " 비교",
    )
    theirs = investors.weights(found[0].rows, cusips)
    own = portfolio.weights(shared.app.calc.holdings(p, prices.close))
    pf_ask(portfolio_fetcher().start_holdings, portfolio.labels(own))
    if portfolio_fetcher().look_progress.running:
        st.info("ETF 구성 데이터 수신 중")
    else:
        look, _ = pf_look(own)
        inv_overlap(theirs, pd.Series(look.totals(), dtype="float64"))
    with st.container(border=True):
        headed(
            f"**위험 지표** · {state.get('pf_window') or '3Y'}",
            "[정의] 동일 기간 내 투자자 복제·내 포트폴리오(보유 기록)·벤치마크 지수"
            " 위험 지표 비교\n\n"
            "[해석] 수익성·변동성·낙폭·시장 민감도 종합 대조\n\n"
            "[산출] 각 시계열의 선택 구간 일간 수익률 기반 성과·위험 통계량 산출",
        )
        held = cached(
            "held",
            pf_trades_key(p),
            partial(shared.app.calc.held_returns, p, prices.close, prices.adjusted),
        )
        sp_days = portfolio.trading_days(prices.adjusted, [portfolio.TOTAL_RETURN])
        sp500 = shared.app.calc.market_returns(prices.adjusted, sp_days).dropna()
        rows = {
            s.label: (s.key, s.rets),
            p.name: (pf_trades_key(p), held),
        }
        if not sp500.empty:
            rows[portfolio.BENCHMARK_LABEL] = (pf_key("benchmark"), sp500)
        stats = {
            name: inv_ratios(key, rets, prices, months)
            for name, (key, rets) in rows.items()
        }
        windows = {
            name: portfolio.window(rets, months).dropna()
            for name, (_, rets) in rows.items()
        }
        betas = {
            name: shared.app.calc.market_beta(
                portfolio.window(rets, months), prices.adjusted
            )
            for name, (_, rets) in rows.items()
        }
        pf_table(
            {
                "구분": list(rows),
                "기간 수익률": [
                    pf_percent(shared.app.calc.compounded(w), signed=True)
                    if len(w)
                    else PF_DASH
                    for w in windows.values()
                ],
                "연 변동성": [
                    pf_percent(r.volatility) if r.volatility is not None else PF_DASH
                    for r in stats.values()
                ],
                "최대 낙폭": [
                    pf_percent(r.under.max_drawdown) if r.under else PF_DASH
                    for r in stats.values()
                ],
                "샤프 (연)": [pf_ratio_or(r.sharpe) for r in stats.values()],
                "베타": [
                    pf_ratio(b) if math.isfinite(b) else PF_DASH for b in betas.values()
                ],
                "일수": [f"{r.days:,}" for r in stats.values()],
            },
            "위험 지표 비교",
            signed=["기간 수익률"],
            helps={
                "기간 수익률": "[정의] 선택 구간 일간 수익률 복리 누적 수익률",
                "연 변동성": "[정의] 일간 초과수익률의 연환산 표준편차(높을수록 등락"
                " 확대)",
                "최대 낙폭": "[정의] 기간 내 고점 대비 최대 낙폭(MDD)",
                "샤프 (연)": "[정의] 연환산 변동성 단위당 무위험수익률 초과수익",
                "베타": f"[정의] {portfolio.BENCHMARK_LABEL} 가격지수 1% 변동 대비"
                " 민감도",
            },
        )


def inv_flags(filer: investors.Filer | None) -> list[str]:
    stopped = investors.stale(filer, datetime.now().astimezone().date())
    late = investors.notice(filer)
    return [
        *([f"최종 보고 분기({stopped}) 이후 후속 공시 없음"] if stopped else []),
        *(
            [f"최근 제출 공시는 보유 내역 미포함 {late.form}({late.filed})"]
            if late
            else []
        ),
    ]


def inv_section(  # noqa: PLR0913  # 섹션별 분석 렌더링 인자 일괄 전달
    section: str,
    key: Key,
    book: portfolio.Book,
    people: list[investors.Investor],
    data: InvData,
    prices: portfolio.Prices,
) -> None:
    months, above = inv_controls(section)
    top, equal = inv_variant()
    if section == "returns":
        inv_returns(key, data, book, prices, top=top, equal=equal)
        return
    s = inv_subject(key, data, prices, top=top, equal=equal)
    if s is None:
        pf_no_prices()
    elif section == "stats":
        pf_stats(s, months)
        inv_correlation(s, book, people, prices, months, top=top, equal=equal)
        pf_component(s, prices, months)
        pf_style_box(s, prices, "inv_style")
    elif section == "simulation":
        pf_simulation(s, prices, months, above)
    else:
        inv_compare(s, data, book, prices, months)


def inv_view_section(  # noqa: PLR0913  # 탭 전환 및 데이터 갱신 콜백 일괄 전달
    key: Key,
    book: portfolio.Book,
    people: list[investors.Investor],
    data: InvData,
    prices: portfolio.Prices,
    more: Callable[[], bool],
) -> None:
    section = st.segmented_control(
        "보기",
        list(INV_SECTIONS),
        format_func=INV_SECTIONS.__getitem__,
        default="returns",
        required=True,
        key="inv_section",
        label_visibility="collapsed",
        bind="query-params",
    )
    if more():
        st.rerun()
    inv_section(section or "returns", key, book, people, data, prices)


def inv_view(
    book: portfolio.Book,
    prices: portfolio.Prices,
    pfetch: portfolio_fetch.Fetcher,
    missing: list[str],
) -> None:
    agent = sec_agent()
    fetcher = investor_fetcher()
    blocked = inv_blocked(agent, missing)
    frozen = shared.app.demo or bool(blocked)
    people = inv_roster()
    person = inv_selector(people, frozen)
    if state.get("inv_dialog"):
        st.dialog("CIK 추가", on_dismiss=inv_close)(inv_dialog_body)(agent)
    cik = person.cik
    stamp = investors.stamp()
    data = inv_load(cik, stamp)
    refresh = bool(state.pop("refresh_page", False)) and not frozen
    if refresh:
        state.inv_reprice = cik
    if refresh or (not frozen and data[0] is None and not fetcher.tried(cik)):
        fetcher.start(cik, agent)
    found = data[1]
    more = partial(
        inv_prices, pfetch, frozen or fetcher.progress.running, cik, found, data[2]
    )
    if found:
        more()
    inv_bar(fetcher, pfetch, people, cik, blocked)
    if not found:
        return
    pf_part(
        "inv_holdings",
        partial(inv_holdings, found, data[2], data[3], prices, inv_flags(data[0])),
    )
    pf_part(
        "inv_section",
        partial(inv_view_section, pf_key(cik, stamp), book, people, data, prices, more),
    )
