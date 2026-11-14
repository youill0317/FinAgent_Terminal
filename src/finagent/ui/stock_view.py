# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false, reportUnknownMemberType=false
"""개별 종목 통계·시뮬레이션 및 포트폴리오 가상 편입(What-if) 효과 분석 화면."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import altair as alt
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent import demo, display, perf, portfolio, quant
from finagent.ui import palette, shared
from finagent.ui.pf_common import (
    CHART_HEIGHT,
    PF_DONUT_SLICES,
    PF_MIN_QUANTITY,
    PF_THIN,
    Key,
    Subject,
    cached,
    cached_draws,
    cached_paths,
    headed,
    pf_bar,
    pf_choose,
    pf_donut,
    pf_gaps,
    pf_key,
    pf_metric,
    pf_money,
    pf_no_prices,
    pf_part,
    pf_percent,
    pf_periods,
    pf_pp,
    pf_ratio,
    pf_shares,
    pf_subject,
)
from finagent.ui.pf_simulation import pf_simulation
from finagent.ui.pf_stats import pf_stats, pf_style_box

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray
    from streamlit.delta_generator import DeltaGenerator

    from finagent import portfolio_fetch

state = st.session_state

STOCK_SECTIONS = {
    "stats": "통계",
    "simulation": "시뮬레이션",
    "whatif": "편입 효과",
}
STOCK_INDICES = (portfolio.BENCHMARK, portfolio.TOTAL_RETURN)
WHATIF_START = 0.05
WHATIF_TOP = 0.3
WHATIF_STEPS = 40
WHATIF_CURVE = 150
WHATIF_BARS = 130
WHATIF_LEVELS = portfolio.REACH_STEPS
WHATIF_WHEN = ("지금", "매수 후")
WHATIF_REST = "기타"


def stock_choose(ticker: str) -> None:
    state.stock_error = ""
    if ticker != state.get("stock_ticker"):
        state.stock_ticker = ticker


def stock_enter() -> None:
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
    return sorted(
        {
            h.ticker
            for p in book.portfolios
            for h in shared.app.calc.holdings(p, prices.close)
            if h.is_open
        }
    )


def stock_fetch(fetcher: portfolio_fetch.Fetcher, ticker: str, *, retry: bool) -> None:
    todo = fetcher.untried([ticker, *STOCK_INDICES])
    if retry and ticker not in todo:
        todo = [ticker, *todo]
    if ticker not in todo:
        return
    with st.spinner(f"{ticker} 가격 수신 중…"):
        try:
            saved = fetcher.fetch_now(todo)
        except Exception as error:  # noqa: BLE001  # 외부 가격 공급자 예외 포착 후 상태 표시
            state.stock_error = f"가격 수신 실패: {type(error).__name__}"
        else:
            if ticker not in saved.tickers:
                state.stock_error = f"{ticker} 가격 수신 실패"
    st.rerun()


def stock_share(ticker: str, prices: portfolio.Prices) -> float:
    price, _ = shared.app.calc.last_close(prices.close, ticker)
    return price


def stock_subject(ticker: str, prices: portfolio.Prices) -> Subject:
    rets = shared.app.calc.asset_returns(prices.adjusted, [ticker])
    value = stock_share(ticker, prices)
    return Subject(
        key=pf_key("stock", ticker),
        label=ticker,
        rets=cast("pd.Series", rets[ticker]).dropna(),
        w=pd.Series({ticker: 1.0}, dtype="float64"),
        value=value,
        bench=partial(shared.app.calc.market_returns, prices.adjusted),
        notes=(f"금액은 1주 기준 (최근 종가 {pf_money(value)})",),
    )


def stock_change(before: float, after: float, text: Callable[[float], str]) -> str:
    return f"{text(before)} → {text(after)}"


def stock_es(value: float) -> str:
    return pf_percent(max(value, 0.0), dec=2)


def whatif_pair(
    label: str, pair: tuple[float, float], text: Callable[[float], str]
) -> str:
    shown = stock_change(*pair, text)
    state.ai_page_values[label] = shown
    return shown


def whatif_shift_chart(
    before: list[dict[str, float]],
    after: list[dict[str, float]],
    field: str,
    title: str,
) -> alt.LayerChart:
    rows = [
        {
            "k": round(-old["x"] * 100),
            "change": new[field] - old[field],
            "up": new[field] > old[field],
            "old": pf_percent(old[field], dec=2),
            "new": pf_percent(new[field], dec=2),
            "moved": pf_pp(new[field] - old[field]),
        }
        for old, new in zip(before, after, strict=True)
        if old["side"] < 0
    ]
    p = palette.current()
    bars = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar()
        .encode(
            x=alt.X(
                "k:O",
                title=title,
                sort="ascending",
                axis=alt.Axis(
                    values=list(range(0, WHATIF_LEVELS + 1, 5)),
                    labelExpr="datum.value + '%'",
                    labelAngle=0,
                ),
            ),
            y=alt.Y(
                "change:Q",
                title="경로 비율 변화",
                axis=alt.Axis(labelExpr="format(datum.value * 100, '+.1f') + '%p'"),
            ),
            color=alt.Color(
                "up:N",
                scale=alt.Scale(domain=[True, False], range=[p.up, p.down]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("k:O", title="기준 (% 이상)"),
                alt.Tooltip("old:N", title=WHATIF_WHEN[0]),
                alt.Tooltip("new:N", title=WHATIF_WHEN[1]),
                alt.Tooltip("moved:N", title="변화"),
            ],
        )
    )
    zero = (
        alt.Chart(alt.Data(values=[{"change": 0.0}]))
        .mark_rule(color=p.guide)
        .encode(y="change:Q")
    )
    return alt.LayerChart(layer=[bars, zero]).properties(height=CHART_HEIGHT)


def whatif_paths(
    key: Key, j: portfolio.Joint, ticker: str, fraction: float, label: str
) -> None:
    days = portfolio.HORIZONS[label]
    block = shared.app.calc.mean_block(j)
    picks = cached_draws(len(j.window), days, block)
    before = cached_paths((*key, days, block), 1.0, j, picks)
    after = cached_paths(
        (*key, days, block, ticker, fraction),
        value=1.0,
        j=j,
        picks=picks,
        w=portfolio.mixed_in(j, ticker, fraction),
    )
    falls = (before.paths.falls, after.paths.falls)
    ten = perf.PATH_LEVELS[0]
    whatif_pair(
        f"{label} 뒤 10% 이상 손실 경로 비율",
        (falls[0][ten].end, falls[1][ten].end),
        display.share,
    )
    for level in falls[0] if days > 1 else ():
        whatif_pair(
            f"기간 중 -{level:.0%} 낙폭 경로 비율",
            (falls[0][level].swing, falls[1][level].swing),
            display.share,
        )
    with st.container(border=True):
        headed(
            f"**몬테카를로** · {label}",
            "[정의] 종목 편입 전후 몬테카를로 모의 경로의 손실 도달 비율 증감(%p)"
            " 비교\n\n"
            "[해석] 좌측은 만기 손실 기준, 우측은 기간 중 고점 대비 낙폭 기준(1일"
            " 전망은 좌측만 표시), 0 상회 시 편입 후 해당 손실 경로 비중 증가·하회 시"
            " 감소\n\n"
            "[산출] 평균 수익률 0 가정 및 일별 목표 비중 리밸런싱 조건 하 동일"
            f" {portfolio.PATHS:,}개 부트스트랩 경로 전후 대조",
        )
        ends = whatif_shift_chart(before.reach, after.reach, "end", "기간 끝 손실 기준")
        if days == 1:
            st.altair_chart(ends, alt="기간 끝 손실 경로 비율 변화")
            return
        left, right = st.columns(2)
        left.altair_chart(ends, alt="기간 끝 손실 경로 비율 변화")
        right.altair_chart(
            whatif_shift_chart(
                before.reach, after.reach, "swing", "기간 중 고점 대비 낙폭 기준"
            ),
            alt="기간 중 낙폭 경로 비율 변화",
        )


@dataclass(frozen=True, slots=True)
class Buy:
    shares: float
    price: float
    value: float

    @property
    def amount(self) -> float:
        return self.shares * self.price

    @property
    def fraction(self) -> float:
        return self.fraction_of(self.shares)

    def fraction_of(self, shares: float) -> float:
        amount = shares * self.price
        return amount / (self.value + amount)


def whatif_mix(
    w: pd.Series, ticker: str, fraction: float
) -> list[tuple[str, float, float]]:
    old = {str(t): float(v) for t, v in portfolio.with_candidate(w, ticker).items()}
    new = {t: (1 - fraction) * v + fraction * (t == ticker) for t, v in old.items()}
    ranked = sorted(new, key=new.__getitem__, reverse=True)
    shown = ranked
    if len(ranked) > PF_DONUT_SLICES:
        shown = ranked[: PF_DONUT_SLICES - 1]
        if ticker not in shown:
            shown = [*shown[:-1], ticker]
    rest = [t for t in ranked if t not in shown]
    rows = [(t, old[t], new[t]) for t in shown]
    if rest:
        rows.append((WHATIF_REST, sum(old[t] for t in rest), sum(new[t] for t in rest)))
    return rows


def whatif_donuts(s: Subject, ticker: str, buy: Buy) -> None:
    rows = whatif_mix(s.w, ticker, buy.fraction)
    names = [name for name, _, _ in rows]
    after = buy.value + buy.amount
    sides = (
        [(name, old, old * buy.value) for name, old, _ in rows],
        [(name, new, new * after) for name, _, new in rows],
    )
    for column, label, side in zip(st.columns(2), WHATIF_WHEN, sides, strict=True):
        with column:
            st.caption(label)
            st.altair_chart(pf_donut(side, pf_money, order=names), alt=f"{label} 비중")


def whatif_curve_chart(
    points: list[dict[str, float]], picked: dict[str, float], label: str, number: str
) -> alt.LayerChart:
    p = palette.current()
    tips = [
        alt.Tooltip("shares:Q", title="매수 수량", format=",.4~g"),
        alt.Tooltip("weight:Q", title="매수 후 비중", format=".1%"),
        alt.Tooltip("value:Q", title=label, format=number),
    ]
    x = alt.X("shares:Q", title="매수 수량 (주)", scale=alt.Scale(nice=False))
    y = alt.Y(
        "value:Q",
        title=None,
        axis=alt.Axis(format=number, tickCount=4),
        scale=alt.Scale(zero=False),
    )
    line = (
        alt.Chart(alt.Data(values=points))
        .mark_line(color=p.ink, point=alt.OverlayMarkDef(size=18, color=p.ink))
        .encode(x=x, y=y, tooltip=tips)
    )
    start = (
        alt.Chart(alt.Data(values=points[:1]))
        .mark_rule(color=p.guide, strokeDash=[4, 4])
        .encode(y="value:Q")
    )
    dot = (
        alt.Chart(alt.Data(values=[picked]))
        .mark_point(filled=True, size=110, opacity=1, color=p.ink)
        .encode(x="shares:Q", y="value:Q", tooltip=tips)
    )
    return alt.LayerChart(layer=[start, line, dot]).properties(height=WHATIF_CURVE)


def whatif_curves(  # noqa: PLR0913  # 매수 수량별 민감도 곡선 산출 인자 일괄 전달
    key: Key,
    j: portfolio.Joint,
    cov: NDArray[np.float64],
    ticker: str,
    buy: Buy,
    added: portfolio.Added,
) -> None:
    most = WHATIF_TOP / (1 - WHATIF_TOP) * buy.value / buy.price
    top = max(most, buy.shares)
    grid = [top * i / WHATIF_STEPS for i in range(WHATIF_STEPS + 1)]
    fractions = [buy.fraction_of(n) for n in grid]
    curve = cached(
        "curve",
        (*key, top),
        partial(shared.app.calc.added_curve, j, cov, ticker, fractions),
    )
    worst = f"{1 - quant.TAIL_LEVEL:.0%}"
    panels = (
        (
            "연 변동성",
            "vol",
            added.vol,
            pf_percent,
            ".1%",
            "[정의] 편입 수량 변화에 따른 포트폴리오 연환산 변동성 추이\n\n"
            "[해석] 우하향 시 편입에 따른 변동성 절감 효과 존재\n\n"
            "[산출] √(비중 벡터ᵀ × 축소 공분산 행렬 × 비중 벡터) × √연간 거래일수",
        ),
        (
            f"ES {quant.TAIL_LEVEL:.1%} (1일)",
            "es",
            added.es,
            stock_es,
            ".2%",
            f"[정의] 기간 내 하위 {worst} 최악 거래일의 평균 1일 손실률(예상부족액)\n\n"
            "[해석] 낮을수록 극단적 하락장 꼬리 손실 축소\n\n"
            f"[산출] 공통 거래일 {quant.min_samples(quant.TAIL_LEVEL):,}일 이상 시 하위"
            f" {worst} 일별 손실 산술평균",
        ),
        (
            "베타",
            "beta",
            added.beta,
            pf_ratio,
            ".2f",
            f"[정의] {portfolio.BENCHMARK_LABEL} 가격지수 1% 변동 대비 포트폴리오"
            " 민감도\n\n"
            "[해석] 1 초과는 시장 대비 고변동, 1 미만은 저변동\n\n"
            "[산출] 종목별 지수가중(Welch) 베타의 포트폴리오 비중 가중합",
        ),
        (
            "분산 비율",
            "ratio",
            added.ratio,
            pf_ratio,
            ".2f",
            "[정의] 개별 종목 가중평균 변동성 대비 포트폴리오 전체 변동성 비율\n\n"
            "[해석] 1은 분산 효과 부재, 높을수록 상관관계 상쇄를 통한 분산 효과"
            " 확대\n\n"
            "[산출] Σ(종목 비중 × 개별 변동성) ÷ 포트폴리오 변동성",
        ),
    )
    columns = st.columns(len(panels))
    for column, (label, field, pair, text, number, tip) in zip(
        columns, panels, strict=True
    ):
        with column:
            if pair is None:
                headed(f"**{label}** · {PF_THIN}", tip)
                continue
            headed(f"**{label}** · {whatif_pair(label, pair, text)}", tip)
            points = [
                {"shares": n, "weight": f, "value": getattr(m, field)}
                for n, f, m in zip(grid, fractions, curve, strict=True)
            ]
            picked = {"shares": buy.shares, "weight": buy.fraction, "value": pair[1]}
            st.altair_chart(
                whatif_curve_chart(points, picked, label, number),
                alt=f"매수 수량에 따른 {label}",
            )


def whatif_share_chart(weight: float, share: float, ticker: str) -> alt.LayerChart:
    names = ["평가액 비중", "꼬리 손실 몫"]
    values = (weight, share)
    rows = [
        {
            "name": name,
            "value": demo.SHARE if shared.app.demo else value,
            "text": pf_percent(value),
        }
        for name, value in zip(names, values, strict=True)
    ]
    p = palette.current()
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("value:Q", title=None, axis=alt.Axis(format=".0%", tickCount=5)),
        y=alt.Y("name:N", sort=names, title=None),
    )
    bars = base.mark_bar().encode(
        color=alt.Color(
            "name:N",
            sort=names,
            scale=alt.Scale(domain=names, range=[p.muted, p.ink]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("name:N", title=ticker),
            alt.Tooltip("text:N", title="매수 후"),
        ],
    )
    labels = base.mark_text(align="left", dx=6, color=p.muted).encode(text="text:N")
    return alt.LayerChart(layer=[bars, labels]).properties(height=WHATIF_BARS)


def whatif_buy(ticker: str, s: Subject, buy: Buy) -> float:
    held = float(cast("float", s.w.get(ticker, 0.0)))
    after = (1 - buy.fraction) * held + buy.fraction
    with st.container(horizontal=True, key="pf_whatif_buy_cards"):
        pf_metric(
            "매수 금액",
            pf_money(buy.amount),
            f"{pf_shares(buy.shares)}주 × {pf_money(buy.price)}",
            delta_color="off",
            delta_arrow="off",
            help="[정의] 가상 편입 매수 총액\n\n"
            "[해석] 신규 투입 가정 자금 규모\n\n"
            "[산출] 입력 매수 수량 × 최근 종가",
        )
        pf_metric(
            f"{ticker} 비중",
            stock_change(held, after, pf_percent),
            pf_pp(after - held),
            delta_color="off",
            delta_arrow="off",
            help="[정의] 가상 매수 전후 포트폴리오 내 대상 종목 평가액 비중 변화\n\n"
            "[해석] 좌측은 현재 비중, 우측은 신규 매수 반영 후 비중\n\n"
            "[산출] 매수 후 종목 평가액 ÷ (기존 포트폴리오 총평가액 + 신규 매수 금액)",
        )
    return after


def whatif_share(ticker: str, weight: float, added: portfolio.Added) -> None:
    worst = f"{1 - quant.TAIL_LEVEL:.0%}"
    label = f"매수 후 {ticker}의 꼬리 손실 몫"
    tip = (
        f"[정의] 매수 후 포트폴리오 하위 {worst} 최악 손실일(ES) 내 대상 종목 손실 기여"
        " 비중과 평가액 비중 비교\n\n"
        "[해석] 꼬리 손실 몫이 평가액 비중 초과 시 극단적 하락장 손실이 해당 종목에"
        " 집중\n\n"
        f"[산출] 대상 종목 1일 {quant.TAIL_LEVEL:.1%} 구성요소 ES ÷ 포트폴리오 전체"
        " 1일 ES"
    )
    with st.container(border=True, height="stretch"):
        if added.es_share is None:
            headed(f"**{label}** · {PF_THIN}", tip)
            return
        state.ai_page_values[label] = pf_percent(added.es_share)
        headed(f"**{label}**", tip)
        st.altair_chart(
            whatif_share_chart(weight, added.es_share, ticker),
            alt="비중과 꼬리 손실 몫",
        )


def stock_whatif(
    ticker: str, book: portfolio.Book, prices: portfolio.Prices, months: int
) -> None:
    names = {p.id: p.name for p in book.portfolios if p.trades}
    if not names:
        st.info("거래 내역 보유 포트폴리오 없음")
        return
    price = stock_share(ticker, prices)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        pf_choose("포트폴리오", names, "whatif_pf", state.get("pf_selected"))
        p = cast("portfolio.Portfolio", book.get(state.whatif_pf))
        s = pf_subject(p, prices)
        shares = 0.0
        if not s.w.empty:
            shares = st.number_input(
                "매수 수량",
                min_value=PF_MIN_QUANTITY,
                value=max(1.0, float(round(WHATIF_START * s.value / price))),
                step=1.0,
                format="%g",
                key=f"whatif_shares_{p.id}_{ticker}",
                width=150,
                help="[정의] 최근 종가 기준 가상 신규 매수 수량\n\n"
                "[해석] 기존 보유 종목 수량 유지 하 신규 자금 투입 가정\n\n"
                "[산출] 총평가액 증가분 비례 기존 종목 비중 희석 반영",
            )
    if s.w.empty:
        st.info("가격 확인 보유 종목 없음")
        return
    buy = Buy(shares, price, s.value)
    fraction = buy.fraction
    key = (*s.key, months, "whatif", ticker)
    w = portfolio.with_candidate(s.w, ticker)
    j = cached("joint", key, partial(shared.app.calc.joint, prices.adjusted, w, months))
    if len(j.window) < portfolio.MIN_DAYS:
        st.info(
            f"보유 종목·{ticker} 공통 수익률 거래일 {len(j.window)}일"
            f" (최소 {portfolio.MIN_DAYS}일 필요)"
        )
        return
    if s.notes:
        st.caption(" · ".join(s.notes))
    cov = cached("cov", key, partial(shared.app.calc.covariance, j))
    added = cached(
        "added",
        (*key, fraction),
        partial(shared.app.calc.added, j, cov, ticker, fraction),
    )
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed(
            "**비중 변화**",
            "[정의] 가상 매수 전후 종목별 평가액 구성 비중 비교\n\n"
            f"[해석] 양측 도넛 동일 색상 매칭, {PF_DONUT_SLICES}종목 초과 시 매수 후"
            f" 상위 {PF_DONUT_SLICES - 1}종목 외 기타 병합\n\n"
            "[산출] 매수 전후 종목별 평가액 ÷ 포트폴리오 총평가액",
        )
        whatif_donuts(s, ticker, buy)
    with right:
        whatif_share(ticker, whatif_buy(ticker, s, buy), added)
    with st.container(border=True):
        headed(
            "**매수 수량에 따른 포트폴리오 위험**",
            "[정의] 매수 수량(0주부터 증분)에 따른 포트폴리오 위험 지표 민감도 곡선\n\n"
            "[해석] 굵은 점은 현재 선택 수량, 수평 점선은 매수 전 기준선, 제목 우측"
            " 수치는 매수 전→후 변화\n\n"
            "[산출] 수량 격자별 편입 비중 반영 포트폴리오 변동성·ES·베타·분산 비율"
            " 산출",
            pf_gaps([], j.late),
        )
        whatif_curves(key, j, cov, ticker, buy, added)
    whatif_paths(key, j, ticker, fraction, state.get("whatif_horizon") or "1M")


def stock_section(  # noqa: PLR0913  # 종목 분석 섹션별 렌더링 인자 일괄 전달
    section: str,
    ticker: str,
    book: portfolio.Book,
    prices: portfolio.Prices,
    months: int,
    top: DeltaGenerator,
) -> None:
    if section == "whatif":
        stock_whatif(ticker, book, prices, months)
        return
    s = stock_subject(ticker, prices)
    if len(s.rets) < 2:  # noqa: PLR2004  # 수익률 산출 최소 2거래일 기준
        pf_no_prices()
    elif section == "stats":
        pf_stats(s, months)
        pf_style_box(s, prices, "stock_style")
    else:
        pf_simulation(s, prices, months, top)


def stock_view(
    book: portfolio.Book,
    prices: portfolio.Prices,
    fetcher: portfolio_fetch.Fetcher,
    missing: list[str],
) -> None:
    owned = stock_owned(book, prices)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        typed = st.text_input(
            "티커",
            key="stock_input",
            width=170,
            on_change=stock_enter,
            bind="query-params",
        )
        st.button("분석", type="primary", key="stock_go", on_click=stock_enter)
        if typed and not state.get("stock_ticker"):
            stock_enter()
        if owned:
            st.pills("보유 종목", owned, key="stock_held", on_change=stock_held)
    if state.get("stock_error"):
        st.error(state.stock_error)
    ticker: str | None = state.get("stock_ticker")
    blocked = shared.app.demo or bool(missing)
    retry = bool(state.pop("stock_retry", False))
    if (
        ticker
        and not blocked
        and (ticker not in prices.tickers or fetcher.untried([ticker]))
    ):
        stock_fetch(fetcher, ticker, retry=retry)
    pf_bar(fetcher, book, missing, [ticker] if ticker else [])
    if not ticker:
        if not state.get("stock_error"):
            st.info("분석 대상 종목 없음")
        return
    if ticker not in prices.tickers:
        if not state.get("stock_error"):
            st.info(f"{ticker} 수신 가격 데이터 없음")
        return
    headed(f"**{ticker}**")
    pf_part("stock_section", partial(stock_view_section, ticker, book, prices))


def stock_view_section(
    ticker: str, book: portfolio.Book, prices: portfolio.Prices
) -> None:
    section = st.segmented_control(
        "보기",
        list(STOCK_SECTIONS),
        format_func=STOCK_SECTIONS.__getitem__,
        default="stats",
        required=True,
        key="stock_section",
        label_visibility="collapsed",
        bind="query-params",
    )
    horizons = {"simulation": "pf_horizon", "whatif": "whatif_horizon"}
    top = st.container(horizontal=True, vertical_alignment="bottom")
    row = st.container(horizontal=True, vertical_alignment="bottom")
    with row:
        months = pf_periods(horizons.get(section or "stats"))
    stock_section(section or "stats", ticker, book, prices, months, top)
