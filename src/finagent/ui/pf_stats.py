# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false, reportUnknownMemberType=false
"""포트폴리오 위험·성과 지표, 상관행렬, 스타일 분석 및 꼬리 위험 기여도 화면."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import altair as alt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent import demo, perf, portfolio, quant
from finagent.ui import shared
from finagent.ui.pf_common import (
    CHART_HEIGHT,
    PF_BETA_HELP,
    PF_DASH,
    PF_HEAT,
    PF_LONG_MONTHS,
    PF_SHORT,
    Key,
    Subject,
    cached,
    headed,
    pf_compact,
    pf_gaps,
    pf_joint,
    pf_key,
    pf_metric,
    pf_money,
    pf_num,
    pf_percent,
    pf_rates,
    pf_rates_key,
    pf_ratio,
    pf_ratio_or,
    pf_table,
    pf_weight_chart,
    pf_weights,
    portfolio_fetcher,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def pf_drawdown(stats: portfolio.Ratios) -> None:
    under = stats.under
    fell = under is not None and under.max_drawdown < 0
    pf_metric(
        "최대 낙폭",
        pf_percent(under.max_drawdown) if under else PF_DASH,
        f"{under.peak:%Y-%m-%d} → {under.trough:%Y-%m-%d}" if under and fell else None,
        delta_color="off",
        delta_arrow="off",
        help="[정의] 분석 기간 내 직전 고점 대비 최대 낙폭(MDD), 하단은 고점→저점"
        " 발생 일자\n\n"
        "[해석] 0% 근접 시 낙폭 방어력 우위\n\n"
        "[산출] 기간 내 min((일별 누적가치 ÷ 직전 최고 누적가치) − 1)",
    )
    back = under.recovery if under and fell else None
    pf_metric(
        "최대 낙폭 회복",
        PF_DASH if not fell else "미회복" if back is None else f"{back:%Y-%m-%d}",
        f"저점부터 {pf_num(under.recovery_days, ',.0f')}일"
        if under and back is not None and under.recovery_days is not None
        else None,
        delta_color="off",
        delta_arrow="off",
        help="[정의] 최대 낙폭 저점 이후 직전 고점 재돌파 일자 및 소요 기간\n\n"
        "[해석] 미회복 표기 시 현재 직전 고점 하회 상태 지속\n\n"
        "[산출] 최대 낙폭 저점일로부터 고점 수준 회복일까지의 달력일수",
    )
    pf_metric(
        "고점 아래 기간",
        f"{pf_num(under.current_days, ',.0f')}일" if under else PF_DASH,
        f"최장 {pf_num(under.longest_days, ',.0f')}일" if under else None,
        delta_color="off",
        delta_arrow="off",
        help="[정의] 최근 고점 이후 경과 달력일수(현재 고점 경신 시 0일) 및 기간 내"
        " 최장 고점 아래 기간(하단)\n\n"
        "[산출] 기준일 − 최근 고점일(현재) 및 고점 간 최대 달력일수(최장)",
    )


def pf_sharpe(stats: portfolio.Ratios) -> None:
    ci = stats.ci
    pf_metric(
        "샤프 지수",
        pf_ratio_or(stats.sharpe),
        f"95% {pf_ratio(ci.low)} \\~ {pf_ratio(ci.high)}" if ci else PF_SHORT,
        delta_color="off",
        delta_arrow="off",
        help="[정의] 총변동성 대비 연환산 초과수익률(Sharpe Ratio)\n\n"
        "[해석] 높을수록 위험 대비 성과 우위(1 이상 양호), 95% 신뢰구간 하한이 0 초과"
        " 시 통계적 유의성 확보\n\n"
        f"[산출] (일간 초과수익률 평균 ÷ 표준편차) × √연간 거래일수({perf.MIN_OBS}일"
        " 이상 시 신뢰구간 산출)",
    )


def pf_advanced(stats: portfolio.Ratios) -> None:
    why = PF_SHORT if stats.span < portfolio.YEAR_DAYS else None
    index = f"{portfolio.BENCHMARK_LABEL} 총수익지수"
    with st.container(horizontal=True, key="pf_more_cards"):
        pf_metric(
            "소르티노",
            pf_ratio_or(stats.sortino),
            help="[정의] 하방변동성 대비 연환산 초과수익률(Sortino Ratio)\n\n"
            "[해석] 높을수록 하락 위험 대비 초과수익 우위(대칭 분포 시 샤프 지수의 약"
            " 1.41배, 초과 시 상방 변동 우세)\n\n"
            "[산출] 연환산 일간 초과수익률 평균 ÷ 연환산 하방편차(음수 초과수익률"
            " 제곱평균제곱근)",
        )
        pf_metric(
            "칼마",
            pf_ratio_or(stats.calmar),
            None if stats.calmar is not None else why,
            delta_color="off",
            delta_arrow="off",
            help="[정의] 최대 낙폭(MDD) 대비 연복리 수익률 비율(Calmar Ratio)\n\n"
            "[해석] 높을수록 극단적 낙폭 대비 장기 복리 성과 우위\n\n"
            "[산출] 연복리 수익률(CAGR) ÷ |최대 낙폭|(기간 365일 이상 시 산출)",
        )
        pf_metric(
            "CDaR 95%",
            pf_percent(stats.cdar),
            help=f"[정의] 조건부 낙폭 위험(Conditional Drawdown at Risk, 하위"
            f" {1 - perf.CDAR_LEVEL:.0%})\n\n"
            "[해석] 0% 근접 시 심층 낙폭 구간 방어력 우위\n\n"
            f"[산출] 일별 낙폭 중 가장 깊은 하위 {1 - perf.CDAR_LEVEL:.0%} 구간"
            " 산술평균(평균 낙폭과 최대 낙폭 사이 값)",
        )
        pf_metric(
            "추적오차",
            pf_percent(stats.te) if stats.te is not None else PF_DASH,
            None if stats.te is not None else PF_SHORT,
            delta_color="off",
            delta_arrow="off",
            help=f"[정의] 벤치마크({index}) 대비 초과수익률 연환산 변동성(Tracking"
            " Error)\n\n"
            "[해석] 낮을수록 벤치마크 추종성 밀착, 높을수록 액티브 괴리 확대\n\n"
            f"[산출] (포트폴리오 − {index}) 일간 수익률 표준편차 × √연간 거래일수"
            f"({perf.MIN_OBS}일 이상 시 산출)",
        )
        pf_metric(
            "정보비율",
            pf_ratio_or(stats.ir),
            None if stats.ir is not None else PF_SHORT,
            delta_color="off",
            delta_arrow="off",
            help=f"[정의] 추적오차 단위당 벤치마크({index}) 대비 연환산"
            " 초과수익(Information Ratio)\n\n"
            "[해석] 높을수록 일관된 액티브 초과성과 창출(0.5 이상 양호, 1.0 이상"
            " 우수)\n\n"
            f"[산출] 연환산 액티브 수익률 평균 ÷ 추적오차({perf.MIN_OBS}일 이상 시"
            " 산출)",
        )


def pf_stats(s: Subject, months: int) -> None:
    rets = portfolio.window(s.rets, months)
    rates = pf_rates()
    rf = shared.app.calc.risk_free(rates, pd.DatetimeIndex(rets.index))
    key = (*s.key, months, pf_rates_key(rates))
    bench = partial(s.bench, pd.DatetimeIndex(s.rets.index))
    cut = len(s.rets) - len(rets)
    start = cast("pd.Timestamp", s.rets.index[cut - 1]) if cut > 0 else None
    stats = cached(
        "ratios", key, lambda: shared.app.calc.ratios(rets, rf, bench(), start)
    )
    with st.container(horizontal=True, key="pf_risk_cards"):
        pf_metric(
            "연 변동성",
            pf_percent(stats.volatility) if stats.volatility is not None else PF_DASH,
            help="[정의] 일간 초과수익률의 연환산 표준편차\n\n"
            "[해석] 높을수록 수익률 진폭 및 가격 변동 위험 확대\n\n"
            "[산출] (일간 수익률 − 무위험수익률) 표준편차 × √연간 거래일수",
        )
        pf_drawdown(stats)
        pf_sharpe(stats)
    more = st.expander("고급", key="pf_advanced", on_change="rerun")
    if more.open:
        with more:
            pf_advanced(stats)


def pf_style_box(s: Subject, prices: portfolio.Prices, key: str) -> None:
    if s.w.empty:
        return
    box = st.expander("스타일 분석", key=key, on_change="rerun")
    if box.open:
        with box:
            pf_style_fit(s, prices)


def pf_style_fit(s: Subject, prices: portfolio.Prices) -> None:
    absent = [t for t in portfolio.STYLE_PROXIES if t not in prices.tickers]
    fetcher = portfolio_fetcher()
    can = bool(absent) and not shared.app.demo and not portfolio.missing_packages()
    if can and st.button(
        "비교 지수 수신",
        icon=":material/download:",
        key="pf_style_fetch",
        disabled=fetcher.progress.running,
    ):
        fetcher.start(absent)
        st.rerun()
    w, _, _ = pf_weights(s, prices, PF_LONG_MONTHS)
    fit = cached(
        "style",
        s.key,
        partial(shared.app.calc.style, prices.adjusted, w),
    )
    about = (
        "[정의] 수익률 기반 스타일 분석(RBSA) 복제 팩터·현금 조합 비중\n\n"
        "[해석] 특정 지수 비중이 높을수록 해당 스타일 요인 노출 집중\n\n"
        "[산출] 주간 수익률 오차 제곱합 최소화 제약 회귀(비중 ≥ 0, 합계 = 100%)"
    )
    if fit is None:
        headed("**스타일 비중**", about)
        st.caption(
            f"현 비중 주간 수익률 {portfolio.WEEKS}주 미만 또는 비교 지수 데이터 미수신"
        )
        return
    shown = pd.Series(fit.weights, index=list(fit.names), dtype="float64")
    shown = cast("pd.Series", shown[shown > 0.0005])  # noqa: PLR2004  # 반올림 0.1% 미만 미소 비중 절사 임계치
    with st.container(horizontal=True, key="pf_style_cards"):
        pf_metric(
            "설명력 R²",
            pf_percent(fit.r2, dec=0) if fit.r2 is not None else PF_DASH,
            help="[정의] 스타일 조합의 포트폴리오 주간 수익률 변동 설명 비율"
            "(결정계수)\n\n"
            "[해석] 높을수록 스타일 벤치마크 조합과의 동조성 우위\n\n"
            "[산출] 1 − (제약 회귀 잔차 분산 ÷ 포트폴리오 주간 수익률 분산)",
        )
        pf_metric(
            "조합 대비 차이 (연)",
            pf_percent(fit.selection, signed=True)
            if fit.selection is not None
            else PF_DASH,
            help="[정의] 스타일 복제 조합 대비 연환산 종목 선택 초과수익(Selection"
            " Return)\n\n"
            "[해석] 양(+)은 동일 스타일 조합 대비 초과성과, 음(-)은 하회\n\n"
            "[산출] 직전 구간 추정 스타일 조합 대비 주간 표본외(Out-of-Sample)"
            " 초과수익 연환산",
        )
    headed(
        "**스타일 비중**",
        about,
        [f"제외된 비교 지수: {', '.join(fit.dropped)}"] if fit.dropped else [],
    )
    st.altair_chart(pf_weight_chart(shown, "지수"), alt="스타일 비중")


PF_CELL_TEXT = "luminance(scale('color', datum.value)) > 0.179 ? '#000000' : '#FFFFFF'"


def pf_heatmap(matrix: pd.DataFrame) -> alt.LayerChart:
    if shared.app.demo:
        matrix = matrix.mask(matrix.notna(), demo.VALUE)
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
    labels = base.mark_text(color=alt.ExprRef(PF_CELL_TEXT)).encode(
        text=alt.Text("value:Q", format=".2f")
    )
    size = max(CHART_HEIGHT, 44 * len(order))
    return alt.LayerChart(layer=[cells, labels]).properties(height=size)


def pf_correlation(
    s: Subject,
    prices: portfolio.Prices,
    months: int,
    compare: Callable[[], None],
) -> None:
    j, _, dropped = pf_joint(s, prices, months)
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        title = "**보유 종목** · 일간 수익률"
        about = (
            "[정의] 보유 종목 간 일간 수익률 상관계수 행렬(비중 상위"
            f" {PF_HEAT}종목)\n\n"
            "[해석] +1 근접 시 동조 움직임, 음(-)은 역방향 헤지, 0 근접은 독립 변동\n\n"
            "[산출] Ledoit-Wolf 축소 추정 공분산 행렬 기반 상관계수"
        )
        if j is None or len(j.names) < 2 or len(j.window) < portfolio.MIN_DAYS:  # noqa: PLR2004  # 상관계수 산출 최소 2종목 기준
            names, days = (len(j.names), len(j.window)) if j is not None else (0, 0)
            headed(title, about)
            st.caption(
                f"수익률 보유 종목 {names}개 · 공통 거래일 {days}일"
                f" (최소 2종목 · {portfolio.MIN_DAYS}일 필요)"
            )
        else:
            cov = cached(
                "cov", (*s.key, months), partial(shared.app.calc.covariance, j)
            )
            shown = pf_largest(j)
            full = shared.app.calc.correlation_frame(cov, j.names)
            matrix = cast("pd.DataFrame", full.loc[shown, shown])
            headed(title, about, pf_gaps(dropped, j.late))
            st.altair_chart(pf_heatmap(matrix), alt="종목 상관")
    with right, st.container(border=True, height="stretch"):
        compare()
    if j is None or len(j.window) < portfolio.MIN_DAYS:
        return
    left, right = st.columns(2)
    with left, st.container(border=True, height="stretch"):
        headed("**베타** · 지금 비중, 일간 수익률", PF_BETA_HELP)
        shown = set(pf_largest(j))
        rows = shared.app.calc.betas(j)
        pf_betas(
            [b for b in rows if b.name in shown or b.name == portfolio.PORTFOLIO_LABEL]
        )
    with right:
        cov = cached("cov", (*s.key, months), partial(shared.app.calc.covariance, j))
        pf_spread(j, shared.app.calc.diversification(j, cov))


def pf_largest(j: portfolio.Joint) -> list[str]:
    if len(j.names) <= PF_HEAT:
        return list(j.names)
    cut = sorted(float(v) for v in j.w)[-PF_HEAT]
    return [n for n, v in zip(j.names, j.w, strict=True) if v >= cut][:PF_HEAT]


def pf_book_heatmap(
    book: portfolio.Book, prices: portfolio.Prices, months: int
) -> None:
    key = pf_key(book.model_dump_json())
    frame = cached(
        "lineup", (*key, months), partial(shared.app.calc.lineup, book, prices, months)
    )
    pf_compare_heatmap(
        (*key, months),
        frame,
        f"**포트폴리오와 {portfolio.BENCHMARK_LABEL}**",
        f"[정의] 포트폴리오별 현재 비중 고정 일간 수익률과 {portfolio.BENCHMARK_LABEL}"
        " 총수익지수 간 상관계수(미보유 포트폴리오 제외)\n\n"
        "[해석] +1 근접 시 동행, 0 근접 시 무관\n\n"
        "[산출] 공통 거래일 일간 수익률 축소 추정 상관계수",
    )


def pf_betas(rows: list[portfolio.Beta]) -> None:
    pf_table(
        {
            "구분": [b.name for b in rows],
            portfolio.BENCHMARK_LABEL: [pf_ratio(b.welch) for b in rows],
        },
        "베타",
    )


def pf_spread(j: portfolio.Joint, spread: portfolio.Spread) -> None:
    with st.container(border=True, height="stretch"):
        headed(
            f"**분산** · 지금 비중 · 분산 비율 {pf_ratio(spread.ratio)}",
            "[정의] 포트폴리오 분산 내 종목별 위험 기여 비중(합계 100%) 및 분산"
            " 비율\n\n"
            "[해석] 위험 기여도가 보유 비중 초과 시 변동성 집중, 분산 비율이 1보다"
            " 클수록 종목 간 상쇄를 통한 변동성 절감 효과 확대\n\n"
            "[산출] 위험 기여 = (비중 × 공분산 행렬 곱) ÷ 포트폴리오 분산, 분산 비율 ="
            " 가중평균 개별 변동성 ÷ 포트폴리오 변동성",
        )
        order = np.argsort(-spread.risk, kind="stable")
        pf_table(
            {
                "종목": [j.names[i] for i in order],
                "비중": [pf_percent(float(j.w[i])) for i in order],
                "위험 기여": [pf_percent(float(spread.risk[i])) for i in order],
            },
            "위험 기여",
        )


def pf_compare_heatmap(key: Key, frame: pd.DataFrame, title: str, about: str) -> None:
    found = cached("shrunk", key, partial(shared.app.calc.shrunk_correlation, frame))
    headed(title, about)
    if found is None:
        st.caption(
            f"비교 대상 수익률 2개 미만 또는 공통 거래일 {portfolio.MIN_DAYS}일 미만"
        )
        return
    st.altair_chart(pf_heatmap(found), alt="포트폴리오 상관")


def pf_component(s: Subject, prices: portfolio.Prices, months: int) -> None:
    j, share, dropped = pf_joint(s, prices, months)
    if j is None or len(j.names) == 1:
        return
    component = quant.component_es(j.window.to_numpy(dtype="float64"), j.w)
    value = s.value * share
    with st.container(border=True):
        headed(
            f"**종목별 ES 기여** · 1일 {quant.TAIL_LEVEL:.1%}",
            f"[정의] 포트폴리오 1일 {quant.TAIL_LEVEL:.1%} 예상부족액(ES) 내 종목별"
            " 손실 기여도\n\n"
            "[해석] 양(+)의 값이 클수록 최악 손실일 손실 집중, 음(-)은 꼬리 위험 완충"
            " 역할\n\n"
            f"[산출] 포트폴리오 하위 {1 - quant.TAIL_LEVEL:.0%} 최악 손실일 내 종목별"
            " (비중 × 평균 손실률), 전 종목 합산 시 포트폴리오 1일 ES 일치",
            pf_gaps(dropped, j.late),
        )
        if component is None:
            st.caption(
                f"기간 수익률 {len(j.window):,}일"
                f" (최소 {quant.min_samples(quant.TAIL_LEVEL):,}일 필요)"
            )
            return
        order = np.argsort(-component, kind="stable")
        pf_table(
            {
                "종목": [j.names[i] for i in order],
                "ES 기여": [pf_percent(float(component[i]), dec=2) for i in order],
                "ES 기여액": [
                    pf_money(float(component[i]) * value, compact=pf_compact(value))
                    for i in order
                ],
            },
            "종목별 ES",
        )
