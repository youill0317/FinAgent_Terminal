# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false, reportUnknownMemberType=false
"""몬테카를로 경로 시뮬레이션, 블랙-리터만 기대수익 및 사후검정 화면."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import altair as alt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent import portfolio, quant
from finagent.ui import palette, shared
from finagent.ui.pf_common import (
    CHART_HEIGHT,
    PF_AXIS,
    Key,
    Subject,
    cached,
    cached_draws,
    cached_paths,
    headed,
    pf_compact,
    pf_funds,
    pf_gaps,
    pf_joint,
    pf_money,
    pf_num,
    pf_percent,
    pf_rates,
    pf_ratio,
    pf_store,
    pf_table,
    portfolio_fetcher,
)

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

state = st.session_state

PF_FIT_PAD = 0.05
PF_BREAK = "≈"
PF_REACH_LINES = (
    (
        "end",
        "기간 끝",
        ("기간 끝", "기간 끝"),
        ("기간 끝 지금 대비 {} 이하", "기간 끝 지금 대비 {} 이상"),
    ),
    (
        "touched",
        "기간 중 도달",
        ("기간 중 도달", "기간 중 도달"),
        (
            "기간 중 한 번이라도 지금 대비 {} 이하",
            "기간 중 한 번이라도 지금 대비 {} 이상",
        ),
    ),
    (
        "swing",
        "고점·저점 대비",
        ("고점 대비 하락", "저점 대비 상승"),
        (
            "기간 중 고점(지금 포함)에서 {} 이상 하락",
            "기간 중 저점(지금 포함)에서 {} 이상 상승",
        ),
    ),
)
PF_END_MARKS: tuple[tuple[str, float, list[int]], ...] = (
    ("VaR", 0.95, []),
    ("ES", 0.95, [6, 3]),
    ("VaR", 0.99, []),
)
PF_END_ROOM = 1.35
PF_END_LABEL = 15


def pf_fit(low: float, high: float) -> list[float]:
    pad = PF_FIT_PAD * ((high - low) or abs(high) or 1.0)
    return [low - pad, high + pad]


def pf_break(x: float, y: float, *, angle: int) -> alt.Chart:
    return (
        alt.Chart(alt.Data(values=[{}]))
        .mark_text(
            text=PF_BREAK,
            angle=angle,
            fontSize=14,
            dy=-10,
            color=palette.current().ink,
        )
        .encode(x=alt.datum(x), y=alt.datum(y))
    )


def pf_value_axis(high: float) -> alt.Axis:
    if not pf_compact(high):
        return alt.Axis(format=PF_AXIS)
    return alt.Axis(
        labelExpr="abs(datum.value) >= 1e9"
        " ? format(datum.value / 1e9, ',.1~f') + 'B'"
        " : format(datum.value / 1e6, ',.1~f') + 'M'"
    )


def pf_fan_chart(rows: list[dict[str, float]]) -> alt.LayerChart:
    low, high = pf_fit(min(r["p5"] for r in rows), max(r["p95"] for r in rows))
    compact = pf_compact(high)
    shown = [
        r
        | {
            f"t{q}": pf_money(r[f"p{q}"], compact=compact)
            for q in portfolio.PERCENTILES
        }
        for r in rows
    ]
    chart = alt.Chart(alt.Data(values=shown)).encode(x=alt.X("day:Q", title="거래일"))
    wide = chart.mark_area(opacity=0.18).encode(
        y=alt.Y(
            "p5:Q",
            title=None,
            axis=pf_value_axis(high),
            scale=alt.Scale(domain=[low, high], zero=False, nice=False),
        ),
        y2="p95:Q",
    )
    mid = chart.mark_area(opacity=0.35).encode(y="p25:Q", y2="p75:Q")
    median = chart.mark_line().encode(
        y="p50:Q",
        tooltip=[
            alt.Tooltip("day:Q", title="거래일"),
            *(alt.Tooltip(f"t{q}:N", title=f"{q}%") for q in portfolio.PERCENTILES),
        ],
    )
    return alt.LayerChart(
        layer=[wide, mid, median, pf_break(0, low, angle=0)]
    ).properties(height=CHART_HEIGHT)


def pf_end_chart(
    rows: list[dict[str, float]],
    value: float,
    tails: dict[float, tuple[float, float]],
) -> alt.LayerChart:
    compact = pf_compact(value)
    marks: list[dict[str, object]] = []
    dashes: list[list[int]] = []
    for i, (kind, level, dash) in enumerate(PF_END_MARKS):
        if level not in tails:
            continue
        loss = tails[level][kind == "ES"]
        name = f"{kind} {level:.0%}"
        amount = pf_money(loss * value, compact=compact)
        marks.append(
            {
                "x": value * (1 - loss),
                "row": PF_END_LABEL * (i + 0.5),
                "name": name,
                "loss": pf_percent(loss),
                "amount": amount,
                "worth": pf_money(value * (1 - loss), compact=compact),
                "text": f"{name} {pf_percent(loss)} · {amount}",
            }
        )
        dashes.append(dash)
    xs = [cast("float", m["x"]) for m in marks]
    low, high = pf_fit(
        min(value, *xs, *(r["start"] for r in rows)),
        max(value, *xs, *(r["end"] for r in rows)),
    )
    top = max((r["count"] for r in rows), default=0.0) * PF_END_ROOM or 1.0
    shown = [
        r
        | {
            "from": pf_money(r["start"], compact=compact),
            "to": pf_money(r["end"], compact=compact),
        }
        for r in rows
    ]
    bars = (
        alt.Chart(alt.Data(values=shown))
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(
                "start:Q",
                title=None,
                axis=pf_value_axis(high),
                scale=alt.Scale(domain=[low, high], zero=False, nice=False),
            ),
            x2="end:Q",
            y=alt.Y(
                "count:Q",
                title="경로 수",
                scale=alt.Scale(domain=[0, top], nice=False),
            ),
            y2=alt.datum(0),
            tooltip=[
                alt.Tooltip("from:N", title="부터"),
                alt.Tooltip("to:N", title="까지"),
                alt.Tooltip("count:Q", title="경로 수"),
            ],
        )
    )
    p = palette.current()
    now = (
        alt.Chart(alt.Data(values=[{"x": value}]))
        .mark_rule(strokeDash=[3, 3], color=p.ink)
        .encode(x="x:Q")
    )
    tooltip = [
        alt.Tooltip("name:N", title="측정"),
        alt.Tooltip("loss:N", title="손실률"),
        alt.Tooltip("amount:N", title="손실액"),
        alt.Tooltip("worth:N", title="평가액"),
    ]
    shade = [{"x": low, "x2": m["x"]} for m in marks if m["name"] == "VaR 95%"]
    layers: list[alt.Chart] = [
        alt.Chart(alt.Data(values=shade))
        .mark_rect(color=p.down, opacity=0.12)
        .encode(x="x:Q", x2="x2:Q"),
        bars,
        now,
        *(
            alt.Chart(alt.Data(values=[m]))
            .mark_rule(color=p.down, strokeWidth=2, strokeDash=dash)
            .encode(x="x:Q", tooltip=tooltip)
            for m, dash in zip(marks, dashes, strict=True)
        ),
        *(
            alt.Chart(alt.Data(values=marks))
            .mark_text(
                align="left",
                baseline="middle",
                dx=4,
                fontSize=12,
                color=color,
                stroke=color,
                strokeWidth=width,
            )
            .encode(
                x="x:Q", y=alt.Y("row:Q", scale=None), text="text:N", tooltip=tooltip
            )
            for color, width in ((p.paper, 4), (p.ink, 0))
        ),
        pf_break(low, 0, angle=90),
    ]
    return alt.LayerChart(layer=layers).properties(height=CHART_HEIGHT)


def pf_reach_chart(
    curve: list[dict[str, float]], *, single: bool = False
) -> alt.LayerChart:
    measures = PF_REACH_LINES[:1] if single else PF_REACH_LINES
    rows: list[dict[str, object]] = []
    for r in curve:
        part = int(r["side"] > 0)
        threshold = pf_percent(r["x"], signed=True, dec=0)
        for field, legend, names, means in measures:
            rows.append(
                {
                    "x": r["x"],
                    "side": r["side"],
                    "line": legend,
                    "name": names[part],
                    "mean": means[part].format(
                        threshold if field != "swing" else f"{abs(r['x']):.0%}"
                    ),
                    "p": r[field],
                }
            )
    lines = [legend for _, legend, _, _ in measures]
    p = palette.current()
    dashes = [[1, 0], [5, 3], [2, 2]][: len(lines)]
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=alt.OverlayMarkDef(size=18))
        .encode(
            x=alt.X(
                "x:Q",
                title="지금 대비 수익률 기준",
                axis=alt.Axis(format="+.0%"),
                scale=alt.Scale(nice=False),
            ),
            y=alt.Y(
                "p:Q",
                title="경로 비율",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "line:N",
                sort=lines,
                scale=alt.Scale(domain=lines, range=p.tones(len(lines))),
                title=None,
                legend=alt.Legend(orient="bottom", symbolType="stroke"),
            ),
            strokeDash=alt.StrokeDash(
                "line:N",
                sort=lines,
                scale=alt.Scale(domain=lines, range=dashes),
                legend=None,
            ),
            detail="side:N",
            tooltip=[
                alt.Tooltip("name:N", title="측정"),
                alt.Tooltip("mean:N", title="정의"),
                alt.Tooltip("p:Q", title="경로 비율", format=".2%"),
            ],
        )
    )
    zero = (
        alt.Chart(alt.Data(values=[{"x": 0.0}]))
        .mark_rule(strokeDash=[2, 3], color=p.guide)
        .encode(x="x:Q")
    )
    return alt.LayerChart(layer=[zero, chart]).properties(height=CHART_HEIGHT)


PF_REACH_HELP = (
    "[정의] 수익률 임계치별 몬테카를로 모의 경로 도달 비율 곡선(좌측 하락·우측 상승)"
)
PF_REACH_LINES_HELP = (
    "\n\n[해석] 기간 끝 대비 기간 중 도달 비율 격차는 기간 중 겪었다가 사라진"
    " 손실·기회 폭\n\n"
    "[산출] 전체 모의 경로 중 각 기준 충족 경로 비중(기간 끝: 마지막 날 수준 기준,"
    " 기간 중 도달: 하루 이상 기준 도달, 고점·저점 대비: 그 시점까지 고점 대비"
    " 하락폭·저점 대비 상승폭 기준)"
)
PF_REACH_ONE_HELP = (
    "\n\n[산출] 전체 모의 경로 중 마지막 날 수준 기준 충족 경로 비중(1일 전망은 세"
    " 기준 동일로 기간 끝 곡선만 표기)"
)
PF_OWN_MEAN = "경로 평균 수익률: 과거 표본 평균 적용"
PF_BL_MEAN = "경로 평균 수익률: 블랙-리터만 기대수익 적용"
PF_ZERO_MEAN = "경로 평균 수익률: 0% 가정"
PF_P_NOTE = "0.05 미만 시"
PF_YELLOW, PF_RED = (
    next(
        k
        for k in range(quant.BACKTEST_DAYS + 1)
        if quant.traffic_light(k, quant.BACKTEST_DAYS, 1 - quant.BACKTEST_LEVEL)
        == light
    )
    for light in ("yellow", "red")
)
PF_LIGHT_NOTE = (
    f"0\\~{PF_YELLOW - 1}회 녹색, {PF_YELLOW}\\~{PF_RED - 1}회 황색, {PF_RED}회"
    " 이상 적색"
)
PF_LIGHTS: dict[str, str] = {"green": "녹색", "yellow": "황색", "red": "적색"}


def pf_density(d: portfolio.Density, days: int) -> None:
    span = pf_span(days)
    headed(
        f"**분포** · {span} 분포 검증",
        f"[정의] 비중첩 {span} 구간별 예측 수익률 분포 적합성 검정(PIT)\n\n"
        f"[해석] p값 {PF_P_NOTE} 예측 분포와 실제 수익률 분포 간 유의미한 괴리 시사\n\n"
        "[산출] 직전 구간 평균 0 조정 부트스트랩 예측 분포 내 실제 수익률 백분위(PIT)"
        " 정규성·독립성 검정",
    )
    if d.test is None:
        st.caption(f"{span} 구간 {d.blocks}개 (최소 {quant.MIN_MONTHS}개 필요)")
        return
    pf_table(
        {
            "구간": [f"{d.blocks}개"],
            "LR": [pf_num(d.test.lr, ".2f")],
            "p": [pf_num(d.test.p, ".2f")],
            "평균": [pf_ratio(d.test.mean)],
            "표준편차": [pf_num(d.test.sd, ".2f")],
            "자기상관": [pf_ratio(d.test.rho)],
        },
        "분포 검증",
        helps={
            "LR": "[정의] 예측 분포 적합성 우도비(Likelihood Ratio) 검정 통계량\n\n"
            "[해석] 클수록 예측 분포 대비 실제 분포 이탈 확대\n\n"
            "[산출] Berkowitz 우도비 검정 통계량",
            "p": "[정의] 귀무가설(예측 분포 적합) 하 우도비 관측 유의확률\n\n"
            f"[해석] {PF_P_NOTE} 예측 분포 부적합 판정\n\n"
            "[산출] 카이제곱 분포 기반 우도비 p값",
            "평균": "[정의] 역정규 변환 PIT 계열의 표본 평균\n\n"
            "[해석] 0 근접 정상, 양(+)은 실제 수익률이 예측 대비 상방 편향\n\n"
            "[산출] Φ⁻¹(PIT) 시계열 산술평균",
            "표준편차": "[정의] 역정규 변환 PIT 계열의 표본 표준편차\n\n"
            "[해석] 1 근접 정상, 1 초과는 예측 분포 과소 산정(협소), 1 미만은 과대"
            " 산정\n\n"
            "[산출] Φ⁻¹(PIT) 시계열 표준편차",
            "자기상관": "[정의] 역정규 변환 PIT 계열의 1차 자기상관계수\n\n"
            "[해석] 0 근접 정상, 0 이탈 시 구간 간 예측 오차 지속성 존재\n\n"
            "[산출] Φ⁻¹(PIT) 시계열 AR(1) 상관계수",
        },
    )


def pf_sim(
    key: Key, j: portfolio.Joint, value: float, drift: float, *, demean: bool
) -> portfolio.Simulation:
    days = portfolio.HORIZONS[state.get("pf_horizon") or "1M"]
    block = shared.app.calc.mean_block(j)
    picks = cached_draws(len(j.window), days, block)
    return cached_paths(
        (*key, days, block),
        value,
        j,
        picks,
        drift=drift,
        demean=demean,
    )


def pf_path_charts(
    sim: portfolio.Simulation, value: float, mean: str, warns: list[str]
) -> None:
    label = state.get("pf_horizon") or "1M"
    one = portfolio.HORIZONS[label] == 1
    with st.container(border=True):
        headed(
            f"**{label} 뒤 평가액 분포** · VaR·ES",
            f"[정의] 모의 경로 {portfolio.PATHS:,}개의 기간 끝 평가액 분포 및 꼬리"
            " 위험 지표(VaR·ES), 점선은 현재 평가액, 실선은 VaR 95%·99%(하위 5%·1%"
            " 손실 임계치), 파선은 ES 95%(하위 5% 평균 손실)\n\n"
            "[해석] VaR-ES 간격 확대 시 극단 꼬리 손실 심화\n\n"
            "[산출] 정상 블록 부트스트랩 기간 끝 평가액 분위수 및 꼬리 평균"
            f" ({mean})",
            warns,
        )
        st.altair_chart(pf_end_chart(sim.ends, value, sim.tails), alt="분포")
    if one:
        right = st.container()
    else:
        left, right = st.columns(2)
        with left, st.container(border=True, height="stretch"):
            headed(
                "**시뮬레이션 범위** · 5–95%, 25–75%, 중앙값",
                "[정의] 거래일별 모의 평가액 백분위(5·25·50·75·95%) 팬 차트\n\n"
                "[해석] 진한 영역은 경로 중위 50%(25\\~75%), 옅은 영역은"
                " 90%(5\\~95%) 분포 범위\n\n"
                f"[산출] 거래일별 {portfolio.PATHS:,}개 모의 경로 평가액 분위수 집계",
            )
            st.altair_chart(pf_fan_chart(sim.fan), alt="범위")
    with right, st.container(border=True, height="stretch"):
        headed(
            f"**도달 확률 곡선** · {label}",
            PF_REACH_HELP + (PF_REACH_ONE_HELP if one else PF_REACH_LINES_HELP),
        )
        st.altair_chart(pf_reach_chart(sim.reach, single=one), alt="도달 확률")


def pf_backtest(t: portfolio.Tails) -> None:
    expected = quant.BACKTEST_DAYS * (1 - quant.BACKTEST_LEVEL)
    headed(
        f"**꼬리** · 1일 95% VaR·ES 사후검정, 최근 {quant.BACKTEST_DAYS}일",
        f"[정의] 최근 {quant.BACKTEST_DAYS}거래일 역사적 1일 95% VaR·ES 예측"
        " 사후검정(위 경로 VaR 아닌 역사적 1일 예측 대상)\n\n"
        "[해석] 초과 빈도·군집성·크기 기준 예측 적합성 판정\n\n"
        "[산출] 일별 전일까지 실제 수익률 분포 기반 1일 95% 역사적 VaR·ES 산출 후"
        f" 당일 실제 손실 대조, 기대 초과 횟수 {expected:.1f}회, 바젤 신호등"
        f" 등급({PF_LIGHT_NOTE})",
    )
    bt = t.backtest
    if bt is None:
        st.caption(
            f"전체 기간 수익률 {t.history:,}일 (최소 {2 * quant.BACKTEST_DAYS}일 필요)"
        )
        return
    pf_table(
        {
            "초과 횟수": [f"{bt.exceptions}회"],
            "기대": [f"{pf_num(bt.expected, '.1f')}회"],
            "Kupiec p": [pf_num(bt.kupiec, ".2f")],
            "Christoffersen p": [pf_num(bt.christoffersen, ".2f")],
            "ES Z2": [pf_ratio(bt.z2)],
            "Z2 p": [pf_num(bt.es, ".2f")],
            "U p": [pf_num(bt.de_u, ".2f")],
            "C p": [pf_num(bt.de_c, ".2f")],
            "신호등": [PF_LIGHTS[bt.light]],
        },
        "사후검정",
        helps={
            "초과 횟수": "[정의] 실제 일간 손실의 당일 1일 VaR 예측치 초과 거래일수\n\n"
            "[해석] 기대 횟수 대비 과다 시 위험 과소평가 시사\n\n"
            "[산출] 검정 기간 내 (실제 손실 > 예측 VaR) 발생 빈도 합산",
            "기대": "[정의] 모형 적정 가정 시 예상 VaR 초과 발생 횟수\n\n"
            "[해석] 초과 횟수 적정성 판단 기준선\n\n"
            "[산출] 검정 거래일수 × (1 − 신뢰수준 95%)",
            "Kupiec p": "[정의] VaR 초과 빈도 적정성(무조건부 커버리지) 검정 p값\n\n"
            f"[해석] {PF_P_NOTE} VaR 과소·과대 산정 시사\n\n"
            "[산출] Kupiec POF 우도비 검정 유의확률",
            "Christoffersen p": "[정의] VaR 초과 발생의 연속 군집 여부(독립성) 검정"
            " p값\n\n"
            f"[해석] {PF_P_NOTE} 손실 초과일 연속 집중 발생 시사\n\n"
            "[산출] Christoffersen 독립성 마르코프 전이 우도비 검정 유의확률",
            "ES Z2": "[정의] Acerbi-Szekely 비모수 ES 적합성 검정 통계량(Z2)\n\n"
            "[해석] 0 근접 정상, 음(-)의 값 확대 시 실제 꼬리 손실 대비 ES 과소"
            " 산정\n\n"
            "[산출] 1 − Σ(VaR 초과일 실제 손실 ÷ 예측 ES) ÷ 기대 초과 횟수(검정"
            " 거래일수 × 5%)",
            "Z2 p": "[정의] ES Z2 통계량 좌측 꼬리 유의확률\n\n"
            f"[해석] {PF_P_NOTE} ES 위험 과소 산정 판정\n\n"
            "[산출] 부트스트랩 귀무분포 기반 Z2 좌측 꼬리 확률",
            "U p": "[정의] Du-Escanciano 무조건부 꼬리 손실 크기 검정 p값\n\n"
            f"[해석] {PF_P_NOTE} VaR 초과 구간의 손실 분포 크기 부적합 시사\n\n"
            "[산출] 누적 위반 프로세스 평균 검정 유의확률",
            "C p": f"[정의] Du-Escanciano 조건부 꼬리 손실 자기상관(1\\~{quant.DE_LAGS}"
            "일 시차) 검정 p값\n\n"
            f"[해석] {PF_P_NOTE} 극단적 손실의 시차 군집성 존재 시사\n\n"
            "[산출] 누적 위반 프로세스 Box-Pierce 시차 결합 검정 유의확률",
            "신호등": "[정의] 바젤 위원회 기준 VaR 사후검정 신호등 등급\n\n"
            "[산출] 이항분포 누적확률 기반 초과 횟수 구간 분류"
            f"({PF_LIGHT_NOTE})",
        },
    )


def pf_risk_inputs(
    s: Subject, prices: portfolio.Prices, months: int
) -> tuple[Key, portfolio.Joint, float, list[str]] | None:
    j, share, dropped = pf_joint(s, prices, months)
    if j is None:
        st.info("가격 확인 보유 종목 없음")
        return None
    value = s.value * share
    if len(j.window) < portfolio.MIN_DAYS:
        st.info(
            f"공통 수익률 거래일 {len(j.window)}일 (최소 {portfolio.MIN_DAYS}일 필요)"
        )
        return None
    key = (*s.key, months)
    return key, j, value, pf_gaps(dropped, j.late)


def pf_simulation(
    s: Subject,
    prices: portfolio.Prices,
    months: int,
    top: DeltaGenerator,
    p: portfolio.Portfolio | None = None,
) -> None:
    found = pf_risk_inputs(s, prices, months)
    if found is None:
        return
    key, j, value, warns = found
    if s.notes:
        st.caption(" · ".join(s.notes))
    drift, mean = 0.0, PF_OWN_MEAN
    if p is not None:
        with top:
            pf_views_editor(p, j)
        expected = pf_expected(j, p)
        if expected is None:
            pf_checks(key, j)
            return
        drift, mean = expected
    sim = pf_sim(key, j, value, drift, demean=p is not None)
    pf_path_charts(sim, value, mean, warns)
    pf_checks(key, j)


def pf_checks(key: Key, j: portfolio.Joint) -> None:
    days = pf_check_days()
    with st.container(border=True):
        headed(
            "**모형 검증**",
            "[정의] 역사적 1일 95% VaR·ES 꼬리 사후검정 및 부트스트랩 분포 적합성"
            " 검증\n\n"
            f"[해석] 각 검정 p값 {PF_P_NOTE} 모형 예측과 실제 수익률 분포 간 괴리"
            " 시사\n\n"
            "[산출] 과거 시점별 표본외(Out-of-Sample) 예측치와 후속 실현 수익률 대조"
            " 검정",
        )
        pf_backtest(cached("tails", key, partial(shared.app.calc.tails, j)))
        pf_density(
            cached(
                "density", (*key, days), partial(shared.app.calc.density, j, days=days)
            ),
            days,
        )


def pf_check_days() -> int:
    return 1 if (state.get("pf_horizon") or "1M") == "1D" else quant.MONTH_DAYS


def pf_span(days: int) -> str:
    return "1일" if days == 1 else "1개월"


PF_CONFIDENCE = 50.0
PF_VIEWS_WIDTH = 380
PF_BL_DOMINANT = 0.8


def pf_views_editor(p: portfolio.Portfolio, j: portfolio.Joint) -> None:
    if shared.app.demo:
        return
    saved = {v.ticker: v for v in p.views}
    rows = pd.DataFrame(
        {
            "종목": list(j.names),
            "연 기대수익 (%)": [
                saved[n].annual * 100 if n in saved else None for n in j.names
            ],
            "확신도 (%)": [
                saved[n].confidence * 100 if n in saved else PF_CONFIDENCE
                for n in j.names
            ],
        }
    )
    with (
        st.popover(
            "내 견해",
            icon=":material/edit_note:",
            help="[정의] 블랙-리터만 모형 입력용 종목별 주관적 연 기대수익률 및"
            " 확신도(미입력 종목은 견해 없음 처리, 입력값은 포트폴리오별 저장)\n\n"
            "[산출] 사전 균형 기대수익과 견해의 확신도 가중 베이지안 결합",
        ),
        st.form(f"pf_views_form_{p.id}", border=False),
    ):
        edited = st.data_editor(
            rows,
            width=PF_VIEWS_WIDTH,
            hide_index=True,
            disabled=["종목"],
            column_config={
                "연 기대수익 (%)": st.column_config.NumberColumn(format="%.1f"),
                "확신도 (%)": st.column_config.NumberColumn(
                    min_value=1, max_value=100, step=1, format="%d"
                ),
            },
            key=f"pf_views_{p.id}",
        )
        premium = st.number_input(
            "시장 위험프리미엄 (연 %)",
            min_value=0.0,
            value=p.premium * 100,
            step=0.5,
            format="%.1f",
            key=f"pf_premium_{p.id}",
            help="[정의] 시장 포트폴리오(규모 비중 합산)의 무위험수익률 대비 연간 기대"
            " 초과수익률\n\n"
            "[해석] 높을수록 견해 미입력 종목을 포함한 전반적 균형 기대수익 상향\n\n"
            "[산출] 균형 기대수익 = 무위험수익률 + 위험회피계수(시장 위험프리미엄 ÷"
            " 시장 분산) × 공분산 × 시장 비중",
        )
        if st.form_submit_button("저장", key=f"pf_views_save_{p.id}"):
            records = cast("list[dict[str, object]]", edited.to_dict("records"))
            views = tuple(
                portfolio.View(
                    ticker=str(r["종목"]),
                    annual=cast("float", r["연 기대수익 (%)"]) / 100,
                    confidence=cast("float", r["확신도 (%)"]) / 100,
                )
                for r in records
                if pd.notna(cast("float", r["연 기대수익 (%)"]))
                and pd.notna(cast("float", r["확신도 (%)"]))
            )
            kept = tuple(v for v in p.views if v.ticker not in j.names)
            error = pf_store(
                lambda book: portfolio.set_views(
                    book, p.id, (*kept, *views), premium / 100
                )
            )
            if error:
                st.error(error)
            else:
                state.toast = f"견해 {len(views)}건 저장 완료"
                st.rerun()


def pf_expected(j: portfolio.Joint, p: portfolio.Portfolio) -> tuple[float, str] | None:
    sized = portfolio.sizes(shared.app.calc.load_facts(), pf_funds())
    mkt_w, missing = shared.app.calc.market_weights(sized, j.names)
    if mkt_w is None:
        if portfolio_fetcher().facts_progress.running:
            st.info("규모 데이터 수신 중")
        else:
            st.info(f"규모 데이터 없음: {', '.join(missing)}")
        return None
    rf = shared.app.calc.risk_free(pf_rates(), pd.DatetimeIndex(j.full.index))
    found = shared.app.calc.black_litterman(j, rf, mkt_w, p.views, p.premium)
    if found is None:
        return 0.0, PF_ZERO_MEAN
    year = shared.app.calc.per_year(j.full.index)

    def annual(daily: float) -> str:
        return pf_percent((1 + daily) ** year - 1, signed=True)

    top = int(np.argmax(mkt_w))
    columns = ("균형 기대수익", "견해 반영 기대수익")
    with st.container(border=True):
        headed(
            "**기대수익** · 연",
            "[정의] 시장 규모 비중 역산 균형 기대수익 및 견해 결합 사후 기대수익\n\n"
            "[산출] 시가총액·운용자산 가중 역최적화 균형 수익률과 견해의 블랙-리터만"
            " 사후 결합(견해 미입력 시 두 수치 동일, 포트폴리오 행의 견해 반영"
            " 기대수익을 모의 경로 드리프트로 적용)",
            [
                f"보유 자산 규모 비중의 {pf_percent(float(mkt_w[top]), dec=0)}가"
                f" {j.names[top]}에 집중되어 균형 기대수익이 해당 자산 공분산에 좌우됨"
            ]
            if mkt_w[top] >= PF_BL_DOMINANT
            else [],
        )
        pf_table(
            {
                "구분": [portfolio.PORTFOLIO_LABEL, *j.names],
                **{
                    name: [annual(float(mu @ j.w)), *(annual(float(m)) for m in mu)]
                    for name, mu in zip(columns, found, strict=True)
                },
            },
            "기대수익",
            signed=columns,
        )
    return float(found[1] @ j.w), PF_BL_MEAN
