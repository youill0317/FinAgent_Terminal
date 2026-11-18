# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false, reportUnknownMemberType=false
# pyright: reportExplicitAny=false
"""포트폴리오·종목·투자자 뷰 공용 위젯, 차트, 캐시 및 포맷 헬퍼."""

from __future__ import annotations

import dataclasses
import math
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import altair as alt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent import (
    catalog,
    demo,
    etf,
    investors,
    lookthrough,
    perf,
    portfolio,
    portfolio_fetch,
    quant,
)
from finagent.fetch import SourceError
from finagent.ui import palette, shared
from finagent.ui.components import DATE_AXIS, clear_selection, notice
from finagent.ui.shared import etf_updater, limit_note, market_keys, sec_agent

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime

    from numpy.typing import NDArray

    from finagent.sources import MacroFetch
    from finagent.sources import Series as Points

state = st.session_state

CHART_HEIGHT = 240


def headed(text: str, help_: str | None = None, warns: Sequence[str] = ()) -> None:
    if not warns:
        st.markdown(text, help=help_)
        return
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(text, help=help_)
        notice("\n\n".join(warns))


PF_WINDOW_HELP = (
    "[정의] 통계 지표 산출 대상 과거 일간 수익률 표본 구간\n\n"
    "[산출] 기준일 역산 해당 기간 일별 수익률 시계열 추출(시뮬레이션 및 편입"
    f" 효과는 선택 기간과 무관하게 최근 {portfolio.SIM_MONTHS}개월 고정)"
)
PF_HORIZON_HELP = (
    "[정의] 몬테카를로 모의 경로 전망 기간("
    + ", ".join(f"{k} {v}거래일" for k, v in portfolio.HORIZONS.items())
    + ")\n\n"
    "[해석] 기간 확대 시 미래 평가액 분포 폭 및 꼬리 위험 범위 확장\n\n"
    f"[산출] 현재 평가액 기준 최근 {portfolio.SIM_MONTHS}개월 일간 수익률 정상 블록"
    " 부트스트랩 재표집"
)


@st.cache_resource(show_spinner=False)
def portfolio_fetcher() -> portfolio_fetch.Fetcher:
    official = partial(portfolio_fetch.Official, keys=market_keys(), agent=sec_agent())
    return portfolio_fetch.Fetcher(official)


def pf_metric(
    label: str,
    value: str,
    delta: str | None = None,
    **options: Any,  # noqa: ANN401  # st.metric 가변 키워드 인자 전달
) -> None:
    st.metric(label, value, delta, border=True, height="stretch", **options)


@st.fragment
def pf_part(render: Callable[[], None]) -> None:
    render()


def pf_store(change: Callable[[portfolio.Book], portfolio.Book]) -> str:
    try:
        portfolio.save_book(
            shared.app.root, change(portfolio.load_book(shared.app.root))
        )
    except (portfolio.BookError, OSError) as error:
        return str(error)
    return ""


def pf_style(text: str) -> str:
    text = text.removeprefix("연 ")
    if text.startswith("+") or text == "익절":
        return f"color: {palette.current().up}"
    if text.startswith("-") or text == "손절":
        return f"color: {palette.current().down}"
    return ""


def pf_color(value: float) -> Literal["red", "blue", "off"]:
    if not value or math.isnan(value):
        return "off"
    return "red" if value > 0 else "blue"


def pf_table(
    data: dict[str, list[str]],
    alt_text: str,
    *,
    signed: Sequence[str] = (),
    helps: Mapping[str, str] | None = None,
    **options: Any,  # noqa: ANN401  # st.dataframe 가변 키워드 인자 전달
) -> None:
    st.dataframe(
        pd.DataFrame(data).style.map(pf_style, subset=list(signed)),
        hide_index=True,
        column_config={
            name: st.column_config.Column(help=text)
            for name, text in (helps or {}).items()
        },
        alt=alt_text,
        **options,
    )


def pf_picked(key: str) -> int | None:
    cells = state[key]["selection"]["cells"]
    clear_selection(key)
    return int(cells[0][0]) if cells else None


def pf_choose(
    label: str, names: dict[str, str], key: str, first: str | None = None
) -> None:
    if state.get(key) not in names:
        state[key] = first if first in names else next(iter(names), None)
    if names:
        st.selectbox(
            label,
            list(names),
            format_func=names.__getitem__,
            key=key,
            persist_state="session",
            width=280,
        )


def pf_ratio(value: float) -> str:
    if shared.app.demo:
        value = demo.VALUE
    if not math.isfinite(value):
        return portfolio.UNKNOWN
    if round(value, 2) == 0:
        return "0.00"
    return f"{value:.2f}"


def pf_num(value: float, spec: str) -> str:
    if shared.app.demo:
        value = demo.VALUE
    if not math.isfinite(value):
        return portfolio.UNKNOWN
    return format(value, spec)


def pf_shares(quantity: float) -> str:
    return portfolio.shares(demo.VALUE if shared.app.demo else quantity)


def pf_percent(value: float, *, signed: bool = False, dec: int = 1) -> str:
    shown = demo.SHARE if shared.app.demo else value
    return portfolio.percent(shown, signed=signed, dec=dec)


def pf_pp(value: float) -> str:
    text = pf_percent(value, signed=True)
    return text if text == portfolio.UNKNOWN else f"{text}p"


def pf_periods(horizon: str | None = None) -> int:
    if horizon is None:
        window = st.segmented_control(
            "추정 기간",
            list(portfolio.WINDOWS),
            default="3Y",
            required=True,
            key="pf_window",
            help=PF_WINDOW_HELP,
            bind="query-params",
        )
        return portfolio.WINDOWS[window or "3Y"]
    st.segmented_control(
        "전망 기간",
        list(portfolio.HORIZONS),
        default="1M",
        required=True,
        key=horizon,
        help=PF_HORIZON_HELP,
        bind="query-params",
    )
    return portfolio.SIM_MONTHS


type Key = tuple[object, ...]


@st.cache_resource(show_spinner=False, max_entries=2)
def pf_prices(root: str, stamp: int) -> portfolio.Prices:
    del stamp
    return shared.app.calc.load_prices(Path(root), portfolio_fetcher().tickers())


def pf_current() -> portfolio.Prices:
    root = str(shared.app.root)
    return pf_prices(root, portfolio_fetcher().stamp())


@st.cache_data(show_spinner=False, max_entries=2)
def pf_saved_at(stamp: int) -> datetime | None:
    del stamp
    names = portfolio_fetcher().tickers()
    return shared.app.calc.prices_saved_at(names)


def pf_funds() -> dict[str, dict[str, etf.Cell]]:
    return (
        shared.app.calc.load_funds() if shared.app.demo else etf_updater().store.funds()
    )


def pf_key(*parts: object) -> Key:
    return (state.get("pf_prices_stamp", ()), *parts)


def pf_trades_key(p: portfolio.Portfolio) -> Key:
    return pf_key(p.model_dump_json(include={"id", "trades"}))


@st.cache_data(show_spinner=False, max_entries=256)
def cached[T](tag: str, key: Key, _compute: Callable[[], T]) -> T:
    del tag, key
    return _compute()


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_draws(n: int, horizon: int, block: float) -> NDArray[np.int64]:
    if shared.app.demo:
        return np.zeros((1, horizon), dtype=np.int64)
    picks = quant.stationary_indices(n, portfolio.PATHS, horizon, block)
    picks.flags.writeable = False
    return picks


def cached_paths(  # noqa: PLR0913  # 시뮬레이션 경로 캐시 키·입력 인자 일괄 전달
    key: Key,
    value: float,
    j: portfolio.Joint,
    picks: NDArray[np.int64],
    w: NDArray[np.float64] | None = None,
    drift: float = 0.0,
    *,
    demean: bool = True,
) -> portfolio.Simulation:
    def run() -> portfolio.Simulation:
        growth = shared.app.calc.growth(j, picks, w, demean=demean)
        return shared.app.calc.simulation(growth, value, drift)

    return cached("paths", (*key, value, drift, demean), run)


PF_HEAT = 15
PF_LONG_MONTHS = 40
PF_AXIS = f",.{portfolio.DECIMALS}f"


@dataclasses.dataclass(frozen=True, slots=True)
class Subject:
    key: Key
    label: str
    rets: pd.Series
    w: pd.Series
    value: float
    bench: Callable[[pd.DatetimeIndex], pd.Series]
    trim: bool = False
    notes: tuple[str, ...] = ()


def pf_subject(p: portfolio.Portfolio, prices: portfolio.Prices) -> Subject:
    key = pf_trades_key(p)
    items = shared.app.calc.holdings(p, prices.close)
    w = portfolio.weights(items)
    unknown = portfolio.unpriced(items)
    return Subject(
        key=key,
        label="보유 기록",
        rets=cached(
            "held",
            key,
            partial(shared.app.calc.held_returns, p, prices.close, prices.adjusted),
        ),
        w=w,
        value=sum(h.value for h in items if h.ticker in w.index),
        bench=partial(shared.app.calc.market_returns, prices.close),
        notes=(f"제외 (평가액 미확인): {', '.join(unknown)}",) if unknown else (),
    )


def pf_weights(
    s: Subject, prices: portfolio.Prices, months: int
) -> tuple[pd.Series, float, list[str]]:
    if not s.trim or s.w.empty:
        return s.w, 1.0, []

    def trimmed() -> tuple[pd.Series, list[str]]:
        names = portfolio.labels(s.w)
        rets = shared.app.calc.asset_returns(prices.adjusted, names)
        return shared.app.calc.covered(portfolio.window(rets, months), s.w)

    w, dropped = cached("trim", (*s.key, months), trimmed)
    kept = cast("float", s.w.reindex(portfolio.labels(w)).sum())
    return w, float(kept), dropped


def pf_empty() -> pd.Series:
    return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))


def pf_series(points: Points) -> pd.Series:
    return pd.Series(
        [value for _, value in points],
        index=pd.DatetimeIndex([pd.Timestamp(day) for day, _ in points]),
        dtype="float64",
    )


@st.cache_data(show_spinner=False, max_entries=16)
def pf_points(fetch: MacroFetch, day: str, refreshes: int) -> pd.Series:
    del day, refreshes

    async def get() -> Points:
        async with shared.app.rt.dashboard.fetch_context() as ctx:
            return await shared.app.rt.dashboard.series_store.through(
                fetch, ctx, shared.app.rt.dashboard.today()
            )

    return pf_series(shared.app.rt.call(get()))


def pf_rates() -> pd.Series:
    if shared.app.demo:
        return pf_empty()
    day = shared.app.rt.dashboard.today().isoformat()
    try:
        return pf_points(catalog.RISK_FREE, day, shared.app.rt.dashboard.refreshes)
    except (SourceError, OSError, ValueError):
        return pf_empty()


def pf_rates_key(rates: pd.Series) -> tuple[int, str]:
    last = cast("pd.Timestamp", rates.index[-1]) if len(rates) else None
    return len(rates), str(last)


def pf_update(
    fetcher: portfolio_fetch.Fetcher,
    book: portfolio.Book,
    missing: list[str],
    extra: Sequence[str] = (),
) -> None:
    tickers = [*portfolio.book_tickers(book), *extra]
    if not missing and not shared.app.demo and tickers:
        _ = fetcher.start(
            list(dict.fromkeys([*portfolio.wanted(book), *extra])), force=True
        )


def pf_ask(start: Callable[[Sequence[str]], bool], names: Sequence[str]) -> None:
    if not shared.app.demo and not portfolio.missing_packages() and start(names):
        st.rerun()


def pf_jobs(jobs: Sequence[tuple[str, portfolio_fetch.Progress]]) -> None:
    for name, progress in jobs:
        if progress.running:
            st.caption(
                f":material/progress_activity: {name} 수신 중 · {progress.message}"
                f"{limit_note()}"
            )
        elif progress.failed:
            st.warning(progress.message)


def pf_bar(
    fetcher: portfolio_fetch.Fetcher,
    book: portfolio.Book,
    missing: list[str],
    extra: Sequence[str] = (),
) -> None:
    if state.pop("refresh_page", False):
        pf_update(fetcher, book, missing, extra)

    def jobs() -> list[tuple[str, portfolio_fetch.Progress]]:
        return [
            ("가격", fetcher.progress),
            ("ETF 구성", fetcher.look_progress),
            ("규모", fetcher.facts_progress),
        ]

    polling = any(progress.running for _, progress in jobs())

    @st.fragment(run_every=1.0 if polling else None)
    def body() -> None:
        now = jobs()
        ended = polling and not any(progress.running for _, progress in now)
        if ended and not state.get("settings_open"):
            st.rerun()
        saved = pf_saved_at(fetcher.stamp())
        if saved:
            st.caption(f"가격 {saved:%Y-%m-%d %H:%M} 수신")
        elif portfolio.book_tickers(book) or extra:
            st.caption("수신된 가격 데이터 없음")
        pf_jobs(now)
        if missing and not shared.app.demo:
            st.info(
                f"가격 수신 필수 패키지 누락({', '.join(missing)}):"
                f" `{portfolio.INSTALL_HINT}`"
            )

    body()


def pf_no_prices() -> None:
    if portfolio_fetcher().progress.running:
        st.info("가격 데이터 수신 중")
    else:
        st.info("가격 데이터 없음")


PF_DONUT_SLICES = 8
PF_MIN_QUANTITY = 0.0001


def pf_donut(
    rows: list[tuple[str, float, float]],
    text: Callable[[float], str],
    amount: str = "평가액",
    order: Sequence[str] | None = None,
) -> alt.Chart:
    if shared.app.demo:
        rows = [(name, demo.SHARE, value) for name, _, value in rows]
    merged: dict[str, tuple[float, float]] = {}
    for name, weight, value in rows:
        w, v = merged.get(name, (0.0, 0.0))
        merged[name] = (w + weight, v + value)
    ranked = sorted(merged.items(), key=lambda item: item[1][0], reverse=True)
    if order is not None:
        ranked = [(name, merged[name]) for name in order]
    if len(ranked) > PF_DONUT_SLICES:
        rest = ranked[PF_DONUT_SLICES - 1 :]
        other = (sum(w for _, (w, _) in rest), sum(v for _, (_, v) in rest))
        ranked = [*ranked[: PF_DONUT_SLICES - 1], ("기타", other)]
    points = [
        {"name": name, "weight": w, "value": text(v), "rank": i}
        for i, (name, (w, v)) in enumerate(ranked)
    ]
    names = [name for name, _ in ranked]
    return (
        alt.Chart(alt.Data(values=points))
        .mark_arc(innerRadius=50, padAngle=0.01)
        .encode(
            theta=alt.Theta("weight:Q"),
            order=alt.Order("rank:Q"),
            color=alt.Color(
                "name:N",
                sort=names,
                scale=alt.Scale(
                    domain=names, range=palette.current().tones(len(names))
                ),
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


def pf_line_chart(
    points: list[dict[str, str | float]], number: str, height: int = CHART_HEIGHT
) -> alt.Chart:
    if shared.app.demo:
        shown = demo.SHARE if "%" in number else demo.VALUE
        points = [point | {"value": shown} for point in points]
    order = list(dict.fromkeys(str(point["series"]) for point in points))
    return (
        alt.Chart(pf_rows(points))
        .mark_line()
        .encode(
            x=alt.X("day:T", title=None, axis=DATE_AXIS),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format=number)),
            color=alt.Color(
                "series:N",
                title=None,
                legend=alt.Legend(orient="bottom"),
                scale=alt.Scale(domain=order, range=palette.current().series(order)),
            ),
            tooltip=[
                alt.Tooltip("day:T", title="날짜", format="%Y-%m-%d"),
                alt.Tooltip("series:N", title="구분"),
                alt.Tooltip("value:Q", title="값", format=number),
            ],
        )
        .properties(height=height)
    )


def pf_rows(points: list[dict[str, str | float]]) -> pd.DataFrame:
    return pd.DataFrame(points, columns=["day", "series", "value"])


PF_DASH = "—"
PF_SHORT = "기간 부족"


def pf_annual(value: perf.Annualised | None) -> str:
    if value is None:
        return PF_DASH
    if value.annual is not None:
        return f"연 {pf_percent(value.annual, signed=True)}"
    return pf_percent(value.cumulative, signed=True)


def pf_weight_chart(w: pd.Series, label: str = "종목") -> alt.Chart:
    points = [
        {"ticker": str(t), "weight": demo.SHARE if shared.app.demo else float(v)}
        for t, v in w.items()
    ]
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


def pf_joint(
    s: Subject, prices: portfolio.Prices, months: int
) -> tuple[portfolio.Joint | None, float, list[str]]:
    w, share, dropped = pf_weights(s, prices, months)
    if w.empty:
        return None, share, dropped
    j = cached(
        "joint",
        (*s.key, months),
        partial(shared.app.calc.joint, prices.adjusted, w, months),
    )
    return j, share, dropped


def pf_gaps(dropped: list[str], late: str) -> list[str]:
    warns: list[str] = []
    if dropped:
        more = " 외" if len(dropped) > PF_HEAT else ""
        warns.append(
            f"기간 내 데이터 결측 제외 종목({len(dropped)}개):"
            f" {', '.join(dropped[:PF_HEAT])}{more}"
        )
    if late:
        warns.append(f"기간 중 데이터 개시 종목: {late}")
    return warns


PF_BETA_HELP = (
    f"[정의] {portfolio.BENCHMARK_LABEL} 가격지수 일간 수익률 대비 민감도\n\n"
    "[해석] 1은 지수 수준 변동, 1 초과는 지수 대비 고변동, 0 근접은 지수 무관\n\n"
    "[산출] 최근일 가중(지수적 감쇠)·이상치 절단 적용 Welch 일간 수익률 회귀 기울기"
)

PF_COMPACT = 1e8


def pf_compact(total: float) -> bool:
    return abs(total) >= PF_COMPACT


def pf_money(value: float, *, compact: bool = False, signed: bool = False) -> str:
    if shared.app.demo:
        value = demo.VALUE
    if not compact or not math.isfinite(value):
        return portfolio.money(value, signed=signed)
    sign = "-" if value < 0 else ("+" if signed and value > 0 else "")
    return sign + investors.dollars(abs(value))


PF_THIN = "표본 부족"


def pf_look(
    w: pd.Series,
) -> tuple[lookthrough.Lookthrough, dict[str, etf.Composition]]:
    comps = shared.app.calc.compositions(portfolio.labels(w))
    weights = {str(t): float(v) for t, v in w.items()}
    return lookthrough.effective_holdings(weights, comps), comps
