# pyright: reportUnusedCallResult=false, reportImplicitStringConcatenation=false
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""ETF 유니버스 수익률 트리맵 및 구성 종목 상세 화면."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, cast

import altair as alt
import pandas as pd
import streamlit as st

from finagent import etf, etf_universe, fund_filings
from finagent.analysis import documents
from finagent.etf_universe import Product
from finagent.settings import EtfChoice, load_etf, save_etf
from finagent.treemap import Rect, squarify
from finagent.ui import components, palette

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from finagent.fundamentals import Company

ALL: Final = "전체"
MAP_SIZE: Final = (1200.0, 560.0)
MAP_HEAD: Final = 18.0
MAP_GAP: Final = 4.0
MAP_FONT: Final = (9.0, 26.0)
MAP_POWER: Final = 0.5
MAP_FLAT: Final = 0.3
MAP_BOUND: Final = 0.9
MAP_FLOOR: Final = 0.004
DETAIL_HEIGHT: Final = 240
WEIGHT_STEP: Final = 24
WEIGHT_ROOM: Final = 0.12
MAJORITY: Final = 50.0
MARKET: Final = "VOO"
ASSET_LABELS: Final = {
    "stock": "주식",
    "bond": "채권",
    "cash": "현금",
    "collateral": "담보",
    "fund": "펀드",
    "other": "기타",
    "preferred": "우선주",
    "convertible": "전환사채",
}
RATING_LABELS: Final = {
    "us_government": "미국 정부",
    "aaa": "AAA",
    "aa": "AA",
    "a": "A",
    "bbb": "BBB",
    "bb": "BB",
    "b": "B",
    "below_b": "B 미만",
    "other": "기타",
}
YEAR_BARS: Final = 252
SPREADS: Final[tuple[tuple[str, str, Callable[[str], str], str], ...]] = (
    ("섹터 비중", "sector_weights", components.sector_label, "주식"),
    (
        "신용등급 비중",
        "credit_weights",
        lambda key: RATING_LABELS.get(key, key),
        "채권",
    ),
)
DESCRIPTION_LIMIT: Final = 600
FORM_DEFAULTS: Final[dict[str, object]] = {
    "etf_add_ticker": "",
    "etf_add_group": None,
    "etf_add_category": None,
    "etf_add_multiple": 1.0,
}
STOCK_FACTS: Final = (("sector", "sector"), ("industry", "text"), ("country", "text"))
STOCK_METRICS: Final[tuple[tuple[str, str, str, str | None], ...]] = (
    ("market_cap", "시가총액", "big", None),
    (
        "per",
        "PER",
        "num",
        "[정의] 주가수익비율(Price-to-Earnings Ratio)\n\n"
        "[해석] 높을수록 이익 대비 고평가 상태\n\n"
        "[산출] 현재 주가 ÷ 최근 12개월 주당순이익(EPS)",
    ),
    (
        "forward_per",
        "선행 PER",
        "num",
        "[정의] 향후 12개월 예상 이익 기준 주가수익비율\n\n"
        "[해석] 현행 PER 하회 시 향후 이익 성장 기대 반영\n\n"
        "[산출] 현재 주가 ÷ 향후 12개월 컨센서스 EPS",
    ),
    (
        "dividend_yield",
        "배당수익률",
        "pct",
        "[정의] 주가 대비 연간 현금배당 비율\n\n"
        "[해석] 높을수록 현금 배당 매력 우위\n\n"
        "[산출] 연간 주당배당금 ÷ 현재 주가",
    ),
    (
        "roe",
        "ROE",
        "pct",
        "[정의] 자기자본이익률(Return on Equity)\n\n"
        "[해석] 높을수록 자기자본 활용 수익성 우위\n\n"
        "[산출] 당기순이익 ÷ 자기자본",
    ),
    (
        "net_margin",
        "순이익률",
        "pct",
        "[정의] 매출액 대비 당기순이익 비율\n\n"
        "[해석] 높을수록 최종 비용 차감 후 마진 구조 우위\n\n"
        "[산출] 당기순이익 ÷ 매출액",
    ),
    (
        "revenue_growth",
        "매출 성장률",
        "pct",
        "[정의] 전년 동기 대비 매출액 증감률\n\n"
        "[해석] 양(+)의 값 확대 시 외형 성장 가속\n\n"
        "[산출] (당기 매출액 − 전년 동기 매출액) ÷ 전년 동기 매출액",
    ),
)
VIEW_HELP: Final = (
    "[정의] 분류·카테고리별 ETF 운용자산 및 기간 수익률 트리맵\n\n"
    "[해석] 상승 시 빨강, 하락 시 파랑(변동폭 비례 채도), 점선 테두리는 레버리지·인버스"
    " 등 비등배 상품으로 같은 분류 안에서 색 반전, 셀 호버 시 전 기간 수익률·운용자산·"
    "거래대금·총보수 표시, 셀 클릭 시 상세 모달 표시\n\n"
    "[산출] 셀 면적은 운용자산(AUM) 제곱근 비례(미확인·소규모 상품은 최소 면적 적용)"
)
EXPENSE_HELP: Final = (
    "[정의] 순자산 대비 연간 총보수율(Expense Ratio)\n\n"
    "[해석] 동일 카테고리 내 낮을수록 장기 복리 비용 우위\n\n"
    "[산출] 연간 운용·관리 제반 비용 ÷ 펀드 순자산총액(일별 기준가 차감 반영)"
)
AUM_HELP: Final = (
    "[정의] 펀드 전체 운용 순자산총액(AUM, USD)\n\n"
    "[해석] 규모가 작을수록 호가 스프레드 확대 및 유동성 저하 가능성\n\n"
    "[산출] 발행 증권 클래스 전체 순자산가치 합산"
)
PICK_HELP: Final = "기존 항목 선택 또는 신규 명칭 직접 입력"
YIELD_HELP: Final = (
    "[정의] 현재 가격 대비 연간 분배금 지급 비율\n\n"
    "[해석] 높을수록 현금흐름 수익 비중 우위\n\n"
    "[산출] 최근 12개월 누적 분배금 ÷ 현재 가격"
)
TURNOVER_HELP: Final = (
    f"[정의] 최근 {etf.TURNOVER_DAYS}거래일 일평균 거래대금(USD)\n\n"
    "[해석] 클수록 대량 매매 시 체결 용이성 및 유동성 우위\n\n"
    f"[산출] 최근 {etf.TURNOVER_DAYS}거래일 (일별 종가 × 거래량) 산술평균"
)
GAP_HELP: Final = (
    "[정의] 순자산가치(NAV) 대비 시장 종가 괴리율\n\n"
    "[해석] 양(+)은 할증, 음(-)은 할인 거래, 괴리 지속 시 기초자산 유동성 제약 시사\n\n"
    "[산출] (전일 종가 ÷ 주당 순자산가치 − 1) × 100"
)
PRICE_HELP: Final = (
    "[정의] 배당·분할 반영 수정 종가 시계열\n\n"
    "[해석] 분배금 재투자 가정 총수익 기준 가격 추이\n\n"
    "[산출] 원시 종가에 배당·액면분할 조정계수 반영"
)
MIX_HELP: Final = (
    "[정의] 순자산 내 주식·채권·현금 등 자산군별 구성 비중(%)\n\n"
    "[해석] 펀드 자산배분 성격 및 주된 위험 노출 자산군 파악\n\n"
    "[산출] 자산군별 평가액 합계 ÷ 펀드 순자산총액"
)
TOP_HELP: Final = (
    "[정의] 순자산 대비 편입 비중 상위 보유 종목 목록\n\n"
    "[해석] 상위 종목 집중도 확인 및 주식 행 선택 시 하단 기업 상세 열람\n\n"
    "[산출] 개별 보유 종목 평가액 ÷ 펀드 순자산총액"
)
WEIGHT_HELP: Final = (
    "[정의] 해당 ETF 순자산 내 개별 종목 편입 비중\n\n"
    "[해석] 높을수록 ETF 성과에 대한 개별 종목 영향력 확대\n\n"
    "[산출] 종목 보유 평가액 ÷ ETF 순자산총액"
)
MOVES_HELP: Final = (
    f"[정의] 기간별 총수익률 및 시장({MARKET}) 대비 민감도(베타)\n\n"
    "[해석] 수익률 양(+)은 빨강·음(-)은 파랑, 베타 1은 시장 동행·1 초과는 고변동\n\n"
    f"[산출] 수정 종가 기간 수익률 및 최근 {YEAR_BARS}거래일 일간 수익률 공분산 ÷"
    f" {MARKET} 분산"
)
MOVES_GAP: Final = "\u2003"
REFUSED: Final = {
    "개별 주식": "개별 주식 추가 불가 (ETF·ETN 전용)",
    "찾을 수 없음": "미확인 티커",
    "확인 실패": "티커 확인 실패",
}


@dataclass(frozen=True, slots=True)
class EtfContext:
    settings: Path
    updater: etf.Updater | None
    held: Callable[[], Iterable[str]]
    confirm: Callable[[str], etf.Verdict]
    note: Callable[[], str]
    rows: Callable[[Sequence[str]], pd.DataFrame]
    funds: Callable[[], dict[str, dict[str, etf.Cell]]]
    stamp: Callable[[], object]
    detail: Callable[[str], etf.Composition | None]
    company: Callable[[str], Company]
    stock_rows: Callable[[str], pd.DataFrame]


def _save(ctx: EtfContext, choice: EtfChoice) -> bool:
    try:
        save_etf(ctx.settings, choice)
    except OSError as error:
        st.error(f"설정 저장 실패: {error}")
        return False
    st.session_state.etf_universe_changed = True
    if ctx.updater is not None:
        products = etf_universe.effective(choice.added, choice.removed)
        ctx.updater.prune(products, [*ctx.held(), MARKET])
    return True


def _add(ctx: EtfContext, product: Product) -> bool:
    choice = load_etf(ctx.settings)
    known = {p.ticker for p in etf_universe.effective(choice.added, choice.removed)}
    if product.ticker not in known:
        with st.spinner(f"{product.ticker} 확인 중…"):
            verdict = ctx.confirm(product.ticker)
        if verdict != "확인됨":
            st.error(REFUSED[verdict])
            return False
    added = (*(p for p in choice.added if p.ticker != product.ticker), product)
    removed = tuple(t for t in choice.removed if t != product.ticker)
    if not _save(ctx, replace(choice, added=added, removed=removed)):
        return False
    if ctx.updater is not None and product.ticker not in known:
        ctx.updater.add(product.ticker)
    st.toast(f"{product.ticker} 유니버스 추가 완료")
    return True


def _remove(ctx: EtfContext, tickers: list[str]) -> bool:
    choice = load_etf(ctx.settings)
    defaults = {p.ticker for p in etf_universe.DEFAULTS}
    added = tuple(p for p in choice.added if p.ticker not in tickers)
    removed = (*choice.removed, *(t for t in tickers if t in defaults))
    saved = _save(
        ctx, replace(choice, added=added, removed=tuple(dict.fromkeys(removed)))
    )
    if saved:
        st.toast(f"{len(tickers)}개 종목 유니버스 제외 완료")
    return saved


def _restore(ctx: EtfContext) -> None:
    choice = load_etf(ctx.settings)
    if _save(ctx, replace(choice, added=(), removed=())):
        st.toast("기본 유니버스 복원 완료")


def _submit(ctx: EtfContext) -> None:
    state = st.session_state
    name = str(state.etf_add_ticker).strip().upper()
    group, category = state.etf_add_group, state.etf_add_category
    if not name or not group or not category:
        st.error("티커·분류·카테고리 필수 입력")
        return
    product = Product(name, str(group), str(category), float(state.etf_add_multiple))
    if _add(ctx, product):
        state.update(FORM_DEFAULTS)


def _ask(action: str) -> None:
    st.session_state.etf_action = action


def _perform(ctx: EtfContext, action: str) -> None:
    state = st.session_state
    if action == "add":
        _submit(ctx)
    elif action == "restore":
        _restore(ctx)
    elif _remove(ctx, list(state.etf_remove)):
        state.etf_remove = []


def universe_settings(ctx: EtfContext) -> None:
    action = st.session_state.pop("etf_action", None)
    if action is not None:
        _perform(ctx, action)
    choice = load_etf(ctx.settings)
    products = etf_universe.effective(choice.added, choice.removed)
    changed = len(choice.added) + len(choice.removed)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(
            f"{len(products)}개 · 분류 {len(etf_universe.groups(products))}개"
            f" · 카테고리 {len(etf_universe.categories(products))}개"
        )
        st.button(
            "기본값으로",
            icon=":material/restart_alt:",
            key="etf_restore",
            disabled=not changed,
            on_click=_ask,
            args=("restore",),
        )
    with st.form("etf_add", border=True):
        st.markdown("**상품 추가**")
        cols = st.columns([1, 2, 2, 1])
        cols[0].text_input("티커", key="etf_add_ticker")
        cols[1].selectbox(
            "분류",
            etf_universe.groups(products),
            index=None,
            accept_new_options=True,
            key="etf_add_group",
            placeholder="분류 선택",
            help=PICK_HELP,
        )
        cols[2].selectbox(
            "카테고리",
            etf_universe.categories(products),
            index=None,
            accept_new_options=True,
            key="etf_add_category",
            placeholder="카테고리 선택",
            help=PICK_HELP,
        )
        st.session_state.setdefault("etf_add_multiple", 1.0)
        cols[3].number_input("배수", step=1.0, format="%g", key="etf_add_multiple")
        st.form_submit_button("추가", type="primary", on_click=_ask, args=("add",))
    with st.container(border=True):
        st.markdown("**상품 제거**")
        picked = st.multiselect(
            "제거할 상품",
            [p.ticker for p in products],
            key="etf_remove",
            label_visibility="collapsed",
            placeholder="티커 선택",
        )
        st.button(
            "제거",
            key="etf_remove_button",
            disabled=not picked,
            on_click=_ask,
            args=("remove",),
        )


def _bar(ctx: EtfContext) -> None:
    updater = ctx.updater
    if updater is None:
        return
    state = st.session_state
    for message in state.pop("etf_finished", ()):
        st.toast(message)
    was = (updater.progress.running, updater.slow.running)

    @st.fragment(run_every=1.0 if any(was) else None)
    def body() -> None:
        items = (updater.progress, updater.slow)
        ended = [
            item
            for ran, item in zip(was, items, strict=True)
            if ran and not item.running
        ]
        if ended and not state.get("settings_open"):
            state.etf_finished = [
                item.message for item in ended if item.message and not item.failed
            ]
            st.rerun()
        for item in items:
            if item.running:
                done = item.done / item.total if item.total else 0.0
                count = f" {item.done}/{item.total}" if item.total else ""
                st.progress(done, text=f"{item.stage}{count}{ctx.note()}")
            elif item.failed:
                st.warning(item.message)

    body()


def _fmt(value: object, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if kind == "sector":
        value = components.sector_label(str(value))
    if kind == "text" or not isinstance(value, int | float):
        return str(value)
    if kind == "pct":
        return f"{value:,.2f}%"
    if kind == "big":
        for size, unit in ((1e12, "조"), (1e9, "B"), (1e6, "M")):
            if abs(value) >= size:
                return f"{value / size:,.1f}{unit}"
        return f"{value:,.0f}"
    return f"{value:,.2f}"


@st.cache_data(show_spinner=False, max_entries=4)
def _prepared(
    stamp: object,
    tickers: tuple[str, ...],
    _ctx: EtfContext,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, etf.Cell]]]:
    _ = stamp
    rows = _ctx.rows(list(tickers))
    return rows, etf.returns_frame(rows), _ctx.funds()


def _choice_filters(
    ctx: EtfContext, products: Sequence[Product], choice: EtfChoice
) -> EtfChoice:
    groups = etf_universe.groups(products)
    group_options = [ALL, *groups]
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.subheader("수익률 지도", help=VIEW_HELP, width="content")
        st.space("stretch")
        group = st.selectbox(
            "분류",
            group_options,
            index=group_options.index(choice.group) if choice.group in groups else 0,
            key="etf_group",
            width=240,
        )
        picked_group = None if group == ALL else group
        categories = etf_universe.categories(products, picked_group)
        category_options = [ALL, *categories]
        category = st.selectbox(
            "카테고리",
            category_options,
            index=category_options.index(choice.category)
            if choice.category in categories
            else 0,
            key="etf_category",
            width=280,
        )
    period = st.segmented_control(
        "기간",
        list(etf.PERIODS),
        default=choice.period if choice.period in etf.PERIODS else EtfChoice().period,
        required=True,
        key="etf_period",
        label_visibility="collapsed",
    )
    picked = replace(
        choice,
        group=picked_group,
        category=None if category == ALL else category,
        period=str(period),
    )
    if picked != choice:
        try:
            save_etf(ctx.settings, picked)
        except OSError as error:
            st.error(f"설정 저장 실패: {error}")
    return picked


def _map_cells(
    table: pd.DataFrame, period: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    nested: dict[str, dict[str, list[dict[str, object]]]] = {}
    for _, found in table.iterrows():
        row = cast("dict[str, object]", found.to_dict())
        inside = nested.setdefault(str(row["분류"]), {})
        inside.setdefault(str(row["카테고리"]), []).append(row)
    known = cast("pd.Series", table["운용자산"]).fillna(0.0).clip(lower=0.0)
    least = (float((known**MAP_POWER).sum()) or 1.0) * MAP_FLOOR

    def area(row: dict[str, object]) -> float:
        aum = _number(row["운용자산"])
        return max(aum**MAP_POWER, least) if aum > 0 else least

    def size(rows: list[dict[str, object]]) -> float:
        return sum(area(row) for row in rows)

    cells: list[dict[str, object]] = []
    heads: list[dict[str, object]] = []
    groups = list(nested.items())
    sizes = [sum(size(rows) for rows in inside.values()) for _, inside in groups]
    for (group, inside), outer in zip(
        groups, squarify(sizes, Rect(0.0, 0.0, *MAP_SIZE)), strict=True
    ):
        titled = outer.h >= 3 * MAP_HEAD and outer.w >= MAP_HEAD * len(group)
        box = outer.inset(MAP_GAP, MAP_HEAD if titled else 0.0)
        if titled:
            heads.append({"x": box.x, "y": outer.y + MAP_GAP / 2, "text": group})
        categories = list(inside.values())
        for rows, part in zip(
            categories, squarify([size(rows) for rows in categories], box), strict=True
        ):
            placed = squarify([area(row) for row in rows], part)
            cells.extend(
                _map_cell(row, r, period) for row, r in zip(rows, placed, strict=True)
            )
    return cells, heads


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else math.nan


def _move(value: object) -> str:
    moved = _number(value)
    return "—" if math.isnan(moved) else f"{moved:+.1f}%"


def _map_cell(row: Mapping[str, object], r: Rect, period: str) -> dict[str, object]:
    ticker = str(row["티커"])
    moved = row.get(period)
    known = isinstance(moved, int | float) and not math.isnan(moved)
    multiple = _number(row["배수"])
    font = min(r.w / (0.62 * len(ticker) + 0.8), r.h / 1.5, MAP_FONT[1])
    return {
        "ticker": ticker,
        "name": _fmt(row["이름"], "text"),
        "group": row["분류"],
        "category": row["카테고리"],
        "aum": _fmt(row["운용자산"], "big"),
        "turnover": _fmt(row["거래대금"], "big"),
        "expense": _fmt(row["총보수"], "pct"),
        "multiple": f"{multiple:g}",
        "odd": multiple != 1,
        **{name: _move(row.get(name)) for name in etf.PERIODS},
        "ret": float(cast("float", moved)) if known else 0.0,
        "x": r.x,
        "x2": r.x + r.w,
        "y": r.y,
        "y2": r.y + r.h,
        "cx": r.x + r.w / 2,
        "cy": r.y + r.h / 2,
        "font": font if font >= MAP_FONT[0] else 0.0,
    }


def _map(
    cells: list[dict[str, object]], heads: list[dict[str, object]]
) -> alt.LayerChart:
    colors = palette.current()
    flat = palette.mix(colors.paper, colors.guide, MAP_FLAT)
    spread = sorted(abs(cast("float", cell["ret"])) for cell in cells)
    bound = max(1.0, float(math.ceil(spread[int(MAP_BOUND * (len(spread) - 1))])))
    for cell in cells:
        strong = abs(cast("float", cell["ret"])) > bound / 2
        cell["tone"] = colors.paper if strong else colors.ink
    width, height = MAP_SIZE
    x = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, width], nice=False))
    y = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[height, 0], nice=False))
    base = alt.Chart(alt.Data(values=cells))
    boxes = base.mark_rect(stroke=colors.paper, strokeWidth=1).encode(
        x=x,
        x2="x2:Q",
        y=y,
        y2="y2:Q",
        color=alt.Color(
            "ret:Q",
            scale=alt.Scale(
                domain=[-bound, 0, bound],
                range=[colors.down, flat, colors.up],
                interpolate="rgb",
                clamp=True,
            ),
            legend=alt.Legend(
                orient="bottom", title=None, labelExpr="datum.label + '%'"
            ),
        ),
    )
    tickers = base.mark_text(fontWeight="bold", baseline="middle").encode(
        x=alt.X("cx:Q", axis=None, scale=alt.Scale(domain=[0, width], nice=False)),
        y=alt.Y("cy:Q", axis=None, scale=alt.Scale(domain=[height, 0], nice=False)),
        text="ticker:N",
        size=alt.Size("font:Q", scale=None, legend=None),
        color=alt.Color("tone:N", scale=None),
    )
    titles = (
        alt.Chart(alt.Data(values=heads))
        .mark_text(
            align="left",
            baseline="top",
            dx=2,
            dy=2,
            fontWeight="bold",
            fontSize=MAP_FONT[0] + 2,
            color=colors.ink,
        )
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    odd = (
        base.transform_filter(alt.datum.odd)
        .mark_rect(fillOpacity=0, stroke=colors.ink, strokeWidth=1.5, strokeDash=[4, 3])
        .encode(x=x, x2="x2:Q", y=y, y2="y2:Q")
    )
    pick = alt.selection_point(name="pick", fields=["ticker"], on="click")
    hit = (
        base.mark_rect(opacity=0)
        .encode(
            x="x:Q",
            x2="x2:Q",
            y="y:Q",
            y2="y2:Q",
            tooltip=[
                alt.Tooltip("ticker:N", title="티커"),
                alt.Tooltip("name:N", title="이름"),
                alt.Tooltip("group:N", title="분류"),
                alt.Tooltip("category:N", title="카테고리"),
                *(alt.Tooltip(f"{name}:N", title=name) for name in etf.PERIODS),
                alt.Tooltip("aum:N", title="운용자산"),
                alt.Tooltip("turnover:N", title="거래대금(20일 평균)"),
                alt.Tooltip("expense:N", title="총보수"),
                alt.Tooltip("multiple:N", title="배수"),
            ],
        )
        .add_params(pick)
    )
    return alt.LayerChart(layer=[boxes, odd, tickers, titles, hit]).properties(
        height=height
    )


def _picked(event: object) -> str | None:
    state = cast("Mapping[str, object]", event)
    selection = cast("Mapping[str, object]", state.get("selection") or {})
    found = selection.get("pick")
    items = cast("list[Mapping[str, object]]", found if isinstance(found, list) else [])
    return str(items[0]["ticker"]) if items and "ticker" in items[0] else None


def _metric(
    label: str,
    cell: etf.Cell | None,
    kind: str,
    help: str | None = None,  # noqa: A002  # st.metric 인자명 대응
) -> None:
    st.metric(
        label,
        _fmt(None if cell is None else cell.value, kind),
        help=help,
        border=True,
        height="stretch",
    )


def _weights(weights: Mapping[str, float]) -> alt.LayerChart:
    colors = palette.current()
    values = [round(v, 2) for v in weights.values()]
    frame = pd.DataFrame(
        {
            "항목": list(weights),
            "비중(%)": values,
            "끝": [max(v, 0.0) for v in values],
        }
    )
    low, top = min(*values, 0.0), max(*values, 0.0)
    span = (top - low) or 1.0
    base = alt.Chart(frame).encode(
        y=alt.Y("항목:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180)),
        tooltip=["항목", "비중(%)"],
    )
    reach = alt.Scale(domain=[low, top + span * WEIGHT_ROOM], nice=False)
    bars = base.mark_bar(color=colors.muted).encode(
        x=alt.X("비중(%):Q", axis=None, scale=reach)
    )
    labels = base.mark_text(align="left", dx=4, color=colors.ink).encode(
        x=alt.X("끝:Q", axis=None, scale=reach),
        text=alt.Text("비중(%):Q", format=".1f"),
    )
    return alt.LayerChart(layer=[bars, labels]).properties(height=alt.Step(WEIGHT_STEP))


def _donut(weights: Mapping[str, float]) -> alt.Chart:
    names = sorted(weights, key=lambda name: -weights[name])
    points = [
        {"name": name, "weight": weights[name], "rank": i}
        for i, name in enumerate(names)
    ]
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
                alt.Tooltip("name:N", title="자산"),
                alt.Tooltip("weight:Q", title="비중(%)", format=".2f"),
            ],
        )
        .properties(height=DETAIL_HEIGHT)
    )


def _price_line(rows: pd.DataFrame, ticker: str) -> None:
    series = etf.wide(rows, "adj_close")
    if ticker not in series.columns:
        return
    line = cast("pd.Series", series[ticker]).dropna().tail(260)
    frame = pd.DataFrame(
        {
            "date": cast("pd.DatetimeIndex", line.index).strftime("%Y-%m-%d"),
            "value": line.to_numpy(),
        }
    )
    st.altair_chart(
        alt.Chart(frame)
        .mark_line(color=palette.current().ink)
        .encode(
            x=alt.X("date:T", title=None, axis=components.DATE_AXIS),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
                alt.Tooltip("value:Q", title="수정 종가", format=",.2f"),
            ],
        )
        .properties(height=DETAIL_HEIGHT),
    )


def _mix(found: etf.Composition | None) -> dict[str, float]:
    if found is None:
        return {}
    if found.full:
        mix = fund_filings.asset_mix(found.rows)
        return {ASSET_LABELS.get(k, k): v for k, v in mix.items() if v}
    return {ASSET_LABELS.get(k, k): v for k, v in (found.assets or {}).items()}


def _beta(stock: pd.DataFrame, market: pd.DataFrame, symbol: str) -> float | None:
    prices = etf.wide(pd.concat([stock, market]), "adj_close")
    if symbol not in prices.columns or MARKET not in prices.columns:
        return None
    both = list(dict.fromkeys((symbol, MARKET)))
    daily = prices[both].pct_change(fill_method=None).dropna()
    own, market_daily = (
        cast("pd.Series", daily[name]).tail(YEAR_BARS) for name in (symbol, MARKET)
    )
    spread = float(cast("float", market_daily.var()))
    if not spread or math.isnan(spread):
        return None
    return float(own.cov(market_daily)) / spread


def _toned(value: object) -> str:
    moved = _number(value)
    if math.isnan(moved) or moved == 0:
        return _move(value)
    return f":{'red' if moved > 0 else 'blue'}[{_move(value)}]"


def _market(ctx: EtfContext, rows: pd.DataFrame, symbol: str, title: str) -> pd.Series:
    found = etf.returns_frame(rows)
    measured = (
        cast("pd.Series", found.loc[symbol]) if symbol in found.index else pd.Series()
    )
    beta = _fmt(_beta(rows, ctx.rows([MARKET]), symbol), "num")
    st.markdown(f"**{title}**", help=PRICE_HELP)
    st.markdown(
        MOVES_GAP.join(
            [
                *(
                    f":gray[{name}] **{_toned(measured.get(name))}**"
                    for name in etf.PERIODS
                ),
                f":gray[베타] **{beta}**",
            ]
        ),
        help=MOVES_HELP,
    )
    _price_line(rows, symbol)
    return measured


def _company(ctx: EtfContext, row: fund_filings.Row) -> None:
    ticker = row.ticker or ""
    with st.spinner(f"{ticker} 데이터 수신 중…"):
        found = ctx.company(ticker)
    cells = found.cells
    currency = cells.get("currency")
    st.title(f"{ticker} · {row.name}", anchor=False)
    facts = [
        _fmt(cell.value, kind)
        for name, kind in STOCK_FACTS
        if (cell := cells.get(name)) is not None
    ]
    if facts:
        st.caption(" · ".join(facts))
    description = cells.get("description")
    if description is not None:
        text = str(description.value)
        if len(text) > DESCRIPTION_LIMIT:
            text = text[:DESCRIPTION_LIMIT].rsplit(". ", 1)[0] + "."
        st.caption(text)
    with st.container(horizontal=True, key="etf_stock_cards"):
        st.metric(
            "ETF 내 비중",
            _fmt(row.weight, "pct"),
            help=WEIGHT_HELP,
            border=True,
            height="stretch",
        )
        for name, label, kind, help_ in STOCK_METRICS:
            cell = cells.get(name)
            text = _fmt(None if cell is None else cell.value, kind)
            if name == "market_cap" and currency is not None and text != "—":
                text = f"{text} {currency.value}"
            st.metric(label, text, help=help_, border=True, height="stretch")
    days = [
        cell.as_of
        for name in (*(n for n, _ in STOCK_FACTS), *(n for n, *_ in STOCK_METRICS))
        if (cell := cells.get(name)) is not None
    ]
    if days:
        st.caption(f"기준일 {max(days)}")
    _ = _market(ctx, ctx.stock_rows(found.symbol), found.symbol, "1년 주가 · 수정 종가")


def _holdings(ticker: str, found: etf.Composition) -> fund_filings.Row | None:
    top = fund_filings.top(found.rows)
    total = sum(r.weight for r in top)
    st.markdown(
        f"**상위 구성** · 상위 {len(top)}개 합계 {total:.1f}% · 기준일 {found.as_of}",
        help=TOP_HELP,
    )
    frame = pd.DataFrame(
        {
            "이름": [r.name for r in top],
            "티커": [r.ticker or "" for r in top],
            "비중": [round(r.weight, 2) for r in top],
            "자산": [ASSET_LABELS[r.asset] for r in top],
            "국가": [r.country or "" for r in top],
        }
    )
    largest = max(r.weight for r in top)
    event = st.dataframe(
        frame,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"etf_holdings_{ticker}",
        alt="상위 구성",
        column_config={
            "비중": st.column_config.ProgressColumn(
                "비중",
                format="%.2f%%",
                min_value=0.0,
                max_value=largest if largest > 0 else 1.0,
            )
        },
    )
    rows = event.selection.rows
    chosen = top[rows[0]] if rows else None
    return chosen if chosen is not None and chosen.asset == "stock" else None


def _fund(
    ctx: EtfContext, ticker: str, cells: Mapping[str, etf.Cell], own: pd.DataFrame
) -> None:
    family, inception = cells.get("family"), cells.get("inception")
    nav, close = cells.get("nav"), cells.get("previous_close")
    gap = (
        (float(cast("float", close.value)) / float(cast("float", nav.value)) - 1) * 100
        if nav is not None and close is not None and nav.value
        else None
    )
    notes = [
        *([] if family is None else [str(family.value)]),
        *([] if inception is None else [f"설정일 {inception.value}"]),
        *([] if nav is None else [f"NAV 기준일 {nav.as_of}"]),
    ]
    if notes:
        st.caption(" · ".join(notes))
    measured = _market(ctx, own, ticker, "1년 가격 · 수정 종가")
    with st.container(horizontal=True, key="etf_fund_cards"):
        _metric("운용자산", cells.get("aum"), "big", AUM_HELP)
        st.metric(
            "거래대금",
            _fmt(measured.get("거래대금"), "big"),
            help=TURNOVER_HELP,
            border=True,
            height="stretch",
        )
        _metric("총보수", cells.get("expense"), "pct", EXPENSE_HELP)
        _metric("분배수익률", cells.get("yield"), "pct", YIELD_HELP)
        st.metric(
            "NAV 대비 괴리",
            _fmt(gap, "pct"),
            help=GAP_HELP,
            border=True,
            height="stretch",
        )


def _composition(
    ticker: str, found: etf.Composition, cells: Mapping[str, etf.Cell]
) -> fund_filings.Row | None:
    mix = _mix(found)
    spreads = [
        (title, cell.as_of, _named(cell, label))
        for title, name, label, asset in SPREADS
        if (cell := cells.get(name)) is not None and mix.get(asset, 0) >= MAJORITY
    ]
    chosen = None
    left, right = st.columns([1, 2])
    if mix:
        with left:
            st.markdown(f"**자산 구성** · 기준일 {found.as_of}", help=MIX_HELP)
            st.altair_chart(_donut(mix) if min(mix.values()) >= 0 else _weights(mix))
    with right:
        for title, as_of, weights in spreads:
            st.markdown(f"**{title}** · 기준일 {as_of}")
            st.altair_chart(_weights(weights))
        if found.rows and not spreads:
            chosen = _holdings(ticker, found)
    if found.rows and spreads:
        chosen = _holdings(ticker, found)
    return chosen


def _detail(
    ctx: EtfContext, ticker: str, funds: Mapping[str, etf.Cell], own: pd.DataFrame
) -> None:
    memo = cast(
        "dict[str, etf.Composition]",
        st.session_state.setdefault("etf_detail", {}),
    )
    found = memo.get(ticker)
    if found is None:
        with st.spinner("구성 및 NAV 수신 중…"):
            found = ctx.detail(ticker)
        if found is not None:
            memo[ticker] = found
    cells = {**funds, **ctx.funds().get(ticker, {})}
    _fund(ctx, ticker, cells, own)
    if found is None:
        components.empty("구성 데이터 없음")
        return
    chosen = _composition(ticker, found, cells)
    if chosen is not None and chosen.ticker:
        _company(ctx, chosen)


def _named(cell: etf.Cell, label: Callable[[str], str]) -> dict[str, float]:
    weights = cast("Mapping[str, float]", cell.value)
    return {label(key): value for key, value in weights.items()}


def _closed() -> None:
    state = st.session_state
    state.pop("etf_open", None)
    state.etf_chart_gen = state.get("etf_chart_gen", 0) + 1


def open_detail(
    ctx: EtfContext,
    ticker: str,
    funds: Mapping[str, etf.Cell],
    rows: pd.DataFrame,
) -> None:
    st.session_state.ai_last_etf = ticker
    name = funds.get("name")
    title = ticker if name is None else f"{ticker} · {name.value}"
    own = cast("pd.DataFrame", rows[rows["ticker"] == ticker])
    st.dialog(title, width="large", on_dismiss=_closed)(_detail)(
        ctx, ticker, funds, own
    )


def update(ctx: EtfContext, *, force: bool = False) -> None:
    if ctx.updater is None:
        return
    saved = load_etf(ctx.settings)
    products = etf_universe.effective(saved.added, saved.removed)
    ctx.updater.prune(products, [*ctx.held(), MARKET])
    _ = ctx.updater.start([*(p.ticker for p in products), MARKET], force=force)


def page(ctx: EtfContext) -> None:
    if st.session_state.pop("refresh_page", False):
        st.session_state.pop("etf_detail", None)
        update(ctx, force=True)
    _bar(ctx)
    _view(ctx)


@st.fragment
def _view(ctx: EtfContext) -> None:
    saved = load_etf(ctx.settings)
    products = etf_universe.effective(saved.added, saved.removed)
    choice = _choice_filters(ctx, products, saved)
    tickers = tuple(p.ticker for p in products)
    rows, returns, funds = _prepared(ctx.stamp(), tickers, ctx)
    leaders = etf.representatives(products, funds)
    scope = etf.scoped(products, choice.group, choice.category)
    table = etf.table_frame(funds, scope, returns, leaders)
    cells, heads = _map_cells(table, choice.period)
    state = st.session_state
    picked = None
    if not cells:
        components.empty("해당 범위 상품 없음")
    else:
        event = st.altair_chart(
            _map(cells, heads),
            on_select="rerun",
            selection_mode="pick",
            key=f"etf_chart_{state.get('etf_chart_gen', 0)}",
        )
        picked = _picked(event)
    state.ai_etf_view = documents.EtfView(
        choice.group, choice.category, choice.period, table
    )
    opened = cast("str | None", state.get("etf_open")) or picked
    if opened is not None:
        state.etf_open = opened
        open_detail(ctx, opened, funds.get(opened, {}), rows)
