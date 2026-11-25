# pyright: reportAny=false, reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false
"""개별 기업 개요, 핵심 재무비율, 연간 재무제표 및 SEC 공시 목록 렌더링."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import streamlit as st

from finagent.ui import components

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from finagent.etf import Cell
    from finagent.fundamentals import Company, FilingRow

DESCRIPTION_LIMIT: Final = 600
METRIC_COLUMNS: Final = 4
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
ANNUAL_ITEMS: Final = (
    ("sales", "매출액"),
    ("ni", "순이익"),
    ("ocf", "영업현금흐름"),
    ("ta", "총자산"),
    ("ca", "유동자산"),
    ("cl", "유동부채"),
    ("cash_sti", "현금·단기투자"),
    ("std", "단기차입금"),
    ("ltd", "장기차입금"),
    ("tl", "총부채"),
)


def fmt_value(value: object, kind: str) -> str:
    """지표 유형에 맞춰 수치·섹터·문자열을 화면 표시 형식으로 변환한다."""
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


def metric_text(name: str, cells: Mapping[str, Cell], kind: str) -> str:
    """셀 값과 통화 단위를 결합한 지표 카드 표시 문자열을 반환한다."""
    cell = cells.get(name)
    text = fmt_value(None if cell is None else cell.value, kind)
    currency = cells.get("currency")
    if name == "market_cap" and currency is not None and text != "—":
        return f"{text} {currency.value}"
    return text


def fact_line(cells: Mapping[str, Cell]) -> str:
    """섹터·산업·국가 요약 문자열을 조합한다."""
    facts = [
        fmt_value(cell.value, kind)
        for name, kind in STOCK_FACTS
        if (cell := cells.get(name)) is not None
    ]
    return " · ".join(facts)


def short_description(text: str) -> str:
    """긴 기업 설명을 제한 길이 내 문장 단위로 줄인다."""
    if len(text) <= DESCRIPTION_LIMIT:
        return text
    return text[:DESCRIPTION_LIMIT].rsplit(". ", 1)[0] + "."


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_help(help_: str | None, cell: Cell | None) -> str | None:
    source = (
        f"출처: {cell.source} ({cell.as_of})"
        if cell is not None and cell.source
        else ""
    )
    parts = [part for part in (help_, source) if part]
    return "\n\n".join(parts) if parts else None


def _render_annual(annual: Mapping[str, object]) -> None:
    fy_end = str(annual.get("fy_end") or "")
    rows: list[tuple[str, str, str, str]] = []
    for stem, label in ANNUAL_ITEMS:
        v0 = _finite(annual.get(f"{stem}_0"))
        v1 = _finite(annual.get(f"{stem}_1"))
        if v0 is None and v1 is None:
            continue
        change = (
            "—"
            if v0 is None or v1 is None or v1 == 0
            else f"{(v0 - v1) / abs(v1) * 100:+.1f}%"
        )
        rows.append((label, fmt_value(v0, "big"), fmt_value(v1, "big"), change))
    with st.container(border=True):
        st.markdown("**연간 재무 요약**")
        if not fy_end or not rows:
            components.empty("재무제표 데이터 없음")
            return
        notes = [
            part
            for part in (
                f"결산일 {fy_end}",
                str(annual.get("currency") or ""),
                str(annual.get("source") or ""),
            )
            if part
        ]
        if notes:
            st.caption(" · ".join(notes))
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            pd.DataFrame(
                {
                    "항목": [r[0] for r in rows],
                    "당기": [r[1] for r in rows],
                    "전기": [r[2] for r in rows],
                    "증감": [r[3] for r in rows],
                }
            ),
            hide_index=True,
            width="stretch",
        )


def _render_filings(filings: Sequence[FilingRow]) -> None:
    with st.container(border=True):
        st.markdown("**SEC 공시**")
        if not filings:
            components.empty("공시 없음")
            return
        frame = pd.DataFrame(
            {
                "유형": [row.form for row in filings],
                "제출일": [row.filed for row in filings],
                "항목": [row.items or "—" for row in filings],
                "원문": [row.url for row in filings],
            }
        )
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            frame,
            hide_index=True,
            width="stretch",
            column_config={
                "원문": st.column_config.LinkColumn("원문", display_text="열기"),
            },
        )


def render_profile(
    found: Company,
    annual: Mapping[str, object],
    filings: Sequence[FilingRow],
) -> None:
    """기업 개요, 핵심 재무비율 카드, 연간 재무제표 및 SEC 공시 목록을 렌더링한다."""
    cells = found.cells
    name_cell = cells.get("name")
    title = found.ticker if name_cell is None else str(name_cell.value)
    if title != found.ticker:
        st.markdown(f"**{title}**")
    if facts := fact_line(cells):
        st.caption(facts)
    if (desc := cells.get("description")) is not None:
        st.caption(short_description(str(desc.value)))
    for start in range(0, len(STOCK_METRICS), METRIC_COLUMNS):
        row_specs = STOCK_METRICS[start : start + METRIC_COLUMNS]
        for col, (name, label, kind, help_) in zip(
            st.columns(METRIC_COLUMNS, gap="small"), row_specs, strict=False
        ):
            col.metric(
                label,
                metric_text(name, cells, kind),
                help=_metric_help(help_, cells.get(name)),
                border=True,
                height="stretch",
            )
    left, right = st.columns(2)
    with left:
        _render_annual(annual)
    with right:
        _render_filings(filings)
