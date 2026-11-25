"""5대 중앙은행 통화정책 기준금리, 대차대조표 자산 및 통화량 화면."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import streamlit as st

from finagent.ui import components as ui

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finagent.models import Dashboard, IndicatorRow, Panel
    from finagent.ui.context import UiContext

BANKS: Final = ("mp_fed", "mp_ecb", "mp_boj", "mp_pboc", "mp_bok")
POLICY_IDS: Final = (
    "us_policy_rate",
    "ea_policy_rate",
    "jp_policy_rate",
    "cn_policy_rate",
    "kr_policy_rate",
)
ASSET_IDS: Final = frozenset(
    {"us_fed_assets_mp", "ea_ecb_assets", "jp_boj_assets", "kr_bok_base"}
)
PAGE: Final = "monetary"
BANKS_HELP: Final = (
    "[정의] 5대 중앙은행 정책금리 및 Fed·ECB·BOJ 총자산·한국은행 본원통화 현황"
    "\n\n[해석] 금리 인상·자산 축소 시 긴축, 금리 인하·자산 확대 시 완화"
    "\n\n[산출] 금리 카드 변화값은 최근 조정폭·변경일 및 최근 12개월 변경 횟수"
)
POLICY_HELP: Final = (
    "[정의] 주요국 중앙은행 기준금리 추이"
    "\n\n[해석] 회의 결정 시에만 변경되어 계단 모양으로 표시, 인상은 긴축·인하는 완화"
    "\n\n[산출] 통화정책회의 의결 기준금리 계단식(Step) 시계열"
)
ASSETS_HELP: Final = (
    "[정의] Fed($T)·ECB(€T)·BOJ(¥T) 총자산 및 한국은행 본원통화(조원) 추이"
    "\n\n[해석] 통화 단위 상이에 따라 선택 기간 첫날을 100으로 맞춰 상대 증감 추세"
    " 비교"
    "\n\n[산출] 각 중앙은행 대차대조표 자산 및 본원통화 지수화 시계열"
)


@st.fragment
def render(d: Dashboard, ctx: UiContext) -> None:
    months = ui.page_range(PAGE)
    central_banks(d, ctx, months)
    rates(ui.panel(d, "mp_rates"), ctx, months)


def pick(rows: Sequence[IndicatorRow], ids: Sequence[str]) -> list[IndicatorRow]:
    by_id = {row.id: row for row in rows}
    return [by_id[i] for i in ids if i in by_id]


def central_banks(d: Dashboard, ctx: UiContext, months: int) -> None:
    panels = [ui.panel(d, panel_id) for panel_id in BANKS]
    found = [p for p in panels if p is not None]
    rows = [row for p in found for row in p.rows]
    ui.section("통화 정책", help=BANKS_HELP, anchor=BANKS[0])
    with st.container(border=True):
        if not rows:
            ui.empty()
            return
        policy = pick(rows, POLICY_IDS)
        assets = [row for row in rows if row.id in ASSET_IDS]
        with st.container(key="mp_banks_metrics", gap="small"):
            if d.policy_changes:
                ui.policy_cards(d.policy_changes, policy)
            for group in (assets,) if d.policy_changes else (policy, assets):
                if group:
                    cols = st.columns(len(group), gap="small")
                    for col, row in zip(cols, group, strict=True):
                        with col:
                            ui.metric(row)
        left, right = st.columns(2, gap="medium")
        with left:
            _ = st.caption(ui.chart_title(policy, "정책금리"), help=POLICY_HELP)
            ui.rows_chart(
                policy,
                ctx,
                months,
                alt="주요국 정책금리 비교",
                step=True,
                height=ui.SMALL_CHART_HEIGHT,
            )
        with right:
            _ = st.caption(
                "중앙은행 자산·유동성 · 선택 기간 첫날 = 100", help=ASSETS_HELP
            )
            ui.rows_chart(
                assets,
                ctx,
                months,
                alt="Fed·ECB·BOJ 자산 및 한국은행 본원통화 비교",
                height=ui.SMALL_CHART_HEIGHT,
                rebase=True,
            )
        for p in found:
            ui.problems(p)
        ui.table_expander(rows, ctx, key="mp_banks")


def rates(p: Panel | None, ctx: UiContext, months: int) -> None:
    with ui.card(p):
        if p is not None:
            ui.standard(p, ctx, months)
