"""주가지수, 섹터·자산군 ETF, 금리, 환율 및 원자재 금융시장 화면."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import streamlit as st

from finagent.ui import components as ui

if TYPE_CHECKING:
    from finagent.models import Dashboard
    from finagent.ui.context import UiContext

PAGE: Final = "finance"
REBASED: Final = frozenset({"fin_sectors", "fin_etfs"})


@st.fragment
def render(d: Dashboard, ctx: UiContext) -> None:
    panels = ui.panels_of(d, PAGE)
    if not panels:
        ui.empty("금융시장 데이터 없음")
        return
    months = ui.page_range(PAGE)
    for p in panels:
        with ui.card(p):
            ui.standard(p, ctx, months, rebase=p.id in REBASED)
