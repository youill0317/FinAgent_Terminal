"""주요국(미국·한국·유로존·일본·중국) 실물경제 지표 비교 및 국가별 화면."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import streamlit as st

from finagent.ui import components as ui

if TYPE_CHECKING:
    from finagent.models import Area, Dashboard
    from finagent.ui.context import UiContext

PAGE: Final = "economy"
COMPARE: Final = "비교"
CHOICES: Final[dict[str, Area | None]] = {
    COMPARE: None,
    "미국": "US",
    "한국": "KR",
    "유로존": "EA",
    "일본": "JP",
    "중국": "CN",
}


@st.fragment
def render(d: Dashboard, ctx: UiContext) -> None:
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        picked = st.segmented_control(
            "국가",
            list(CHOICES),
            default=COMPARE,
            required=True,
            key="economy_area",
            label_visibility="collapsed",
            bind="query-params",
        )
        months = ui.range_control(f"{PAGE}_range")
    area = CHOICES.get(picked)
    panels = ui.panels_of(d, PAGE)
    if not panels:
        ui.empty("경제 지표 데이터 없음")
        return
    for p in panels:
        with ui.card(p):
            if area is None:
                lead = ui.headline(p.rows)
                ui.standard(p, ctx, months, lead, lead=lead)
                continue
            rows = ui.rows_by_area(p).get(area, [])
            if rows:
                ui.standard(p, ctx, months, rows, key=f"{p.id}_{area}")
            else:
                ui.empty(f"{picked} 데이터 없음")
