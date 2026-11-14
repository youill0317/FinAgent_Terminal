from __future__ import annotations

import time
from datetime import date
from typing import TYPE_CHECKING, Final

import streamlit as st

from finagent import catalog, display, limits
from finagent.dashboard import SOURCE_NAMES
from finagent.models import AREA_LABEL
from finagent.sources import (
    BIS_RELEASE_CALENDAR_URL,
    ECB_MEETINGS_URL,
    ECB_STATS_CALENDAR_URL,
    OECD_EXPLORER_URL,
)
from finagent.ui import components

if TYPE_CHECKING:
    from finagent.models import Dashboard, IndicatorRow, SourceSummary
    from finagent.ui.context import UiContext

ALL_SOURCES: Final = "전체"
FREQ_LABEL: Final = {"D": "일", "W": "주", "M": "월", "Q": "분기"}
ROW_STATUS: Final = {"ok": "정상", "missing_key": "키 없음", "error": "수신 실패"}
STALE_STATUS: Final = "이전 값 유지"
SOURCE_STATE: Final = {
    "ok": "정상",
    "partial": "일부 실패",
    "missing_key": "키 없음",
    "error": "수신 실패",
}
SOURCE_LABEL: Final[dict[str, str]] = {str(sid): name for sid, name in SOURCE_NAMES}
FRED_ATTRIBUTION: Final = (
    "This product uses the FRED® API but is not endorsed or certified by the "
    "Federal Reserve Bank of St. Louis."
)
CALENDAR_HELP: Final = (
    "미국 주요 거시 경제 지표 발표 예정 일정\n\n"
    "주요국 세부 지표 일정은 하단 외부 링크 참조"
)
MEETING_HELP: Final = (
    "경제전망(SEP) 동반 연 4회 FOMC 정례회의 일정\n\n"
    "중간 회의 및 ECB 회의 일정은 하단 외부 링크 참조"
)
NEXT_FOMC_HELP: Final = "경제전망(SEP) 발표 예정 차기 FOMC 회의일 및 D-Day 잔여일수"
LINKS_HELP: Final = "주요국 중앙은행 및 국제기구 공식 통계 발표 일정 안내"
SOURCES_HELP: Final = (
    "출처별 최근 데이터 수신 현황(성공/요청 시리즈 수)\n\n"
    "하단 테이블: 전 지표별 출처 코드, 갱신 주기, 최신 관측 시점"
)
KEY_HELP: Final = (
    "`{env}` 환경변수 또는 `.env` 파일 내 API 키 설정 필요  \n"
    " [여기서]({url}) 무료 발급 가능"
)


def render(d: Dashboard, ctx: UiContext) -> None:
    _ = ctx
    left, right = st.columns(2)
    with left:
        _calendar(d)
    with right:
        _meetings(d)
    _links()
    _sources(d)


def _calendar(d: Dashboard) -> None:
    components.section(
        f"향후 {catalog.CALENDAR_DAYS}일 지표 발표 일정",
        help=CALENDAR_HELP,
        anchor="calendar",
    )
    with st.container(border=True, height="stretch"):
        if d.calendar:
            _ = st.dataframe(  # pyright: ignore[reportUnknownMemberType]
                {
                    "날짜": [display.weekday(entry.date) for entry in d.calendar],
                    "발표": [entry.name for entry in d.calendar],
                    "출처": [entry.source for entry in d.calendar],
                },
                hide_index=True,
                width="stretch",
                alt=f"향후 {catalog.CALENDAR_DAYS}일 지표 발표 일정",
            )
        elif d.calendar_message:
            components.empty(f"발표 일정 수신 실패 · {d.calendar_message}")
        else:
            components.empty(
                f"향후 {catalog.CALENDAR_DAYS}일 내 예정된 주요 지표 발표 일정 없음"
            )


def _meetings(d: Dashboard) -> None:
    components.section("중앙은행 회의", help=MEETING_HELP, anchor="meetings")
    with st.container(border=True, height="stretch"):
        today = date.fromisoformat(d.generated_at[:10])
        upcoming = sorted(
            day for day in d.fomc_dates if date.fromisoformat(day) >= today
        )
        if upcoming:
            first, *later = upcoming
            days = (date.fromisoformat(first) - today).days
            _ = st.metric(
                "차기 FOMC(SEP)",
                display.weekday(first),
                f"D-{days}" if days else "오늘",
                delta_color="blue",
                delta_arrow="off",
                delta_description=f"{first[:4]}년",
                help=NEXT_FOMC_HELP,
                border=True,
            )
            if later:
                with st.container(horizontal=True, gap="xsmall"):
                    for day in later:
                        _ = st.badge(
                            f"{day[2:4]}/{display.weekday(day)}",
                            icon=":material/event:",
                            color="gray",
                        )
        elif d.fomc_message:
            components.empty(f"FOMC 일정 수신 실패 · {d.fomc_message}")
        else:
            components.empty("FOMC 일정 데이터 없음")
        _ = st.link_button(
            "ECB 통화정책 회의 일정",
            ECB_MEETINGS_URL,
            icon=":material/open_in_new:",
            width="stretch",
        )


def _links() -> None:
    components.section("외부 발표 일정", help=LINKS_HELP, anchor="calendars")
    with st.container(border=True, height="stretch"):
        _ = st.link_button(
            "ECB 통계 발표 일정",
            ECB_STATS_CALENDAR_URL,
            icon=":material/open_in_new:",
            width="stretch",
        )
        _ = st.link_button(
            "BIS 데이터 공표 일정",
            BIS_RELEASE_CALENDAR_URL,
            icon=":material/open_in_new:",
            width="stretch",
        )


def failed_count(d: Dashboard) -> int:
    return sum(s.total - s.ok for s in d.sources if s.state != "missing_key")


def _ask_retry() -> None:
    st.session_state.retry_failed = True


def _sources(d: Dashboard) -> None:
    components.section("데이터 출처", help=SOURCES_HELP, anchor="sources")
    with st.container(border=True):
        if d.sources:
            with st.container(horizontal=True):
                for source in d.sources:
                    _source_card(source)
        else:
            components.empty("출처 상태 데이터 없음")
        failed = failed_count(d)
        if failed and not d.demo:
            _ = st.button(
                f"실패 {failed}건 재시도",
                icon=":material/replay:",
                key="retry_failed_button",
                on_click=_ask_retry,
            )
        _inventory(d)
        _ = st.caption(FRED_ATTRIBUTION)
        _ = st.link_button(
            "OECD 데이터 탐색기",
            OECD_EXPLORER_URL,
            icon=":material/open_in_new:",
        )


def _source_card(source: SourceSummary) -> None:
    _ = st.metric(
        source.name,
        f"{source.ok}/{source.total}",
        _source_delta(source),
        delta_color=display.SOURCE_COLOR[source.state],
        delta_arrow="off",
        help=_source_help(source),
        border=True,
        height="stretch",
    )


def _source_help(source: SourceSummary) -> str | None:
    key = display.KEY_HELP.get(source.id)
    if source.state == "missing_key" and key:
        env, url = key
        return KEY_HELP.format(env=env, url=url)
    return source.message


def _source_delta(source: SourceSummary) -> str:
    states = [s for s in limits.current().state() if s.source == source.id]
    return display.limit_lines(states, time.time()) or SOURCE_STATE[source.state]


@st.fragment
def _inventory(d: Dashboard) -> None:
    rows = _unique_rows(d)
    present = [sid for sid, _ in SOURCE_NAMES if any(row.source == sid for row in rows)]
    options: list[str] = [ALL_SOURCES, *present]
    picked = st.segmented_control(
        "출처",
        options,
        default=ALL_SOURCES,
        required=True,
        format_func=_source_name,
        key="schedule_source",
        label_visibility="collapsed",
        bind="query-params",
    )
    shown = [row for row in rows if picked in {ALL_SOURCES, row.source}]
    if not shown:
        components.empty("등록된 지표 없음")
        return
    _ = st.dataframe(  # pyright: ignore[reportUnknownMemberType]
        {
            "지표": [row.label for row in shown],
            "국가": [AREA_LABEL[row.area] for row in shown],
            "출처 코드": [row.code for row in shown],
            "주기": [FREQ_LABEL[row.freq] for row in shown],
            "최신일": [display.short_date(row) or "—" for row in shown],
            "상태": [_row_status(row) for row in shown],
            "©": [row.copyright or "" for row in shown],
        },
        hide_index=True,
        width="stretch",
        column_config={
            "지표": st.column_config.TextColumn(width="medium", pinned=True),
            "출처 코드": st.column_config.TextColumn(width="medium"),
        },
        alt="대시보드 전체 지표 출처 및 최신 관측일 목록",
    )
    _ = st.caption(f"총 {len(shown)}개 지표")


def _source_name(option: str) -> str:
    return SOURCE_LABEL.get(option, option)


def _unique_rows(d: Dashboard) -> list[IndicatorRow]:
    seen: set[str] = set()
    rows: list[IndicatorRow] = []
    for panel in d.panels:
        for row in panel.rows:
            if row.code not in seen:
                seen.add(row.code)
                rows.append(row)
    return rows


def _row_status(row: IndicatorRow) -> str:
    if row.status == "ok" and row.message:
        return STALE_STATUS
    return ROW_STATUS[row.status]
