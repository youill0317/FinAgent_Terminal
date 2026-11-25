from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

import streamlit as st

from finagent import (
    catalog,
    display,
    investors,
    limits,
    markets,
    memory,
    portfolio,
    prices,
)
from finagent.cli import resolve_env
from finagent.dashboard import SOURCE_NAMES
from finagent.models import AREA_LABEL
from finagent.sources import (
    BIS_RELEASE_CALENDAR_URL,
    ECB_MEETINGS_URL,
    ECB_STATS_CALENDAR_URL,
    OECD_EXPLORER_URL,
)
from finagent.ui import components, shared

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent.models import Dashboard, IndicatorRow, SourceId, SourceSummary
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
    "출처별 최근 데이터 수신 현황(성공/요청 시리즈 수) 및 시세·공시·매핑 상태\n\n"
    "하단 테이블: 전 지표별 실제 수신 출처 코드, 갱신 주기, 최신 관측 시점"
)
KEY_HELP: Final = (
    "`{env}` 환경변수 또는 `.env` 파일 내 API 키 설정 필요  \n"
    " [여기서]({url}) 무료 발급 가능"
)


@dataclass(frozen=True, slots=True)
class ExtraCard:
    source: SourceId
    name: str
    value: str
    delta: str
    color: display.Color
    help: str | None = None


def _env() -> Mapping[str, str]:
    try:
        env_file = shared.app.env_file
    except AttributeError:
        return resolve_env(None, os.environ)
    return shared.env_values(env_file)


def _limit_delta(source: SourceId) -> str:
    states = [s for s in limits.current().state() if s.source == source]
    return display.limit_lines(states, time.time())


def price_sources() -> dict[str, str]:
    """시세 저장소에 적재된 티커별 실제 시세 출처 문자열을 반환한다."""
    return {
        ticker: "+".join(sources)
        for ticker, sources in prices.current().sources_by_ticker().items()
    }


def display_code(row: IndicatorRow, resolved: Mapping[str, str]) -> str:
    """실제 조회된 시세 출처를 반영해 지표 출처 코드를 표시한다."""
    if row.source != "ALPACA" or ":" not in row.code:
        return row.code
    _, ticker = row.code.split(":", 1)
    actual = resolved.get(ticker)
    return f"{actual}:{ticker}" if actual else row.code


def _key_help(source: str) -> str | None:
    key = display.KEY_HELP.get(source)
    if not key:
        return None
    env_name, url = key
    return KEY_HELP.format(env=env_name, url=url)


def extra_cards(
    *, demo: bool, env: Mapping[str, str] | None = None
) -> tuple[ExtraCard, ...]:
    """온디맨드·확장 데이터 소스 상태 카드 목록을 생성한다."""
    if demo:
        return (
            ExtraCard("ALPACA", "Alpaca", "데모", "데모", "gray"),
            ExtraCard("YAHOO", "Yahoo", "데모", "데모", "gray"),
            ExtraCard("SEC", "SEC EDGAR", "데모", "데모", "gray"),
            ExtraCard("OPENFIGI", "OpenFIGI", "데모", "데모", "gray"),
        )
    found_env = _env() if env is None else env
    store = memory.current()
    by_ticker = prices.current().sources_by_ticker()

    def price_count(provider: str) -> int:
        return sum(provider in providers for providers in by_ticker.values())

    keys = markets.keys(found_env)
    alpaca_count = price_count("Alpaca")
    alpaca_limit = _limit_delta("ALPACA")
    alpaca_card = ExtraCard(
        "ALPACA",
        "Alpaca",
        f"{alpaca_count}종목" if keys.alpaca else "미설정",
        alpaca_limit or ("준비됨" if keys.alpaca else "Yahoo 대체"),
        "orange" if alpaca_limit else ("green" if keys.alpaca else "gray"),
        None if keys.alpaca else _key_help("ALPACA"),
    )

    installed = not portfolio.missing_packages()
    yahoo_prices = price_count("Yahoo")
    yahoo_infos = len(store.yahoo_info.rows())
    yahoo_limit = _limit_delta("YAHOO")
    yahoo_card = ExtraCard(
        "YAHOO",
        "Yahoo",
        f"시세 {yahoo_prices} · 정보 {yahoo_infos}" if installed else "미설치",
        yahoo_limit or ("정상" if installed else "패키지 없음"),
        "red" if not installed else ("orange" if yahoo_limit else "green"),
        None if installed else f"`{portfolio.INSTALL_HINT}`",
    )

    agent = found_env.get(investors.AGENT_KEY, "").strip()
    valid_agent = bool(agent and agent.isascii())
    facts_count = len(store.company_facts.rows())
    subs_count = len(store.submissions.rows())
    funds_count = len(store.holdings.rows()) + len(store.expenses.rows())
    filers_count = len(store.filers.rows())
    sec_total = facts_count + subs_count + funds_count + filers_count
    sec_limit = _limit_delta("SEC")
    sec_counts = " · ".join(
        (
            f"기업 재무 {facts_count}",
            f"공시 {subs_count}",
            f"펀드 {funds_count}",
            f"13F {filers_count}",
        )
    )
    sec_needed = (
        f"`{investors.AGENT_KEY}` 환경변수 또는 `.env` 내 영문 이름·이메일 설정 필요"
    )
    sec_help = sec_counts if valid_agent else sec_needed
    sec_state = "형식 오류" if agent else "미설정"
    sec_card = ExtraCard(
        "SEC",
        "SEC EDGAR",
        f"캐시 {sec_total}건" if valid_agent else ("ASCII 오류" if agent else "미설정"),
        sec_limit or ("준비됨" if valid_agent else sec_state),
        "orange"
        if valid_agent and sec_limit
        else ("green" if valid_agent else ("red" if agent else "gray")),
        sec_help,
    )

    has_figi = bool(found_env.get(investors.FIGI_KEY, "").strip())
    figi_limit = _limit_delta("OPENFIGI")
    figi_help = (
        "분당 200회 확장 한도 적용"
        if has_figi
        else f"{_key_help('OPENFIGI') or ''}  \n미설정 시 분당 20회 기본 한도 적용"
    )
    figi_card = ExtraCard(
        "OPENFIGI",
        "OpenFIGI",
        f"{len(store.cusips.rows())}건",
        figi_limit or ("키 확장" if has_figi else "기본 한도"),
        "orange" if figi_limit else ("green" if has_figi else "gray"),
        figi_help,
    )
    return (alpaca_card, yahoo_card, sec_card, figi_card)


def _extra_sources(demo: bool) -> None:
    _ = st.caption("시세 · 공시 · 매핑")
    with st.container(horizontal=True):
        for card in extra_cards(demo=demo):
            _ = st.metric(
                card.name,
                card.value,
                card.delta,
                delta_color=card.color,
                delta_arrow="off",
                help=card.help,
                border=True,
                height="stretch",
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
                    _source_card(source, demo=d.demo)
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
        _extra_sources(d.demo)
        _inventory(d)
        _ = st.caption(FRED_ATTRIBUTION)
        _ = st.link_button(
            "OECD 데이터 탐색기",
            OECD_EXPLORER_URL,
            icon=":material/open_in_new:",
        )


def _source_card(source: SourceSummary, *, demo: bool = False) -> None:
    _ = st.metric(
        source.name,
        f"{source.ok}/{source.total}",
        _source_delta(source, demo=demo),
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


def _source_delta(source: SourceSummary, *, demo: bool = False) -> str:
    if limited := _limit_delta(source.id):
        return limited
    if (
        not demo
        and source.id == "ALPACA"
        and source.state == "ok"
        and not markets.keys(_env()).alpaca
    ):
        return "Yahoo 대체"
    return SOURCE_STATE[source.state]


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
    resolved = {} if d.demo else price_sources()
    _ = st.dataframe(  # pyright: ignore[reportUnknownMemberType]
        {
            "지표": [row.label for row in shown],
            "국가": [AREA_LABEL[row.area] for row in shown],
            "출처 코드": [display_code(row, resolved) for row in shown],
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
