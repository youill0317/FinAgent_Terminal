"""S&P 500 and S&P 400 screening on yfinance data, kept under ``data/screen/``.

This module is an exception to the dashboard's data rules
(``tests/test_guardrails.py``): it imports ``yfinance`` (unofficial Yahoo
Finance data), reads the constituents lists from Wikipedia, asks SEC EDGAR
for 10-K operating margins when the user set ``SEC_USER_AGENT``, and writes
files, all under ``data/screen/`` in the folder the app runs from: S&P 500's
at its top, S&P 400's in ``sp400/``. yfinance's own caches go there too.
Nothing here runs until the header's 갱신 button starts an ``Updater``; the page
only reads the stored files.

Stored parts per universe and how often they are fetched:

- ``universe-YYYY-MM-DD.csv``: constituents with GICS sector and sub-industry,
  fetched when the newest one is ``UNIVERSE_MAX_AGE_DAYS`` old.
- ``prices.parquet``: daily OHLCV with Adj Close, one row per ticker and day.
  Each update fetches only the days since the last one (plus an overlap);
  a ticker whose overlap no longer matches (split or dividend adjustment) gets
  its full history again.
- ``fundamentals-YYYY-Qn.csv``: once per calendar quarter, and again in full
  when the stored file lacks a column this version writes.
- ``estimates-YYYY-MM-DD.csv``: once per day.
- ``ciks-YYYY-MM-DD.csv``: SEC's ticker → CIK list, like the universe.
- ``margins-YYYY-Qn.csv``: each non-financial stock's 10-K full-year revenue
  and operating income from SEC companyfacts, one row per fiscal year, once
  per calendar quarter. The companyfacts JSON itself is not kept.

Only the newest dated file of each kind is read; an update that finishes
deletes the older ones (``prune``).

The screening itself (``evaluate``) is plain pandas over those files.
"""

# pandas, numpy and yfinance ship no complete type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportUnknownParameterType=false
# Page text is split across lines.
# pyright: reportImplicitStringConcatenation=false

from __future__ import annotations

import importlib.util
import io
import json
import math
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as clock
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, final
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

SNAPSHOT_DIR: Final = Path("data") / "screen"
YFINANCE_CACHE: Final = ".yfinance"
PRICES_FILE: Final = "prices.parquet"
UNIVERSE_GLOB: Final = "universe-????-??-??.csv"
FUNDAMENTALS_GLOB: Final = "fundamentals-????-Q?.csv"
ESTIMATES_GLOB: Final = "estimates-????-??-??.csv"
CIKS_GLOB: Final = "ciks-????-??-??.csv"
MARGINS_GLOB: Final = "margins-????-Q?.csv"
# Dated parts: only the newest of each is read, so a finished update deletes
# the rest.
DATED_GLOBS: Final = (
    UNIVERSE_GLOB,
    FUNDAMENTALS_GLOB,
    ESTIMATES_GLOB,
    CIKS_GLOB,
    MARGINS_GLOB,
)
SEC_TICKERS_URL: Final = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL: Final = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
USER_AGENT: Final = "finagent/0.2 (local screener)"
REQUEST_TIMEOUT_SECONDS: Final = 20.0
PACKAGES: Final = ("yfinance", "lxml")
INSTALL_HINT: Final = "uv sync --extra screen"


@dataclass(frozen=True, slots=True)
class Universe:
    name: str
    url: str  # the Wikipedia list with GICS sector and sub-industry
    folder: str  # under ``SNAPSHOT_DIR``; S&P 500's files stay at its top


UNIVERSES: Final = {
    "sp500": Universe(
        "S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ""
    ),
    "sp400": Universe(
        "S&P 400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "sp400"
    ),
}

# ---------------------------------------------------------------- refresh
UNIVERSE_MAX_AGE_DAYS: Final = 30
HISTORY_DAYS: Final = 1130  # first fetch and kept window: 3 years and slack
# A store starting this many days after the wanted first day is fetched again
# in full: a longer window than the one it was fetched with.
BACKFILL_SLACK_DAYS: Final = 30
OVERLAP_DAYS: Final = 7  # refetched on each update to catch adjustments
ADJUST_TOLERANCE: Final = 0.001  # overlap Adj Close drift that means a refetch
MARKET_ZONE: Final = ZoneInfo("America/New_York")
SESSION_SETTLED: Final = clock(17, 0)  # a day's bar is final after this NY time
CHUNK: Final = 100  # tickers per price download
PAUSE_SECONDS: Final = 0.2  # between per-ticker calls
RETRY_WAITS: Final = (10.0, 30.0, 60.0)  # after Yahoo answers "too many requests"
# Seconds after each SEC request, one at a time across updates: with the 13F
# page's own pace, under SEC's 10 requests a second.
SEC_PAUSE: Final = 0.3
SEC_TIMEOUT_SECONDS: Final = 60.0  # companyfacts runs to megabytes
NOT_FOUND: Final = 404
REFUSED: Final = frozenset({403, 429})  # SEC's answers to a missing or fast agent

# ---------------------------------------------------------------- rules
# Every rule here is its source's own (docs/screening.md); the rest of the
# screen only computes the source's measures.
# Jegadeesh and Titman (2001): "We also exclude all stocks priced below $5 at
# the beginning of the holding period."
MIN_PRICE: Final = 5.0
# FTSE Global Factor Index Series Ground Rules v8.8 (July 2026), rule 5.2.1:
# Z-scores beyond three are truncated to three and renormalised, repeatedly.
Z_LIMIT: Final = 3.0
# Rule 5.4.2: leverage is not taken relative to its industry's median when
# the industry has fewer than three securities.
MEDIAN_MIN: Final = 3
TTM_QUARTERS: Final = 4  # quarters in the TTM operating margin
SEC_YEARS: Final = 5  # 10-K fiscal years in SEC's operating margin median
# When the screening definitions last changed, kept in each snapshot.
DEFINITIONS: Final = "2026-10-27"
YEAR_DAYS: Final = (350, 380)  # a 52- or 53-week fiscal year, in days
QUARTER_DAYS: Final = (80, 120)  # a fiscal quarter of 12 to 17 weeks
# SEC's TTM may end this many days before Yahoo's newest quarter and still
# count as the same quarter: 52/53-week calendars against month ends (Yahoo
# dates AutoZone's quarter to 2026-05-09 as 05-31), well short of a quarter.
STALE_SLACK_DAYS: Final = 31
# A 10-Q's year to date: one quarter, a half of 24 to 28 weeks, or three
# quarters of 36 to 41 weeks. The last also bounds a TTM's three gaps.
NINE_MONTH_DAYS: Final = (245, 290)
YTD_DAYS: Final = (QUARTER_DAYS, (160, 200), NINE_MONTH_DAYS)
# A quarter's revenue above this share of its year is a mistag (a 10-K's year
# on a quarter); H&R Block's tax season is a real 0.61.
QUARTER_SHARE: Final = 0.75
# us-gaap revenue concepts, in the order a tie between them is broken.
REVENUE_CONCEPTS: Final = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
OPERATING_INCOME: Final = "OperatingIncomeLoss"
FINANCIALS: Final = "Financials"
REAL_ESTATE: Final = "Real Estate"
# FTSE rule 5.4.3: financials and real estate (ICB industries 30 and 35) use
# ROA as the sole measure of quality; here the GICS sectors of those names.
ROA_ONLY: Final = frozenset({FINANCIALS, REAL_ESTATE})
SMA_DAYS: Final = (20, 60, 200)
YEAR_BARS: Final = 252
MONTH_BARS: Final = 21
QUARTER_BARS: Final = 63
TECH_COLUMNS: Final = (
    "bars",
    "price",  # last close as traded, for the price floor and the table
    *(f"sma{days}" for days in SMA_DAYS),
    "close",  # last adjusted close, which the indicators use
    "near_high",  # close / the highest close of 252 days (George and Hwang)
    "ma_dist",  # close / 200-day SMA (Brock, Lakonishok and LeBaron)
    "mom12_1",  # 12 months back to 1 month back (Jegadeesh and Titman)
    "ret_1m",
    "ret_3m",
)

# Status words for a measure without a value of its own.
UNKNOWN: Final = "미확인"  # no data
NOT_APPLICABLE: Final = "해당 없음"  # a measure FTSE does not use for the stock
UNUSED: Final = "미적용"  # a measure turned off in the 기준 panel

PRICE_COLUMNS: Final = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)
YAHOO_PRICES: Final = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
TEXT_COLUMNS: Final = ("ticker", "name", "sector", "industry")
# Fundamentals column → ``Ticker.info`` field.
INFO_FIELDS: Final = {"mcap": "marketCap"}
# FTSE's annual items (rules 5.4 and 5.6) under Yahoo's yearly statement
# rows: column stem → (statement, row). Each is stored for the newest fiscal
# year ("_0") and the one before ("_1").
ANNUAL_ROWS: Final = {
    "ni": ("income", "NetIncome"),
    "sales": ("income", "TotalRevenue"),
    "ocf": ("cash", "OperatingCashFlow"),
    "ta": ("balance", "TotalAssets"),
    "ca": ("balance", "CurrentAssets"),
    "cl": ("balance", "CurrentLiabilities"),
    "cash_sti": ("balance", "CashCashEquivalentsAndShortTermInvestments"),
    "sti": ("balance", "OtherShortTermInvestments"),
    "std": ("balance", "CurrentDebt"),
    "ltd": ("balance", "LongTermDebt"),
    "tl": ("balance", "TotalLiabilitiesNetMinorityInterest"),
    "pref": ("balance", "PreferredStock"),
    "inv_adv": ("balance", "InvestmentsAndAdvances"),
}
ANNUAL_COLUMNS: Final = tuple(
    f"{stem}_{year}" for year in (0, 1) for stem in ANNUAL_ROWS
)
FUNDAMENTAL_COLUMNS: Final = (
    "ticker",
    *INFO_FIELDS,
    "margin_avg",
    "margin_years",
    "op_margin",
    "op_margin_quarters",
    "ttm_end",  # the end of the newest quarter in Yahoo's TTM
    "fy_end",  # the newest fiscal year end of the "_0" columns
    *ANNUAL_COLUMNS,
    "last_fy_end",  # the newest fiscal year reported
    "next_fy_end",  # the end of the fiscal year after it
    "currency",
    "error",
)
MARGIN_COLUMNS: Final = (
    "ticker",
    "period",  # "FY" for a fiscal year, "Q" for one of the TTM quarters
    "end",  # period end
    "revenue",
    "op_income",
    "concept",  # the revenue concept
    "filed",
    "error",
)
# FY1 is the fiscal year in progress (or ended and not yet reported), FY2 the
# one after: Yahoo's "0y" and "+1y".
ESTIMATE_PERIODS: Final = {"fy1": "0y", "fy2": "+1y"}
TREND_COLUMNS: Final = {"now": "current", "30d": "30daysAgo", "90d": "90daysAgo"}
REVISION_COLUMNS: Final = {"up30": "upLast30days", "down30": "downLast30days"}
ANALYST_COLUMNS: Final = {"n": "numberOfAnalysts"}
ESTIMATE_COLUMNS: Final = (
    "ticker",
    *(
        f"{period}_{key}"
        for period in ESTIMATE_PERIODS
        for key in (*TREND_COLUMNS, *REVISION_COLUMNS, *ANALYST_COLUMNS)
    ),
    "est_currency",
    "error",
)

# FTSE rule 5.4 퀄리티: three profitability measures and a leverage measure,
# column → name, higher is better.
PROFITABILITY: Final = {
    "q_roa": "ROA",
    "q_turnover": "자산회전율 변화",
    "q_accruals": "발생액",
}
QUALITY: Final = {**PROFITABILITY, "q_leverage": "레버리지"}
# FTSE rule 5.6 밸류에이션: three yields on the full market capitalisation, higher
# is cheaper.
VALUE: Final = {
    "v_cash": "현금흐름 수익률",
    "v_earnings": "이익 수익률",
    "v_sales": "매출 ÷ 시가총액",
}
FORMATS: Final = {
    "pct": "{:.1%}",
    "signed": "{:+.1%}",
    "signed2": "{:+.2%}",
    "usd": "${:,.2f}",
    "big": "${:,.1f}{}",  # billions or millions, as ``fmt`` picks
    "x": "{:,.1f}x",
    "int": "{:,.0f}",
    "num": "{:,.2f}",
    "z": "{:+.2f}",
}
# The results table: column, label and ``fmt`` kind ("" for text).
TABLE: Final = (
    ("ticker", "티커", ""),
    ("name", "종목", ""),
    ("sector", "섹터", ""),
    ("industry", "산업", ""),
    ("price", "주가", "usd"),
    ("mcap", "시가총액", "big"),
    ("quality", "퀄리티", "z"),
    ("value", "밸류에이션", "z"),
    ("counted", "산출 항목 수", ""),
    ("roa", "ROA", "pct"),
    ("leverage", "영업현금흐름 ÷ 총부채", "num"),
    ("earnings_yield", "이익 수익률", "pct"),
    ("cash_yield", "현금흐름 수익률", "pct"),
    ("sales_yield", "매출 ÷ 시가총액", "num"),
    ("op_margin", "영업이익률 (TTM)", "pct"),
    ("ret_1m", "1개월", "signed"),
    ("ret_3m", "3개월", "signed"),
    ("mom12_1", "12-1 모멘텀", "signed"),
)
ESTIMATE_HELP: Final = (
    "**정의** 애널리스트 컨센서스 EPS · FY1 진행 중인(또는 끝나고 결산 전인)"
    " 회계연도 · FY2 그다음 회계연도 · 30일·90일 전 값 · 최근 30일 상향·하향"
    " 애널리스트 수 · 추정 애널리스트 수\n\n"
    "**해석** 수정 강도 = (현재 − 90일 전) ÷ 주가, 같은 회계연도끼리 비교해 FY1·FY2"
    " 평균(결산 대기면 FY2만) · 순수정 비율 = (30일 상향 − 하향) ÷ 애널리스트 수 ·"
    " 퀄리티·밸류에이션 계산에는 쓰지 않음"
)
# The detail dialog's price indicators: column → (name, definition).
TECH_LINES: Final = {
    "mom12_1": (
        "12-1 모멘텀",
        f"수정 종가 기준 {YEAR_BARS}거래일 전부터 {MONTH_BARS}거래일 전까지의 수익률"
        " (Jegadeesh·Titman 1993)",
    ),
    "near_high": (
        "52주 고가 근접",
        f"종가 ÷ 최근 {YEAR_BARS}거래일 최고 종가 (George·Hwang 2004)",
    ),
    "ma_dist": (
        "200일선 대비",
        "종가 ÷ 200일 단순이동평균 (Brock·Lakonishok·LeBaron 1992)",
    ),
    "ret_1m": ("1개월 수익률", f"{MONTH_BARS}거래일 수익률"),
    "ret_3m": ("3개월 수익률", f"{QUARTER_BARS}거래일 수익률"),
}

# ---------------------------------------------------------------- criteria


@dataclass(frozen=True, slots=True)
class Criteria:
    """The switches in force: ``off`` holds the ``RULES`` ids turned off."""

    off: tuple[str, ...] = ()

    def on(self, key: str) -> bool:
        return key not in self.off

    def kept[V](self, items: Mapping[str, V]) -> dict[str, V]:
        """``items`` without the ids turned off."""
        return {key: value for key, value in items.items() if key not in self.off}


DEFAULT: Final = Criteria()


@dataclass(frozen=True, slots=True)
class Rule:
    """A switch of the 기준 panel: ``key`` is its id in ``Criteria.off``."""

    key: str
    section: str
    name: str
    help: str


FTSE: Final = "FTSE Global Factor Index Series Ground Rules v8.8(2026-07)"
AVERAGE_ASSETS: Final = "평균 총자산 = (올해 총자산 + 전년 총자산) ÷ 2"
QUALITY_HELP: Final = {
    "q_roa": f"순이익 ÷ 평균 총자산 · {AVERAGE_ASSETS} · 연간\n\n**근거** {FTSE} 5.4.1",
    "q_turnover": "올해 매출 ÷ 올해 총자산 − 전년 매출 ÷ 전년 총자산 · 연간 ·"
    f" 금융·부동산 해당 없음\n\n**근거** {FTSE} 5.4.1",
    "q_accruals": "−(Δ운전자본 + Δ비유동 순영업자산 + Δ순금융자산) ÷ 평균 총자산 ·"
    " 운전자본 = (유동자산 − 현금·단기투자) − (유동부채 − 단기부채) · 비유동 순영업자산"
    " = (총자산 − 유동자산 − 투자자산) − (총부채 − 유동부채 − 장기부채) · 순금융자산 ="
    " (단기투자 + 투자자산) − (장기부채 + 단기부채 + 우선주) · 연간 · 금융·부동산 해당"
    f" 없음\n\n**근거** {FTSE} 5.4.1",
    "q_leverage": "영업현금흐름 ÷ 총부채(단기부채 + 장기부채), 부채가 없거나"
    " 영업현금흐름이 총부채보다 크면 1 · GICS 섹터 중앙값을 뺀 값(섹터 종목이"
    f" {MEDIAN_MIN}개 미만이면 빼지 않음) · 연간 · 금융·부동산 해당 없음\n\n"
    f"**근거** {FTSE} 5.4.2",
}
VALUE_HELP: Final = {
    "v_cash": "최근 연간 영업현금흐름 ÷ 시가총액",
    "v_earnings": "최근 연간 순이익 ÷ 시가총액",
    "v_sales": "최근 연간 매출 ÷ 시가총액",
}
RULES: Final = (
    Rule(
        "min_price",
        "관문",
        "주가",
        f"**정의** 최근 종가(수정 전) < ${MIN_PRICE:.0f}이면 맵에서 제외\n\n"
        "**근거** Jegadeesh·Titman(2001)의 표본 조건(보유 시작 시점 주가 $5 미만"
        " 제외)",
    ),
    *(
        Rule(key, "퀄리티", name, f"**정의** {QUALITY_HELP[key]}")
        for key, name in QUALITY.items()
    ),
    *(
        Rule(
            key,
            "밸류에이션",
            name,
            f"**정의** {VALUE_HELP[key]}\n\n**근거** {FTSE} 5.6",
        )
        for key, name in VALUE.items()
    ),
)


def criteria(flipped: Iterable[str]) -> Criteria:
    """The defaults with the ``flipped`` switches turned the other way.

    An unknown id is dropped.
    """
    ids = set(DEFAULT.off) ^ set(flipped)
    return Criteria(off=tuple(rule.key for rule in RULES if rule.key in ids))


def changed(c: Criteria) -> list[str]:
    """``RULES`` ids switched from the defaults: what the settings file keeps."""
    return [rule.key for rule in RULES if c.on(rule.key) != DEFAULT.on(rule.key)]


# ---------------------------------------------------------------- tooltips
# The stock dialog's cards and headings, and the table's columns, as the
# criteria in force set them.


def basis_note() -> str:
    return (
        f"{FTSE} 5.2 · 비교 범위: 주가 관문을 통과한 유니버스 전체 · 지표마다 z 점수"
        f" = (값 − 평균) ÷ 표준편차, ±{Z_LIMIT:g}을 넘으면 {Z_LIMIT:g}로 자르고 다시"
        " 표준화 · 하위 지표는 값이 있는 것의 z 평균을 다시 표준화 · 값이 없으면 0"
    )


def _listed(names: Iterable[str]) -> str:
    return " · ".join(names) or "없음"


def quality_help(c: Criteria) -> str:
    profit = _listed(c.kept(PROFITABILITY).values())
    return (
        f"**정의** 수익성({profit})과 레버리지의 z 평균을 다시 표준화 · 금융·부동산은"
        " ROA만\n\n**해석** 높을수록 우수 · 0이 유니버스 평균\n\n"
        f"**기준** {basis_note()}"
    )


def value_help(c: Criteria) -> str:
    return (
        f"**정의** {_listed(c.kept(VALUE).values())}의 z 평균을 다시 표준화\n\n"
        "**해석** 높을수록 쌈 · 0이 유니버스 평균\n\n"
        f"**기준** {basis_note()}"
    )


def column_help(c: Criteria) -> dict[str, str]:
    """The results table's column tooltips."""
    return {
        "quality": quality_help(c),
        "value": value_help(c),
        "counted": "**정의** 값이 있는 퀄리티 / 밸류에이션 지표 수",
        "roa": f"**정의** {QUALITY_HELP['q_roa']}",
        "leverage": "**정의** 영업현금흐름 ÷ 총부채, 부채가 없거나 영업현금흐름이"
        " 총부채보다 크면 1 · 연간",
        "earnings_yield": f"**정의** {VALUE_HELP['v_earnings']}",
        "cash_yield": f"**정의** {VALUE_HELP['v_cash']}",
        "sales_yield": f"**정의** {VALUE_HELP['v_sales']}",
        "mom12_1": f"**정의** {TECH_LINES['mom12_1'][1]}",
    }


@dataclass(frozen=True, slots=True)
class Progress:
    running: bool = False
    stage: str = ""
    done: int = 0
    total: int = 0
    message: str = ""  # how the last update ended
    failed: bool = False


@dataclass(frozen=True, slots=True)
class Parts:
    """Which data the stored files hold: the dates shown above the table."""

    universe: date | None = None
    prices: date | None = None  # last daily close
    fundamentals: str | None = None  # "2026-Q4"
    estimates: date | None = None
    margins: str | None = None  # SEC 10-K margins, "2026-Q4"


@dataclass(frozen=True, slots=True)
class Inputs:
    universe: pd.DataFrame
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    estimates: pd.DataFrame
    margins: pd.DataFrame


class RateLimitedError(Exception):
    """Yahoo refused with "too many requests"."""


class SecUnavailableError(Exception):
    """SEC could not be reached or refused (403, 429); the SEC stage stops."""


class EmptyAnswerError(Exception):
    """Yahoo answered without data, which is also how yfinance hides some limits.

    The row is recorded as failed, so the next update asks again.
    """


class Source(Protocol):
    def constituents(self) -> pd.DataFrame: ...
    def prices(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Daily ``PRICE_COLUMNS`` rows from ``start`` up to, not including, ``end``."""
        ...

    def fundamentals(self, ticker: str) -> Mapping[str, object]:
        """One ``FUNDAMENTAL_COLUMNS`` row without the ticker and error."""
        ...

    def estimates(self, ticker: str) -> Mapping[str, object]:
        """One ``ESTIMATE_COLUMNS`` row without the ticker and error."""
        ...


class Filings(Protocol):
    def tickers(self) -> pd.DataFrame:
        """SEC's ticker (Yahoo form) and ten-digit CIK list."""
        ...

    def facts(self, cik: str) -> Mapping[str, object] | None:
        """One companyfacts JSON, or ``None`` when SEC has none."""
        ...


type Report = Callable[[str, int, int], None]

# ---------------------------------------------------------------- parsing


def yahoo_symbol(symbol: str) -> str:
    """Wikipedia's ``BRK.B`` is Yahoo's ``BRK-B``."""
    return symbol.strip().upper().replace(".", "-")


def missing_packages() -> list[str]:
    return [name for name in PACKAGES if importlib.util.find_spec(name) is None]


def parse_constituents(html: str) -> pd.DataFrame:
    """Ticker, name, GICS sector and sub-industry from the Wikipedia page."""
    raw = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    table = cast(
        "pd.DataFrame",
        raw.rename(
            columns={
                "Symbol": "ticker",
                "Security": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "industry",
            }
        )[list(TEXT_COLUMNS)],
    )
    table["ticker"] = cast("pd.Series", table["ticker"]).map(yahoo_symbol)
    return table.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def empty_prices() -> pd.DataFrame:
    frame = pd.DataFrame({column: [] for column in PRICE_COLUMNS})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].astype("str")
    return frame


def long_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """``yf.download``'s (field, ticker) columns as one row per ticker and day."""
    if frame.empty:
        return empty_prices()
    stacked = cast(
        "pd.DataFrame",
        frame.stack(level="Ticker", future_stack=True),  # noqa: PD013
    )
    stacked = stacked.rename(columns=YAHOO_PRICES).reset_index()
    stacked = stacked.rename(columns={"Date": "date", "Ticker": "ticker"})
    stacked["date"] = pd.to_datetime(stacked["date"]).dt.normalize()
    stacked = stacked.dropna(subset=["close", "adj_close"])
    return cast("pd.DataFrame", stacked[list(PRICE_COLUMNS)]).reset_index(drop=True)


def _number(value: object) -> float:
    """A finite number as a float, else NaN (Yahoo sends "Infinity" and None)."""
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        return math.nan
    number = float(value)
    return number if math.isfinite(number) else math.nan


def _row_value(table: pd.DataFrame, row: str, column: int = 0) -> float:
    if table.empty or row not in table.index or column >= len(table.columns):
        return math.nan
    return _number(table.loc[row].iloc[column])


@dataclass(frozen=True, slots=True)
class Statements:
    """One stock's Yahoo statements; columns are period ends.

    ``income``, ``balance`` and ``cash`` are yearly, ``quarterly`` the
    quarterly income statement. A statement Yahoo did not send is an empty
    frame.
    """

    income: pd.DataFrame
    quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash: pd.DataFrame = field(default_factory=pd.DataFrame)


def _epoch_day(value: object) -> str:
    """Yahoo's epoch seconds as an ISO date, "" when missing."""
    seconds = _number(value)
    if math.isnan(seconds):
        return ""
    return datetime.fromtimestamp(seconds, tz=ZoneInfo("UTC")).date().isoformat()


def _at(table: pd.DataFrame, row: str, end: object) -> float:
    """``row`` of ``table`` in the period ending ``end``, NaN when absent."""
    if row not in table.index or end not in table.columns:
        return math.nan
    return _number(table.loc[row, end])


def fundamentals_row(info: Mapping[str, object], s: Statements) -> dict[str, object]:
    """``info`` numbers, statement values and ratios for one stock.

    ``margin_avg`` (an older name, kept for the stored files) is the median
    operating margin of the fiscal years with both values and revenue above
    zero, and ``margin_years`` how many there were. ``op_margin`` is the TTM
    operating margin from the quarterly income statement: OperatingIncome over
    TotalRevenue summed across the latest four quarters, NaN unless all four
    have both values and the revenue is above zero (``info["operatingMargins"]``
    is only the latest quarter). The ``ANNUAL_ROWS`` columns are the yearly
    statements' values at the yearly balance sheet's newest two year ends.
    """
    row: dict[str, object] = {
        column: _number(info.get(field)) for column, field in INFO_FIELDS.items()
    }
    income = s.income
    years = [
        (
            _row_value(income, "OperatingIncome", year),
            _row_value(income, "TotalRevenue", year),
        )
        for year in range(len(income.columns))
    ]
    margins = [op / sales for op, sales in years if sales > 0 and not math.isnan(op)]
    row["margin_avg"] = statistics.median(margins) if margins else math.nan
    row["margin_years"] = len(margins)
    row["op_margin"], row["op_margin_quarters"], row["ttm_end"] = ttm_margin(
        s.quarterly
    )
    tables = {"income": s.income, "balance": s.balance, "cash": s.cash}
    ends = sorted(s.balance.columns, reverse=True)[:2]
    row["fy_end"] = str(cast("pd.Timestamp", ends[0]).date()) if ends else ""
    for year in (0, 1):
        end = ends[year] if year < len(ends) else None
        for stem, (statement, name) in ANNUAL_ROWS.items():
            row[f"{stem}_{year}"] = _at(tables[statement], name, end)
    row["last_fy_end"] = _epoch_day(info.get("lastFiscalYearEnd"))
    row["next_fy_end"] = _epoch_day(info.get("nextFiscalYearEnd"))
    currency = info.get("financialCurrency")
    row["currency"] = currency if isinstance(currency, str) else ""
    return row


def ttm_margin(quarterly: pd.DataFrame | None) -> tuple[float, int, str]:
    """TTM operating margin, the quarters it sums and the newest one's end.

    NaN, 0 and "" when the latest four quarters are not all there.
    """
    if quarterly is None or len(quarterly.columns) < TTM_QUARTERS:
        return math.nan, 0, ""
    latest = quarterly.sort_index(axis=1, ascending=False).iloc[:, :TTM_QUARTERS]
    income = [_row_value(latest, "OperatingIncome", i) for i in range(TTM_QUARTERS)]
    sales = [_row_value(latest, "TotalRevenue", i) for i in range(TTM_QUARTERS)]
    if any(math.isnan(v) for v in (*income, *sales)) or sum(sales) <= 0:
        return math.nan, 0, ""
    end = str(cast("pd.Timestamp", latest.columns[0]).date())
    return sum(income) / sum(sales), TTM_QUARTERS, end


def estimates_row(
    trend: pd.DataFrame,
    revisions: pd.DataFrame,
    estimate: pd.DataFrame | None = None,
) -> dict[str, object]:
    """FY1/FY2 EPS consensus now, 30 and 90 days ago, 30-day revisions and analysts."""
    analysts = estimate if estimate is not None else pd.DataFrame()
    row: dict[str, object] = {}
    for period, yahoo in ESTIMATE_PERIODS.items():
        for key, column in TREND_COLUMNS.items():
            row[f"{period}_{key}"] = _cell(trend, yahoo, column)
        for key, column in REVISION_COLUMNS.items():
            row[f"{period}_{key}"] = _cell(revisions, yahoo, column)
        for key, column in ANALYST_COLUMNS.items():
            row[f"{period}_{key}"] = _cell(analysts, yahoo, column)
    currencies = (
        cast("pd.Series", trend["currency"]).dropna()
        if "currency" in trend.columns
        else pd.Series()
    )
    row["est_currency"] = str(currencies.iloc[0]) if len(currencies) else ""
    return row


def _cell(table: pd.DataFrame, row: str, column: str) -> float:
    if row not in table.index or column not in table.columns:
        return math.nan
    return _number(table.loc[row, column])


def parse_ciks(raw: object) -> pd.DataFrame:
    """SEC's ``company_tickers.json`` as Yahoo tickers and ten-digit CIKs."""
    entries = cast("dict[str, object]", raw).values() if isinstance(raw, dict) else []
    rows = [
        (yahoo_symbol(str(item["ticker"])), f"{int(str(item['cik_str'])):010d}")
        for item in cast("list[dict[str, object]]", list(entries))
    ]
    frame = pd.DataFrame(rows, columns=["ticker", "cik"])
    return frame.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def _get(value: object, key: str) -> object:
    return (
        cast("dict[str, object]", value).get(key) if isinstance(value, dict) else None
    )


type Values = dict[str, tuple[float, str]]  # period end → (value, filed date)


def _entries(
    facts: Mapping[str, object], concept: str
) -> list[tuple[str, str, int, float, str, str, str]]:
    """A us-gaap concept's USD facts: start, end, days, value, filed, form, fp."""
    units = _get(_get(_get(facts.get("facts"), "us-gaap"), concept), "units")
    entries = _get(units, "USD")
    out: list[tuple[str, str, int, float, str, str, str]] = []
    for item in cast("list[object]", entries if isinstance(entries, list) else []):
        start, end, filed = (_get(item, key) for key in ("start", "end", "filed"))
        value = _number(_get(item, "val"))
        if (
            isinstance(start, str)
            and isinstance(end, str)
            and isinstance(filed, str)
            and not math.isnan(value)
        ):
            days = (date.fromisoformat(end) - date.fromisoformat(start)).days
            form, fp = str(_get(item, "form")), str(_get(item, "fp"))
            out.append((start, end, days, value, filed, form, fp))
    return out


def _keep[K](out: dict[K, tuple[float, str]], key: K, value: float, filed: str) -> None:
    """The latest filed value of a period wins."""
    if filed >= out.get(key, (math.nan, ""))[1]:
        out[key] = (value, filed)


def _within(days: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= days <= bounds[1]


def annual(facts: Mapping[str, object], concept: str) -> Values:
    """A us-gaap concept's 10-K full-year USD values by fiscal year end.

    Only form 10-K, fp FY and a ``YEAR_DAYS`` duration count. Later 10-Ks
    repeat earlier years, restated or not, so the latest filed value of each
    year end wins.
    """
    out: Values = {}
    for _, end, days, value, filed, form, fp in _entries(facts, concept):
        if form == "10-K" and fp == "FY" and _within(days, YEAR_DAYS):
            _keep(out, end, value, filed)
    return out


def quarterly(facts: Mapping[str, object], concept: str) -> Values:
    """A us-gaap concept's fiscal-quarter USD values by quarter end.

    Three-month (``QUARTER_DAYS``) values from 10-Qs, else from 10-Ks; latest
    filed wins per period. A quarter with none filed is the difference of two
    year-to-date values with the same start a quarter apart: six months less
    three, nine less six, the 10-K year less nine. The earlier one is the
    version filed last by the later one's filing, else the first filed.
    """
    tenq: Values = {}
    tenk: Values = {}
    ytd: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for start, end, days, value, filed, form, fp in _entries(facts, concept):
        if form in {"10-Q", "10-K"} and _within(days, QUARTER_DAYS):
            _keep(tenq if form == "10-Q" else tenk, end, value, filed)
        if (form == "10-Q" and any(_within(days, span) for span in YTD_DAYS)) or (
            form == "10-K" and fp == "FY" and _within(days, YEAR_DAYS)
        ):
            ytd.setdefault((start, end), []).append((filed, value))
    out = tenk | tenq
    derived: Values = {}
    for (start, end), versions in ytd.items():
        filed, value = max(versions)
        stop = date.fromisoformat(end)
        earlier = [
            sorted(v)
            for (s, e), v in ytd.items()
            if s == start and _within((stop - date.fromisoformat(e)).days, QUARTER_DAYS)
        ]
        if end not in out and earlier:
            prior = [v for v in earlier[0] if v[0] <= filed] or earlier[0][:1]
            _keep(derived, end, value - prior[-1][1], max(filed, prior[-1][0]))
    return out | derived


def sec_window(ends: Sequence[str], years: int = SEC_YEARS) -> list[str]:
    """The newest ``years`` year ends within that many years of the newest."""
    newest = sorted(ends, reverse=True)
    if not newest:
        return []
    cutoff = date.fromisoformat(newest[0]) - timedelta(days=365 * years - 180)
    return [end for end in newest if date.fromisoformat(end) > cutoff][:years]


def margin_history(facts: Mapping[str, object]) -> list[dict[str, object]]:
    """Each fiscal year's 10-K revenue and operating income, then four quarters.

    Every year end of ``OPERATING_INCOME`` or a revenue concept is an "FY"
    row, newest first. The revenue concept is the one in ``REVENUE_CONCEPTS``
    with the most years in ``sec_window`` that also have operating income,
    the earlier on a tie; a year without it takes the next concept that has
    one. The newest ``TTM_QUARTERS`` quarter ends of any value are "Q" rows,
    revenue in the first concept of that order with all of them, else in the
    first with that quarter. A quarter's revenue above ``QUARTER_SHARE`` of
    its fiscal year's in the same concept is dropped as mistagged.
    """
    income = annual(facts, OPERATING_INCOME)
    revenues = {concept: annual(facts, concept) for concept in REVENUE_CONCEPTS}
    ends = sorted(set(income).union(*revenues.values()), reverse=True)
    window = sec_window(ends)
    first = max(
        REVENUE_CONCEPTS,
        key=lambda c: sum(end in revenues[c] and end in income for end in window),
    )
    order = [first, *(c for c in REVENUE_CONCEPTS if c != first)]
    rows: list[dict[str, object]] = []
    for end in ends:
        concept = next((c for c in order if end in revenues[c]), "")
        revenue, filed = revenues[concept][end] if concept else (math.nan, "")
        op_income, op_filed = income.get(end, (math.nan, ""))
        rows.append(
            {
                "period": "FY",
                "end": end,
                "revenue": revenue,
                "op_income": op_income,
                "concept": concept,
                "filed": max(filed, op_filed),
            }
        )
    if not rows:
        return rows
    sales = {concept: quarterly(facts, concept) for concept in order}
    profit = quarterly(facts, OPERATING_INCOME)
    recent = sorted(set(profit).union(*sales.values()), reverse=True)[:TTM_QUARTERS]
    covering = [c for c in order if all(end in sales[c] for end in recent)]
    for end in recent:
        concept = next((c for c in covering or order if end in sales[c]), "")
        revenue, filed = sales[concept][end] if concept else (math.nan, "")
        year = [
            v
            for e, (v, _) in revenues.get(concept, {}).items()
            if 0
            <= (date.fromisoformat(e) - date.fromisoformat(end)).days
            < YEAR_DAYS[0]
        ]
        if year and revenue > QUARTER_SHARE * year[0]:
            revenue = math.nan
        op_income, op_filed = profit.get(end, (math.nan, ""))
        rows.append(
            {
                "period": "Q",
                "end": end,
                "revenue": revenue,
                "op_income": op_income,
                "concept": concept,
                "filed": max(filed, op_filed),
            }
        )
    return rows


def sec_ttm(quarters: pd.DataFrame, year_end: str) -> float:
    """Operating margin over the four "Q" rows, or NaN.

    All four need both values, a quarter apart each and three quarters
    (``NINE_MONTH_DAYS``) in all, the newest no older than ``year_end`` (the
    newest fiscal year), and revenue summing above zero.
    """
    rows = quarters.sort_values("end", ascending=False)
    ends = [date.fromisoformat(str(end)) for end in rows["end"]]
    gaps = [(newer - older).days for newer, older in pairwise(ends)]
    if (
        len(rows) != TTM_QUARTERS
        or bool(rows[["revenue", "op_income"]].isna().to_numpy().any())
        or not all(_within(gap, QUARTER_DAYS) for gap in gaps)
        or not _within(sum(gaps), NINE_MONTH_DAYS)
        or str(rows["end"].iloc[0]) < year_end
        or not rows["revenue"].sum() > 0
    ):
        return math.nan
    return float(rows["op_income"].sum() / rows["revenue"].sum())


SEC_SUMMARY: Final = (
    "ticker",
    "sec_avg",
    "sec_years",
    "sec_ttm",
    "sec_ttm_end",
)


def _kept(values: pd.Series, mask: pd.Series) -> pd.Series:
    """``values`` where ``mask`` holds, without the missing ones."""
    return cast("pd.Series", values[mask]).dropna()


def sec_row(years: pd.DataFrame, quarters: pd.DataFrame) -> dict[str, object]:
    """One ticker's SEC figures from its "FY" and "Q" rows.

    Margins count in years with operating income and revenue above zero: the
    ``sec_window`` median and its years.
    """
    years = years.sort_values("end", ascending=False)
    ends = cast("pd.Series", years["end"]).astype("str")
    recent = ends.isin(sec_window(list(ends)))
    margin = cast("pd.Series", years["op_income"] / years["revenue"]).where(
        (years["revenue"] > 0) & years["op_income"].notna()
    )
    window = _kept(margin, recent)
    return {
        "sec_avg": float(window.median()),
        "sec_years": len(window),
        "sec_ttm": sec_ttm(quarters, str(ends.max()) if len(ends) else ""),
        "sec_ttm_end": quarters["end"].max(),
    }


def sec_margins(margins: pd.DataFrame) -> pd.DataFrame:
    """Per ticker, ``sec_row`` of its stored rows (``SEC_SUMMARY`` columns).

    A file without every ``MARGIN_COLUMNS`` column is from an older version
    and gives nothing, so Yahoo's figures stand in until it is fetched again.
    """
    if not set(MARGIN_COLUMNS) <= set(margins.columns):  # an older file
        margins = pd.DataFrame(columns=list(MARGIN_COLUMNS))
    rows: list[dict[str, object]] = []
    for ticker, group in margins.dropna(subset=["end"]).groupby("ticker"):
        years = cast("pd.DataFrame", group[group["period"] == "FY"])
        quarters = cast("pd.DataFrame", group[group["period"] == "Q"])
        rows.append({"ticker": ticker, **sec_row(years, quarters)})
    return pd.DataFrame(rows, columns=list(SEC_SUMMARY))


# ---------------------------------------------------------------- indicators
# Every indicator takes adjusted daily series (``adjusted``).


def sma(values: pd.Series, days: int) -> pd.Series:
    return cast("pd.Series", values.rolling(days).mean())


def adjusted(frame: pd.DataFrame) -> pd.DataFrame:
    """Open, high and low scaled by Adj Close / Close, so all four agree."""
    factor = frame["adj_close"] / frame["close"]
    return pd.DataFrame(
        {
            "open": frame["open"] * factor,
            "high": frame["high"] * factor,
            "low": frame["low"] * factor,
            "close": frame["adj_close"],
            "volume": frame["volume"],
        },
        index=frame.index,
    )


def tech_row(frame: pd.DataFrame) -> dict[str, float]:
    """One ticker's indicators on its last bar; NaN where the bars run short.

    ``frame`` is sorted by date.
    """
    bars = len(frame)
    raw = cast("pd.Series", frame["close"])
    close = cast("pd.Series", adjusted(frame)["close"])
    last = float(close.iloc[-1])
    row = {"bars": float(bars), "price": float(raw.iloc[-1]), "close": last}
    for days in SMA_DAYS:
        row[f"sma{days}"] = float(sma(close, days).iloc[-1])

    def back(days: int) -> float:
        return float(close.iloc[-days - 1]) if bars > days else math.nan

    row |= {
        "near_high": last / float(close.tail(YEAR_BARS).max())
        if bars >= YEAR_BARS
        else math.nan,
        "ma_dist": last / row["sma200"],
        "mom12_1": back(MONTH_BARS) / back(YEAR_BARS) - 1,
        "ret_1m": last / back(MONTH_BARS) - 1,
        "ret_3m": last / back(QUARTER_BARS) - 1,
    }
    return row


def technicals(prices: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"ticker": ticker, **tech_row(group.sort_values("date"))}
        for ticker, group in prices.groupby("ticker", sort=False)
    ]
    return pd.DataFrame(rows, columns=["ticker", *TECH_COLUMNS])


# ---------------------------------------------------------------- rules


def liquidity(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.Series:
    """Why a stock is left out ("" when it stays): no prices, then the price.

    No prices at all always leaves a stock out; the price floor only when on.
    """
    reason = pd.Series("", index=table.index, dtype="str")
    reason[table["bars"].isna()] = "시세 없음"
    if c.on("min_price"):
        cheap = (reason == "") & ~(table["price"] >= MIN_PRICE)
        reason[cheap] = f"주가 ${MIN_PRICE:g} 미만"
    return reason


def _days(day: object) -> pd.Series:
    return pd.to_datetime(cast("pd.Series", day), errors="coerce")


def year_state(table: pd.DataFrame) -> pd.DataFrame:
    """Where each stock stands in its fiscal year, as of ``as_of``.

    ``pending``: the year Yahoo calls "0y" (ending ``next_fy_end``) has ended
    and is not yet reported. ``weight`` is FY1's share of the forward 12
    months: the year's time left over a year, 0 when pending, 1 without dates.
    """
    t = table
    as_of = _days(t["as_of"])
    next_end = _days(t["next_fy_end"])
    pending = next_end <= as_of
    left = (next_end - as_of).dt.days / 365.25
    weight = left.clip(0, 1).fillna(1.0).where(~pending, 0.0)
    return pd.DataFrame({"pending": pending, "weight": weight}, index=t.index)


def revisions(table: pd.DataFrame) -> pd.DataFrame:
    """수정 강도 and 순수정 비율, averaged over FY1 and FY2 (FY2 only when pending).

    Strength: (now - 90 days ago) / price, each period against itself. Net:
    (30-day up - down) / the analysts estimating.
    """
    t = table
    strength, net = [], []
    for period in ESTIMATE_PERIODS:
        every = pd.Series(data=True, index=t.index)
        used = ~t["pending"] if period == "fy1" else every
        price = t["price"].where(t["price"] > 0)
        strength.append(((t[f"{period}_now"] - t[f"{period}_90d"]) / price).where(used))
        analysts = t[f"{period}_n"].where(t[f"{period}_n"] > 0)
        net.append(
            ((t[f"{period}_up30"] - t[f"{period}_down30"]) / analysts).where(used)
        )
    return pd.DataFrame(
        {
            "rev_strength": pd.concat(strength, axis=1).mean(axis=1),
            "net_revision": pd.concat(net, axis=1).mean(axis=1),
        },
        index=t.index,
    )


# Yahoo leaves out a row whose balance is zero rather than writing 0 (a
# company with no debt has neither debt row), so these items read as zero.
ZERO_WHEN_MISSING: Final = ("sti", "std", "ltd", "pref", "inv_adv", "cash_sti")


def _item(table: pd.DataFrame, stem: str, year: int) -> pd.Series:
    values = cast("pd.Series", table[f"{stem}_{year}"])
    return values.fillna(0.0) if stem in ZERO_WHEN_MISSING else values


def leverage_ratio(table: pd.DataFrame) -> pd.Series:
    """Operating cash flow / total debt, one at most (FTSE rule 5.4.2).

    Total debt is short-term plus long-term debt; no debt, or cash flow above
    the debt, gives one.
    """
    ocf = cast("pd.Series", table["ocf_0"])
    debt = _item(table, "std", 0) + _item(table, "ltd", 0)
    ratio = (ocf / debt.where(debt > 0)).where(ocf <= debt, 1.0)
    return ratio.where(debt > 0, 1.0).where(ocf.notna())


def measure_values(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Each 퀄리티 and 밸류에이션 measure's value, higher better and cheaper.

    FTSE rules 5.4 and 5.6 on the newest fiscal year ("_0") and the one before
    ("_1"). Financials and real estate (``ROA_ONLY``) have ROA alone for
    퀄리티. A total or average asset at or below zero leaves the measures on
    it without a value (FTSE: a neutral Z-score). Leverage is less its GICS
    sector's median, unless the sector has fewer than ``MEDIAN_MIN``.
    """
    t = table
    roa_only = t["sector"].isin(list(ROA_ONLY))
    ta0, ta1 = t["ta_0"], t["ta_1"]
    average = ((ta0 + ta1) / 2).where((ta0 > 0) & (ta1 > 0))

    def wc(year: int) -> pd.Series:
        return (t[f"ca_{year}"] - _item(t, "cash_sti", year)) - (
            t[f"cl_{year}"] - _item(t, "std", year)
        )

    def nco(year: int) -> pd.Series:
        return (t[f"ta_{year}"] - t[f"ca_{year}"] - _item(t, "inv_adv", year)) - (
            t[f"tl_{year}"] - t[f"cl_{year}"] - _item(t, "ltd", year)
        )

    def fin(year: int) -> pd.Series:
        return (_item(t, "sti", year) + _item(t, "inv_adv", year)) - (
            _item(t, "ltd", year) + _item(t, "std", year) + _item(t, "pref", year)
        )

    change = (wc(0) - wc(1)) + (nco(0) - nco(1)) + (fin(0) - fin(1))
    turnover = t["sales_0"] / ta0.where(ta0 > 0) - t["sales_1"] / ta1.where(ta1 > 0)
    leverage = leverage_ratio(t)
    sector = leverage.groupby(t["sector"])
    sizes = cast("pd.Series", sector.transform("count"))
    median = cast("pd.Series", sector.transform("median")).where(sizes >= MEDIAN_MIN)
    mcap = t["mcap"].where(t["mcap"] > 0)
    values = {
        "q_roa": t["ni_0"] / average,
        "q_turnover": turnover.where(~roa_only),
        "q_accruals": (-change / average).where(~roa_only),
        "q_leverage": (leverage - median.fillna(0.0)).where(~roa_only),
        "v_cash": t["ocf_0"] / mcap,
        "v_earnings": t["ni_0"] / mcap,
        "v_sales": t["sales_0"] / mcap,
    }
    return cast("dict[str, pd.Series]", values)


def normalise(values: pd.Series) -> pd.Series:
    """FTSE rule 5.2.1: cross-sectional Z-scores truncated at ``Z_LIMIT``.

    Truncated scores are renormalised with the rest until all lie within
    the limit. Missing values stay missing; no spread scores zero.
    """
    z = values.astype("float")
    for _ in range(100):  # converges in a few rounds
        spread = float(cast("float", z.std(ddof=0)))
        if not spread > 0:
            return z.where(z.isna(), 0.0)
        z = (z - z.mean()) / spread
        if not (z.abs() > Z_LIMIT).any():
            break
        z = z.clip(-Z_LIMIT, Z_LIMIT)
    return z


def composite(scores: pd.DataFrame) -> pd.Series:
    """FTSE rule 5.2.2: the mean of the non-missing Z-scores, renormalised."""
    return normalise(scores.mean(axis=1))


def axes(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.DataFrame:
    """퀄리티 and 밸류에이션 Z-scores across ``table`` (FTSE rules 5.2, 5.4, 5.6).

    Each measure in force is normalised; profitability is the composite of its
    measures, 퀄리티 that of profitability and leverage, 밸류에이션 that of its
    three yields. A stock without any 퀄리티 (밸류에이션) value scores zero (rule
    5.2.3). The measure columns keep each measure's Z-score, for the dialog.
    """
    values = c.kept(measure_values(table))
    z = {key: normalise(value) for key, value in values.items()}
    out = pd.DataFrame(index=table.index)
    for key in (*QUALITY, *VALUE):
        out[key] = z.get(key, pd.Series(math.nan, index=table.index))

    def scored(keys: Iterable[str]) -> pd.DataFrame:
        return pd.DataFrame({k: z[k] for k in keys if k in z}, index=table.index)

    profit = scored(PROFITABILITY)
    parts = {"profit": composite(profit)} if len(profit.columns) else {}
    if "q_leverage" in z:
        parts["q_leverage"] = z["q_leverage"]
    quality = pd.DataFrame(parts, index=table.index)
    value = scored(VALUE)
    out["q_count"] = scored(QUALITY).count(axis=1)
    out["v_count"] = value.count(axis=1)
    out["quality"] = (
        composite(quality).fillna(0.0) if len(quality.columns) else math.nan
    )
    out["value"] = composite(value).fillna(0.0) if len(value.columns) else math.nan
    return out


def evaluate(inputs: Inputs, criteria: Criteria = DEFAULT) -> pd.DataFrame:
    """Each constituent with its price floor, checks, 퀄리티, 밸류에이션, cell."""
    return judge(prepare(inputs), criteria)


def prepare(inputs: Inputs) -> pd.DataFrame:
    """The stored parts as one row per constituent with its indicators.

    The operating margin median (``margin_avg``, for the dialog) is SEC's
    ``SEC_YEARS`` 10-K years where they are all there, else Yahoo's
    (``margin_source``). A stock on SEC's median takes SEC's TTM too when its
    four quarters are there and end no earlier than Yahoo's newest quarter
    (``STALE_SLACK_DAYS`` aside), else Yahoo's (``op_margin_source``);
    ``op_margin_end`` is the end of the used TTM's newest quarter. ``as_of``
    is the newest price day, which the fiscal year state goes by.
    """
    table = inputs.universe.merge(technicals(inputs.prices), on="ticker", how="left")
    table = table.merge(inputs.fundamentals, on="ticker", how="left")
    table = table.merge(
        inputs.estimates, on="ticker", how="left", suffixes=("", "_estimates")
    )
    table = table.merge(sec_margins(inputs.margins), on="ticker", how="left")
    for column in (
        *TECH_COLUMNS,
        *FUNDAMENTAL_COLUMNS,
        *ESTIMATE_COLUMNS,
        *SEC_SUMMARY,
    ):
        if column not in table.columns:
            table[column] = math.nan
    for column in ("currency", "est_currency", "ttm_end", "next_fy_end"):
        table[column] = table[column].fillna("").astype("str").replace("nan", "")
    newest = (
        cast("pd.Timestamp", inputs.prices["date"].max())
        if len(inputs.prices)
        else None
    )
    table["as_of"] = "" if newest is None else str(newest.date())
    sec = table["sec_years"] >= SEC_YEARS
    table["margin_source"] = np.select(
        [sec, table["margin_years"] > 0], ["SEC", "야후"], ""
    )
    table["margin_avg"] = table["sec_avg"].where(sec, table["margin_avg"])
    table["margin_years"] = table["sec_years"].where(sec, table["margin_years"])
    behind = pd.to_datetime(table["sec_ttm_end"], errors="coerce") < pd.to_datetime(
        table["ttm_end"], errors="coerce"
    ) - pd.Timedelta(days=STALE_SLACK_DAYS)
    ttm = sec & table["sec_ttm"].notna() & ~behind
    table["op_margin_source"] = np.select(
        [ttm, table["op_margin"].notna()], ["SEC", "야후"], ""
    )
    table["op_margin_end"] = (
        table["sec_ttm_end"]
        .where(ttm, table["ttm_end"].where(table["op_margin"].notna(), ""))
        .fillna("")
        .astype("str")
    )
    table["op_margin"] = table["sec_ttm"].where(ttm, table["op_margin"])
    state = year_state(table)
    table[list(state.columns)] = state
    revised = revisions(table)
    table[list(revised.columns)] = revised
    table["error"] = _errors(table)
    return table.drop(columns=["error_estimates"], errors="ignore")


def judge(prepared: pd.DataFrame, c: Criteria) -> pd.DataFrame:
    """``prepare``'s table screened by ``c``: price floor, then both axes.

    The axes rank the stocks past the price floor only.
    """
    table = prepared.copy()
    table["liquidity"] = liquidity(table, c)
    liquid = cast("pd.DataFrame", table[table["liquidity"] == ""]).copy()
    found = axes(liquid, c)
    liquid[list(found.columns)] = found
    values = measure_values(liquid)
    liquid["roa"] = values["q_roa"]
    liquid["leverage"] = leverage_ratio(liquid).where(
        ~liquid["sector"].isin(list(ROA_ONLY))
    )
    liquid["earnings_yield"] = values["v_earnings"]
    liquid["cash_yield"] = values["v_cash"]
    liquid["sales_yield"] = values["v_sales"]

    rest = cast("pd.DataFrame", table[table["liquidity"] != ""])
    out = pd.concat([liquid, rest]) if len(rest) else liquid
    return out.sort_values("mcap", ascending=False, ignore_index=True)


def _errors(table: pd.DataFrame) -> pd.Series:
    """The fundamentals and estimates fetch errors, joined."""
    texts = [
        cast("pd.Series", table[column]).fillna("").astype("str")
        if column in table.columns
        else pd.Series("", index=table.index)
        for column in ("error", "error_estimates")
    ]
    joined = [
        " · ".join(
            f"{name}: {text}"
            for name, text in (("재무", fund), ("추정치", est))
            if text
        )
        for fund, est in zip(*texts, strict=True)
    ]
    return pd.Series(joined, index=table.index, dtype="str")


def liquid(table: pd.DataFrame) -> pd.DataFrame:
    return cast("pd.DataFrame", table[table["liquidity"] == ""])


def pick(table: pd.DataFrame, chosen: Sequence[str] = ()) -> pd.DataFrame:
    """Stocks past the price floor in the ``chosen`` sectors (all when empty)."""
    rows = liquid(table)
    if chosen:
        rows = cast("pd.DataFrame", rows[rows["sector"].isin(list(chosen))])
    return rows.reset_index(drop=True)


def map_rows(table: pd.DataFrame) -> list[dict[str, object]]:
    """The stocks with both axes as map points, with what their tooltip shows."""
    placed = liquid(table).dropna(subset=["quality", "value"])
    return [
        {
            "ticker": str(row["ticker"]),
            "name": str(row["name"]),
            "sector": str(row["sector"]),
            "quality": float(cast("float", row["quality"])),
            "value": float(cast("float", row["value"])),
            "counted": counted(row),
        }
        for row in records(placed)
    ]


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", frame.to_dict(orient="records"))


def view(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows as the results table shows them (``TABLE``), in the same order."""
    shown = frame.reset_index(drop=True)
    rows = records(shown)
    shown = shown.assign(counted=[counted(row) for row in rows])
    return cast("pd.DataFrame", shown[[column for column, _, _ in TABLE]])


def sectors(table: pd.DataFrame) -> list[str]:
    return sorted({str(sector) for sector in table["sector"].dropna()})


def failures(table: pd.DataFrame) -> pd.DataFrame:
    """Stocks left out or not fully fetched, with the reason."""
    reason = table["liquidity"].where(table["error"] == "", table["error"])
    both = (table["liquidity"] != "") & (table["error"] != "")
    reason = reason.where(~both, table["liquidity"] + " · " + table["error"])
    shown = table.assign(reason=reason)
    return cast("pd.DataFrame", shown[reason != ""][["ticker", "name", "reason"]])


# ---------------------------------------------------------------- page text


def fmt(kind: str, value: object) -> str:
    """A number as the page shows it (``FORMATS``); missing values are a dash.

    "big" amounts are in B above a billion, else in M; minus is U+2212.
    """
    number = _number(value)
    if math.isnan(number):
        return "—"
    if kind == "big":
        size, unit = (1e9, "B") if abs(number) >= 1e9 else (1e6, "M")  # noqa: PLR2004
        return FORMATS["big"].format(number / size, unit).replace("-", "−")
    text = FORMATS.get(kind, FORMATS["num"]).format(number)
    if not any(digit in text for digit in "123456789"):  # no signed zero
        text = text.replace("+", "").replace("-", "")
    return text.replace("-", "−")


def counted(row: Mapping[str, object]) -> str:
    """산출 항목 수: 퀄리티 / 밸류에이션 measures with a value."""
    q, v = _number(row.get("q_count")), _number(row.get("v_count"))
    return "" if math.isnan(q) or math.isnan(v) else f"{q:.0f}/{v:.0f}"


MARGIN_SOURCES: Final = {"SEC": "SEC 10-K", "야후": "야후"}
TTM_SOURCES: Final = {"SEC": "SEC 분기", "야후": "야후"}


def _score(value: object) -> str:
    number = _number(value)
    return UNKNOWN if math.isnan(number) else fmt("z", number)


def _items(row: Mapping[str, object], year: int, *stems: str) -> str:
    names = {
        "ni": "순이익",
        "sales": "매출",
        "ocf": "영업현금흐름",
        "ta": "총자산",
        "ca": "유동자산",
        "cl": "유동부채",
        "cash_sti": "현금·단기투자",
        "sti": "단기투자",
        "std": "단기부채",
        "ltd": "장기부채",
        "tl": "총부채",
        "pref": "우선주",
        "inv_adv": "투자자산",
    }
    return " · ".join(
        f"{names[stem]} {fmt('big', row.get(f'{stem}_{year}'))}" for stem in stems
    )


def measure_lines(row: Mapping[str, object], c: Criteria) -> list[tuple[str, str, str]]:
    """Each 퀄리티 and 밸류에이션 measure, then both axes.

    Each line is the measure's Z-score, the test and the values it used; a
    measure turned off is 미적용, one FTSE does not use for the stock 해당 없음.
    """
    year = f"{row.get('fy_end') or '—'} 회계연도"
    measures = {
        "q_roa": (
            "순이익 ÷ 평균 총자산",
            f"{fmt('pct', row.get('roa'))} · {_items(row, 0, 'ni', 'ta')} · 전년"
            f" {_items(row, 1, 'ta')}",
        ),
        "q_turnover": (
            "매출 ÷ 총자산의 전년 대비 변화",
            f"{_items(row, 0, 'sales', 'ta')} · 전년 {_items(row, 1, 'sales', 'ta')}",
        ),
        "q_accruals": (
            "−(Δ운전자본 + Δ비유동 순영업자산 + Δ순금융자산) ÷ 평균 총자산",
            _items(row, 0, "ca", "cl", "cash_sti", "sti", "std", "ltd", "tl", "pref")
            + f" · {_items(row, 0, 'inv_adv')}",
        ),
        "q_leverage": (
            "영업현금흐름 ÷ 총부채 (최대 1, 섹터 중앙값 대비)",
            f"{fmt('num', row.get('leverage'))} ·"
            f" {_items(row, 0, 'ocf', 'std', 'ltd')}",
        ),
        "v_cash": (
            "영업현금흐름 ÷ 시가총액",
            f"{fmt('pct', row.get('cash_yield'))} · {_items(row, 0, 'ocf')}",
        ),
        "v_earnings": (
            "순이익 ÷ 시가총액",
            f"{fmt('pct', row.get('earnings_yield'))} · {_items(row, 0, 'ni')}",
        ),
        "v_sales": (
            "매출 ÷ 시가총액",
            f"{fmt('num', row.get('sales_yield'))} · {_items(row, 0, 'sales')} ·"
            f" 시가총액 {fmt('big', row.get('mcap'))}",
        ),
    }
    roa_only = row.get("sector") in ROA_ONLY
    return [
        *(
            (
                UNUSED
                if not c.on(key)
                else NOT_APPLICABLE
                if roa_only and key in QUALITY and key != "q_roa"
                else _score(row.get(key)),
                f"{QUALITY.get(key) or VALUE[key]} · {test}",
                detail,
            )
            for key, (test, detail) in measures.items()
        ),
        (
            _score(row.get("quality")),
            "퀄리티 = 수익성·레버리지 z의 평균을 다시 표준화 (높을수록 우수)",
            f"{fmt('int', row.get('q_count'))}개 지표 · {year}",
        ),
        (
            _score(row.get("value")),
            "밸류에이션 = 세 수익률 z의 평균을 다시 표준화 (높을수록 쌈)",
            f"{fmt('int', row.get('v_count'))}개 지표 · {year}",
        ),
    ]


def tech_lines(row: Mapping[str, object]) -> list[tuple[str, str, str]]:
    """The price indicators: value, name and definition."""
    kinds = {
        "mom12_1": "signed",
        "near_high": "pct",
        "ma_dist": "num",
        "ret_1m": "signed",
        "ret_3m": "signed",
    }
    return [
        (fmt(kinds[key], row.get(key)), name, definition)
        for key, (name, definition) in TECH_LINES.items()
    ]


def margin_line(row: Mapping[str, object]) -> str:
    """The TTM operating margin beside its multi-year median, for reference."""
    source = MARGIN_SOURCES.get(str(row.get("margin_source")), "—")
    ttm_source = TTM_SOURCES.get(str(row.get("op_margin_source")), "—")
    ttm_end = row.get("op_margin_end") or ""
    basis = f"{ttm_source}, {ttm_end}까지" if ttm_end else ttm_source
    return (
        f"TTM {fmt('pct', row.get('op_margin'))} ({basis}) ·"
        f" {fmt('int', row.get('margin_years'))}개 연도 중앙값"
        f" {fmt('pct', row.get('margin_avg'))} ({source})"
    )


def estimate_table(row: Mapping[str, object]) -> pd.DataFrame:
    """FY1/FY2 consensus EPS, 30-day revisions and analysts as a small table."""
    return pd.DataFrame(
        {
            "기간": ["FY1", "FY2"],
            "현재": [fmt("num", row.get(f"{p}_now")) for p in ESTIMATE_PERIODS],
            "30일 전": [fmt("num", row.get(f"{p}_30d")) for p in ESTIMATE_PERIODS],
            "90일 전": [fmt("num", row.get(f"{p}_90d")) for p in ESTIMATE_PERIODS],
            "30일 상향": [fmt("int", row.get(f"{p}_up30")) for p in ESTIMATE_PERIODS],
            "30일 하향": [fmt("int", row.get(f"{p}_down30")) for p in ESTIMATE_PERIODS],
            "애널리스트": [fmt("int", row.get(f"{p}_n")) for p in ESTIMATE_PERIODS],
        }
    )


def estimate_note(row: Mapping[str, object]) -> str:
    """수정 강도, 순수정 비율 and where the fiscal year stands."""
    state = (
        "결산 대기(FY1 끝, 10-K 전 · 선행 EPS는 FY2)"
        if row.get("pending") is True
        else f"FY1 비중 {fmt('rank', row.get('weight'))}"
    )
    return (
        f"수정 강도 {fmt('signed2', row.get('rev_strength'))}"
        f" · 순수정 비율 {fmt('signed', row.get('net_revision'))} · {state}"
        f" · 회계연도 끝 {row.get('next_fy_end') or '—'}"
    )


def chart_rows(history: pd.DataFrame) -> list[dict[str, str | float]]:
    """The last year of adjusted closes with SMA 20/60/200, as chart rows."""
    if history.empty:
        return []
    frame = history.sort_values("date").reset_index(drop=True)
    close = cast("pd.Series", adjusted(frame)["close"])
    lines = {
        "종가": close,
        "20일선": sma(close, 20),
        "60일선": sma(close, 60),
        "200일선": sma(close, 200),
    }
    start = frame["date"].iloc[-1] - pd.DateOffset(years=1)
    keep = frame["date"] >= start
    return [
        {
            "day": cast("pd.Timestamp", day).date().isoformat(),
            "line": name,
            "value": float(value),
        }
        for name, values in lines.items()
        for day, value in zip(frame["date"][keep], values[keep], strict=True)
        if not math.isnan(value)
    ]


# ---------------------------------------------------------------- storage


def quarter(day: date) -> str:
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def _newest(root: Path, pattern: str) -> Path | None:
    return max(root.glob(pattern), default=None)


def _stem_date(path: Path, prefix: str) -> date:
    return date.fromisoformat(path.stem.removeprefix(prefix))


SNAPSHOT_FOLDER: Final = "snapshots"
SNAPSHOT_COLUMNS: Final = (
    "ticker",
    "sector",
    "quality",
    "value",
    "q_count",
    "v_count",
)


def write_snapshot(root: Path, table: pd.DataFrame, c: Criteria, day: date) -> Path:
    """Each stock's axes on ``day``, with the switches in force.

    One CSV and one JSON (``DEFINITIONS``, the price day and the switches
    turned off) per day.
    """
    folder = root / SNAPSHOT_FOLDER
    rows = liquid(table).reset_index(drop=True)
    frame = cast("pd.DataFrame", rows[list(SNAPSHOT_COLUMNS)])
    path = folder / f"{day.isoformat()}.csv"
    _write(frame, path)
    as_of = str(rows["as_of"].iloc[0]) if len(rows) else ""
    meta = {"definitions": DEFINITIONS, "as_of": as_of, "off": c.off}
    temp = folder / f"{day.isoformat()}.json.tmp"
    _ = temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    _ = temp.replace(folder / f"{day.isoformat()}.json")
    return path


def prune(root: Path) -> None:
    """Delete every dated file in ``root`` but the newest of its kind."""
    for pattern in DATED_GLOBS:
        for path in sorted(root.glob(pattern))[:-1]:
            path.unlink()


def universe_due(root: Path, today: date, pattern: str = UNIVERSE_GLOB) -> bool:
    """Whether the newest dated file of ``pattern`` is missing or a month old."""
    path = _newest(root, pattern)
    if path is None:
        return True
    age = today - _stem_date(path, pattern.split("?", 1)[0])
    return age.days >= UNIVERSE_MAX_AGE_DAYS


def pending(existing: pd.DataFrame, tickers: Sequence[str]) -> list[str]:
    """Tickers with no row yet, or whose row records a failure."""
    if existing.empty:
        return list(tickers)
    done = existing[existing["error"].fillna("") == ""]["ticker"]
    have = set(done.astype("str"))
    return [ticker for ticker in tickers if ticker not in have]


def last_session(now: datetime) -> date:
    """The newest US trading day whose daily bar is final at ``now``."""
    local = now.astimezone(MARKET_ZONE)
    day = local.date()
    return day if local.time() >= SESSION_SETTLED else day - timedelta(days=1)


def price_plan(
    store: pd.DataFrame, tickers: Sequence[str], first: date | None = None
) -> tuple[date | None, list[str]]:
    """Where the incremental fetch starts, and which tickers need full history.

    Every ticker needs it when the store starts ``BACKFILL_SLACK_DAYS`` after
    ``first``, the first day the kept window wants.
    """
    oldest = cast("pd.Timestamp", store["date"].min()).date() if len(store) else None
    if oldest is None or (
        first is not None and oldest > first + timedelta(days=BACKFILL_SLACK_DAYS)
    ):
        return None, list(tickers)
    newest = cast("pd.Timestamp", store["date"].max()).date()
    start = newest - timedelta(days=OVERLAP_DAYS)
    newest_by = cast("pd.Series", store.groupby("ticker")["date"].max())
    last = {str(ticker): pd.Timestamp(day) for ticker, day in newest_by.items()}
    cutoff = pd.Timestamp(start)
    full = [t for t in tickers if t not in last or last[t] < cutoff]
    return start, full


def drifted(store: pd.DataFrame, fresh: pd.DataFrame) -> list[str]:
    """Tickers whose refetched Adj Close no longer matches the stored one.

    A split or a new dividend rescales Yahoo's whole adjusted history, so the
    stored rows would no longer join the new ones.
    """
    if store.empty or fresh.empty:
        return []
    both = store.merge(fresh, on=["ticker", "date"], suffixes=("", "_new"))
    change = (both["adj_close_new"] / both["adj_close"] - 1).abs()
    return sorted(set(both.loc[change > ADJUST_TOLERANCE, "ticker"].astype("str")))


def merge_prices(
    store: pd.DataFrame,
    fresh: Sequence[pd.DataFrame],
    replaced: Sequence[str],
    tickers: Sequence[str],
    last_day: date,
) -> pd.DataFrame:
    """Stored rows, minus ``replaced`` tickers, updated by ``fresh`` rows."""
    kept = cast("pd.DataFrame", store[~store["ticker"].isin(list(replaced))])
    frames = [frame for frame in (kept, *fresh) if not frame.empty]
    if not frames:
        return empty_prices()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker", "date"], keep="last")
    first = pd.Timestamp(last_day - timedelta(days=HISTORY_DAYS))
    keep = (
        merged["ticker"].isin(list(tickers))
        & (merged["date"] >= first)
        & (merged["date"] <= pd.Timestamp(last_day))
    )
    kept_rows = cast("pd.DataFrame", merged[keep])
    return kept_rows.sort_values(["ticker", "date"], ignore_index=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write next to ``path`` and rename, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temp, index=False)
    else:
        frame.to_csv(temp, index=False)
    _ = temp.replace(path)


def _read_csv(path: Path | None, columns: Sequence[str]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame({column: [] for column in columns})
    frame = pd.read_csv(path, dtype={"ticker": "str", "error": "str"})
    frame["error"] = frame["error"].fillna("")
    return frame


def read_universe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="str").fillna("")


def read_prices(root: Path) -> pd.DataFrame:
    path = root / PRICES_FILE
    return pd.read_parquet(path) if path.exists() else empty_prices()


def history(root: Path, ticker: str) -> pd.DataFrame:
    """One ticker's stored daily rows, or none."""
    try:
        return pd.read_parquet(root / PRICES_FILE, filters=[("ticker", "==", ticker)])
    except (OSError, ValueError):
        return empty_prices()


def parts(root: Path) -> Parts:
    universe = _newest(root, UNIVERSE_GLOB)
    fundamentals = _newest(root, FUNDAMENTALS_GLOB)
    estimates = _newest(root, ESTIMATES_GLOB)
    margins = _newest(root, MARGINS_GLOB)
    prices = root / PRICES_FILE
    last = None
    if prices.exists():
        days = pd.read_parquet(prices, columns=["date"])["date"]
        last = cast("pd.Timestamp", days.max()).date() if len(days) else None
    return Parts(
        universe=_stem_date(universe, "universe-") if universe else None,
        prices=last,
        fundamentals=fundamentals.stem.removeprefix("fundamentals-")
        if fundamentals
        else None,
        estimates=_stem_date(estimates, "estimates-") if estimates else None,
        margins=margins.stem.removeprefix("margins-") if margins else None,
    )


def stamp(root: Path) -> tuple[tuple[str, int], ...]:
    """The newest files and their change times: a cache key for ``load``."""
    paths = [
        _newest(root, UNIVERSE_GLOB),
        root / PRICES_FILE,
        _newest(root, FUNDAMENTALS_GLOB),
        _newest(root, ESTIMATES_GLOB),
        _newest(root, MARGINS_GLOB),
    ]
    return tuple(
        (path.name, path.stat().st_mtime_ns)
        for path in paths
        if path is not None and path.exists()
    )


def load(root: Path) -> Inputs | None:
    """The newest stored parts; ``None`` until there is a universe and prices."""
    universe = _newest(root, UNIVERSE_GLOB)
    if universe is None or not (root / PRICES_FILE).exists():
        return None
    return Inputs(
        universe=read_universe(universe),
        prices=read_prices(root),
        fundamentals=_read_csv(_newest(root, FUNDAMENTALS_GLOB), FUNDAMENTAL_COLUMNS),
        estimates=_read_csv(_newest(root, ESTIMATES_GLOB), ESTIMATE_COLUMNS),
        margins=_read_csv(_newest(root, MARGINS_GLOB), MARGIN_COLUMNS),
    )


# ---------------------------------------------------------------- fetching


@final
class Yahoo:
    """The real source: Wikipedia for the members, yfinance for the numbers."""

    def __init__(self, root: Path, universe: str = "sp500") -> None:
        """Point yfinance's caches into ``root`` instead of the user cache."""
        import yfinance as yf  # noqa: PLC0415 - optional dependency

        root.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(root / YFINANCE_CACHE))
        self._url = UNIVERSES[universe].url

    def constituents(self) -> pd.DataFrame:
        response = httpx.get(
            self._url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        _ = response.raise_for_status()
        return parse_constituents(response.text)

    def prices(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf  # noqa: PLC0415

        frame = yf.download(
            list(tickers),
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=True,
        )
        return long_prices(frame if frame is not None else pd.DataFrame())

    def fundamentals(self, ticker: str) -> Mapping[str, object]:
        import yfinance as yf  # noqa: PLC0415

        stock = yf.Ticker(ticker)
        with _rate_limit():
            info = cast("Mapping[str, object]", stock.info)
            if math.isnan(_number(info.get("marketCap"))):
                raise EmptyAnswerError("info 빈 응답")
            statements = Statements(
                income=nonempty(stock.get_income_stmt(freq="yearly"), "손익계산서"),
                quarterly=_frame(stock.get_income_stmt(freq="quarterly")),
                balance=_frame(stock.get_balance_sheet(freq="yearly")),
                cash=_frame(stock.get_cashflow(freq="yearly")),
            )
        return fundamentals_row(info, statements)

    def estimates(self, ticker: str) -> Mapping[str, object]:
        import yfinance as yf  # noqa: PLC0415

        stock = yf.Ticker(ticker)
        with _rate_limit():
            trend = nonempty(stock.eps_trend, "EPS 추정치")
            return estimates_row(
                trend, _frame(stock.eps_revisions), _frame(stock.earnings_estimate)
            )


_SEC_GATE: Final = threading.Lock()  # one SEC request at a time in this process


@final
class Sec:
    """SEC EDGAR with the user's User-Agent; never built without one."""

    def __init__(self, agent: str, sleep: Callable[[float], None] = time.sleep) -> None:
        """``agent`` is ``SEC_USER_AGENT``, the name and contact SEC asks for."""
        if not agent.strip():
            raise ValueError("SEC_USER_AGENT가 비어 있습니다")
        self._client = httpx.Client(
            headers={"User-Agent": agent.strip(), "Accept-Encoding": "gzip, deflate"},
            timeout=SEC_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        self._sleep = sleep

    def _get(self, url: str) -> httpx.Response:
        """One request; ``SecUnavailableError`` when SEC is unreachable or refuses."""
        with _SEC_GATE:
            try:
                response = self._client.get(url)
            except httpx.TransportError as error:
                raise SecUnavailableError(f"{type(error).__name__}: {error}") from error
            finally:
                self._sleep(SEC_PAUSE)
        if response.status_code in REFUSED:
            raise SecUnavailableError(f"HTTP {response.status_code}")
        return response

    def tickers(self) -> pd.DataFrame:
        response = self._get(SEC_TICKERS_URL)
        _ = response.raise_for_status()
        return parse_ciks(response.json())

    def facts(self, cik: str) -> Mapping[str, object] | None:
        response = self._get(COMPANYFACTS_URL.format(cik=cik))
        if response.status_code == NOT_FOUND:
            return None
        _ = response.raise_for_status()
        return cast("Mapping[str, object]", response.json())

    def close(self) -> None:
        self._client.close()


def _frame(value: object) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def nonempty(value: object, what: str) -> pd.DataFrame:
    """``value`` as a table, or ``EmptyAnswerError`` when Yahoo sent none."""
    frame = _frame(value)
    if frame.empty:
        raise EmptyAnswerError(f"{what} 빈 응답")
    return frame


@contextmanager
def _rate_limit() -> Generator[None]:
    """Turn yfinance's rate-limit error into ``RateLimitedError``."""
    from yfinance.exceptions import YFRateLimitError  # noqa: PLC0415

    try:
        yield
    except YFRateLimitError as error:
        raise RateLimitedError(str(error)) from error


def _retry[T](call: Callable[[], T], sleep: Callable[[float], None]) -> T:
    for wait in RETRY_WAITS:
        try:
            return call()
        except RateLimitedError:
            sleep(wait)
    return call()


def _download(
    source: Source,
    tickers: Sequence[str],
    start: date,
    end: date,
    sleep: Callable[[float], None],
) -> pd.DataFrame:
    """One chunk; yfinance hides rate limits, so an empty answer is retried."""
    for wait in RETRY_WAITS:
        frame = source.prices(tickers, start, end)
        if not frame.empty:
            return frame
        sleep(wait)
    frame = source.prices(tickers, start, end)
    if frame.empty:
        raise RateLimitedError("시세가 비어 있습니다")
    return frame


def update_prices(  # noqa: PLR0913 - one stage's inputs
    source: Source,
    root: Path,
    tickers: Sequence[str],
    last_day: date,
    report: Report,
    sleep: Callable[[float], None],
) -> pd.DataFrame:
    """New days for every ticker; full history for new or re-adjusted ones."""
    store = read_prices(root)
    first = last_day - timedelta(days=HISTORY_DAYS)
    start, full = price_plan(store, tickers, first)
    end = last_day + timedelta(days=1)
    recent = [t for t in tickers if t not in full]
    total = len(tickers)
    fresh: list[pd.DataFrame] = []
    if start is not None and recent:
        for i in range(0, len(recent), CHUNK):
            report("시세", i, total)
            fresh.append(_download(source, recent[i : i + CHUNK], start, end, sleep))
        changed = drifted(store, pd.concat(fresh, ignore_index=True))
        full += changed
        fresh = [
            cast("pd.DataFrame", frame[~frame["ticker"].isin(changed)])
            for frame in fresh
        ]
    for i in range(0, len(full), CHUNK):
        report("시세", len(recent) + i, total)
        fresh.append(_download(source, full[i : i + CHUNK], first, end, sleep))
    merged = merge_prices(store, fresh, full, tickers, last_day)
    _write(merged, root / PRICES_FILE)
    return merged


def update_rows(  # noqa: PLR0913 - one stage's inputs
    fetch: Callable[[str], Sequence[Mapping[str, object]]],
    path: Path,
    columns: Sequence[str],
    tickers: Sequence[str],
    stage: str,
    report: Report,
    sleep: Callable[[float], None],
) -> int:
    """Fetch the pending tickers into ``path``; returns the failures recorded.

    ``fetch`` gives a ticker's rows; none still leaves one, so it counts as
    fetched. The file is written even when Yahoo or SEC stops the run, so the
    next update picks up where this one ended. A file without some of
    ``columns`` is from an older version: its rows are marked failed, so all
    are fetched again.
    """
    existing = _read_csv(path, columns)
    missing = [column for column in columns if column not in existing.columns]
    if missing:
        existing["error"] = f"옛 형식 ({', '.join(missing)} 없음)"
    todo = pending(existing, tickers)
    rows: list[dict[str, object]] = []
    try:
        for i, ticker in enumerate(todo):
            report(stage, i, len(todo))
            found: list[dict[str, object]]
            try:
                parts = _retry(partial(fetch, ticker), sleep) or [{}]
                found = [{"ticker": ticker, **part, "error": ""} for part in parts]
            except (RateLimitedError, SecUnavailableError):
                raise
            except Exception as error:  # noqa: BLE001 - recorded per ticker
                found = [
                    {"ticker": ticker, "error": f"{type(error).__name__}: {error}"}
                ]
            rows.extend(found)
            sleep(PAUSE_SECONDS)
    finally:
        if rows or not path.exists():
            done = pd.DataFrame(rows, columns=list(columns))
            kept = cast(
                "pd.DataFrame", existing[~existing["ticker"].isin(list(done["ticker"]))]
            )
            frames = [frame for frame in (kept, done) if not frame.empty]
            merged = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=list(columns))
            )
            _write(merged, path)
    return sum(1 for row in rows if row["error"])


def cik_map(edgar: Filings, root: Path, today: date) -> dict[str, str]:
    """Ticker → CIK, from the stored list unless it is a month old."""
    if universe_due(root, today, CIKS_GLOB):
        _write(edgar.tickers(), root / f"ciks-{today.isoformat()}.csv")
    path = _newest(root, CIKS_GLOB)
    assert path is not None  # noqa: S101 - written just above
    frame = pd.read_csv(path, dtype="str")
    return dict(zip(frame["ticker"], frame["cik"], strict=True))


def update_margins(  # noqa: PLR0913 - one stage's inputs
    edgar: Filings,
    root: Path,
    today: date,
    tickers: Sequence[str],
    report: Report,
    sleep: Callable[[float], None],
) -> int:
    """SEC 10-K yearly history for ``tickers``; returns the failures recorded.

    A ticker missing from SEC's list or without companyfacts gets an empty
    row, so Yahoo's figures stand in for it.
    """
    ciks = cik_map(edgar, root, today)

    def fetch(ticker: str) -> list[dict[str, object]]:
        cik = ciks.get(ticker)
        facts = edgar.facts(cik) if cik else None
        return margin_history(facts) if facts is not None else []

    return update_rows(
        fetch,
        root / f"margins-{quarter(today)}.csv",
        MARGIN_COLUMNS,
        tickers,
        "SEC",
        report,
        sleep,
    )


def update(  # noqa: PLR0913 - the SEC source is optional
    source: Source,
    root: Path,
    now: datetime,
    report: Report,
    sleep: Callable[[float], None] = time.sleep,
    edgar: Filings | None = None,
) -> str:
    """Bring every stored part up to date; returns a one-line summary.

    SEC figures are fetched only with ``edgar``. Raises
    ``RateLimitedError`` when Yahoo keeps refusing after the retries. Only an
    update that gets to the end ``prune``s the superseded dated files.
    """
    today = now.date()
    report("유니버스", 0, 0)
    if universe_due(root, today):
        _write(source.constituents(), root / f"universe-{today.isoformat()}.csv")
    universe_path = _newest(root, UNIVERSE_GLOB)
    assert universe_path is not None  # noqa: S101 - written just above
    universe = read_universe(universe_path)
    tickers = [str(ticker) for ticker in universe["ticker"]]
    listed = set(tickers)
    prices = update_prices(source, root, tickers, last_session(now), report, sleep)
    failed = update_rows(
        lambda t: [source.fundamentals(t)],
        root / f"fundamentals-{quarter(today)}.csv",
        FUNDAMENTAL_COLUMNS,
        tickers,
        "재무",
        report,
        sleep,
    )
    failed += update_rows(
        lambda t: [source.estimates(t)],
        root / f"estimates-{today.isoformat()}.csv",
        ESTIMATE_COLUMNS,
        tickers,
        "추정치",
        report,
        sleep,
    )
    note = f" · 받지 못한 값 {failed}" if failed else ""
    if edgar is not None:
        try:
            missed = update_margins(edgar, root, today, tickers, report, sleep)
        except (httpx.HTTPError, SecUnavailableError) as error:
            note += f" · SEC 받지 못함 ({type(error).__name__}: {error})"
        else:
            note += f" · SEC 받지 못한 값 {missed}" if missed else ""
    prune(root)
    priced = len(set(prices["ticker"]) & listed)
    newest = cast("pd.Timestamp", prices["date"].max()).date() if priced else None
    return f"{len(tickers)}종목 · 시세 {newest or '—'} · 시세 받은 종목 {priced}{note}"


@final
class Updater:
    """Runs one update at a time in a background thread; pages poll ``progress``."""

    __slots__ = ("_edgar", "_lock", "_progress", "_source", "root")

    def __init__(
        self,
        root: Path,
        source: Callable[[Path], Source] = Yahoo,
        edgar: Callable[[str], Filings] = Sec,
    ) -> None:
        """Save under ``root``; ``source`` and ``edgar`` build the fetchers."""
        self.root = root
        self._source = source
        self._edgar = edgar
        self._lock = threading.Lock()
        self._progress = Progress()

    @property
    def progress(self) -> Progress:
        with self._lock:
            return self._progress

    def start(self, agent: str = "", criteria: Criteria = DEFAULT) -> bool:
        """Start an update unless one is running; SEC only with an ``agent``.

        A finished update saves a snapshot screened by ``criteria``.
        """
        with self._lock:
            if self._progress.running:
                return False
            self._progress = Progress(running=True, stage="준비")
        threading.Thread(
            target=self._run,
            args=(agent.strip(), criteria),
            name="finagent-screen",
            daemon=True,
        ).start()
        return True

    def _report(self, stage: str, done: int, total: int) -> None:
        with self._lock:
            self._progress = Progress(running=True, stage=stage, done=done, total=total)

    def _run(self, agent: str, criteria: Criteria) -> None:
        edgar: Filings | None = None
        try:
            edgar = self._edgar(agent) if agent else None
            now = datetime.now().astimezone()
            source = self._source(self.root)
            message = update(source, self.root, now, self._report, edgar=edgar)
            message += self._snapshot(criteria, now.date())
        except RateLimitedError:
            self._finish(
                "야후가 요청을 제한했습니다. 받은 만큼 저장했으니 잠시 뒤 다시"
                " 갱신하세요."
            )
        except Exception as error:  # noqa: BLE001 - shown on the page, not lost
            self._finish(f"갱신하지 못했습니다: {type(error).__name__}: {error}")
        else:
            self._finish(message, failed=False)
        finally:
            close = getattr(edgar, "close", None)
            if callable(close):
                _ = close()

    def _snapshot(self, criteria: Criteria, day: date) -> str:
        """Save the day's snapshot; a note when it could not be."""
        self._report("스냅샷", 0, 0)
        try:
            inputs = load(self.root)
            if inputs is not None:
                _ = write_snapshot(self.root, evaluate(inputs, criteria), criteria, day)
        except (OSError, ValueError, KeyError) as error:
            return f" · 스냅샷 저장 못 함 ({type(error).__name__}: {error})"
        return ""

    def _finish(self, message: str, *, failed: bool = True) -> None:
        with self._lock:
            self._progress = Progress(message=message, failed=failed)
