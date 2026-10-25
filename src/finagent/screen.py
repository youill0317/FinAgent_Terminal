"""S&P 500 and S&P 400 screening on yfinance data, kept under ``data/screen/``.

This module is an exception to the dashboard's data rules
(``tests/test_guardrails.py``): it imports ``yfinance`` (unofficial Yahoo
Finance data), reads the constituents lists from Wikipedia, asks SEC EDGAR
for 10-K operating margins when the user set ``SEC_USER_AGENT``, and writes
files, all under ``data/screen/`` in the folder the app runs from: S&P 500's
at its top, S&P 400's in ``sp400/``. yfinance's own caches go there too.
Nothing here runs until the page's 갱신 button starts an ``Updater``; the page
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
import math
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
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
HISTORY_DAYS: Final = 760  # first fetch and kept window: 2 years and slack
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
# Cutlines are the user's, without a backtest; shares are of the stocks with a
# value, ties at the cutline included.
MIN_BARS: Final = 250
MIN_PRICE: Final = 5.0
ILLIQUID_SHARE: Final = 0.10  # lowest 20-day dollar volume, left out
STRONG_SHARE: Final = 0.20  # top tech scores: 차트 강
WEAK_SHARE: Final = 0.40  # bottom tech scores: 차트 약
SPLIT: Final = 0.5  # 퀄리티 and 밸류에이션 percentiles split at the middle of the peers
QUALITY_MIN: Final = 2  # 퀄리티 measures needed for a 퀄리티 percentile
PEER_MIN: Final = 10  # members a GICS level needs, else the next wider one
REVENUE_FLOOR: Final = -0.05
NET_DEBT_FLOOR: Final = 3.0
NET_DEBT_MEDIAN_TIMES: Final = 2.0
EQUITY_MEDIAN_SHARE: Final = 0.5
PEAK_RATIO: Final = 0.6  # forward PER below trailing times this: 정점 이익
TTM_QUARTERS: Final = 4  # quarters in the TTM operating margin
MARGIN_YEARS: Final = 3  # fiscal years Yahoo's median operating margin needs
SEC_YEARS: Final = 5  # 10-K fiscal years in SEC's median, else Yahoo's
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
# 업황 순풍: TTM operating margin at least this times the multi-year median
# and at least this much above it.
TAILWIND_RATIO: Final = 1.5
TAILWIND_GAP: Final = 0.05
NEAR_HIGH: Final = 0.98  # close at least this share of the 52-week high
OBV_NEAR: Final = 0.05  # of the 60-day OBV range, below its high
TOP_N: Final = 20
FINANCIALS: Final = "Financials"
BANKS: Final = "Banks"
# GICS (March 2023 revision, checked against MSCI's GICS methodology of March
# 2023): industry group → industry → sub-industries, each in code order.
# Wikipedia gives the sector and sub-industry; peers widen through industry
# and group.
GICS_TREE: Final = {
    "Energy": {
        "Energy Equipment & Services": (
            "Oil & Gas Drilling",
            "Oil & Gas Equipment & Services",
        ),
        "Oil, Gas & Consumable Fuels": (
            "Integrated Oil & Gas",
            "Oil & Gas Exploration & Production",
            "Oil & Gas Refining & Marketing",
            "Oil & Gas Storage & Transportation",
            "Coal & Consumable Fuels",
        ),
    },
    "Materials": {
        "Chemicals": (
            "Commodity Chemicals",
            "Diversified Chemicals",
            "Fertilizers & Agricultural Chemicals",
            "Industrial Gases",
            "Specialty Chemicals",
        ),
        "Construction Materials": ("Construction Materials",),
        "Containers & Packaging": (
            "Metal, Glass & Plastic Containers",
            "Paper & Plastic Packaging Products & Materials",
        ),
        "Metals & Mining": (
            "Aluminum",
            "Diversified Metals & Mining",
            "Copper",
            "Gold",
            "Precious Metals & Minerals",
            "Silver",
            "Steel",
        ),
        "Paper & Forest Products": ("Forest Products", "Paper Products"),
    },
    "Capital Goods": {
        "Aerospace & Defense": ("Aerospace & Defense",),
        "Building Products": ("Building Products",),
        "Construction & Engineering": ("Construction & Engineering",),
        "Electrical Equipment": (
            "Electrical Components & Equipment",
            "Heavy Electrical Equipment",
        ),
        "Industrial Conglomerates": ("Industrial Conglomerates",),
        "Machinery": (
            "Construction Machinery & Heavy Transportation Equipment",
            "Agricultural & Farm Machinery",
            "Industrial Machinery & Supplies & Components",
        ),
        "Trading Companies & Distributors": ("Trading Companies & Distributors",),
    },
    "Commercial & Professional Services": {
        "Commercial Services & Supplies": (
            "Commercial Printing",
            "Environmental & Facilities Services",
            "Office Services & Supplies",
            "Diversified Support Services",
            "Security & Alarm Services",
        ),
        "Professional Services": (
            "Human Resource & Employment Services",
            "Research & Consulting Services",
            "Data Processing & Outsourced Services",
        ),
    },
    "Transportation": {
        "Air Freight & Logistics": ("Air Freight & Logistics",),
        "Passenger Airlines": ("Passenger Airlines",),
        "Marine Transportation": ("Marine Transportation",),
        "Ground Transportation": (
            "Rail Transportation",
            "Cargo Ground Transportation",
            "Passenger Ground Transportation",
        ),
        "Transportation Infrastructure": (
            "Airport Services",
            "Highways & Railtracks",
            "Marine Ports & Services",
        ),
    },
    "Automobiles & Components": {
        "Automobile Components": ("Automotive Parts & Equipment", "Tires & Rubber"),
        "Automobiles": ("Automobile Manufacturers", "Motorcycle Manufacturers"),
    },
    "Consumer Durables & Apparel": {
        "Household Durables": (
            "Consumer Electronics",
            "Home Furnishings",
            "Homebuilding",
            "Household Appliances",
            "Housewares & Specialties",
        ),
        "Leisure Products": ("Leisure Products",),
        "Textiles, Apparel & Luxury Goods": (
            "Apparel, Accessories & Luxury Goods",
            "Footwear",
            "Textiles",
        ),
    },
    "Consumer Services": {
        "Hotels, Restaurants & Leisure": (
            "Casinos & Gaming",
            "Hotels, Resorts & Cruise Lines",
            "Leisure Facilities",
            "Restaurants",
        ),
        "Diversified Consumer Services": (
            "Education Services",
            "Specialized Consumer Services",
        ),
    },
    "Consumer Discretionary Distribution & Retail": {
        "Distributors": ("Distributors",),
        "Broadline Retail": ("Broadline Retail",),
        "Specialty Retail": (
            "Apparel Retail",
            "Computer & Electronics Retail",
            "Home Improvement Retail",
            "Other Specialty Retail",
            "Automotive Retail",
            "Homefurnishing Retail",
        ),
    },
    "Consumer Staples Distribution & Retail": {
        "Consumer Staples Distribution & Retail": (
            "Drug Retail",
            "Food Distributors",
            "Food Retail",
            "Consumer Staples Merchandise Retail",
        ),
    },
    "Food, Beverage & Tobacco": {
        "Beverages": (
            "Brewers",
            "Distillers & Vintners",
            "Soft Drinks & Non-alcoholic Beverages",
        ),
        "Food Products": ("Agricultural Products & Services", "Packaged Foods & Meats"),
        "Tobacco": ("Tobacco",),
    },
    "Household & Personal Products": {
        "Household Products": ("Household Products",),
        "Personal Care Products": ("Personal Care Products",),
    },
    "Health Care Equipment & Services": {
        "Health Care Equipment & Supplies": (
            "Health Care Equipment",
            "Health Care Supplies",
        ),
        "Health Care Providers & Services": (
            "Health Care Distributors",
            "Health Care Services",
            "Health Care Facilities",
            "Managed Health Care",
        ),
        "Health Care Technology": ("Health Care Technology",),
    },
    "Pharmaceuticals, Biotechnology & Life Sciences": {
        "Biotechnology": ("Biotechnology",),
        "Pharmaceuticals": ("Pharmaceuticals",),
        "Life Sciences Tools & Services": ("Life Sciences Tools & Services",),
    },
    BANKS: {BANKS: ("Diversified Banks", "Regional Banks")},
    "Financial Services": {
        "Financial Services": (
            "Diversified Financial Services",
            "Multi-Sector Holdings",
            "Specialized Finance",
            "Commercial & Residential Mortgage Finance",
            "Transaction & Payment Processing Services",
        ),
        "Consumer Finance": ("Consumer Finance",),
        "Capital Markets": (
            "Asset Management & Custody Banks",
            "Investment Banking & Brokerage",
            "Diversified Capital Markets",
            "Financial Exchanges & Data",
        ),
        "Mortgage Real Estate Investment Trusts (REITs)": ("Mortgage REITs",),
    },
    "Insurance": {
        "Insurance": (
            "Insurance Brokers",
            "Life & Health Insurance",
            "Multi-line Insurance",
            "Property & Casualty Insurance",
            "Reinsurance",
        ),
    },
    "Software & Services": {
        "IT Services": (
            "IT Consulting & Other Services",
            "Internet Services & Infrastructure",
        ),
        "Software": ("Application Software", "Systems Software"),
    },
    "Technology Hardware & Equipment": {
        "Communications Equipment": ("Communications Equipment",),
        "Technology Hardware, Storage & Peripherals": (
            "Technology Hardware, Storage & Peripherals",
        ),
        "Electronic Equipment, Instruments & Components": (
            "Electronic Equipment & Instruments",
            "Electronic Components",
            "Electronic Manufacturing Services",
            "Technology Distributors",
        ),
    },
    "Semiconductors & Semiconductor Equipment": {
        "Semiconductors & Semiconductor Equipment": (
            "Semiconductor Materials & Equipment",
            "Semiconductors",
        ),
    },
    "Telecommunication Services": {
        "Diversified Telecommunication Services": (
            "Alternative Carriers",
            "Integrated Telecommunication Services",
        ),
        "Wireless Telecommunication Services": ("Wireless Telecommunication Services",),
    },
    "Media & Entertainment": {
        "Media": ("Advertising", "Broadcasting", "Cable & Satellite", "Publishing"),
        "Entertainment": ("Movies & Entertainment", "Interactive Home Entertainment"),
        "Interactive Media & Services": ("Interactive Media & Services",),
    },
    "Utilities": {
        "Electric Utilities": ("Electric Utilities",),
        "Gas Utilities": ("Gas Utilities",),
        "Multi-Utilities": ("Multi-Utilities",),
        "Water Utilities": ("Water Utilities",),
        "Independent Power and Renewable Electricity Producers": (
            "Independent Power Producers & Energy Traders",
            "Renewable Electricity",
        ),
    },
    "Equity Real Estate Investment Trusts (REITs)": {
        "Diversified REITs": ("Diversified REITs",),
        "Industrial REITs": ("Industrial REITs",),
        "Hotel & Resort REITs": ("Hotel & Resort REITs",),
        "Office REITs": ("Office REITs",),
        "Health Care REITs": ("Health Care REITs",),
        "Residential REITs": (
            "Multi-Family Residential REITs",
            "Single-Family Residential REITs",
        ),
        "Retail REITs": ("Retail REITs",),
        "Specialized REITs": (
            "Other Specialized REITs",
            "Self-Storage REITs",
            "Telecom Tower REITs",
            "Timber REITs",
            "Data Center REITs",
        ),
    },
    "Real Estate Management & Development": {
        "Real Estate Management & Development": (
            "Diversified Real Estate Activities",
            "Real Estate Operating Companies",
            "Real Estate Development",
            "Real Estate Services",
        ),
    },
}
# Sub-industry → (industry, industry group).
GICS: Final = {
    sub: (industry, group)
    for group, industries in GICS_TREE.items()
    for industry, subs in industries.items()
    for sub in subs
}
# GICS names the group "Financial Services" and one of its four industries
# too; the group is labelled apart so the two never share medians.
GROUP_LABELS: Final = {
    group: f"{group} 전체" if group in industries and len(industries) > 1 else group
    for group, industries in GICS_TREE.items()
}
BANK_TCE_FLOOR: Final = 0.04  # tangible common equity / tangible assets
CYCLICALS: Final = frozenset(
    {"Energy", "Materials", "Industrials", "Consumer Discretionary"}
)
SMA_DAYS: Final = (5, 20, 60, 120, 200)
YEAR_BARS: Final = 252
TECH_COLUMNS: Final = (
    "bars",
    "price",  # last close as traded, for the price floor and the table
    "dollar_vol20",
    *(f"sma{days}" for days in SMA_DAYS),
    "sma60_lag20",
    "close",  # last adjusted close, which the indicators use
    "rsi",
    "macd",
    "macd_signal",
    "pct_b",
    "atr",
    "stoch_k",
    "stoch_d",
    "obv",
    "obv_lag20",
    "obv_max60",
    "obv_min60",
    "obv_slope60",
    "vol20",
    "vol50",
    "up_vol60",
    "down_vol60",
    "high52",
    "from_high",
    "ret_1m",
    "ret_3m",
)

# Status words the page shows for a check that is not a plain pass or fail.
CLEAR: Final = "통과"
FAIL: Final = "결격"
UNKNOWN: Final = "미확인"  # no data
MEANINGLESS: Final = "해석 부적절"  # negative or zero base
NOT_COMPUTED: Final = "미산출"  # EBITDA ≤ 0
NET_CASH: Final = "순현금"
NOT_APPLICABLE: Final = "해당 없음"  # non-financial definition on a financial
UNUSED: Final = "미적용"  # a check turned off in the 기준 panel
UNSURE: Final = frozenset({UNKNOWN, MEANINGLESS})

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
# Fundamentals column → ``Ticker.info`` field. Ratios are fractions.
INFO_FIELDS: Final = {
    "fcf": "freeCashflow",
    "roe": "returnOnEquity",
    "book_value": "bookValue",
    "total_debt": "totalDebt",
    "total_cash": "totalCash",
    "ebitda": "ebitda",
    "fpe": "forwardPE",
    "tpe": "trailingPE",
    "mcap": "marketCap",
    "ev_ebitda": "enterpriseToEbitda",
    "short_pct": "shortPercentOfFloat",
}
FUNDAMENTAL_COLUMNS: Final = (
    "ticker",
    *INFO_FIELDS,
    "nd_ebitda",
    "eq_ratio",
    "tce_ratio",
    "revenue",
    "revenue_prior",
    "rev_growth",
    "rev_year",
    "margin_avg",
    "margin_years",
    "op_margin",
    "op_margin_quarters",
    "ttm_end",  # the end of the newest quarter in Yahoo's TTM
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
ESTIMATE_PERIODS: Final = {"fy0": "0y", "fy1": "+1y"}
TREND_COLUMNS: Final = {"now": "current", "30d": "30daysAgo", "90d": "90daysAgo"}
REVISION_COLUMNS: Final = {"up30": "upLast30days", "down30": "downLast30days"}
ESTIMATE_COLUMNS: Final = (
    "ticker",
    *(
        f"{period}_{key}"
        for period in ESTIMATE_PERIODS
        for key in (*TREND_COLUMNS, *REVISION_COLUMNS)
    ),
    "est_currency",
    "error",
)

TECH_CHECKS: Final = {
    "t_above200": "종가 > 200일선",
    "t_20over60": "20일선 > 60일선",
    "t_60rising": "60일선이 20거래일 전보다 높음",
    "t_above20": "종가 > 20일선",
    "m_macd_signal": "MACD > 시그널",
    "m_macd_positive": "MACD > 0",
    "m_rsi": "50 ≤ RSI ≤ 70",
    "v_obv20": "OBV가 20거래일 전보다 높음",
    "v_updown": "60일 오른 날 평균 거래량 > 내린 날",
    "v_obv_high": "52주 신고가면 OBV 60일 고점 근처, 아니면 OBV 60일 기울기 > 0",
}
OVERHEAT: Final = {
    "o_rsi": "RSI > 70",
    "o_stoch": "스토캐스틱 %K > 80, %K < %D",
    "o_bb": "볼린저 %B > 1",
}
# 퀄리티 measures: column → name. Each is a percentile within the peer group,
# higher is better; financials have only ROE and revenue growth.
QUALITY: Final = {
    "q_roe": "ROE",
    "q_margin": "영업이익률 5년 중앙값",
    "q_fcf": "FCF 마진",
    "q_growth": "매출 성장",
}
# 밸류에이션 measures: a percentile within the peer group, higher is pricier.
# Financials use PBR in place of EV/EBITDA.
VALUATION: Final = {
    "v_fpe": "선행 PER",
    "v_ev": "EV/EBITDA",
    "v_pb": "PBR",
}
# The four cells of 퀄리티 by 밸류에이션, then the ones that cannot be placed; ``{s}``
# is the split and ``{hold}`` the 보류 reasons in force (``group_help``).
GROUPS: Final = {
    "prime": "결격 없음 · 퀄리티 업종 백분위 {s} 이상 · 밸류에이션 업종 백분위"
    " {s} 미만",
    "pricey": "결격 없음 · 퀄리티 {s} 이상 · 밸류에이션 {s} 이상: 좋은 회사, 가격 확인",
    "trap": "결격 없음 · 퀄리티 {s} 미만 · 밸류에이션 {s} 미만: 싼 이유 점검",
    "weak": "결격 없음 · 퀄리티 {s} 미만 · 밸류에이션 {s} 이상",
    "hold": "결격 없음 · {hold}",
}
# Each group's order: column and whether it ascends.
GROUP_ORDER: Final = {
    "prime": (("edge", False), ("mcap", False)),
    "pricey": (("edge", False), ("mcap", False)),
    "trap": (("edge", False), ("mcap", False)),
    "weak": (("edge", False), ("mcap", False)),
    "hold": (("mcap", False),),
}
FORMATS: Final = {
    "pct": "{:.1%}",
    "signed": "{:+.1%}",
    "usd": "${:,.2f}",
    "big": "${:,.1f}{}",  # billions or millions, as ``fmt`` picks
    "x": "{:,.1f}x",
    "int": "{:,.0f}",
    "num": "{:,.2f}",
    "rank": "{:.0%}",
}
GROUP_NAMES: Final = {
    "prime": "검토 우선",
    "pricey": "밸류에이션 부담",
    "trap": "밸류 트랩 점검",
    "weak": "위험 우세",
    "hold": "보류",
}
# The 퀄리티-by-밸류에이션 map: rows of (row label, group per column).
MAP_ROWS: Final = (
    ("퀄리티 상위", ("prime", "pricey")),
    ("퀄리티 하위", ("trap", "weak")),
)
# FY1 consensus EPS direction, the map's colors.
EST_UP: Final = "상향"
EST_FLAT: Final = "보합"
EST_DOWN: Final = "하향"
EST_ORDER: Final = (EST_UP, EST_FLAT, EST_DOWN, UNKNOWN)
# The results table: column, label and ``fmt`` kind ("" for text).
TABLE: Final = (
    ("ticker", "티커", ""),
    ("name", "종목", ""),
    ("sector", "섹터", ""),
    ("industry", "산업", ""),
    ("price", "주가", "usd"),
    ("ret_1m", "1개월", "signed"),
    ("ret_3m", "3개월", "signed"),
    ("mcap", "시가총액", "big"),
    ("group", "칸", ""),
    ("quality", "퀄리티", "rank"),
    ("valuation", "밸류에이션", "rank"),
    ("est", "추정치", ""),
    ("weak_spot", "약한 고리", ""),
    ("tags", "라벨", ""),
    ("verdict", "결격", ""),
    ("score", "기술 점수", "int"),
    ("chart", "차트", ""),
    ("hot", "과열", ""),
    ("roe", "ROE", "pct"),
    ("op_margin", "영업이익률 (TTM)", "pct"),
    ("fpe", "선행 PER", "num"),
    ("ev_ebitda", "EV/EBITDA", "num"),
    ("rev_growth", "매출 성장 (연간)", "signed"),
    ("nd_ebitda", "순부채/EBITDA", "x"),
    ("fcf", "FCF", "big"),
)
# Measures only one side has; the other side's are 해당 없음.
FINANCIAL_ONLY: Final = frozenset({"v_pb"})
NON_FINANCIAL_ONLY: Final = frozenset({"q_margin", "q_fcf", "v_ev"})
OVERHEAT_RULES: Final = {
    "o_rsi": "RSI(14) > 70",
    "o_stoch": "스토캐스틱 %K > 80 그리고 %K < %D",
    "o_bb": "볼린저 %B > 1",
}
DQ_RULES: Final = {
    "dq_fcf": "FCF(금융업은 ROE·자기자본비율, 은행은 ROE·TCE/TA)",
    "dq_rev": "매출 성장",
    "dq_debt": "순부채/EBITDA(금융업 해당 없음)",
}
FROM_HIGH_HELP: Final = (
    f"**정의** 최근 종가 ÷ 최근 {YEAR_BARS}거래일 최고가(고가 기준) − 1\n\n"
    f"**해석** 0에 가까울수록 52주 고점 부근 · −{1 - NEAR_HIGH:.0%} 이내 신고가 근처"
)
PEERS_NOTE: Final = (
    f"업종: GICS 세부 산업, {PEER_MIN}종목 미만이면 산업 → 산업 그룹 → 섹터 순으로"
    " 확대 · 백분위: Hazen (순위 − 0.5) ÷ 종목 수, 동률은 평균 순위"
)
ESTIMATE_HELP: Final = (
    "**정의** 애널리스트 컨센서스 EPS · FY0 이번 회계연도 · FY1 다음 회계연도 ·"
    " 30일·90일 전 값 · 최근 30일 상향·하향 애널리스트 수\n\n"
    "**해석** FY1 현재 ≥ 90일 전 그리고 30일 하향 ≤ 상향이면 상향 · FY1 현재 < 90일 전"
    " 그리고 하향 > 상향이면 하향 · 그 밖 보합"
)

# ---------------------------------------------------------------- criteria


@dataclass(frozen=True, slots=True)
class Criteria:
    """The screening rules in force; the defaults are the constants above.

    ``off`` holds the ``RULES`` ids turned off, in ``RULES`` order.
    """

    min_bars: int = MIN_BARS
    min_price: float = MIN_PRICE
    illiquid_share: float = ILLIQUID_SHARE
    equity_median_share: float = EQUITY_MEDIAN_SHARE
    bank_tce_floor: float = BANK_TCE_FLOOR
    revenue_floor: float = REVENUE_FLOOR
    net_debt_median_times: float = NET_DEBT_MEDIAN_TIMES
    net_debt_floor: float = NET_DEBT_FLOOR
    quality_min: int = QUALITY_MIN
    split: float = SPLIT
    peak_ratio: float = PEAK_RATIO
    tailwind_ratio: float = TAILWIND_RATIO
    tailwind_gap: float = TAILWIND_GAP
    strong_share: float = STRONG_SHARE
    weak_share: float = WEAK_SHARE
    off: tuple[str, ...] = ()

    def on(self, key: str) -> bool:
        return key not in self.off

    def kept[V](self, items: Mapping[str, V]) -> dict[str, V]:
        """``items`` without the ids turned off."""
        return {key: value for key, value in items.items() if key not in self.off}

    def measures(self, names: Mapping[str, str], *, financial: bool) -> list[str]:
        """The names of the measures in force that a (non-)financial can have."""
        skipped = NON_FINANCIAL_ONLY if financial else FINANCIAL_ONLY
        return [name for key, name in self.kept(names).items() if key not in skipped]

    def need(self, *, financial: bool) -> int:
        """퀄리티 measures a stock needs: ``quality_min``, at most those in force."""
        return min(self.quality_min, len(self.measures(QUALITY, financial=financial)))


DEFAULT: Final = Criteria()


@dataclass(frozen=True, slots=True)
class Limit:
    """One number of a rule: its ``Criteria`` field and input range.

    ``low``, ``high`` and ``step`` are as the panel shows them: a percent is
    shown times 100.
    """

    field: str
    label: str
    low: float
    high: float
    step: float
    percent: bool = False

    @property
    def whole(self) -> bool:
        return isinstance(getattr(DEFAULT, self.field), int)

    def shown(self, value: float) -> float:
        return round(value * 100, 6) if self.percent else value

    def stored(self, value: float) -> float:
        return value / 100 if self.percent else value


@dataclass(frozen=True, slots=True)
class Rule:
    """A row of the 기준 panel.

    ``key`` is the id in ``Criteria.off``; a rule without ``toggle`` is a
    boundary that cannot be off. ``part`` heads a run of rows in a section.
    """

    key: str
    section: str
    name: str
    help: str
    limits: tuple[Limit, ...] = ()
    toggle: bool = True
    part: str = ""


def _check_help(kind: str, text: str) -> str:
    return f"**정의** {kind} 점검 · {text} · 통과 시 기술 점수 1점"


TECH_KINDS: Final = {"t": "추세", "m": "모멘텀", "v": "거래량"}
RULES: Final = (
    Rule(
        "min_bars",
        "거래 가능",
        "일봉 수",
        "**정의** 받은 일봉 수 하한 · 미만이면 제외\n\n"
        f"**해석** 상장 직후 종목 제외 · 기술 지표는 일봉 {MIN_BARS}개 이상에서"
        " 산출\n\n"
        f"**기준** 기본 {MIN_BARS}개 · 끄면 {MIN_BARS}개 미만 종목은 기술 지표 없음,"
        " 점검 모두 미통과",
        (Limit("min_bars", "일봉 (개)", MIN_BARS, 500, 10),),
    ),
    Rule(
        "min_price",
        "거래 가능",
        "주가",
        "**정의** 최근 종가(수정 전) 하한 · 미만이면 제외\n\n"
        "**해석** 호가 단위가 크고 변동이 심한 저가주 제외\n\n"
        f"**기준** 기본 ${MIN_PRICE:.0f}",
        (Limit("min_price", "주가 ($)", 0, 100, 1),),
    ),
    Rule(
        "illiquid",
        "거래 가능",
        "거래대금",
        "**정의** 20일 평균 거래대금 하위 비율 제외 · 일봉·주가 통과 종목 기준, 경계"
        " 동률 포함\n\n"
        "**해석** 사고팔기 어려운 종목 제외\n\n"
        f"**기준** 기본 하위 {ILLIQUID_SHARE:.0%}",
        (Limit("illiquid_share", "하위 (%)", 0, 50, 1, percent=True),),
    ),
    Rule(
        "dq_fcf",
        "결격 점검",
        "현금흐름·자본",
        "**정의** 일반: 잉여현금흐름(TTM) < 0 · 금융업: ROE < 0 또는 자기자본비율 <"
        " 업종 중앙값 × 비율 · 은행: ROE < 0 또는 TCE/TA < 하한 · 해당하면 결격\n\n"
        "**해석** 현금을 못 버는 회사, 자본 완충이 얇은 금융사 제외\n\n"
        f"**기준** 기본 중앙값의 {EQUITY_MEDIAN_SHARE:.0%} · TCE/TA"
        f" {BANK_TCE_FLOOR:.0%}",
        (
            Limit(
                "equity_median_share", "금융업 중앙값 대비 (%)", 0, 100, 5, percent=True
            ),
            Limit("bank_tce_floor", "은행 TCE/TA (%)", 0, 20, 0.5, percent=True),
        ),
    ),
    Rule(
        "dq_rev",
        "결격 점검",
        "매출 성장",
        "**정의** 연간 매출 성장률 ≤ 하한이면 결격 · 전년 매출 0 이하면 해석 부적절\n\n"
        "**해석** 사업이 줄어드는 회사 제외\n\n"
        f"**기준** 기본 {REVENUE_FLOOR:.0%}",
        (Limit("revenue_floor", "하한 (%)", -50, 50, 1, percent=True),),
    ),
    Rule(
        "dq_debt",
        "결격 점검",
        "순부채/EBITDA",
        "**정의** 순부채/EBITDA > max(업종 중앙값 × 배수, 하한)이면 결격 ·"
        " 순현금·EBITDA 0 이하는 통과 · 금융업 해당 없음\n\n"
        "**해석** 업종보다 빚이 무거운 회사 제외\n\n"
        f"**기준** 기본 중앙값 × {NET_DEBT_MEDIAN_TIMES:g} · 하한"
        f" {NET_DEBT_FLOOR:.1f}x",
        (
            Limit("net_debt_median_times", "중앙값 × (배)", 0.5, 10, 0.5),
            Limit("net_debt_floor", "하한 (x)", 0, 20, 0.5),
        ),
    ),
    Rule(
        "q_roe",
        "퀄리티",
        QUALITY["q_roe"],
        "**정의** 자기자본이익률(TTM) 업종 백분위 · 자기자본 0 이하 제외 · 금융업"
        " 포함\n\n**해석** 높을수록 자본 효율 우위",
    ),
    Rule(
        "q_margin",
        "퀄리티",
        QUALITY["q_margin"],
        f"**정의** 연간 영업이익률 중앙값 업종 백분위 · SEC 10-K {SEC_YEARS}년, 없으면"
        f" 야후 {MARGIN_YEARS}년 이상 · 금융업 해당 없음\n\n"
        "**해석** 높을수록 꾸준한 수익성",
    ),
    Rule(
        "q_fcf",
        "퀄리티",
        QUALITY["q_fcf"],
        "**정의** 잉여현금흐름(TTM) ÷ 연간 매출 업종 백분위 · 매출 0 이하 제외 · 금융업"
        " 해당 없음\n\n**해석** 높을수록 현금 창출력 우위",
    ),
    Rule(
        "q_growth",
        "퀄리티",
        QUALITY["q_growth"],
        "**정의** 연간 매출 성장률 업종 백분위 · 금융업 포함\n\n**해석** 높을수록 성장"
        " 우위",
    ),
    Rule(
        "quality_min",
        "퀄리티",
        "최소 항목 수",
        "**정의** 퀄리티 백분위 평균에 필요한 항목 수 · 미달이면 퀄리티 산출 불가,"
        " 보류\n\n"
        "**해석** 켠 항목이 더 적으면 켠 항목 수 · 금융업은 ROE·매출 성장 중 켠 수\n\n"
        f"**기준** 기본 {QUALITY_MIN}개",
        (Limit("quality_min", "항목 (개)", 1, len(QUALITY), 1),),
        toggle=False,
    ),
    Rule(
        "v_fpe",
        "밸류에이션",
        VALUATION["v_fpe"],
        "**정의** 선행 PER 업종 백분위 · 0 이하 제외 · 금융업 포함\n\n"
        "**해석** 높을수록 고평가",
    ),
    Rule(
        "v_ev",
        "밸류에이션",
        VALUATION["v_ev"],
        "**정의** EV/EBITDA(TTM) 업종 백분위 · 0 이하 제외 · 금융업은 PBR로 대체\n\n"
        "**해석** 높을수록 고평가",
    ),
    Rule(
        "v_pb",
        "밸류에이션",
        "PBR (금융업)",
        "**정의** 주가 ÷ 주당 장부가치 업종 백분위 · 금융업 전용, EV/EBITDA 대신 ·"
        " 장부가치 0 이하 제외\n\n**해석** 높을수록 고평가",
    ),
    Rule(
        "split",
        "칸 배정·라벨",
        "퀄리티·밸류에이션 경계",
        "**정의** 칸을 나누는 업종 백분위 · 퀄리티 ≥ 경계면 퀄리티 상위 ·"
        " 밸류에이션 < 경계면 저평가\n\n"
        "**해석** 높일수록 퀄리티 상위는 좁게, 저평가는 넓게\n\n"
        f"**기준** 기본 {SPLIT:.0%}",
        (Limit("split", "업종 백분위 (%)", 10, 90, 5, percent=True),),
        toggle=False,
    ),
    Rule(
        "peak",
        "칸 배정·라벨",
        "정점 이익",
        "**정의** 경기순환주(에너지·소재·산업재·임의소비재) 선행 PER < 후행 PER ×"
        " 비율이면 보류\n\n"
        "**해석** 이익 정점에서 싸 보이는 경기순환주 분리\n\n"
        f"**기준** 기본 {PEAK_RATIO:g}",
        (Limit("peak_ratio", "선행 ÷ 후행 PER", 0.1, 1, 0.05),),
    ),
    Rule(
        "tailwind",
        "칸 배정·라벨",
        "업황 순풍",
        f"**정의** TTM 영업이익률 ≥ {SEC_YEARS}년 중앙값 × 배수 그리고 ≥ 중앙값 +"
        " 차이면 보류 · 금융업 해당 없음\n\n"
        "**해석** 업황 호조로 일시적으로 부푼 이익 분리\n\n"
        f"**기준** 기본 × {TAILWIND_RATIO:g} · +{TAILWIND_GAP * 100:.0f}%p",
        (
            Limit("tailwind_ratio", "중앙값 × (배)", 1, 5, 0.1),
            Limit("tailwind_gap", "중앙값 + (%p)", 0, 30, 1, percent=True),
        ),
    ),
    *(
        Rule(
            key,
            "칸 배정·라벨",
            label,
            _check_help(TECH_KINDS[key[0]], label),
            part="기술 점수",
        )
        for key, label in TECH_CHECKS.items()
    ),
    Rule(
        "strong",
        "칸 배정·라벨",
        "차트 강",
        "**정의** 기술 점수 상위 비율 · 유동성 통과 종목 기준, 경계 동률 포함\n\n"
        f"**기준** 기본 상위 {STRONG_SHARE:.0%}",
        (Limit("strong_share", "상위 (%)", 1, 50, 1, percent=True),),
        part="기술 점수",
    ),
    Rule(
        "weak",
        "칸 배정·라벨",
        "차트 약",
        "**정의** 기술 점수 하위 비율 · 차트 강 제외 · 경계 동률 포함\n\n"
        f"**기준** 기본 하위 {WEAK_SHARE:.0%}",
        (Limit("weak_share", "하위 (%)", 1, 90, 1, percent=True),),
        part="기술 점수",
    ),
    *(
        Rule(
            key,
            "칸 배정·라벨",
            label,
            f"**정의** 단기 과열 신호 · {OVERHEAT_RULES[key]}\n\n"
            "**해석** 과열 표시에만 반영 · 기술 점수와 별개",
            part="과열",
        )
        for key, label in OVERHEAT.items()
    ),
)


def criteria(values: Mapping[str, object], off: Iterable[str]) -> Criteria:
    """Saved numbers and turned-off ids as ``Criteria``.

    A number of the wrong type or outside its input range keeps its default;
    an unknown id is dropped.
    """
    limits = {limit.field: limit for rule in RULES for limit in rule.limits}
    clean: dict[str, int | float] = {}
    for name, value in values.items():
        limit = limits.get(name)
        if limit is None or isinstance(value, bool):
            continue
        if not isinstance(value, int | float) or not math.isfinite(value):
            continue
        if not limit.low <= limit.shown(value) <= limit.high:
            continue
        if limit.whole:
            if not float(value).is_integer():
                continue
            clean[name] = int(value)
        else:
            clean[name] = float(value)
    gone = set(off)
    ids = tuple(rule.key for rule in RULES if rule.toggle and rule.key in gone)
    return replace(DEFAULT, **clean, off=ids)


def saved(c: Criteria) -> tuple[dict[str, int | float], tuple[str, ...]]:
    """The numbers that differ from the defaults, and the ids turned off."""
    values = {
        f.name: getattr(c, f.name)
        for f in fields(c)
        if f.name != "off" and getattr(c, f.name) != getattr(DEFAULT, f.name)
    }
    return values, c.off


def changed(c: Criteria) -> list[str]:
    """``RULES`` ids whose switch or numbers differ from the defaults."""
    return [
        rule.key
        for rule in RULES
        if c.on(rule.key) != DEFAULT.on(rule.key)
        or any(getattr(c, x.field) != getattr(DEFAULT, x.field) for x in rule.limits)
    ]


# ---------------------------------------------------------------- tooltips
# The stock dialog's cards and headings, and the table's columns, as the
# criteria in force set them.


def tech_basis(c: Criteria) -> str:
    return (
        f"**정의** 추세·모멘텀·거래량 점검 {len(c.kept(TECH_CHECKS))}개 중 통과 수\n\n"
        "**해석** 높을수록 추세·모멘텀·거래량 동반 우위 · ✓ 통과 · ✗ 미통과(값 없음"
        " 포함)\n\n"
        "**기준** 수정주가 일봉(시가·고가·저가에 수정종가/종가 비율 적용) · 이동평균은"
        " 단순 · RSI 14일 Wilder 평활 · MACD 12/26/9 EMA · 볼린저 20일 ±2σ(모표준편차)"
        " · 스토캐스틱 14/3/3 · ATR 14일 Wilder · OBV 60일 기울기는 최소제곱"
        f" · 52주 신고가 근처: 종가 ≥ {YEAR_BARS}일 고가의 {NEAR_HIGH:.0%} · OBV 고점"
        f" 근처: 60일 범위의 {OBV_NEAR:.0%} 이내"
    )


def chart_rules(c: Criteria) -> list[str]:
    """차트 강 and 약, where they are on."""
    rules = {
        "strong": f"차트 강: 유동성 통과 종목 중 점수 상위 {c.strong_share:.0%}",
        "weak": f"차트 약: 하위 {c.weak_share:.0%}(강 제외)",
    }
    return list(c.kept(rules).values())


def hold_rules(c: Criteria) -> list[str]:
    """The 보류 labels in force."""
    return list(c.kept({"peak": "정점 이익", "tailwind": "업황 순풍"}).values())


def score_help(c: Criteria) -> str:
    rules = "".join(f" · {rule}" for rule in chart_rules(c))
    return (
        f"**정의** 기술 점검 {len(c.kept(TECH_CHECKS))}개 중 통과 수{rules}\n\n"
        "**해석** 높을수록 추세·모멘텀·거래량 우위"
    )


def _listed(names: Sequence[str]) -> str:
    return " · ".join(names) if names else "없음"


def _needs(c: Criteria) -> str:
    plain, financial = c.need(financial=False), c.need(financial=True)
    return f"{plain}개" if plain == financial else f"{plain}개(금융업 {financial}개)"


def quality_help(c: Criteria) -> str:
    financial = "·".join(c.measures(QUALITY, financial=True)) or "없음"
    return (
        "**정의** 퀄리티 항목 업종 내 백분위의 평균 ·"
        f" {_listed(c.measures(QUALITY, financial=False))} · 금융업은 {financial}\n\n"
        f"**해석** 높을수록 우수 · {c.split:.0%} 이상 퀄리티 상위\n\n"
        f"**기준** 항목 {_needs(c)} 이상 · {PEERS_NOTE}"
    )


def valuation_help(c: Criteria) -> str:
    financial = "·".join(c.measures(VALUATION, financial=True)) or "없음"
    return (
        "**정의** 밸류에이션 항목 업종 내 백분위의 평균 ·"
        f" {_listed(c.measures(VALUATION, financial=False))} · 금융업은 {financial}\n\n"
        f"**해석** 높을수록 고평가 · {c.split:.0%} 미만 저평가\n\n"
        f"**기준** 0 이하 배수 제외 · {PEERS_NOTE}"
    )


def labels_help(c: Criteria) -> str:
    hold = "·".join(hold_rules(c))
    return (
        "**정의** 퀄리티·밸류에이션 항목별 업종 백분위와 보류 사유 점검\n\n"
        "**해석** 퀄리티 높을수록 우수 · 밸류에이션 높을수록 고평가 ·"
        f" 경계 {c.split:.0%}"
        + (f" · {hold}이면 보류" if hold else "")
        + f"\n\n**기준** {PEERS_NOTE}"
    )


def overheat_help(c: Criteria) -> str:
    flags = list(c.kept(OVERHEAT_RULES).values())
    return (
        f"**정의** 단기 과열 신호 {len(flags)}개"
        + "".join(f" · {flag}" for flag in flags)
        + "\n\n**해석** ✓ 많을수록 단기 과열 · 기술 점수와 별개"
    )


def dq_help(c: Criteria) -> str:
    checks = list(c.kept(DQ_RULES).values())
    return (
        f"**정의** 결격 점검 {len(checks)}개"
        + "".join(f" · {check}" for check in checks)
        + "\n\n**해석** 결격·미확인·해석 부적절 1개 이상이면 칸 미배정 ·"
        " 순현금·미산출은 통과"
    )


def column_help(c: Criteria) -> dict[str, str]:
    """The results table's column tooltips."""
    tags = c.kept(
        {
            "peak": f"정점 이익(경기순환주 선행 PER < 후행 PER × {c.peak_ratio:g})",
            "tailwind": "업황 순풍",
        }
    )
    return {
        "group": "**정의** 퀄리티 × 밸류에이션 업종 백분위 칸 · 보류 ·"
        " 결격·미확인은 빈칸",
        "quality": quality_help(c),
        "valuation": valuation_help(c),
        "est": ESTIMATE_HELP,
        "weak_spot": "**정의** 가장 불리한 항목 · 퀄리티는 가장 낮은 백분위,"
        " 밸류에이션은 가장 높은 백분위\n\n"
        f"**해석** (비쌈): 밸류에이션 백분위 {c.split:.0%} 이상",
        "tags": "**정의** "
        + " · ".join([*tags.values(), "퀄리티·밸류에이션 산출 불가"]),
        "verdict": dq_help(c),
        "score": score_help(c),
        "chart": score_help(c),
        "hot": overheat_help(c),
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

    def fundamentals(self, ticker: str, *, financial: bool) -> Mapping[str, object]:
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


def fundamentals_row(
    info: Mapping[str, object],
    income: pd.DataFrame,
    balance: pd.DataFrame | None = None,
    quarterly: pd.DataFrame | None = None,
) -> dict[str, object]:
    """``info`` numbers, annual revenue growth and, for financials, equity/assets.

    ``income`` is the yearly income statement, newest year first. Yahoo keeps
    only five quarters, too few for a TTM year over year, so revenue growth is
    the last fiscal year over the one before. ``margin_avg`` (an older name,
    kept for the stored files) is the median operating margin of the fiscal
    years with both values and revenue above zero, ``margin_years`` how many
    there were. ``op_margin`` is the TTM
    operating margin from ``quarterly`` (the quarterly income statement):
    OperatingIncome over TotalRevenue summed across the latest four quarters,
    NaN unless all four have both values and the revenue is above zero
    (``info["operatingMargins"]`` is only the latest quarter).
    """
    row: dict[str, object] = {
        column: _number(info.get(field)) for column, field in INFO_FIELDS.items()
    }
    debt, cash, ebitda = row["total_debt"], row["total_cash"], row["ebitda"]
    row["nd_ebitda"] = (
        (cast("float", debt) - cast("float", cash)) / cast("float", ebitda)
        if cast("float", ebitda) > 0
        else math.nan
    )
    revenue = _row_value(income, "TotalRevenue", 0)
    prior = _row_value(income, "TotalRevenue", 1)
    row["revenue"] = revenue
    row["revenue_prior"] = prior
    row["rev_growth"] = revenue / prior - 1 if prior > 0 else math.nan
    row["rev_year"] = (
        str(cast("pd.Timestamp", income.columns[0]).date()) if not income.empty else ""
    )
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
    row["op_margin"], row["op_margin_quarters"], row["ttm_end"] = ttm_margin(quarterly)
    row["eq_ratio"], row["tce_ratio"] = (
        capital_ratios(balance) if balance is not None else (math.nan, math.nan)
    )
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


def capital_ratios(balance: pd.DataFrame) -> tuple[float, float]:
    """Equity / assets, and tangible common equity / tangible assets.

    TCE is Yahoo's TangibleBookValue (common equity less goodwill and other
    intangibles, so preferred stock is out); without it, CommonStockEquity less
    GoodwillAndOtherIntangibleAssets. Tangible assets are TotalAssets less the
    same intangibles. Any missing input leaves the ratio NaN.
    """
    assets = _row_value(balance, "TotalAssets")
    equity = _row_value(balance, "StockholdersEquity")
    intangibles = _row_value(balance, "GoodwillAndOtherIntangibleAssets")
    tce = _row_value(balance, "TangibleBookValue")
    if math.isnan(tce):
        tce = _row_value(balance, "CommonStockEquity") - intangibles
    tangible_assets = assets - intangibles
    return (
        equity / assets if assets > 0 else math.nan,
        tce / tangible_assets if tangible_assets > 0 else math.nan,
    )


def estimates_row(trend: pd.DataFrame, revisions: pd.DataFrame) -> dict[str, object]:
    """FY0/FY1 EPS consensus now, 30 and 90 days ago, and 30-day revisions."""
    row: dict[str, object] = {}
    for period, yahoo in ESTIMATE_PERIODS.items():
        for key, column in TREND_COLUMNS.items():
            row[f"{period}_{key}"] = _cell(trend, yahoo, column)
        for key, column in REVISION_COLUMNS.items():
            row[f"{period}_{key}"] = _cell(revisions, yahoo, column)
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


def sec_window(ends: Sequence[str]) -> list[str]:
    """The newest ``SEC_YEARS`` year ends within that many years of the newest."""
    newest = sorted(ends, reverse=True)
    if not newest:
        return []
    cutoff = date.fromisoformat(newest[0]) - timedelta(days=365 * SEC_YEARS - 180)
    return [end for end in newest if date.fromisoformat(end) > cutoff][:SEC_YEARS]


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


def sec_margins(margins: pd.DataFrame) -> pd.DataFrame:
    """Per ticker, the median operating margin over ``sec_window``, its years and TTM.

    A year counts with operating income and revenue above zero; ``sec_years``
    below ``SEC_YEARS`` means the window was not full. ``sec_ttm`` is
    ``sec_ttm`` of the quarter rows, ``sec_ttm_end`` the newest of them.
    """
    if not set(MARGIN_COLUMNS) <= set(margins.columns):  # an older file
        margins = pd.DataFrame(columns=list(MARGIN_COLUMNS))
    rows: list[dict[str, object]] = []
    for ticker, group in margins.dropna(subset=["end"]).groupby("ticker"):
        years = cast("pd.DataFrame", group[group["period"] == "FY"])
        quarters = cast("pd.DataFrame", group[group["period"] == "Q"])
        ends = cast("pd.Series", years["end"]).astype("str")
        used = years[
            ends.isin(sec_window(list(ends)))
            & (years["revenue"] > 0)
            & years["op_income"].notna()
        ]
        rows.append(
            {
                "ticker": ticker,
                "sec_avg": float(
                    cast("pd.Series", used["op_income"] / used["revenue"]).median()
                ),
                "sec_years": len(used),
                "sec_ttm": sec_ttm(quarters, max(ends, default="")),
                "sec_ttm_end": quarters["end"].max(),
            }
        )
    columns = ["ticker", "sec_avg", "sec_years", "sec_ttm", "sec_ttm_end"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------- indicators
# Every indicator takes adjusted daily series (``adjusted``).


def sma(values: pd.Series, days: int) -> pd.Series:
    return cast("pd.Series", values.rolling(days).mean())


def ema(values: pd.Series, span: int) -> pd.Series:
    return cast("pd.Series", values.ewm(span=span, adjust=False).mean())


def rsi(close: pd.Series, days: int = 14) -> pd.Series:
    """Wilder's RSI: gains and losses smoothed with alpha = 1/days."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / days, adjust=False, min_periods=days)
    loss = (-change).clip(lower=0).ewm(alpha=1 / days, adjust=False, min_periods=days)
    up, down = cast("pd.Series", gain.mean()), cast("pd.Series", loss.mean())
    value = cast("pd.Series", 100 - 100 / (1 + up / down))  # pyright: ignore[reportInvalidCast]
    return value.where(down > 0, 100.0).where(up.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """MACD line (EMA fast minus EMA slow) and its EMA signal line."""
    line = ema(close, fast) - ema(close, slow)
    return line, ema(line, signal)


def percent_b(close: pd.Series, days: int = 20, width: float = 2.0) -> pd.Series:
    """Bollinger %B with a population standard deviation."""
    middle = sma(close, days)
    spread = close.rolling(days).std(ddof=0) * width
    return cast("pd.Series", (close - (middle - spread)) / (2 * spread))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, days: int = 14) -> pd.Series:
    """Wilder's average true range."""
    previous = close.shift()
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    smoothed = true_range.ewm(alpha=1 / days, adjust=False, min_periods=days)
    return cast("pd.Series", smoothed.mean())


def stochastic(  # noqa: PLR0913 - the three periods have defaults
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    days: int = 14,
    smooth: int = 3,
    signal: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Slow stochastic %K (SMA of the raw %K) and %D (SMA of %K)."""
    lowest = low.rolling(days).min()
    raw = cast(
        "pd.Series", 100 * (close - lowest) / (high.rolling(days).max() - lowest)
    )
    k = sma(raw, smooth)
    return k, sma(k, signal)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume, starting at 0."""
    direction = pd.Series(np.sign(close.diff()), index=close.index).fillna(0)
    return (direction * volume).cumsum()


def slope(values: pd.Series) -> float:
    """Least-squares slope per bar."""
    y = values.to_numpy(dtype="float64")
    x = np.arange(len(y), dtype="float64")
    x -= x.mean()
    return float(x @ (y - y.mean()) / (x @ x))


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
    """One ticker's indicators on its last bar; ``frame`` is sorted by date.

    Fewer than ``MIN_BARS`` bars give only the liquidity inputs.
    """
    bars = len(frame)
    raw = cast("pd.Series", frame["close"])
    volume = cast("pd.Series", frame["volume"]).astype("float64")
    row = {
        "bars": float(bars),
        "price": float(raw.iloc[-1]),
        "dollar_vol20": float((raw * volume).tail(20).mean()),
    }
    if bars < MIN_BARS:
        return row
    adj = adjusted(frame)
    close = cast("pd.Series", adj["close"])
    high = cast("pd.Series", adj["high"])
    low = cast("pd.Series", adj["low"])
    for days in SMA_DAYS:
        row[f"sma{days}"] = float(sma(close, days).iloc[-1])
    row["sma60_lag20"] = float(sma(close, 60).iloc[-21])
    line, signal = macd(close)
    k, d = stochastic(high, low, close)
    balance = obv(close, volume)
    recent = balance.tail(60)
    change = close.diff().tail(60)
    last60 = volume.tail(60)
    high52 = float(high.tail(YEAR_BARS).max())
    row |= {
        "close": float(close.iloc[-1]),
        "rsi": float(rsi(close).iloc[-1]),
        "macd": float(line.iloc[-1]),
        "macd_signal": float(signal.iloc[-1]),
        "pct_b": float(percent_b(close).iloc[-1]),
        "atr": float(atr(high, low, close).iloc[-1]),
        "stoch_k": float(k.iloc[-1]),
        "stoch_d": float(d.iloc[-1]),
        "obv": float(balance.iloc[-1]),
        "obv_lag20": float(balance.iloc[-21]),
        "obv_max60": float(recent.max()),
        "obv_min60": float(recent.min()),
        "obv_slope60": slope(recent),
        "vol20": float(volume.tail(20).mean()),
        "vol50": float(volume.tail(50).mean()),
        "up_vol60": float(last60[change > 0].mean()),
        "down_vol60": float(last60[change < 0].mean()),
        "high52": high52,
        "from_high": float(close.iloc[-1]) / high52 - 1,
        "ret_1m": float(close.iloc[-1] / close.iloc[-22] - 1),
        "ret_3m": float(close.iloc[-1] / close.iloc[-64] - 1),
    }
    return row


def technicals(prices: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"ticker": ticker, **tech_row(group.sort_values("date"))}
        for ticker, group in prices.groupby("ticker", sort=False)
    ]
    return pd.DataFrame(rows, columns=["ticker", *TECH_COLUMNS])


# ---------------------------------------------------------------- rules


def top_share(values: pd.Series, share: float) -> pd.Series:
    """The highest ``share`` of the values present, ties at the cutline included."""
    count = math.ceil(share * int(values.notna().sum()))
    return values.rank(method="min", ascending=False) <= count


def bottom_share(values: pd.Series, share: float) -> pd.Series:
    """The lowest ``share`` of the values present, ties at the cutline included."""
    count = math.ceil(share * int(values.notna().sum()))
    return values.rank(method="min", ascending=True) <= count


def financial_group(table: pd.DataFrame) -> pd.Series:
    """The GICS industry group of each Financials stock, NaN for the rest."""
    groups = {sub: group for sub, (_, group) in GICS.items()}
    group = cast("pd.Series", table["industry"]).map(groups)
    return group.where(table["sector"] == FINANCIALS)


def gics_levels(table: pd.DataFrame) -> pd.DataFrame:
    """Each stock's sub-industry, industry, group (``GROUP_LABELS``) and sector.

    A sub-industry missing from ``GICS`` has no industry or group.
    """
    industry = cast("pd.Series", table["industry"])
    return pd.DataFrame(
        {
            "sub": industry,
            "industry": industry.map({sub: wide for sub, (wide, _) in GICS.items()}),
            "group": industry.map(
                {sub: GROUP_LABELS[group] for sub, (_, group) in GICS.items()}
            ),
            "sector": table["sector"],
        },
        index=table.index,
    )


def peers(table: pd.DataFrame) -> pd.Series:
    """The narrowest GICS level with ``PEER_MIN`` members here, else the sector."""
    levels = gics_levels(table)
    peer = cast("pd.Series", levels["sector"]).copy()
    for level in ("group", "industry", "sub"):
        labels = cast("pd.Series", levels[level])
        peer = labels.where(labels.map(labels.value_counts()) >= PEER_MIN, peer)
    return peer


def within_peers(
    values: pd.Series, table: pd.DataFrame, how: str, **options: object
) -> pd.Series:
    """``values`` grouped by ``how`` over each stock's whole peer group.

    The group is every stock carrying the ``peer`` label at any GICS level, so
    a stock compared at its industry meets the members its sub-industry kept.
    """
    labels = gics_levels(table).assign(peer=table["peer"])
    pairs = (
        pd.concat([cast("pd.Series", labels[column]) for column in labels.columns])
        .dropna()
        .rename_axis("row")
        .rename("label")
        .reset_index()
        .drop_duplicates()
    )
    pairs["value"] = values.reindex(pairs["row"]).to_numpy()
    pairs["stat"] = pairs.groupby("label")["value"].transform(how, **options)
    own = table["peer"].reindex(pairs["row"]).to_numpy() == pairs["label"].to_numpy()
    return cast("pd.Series", pairs[own].set_index("row")["stat"]).reindex(table.index)


def percentile(values: pd.Series, table: pd.DataFrame) -> pd.Series:
    """Hazen percentile in each stock's peer group: (rank - 0.5) / n, ties averaged.

    The peer median lands on 50%, so ``SPLIT`` halves the group.
    """
    rank = within_peers(values, table, "rank")
    return (rank - 0.5) / within_peers(values, table, "count")


def liquidity(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.Series:
    """Why a stock is left out ("" when it stays): history, price, then volume.

    No prices at all always leaves a stock out; the three filters only when on.
    """
    reason = pd.Series("", index=table.index, dtype="str")
    reason[table["bars"].isna()] = "시세 없음"
    if c.on("min_bars"):
        short = (reason == "") & ~(table["bars"] >= c.min_bars)
        reason[short] = f"일봉 {c.min_bars}개 미만"
    if c.on("min_price"):
        cheap = (reason == "") & ~(table["price"] >= c.min_price)
        reason[cheap] = f"주가 ${c.min_price:g} 미만"
    if c.on("illiquid"):
        rest = reason == ""
        volume = cast("pd.Series", table["dollar_vol20"]).where(rest)
        thin = rest & bottom_share(volume, c.illiquid_share)
        reason[thin] = f"거래대금 하위 {c.illiquid_share:.0%}"
    return reason


def tech_checks(table: pd.DataFrame) -> pd.DataFrame:
    """The ten score checks; a missing value fails its check."""
    t = table
    near_high = t["close"] >= NEAR_HIGH * t["high52"]
    near_obv_high = t["obv"] >= t["obv_max60"] - OBV_NEAR * (
        t["obv_max60"] - t["obv_min60"]
    )
    checks = {
        "t_above200": t["close"] > t["sma200"],
        "t_20over60": t["sma20"] > t["sma60"],
        "t_60rising": t["sma60"] > t["sma60_lag20"],
        "t_above20": t["close"] > t["sma20"],
        "m_macd_signal": t["macd"] > t["macd_signal"],
        "m_macd_positive": t["macd"] > 0,
        "m_rsi": t["rsi"].between(50, 70),
        "v_obv20": t["obv"] > t["obv_lag20"],
        "v_updown": t["up_vol60"] > t["down_vol60"].fillna(0),
        "v_obv_high": near_obv_high.where(near_high, t["obv_slope60"] > 0),
    }
    return pd.DataFrame(checks, index=t.index).fillna(value=False).astype(bool)


def overheat(table: pd.DataFrame) -> pd.DataFrame:
    t = table
    flags = {
        "o_rsi": t["rsi"] > 70,  # noqa: PLR2004
        "o_stoch": (t["stoch_k"] > 80) & (t["stoch_k"] < t["stoch_d"]),  # noqa: PLR2004
        "o_bb": t["pct_b"] > 1,
    }
    return pd.DataFrame(flags, index=t.index)


def fundamentals_status(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.DataFrame:
    """Disqualifiers, their peer medians, 정점 이익 and 업황 순풍, for liquid stocks.

    A check whose input is missing is 미확인, and one with a negative or zero
    base is 해석 부적절; either withholds the stock from every cell. A check
    turned off is 미적용 and a label turned off never set.
    """
    t = table
    financial = t["sector"] == FINANCIALS
    out = pd.DataFrame(index=t.index)

    bank = financial_group(t) == BANKS
    other = financial & ~bank
    eq_ratio = cast("pd.Series", t["eq_ratio"]).where(other)
    out["eq_med"] = within_peers(eq_ratio, t, "median")
    out["eq_limit"] = out["eq_med"] * c.equity_median_share
    other_fail = (t["roe"] < 0) | (eq_ratio < out["eq_limit"])
    other_unknown = t["roe"].isna() | eq_ratio.isna() | out["eq_limit"].isna()
    bank_fail = (t["roe"] < 0) | (t["tce_ratio"] < c.bank_tce_floor)
    bank_unknown = t["roe"].isna() | t["tce_ratio"].isna()
    out["dq_fcf"] = np.select(
        [
            bank & bank_fail,
            bank & bank_unknown,
            bank,
            other & other_fail,
            other & other_unknown,
            other,
            t["fcf"].isna(),
            t["fcf"] < 0,
        ],
        [FAIL, UNKNOWN, CLEAR, FAIL, UNKNOWN, CLEAR, UNKNOWN, FAIL],
        CLEAR,
    )

    growth = t["rev_growth"]
    out["dq_rev"] = np.select(
        [t["revenue_prior"] <= 0, growth.isna(), growth <= c.revenue_floor],
        [MEANINGLESS, UNKNOWN, FAIL],
        CLEAR,
    )

    net_debt = t["total_debt"] - t["total_cash"]
    nd = cast("pd.Series", t["nd_ebitda"]).where(~financial)
    out["nd_med"] = within_peers(nd, t, "median")
    out["nd_limit"] = np.fmax(out["nd_med"] * c.net_debt_median_times, c.net_debt_floor)
    out["dq_debt"] = np.select(
        [
            financial,
            net_debt.isna(),
            net_debt <= 0,
            t["ebitda"].isna(),
            t["ebitda"] <= 0,
            nd > out["nd_limit"],
        ],
        [NOT_APPLICABLE, UNKNOWN, NET_CASH, UNKNOWN, NOT_COMPUTED, FAIL],
        CLEAR,
    )

    for key in DQ_RULES:
        if not c.on(key):
            out[key] = UNUSED
    checks = out[list(DQ_RULES)]
    out["dq_state"] = np.select(
        [(checks == FAIL).any(axis=1), checks.isin(list(UNSURE)).any(axis=1)],
        [FAIL, UNKNOWN],
        CLEAR,
    )

    fpe = cast("pd.Series", t["fpe"]).where(t["fpe"] > 0)
    out["peak"] = (
        t["sector"].isin(list(CYCLICALS))
        & (t["tpe"] > 0)
        & (fpe < t["tpe"] * c.peak_ratio)
        & c.on("peak")
    )
    median = cast("pd.Series", t["margin_avg"]).where(
        ~financial & (t["margin_years"] >= MARGIN_YEARS) & (t["margin_avg"] > 0)
    )
    out["tailwind"] = (
        (t["op_margin"] >= median * c.tailwind_ratio)
        & (t["op_margin"] >= median + c.tailwind_gap)
        & c.on("tailwind")
    )
    return out


def axes(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.DataFrame:
    """퀄리티 and 밸류에이션: each measure's peer percentile, then their mean.

    A measure needs a positive base: ROE positive equity, FCF margin positive
    revenue, a multiple a value above zero; the margin median needs
    ``MARGIN_YEARS`` fiscal years. A measure turned off is NaN. 퀄리티 needs
    ``Criteria.need`` measures and 밸류에이션 one, else it is NaN. ``edge`` is 퀄리티
    less 밸류에이션.
    """
    t = table
    financial = t["sector"] == FINANCIALS
    plain = ~financial
    equity = ~(t["book_value"] <= 0)
    measures = {
        "q_roe": t["roe"].where(equity),
        "q_margin": t["margin_avg"].where(plain & (t["margin_years"] >= MARGIN_YEARS)),
        "q_fcf": (t["fcf"] / t["revenue"]).where(plain & (t["revenue"] > 0)),
        "q_growth": t["rev_growth"],
        "v_fpe": t["fpe"].where(t["fpe"] > 0),
        "v_ev": t["ev_ebitda"].where(plain & (t["ev_ebitda"] > 0)),
        "v_pb": (t["price"] / t["book_value"]).where(financial & (t["book_value"] > 0)),
    }
    out = pd.DataFrame(
        {
            key: percentile(cast("pd.Series", values), t)
            if c.on(key)
            else pd.Series(math.nan, index=t.index)
            for key, values in measures.items()
        },
        index=t.index,
    )
    quality = cast("pd.DataFrame", out[list(QUALITY)])
    need = np.where(financial, c.need(financial=True), c.need(financial=False))
    out["quality"] = quality.mean(axis=1).where(quality.count(axis=1) >= need)
    out["valuation"] = out[list(VALUATION)].mean(axis=1)
    out["edge"] = out["quality"] - out["valuation"]
    return out


def estimate_direction(table: pd.DataFrame) -> pd.Series:
    """FY1 consensus EPS against 90 days ago, with the 30-day revision counts."""
    t = table
    now, before = t["fy1_now"], t["fy1_90d"]
    up, down = t["fy1_up30"], t["fy1_down30"]
    known = now.notna() & before.notna() & up.notna() & down.notna()
    direction = np.select(
        [~known, (now >= before) & (down <= up), (now < before) & (down > up)],
        [UNKNOWN, EST_UP, EST_DOWN],
        EST_FLAT,
    )
    return pd.Series(direction, index=t.index, dtype="str")


def groups(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.Series:
    """Each stock's cell of 퀄리티 by 밸류에이션, "hold", or "" if 결격/미확인."""
    t = table
    placed = t["quality"].notna() & t["valuation"].notna()
    hold = t["peak"].eq(other=True) | t["tailwind"].eq(other=True) | ~placed
    good = t["quality"] >= c.split
    cheap = t["valuation"] < c.split
    key = np.select(
        [t["dq_state"] != CLEAR, hold, good & cheap, good, cheap],
        ["", "hold", "prime", "pricey", "trap"],
        "weak",
    )
    return pd.Series(key, index=t.index, dtype="str")


def evaluate(inputs: Inputs, criteria: Criteria = DEFAULT) -> pd.DataFrame:
    """Each constituent with liquidity, score, checks, 퀄리티, 밸류에이션, cell."""
    return judge(prepare(inputs), criteria)


def prepare(inputs: Inputs) -> pd.DataFrame:
    """The stored parts as one row per constituent with its indicators.

    Nothing here depends on the criteria. The operating margin median
    (``margin_avg``) is SEC's ``SEC_YEARS`` 10-K years where they are all
    there, else Yahoo's (``margin_source``). A stock
    on SEC's median takes SEC's TTM too when its four quarters are there and
    end no earlier than Yahoo's newest quarter (``STALE_SLACK_DAYS`` aside),
    else Yahoo's (``op_margin_source``); ``op_margin_end`` is the end of the
    used TTM's newest quarter.
    """
    table = inputs.universe.merge(technicals(inputs.prices), on="ticker", how="left")
    table = table.merge(inputs.fundamentals, on="ticker", how="left")
    table = table.merge(
        inputs.estimates, on="ticker", how="left", suffixes=("", "_estimates")
    )
    for column in (*FUNDAMENTAL_COLUMNS, *ESTIMATE_COLUMNS):
        if column not in table.columns:
            table[column] = math.nan
    for column in ("currency", "est_currency", "rev_year", "ttm_end"):
        table[column] = table[column].fillna("").astype("str")
    table = table.merge(sec_margins(inputs.margins), on="ticker", how="left")
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
    table["error"] = _errors(table)
    return table.drop(columns=["error_estimates"], errors="ignore")


def judge(prepared: pd.DataFrame, c: Criteria) -> pd.DataFrame:
    """``prepare``'s table screened by ``c``: liquidity, score, checks and cell.

    The score counts the checks in force; with none, no stock is 강 or 약.
    """
    table = prepared.copy()
    table["liquidity"] = liquidity(table, c)
    liquid = cast("pd.DataFrame", table[table["liquidity"] == ""]).copy()

    checks = tech_checks(liquid)
    liquid[list(checks.columns)] = checks
    scored = list(c.kept(TECH_CHECKS))
    liquid["score"] = checks[scored].sum(axis=1).astype("int64")
    score = cast("pd.Series", liquid["score"]).astype("float64")
    none = pd.Series(data=False, index=liquid.index)
    strong = top_share(score, c.strong_share) if c.on("strong") else none
    weak = bottom_share(score, c.weak_share) & ~strong if c.on("weak") else none
    liquid["chart"] = np.select([strong, weak], ["강", "약"], "") if scored else ""
    flags = overheat(liquid)
    for key in OVERHEAT:
        liquid[key] = flags[key] & c.on(key)
    liquid["peer"] = peers(liquid)
    liquid["peer_n"] = within_peers(pd.Series(1, index=liquid.index), liquid, "size")
    status = fundamentals_status(liquid, c)
    liquid[list(status.columns)] = status
    measured = axes(liquid, c)
    liquid[list(measured.columns)] = measured
    liquid["est"] = estimate_direction(liquid)
    liquid["group"] = groups(liquid, c)

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


def members(table: pd.DataFrame, key: str) -> pd.DataFrame:
    return cast("pd.DataFrame", table[table["group"] == key])


def counts(table: pd.DataFrame) -> dict[str, int]:
    return {key: len(members(table, key)) for key in GROUPS}


def group(table: pd.DataFrame, key: str) -> pd.DataFrame:
    """Group ``key``'s top ``TOP_N`` in its stated order."""
    columns, ascending = zip(*GROUP_ORDER[key], strict=True)
    ordered = members(table, key).sort_values(list(columns), ascending=list(ascending))
    return ordered.head(TOP_N)


def pick(table: pd.DataFrame, key: str, chosen: Sequence[str] = ()) -> pd.DataFrame:
    """Liquid stocks in the ``chosen`` sectors (all when empty), then group ``key``.

    The sector filter comes first, so a group shows its top ``TOP_N`` within
    those sectors; ``"all"`` keeps every stock, by market cap.
    """
    rows = liquid(table)
    if chosen:
        rows = cast("pd.DataFrame", rows[rows["sector"].isin(list(chosen))])
    if key != "all":
        rows = group(rows, key)
    return rows.reset_index(drop=True)


def map_rows(table: pd.DataFrame) -> list[dict[str, str | float]]:
    """Stocks in the four cells, as points of 밸류에이션 against 퀄리티."""
    placed = table[table["group"].isin([key for _, keys in MAP_ROWS for key in keys])]
    return [
        {
            "ticker": str(row["ticker"]),
            "name": str(row["name"]),
            "group": str(row["group"]),
            "cell": GROUP_NAMES[str(row["group"])],
            "quality": float(cast("float", row["quality"])),
            "valuation": float(cast("float", row["valuation"])),
            "est": str(row["est"]),
        }
        for row in records(cast("pd.DataFrame", placed))
    ]


def group_help(c: Criteria) -> str:
    """Every group's name and conditions, for a tooltip."""
    labels = ", ".join(
        c.kept({"peak": "정점 이익(경기순환주)", "tailwind": "업황 순풍"}).values()
    )
    reasons = (
        f"{labels} 또는 퀄리티·밸류에이션 산출 불가"
        if labels
        else "퀄리티·밸류에이션 산출 불가"
    )
    return (
        "**정의** 결격 없는 종목의 퀄리티 × 밸류에이션 업종 백분위 분류 · 경계"
        f" {c.split:.0%}\n\n**기준**\n\n"
        + "\n".join(
            f"- **{GROUP_NAMES[key]}**: {rule.format(s=f'{c.split:.0%}', hold=reasons)}"
            for key, rule in GROUPS.items()
        )
    )


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", frame.to_dict(orient="records"))


def view(frame: pd.DataFrame, c: Criteria) -> pd.DataFrame:
    """Rows as the results table shows them (``TABLE``), in the same order."""
    shown = frame.reset_index(drop=True)
    rows = records(shown)
    text = pd.DataFrame(
        {
            "group": [GROUP_NAMES.get(str(row.get("group")), "") for row in rows],
            "weak_spot": [weak_spot(row, c) for row in rows],
            "tags": [tags(row) for row in rows],
            "verdict": [verdict(row) for row in rows],
            "hot": [hot(row) for row in rows],
        },
        index=shown.index,
    )
    joined = pd.concat(
        [shown.drop(columns=list(text.columns), errors="ignore"), text], axis=1
    )
    return cast("pd.DataFrame", joined[[column for column, _, _ in TABLE]])


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


def percent(value: float) -> str:
    """A share as a percent without trailing zeros: 0.04 is "4%", 0.045 "4.5%"."""
    return f"{round(value * 100, 4):g}%"


def peer_level(row: Mapping[str, object]) -> str:
    """Which level ``peers`` compared the stock at."""
    sub = str(row.get("industry"))
    industry, group = GICS.get(sub, ("", ""))
    peer = row.get("peer")
    levels = (
        (sub, "세부 산업"),
        (industry, "산업"),
        (GROUP_LABELS.get(group), "산업 그룹"),
    )
    return next((level for label, level in levels if peer == label), "섹터")


def tags(row: Mapping[str, object]) -> str:
    """Labels for the table: 정점 이익, 업황 순풍, and an axis not computed."""
    out: list[str] = []
    if row.get("peak") is True:
        out.append("정점 이익")
    if row.get("tailwind") is True:
        out.append("업황 순풍")
    if math.isnan(_number(row.get("quality"))):
        out.append("퀄리티 산출 불가")
    if math.isnan(_number(row.get("valuation"))):
        out.append("밸류에이션 산출 불가")
    return " · ".join(out)


def weak_spot(row: Mapping[str, object], c: Criteria) -> str:
    """The measure that stands worst: the lowest 퀄리티 or the highest 밸류에이션."""
    spots = [
        (q, f"{name} 백분위 {q * 100:.0f}")
        for key, name in QUALITY.items()
        if not math.isnan(q := _number(row.get(key)))
    ] + [
        (1 - v, f"{name} 백분위 {v * 100:.0f}{'(비쌈)' if v >= c.split else ''}")
        for key, name in VALUATION.items()
        if not math.isnan(v := _number(row.get(key)))
    ]
    return min(spots, key=lambda spot: spot[0])[1] if spots else ""


MARGIN_SOURCES: Final = {"SEC": "SEC 10-K", "야후": "야후"}
TTM_SOURCES: Final = {"SEC": "SEC 분기", "야후": "야후"}
DQ_NAMES: Final = {"dq_fcf": "FCF", "dq_rev": "매출", "dq_debt": "부채"}


def verdict(row: Mapping[str, object]) -> str:
    """결격 reasons, or what could not be checked."""
    failed = [name for key, name in DQ_NAMES.items() if row.get(key) == FAIL]
    if failed:
        return "결격: " + ", ".join(failed)
    unsure = [
        f"{name} {row.get(key)}"
        for key, name in DQ_NAMES.items()
        if row.get(key) in UNSURE
    ]
    return ", ".join(unsure) if unsure else CLEAR


def hot(row: Mapping[str, object]) -> str:
    names = {"o_rsi": "RSI", "o_stoch": "스토캐스틱", "o_bb": "%B"}
    return " · ".join(name for key, name in names.items() if row.get(key) is True)


def tech_lines(row: Mapping[str, object], c: Criteria) -> list[tuple[bool, str, str]]:
    """Each score check in force with the values it compared."""
    r = {key: fmt("num", value) for key, value in row.items()}
    near = _number(row.get("close")) >= NEAR_HIGH * _number(row.get("high52"))
    details = {
        "t_above200": f"{r['close']} / {r['sma200']}",
        "t_20over60": f"{r['sma20']} / {r['sma60']}",
        "t_60rising": f"{r['sma60']} / {r['sma60_lag20']}",
        "t_above20": f"{r['close']} / {r['sma20']}",
        "m_macd_signal": f"{r['macd']} / {r['macd_signal']}",
        "m_macd_positive": r["macd"],
        "m_rsi": r["rsi"],
        "v_obv20": f"{fmt('int', row.get('obv'))} / {fmt('int', row.get('obv_lag20'))}",
        "v_updown": (
            f"{fmt('int', row.get('up_vol60'))} / {fmt('int', row.get('down_vol60'))}"
        ),
        "v_obv_high": (
            f"52주 고가 근처 · OBV {fmt('int', row.get('obv'))},"
            f" 60일 고점 {fmt('int', row.get('obv_max60'))}"
            if near
            else f"신고가 아님 · 기울기 {fmt('int', row.get('obv_slope60'))}/일"
        ),
    }
    return [
        (row.get(key) is True, label, details[key])
        for key, label in c.kept(TECH_CHECKS).items()
    ]


def overheat_lines(
    row: Mapping[str, object], c: Criteria
) -> list[tuple[bool, str, str]]:
    details = {
        "o_rsi": fmt("num", row.get("rsi")),
        "o_stoch": (
            f"%K {fmt('num', row.get('stoch_k'))} · %D {fmt('num', row.get('stoch_d'))}"
        ),
        "o_bb": fmt("num", row.get("pct_b")),
    }
    return [
        (row.get(key) is True, label, details[key])
        for key, label in c.kept(OVERHEAT).items()
    ]


def dq_lines(row: Mapping[str, object], c: Criteria) -> list[tuple[str, str, str]]:
    """Each disqualifier: status, the test, and the values against it."""
    roe = f"ROE {fmt('pct', row.get('roe'))} (TTM)"
    share = f"{c.equity_median_share:.0%}"
    if GICS.get(str(row.get("industry")), ("", ""))[1] == BANKS:
        fcf = (
            f"은행: ROE < 0 또는 TCE/TA < {percent(c.bank_tce_floor)}",
            f"{roe} · TCE/TA {fmt('pct', row.get('tce_ratio'))} (최근 분기,"
            " 유형 보통주자본 ÷ (총자산 − 영업권·무형자산))",
        )
    elif row.get("sector") == FINANCIALS:
        fcf = (
            f"금융업: ROE < 0 또는 자기자본비율 < 업종 중앙값의 {share}",
            f"{roe} · 자기자본비율 {fmt('pct', row.get('eq_ratio'))} (최근 분기) ·"
            f" 기준 {fmt('pct', row.get('eq_limit'))}"
            f" ({row.get('peer')} 중앙값의 {share})",
        )
    else:
        fcf = ("잉여현금흐름 < 0", f"{fmt('big', row.get('fcf'))} (TTM)")
    year = row.get("rev_year") or "—"
    revenue = (
        f"매출 성장 ≤ {c.revenue_floor:.0%}",
        f"{fmt('signed', row.get('rev_growth'))} (연간, {year} 결산)",
    )
    debt = (
        f"순부채/EBITDA > max(업종 중앙값×{c.net_debt_median_times:g},"
        f" {c.net_debt_floor:.1f})",
        f"{fmt('x', row.get('nd_ebitda'))} · 기준 {fmt('x', row.get('nd_limit'))}"
        f" (EBITDA TTM {fmt('big', row.get('ebitda'))})",
    )
    return [
        (str(row.get("dq_fcf")), fcf[0], fcf[1]),
        (str(row.get("dq_rev")), revenue[0], revenue[1]),
        (str(row.get("dq_debt")), debt[0], debt[1]),
    ]


def _ratio(top: object, base: object) -> float:
    base_value = _number(base)
    return _number(top) / base_value if base_value > 0 else math.nan


def _standing(value: object) -> str:
    number = _number(value)
    return UNKNOWN if math.isnan(number) else fmt("rank", number)


def label_lines(row: Mapping[str, object], c: Criteria) -> list[tuple[str, str, str]]:
    """Each 퀄리티 and 밸류에이션 measure, both means, 정점 이익 and 업황 순풍.

    Each line is the result, the test and the values it used; a measure or
    label turned off is 미적용.
    """
    financial = row.get("sector") == FINANCIALS
    year = row.get("rev_year") or "—"
    median = fmt("pct", row.get("margin_avg"))
    years = f"{fmt('int', row.get('margin_years'))}년"
    source = MARGIN_SOURCES.get(str(row.get("margin_source")), "—")
    ttm_source = TTM_SOURCES.get(str(row.get("op_margin_source")), "—")
    ttm_end = row.get("op_margin_end") or ""
    ttm_basis = f"{ttm_source}, {ttm_end}까지" if ttm_end else ttm_source
    ttm = f"{fmt('pct', row.get('op_margin'))} ({ttm_basis})"
    measures = {
        "q_roe": ("ROE (TTM)", fmt("pct", row.get("roe"))),
        "q_margin": (
            f"영업이익률 {SEC_YEARS}년 중앙값"
            f" (연간, 없으면 야후 {MARGIN_YEARS}년 이상)",
            f"{median} ({source} {years} 결산 중앙값) · TTM {ttm}",
        ),
        "q_fcf": (
            "FCF 마진 = FCF ÷ 연간 매출",
            f"{fmt('pct', _ratio(row.get('fcf'), row.get('revenue')))} ·"
            f" FCF {fmt('big', row.get('fcf'))} (TTM) ·"
            f" 매출 {fmt('big', row.get('revenue'))}",
        ),
        "q_growth": (
            "매출 성장 (연간)",
            f"{fmt('signed', row.get('rev_growth'))} ({year} 결산)",
        ),
        "v_fpe": ("선행 PER (예상)", fmt("num", row.get("fpe"))),
        "v_ev": ("EV/EBITDA (TTM)", fmt("num", row.get("ev_ebitda"))),
        "v_pb": (
            "PBR = 주가 ÷ 주당 장부가치",
            f"{fmt('num', _ratio(row.get('price'), row.get('book_value')))} ·"
            f" 주가 {fmt('usd', row.get('price'))} ·"
            f" 장부가치 {fmt('num', row.get('book_value'))}",
        ),
    }
    skipped = {"q_margin", "q_fcf", "v_ev"} if financial else {"v_pb"}
    known = {
        axis: sum(not math.isnan(_number(row.get(key))) for key in keys)
        for axis, keys in (("quality", QUALITY), ("valuation", VALUATION))
    }
    peak = (
        UNUSED
        if not c.on("peak")
        else "정점 이익"
        if row.get("peak") is True
        else "아님"
        if row.get("sector") in CYCLICALS
        else NOT_APPLICABLE
    )
    tailwind = (
        UNUSED
        if not c.on("tailwind")
        else NOT_APPLICABLE
        if financial
        else UNKNOWN
        if not _number(row.get("margin_years")) >= MARGIN_YEARS
        or math.isnan(_number(row.get("op_margin")))
        else MEANINGLESS
        if not _number(row.get("margin_avg")) > 0
        else "업황 순풍"
        if row.get("tailwind") is True
        else "아님"
    )
    return [
        *(
            (
                UNUSED
                if not c.on(key)
                else NOT_APPLICABLE
                if key in skipped
                else _standing(row.get(key)),
                test,
                detail,
            )
            for key, (test, detail) in measures.items()
        ),
        (
            _standing(row.get("quality")),
            "퀄리티 = 퀄리티 백분위의 평균 (높을수록 좋음,"
            f" {c.need(financial=financial)}개 이상)",
            f"{known['quality']}개 항목 평균 · 기준 {c.split:.0%}",
        ),
        (
            _standing(row.get("valuation")),
            "밸류에이션 = 밸류에이션 백분위의 평균 (높을수록 비쌈)",
            f"{known['valuation']}개 항목 평균 · 기준 {c.split:.0%}",
        ),
        (
            peak,
            f"경기순환주: 선행 PER < 후행 PER × {c.peak_ratio:g}",
            f"선행 {fmt('num', row.get('fpe'))} ·"
            f" 후행 {fmt('num', row.get('tpe'))} (TTM)",
        ),
        (
            tailwind,
            f"업황 순풍: TTM 영업이익률 ≥ {SEC_YEARS}년 중앙값 × {c.tailwind_ratio:g}"
            f" 그리고 ≥ {SEC_YEARS}년 중앙값 + {c.tailwind_gap * 100:g}%p",
            f"TTM {ttm} · 중앙값 {median} ({source} {years})",
        ),
    ]


def estimate_table(row: Mapping[str, object]) -> pd.DataFrame:
    """FY0/FY1 consensus EPS and 30-day revisions as a small table."""
    return pd.DataFrame(
        {
            "기간": ["FY0", "FY1"],
            "현재": [fmt("num", row.get(f"{p}_now")) for p in ESTIMATE_PERIODS],
            "30일 전": [fmt("num", row.get(f"{p}_30d")) for p in ESTIMATE_PERIODS],
            "90일 전": [fmt("num", row.get(f"{p}_90d")) for p in ESTIMATE_PERIODS],
            "30일 상향": [fmt("int", row.get(f"{p}_up30")) for p in ESTIMATE_PERIODS],
            "30일 하향": [fmt("int", row.get(f"{p}_down30")) for p in ESTIMATE_PERIODS],
        }
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
    store: pd.DataFrame, tickers: Sequence[str]
) -> tuple[date | None, list[str]]:
    """Where the incremental fetch starts, and which tickers need full history."""
    if store.empty:
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

    def fundamentals(self, ticker: str, *, financial: bool) -> Mapping[str, object]:
        import yfinance as yf  # noqa: PLC0415

        stock = yf.Ticker(ticker)
        with _rate_limit():
            info = cast("Mapping[str, object]", stock.info)
            if math.isnan(_number(info.get("marketCap"))):
                raise EmptyAnswerError("info 빈 응답")
            income = nonempty(stock.get_income_stmt(freq="yearly"), "손익계산서")
            quarterly = _frame(stock.get_income_stmt(freq="quarterly"))
            balance = (
                nonempty(stock.get_balance_sheet(freq="quarterly"), "재무상태표")
                if financial
                else None
            )
        return fundamentals_row(info, income, balance, quarterly)

    def estimates(self, ticker: str) -> Mapping[str, object]:
        import yfinance as yf  # noqa: PLC0415

        stock = yf.Ticker(ticker)
        with _rate_limit():
            trend = nonempty(stock.eps_trend, "EPS 추정치")
            return estimates_row(trend, _frame(stock.eps_revisions))


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
    start, full = price_plan(store, tickers)
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
    first = last_day - timedelta(days=HISTORY_DAYS)
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
    """SEC 10-K margin history for ``tickers``; returns the failures recorded.

    A ticker missing from SEC's list or without companyfacts gets an empty
    row, so Yahoo's median stands in for it.
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

    SEC margins are fetched only with ``edgar``, for non-financials. Raises
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
    financial = set(universe.loc[universe["sector"] == FINANCIALS, "ticker"])

    prices = update_prices(source, root, tickers, last_session(now), report, sleep)
    failed = update_rows(
        lambda t: [source.fundamentals(t, financial=t in financial)],
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
        plain = [t for t in tickers if t not in financial]
        try:
            missed = update_margins(edgar, root, today, plain, report, sleep)
        except (httpx.HTTPError, SecUnavailableError) as error:
            note += f" · SEC 받지 못함 ({type(error).__name__}: {error})"
        else:
            note += f" · SEC 받지 못한 값 {missed}" if missed else ""
    prune(root)
    priced = len(set(prices["ticker"]))
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

    def start(self, agent: str = "") -> bool:
        """Start an update unless one is running; SEC only with an ``agent``."""
        with self._lock:
            if self._progress.running:
                return False
            self._progress = Progress(running=True, stage="준비")
        threading.Thread(
            target=self._run, args=(agent.strip(),), name="finagent-screen", daemon=True
        ).start()
        return True

    def _report(self, stage: str, done: int, total: int) -> None:
        with self._lock:
            self._progress = Progress(running=True, stage=stage, done=done, total=total)

    def _run(self, agent: str) -> None:
        edgar: Filings | None = None
        try:
            edgar = self._edgar(agent) if agent else None
            now = datetime.now().astimezone()
            source = self._source(self.root)
            message = update(source, self.root, now, self._report, edgar=edgar)
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

    def _finish(self, message: str, *, failed: bool = True) -> None:
        with self._lock:
            self._progress = Progress(message=message, failed=failed)
