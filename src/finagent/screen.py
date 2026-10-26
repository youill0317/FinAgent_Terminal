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
import json
import math
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
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
# Cutlines are the user's, without a backtest; shares are of the stocks with a
# value, ties at the cutline included.
MIN_BARS: Final = 250
MIN_PRICE: Final = 5.0
ILLIQUID_SHARE: Final = 0.05  # lowest 20-day dollar volume, left out
STRONG_SHARE: Final = 0.20  # top tech scores: 차트 강
WEAK_SHARE: Final = 0.40  # bottom tech scores: 차트 약
SURGE_SHARE: Final = 0.10  # top 21-day return or MAX: 단기 급등
SPLIT: Final = 0.5  # 퀄리티 and 밸류에이션 percentiles split at the middle of the peers
NEAR_BAND: Final = 0.05  # 경계 근처: within this of the split
QUALITY_MIN: Final = 3  # 퀄리티 measures a non-financial needs
QUALITY_MIN_FIN: Final = 2  # and a financial
VALUATION_MIN: Final = 2  # 밸류에이션 measures needed
PEER_MIN: Final = 15  # members a GICS level needs, else the next wider one
FCF_YEARS: Final = 3  # fiscal years of the cumulative FCF disqualifier
FCF_YEARS_LEAST: Final = 2  # fewer than N years: the ones there, at least this many
NET_DEBT_FLOOR: Final = 4.0
NET_DEBT_MEDIAN_TIMES: Final = 2.0
EQUITY_MEDIAN_SHARE: Final = 0.5
COVERAGE_FLOOR: Final = 1.5  # EBIT / interest expense
# 정점 이익: TTM operating margin at least this times the normal (10-year
# median) one, and forward EPS at most trailing EPS times the second.
PEAK_RATIO: Final = 1.5
PEAK_EPS_RATIO: Final = 1.05
# 업황 순풍: the TTM margin's robust z against the 10-year median and MAD, and
# its gap over that median.
TAILWIND_Z: Final = 2.0
TAILWIND_GAP: Final = 0.03
MAD_SCALE: Final = 1.4826  # MAD to a normal standard deviation
# With fewer than ``LONG_YEARS`` years in ``LONG_WINDOW``, the 5-year median
# stands in (짧은 창): TTM at least these times and above it.
SHORT_RATIO: Final = 1.5
SHORT_GAP: Final = 0.05
LONG_WINDOW: Final = 10
LONG_YEARS: Final = 7
SHORT_YEARS: Final = 4  # the least a 5-year median needs for the hold rules
REVENUE_FLOOR: Final = -0.15  # 매출 급감 label
RELIANCE_RATIO: Final = 1.5  # 추정 의존 label: forward EPS against trailing
WEAK_TREND_SHARE: Final = 0.20  # 가격·추정치 약세: bottom 12-1 momentum
CYCLICAL_STD_SHARE: Final = 1 / 3  # top 10-year margin deviations: 경기순환
IMPROVE_YEARS: Final = 3  # 수익성 개선: the change over this many years
TTM_QUARTERS: Final = 4  # quarters in the TTM operating margin
MARGIN_YEARS: Final = 3  # fiscal years Yahoo's margin median and deviation need
SEC_YEARS: Final = 5  # 10-K fiscal years in SEC's median, else Yahoo's
STABILITY_YEARS: Final = 3  # fiscal years a 5-year deviation needs
# 연도 교체 주의: a fiscal year ended and not yet reported, or reported in a
# 10-K filed this many days ago.
YEAR_TURN_DAYS: Final = 90
YEAR_END_SLACK_DAYS: Final = 7  # 52/53-week years end a few days apart
# Momentum crash notice (Daniel and Moskowitz): the market's 24-month return
# below zero and its 126-day variance above the median over the stored days.
BEAR_BARS: Final = 504
VARIANCE_BARS: Final = 126
# When the screening definitions last changed, kept in each snapshot.
DEFINITIONS: Final = "2026-10-26"
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
# Other us-gaap concepts per fiscal year: column → concepts, the first with a
# value for a year end wins. Assets and equity are year-end instants.
SEC_FLOWS: Final = {
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "ocf": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "interest": (
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
    ),
    "interest_paid": ("InterestPaidNet", "InterestPaid"),
}
SEC_INSTANTS: Final = {
    "assets": ("Assets",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
}
TOP_N: Final = 20
FINANCIALS: Final = "Financials"
BANKS: Final = "Banks"
# Sectors that report operating cash flow but whose FCF is capex by design.
CASH_ONLY: Final = frozenset({"Utilities", "Real Estate"})
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
BANK_TCE_FLOOR: Final = 0.05  # tangible common equity / tangible assets
CYCLICALS: Final = frozenset(
    {"Energy", "Materials", "Industrials", "Consumer Discretionary"}
)
CYCLICAL_GROUPS: Final = frozenset(
    {"Semiconductors & Semiconductor Equipment", "Technology Hardware & Equipment"}
)
SMA_DAYS: Final = (20, 21, 60, 200)
YEAR_BARS: Final = 252
MONTH_BARS: Final = 21
MOMENTUM_BARS: Final = 273  # 12-1 momentum: a year back, skipping the last month
TECH_COLUMNS: Final = (
    "bars",
    "price",  # last close as traded, for the price floor and the table
    "dollar_vol20",
    *(f"sma{days}" for days in SMA_DAYS),
    "close",  # last adjusted close, which the indicators use
    "high52",
    "from_high",
    "near_high",  # close / 52-week high
    "ma_dist",  # 21-day SMA / 200-day SMA
    "mom12_1",
    "ret_1m",
    "ret_3m",
    "max_1m",  # the largest daily return of the last month
)

# Status words the page shows for a check that is not a plain pass or fail.
CLEAR: Final = "통과"
FAIL: Final = "결격"
UNKNOWN: Final = "미확인"  # no data
MEANINGLESS: Final = "해석 부적절"  # negative or zero base
NOT_COMPUTED: Final = "미산출"
NET_CASH: Final = "순현금"
NOT_APPLICABLE: Final = "해당 없음"  # non-financial definition on a financial
UNUSED: Final = "미적용"  # a check turned off in the 기준 panel
# A check left out for want of data, without withholding the stock.
NO_INTEREST: Final = "미적용(이자비용 미확인)"
NO_EBIT: Final = "미적용(EBIT 미확인)"
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
    "ocf": "operatingCashflow",
    "roe": "returnOnEquity",
    "book_value": "bookValue",
    "total_debt": "totalDebt",
    "total_cash": "totalCash",
    "ebitda": "ebitda",
    "ev": "enterpriseValue",
    "trailing_eps": "trailingEps",
    "fpe": "forwardPE",
    "tpe": "trailingPE",
    "mcap": "marketCap",
    "ev_ebitda": "enterpriseToEbitda",
    "short_pct": "shortPercentOfFloat",
}
# Column → row of Yahoo's trailing (TTM) income statement.
TTM_ROWS: Final = {
    "revenue_ttm": "TotalRevenue",
    "gross_profit": "GrossProfit",
    "cost_of_revenue": "CostOfRevenue",
    "op_income_ttm": "OperatingIncome",
    "rnd": "ResearchAndDevelopment",
    "ebit": "EBIT",
    "interest": "InterestExpense",
    "net_income": "NetIncome",
}
# Column → row of the trailing cash flow statement. Cash-flow signed: a
# working capital increase is negative.
CASH_ROWS: Final = {"wc_change": "ChangeInWorkingCapital"}
# Column → row of the newest quarterly balance sheet.
BALANCE_ROWS: Final = {
    "total_assets": "TotalAssets",
    "tbv": "TangibleBookValue",
    "shares": "OrdinarySharesNumber",
}
ANNUAL_YEARS: Final = 5  # yearly FCF and operating income / assets kept, newest first
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
    "margin_std",
    "op_margin",
    "op_margin_quarters",
    "ttm_end",  # the end of the newest quarter in Yahoo's TTM
    *TTM_ROWS,
    *CASH_ROWS,
    *BALANCE_ROWS,
    *(f"fcf_y{year}" for year in range(ANNUAL_YEARS)),
    *(f"opa_y{year}" for year in range(ANNUAL_YEARS)),
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
    *SEC_FLOWS,
    *SEC_INSTANTS,
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

# 기술 점수 measures: id → name, and the column each ranks (higher is better).
TECH_ITEMS: Final = {
    "t_mom": "12-1 모멘텀",
    "t_high": "52주 고가 근접",
    "t_trend": "이동평균 거리",
}
TECH_VALUES: Final = {"t_mom": "mom12_1", "t_high": "near_high", "t_trend": "ma_dist"}
SURGE: Final = {
    "s_ret": "21거래일 수익률",
    "s_max": "21거래일 최대 일간 수익률",
}
SURGE_VALUES: Final = {"s_ret": "ret_1m", "s_max": "max_1m"}
# 퀄리티 measures: column → name. Each is a percentile within the peer group,
# higher is better.
QUALITY: Final = {
    "q_cbop": "현금기반 영업수익성",
    "q_gpa": "총이익/자산",
    "q_improve": "수익성 개선",
    "q_stability": "영업이익률 5년 안정성",
    "q_leverage": "레버리지",
    "q_roe": "ROE",
    "q_roa": "ROA",
    "q_capital": "자본 비율",
    "q_roe_stability": "ROE 5년 안정성",
}
QUALITY_CORE: Final = ("q_cbop", "q_gpa")  # a non-financial needs one of them
# 밸류에이션 measures, all yields on the price: the percentile of the yield
# turned around, so higher is pricier and a negative yield the priciest.
VALUATION: Final = {
    "v_fwd": "선행 E/P",
    "v_ttm": "TTM E/P",
    "v_ebitda": "EBITDA/EV",
    "v_fcf": "FCF/EV",
    "v_book": "장부가치 수익률",
}
# Display-only labels the 기준 panel can turn off.
LABEL_RULES: Final = {
    "l_revdrop": "매출 급감",
    "l_reliance": "추정 의존",
    "l_margin_jump": "마진 급등",
    "l_near": "경계 근처",
    "l_weak_trend": "가격·추정치 약세",
    "l_year_turn": "연도 교체 주의",
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
    "all": (("combined", False), ("mcap", False)),
    "prime": (("edge", False), ("mcap", False)),
    "pricey": (("edge", False), ("mcap", False)),
    "trap": (("edge", False), ("mcap", False)),
    "weak": (("edge", False), ("mcap", False)),
    "hold": (("mcap", False),),
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
# FY1 consensus EPS direction.
EST_UP: Final = "상향"
EST_FLAT: Final = "보합"
EST_DOWN: Final = "하향"
EST_ORDER: Final = (EST_UP, EST_FLAT, EST_DOWN, UNKNOWN)
# 기술 점수 bands, a map coloring.
CHART_STRONG: Final = "차트 강"
CHART_MIDDLE: Final = "중간"
CHART_WEAK: Final = "차트 약"
CHART_ORDER: Final = (CHART_STRONG, CHART_MIDDLE, CHART_WEAK, NOT_COMPUTED)
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
    ("hot", "단기 급등", ""),
    ("roe", "ROE", "pct"),
    ("op_margin", "영업이익률 (TTM)", "pct"),
    ("fwd_pe", "선행 PER", "num"),
    ("ev_ebitda", "EV/EBITDA", "num"),
    ("rev_growth", "매출 성장 (연간)", "signed"),
    ("nd_ebitda", "순부채/EBITDA", "x"),
    ("fcf", "FCF", "big"),
    ("combined", "통합 점수", "int"),
    ("peer_level", "비교 업종 단계", ""),
    ("counted", "산출 항목 수", ""),
    ("mom12_1", "12-1 모멘텀", "signed"),
    ("rev_strength", "수정 강도", "signed2"),
    ("cbop", "현금기반 영업수익성", "pct"),
    ("coverage", "이자보상배율", "x"),
    ("fcf_cum", "누적 FCF", "big"),
    ("sources", "출처", ""),
)
# Measures only one side has; the other side's are 해당 없음.
FINANCIAL_ONLY: Final = frozenset(
    {"q_roe", "q_roa", "q_capital", "q_roe_stability", "v_book"}
)
NON_FINANCIAL_ONLY: Final = frozenset(
    {"q_cbop", "q_gpa", "q_improve", "q_stability", "q_leverage", "v_ebitda", "v_fcf"}
)
DQ_RULES: Final = {
    "dq_cash": "현금흐름·자본(금융업은 ROE·자기자본비율, 은행은 ROE·TCE/TA)",
    "dq_debt": "순부채/EBITDA(금융업 해당 없음)",
    "dq_cover": "이자보상배율(금융업 해당 없음)",
}
FROM_HIGH_HELP: Final = (
    f"**정의** 최근 종가 ÷ 최근 {YEAR_BARS}거래일 최고가(고가 기준) − 1\n\n"
    "**해석** 0에 가까울수록 52주 고점 부근"
)
ESTIMATE_HELP: Final = (
    "**정의** 애널리스트 컨센서스 EPS · FY1 진행 중인(또는 끝나고 결산 전인)"
    " 회계연도 · FY2 그다음 회계연도 · 30일·90일 전 값 · 최근 30일 상향·하향"
    " 애널리스트 수 · 추정 애널리스트 수\n\n"
    "**해석** 수정 강도 = (현재 − 90일 전) ÷ 주가, 같은 회계연도끼리 비교해 FY1·FY2"
    " 평균(결산 대기면 FY2만) · 순수정 비율 = (30일 상향 − 하향) ÷ 애널리스트 수 ·"
    " 수정 강도 > 0 그리고 순수정 ≥ 0이면 상향 · 수정 강도 < 0 그리고 순수정 < 0이면"
    " 하향 · 그 밖 보합"
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
    peer_min: int = PEER_MIN
    fcf_years: int = FCF_YEARS
    equity_median_share: float = EQUITY_MEDIAN_SHARE
    bank_tce_floor: float = BANK_TCE_FLOOR
    net_debt_median_times: float = NET_DEBT_MEDIAN_TIMES
    net_debt_floor: float = NET_DEBT_FLOOR
    coverage_floor: float = COVERAGE_FLOOR
    quality_min: int = QUALITY_MIN
    quality_min_fin: int = QUALITY_MIN_FIN
    split: float = SPLIT
    peak_ratio: float = PEAK_RATIO
    peak_eps_ratio: float = PEAK_EPS_RATIO
    tailwind_z: float = TAILWIND_Z
    tailwind_gap: float = TAILWIND_GAP
    strong_share: float = STRONG_SHARE
    weak_share: float = WEAK_SHARE
    revenue_floor: float = REVENUE_FLOOR
    reliance_ratio: float = RELIANCE_RATIO
    near_band: float = NEAR_BAND
    weak_trend_share: float = WEAK_TREND_SHARE
    surge_share: float = SURGE_SHARE
    valuation_min: int = VALUATION_MIN
    cyclical_share: float = CYCLICAL_STD_SHARE
    long_years: int = LONG_YEARS
    short_years: int = SHORT_YEARS
    improve_years: int = IMPROVE_YEARS
    year_turn_days: int = YEAR_TURN_DAYS
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
        """퀄리티 measures a stock needs, at most those in force."""
        least = self.quality_min_fin if financial else self.quality_min
        return min(least, len(self.measures(QUALITY, financial=financial)))

    def need_valuation(self, *, financial: bool) -> int:
        """밸류에이션 measures a stock needs, at most those in force."""
        least = self.valuation_min
        return min(least, len(self.measures(VALUATION, financial=financial)))

    @property
    def core(self) -> list[str]:
        """The core 퀄리티 measures in force; a non-financial needs one of them."""
        return [key for key in QUALITY_CORE if self.on(key)]


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
        """The field's value; the default as shown is the default exactly (1/3)."""
        default = getattr(DEFAULT, self.field)
        if value == self.shown(default):
            return default
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


CRASH_HELP: Final = (
    "**정의** 시장(유니버스 종목 시가총액 가중 일간 수익률)의 지난"
    f" {BEAR_BARS}거래일 누적 수익률 < 0 그리고 직전 {VARIANCE_BARS}거래일 일간"
    " 수익률 분산 > 저장 기간 중앙값이면 주의\n\n"
    "**해석** 큰 폭 하락 뒤 변동성 높은 반등기에는 모멘텀 급락 위험이 큼(Daniel·"
    "Moskowitz) · 현재 구성 종목으로 만든 지수라 과거 수익률이 조금 좋게 나옴"
)
# The numbers of a 퀄리티 measure's rule.
QUALITY_LIMITS: Final = {
    "q_improve": (Limit("improve_years", "변화 (년)", 2, ANNUAL_YEARS - 1, 1),),
}
QUALITY_HELP: Final = {
    "q_cbop": "(영업이익 + 연구개발비 − 운전자본 증가분) ÷ 총자산, TTM · 연구개발비가"
    " 없으면 0, 운전자본 변동이 없으면 미산출 · 금융업 해당 없음\n\n"
    "**해석** 높을수록 현금으로 뒷받침되는 영업 수익성 우위",
    "q_gpa": "매출총이익(TTM) ÷ 총자산 · 매출총이익이 없으면 매출 − 매출원가 · 금융업"
    " 해당 없음\n\n**해석** 높을수록 자산 대비 총이익 우위",
    "q_improve": "영업이익 ÷ 총자산의 N년 변화 · SEC 10-K, 없으면 야후 연간 · 1개 연도"
    " 모자라면 N − 1년 변화(짧은 창) · 금융업 해당 없음\n\n"
    "**해석** 높을수록 수익성이 좋아지는 중\n\n"
    f"**기준** 기본 {IMPROVE_YEARS}년",
    "q_stability": f"영업이익률 {SEC_YEARS}년 표준편차 · SEC 10-K, 없으면 야후 연간"
    f" {MARGIN_YEARS}년 이상 · 금융업 해당 없음\n\n**해석** 낮을수록(백분위 높을수록)"
    " 꾸준한 수익성",
    "q_leverage": "총차입금 ÷ 총자산 · 금융업 해당 없음\n\n**해석** 낮을수록(백분위"
    " 높을수록) 빚 부담이 가벼움",
    "q_roe": "자기자본이익률(TTM) · 자기자본 0 이하 제외 · 금융업 전용\n\n"
    "**해석** 높을수록 자본 효율 우위",
    "q_roa": "순이익(TTM) ÷ 총자산 · 금융업 전용\n\n**해석** 높을수록 자산 효율 우위",
    "q_capital": "은행: TCE/TA · 그 밖의 금융업: 자기자본 ÷ 총자산 · 금융업 전용\n\n"
    "**해석** 높을수록 자본 완충이 두꺼움",
    "q_roe_stability": f"SEC 10-K {SEC_YEARS}년 ROE(순이익 ÷ 연말 자기자본)의 표준편차"
    " · 금융업 전용\n\n**해석** 낮을수록(백분위 높을수록) 꾸준한 수익성",
}
VALUATION_HELP: Final = {
    "v_fwd": "선행 EPS ÷ 주가 · 선행 EPS = w × FY1 + (1 − w) × FY2, w = 이번 회계연도"
    " 남은 기간 ÷ 1년(결산 대기면 0) · 음수 포함 · 표시 열은 선행 PER",
    "v_ttm": "후행 12개월 EPS ÷ 주가 · 음수 포함",
    "v_ebitda": "EBITDA(TTM) ÷ EV · 음수 포함, EV 0 이하 제외 · 표시 열은 EV/EBITDA ·"
    " 금융업은 장부가치 수익률로 대체",
    "v_fcf": "FCF(TTM) ÷ EV · 음수 포함, EV 0 이하 제외 · 금융업 해당 없음",
    "v_book": "은행: 주당 유형장부가치 ÷ 주가 · 그 밖의 금융업: 주당 장부가치 ÷ 주가 ·"
    " 장부가치 0 이하는 가장 비싼 쪽 · 금융업 전용",
}
TECH_HELP: Final = {
    "t_mom": f"수정 종가 기준 {YEAR_BARS}거래일 전부터 {MONTH_BARS}거래일 전까지의"
    f" 수익률(최근 1개월 제외) · 일봉 {MOMENTUM_BARS}개 미만이면 미산출",
    "t_high": f"종가 ÷ 최근 {YEAR_BARS}거래일 최고가",
    "t_trend": "21일선 ÷ 200일선",
}
RULES: Final = (
    Rule(
        "min_bars",
        "거래 가능",
        "일봉 수",
        "**정의** 받은 일봉 수 하한 · 미만이면 이력 부족(분사·신규 상장 가능)으로"
        " 제외\n\n"
        f"**기준** 기본 {MIN_BARS}개 · 끄면 {MIN_BARS}개 미만 종목은 기술 지표 없음",
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
        "peer_min",
        "비교 업종",
        "비교 업종 크기",
        "**정의** 유동성 통과 종목 중 GICS 세부 산업 → 산업 → 산업 그룹 → 섹터 순으로,"
        " 이 수 이상인 가장 좁은 단계\n\n"
        "**해석** 키울수록 넓은 업종과 비교\n\n"
        f"**기준** 기본 {PEER_MIN}종목",
        (Limit("peer_min", "종목 (개)", 10, 30, 1),),
        toggle=False,
    ),
    Rule(
        "dq_cash",
        "결격 점검",
        "현금흐름·자본",
        "**정의** 일반: 영업현금흐름(TTM) < 0 또는 최근 N개 회계연도 누적 FCF < 0 ·"
        " 유틸리티·부동산은 영업현금흐름만 · 금융업: ROE < 0 또는 자기자본비율 <"
        " 업종 중앙값 × 비율 · 은행: ROE < 0 또는 TCE/TA < 하한 · 해당하면 결격\n\n"
        f"**해석** 현금을 못 버는 회사, 자본 완충이 얇은 금융사 제외 · 연도가 모자라면"
        f" 있는 연도({FCF_YEARS_LEAST}개 이상), 그것도 없으면 FCF(TTM) < 0 · SEC 10-K"
        " 영업현금흐름 − 설비투자 우선, 없으면 야후 연간\n\n"
        f"**기준** 기본 {FCF_YEARS}개 연도 · 중앙값의 {EQUITY_MEDIAN_SHARE:.0%} ·"
        f" TCE/TA {BANK_TCE_FLOOR:.0%}",
        (
            Limit("fcf_years", "누적 FCF (년)", 2, 5, 1),
            Limit(
                "equity_median_share", "금융업 중앙값 대비 (%)", 0, 100, 5, percent=True
            ),
            Limit("bank_tce_floor", "은행 TCE/TA (%)", 0, 20, 0.5, percent=True),
        ),
    ),
    Rule(
        "dq_debt",
        "결격 점검",
        "순부채/EBITDA",
        "**정의** 순부채/EBITDA > max(업종 중앙값 × 배수, 하한)이면 결격 · 순현금은"
        " 통과 · EBITDA 0 이하이면서 순부채 > 0이면 결격 · 금융업 해당 없음\n\n"
        "**해석** 업종보다 빚이 무거운 회사 제외\n\n"
        f"**기준** 기본 중앙값 × {NET_DEBT_MEDIAN_TIMES:g} · 하한"
        f" {NET_DEBT_FLOOR:.1f}x",
        (
            Limit("net_debt_median_times", "중앙값 × (배)", 0.5, 10, 0.5),
            Limit("net_debt_floor", "하한 (x)", 0, 20, 0.5),
        ),
    ),
    Rule(
        "dq_cover",
        "결격 점검",
        "이자보상배율",
        "**정의** EBIT(TTM) ÷ 이자비용 < 하한이면 결격 · 이자비용 0이면 통과 ·"
        " 금융업 해당 없음\n\n"
        "**해석** 이익으로 이자를 감당하기 어려운 회사 제외 · 이자비용은 야후 → SEC"
        " 10-K 이자비용 → SEC 이자 지급액 순, 끝까지 없으면 이 점검만"
        " 미적용(칸 배정은 유지)\n\n"
        f"**기준** 기본 {COVERAGE_FLOOR:g}x",
        (Limit("coverage_floor", "하한 (x)", 0, 5, 0.5),),
    ),
    *(
        Rule(
            key,
            "퀄리티",
            name,
            f"**정의** 업종 백분위 · {QUALITY_HELP[key]}",
            QUALITY_LIMITS.get(key, ()),
        )
        for key, name in QUALITY.items()
    ),
    Rule(
        "quality_min",
        "퀄리티",
        "최소 항목 수",
        "**정의** 퀄리티 백분위 평균에 필요한 항목 수 · 미달이면 퀄리티 산출 불가,"
        " 보류\n\n"
        "**해석** 켠 항목이 더 적으면 켠 항목 수 · 비금융은 현금기반 영업수익성 또는"
        " 총이익/자산 중 1개 필수\n\n"
        f"**기준** 기본 비금융 {QUALITY_MIN}개 · 금융업 {QUALITY_MIN_FIN}개",
        (
            Limit("quality_min", "비금융 (개)", 1, 5, 1),
            Limit("quality_min_fin", "금융업 (개)", 1, 4, 1),
        ),
        toggle=False,
    ),
    *(
        Rule(
            key,
            "밸류에이션",
            name,
            f"**정의** 업종 백분위(수익률이 낮을수록 높음) · {VALUATION_HELP[key]}\n\n"
            "**해석** 높을수록 고평가",
        )
        for key, name in VALUATION.items()
    ),
    Rule(
        "valuation_min",
        "밸류에이션",
        "최소 항목 수",
        "**정의** 밸류에이션 백분위 평균에 필요한 항목 수 · 미달이면 밸류에이션 산출"
        " 불가, 보류\n\n"
        "**해석** 켠 항목이 더 적으면 켠 항목 수\n\n"
        f"**기준** 기본 {VALUATION_MIN}개",
        (Limit("valuation_min", "항목 (개)", 1, 5, 1),),
        toggle=False,
    ),
    Rule(
        "split",
        "칸 배정·보류",
        "퀄리티·밸류에이션 경계",
        "**정의** 칸을 나누는 업종 백분위 · 퀄리티 ≥ 경계면 퀄리티 상위 ·"
        " 밸류에이션 < 경계면 저평가\n\n"
        "**해석** 높일수록 퀄리티 상위는 좁게, 저평가는 넓게\n\n"
        f"**기준** 기본 {SPLIT:.0%}",
        (Limit("split", "업종 백분위 (%)", 10, 90, 5, percent=True),),
        toggle=False,
    ),
    Rule(
        "cyclical",
        "칸 배정·보류",
        "경기순환 기업",
        "**정의** GICS 에너지·소재·산업재·임의소비재 섹터, 반도체·반도체 장비, 기술"
        " 하드웨어·장비 산업 그룹, 또는 10년 영업이익률 표준편차가 유동성 통과 종목"
        " 상위 비율 · 금융업 제외 · 경계 동률 포함\n\n"
        "**해석** 정점 이익·업황 순풍을 보는 대상 · 0%면 업종으로만 판정\n\n"
        f"**기준** 기본 상위 {CYCLICAL_STD_SHARE:.0%}",
        (Limit("cyclical_share", "표준편차 상위 (%)", 0, 50, 1, percent=True),),
        toggle=False,
    ),
    Rule(
        "history",
        "칸 배정·보류",
        "이익률 이력",
        f"**정의** {LONG_WINDOW}년 영업이익률 중앙값에 필요한 연도 수 · 미달이면 5년"
        " 중앙값으로 판정하고 짧은 창 표시 · 5년 중앙값도 연도 수 미달이면 정점"
        " 이익·업황 순풍 미적용\n\n"
        "**해석** 줄일수록 짧은 이력으로도 정상 이익률을 추정\n\n"
        f"**기준** 기본 {LONG_YEARS}개 · {SHORT_YEARS}개 연도",
        (
            Limit("long_years", "10년 창 (년)", 5, 10, 1),
            Limit("short_years", "5년 창 (년)", 3, 5, 1),
        ),
        toggle=False,
    ),
    Rule(
        "peak",
        "칸 배정·보류",
        "정점 이익",
        "**정의** 경기순환 기업에서 TTM 영업이익률 ≥ 10년 중앙값 × 배수 그리고 선행"
        " EPS ≤ 후행 EPS × 비율이면 보류 · 후행 EPS > 0, 중앙값 > 0일 때만\n\n"
        "**해석** 이익 정점에서 싸 보이는 경기순환주 분리 · 10년 이력이 모자라면 5년"
        " 중앙값(짧은 창, 이익률 이력 참고)\n\n"
        f"**기준** 기본 × {PEAK_RATIO:g} · {PEAK_EPS_RATIO:g}",
        (
            Limit("peak_ratio", "중앙값 × (배)", 1.1, 3, 0.1),
            Limit("peak_eps_ratio", "선행 ÷ 후행 EPS", 0.8, 1.3, 0.05),
        ),
    ),
    Rule(
        "tailwind",
        "칸 배정·보류",
        "업황 순풍",
        f"**정의** 경기순환 기업에서 z = (TTM 영업이익률 − 10년 중앙값) ÷ ({MAD_SCALE}"
        " × 10년 MAD) ≥ 하한 그리고 TTM − 중앙값 ≥ 차이면 보류 · 금융업 해당 없음\n\n"
        "**해석** 업황 호조로 일시적으로 부푼 이익 분리 · 10년 이력이 모자라면"
        f" TTM ≥ 5년 중앙값 × {SHORT_RATIO:g} 그리고 ≥ 중앙값 +"
        f" {SHORT_GAP * 100:.0f}%p(짧은 창) · 비경기순환 기업은 같은 조건이면 마진"
        " 급등 라벨\n\n"
        f"**기준** 기본 z {TAILWIND_Z:g} · +{TAILWIND_GAP * 100:.0f}%p",
        (
            Limit("tailwind_z", "z 하한", 1, 4, 0.5),
            Limit("tailwind_gap", "중앙값 + (%p)", 0, 15, 1, percent=True),
        ),
    ),
    *(
        Rule(
            key,
            "표시 지표",
            name,
            f"**정의** {TECH_HELP[key]} · 유동성 통과 종목 전체 안 백분위\n\n"
            "**해석** 기술 점수 = 켠 항목 백분위의 평균",
            part="기술 점수",
        )
        for key, name in TECH_ITEMS.items()
    ),
    Rule(
        "strong",
        "표시 지표",
        "차트 강",
        "**정의** 기술 점수 상위 비율 · 유동성 통과 종목 기준, 경계 동률 포함\n\n"
        f"**기준** 기본 상위 {STRONG_SHARE:.0%}",
        (Limit("strong_share", "상위 (%)", 1, 50, 1, percent=True),),
        part="기술 점수",
    ),
    Rule(
        "weak",
        "표시 지표",
        "차트 약",
        "**정의** 기술 점수 하위 비율 · 차트 강 제외 · 경계 동률 포함\n\n"
        f"**기준** 기본 하위 {WEAK_SHARE:.0%}",
        (Limit("weak_share", "하위 (%)", 1, 90, 1, percent=True),),
        part="기술 점수",
    ),
    *(
        Rule(
            key,
            "표시 지표",
            name,
            f"**정의** {name} 상위 비율 · 유동성 통과 종목 기준, 경계 동률 포함\n\n"
            "**해석** 단기 급등 표시에만 반영 · 매도 신호가 아니라 진입 시점 참고용",
            part="단기 급등",
        )
        for key, name in SURGE.items()
    ),
    Rule(
        "surge",
        "표시 지표",
        "급등 기준",
        "**정의** 21거래일 수익률·최대 일간 수익률 각각 상위 비율이면 단기 급등\n\n"
        f"**기준** 기본 상위 {SURGE_SHARE:.0%}",
        (Limit("surge_share", "상위 (%)", 1, 30, 1, percent=True),),
        toggle=False,
        part="단기 급등",
    ),
    Rule(
        "l_revdrop",
        "표시 지표",
        LABEL_RULES["l_revdrop"],
        "**정의** 연간 매출 성장률 ≤ 하한 그리고 추정치 방향 하향 · 전년 매출 0 이하면"
        " 해석 부적절\n\n**해석** 라벨만 표시, 칸 배정 그대로\n\n"
        f"**기준** 기본 {REVENUE_FLOOR:.0%}",
        (Limit("revenue_floor", "하한 (%)", -50, 0, 1, percent=True),),
        part="점검 라벨",
    ),
    Rule(
        "l_reliance",
        "표시 지표",
        LABEL_RULES["l_reliance"],
        "**정의** 선행 EPS ≥ 후행 EPS × 배수, 또는 후행 EPS ≤ 0이고 선행 EPS > 0 ·"
        " 전 섹터\n\n**해석** 라벨만 표시, 칸 배정 그대로\n\n"
        f"**기준** 기본 × {RELIANCE_RATIO:g}",
        (Limit("reliance_ratio", "후행 EPS × (배)", 1.1, 3, 0.1),),
        part="점검 라벨",
    ),
    Rule(
        "l_margin_jump",
        "표시 지표",
        LABEL_RULES["l_margin_jump"],
        "**정의** 비경기순환 기업이 업황 순풍 조건을 충족 · 숫자는 업황 순풍과 같음\n\n"
        "**해석** 라벨만 표시, 칸 배정 그대로",
        part="점검 라벨",
    ),
    Rule(
        "l_near",
        "표시 지표",
        LABEL_RULES["l_near"],
        "**정의** 퀄리티나 밸류에이션이 경계 ± 폭 안\n\n"
        "**해석** 라벨만 표시, 칸 배정 그대로\n\n"
        f"**기준** 기본 ±{NEAR_BAND * 100:.0f}%p",
        (Limit("near_band", "폭 (%p)", 0, 15, 1, percent=True),),
        part="점검 라벨",
    ),
    Rule(
        "l_weak_trend",
        "표시 지표",
        LABEL_RULES["l_weak_trend"],
        "**정의** 검토 우선·밸류 트랩 점검 칸에서 12-1 모멘텀이 유동성 통과 종목 하위"
        " 비율 그리고 추정치 방향 하향\n\n"
        "**해석** 라벨만 표시, 칸 배정 그대로\n\n"
        f"**기준** 기본 하위 {WEAK_TREND_SHARE:.0%}",
        (Limit("weak_trend_share", "하위 (%)", 5, 50, 1, percent=True),),
        part="점검 라벨",
    ),
    Rule(
        "l_year_turn",
        "표시 지표",
        LABEL_RULES["l_year_turn"],
        "**정의** 회계연도가 끝나고 10-K 전(결산 대기), 또는 그 연도 10-K 제출 후"
        " 일수 이내 · SEC에 그 연도 10-K가 없으면 결산일부터\n\n"
        "**해석** 라벨만 표시 · 결산 대기면 끄더라도 선행 EPS는 FY2만 사용\n\n"
        f"**기준** 기본 {YEAR_TURN_DAYS}일",
        (Limit("year_turn_days", "제출 후 (일)", 30, 180, 10),),
        part="점검 라벨",
    ),
    Rule(
        "crash",
        "표시 지표",
        "모멘텀 붕괴 주의",
        CRASH_HELP + " · 분포 위에 경고만 표시, 점수·칸 배정 그대로",
        part="시장",
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


def peers_note(c: Criteria) -> str:
    return (
        f"업종: GICS 세부 산업, {c.peer_min}종목 미만이면 산업 → 산업 그룹 → 섹터"
        " 순으로 확대 · 백분위: Hazen (순위 − 0.5) ÷ 종목 수, 동률은 평균 순위"
    )


def tech_basis(c: Criteria) -> str:
    items = c.kept(TECH_ITEMS)
    return (
        "**정의** 켠 항목의 유동성 통과 종목 전체 안 백분위 평균(0–100) · "
        + (
            " · ".join(f"{name}: {TECH_HELP[key]}" for key, name in items.items())
            or "없음"
        )
        + "\n\n**해석** 높을수록 가격 추세 우위 · 값이 없는 항목은 평균에서 빠짐\n\n"
        "**기준** 수정주가 일봉(시가·고가·저가에 수정종가/종가 비율 적용) · 이동평균은"
        " 단순"
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
    names = _listed(list(c.kept(TECH_ITEMS).values()))
    return (
        f"**정의** {names}의 유동성 통과 종목 전체 안 백분위 평균(0–100){rules}\n\n"
        "**해석** 높을수록 가격 추세 우위 · 칸 배정에는 쓰지 않음"
    )


def _listed(names: Sequence[str]) -> str:
    return " · ".join(names) if names else "없음"


def _needs(c: Criteria) -> str:
    plain, financial = c.need(financial=False), c.need(financial=True)
    return f"{plain}개(금융업 {financial}개)"


def quality_help(c: Criteria) -> str:
    financial = "·".join(c.measures(QUALITY, financial=True)) or "없음"
    core = "·".join(QUALITY[key] for key in c.core)
    return (
        "**정의** 퀄리티 항목 업종 내 백분위의 평균을 업종 안에서 다시 백분위로 ·"
        f" {_listed(c.measures(QUALITY, financial=False))} · 금융업은 {financial}\n\n"
        f"**해석** 높을수록 우수 · {c.split:.0%} 이상 퀄리티 상위\n\n"
        f"**기준** 항목 {_needs(c)} 이상"
        + (f" · 비금융은 {core} 중 1개 필수" if core else "")
        + f" · {peers_note(c)}"
    )


def valuation_help(c: Criteria) -> str:
    financial = "·".join(c.measures(VALUATION, financial=True)) or "없음"
    return (
        "**정의** 밸류에이션 항목(가격 대비 수익률) 업종 내 백분위의 평균을 업종 안에서"
        f" 다시 백분위로 · {_listed(c.measures(VALUATION, financial=False))} · 금융업은"
        f" {financial}\n\n"
        f"**해석** 높을수록 고평가(수익률이 낮음) · {c.split:.0%} 미만 저평가\n\n"
        f"**기준** 항목 {c.need_valuation(financial=False)}개 이상 · 음수 수익률은 가장"
        f" 비싼 쪽 · {peers_note(c)}"
    )


def labels_help(c: Criteria) -> str:
    hold = "·".join(hold_rules(c))
    return (
        "**정의** 퀄리티·밸류에이션 항목별 업종 백분위와 보류 사유 점검\n\n"
        "**해석** 퀄리티 높을수록 우수 · 밸류에이션 높을수록 고평가 ·"
        f" 경계 {c.split:.0%}"
        + (f" · {hold}이면 보류" if hold else "")
        + f"\n\n**기준** {peers_note(c)}"
    )


def surge_help(c: Criteria) -> str:
    flags = list(c.kept(SURGE).values())
    return (
        f"**정의** 단기 급등 신호 {len(flags)}개, 각각 상위 {c.surge_share:.0%}"
        + "".join(f" · {flag}" for flag in flags)
        + "\n\n**해석** 기술 점수와 별개 · 매도 신호가 아니라 진입 시점 참고용"
    )


def dq_help(c: Criteria) -> str:
    checks = list(c.kept(DQ_RULES).values())
    return (
        f"**정의** 결격 점검 {len(checks)}개"
        + "".join(f" · {check}" for check in checks)
        + "\n\n**해석** 결격·미확인·해석 부적절 1개 이상이면 칸 미배정 · 순현금은 통과"
        " · 이자비용을 끝까지 못 찾은 이자보상배율은 미적용(칸 배정 유지)"
    )


def tag_names(c: Criteria) -> list[str]:
    """Every label the 라벨 column can show, as the criteria in force set them."""
    return [
        *hold_rules(c),
        "퀄리티·밸류에이션 산출 불가",
        *c.kept(LABEL_RULES).values(),
        "짧은 창",
    ]


def column_help(c: Criteria) -> dict[str, str]:
    """The results table's column tooltips."""
    return {
        "group": "**정의** 퀄리티 × 밸류에이션 업종 백분위 칸 · 보류 ·"
        " 결격·미확인은 빈칸",
        "quality": quality_help(c),
        "valuation": valuation_help(c),
        "est": ESTIMATE_HELP,
        "weak_spot": "**정의** 가장 불리한 항목 · 퀄리티는 가장 낮은 백분위,"
        " 밸류에이션은 가장 높은 백분위\n\n"
        f"**해석** (비쌈): 밸류에이션 백분위 {c.split:.0%} 이상",
        "tags": "**정의** " + " · ".join(tag_names(c)),
        "verdict": dq_help(c),
        "score": score_help(c),
        "chart": score_help(c),
        "hot": surge_help(c),
        "fwd_pe": "**정의** 주가 ÷ 선행 EPS(FY1·FY2를 남은 기간으로 섞은 12개월 값)",
        "combined": "**정의** 퀄리티 + (100 − 밸류에이션), 0–200\n\n"
        "**해석** 높을수록 좋은 회사를 싸게 · 전체 칸의 정렬 기준",
        "peer_level": f"**정의** 백분위를 낸 비교 업종 단계 · {peers_note(c)}",
        "counted": "**정의** 산출된 퀄리티 / 밸류에이션 항목 수",
        "mom12_1": f"**정의** {TECH_HELP['t_mom']}",
        "rev_strength": "**정의** (현재 − 90일 전 컨센서스 EPS) ÷ 주가 · FY1·FY2 평균"
        "(결산 대기면 FY2만)",
        "cbop": f"**정의** {QUALITY_HELP['q_cbop']}",
        "coverage": "**정의** EBIT(TTM) ÷ 이자비용 · 이자비용 0이면 비움",
        "fcf_cum": f"**정의** 최근 {c.fcf_years}개 회계연도 FCF 합계 · SEC 10-K"
        " 영업현금흐름 − 설비투자, 없으면 야후 연간",
        "sources": "**정의** SEC 10-K 값으로 대체하거나 교차 확인한 항목",
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
    """One stock's Yahoo statements; columns are period ends, newest first.

    ``income`` and ``cash_yearly`` are yearly, ``quarterly`` the quarterly
    income statement, ``trailing`` and ``cash`` the TTM income and cash flow
    statements, ``balance`` the quarterly and ``balance_yearly`` the yearly
    balance sheet. A statement Yahoo did not send is an empty frame.
    """

    income: pd.DataFrame
    quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)
    trailing: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash_yearly: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_yearly: pd.DataFrame = field(default_factory=pd.DataFrame)


def _epoch_day(value: object) -> str:
    """Yahoo's epoch seconds as an ISO date, "" when missing."""
    seconds = _number(value)
    if math.isnan(seconds):
        return ""
    return datetime.fromtimestamp(seconds, tz=ZoneInfo("UTC")).date().isoformat()


def _newest_first(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_index(axis=1, ascending=False) if not table.empty else table


def fundamentals_row(info: Mapping[str, object], s: Statements) -> dict[str, object]:
    """``info`` numbers, statement values and ratios for one stock.

    Yahoo keeps only five quarters, too few for a TTM year over year, so
    revenue growth is the last fiscal year over the one before. ``margin_avg``
    (an older name, kept for the stored files) is the median operating margin
    of the fiscal years with both values and revenue above zero,
    ``margin_years`` how many there were and ``margin_std`` their standard
    deviation. ``op_margin`` is the TTM operating margin from the quarterly
    income statement: OperatingIncome over TotalRevenue summed across the
    latest four quarters, NaN unless all four have both values and the
    revenue is above zero (``info["operatingMargins"]`` is only the latest
    quarter). ``fcf_y*`` are yearly free cash flows and ``opa_y*`` yearly
    operating income over the same year's total assets, newest first.
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
    income = s.income
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
    row["margin_std"] = statistics.stdev(margins) if len(margins) > 1 else math.nan
    row["op_margin"], row["op_margin_quarters"], row["ttm_end"] = ttm_margin(
        s.quarterly
    )
    row["eq_ratio"], row["tce_ratio"] = capital_ratios(s.balance)
    for column, name in TTM_ROWS.items():
        row[column] = _row_value(s.trailing, name)
    for column, name in CASH_ROWS.items():
        row[column] = _row_value(s.cash, name)
    balance = _newest_first(s.balance)
    for column, name in BALANCE_ROWS.items():
        row[column] = _row_value(balance, name)
    flows = _newest_first(s.cash_yearly)
    for year in range(ANNUAL_YEARS):
        row[f"fcf_y{year}"] = _row_value(flows, "FreeCashFlow", year)
    yearly = _newest_first(income)
    for year in range(ANNUAL_YEARS):
        end = yearly.columns[year] if year < len(yearly.columns) else None
        same = (
            cast("pd.DataFrame", s.balance_yearly[[end]])
            if end in s.balance_yearly.columns
            else pd.DataFrame()
        )
        total = _row_value(same, "TotalAssets")
        profit = _row_value(yearly, "OperatingIncome", year)
        row[f"opa_y{year}"] = profit / total if total > 0 else math.nan
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


def instants(facts: Mapping[str, object], concept: str) -> Values:
    """A us-gaap concept's 10-K year-end USD values (a balance), by date.

    Only form 10-K, fp FY and no start count; the latest filed value of each
    date wins.
    """
    units = _get(_get(_get(facts.get("facts"), "us-gaap"), concept), "units")
    entries = _get(units, "USD")
    out: Values = {}
    for item in cast("list[object]", entries if isinstance(entries, list) else []):
        end, filed = _get(item, "end"), _get(item, "filed")
        value = _number(_get(item, "val"))
        if (
            _get(item, "start") is None
            and isinstance(end, str)
            and isinstance(filed, str)
            and _get(item, "form") == "10-K"
            and _get(item, "fp") == "FY"
            and not math.isnan(value)
        ):
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

    Every year end of ``OPERATING_INCOME``, a revenue concept or one of
    ``SEC_FLOWS`` and ``SEC_INSTANTS`` is an "FY" row, newest first, with the
    first of each column's concepts that has that year. The revenue concept
    is the one in ``REVENUE_CONCEPTS``
    with the most years in ``sec_window`` that also have operating income,
    the earlier on a tie; a year without it takes the next concept that has
    one. The newest ``TTM_QUARTERS`` quarter ends of any value are "Q" rows,
    revenue in the first concept of that order with all of them, else in the
    first with that quarter. A quarter's revenue above ``QUARTER_SHARE`` of
    its fiscal year's in the same concept is dropped as mistagged.
    """
    income = annual(facts, OPERATING_INCOME)
    revenues = {concept: annual(facts, concept) for concept in REVENUE_CONCEPTS}
    others = {
        column: [annual(facts, concept) for concept in concepts]
        for column, concepts in SEC_FLOWS.items()
    } | {
        column: [instants(facts, concept) for concept in concepts]
        for column, concepts in SEC_INSTANTS.items()
    }
    found = [values for series in others.values() for values in series]
    ends = sorted(set(income).union(*revenues.values(), *found), reverse=True)
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
        extra = {
            column: next((v[end] for v in series if end in v), (math.nan, ""))
            for column, series in others.items()
        }
        rows.append(
            {
                "period": "FY",
                "end": end,
                "revenue": revenue,
                "op_income": op_income,
                **{column: value for column, (value, _) in extra.items()},
                "concept": concept,
                "filed": max(filed, op_filed, *(f for _, f in extra.values())),
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
    "sec_std",
    "sec_ttm",
    "sec_ttm_end",
    "long_med",
    "long_mad",
    "long_std",
    "long_years",
    *(f"sec_opa_y{year}" for year in range(ANNUAL_YEARS)),
    *(f"sec_fcf_y{year}" for year in range(ANNUAL_YEARS)),
    "sec_roe_std",
    "sec_interest",
    "sec_interest_paid",
    "sec_fy_end",
    "sec_filed",
)


def _spread(values: pd.Series, least: int = 2) -> float:
    return float(cast("float", values.std())) if len(values) >= least else math.nan


def _kept(values: pd.Series, mask: pd.Series) -> pd.Series:
    """``values`` where ``mask`` holds, without the missing ones."""
    return cast("pd.Series", values[mask]).dropna()


def sec_row(years: pd.DataFrame, quarters: pd.DataFrame) -> dict[str, object]:
    """One ticker's SEC figures from its "FY" and "Q" rows.

    Margins count in years with operating income and revenue above zero:
    the ``sec_window`` median, its years and standard deviation, and over
    ``LONG_WINDOW`` years the median, median absolute deviation and standard
    deviation. By year back in the window, ``sec_opa_y*`` is operating income
    over year-end assets and ``sec_fcf_y*`` operating cash flow less capex;
    ``sec_roe_std`` the deviation of net income over year-end equity.
    ``sec_interest``, ``sec_interest_paid``, ``sec_fy_end`` and ``sec_filed``
    are the newest year's.
    """
    years = years.sort_values("end", ascending=False)
    ends = cast("pd.Series", years["end"]).astype("str")
    margin = cast("pd.Series", years["op_income"] / years["revenue"]).where(
        (years["revenue"] > 0) & years["op_income"].notna()
    )
    window = _kept(margin, ends.isin(sec_window(list(ends))))
    long = _kept(margin, ends.isin(sec_window(list(ends), LONG_WINDOW)))
    long_med = float(long.median())
    profit = cast("pd.Series", years["op_income"] / years["assets"]).where(
        years["assets"] > 0
    )
    recent = ends.isin(sec_window(list(ends)))
    opa = cast("pd.Series", profit[recent]).tolist()
    fcf = cast("pd.Series", years["ocf"] - years["capex"])[recent].tolist()
    roe = cast("pd.Series", years["net_income"] / years["equity"]).where(
        years["equity"] > 0
    )
    newest = years.iloc[0] if len(years) else pd.Series(dtype="object")
    return {
        "sec_avg": float(window.median()),
        "sec_years": len(window),
        "sec_std": _spread(window),
        "sec_ttm": sec_ttm(quarters, str(ends.max()) if len(ends) else ""),
        "sec_ttm_end": quarters["end"].max(),
        "long_med": long_med,
        "long_mad": float((long - long_med).abs().median()),
        "long_std": _spread(long),
        "long_years": len(long),
        **{
            f"sec_opa_y{year}": opa[year] if year < len(opa) else math.nan
            for year in range(ANNUAL_YEARS)
        },
        **{
            f"sec_fcf_y{year}": fcf[year] if year < len(fcf) else math.nan
            for year in range(ANNUAL_YEARS)
        },
        "sec_roe_std": _spread(_kept(roe, recent), STABILITY_YEARS),
        "sec_interest": _number(newest.get("interest")),
        "sec_interest_paid": _number(newest.get("interest_paid")),
        "sec_fy_end": str(newest.get("end", "")),
        "sec_filed": str(newest.get("filed", "")),
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
    """One ticker's indicators on its last bar; ``frame`` is sorted by date.

    Fewer than ``MIN_BARS`` bars give only the liquidity inputs, fewer than
    ``MOMENTUM_BARS`` no 12-1 momentum.
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
    for days in SMA_DAYS:
        row[f"sma{days}"] = float(sma(close, days).iloc[-1])
    high52 = float(cast("pd.Series", adj["high"]).tail(YEAR_BARS).max())
    last = float(close.iloc[-1])
    month = close.iloc[-MONTH_BARS - 1 :]
    row |= {
        "close": last,
        "high52": high52,
        "from_high": last / high52 - 1,
        "near_high": last / high52,
        "ma_dist": row["sma21"] / row["sma200"],
        "mom12_1": float(close.iloc[-MONTH_BARS - 1] / close.iloc[-YEAR_BARS - 1] - 1)
        if bars >= MOMENTUM_BARS
        else math.nan,
        "ret_1m": float(last / close.iloc[-MONTH_BARS - 1] - 1),
        "ret_3m": float(last / close.iloc[-64] - 1),
        "max_1m": float(month.pct_change().max()),
    }
    return row


@dataclass(frozen=True, slots=True)
class Crash:
    """The momentum crash notice: ``state`` is 주의, 정상 or 미산출."""

    state: str = NOT_COMPUTED
    market_return: float = math.nan  # over ``BEAR_BARS`` days
    variance: float = math.nan  # of daily returns over ``VARIANCE_BARS`` days
    median: float = math.nan  # of that variance over the stored days


CRASH_WATCH: Final = "주의"
CRASH_CLEAR: Final = "정상"


def momentum_crash(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> Crash:
    """The market's 24-month return and variance state, from the stored prices.

    The market is the cap-weighted daily return of the stocks with a market
    cap: each day weighted by the cap implied by today's cap and the adjusted
    close's path. Fewer than ``BEAR_BARS`` returns leave it 미산출.
    """
    if prices.empty or "mcap" not in fundamentals.columns:
        return Crash()
    every = cast("pd.Series", fundamentals.set_index("ticker")["mcap"])
    caps = cast("pd.Series", every[every > 0])
    wide = prices.pivot_table(index="date", columns="ticker", values="adj_close")
    wide = cast("pd.DataFrame", wide[[t for t in wide.columns if t in caps.index]])
    if wide.empty:
        return Crash()
    implied = wide / wide.ffill().iloc[-1] * caps.reindex(wide.columns)
    returns = wide.pct_change(fill_method=None)
    weights = implied.shift(1).where(returns.notna())
    market = ((returns * weights).sum(axis=1) / weights.sum(axis=1)).iloc[1:].dropna()
    if len(market) < BEAR_BARS:
        return Crash()
    total = float((1 + market.tail(BEAR_BARS)).prod() - 1)
    variance = market.rolling(VARIANCE_BARS).var().dropna()
    now, median = float(variance.iloc[-1]), float(variance.median())
    watch = total < 0 and now > median
    return Crash(CRASH_WATCH if watch else CRASH_CLEAR, total, now, median)


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


def peers(table: pd.DataFrame, least: int = PEER_MIN) -> pd.Series:
    """The narrowest GICS level with ``least`` members here, else the sector."""
    levels = gics_levels(table)
    peer = cast("pd.Series", levels["sector"]).copy()
    for level in ("group", "industry", "sub"):
        labels = cast("pd.Series", levels[level])
        peer = labels.where(labels.map(labels.value_counts()) >= least, peer)
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


def broad_percentile(values: pd.Series) -> pd.Series:
    """Hazen percentile among all the rows (the liquid stocks), not the peers."""
    return (values.rank(method="average") - 0.5) / int(values.notna().sum())


SHORT_HISTORY: Final = "이력 부족(분사·신규 상장 가능)"


def liquidity(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.Series:
    """Why a stock is left out ("" when it stays): history, price, then volume.

    No prices at all always leaves a stock out; the three filters only when on.
    """
    reason = pd.Series("", index=table.index, dtype="str")
    reason[table["bars"].isna()] = "시세 없음"
    if c.on("min_bars"):
        short = (reason == "") & ~(table["bars"] >= c.min_bars)
        reason[short] = f"{SHORT_HISTORY} · 일봉 {c.min_bars}개 미만"
    if c.on("min_price"):
        cheap = (reason == "") & ~(table["price"] >= c.min_price)
        reason[cheap] = f"주가 ${c.min_price:g} 미만"
    if c.on("illiquid"):
        rest = reason == ""
        volume = cast("pd.Series", table["dollar_vol20"]).where(rest)
        thin = rest & bottom_share(volume, c.illiquid_share)
        reason[thin] = f"거래대금 하위 {c.illiquid_share:.0%}"
    return reason


def tech_score(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.DataFrame:
    """Each 기술 점수 measure's percentile, the score (0 to 100) and 차트 강/약.

    Percentiles are over the rows (the liquid stocks). The score is the mean
    of the measures' percentiles in force, NaN without any; with none in
    force no stock is 강 or 약.
    """
    t = table
    ranks = pd.DataFrame(
        {
            key: broad_percentile(cast("pd.Series", t[column]))
            for key, column in TECH_VALUES.items()
        },
        index=t.index,
    )
    kept = list(c.kept(TECH_ITEMS))
    score = (
        ranks[kept].mean(axis=1) * 100 if kept else pd.Series(math.nan, index=t.index)
    )
    none = pd.Series(data=False, index=t.index)
    strong = top_share(score, c.strong_share) if c.on("strong") else none
    weak = bottom_share(score, c.weak_share) & ~strong if c.on("weak") else none
    ranks["score"] = score
    ranks["chart"] = np.select([strong, weak], ["강", "약"], "")
    return ranks


def surge(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.DataFrame:
    """단기 급등: the top ``surge_share`` of 21-day return and MAX, where on."""
    return pd.DataFrame(
        {
            key: top_share(cast("pd.Series", table[column]), c.surge_share) & c.on(key)
            for key, column in SURGE_VALUES.items()
        },
        index=table.index,
    )


def chart_band(table: pd.DataFrame) -> pd.Series:
    """차트 강 · 중간 · 차트 약 · 미산출, the map's default coloring."""
    t = table
    band = np.select(
        [t["score"].isna(), t["chart"] == "강", t["chart"] == "약"],
        [NOT_COMPUTED, CHART_STRONG, CHART_WEAK],
        CHART_MIDDLE,
    )
    return pd.Series(band, index=t.index, dtype="str")


def _days(day: object) -> pd.Series:
    return pd.to_datetime(cast("pd.Series", day), errors="coerce")


def year_state(table: pd.DataFrame) -> pd.DataFrame:
    """Where each stock stands in its fiscal year, as of ``as_of``.

    ``pending``: the year Yahoo calls "0y" (ending ``next_fy_end``) has ended
    and is not yet reported. ``weight`` is FY1's share of the forward 12
    months: the year's time left over a year, 0 when pending, 1 without dates.
    ``turn_days`` (for 연도 교체 주의) is the days since the newest year's
    10-K was filed; without SEC's 10-K for that year (none, or still the year
    before), since that year ended.
    """
    t = table
    as_of = _days(t["as_of"])
    next_end, last_end = _days(t["next_fy_end"]), _days(t["last_fy_end"])
    pending = next_end <= as_of
    left = (next_end - as_of).dt.days / 365.25
    weight = left.clip(0, 1).fillna(1.0).where(~pending, 0.0)
    same_year = (_days(t["sec_fy_end"]) - last_end).abs().dt.days <= (
        YEAR_END_SLACK_DAYS
    )
    since = _days(t["sec_filed"]).where(same_year).fillna(last_end)
    return pd.DataFrame(
        {"pending": pending, "weight": weight, "turn_days": (as_of - since).dt.days},
        index=t.index,
    )


def forward_eps(table: pd.DataFrame) -> pd.Series:
    """12-month forward EPS: ``weight`` * FY1 + (1 - ``weight``) * FY2.

    With one of the two missing, the other alone.
    """
    t = table
    fy1, fy2 = t["fy1_now"], t["fy2_now"]
    blend = t["weight"] * fy1 + (1 - t["weight"]) * fy2
    return cast("pd.Series", blend.fillna(fy1).fillna(fy2))


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


def estimate_direction(table: pd.DataFrame) -> pd.Series:
    """상향, 하향 or 보합 from 수정 강도 and 순수정 비율; 미확인 without either."""
    t = table
    strength, net = t["rev_strength"], t["net_revision"]
    direction = np.select(
        [
            strength.isna() | net.isna(),
            (strength > 0) & (net >= 0),
            (strength < 0) & (net < 0),
        ],
        [UNKNOWN, EST_UP, EST_DOWN],
        EST_FLAT,
    )
    return pd.Series(direction, index=t.index, dtype="str")


def cumulative_fcf(table: pd.DataFrame, years: int) -> pd.DataFrame:
    """FCF summed over the newest ``years`` fiscal years, and where it came from.

    SEC (operating cash flow less capex) when it has all of them, else Yahoo
    when it does, else the one with more years if at least
    ``FCF_YEARS_LEAST`` (SEC on a tie), else the TTM FCF alone.
    """
    t = table
    sec = t[[f"sec_fcf_y{year}" for year in range(years)]]
    yahoo = t[[f"fcf_y{year}" for year in range(years)]]
    sec_n, yahoo_n = sec.notna().sum(axis=1), yahoo.notna().sum(axis=1)
    use_sec = (sec_n == years) | ((yahoo_n < years) & (sec_n >= yahoo_n))
    count = sec_n.where(use_sec, yahoo_n)
    some = count >= FCF_YEARS_LEAST
    total = sec.sum(axis=1).where(use_sec, yahoo.sum(axis=1))
    return pd.DataFrame(
        {
            "fcf_cum": total.where(some, t["fcf"]),
            "fcf_cum_years": count.where(some, 0),
            "fcf_source": np.select(
                [some & use_sec, some, t["fcf"].notna()], ["SEC", "야후", "TTM"], ""
            ),
        },
        index=t.index,
    )


def interest_expense(table: pd.DataFrame) -> pd.DataFrame:
    """Interest expense: Yahoo's TTM, else SEC's 10-K expense, else interest paid."""
    t = table
    yahoo, sec, paid = t["interest"].abs(), t["sec_interest"], t["sec_interest_paid"]
    return pd.DataFrame(
        {
            "interest_used": yahoo.fillna(sec).fillna(paid),
            "interest_source": np.select(
                [yahoo.notna(), sec.notna(), paid.notna()],
                ["야후", "SEC", "SEC 이자 지급액"],
                "",
            ),
        },
        index=t.index,
    )


def fundamentals_status(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.DataFrame:
    """Disqualifiers and their peer medians, for liquid stocks.

    A check whose input is missing is 미확인, and one with a negative or zero
    base is 해석 부적절; either withholds the stock from every cell. The
    interest coverage without an interest expense (or EBIT) is left out
    instead, without withholding. A check turned off is 미적용.
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
    flows = cumulative_fcf(t, c.fcf_years)
    out[list(flows.columns)] = flows
    cash_only = t["sector"].isin(list(CASH_ONLY))
    cash_fail = (t["ocf"] < 0) | (~cash_only & (out["fcf_cum"] < 0))
    cash_unknown = t["ocf"].isna() | (~cash_only & out["fcf_cum"].isna())
    out["dq_cash"] = np.select(
        [
            bank & bank_fail,
            bank & bank_unknown,
            bank,
            other & other_fail,
            other & other_unknown,
            other,
            cash_fail,
            cash_unknown,
        ],
        [FAIL, UNKNOWN, CLEAR, FAIL, UNKNOWN, CLEAR, FAIL, UNKNOWN],
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
        [NOT_APPLICABLE, UNKNOWN, NET_CASH, UNKNOWN, FAIL, FAIL],
        CLEAR,
    )

    interest = interest_expense(t)
    out[list(interest.columns)] = interest
    ebit = t["ebit"].fillna(t["op_income_ttm"])
    used = out["interest_used"]
    out["coverage"] = (ebit / used).where(used > 0)
    out["dq_cover"] = np.select(
        [
            financial,
            used.isna(),
            used == 0,
            ebit.isna(),
            out["coverage"] < c.coverage_floor,
        ],
        [NOT_APPLICABLE, NO_INTEREST, CLEAR, NO_EBIT, FAIL],
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
    return out


def hold_flags(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.DataFrame:
    """경기순환, the margin basis, 정점 이익, 업황 순풍 and 마진 급등.

    The basis is the ``LONG_WINDOW``-year median (``long_med``) with
    ``long_years`` years, else the 5-year one (``margin_avg``) with
    ``short_years`` (``short``: 짧은 창), else none and no flag. The surge test
    is the robust z and gap on the long basis, the ratio and gap on the short.
    Financials have none of these.
    """
    t = table
    plain = t["sector"] != FINANCIALS
    groups = {sub: group for sub, (_, group) in GICS.items()}
    long_ok = plain & (t["long_years"] >= c.long_years)
    short = (
        plain
        & ~long_ok
        & (t["margin_years"] >= c.short_years)
        & t["margin_avg"].notna()
    )
    median = cast(
        "pd.Series", t["long_med"].where(long_ok, t["margin_avg"].where(short))
    )
    deviation = cast("pd.Series", t["long_std"]).where(long_ok)
    cyclical = plain & (
        t["sector"].isin(list(CYCLICALS))
        | cast("pd.Series", t["industry"]).map(groups).isin(list(CYCLICAL_GROUPS))
        | top_share(deviation, c.cyclical_share)
    )
    ttm = t["op_margin"]
    z = (ttm - median) / (MAD_SCALE * t["long_mad"])
    jumped = (long_ok & (z >= c.tailwind_z) & (ttm - median >= c.tailwind_gap)) | (
        short
        & (median > 0)
        & (ttm >= median * SHORT_RATIO)
        & (ttm >= median + SHORT_GAP)
    )
    eps, forward = t["trailing_eps"], t["fwd_eps"]
    peak = (
        cyclical
        & (eps > 0)
        & (median > 0)
        & (ttm >= median * c.peak_ratio)
        & (forward <= eps * c.peak_eps_ratio)
    )
    return pd.DataFrame(
        {
            "cyclical": cyclical,
            "hold_median": median,
            "hold_z": z.where(long_ok),
            "short": short,
            "peak": peak & c.on("peak"),
            "tailwind": cyclical & jumped & c.on("tailwind"),
            "margin_jump": plain & ~cyclical & jumped & c.on("l_margin_jump"),
        },
        index=t.index,
    )


def measure_values(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Each 퀄리티 and 밸류에이션 measure's value, higher better and cheaper.

    Lower-is-better measures (deviations, leverage) are negated; valuation
    yields are negated so their percentile rises with the price.
    """
    t = table
    financial = t["sector"] == FINANCIALS
    plain = ~financial
    bank = financial_group(t) == BANKS
    assets = t["total_assets"].where(t["total_assets"] > 0)
    gross = t["gross_profit"].fillna(t["revenue_ttm"] - t["cost_of_revenue"])
    price = t["price"].where(t["price"] > 0)
    ev = t["ev"].where(t["ev"] > 0)
    book = (t["tbv"] / t["shares"].where(t["shares"] > 0)).where(bank, t["book_value"])
    values = {
        "q_cbop": (
            (t["op_income_ttm"] + t["rnd"].fillna(0) + t["wc_change"]) / assets
        ).where(plain),
        "q_gpa": (gross / assets).where(plain),
        "q_improve": t["improve"].where(plain),
        "q_stability": -t["margin_spread"].where(plain),
        "q_leverage": -(t["total_debt"] / assets).where(plain),
        "q_roe": t["roe"].where(financial & (t["book_value"] > 0)),
        "q_roa": (t["net_income"] / assets).where(financial),
        "q_capital": t["tce_ratio"].where(bank, t["eq_ratio"]).where(financial),
        "q_roe_stability": -t["sec_roe_std"].where(financial),
        "v_fwd": -(t["fwd_eps"] / price),
        "v_ttm": -(t["trailing_eps"] / price),
        "v_ebitda": -(t["ebitda"] / ev).where(plain),
        "v_fcf": -(t["fcf"] / ev).where(plain),
        "v_book": -(book / price).where(financial),
    }
    return cast("dict[str, pd.Series]", values)


def axes(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.DataFrame:
    """퀄리티 and 밸류에이션: each measure's peer percentile, their mean, re-ranked.

    A measure turned off is NaN. 퀄리티 needs ``Criteria.need`` measures, a
    non-financial one of ``Criteria.core`` among them; 밸류에이션 needs
    ``Criteria.need_valuation``; else NaN. The mean is then a Hazen percentile
    within the peers again. ``edge`` is 퀄리티 less 밸류에이션 and ``combined``
    퀄리티 + (1 - 밸류에이션), in points (0 to 200).
    """
    t = table
    financial = t["sector"] == FINANCIALS
    values = measure_values(t)
    out = pd.DataFrame(
        {
            key: percentile(value, t)
            if c.on(key)
            else pd.Series(math.nan, index=t.index)
            for key, value in values.items()
        },
        index=t.index,
    )
    out["cbop"] = values["q_cbop"]
    quality = cast("pd.DataFrame", out[list(QUALITY)])
    valuation = cast("pd.DataFrame", out[list(VALUATION)])
    out["q_count"] = quality.count(axis=1)
    out["v_count"] = valuation.count(axis=1)
    need = np.where(financial, c.need(financial=True), c.need(financial=False))
    core = (
        financial | out[c.core].notna().any(axis=1)
        if c.core
        else pd.Series(data=True, index=t.index)
    )
    need_v = np.where(
        financial, c.need_valuation(financial=True), c.need_valuation(financial=False)
    )
    quality_mean = quality.mean(axis=1).where((out["q_count"] >= need) & core)
    valuation_mean = valuation.mean(axis=1).where(out["v_count"] >= need_v)
    out["quality"] = percentile(quality_mean, t)
    out["valuation"] = percentile(valuation_mean, t)
    out["edge"] = out["quality"] - out["valuation"]
    out["combined"] = (out["quality"] + 1 - out["valuation"]) * 100
    return out


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


def labels(table: pd.DataFrame, c: Criteria = DEFAULT) -> pd.DataFrame:
    """The display-only labels; 짧은 창 from the hold basis or 수익성 개선."""
    t = table
    eps, forward = t["trailing_eps"], t["fwd_eps"]
    band = c.near_band
    near = (t["dq_state"] == CLEAR) & (
        ((t["quality"] - c.split).abs() <= band)
        | ((t["valuation"] - c.split).abs() <= band)
    )
    down = t["est"] == EST_DOWN
    lagging = bottom_share(cast("pd.Series", t["mom12_1"]), c.weak_trend_share)
    improve_short = t["q_improve"].notna() & (t["improve_years"] < c.improve_years)
    return pd.DataFrame(
        {
            "revdrop": (t["revenue_prior"] > 0)
            & (t["rev_growth"] <= c.revenue_floor)
            & down
            & c.on("l_revdrop"),
            "reliance": (
                ((eps > 0) & (forward >= eps * c.reliance_ratio))
                | ((eps <= 0) & (forward > 0))
            )
            & c.on("l_reliance"),
            "near": near & c.on("l_near"),
            "weak_trend": t["group"].isin(["prime", "trap"])
            & lagging
            & down
            & c.on("l_weak_trend"),
            "short_window": t["short"] | improve_short,
        },
        index=t.index,
    )


def evaluate(inputs: Inputs, criteria: Criteria = DEFAULT) -> pd.DataFrame:
    """Each constituent with liquidity, score, checks, 퀄리티, 밸류에이션, cell."""
    return judge(prepare(inputs), criteria)


def prepare(inputs: Inputs) -> pd.DataFrame:
    """The stored parts as one row per constituent with its indicators.

    Nothing here depends on the criteria. The operating margin median
    (``margin_avg``) and its deviation (``margin_spread``) are SEC's
    ``SEC_YEARS`` 10-K years where they are all there, else Yahoo's
    (``margin_source``). A stock on SEC's median takes SEC's TTM too when its
    four quarters are there and end no earlier than Yahoo's newest quarter
    (``STALE_SLACK_DAYS`` aside), else Yahoo's (``op_margin_source``);
    ``op_margin_end`` is the end of the used TTM's newest quarter.
    ``improve`` (수익성 개선) is SEC's, else Yahoo's yearly operating income
    over assets: three years back, else two. ``as_of`` is the newest price
    day, which the fiscal year state and the forward EPS go by.
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
    for column in (
        "currency",
        "est_currency",
        "rev_year",
        "ttm_end",
        "last_fy_end",
        "next_fy_end",
        "sec_fy_end",
        "sec_filed",
    ):
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
    table["margin_spread"] = table["sec_std"].where(
        sec, table["margin_std"].where(table["margin_years"] >= MARGIN_YEARS)
    )
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
    table["fwd_eps"] = forward_eps(table)
    table["fwd_pe"] = (table["price"] / table["fwd_eps"]).where(table["fwd_eps"] > 0)
    revised = revisions(table)
    table[list(revised.columns)] = revised
    table["est"] = estimate_direction(table)
    table["error"] = _errors(table)
    return table.drop(columns=["error_estimates"], errors="ignore")


def improvement(table: pd.DataFrame, years: int) -> pd.DataFrame:
    """수익성 개선: operating income / assets, newest less ``years`` back.

    A year short, ``years`` - 1 back (짧은 창); SEC's 10-K first, else
    Yahoo's yearly statements.
    """
    t = table
    changes: dict[str, tuple[pd.Series, pd.Series]] = {}
    for source, prefix in (("SEC", "sec_opa_y"), ("야후", "opa_y")):
        full = t[f"{prefix}0"] - t[f"{prefix}{years}"]
        short = t[f"{prefix}0"] - t[f"{prefix}{years - 1}"]
        span = pd.Series(np.where(full.notna(), years, years - 1), index=t.index)
        changes[source] = (full.fillna(short), span)
    (sec, sec_span), (yahoo, yahoo_span) = changes["SEC"], changes["야후"]
    from_sec = sec.notna()
    found = sec.where(from_sec, yahoo)
    return pd.DataFrame(
        {
            "improve": found,
            "improve_years": sec_span.where(from_sec, yahoo_span).where(found.notna()),
            "improve_source": np.select([from_sec, yahoo.notna()], ["SEC", "야후"], ""),
        },
        index=t.index,
    )


def judge(prepared: pd.DataFrame, c: Criteria) -> pd.DataFrame:
    """``prepare``'s table screened by ``c``: liquidity, score, checks and cell."""
    table = prepared.copy()
    table["liquidity"] = liquidity(table, c)
    liquid = cast("pd.DataFrame", table[table["liquidity"] == ""]).copy()

    for part in (tech_score(liquid, c), surge(liquid, c)):
        liquid[list(part.columns)] = part
    liquid["band"] = chart_band(liquid)
    liquid["peer"] = peers(liquid, c.peer_min)
    liquid["peer_n"] = within_peers(pd.Series(1, index=liquid.index), liquid, "size")
    turn = liquid["pending"] | (liquid["turn_days"] <= c.year_turn_days)
    liquid["year_turn"] = turn & c.on("l_year_turn")
    improved = improvement(liquid, c.improve_years)
    liquid[list(improved.columns)] = improved
    for part in (
        fundamentals_status(liquid, c),
        hold_flags(liquid, c),
        axes(liquid, c),
    ):
        liquid[list(part.columns)] = part
    liquid["group"] = groups(liquid, c)
    found = labels(liquid, c)
    liquid[list(found.columns)] = found

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
    those sectors; ``"all"`` keeps every stock, by 통합 점수 then market cap.
    """
    rows = liquid(table)
    if chosen:
        rows = cast("pd.DataFrame", rows[rows["sector"].isin(list(chosen))])
    if key == "all":
        columns, ascending = zip(*GROUP_ORDER[key], strict=True)
        rows = rows.sort_values(list(columns), ascending=list(ascending))
    else:
        rows = group(rows, key)
    return rows.reset_index(drop=True)


def map_rows(table: pd.DataFrame) -> list[dict[str, object]]:
    """Stocks in the four cells, and held ones with both axes, as map points.

    Each point carries what the map colors, highlights and its tooltip show.
    """
    cells = [key for _, keys in MAP_ROWS for key in keys]
    placed = table[
        table["group"].isin(cells)
        | (
            (table["group"] == "hold")
            & table["quality"].notna()
            & table["valuation"].notna()
        )
    ]
    return [
        {
            "ticker": str(row["ticker"]),
            "name": str(row["name"]),
            "group": str(row["group"]),
            "cell": GROUP_NAMES[str(row["group"])],
            "held": row["group"] == "hold",
            "quality": float(cast("float", row["quality"])),
            "valuation": float(cast("float", row["valuation"])),
            "counted": counted(row),
            "peer": f"{peer_level(row)} · {row['peer']}",
            "score": _number(row.get("score")),
            "band": str(row.get("band", NOT_COMPUTED)),
            "est": str(row["est"]),
            "revision": _number(row.get("rev_strength")),
            "tags": tags(row),
            "labels": tag_list(row),
        }
        for row in records(cast("pd.DataFrame", placed))
    ]


def group_help(c: Criteria) -> str:
    """Every group's name and conditions, for a tooltip."""
    held = ", ".join(
        c.kept({"peak": "정점 이익(경기순환 기업)", "tailwind": "업황 순풍"}).values()
    )
    reasons = (
        f"{held} 또는 퀄리티·밸류에이션 산출 불가"
        if held
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
            "peer_level": [peer_level(row) for row in rows],
            "counted": [counted(row) for row in rows],
            "sources": [sources(row) for row in rows],
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


TAGS: Final = {
    "peak": "정점 이익",
    "tailwind": "업황 순풍",
    "margin_jump": "마진 급등",
    "reliance": "추정 의존",
    "revdrop": "매출 급감",
    "near": "경계 근처",
    "weak_trend": "가격·추정치 약세",
    "short_window": "짧은 창",
    "year_turn": "연도 교체 주의",
}


def tag_list(row: Mapping[str, object]) -> list[str]:
    """A stock's labels in ``TAGS`` order, axes not computed after the holds."""
    out = [name for key, name in TAGS.items() if row.get(key) is True]
    missing = [
        f"{name} 산출 불가"
        for key, name in (("quality", "퀄리티"), ("valuation", "밸류에이션"))
        if math.isnan(_number(row.get(key)))
    ]
    holds = sum(row.get(key) is True for key in ("peak", "tailwind"))
    return [*out[:holds], *missing, *out[holds:]]


def tags(row: Mapping[str, object]) -> str:
    return " · ".join(tag_list(row))


def counted(row: Mapping[str, object]) -> str:
    """산출 항목 수: 퀄리티 / 밸류에이션 measures computed."""
    q, v = _number(row.get("q_count")), _number(row.get("v_count"))
    return "" if math.isnan(q) or math.isnan(v) else f"{q:.0f}/{v:.0f}"


def sources(row: Mapping[str, object]) -> str:
    """The values taken or checked against SEC 10-K filings."""
    used = [
        name
        for key, name in (
            ("margin_source", "영업이익률"),
            ("op_margin_source", "TTM 영업이익률"),
            ("fcf_source", "누적 FCF"),
            ("improve_source", "수익성 개선"),
        )
        if row.get(key) == "SEC"
    ]
    if str(row.get("interest_source", "")).startswith("SEC"):
        used.append("이자비용")
    if row.get("short") is False and not math.isnan(_number(row.get("hold_median"))):
        used.append(f"{LONG_WINDOW}년 이익률")
    return f"SEC: {' · '.join(used)}" if used else "야후"


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
DQ_NAMES: Final = {"dq_cash": "현금흐름", "dq_debt": "부채", "dq_cover": "이자보상"}
LEFT_OUT: Final = frozenset({NO_INTEREST, NO_EBIT})


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
    if unsure:
        return ", ".join(unsure)
    left = [name for key, name in DQ_NAMES.items() if row.get(key) in LEFT_OUT]
    return f"{CLEAR}({', '.join(left)} 미적용)" if left else CLEAR


def hot(row: Mapping[str, object]) -> str:
    names = {"s_ret": "21일 수익률", "s_max": "MAX"}
    return " · ".join(name for key, name in names.items() if row.get(key) is True)


def tech_lines(row: Mapping[str, object], c: Criteria) -> list[tuple[str, str, str]]:
    """Each 기술 점수 measure in force: its percentile, the test and its value."""
    values = {
        "t_mom": fmt("signed", row.get("mom12_1")),
        "t_high": fmt("pct", row.get("near_high")),
        "t_trend": fmt("num", row.get("ma_dist"))
        + f" (21일선 {fmt('num', row.get('sma21'))} ÷ 200일선"
        f" {fmt('num', row.get('sma200'))})",
    }
    return [
        (_standing(row.get(key)), f"{name} · {TECH_HELP[key]}", values[key])
        for key, name in c.kept(TECH_ITEMS).items()
    ]


def surge_lines(row: Mapping[str, object], c: Criteria) -> list[tuple[bool, str, str]]:
    details = {
        "s_ret": fmt("signed", row.get("ret_1m")),
        "s_max": fmt("signed", row.get("max_1m")),
    }
    return [
        (row.get(key) is True, label, details[key])
        for key, label in c.kept(SURGE).items()
    ]


def dq_lines(row: Mapping[str, object], c: Criteria) -> list[tuple[str, str, str]]:
    """Each disqualifier: status, the test, and the values against it."""
    roe = f"ROE {fmt('pct', row.get('roe'))} (TTM)"
    share = f"{c.equity_median_share:.0%}"
    if GICS.get(str(row.get("industry")), ("", ""))[1] == BANKS:
        cash = (
            f"은행: ROE < 0 또는 TCE/TA < {percent(c.bank_tce_floor)}",
            f"{roe} · TCE/TA {fmt('pct', row.get('tce_ratio'))} (최근 분기,"
            " 유형 보통주자본 ÷ (총자산 − 영업권·무형자산))",
        )
    elif row.get("sector") == FINANCIALS:
        cash = (
            f"금융업: ROE < 0 또는 자기자본비율 < 업종 중앙값의 {share}",
            f"{roe} · 자기자본비율 {fmt('pct', row.get('eq_ratio'))} (최근 분기) ·"
            f" 기준 {fmt('pct', row.get('eq_limit'))}"
            f" ({row.get('peer')} 중앙값의 {share})",
        )
    elif row.get("sector") in CASH_ONLY:
        cash = (
            "영업현금흐름 < 0 (유틸리티·부동산)",
            f"영업현금흐름 {fmt('big', row.get('ocf'))} (TTM)",
        )
    else:
        source = str(row.get("fcf_source") or "—")
        years = _number(row.get("fcf_cum_years"))
        basis = (
            "FCF TTM"
            if source == "TTM"
            else f"{source} {years:.0f}개 연도"
            if not math.isnan(years)
            else source
        )
        cash = (
            f"영업현금흐름 < 0 또는 최근 {c.fcf_years}개 회계연도 누적 FCF < 0",
            f"영업현금흐름 {fmt('big', row.get('ocf'))} (TTM) · 누적 FCF"
            f" {fmt('big', row.get('fcf_cum'))} ({basis})",
        )
    debt = (
        f"순부채/EBITDA > max(업종 중앙값×{c.net_debt_median_times:g},"
        f" {c.net_debt_floor:.1f}) · EBITDA 0 이하면서 순부채 > 0",
        f"{fmt('x', row.get('nd_ebitda'))} · 기준 {fmt('x', row.get('nd_limit'))}"
        f" (EBITDA TTM {fmt('big', row.get('ebitda'))})",
    )
    interest = str(row.get("interest_source") or "—")
    cover = (
        f"EBIT ÷ 이자비용 < {c.coverage_floor:g}x",
        f"{fmt('x', row.get('coverage'))} · 이자비용"
        f" {fmt('big', row.get('interest_used'))} ({interest})",
    )
    return [
        (str(row.get("dq_cash")), cash[0], cash[1]),
        (str(row.get("dq_debt")), debt[0], debt[1]),
        (str(row.get("dq_cover")), cover[0], cover[1]),
    ]


def _ratio(top: object, base: object) -> float:
    base_value = _number(base)
    return _number(top) / base_value if base_value > 0 else math.nan


def _standing(value: object) -> str:
    number = _number(value)
    return UNKNOWN if math.isnan(number) else fmt("rank", number)


def _hold_line(row: Mapping[str, object], key: str, c: Criteria) -> str:
    """정점 이익 or 업황 순풍's result for the dialog (업황 순풍 also 마진 급등)."""
    if not c.on(key):
        return UNUSED
    shown = (key, "margin_jump") if key == "tailwind" else (key,)
    hit = next((TAGS[k] for k in shown if row.get(k) is True), "")
    if hit:
        return hit
    if row.get("sector") == FINANCIALS or (
        key == "peak" and row.get("cyclical") is not True
    ):
        return NOT_APPLICABLE
    missing = math.isnan(_number(row.get("hold_median"))) or math.isnan(
        _number(row.get("op_margin"))
    )
    return UNKNOWN if missing else "아님"


def label_lines(row: Mapping[str, object], c: Criteria) -> list[tuple[str, str, str]]:
    """Each 퀄리티 and 밸류에이션 measure, both axes, 정점 이익 and 업황 순풍.

    Each line is the result, the test and the values it used; a measure or
    label turned off is 미적용.
    """
    financial = row.get("sector") == FINANCIALS
    median = fmt("pct", row.get("margin_avg"))
    years = f"{fmt('int', row.get('margin_years'))}년"
    source = MARGIN_SOURCES.get(str(row.get("margin_source")), "—")
    ttm_source = TTM_SOURCES.get(str(row.get("op_margin_source")), "—")
    ttm_end = row.get("op_margin_end") or ""
    ttm_basis = f"{ttm_source}, {ttm_end}까지" if ttm_end else ttm_source
    ttm = f"{fmt('pct', row.get('op_margin'))} ({ttm_basis})"
    assets = f"총자산 {fmt('big', row.get('total_assets'))}"
    improve_years = _number(row.get("improve_years"))
    price = f"주가 {fmt('usd', row.get('price'))}"
    measures = {
        "q_cbop": (
            "(영업이익 + 연구개발비 − 운전자본 증가분) ÷ 총자산 (TTM)",
            f"영업이익 {fmt('big', row.get('op_income_ttm'))} · 연구개발비"
            f" {fmt('big', row.get('rnd'))} · 운전자본 변동(현금흐름 부호)"
            f" {fmt('big', row.get('wc_change'))} · {assets}",
        ),
        "q_gpa": (
            "매출총이익 ÷ 총자산 (TTM)",
            f"매출총이익 {fmt('big', row.get('gross_profit'))} · {assets}",
        ),
        "q_improve": (
            f"영업이익 ÷ 총자산의 {c.improve_years}년 변화",
            f"{fmt('signed', row.get('improve'))}"
            + (
                f" ({row.get('improve_source')}, {improve_years:.0f}년 변화)"
                if not math.isnan(improve_years) and row.get("improve_source")
                else ""
            ),
        ),
        "q_stability": (
            f"영업이익률 {SEC_YEARS}년 표준편차 (낮을수록 좋음)",
            f"{fmt('pct', row.get('margin_spread'))} ({source} {years})",
        ),
        "q_leverage": (
            "총차입금 ÷ 총자산 (낮을수록 좋음)",
            f"{fmt('pct', _ratio(row.get('total_debt'), row.get('total_assets')))} ·"
            f" 총차입금 {fmt('big', row.get('total_debt'))} · {assets}",
        ),
        "q_roe": ("ROE (TTM)", fmt("pct", row.get("roe"))),
        "q_roa": (
            "순이익 ÷ 총자산 (TTM)",
            f"{fmt('pct', _ratio(row.get('net_income'), row.get('total_assets')))}",
        ),
        "q_capital": (
            "은행 TCE/TA · 그 밖 자기자본 ÷ 총자산",
            f"TCE/TA {fmt('pct', row.get('tce_ratio'))} · 자기자본비율"
            f" {fmt('pct', row.get('eq_ratio'))}",
        ),
        "q_roe_stability": (
            f"ROE {SEC_YEARS}년 표준편차 (SEC 10-K, 낮을수록 좋음)",
            fmt("pct", row.get("sec_roe_std")),
        ),
        "v_fwd": (
            "선행 EPS ÷ 주가 (FY1·FY2 12개월 혼합)",
            f"{fmt('pct', _ratio(row.get('fwd_eps'), row.get('price')))} · 선행 EPS"
            f" {fmt('num', row.get('fwd_eps'))} (FY1 비중"
            f" {fmt('rank', row.get('weight'))}) · {price}",
        ),
        "v_ttm": (
            "후행 EPS ÷ 주가",
            f"{fmt('pct', _ratio(row.get('trailing_eps'), row.get('price')))} · 후행"
            f" EPS {fmt('num', row.get('trailing_eps'))} · {price}",
        ),
        "v_ebitda": (
            "EBITDA ÷ EV (TTM)",
            f"{fmt('pct', _ratio(row.get('ebitda'), row.get('ev')))} · EV/EBITDA"
            f" {fmt('num', row.get('ev_ebitda'))}",
        ),
        "v_fcf": (
            "FCF ÷ EV (TTM)",
            f"{fmt('pct', _ratio(row.get('fcf'), row.get('ev')))} · FCF"
            f" {fmt('big', row.get('fcf'))} · EV {fmt('big', row.get('ev'))}",
        ),
        "v_book": (
            "주당 (유형)장부가치 ÷ 주가",
            f"주당 장부가치 {fmt('num', row.get('book_value'))} · 유형장부가치"
            f" {fmt('big', row.get('tbv'))} · {price}",
        ),
    }
    skipped = NON_FINANCIAL_ONLY if financial else FINANCIAL_ONLY
    basis = (
        f"{LONG_WINDOW}년 중앙값 {fmt('pct', row.get('long_med'))}"
        f" ({fmt('int', row.get('long_years'))}개 연도, SEC 10-K)"
        if row.get("short") is not True
        else f"5년 중앙값 {median} ({source} {years}, 짧은 창)"
    )
    return [
        *(
            (
                UNUSED
                if not c.on(key)
                else NOT_APPLICABLE
                if key in skipped
                else _standing(row.get(key)),
                f"{QUALITY.get(key) or VALUATION[key]} · {test}",
                detail,
            )
            for key, (test, detail) in measures.items()
        ),
        (
            _standing(row.get("quality")),
            "퀄리티 = 항목 백분위 평균의 업종 백분위 (높을수록 좋음,"
            f" {c.need(financial=financial)}개 이상)",
            f"{fmt('int', row.get('q_count'))}개 항목 · 기준 {c.split:.0%}",
        ),
        (
            _standing(row.get("valuation")),
            "밸류에이션 = 항목 백분위 평균의 업종 백분위 (높을수록 비쌈,"
            f" {c.need_valuation(financial=financial)}개 이상)",
            f"{fmt('int', row.get('v_count'))}개 항목 · 기준 {c.split:.0%}",
        ),
        (
            _hold_line(row, "peak", c),
            f"경기순환 기업: TTM 영업이익률 ≥ 중앙값 × {c.peak_ratio:g} 그리고 선행 EPS"
            f" ≤ 후행 EPS × {c.peak_eps_ratio:g}",
            f"TTM {ttm} · {basis} · 선행 EPS {fmt('num', row.get('fwd_eps'))} ·"
            f" 후행 EPS {fmt('num', row.get('trailing_eps'))}"
            f" · 경기순환 {'예' if row.get('cyclical') is True else '아님'}",
        ),
        (
            _hold_line(row, "tailwind", c),
            f"업황 순풍: z ≥ {c.tailwind_z:g} 그리고 TTM − 중앙값 ≥"
            f" {c.tailwind_gap * 100:g}%p (짧은 창: × {SHORT_RATIO:g}, +"
            f"{SHORT_GAP * 100:g}%p)",
            f"TTM {ttm} · {basis} · z {fmt('num', row.get('hold_z'))}",
        ),
    ]


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
    turn = " · 연도 교체 주의" if row.get("year_turn") is True else ""
    return (
        f"{row.get('est') or UNKNOWN} · 수정 강도"
        f" {fmt('signed2', row.get('rev_strength'))}"
        f" · 순수정 비율 {fmt('signed', row.get('net_revision'))} · {state}"
        f" · 회계연도 끝 {row.get('next_fy_end') or '—'}{turn}"
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
    "group",
    "quality",
    "valuation",
    "combined",
    "q_count",
    "v_count",
    "score",
    "chart",
    "est",
    "rev_strength",
    "dq_state",
)


def write_snapshot(root: Path, table: pd.DataFrame, c: Criteria, day: date) -> Path:
    """Each liquid stock's cell, scores and labels on ``day``, with the criteria.

    For checking the screen later; the page never reads these back. One CSV
    and one JSON (the criteria, ``DEFINITIONS`` and the price day) per day.
    """
    folder = root / SNAPSHOT_FOLDER
    rows = liquid(table).reset_index(drop=True)
    texts = records(rows)
    frame = cast("pd.DataFrame", rows[list(SNAPSHOT_COLUMNS)]).assign(
        tags=[tags(row) for row in texts], verdict=[verdict(row) for row in texts]
    )
    path = folder / f"{day.isoformat()}.csv"
    _write(frame, path)
    values, off = saved(c)
    as_of = str(rows["as_of"].iloc[0]) if len(rows) else ""
    meta = {"definitions": DEFINITIONS, "as_of": as_of, "values": values, "off": off}
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
                trailing=_frame(stock.get_income_stmt(freq="trailing")),
                cash=_frame(stock.get_cashflow(freq="trailing")),
                cash_yearly=_frame(stock.get_cashflow(freq="yearly")),
                balance=_frame(stock.get_balance_sheet(freq="quarterly")),
                balance_yearly=_frame(stock.get_balance_sheet(freq="yearly")),
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
