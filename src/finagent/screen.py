"""S&P 500 screening on yfinance data, kept under ``data/screen/``.

This module is the one exception to the dashboard's data rules
(``tests/test_guardrails.py``): it alone imports ``yfinance`` (unofficial
Yahoo Finance data), reads the constituents list from Wikipedia and writes
files, all under ``data/screen/`` in the folder the app runs from. yfinance's
own caches go there too. Nothing here runs until the page's 갱신 button starts
an ``Updater``; the page only reads the stored files.

Stored parts and how often they are fetched:

- ``universe-YYYY-MM-DD.csv``: constituents with GICS sector and sub-industry,
  fetched when the newest one is ``UNIVERSE_MAX_AGE_DAYS`` old.
- ``prices.parquet``: daily OHLCV with Adj Close, one row per ticker and day.
  Each update fetches only the days since the last one (plus an overlap);
  a ticker whose overlap no longer matches (split or dividend adjustment) gets
  its full history again.
- ``fundamentals-YYYY-Qn.csv``: once per calendar quarter.
- ``estimates-YYYY-MM-DD.csv``: once per day.

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
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as clock
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, final
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

SNAPSHOT_DIR: Final = Path("data") / "screen"
YFINANCE_CACHE: Final = ".yfinance"
PRICES_FILE: Final = "prices.parquet"
UNIVERSE_GLOB: Final = "universe-????-??-??.csv"
FUNDAMENTALS_GLOB: Final = "fundamentals-????-Q?.csv"
ESTIMATES_GLOB: Final = "estimates-????-??-??.csv"
CONSTITUENTS_URL: Final = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT: Final = "finagent/0.2 (local screener)"
REQUEST_TIMEOUT_SECONDS: Final = 20.0
PACKAGES: Final = ("yfinance", "lxml")
INSTALL_HINT: Final = "uv sync --extra screen"

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

# ---------------------------------------------------------------- rules
# Cutlines are the user's, without a backtest; shares are of the stocks with a
# value, ties at the cutline included.
MIN_BARS: Final = 250
MIN_PRICE: Final = 5.0
ILLIQUID_SHARE: Final = 0.10  # lowest 20-day dollar volume, left out
STRONG_SHARE: Final = 0.20  # top tech scores: 차트 강
WEAK_SHARE: Final = 0.40  # bottom tech scores: 차트 약
VALUE_SHARE: Final = 0.20  # 저렴 / 비쌈 within the peer group
PEER_MIN: Final = 10  # sub-industry members needed, else the sector
REVENUE_FLOOR: Final = -0.05
NET_DEBT_FLOOR: Final = 3.0
NET_DEBT_MEDIAN_TIMES: Final = 2.0
EQUITY_MEDIAN_SHARE: Final = 0.5
PEAK_RATIO: Final = 0.6  # forward PER below trailing times this: 정점 이익
NEAR_HIGH: Final = 0.98  # close at least this share of the 52-week high
OBV_NEAR: Final = 0.05  # of the 60-day OBV range, below its high
D_SCORES: Final = (3, 6)
D_FROM_HIGH: Final = -0.20
D_SHORT: Final = 0.05
TOP_N: Final = 20
FINANCIALS: Final = "Financials"
BANKS: Final = "Banks"
# GICS sub-industry to industry group, for Financials only (Wikipedia gives the
# sub-industry). A Financials sub-industry with too few members is compared
# within this group, so banks never share medians with payments or exchanges.
# One not listed here falls back to the sector.
FINANCIAL_GROUPS: Final = {
    "Diversified Banks": BANKS,
    "Regional Banks": BANKS,
    "Transaction & Payment Processing Services": "Financial Services",
    "Multi-Sector Holdings": "Financial Services",
    "Diversified Financial Services": "Financial Services",
    "Specialized Finance": "Financial Services",
    "Commercial & Residential Mortgage Finance": "Financial Services",
    "Consumer Finance": "Financial Services",
    "Asset Management & Custody Banks": "Financial Services",
    "Investment Banking & Brokerage": "Financial Services",
    "Diversified Capital Markets": "Financial Services",
    "Financial Exchanges & Data": "Financial Services",
    "Mortgage REITs": "Financial Services",
    "Insurance Brokers": "Insurance",
    "Life & Health Insurance": "Insurance",
    "Multi-line Insurance": "Insurance",
    "Property & Casualty Insurance": "Insurance",
    "Reinsurance": "Insurance",
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
    "op_margin": "operatingMargins",
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
    "currency",
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
GROUPS: Final = {
    "A": "차트 강 · 결격 없음 · ROE 우수 · 비쌈 아님",
    "B": "차트 강 · 결격 없음 · 비쌈",
    "C": "결격 없음 · ROE 우수 · 저렴 · 차트 약",
    "D": (
        "결격 없음 · A 아님 · 선행 PER 또는 EV/EBITDA 업종 하위 20% ·"
        " FY1 EPS 컨센서스 ≥ 90일 전 · 30일 하향 ≤ 상향 · 기술 점수 3–6"
    ),
}
# Each group's order: column and whether it ascends.
GROUP_ORDER: Final = {
    "A": (("score", False), ("roe", False)),
    "B": (("score", False), ("mcap", False)),
    "C": (("fpe_pct", True), ("roe", False)),
    "D": (("signals", False), ("score", False), ("mcap", False)),
}
FORMATS: Final = {
    "pct": "{:.1%}",
    "signed": "{:+.1%}",
    "usd": "${:,.2f}",
    "big": "${:,.1f}B",  # billions
    "x": "{:,.1f}x",
    "int": "{:,.0f}",
    "num": "{:,.2f}",
}
GROUP_NAMES: Final = {
    "A": "우량 추세",
    "B": "고평가 추세",
    "C": "소외 우량",
    "D": "저평가 개선",
}
# The chart-by-valuation map: rows of (row label, group per column, None for a
# cell without a group). D sits outside it: its axis is estimate revisions.
MAP_COLUMNS: Final = ("비쌈 아님", "비쌈")
MAP_ROWS: Final = (("차트 강", ("A", "B")), ("차트 약", ("C", None)))
GROUP_SORT: Final = {
    "A": "기술 점수, ROE 순",
    "B": "기술 점수, 시가총액 순",
    "C": "업종 내 선행 PER 낮은 순, ROE 순",
    "D": "부가 신호 수, 기술 점수, 시가총액 순",
}
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
    ("score", "기술 점수", "int"),
    ("chart", "차트", ""),
    ("tags", "라벨", ""),
    ("verdict", "결격", ""),
    ("hot", "과열", ""),
    ("roe", "ROE", "pct"),
    ("fpe", "선행 PER", "num"),
    ("ev_ebitda", "EV/EBITDA", "num"),
    ("rev_growth", "매출 성장(연간)", "signed"),
    ("nd_ebitda", "순부채/EBITDA", "x"),
    ("fcf", "FCF", "big"),
    ("signals", "부가 신호", "int"),
)
RETURN_COLUMNS: Final = ("ret_1m", "ret_3m")
TECH_BASIS: Final = (
    "수정주가 일봉(시가·고가·저가에 수정종가/종가 비율 적용) · 이동평균은 단순"
    " · RSI 14일 Wilder 평활 · MACD 12/26/9 EMA · 볼린저 20일 ±2σ(모표준편차)"
    " · 스토캐스틱 14/3/3 · ATR 14일 Wilder · OBV 60일 기울기는 최소제곱"
    " · 52주 신고가는 종가 ≥ 252일 고가의 98%, OBV 고점 근처는 60일 범위의 5% 이내"
)
D_SIGNALS: Final = {
    "s_macd": "MACD > 시그널",
    "s_high": "52주 고점 대비 −20% 이내",
    "s_short": "공매도 비율 5% 이상",
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


@dataclass(frozen=True, slots=True)
class Inputs:
    universe: pd.DataFrame
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    estimates: pd.DataFrame


class RateLimitedError(Exception):
    """Yahoo refused with "too many requests"."""


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
) -> dict[str, object]:
    """``info`` numbers, annual revenue growth and, for financials, equity/assets.

    ``income`` is the yearly income statement, newest year first. Yahoo keeps
    only five quarters, too few for a TTM year over year, so revenue growth is
    the last fiscal year over the one before.
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
    row["eq_ratio"], row["tce_ratio"] = (
        capital_ratios(balance) if balance is not None else (math.nan, math.nan)
    )
    currency = info.get("financialCurrency")
    row["currency"] = currency if isinstance(currency, str) else ""
    return row


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
    group = cast("pd.Series", table["industry"]).map(FINANCIAL_GROUPS)
    return group.where(table["sector"] == FINANCIALS)


def peers(table: pd.DataFrame) -> pd.Series:
    """The sub-industry where it has ``PEER_MIN`` members here, else the wider group.

    The wider group is the industry group for Financials (``FINANCIAL_GROUPS``)
    and the sector for everything else, or for an unmapped Financials stock.
    """
    industry = cast("pd.Series", table["industry"])
    counts = industry.map(industry.value_counts())
    wider = financial_group(table).fillna(table["sector"])
    return industry.where(counts >= PEER_MIN, wider)


def liquidity(table: pd.DataFrame) -> pd.Series:
    """Why a stock is left out ("" when it stays): history, price, then volume."""
    reason = pd.Series("", index=table.index, dtype="str")
    reason[table["bars"].isna()] = "시세 없음"
    reason[(reason == "") & ~(table["bars"] >= MIN_BARS)] = f"일봉 {MIN_BARS}개 미만"
    reason[(reason == "") & ~(table["price"] >= MIN_PRICE)] = (
        f"주가 ${MIN_PRICE:.0f} 미만"
    )
    rest = reason == ""
    volume = cast("pd.Series", table["dollar_vol20"]).where(rest)
    reason[rest & bottom_share(volume, ILLIQUID_SHARE)] = "거래대금 하위 10%"
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


def _median(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values.groupby(groups).transform("median")


def _within(
    values: pd.Series, groups: pd.Series, pick: Callable[..., pd.Series]
) -> pd.Series:
    out = pd.Series(data=False, index=values.index)
    for _, members in values.groupby(groups):
        out[members.index] = pick(members, VALUE_SHARE)
    return out


def fundamentals_status(table: pd.DataFrame) -> pd.DataFrame:
    """Disqualifiers, labels and their peer medians, for liquid stocks only.

    A check whose input is missing is 미확인, and one with a negative or zero
    base is 해석 부적절; either withholds the stock from every group.
    """
    t = table
    peer = cast("pd.Series", t["peer"])
    financial = t["sector"] == FINANCIALS
    out = pd.DataFrame(index=t.index)

    negative_equity = t["book_value"] <= 0
    roe = cast("pd.Series", t["roe"]).where(~negative_equity)
    out["roe_med"] = _median(roe, peer)
    out["roe_label"] = np.select(
        [negative_equity, roe.isna(), roe >= out["roe_med"]],
        [MEANINGLESS, UNKNOWN, "우수"],
        "",
    )

    bank = financial_group(t) == BANKS
    other = financial & ~bank
    eq_ratio = cast("pd.Series", t["eq_ratio"]).where(other)
    out["eq_med"] = _median(eq_ratio, peer)
    out["eq_limit"] = out["eq_med"] * EQUITY_MEDIAN_SHARE
    other_fail = (t["roe"] < 0) | (eq_ratio < out["eq_limit"])
    other_unknown = t["roe"].isna() | eq_ratio.isna() | out["eq_limit"].isna()
    bank_fail = (t["roe"] < 0) | (t["tce_ratio"] < BANK_TCE_FLOOR)
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
        [t["revenue_prior"] <= 0, growth.isna(), growth <= REVENUE_FLOOR],
        [MEANINGLESS, UNKNOWN, FAIL],
        CLEAR,
    )

    net_debt = t["total_debt"] - t["total_cash"]
    nd = cast("pd.Series", t["nd_ebitda"]).where(~financial)
    out["nd_med"] = _median(nd, peer)
    out["nd_limit"] = np.fmax(out["nd_med"] * NET_DEBT_MEDIAN_TIMES, NET_DEBT_FLOOR)
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

    checks = out[["dq_fcf", "dq_rev", "dq_debt"]]
    out["dq_state"] = np.select(
        [(checks == FAIL).any(axis=1), checks.isin(list(UNSURE)).any(axis=1)],
        [FAIL, UNKNOWN],
        CLEAR,
    )

    fpe = cast("pd.Series", t["fpe"]).where(t["fpe"] > 0)
    out["fpe_pct"] = fpe.groupby(peer).rank(pct=True)
    cheap = _within(fpe, peer, bottom_share)
    dear = _within(fpe, peer, top_share)
    out["value_label"] = np.select(
        [t["fpe"] <= 0, fpe.isna(), cheap, dear],
        [MEANINGLESS, UNKNOWN, "저렴", "비쌈"],
        "보통",
    )
    ev = cast("pd.Series", t["ev_ebitda"]).where((t["ev_ebitda"] > 0) & ~financial)
    out["ev_cheap"] = _within(ev, peer, bottom_share)
    out["peak"] = (
        t["sector"].isin(list(CYCLICALS))
        & (t["tpe"] > 0)
        & (fpe < t["tpe"] * PEAK_RATIO)
    )
    return out


def groups(table: pd.DataFrame) -> pd.DataFrame:
    """Membership in A to D (D may overlap B or C, as the rules allow)."""
    t = table
    strong = t["chart"] == "강"
    clean = t["dq_state"] == CLEAR
    good = t["roe_label"] == "우수"
    value = t["value_label"]
    estimates = (t["fy1_now"] >= t["fy1_90d"]) & (t["fy1_down30"] <= t["fy1_up30"])
    low, high = D_SCORES
    a = strong & clean & good & value.isin(["저렴", "보통"])
    return pd.DataFrame(
        {
            "in_A": a,
            "in_B": strong & clean & (value == "비쌈"),
            "in_C": clean & good & (value == "저렴") & (t["chart"] == "약"),
            "in_D": clean
            & ~a
            & ((value == "저렴") | t["ev_cheap"])
            & estimates
            & t["score"].between(low, high),
        },
        index=t.index,
    )


def evaluate(inputs: Inputs) -> pd.DataFrame:
    """Every constituent with its liquidity, score, checks, labels and groups."""
    table = inputs.universe.merge(technicals(inputs.prices), on="ticker", how="left")
    table = table.merge(inputs.fundamentals, on="ticker", how="left")
    table = table.merge(
        inputs.estimates, on="ticker", how="left", suffixes=("", "_estimates")
    )
    for column in (*FUNDAMENTAL_COLUMNS, *ESTIMATE_COLUMNS):
        if column not in table.columns:
            table[column] = math.nan
    for column in ("currency", "est_currency", "rev_year"):
        table[column] = table[column].fillna("").astype("str")
    table["error"] = _errors(table)
    table = table.drop(columns=["error_estimates"], errors="ignore")
    table["liquidity"] = liquidity(table)
    liquid = cast("pd.DataFrame", table[table["liquidity"] == ""]).copy()

    checks = tech_checks(liquid)
    liquid[list(checks.columns)] = checks
    liquid["score"] = checks.sum(axis=1).astype("int64")
    score = cast("pd.Series", liquid["score"]).astype("float64")
    strong = top_share(score, STRONG_SHARE)
    weak = bottom_share(score, WEAK_SHARE) & ~strong
    liquid["chart"] = np.select([strong, weak], ["강", "약"], "")
    liquid[list(OVERHEAT)] = overheat(liquid)
    liquid["peer"] = peers(liquid)
    peer = cast("pd.Series", liquid["peer"])
    liquid["peer_n"] = peer.map(peer.value_counts().to_dict())
    status = fundamentals_status(liquid)
    liquid[list(status.columns)] = status
    liquid["s_macd"] = liquid["macd"] > liquid["macd_signal"]
    liquid["s_high"] = liquid["from_high"] >= D_FROM_HIGH
    liquid["s_short"] = liquid["short_pct"] >= D_SHORT
    liquid["signals"] = liquid[list(D_SIGNALS)].sum(axis=1).astype("int64")
    member = groups(liquid)
    liquid[list(member.columns)] = member

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
    return cast("pd.DataFrame", table[table[f"in_{key}"].eq(other=True)])


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


def group_help() -> str:
    """Every group's name and conditions, for a tooltip."""
    return "\n".join(
        f"- **{GROUP_NAMES[key]}**: {rule}" for key, rule in GROUPS.items()
    )


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", frame.to_dict(orient="records"))


def view(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows as the results table shows them (``TABLE``), in the same order."""
    shown = frame.reset_index(drop=True)
    rows = records(shown)
    text = pd.DataFrame(
        {
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
    """A number as the page shows it (``FORMATS``); missing values are a dash."""
    number = _number(value)
    if math.isnan(number):
        return "—"
    if kind == "big":
        number /= 1e9
    return FORMATS.get(kind, FORMATS["num"]).format(number)


def peer_level(row: Mapping[str, object]) -> str:
    """Which level ``peers`` compared the stock at."""
    peer = row.get("peer")
    if peer == row.get("industry"):
        return "세부 산업"
    return "산업 그룹" if peer in FINANCIAL_GROUPS.values() else "섹터"


def tags(row: Mapping[str, object]) -> str:
    """Labels for the table: 우수, 저렴/비쌈, 정점 이익, and why one is missing."""
    out: list[str] = []
    roe, value = row.get("roe_label"), row.get("value_label")
    if roe == "우수":
        out.append("우수")
    elif roe in UNSURE:
        out.append(f"ROE {roe}")
    if value in {"저렴", "비쌈"}:
        out.append(str(value))
    elif value in UNSURE:
        out.append(f"PER {value}")
    if row.get("peak") is True:
        out.append("정점 이익")
    return " · ".join(out)


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


def tech_lines(row: Mapping[str, object]) -> list[tuple[bool, str, str]]:
    """Each score check with the values it compared."""
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
        for key, label in TECH_CHECKS.items()
    ]


def overheat_lines(row: Mapping[str, object]) -> list[tuple[bool, str, str]]:
    details = {
        "o_rsi": fmt("num", row.get("rsi")),
        "o_stoch": (
            f"%K {fmt('num', row.get('stoch_k'))} · %D {fmt('num', row.get('stoch_d'))}"
        ),
        "o_bb": fmt("num", row.get("pct_b")),
    }
    return [
        (row.get(key) is True, label, details[key]) for key, label in OVERHEAT.items()
    ]


def dq_lines(row: Mapping[str, object]) -> list[tuple[str, str, str]]:
    """Each disqualifier: status, the test, and the values against it."""
    roe = f"ROE {fmt('pct', row.get('roe'))} (TTM)"
    if FINANCIAL_GROUPS.get(str(row.get("industry"))) == BANKS:
        fcf = (
            f"은행: ROE < 0 또는 TCE/TA < {BANK_TCE_FLOOR:.0%}",
            f"{roe} · TCE/TA {fmt('pct', row.get('tce_ratio'))} (최근 분기,"
            " 유형 보통주자본 ÷ (총자산 − 영업권·무형자산))",
        )
    elif row.get("sector") == FINANCIALS:
        fcf = (
            "금융업: ROE < 0 또는 자기자본비율 < 업종 중앙값의 절반",
            f"{roe} · 자기자본비율 {fmt('pct', row.get('eq_ratio'))} (최근 분기) ·"
            f" 기준 {fmt('pct', row.get('eq_limit'))}"
            f" ({row.get('peer')} 중앙값의 절반)",
        )
    else:
        fcf = ("잉여현금흐름 < 0", f"{fmt('big', row.get('fcf'))} (TTM)")
    year = row.get("rev_year") or "—"
    revenue = (
        f"매출 성장 ≤ {REVENUE_FLOOR:.0%}",
        f"{fmt('signed', row.get('rev_growth'))} (연간, {year} 결산)",
    )
    debt = (
        "순부채/EBITDA > max(업종 중앙값×2, 3.0)",
        f"{fmt('x', row.get('nd_ebitda'))} · 기준 {fmt('x', row.get('nd_limit'))}"
        f" (EBITDA TTM {fmt('big', row.get('ebitda'))})",
    )
    return [
        (str(row.get("dq_fcf")), fcf[0], fcf[1]),
        (str(row.get("dq_rev")), revenue[0], revenue[1]),
        (str(row.get("dq_debt")), debt[0], debt[1]),
    ]


def label_lines(row: Mapping[str, object]) -> list[tuple[str, str, str]]:
    """Each label: result, the test, and the values and medians it used."""
    pct = _number(row.get("fpe_pct"))
    rank = f"업종 내 하위 {pct:.0%}" if math.isfinite(pct) else "순위 없음"
    peak = (
        "정점 이익"
        if row.get("peak") is True
        else "아님"
        if row.get("sector") in CYCLICALS
        else NOT_APPLICABLE
    )
    return [
        (
            str(row.get("roe_label") or "보통 이하"),
            "ROE ≥ 업종 중앙값",
            f"{fmt('pct', row.get('roe'))} (TTM) ·"
            f" 중앙값 {fmt('pct', row.get('roe_med'))}",
        ),
        (
            str(row.get("value_label")),
            "선행 PER 업종 하위 20% 저렴 · 상위 20% 비쌈",
            f"{fmt('num', row.get('fpe'))} (예상) · {rank}",
        ),
        (
            peak,
            "경기순환주: 선행 PER < 후행 PER × 0.6",
            f"선행 {fmt('num', row.get('fpe'))} ·"
            f" 후행 {fmt('num', row.get('tpe'))} (TTM)",
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


def universe_due(root: Path, today: date) -> bool:
    path = _newest(root, UNIVERSE_GLOB)
    if path is None:
        return True
    age = today - _stem_date(path, "universe-")
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
    )


def stamp(root: Path) -> tuple[tuple[str, int], ...]:
    """The newest files and their change times: a cache key for ``load``."""
    paths = [
        _newest(root, UNIVERSE_GLOB),
        root / PRICES_FILE,
        _newest(root, FUNDAMENTALS_GLOB),
        _newest(root, ESTIMATES_GLOB),
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
    )


# ---------------------------------------------------------------- fetching


@final
class Yahoo:
    """The real source: Wikipedia for the members, yfinance for the numbers."""

    def __init__(self, root: Path) -> None:
        """Point yfinance's caches into ``root`` instead of the user cache."""
        import yfinance as yf  # noqa: PLC0415 - optional dependency

        root.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(root / YFINANCE_CACHE))

    def constituents(self) -> pd.DataFrame:
        response = httpx.get(
            CONSTITUENTS_URL,
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
            balance = (
                nonempty(stock.get_balance_sheet(freq="quarterly"), "재무상태표")
                if financial
                else None
            )
        return fundamentals_row(info, income, balance)

    def estimates(self, ticker: str) -> Mapping[str, object]:
        import yfinance as yf  # noqa: PLC0415

        stock = yf.Ticker(ticker)
        with _rate_limit():
            trend = nonempty(stock.eps_trend, "EPS 추정치")
            return estimates_row(trend, _frame(stock.eps_revisions))


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
    fetch: Callable[[str], Mapping[str, object]],
    path: Path,
    columns: Sequence[str],
    tickers: Sequence[str],
    stage: str,
    report: Report,
    sleep: Callable[[float], None],
) -> int:
    """Fetch the pending tickers into ``path``; returns the failures recorded.

    The file is written even when Yahoo stops the run, so the next update
    picks up where this one ended.
    """
    existing = _read_csv(path, columns)
    todo = pending(existing, tickers)
    rows: list[dict[str, object]] = []
    try:
        for i, ticker in enumerate(todo):
            report(stage, i, len(todo))
            row: dict[str, object]
            try:
                row = {"ticker": ticker, **_retry(partial(fetch, ticker), sleep)}
                row["error"] = ""
            except RateLimitedError:
                raise
            except Exception as error:  # noqa: BLE001 - recorded per ticker
                row = {"ticker": ticker, "error": f"{type(error).__name__}: {error}"}
            rows.append(row)
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


def update(
    source: Source,
    root: Path,
    now: datetime,
    report: Report,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Bring every stored part up to date; returns a one-line summary.

    Raises ``RateLimitedError`` when Yahoo keeps refusing after the retries.
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
        lambda t: source.fundamentals(t, financial=t in financial),
        root / f"fundamentals-{quarter(today)}.csv",
        FUNDAMENTAL_COLUMNS,
        tickers,
        "재무",
        report,
        sleep,
    )
    failed += update_rows(
        source.estimates,
        root / f"estimates-{today.isoformat()}.csv",
        ESTIMATE_COLUMNS,
        tickers,
        "추정치",
        report,
        sleep,
    )
    priced = len(set(prices["ticker"]))
    note = f" · 받지 못한 값 {failed}" if failed else ""
    newest = cast("pd.Timestamp", prices["date"].max()).date() if priced else None
    return f"{len(tickers)}종목 · 시세 {newest or '—'} · 시세 받은 종목 {priced}{note}"


@final
class Updater:
    """Runs one update at a time in a background thread; pages poll ``progress``."""

    __slots__ = ("_lock", "_progress", "_source", "root")

    def __init__(self, root: Path, source: Callable[[Path], Source] = Yahoo) -> None:
        """Save under ``root``; ``source`` builds the fetcher per update."""
        self.root = root
        self._source = source
        self._lock = threading.Lock()
        self._progress = Progress()

    @property
    def progress(self) -> Progress:
        with self._lock:
            return self._progress

    def start(self) -> bool:
        """Start an update unless one is already running."""
        with self._lock:
            if self._progress.running:
                return False
            self._progress = Progress(running=True, stage="준비")
        threading.Thread(target=self._run, name="finagent-screen", daemon=True).start()
        return True

    def _report(self, stage: str, done: int, total: int) -> None:
        with self._lock:
            self._progress = Progress(running=True, stage=stage, done=done, total=total)

    def _run(self) -> None:
        try:
            now = datetime.now().astimezone()
            message = update(self._source(self.root), self.root, now, self._report)
        except RateLimitedError:
            self._finish(
                "야후가 요청을 제한했습니다. 받은 만큼 저장했으니 잠시 뒤 다시"
                " 갱신하세요."
            )
        except Exception as error:  # noqa: BLE001 - shown on the page, not lost
            self._finish(f"갱신하지 못했습니다: {type(error).__name__}: {error}")
        else:
            self._finish(message, failed=False)

    def _finish(self, message: str, *, failed: bool = True) -> None:
        with self._lock:
            self._progress = Progress(message=message, failed=failed)
