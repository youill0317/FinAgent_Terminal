"""Portfolios from a trade log, with analytics and risk on yfinance prices.

Like the screener, this module is an exception to the dashboard's data rules
(``tests/test_guardrails.py``): it imports ``yfinance`` (unofficial Yahoo
Finance data), downloads Ken French's daily factors from his Data Library
when asked, and writes files, all under ``data/portfolio/`` in the folder
the app runs from. yfinance's own caches go there too. Of the other finagent
modules it imports only the pure math in ``quant`` and ``perf``; rates from
the dashboard's sources (CD 91-day, FRED DEXKOUS) are passed in by app.py.
Prices are fetched only by a ``Fetcher`` thread.

Two price bases, never mixed: valuation and P&L use closes (split-adjusted,
not dividend-adjusted) against the trade prices the user entered; returns,
correlations and risk use adjusted closes (dividends and splits). Amounts are
converted to the portfolio's base currency with daily USD/KRW (``KRW=X``).
A missing price stays missing: a close is carried over a series' own gap
(holidays, a short outage) for at most ``FILL_DAYS`` calendar days and past
its last close for ``GAP_DAYS``, never further (``carry_over``); a 13F clone
buys only at a close at most ``GAP_DAYS`` old.
"""

# pandas and yfinance ship no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportImplicitStringConcatenation=false

from __future__ import annotations

import importlib.util
import io
import math
import re
import threading
import uuid
import zipfile
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Protocol, cast, final

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# The guardrail test reads these two dotted forms as the math modules.
import finagent.perf as perf  # noqa: PLR0402
import finagent.quant as quant  # noqa: PLR0402

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    from numpy.typing import NDArray

SCHEMA_VERSION: Final = 1
DATA_DIR: Final = Path("data") / "portfolio"
BOOK_FILE: Final = "portfolios.json"
CLOSE_FILE: Final = "close.parquet"
ADJUSTED_FILE: Final = "adjusted.parquet"
RANGE_FILES: Final = ("open.parquet", "high.parquet", "low.parquet")  # on request
YFINANCE_CACHE: Final = ".yfinance"
PACKAGES: Final = ("yfinance",)
INSTALL_HINT: Final = "uv sync --extra screen"
FX: Final = "KRW=X"  # won per dollar
BENCHMARK: Final = "^GSPC"  # price index, from 1987: stress, index x beta, betas
BENCHMARK_LABEL: Final = "S&P 500"
TOTAL_RETURN: Final = "^SP500TR"  # dividends reinvested, from 1988: return comparisons
KOSPI: Final = "^KS11"  # stand-in for Korean holdings in stress windows
KOSDAQ: Final = "^KQ11"
KRW_INDICES: Final = (KOSPI, KOSDAQ)  # quoted in won
RF_USD: Final = "^IRX"  # 13-week T-bill, annual percent: the USD risk-free rate
VIX: Final = "^VIX"  # fetched on request, for the HAR filter
HISTORY_START: Final = date(2007, 1, 1)  # holdings' history
LONG_TICKERS: Final = (BENCHMARK, KOSPI, FX)  # kept back to LONG_START for stress
LONG_START: Final = date(1987, 1, 1)
FILL_DAYS: Final = 31  # calendar days a close is carried over a gap it resumes from
GAP_DAYS: Final = 7  # calendar days a rate, window edge or last close may sit off
TRADING_DAYS: Final = 252
YEAR_DAYS: Final = 365
MIN_DAYS: Final = 20  # fewer return days and a statistic is left out
WEEK: Final = "W-FRI"  # weekly returns for turbulence and style
KOREAN_MAJORITY: Final = 0.5  # a book more Korean than this by value closes in Seoul
PATHS: Final = 20_000  # a path share's Monte Carlo SD <= 0.35%p; 1Y peak ~150 MB
PERCENTILES: Final = (5, 25, 50, 75, 95)
BINS: Final = 40
WINDOWS: Final = {"6M": 6, "1Y": 12, "3Y": 36}
HORIZONS: Final = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
FACTS_FILE: Final = "facts.json"  # on request: each holding's Yahoo market cap
FACTORS_FILE: Final = "factors.parquet"  # on request: Ken French's monthly US factors
FRENCH_URL: Final = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES: Final = ("F-F_Research_Data_Factors_CSV.zip",)  # Mkt-RF, SMB, HML, RF
FRENCH_START: Final = "1963-07-01"  # Fama-French (1997) factor history start
FACTORS: Final = ("Mkt-RF", "SMB", "HML")
FACTOR_MIN_MONTHS: Final = 36  # Fama-French (1997) Table 5: shortest window they fit
MONTHS_PER_YEAR: Final = 12
MOP_COM: Final = 60  # Moskowitz-Ooi-Pedersen (2012) eq. 1 EWMA center of mass (days)
MOP_COVERAGE: Final = 0.99  # eq. 1's weights summed before a volatility is used
# Days until the weights (1 - delta) delta^i, delta = 60/61, sum to MOP_COVERAGE: 279
MOP_BURN: Final = math.ceil(
    math.log(1 - MOP_COVERAGE) / math.log(MOP_COM / (MOP_COM + 1))
)
MOP_DAYS: Final = 261  # Moskowitz-Ooi-Pedersen (2012) eq. 1 annualisation factor
MOP_LAGS: Final = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}  # eq. 3 lags h, user-chosen
BL_TAU: Final = 0.025  # Idzorek (2005) prior covariance scalar tau
BL_PREMIUM: Final = 0.03  # Idzorek (2005): his example's specified market premium
FRENCH_TIMEOUT: Final = 60.0
KRW_SUFFIXES: Final = (".KS", ".KQ")
VAR_LEVELS: Final = (0.95, 0.975, 0.99)
REVERSE_LOSS: Final = 0.30
CHART_POINTS: Final = 1000  # most points a chart line shows; calculations use all
PROXY_MIN_DAYS: Final = 60  # fewer overlapping returns: a stand-in index at beta 1
WEEKS: Final = 156  # turbulence and style windows
TURBULENCE_REFIT: Final = 4  # weeks between covariance refits
STYLE_PROXIES: Final = (
    "SPY",
    "IWM",
    "IVW",
    "IVE",
    "MTUM",
    "QUAL",
    "USMV",
    "AGG",
    "GLD",
    KOSPI,
    KOSDAQ,
)
CASH: Final = "현금"
TICKER_PATTERN: Final = re.compile(r"[A-Z0-9^][A-Z0-9.=^-]{0,19}")
EPSILON: Final = 1e-9
MEMO_LENGTH: Final = 200
SIDES: Final = {"buy": "매수", "sell": "매도"}
SYMBOLS: Final = {"KRW": "₩", "USD": "$"}
DECIMALS: Final = {"KRW": 0, "USD": 2}
UNKNOWN: Final = "—"

Side = Literal["buy", "sell"]
Currency = Literal["KRW", "USD"]


type Span = tuple[date, date]  # stated peak and trough


@dataclass(frozen=True, slots=True)
class Scenario:
    """A past fall; each index's window is searched around its own dates.

    ``start`` and ``end`` are the union's approximate edges. ``us`` and ``kr``
    are the S&P 500's and KOSPI's stated peak and trough, None when that
    index is not used.
    """

    name: str
    start: date
    end: date
    us: Span | None
    kr: Span | None

    @property
    def us_only(self) -> bool:
        return self.kr is None


def _both(name: str, start: date, end: date) -> Scenario:
    return Scenario(name, start, end, (start, end), (start, end))


SCENARIOS: Final = (
    Scenario(
        "1987 블랙먼데이",
        date(1987, 8, 25),
        date(1987, 12, 4),
        (date(1987, 8, 25), date(1987, 12, 4)),
        None,
    ),
    Scenario(
        "1997 IMF",
        date(1997, 6, 17),
        date(1998, 6, 16),
        None,
        (date(1997, 6, 17), date(1998, 6, 16)),
    ),
    Scenario(
        "닷컴",
        date(2000, 1, 4),
        date(2002, 10, 9),
        (date(2000, 3, 24), date(2002, 10, 9)),
        (date(2000, 1, 4), date(2001, 9, 17)),
    ),
    Scenario(
        "금융위기",
        date(2007, 10, 9),
        date(2009, 3, 9),
        (date(2007, 10, 9), date(2009, 3, 9)),
        (date(2007, 10, 31), date(2008, 10, 24)),
    ),
    _both("2011 미국 신용등급", date(2011, 4, 29), date(2011, 10, 3)),
    _both("2015 중국", date(2015, 5, 21), date(2016, 2, 11)),
    _both("2018 4분기", date(2018, 9, 20), date(2018, 12, 24)),
    _both("코로나", date(2020, 2, 19), date(2020, 3, 23)),
    Scenario(
        "2022",
        date(2021, 7, 6),
        date(2022, 10, 12),
        (date(2022, 1, 3), date(2022, 10, 12)),
        (date(2021, 7, 6), date(2022, 10, 12)),
    ),
    _both("2024 한국 블랙먼데이", date(2024, 7, 11), date(2024, 8, 5)),
    _both("2025 관세", date(2025, 2, 19), date(2025, 4, 8)),
)


@dataclass(frozen=True, slots=True)
class Shock:
    """A made-up move: each index in its own currency, and USD/KRW."""

    name: str
    sp500: float
    kospi: float
    usdkrw: float


SHOCKS: Final = (
    Shock("한국발 위기", 0.0, -0.30, 0.15),
    Shock("미국발 위기", -0.25, 0.0, 0.10),
    Shock("환율 −15%", 0.0, 0.0, -0.15),
    Shock("환율 +15%", 0.0, 0.0, 0.15),
)


class BookError(ValueError):
    """A change the trade log cannot take; the message is shown on the page."""


# ---------------------------------------------------------------- the log


class _Frozen(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class Trade(_Frozen):
    id: str
    day: date
    ticker: str
    side: Side
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)  # per share, in the ticker's currency
    fee: float = Field(default=0.0, ge=0)
    memo: str = Field(default="", max_length=MEMO_LENGTH)  # absent in older files


class View(_Frozen):
    """The user's Black-Litterman view: one holding's annual return, in base."""

    ticker: str
    annual: float
    confidence: float = Field(gt=0, le=1)  # Idzorek's, 1 = the view itself


class Portfolio(_Frozen):
    id: str
    name: str
    base: Currency = "KRW"
    trades: tuple[Trade, ...] = ()
    views: tuple[View, ...] = ()  # absent in older files
    premium: float = BL_PREMIUM  # BL's annual market premium; absent in older files


class Book(_Frozen):
    """Everything in ``portfolios.json``."""

    version: int = SCHEMA_VERSION
    portfolios: tuple[Portfolio, ...] = ()

    def get(self, pid: str) -> Portfolio | None:
        return next((p for p in self.portfolios if p.id == pid), None)


@dataclass(frozen=True, slots=True)
class Position:
    """One ticker after the log, at moving-average cost with fees included.

    ``*_base`` amounts are in the portfolio's currency at each trade day's
    USD/KRW, so their gap to the native figures is the currency effect.
    """

    ticker: str
    quantity: float = 0.0
    cost: float = 0.0  # basis of the shares held, native
    cost_base: float = 0.0
    realized: float = 0.0  # every sale so far, native
    realized_base: float = 0.0
    sold_base: float = 0.0  # basis of the shares sold
    bought_base: float = 0.0  # every buy with its fee
    first: date | None = None  # first buy
    opened: date | None = None  # first buy since the position was last empty
    last_sell: date | None = None

    @property
    def average(self) -> float:
        return self.cost / self.quantity if self.quantity > EPSILON else math.nan


@dataclass(frozen=True, slots=True)
class Sale:
    """One sell; ``basis`` is the moving-average cost of the shares sold."""

    trade: Trade
    opened: date  # first buy of the holding the shares came from
    basis: float  # native
    basis_base: float
    pnl: float  # proceeds net of the fee minus basis, native
    pnl_base: float

    @property
    def average(self) -> float:
        return self.basis / self.trade.quantity

    @property
    def native_return(self) -> float:
        return self.pnl / self.basis

    @property
    def base_return(self) -> float:
        return self.pnl_base / self.basis_base if self.basis_base else math.nan

    @property
    def held_days(self) -> int:
        return (self.trade.day - self.opened).days


@dataclass(frozen=True, slots=True)
class Ledger:
    positions: dict[str, Position]
    sales: tuple[Sale, ...]


def missing_packages() -> list[str]:
    return [name for name in PACKAGES if importlib.util.find_spec(name) is None]


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def clean_ticker(text: str) -> str:
    """The ticker as Yahoo spells it; raises ``BookError`` for one not valued."""
    ticker = text.strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise BookError("티커 형식이 맞지 않습니다.")
    if "." in ticker and not ticker.endswith(KRW_SUFFIXES):
        raise BookError("미국(BRK-B 형식)과 한국(.KS/.KQ) 종목만 받습니다.")
    return ticker


def currency(ticker: str) -> Currency:
    """In won for .KS/.KQ tickers and the Korean indices, otherwise in dollars."""
    return "KRW" if korean(ticker) else "USD"


def ordered(trades: Iterable[Trade]) -> list[Trade]:
    """Ledger order: by day, buys before sells on a day, then as entered."""
    numbered = sorted(
        enumerate(trades), key=lambda it: (it[1].day, it[1].side != "buy", it[0])
    )
    return [trade for _, trade in numbered]


def replay(
    trades: Iterable[Trade], base: Currency = "KRW", fx: pd.Series | None = None
) -> Ledger:
    """Positions after the log and every sale, by the moving-average method.

    A buy's fee is added to its cost and a sell's fee taken from its proceeds.
    A position sold down to zero starts a new cost basis on its next buy.
    ``fx`` (USD/KRW closes) prices the ``*_base`` amounts; without it they are
    nan for foreign tickers. Raises ``BookError`` at the first sell of more
    than was held that day.
    """
    rates = fx if fx is not None else _no_rates()
    held: dict[str, Position] = {}
    sales: list[Sale] = []
    for trade in ordered(trades):
        old = held.get(trade.ticker, Position(trade.ticker))
        native = currency(trade.ticker)
        rate = 1.0 if native == base else rate_on(rates, trade.day)
        amount = trade.quantity * trade.price
        if trade.side == "buy":
            paid = amount + trade.fee
            paid_base = convert(paid, native, base, rate)
            held[trade.ticker] = replace(
                old,
                quantity=old.quantity + trade.quantity,
                cost=old.cost + paid,
                cost_base=old.cost_base + paid_base,
                bought_base=old.bought_base + paid_base,
                first=old.first or trade.day,
                opened=old.opened or trade.day,
            )
            continue
        if trade.quantity > old.quantity + EPSILON:
            raise BookError(
                f"{trade.day} {trade.ticker} 매도 {shares(trade.quantity)}주가 그날"
                f" 보유 {shares(old.quantity)}주보다 많습니다."
            )
        part = min(trade.quantity / old.quantity, 1.0)
        got = amount - trade.fee
        sale = Sale(
            trade=trade,
            opened=old.opened or trade.day,
            basis=old.cost * part,
            basis_base=old.cost_base * part,
            pnl=got - old.cost * part,
            pnl_base=convert(got, native, base, rate) - old.cost_base * part,
        )
        sales.append(sale)
        closed = old.quantity - trade.quantity <= EPSILON
        held[trade.ticker] = replace(
            old,
            quantity=0.0 if closed else old.quantity - trade.quantity,
            cost=0.0 if closed else old.cost - sale.basis,
            cost_base=0.0 if closed else old.cost_base - sale.basis_base,
            realized=old.realized + sale.pnl,
            realized_base=old.realized_base + sale.pnl_base,
            sold_base=old.sold_base + sale.basis_base,
            opened=None if closed else old.opened,
            last_sell=trade.day,
        )
    return Ledger(held, tuple(sales))


def _replace(book: Book, portfolio: Portfolio) -> Book:
    return book.model_copy(
        update={
            "portfolios": tuple(
                portfolio if p.id == portfolio.id else p for p in book.portfolios
            )
        }
    )


def _named(book: Book, name: str, skip: str = "") -> str:
    name = name.strip()
    if not name:
        raise BookError("이름을 입력하세요.")
    if any(p.name == name and p.id != skip for p in book.portfolios):
        raise BookError("같은 이름의 포트폴리오가 있습니다.")
    return name


def _portfolio(book: Book, pid: str) -> Portfolio:
    found = book.get(pid)
    if found is None:
        raise BookError("포트폴리오가 없습니다.")
    return found


def add_portfolio(book: Book, name: str, base: Currency) -> tuple[Book, str]:
    portfolio = Portfolio(id=new_id(), name=_named(book, name), base=base)
    return book.model_copy(update={"portfolios": (*book.portfolios, portfolio)}), (
        portfolio.id
    )


def edit_portfolio(book: Book, pid: str, name: str, base: Currency) -> Book:
    old = _portfolio(book, pid)
    new = old.model_copy(update={"name": _named(book, name, skip=pid), "base": base})
    return _replace(book, new)


def remove_portfolio(book: Book, pid: str) -> Book:
    kept = tuple(p for p in book.portfolios if p.id != pid)
    return book.model_copy(update={"portfolios": kept})


def add_trade(book: Book, pid: str, trade: Trade) -> Book:
    """The book with ``trade``; raises ``BookError`` if the log no longer adds up."""
    old = _portfolio(book, pid)
    trades = (*old.trades, trade)
    _ = replay(trades)
    return _replace(book, old.model_copy(update={"trades": trades}))


def remove_trades(book: Book, pid: str, ids: Collection[str]) -> Book:
    """The book without ``ids``; raises ``BookError`` if a later sell is left short."""
    old = _portfolio(book, pid)
    trades = tuple(t for t in old.trades if t.id not in ids)
    try:
        _ = replay(trades)
    except BookError as error:
        raise BookError(f"지우면 매도가 보유 수량을 넘습니다. {error}") from error
    return _replace(book, old.model_copy(update={"trades": trades}))


def set_views(
    book: Book, pid: str, views: tuple[View, ...], premium: float = BL_PREMIUM
) -> Book:
    old = _portfolio(book, pid)
    return _replace(book, old.model_copy(update={"views": views, "premium": premium}))


def book_tickers(book: Book) -> list[str]:
    return sorted({t.ticker for p in book.portfolios for t in p.trades})


def wanted(book: Book, saved: Collection[str] = ()) -> list[str]:
    """Every ticker the page values, USD/KRW, the indices and the USD rate.

    With VIX once it is among the ``saved`` tickers (it is fetched on request).
    """
    vix = [VIX] if VIX in saved else []
    return [*book_tickers(book), FX, BENCHMARK, TOTAL_RETURN, KOSPI, RF_USD, *vix]


# ---------------------------------------------------------------- storage


def load_book(root: Path) -> Book:
    """The saved portfolios, or none yet.

    Raises ``BookError`` when the file cannot be read; the page then stops,
    so a damaged file is never overwritten.
    """
    path = root / BOOK_FILE
    if not path.is_file():
        return Book()
    try:
        return Book.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise BookError(f"{path} 파일을 읽지 못했습니다: {error}") from error


def save_book(root: Path, book: Book) -> None:
    """Write atomically; raises ``OSError`` if the file cannot be written."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / BOOK_FILE
    temp = path.with_name(f"{path.name}.tmp")
    _ = temp.write_text(book.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _ = temp.replace(path)


class Facts(_Frozen):
    """One stock's Yahoo quote figures (``Ticker.info``) on the day fetched.

    ``market_cap`` is in ``currency``, the quote's; NaN where Yahoo has none.
    """

    # NaN is written as NaN, which reads back, not as null, which does not.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, ser_json_inf_nan="constants"
    )

    day: date
    currency: str
    market_cap: float


def parse_french(text: str) -> pd.DataFrame:
    """The monthly table of a Ken French CSV from ``FRENCH_START``: fractions."""
    lines = text.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith(","))
    columns = [name.strip() for name in lines[head].split(",")[1:]]
    months: list[str] = []
    values: list[list[float]] = []
    for line in lines[head + 1 :]:
        cells = line.split(",")
        stamp = cells[0].strip()
        if not re.fullmatch(r"\d{6}", stamp):
            break
        months.append(stamp)
        values.append([float(cell) / 100 for cell in cells[1:]])
    index = pd.to_datetime(months, format="%Y%m")
    frame = pd.DataFrame(values, index=index, columns=columns)
    return cast("pd.DataFrame", frame.loc[FRENCH_START:])


def load_factors(root: Path) -> pd.DataFrame:
    """The saved monthly factors, empty when there are none or columns differ."""
    path = root / FACTORS_FILE
    if not path.is_file():
        return _empty()
    frame = _read(path)
    return frame if list(frame.columns) == [*FACTORS, "RF"] else _empty()


class _FactsFile(_Frozen):
    stocks: dict[str, Facts] = Field(default_factory=dict)


def load_facts(root: Path) -> dict[str, Facts]:
    """The saved quote figures by ticker; none when the file is missing or bad."""
    path = root / FACTS_FILE
    if not path.is_file():
        return {}
    try:
        return dict(_FactsFile.model_validate_json(path.read_bytes()).stocks)
    except (OSError, ValidationError):
        return {}


def save_facts(root: Path, fresh: Mapping[str, Facts]) -> None:
    """Merge ``fresh`` into the saved figures, atomically."""
    root.mkdir(parents=True, exist_ok=True)
    body = _FactsFile(stocks={**load_facts(root), **fresh})
    path = root / FACTS_FILE
    temp = path.with_name(f"{path.name}.tmp")
    _ = temp.write_text(body.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _ = temp.replace(path)


def facts_of(info: Mapping[str, object], day: date) -> Facts:
    """``Facts`` from a ``Ticker.info`` mapping."""
    cap = info.get("marketCap")
    currency = info.get("currency")
    return Facts(
        day=day,
        currency=currency if isinstance(currency, str) else "",
        market_cap=float(cap) if isinstance(cap, int | float) else math.nan,
    )


def _empty() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([]), dtype="float64")


@dataclass(frozen=True, slots=True)
class Prices:
    close: pd.DataFrame  # daily closes, one column per ticker
    adjusted: pd.DataFrame  # daily adjusted closes, same columns
    # Raw daily opens, highs and lows (split- not dividend-adjusted, like
    # ``close``), only for the tickers they were asked for.
    open: pd.DataFrame = field(default_factory=_empty)
    high: pd.DataFrame = field(default_factory=_empty)
    low: pd.DataFrame = field(default_factory=_empty)

    @property
    def ranges(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.open, self.high, self.low

    @property
    def tickers(self) -> frozenset[str]:
        return frozenset(str(c) for c in self.close.columns) & frozenset(
            str(c) for c in self.adjusted.columns
        )


def no_prices() -> Prices:
    return Prices(_empty(), _empty())


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return _empty()


def load_prices(root: Path) -> Prices:
    try:
        close = pd.read_parquet(root / CLOSE_FILE)
        adjusted = pd.read_parquet(root / ADJUSTED_FILE)
    except (OSError, ValueError):
        return no_prices()
    found = [
        _read(root / name) if (root / name).is_file() else _empty()
        for name in RANGE_FILES
    ]
    return Prices(close, adjusted, *found)


def save_prices(root: Path, prices: Prices) -> None:
    """The closes, and the opens, highs and lows once any were asked for."""
    root.mkdir(parents=True, exist_ok=True)
    _write(prices.adjusted, root / ADJUSTED_FILE)
    _write(prices.close, root / CLOSE_FILE)
    for frame, name in zip(prices.ranges, RANGE_FILES, strict=True):
        if len(frame.columns):
            _write(frame, root / name)


def prices_saved_at(root: Path) -> datetime | None:
    path = root / CLOSE_FILE
    if not path.is_file():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).astimezone()


def merge(old: Prices, new: Prices) -> Prices:
    """``old`` with the tickers in ``new`` replaced by their fresh history.

    For ``LONG_TICKERS`` (indices and USD/KRW, no dividends) the saved days
    before the fresh history starts are kept, so a refresh from
    ``HISTORY_START`` leaves the years back to ``LONG_START`` in place.
    """

    def one(before: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        if not len(fresh.columns):  # e.g. no opens, highs or lows kept
            return before
        kept = before.drop(columns=list(fresh.columns), errors="ignore")
        columns: dict[str, pd.Series] = {}
        for name in fresh.columns:
            new_ = cast("pd.Series", fresh[name]).dropna()
            if name in LONG_TICKERS and name in before.columns:
                older = cast("pd.Series", before[name]).dropna()
                if not new_.empty:
                    older = cast("pd.Series", older[older.index < new_.index[0]])
                new_ = pd.concat([older, new_])
            columns[str(name)] = new_
        added = pd.DataFrame(columns, dtype="float64")
        return pd.concat([kept, added], axis=1).sort_index()

    return Prices(
        one(old.close, new.close),
        one(old.adjusted, new.adjusted),
        *(one(a, b) for a, b in zip(old.ranges, new.ranges, strict=True)),
    )


def kept_ranges(fresh: Prices, saved: Prices, asked: bool) -> Prices:
    """``fresh`` with opens, highs and lows only where ``asked`` or already saved.

    Yahoo sends them with every download; they are kept for a ticker only
    once the user asked for them, then refreshed with its closes.
    """
    if asked:
        return fresh
    have = set(saved.high.columns)
    return replace(
        fresh,
        **{
            name: frame[[c for c in frame.columns if c in have]]
            for name, frame in zip(("open", "high", "low"), fresh.ranges, strict=True)
        },
    )


def _write(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(temp)
    _ = temp.replace(path)


# ---------------------------------------------------------------- currency


def convert(amount: float, native: Currency, base: Currency, usdkrw: float) -> float:
    """``amount`` in ``native`` as ``base``, at ``usdkrw`` won per dollar."""
    if native == base:
        return amount
    return amount * usdkrw if base == "KRW" else amount / usdkrw


def _no_rates() -> pd.Series:
    return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))


def fx_series(close: pd.DataFrame) -> pd.Series:
    if FX not in close.columns:
        return _no_rates()
    return cast("pd.Series", close[FX]).dropna()


def rate_on(fx: pd.Series, day: date) -> float:
    """USD/KRW on ``day``, or the last close before it within ``GAP_DAYS``."""
    known = fx.loc[: pd.Timestamp(day)]
    if known.empty:
        return math.nan
    last = cast("pd.Timestamp", known.index[-1])
    if (pd.Timestamp(day) - last).days > GAP_DAYS:
        return math.nan
    return float(known.iloc[-1])


def last_close(close: pd.DataFrame, ticker: str) -> tuple[float, date | None]:
    if ticker not in close.columns:
        return math.nan, None
    series = cast("pd.Series", close[ticker]).dropna()
    if series.empty:
        return math.nan, None
    return float(series.iloc[-1]), cast("pd.Timestamp", series.index[-1]).date()


def close_on(close: pd.DataFrame, ticker: str, day: date) -> tuple[float, date | None]:
    """The close on ``day``, or the last one before it, and its date."""
    if ticker not in close.columns:
        return math.nan, None
    known = cast("pd.Series", close[ticker]).dropna().loc[: pd.Timestamp(day)]
    if known.empty:
        return math.nan, None
    return float(known.iloc[-1]), cast("pd.Timestamp", known.index[-1]).date()


def trading_days(frame: pd.DataFrame, names: Sequence[str]) -> pd.DatetimeIndex:
    """Days on which any of ``names`` has a price."""
    return cast(
        "pd.DatetimeIndex", frame.reindex(columns=list(names)).dropna(how="all").index
    )


def carry_over(frame: pd.DataFrame, limit: int = FILL_DAYS) -> pd.DataFrame:
    """Each column's last close carried over its own gaps, by calendar days.

    Up to ``limit`` since its last close, so a market's first return after
    its holiday (or USD/KRW's after an outage) spans the whole gap and is
    kept; past a column's last close only ``GAP_DAYS``, so a series that
    stopped is not carried on. ``frame`` is sorted by date.
    """
    if frame.empty:
        return frame
    known = frame.notna().to_numpy()
    day = pd.DatetimeIndex(frame.index).to_numpy("datetime64[D]").astype(np.int64)
    seen = np.where(known, day[:, None], np.iinfo(np.int64).min // 2)
    age = day[:, None] - np.maximum.accumulate(seen, axis=0)
    last = seen.max(axis=0)
    ok = (age <= limit) & ((day[:, None] <= last) | (age <= GAP_DAYS))
    return frame.ffill().where(ok)


def in_base(
    frame: pd.DataFrame,
    names: Sequence[str],
    base: Currency,
    days: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Prices of ``names`` in ``base`` on ``days`` (by default when any traded)."""
    on = trading_days(frame, names) if days is None else days
    columns = [*dict.fromkeys([*names, FX])]
    filled = carry_over(frame.reindex(columns=columns).sort_index())
    filled = filled.reindex(on)
    fx = cast("pd.Series", filled[FX])
    out: dict[str, pd.Series] = {}
    for name in names:
        values = cast("pd.Series", filled[name])
        if currency(name) == base:
            out[name] = values
        else:
            out[name] = values * fx if base == "KRW" else values / fx
    return pd.DataFrame(out, index=on, columns=list(names), dtype="float64")


# ---------------------------------------------------------------- valuation


@dataclass(frozen=True, slots=True)
class Holding:
    position: Position
    price: float  # last close, native; nan when unknown
    day: date | None  # of that close
    value: float  # base, at the latest USD/KRW; 0 once sold out

    @property
    def ticker(self) -> str:
        return self.position.ticker

    @property
    def currency(self) -> Currency:
        return currency(self.ticker)

    @property
    def is_open(self) -> bool:
        return self.position.quantity > EPSILON

    @property
    def unrealized(self) -> float:
        """평가손익 in base: value now minus the basis at buy-day rates."""
        return self.value - self.position.cost_base if self.is_open else 0.0

    @property
    def native_return(self) -> float:
        return self.price / self.position.average - 1 if self.is_open else math.nan

    @property
    def base_return(self) -> float:
        cost = self.position.cost_base
        return self.value / cost - 1 if self.is_open and cost else math.nan

    @property
    def contribution(self) -> float:
        """Realized plus unrealized P&L in base, sold-out tickers included."""
        return self.position.realized_base + self.unrealized


@dataclass(frozen=True, slots=True)
class Totals:
    value: float  # open positions, base, latest closes and USD/KRW
    cost: float  # their basis at buy-day USD/KRW
    realized: float  # every sale, base
    sold: float  # basis of the shares sold, base
    bought: float  # every buy with its fee, base

    @property
    def unrealized(self) -> float:
        return self.value - self.cost

    @property
    def pnl(self) -> float:
        return self.realized + self.unrealized

    @property
    def realized_return(self) -> float:
        return self.realized / self.sold if self.sold else math.nan

    @property
    def open_return(self) -> float:
        return self.unrealized / self.cost if self.cost else math.nan

    @property
    def total_return(self) -> float:
        """Total P&L over everything ever bought (fees included)."""
        return self.pnl / self.bought if self.bought else math.nan


def ledger(portfolio: Portfolio, close: pd.DataFrame) -> Ledger:
    return replay(portfolio.trades, portfolio.base, fx_series(close))


def holdings(portfolio: Portfolio, close: pd.DataFrame) -> list[Holding]:
    """Every ticker traded: open positions by base value, then sold-out ones."""
    usdkrw, _ = last_close(close, FX)
    out: list[Holding] = []
    for ticker, position in ledger(portfolio, close).positions.items():
        price, day = last_close(close, ticker)
        open_ = position.quantity > EPSILON
        worth = convert(
            position.quantity * price, currency(ticker), portfolio.base, usdkrw
        )
        out.append(Holding(position, price, day, worth if open_ else 0.0))
    return sorted(out, key=lambda h: (not h.is_open, -_finite(h.value), h.ticker))


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def totals(items: Iterable[Holding]) -> Totals:
    """Sums in base; any unknown value leaves its total unknown."""
    items = list(items)
    held = [h for h in items if h.is_open]
    return Totals(
        value=sum(h.value for h in held),
        cost=sum(h.position.cost_base for h in held),
        realized=sum(h.position.realized_base for h in items),
        sold=sum(h.position.sold_base for h in items),
        bought=sum(h.position.bought_base for h in items),
    )


def weights(items: Iterable[Holding]) -> pd.Series:
    """Current weights by base value, open positions with a known value only."""
    values = {h.ticker: h.value for h in items if h.is_open and math.isfinite(h.value)}
    total = sum(values.values())
    if total <= 0:
        return pd.Series(dtype="float64")
    return pd.Series({t: v / total for t, v in values.items()}, dtype="float64")


def unpriced(items: Iterable[Holding]) -> list[str]:
    """Open positions whose value is unknown (no close or no USD/KRW)."""
    return [h.ticker for h in items if h.is_open and not math.isfinite(h.value)]


def realized_by(
    sales: Iterable[Sale], *, monthly: bool = False
) -> list[tuple[str, float]]:
    """Realized P&L in base per sale year (or month), oldest first."""
    out: dict[str, float] = {}
    for sale in sorted(sales, key=lambda s: s.trade.day):
        label = f"{sale.trade.day:%Y-%m}" if monthly else f"{sale.trade.day:%Y}"
        out[label] = out.get(label, 0.0) + sale.pnl_base
    return list(out.items())


def flow(trade: Trade, base: Currency, fx: pd.Series) -> float:
    """Cash into the portfolio for ``trade`` in ``base``; sells are negative."""
    amount = trade.quantity * trade.price
    native = amount + trade.fee if trade.side == "buy" else trade.fee - amount
    rate = rate_on(fx, trade.day) if currency(trade.ticker) != base else 1.0
    return convert(native, currency(trade.ticker), base, rate)


def quantities(trades: Sequence[Trade], days: pd.DatetimeIndex) -> pd.DataFrame:
    """Shares held at the end of each of ``days``, one column per ticker."""
    rows = [
        (pd.Timestamp(t.day), t.ticker, t.quantity if t.side == "buy" else -t.quantity)
        for t in trades
    ]
    if not rows:
        return pd.DataFrame(index=days, dtype="float64")
    signed = pd.DataFrame(rows, columns=["day", "ticker", "quantity"])
    daily = signed.pivot_table(
        index="day", columns="ticker", values="quantity", aggfunc="sum"
    ).fillna(0.0)
    held = daily.cumsum()
    return held.reindex(days.union(held.index)).ffill().fillna(0.0).reindex(days)


def _running(amounts: pd.Series, days: pd.Index) -> pd.Series:
    """The running total of dated ``amounts`` on each of ``days``.

    Zero before the first amount; unknown from an unknown amount on.
    """
    if amounts.empty:
        return pd.Series(0.0, index=days)
    summed = cast("pd.Series", amounts.groupby(level=0).sum(min_count=1))
    total = summed.sort_index().cumsum(skipna=False)
    out = total.reindex(days, method="ffill")
    return out.where(days >= total.index[0], 0.0)


def history(portfolio: Portfolio, close: pd.DataFrame) -> pd.DataFrame:
    """Daily base ``value``, net ``invested`` cash and ``realized`` P&L.

    From the first trade, on days the holdings traded; days on which an open
    position has no close are left out.
    """
    names = sorted({t.ticker for t in portfolio.trades})
    if not names:
        return pd.DataFrame(columns=["value", "invested", "realized"], dtype="float64")
    first = pd.Timestamp(min(t.day for t in portfolio.trades))
    prices = in_base(close, names, portfolio.base)
    prices = prices[prices.index >= first]
    held = quantities(portfolio.trades, cast("pd.DatetimeIndex", prices.index))
    held = held.reindex(columns=names).fillna(0.0)
    unknown = ((held > EPSILON) & prices.isna()).any(axis=1)
    value = (held * prices).sum(axis=1)[~unknown]
    fx = fx_series(close)
    flows = pd.Series(
        [flow(t, portfolio.base, fx) for t in portfolio.trades],
        index=pd.DatetimeIndex([pd.Timestamp(t.day) for t in portfolio.trades]),
        dtype="float64",
    )
    sales = replay(portfolio.trades, portfolio.base, fx).sales
    gains = pd.Series(
        [s.pnl_base for s in sales],
        index=pd.DatetimeIndex([pd.Timestamp(s.trade.day) for s in sales]),
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "value": value,
            "invested": _running(flows, value.index),
            "realized": _running(gains, value.index),
        }
    )


def twr(portfolio: Portfolio, close: pd.DataFrame) -> pd.Series:
    """Cumulative time-weighted return in base since the first trade.

    Each day's return is (value - previous value - flow) ÷ (previous value +
    net inflow). Buys and sells on the same day net out; a net inflow counts
    from the start of the day, a net outflow leaves at its end. Closes
    without dividends. Empty when a flow is unknown.
    """
    frame = history(portfolio, close)
    value = cast("pd.Series", frame["value"])
    if value.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    days = cast("pd.DatetimeIndex", value.index)
    fx = fx_series(close)
    flows = np.zeros(len(days))
    for trade in portfolio.trades:
        at = int(np.searchsorted(days.to_numpy(), np.datetime64(trade.day, "ns")))
        if at < len(days):
            flows[at] += flow(trade, portfolio.base, fx)
    if not np.isfinite(flows).all():
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    now = value.to_numpy(dtype="float64")
    before = np.concatenate([[0.0], now[:-1]])
    start = before + np.clip(flows, 0.0, None)
    daily = np.divide(
        now - before - flows, start, out=np.zeros(len(days)), where=start > 0
    )
    return pd.Series(np.cumprod(1 + daily) - 1, index=days, dtype="float64")


def money_weighted(
    p: Portfolio, close: pd.DataFrame, value: float
) -> perf.Annualised | None:
    """XIRR of the trades and today's ``value`` (``perf.mwr``).

    Cumulative only under a year; None when it cannot be solved.
    """
    fx = fx_series(close)
    last = max((h.day for h in holdings(p, close) if h.day), default=None)
    if not p.trades or last is None or not math.isfinite(value):
        return None
    first = min(t.day for t in p.trades)
    days = [(t.day - first).days for t in p.trades] + [(last - first).days]
    cash = [-flow(t, p.base, fx) for t in p.trades] + [value]
    if not all(math.isfinite(c) for c in cash):
        return None
    return perf.mwr(np.array(days, dtype="float64"), np.array(cash, dtype="float64"))


def span_days(index: pd.Index) -> float:
    """Calendar days from the first to the last date of ``index``."""
    if len(index) < 2:  # noqa: PLR2004 - a span needs two dates
        return 0.0
    first = cast("pd.Timestamp", index[0])
    last = cast("pd.Timestamp", index[-1])
    return float((last - first).days)


def per_year(index: pd.Index) -> float:
    """Return observations a year in ``index``: steps x 365 / their calendar days.

    A mix's days are the union of the Korean and US calendars (about 258 a
    year; KOSPI alone about 243, the S&P 500 251), so annualising by a fixed
    252 would misstate both. Steps longer than ``FILL_DAYS`` (a sold-out
    stretch ``held_returns`` leaves out, not a holiday) are not counted.
    ``TRADING_DAYS`` without a step.
    """
    days = pd.DatetimeIndex(index).to_numpy("datetime64[D]").astype(np.int64)
    steps = np.diff(days)
    steps = steps[steps <= FILL_DAYS]
    return len(steps) * YEAR_DAYS / steps.sum() if steps.sum() > 0 else TRADING_DAYS


def time_weighted(cumulative: pd.Series) -> perf.Annualised | None:
    """A cumulative return over its span; annual only from a year on."""
    known = cumulative.dropna()
    if known.empty:
        return None
    return perf.annualise(float(known.iloc[-1]), span_days(known.index))


def gap(first: perf.Annualised, second: perf.Annualised) -> perf.Gap:
    """``first`` minus ``second``: annual rates from a year on, else cumulative."""
    if first.annual is not None and second.annual is not None:
        return perf.Gap(first.annual - second.annual, annualised=True)
    return perf.Gap(first.cumulative - second.cumulative, annualised=False)


# ---------------------------------------------------------------- statistics


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).iloc[1:]


def window[T: (pd.Series, pd.DataFrame)](data: T, months: int) -> T:
    """The last ``months`` of ``data``."""
    if data.empty:
        return data
    last = cast("pd.Timestamp", data.index[-1])
    return data.truncate(before=last - pd.DateOffset(months=months))


def held_returns(
    portfolio: Portfolio, close: pd.DataFrame, adjusted: pd.DataFrame
) -> pd.Series:
    """Daily returns of the positions held at the previous close, in base.

    Each position weighs its value at the previous close (closes, as the
    book is valued) and moves by its adjusted close, so a dividend counts on
    its day without tilting the weights before it. Time-weighted, so buys
    and sells do not count as gains. Days on which an open position has no
    price are left out.
    """
    names = sorted({t.ticker for t in portfolio.trades})
    if not names:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    prices = in_base(close, names, portfolio.base)
    days = cast("pd.DatetimeIndex", prices.index)
    growth = in_base(adjusted, names, portfolio.base, days)
    held = quantities(portfolio.trades, days).reindex(columns=names).fillna(0.0)
    held = held.shift(1).fillna(0.0)
    on = held > EPSILON
    worth = held * prices.shift(1)
    moved = growth / growth.shift(1)
    unknown = (on & (worth.isna() | moved.isna())).any(axis=1)
    start = worth.where(on, 0.0).sum(axis=1)
    keep = ~unknown & (start > 0)
    end = (worth * moved).where(on, 0.0).sum(axis=1)
    return (end[keep] / start[keep] - 1).rename("portfolio")


def asset_returns(
    adjusted: pd.DataFrame, names: Sequence[str], base: Currency
) -> pd.DataFrame:
    """Daily returns of ``names`` in ``base`` on the days any of them traded."""
    return returns(in_base(adjusted, names, base))


def late_starts(rets: pd.DataFrame) -> str:
    """Columns whose returns start after the window does, as page text."""
    if rets.empty:
        return ""
    first = cast("pd.Timestamp", rets.index[0])
    late: list[str] = []
    for name in rets.columns:
        start = cast("pd.Timestamp | None", rets[name].first_valid_index())
        if start is None:
            late.append(f"{name} 없음")
        elif start > first:
            late.append(f"{name} {start.date()}")
    return ", ".join(late)


def weighted(rets: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Daily returns at fixed weights ``w``, on days every holding has one."""
    joint = rets.reindex(columns=list(w.index)).dropna()
    return pd.Series(
        joint.to_numpy(dtype="float64") @ w.to_numpy(dtype="float64"),
        index=joint.index,
        dtype="float64",
    )


def covered(rets: pd.DataFrame, w: pd.Series) -> tuple[pd.Series, list[str]]:
    """``w`` without holdings whose returns start late or stop early, scaled to 1.

    For a window with many holdings, where one listing (or delisting) inside
    it would otherwise leave ``weighted`` almost no days. Returns the dropped
    names.
    """
    if rets.empty:
        return w.iloc[:0], [str(t) for t in w.index]
    first = cast("pd.Timestamp", rets.index[0])
    end = cast("pd.Timestamp", rets.index[-1])
    kept: list[str] = []
    dropped: list[str] = []
    for name in w.index:
        column = cast("pd.Series", rets[name]) if name in rets.columns else None
        whole = (
            column is not None
            and column.first_valid_index() == first
            and column.last_valid_index() == end
        )
        (kept if whole else dropped).append(str(name))
    part = w.reindex(kept)
    total = float(part.sum())
    return (part / total if total > 0 else part.iloc[:0]), dropped


@dataclass(frozen=True, slots=True)
class Step:
    """One purchase of a weight set by ``clone``."""

    day: date  # the close it was bought at
    names: int  # holdings bought
    turnover: float | None  # one-way, against the drifted set before; None from cash


@dataclass(frozen=True, slots=True)
class Clone:
    cumulative: pd.Series  # from 0 at the first purchase; empty when none
    steps: tuple[Step, ...]
    weights: pd.Series  # the last set drifted to the last day, priced holdings only


def clone(prices: pd.DataFrame, schedule: Sequence[tuple[date, pd.Series]]) -> Clone:
    """Buy each weight set and hold it to the next (``rebalanced``), step by step.

    Each step's turnover is ``perf.turnover`` of the new set against the
    previous one drifted to that close; the first purchase, or one after a
    period in cash, has none. A holding is bought only at a close at most
    ``GAP_DAYS`` old: these are US stocks on one calendar, so an older one
    means a halt, and a weight set is not bought at a stale price.
    """
    names = sorted({str(t) for _, w in schedule for t in w.index})
    frame = prices.reindex(columns=names).sort_index()
    frame = frame[frame.notna().any(axis=1)]
    days = cast("pd.DatetimeIndex", frame.index)
    near = carry_over(cast("pd.DataFrame", frame), GAP_DAYS)
    carried = frame.ffill()
    starts: dict[int, pd.Series] = {}
    for day, w in sorted(schedule, key=lambda it: it[0]):
        at = int(np.searchsorted(days.to_numpy(), np.datetime64(day, "ns")))
        if at < len(days):
            starts[at] = w  # a later set on the same day replaces an earlier one
    order = sorted(starts)
    pieces: list[pd.Series] = []
    steps: list[Step] = []
    drifted: pd.Series | None = None
    level = 1.0
    for i, at in enumerate(order):
        end = order[i + 1] if i + 1 < len(order) else len(days) - 1
        w = starts[at]
        first = cast("pd.Series", near.iloc[at])
        held = [
            str(t)
            for t in w.index
            if math.isfinite(float(cast("float", first[t])))
            and float(cast("float", w[t])) > 0
        ]
        part = w.reindex(held)
        total = float(part.sum())
        if total <= 0:
            if pieces:  # nothing to buy: the money waits as cash
                pieces.append(pd.Series(level, index=days[at + 1 : end + 1]))
            drifted = None
            continue
        mix = part / total
        relative = carried.iloc[at : end + 1][held] / first[held]
        growth = relative.to_numpy(dtype="float64") @ mix.to_numpy(dtype="float64")
        path = pd.Series(level * growth, index=relative.index, dtype="float64")
        pieces.append(path if not pieces else path.iloc[1:])
        level = float(path.iloc[-1])
        day = cast("pd.Timestamp", days[at]).date()
        moved = None if drifted is None else perf.turnover(mix, drifted)
        steps.append(Step(day, len(held), moved))
        drifted = perf.drift(mix, cast("pd.Series", relative.iloc[-1]))
    if not pieces:
        empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        return Clone(empty, (), pd.Series(dtype="float64"))
    now = _priced_now(drifted, near)
    return Clone(pd.concat(pieces) - 1, tuple(steps), now)


def _priced_now(w: pd.Series | None, near: pd.DataFrame) -> pd.Series:
    """``w`` without holdings whose price has stopped, scaled back to 1."""
    if w is None or near.empty:
        return pd.Series(dtype="float64")
    now = cast("pd.Series", near.iloc[-1]).reindex(w.index)
    part = cast("pd.Series", w[now.notna() & (w > 0)])
    total = float(part.sum())
    return part / total if total > 0 else part.iloc[:0]


def rebalanced(
    prices: pd.DataFrame, schedule: Sequence[tuple[date, pd.Series]]
) -> pd.Series:
    """Cumulative return of buying each weight set and holding it to the next.

    Each set is bought at the close of the first day on or after its date
    that any of its holdings traded, then left to drift. Holdings without a
    price that day are left out and the rest scaled back to 1. A price that
    stops inside a period is carried at its last value (as cash). Empty when
    no set can be bought.
    """
    return clone(prices, schedule).cumulative


def since(cumulative: pd.Series, start: pd.Timestamp) -> pd.Series:
    """A cumulative return restarted at 0 on its first day on or after ``start``."""
    later = cumulative.dropna().loc[start:]
    if later.empty:
        return later
    return (1 + later) / (1 + float(later.iloc[0])) - 1


def daily(cumulative: pd.Series) -> pd.Series:
    """Daily returns of a cumulative return series."""
    return (1 + cumulative).pct_change(fill_method=None).iloc[1:]


def korean(name: str) -> bool:
    """A Korean listing or index: it closes hours before New York opens."""
    return name.endswith(KRW_SUFFIXES) or name in KRW_INDICES


def mixed(names: Iterable[str]) -> bool:
    """Korean and other tickers together: their closes are hours apart."""
    names = list(names)
    return any(korean(n) for n in names) and not all(korean(n) for n in names)


def weekly[T: (pd.Series, pd.DataFrame)](rets: T) -> T:
    """Daily returns compounded into Friday-ending weeks."""
    return (1 + rets).resample(WEEK).prod(min_count=1) - 1


def korean_weight(w: pd.Series) -> float:
    """The Korean listings' share of ``w``."""
    return float(sum(float(v) for t, v in w.items() if korean(str(t))))


def us_sleeve(w: pd.Series) -> pd.Series:
    """The dollar-listed part of ``w``, scaled back to 1; empty without one."""
    part = cast("pd.Series", w[[currency(str(t)) == "USD" for t in w.index]])
    total = float(part.sum())
    return part / total if total > 0 else part.iloc[:0]


def lineup(
    book: Book, prices: Prices, base: Currency, months: int
) -> tuple[pd.DataFrame, list[bool]]:
    """Each portfolio's current-weights returns and the S&P 500's, in ``base``.

    The S&P 500 with dividends (``TOTAL_RETURN``), as the holdings' adjusted
    closes have them, when it has any. Holdings whose returns start or stop
    inside the window are left out (``covered``). The flags say which columns
    close in Seoul: a portfolio more than ``KOREAN_MAJORITY`` Korean by value.
    """
    names = [*book_tickers(book), TOTAL_RETURN]
    rets = window(asset_returns(prices.adjusted, names, base), months)
    columns: dict[str, pd.Series] = {}
    flags: list[bool] = []
    for p in book.portfolios:
        w, _ = covered(rets, weights(holdings(p, prices.close)))
        if not w.empty:
            columns[p.name] = weighted(rets, w)
            flags.append(korean_weight(w) > KOREAN_MAJORITY)
    index = cast("pd.Series", rets[TOTAL_RETURN])
    if index.notna().any():  # none until ^SP500TR is fetched
        columns[BENCHMARK_LABEL] = index
        flags.append(False)
    return pd.DataFrame(columns), flags


# ---------------------------------------------------------------- returns panel


def held_values(
    p: Portfolio, frame: pd.DataFrame, days: pd.DatetimeIndex
) -> pd.DataFrame:
    """Each holding's value in base at the end of each of ``days``."""
    names = sorted({t.ticker for t in p.trades})
    prices = in_base(frame, names, p.base, days)
    held = quantities(p.trades, days).reindex(columns=names).fillna(0.0)
    return held * prices


def korean_share(
    p: Portfolio, frame: pd.DataFrame, days: pd.DatetimeIndex
) -> pd.Series:
    """The Korean holdings' share of the base value at each day's prior close."""
    values = held_values(p, frame, days)
    total = values.sum(axis=1, min_count=1)
    kr = values[[c for c in values.columns if korean(str(c))]].sum(axis=1)
    return (kr / total.where(total > 0)).shift(1)


def index_returns(
    frame: pd.DataFrame, base: Currency, days: pd.DatetimeIndex
) -> pd.DataFrame:
    """Daily returns of the S&P 500 (total return) and KOSPI (price) on ``days``.

    In ``base``; Yahoo has no free KOSPI total-return index.
    """
    names = [TOTAL_RETURN, KOSPI]
    return in_base(frame, names, base, days).pct_change(fill_method=None)


def blended(p: Portfolio, frame: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """The KOSPI (price) / S&P 500 (total return) mix in base, drifting with it.

    Each day's Korean weight is the portfolio's Korean share at the prior
    close (``perf.blend``); a leg with no weight that day is not needed.
    Returns and shares are taken on the indices' own days, so a gap in
    ``days`` never stretches one over several days.
    """
    legs = [TOTAL_RETURN, KOSPI]
    grid = cast("pd.DatetimeIndex", trading_days(frame, legs).union(days))
    rets = index_returns(frame, p.base, grid).reindex(days)
    share = korean_share(p, frame, grid).reindex(days)
    kr = cast("pd.Series", rets[KOSPI]).where(share > 0, 0.0)
    us = cast("pd.Series", rets[TOTAL_RETURN]).where(share < 1, 0.0)
    return pd.Series(perf.blend(share, kr, us), index=days, dtype="float64")


def cumulative(daily_rets: pd.Series) -> pd.Series:
    """Growth from 0 on the first day; the first day's own return is left out."""
    if daily_rets.empty:
        return daily_rets
    steps = daily_rets.copy()
    steps.iloc[0] = 0.0
    out = (1 + steps).cumprod() - 1
    return out.where(steps.notna())


def benchmark_lines(
    p: Portfolio, close: pd.DataFrame, days: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    """Cumulative mix, S&P 500 (total return) and KOSPI (price) in base on ``days``."""
    rets = index_returns(close, p.base, days)
    return {
        "혼합 벤치마크": cumulative(blended(p, close, days)),
        BENCHMARK_LABEL: cumulative(cast("pd.Series", rets[TOTAL_RETURN])),
        "KOSPI": cumulative(cast("pd.Series", rets[KOSPI])),
    }


def currency_split(p: Portfolio, close: pd.DataFrame) -> perf.FxSplit | None:
    """Local, currency and cross parts of the held positions' chained return.

    Closes, positions held at each prior close (as ``held_returns``), so the
    base part is that chained return. None when every holding is in the base
    currency or no day can be split.
    """
    names = sorted({t.ticker for t in p.trades})
    if not names or all(currency(n) == p.base for n in names):
        return None
    first = pd.Timestamp(min(t.day for t in p.trades))
    days = trading_days(close, names)
    days = cast("pd.DatetimeIndex", days[days >= first])
    filled = carry_over(close.reindex(columns=[*names, FX]).sort_index())
    filled = filled.reindex(days)
    rate = cast("pd.Series", filled[FX])
    per_dollar = rate if p.base == "KRW" else 1 / rate  # base per dollar
    moved = per_dollar.pct_change(fill_method=None)
    prices = in_base(close, names, p.base, days)
    held = quantities(p.trades, days).reindex(columns=names).fillna(0.0)
    held = held.shift(1).fillna(0.0)
    start = held * prices.shift(1)
    total = start.sum(axis=1)
    unknown = ((held > EPSILON) & (prices.isna() | prices.shift(1).isna())).any(axis=1)
    keep = ~unknown & (total > 0)
    if not keep.any():
        return None
    on = held[keep] > EPSILON
    local = cast("pd.DataFrame", filled[names].pct_change(fill_method=None)[keep])
    local = local.where(on, 0.0)
    fx = pd.DataFrame(
        {n: moved if currency(n) != p.base else 0.0 for n in names}, index=days
    )[keep].where(on, 0.0)
    w = start[keep].div(total[keep], axis=0)
    return perf.fx_split(w.to_numpy(), local.to_numpy(), fx.to_numpy())


@dataclass(frozen=True, slots=True)
class Trading:
    actual: float  # value now, base
    held: float  # the do-nothing book's value now
    effect: perf.Gap | None  # XIRR of the real book minus the do-nothing one


def trading(p: Portfolio, close: pd.DataFrame) -> Trading | None:
    """The real book against holding the first day's shares with no trades.

    Later buys and sells become pro-rata buys and sells of that first mix
    (``perf.counterfactual``); None without two valued days.
    """
    frame = history(p, close)
    value = cast("pd.Series", frame["value"])
    if len(value) < 2:  # noqa: PLR2004 - a path needs two days
        return None
    days = cast("pd.DatetimeIndex", value.index)
    fx = fx_series(close)
    flows = np.zeros(len(days))
    for trade in p.trades:
        at = int(np.searchsorted(days.to_numpy(), np.datetime64(trade.day, "ns")))
        if at < len(days):
            flows[at] += flow(trade, p.base, fx)
    flows[0] = 0.0  # the first day's trades are the starting shares
    shares = quantities(p.trades, days).iloc[0]
    mix = [str(t) for t, q in shares.items() if float(q) > EPSILON]
    prices = in_base(close, mix, p.base, days).ffill()
    if not mix or not np.isfinite(flows).all() or prices.iloc[0].isna().any():
        return None
    held = perf.counterfactual(
        prices.to_numpy(dtype="float64"), shares[mix].to_numpy(dtype="float64"), flows
    )
    offsets = (days - days[0]).days.to_numpy(dtype="float64")
    cash = -flows
    cash[0] = -float(value.iloc[0])
    actual = float(value.iloc[-1])
    effect = perf.trading_effect(offsets, cash, actual, float(held[-1]))
    return Trading(actual, float(held[-1]), effect)


# ---------------------------------------------------------------- risk-adjusted


def risk_free(
    rates: pd.Series, index: pd.DatetimeIndex, quote: perf.Quote
) -> pd.Series:
    """Daily risk-free returns on ``index`` from annual percent ``rates``.

    ``quote`` is how the series states them: ``discount`` for ^IRX,
    ``simple`` for CD 91-day (``perf.effective_annual``), compounded over
    ``index``'s own days a year (``per_year``). 0 every day when there are
    no rates; NaN before the first rate, and ``ratios`` leaves those days
    out.
    """
    known = rates.dropna()
    if known.empty:
        return pd.Series(0.0, index=index, dtype="float64")
    return perf.daily_rf(known, index, quote, per_year(index))


def wealth(rets: pd.Series) -> pd.Series:
    """Growth of 1 from the business day before the first return."""
    if rets.empty:
        return rets
    before = cast("pd.Timestamp", rets.index[0]) - pd.offsets.BDay(1)
    growth = (1 + rets).cumprod()
    return pd.concat([pd.Series([1.0], index=pd.DatetimeIndex([before])), growth])


def standing(rets: pd.Series) -> tuple[float, float]:
    """Today's wealth and its highest point, from ``rets``' growth of 1."""
    level = wealth(rets)
    if level.empty:
        return 1.0, 1.0
    return float(level.iloc[-1]), float(level.max())


def _annual(value: float | None, periods: float) -> float | None:
    return None if value is None else value * math.sqrt(periods)


@dataclass(frozen=True, slots=True)
class Ratios:
    days: int  # returns with a known risk-free rate
    span: float  # calendar days
    volatility: float | None
    under: perf.Underwater | None
    sharpe: float | None  # annualised
    ci: perf.SharpeCI | None  # None under perf.MIN_OBS returns
    psr: float | None
    min_trl: float | None  # trading days
    sortino: float | None  # annualised
    calmar: float | None  # None under a year
    ulcer: float
    upi: float | None
    omega: float | None
    cdar: float
    te: float | None
    ir: float | None


def ratios(rets: pd.Series, rf: pd.Series, bench: pd.Series) -> Ratios:
    """Risk-adjusted statistics of daily ``rets`` over daily ``rf``.

    Days without a rate (NaN in ``rf``) are left out. ``bench`` is daily
    benchmark returns for tracking error and the information ratio. Annual
    figures count the returns' own days a year (``per_year``).
    """
    rate = rf.reindex(rets.index)
    known = rate.notna() & rets.notna()
    r = cast("pd.Series", rets[known])
    f = cast("pd.Series", rate[known])
    x = (r - f).to_numpy(dtype="float64")
    n = len(r)
    periods = per_year(r.index)
    level = wealth(r)
    span = span_days(level.index)
    sr = perf.sharpe(x)
    moments = perf.skew_kurt(x) if n else None
    trl = None
    if n >= perf.MIN_OBS and sr is not None and moments is not None:
        trl = perf.min_trl(sr, *moments)
    mean_rf = float(f.mean()) if n else 0.0
    b = bench.reindex(r.index)
    paired = b.notna()
    return Ratios(
        days=n,
        span=span,
        volatility=float(x.std(ddof=1)) * math.sqrt(periods) if n > 1 else None,
        under=perf.underwater(level) if n else None,
        sharpe=_annual(sr, periods),
        ci=perf.sharpe_ci(x, periods=periods),
        psr=perf.psr(x),
        min_trl=trl,
        sortino=_annual(perf.sortino(r, f), periods),
        calmar=perf.calmar(r, span),
        ulcer=perf.ulcer_index(r),
        upi=perf.ulcer_performance(r, span, (1 + mean_rf) ** periods - 1),
        omega=perf.omega(x),
        cdar=perf.cdar(r),
        te=perf.tracking_error(r[paired], b[paired], periods),
        ir=perf.information_ratio(r[paired], b[paired], periods),
    )


def market_returns(
    adjusted: pd.DataFrame,
    base: Currency,
    days: pd.DatetimeIndex,
    index: str = TOTAL_RETURN,
) -> pd.Series:
    """The S&P 500's (or another index's) daily returns in ``base`` on ``days``.

    Each is from the index's own previous close, so a gap in ``days`` never
    stretches one over several days.
    """
    grid = trading_days(adjusted, [index])
    rets = returns(in_base(adjusted, [index], base, grid))
    return cast("pd.Series", rets[index]).reindex(days)


def market_beta(rets: pd.Series, adjusted: pd.DataFrame, base: Currency) -> float:
    """Welch beta of daily ``rets`` against the S&P 500 in ``base``, same days.

    The price index, as ``betas`` and the stress betas: dividends add a near
    constant to its returns, which leaves a slope alone.
    """
    sp = market_returns(adjusted, base, pd.DatetimeIndex(rets.index), BENCHMARK)
    ok = rets.notna() & sp.notna()
    if int(ok.sum()) <= 2:  # noqa: PLR2004 - a slope needs three days
        return math.nan
    return quant.welch_beta(
        cast("pd.Series", rets[ok]).to_numpy(dtype="float64"),
        cast("pd.Series", sp[ok]).to_numpy(dtype="float64"),
    )


@dataclass(frozen=True, slots=True)
class Peer:
    name: str
    days: int  # returns with a known risk-free rate
    total: float  # compounded over those days
    sharpe: float | None  # annualised, over the risk-free rate
    shrunk: float | None  # empirical Bayes; None with fewer than two Sharpes


@dataclass(frozen=True, slots=True)
class Peers:
    rows: tuple[Peer, ...]  # highest Sharpe first, those without one last
    count: int  # series with a Sharpe: the N compared
    deflated: float | None  # DSR of the first row; None with N < 2 or a short record


def peers(series: Mapping[str, pd.Series], rf: pd.Series) -> Peers:
    """Sharpe ratios of several daily series side by side, ranked.

    Each is shrunk toward their mean by its Mertens standard error
    (``perf.shrink``). ``deflated`` asks whether the highest beats the
    highest of N expected by luck (``perf.expected_max_sr``), from
    ``perf.MIN_OBS`` returns on. ``rf`` is daily; days without one are left out.
    """
    found: list[_Rated] = []
    for name, rets in series.items():
        rate = rf.reindex(rets.index)
        known = rate.notna() & rets.notna()
        r = cast("pd.Series", rets[known])
        x = (r - cast("pd.Series", rate[known])).to_numpy(dtype="float64")
        total = float(np.prod(1 + r.to_numpy(dtype="float64")) - 1)
        sr, moments = perf.sharpe(x), perf.skew_kurt(x)
        skew, kurt = moments or (0.0, perf.NORMAL_KURT)
        rated_sr = sr if moments else None
        found.append(
            _Rated(name, len(x), total, rated_sr, skew, kurt, per_year(r.index))
        )
    rated = [f for f in found if f.sr is not None]
    shrunk: dict[str, float] = {}
    if len(rated) > 1:
        srs = [f.per for f in rated]
        ses = [perf.sharpe_se(f.per, f.n, f.skew, f.kurt) for f in rated]
        shrunk = dict(zip((f.name for f in rated), perf.shrink(srs, ses), strict=True))
    found.sort(key=lambda f: math.inf if f.sr is None else -f.sr)
    deflated = None
    if len(rated) > 1 and found[0].n >= perf.MIN_OBS:
        top = found[0]
        trials = np.array([f.per for f in rated], dtype="float64")
        bar = perf.expected_max_sr(len(rated), float(trials.var(ddof=1)))
        if bar is not None:
            deflated = perf.psr_from(top.per, top.n, top.skew, top.kurt, bar)
    rows = tuple(
        Peer(
            f.name,
            f.n,
            f.total,
            _annual(f.sr, f.periods),
            _annual(float(shrunk[f.name]), f.periods) if f.name in shrunk else None,
        )
        for f in found
    )
    return Peers(rows, len(rated), deflated)


@dataclass(frozen=True, slots=True)
class _Rated:
    name: str
    n: int
    total: float
    sr: float | None  # per period
    skew: float
    kurt: float  # raw
    periods: float  # returns a year (``per_year``)

    @property
    def per(self) -> float:
        return self.sr if self.sr is not None else math.nan


# ---------------------------------------------------------------- joint returns


@dataclass(frozen=True, slots=True)
class Joint:
    """Base-currency daily returns of the current holdings on days all have one.

    ``full`` runs from the latest first price; ``window`` is its last months
    and ``market`` the S&P 500 and KOSPI on the window's days.
    """

    names: tuple[str, ...]
    w: NDArray[np.float64]  # current weights, sum 1
    full: pd.DataFrame
    window: pd.DataFrame
    market: pd.DataFrame
    late: str  # holdings whose returns start inside the window

    @property
    def korean(self) -> NDArray[np.bool_]:
        return np.array([korean(n) for n in self.names], dtype=np.bool_)

    @property
    def mixed(self) -> bool:
        return mixed(self.names)

    @property
    def portfolio(self) -> NDArray[np.float64]:
        return self.window.to_numpy(dtype="float64") @ self.w


def joint(adjusted: pd.DataFrame, w: pd.Series, base: Currency, months: int) -> Joint:
    """``Joint`` returns of the holdings in ``w`` over the last ``months``."""
    names = labels(w)
    grid = trading_days(adjusted, names)
    extra = [i for i in (BENCHMARK, KOSPI) if i not in names]
    rets = returns(in_base(adjusted, [*names, *extra], base, grid))
    own = cast("pd.DataFrame", rets[names])
    full = own.dropna()
    part = window(full, months)
    return Joint(
        names=tuple(names),
        w=w.to_numpy(dtype="float64") / float(w.sum()),
        full=full,
        window=part,
        market=cast("pd.DataFrame", rets[[BENCHMARK, KOSPI]]).reindex(part.index),
        late=late_starts(window(own, months)),
    )


def covariance(j: Joint) -> tuple[NDArray[np.float64], float, float]:
    """Lag-adjusted, Ledoit-Wolf shrunk daily covariance of the window.

    Returns (covariance, lag factor f, shrinkage weight).
    """
    return _shrunk(j.window.to_numpy(dtype="float64"), j.korean)


def _shrunk(
    x: NDArray[np.float64], korean_: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], float, float]:
    lagged, f = quant.lagged_covariance(x, korean_)
    cov, weight = quant.ledoit_wolf(x, sample=lagged)
    return cov, f, weight


def correlation_frame(cov: NDArray[np.float64], names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(quant.correlation(cov), index=list(names), columns=list(names))


@dataclass(frozen=True, slots=True)
class Correlation:
    matrix: pd.DataFrame
    lag: float  # factor f on the Korea/US lag terms (``quant.lagged_covariance``)
    shrink: float  # Ledoit-Wolf weight
    days: int


def lagged_correlation(
    frame: pd.DataFrame, korean_: Sequence[bool]
) -> Correlation | None:
    """Columns' daily correlation as ``covariance``, on days all have a return.

    ``korean_`` flags the columns that close in Seoul. Columns without any
    return are left out first. None with fewer than two columns or
    ``MIN_DAYS`` days.
    """
    some = frame.notna().any().to_numpy()
    x = frame.loc[:, some].dropna()
    korean_ = [k for k, keep in zip(korean_, some, strict=True) if keep]
    if x.shape[1] < 2 or len(x) < MIN_DAYS:  # noqa: PLR2004 - a pair needs two
        return None
    cov, f, weight = _shrunk(x.to_numpy(dtype="float64"), np.array(korean_, dtype=bool))
    return Correlation(correlation_frame(cov, labels(x)), f, weight, len(x))


@dataclass(frozen=True, slots=True)
class Beta:
    name: str
    welch: float  # robust, against the S&P 500
    dimson: float  # same day plus previous day, Korean series only
    kospi: float  # Korean series only
    days: int


PORTFOLIO_LABEL: Final = "포트폴리오"


def betas(j: Joint) -> list[Beta]:
    """Each holding's and the portfolio's betas, in base currency."""
    sp = cast("pd.Series", j.market[BENCHMARK]).to_numpy(dtype="float64")
    ks = cast("pd.Series", j.market[KOSPI]).to_numpy(dtype="float64")
    rows = [
        (
            name,
            cast("pd.Series", j.window[name]).to_numpy(dtype="float64"),
            korean(name),
        )
        for name in j.names
    ]
    rows.append((PORTFOLIO_LABEL, j.portfolio, bool(j.korean.any())))
    out: list[Beta] = []
    for name, r, kr in rows:
        ok = np.isfinite(r) & np.isfinite(sp)
        both = np.isfinite(r) & np.isfinite(ks)
        nan = math.nan
        out.append(
            Beta(
                name=name,
                welch=quant.welch_beta(r[ok], sp[ok]) if ok.sum() > 2 else nan,  # noqa: PLR2004
                dimson=quant.dimson_beta(r, sp) if kr else nan,
                kospi=quant.welch_beta(r[both], ks[both])
                if kr and both.sum() > 2  # noqa: PLR2004
                else nan,
                days=int(ok.sum()),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class Spread:
    risk: NDArray[np.float64]  # Euler shares of variance, sum 1
    ratio: float  # diversification ratio
    effective: float  # 1 / HHI of the weights


def diversification(j: Joint, cov: NDArray[np.float64]) -> Spread:
    return Spread(
        quant.risk_contributions(j.w, cov),
        quant.diversification_ratio(j.w, cov),
        quant.effective_holdings(j.w),
    )


# ---------------------------------------------------------------- what-if


def with_candidate(w: pd.Series, ticker: str) -> pd.Series:
    """``w`` with ``ticker`` at weight 0 when it is not held."""
    if ticker in w.index:
        return w
    return pd.concat([w, pd.Series({ticker: 0.0}, dtype="float64")])


def mixed_in(j: Joint, ticker: str, fraction: float) -> NDArray[np.float64]:
    """``j.w`` after putting ``fraction`` into ``ticker``, the rest scaled pro rata.

    A ticker already held gets ``fraction`` on top of its scaled weight.
    """
    after = (1 - fraction) * j.w
    after[j.names.index(ticker)] += fraction
    return after


@dataclass(frozen=True, slots=True)
class Added:
    """Before and after putting a fraction of the portfolio into one stock."""

    vol: tuple[float, float]  # annualised
    beta: tuple[float, float]  # S&P 500 in base; Korean holdings by Dimson
    ratio: tuple[float, float]  # diversification ratio
    correlation: float  # of the stock with the portfolio before
    threshold: float  # sigma_p / sigma_c: a lower correlation lowers volatility
    es: tuple[float, float] | None  # 1-day 97.5%; None below quant.min_samples
    es_share: float | None  # the stock's share of the ES after (Tasche)


def asset_betas(j: Joint) -> NDArray[np.float64]:
    """Each holding's beta to the S&P 500 in base: Dimson if Korean, else Welch."""
    return np.array(
        [b.dimson if korean(b.name) else b.welch for b in betas(j)[: len(j.names)]],
        dtype="float64",
    )


def market_weights(
    facts: Mapping[str, Facts], names: Sequence[str], close: pd.DataFrame
) -> NDArray[np.float64] | None:
    """Market-capitalisation weights of ``names`` in a common currency.

    Converts USD figures by the latest USD/KRW close when ``names`` mix
    currencies. None unless every holding has a positive market cap.
    """
    held: list[Facts] = []
    for name in names:
        f = facts.get(name)
        if f is None or not f.market_cap > 0 or f.currency not in {"USD", "KRW"}:
            return None
        held.append(f)
    currencies = {f.currency for f in held}
    rate = 1.0
    if len(currencies) > 1:
        fx = fx_series(close)
        if fx.empty:
            return None
        rate = float(fx.iloc[-1])
    caps = np.array(
        [f.market_cap * (rate if f.currency == "USD" else 1.0) for f in held],
        dtype="float64",
    )
    return caps / float(caps.sum())


def black_litterman(
    j: Joint,
    rf: pd.Series,
    mkt_w: NDArray[np.float64] | None,
    views: Sequence[View],
    premium: float = BL_PREMIUM,
) -> NDArray[np.float64] | None:
    """Each holding's daily combined expected return, Idzorek's formula 19.

    Black-Litterman (1992) and Idzorek (2005): reverse-optimised prior
    ``rf + lambda Sigma w_mkt`` (formula 1) with ``lambda`` the specified
    annual market ``premium``, a day at the views' rate, over the market-cap
    portfolio's variance in ``j.window``, Sigma the window's excess-return
    covariance and ``tau`` = ``BL_TAU`` (0.025). Each view on a
    holding is P = its unit row, Q its ``annual`` return a day; Omega
    diagonal, (1 - c) / c times the view's own prior variance (Idzorek's
    confidence c, in Walters' closed form), so c = 1 takes the view as is
    and tau cancels. The paper gives no standard error. None without
    ``mkt_w`` or a view on a holding.
    """
    on = [v for v in views if v.ticker in j.names]
    if not on or mkt_w is None:
        return None
    known = rf.reindex(j.window.index).fillna(0.0).to_numpy(dtype="float64")
    rf_today = float(known[-1]) if len(known) else 0.0
    excess = j.window.to_numpy(dtype="float64") - known[:, None]
    sigma = np.cov(excess, rowvar=False).reshape(len(j.names), -1)
    mkt_var = float(mkt_w @ sigma @ mkt_w)
    if not mkt_var > 0:
        return None
    year = per_year(j.full.index)
    lam = ((1 + premium) ** (1 / year) - 1) / mkt_var
    prior = rf_today + lam * (sigma @ mkt_w)
    prior_cov = BL_TAU * sigma
    p = np.zeros((len(on), len(j.names)))
    for row, v in enumerate(on):
        p[row, j.names.index(v.ticker)] = 1.0
    q = (1 + np.array([v.annual for v in on])) ** (1 / year) - 1
    c = np.array([v.confidence for v in on])
    seen = p @ prior_cov
    spread = seen @ p.T
    omega = np.diag((1 - c) / c * np.diag(spread))
    gain = np.linalg.solve(spread + omega, seen).T
    return prior + gain @ (q - p @ prior)


def trailing_means(j: Joint) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each holding's mean daily return over the window's length, and its variance.

    Row t of ``len(j.full) + 1`` averages the ``len(j.window)`` rows of
    ``j.full`` before it, so it forecasts day t from earlier days only (as
    ``quant.blocks`` needs) and the last row, the window's mean, is today's.
    The variance is the mean's sampling variance, s^2 / n. NaN where fewer
    rows precede.
    """
    x = j.full.to_numpy(dtype="float64")
    n = len(j.window)
    zero = np.zeros((1, x.shape[1]))
    s1 = np.concatenate([zero, np.cumsum(x, axis=0)])
    s2 = np.concatenate([zero, np.cumsum(x**2, axis=0)])
    mean = np.full(s1.shape, np.nan)
    var = np.full(s1.shape, np.nan)
    t = np.arange(n, len(s1))
    mean[t] = (s1[t] - s1[t - n]) / n
    var[t] = ((s2[t] - s2[t - n]) / n - mean[t] ** 2) / (n - 1)
    return mean, var


def _rolling_moments(
    j: Joint,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rolling ``len(j.window)``-day sample means and covariances (ddof=1)."""
    x = j.full.to_numpy(dtype="float64")
    n = len(j.window)
    k = x.shape[1]
    s1 = np.concatenate([np.zeros((1, k)), np.cumsum(x, axis=0)])
    mean = (s1[n:] - s1[:-n]) / n
    s2 = np.concatenate(
        [np.zeros((1, k, k)), np.cumsum(x[:, :, None] * x[:, None, :], axis=0)]
    )
    cov = (s2[n:] - s2[:-n] - n * mean[:, :, None] * mean[:, None, :]) / (n - 1)
    return mean, cov


def james_stein(j: Joint) -> NDArray[np.float64] | None:
    """James-Stein (1961) shrunk means under an unknown covariance.

    With X = sqrt(n) times each ``len(j.window)``-day sample mean and the
    Wishart A = (n - 1) S on m = n - 1 degrees of freedom, eq. 23 with the
    optimum c, ``(1 - (k - 2) / (m - k + 3) / (X' A^-1 X)) X``, shrunk toward 0 and not
    truncated at 0 (positive-part is Baranchik's). The paper gives no
    standard error. Rows match ``trailing_means``; None with fewer than three
    assets or ``n <= k``.
    """
    k = len(j.names)
    n = len(j.window)
    if k < 3 or n <= k:  # noqa: PLR2004
        return None
    mean, cov = _rolling_moments(j)
    sol = (np.linalg.pinv(cov) @ mean[..., None]).squeeze(-1)
    quad = (n / (n - 1)) * np.sum(mean * sol, axis=1)
    factor = np.where(quad > 0, 1.0 - (k - 2) / ((n - k + 2) * quad), 0.0)
    out = np.full((len(j.full) + 1, k), np.nan)
    out[n:] = factor[:, None] * mean
    return out


def bayes_stein(j: Joint) -> tuple[NDArray[np.float64], float] | None:
    """Jorion (1986) Bayes-Stein means and today's portfolio SE.

    Shrinks each ``len(j.window)``-day sample mean Y toward the minimum-
    variance portfolio mean ``Y_0 = (1' Sigma^-1 Y) / (1' Sigma^-1 1)``
    (eq. 14) with weight ``v = (k + 2) / ((k + 2) + n (Y - Y_0 1)' Sigma^-1
    (Y - Y_0 1))`` (eq. 17) and ``Sigma = (n - 1) / (n - k - 2) * S``
    (eq. 18). Today's SE is from eq. 15's predictive covariance less Sigma,
    ``Sigma / (n + lambda) + lambda / (n (n + 1 + lambda)) 1 1' / (1' Sigma^-1
    1)`` with v = lambda / (n + lambda). None with fewer than three assets or
    ``n <= k + 2``.
    """
    k = len(j.names)
    n = len(j.window)
    if k < 3 or n <= k + 2:  # noqa: PLR2004
        return None
    mean, cov = _rolling_moments(j)
    sigma = cov * ((n - 1) / (n - k - 2))
    inv = np.linalg.pinv(sigma)
    inv_ones = inv.sum(axis=2)
    denom = inv_ones.sum(axis=1)
    y0 = np.sum(inv_ones * mean, axis=1) / denom
    gap = mean - y0[:, None]
    quad = np.einsum("bi,bij,bj->b", gap, inv, gap)
    v = (k + 2) / ((k + 2) + n * np.maximum(quad, 0.0))
    out = np.full((len(j.full) + 1, k), np.nan)
    out[n:] = (1.0 - v[:, None]) * mean + v[:, None] * y0[:, None]
    v_last = float(v[-1])
    # lambda / (n (n + 1 + lambda)) with v = lambda / (n + lambda)
    var_p = (1.0 - v_last) / n * float(j.w @ sigma[-1] @ j.w) + v_last / (
        (n + 1 - v_last) * float(denom[-1])
    )
    return out, math.sqrt(max(var_p, 0.0))


def ewma_volatility(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Moskowitz-Ooi-Pedersen (2012) eq. 1 ex-ante annualised volatility.

    Row t of ``len(r) + 1`` uses days before t with exponential weights of
    center of mass ``MOP_COM`` (delta = 60/61):
    ``sigma_t^2 = 261 sum_{i=0}^{t-1} (1 - delta) delta^i (r_{t-1-i} - r_bar_t)^2``
    where ``r_bar_t = sum_{i=0}^{t-1} (1 - delta) delta^i r_{t-1-i}``.
    NaN before ``MOP_BURN`` days, until the weights sum to ``MOP_COVERAGE``
    (this app's choice; the paper's sum runs over an unbounded past).
    """
    delta = MOP_COM / (MOP_COM + 1)
    n, k = r.shape
    s1 = np.zeros((n + 1, k), dtype="float64")
    s2 = np.zeros((n + 1, k), dtype="float64")
    for t in range(n):
        s1[t + 1] = delta * s1[t] + (1.0 - delta) * r[t]
        s2[t + 1] = delta * s2[t] + (1.0 - delta) * r[t] ** 2
    power = delta ** np.arange(n + 1)[:, None]
    var = MOP_DAYS * np.maximum(s2 - (1.0 + power) * s1**2, 0.0)
    sd = np.sqrt(var)
    sd[:MOP_BURN] = np.nan
    return sd


def momentum_returns(
    j: Joint, rf: pd.Series, lag_months: int
) -> tuple[NDArray[np.float64], float, int]:
    """Moskowitz-Ooi-Pedersen (2012) eq. 1 and eq. 3 daily expected returns.

    Fits eq. 3 at lag h = ``lag_months`` (one of ``MOP_LAGS``),
    ``r_t^s / sigma_{t-1}^s = alpha + beta sign(r_{t-h}^s) + eps_t^s``,
    pooled across holdings on the calendar months completed before each day,
    where ``sigma`` is the eq. 1 EWMA ex-ante volatility at the month's start
    and ``r^s`` a month's compounded return less the compounded risk-free
    rate. The first and last months may be partial and are left out. With a
    +-1 regressor, ``alpha + beta`` is the mean scaled return after a +1 sign
    and ``alpha - beta`` after -1. The daily rate compounds over a month's
    share of the year's days (``per_year`` / 12) to the month's expected total
    return, ``(1 + rf)^(days) + sigma (alpha + beta sign(r_{t-h}))``, the sign
    from h months before the day's own; NaN until both signs have been seen.
    Second, today's portfolio daily SE from standard errors clustered by
    month (MOP 2012 Section 3.1), with the usual CR1 small-sample factor
    G / (G - 1) (N - 1) / (N - 2), carried to the daily rate by its
    derivative; 0 with fewer than two months or three observations. Third,
    the fewer of the two signs' distinct months today, for a thin-data
    warning.
    """
    x = j.full.to_numpy(dtype="float64")
    month = per_year(j.full.index) / MONTHS_PER_YEAR  # days to compound over
    known = rf.reindex(j.full.index).fillna(0.0).to_numpy(dtype="float64")
    sd = ewma_volatility(x - known[:, None])
    growth = np.concatenate([np.ones((1, x.shape[1])), np.cumprod(1 + x, axis=0)])
    cash = np.concatenate([[1.0], np.cumprod(1 + known)])
    periods = pd.DatetimeIndex(j.full.index).to_period("M")
    starts = np.flatnonzero(np.r_[True, periods[1:] != periods[:-1]])
    ends = np.append(starts[1:], len(x))
    count = len(starts)
    excess = growth[ends] / growth[starts] - (cash[ends] / cash[starts])[:, None]
    excess[[0, -1]] = np.nan  # possibly partial months
    scaled = excess / sd[starts]
    signs = np.full_like(excess, np.nan)
    signs[lag_months:] = np.sign(excess[: count - lag_months])
    rows = np.arange(len(growth))
    own = np.searchsorted(starts, rows, side="right") - 1  # today: the last month
    done = np.searchsorted(ends, rows, side="right")
    means: list[NDArray[np.float64]] = []
    sums: list[NDArray[np.float64]] = []
    counts: list[NDArray[np.int64]] = []
    for sign in (1.0, -1.0):
        hit = (signs == sign) & np.isfinite(scaled)
        s_m = np.where(hit, scaled, 0.0).sum(axis=1)
        n_m = hit.sum(axis=1)
        total = np.concatenate([[0.0], np.cumsum(s_m)])[done]
        seen = np.concatenate([[0], np.cumsum(n_m)])[done]
        mean = total / np.maximum(seen, 1)
        means.append(np.where(seen > 0, mean, np.nan))
        sums.append(s_m)
        counts.append(n_m)
    alpha, beta = (means[0] + means[1]) / 2, (means[0] - means[1]) / 2
    rates = np.append(known, known[-1] if len(known) else 0.0)
    fitted = alpha[:, None] + beta[:, None] * signs[own]
    total = (1 + rates[:, None]) ** month + sd * fitted
    daily = total ** (1 / month) - 1
    se = 0.0
    if np.isfinite(daily[-1]).all():
        sign_now = signs[-1]
        # d(daily) / d(fitted) for each holding today
        weights = j.w * sd[-1] * total[-1] ** (1 / month - 1) / month
        score = np.zeros(count)
        for idx, sign in enumerate((1.0, -1.0)):
            u_m = sums[idx] - counts[idx] * float(means[idx][-1])
            c_g = float(weights[sign_now == sign].sum())
            score += c_g * u_m / int(counts[idx].sum())
        clusters = int(((counts[0] + counts[1]) > 0).sum())
        obs = int(counts[0].sum() + counts[1].sum())
        if clusters > 1 and obs > 2:  # noqa: PLR2004 - else no clustered error
            cr1 = clusters / (clusters - 1) * (obs - 1) / (obs - 2)
            se = math.sqrt(cr1 * float(score @ score))
    return daily, se, min(int((c > 0).sum()) for c in counts)


def factor_returns(
    adjusted: pd.DataFrame,
    names: Sequence[str],
    w: NDArray[np.float64],
    factors: pd.DataFrame,
) -> tuple[NDArray[np.float64], float] | None:
    """Fama-French (1997) three-factor expected monthly returns and portfolio SE.

    Regresses each holding's monthly excess returns (month-end ``adjusted``
    closes over ``RF``) on ``Mkt-RF``, ``SMB`` and ``HML`` over every month
    both have (the full-period constant-slope regression, eq. 4). Each
    holding's monthly cost of equity is eq. 3, ``RF + b_i E[Mkt-RF] + s_i
    E[SMB] + h_i E[HML]``, with ``RF`` the latest month's one-month T-bill in
    ``factors`` and ``E[.]`` the average monthly premium since July 1963
    (``FRENCH_START``), as in Table 7. The portfolio's monthly SE is Table
    8's, ``Prem' var(u_g) Prem + g' var(u_Prem) g + 1'[var(u_Prem) o
    var(u_g)]1``, with var(u_g) the slopes' OLS covariance and var(u_Prem)
    the premiums' mean covariance. None with a Korean holding, without
    factors or under ``FACTOR_MIN_MONTHS`` months (this app's floor: the
    paper sets none, and its shortest rolling window is three years).
    """
    if (
        factors.empty
        or any(korean(n) for n in names)
        or any(n not in adjusted.columns for n in names)
    ):
        return None
    closes = adjusted[list(names)].dropna(how="all")
    if closes.empty:
        return None
    monthly = (
        closes.groupby(pd.DatetimeIndex(closes.index).to_period("M"))
        .last()
        .pct_change(fill_method=None)
    )
    fac = factors.copy()
    fac.index = pd.DatetimeIndex(fac.index).to_period("M")
    both = cast("pd.DataFrame", monthly.join(fac, how="inner")).dropna()
    if len(both) < FACTOR_MIN_MONTHS:
        return None
    x = np.column_stack([np.ones(len(both)), both[list(FACTORS)].to_numpy()])
    y = both[list(names)].to_numpy() - both[["RF"]].to_numpy()
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    betas = coef[1:].T
    hist = factors[list(FACTORS)].to_numpy(dtype="float64")
    mean_f = hist.mean(axis=0)
    cost = float(factors["RF"].iloc[-1]) + betas @ mean_f
    b_p = betas.T @ w
    resid = (y - x @ coef) @ w
    s2 = float(resid @ resid) / (len(both) - x.shape[1])
    cov_b = s2 * np.linalg.pinv(x.T @ x)[1:, 1:]
    cov_f = np.cov(hist, rowvar=False) / len(hist)
    var_m = float(b_p @ cov_f @ b_p + mean_f @ cov_b @ mean_f + np.trace(cov_b @ cov_f))
    return cost, math.sqrt(max(var_m, 0.0))


def added(j: Joint, cov: NDArray[np.float64], ticker: str, fraction: float) -> Added:
    """``quant.what_if`` and ``quant.incremental_es`` for ``ticker`` in ``j``.

    ``cov`` is ``j``'s daily covariance (``covariance``); the ES is a full
    revaluation on the window's days.
    """
    at = j.names.index(ticker)
    b = asset_betas(j)
    found = quant.what_if(j.w, cov, b, at, fraction)
    year = math.sqrt(per_year(j.window.index))
    vol = math.sqrt(float(j.w @ cov @ j.w)) * year
    beta = float(j.w @ b)
    ratio = quant.diversification_ratio(j.w, cov)
    x = j.window.to_numpy(dtype="float64")
    tail = quant.var_es(x @ j.w, quant.TAIL_LEVEL)
    more = quant.incremental_es(x, j.w, x[:, at], fraction)
    parts = quant.component_es(x, mixed_in(j, ticker, fraction))
    total = float(parts.sum()) if parts is not None else 0.0
    return Added(
        vol=(vol, vol + found.vol * year),
        beta=(beta, beta + found.beta),
        ratio=(ratio, ratio + found.ratio),
        correlation=found.correlation,
        threshold=found.threshold,
        es=(tail[1], tail[1] + more) if tail and more is not None else None,
        es_share=float(parts[at]) / total if parts is not None and total > 0 else None,
    )


# ---------------------------------------------------------------- Monte Carlo


def mean_block(j: Joint) -> float:
    """The window's mean bootstrap block, from its portfolio returns."""
    return quant.block_length(j.portfolio)


@dataclass(frozen=True, slots=True)
class Vol:
    """The volatility filter's model and inputs; by default GJR-GARCH on r^2.

    ``proxy`` (``range_proxy``) and ``x`` (``vix_regressor``) are on the last
    rows of ``Joint.full`` (``FIT_ROWS`` covers every fit), one column per
    holding.
    """

    model: quant.Filter = "gjr"
    proxy: NDArray[np.float64] | None = None
    x: NDArray[np.float64] | None = None

    def fit(
        self, r: NDArray[np.float64], rows: int | None = None
    ) -> quant.Devolatilised:
        """``quant.devolatilise`` of ``r``, the last rows of ``Joint.full``.

        The filter is estimated on its first ``rows`` (all by default).
        """
        n = len(r)
        proxy = None if self.proxy is None else self.proxy[-n:]
        x = None if self.x is None else self.x[-n:]
        return quant.devolatilise(r, rows, self.model, proxy, x)


FIT_ROWS: Final = quant.FILTER_DAYS + quant.BACKTEST_DAYS  # rows a filter fit reads
MODEL_NAMES: Final[dict[str, str]] = {"gjr": "GJR-GARCH", "har": "HAR", "pdv": "PDV"}


def no_ranges(prices: Prices, names: Iterable[str]) -> list[str]:
    """``names`` without saved opens, highs and lows."""
    have = set.intersection(*(set(map(str, f.columns)) for f in prices.ranges))
    return [n for n in names if n not in have]


def range_proxy(
    prices: Prices, names: Sequence[str], base: Currency, index: pd.Index
) -> NDArray[np.float64] | None:
    """Garman-Klass daily variances of ``names`` in ``base`` on ``index``.

    ``index`` is ``Joint.full``'s; the previous close is the adjusted close
    on the grid day before, as the joint returns take it, so the overnight
    term spans holidays and, in another base currency, the exchange rate's
    move. nan where a holding has no open, high or low that day (the
    filter then takes r^2); None when none has any.
    """
    if len(no_ranges(prices, names)) == len(names):
        return None
    grid = trading_days(prices.adjusted, names)
    native = carry_over(prices.adjusted.reindex(columns=list(names)).sort_index())
    native = native.reindex(grid)
    based = in_base(prices.adjusted, names, base, grid)
    rate = based / native  # 1 in the holding's own currency
    close = prices.close.reindex(index=grid, columns=list(names))
    o, h, lo = (
        f.reindex(index=grid, columns=list(names)).to_numpy(dtype="float64")
        for f in prices.ranges
    )
    o, h, lo, c = quant.adjusted_ohlc(
        o, h, lo, close.to_numpy(dtype="float64"), native.to_numpy(dtype="float64")
    )
    fx = rate.to_numpy(dtype="float64")
    previous = based.shift(1).to_numpy(dtype="float64")
    proxy = quant.garman_klass(o * fx, h * fx, lo * fx, c * fx, previous)
    frame = pd.DataFrame(proxy, index=grid, columns=list(names))
    return frame.reindex(index).to_numpy(dtype="float64")


def vix_regressor(
    adjusted: pd.DataFrame, names: Sequence[str], index: pd.Index
) -> NDArray[np.float64] | None:
    """VIX as daily variance (``quant.vix_variance``) for each holding's day.

    On ``index`` (``Joint.full``'s): the last VIX close on or before the
    day, for a Korean holding before it (Seoul closes before New York
    opens). None without a saved VIX that covers ``index`` to within
    ``GAP_DAYS`` of its last day.
    """
    if VIX not in adjusted.columns or not len(index):
        return None
    vix = cast("pd.Series", adjusted[VIX]).dropna().sort_index()
    days = pd.DatetimeIndex(index)
    if vix.empty or (days[-1] - vix.index[-1]).days > GAP_DAYS:
        return None
    late = pd.Series(vix.to_numpy(), index=vix.index + pd.Timedelta(days=1))
    columns = [
        (late if korean(name) else vix).reindex(days, method="ffill") for name in names
    ]
    out = quant.vix_variance(np.column_stack(columns).astype("float64"))
    return out if np.isfinite(out).all() else None


def volatility(j: Joint, v: Vol | None = None) -> quant.Devolatilised:
    """The holdings' volatility filter on their last ``quant.FILTER_DAYS``."""
    full = j.full.to_numpy(dtype="float64")[-quant.FILTER_DAYS :]
    return (v or Vol()).fit(full)


def filter_label(j: Joint, f: quant.Devolatilised) -> str:
    """Which filter each holding got: HAR or PDV, GJR-GARCH, else EWMA."""
    if f.model is not None:
        return MODEL_NAMES[f.model.kind]
    ewma = f"EWMA(λ={quant.EWMA_LAMBDA})"
    if not f.garch.any():
        return ewma
    rest = [name for name, garch in zip(j.names, f.garch, strict=True) if not garch]
    return "GJR-GARCH(1,1)" + (f", {ewma}: {', '.join(rest)}" if rest else "")


def growth(
    j: Joint,
    picks: NDArray[np.int64],
    f: quant.Devolatilised | None = None,
    w: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Growth of 1 along the drawn days, shape (paths, horizon + 1).

    ``picks`` index into the window's days; with ``f`` (``volatility``),
    into their residuals: filtered HS from today's volatility, which then
    follows the filter along each path (``quant.growth_paths``).
    ``w`` replaces ``j.w``, e.g. after a what-if, on the same draws.
    """
    mix = j.w if w is None else w
    return quant.growth_paths(j.window.to_numpy(dtype="float64"), mix, picks, f)


def fan(growth: NDArray[np.float64], value: float) -> list[dict[str, float]]:
    """Percentiles of the simulated value on each day, as chart rows."""
    bands = np.percentile(growth, PERCENTILES, axis=0) * value
    return [
        {
            "day": float(day),
            **{f"p{p}": float(bands[i, day]) for i, p in enumerate(PERCENTILES)},
        }
        for day in range(growth.shape[1])
    ]


def histogram(growth: NDArray[np.float64], value: float) -> list[dict[str, float]]:
    counts, edges = np.histogram(growth[:, -1] * value, bins=BINS)
    return [
        {"start": float(edges[i]), "end": float(edges[i + 1]), "count": float(n)}
        for i, n in enumerate(counts)
    ]


@dataclass(frozen=True, slots=True)
class Simulation:
    paths: perf.PathRisk
    fan: list[dict[str, float]]
    ends: list[dict[str, float]]
    outcomes: NDArray[np.float64]  # each path's growth of 1 at zero drift


@dataclass(frozen=True, slots=True)
class Tradeoff:
    """Simulated returns over the horizon under one expected return."""

    mean: float
    median: float
    gain: float  # share of paths ending above the start
    low: float  # 5th percentile
    es: float | None  # mean loss of the worst 5%
    high: float  # 95th percentile


def simulation(
    paths: NDArray[np.float64],
    value: float,
    current: float,
    peak: float,
    drift: float = 0.0,
) -> Simulation:
    """Summary of growth ``paths``; ``current`` and ``peak`` are past wealth.

    The paths are zero drift; the daily ``drift`` compounds onto them.
    """
    outcomes = paths[:, -1].copy()
    paths = paths * (1 + drift) ** np.arange(paths.shape[1])
    return Simulation(
        perf.path_risk(paths[:, 1:], current, peak),
        fan(paths, value),
        histogram(paths, value),
        outcomes,
    )


def tradeoff(outcomes: NDArray[np.float64], days: int, drift: float) -> Tradeoff:
    """``Simulation.outcomes`` over ``days`` with the daily ``drift`` compounded on."""
    r = outcomes * (1 + drift) ** days - 1
    low, median, high = np.percentile(r, (5, 50, 95))
    tail = quant.var_es(r, 0.95)
    return Tradeoff(
        float(r.mean()),
        float(median),
        float((r > 0).mean()),
        float(low),
        None if tail is None else tail[1],
        float(high),
    )


def generated_returns(
    j: Joint,
    native: NDArray[np.float64],
    fx: NDArray[np.float64] | None,
    base: Currency,
) -> NDArray[np.float64]:
    """A generator's paths as daily portfolio returns in ``base``, shape (n, h).

    ``native`` (n, h, len(j.names)) are simple returns in each holding's own
    currency, columns in ``j.names`` order; ``fx`` (n, h) the same paths'
    USD/KRW returns, needed when a holding is not in ``base``. Converted as
    ``in_base`` converts prices (times USD/KRW into won, divided into
    dollars) and weighted by ``j.w``, reset daily.
    """
    out = native.copy()
    for i, name in enumerate(j.names):
        if currency(name) == base:
            continue
        if fx is None:
            raise ValueError(f"{name}: 환율 경로가 없습니다.")
        grown = 1.0 + native[..., i]
        out[..., i] = (grown * (1.0 + fx) if base == "KRW" else grown / (1.0 + fx)) - 1
    return out @ j.w


def path_growth(daily: NDArray[np.float64]) -> NDArray[np.float64]:
    """Growth of 1 along daily portfolio returns (n, h), shape (n, h + 1).

    Demeaned (zero drift) and floored at -100%, as ``quant.growth_paths``.
    """
    steps = np.maximum(daily - daily.mean(), -1.0) + 1.0
    growth = np.ones((len(daily), daily.shape[1] + 1))
    _ = np.cumprod(steps, axis=1, out=growth[:, 1:])
    return growth


# ---------------------------------------------------------------- VaR and ES


@dataclass(frozen=True, slots=True)
class Tail:
    method: str
    level: float
    var: float | None  # loss fraction; None below quant.min_samples
    es: float | None
    var_ci: tuple[float, float] | None = None  # EVT: evt_intervals; else bootstrap
    es_ci: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class Tails:
    rows: tuple[Tail, ...]
    # By method, full history; None for a method short of data; {} if all are
    backtests: dict[str, quant.Backtest | None]
    lookback: int | None  # days behind each backtest forecast (``backtest_lookback``)
    component: NDArray[np.float64] | None  # 97.5% ES by holding
    days: int
    history: int


def backtest_lookback(window: int, history: int) -> int | None:
    """Days behind each backtest forecast: the page window's, else a year.

    The window's when it holds ``quant.min_samples`` at the backtest level
    and the history reaches that far before the ``quant.BACKTEST_DAYS``
    tested; else ``quant.BACKTEST_DAYS``; None when even that does not fit.
    """
    days = quant.BACKTEST_DAYS
    if window >= quant.min_samples(quant.BACKTEST_LEVEL) and history >= days + window:
        return window
    return days if history >= 2 * days else None


def tails(j: Joint, f: quant.Devolatilised, v: Vol | None = None) -> Tails:
    """Historical, filtered-HS and EVT VaR and ES at current weights.

    Historical uses the window's returns; filtered HS and EVT its residuals
    under ``f`` (``volatility(j)``). Every row carries a 95% interval: EVT's
    by profile likelihood (``quant.evt_intervals``), the others' by
    bootstrap (``quant.bootstrap_intervals``). The backtests' forecasts look back as
    far as the window (``backtest_lookback``); the filtered-HS one refits
    the filter on the ``quant.FILTER_DAYS`` before the tested days, and the
    EVT one puts a GPD on its tails (``quant.evt_windows``). ``v``
    (GJR-GARCH by default) is the filter ``f`` was fitted with.
    """
    v = v or Vol()
    x = j.window.to_numpy(dtype="float64")
    full = j.full.to_numpy(dtype="float64")
    r = x @ j.w
    held = full @ j.w
    filtered = quant.filtered_returns(f)[-len(x) :] @ j.w
    rows: list[Tail] = []
    for method, series, measure in (
        ("역사적", r, quant.var_es),
        ("FHS", filtered, quant.var_es),
        ("EVT", filtered, quant.evt_var_es),
    ):
        cis = (
            [quant.evt_intervals(series, level) for level in VAR_LEVELS]
            if method == "EVT"
            else quant.bootstrap_intervals(series, VAR_LEVELS)
        )
        for level, ci in zip(VAR_LEVELS, cis, strict=True):
            got = measure(series, level)
            rows.append(
                Tail(method, level, *(got or (None, None)), *(ci or (None, None)))
            )
    lookback = backtest_lookback(len(x), len(full))
    backtests: dict[str, quant.Backtest | None] = {}
    if lookback is not None:
        days = quant.BACKTEST_DAYS
        windows = quant.past_windows(held, days, lookback)
        fhs = refit(j, v, lookback)
        evt = None if fhs is None else quant.evt_windows(fhs)
        for method, past in (("역사적", windows), ("FHS", fhs), ("EVT", evt)):
            backtests[method] = (
                None if past is None else quant.backtest(held[-days:], past)
            )
    return Tails(
        rows=tuple(rows),
        backtests=backtests,
        lookback=lookback,
        component=quant.component_es(x, j.w),
        days=len(x),
        history=len(full),
    )


def tail_of(t: Tails, method: str, level: float) -> Tail:
    return next(row for row in t.rows if row.method == method and row.level == level)


def refit(j: Joint, v: Vol, lookback: int) -> NDArray[np.float64] | None:
    """Filtered-HS forecasts of the last ``quant.BACKTEST_DAYS``, out of sample.

    The filter ``v`` is fitted on the ``quant.FILTER_DAYS`` before them.
    """
    days = quant.BACKTEST_DAYS
    recent = j.full.to_numpy(dtype="float64")[-quant.FILTER_DAYS - days :]
    before = v.fit(recent, len(recent) - days)
    return quant.filtered_windows(before, j.w, days, lookback)


DENSITY_PATHS: Final = 2000  # draws per month: a PIT's Monte Carlo SD <= 0.011


@dataclass(frozen=True, slots=True)
class Density:
    test: quant.Berkowitz | None  # None under quant.MIN_MONTHS blocks
    blocks: int


def density(
    j: Joint,
    f: quant.Devolatilised | None = None,
    paths: int = DENSITY_PATHS,
    drift: NDArray[np.float64] | None = None,
) -> Density:
    """Berkowitz test of 1-month forecasts made as the 1M fan makes them.

    Each non-overlapping 21-day block (``quant.blocks``) of the portfolio's
    returns is set against ``paths`` compounded months. With ``f``, over its
    own days, they are simulated from the filter's state on the block's
    first day, as the FHS fan simulates from today's. Without, they are
    stationary-bootstrapped from the window's length of days before the
    block, its mean block re-estimated there (``quant.block_length``), as
    the fan resamples the window; blocks with fewer days before them are
    left out. ``drift`` (daily, on ``j.full``'s days) compounds onto each
    block's months at its first day's rate, as ``simulation`` onto the fan's;
    blocks where it is NaN are left out.
    """
    held = j.full.to_numpy(dtype="float64")
    if f is not None:
        held = held[-len(f.z) :]
    outcomes, starts = quant.blocks(held @ j.w)
    n = len(j.window)
    if f is None:
        outcomes, starts = outcomes[starts >= n], starts[starts >= n]
    rates = None if drift is None else drift[-len(held) :][starts]
    if rates is not None:  # blocks with no forecast of the drift are left out
        known = np.isfinite(rates)
        outcomes, starts, rates = outcomes[known], starts[known], rates[known]
    rng = np.random.default_rng(quant.SEED)
    samples = np.empty((len(starts), paths))
    for i, start in enumerate(starts):
        if f is None:
            before = held[start - n : start]
            block = quant.block_length(before @ j.w)
            picks = quant.stationary_indices(n, paths, quant.MONTH_DAYS, block, rng)
            samples[i] = quant.growth_paths(before, j.w, picks)[:, -1] - 1
        else:
            picks = rng.integers(0, len(f.z), size=(paths, quant.MONTH_DAYS))
            daily = (
                quant.filtered_paths(f, j.w, picks, f.sigma[start] ** 2)
                if f.model is None
                else quant.model_paths(f, j.w, picks, f.model.states[start])
            )
            samples[i] = np.prod(1 + np.maximum(daily, -1.0), axis=1) - 1
    if rates is not None:
        samples = (1 + samples) * (1 + rates[:, None]) ** quant.MONTH_DAYS - 1
    return Density(quant.berkowitz(quant.pit(outcomes, samples)), len(starts))


# ---------------------------------------------------------------- scenario replay


def long_fx(adjusted: pd.DataFrame, older: pd.Series) -> pd.Series:
    """USD/KRW: Yahoo's ``KRW=X``, with ``older`` (FRED DEXKOUS) before it starts."""
    yahoo = fx_series(adjusted)
    known = older.dropna()
    if known.empty:
        return yahoo
    if not yahoo.empty:
        known = cast("pd.Series", known[known.index < yahoo.index[0]])
    return pd.concat([known, yahoo]).sort_index()


def benchmark_of(ticker: str) -> str:
    """The index a stock's returns are compared with: KOSPI or the S&P 500 TR."""
    return KOSPI if korean(ticker) else TOTAL_RETURN


def stand_in(ticker: str) -> str:
    """The index whose move a holding without history takes: KOSPI or S&P 500."""
    return KOSPI if korean(ticker) else BENCHMARK


def proxy_beta(adjusted: pd.DataFrame, name: str, index: str) -> float:
    """``name``'s beta to ``index`` from all its saved days, shrunk toward 1.

    Every day both have counts alike: Dimson when a Korean name stands in
    for a US index (Seoul closes first), else Welch's winsorised slope
    without its age decay, in local currency; then ``quant.shrunk_beta`` by
    those days. 1 with fewer than ``PROXY_MIN_DAYS`` of them.
    """
    pair = adjusted.reindex(columns=[name, index]).dropna()
    rets = returns(pair).dropna()
    if len(rets) < PROXY_MIN_DAYS:
        return 1.0
    r = cast("pd.Series", rets[name]).to_numpy(dtype="float64")
    m = cast("pd.Series", rets[index]).to_numpy(dtype="float64")
    lagged = korean(name) and not korean(index)
    beta = quant.dimson_beta(r, m) if lagged else quant.welch_beta(r, m, decay=0.0)
    return quant.shrunk_beta(beta, len(rets)) if math.isfinite(beta) else 1.0


def _covers(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    inside = series.dropna().loc[start:end]
    if inside.empty:
        return False
    first = cast("pd.Timestamp", inside.index[0])
    last = cast("pd.Timestamp", inside.index[-1])
    return (first - start).days <= GAP_DAYS and (end - last).days <= GAP_DAYS


def resolve(
    scenario: Scenario, adjusted: pd.DataFrame
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Earliest index peak to latest index trough (``quant.union_window``).

    Each index is searched around its own stated dates; None when no index
    used by the scenario has closes there.
    """
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for ticker, stated in ((BENCHMARK, scenario.us), (KOSPI, scenario.kr)):
        if stated is None or ticker not in adjusted.columns:
            continue
        prices = cast("pd.Series", adjusted[ticker]).dropna()
        peak = cast("pd.Timestamp", pd.Timestamp(stated[0]))
        trough = cast("pd.Timestamp", pd.Timestamp(stated[1]))
        found = quant.union_window(prices, prices, peak, trough)
        if found is not None:
            spans.append(found)
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


@dataclass(frozen=True, slots=True)
class Replayed:
    scenario: Scenario
    window: tuple[date, date] | None
    change: float  # base, first to last day; nan when unknown
    worst: float  # deepest fall from a running peak inside the window
    fx: float  # the currency's factor on change: (1 + change) / (1 + local) - 1
    pnl: float  # base: today's value times ``worst``, the peak-to-trough loss
    sp500: float  # the index's own move, local currency
    kospi: float
    proxied: tuple[str, ...]  # holdings without data: their index stands in
    no_fx: bool  # a foreign leg had no USD/KRW over the window: local terms
    proxy_betas: tuple[float, ...] = ()  # each proxied holding's (``proxy_beta``)


def _replay_one(  # noqa: PLR0913 - one scenario's inputs
    scenario: Scenario,
    values: pd.Series,
    adjusted: pd.DataFrame,
    base: Currency,
    fx: pd.Series,
    span: tuple[pd.Timestamp, pd.Timestamp],
    beta_of: Callable[[str, str], float],
) -> Replayed:
    start, end = span
    window_ = (start.date(), end.date())
    moves = {
        i: float(cast("pd.Series", adjusted[i]).dropna().loc[start:end].pipe(_move))
        if i in adjusted.columns and _covers(cast("pd.Series", adjusted[i]), start, end)
        else math.nan
        for i in (BENCHMARK, KOSPI)
    }
    names = labels(values)
    present = [c for c in dict.fromkeys([*names, BENCHMARK, KOSPI]) if c in adjusted]
    local = carry_over(adjusted.reindex(columns=present)).loc[start:end]
    sources: list[str] = []
    proxied: list[str] = []
    scale: list[float] = []  # 1 for a holding's own prices
    proxy_betas: list[float] = []
    for name in names:
        # Its own closes (the last one before the window carried in), or,
        # with none on the first day, its index for the whole window.
        own = name in local.columns and not local[name].isna().any()
        if own:
            sources.append(name)
            scale.append(1.0)
            continue
        index = BENCHMARK if scenario.us_only else stand_in(name)
        proxied.append(name)
        sources.append(index)
        scale.append(beta_of(name, index))
        proxy_betas.append(scale[-1])
    unknown = _unknown(scenario, window_, moves, tuple(proxied), tuple(proxy_betas))
    if any(not math.isfinite(moves.get(s, 0.0)) for s in sources if s in moves):
        return unknown
    local = local.reindex(columns=list(dict.fromkeys(sources)))
    if local.isna().any().any():
        return unknown
    # A stand-in's beta is on the holding's own returns: its currency decides.
    foreign = [currency(name) != base for name in names]
    rate = fx.loc[:end]
    per = rate.reindex(rate.index.union(local.index)).ffill().reindex(local.index)
    no_fx = any(foreign) and (bool(per.isna().any()) or not _covers(fx, start, end))
    factor = per if base == "KRW" else 1 / per
    plain = np.column_stack(
        [
            _scaled(local[s].to_numpy(dtype="float64"), b)
            for s, b in zip(sources, scale, strict=True)
        ]
    )
    matrix = np.column_stack(
        [
            plain[:, i]
            * (factor.to_numpy(dtype="float64") if away and not no_fx else 1.0)
            for i, away in enumerate(foreign)
        ]
    )
    w = values.to_numpy(dtype="float64")
    both = quant.replay(matrix, w)
    own = quant.replay(plain, w)
    return Replayed(
        scenario=scenario,
        window=window_,
        change=both.change,
        worst=both.worst,
        fx=(1 + both.change) / (1 + own.change) - 1 if own.change > -1 else math.nan,
        pnl=float(w.sum()) * both.worst,
        sp500=moves[BENCHMARK],
        kospi=moves[KOSPI],
        proxied=tuple(proxied),
        no_fx=no_fx,
        proxy_betas=tuple(proxy_betas),
    )


def _scaled(prices: NDArray[np.float64], beta: float) -> NDArray[np.float64]:
    """A path whose daily returns are ``beta`` times those of ``prices``."""
    if beta == 1:
        return prices
    steps = np.maximum(1 + beta * (prices[1:] / prices[:-1] - 1), 0.0)
    return np.concatenate([[1.0], np.cumprod(steps)])


def _unknown(
    scenario: Scenario,
    window_: tuple[date, date] | None,
    moves: Mapping[str, float],
    proxied: tuple[str, ...],
    proxy_betas: tuple[float, ...] = (),
) -> Replayed:
    nan = math.nan
    return Replayed(
        scenario,
        window_,
        change=nan,
        worst=nan,
        fx=nan,
        pnl=nan,
        sp500=moves.get(BENCHMARK, nan),
        kospi=moves.get(KOSPI, nan),
        proxied=proxied,
        no_fx=False,
        proxy_betas=proxy_betas,
    )


def _move(prices: pd.Series) -> float:
    return float(prices.iloc[-1] / prices.iloc[0] - 1)


def scenario_replays(
    values: Mapping[str, float], adjusted: pd.DataFrame, base: Currency, fx: pd.Series
) -> list[Replayed]:
    """Today's positions (ticker -> base value) through each past window.

    Buy and hold from the window's first day (``quant.replay``), each price
    the last close on or before the day (``carry_over``). A holding without
    prices across a window (or listed inside it) takes its index (KOSPI or
    S&P 500; the S&P 500 in a US-only scenario), each day's move times its
    ``proxy_beta``, as a return in the holding's own currency. ``fx`` is
    USD/KRW, e.g. ``long_fx``.
    """
    amounts = pd.Series(dict(values), dtype="float64")
    amounts = cast("pd.Series", amounts[amounts > 0])
    found: dict[tuple[str, str], float] = {}

    def beta_of(name: str, index: str) -> float:
        if (name, index) not in found:
            found[name, index] = proxy_beta(adjusted, name, index)
        return found[name, index]

    out: list[Replayed] = []
    for scenario in SCENARIOS:
        span = resolve(scenario, adjusted)
        if span is None or amounts.empty:
            out.append(_unknown(scenario, None, {}, ()))
            continue
        out.append(_replay_one(scenario, amounts, adjusted, base, fx, span, beta_of))
    return out


def bad_print_days(
    adjusted: pd.DataFrame, names: Iterable[str]
) -> dict[str, list[date]]:
    """Closes that jump and come straight back (``quant.bad_prints``), by ticker."""
    out: dict[str, list[date]] = {}
    for name in names:
        if name not in adjusted.columns:
            continue
        prices = cast("pd.Series", adjusted[name]).dropna()
        prices = cast("pd.Series", prices[prices > 0])
        flags = quant.bad_prints(prices.to_numpy(dtype="float64"))
        if flags.any():
            days = pd.DatetimeIndex(prices.index)[flags]
            out[name] = [cast("pd.Timestamp", d).date() for d in days]
    return out


# ---------------------------------------------------------------- made-up shocks


def local_betas(
    adjusted: pd.DataFrame, names: Sequence[str], months: int
) -> NDArray[np.float64]:
    """Each holding's Welch beta to its own index in local currency.

    Over the last ``months``, pulled toward 1 by 250 / (n + 250)
    (``quant.shrunk_beta``), n the Kish effective size of Welch's age
    weights over the window's days (``quant.effective_days``: about 255
    for 3Y, not 750); 1 without data, and for the index itself.
    """
    out: list[float] = []
    for name in names:
        index = KOSPI if korean(name) else BENCHMARK
        if name == index:
            out.append(1.0)
            continue
        pair = adjusted.reindex(columns=[name, index]).dropna()
        rets = window(returns(pair), months).dropna()
        beta = math.nan
        if len(rets) > 2:  # noqa: PLR2004
            beta = quant.welch_beta(
                cast("pd.Series", rets[name]).to_numpy(dtype="float64"),
                cast("pd.Series", rets[index]).to_numpy(dtype="float64"),
            )
        if math.isfinite(beta):
            beta = quant.shrunk_beta(beta, quant.effective_days(len(rets)))
        out.append(beta if math.isfinite(beta) else 1.0)
    return np.array(out, dtype="float64")


def fx_moves(
    names: Sequence[str], base: Currency, usdkrw: float
) -> NDArray[np.float64]:
    """Each holding's currency move against ``base`` when USD/KRW moves."""
    against = usdkrw if base == "KRW" else 1 / (1 + usdkrw) - 1
    return np.array(
        [0.0 if currency(n) == base else against for n in names], dtype="float64"
    )


def shocked(
    values: pd.Series, betas_: NDArray[np.float64], base: Currency
) -> list[tuple[Shock, float]]:
    """Each of ``SHOCKS`` on positions ``values`` (``quant.hypothetical``)."""
    names = labels(values)
    w = values.to_numpy(dtype="float64")
    return [
        (
            s,
            quant.hypothetical(
                w,
                betas_,
                np.array([s.kospi if korean(n) else s.sp500 for n in names]),
                fx_moves(names, base, s.usdkrw),
            ),
        )
        for s in SHOCKS
    ]


def reverse(values: pd.Series, betas_: NDArray[np.float64]) -> float | None:
    """The same fall in every index that costs ``REVERSE_LOSS``, currencies still."""
    w = values.to_numpy(dtype="float64")
    return quant.reverse_stress(w, betas_, REVERSE_LOSS, np.zeros(len(w)))


# ---------------------------------------------------------------- turbulence


@dataclass(frozen=True, slots=True)
class Turbulence:
    week: date  # the latest week, possibly still running
    weeks: int  # weeks it is ranked against
    level: float  # about 1 in a typical week
    rank: float  # share of past weeks at or below it
    surprise: float  # level over its correlation-blind part
    surprise_rank: float


def turbulence(
    adjusted: pd.DataFrame, names: Sequence[str], base: Currency
) -> Turbulence | None:
    """This week's turbulence of the holdings' weekly returns, ranked.

    ``quant.turbulence`` over ``WEEKS`` weeks, refitted every
    ``TURBULENCE_REFIT``; None with one holding or too few weeks.
    """
    rets = weekly(asset_returns(adjusted, list(names), base)).dropna()
    if len(names) < 2 or len(rets) <= WEEKS + 1:  # noqa: PLR2004
        return None
    t = quant.turbulence(rets.to_numpy(dtype="float64"), WEEKS, TURBULENCE_REFIT)
    level = float(t.index[-1])
    surprise = float(t.surprise[-1])
    past = t.surprise[:-1]
    past = past[np.isfinite(past)]
    return Turbulence(
        week=cast("pd.Timestamp", rets.index[-1]).date(),
        weeks=len(t.index) - 1,
        level=level,
        rank=quant.percentile_rank(t.index[:-1], level),
        surprise=surprise,
        surprise_rank=quant.percentile_rank(past, surprise)
        if len(past) and math.isfinite(surprise)
        else math.nan,
    )


# ---------------------------------------------------------------- style


@dataclass(frozen=True, slots=True)
class Style:
    names: tuple[str, ...]  # proxies used, then CASH
    weights: NDArray[np.float64]
    r2: float | None
    weeks: int
    dropped: tuple[str, ...]  # proxies without prices across the window
    selection: float | None  # mean out-of-sample selection return, a year
    selection_weeks: int


def style(adjusted: pd.DataFrame, w: pd.Series, base: Currency) -> Style | None:
    """Returns-based style of the current weights' weekly returns in base.

    ``perf.style_fit`` on the last ``WEEKS`` weeks against ``STYLE_PROXIES``
    and cash; None with fewer weeks.
    """
    y_all = weekly(weighted(asset_returns(adjusted, labels(w), base), w)).dropna()
    if len(y_all) < WEEKS:
        return None
    y = y_all.iloc[-WEEKS:]
    present = [t for t in STYLE_PROXIES if t in adjusted.columns]
    proxies = weekly(asset_returns(adjusted, present, base)) if present else _empty()
    shown = proxies.reindex(index=y.index, columns=present)
    keep = [t for t in present if bool(shown[t].notna().all())]
    if not keep:
        return None
    x = shown[keep].assign(**{CASH: 0.0})
    fit = perf.style_fit(y.to_numpy(dtype="float64"), x.to_numpy(dtype="float64"))
    joined = pd.concat(
        [y_all.rename("y"), proxies[keep]], axis=1, join="inner"
    ).dropna()
    sel = perf.selection_returns(
        joined["y"].to_numpy(dtype="float64"),
        joined[keep].assign(**{CASH: 0.0}).to_numpy(dtype="float64"),
        WEEKS,
    )
    return Style(
        names=(*keep, CASH),
        weights=fit.weights,
        r2=fit.r2,
        weeks=len(y),
        dropped=tuple(t for t in STYLE_PROXIES if t not in keep),
        selection=float(sel.mean()) * 52 if len(sel) else None,
        selection_weeks=len(sel),
    )


# ---------------------------------------------------------------- text

# ``--demo``: the number every amount, return and quantity below shows instead.
demo_value: float | None = None


def shares(quantity: float) -> str:
    if demo_value is not None:
        quantity = demo_value
    return f"{quantity:,.4f}".rstrip("0").rstrip(".")


def money(value: float, unit: Currency, *, signed: bool = False) -> str:
    if demo_value is not None:
        value = demo_value
    if not math.isfinite(value):
        return UNKNOWN
    sign = "−" if value < 0 else ("+" if signed and value > 0 else "")
    return f"{sign}{SYMBOLS[unit]}{abs(value):,.{DECIMALS[unit]}f}"


def percent(value: float, *, signed: bool = False, dec: int = 1) -> str:
    if demo_value is not None:
        value = demo_value / 100
    if not math.isfinite(value):
        return UNKNOWN
    if round(value * 100, dec) == 0:  # no signed zero
        value = 0.0
    sign = "−" if value < 0 else ("+" if signed and value > 0 else "")
    return f"{sign}{abs(value):.{dec}%}"


def points(value: float) -> str:
    """A difference of two returns in percentage points."""
    if demo_value is not None:
        value = demo_value / 100
    if not math.isfinite(value):
        return UNKNOWN
    if round(value * 100, 1) == 0:  # no signed zero
        value = 0.0
    sign = "−" if value < 0 else ("+" if value > 0 else "")
    return f"{sign}{abs(value) * 100:.1f}%p"


def holdings_table(items: Sequence[Holding], base: Currency) -> dict[str, list[str]]:
    """Open positions; the currency effect shows when a holding is foreign."""
    held = [h for h in items if h.is_open]
    w = weights(held)
    table = {
        "종목": [h.ticker for h in held],
        "수량": [shares(h.position.quantity) for h in held],
        "평균단가": [money(h.position.average, h.currency) for h in held],
        "종가": [money(h.price, h.currency) for h in held],
        "종가일": [h.day.isoformat() if h.day else UNKNOWN for h in held],
        f"평가액 ({base})": [money(h.value, base) for h in held],
        "비중": [
            percent(float(cast("float", w[h.ticker])))
            if h.ticker in w.index
            else UNKNOWN
            for h in held
        ],
        f"평가손익 ({base})": [money(h.unrealized, base, signed=True) for h in held],
        "수익률 (현지)": [percent(h.native_return, signed=True) for h in held],
    }
    if any(h.currency != base for h in held):
        table[f"수익률 ({base})"] = [percent(h.base_return, signed=True) for h in held]
        table["환율 효과"] = [points(h.base_return - h.native_return) for h in held]
    return table


def realized_table(sales: Sequence[Sale], base: Currency) -> dict[str, list[str]]:
    """One row per sell, newest first."""
    rows = sorted(sales, key=lambda s: s.trade.day, reverse=True)
    table = {
        "매도일": [s.trade.day.isoformat() for s in rows],
        "종목": [s.trade.ticker for s in rows],
        "수량": [shares(s.trade.quantity) for s in rows],
        "평균단가": [money(s.average, currency(s.trade.ticker)) for s in rows],
        "매도가": [money(s.trade.price, currency(s.trade.ticker)) for s in rows],
        f"실현손익 ({base})": [money(s.pnl_base, base, signed=True) for s in rows],
        "수익률 (현지)": [percent(s.native_return, signed=True) for s in rows],
    }
    if any(currency(s.trade.ticker) != base for s in rows):
        table[f"수익률 ({base})"] = [percent(s.base_return, signed=True) for s in rows]
    table["보유일"] = [f"{s.held_days:,}" for s in rows]
    table["구분"] = [_tag(s.pnl_base) for s in rows]
    table["메모"] = [s.trade.memo for s in rows]
    return table


def _tag(pnl: float) -> str:
    if not math.isfinite(pnl):
        return UNKNOWN
    return "익절" if pnl > 0 else ("손절" if pnl < 0 else "본전")


def closed_table(items: Iterable[Holding], base: Currency) -> dict[str, list[str]]:
    """Tickers sold out, with what their sales realized."""
    done = [h.position for h in items if not h.is_open]
    return {
        "종목": [p.ticker for p in done],
        f"실현손익 ({base})": [money(p.realized_base, base, signed=True) for p in done],
        "원가 대비": [
            percent(
                p.realized_base / p.sold_base if p.sold_base else math.nan, signed=True
            )
            for p in done
        ],
        "첫 매수": [p.first.isoformat() if p.first else "—" for p in done],
        "마지막 매도": [p.last_sell.isoformat() if p.last_sell else "—" for p in done],
    }


def trades_table(portfolio: Portfolio) -> tuple[pd.DataFrame, list[str]]:
    """The log newest first, and the trade id of each row."""
    trades = list(reversed(ordered(portfolio.trades)))
    table = pd.DataFrame(
        {
            "날짜": [t.day.isoformat() for t in trades],
            "종목": [t.ticker for t in trades],
            "구분": [SIDES[t.side] for t in trades],
            "수량": [shares(t.quantity) for t in trades],
            "가격": [money(t.price, currency(t.ticker)) for t in trades],
            "수수료": [money(t.fee, currency(t.ticker)) for t in trades],
            "메모": [t.memo for t in trades],
        }
    )
    return table, [t.id for t in trades]


def thinned(series: pd.Series, most: int = CHART_POINTS) -> pd.Series:
    """At most ``most`` points of ``series`` to draw, for charts only.

    The first and last, and each stretch's low and high, so peaks and troughs
    stay on the chart.
    """
    if len(series) <= most:
        return series
    values = series.to_numpy(dtype="float64")
    keep = {0, len(values) - 1}
    for part in np.array_split(np.arange(len(values)), most // 2 - 1):
        keep.add(int(part[np.argmin(values[part])]))
        keep.add(int(part[np.argmax(values[part])]))
    return cast("pd.Series", series.iloc[sorted(keep)])


def line_points(lines: Mapping[str, pd.Series]) -> list[dict[str, str | float]]:
    """Dated series as long chart rows, ``thinned``; unknown days are left out."""
    return [
        {
            "day": cast("pd.Timestamp", day).date().isoformat(),
            "series": label,
            "value": float(value),
        }
        for label, series in lines.items()
        for day, value in thinned(series.dropna()).items()
    ]


def last(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) else math.nan


def labels(data: pd.Series | pd.DataFrame) -> list[str]:
    """A series' index or a frame's columns as text."""
    keys = data.index if isinstance(data, pd.Series) else data.columns
    return [str(key) for key in keys]


def value_lines(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Value and net invested capital from ``history``."""
    return {
        "평가액": cast("pd.Series", frame["value"]),
        "순투입": cast("pd.Series", frame["invested"]),
    }


def pnl_lines(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Cumulative realized, unrealized and total P&L from ``history``."""
    total = cast("pd.Series", frame["value"] - frame["invested"])
    realized = cast("pd.Series", frame["realized"])
    return {"실현손익": realized, "평가손익": total - realized, "총손익": total}


def contribution_points(items: Iterable[Holding]) -> list[dict[str, str | float]]:
    return [
        {"ticker": h.ticker, "pnl": h.contribution}
        for h in items
        if math.isfinite(h.contribution)
    ]


def matrix_points(matrix: pd.DataFrame) -> list[dict[str, str | float]]:
    """Correlation cells as chart rows; unknown pairs are left out."""
    return [
        {"a": str(a), "b": str(b), "value": float(value)}
        for a, row in matrix.iterrows()
        for b, value in row.items()
        if math.isfinite(float(value))
    ]


# ---------------------------------------------------------------- fetching


@dataclass(frozen=True, slots=True)
class Progress:
    running: bool = False
    message: str = ""
    failed: bool = False


class Source(Protocol):
    def history(self, tickers: Sequence[str], start: date) -> Prices: ...


class FactsSource(Protocol):
    def facts(self, tickers: Sequence[str]) -> dict[str, Facts]: ...


class FactorsSource(Protocol):
    def factors(self) -> pd.DataFrame: ...


@final
class French:
    """Ken French's Data Library: the daily US factors, on the days all have."""

    def __init__(self, root: Path) -> None:
        """Nothing is cached under ``root``, unlike ``Yahoo``."""
        del root

    def factors(self) -> pd.DataFrame:
        tables: list[pd.DataFrame] = []
        for name in FRENCH_FILES:
            response = httpx.get(
                FRENCH_URL + name, timeout=FRENCH_TIMEOUT, follow_redirects=True
            )
            _ = response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                text = archive.read(archive.namelist()[0]).decode("latin-1")
            tables.append(parse_french(text))
        return pd.concat(tables, axis=1, join="inner")


@final
class Yahoo:
    """Daily closes and adjusted closes from yfinance."""

    def __init__(self, root: Path) -> None:
        """Point yfinance's caches into ``root`` instead of the user cache."""
        import yfinance as yf  # noqa: PLC0415 - optional dependency

        root.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(root / YFINANCE_CACHE))

    def history(self, tickers: Sequence[str], start: date) -> Prices:
        import yfinance as yf  # noqa: PLC0415

        frame = yf.download(
            list(tickers),
            start=start.isoformat(),
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if frame is None or frame.empty:
            return no_prices()
        return Prices(
            *(
                _daily(cast("pd.DataFrame", frame[name]))
                for name in ("Close", "Adj Close", "Open", "High", "Low")
            )
        )

    def facts(self, tickers: Sequence[str]) -> dict[str, Facts]:
        """Each ticker's ``Facts``; one Yahoo has no market cap for is left out."""
        import yfinance as yf  # noqa: PLC0415

        today = datetime.now(UTC).date()
        out: dict[str, Facts] = {}
        for ticker in tickers:
            info = cast("Mapping[str, object]", yf.Ticker(ticker).info)
            found = facts_of(info, today)
            if found.market_cap > 0:
                out[ticker] = found
        return out


def market_series(ticker: str) -> bool:
    """A Yahoo index or rate (``^``) or exchange rate (``=X``), not a stock."""
    return ticker.startswith("^") or ticker.endswith("=X")


def saved_message(got: Sequence[str], lost: Sequence[str]) -> str:
    """What a fetch saved: stocks and the exchange rate, indices and rates apart."""
    stocks = [t for t in got if not market_series(t)]
    markets = len(got) - len(stocks)
    parts = [f"{len(stocks)}종목 가격 저장"] if stocks else []
    if markets:
        parts.append(f"환율·지수·금리 {markets}개 갱신")
    if lost:
        parts.append(f"받지 못함: {', '.join(lost)}")
    return " · ".join(parts) or "받은 가격이 없습니다"


def shallow(prices: Prices) -> bool:
    """The saved S&P 500 does not reach back before ``HISTORY_START``."""
    if BENCHMARK not in prices.adjusted.columns:
        return True
    first = cast("pd.Timestamp | None", prices.adjusted[BENCHMARK].first_valid_index())
    return first is None or first.date() >= HISTORY_START


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.dropna(axis=1, how="all").astype("float64")
    index = pd.DatetimeIndex(out.index)
    out.index = index.tz_localize(None) if index.tz is not None else index
    out.columns = [str(c) for c in out.columns]
    return out


@final
class Fetcher:
    """Fetches price history in a background thread; pages poll ``progress``.

    Tickers a fetch has tried are not fetched again on their own, so a ticker
    Yahoo does not know is not requested on every rerun; the header's 갱신
    button fetches everything again.
    """

    __slots__ = (
        "_deepened",
        "_factors",
        "_facts",
        "_lock",
        "_progress",
        "_saving",
        "_source",
        "_tried",
        "root",
    )

    def __init__(
        self,
        root: Path,
        source: Callable[[Path], Source] = Yahoo,
        facts: Callable[[Path], FactsSource] = Yahoo,
        factors: Callable[[Path], FactorsSource] = French,
    ) -> None:
        """Save under ``root``; ``source`` and ``facts`` build the fetchers per run."""
        self.root = root
        self._source = source
        self._facts = facts
        self._factors = factors
        self._lock = threading.Lock()
        self._saving = threading.Lock()  # one read-merge-write of the files at a time
        self._progress = Progress()
        self._tried: set[str] = set()
        self._deepened = False  # the one-time fetch back to LONG_START was tried

    @property
    def progress(self) -> Progress:
        with self._lock:
            return self._progress

    def untried(self, tickers: Iterable[str], cached: Collection[str]) -> list[str]:
        """``tickers`` without cached prices that no fetch has tried."""
        with self._lock:
            return [t for t in tickers if t not in cached and t not in self._tried]

    def start(
        self,
        tickers: Sequence[str],
        since: date = HISTORY_START,
        *,
        ranges: bool = False,
    ) -> bool:
        """Fetch ``tickers`` from ``since`` unless a fetch is already running.

        ``ranges``: also save their opens, highs and lows (``kept_ranges``).
        """
        with self._lock:
            if self._progress.running or not tickers:
                return False
            self._tried.update(tickers)
            self._progress = Progress(running=True, message=", ".join(tickers))
        threading.Thread(
            target=self._run,
            args=(list(tickers), since, ranges),
            name="finagent-portfolio",
            daemon=True,
        ).start()
        return True

    def start_facts(self, tickers: Sequence[str]) -> bool:
        """Fetch ``tickers``' ``Facts`` unless a fetch is already running."""
        with self._lock:
            if self._progress.running or not tickers:
                return False
            self._progress = Progress(running=True, message=", ".join(tickers))
        threading.Thread(
            target=self._run_facts,
            args=(list(tickers),),
            name="finagent-facts",
            daemon=True,
        ).start()
        return True

    def _run_facts(self, tickers: list[str]) -> None:
        try:
            got = self._facts(self.root).facts(tickers)
            with self._saving:
                save_facts(self.root, got)
        except Exception as error:  # noqa: BLE001 - shown on the page, not lost
            self._finish(
                f"기본 정보를 받지 못했습니다: {type(error).__name__}: {error}"
            )
            return
        lost = [t for t in tickers if t not in got]
        parts = [f"{len(got)}종목 기본 정보 저장"] if got else []
        if lost:
            parts.append(f"받지 못함: {', '.join(lost)}")
        self._finish(" · ".join(parts), failed=bool(lost))

    def start_factors(self) -> bool:
        """Fetch Ken French's monthly factors unless a fetch is already running."""
        with self._lock:
            if self._progress.running:
                return False
            self._progress = Progress(running=True, message="Fama–French 요인")
        threading.Thread(
            target=self._run_factors, name="finagent-factors", daemon=True
        ).start()
        return True

    def _run_factors(self) -> None:
        try:
            frame = self._factors(self.root).factors()
            with self._saving:
                self.root.mkdir(parents=True, exist_ok=True)
                _write(frame, self.root / FACTORS_FILE)
        except Exception as error:  # noqa: BLE001 - shown on the page, not lost
            self._finish(
                f"요인 자료를 받지 못했습니다: {type(error).__name__}: {error}"
            )
            return
        last = cast("pd.Timestamp", frame.index[-1]).date()
        self._finish(f"Fama–French 요인 저장({last}까지)", failed=False)

    def deepen(self, prices: Prices) -> bool:
        """Once per session, fetch the indices and USD/KRW back to ``LONG_START``.

        Only while the saved S&P 500 starts at ``HISTORY_START`` or later; the
        refreshes after it fetch from ``HISTORY_START`` and ``merge`` keeps the
        older years.
        """
        with self._lock:
            if self._deepened or self._progress.running or not shallow(prices):
                return False
            self._deepened = True
        return self.start(LONG_TICKERS, LONG_START)

    def fetch_now(self, tickers: Sequence[str]) -> Prices:
        """Fetch ``tickers`` in the calling thread, save them, return all prices.

        Raises whatever the source raises.
        """
        with self._lock:
            self._tried.update(tickers)
        return self._save(self._source(self.root).history(tickers, HISTORY_START))

    def _save(self, fresh: Prices, ranges: bool = False) -> Prices:
        with self._saving:
            prices = load_prices(self.root)
            if not fresh.tickers:
                return prices
            prices = merge(prices, kept_ranges(fresh, prices, ranges))
            save_prices(self.root, prices)
            return prices

    def _run(self, tickers: list[str], since: date, ranges: bool) -> None:
        try:
            fresh = self._source(self.root).history(tickers, since)
            _ = self._save(fresh, ranges)
        except Exception as error:  # noqa: BLE001 - shown on the page, not lost
            self._finish(f"가격을 받지 못했습니다: {type(error).__name__}: {error}")
            return
        lost = [t for t in tickers if t not in fresh.tickers]
        got = [t for t in tickers if t in fresh.tickers]
        self._finish(saved_message(got, lost), failed=bool(lost))

    def _finish(self, message: str, *, failed: bool = True) -> None:
        with self._lock:
            self._progress = Progress(message=message, failed=failed)
