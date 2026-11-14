# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportImplicitStringConcatenation=false
"""포트폴리오 거래 원장·수익률·위험 지표·시뮬레이션 및 블랙-리터만 분석 엔진."""

from __future__ import annotations

import importlib.util
import math
import re
import uuid
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import TYPE_CHECKING, ClassVar, Final, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import finagent.perf as perf  # noqa: PLR0402  # 도메인 네임스페이스 명시적 참조
import finagent.quant as quant  # noqa: PLR0402  # 도메인 네임스페이스 명시적 참조
from finagent import etf, investors, layout, memory, prices, yahoo

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

BOOK_FILE: Final = layout.USER_DIR / "portfolios.json"
PACKAGES: Final = ("yfinance",)
INSTALL_HINT: Final = "uv sync --extra yahoo"
BENCHMARK: Final = "^GSPC"
BENCHMARK_LABEL: Final = "S&P 500"
TOTAL_RETURN: Final = "^SP500TR"
HISTORY_START: Final = date(2007, 1, 1)
FILL_DAYS: Final = 31
GAP_DAYS: Final = 7
TRADING_DAYS: Final = 252
YEAR_DAYS: Final = 365
MIN_DAYS: Final = 20
WEEK: Final = "W-FRI"
PATHS: Final = 50_000
PERCENTILES: Final = (5, 25, 50, 75, 95)
BINS: Final = 40
END_LEVELS: Final = (0.95, 0.99)
REACH_STEPS: Final = 30
WINDOWS: Final = {"6M": 6, "1Y": 12, "3Y": 36, "5Y": 60}
SIM_MONTHS: Final = 120
HORIZONS: Final = {"1D": 1, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
BL_TAU: Final = 0.025
BL_PREMIUM: Final = 0.03
LOOK_DAYS: Final = 380
CHART_POINTS: Final = 1000
WEEKS: Final = 156
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
)
CASH: Final = "현금"
TICKER_PATTERN: Final = re.compile(r"[A-Z0-9^][A-Z0-9.=^-]{0,19}")
EPSILON: Final = 1e-9
MEMO_LENGTH: Final = 200
SIDES: Final = {"buy": "매수", "sell": "매도"}
DECIMALS: Final = 2
UNKNOWN: Final = "—"

Side = Literal["buy", "sell"]


class BookError(ValueError): ...


class _Frozen(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class Trade(_Frozen):
    id: str
    day: date
    ticker: str
    side: Side
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    fee: float = Field(default=0.0, ge=0)
    memo: str = Field(default="", max_length=MEMO_LENGTH)


class View(_Frozen):
    ticker: str
    annual: float
    confidence: float = Field(gt=0, le=1)


class Portfolio(_Frozen):
    id: str
    name: str
    trades: tuple[Trade, ...] = ()
    views: tuple[View, ...] = ()
    premium: float = BL_PREMIUM


class Book(_Frozen):
    portfolios: tuple[Portfolio, ...] = ()

    def get(self, pid: str) -> Portfolio | None:
        return next((p for p in self.portfolios if p.id == pid), None)


@dataclass(frozen=True, slots=True)
class Position:
    ticker: str
    quantity: float = 0.0
    cost: float = 0.0
    realized: float = 0.0
    sold: float = 0.0
    bought: float = 0.0
    first: date | None = None
    opened: date | None = None
    last_sell: date | None = None

    @property
    def average(self) -> float:
        return self.cost / self.quantity if self.quantity > EPSILON else math.nan


@dataclass(frozen=True, slots=True)
class Sale:
    trade: Trade
    opened: date
    basis: float
    pnl: float

    @property
    def average(self) -> float:
        return self.basis / self.trade.quantity

    @property
    def return_rate(self) -> float:
        return self.pnl / self.basis

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
    ticker = text.strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise BookError("티커 형식 오류")
    if "." in ticker:
        raise BookError("티커 구분자 형식 오류 (예: BRK-B)")
    return ticker


def ordered(trades: Iterable[Trade]) -> list[Trade]:
    numbered = sorted(enumerate(trades), key=lambda it: (it[1].day, it[0]))
    return [trade for _, trade in numbered]


def replay(trades: Iterable[Trade]) -> Ledger:
    held: dict[str, Position] = {}
    sales: list[Sale] = []
    for trade in ordered(trades):
        old = held.get(trade.ticker, Position(trade.ticker))
        amount = trade.quantity * trade.price
        if trade.side == "buy":
            paid = amount + trade.fee
            held[trade.ticker] = replace(
                old,
                quantity=old.quantity + trade.quantity,
                cost=old.cost + paid,
                bought=old.bought + paid,
                first=old.first or trade.day,
                opened=old.opened or trade.day,
            )
            continue
        if trade.quantity > old.quantity + EPSILON:
            raise BookError(
                f"{trade.day} {trade.ticker} 매도 수량({shares(trade.quantity)}주)이"
                f" 입력 순서상 그 시점 보유 수량({shares(old.quantity)}주) 초과"
            )
        part = min(trade.quantity / old.quantity, 1.0)
        got = amount - trade.fee
        sale = Sale(
            trade=trade,
            opened=old.opened or trade.day,
            basis=old.cost * part,
            pnl=got - old.cost * part,
        )
        sales.append(sale)
        closed = old.quantity - trade.quantity <= EPSILON
        held[trade.ticker] = replace(
            old,
            quantity=0.0 if closed else old.quantity - trade.quantity,
            cost=0.0 if closed else old.cost - sale.basis,
            realized=old.realized + sale.pnl,
            sold=old.sold + sale.basis,
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
        raise BookError("포트폴리오 이름 필수 입력")
    if any(p.name == name and p.id != skip for p in book.portfolios):
        raise BookError("동일 이름 포트폴리오 존재")
    return name


def _portfolio(book: Book, pid: str) -> Portfolio:
    found = book.get(pid)
    if found is None:
        raise BookError("대상 포트폴리오 미존재")
    return found


def add_portfolio(book: Book, name: str) -> tuple[Book, str]:
    portfolio = Portfolio(id=new_id(), name=_named(book, name))
    return book.model_copy(update={"portfolios": (*book.portfolios, portfolio)}), (
        portfolio.id
    )


def edit_portfolio(book: Book, pid: str, name: str) -> Book:
    old = _portfolio(book, pid)
    new = old.model_copy(update={"name": _named(book, name, skip=pid)})
    return _replace(book, new)


def remove_portfolio(book: Book, pid: str) -> Book:
    kept = tuple(p for p in book.portfolios if p.id != pid)
    return book.model_copy(update={"portfolios": kept})


def add_trade(
    book: Book,
    pid: str,
    trade: Trade,
    splits: Mapping[str, Sequence[Split]],
) -> Book:
    old = _portfolio(book, pid)
    new = old.model_copy(update={"trades": (*old.trades, trade)})
    _ = replay(split_adjusted(new, splits).trades)
    return _replace(book, new)


def remove_trades(
    book: Book, pid: str, ids: Collection[str], splits: Mapping[str, Sequence[Split]]
) -> Book:
    old = _portfolio(book, pid)
    new = old.model_copy(
        update={"trades": tuple(t for t in old.trades if t.id not in ids)}
    )
    try:
        _ = replay(split_adjusted(new, splits).trades)
    except BookError as error:
        raise BookError(f"거래 삭제 시 매도 수량의 보유 수량 초과: {error}") from error
    return _replace(book, new)


def set_views(
    book: Book, pid: str, views: tuple[View, ...], premium: float = BL_PREMIUM
) -> Book:
    old = _portfolio(book, pid)
    return _replace(book, old.model_copy(update={"views": views, "premium": premium}))


def book_tickers(book: Book) -> list[str]:
    return sorted({t.ticker for p in book.portfolios for t in p.trades})


def wanted(book: Book) -> list[str]:
    return [*book_tickers(book), BENCHMARK, TOTAL_RETURN]


def load_book(root: Path) -> Book:
    path = root / BOOK_FILE
    if not path.is_file():
        return Book()
    try:
        return Book.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise BookError(f"{path} 파일 읽기 실패: {error}") from error


def save_book(root: Path, book: Book) -> None:
    layout.write_text(root / BOOK_FILE, book.model_dump_json(indent=2) + "\n")


class Facts(_Frozen):
    day: date
    currency: str
    market_cap: float


def load_facts() -> dict[str, Facts]:
    return memory.current().facts.rows()


class Split(_Frozen):
    ex_date: date
    ratio: float = Field(gt=0)


type Splits = dict[str, tuple[Split, ...]]


def load_splits() -> Splits:
    return memory.current().splits.rows()


def save_splits(fresh: Mapping[str, Sequence[Split]]) -> None:
    memory.current().splits.update(
        {t: tuple(sorted(v, key=lambda s: s.ex_date)) for t, v in fresh.items()}
    )


def split_factor(splits: Sequence[Split], day: date) -> float:
    return math.prod(s.ratio for s in splits if s.ex_date > day)


def split_adjusted(p: Portfolio, splits: Mapping[str, Sequence[Split]]) -> Portfolio:
    trades: list[Trade] = []
    for t in p.trades:
        factor = split_factor(splits.get(t.ticker, ()), t.day)
        trades.append(
            t
            if factor == 1
            else t.model_copy(
                update={"quantity": t.quantity * factor, "price": t.price / factor}
            )
        )
    return p.model_copy(update={"trades": tuple(trades)})


def adjusted_book(book: Book, splits: Mapping[str, Sequence[Split]]) -> Book:
    if not splits:
        return book
    return book.model_copy(
        update={"portfolios": tuple(split_adjusted(p, splits) for p in book.portfolios)}
    )


def _empty() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([]), dtype="float64")


@dataclass(frozen=True, slots=True)
class Prices:
    close: pd.DataFrame
    adjusted: pd.DataFrame

    @property
    def tickers(self) -> frozenset[str]:
        return frozenset(str(c) for c in self.close.columns) & frozenset(
            str(c) for c in self.adjusted.columns
        )


def no_prices() -> Prices:
    return Prices(_empty(), _empty())


def load_prices(_root: Path, tickers: Collection[str]) -> Prices:
    return wide_prices(prices.current().rows(list(tickers)))


def load_funds() -> dict[str, dict[str, etf.Cell]]:
    return etf.FundStore().funds()


def compositions(names: Sequence[str]) -> dict[str, etf.Composition]:
    return {
        name: found for name in names if (found := etf.load_holdings(name)) is not None
    }


def look_returns(
    adjusted: pd.DataFrame, names: Sequence[str], days: pd.DatetimeIndex
) -> pd.DataFrame:
    known = [n for n in names if n in adjusted.columns]
    return returns(aligned(adjusted, known)).reindex(days)


def look_sectors() -> dict[str, str | None]:
    return {t: s.sector for t, s in investors.load_sectors().items()}


def prices_saved_at(tickers: Collection[str]) -> datetime | None:
    return prices.current().saved_at(list(tickers))


def last_close(close: pd.DataFrame, ticker: str) -> tuple[float, date | None]:
    if ticker not in close.columns:
        return math.nan, None
    series = cast("pd.Series", close[ticker]).dropna()
    if series.empty:
        return math.nan, None
    return float(series.iloc[-1]), cast("pd.Timestamp", series.index[-1]).date()


def close_on(close: pd.DataFrame, ticker: str, day: date) -> tuple[float, date | None]:
    if ticker not in close.columns:
        return math.nan, None
    known = cast("pd.Series", close[ticker]).dropna().loc[: pd.Timestamp(day)]
    if known.empty:
        return math.nan, None
    return float(known.iloc[-1]), cast("pd.Timestamp", known.index[-1]).date()


def trading_days(frame: pd.DataFrame, names: Sequence[str]) -> pd.DatetimeIndex:
    return cast(
        "pd.DatetimeIndex", frame.reindex(columns=list(names)).dropna(how="all").index
    )


def carry_over(frame: pd.DataFrame, limit: int = FILL_DAYS) -> pd.DataFrame:
    if frame.empty:
        return frame
    known = frame.notna().to_numpy()
    day = pd.DatetimeIndex(frame.index).to_numpy("datetime64[D]").astype(np.int64)
    seen = np.where(known, day[:, None], np.iinfo(np.int64).min // 2)
    age = day[:, None] - np.maximum.accumulate(seen, axis=0)
    last = seen.max(axis=0)
    ok = (age <= limit) & ((day[:, None] <= last) | (age <= GAP_DAYS))
    return frame.ffill().where(ok)


def aligned(
    frame: pd.DataFrame,
    names: Sequence[str],
    days: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    on = trading_days(frame, names) if days is None else days
    filled = carry_over(frame.reindex(columns=list(names)).sort_index())
    return filled.reindex(on).astype("float64")


@dataclass(frozen=True, slots=True)
class Holding:
    position: Position
    price: float
    day: date | None
    value: float

    @property
    def ticker(self) -> str:
        return self.position.ticker

    @property
    def is_open(self) -> bool:
        return self.position.quantity > EPSILON

    @property
    def unrealized(self) -> float:
        return self.value - self.position.cost if self.is_open else 0.0

    @property
    def return_rate(self) -> float:
        return self.price / self.position.average - 1 if self.is_open else math.nan

    @property
    def contribution(self) -> float:
        return self.position.realized + self.unrealized


@dataclass(frozen=True, slots=True)
class Totals:
    value: float
    cost: float
    realized: float
    sold: float
    bought: float

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
        return self.pnl / self.bought if self.bought else math.nan


def ledger(portfolio: Portfolio) -> Ledger:
    return replay(portfolio.trades)


def holdings(portfolio: Portfolio, close: pd.DataFrame) -> list[Holding]:
    out: list[Holding] = []
    for ticker, position in ledger(portfolio).positions.items():
        price, day = last_close(close, ticker)
        open_ = position.quantity > EPSILON
        worth = position.quantity * price
        out.append(Holding(position, price, day, worth if open_ else 0.0))
    return sorted(out, key=lambda h: (not h.is_open, -_finite(h.value), h.ticker))


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def totals(items: Iterable[Holding]) -> Totals:
    items = list(items)
    held = [h for h in items if h.is_open]
    return Totals(
        value=sum(h.value for h in held),
        cost=sum(h.position.cost for h in held),
        realized=sum(h.position.realized for h in items),
        sold=sum(h.position.sold for h in items),
        bought=sum(h.position.bought for h in items),
    )


def weights(items: Iterable[Holding]) -> pd.Series:
    values = {h.ticker: h.value for h in items if h.is_open and math.isfinite(h.value)}
    total = sum(values.values())
    if total <= 0:
        return pd.Series(dtype="float64")
    return pd.Series({t: v / total for t, v in values.items()}, dtype="float64")


def unpriced(items: Iterable[Holding]) -> list[str]:
    return [h.ticker for h in items if h.is_open and not math.isfinite(h.value)]


def realized_by(
    sales: Iterable[Sale], *, monthly: bool = False
) -> list[tuple[str, float]]:
    out: dict[str, float] = {}
    for sale in sorted(sales, key=lambda s: s.trade.day):
        label = f"{sale.trade.day:%Y-%m}" if monthly else f"{sale.trade.day:%Y}"
        out[label] = out.get(label, 0.0) + sale.pnl
    return list(out.items())


def flow(trade: Trade) -> float:
    amount = trade.quantity * trade.price
    return amount + trade.fee if trade.side == "buy" else trade.fee - amount


def quantities(trades: Sequence[Trade], days: pd.DatetimeIndex) -> pd.DataFrame:
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
    if amounts.empty:
        return pd.Series(0.0, index=days)
    summed = cast("pd.Series", amounts.groupby(level=0).sum(min_count=1))
    total = summed.sort_index().cumsum(skipna=False)
    out = total.reindex(days, method="ffill")
    return out.where(days >= total.index[0], 0.0)


def _names(p: Portfolio) -> list[str]:
    return sorted({t.ticker for t in p.trades})


def _held(p: Portfolio, days: pd.DatetimeIndex, names: Sequence[str]) -> pd.DataFrame:
    return quantities(p.trades, days).reindex(columns=list(names)).fillna(0.0)


def _flows(p: Portfolio, days: pd.DatetimeIndex) -> NDArray[np.float64]:
    flows = np.zeros(len(days))
    for trade in p.trades:
        at = int(np.searchsorted(days.to_numpy(), np.datetime64(trade.day, "ns")))
        if at < len(days):
            flows[at] += flow(trade)
    return flows


def history(portfolio: Portfolio, close: pd.DataFrame) -> pd.DataFrame:
    names = _names(portfolio)
    if not names:
        return pd.DataFrame(columns=["value", "invested", "realized"], dtype="float64")
    first = pd.Timestamp(min(t.day for t in portfolio.trades))
    prices = aligned(close, names)
    prices = prices[prices.index >= first]
    held = _held(portfolio, cast("pd.DatetimeIndex", prices.index), names)
    unknown = ((held > EPSILON) & prices.isna()).any(axis=1)
    value = (held * prices).sum(axis=1)[~unknown]
    flows = pd.Series(
        [flow(t) for t in portfolio.trades],
        index=pd.DatetimeIndex([pd.Timestamp(t.day) for t in portfolio.trades]),
        dtype="float64",
    )
    sales = replay(portfolio.trades).sales
    gains = pd.Series(
        [s.pnl for s in sales],
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


def earned(
    names: Sequence[str],
    close: pd.DataFrame,
    adjusted: pd.DataFrame,
    days: pd.DatetimeIndex,
) -> pd.DataFrame:
    prices = aligned(close, names, days)
    growth = aligned(adjusted, names, days)
    return prices.shift(1) * growth / growth.shift(1) - prices


def paid(
    portfolio: Portfolio,
    close: pd.DataFrame,
    adjusted: pd.DataFrame,
    days: pd.DatetimeIndex,
) -> pd.Series:
    names = _names(portfolio)
    held = _held(portfolio, days, names)
    cash = earned(names, close, adjusted, days)
    return (held.shift(1).fillna(0.0) * cash).fillna(0.0).sum(axis=1)


def twr(portfolio: Portfolio, close: pd.DataFrame, adjusted: pd.DataFrame) -> pd.Series:
    frame = history(portfolio, close)
    value = cast("pd.Series", frame["value"])
    if value.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    if value.iloc[-1] <= 0:
        last = np.datetime64(max(t.day for t in portfolio.trades), "ns")
        value = value.iloc[: int(np.searchsorted(value.index.to_numpy(), last)) + 1]
    days = cast("pd.DatetimeIndex", value.index)
    flows = _flows(portfolio, days)
    if not np.isfinite(flows).all():
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    now = value.to_numpy(dtype="float64")
    before = np.concatenate([[0.0], now[:-1]])
    start = before + np.clip(flows, 0.0, None)
    gain = now - before - flows + paid(portfolio, close, adjusted, days).to_numpy()
    daily = np.divide(gain, start, out=np.zeros(len(days)), where=start > 0)
    return pd.Series(np.cumprod(1 + daily) - 1, index=days, dtype="float64")


def money_weighted(
    p: Portfolio,
    close: pd.DataFrame,
    adjusted: pd.DataFrame,
    value: float,
    valued: pd.DatetimeIndex,
) -> perf.Annualised | None:
    items = holdings(p, close)
    last = max((h.day for h in items if h.day), default=None)
    if not p.trades or last is None or not math.isfinite(value):
        return None
    if not any(h.is_open for h in items):
        last = max(t.day for t in p.trades)
    first = min(t.day for t in p.trades)
    got = paid(p, close, adjusted, valued[valued <= pd.Timestamp(last)])
    days = [(t.day - first).days for t in p.trades]
    days += [(cast("pd.Timestamp", d).date() - first).days for d in got.index]
    days += [(last - first).days]
    cash = [-flow(t) for t in p.trades] + got.tolist() + [value]
    if not all(math.isfinite(c) for c in cash):
        return None
    return perf.mwr(np.array(days, dtype="float64"), np.array(cash, dtype="float64"))


def span_days(index: pd.Index) -> float:
    if len(index) < 2:  # noqa: PLR2004
        return 0.0
    first = cast("pd.Timestamp", index[0])
    last = cast("pd.Timestamp", index[-1])
    return float((last - first).days)


def per_year(index: pd.Index) -> float:
    days = pd.DatetimeIndex(index).to_numpy("datetime64[D]").astype(np.int64)
    steps = np.diff(days)
    steps = steps[steps <= FILL_DAYS]
    return len(steps) * YEAR_DAYS / steps.sum() if steps.sum() > 0 else TRADING_DAYS


gap = perf.gap


def time_weighted(cumulative: pd.Series) -> perf.Annualised | None:
    known = cumulative.dropna()
    if known.empty:
        return None
    return perf.annualise(float(known.iloc[-1]), span_days(known.index))


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).iloc[1:]


def window[T: (pd.Series, pd.DataFrame)](data: T, months: int) -> T:
    if data.empty:
        return data
    last = cast("pd.Timestamp", data.index[-1])
    return data.truncate(before=last - pd.DateOffset(months=months))


def held_returns(
    portfolio: Portfolio, close: pd.DataFrame, adjusted: pd.DataFrame
) -> pd.Series:
    names = _names(portfolio)
    if not names:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    prices = aligned(close, names)
    days = cast("pd.DatetimeIndex", prices.index)
    growth = aligned(adjusted, names, days)
    held = _held(portfolio, days, names).shift(1).fillna(0.0)
    on = held > EPSILON
    worth = held * prices.shift(1)
    moved = growth / growth.shift(1)
    unknown = (on & (worth.isna() | moved.isna())).any(axis=1)
    start = worth.where(on, 0.0).sum(axis=1)
    keep = ~unknown & (start > 0)
    end = (worth * moved).where(on, 0.0).sum(axis=1)
    return (end[keep] / start[keep] - 1).rename("portfolio")


def asset_returns(adjusted: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    return returns(aligned(adjusted, names))


def late_starts(rets: pd.DataFrame) -> str:
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
    joint = rets.reindex(columns=list(w.index)).dropna()
    return pd.Series(
        joint.to_numpy(dtype="float64") @ w.to_numpy(dtype="float64"),
        index=joint.index,
        dtype="float64",
    )


def covered(rets: pd.DataFrame, w: pd.Series) -> tuple[pd.Series, list[str]]:
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
    day: date
    names: int
    turnover: float | None


@dataclass(frozen=True, slots=True)
class Clone:
    cumulative: pd.Series
    steps: tuple[Step, ...]
    weights: pd.Series


def clone(prices: pd.DataFrame, schedule: Sequence[tuple[date, pd.Series]]) -> Clone:
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
            starts[at] = w
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
            if pieces:
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
        moved = None if drifted is None else perf.active_share(mix, drifted)
        steps.append(Step(day, len(held), moved))
        drifted = perf.drift(mix, cast("pd.Series", relative.iloc[-1]))
    if not pieces:
        empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        return Clone(empty, (), pd.Series(dtype="float64"))
    now = _priced_now(drifted, near)
    return Clone(pd.concat(pieces) - 1, tuple(steps), now)


def _priced_now(w: pd.Series | None, near: pd.DataFrame) -> pd.Series:
    if w is None or near.empty:
        return pd.Series(dtype="float64")
    now = cast("pd.Series", near.iloc[-1]).reindex(w.index)
    part = cast("pd.Series", w[now.notna() & (w > 0)])
    total = float(part.sum())
    return part / total if total > 0 else part.iloc[:0]


def rebalanced(
    prices: pd.DataFrame, schedule: Sequence[tuple[date, pd.Series]]
) -> pd.Series:
    return clone(prices, schedule).cumulative


def since(cumulative: pd.Series, start: pd.Timestamp) -> pd.Series:
    later = cumulative.dropna().loc[start:]
    if later.empty:
        return later
    return (1 + later) / (1 + float(later.iloc[0])) - 1


def compounded(rets: pd.Series) -> float:
    return float(np.prod(1 + rets.to_numpy(dtype="float64")) - 1)


def daily(cumulative: pd.Series) -> pd.Series:
    return (1 + cumulative).pct_change(fill_method=None).iloc[1:]


def weekly[T: (pd.Series, pd.DataFrame)](rets: T) -> T:
    return (1 + rets).resample(WEEK).prod(min_count=1) - 1


def lineup(book: Book, prices: Prices, months: int) -> pd.DataFrame:
    names = [*book_tickers(book), TOTAL_RETURN]
    rets = window(asset_returns(prices.adjusted, names), months)
    columns: dict[str, pd.Series] = {}
    for p in book.portfolios:
        w, _ = covered(rets, weights(holdings(p, prices.close)))
        if not w.empty:
            columns[p.name] = weighted(rets, w)
    index = cast("pd.Series", rets[TOTAL_RETURN])
    if index.notna().any():
        columns[BENCHMARK_LABEL] = index
    return pd.DataFrame(columns)


def cumulative(daily_rets: pd.Series) -> pd.Series:
    if daily_rets.empty:
        return daily_rets
    steps = daily_rets.copy()
    steps.iloc[0] = 0.0
    out = (1 + steps).cumprod() - 1
    return out.where(steps.notna())


def benchmark_lines(
    close: pd.DataFrame, days: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    rets = aligned(close, [TOTAL_RETURN], days).pct_change(fill_method=None)
    return {BENCHMARK_LABEL: cumulative(cast("pd.Series", rets[TOTAL_RETURN]))}


@dataclass(frozen=True, slots=True)
class Trading:
    actual: float
    held: float
    effect: perf.Gap | None


def trading(
    p: Portfolio, close: pd.DataFrame, adjusted: pd.DataFrame
) -> Trading | None:
    frame = history(p, close)
    value = cast("pd.Series", frame["value"])
    if len(value) < 2:  # noqa: PLR2004
        return None
    days = cast("pd.DatetimeIndex", value.index)
    flows = _flows(p, days)
    flows[0] = 0.0
    shares = quantities(p.trades, days).iloc[0]
    mix = [str(t) for t, q in shares.items() if float(q) > EPSILON]
    prices = aligned(close, mix, days).ffill()
    if not mix or not np.isfinite(flows).all() or prices.iloc[0].isna().any():
        return None
    first = shares[mix].to_numpy(dtype="float64")
    held = perf.counterfactual(prices.to_numpy(dtype="float64"), first, flows)
    units = held / (prices.to_numpy(dtype="float64") @ first)
    cash = earned(mix, close, adjusted, days).fillna(0.0)
    still_paid = np.concatenate([[0.0], units[:-1]]) * (
        cash.to_numpy(dtype="float64") @ first
    )
    real_paid = paid(p, close, adjusted, days).to_numpy(dtype="float64")
    offsets = (days - days[0]).days.to_numpy(dtype="float64")
    real = -flows + real_paid
    still = -flows + still_paid
    real[0] = still[0] = -float(value.iloc[0])
    actual = float(value.iloc[-1])
    real[-1] += actual
    still[-1] += float(held[-1])
    effect = perf.trading_effect(offsets, real, still)
    return Trading(actual, float(held[-1]), effect)


def risk_free(rates: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    known = rates.dropna()
    if known.empty:
        return pd.Series(0.0, index=index, dtype="float64")
    return perf.daily_rf(known, index, per_year(index))


def wealth(rets: pd.Series, start: pd.Timestamp | None = None) -> pd.Series:
    if rets.empty:
        return rets
    first = cast("pd.Timestamp", rets.index[0])
    before = first - pd.offsets.BDay(1) if start is None else start
    growth = (1 + rets).cumprod()
    return pd.concat([pd.Series([1.0], index=pd.DatetimeIndex([before])), growth])


def _annual(value: float | None, periods: float) -> float | None:
    return None if value is None else value * math.sqrt(periods)


@dataclass(frozen=True, slots=True)
class Ratios:
    days: int
    span: float
    volatility: float | None
    under: perf.Underwater | None
    sharpe: float | None
    ci: perf.SharpeCI | None
    sortino: float | None
    calmar: float | None
    cdar: float
    te: float | None
    ir: float | None


def _excess(
    rets: pd.Series, rf: pd.Series
) -> tuple[pd.Series, pd.Series, NDArray[np.float64]]:
    rate = rf.reindex(rets.index)
    known = rate.notna() & rets.notna()
    r = cast("pd.Series", rets[known])
    f = cast("pd.Series", rate[known])
    return r, f, (r - f).to_numpy(dtype="float64")


def ratios(
    rets: pd.Series,
    rf: pd.Series,
    bench: pd.Series,
    start: pd.Timestamp | None = None,
) -> Ratios:
    r, f, x = _excess(rets, rf)
    n = len(r)
    periods = per_year(r.index)
    same = n > 0 and r.index[0] == rets.index[0]
    level = wealth(r, start if same else None)
    span = span_days(level.index)
    sr = perf.sharpe(x)
    b = bench.reindex(r.index)
    paired = b.notna()
    return Ratios(
        days=n,
        span=span,
        volatility=float(x.std(ddof=1)) * math.sqrt(periods) if n > 1 else None,
        under=perf.underwater(level) if n else None,
        sharpe=_annual(sr, periods),
        ci=perf.sharpe_ci(x, periods=periods),
        sortino=_annual(perf.sortino(r, f), periods),
        calmar=perf.calmar(r, span),
        cdar=perf.cdar(r),
        te=perf.tracking_error(r[paired], b[paired], periods),
        ir=perf.information_ratio(r[paired], b[paired], periods),
    )


def market_returns(
    adjusted: pd.DataFrame,
    days: pd.DatetimeIndex,
    index: str = TOTAL_RETURN,
) -> pd.Series:
    grid = trading_days(adjusted, [index])
    rets = returns(aligned(adjusted, [index], grid))
    return cast("pd.Series", rets[index]).reindex(days)


def market_beta(rets: pd.Series, adjusted: pd.DataFrame) -> float:
    sp = market_returns(adjusted, pd.DatetimeIndex(rets.index), BENCHMARK)
    ok = rets.notna() & sp.notna()
    if int(ok.sum()) <= 2:  # noqa: PLR2004
        return math.nan
    return quant.welch_beta(
        cast("pd.Series", rets[ok]).to_numpy(dtype="float64"),
        cast("pd.Series", sp[ok]).to_numpy(dtype="float64"),
    )


@dataclass(frozen=True, slots=True)
class Peer:
    name: str
    days: int
    total: float
    sharpe: float | None
    shrunk: float | None


@dataclass(frozen=True, slots=True)
class Peers:
    rows: tuple[Peer, ...]
    count: int
    deflated: float | None


def peers(series: Mapping[str, pd.Series], rf: pd.Series) -> Peers:
    found: list[_Rated] = []
    for name, rets in series.items():
        r, _, x = _excess(rets, rf)
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
    sr: float | None
    skew: float
    kurt: float
    periods: float

    @property
    def per(self) -> float:
        return self.sr if self.sr is not None else math.nan


@dataclass(frozen=True, slots=True)
class Joint:
    names: tuple[str, ...]
    w: NDArray[np.float64]
    full: pd.DataFrame
    window: pd.DataFrame
    market: pd.DataFrame
    late: str

    @property
    def portfolio(self) -> NDArray[np.float64]:
        return self.window.to_numpy(dtype="float64") @ self.w


def joint(adjusted: pd.DataFrame, w: pd.Series, months: int) -> Joint:
    names = labels(w)
    grid = trading_days(adjusted, names)
    extra = [BENCHMARK] if BENCHMARK not in names else []
    rets = returns(aligned(adjusted, [*names, *extra], grid))
    own = cast("pd.DataFrame", rets[names])
    full = own.dropna()
    part = window(full, months)
    return Joint(
        names=tuple(names),
        w=w.to_numpy(dtype="float64") / float(w.sum()),
        full=full,
        window=part,
        market=cast("pd.DataFrame", rets[[BENCHMARK]]).reindex(part.index),
        late=late_starts(window(own, months)),
    )


def covariance(j: Joint) -> NDArray[np.float64]:
    return quant.ledoit_wolf(j.window.to_numpy(dtype="float64"))[0]


def correlation_frame(cov: NDArray[np.float64], names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(quant.correlation(cov), index=list(names), columns=list(names))


def shrunk_correlation(frame: pd.DataFrame) -> pd.DataFrame | None:
    some = frame.notna().any().to_numpy()
    x = frame.loc[:, some].dropna()
    if x.shape[1] < 2 or len(x) < MIN_DAYS:  # noqa: PLR2004
        return None
    cov, _ = quant.ledoit_wolf(x.to_numpy(dtype="float64"))
    return correlation_frame(cov, labels(x))


@dataclass(frozen=True, slots=True)
class Beta:
    name: str
    welch: float


PORTFOLIO_LABEL: Final = "포트폴리오"


def betas(j: Joint) -> list[Beta]:
    sp = cast("pd.Series", j.market[BENCHMARK]).to_numpy(dtype="float64")
    rows = [
        (name, cast("pd.Series", j.window[name]).to_numpy(dtype="float64"))
        for name in j.names
    ]
    rows.append((PORTFOLIO_LABEL, j.portfolio))
    out: list[Beta] = []
    for name, r in rows:
        ok = np.isfinite(r) & np.isfinite(sp)
        out.append(
            Beta(
                name=name,
                welch=quant.welch_beta(r[ok], sp[ok]) if ok.sum() > 2 else math.nan,  # noqa: PLR2004
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class Spread:
    risk: NDArray[np.float64]
    ratio: float


def diversification(j: Joint, cov: NDArray[np.float64]) -> Spread:
    return Spread(
        quant.risk_contributions(j.w, cov),
        quant.diversification_ratio(j.w, cov),
    )


def with_candidate(w: pd.Series, ticker: str) -> pd.Series:
    if ticker in w.index:
        return w
    return pd.concat([w, pd.Series({ticker: 0.0}, dtype="float64")])


def mixed_in(j: Joint, ticker: str, fraction: float) -> NDArray[np.float64]:
    after = (1 - fraction) * j.w
    after[j.names.index(ticker)] += fraction
    return after


@dataclass(frozen=True, slots=True)
class Mixed:
    vol: float
    beta: float
    ratio: float
    es: float | None


@dataclass(frozen=True, slots=True)
class Added:
    vol: tuple[float, float]
    beta: tuple[float, float]
    ratio: tuple[float, float]
    es: tuple[float, float] | None
    es_share: float | None


def asset_betas(j: Joint) -> NDArray[np.float64]:
    return np.array(
        [b.welch for b in betas(j)[: len(j.names)]],
        dtype="float64",
    )


def is_fund(cells: Mapping[str, etf.Cell] | None) -> bool:
    kind = None if cells is None else cells.get("kind")
    return kind is not None and str(kind.value).upper() in yahoo.FUND_KINDS


def sizes(
    facts: Mapping[str, Facts], funds: Mapping[str, Mapping[str, etf.Cell]]
) -> dict[str, Facts]:
    out = dict(facts)
    for ticker, cells in funds.items():
        if not is_fund(cells):
            continue
        _ = out.pop(ticker, None)
        aum = cells.get("aum")
        value = None if aum is None else aum.value
        if aum is not None and isinstance(value, int | float) and value > 0:
            out[ticker] = Facts(
                day=date.fromisoformat(aum.as_of[:10]),
                currency="USD",
                market_cap=float(value),
            )
    return out


def market_weights(
    sized: Mapping[str, Facts], names: Sequence[str]
) -> tuple[NDArray[np.float64] | None, list[str]]:
    missing = [
        name
        for name in names
        if name not in sized
        or not sized[name].market_cap > 0
        or sized[name].currency != "USD"
    ]
    if missing:
        return None, missing
    caps = np.array([sized[name].market_cap for name in names], dtype="float64")
    return caps / float(caps.sum()), []


def black_litterman(
    j: Joint,
    rf: pd.Series,
    mkt_w: NDArray[np.float64] | None,
    views: Sequence[View],
    premium: float = BL_PREMIUM,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    on = [v for v in views if v.ticker in j.names]
    if mkt_w is None:
        return None
    known = rf.reindex(j.window.index).fillna(0.0).to_numpy(dtype="float64")
    rf_today = float(known[-1]) if len(known) else 0.0
    excess = j.window.to_numpy(dtype="float64") - known[:, None]
    sigma = quant.ledoit_wolf(excess)[0]
    mkt_var = float(mkt_w @ sigma @ mkt_w)
    if not mkt_var > 0:
        return None
    year = per_year(j.full.index)
    lam = ((1 + premium) ** (1 / year) - 1) / mkt_var
    prior = rf_today + lam * (sigma @ mkt_w)
    prior_cov = BL_TAU * sigma
    if not on:
        return prior, prior
    p = np.zeros((len(on), len(j.names)))
    for row, v in enumerate(on):
        p[row, j.names.index(v.ticker)] = 1.0
    q = (1 + np.array([v.annual for v in on])) ** (1 / year) - 1
    c = np.array([v.confidence for v in on])
    seen = p @ prior_cov
    spread = seen @ p.T
    omega = np.diag((1 - c) / c * np.diag(spread))
    gain = np.linalg.solve(spread + omega, seen).T
    return prior, prior + gain @ (q - p @ prior)


def added(j: Joint, cov: NDArray[np.float64], ticker: str, fraction: float) -> Added:
    old, new = added_curve(j, cov, ticker, (0.0, fraction))
    x = j.window.to_numpy(dtype="float64")
    parts = quant.component_es(x, mixed_in(j, ticker, fraction))
    total = float(parts.sum()) if parts is not None else 0.0
    at = j.names.index(ticker)
    return Added(
        vol=(old.vol, new.vol),
        beta=(old.beta, new.beta),
        ratio=(old.ratio, new.ratio),
        es=None if old.es is None or new.es is None else (old.es, new.es),
        es_share=float(parts[at]) / total if parts is not None and total > 0 else None,
    )


def added_curve(
    j: Joint, cov: NDArray[np.float64], ticker: str, fractions: Sequence[float]
) -> list[Mixed]:
    x = j.window.to_numpy(dtype="float64")
    year = math.sqrt(per_year(j.window.index))
    b = asset_betas(j)
    rows: list[Mixed] = []
    for fraction in fractions:
        after = mixed_in(j, ticker, fraction)
        tail = quant.var_es(x @ after, quant.TAIL_LEVEL)
        rows.append(
            Mixed(
                vol=math.sqrt(float(after @ cov @ after)) * year,
                beta=float(after @ b),
                ratio=quant.diversification_ratio(after, cov),
                es=tail[1] if tail else None,
            )
        )
    return rows


def mean_block(j: Joint) -> float:
    return quant.block_length(j.portfolio)


def growth(
    j: Joint,
    picks: NDArray[np.int64],
    w: NDArray[np.float64] | None = None,
    *,
    demean: bool = True,
) -> NDArray[np.float64]:
    mix = j.w if w is None else w
    return quant.growth_paths(
        j.window.to_numpy(dtype="float64"), mix, picks, demean=demean
    )


def fan(growth: NDArray[np.float64], value: float) -> list[dict[str, float]]:
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


def reach_curve(risk: perf.PathRisk) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for side, reach in ((-1.0, risk.falls), (1.0, risk.rises)):
        steps = range(REACH_STEPS, -1, -1) if side < 0 else range(REACH_STEPS + 1)
        for k in steps:
            r = reach[k / 100]
            rows.append(
                {
                    "x": side * k / 100,
                    "side": side,
                    "end": r.end,
                    "touched": r.touched,
                    "swing": r.swing,
                }
            )
    return rows


@dataclass(frozen=True, slots=True)
class Simulation:
    paths: perf.PathRisk
    fan: list[dict[str, float]]
    ends: list[dict[str, float]]
    tails: dict[float, tuple[float, float]]
    reach: list[dict[str, float]]


def simulation(
    paths: NDArray[np.float64], value: float, drift: float = 0.0
) -> Simulation:
    paths = paths * drift_factor(paths.shape[1] - 1, drift)
    grid = (*perf.PATH_LEVELS, *(k / 100 for k in range(REACH_STEPS + 1)))
    risk = perf.path_risk(paths[:, 1:], tuple(dict.fromkeys(grid)))
    shown = replace(
        risk,
        falls={level: risk.falls[level] for level in perf.PATH_LEVELS},
        rises={level: risk.rises[level] for level in perf.PATH_LEVELS},
    )
    tails = {level: quant.var_es(paths[:, -1] - 1, level) for level in END_LEVELS}
    return Simulation(
        shown,
        fan(paths, value),
        histogram(paths, value),
        {level: tail for level, tail in tails.items() if tail is not None},
        reach_curve(risk),
    )


def drift_factor(days: int, drift: float) -> NDArray[np.float64]:
    return np.concatenate([[1.0], np.cumprod(np.full(days, 1 + drift))])


@dataclass(frozen=True, slots=True)
class Tails:
    backtest: quant.Backtest | None
    history: int


def backtest_lookback(window: int, history: int) -> int | None:
    days = quant.BACKTEST_DAYS
    if window >= quant.min_samples(quant.BACKTEST_LEVEL) and history >= days + window:
        return window
    return days if history >= 2 * days else None


def tails(j: Joint) -> Tails:
    full = j.full.to_numpy(dtype="float64")
    held = full @ j.w
    lookback = backtest_lookback(len(j.window), len(full))
    days = quant.BACKTEST_DAYS
    past = None if lookback is None else quant.past_windows(held, days, lookback)
    return Tails(
        backtest=None if past is None else quant.backtest(held[-days:], past),
        history=len(full),
    )


DENSITY_PATHS: Final = 5000


@dataclass(frozen=True, slots=True)
class Density:
    test: quant.Berkowitz | None
    blocks: int


def density(
    j: Joint, paths: int = DENSITY_PATHS, days: int = quant.MONTH_DAYS
) -> Density:
    held = j.full.to_numpy(dtype="float64")
    n = len(j.window)
    outcomes, starts = quant.blocks(held @ j.w, days)
    outcomes, starts = outcomes[starts >= n], starts[starts >= n]
    last = slice(-quant.BACKTEST_DAYS, None)
    outcomes, starts = outcomes[last], starts[last]
    rng = np.random.default_rng(quant.SEED)
    samples = np.empty((len(starts), paths))
    for i, start in enumerate(starts):
        before = held[start - n : start]
        block = quant.block_length(before @ j.w)
        picks = quant.stationary_indices(n, paths, days, block, rng)
        samples[i] = quant.growth_paths(before, j.w, picks)[:, -1] - 1
    return Density(quant.berkowitz(quant.pit(outcomes, samples)), len(starts))


@dataclass(frozen=True, slots=True)
class Style:
    names: tuple[str, ...]
    weights: NDArray[np.float64]
    r2: float | None
    dropped: tuple[str, ...]
    selection: float | None


def style(adjusted: pd.DataFrame, w: pd.Series) -> Style | None:
    y_all = weekly(weighted(asset_returns(adjusted, labels(w)), w)).dropna()
    if len(y_all) < WEEKS:
        return None
    y = y_all.iloc[-WEEKS:]
    present = [t for t in STYLE_PROXIES if t in adjusted.columns]
    proxies = weekly(asset_returns(adjusted, present)) if present else _empty()
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
        dropped=tuple(t for t in STYLE_PROXIES if t not in keep),
        selection=float(sel.mean()) * 52 if len(sel) else None,
    )


def shares(quantity: float) -> str:
    return f"{quantity:,.4f}".rstrip("0").rstrip(".")


def money(value: float, *, signed: bool = False) -> str:
    if not math.isfinite(value):
        return UNKNOWN
    if round(value, DECIMALS) == 0:
        value = 0.0
    sign = "-" if value < 0 else ("+" if signed and value > 0 else "")
    return f"{sign}${abs(value):,.{DECIMALS}f}"


def percent(value: float, *, signed: bool = False, dec: int = 1) -> str:
    if not math.isfinite(value):
        return UNKNOWN
    if round(value * 100, dec) == 0:
        value = 0.0
    sign = "-" if value < 0 else ("+" if signed and value > 0 else "")
    return f"{sign}{abs(value):.{dec}%}"


def thinned(series: pd.Series, most: int = CHART_POINTS) -> pd.Series:
    if len(series) <= most:
        return series
    values = series.to_numpy(dtype="float64")
    keep = {0, len(values) - 1}
    for part in np.array_split(np.arange(len(values)), most // 2 - 1):
        keep.add(int(part[np.argmin(values[part])]))
        keep.add(int(part[np.argmax(values[part])]))
    return cast("pd.Series", series.iloc[sorted(keep)])


def line_points(lines: Mapping[str, pd.Series]) -> list[dict[str, str | float]]:
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
    keys = data.index if isinstance(data, pd.Series) else data.columns
    return [str(key) for key in keys]


def value_lines(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "평가액": cast("pd.Series", frame["value"]),
        "순투입": cast("pd.Series", frame["invested"]),
    }


def pnl_lines(frame: pd.DataFrame) -> dict[str, pd.Series]:
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
    return [
        {"a": str(a), "b": str(b), "value": float(value)}
        for a, row in matrix.iterrows()
        for b, value in row.items()
        if math.isfinite(float(value))
    ]


def wide_prices(rows: pd.DataFrame) -> Prices:
    if rows.empty:
        return no_prices()
    unique = rows.drop_duplicates(["ticker", "date"], keep="last")

    def wide(field_: str) -> pd.DataFrame:
        table = unique.pivot(index="date", columns="ticker", values=field_)  # noqa: PD010
        return _daily(table).dropna(how="all")

    return Prices(wide("close"), wide("adj_close"))


def market_series(ticker: str) -> bool:
    return ticker.startswith("^") or ticker.endswith(("=X", "=F"))


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.dropna(axis=1, how="all").astype("float64")
    index = pd.DatetimeIndex(out.index)
    out.index = index.tz_localize(None) if index.tz is not None else index
    out.columns = [str(c) for c in out.columns]
    return out
