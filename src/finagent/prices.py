# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportPrivateUsage=false
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Alpaca·Yahoo 일봉 시세 병합·액면분할·시가총액 조회 및 인메모리 시세 저장소."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, cast, final
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from finagent import limits, markets, memory, yahoo

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

Kind = Literal["daily", "splits"]
ProviderName = Literal["Alpaca", "Yahoo"]
Basis = Literal["split", "total"]
type RowFetch = Callable[[Sequence[str], date, date], pd.DataFrame]

OVERLAP_DAYS: Final = 7
ADJUST_TOLERANCE: Final = 0.001
GAP_DAYS: Final = 7
ALPACA_SEAM: Final = markets.ALPACA_START + timedelta(days=GAP_DAYS)
ROW_COLUMNS: Final = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
)
STORED_COLUMNS: Final = (*ROW_COLUMNS, "fetched_at")
PRICE_FIELDS: Final = ("open", "high", "low", "close", "adj_close", "volume")
KEY: Final = ["ticker", "date"]
CHUNK: Final = 100
NO_PRICE: Final = "시세 없음"
SESSION_SETTLED: Final = clock_time(17, 0)
SEOUL: Final = ZoneInfo("Asia/Seoul")
TOKYO: Final = ZoneInfo("Asia/Tokyo")
SHANGHAI: Final = ZoneInfo("Asia/Shanghai")
LONDON: Final = ZoneInfo("Europe/London")
MARKET_CLOSES: Final[tuple[tuple[tuple[str, ...], ZoneInfo, clock_time], ...]] = (
    ((".KS", ".KQ"), SEOUL, clock_time(16, 0)),
    ((".T",), TOKYO, clock_time(16, 0)),
    ((".HK", ".SS", ".SZ"), SHANGHAI, clock_time(16, 30)),
    ((".L", ".DE", ".PA", ".AS", ".SW", ".MI"), LONDON, clock_time(17, 30)),
)
INDEX_CLOSES: Final[Mapping[str, tuple[ZoneInfo, clock_time]]] = {
    "^KS11": (SEOUL, clock_time(16, 0)),
    "^KQ11": (SEOUL, clock_time(16, 0)),
    "^N225": (TOKYO, clock_time(16, 0)),
    "^HSI": (SHANGHAI, clock_time(16, 30)),
    "^FTSE": (LONDON, clock_time(17, 30)),
    "^GDAXI": (LONDON, clock_time(17, 30)),
    "^FCHI": (LONDON, clock_time(17, 30)),
    "^STOXX50E": (LONDON, clock_time(17, 30)),
}
YAHOO_FIELDS: Final = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
PROVIDER_ERRORS: Final = (
    httpx.HTTPStatusError,
    httpx.TransportError,
    limits.LimitReached,
)


@dataclass(frozen=True, slots=True)
class Route:
    kind: Kind
    target: str
    order: tuple[ProviderName, ...]


US: Final = "미국 상장"
OTHER: Final = "그 밖"
ROUTES: Final[tuple[Route, ...]] = (
    Route("daily", US, ("Alpaca", "Yahoo")),
    Route("daily", OTHER, ("Yahoo",)),
    Route("splits", US, ("Alpaca", "Yahoo")),
    Route("splits", OTHER, ("Yahoo",)),
)


def route(kind: Kind, ticker: str, keys: markets.Keys) -> tuple[ProviderName, ...]:
    target = US if markets.us_listed(ticker) else OTHER
    found = next(row for row in ROUTES if row.kind == kind and row.target == target)
    return tuple(p for p in found.order if p != "Alpaca" or keys.alpaca)


def empty_rows() -> pd.DataFrame:
    frame = pd.DataFrame({column: [] for column in ROW_COLUMNS})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].astype("str")
    frame["source"] = frame["source"].astype("str")
    return frame


def alpaca_rows(
    alpaca: markets.Alpaca, tickers: Sequence[str], start: date, last: date
) -> pd.DataFrame:
    with ThreadPoolExecutor(2) as pool:
        split, whole = pool.map(
            partial(alpaca.bars, tickers, start, last), ("split", "all")
        )
    if split.empty:
        return empty_rows()
    adjusted = cast("pd.DataFrame", whole[["ticker", "date", "close"]]).rename(
        columns={"close": "adj_close"}
    )
    frame = split.merge(adjusted, on=["ticker", "date"], how="inner")
    frame = frame.dropna(subset=["close", "adj_close"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["source"] = "Alpaca"
    return cast("pd.DataFrame", frame[list(ROW_COLUMNS)]).reset_index(drop=True)


def yahoo_rows(
    tickers: Sequence[str],
    start: date,
    end: date | None = None,
    max_wait: float = limits.BACKGROUND_WAIT,
) -> pd.DataFrame:
    frame = yahoo.download(tickers, start, end, max_wait)
    if frame.empty:
        return empty_rows()
    stacked = cast(
        "pd.DataFrame",
        frame.stack(level="Ticker", future_stack=True),  # noqa: PD013
    )
    stacked = stacked.rename(columns=YAHOO_FIELDS).reset_index()
    stacked = stacked.rename(columns={"Date": "date", "Ticker": "ticker"})
    stacked["date"] = (
        pd.to_datetime(stacked["date"]).dt.tz_localize(None).dt.normalize()
    )
    stacked = stacked.dropna(subset=["close", "adj_close"])
    stacked["source"] = "Yahoo"
    return cast("pd.DataFrame", stacked[list(ROW_COLUMNS)]).reset_index(drop=True)


def _head(recent: pd.Series, older: pd.Series) -> pd.Series:
    later = recent.dropna().sort_index()
    if later.empty:
        return later
    before = older.dropna().sort_index()
    seam = cast("pd.Timestamp", later.index[0])
    meet = float(cast("float", before.get(seam, math.nan)))
    if not meet > 0:
        return later.iloc[:0]
    scale = float(cast("float", later.iloc[0])) / meet
    return cast("pd.Series", before[before.index < seam]) * scale


def spliced(recent: pd.DataFrame, older: pd.DataFrame) -> pd.DataFrame:
    kept: list[pd.DataFrame] = []
    for ticker, rows in recent.groupby("ticker"):
        past = cast("pd.DataFrame", older[older["ticker"] == ticker]).set_index("date")
        now = rows.set_index("date")
        close = _head(cast("pd.Series", now["close"]), cast("pd.Series", past["close"]))
        if close.empty:
            continue
        adjusted = _head(
            cast("pd.Series", now["adj_close"]), cast("pd.Series", past["adj_close"])
        )
        head = pd.DataFrame({"close": close, "adj_close": adjusted})
        head = head.rename_axis("date").reset_index()
        head["ticker"] = ticker
        head["source"] = "Yahoo"
        kept += [head.reindex(columns=list(ROW_COLUMNS)), rows]
    return (
        pd.concat(kept, ignore_index=True).sort_values(["ticker", "date"])
        if kept
        else empty_rows()
    )


def _with_early_years(
    bars: pd.DataFrame, start: date, wanted: Sequence[str], max_wait: float
) -> pd.DataFrame:
    if start >= markets.ALPACA_START or bars.empty:
        return bars
    first = bars.groupby("ticker")["date"].min()
    seam = pd.Timestamp(ALPACA_SEAM)
    early = [
        t
        for t in wanted
        if t in first.index and cast("pd.Timestamp", first.loc[t]) <= seam
    ]
    if not early:
        return bars
    older = yahoo_rows(early, start, ALPACA_SEAM + timedelta(days=1), max_wait)
    joined = spliced(cast("pd.DataFrame", bars[bars["ticker"].isin(early)]), older)
    late = cast("pd.DataFrame", bars[~bars["ticker"].isin(early)])
    return pd.concat([joined, late], ignore_index=True)


def daily(
    tickers: Sequence[str],
    start: date,
    end: date | None,
    keys: markets.Keys,
    max_wait: float = limits.BACKGROUND_WAIT,
) -> pd.DataFrame:
    us = [t for t in tickers if route("daily", t, keys)[0] == "Alpaca"]
    parts: list[pd.DataFrame] = []
    handled: set[str] = set()
    if us:
        last = (
            end - timedelta(days=1)
            if end is not None
            else datetime.now(markets.NEW_YORK).date()
        )
        try:
            with httpx.Client(timeout=markets.TIMEOUT) as client:
                alpaca = markets.Alpaca(keys, client, max_wait)
                bars = alpaca_rows(alpaca, us, start, last)
        except PROVIDER_ERRORS:
            bars = empty_rows()
        handled |= set(bars["ticker"])
        parts.append(_with_early_years(bars, start, us, max_wait))
    rest = [t for t in tickers if t not in handled]
    if rest:
        parts.insert(0, yahoo_rows(rest, start, end, max_wait))
    filled = [part for part in parts if not part.empty]
    return pd.concat(filled, ignore_index=True) if filled else empty_rows()


@dataclass(frozen=True, slots=True)
class SplitRow:
    ex_date: date
    ratio: float


def _yahoo_splits(ticker: str) -> tuple[SplitRow, ...]:
    with yahoo.ticker(ticker) as stock:
        found = stock.splits
    return tuple(
        SplitRow(cast("pd.Timestamp", day).date(), float(ratio))
        for day, ratio in found.items()
        if float(ratio) > 0
    )


def splits(
    tickers: Sequence[str], start: date, last: date, keys: markets.Keys
) -> dict[str, tuple[SplitRow, ...]]:
    us = [t for t in tickers if route("splits", t, keys)[0] == "Alpaca"]
    out: dict[str, tuple[SplitRow, ...]] = {}
    if us:
        try:
            table = markets.Alpaca(keys).splits(us, start, last)
        except PROVIDER_ERRORS:
            table = None
        if table is not None:
            for ticker in us:
                rows = table[table["ticker"] == ticker]
                out[ticker] = tuple(
                    SplitRow(date.fromisoformat(str(day)), float(ratio))
                    for day, ratio in zip(rows["ex_date"], rows["ratio"], strict=True)
                )
    for ticker in tickers:
        if ticker not in out:
            out[ticker] = _yahoo_splits(ticker)
    return out


@dataclass(frozen=True, slots=True)
class Cap:
    day: date
    market_cap: float
    currency: str


def _yahoo_cap(ticker: str, day: date, max_wait: float) -> Cap:
    info = yahoo.info(ticker, max_wait)
    cap = info.get("marketCap")
    currency = info.get("currency")
    return Cap(
        day,
        float(cap) if isinstance(cap, int | float) else math.nan,
        currency if isinstance(currency, str) else "",
    )


def market_caps(
    tickers: Sequence[str],
    max_wait: float = limits.BACKGROUND_WAIT,
) -> dict[str, Cap]:
    today = datetime.now(UTC).date()
    with ThreadPoolExecutor(yahoo.WORKERS) as pool:
        jobs = {
            ticker: pool.submit(_yahoo_cap, ticker, today, max_wait)
            for ticker in tickers
        }
    caps = {t: job.result() for t, job in jobs.items() if job.exception() is None}
    return {ticker: cap for ticker, cap in caps.items() if cap.market_cap > 0}


def points(values: pd.Series) -> tuple[tuple[date, float], ...]:
    return tuple(
        (cast("pd.Timestamp", day).date(), float(value))
        for day, value in values.items()
    )


def _is_continuous(ticker: str) -> bool:
    return ticker.endswith(("=X", "=F")) or ticker == "DX-Y.NYB"


def _market_close(ticker: str) -> tuple[ZoneInfo, clock_time]:
    if ticker in INDEX_CLOSES:
        return INDEX_CLOSES[ticker]
    for suffixes, zone, settled in MARKET_CLOSES:
        if ticker.endswith(suffixes):
            return zone, settled
    return markets.NEW_YORK, SESSION_SETTLED


def last_session(now: datetime, ticker: str = "") -> date:
    """티커의 거래소 현지 마감 시각 기준으로 최신 종료 거래일을 반환한다."""
    upper = ticker.strip().upper()
    if upper and _is_continuous(upper):
        return now.astimezone(UTC).date()
    zone, settled = _market_close(upper)
    local = now.astimezone(zone)
    day = local.date()
    return day if local.time() >= settled else day - timedelta(days=1)


def drifted(store: pd.DataFrame, fresh: pd.DataFrame) -> list[str]:
    if store.empty or fresh.empty:
        return []
    both = store.merge(fresh, on=KEY, suffixes=("", "_new"))
    change = (both["adj_close_new"] / both["adj_close"] - 1).abs()
    return sorted(set(both.loc[change > ADJUST_TOLERANCE, "ticker"].astype("str")))


def empty_stored() -> pd.DataFrame:
    frame = empty_rows()
    frame["fetched_at"] = pd.to_datetime(frame["date"])
    return frame


def _stored(frame: pd.DataFrame, fetched_at: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return empty_stored()
    out = frame.reindex(columns=list(STORED_COLUMNS))
    out["date"] = pd.DatetimeIndex(out["date"]).tz_localize(None).normalize()
    out["ticker"] = out["ticker"].astype("str")
    out["source"] = out["source"].astype("str")
    out["fetched_at"] = fetched_at
    for name in PRICE_FIELDS:
        out[name] = pd.to_numeric(out[name]).astype("float64")
    out = out.dropna(subset=["close", "adj_close"], how="all")
    return out.drop_duplicates(KEY, keep="last").reset_index(drop=True)


def _combined(stored: pd.DataFrame, rows: pd.DataFrame, *, whole: bool) -> pd.DataFrame:
    kept = stored
    if whole and not rows.empty:
        firsts = rows.groupby("ticker")["date"].min()
        replaced = stored["date"] >= stored["ticker"].map(firsts)
        kept = cast("pd.DataFrame", stored[~replaced])
    frames = [frame for frame in (kept, rows) if not frame.empty]
    if not frames:
        return empty_stored()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(KEY, keep="last")
    return merged.sort_values(KEY, ignore_index=True)


@dataclass(frozen=True, slots=True)
class PriceResult:
    got: tuple[str, ...]
    lost: Mapping[str, str]
    reached: limits.LimitReached | None = None


@dataclass(frozen=True, slots=True)
class PriceSeries:
    series: pd.Series
    source: str


@dataclass(frozen=True, slots=True)
class _Ask:
    tickers: tuple[str, ...]
    start: date
    whole: bool


@dataclass(slots=True)
class _Run:
    start: date
    end: date
    force: bool
    get: RowFetch
    lost: dict[str, str] = field(default_factory=dict)
    reached: limits.LimitReached | None = None


@final
class PriceStore:
    def __init__(
        self,
        keys: markets.Keys | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.keys = keys or markets.Keys()
        self._clock = clock
        self._lock = threading.RLock()
        self._busy_guard = threading.Lock()
        self._ticker_locks: dict[str, threading.Lock] = {}

    def stamp(self) -> int:
        return memory.current().price_version

    def rows(self, tickers: Sequence[str] | None = None) -> pd.DataFrame:
        found = memory.current().prices
        frame = empty_stored() if found is None else found
        if tickers is None:
            return frame
        kept = cast("pd.DataFrame", frame[frame["ticker"].isin(list(tickers))])
        return kept.reset_index(drop=True)

    def _ranges(self, tickers: Sequence[str]) -> dict[str, tuple[date, date]]:
        found = self.rows(tickers)
        priced = found[found["close"].notna()]
        spans = priced.groupby("ticker")["date"].agg(["min", "max"])
        return {
            str(ticker): (
                cast("pd.Timestamp", row["min"]).date(),
                cast("pd.Timestamp", row["max"]).date(),
            )
            for ticker, row in spans.iterrows()
        }

    def _download(
        self, tickers: Sequence[str], start: date, end: date, *, max_wait: float
    ) -> pd.DataFrame:
        try:
            return daily(tickers, start, end, self.keys, max_wait)
        except (limits.SourceError, *PROVIDER_ERRORS):
            raise
        except Exception as error:
            raise limits.SourceError(f"시세 수신 실패 ({error})") from error

    def _plan(self, tickers: Sequence[str], run: _Run) -> list[_Ask]:
        ranges = self._ranges(tickers)
        now = self._clock()
        tails: dict[date, list[str]] = {}
        wholes: dict[date, list[str]] = {}
        for ticker in tickers:
            wanted_to = min(run.end, last_session(now, ticker))
            span = ranges.get(ticker)
            if span is None:
                wholes.setdefault(run.start, []).append(ticker)
                continue
            covered = memory.current().covered.get(ticker)
            short = (
                span[0] > run.start + timedelta(days=GAP_DAYS)
                if covered is None
                else covered[0] > run.start
            )
            if short:
                wholes.setdefault(min(run.start, span[0]), []).append(ticker)
            elif run.force or (covered[1] if covered else span[1]) < wanted_to:
                tails.setdefault(span[1] - timedelta(days=OVERLAP_DAYS), []).append(
                    ticker
                )
        return [
            *(
                _Ask(tuple(names), day, whole=False)
                for day, names in sorted(tails.items(), reverse=True)
            ),
            *(
                _Ask(tuple(names), day, whole=True)
                for day, names in sorted(wholes.items())
            ),
        ]

    @contextmanager
    def _busy(self, tickers: Sequence[str]) -> Generator[None]:
        with self._busy_guard:
            held = [
                self._ticker_locks.setdefault(ticker, threading.Lock())
                for ticker in sorted(tickers)
            ]
        for lock in held:
            _ = lock.acquire()
        try:
            yield
        finally:
            for lock in held:
                lock.release()

    def ensure(  # noqa: PLR0913
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
        *,
        force: bool = False,
        max_wait: float = limits.BACKGROUND_WAIT,
        fetch: RowFetch | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> PriceResult:
        names = list(dict.fromkeys(tickers))
        get = fetch or partial(self._download, max_wait=max_wait)
        run = _Run(start, end, force, get)
        for i in range(0, len(names), CHUNK):
            group = names[i : i + CHUNK]
            if progress is not None:
                progress(i, len(names))
            with self._busy(group):
                for ask in self._plan(group, run):
                    self._answer(ask, run)
        got = tuple(t for t in names if t not in run.lost)
        return PriceResult(got, run.lost, run.reached)

    def _answer(self, ask: _Ask, run: _Run) -> None:
        rows = self._fetched(ask, run)
        if rows is None:
            return
        if not ask.whole:
            before = self.rows(ask.tickers)
            before = before[before["date"] >= pd.Timestamp(ask.start)]
            settled = before.sort_values(KEY).groupby("ticker").head(-1)
            changed = drifted(settled, rows)
            if changed:
                self._commit(
                    cast("pd.DataFrame", rows[~rows["ticker"].isin(changed)]), ask, run
                )
                ranges = self._ranges(changed)
                first = min((span[0] for span in ranges.values()), default=run.start)
                whole = _Ask(tuple(changed), min(run.start, first), whole=True)
                self._answer(whole, run)
                return
        self._commit(rows, ask, run)

    def _fetched(self, ask: _Ask, run: _Run) -> pd.DataFrame | None:
        try:
            frame = run.get(ask.tickers, ask.start, run.end + timedelta(days=1))
        except limits.LimitReached as error:
            run.reached = error
            run.lost.update(dict.fromkeys(ask.tickers, str(error)))
            return None
        except (limits.SourceError, *PROVIDER_ERRORS) as error:
            run.lost.update(dict.fromkeys(ask.tickers, str(error)))
            return None
        now = self._clock().astimezone(UTC).replace(tzinfo=None)
        rows = _stored(frame, cast("pd.Timestamp", pd.Timestamp(now)))
        found = set(rows["ticker"])
        run.lost.update({t: NO_PRICE for t in ask.tickers if t not in found})
        return rows

    def _commit(self, rows: pd.DataFrame, ask: _Ask, run: _Run) -> None:
        with self._lock:
            if not rows.empty:
                merged = _combined(self.rows(), rows, whole=ask.whole)
                memory.current().prices = merged
                self._cover(ask, run, merged, set(rows["ticker"]))
                memory.current().price_version += 1

    def _cover(self, ask: _Ask, run: _Run, merged: pd.DataFrame, got: set[str]) -> None:
        now = self._clock()
        firsts = merged[merged["ticker"].isin(got)].groupby("ticker")["date"].min()
        for ticker in ask.tickers:
            to = min(run.end, last_session(now, ticker))
            known = memory.current().covered.get(ticker)
            if ticker not in firsts.index:
                continue
            first = cast("pd.Timestamp", firsts.loc[ticker]).date()
            start = known[0] if known else first
            memory.current().covered[ticker] = (
                min(start, ask.start) if ask.whole else start,
                max(to, known[1]) if known else to,
            )

    def series(self, ticker: str, basis: Basis) -> pd.Series:
        name = "adj_close" if basis == "total" else "close"
        rows = self.rows([ticker]).dropna(subset=[name])
        return pd.Series(
            rows[name].to_numpy(), index=pd.DatetimeIndex(rows["date"]), dtype="float64"
        )

    def saved_at(self, tickers: Sequence[str]) -> datetime | None:
        found = self.rows(tickers)
        if found.empty:
            return None
        newest = cast("pd.Timestamp", found["fetched_at"].max())
        return newest.to_pydatetime().replace(tzinfo=UTC).astimezone()

    def latest(
        self, tickers: Sequence[str], start: date, end: date, basis: Basis
    ) -> dict[str, PriceSeries]:
        name = "adj_close" if basis == "total" else "close"
        found = self.rows(tickers)
        kept = found[
            (found["date"] >= pd.Timestamp(start))
            & (found["date"] <= pd.Timestamp(end))
        ].dropna(subset=[name])
        out: dict[str, PriceSeries] = {}
        for ticker, group in kept.sort_values(KEY).groupby("ticker"):
            values = pd.Series(
                group[name].to_numpy(),
                index=pd.DatetimeIndex(group["date"]),
                dtype="float64",
            )
            sources = dict.fromkeys(str(s) for s in group["source"])
            out[str(ticker)] = PriceSeries(values, "+".join(sources))
        return out

    def sources_by_ticker(self) -> dict[str, tuple[str, ...]]:
        """티커별 저장된 시세 출처 목록을 반환한다."""
        found = self.rows()
        if found.empty:
            return {}
        out: dict[str, tuple[str, ...]] = {}
        for ticker, group in found.groupby("ticker"):
            sources = tuple(dict.fromkeys(str(src) for src in group["source"] if src))
            if sources:
                out[str(ticker)] = sources
        return out


_current = PriceStore()


def current() -> PriceStore:
    return _current


def configure(store: PriceStore) -> PriceStore:
    global _current  # noqa: PLW0603
    _current = store
    return store
