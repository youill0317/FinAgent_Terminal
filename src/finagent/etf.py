# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportExplicitAny=false
"""ETF 상품 메타데이터·수익률·공시(N-PORT/N-CSR) 보유 구성 수집 및 저장소."""

from __future__ import annotations

import contextlib
import math
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import date, timedelta
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, cast, final

import httpx
import pandas as pd

from finagent import (
    fund_filings,
    investors,
    limits,
    lookthrough,
    markets,
    memory,
    prices,
    sec,
    yahoo,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from finagent.etf_universe import Product

HISTORY_YEARS: Final = 4
YEAR_DAYS: Final = 365
ACTION_DAYS: Final = YEAR_DAYS + 30
NCSR_FORM: Final = "N-CSR"
NCSR_SUFFIX: Final = "_htm.xml"

NPORT: Final = "SEC N-PORT"
NCSR: Final = "SEC N-CSR"
ALPACA: Final = "Alpaca"
YAHOO: Final = "Yahoo"

ROUTES: Final[Mapping[str, tuple[str, ...]]] = {
    "kind": (YAHOO,),
    "name": (YAHOO,),
    "inception": (YAHOO,),
    "aum": (YAHOO,),
    "expense": (NCSR, YAHOO),
    "yield": (ALPACA, YAHOO),
    "family": (YAHOO,),
    "nav": (YAHOO,),
    "previous_close": (YAHOO,),
    "sector_weights": (YAHOO,),
    "credit_weights": (YAHOO,),
}
YAHOO_FIELDS: Final[Mapping[str, str]] = {
    "kind": "kind",
    "name": "name",
    "inception": "inception",
    "family": "family",
    "aum": "total_assets",
    "expense": "expense",
    "yield": "yield",
    "sector_weights": "sector_weights",
    "credit_weights": "credit_weights",
}

type Action = tuple[str, float]
type Verdict = Literal["확인됨", "개별 주식", "찾을 수 없음", "확인 실패"]
FAILURES: Final = (
    httpx.HTTPError,
    limits.SourceError,
    ValueError,
    KeyError,
    TypeError,
    ET.ParseError,
)


@dataclass(frozen=True, slots=True)
class Cell:
    value: object
    source: str
    as_of: str


_MERGING: Final = threading.Lock()


@final
class FundStore:
    @property
    def version(self) -> int:
        return memory.current().funds.version

    def funds(self) -> dict[str, dict[str, Cell]]:
        return memory.current().funds.rows()

    def merge(self, ticker: str, name: str, cell: Cell) -> bool:
        route = ROUTES[name]
        if cell.value is None or cell.source not in route:
            return False
        table = memory.current().funds
        with _MERGING:
            fields = table.get(ticker) or {}
            old = fields.get(name)
            if (
                old is not None
                and old.source in route
                and route.index(old.source) < route.index(cell.source)
            ):
                return False
            table.put(ticker, {**fields, name: cell})
            return True

    def checked(self, ticker: str, key: str) -> str | None:
        return (memory.current().checked.get(ticker) or {}).get(key)

    def check(self, ticker: str, key: str, value: str) -> None:
        table = memory.current().checked
        with _MERGING:
            table.put(ticker, {**(table.get(ticker) or {}), key: value})

    def prune(self, keep: Iterable[str], held: Iterable[str]) -> None:
        wanted = set(keep) | set(held)
        found = memory.current()
        for ticker in (set(found.funds.rows()) | set(found.checked.rows())) - wanted:
            found.funds.drop(ticker)
            found.checked.drop(ticker)


def split_factor(splits: Sequence[Action], after: str) -> float:
    return math.prod(ratio for day, ratio in splits if day > after)


def distribution_yield(
    dividends: Sequence[Action], splits: Sequence[Action], close: float, today: date
) -> float | None:
    start = (today - timedelta(days=YEAR_DAYS)).isoformat()
    recent = [(day, rate) for day, rate in dividends if day > start]
    if not recent or not close > 0:
        return None
    total = sum(rate / split_factor(splits, day) for day, rate in recent)
    return total / close * 100


def by_ticker(frame: pd.DataFrame, column: str) -> dict[str, list[Action]]:
    tickers = cast("list[str]", cast("pd.Series", frame["ticker"]).tolist())
    days = cast("list[str]", cast("pd.Series", frame["ex_date"]).tolist())
    values = cast("list[float]", cast("pd.Series", frame[column]).tolist())
    found: dict[str, list[Action]] = {}
    for ticker, day, value in zip(tickers, days, values, strict=True):
        found.setdefault(ticker, []).append((day, float(value)))
    return found


def confirm(ticker: str, quote: Callable[[str], object]) -> Verdict:
    try:
        quoted = str(quote(ticker) or "").upper()
    except FAILURES:
        return "확인 실패"
    if quoted in yahoo.FUND_KINDS:
        return "확인됨"
    if quoted in yahoo.STOCK_KINDS:
        return "개별 주식"
    return "찾을 수 없음"


@dataclass(frozen=True, slots=True)
class Progress:
    running: bool = False
    stage: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    failed: bool = False


def _patient[T](call: Callable[[], T]) -> T:
    while True:
        try:
            return call()
        except limits.LimitReached as error:
            limits.current().pause(max(1.0, error.next_at - time.time()))


@dataclass(frozen=True, slots=True)
class _Actions:
    dividends: Mapping[str, list[Action]]
    splits: Mapping[str, list[Action]]


@final
class Updater:
    __slots__ = (
        "_actions",
        "_agent",
        "_charted",
        "_keys",
        "_late",
        "_lock",
        "_pending",
        "_progress",
        "_slow",
        "_today",
        "store",
    )

    def __init__(
        self,
        keys: markets.Keys,
        charted: Callable[[date], Mapping[str, date]],
        agent: str = "",
        today: Callable[[], date] = date.today,
    ) -> None:
        self.store = FundStore()
        self._keys = keys
        self._charted = charted
        self._agent = agent.strip()
        self._today = today
        self._lock = threading.Lock()
        self._progress = Progress()
        self._slow = Progress()
        self._pending: list[str] = []
        self._late: list[str] = []
        self._actions = _Actions({}, {})

    @property
    def progress(self) -> Progress:
        with self._lock:
            return self._progress

    @property
    def slow(self) -> Progress:
        with self._lock:
            return self._slow

    def start(self, tickers: Sequence[str], *, force: bool = False) -> bool:
        names = list(dict.fromkeys(tickers))
        with self._lock:
            if self._progress.running:
                return False
            self._progress = Progress(running=True, stage="준비")
        threading.Thread(
            target=self._run, args=(names, force), name="finagent-etf", daemon=True
        ).start()
        return True

    def add(self, ticker: str) -> None:
        with self._lock:
            self._pending.append(ticker)
        _ = self.start([])

    def prune(self, products: Sequence[Product], held: Iterable[str]) -> None:
        self.store.prune((p.ticker for p in products), held)

    def _report(self, stage: str, done: int, total: int) -> None:
        with self._lock:
            self._progress = Progress(running=True, stage=stage, done=done, total=total)

    def _report_slow(self, stage: str, done: int, total: int) -> None:
        with self._lock:
            self._slow = Progress(running=True, stage=stage, done=done, total=total)

    def _run(self, tickers: list[str], force: bool) -> None:
        errors: list[str] = []
        message = ""
        while True:
            with self._lock:
                tickers = list(dict.fromkeys([*tickers, *self._pending]))
                self._pending = []
                if not tickers:
                    self._progress = Progress(
                        message=" · ".join(filter(None, [*errors, message])),
                        failed=bool(errors),
                    )
                    return
            try:
                text, failed = self._foreground(tickers, force)
            except Exception as error:  # noqa: BLE001  # 백그라운드 스레드 최상위 예외 포착
                errors.append(f"갱신 실패: {type(error).__name__}: {error}")
            else:
                if failed:
                    errors.append(text)
                else:
                    message = text
                self._later(tickers)
            tickers, force = [], False

    def _later(self, tickers: list[str]) -> None:
        with self._lock:
            self._late.extend(tickers)
            if self._slow.running:
                return
            self._slow = Progress(running=True, stage="준비")
        threading.Thread(
            target=self._run_slow, name="finagent-etf-slow", daemon=True
        ).start()

    def _run_slow(self) -> None:
        errors: list[str] = []
        message = ""
        while True:
            with self._lock:
                tickers = list(dict.fromkeys(self._late))
                self._late = []
                if not tickers:
                    self._slow = Progress(
                        message=" · ".join(filter(None, [*errors, message])),
                        failed=bool(errors),
                    )
                    return
            try:
                text, failed = self._background(tickers)
            except Exception as error:  # noqa: BLE001
                text, failed = f"{type(error).__name__}: {error}", True
            if failed:
                errors.append(text)
            else:
                message = text

    def _foreground(self, tickers: list[str], force: bool) -> tuple[str, bool]:
        today = self._today()
        first = today - timedelta(days=HISTORY_YEARS * YEAR_DAYS)
        charted = self._charted(today)
        starts: dict[date, list[str]] = {}
        for ticker in tickers:
            start = min(first, charted.get(ticker, first))
            starts.setdefault(start, []).append(ticker)
        got = sum(
            self._bars(names, start, today, force) for start, names in starts.items()
        )
        if self._keys.alpaca:
            self._report("분배·분할", 0, 0)
            start = today - timedelta(days=ACTION_DAYS)
            client = markets.Alpaca(self._keys)
            paid, ratios = _patient(lambda: client.payouts(tickers, start, today))
            self._actions = _Actions(
                by_ticker(paid, "rate"), by_ticker(ratios, "ratio")
            )
        stale = [
            t for t in tickers if self.store.checked(t, "yahoo") != today.isoformat()
        ]
        self._report("Yahoo 펀드 값", 0, len(stale))
        with ThreadPoolExecutor(yahoo.WORKERS, "finagent-etf-yahoo") as pool:
            jobs = [pool.submit(self._yahoo, ticker, today) for ticker in stale]
            failed = 0
            for i, job in enumerate(as_completed(jobs)):
                failed += not job.result()
                self._report("Yahoo 펀드 값", i + 1, len(stale))
        self._derive(tickers, today)
        lost = f" · Yahoo 펀드 값 미수신 {failed}개" if failed else ""
        return f"{len(tickers)}개 · 일봉 {got}개{lost}", bool(failed)

    def _bars(self, tickers: list[str], start: date, today: date, force: bool) -> int:
        store = prices.current()
        self._report("일봉", 0, len(tickers))
        result = store.ensure(
            tickers,
            start,
            today,
            force=force,
            progress=lambda done, total: self._report("일봉", done, total),
        )
        got = set(result.got)
        while result.reached is not None:
            limits.current().pause(max(1.0, result.reached.next_at - time.time()))
            left = [t for t, why in result.lost.items() if why != prices.NO_PRICE]
            result = store.ensure(left, start, today, force=force)
            got.update(result.got)
        return len(got)

    def _yahoo(self, ticker: str, today: date) -> bool:
        try:
            info = _patient(lambda: yahoo.fund_info(ticker, limits.BACKGROUND_WAIT))
        except FAILURES:
            return False
        for name, key in YAHOO_FIELDS.items():
            _ = self.store.merge(
                ticker, name, Cell(info.get(key), YAHOO, today.isoformat())
            )
        self.store.check(ticker, "yahoo", today.isoformat())
        return True

    def _derive(self, tickers: Iterable[str], today: date) -> None:
        actions = self._actions
        rows = prices.current().rows(list(tickers))
        last = rows.dropna(subset=["close"]).groupby("ticker").tail(1)
        for ticker, day, close in zip(
            last["ticker"], last["date"], last["close"], strict=True
        ):
            found = distribution_yield(
                actions.dividends.get(ticker, []),
                actions.splits.get(ticker, []),
                float(close),
                today,
            )
            if found is not None:
                _ = self.store.merge(
                    ticker, "yield", Cell(found, ALPACA, day.date().isoformat())
                )

    def _background(self, tickers: list[str]) -> tuple[str, bool]:
        if not (self._agent and self._agent.isascii()):
            return "", False
        edgar = sec.Client(self._agent)
        try:
            done, failed = self._expenses(edgar, tickers, self._today())
        finally:
            edgar.close()
        lost = f" · 미수신 {failed}개" if failed else ""
        return f"N-CSR {done}개{lost}", bool(failed)

    def _expenses(
        self, edgar: sec.Client, tickers: list[str], today: date
    ) -> tuple[int, int]:
        table = series_table(edgar, today)
        day = today.isoformat()
        listed = [
            t
            for t in tickers
            if t in table and self.store.checked(t, "ncsr_day") != day
        ]
        done = failed = 0
        for i, ticker in enumerate(listed):
            self._report_slow("N-CSR", i, len(listed))
            series = table[ticker]
            try:
                filing = _patient(
                    partial(edgar.latest_filing, series.series, NCSR_FORM)
                )
                if filing is None or filing.accession == self.store.checked(
                    ticker, "ncsr"
                ):
                    self.store.check(ticker, "ncsr_day", day)
                    continue
                ratios = self._ncsr(edgar, series.cik, filing.accession)
            except FAILURES:
                failed += 1
                continue
            self.store.check(ticker, "ncsr_day", day)
            self.store.check(ticker, "ncsr", filing.accession)
            ratio = ratios.get(series.class_id)
            if ratio is None:
                continue
            _ = self.store.merge(
                ticker, "expense", Cell(ratio * 100, NCSR, filing.filed)
            )
            done += 1
        return done, failed

    def _ncsr(self, edgar: sec.Client, cik: str, accession: str) -> dict[str, float]:
        table = memory.current().expenses
        saved = table.get(accession)
        if saved is not None:
            return saved
        names = _patient(lambda: edgar.archive_names(cik, accession))
        name = next((n for n in names if n.endswith(NCSR_SUFFIX)), None)
        found = (
            {}
            if name is None
            else fund_filings.parse_ncsr(
                _patient(lambda: edgar.archive(cik, accession, name))
            )
        )
        table.put(accession, found)
        return found


PERIODS: Final[Mapping[str, pd.DateOffset]] = {
    "1주": pd.DateOffset(weeks=1),
    "1개월": pd.DateOffset(months=1),
    "3개월": pd.DateOffset(months=3),
    "6개월": pd.DateOffset(months=6),
    "1년": pd.DateOffset(years=1),
}
TURNOVER_DAYS: Final = 20
TABLE_COLUMNS: Final = (
    "티커",
    "이름",
    "분류",
    "카테고리",
    "배수",
    *PERIODS,
    "총보수",
    "운용자산",
    "거래대금",
    "대표",
)


def wide(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    frame = rows.pivot_table(index="date", columns="ticker", values=column)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.sort_index()


def _period_return(series: pd.Series, offset: pd.DateOffset) -> float | None:
    kept = series.dropna()
    if kept.empty:
        return None
    end = cast("pd.Timestamp", kept.index[-1])
    start = end - offset
    before = cast("pd.Series", kept[kept.index <= start])
    if before.empty:
        return None
    return (float(kept.iloc[-1]) / float(before.iloc[-1]) - 1) * 100


def returns_frame(rows: pd.DataFrame) -> pd.DataFrame:
    adjusted = wide(rows, "adj_close")
    traded = wide(rows.assign(value=rows["close"] * rows["volume"]), "value")
    out: dict[str, dict[str, float | None]] = {}
    for ticker in adjusted.columns:
        series = cast("pd.Series", adjusted[ticker])
        found: dict[str, float | None] = {
            name: _period_return(series, offset) for name, offset in PERIODS.items()
        }
        if ticker in traded.columns:
            recent = cast("pd.Series", traded[ticker]).dropna().tail(TURNOVER_DAYS)
            found["거래대금"] = float(recent.mean()) if not recent.empty else None
        out[str(ticker)] = found
    return pd.DataFrame.from_dict(out, orient="index")


def scoped(
    products: Sequence[Product], group: str | None, category: str | None
) -> list[Product]:
    return [
        p
        for p in products
        if (group is None or p.group == group)
        and (category is None or p.category == category)
    ]


def _number(cell: Cell | None) -> float | None:
    if cell is None or not isinstance(cell.value, int | float):
        return None
    return float(cell.value)


def representatives(
    products: Sequence[Product], funds: Mapping[str, Mapping[str, Cell]]
) -> set[str]:
    best: dict[str, tuple[float, str]] = {}
    for p in products:
        size = _number(funds.get(p.ticker, {}).get("aum"))
        if p.multiple != 1 or size is None:
            continue
        if p.category not in best or size > best[p.category][0]:
            best[p.category] = (size, p.ticker)
    return {ticker for _, ticker in best.values()}


def table_frame(
    funds: Mapping[str, Mapping[str, Cell]],
    products: Sequence[Product],
    returns: pd.DataFrame,
    leaders: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p in products:
        cells = funds.get(p.ticker, {})
        name = cells.get("name")
        found = (
            cast("pd.Series", returns.loc[p.ticker])
            if p.ticker in returns.index
            else pd.Series(dtype="float64")
        )
        rows.append(
            {
                "티커": p.ticker,
                "이름": None if name is None else name.value,
                "분류": p.group,
                "카테고리": p.category,
                "배수": p.multiple,
                **{period: found.get(period) for period in PERIODS},
                "총보수": _number(cells.get("expense")),
                "운용자산": _number(cells.get("aum")),
                "거래대금": found.get("거래대금"),
                "대표": p.ticker in leaders,
            }
        )
    frame = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    return frame.sort_values(
        "운용자산", ascending=False, na_position="last", kind="stable"
    ).reset_index(drop=True)


NPORT_FORM: Final = "NPORT-P"
NPORT_FILE: Final = "primary_doc.xml"
HOME: Final = "US"


@dataclass(frozen=True, slots=True)
class Composition:
    ticker: str
    as_of: str
    source: str
    accession: str | None
    full: bool
    rows: tuple[fund_filings.Row, ...]
    assets: Mapping[str, float] | None = None


def load_holdings(ticker: str) -> Composition | None:
    return memory.current().holdings.get(ticker)


def _save_holdings(found: Composition) -> None:
    memory.current().holdings.put(found.ticker, found)


def series_table(edgar: sec.Client, today: date) -> dict[str, sec.FundSeries]:
    saved = memory.current().fund_series
    if saved is not None and saved[0] == today:
        return saved[1]
    found = _patient(edgar.fund_series)
    memory.current().fund_series = (today, found)
    return found


def _quiet(_stage: str, _done: int, _total: int) -> None:
    return


def _code(row: fund_filings.Row) -> str | None:
    if row.isin and not row.isin.startswith(HOME):
        return row.isin
    return row.cusip or row.isin


def _filed(row: fund_filings.Row) -> fund_filings.Row:
    ticker = (
        investors.yahoo_ticker(row.ticker)
        if row.ticker and row.country == HOME
        else None
    )
    return row if ticker == row.ticker else replace(row, ticker=ticker)


def with_tickers(
    rows: tuple[fund_filings.Row, ...], figi: investors.Source, *, every: bool = False
) -> tuple[fund_filings.Row, ...]:
    filed = tuple(_filed(row) for row in rows)
    pool = [*fund_filings.top(filed), *(lookthrough.expanded(filed) if every else ())]
    wanted = [row for row in pool if row.asset == "stock" and row.ticker is None]
    codes = [code for row in wanted if (code := _code(row))]
    if codes:
        _ = investors.map_cusips(figi, codes, _quiet)
    links = investors.load_cusips()
    out: list[fund_filings.Row] = []
    for row in filed:
        code = _code(row)
        link = links.get(code) if row.ticker is None and code else None
        out.append(row if link is None else replace(row, ticker=link.ticker))
    return tuple(out)


def expand_tickers(ticker: str, figi: investors.Source) -> Composition | None:
    saved = load_holdings(ticker)
    if saved is None or saved.source != NPORT:
        return saved
    rows = with_tickers(saved.rows, figi, every=True)
    if rows == saved.rows:
        return saved
    found = replace(saved, rows=rows)
    _save_holdings(found)
    return found


def _nport(ticker: str, agent: str, today: date, *, every: bool) -> Composition | None:
    saved = load_holdings(ticker)
    edgar = sec.Client(agent)
    try:
        table = series_table(edgar, today)
        series = table.get(ticker)
        if series is None:
            return None
        filing = edgar.latest_filing(series.series, NPORT_FORM, limits.INTERACTIVE_WAIT)
        if filing is None or (
            saved is not None and saved.accession == filing.accession
        ):
            return saved
        body = edgar.archive(
            series.cik, filing.accession, NPORT_FILE, limits.INTERACTIVE_WAIT
        )
    finally:
        edgar.close()
    parsed = fund_filings.parse_nport(body)
    web = investors.Web(agent)
    try:
        rows = with_tickers(parsed.rows, web, every=every)
    finally:
        web.close()
    found = Composition(
        ticker, parsed.as_of, NPORT, filing.accession, full=True, rows=rows
    )
    _save_holdings(found)
    return found


def _yahoo_holdings(ticker: str, today: date) -> Composition | None:
    raw = yahoo.fund(ticker, limits.INTERACTIVE_WAIT)
    top_rows = cast("list[dict[str, object]]", raw.get("top_holdings") or [])
    if not top_rows and not raw.get("asset_classes_pct"):
        return None
    rows = tuple(
        fund_filings.Row(
            name=str(row.get("name") or row.get("ticker") or ""),
            cusip=None,
            isin=None,
            ticker=str(row.get("ticker") or "") or None,
            weight=float(cast("float", row.get("weight_pct") or 0)),
            asset="stock",
            country=None,
            maturity=None,
        )
        for row in top_rows
    )
    found = Composition(
        ticker,
        today.isoformat(),
        YAHOO,
        None,
        full=False,
        rows=rows,
        assets=cast("Mapping[str, float] | None", raw.get("asset_classes_pct")),
    )
    _save_holdings(found)
    return found


def composition(
    ticker: str, agent: str, today: date, *, every: bool = False
) -> Composition | None:
    if agent and agent.isascii():
        try:
            found = _nport(ticker, agent, today, every=every)
        except FAILURES:
            if load_holdings(ticker) is None:
                _ = _yahoo_holdings(ticker, today)
            raise
        if found is not None:
            return found
    return _yahoo_holdings(ticker, today) or load_holdings(ticker)


def open_detail(
    ticker: str, agent: str, store: FundStore, today: date
) -> Composition | None:
    with contextlib.suppress(*FAILURES):
        quote = yahoo.fund_quote(ticker, limits.INTERACTIVE_WAIT)
        for name, value in quote.items():
            _ = store.merge(ticker, name, Cell(value, YAHOO, today.isoformat()))
    try:
        return composition(ticker, agent, today)
    except FAILURES:
        return load_holdings(ticker)


MONEY_COLUMNS: Final = ("운용자산", "거래대금")


def cell_text(column: str, value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if column == "대표":
        return "예" if value else "아니요"
    if not isinstance(value, int | float):
        return str(value)
    if column in PERIODS or column == "총보수":
        return f"{value:,.2f}%"
    if column in MONEY_COLUMNS:
        return f"{value:,.0f} USD"
    return f"{value:g}"
