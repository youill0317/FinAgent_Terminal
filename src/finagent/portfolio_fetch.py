# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""포트폴리오 종목 일봉·액면분할·시가총액·ETF 구성 백그라운드 수집기."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, final

from finagent import (
    etf,
    fund_filings,
    investors,
    limits,
    lookthrough,
    markets,
    memory,
    prices,
    yahoo,
)
from finagent.portfolio import (
    HISTORY_START,
    LOOK_DAYS,
    Facts,
    Prices,
    Split,
    Splits,
    is_fund,
    market_series,
    save_splits,
    sizes,
    wide_prices,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Progress:
    running: bool = False
    message: str = ""
    failed: bool = False


@final
class Official:
    def __init__(self, keys: markets.Keys, agent: str = "") -> None:
        self._keys = keys
        self._agent = agent.strip()

    def facts(self, tickers: Sequence[str]) -> dict[str, Facts]:
        return {
            ticker: Facts(day=cap.day, currency=cap.currency, market_cap=cap.market_cap)
            for ticker, cap in prices.market_caps(tickers).items()
        }

    def splits(self, tickers: Sequence[str], start: date) -> Splits:
        last = datetime.now(markets.NEW_YORK).date()
        return {
            ticker: tuple(Split(ex_date=row.ex_date, ratio=row.ratio) for row in rows)
            for ticker, rows in prices.splits(tickers, start, last, self._keys).items()
        }

    def fund(self, ticker: str) -> Mapping[str, object]:
        return yahoo.fund_info(ticker, limits.BACKGROUND_WAIT)

    def composition(self, ticker: str, today: date) -> etf.Composition | None:
        saved = etf.load_holdings(ticker)
        if saved is None or (self._agent and saved.source != etf.NPORT):
            return etf.composition(ticker, self._agent, today, every=True)
        if not self._agent:
            return saved
        web = investors.Web(self._agent)
        try:
            return etf.expand_tickers(ticker, web)
        finally:
            web.close()

    def sectors(self, names: Sequence[str]) -> None:
        if not self._agent or not names:
            return
        web = investors.Web(self._agent)
        try:
            _ = investors.fetch_sectors(web, names, _quiet)
        finally:
            web.close()


def _quiet(_stage: str, _done: int, _total: int) -> None:
    return


def saved_message(got: Sequence[str], lost: Sequence[str]) -> str:
    stocks = [t for t in got if not market_series(t)]
    markets = len(got) - len(stocks)
    parts = [f"{len(stocks)}종목 가격 수신"] if stocks else []
    if markets:
        parts.append(f"지수 {markets}개 갱신")
    if lost:
        parts.append(f"미수신: {', '.join(lost)}")
    return " · ".join(parts) or "수신된 가격 없음"


@final
class Fetcher:
    __slots__ = (
        "_asked",
        "_facts_progress",
        "_lock",
        "_look_progress",
        "_looked",
        "_lost",
        "_progress",
        "_saving",
        "_source",
        "_tried",
    )

    def __init__(
        self,
        source: Callable[[], Official],
    ) -> None:
        self._source = source
        self._lock = threading.Lock()
        self._saving = threading.Lock()
        self._progress = Progress()
        self._facts_progress = Progress()
        self._look_progress = Progress()
        self._tried: dict[str, date] = {}
        self._asked: set[str] = set()
        self._looked: set[str] = set()
        self._lost: set[str] = set()

    @property
    def progress(self) -> Progress:
        with self._lock:
            return self._progress

    @property
    def facts_progress(self) -> Progress:
        with self._lock:
            return self._facts_progress

    @property
    def look_progress(self) -> Progress:
        with self._lock:
            return self._look_progress

    def tickers(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._tried)

    def stamp(self) -> int:
        found = memory.current()
        tables = (found.holdings, found.facts, found.splits)
        versions = sum(table.version for table in tables)
        return found.price_version + len(self.tickers()) + versions

    def untried(self, tickers: Iterable[str], since: date = HISTORY_START) -> list[str]:
        with self._lock:
            return [t for t in tickers if self._tried.get(t, date.max) > since]

    def _try(self, tickers: Iterable[str], since: date) -> None:
        for ticker in tickers:
            self._tried[ticker] = min(since, self._tried.get(ticker, since))

    def start(
        self,
        tickers: Sequence[str],
        since: date = HISTORY_START,
        *,
        force: bool = False,
    ) -> bool:
        with self._lock:
            if self._progress.running or not tickers:
                return False
            if force:
                self._asked.intersection_update(memory.current().facts.rows())
                self._looked.difference_update(self._lost)
                self._lost.clear()
            self._try(tickers, since)
            self._progress = Progress(running=True, message=", ".join(tickers))
        threading.Thread(
            target=self._run,
            args=(list(tickers), since, force),
            name="finagent-portfolio",
            daemon=True,
        ).start()
        return True

    def start_facts(self, tickers: Sequence[str]) -> bool:
        with self._lock:
            todo = [t for t in dict.fromkeys(tickers) if t not in self._asked]
            if self._facts_progress.running or not todo:
                return False
            self._asked.update(todo)
            self._facts_progress = Progress(running=True, message=", ".join(todo))
        threading.Thread(
            target=self._run_facts,
            args=(todo,),
            name="finagent-facts",
            daemon=True,
        ).start()
        return True

    def _run_facts(self, tickers: list[str]) -> None:
        try:
            source = self._source()
            lost = self._funds(source, tickers, datetime.now(markets.NEW_YORK).date())
            funds = etf.FundStore().funds()
            got = source.facts(
                [t for t in tickers if t not in lost and not is_fund(funds.get(t))]
            )
            with self._saving:
                memory.current().facts.update(got)
        except Exception as error:  # noqa: BLE001  # 백그라운드 스레드 최상위 예외 포착
            self._finish(
                f"규모 수신 실패: {type(error).__name__}: {error}",
                facts=True,
            )
            return
        sized = sizes(memory.current().facts.rows(), funds)
        lost = [t for t in tickers if t not in sized]
        count = len(tickers) - len(lost)
        parts = [f"{count}종목 규모 수신"] if count else []
        if lost:
            parts.append(f"미수신: {', '.join(lost)}")
        self._finish(" · ".join(parts), failed=bool(lost), facts=True)

    def start_holdings(self, tickers: Sequence[str]) -> bool:
        with self._lock:
            todo = [t for t in dict.fromkeys(tickers) if t not in self._looked]
            if self._look_progress.running or not todo:
                return False
            self._looked.update(todo)
            self._look_progress = Progress(running=True, message=", ".join(todo))
        threading.Thread(
            target=self._run_holdings,
            args=(todo,),
            name="finagent-holdings",
            daemon=True,
        ).start()
        return True

    def look_now(self, tickers: Sequence[str]) -> None:
        with self._lock:
            todo = [t for t in dict.fromkeys(tickers) if t not in self._looked]
            self._looked.update(todo)
        if todo:
            self._run_holdings(todo)

    def _run_holdings(self, tickers: list[str]) -> None:
        try:
            message, lost = self._look(tickers)
        except Exception as error:  # noqa: BLE001  # 백그라운드 스레드 최상위 예외 포착
            with self._lock:
                self._lost.update(tickers)
            self._finish(f"구성 수신 실패: {type(error).__name__}: {error}", look=True)
            return
        with self._lock:
            self._lost.update(lost)
        self._finish(message, failed=bool(lost), look=True)

    def _funds(self, source: Official, tickers: list[str], today: date) -> list[str]:
        store = etf.FundStore()
        funds = store.funds()
        known = {
            t
            for t, cells in funds.items()
            if "kind" in cells and (not is_fund(cells) or "aum" in cells)
        }
        with ThreadPoolExecutor(yahoo.WORKERS, "finagent-funds") as pool:
            jobs = {t: pool.submit(source.fund, t) for t in tickers if t not in known}
        lost: list[str] = []
        for ticker, job in jobs.items():
            try:
                info = job.result()
            except etf.FAILURES:
                lost.append(ticker)
                continue
            for name, key in etf.YAHOO_FIELDS.items():
                cell = etf.Cell(info.get(key), etf.YAHOO, today.isoformat())
                _ = store.merge(ticker, name, cell)
        return lost

    def _look(self, tickers: list[str]) -> tuple[str, list[str]]:
        source = self._source()
        store = etf.FundStore()
        today = datetime.now(markets.NEW_YORK).date()
        lost = self._funds(source, tickers, today)
        funds = store.funds()
        found: dict[str, etf.Composition] = {}
        for ticker in [t for t in tickers if is_fund(funds.get(t))]:
            try:
                comp = source.composition(ticker, today)
            except etf.FAILURES:
                comp = etf.load_holdings(ticker)
                lost.append(ticker)
            if comp is not None:
                found[ticker] = comp
        names = list(
            dict.fromkeys(
                row.ticker
                for comp in found.values()
                if fund_filings.expandable(comp.rows)
                for row in lookthrough.expanded(comp.rows, full=comp.full)
                if row.ticker
            )
        )
        extra = [t for t in names if t not in tickers]
        result = prices.current().ensure(
            extra, today - timedelta(days=LOOK_DAYS), today
        )
        stocks = [t for t in tickers if t not in found and t not in lost]
        source.sectors(list(dict.fromkeys([*names, *stocks])))
        message = f"ETF {len(found)}개 구성 · 구성 종목 일봉 {len(result.got)}개"
        if lost:
            message += f" · 미수신: {', '.join(lost)}"
        return message, lost

    def fetch_now(self, tickers: Sequence[str]) -> Prices:
        with self._lock:
            self._try(tickers, HISTORY_START)
        result = self._ensure(tickers, HISTORY_START, force=False)
        self._splits([t for t in tickers if t in result.got])
        return wide_prices(prices.current().rows(list(self.tickers())))

    def _ensure(
        self, tickers: Sequence[str], since: date, *, force: bool
    ) -> prices.PriceResult:
        return prices.current().ensure(
            tickers, since, datetime.now(markets.NEW_YORK).date(), force=force
        )

    def _splits(self, tickers: Sequence[str]) -> None:
        stocks = [t for t in tickers if not market_series(t)]
        if stocks:
            found = self._source().splits(stocks, HISTORY_START)
            with self._saving:
                save_splits(found)

    def _run(self, tickers: list[str], since: date, force: bool) -> None:
        try:
            result = self._ensure(tickers, since, force=force)
        except Exception as error:  # noqa: BLE001  # 백그라운드 스레드 최상위 예외 포착
            self._finish(f"가격 수신 실패: {type(error).__name__}: {error}")
            return
        lost = [t for t in tickers if t not in result.got]
        got = [t for t in tickers if t in result.got]
        message = saved_message(got, lost)
        try:
            self._splits(got)
        except Exception as error:  # noqa: BLE001  # 백그라운드 스레드 최상위 예외 포착
            self._finish(f"{message} · 분할 수신 실패: {type(error).__name__}: {error}")
            return
        self._finish(message, failed=bool(lost))

    def _finish(
        self,
        message: str,
        *,
        failed: bool = True,
        facts: bool = False,
        look: bool = False,
    ) -> None:
        with self._lock:
            done = Progress(message=message, failed=failed)
            if look:
                self._look_progress = done
            elif facts:
                self._facts_progress = done
            else:
                self._progress = done
