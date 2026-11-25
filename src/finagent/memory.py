# pyright: reportMissingTypeStubs=false
"""프로세스 내 스레드 안전(Thread-safe) 인메모리 테이블 및 전역 상태 저장소."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date, datetime

    import pandas as pd

    from finagent import etf, fundamentals, investors, portfolio, sec
    from finagent.service import SeriesRecord
    from finagent.sources import Fetch, Series


@final
class Table[K, V]:
    def __init__(self) -> None:
        self._rows: dict[K, V] = {}
        self._lock = threading.Lock()
        self.version = 0

    def get(self, key: K) -> V | None:
        return self._rows.get(key)

    def rows(self) -> dict[K, V]:
        with self._lock:
            return dict(self._rows)

    def put(self, key: K, value: V) -> None:
        self.update({key: value})

    def update(self, rows: Mapping[K, V]) -> None:
        with self._lock:
            self._rows.update(rows)
            self.version += 1

    def drop(self, key: K) -> None:
        with self._lock:
            if key in self._rows:
                del self._rows[key]
                self.version += 1


@final
class Store:
    def __init__(self) -> None:
        self.prices: pd.DataFrame | None = None
        self.covered: dict[str, tuple[date, date]] = {}
        self.price_version = 0
        self.macro: dict[Fetch, Series] = {}
        self.macro_at: datetime | None = None
        self.extra: Table[str, SeriesRecord] = Table()
        self.funds: Table[str, dict[str, etf.Cell]] = Table()
        self.checked: Table[str, dict[str, str]] = Table()
        self.holdings: Table[str, etf.Composition] = Table()
        self.expenses: Table[str, dict[str, float]] = Table()
        self.fund_series: tuple[date, dict[str, sec.FundSeries]] | None = None
        self.companies: Table[str, fundamentals.Company] = Table()
        self.annuals: Table[str, tuple[date, dict[str, object]]] = Table()
        self.submissions: Table[
            str, tuple[date, tuple[fundamentals.FilingRow, ...]]
        ] = Table()
        self.facts: Table[str, portfolio.Facts] = Table()
        self.splits: Table[str, tuple[portfolio.Split, ...]] = Table()
        self.ciks: tuple[date, dict[str, str]] | None = None
        self.company_facts: Table[str, tuple[date, Mapping[str, object] | None]] = (
            Table()
        )
        self.yahoo_info: Table[str, tuple[date, Mapping[str, object]]] = Table()
        self.documents: Table[str, list[str]] = Table()
        self.mains: Table[str, str] = Table()
        self.bodies: Table[tuple[str, str], str] = Table()
        self.filers: Table[str, investors.Filer] = Table()
        self.filings: Table[str, investors.Table] = Table()
        self.cusips: Table[str, investors.Link] = Table()
        self.sectors: Table[str, investors.Sector] = Table()


_current = Store()


def current() -> Store:
    return _current


def reset() -> None:
    global _current  # noqa: PLW0603
    _current = Store()
