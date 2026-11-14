# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""ETF 보유 구성을 관통한 실질 종목 노출·분산 기여도·충격 시나리오·섹터 비중 산출."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from finagent import fund_filings, investors

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from finagent.etf import Composition
    from finagent.fund_filings import Row

RESIDUAL: Final = "잔여"
UNKNOWN_SECTOR: Final = "미확인"


@dataclass(frozen=True, slots=True)
class Effective:
    ticker: str
    direct: float
    via: Mapping[str, float]

    @property
    def total(self) -> float:
        return self.direct + sum(self.via.values())


@dataclass(frozen=True, slots=True)
class Lookthrough:
    rows: tuple[Effective, ...]
    residual: Mapping[str, float]

    def totals(self) -> dict[str, float]:
        return {row.ticker: row.total for row in self.rows}


def expanded(rows: Sequence[Row], *, full: bool = True) -> tuple[Row, ...]:
    stocks = sorted(
        (row for row in rows if row.asset == "stock"), key=lambda row: -row.weight
    )
    return investors.cover(stocks, lambda row: row.weight) if full else tuple(stocks)


def effective_holdings(
    book_weights: Mapping[str, float], compositions: Mapping[str, Composition]
) -> Lookthrough:
    direct: dict[str, float] = {}
    via: dict[str, dict[str, float]] = {}
    residual: dict[str, float] = {}
    for name, weight in book_weights.items():
        found = compositions.get(name)
        if found is None or not fund_filings.expandable(found.rows):
            direct[name] = direct.get(name, 0.0) + weight
            continue
        spread = 0.0
        for row in expanded(found.rows, full=found.full):
            if row.ticker is None:
                continue
            part = weight * row.weight / 100
            through = via.setdefault(row.ticker, {})
            through[name] = through.get(name, 0.0) + part
            spread += part
        residual[name] = weight - spread
    found = [
        Effective(t, direct.get(t, 0.0), via.get(t, {}))
        for t in dict.fromkeys([*direct, *via])
    ]
    found.sort(key=lambda row: -row.total)
    return Lookthrough(tuple(found), residual)


def risk_contributions(
    totals: Mapping[str, float], stocks: pd.DataFrame, portfolio: pd.Series
) -> tuple[dict[str, float], float]:
    held = portfolio.dropna()
    x = held.to_numpy(dtype="float64")
    x = x - x.mean() if len(x) else x
    spread = float(x @ x)
    frame = stocks.reindex(index=held.index)
    out: dict[str, float] = {}
    if spread <= 0:
        return out, 1.0
    for ticker, weight in totals.items():
        if ticker not in frame.columns:
            continue
        r = frame[ticker].to_numpy(dtype="float64")
        if not np.isfinite(r).all():
            continue
        out[ticker] = weight * float((r - r.mean()) @ x) / spread
    return out, 1 - sum(out.values())


def stock_shock(totals: Mapping[str, float], ticker: str, move: float) -> float:
    return totals.get(ticker, 0.0) * move


def sector_weights(
    look: Lookthrough, sectors: Mapping[str, str | None]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in look.rows:
        name = sectors.get(row.ticker) or UNKNOWN_SECTOR
        out[name] = out.get(name, 0.0) + row.total
    ordered = dict(sorted(out.items(), key=lambda item: -item[1]))
    left = sum(look.residual.values())
    return {**ordered, RESIDUAL: left} if look.residual else ordered
