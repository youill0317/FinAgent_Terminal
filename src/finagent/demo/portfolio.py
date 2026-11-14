# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""오프라인·데모 모드용 포트폴리오 평가·수익률·시뮬레이션·13F 투자자 합성 계산기."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

from finagent import etf, fund_filings, investors, perf, portfolio, quant
from finagent.demo import COUNT, SHARE, VALUE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

YEARS: Final = 3
FUND: Final = "DEMOB"
SOURCE: Final = "데모"
INVESTOR: Final = investors.Investor("0000000999", "데모 투자자")
REPORTED: Final = (("DEMOA", "DEMOB", "DEMOC"), ("DEMOB", "DEMOC", "DEMOD"))


def _today() -> date:
    return datetime.now().astimezone().date()


def _days() -> pd.DatetimeIndex:
    today = _today()
    return pd.bdate_range(today - timedelta(days=365 * YEARS), today)


def _ago(days: int) -> date:
    return _today() - timedelta(days=days)


def _stamp(days: int) -> pd.Timestamp:
    return cast("pd.Timestamp", pd.Timestamp(_ago(days)))


def _flat(columns: Sequence[str], days: pd.Index | None = None) -> pd.DataFrame:
    index = _days() if days is None else days
    return pd.DataFrame(VALUE, index=index, columns=list(columns))


def _line(days: pd.Index | None = None) -> pd.Series:
    return pd.Series(VALUE, index=_days() if days is None else days, dtype="float64")


def load_prices(root: Path, _tickers: object) -> portfolio.Prices:
    names = portfolio.wanted(portfolio.load_book(root))
    empty = pd.DataFrame(columns=names, index=pd.DatetimeIndex([]), dtype="float64")
    return portfolio.Prices(empty, empty.copy())


def _cell(value: object) -> etf.Cell:
    day = _today().isoformat()
    return etf.Cell(value, SOURCE, day)


def load_funds(*_: object) -> dict[str, dict[str, etf.Cell]]:
    return {FUND: {"kind": _cell("ETF"), "aum": _cell(VALUE)}}


def _row(ticker: str, weight: float, asset: fund_filings.Asset) -> fund_filings.Row:
    return fund_filings.Row(ticker, None, None, ticker, weight, asset, "US", None)


def compositions(names: Sequence[str]) -> dict[str, etf.Composition]:
    if FUND not in names:
        return {}
    rows = (
        _row("DEMOA", 30.0, "stock"),
        _row("DEMOC", 25.0, "stock"),
        _row("DEMOD", 20.0, "stock"),
        _row("DEMOT", 25.0, "bond"),
    )
    return {
        FUND: etf.Composition(
            FUND, _today().isoformat(), SOURCE, None, full=True, rows=rows
        )
    }


def look_sectors(*_: object) -> dict[str, str | None]:
    return {"DEMOA": "기술", "DEMOC": "헬스케어", "DEMOD": "금융"}


def look_returns(
    adjusted: pd.DataFrame, names: Sequence[str], days: pd.DatetimeIndex
) -> pd.DataFrame:
    del adjusted
    return _flat(names, days)


def prices_saved_at(_tickers: object) -> datetime:
    return datetime.now().astimezone()


def last_close(*_: object) -> tuple[float, date]:
    return VALUE, _today()


def close_on(close: pd.DataFrame, ticker: str, day: date) -> tuple[float, date]:
    del close, ticker
    return VALUE, day


def asset_returns(adjusted: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    del adjusted
    return _flat(names)


def _positions(p: portfolio.Portfolio) -> dict[str, portfolio.Position]:
    trades = portfolio.ordered(p.trades)
    out: dict[str, portfolio.Position] = {}
    for t in trades:
        mine = [x for x in trades if x.ticker == t.ticker]
        sells = [x.day for x in mine if x.side == "sell"]
        out[t.ticker] = portfolio.Position(
            t.ticker,
            quantity=0.0 if mine[-1].side == "sell" else VALUE,
            cost=VALUE,
            realized=VALUE,
            sold=VALUE,
            bought=VALUE,
            first=mine[0].day,
            opened=mine[0].day,
            last_sell=sells[-1] if sells else None,
        )
    return out


def ledger(p: portfolio.Portfolio) -> portfolio.Ledger:
    sales = tuple(
        portfolio.Sale(t, t.day - timedelta(days=COUNT), VALUE, VALUE)
        for t in portfolio.ordered(p.trades)
        if t.side == "sell"
    )
    return portfolio.Ledger(_positions(p), sales)


def holdings(p: portfolio.Portfolio, *_: object) -> list[portfolio.Holding]:
    return [
        portfolio.Holding(pos, VALUE, _today(), VALUE if pos.quantity else 0.0)
        for pos in _positions(p).values()
    ]


def history(*_: object) -> pd.DataFrame:
    return _flat(["value", "invested", "realized"])


def twr(*_: object) -> pd.Series:
    return _line()


def held_returns(*_: object) -> pd.Series:
    return _line()


def time_weighted(*_: object) -> perf.Annualised:
    return perf.Annualised(SHARE, SHARE, annualised=True)


def money_weighted(*_: object) -> perf.Annualised:
    return perf.Annualised(SHARE, SHARE, annualised=True)


def gap(*_: object) -> perf.Gap:
    return perf.Gap(SHARE, annualised=True)


def benchmark_lines(
    close: pd.DataFrame, days: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    del close
    return {portfolio.BENCHMARK_LABEL: _line(days)}


def market_returns(
    adjusted: pd.DataFrame,
    days: pd.DatetimeIndex,
    index: str = portfolio.TOTAL_RETURN,
) -> pd.Series:
    del adjusted, index
    return _line(days)


def trading(*_: object) -> portfolio.Trading:
    return portfolio.Trading(VALUE, VALUE, perf.Gap(SHARE, annualised=True))


def risk_free(rates: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    del rates
    return _line(index)


def ratios(*_: object) -> portfolio.Ratios:
    under = perf.Underwater(
        max_drawdown=-SHARE,
        peak=_stamp(3 * COUNT // 2),
        trough=_stamp(COUNT),
        recovery=_stamp(COUNT // 2),
        recovery_days=COUNT,
        longest_days=COUNT,
        current_days=COUNT,
    )
    return portfolio.Ratios(
        days=COUNT,
        span=VALUE,
        volatility=SHARE,
        under=under,
        sharpe=VALUE,
        ci=perf.SharpeCI(VALUE, VALUE, VALUE, COUNT),
        sortino=VALUE,
        calmar=VALUE,
        cdar=SHARE,
        te=SHARE,
        ir=VALUE,
    )


def market_beta(*_: object) -> float:
    return VALUE


def lineup(book: portfolio.Book, *_: object) -> pd.DataFrame:
    names = [p.name for p in book.portfolios if p.trades]
    return _flat([*names, portfolio.BENCHMARK_LABEL])


def shrunk_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    names = [str(c) for c in frame.columns]
    return pd.DataFrame(VALUE, index=names, columns=names)


def joint(adjusted: pd.DataFrame, w: pd.Series, months: int) -> portfolio.Joint:
    del adjusted, months
    names = tuple(portfolio.labels(w))
    window = _flat(names)
    market = _flat([portfolio.BENCHMARK])
    return portfolio.Joint(
        names, w.to_numpy(dtype="float64"), window, window, market, ""
    )


def covariance(j: portfolio.Joint) -> NDArray[np.float64]:
    n = len(j.names)
    return np.full((n, n), VALUE)


def correlation_frame(cov: NDArray[np.float64], names: Sequence[str]) -> pd.DataFrame:
    del cov
    return pd.DataFrame(VALUE, index=list(names), columns=list(names))


def betas(j: portfolio.Joint) -> list[portfolio.Beta]:
    names = (*j.names, portfolio.PORTFOLIO_LABEL)
    return [portfolio.Beta(name, VALUE) for name in names]


def diversification(j: portfolio.Joint, cov: NDArray[np.float64]) -> portfolio.Spread:
    del cov
    return portfolio.Spread(np.full(len(j.names), SHARE), VALUE)


def style(*_: object) -> portfolio.Style:
    names = (*portfolio.STYLE_PROXIES, portfolio.CASH)
    return portfolio.Style(names, np.full(len(names), SHARE), SHARE, (), SHARE)


def covered(rets: pd.DataFrame, w: pd.Series) -> tuple[pd.Series, list[str]]:
    del rets
    return w, []


def added(*_: object) -> portfolio.Added:
    pair = (VALUE, VALUE)
    shares = (SHARE, SHARE)
    return portfolio.Added(shares, pair, pair, shares, SHARE)


def added_curve(
    j: portfolio.Joint, cov: object, ticker: str, fractions: Sequence[float]
) -> list[portfolio.Mixed]:
    del j, cov, ticker
    return [portfolio.Mixed(SHARE, VALUE, VALUE, SHARE)] * len(fractions)


def mean_block(*_: object) -> float:
    return VALUE


def per_year(*_: object) -> float:
    return VALUE


def growth(
    j: portfolio.Joint,
    picks: NDArray[np.int64],
    *_: object,
    demean: bool = True,
) -> NDArray[np.float64]:
    del j, demean
    return np.full((1, picks.shape[1] + 1), VALUE)


def simulation(paths: NDArray[np.float64], *_: object) -> portfolio.Simulation:
    fan = [
        {"day": float(day), **{f"p{q}": VALUE for q in portfolio.PERCENTILES}}
        for day in range(cast("int", paths.shape[1]))
    ]
    ends = [{"start": VALUE, "end": VALUE, "count": COUNT}] * portfolio.BINS
    levels = dict.fromkeys(perf.PATH_LEVELS, perf.Reach(SHARE, SHARE, SHARE))
    risk = perf.PathRisk(levels, levels)
    tails = dict.fromkeys(portfolio.END_LEVELS, (SHARE, SHARE))
    reach = [
        {
            "x": side * level,
            "side": side,
            "end": SHARE,
            "touched": SHARE,
            "swing": SHARE,
        }
        for side in (-1.0, 1.0)
        for level in (0.0, *perf.PATH_LEVELS)
    ]
    return portfolio.Simulation(risk, fan, ends, tails, reach)


def tails(j: portfolio.Joint, *_: object) -> portfolio.Tails:
    test = quant.Backtest(
        COUNT, COUNT, VALUE, VALUE, VALUE, "green", VALUE, VALUE, VALUE, VALUE
    )
    return portfolio.Tails(test, len(j.window))


def density(*_: object, days: int = 0) -> portfolio.Density:
    del days
    return portfolio.Density(quant.Berkowitz(VALUE, VALUE, VALUE, VALUE, VALUE), COUNT)


def load_facts(*_: object) -> dict[str, portfolio.Facts]:
    names = ("DEMOA", "DEMOB", "DEMOC")
    return {
        name: portfolio.Facts(day=_today(), currency="USD", market_cap=VALUE)
        for name in names
    }


def market_weights(
    _: object, names: Sequence[str]
) -> tuple[NDArray[np.float64], list[str]]:
    return np.full(len(names), 1.0 / len(names)), []


def investor() -> tuple[
    investors.Filer,
    list[investors.Quarter],
    dict[str, investors.Link],
    dict[str, investors.Sector],
]:
    today = _today()
    last = investors.expected_period(today)
    tables = [
        investors.Table(
            accession=f"{INVESTOR.cik}-{n}",
            form="13F-HR",
            filed=period + timedelta(days=investors.DEADLINE_DAYS),
            period=period,
            rows=tuple(
                investors.Row(cusip=name, name=name, shares=VALUE, value=VALUE)
                for name in names
            ),
        )
        for n, (period, names) in enumerate(
            zip((last, investors.expected_period(last)), REPORTED, strict=True)
        )
    ]
    filer = investors.Filer(
        cik=INVESTOR.cik,
        name=INVESTOR.name,
        fetched=datetime.now().astimezone(),
        filings=tuple(
            investors.Filing(
                accession=t.accession, form=t.form, filed=t.filed, period=t.period
            )
            for t in tables
        ),
    )
    names = sorted({name for group in REPORTED for name in group})
    return (
        filer,
        investors.quarters(tables),
        {name: investors.Link(ticker=name) for name in names},
        {
            name: investors.Sector(sector=sector)
            for name, sector in look_sectors().items()
        },
    )


def clone(
    prices: pd.DataFrame, schedule: Sequence[tuple[date, pd.Series]]
) -> portfolio.Clone:
    del prices
    steps = tuple(portfolio.Step(day, len(w), SHARE) for day, w in schedule)
    w = schedule[-1][1]
    return portfolio.Clone(_line(), steps, w / float(w.sum()))


def rebalanced(*_: object) -> pd.Series:
    return _line()


def cumulative(daily_rets: pd.Series) -> pd.Series:
    return daily_rets


def compounded(*_: object) -> float:
    return SHARE


def peers(series: Mapping[str, pd.Series], *_: object) -> portfolio.Peers:
    rows = tuple(portfolio.Peer(name, COUNT, SHARE, VALUE, VALUE) for name in series)
    return portfolio.Peers(rows, len(rows), SHARE)


def black_litterman(
    j: portfolio.Joint, *_: object
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k = len(j.names)
    return np.zeros(k), np.zeros(k)
