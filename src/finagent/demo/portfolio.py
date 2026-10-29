"""The demo's portfolio: ``portfolio``'s price readers and calculations, canned.

Each function here has the name and result type of the ``portfolio`` one
the page calls, so the page swaps the module (``calc`` in the app). None of
them reads a price or computes: the prices have columns and no rows, and
every result is built at ``VALUE`` (``SHARE`` for shares and returns,
``COUNT`` for counts), sized only by the names and days it is asked about.
The bookkeeping on these results (totals, weights, tables) stays the
page's, which prints each number as ``VALUE`` (``portfolio.demo_value``).
"""

# pandas ships no complete type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

from finagent import generative, perf, portfolio, quant
from finagent.demo import COUNT, SHARE, VALUE

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

YEARS: Final = 3  # of days every series runs over, for the longest window
METHODS: Final = ("역사적", "FHS", "EVT")


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


# ---------------------------------------------------------------- prices


def load_prices(root: Path) -> portfolio.Prices:
    """Columns for every ticker the book needs, and no rows."""
    names = portfolio.wanted(portfolio.load_book(root))
    empty = pd.DataFrame(columns=names, index=pd.DatetimeIndex([]), dtype="float64")
    return portfolio.Prices(empty, empty.copy())


def prices_saved_at(*_: object) -> datetime:
    return datetime.now().astimezone()


def last_close(*_: object) -> tuple[float, date]:
    return VALUE, _today()


def close_on(close: pd.DataFrame, ticker: str, day: date) -> tuple[float, date]:
    del close, ticker
    return VALUE, day


def convert(*_: object) -> float:
    return VALUE


def long_fx(*_: object) -> pd.Series:
    return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))


def asset_returns(
    adjusted: pd.DataFrame, names: Sequence[str], base: portfolio.Currency
) -> pd.DataFrame:
    del adjusted, base
    return _flat(names)


# ---------------------------------------------------------------- holdings


def _positions(p: portfolio.Portfolio) -> dict[str, portfolio.Position]:
    """Each ticker traded, open unless its last trade sold."""
    trades = portfolio.ordered(p.trades)
    out: dict[str, portfolio.Position] = {}
    for t in trades:
        mine = [x for x in trades if x.ticker == t.ticker]
        sells = [x.day for x in mine if x.side == "sell"]
        out[t.ticker] = portfolio.Position(
            t.ticker,
            quantity=0.0 if mine[-1].side == "sell" else VALUE,
            cost=VALUE,
            cost_base=VALUE,
            realized=VALUE,
            realized_base=VALUE,
            sold_base=VALUE,
            bought_base=VALUE,
            first=mine[0].day,
            opened=mine[0].day,
            last_sell=sells[-1] if sells else None,
        )
    return out


def ledger(p: portfolio.Portfolio, *_: object) -> portfolio.Ledger:
    """The positions and every sell, each held ``COUNT`` days."""
    sales = tuple(
        portfolio.Sale(t, t.day - timedelta(days=COUNT), VALUE, VALUE, VALUE, VALUE)
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
    p: portfolio.Portfolio, close: pd.DataFrame, days: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    del p, close
    names = ("혼합 벤치마크", portfolio.BENCHMARK_LABEL, "KOSPI")
    return {name: _line(days) for name in names}


def blended(
    p: portfolio.Portfolio, frame: pd.DataFrame, days: pd.DatetimeIndex
) -> pd.Series:
    del p, frame
    return _line(days)


def market_returns(
    adjusted: pd.DataFrame,
    base: portfolio.Currency,
    days: pd.DatetimeIndex,
    index: str = portfolio.TOTAL_RETURN,
) -> pd.Series:
    del adjusted, base, index
    return _line(days)


def currency_split(*_: object) -> perf.FxSplit:
    return perf.FxSplit(SHARE, SHARE, SHARE, SHARE)


def trading(*_: object) -> portfolio.Trading:
    return portfolio.Trading(VALUE, VALUE, perf.Gap(SHARE, annualised=True))


# ---------------------------------------------------------------- statistics


def risk_free(
    rates: pd.Series, index: pd.DatetimeIndex, quote: perf.Quote
) -> pd.Series:
    del rates, quote
    return _line(index)


def ratios(*_: object) -> portfolio.Ratios:
    under = perf.Underwater(
        max_drawdown=-SHARE,  # below 0, so the page shows its dates
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
        ci=perf.SharpeCI(VALUE, VALUE, VALUE, VALUE, COUNT),
        psr=SHARE,
        min_trl=VALUE,
        sortino=VALUE,
        calmar=VALUE,
        ulcer=VALUE,
        upi=VALUE,
        omega=VALUE,
        cdar=SHARE,
        te=SHARE,
        ir=VALUE,
    )


def market_beta(*_: object) -> float:
    return VALUE


def lineup(book: portfolio.Book, *_: object) -> tuple[pd.DataFrame, list[bool]]:
    names = [p.name for p in book.portfolios if p.trades]
    frame = _flat([*names, portfolio.BENCHMARK_LABEL])
    return frame, [False] * len(frame.columns)


def lagged_correlation(
    frame: pd.DataFrame, korean_: Sequence[bool]
) -> portfolio.Correlation:
    del korean_
    names = [str(c) for c in frame.columns]
    matrix = pd.DataFrame(VALUE, index=names, columns=names)
    return portfolio.Correlation(matrix, VALUE, SHARE, COUNT)


def joint(
    adjusted: pd.DataFrame, w: pd.Series, base: portfolio.Currency, months: int
) -> portfolio.Joint:
    del adjusted, base, months
    names = tuple(portfolio.labels(w))
    window = _flat(names)
    market = _flat([portfolio.BENCHMARK, portfolio.KOSPI])
    return portfolio.Joint(
        names, w.to_numpy(dtype="float64"), window, window, market, ""
    )


def covariance(j: portfolio.Joint) -> tuple[NDArray[np.float64], float, float]:
    n = len(j.names)
    return np.full((n, n), VALUE), VALUE, SHARE


def correlation_frame(cov: NDArray[np.float64], names: Sequence[str]) -> pd.DataFrame:
    del cov
    return pd.DataFrame(VALUE, index=list(names), columns=list(names))


def betas(j: portfolio.Joint) -> list[portfolio.Beta]:
    names = (*j.names, portfolio.PORTFOLIO_LABEL)
    return [portfolio.Beta(name, VALUE, VALUE, VALUE, COUNT) for name in names]


def diversification(j: portfolio.Joint, cov: NDArray[np.float64]) -> portfolio.Spread:
    del cov
    return portfolio.Spread(np.full(len(j.names), SHARE), VALUE, VALUE)


def style(*_: object) -> portfolio.Style:
    names = (*portfolio.STYLE_PROXIES, portfolio.CASH)
    return portfolio.Style(
        names, np.full(len(names), SHARE), SHARE, COUNT, (), SHARE, COUNT
    )


def covered(rets: pd.DataFrame, w: pd.Series) -> tuple[pd.Series, list[str]]:
    del rets
    return w, []


def added(*_: object) -> portfolio.Added:
    pair = (VALUE, VALUE)
    shares = (SHARE, SHARE)
    return portfolio.Added(shares, pair, pair, VALUE, VALUE, shares, SHARE)


# ---------------------------------------------------------------- risk


def volatility(j: portfolio.Joint, *_: object) -> quant.Devolatilised:
    """A GJR-GARCH filter at a flat volatility; the demo never fits one."""
    days, assets = min(len(j.full), quant.FILTER_DAYS), len(j.names)
    return quant.Devolatilised(
        z=np.zeros((days, assets)),
        sigma=np.full((days, assets), SHARE),
        now=np.full(assets, SHARE),
        step=np.zeros((4, assets)),
        garch=np.ones(assets, dtype=np.bool_),
    )


def vix_regressor(*_: object) -> None:
    return None


def range_proxy(*_: object) -> None:
    return None


def standing(*_: object) -> tuple[float, float]:
    return VALUE, VALUE


def mean_block(*_: object) -> float:
    return VALUE


def per_year(*_: object) -> float:
    return VALUE


def growth(
    j: portfolio.Joint, picks: NDArray[np.int64], *_: object
) -> NDArray[np.float64]:
    del j
    return np.full((1, picks.shape[1] + 1), VALUE)


def simulation(paths: NDArray[np.float64], *_: object) -> portfolio.Simulation:
    return simulated(cast("int", paths.shape[1]) - 1)


def simulated(days: int) -> portfolio.Simulation:
    """A fan, histogram and path shares at ``VALUE`` over ``days``."""
    fan = [
        {"day": float(day), **{f"p{q}": VALUE for q in portfolio.PERCENTILES}}
        for day in range(days + 1)
    ]
    ends = [{"start": VALUE, "end": VALUE, "count": COUNT}] * portfolio.BINS
    levels = dict.fromkeys(perf.PATH_LEVELS, perf.Reach(SHARE, SHARE, SHARE))
    paths = perf.PathRisk(levels, levels, COUNT, COUNT, COUNT, SHARE)
    return portfolio.Simulation(paths, fan, ends, np.ones(COUNT))


def tails(j: portfolio.Joint, *_: object) -> portfolio.Tails:
    """VaR and ES rows, backtests and the ES split by holding."""
    rows = tuple(
        portfolio.Tail(method, level, SHARE, SHARE, (SHARE, SHARE), (SHARE, SHARE))
        for method in METHODS
        for level in portfolio.VAR_LEVELS
    )
    test = quant.Backtest(
        COUNT, COUNT, VALUE, VALUE, VALUE, "green", VALUE, VALUE, VALUE, VALUE
    )
    return portfolio.Tails(
        rows,
        dict.fromkeys(METHODS, test),
        quant.BACKTEST_DAYS,
        np.full(len(j.names), SHARE),
        len(j.window),
        len(j.window),
    )


def density(*_: object, drift: object = None) -> portfolio.Density:
    del drift
    return portfolio.Density(quant.Berkowitz(VALUE, VALUE, VALUE, VALUE, VALUE), COUNT)


def generator(
    method: generative.Method, tickers: tuple[str, ...]
) -> generative.Generator:
    """A generator to choose, without weights: the demo never draws from it."""
    meta = {"train_start": _ago(COUNT).isoformat(), "train_end": _today().isoformat()}
    return generative.Generator(method, tickers, None, meta, {})


# ---------------------------------------------------------------- stress


def scenario_replays(*_: object) -> list[portfolio.Replayed]:
    return [
        portfolio.Replayed(
            s,
            (s.start, s.end),
            SHARE,
            SHARE,
            SHARE,
            VALUE,
            SHARE,
            SHARE,
            (),
            no_fx=False,
        )
        for s in portfolio.SCENARIOS
    ]


def bad_print_days(*_: object) -> dict[str, list[date]]:
    return {}


def local_betas(
    adjusted: pd.DataFrame, names: Sequence[str], months: int
) -> NDArray[np.float64]:
    del adjusted, months
    return np.full(len(names), VALUE)


def shocked(*_: object) -> list[tuple[portfolio.Shock, float]]:
    return [(shock, SHARE) for shock in portfolio.SHOCKS]


def reverse(*_: object) -> float:
    return SHARE


def turbulence(*_: object) -> portfolio.Turbulence:
    return portfolio.Turbulence(_today(), COUNT, VALUE, SHARE, VALUE, SHARE)


def trailing_means(
    j: portfolio.Joint,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    shape = (len(j.full) + 1, len(j.names))
    return np.zeros(shape), np.ones(shape)


def james_stein(j: portfolio.Joint) -> NDArray[np.float64] | None:
    if len(j.names) < 3:  # noqa: PLR2004
        return None
    return np.zeros((len(j.full) + 1, len(j.names)))


def bayes_stein(
    j: portfolio.Joint,
) -> tuple[NDArray[np.float64], float] | None:
    if len(j.names) < 3:  # noqa: PLR2004
        return None
    return np.zeros((len(j.full) + 1, len(j.names))), 0.0


def tradeoff(*_: object) -> portfolio.Tradeoff:
    return portfolio.Tradeoff(SHARE, SHARE, SHARE, SHARE, SHARE, SHARE)


def momentum_returns(
    j: portfolio.Joint, rf: pd.Series, lag_months: int
) -> tuple[NDArray[np.float64], float, int]:
    del rf, lag_months
    return np.zeros((len(j.full) + 1, len(j.names))), 0.0, COUNT


def load_facts(*_: object) -> dict[str, portfolio.Facts]:
    """Every demo ticker's figures at ``VALUE``, as if fetched today."""
    names = ("DEMOA", "DEMOB", "DEMOC", "DEMOK.KS")
    return {
        name: portfolio.Facts(
            day=_today(),
            currency="KRW" if portfolio.korean(name) else "USD",
            market_cap=VALUE,
        )
        for name in names
    }


def market_weights(
    _: object, names: Sequence[str], close: object
) -> NDArray[np.float64]:
    del close
    return np.full(len(names), 1.0 / len(names))


def load_factors(*_: object) -> pd.DataFrame:
    """One month of zero factors, so the factor model is offered."""
    columns = [*portfolio.FACTORS, "RF"]
    return pd.DataFrame([[0.0] * len(columns)], index=_days()[-1:], columns=columns)


def factor_returns(
    _: object, names: Sequence[str], w: object, factors: object
) -> tuple[NDArray[np.float64], float]:
    del w, factors
    return np.zeros(len(names)), 0.0


def black_litterman(j: portfolio.Joint, *_: object) -> NDArray[np.float64]:
    return np.zeros(len(j.names))
