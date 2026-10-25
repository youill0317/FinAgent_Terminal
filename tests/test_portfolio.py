"""Portfolios without the network: trade-log math, analytics, risk, the page."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import math
import shutil
import sys
import threading
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

from finagent import perf, quant
from finagent import portfolio as pf
from finagent.cli import APP, build_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from finagent.service import DashboardService

D1, D2, D3, D4, D5 = (date(2026, 3, d) for d in (2, 3, 4, 5, 6))


def trade(  # noqa: PLR0913 - a trade has six fields
    day: date,
    side: pf.Side,
    quantity: float,
    price: float,
    *,
    ticker: str = "AAA",
    fee: float = 0.0,
) -> pf.Trade:
    return pf.Trade(
        id=pf.new_id(),
        day=day,
        ticker=ticker,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
    )


def folio(*trades: pf.Trade, base: pf.Currency = "USD") -> pf.Portfolio:
    return pf.Portfolio(id="p1", name="테스트", base=base, trades=trades)


def frame(days: Sequence[date], **columns: Sequence[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    return pd.DataFrame(dict(columns), index=index, dtype="float64")


def market(start: str = "2006-12-01", end: str = "2026-10-21") -> pf.Prices:
    """Twenty years of made-up business-day prices; NEW lists in 2021."""
    days = pd.bdate_range(start, end)
    t = np.arange(len(days), dtype="float64")
    noise = np.random.default_rng(1).normal(0, 0.01, size=(len(days), 3))
    walk = np.exp(np.cumsum(noise, axis=0))
    adjusted = pd.DataFrame(
        {
            "AAA": 100 * walk[:, 0],
            "BBB.KS": 50_000 * walk[:, 1],
            pf.BENCHMARK: 1_000 * walk[:, 2],
            pf.TOTAL_RETURN: 1_000 * walk[:, 2] * np.exp(6e-5 * t),  # 1.5% a year
            pf.FX: 1_300 + 50 * np.sin(t / 200),
        },
        index=days,
    )
    adjusted["NEW"] = adjusted["AAA"].where(days >= pd.Timestamp("2021-01-04"))
    close = adjusted * 1.0
    close["AAA"] = adjusted["AAA"] * 1.02  # dividends paid: closes sit above
    return pf.Prices(close, adjusted)


# ---------------------------------------------------------------- the log


def test_moving_average_cost_with_partial_sells_and_fees() -> None:
    ledger = pf.replay(
        [
            trade(D1, "buy", 10, 100, fee=10),
            trade(D2, "buy", 10, 120, fee=10),
            trade(D3, "sell", 5, 130, fee=5),
        ]
    )
    position = ledger.positions["AAA"]
    (sale,) = ledger.sales

    # Cost 1,010 + 1,210 = 2,220 for 20 shares: 111 each.
    assert sale.average == pytest.approx(111)
    assert sale.basis == pytest.approx(555)
    assert sale.pnl == pytest.approx(650 - 5 - 555)
    assert sale.native_return == pytest.approx(90 / 555)
    assert sale.held_days == 2
    assert position.quantity == 15
    assert position.average == pytest.approx(111)
    assert position.realized == pytest.approx(90)


def test_a_full_close_moves_to_the_closed_positions() -> None:
    p = folio(
        trade(D1, "buy", 10, 100),
        trade(D2, "sell", 4, 130),
        trade(D3, "sell", 6, 90),
    )
    close = frame([D3], AAA=[95.0])
    items = pf.holdings(p, close)
    (held,) = items

    assert not held.is_open
    assert held.value == 0
    assert held.position.realized_base == pytest.approx(120 - 60)
    assert pf.holdings_table(items, "USD")["종목"] == []
    closed = pf.closed_table(items, "USD")
    assert closed["종목"] == ["AAA"]
    assert closed["실현손익 (USD)"] == ["+$60.00"]
    assert closed["원가 대비"] == ["+6.0%"]
    assert closed["첫 매수"] == [D1.isoformat()]
    assert closed["마지막 매도"] == [D3.isoformat()]
    log = pf.realized_table(pf.ledger(p, close).sales, "USD")
    assert log["구분"] == ["손절", "익절"]  # newest first
    assert log["실현손익 (USD)"] == ["−$60.00", "+$120.00"]
    totals = pf.totals(items)
    assert totals.pnl == pytest.approx(60)
    assert totals.total_return == pytest.approx(60 / 1000)


def test_buying_again_after_a_full_close_starts_a_new_cost_basis() -> None:
    ledger = pf.replay(
        [
            trade(D1, "buy", 10, 100),
            trade(D2, "sell", 10, 120),
            trade(D4, "buy", 2, 50),
            trade(D5, "sell", 1, 60),
        ]
    )
    position = ledger.positions["AAA"]

    assert position.average == pytest.approx(50)
    assert position.opened == D4
    assert position.first == D1
    assert ledger.sales[-1].held_days == 1  # from the new first buy
    assert position.realized == pytest.approx(200 + 10)


def test_sells_beyond_the_holding_on_that_day_are_rejected() -> None:
    book, pid = pf.add_portfolio(pf.Book(), "연금", "KRW")
    book = pf.add_trade(book, pid, trade(D2, "buy", 10, 100))

    with pytest.raises(pf.BookError, match="보유 10주보다"):
        _ = pf.add_trade(book, pid, trade(D3, "sell", 11, 100))
    with pytest.raises(pf.BookError, match="보유 0주보다"):
        _ = pf.add_trade(book, pid, trade(D1, "sell", 1, 100))
    # A same-day sell counts that day's buys.
    book = pf.add_trade(book, pid, trade(D2, "sell", 10, 100))
    sell_id = book.portfolios[0].trades[-1].id
    buy_id = book.portfolios[0].trades[0].id
    with pytest.raises(pf.BookError, match="지우면"):
        _ = pf.remove_trades(book, pid, {buy_id})
    book = pf.remove_trades(book, pid, {buy_id, sell_id})
    assert book.portfolios[0].trades == ()


def test_portfolios_are_named_once() -> None:
    book, pid = pf.add_portfolio(pf.Book(), " 연금 ", "KRW")
    book, other = pf.add_portfolio(book, "해외", "USD")

    assert [p.name for p in book.portfolios] == ["연금", "해외"]
    with pytest.raises(pf.BookError):
        _ = pf.add_portfolio(book, "연금", "KRW")
    with pytest.raises(pf.BookError):
        _ = pf.edit_portfolio(book, other, "", "USD")
    book = pf.edit_portfolio(book, other, "미국", "KRW")
    assert book.get(other) == pf.Portfolio(id=other, name="미국", base="KRW")
    assert [p.id for p in pf.remove_portfolio(book, pid).portfolios] == [other]


def test_tickers_and_their_currency() -> None:
    assert pf.clean_ticker(" aapl ") == "AAPL"
    assert pf.clean_ticker("005930.ks") == "005930.KS"
    assert pf.currency("005930.KS") == "KRW"
    assert pf.currency("035720.KQ") == "KRW"
    assert pf.currency("BRK-B") == "USD"
    with pytest.raises(pf.BookError):
        _ = pf.clean_ticker("7203.T")
    with pytest.raises(pf.BookError):
        _ = pf.clean_ticker("A B")


# ---------------------------------------------------------------- currency


def test_foreign_positions_split_the_currency_effect() -> None:
    p = folio(trade(D1, "buy", 10, 100), base="KRW")
    close = frame([D1, D2], AAA=[100.0, 110.0], **{pf.FX: [1_000.0, 1_100.0]})
    (held,) = pf.holdings(p, close)

    assert held.position.cost_base == pytest.approx(1_000_000)
    assert held.value == pytest.approx(1_210_000)
    assert held.native_return == pytest.approx(0.10)
    assert held.base_return == pytest.approx(0.21)
    table = pf.holdings_table([held], "KRW")
    assert table["환율 효과"] == ["+11.0%p"]
    assert table["평가손익 (KRW)"] == ["+₩210,000"]
    assert pf.convert(100, "USD", "KRW", 1_300) == 130_000
    assert pf.convert(130_000, "KRW", "USD", 1_300) == 100
    assert pf.convert(5, "KRW", "KRW", math.nan) == 5


def test_a_missing_rate_leaves_base_amounts_unknown() -> None:
    p = folio(trade(D1, "buy", 1, 100), base="KRW")
    (held,) = pf.holdings(p, frame([D1], AAA=[100.0]))

    assert math.isnan(held.value)
    assert pf.unpriced([held]) == ["AAA"]
    assert pf.holdings_table([held], "KRW")["평가액 (KRW)"] == ["—"]
    rates = cast("pd.Series", frame([D1], **{pf.FX: [1_000.0]})[pf.FX])
    assert pf.rate_on(rates, D5) == 1_000
    assert math.isnan(pf.rate_on(rates, date(2026, 3, 20)))  # too old


# ---------------------------------------------------------------- performance


def test_twr_ignores_buys_and_sells() -> None:
    p = folio(
        trade(D1, "buy", 10, 100),
        trade(D3, "buy", 10, 110),
        trade(D4, "sell", 10, 121),
    )
    close = frame(
        [D1, D2, D3, D4, D5],
        AAA=[100.0, 110.0, 121.0, 121.0, 133.1],
        **{pf.TOTAL_RETURN: [10.0, 11.0, 12.0, 13.0, 14.0]},
    )
    cumulative = pf.twr(p, close)

    # Day returns 0, +10%, +10% (the buy at 110 closes at 121), 0, +10%.
    assert cumulative.tolist() == pytest.approx([0, 0.1, 0.21, 0.21, 0.331])
    days = pd.DatetimeIndex(cumulative.index)
    sp = pf.market_returns(close, "USD", days)
    assert pf.cumulative(sp).iloc[-1] == pytest.approx(0.4)
    history = pf.history(p, close)
    assert history["invested"].tolist() == pytest.approx([1000, 1000, 2100, 890, 890])
    assert history["realized"].tolist() == pytest.approx([0, 0, 0, 1210 - 1050, 160])
    lines = pf.pnl_lines(history)
    assert lines["총손익"].iloc[-1] == pytest.approx(1331 - 890)
    assert lines["평가손익"].iloc[-1] == pytest.approx(1331 - 890 - 160)


def test_money_weighted_return_is_annual_only_from_a_year() -> None:
    p = folio(trade(date(2025, 1, 1), "buy", 1, 100))
    close = frame([date(2026, 1, 1)], AAA=[200.0])

    year = pf.money_weighted(p, close, 200.0)
    assert year is not None
    assert year.annual == pytest.approx(1.0, abs=1e-6)
    later = date(2026, 6, 10)  # 100 days after D1
    short = pf.money_weighted(
        folio(trade(D1, "buy", 1, 100)), frame([later], AAA=[110.0]), 110.0
    )
    assert short is not None
    assert short.annual is None
    assert short.cumulative == pytest.approx(0.1, abs=1e-6)


def test_peers_shrink_and_deflate_only_from_two() -> None:
    days = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(9)
    series = {
        name: pd.Series(rng.normal(mean, 0.01, len(days)), index=days)
        for name, mean in (("A", 0.002), ("B", 0.0), ("C", -0.001))
    }
    rf = pd.Series(0.0, index=days)

    alone = pf.peers({"A": series["A"]}, rf)
    assert alone.count == 1
    assert alone.rows[0].shrunk is None
    assert alone.deflated is None

    many = pf.peers(series, rf)
    assert [r.name for r in many.rows] == ["A", "B", "C"]  # highest Sharpe first
    assert many.count == 3
    sharpes = [cast("float", r.sharpe) for r in many.rows]
    shrunk = [cast("float", r.shrunk) for r in many.rows]
    mean = sum(sharpes) / 3
    for raw, pulled in zip(sharpes, shrunk, strict=True):
        assert abs(pulled - mean) <= abs(raw - mean) + 1e-12  # toward the mean
    x = series["A"].to_numpy()
    sr = cast("float", perf.sharpe(x))
    year = pf.per_year(days)  # each Sharpe is annualised by its own days a year
    trials = np.array([s_ / math.sqrt(year) for s_ in sharpes])
    bar = perf.expected_max_sr(3, float(trials.var(ddof=1)))
    assert bar is not None
    moments = perf.skew_kurt(x)
    assert moments is not None
    assert many.deflated == pytest.approx(perf.psr_from(sr, len(x), *moments, bar))
    grown = cast("float", np.prod(1 + x))
    assert many.rows[0].total == pytest.approx(grown - 1)
    short = {k: v.iloc[:100] for k, v in series.items()}
    assert pf.peers(short, rf).deflated is None  # under perf.MIN_OBS
    assert pf.peers(short, rf).rows[0].shrunk is not None


def test_held_weights_come_from_closes_and_returns_from_adjusted_closes() -> None:
    # Equal money in A and B on D1. A's adjusted closes sit at half its
    # closes (a large dividend to come); that must not halve its weight.
    close = frame([D1, D2, D3], A=[100.0, 110.0, 121.0], B=[100.0, 90.0, 81.0])
    adjusted = frame([D1, D2, D3], A=[50.0, 55.0, 60.5], B=[100.0, 90.0, 81.0])
    p = folio(
        trade(D1, "buy", 10, 100, ticker="A"), trade(D1, "buy", 10, 100, ticker="B")
    )

    rets = pf.held_returns(p, close, adjusted)

    # Day 2: half and half, +10% and -10%. Day 3: 1,100 and 900 of 2,000.
    assert rets.tolist() == pytest.approx([0.0, (1_100 * 0.1 - 900 * 0.1) / 2_000])
    # Adjusted closes as weights would have said 1/3 A: -3.3% on day 2.
    assert pf.held_returns(p, adjusted, adjusted).iloc[0] == pytest.approx(-0.1 / 3)


# ---------------------------------------------------------------- risk


def test_portfolio_correlations_put_back_the_seoul_lag() -> None:
    days = pd.bdate_range("2025-01-03", periods=301)
    move = np.random.default_rng(2).normal(0, 0.01, len(days))
    # Seoul closes first: today's US move shows up in Korea tomorrow.
    rets = pd.DataFrame(
        {"US": move[1:], "KR": move[:-1]}, index=days[1:], dtype="float64"
    )

    plain = pf.lagged_correlation(rets, [False, False])
    lagged = pf.lagged_correlation(rets, [False, True])

    assert plain is not None
    assert lagged is not None
    assert abs(float(cast("float", plain.matrix.loc["US", "KR"]))) < 0.2
    assert float(cast("float", lagged.matrix.loc["US", "KR"])) > 0.6
    assert lagged.days == 300
    assert pf.lagged_correlation(rets.head(10), [False, True]) is None
    assert pf.lagged_correlation(rets.drop(columns="KR"), [False]) is None
    assert pf.mixed(["AAA", "BBB.KS"])
    assert not pf.mixed(["BBB.KS", "CCC.KQ"])


def long_indices() -> pd.DataFrame:
    """Made-up closes: S&P 500 from 1986, KOSPI from 1996-07, KRW=X from 2003-12."""
    days = pd.bdate_range("1986-01-02", "2026-10-21")
    noise = np.random.default_rng(4).normal(0, 0.01, size=(len(days), 3))
    walk = np.exp(np.cumsum(noise, axis=0))
    frame_ = pd.DataFrame(
        {
            pf.BENCHMARK: 300 * walk[:, 0],
            pf.KOSPI: 800 * walk[:, 1],
            pf.FX: 1_000 * walk[:, 2],
        },
        index=days,
    )
    frame_.loc[: pd.Timestamp("1996-06-30"), pf.KOSPI] = np.nan
    frame_.loc[: pd.Timestamp("2003-11-30"), pf.FX] = np.nan
    return frame_


def test_every_scenario_window_lies_inside_the_stored_series() -> None:
    adjusted = long_indices()

    for scenario in pf.SCENARIOS:
        span = pf.resolve(scenario, adjusted)
        assert span is not None, scenario.name
        start, end = span
        assert start < end, scenario.name
        for ticker, stated in ((pf.BENCHMARK, scenario.us), (pf.KOSPI, scenario.kr)):
            if stated is None:
                continue
            series = pd.DatetimeIndex(adjusted[ticker].dropna().index)
            assert series[0] <= start, (scenario.name, ticker)
            assert end <= series[-1], (scenario.name, ticker)
        stated = [d for pair in (scenario.us, scenario.kr) if pair for d in pair]
        assert abs((start - pd.Timestamp(min(stated))).days) <= quant.SEARCH_DAYS
        assert abs((end - pd.Timestamp(max(stated))).days) <= quant.SEARCH_DAYS


def test_old_windows_take_the_won_from_fred_or_say_it_is_missing() -> None:
    adjusted = long_indices()
    values = {"NEW.KS": 1_000.0}
    imf = next(s for s in pf.SCENARIOS if s.name == "1997 IMF")
    span = pf.resolve(imf, adjusted)
    assert span is not None
    fred = pd.Series(
        np.linspace(900.0, 1_800.0, 20 * 260),
        index=pd.bdate_range("1990-01-01", periods=20 * 260),
    )

    with_fred = pf.scenario_replays(values, adjusted, "USD", pf.long_fx(adjusted, fred))
    without = pf.scenario_replays(values, adjusted, "USD", pf.fx_series(adjusted))
    row = next(r for r in with_fred if r.scenario.name == "1997 IMF")
    bare = next(r for r in without if r.scenario.name == "1997 IMF")

    assert row.proxied == ("NEW.KS",)
    assert not row.no_fx
    assert bare.no_fx
    kospi = adjusted[pf.KOSPI].loc[span[0] : span[1]]
    won = fred.loc[span[0] : span[1]]
    local = float(kospi.iloc[-1] / kospi.iloc[0] - 1)
    assert bare.change == pytest.approx(local)
    assert row.change == pytest.approx(
        (1 + local) * float(won.loc[: span[0]].iloc[-1] / won.iloc[-1]) - 1, rel=1e-3
    )
    assert (1 + row.fx) * (1 + local) == pytest.approx(1 + row.change)
    assert row.kospi == pytest.approx(local)
    spliced = pf.long_fx(adjusted, fred)
    assert spliced.index[0] == fred.index[0]
    assert spliced.loc["2010-01-04"] == adjusted.loc["2010-01-04", pf.FX]


# ---------------------------------------------------------------- method review


def test_beta_is_one_for_a_copy_of_the_index_held_with_a_gap() -> None:
    days = pd.bdate_range("2025-01-01", periods=300)
    walk = np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.01, len(days))))
    adjusted = pd.DataFrame({"AAA": 100 * walk, pf.BENCHMARK: 1_000 * walk}, index=days)
    dates = [day.date() for day in days]
    p = folio(
        trade(dates[10], "buy", 1, 100),
        trade(dates[100], "sell", 1, 100),
        trade(dates[200], "buy", 1, 100),
    )

    rets = pf.held_returns(p, adjusted, adjusted)

    assert len(rets) < len(days) - 100  # the sold-out stretch is left out
    assert pf.market_beta(rets, adjusted, "USD") == pytest.approx(1.0, abs=0.05)
    assert math.isnan(pf.market_beta(rets.head(2), adjusted, "USD"))


def test_monte_carlo_does_not_extrapolate_a_strong_year() -> None:
    days = pd.bdate_range("2023-01-02", periods=760)
    drift = np.exp(np.cumsum(np.random.default_rng(11).normal(0.003, 0.01, len(days))))
    adjusted = pd.DataFrame({"AAA": 100 * drift, pf.BENCHMARK: 1_000.0}, index=days)
    j = pf.joint(adjusted, pd.Series({"AAA": 1.0}), "USD", 36)  # +75% a year
    picks = quant.stationary_indices(len(j.window), 2_000, 252, pf.mean_block(j))

    ends = pf.growth(j, picks)[:, -1]

    assert 0.4 < float((ends < 1).mean()) < 0.65
    assert float(np.median(ends)) < 1.1


def test_the_lineup_flags_mostly_korean_books() -> None:
    prices = market()
    kr = folio(
        trade(date(2023, 1, 3), "buy", 1, 100),
        trade(date(2023, 1, 3), "buy", 40, 50_000, ticker="BBB.KS"),
        base="KRW",
    )
    new = trade(date(2023, 1, 3), "buy", 1, 100, ticker="NEW")
    us = pf.Portfolio(
        id="p2", name="미국", base="USD", trades=(trade(D1, "buy", 1, 100), new)
    )
    book = pf.Book(portfolios=(kr, us))

    frame_, flags = pf.lineup(book, prices, "KRW", 36)

    assert list(frame_.columns) == ["테스트", "미국", pf.BENCHMARK_LABEL]
    assert flags == [True, False, False]
    # The S&P 500 column is the total-return index, not the price one.
    tr = pf.window(pf.asset_returns(prices.adjusted, [pf.TOTAL_RETURN], "KRW"), 36)
    column = frame_[pf.BENCHMARK_LABEL].dropna()
    assert np.allclose(column, tr[pf.TOTAL_RETURN].loc[column.index])
    # NEW lists in 2021, inside a 6-year window: left out, AAA stands alone.
    six, _ = pf.lineup(book, prices, "KRW", 72)
    rets = pf.window(pf.asset_returns(prices.adjusted, ["AAA"], "KRW"), 72)
    assert np.allclose(six["미국"].dropna(), rets["AAA"].dropna().loc[six.index])
    found = pf.lagged_correlation(frame_, flags)
    assert found is not None
    assert found.matrix.loc["테스트", "테스트"] == pytest.approx(1.0)
    assert 0 <= found.shrink <= 1


def test_a_stand_in_takes_a_shrunk_beta_from_a_short_history() -> None:
    days = pd.bdate_range("2007-01-02", "2026-10-21")
    index = np.exp(np.cumsum(np.random.default_rng(9).normal(0, 0.01, len(days))))
    rets = np.diff(index) / index[:-1]
    late = np.full(len(days), np.nan)  # lists 120 days before the end, beta 1.5
    late[-121:] = 50 * np.cumprod(np.concatenate([[1.0], 1 + 1.5 * rets[-120:]]))
    short = np.full(len(days), np.nan)  # 40 days: too few, beta 1
    short[-41:] = 10 * np.cumprod(np.concatenate([[1.0], 1 + 1.5 * rets[-40:]]))
    adjusted = pd.DataFrame(
        {pf.BENCHMARK: 1_000 * index, "LATE": late, "SHORT": short, pf.FX: 1_300.0},
        index=days,
    )
    fx = pf.fx_series(adjusted)

    rows = pf.scenario_replays({"LATE": 600.0, "SHORT": 400.0}, adjusted, "USD", fx)
    gfc = next(r for r in rows if r.scenario.name == "금융위기")

    expected = quant.shrunk_beta(1.5, 120)  # about 1.16
    assert gfc.proxied == ("LATE", "SHORT")
    assert gfc.proxy_betas == pytest.approx((expected, 1.0), abs=1e-9)
    assert pf.proxy_beta(adjusted, "SHORT", pf.BENCHMARK) == 1.0
    assert gfc.window is not None
    start, end = (pd.Timestamp(d) for d in gfc.window)
    sp = adjusted[pf.BENCHMARK].loc[start:end].to_numpy()
    steps = sp[1:] / sp[:-1] - 1
    late_path = np.concatenate([[1.0], np.cumprod(1 + expected * steps)])
    path = (600 * late_path + 400 * sp / sp[0]) / 1_000
    assert gfc.change == pytest.approx(path[-1] - 1)
    assert gfc.worst == pytest.approx((path / np.maximum.accumulate(path) - 1).min())
    assert gfc.sp500 == pytest.approx(sp[-1] / sp[0] - 1)
    old = next(r for r in rows if r.scenario.name == "1987 블랙먼데이")
    assert old.window is None  # no index history back then
    assert math.isnan(old.change)


def test_a_stand_in_beta_uses_dimson_only_across_the_time_zones() -> None:
    days = pd.bdate_range("2024-01-02", periods=300)
    rng = np.random.default_rng(3)
    kospi = 1_000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    sp = 1_000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    noise = np.exp(np.cumsum(rng.normal(0, 0.005, len(days))))
    stock = 0.05 * noise * kospi
    adjusted = pd.DataFrame(
        {"005930.KS": stock, pf.KOSPI: kospi, pf.BENCHMARK: sp}, index=days
    )

    def expected(index: str, *, lagged: bool) -> float:
        pair = adjusted.reindex(columns=["005930.KS", index])
        rets = pf.returns(pair).dropna().to_numpy(dtype="float64")
        r, m = rets[:, 0], rets[:, 1]
        beta = quant.dimson_beta(r, m) if lagged else quant.welch_beta(r, m, decay=0)
        return quant.shrunk_beta(beta, len(rets))

    on_kospi = pf.proxy_beta(adjusted, "005930.KS", pf.KOSPI)
    on_sp = pf.proxy_beta(adjusted, "005930.KS", pf.BENCHMARK)

    assert on_kospi == pytest.approx(expected(pf.KOSPI, lagged=False))
    assert on_sp == pytest.approx(expected(pf.BENCHMARK, lagged=True))


def test_a_korean_stand_in_takes_kospi_through_the_won() -> None:
    days = pd.bdate_range("2019-11-01", "2020-06-30")
    t = np.arange(len(days), dtype="float64")
    after = days >= pd.Timestamp("2020-03-23")
    adjusted = pd.DataFrame(
        {
            pf.KOSPI: np.where(after, 75.0, 100.0) + np.sin(t) * 1e-3,  # -25% in won
            pf.FX: np.where(after, 1_200.0, 1_000.0),  # the dollar +20%
            pf.BENCHMARK: 1_000.0 + np.sin(t),
        },
        index=days,
    )
    fx = pf.fx_series(adjusted)

    rows = pf.scenario_replays({"NEW.KS": 1_000.0, "NEW": 500.0}, adjusted, "USD", fx)
    covid = next(r for r in rows if r.scenario.name == "코로나")

    assert covid.proxied == ("NEW.KS", "NEW")
    assert pf.stand_in("NEW.KS") == pf.KOSPI
    assert pf.currency(pf.KOSPI) == "KRW"
    assert covid.kospi == pytest.approx(-0.25, abs=1e-4)
    korean_part = (1 - 0.25) / 1.2 - 1  # -37.5% in dollars
    assert covid.change * 1_500 == pytest.approx(
        1_000 * korean_part + 500 * covid.sp500, abs=0.5
    )
    assert covid.pnl == pytest.approx(covid.worst * 1_500)
    assert covid.fx == pytest.approx(
        (1 + covid.change) / (1 + (1_000 * -0.25 + 500 * covid.sp500) / 1_500) - 1,
        abs=1e-3,
    )
    assert pf.KOSPI in pf.wanted(pf.Book())
    assert pf.RF_USD in pf.wanted(pf.Book())
    assert pf.TOTAL_RETURN in pf.wanted(pf.Book())
    assert pf.VIX not in pf.wanted(pf.Book(), {"AAA"})  # never asked for
    assert pf.wanted(pf.Book(), {pf.VIX})[-1] == pf.VIX  # saved: refreshed
    assert pf.market_series(pf.TOTAL_RETURN)
    assert pf.benchmark_of("AAPL") == pf.TOTAL_RETURN
    assert pf.benchmark_of("005930.KS") == pf.KOSPI


def test_a_stand_in_beta_weighs_every_day_alike() -> None:
    # Beta 0.8 for years, then -0.2 for the last year: the full history's
    # slope, not the last year's, and shrunk by all its days.
    days = pd.bdate_range("2017-01-02", periods=2_000)
    m = np.random.default_rng(12).normal(0, 0.01, len(days) - 1)
    r = np.where(np.arange(len(m)) < len(m) - 250, 0.8 * m, -0.2 * m)
    adjusted = pd.DataFrame(
        {
            pf.BENCHMARK: 1_000 * np.cumprod(np.concatenate([[1.0], 1 + m])),
            "AAA": 50 * np.cumprod(np.concatenate([[1.0], 1 + r])),
        },
        index=days,
    )
    slope = cast("float", np.polyfit(m, r, 1)[0])  # plain least squares, about 0.67

    beta = pf.proxy_beta(adjusted, "AAA", pf.BENCHMARK)

    assert beta == pytest.approx(quant.shrunk_beta(slope, len(m)))
    assert 0.6 < beta < 0.75
    assert quant.welch_beta(r, m) < 0.1  # the age decay: about the last year


def test_a_korean_holding_in_a_us_only_window_converts_by_its_own_currency() -> None:
    # 1987 has only the S&P 500: NEW.KS takes its move (beta 1, no data) as
    # a won return. In won nothing converts; in dollars the won does.
    adjusted = long_indices()
    fred = pd.Series(
        np.linspace(700.0, 900.0, 5_000),
        index=pd.bdate_range("1985-01-01", periods=5_000),
    )
    fx = pf.long_fx(adjusted, fred)
    values = {"NEW.KS": 1_000.0}

    won = pf.scenario_replays(values, adjusted, "KRW", fx)[0]
    dollar = pf.scenario_replays(values, adjusted, "USD", fx)[0]

    assert won.scenario.us_only
    assert won.proxied == dollar.proxied == ("NEW.KS",)
    assert won.window is not None
    start, end = (pd.Timestamp(d) for d in won.window)
    sp = adjusted[pf.BENCHMARK].loc[start:end]
    move = float(sp.iloc[-1] / sp.iloc[0] - 1)
    rate = fred.loc[:end]
    first, last = float(rate.loc[:start].iloc[-1]), float(rate.iloc[-1])
    assert won.change == pytest.approx(move)
    assert won.fx == pytest.approx(0.0)
    assert not dollar.no_fx
    assert dollar.change == pytest.approx((1 + move) * first / last - 1)


def stress_prices() -> pd.DataFrame:
    """2025 관세: the S&P 500 falls 20% to Apr 8 and is back to -10% when
    KOSPI bottoms on May 8, the window's end."""
    days = pd.bdate_range("2024-10-01", "2025-12-31")

    def line(*points: tuple[str, float]) -> NDArray[np.float64]:
        at = [cast("float", days.get_indexer([pd.Timestamp(d)])[0]) for d, _ in points]
        return np.interp(
            np.arange(len(days), dtype=np.float64), at, [v for _, v in points]
        )

    sp = line(("2024-10-01", 90), ("2025-02-19", 100), ("2025-04-08", 80),
              ("2025-05-08", 90), ("2025-12-31", 95))  # fmt: skip
    kospi = line(("2024-10-01", 90), ("2025-02-19", 100), ("2025-05-08", 70),
                 ("2025-12-31", 95))  # fmt: skip
    frame_ = pd.DataFrame(
        {pf.BENCHMARK: sp, pf.KOSPI: kospi, pf.FX: 1_300.0}, index=days
    )
    frame_["AAA"] = frame_[pf.BENCHMARK]
    frame_["BBB.KS"] = frame_[pf.KOSPI]
    frame_.loc["2025-02-19":"2025-02-20", "BBB.KS"] = np.nan  # Seoul closed
    frame_["NEW"] = frame_[pf.BENCHMARK].where(days >= pd.Timestamp("2025-02-24"))
    return frame_


def test_the_stress_headline_is_the_fall_from_peak_to_trough() -> None:
    adjusted = stress_prices()
    fx = pf.fx_series(adjusted)

    rows = pf.scenario_replays({"AAA": 1_000.0}, adjusted, "USD", fx)
    tariff = next(r for r in rows if r.scenario.name == "2025 관세")

    assert tariff.window == (date(2025, 2, 19), date(2025, 5, 8))
    assert tariff.change == pytest.approx(-0.10)  # 100 -> 90 at the window's end
    assert tariff.worst == pytest.approx(-0.20)  # 100 -> 80 on Apr 8
    assert tariff.pnl == pytest.approx(-200.0)


def test_a_replay_starts_each_holding_at_its_last_close_not_a_later_one() -> None:
    adjusted = stress_prices()
    fx = pf.fx_series(adjusted)
    kospi = adjusted[pf.KOSPI]

    rows = pf.scenario_replays({"BBB.KS": 1_000.0}, adjusted, "KRW", fx)
    tariff = next(r for r in rows if r.scenario.name == "2025 관세")

    # Seoul is shut on the first day: Feb 18's close, not Feb 21's.
    before = float(kospi.loc["2025-02-18"])
    assert tariff.proxied == ()
    assert tariff.change == pytest.approx(70 / before - 1)
    assert tariff.change < 70 / float(kospi.loc["2025-02-21"]) - 1


def test_a_holding_listed_inside_the_window_takes_its_index_from_the_start() -> None:
    adjusted = stress_prices()
    fx = pf.fx_series(adjusted)

    rows = pf.scenario_replays({"NEW": 1_000.0}, adjusted, "USD", fx)
    tariff = next(r for r in rows if r.scenario.name == "2025 관세")

    # NEW lists on Feb 24, three days in: the S&P 500 (its beta ~1) stands
    # in for the whole window rather than NEW's later first close.
    beta = pf.proxy_beta(adjusted, "NEW", pf.BENCHMARK)
    assert tariff.proxied == ("NEW",)
    assert beta == pytest.approx(1.0)
    assert tariff.change == pytest.approx(-0.10)
    assert tariff.worst == pytest.approx(-0.20)


def test_a_return_after_a_long_holiday_or_a_rate_outage_is_kept() -> None:
    days = pd.bdate_range("2017-09-25", periods=30)
    us = 100 * np.cumprod(np.full(len(days), 1.001))
    kr = pd.Series(np.linspace(2_400.0, 2_500.0, len(days)), index=days)
    shut = (days >= pd.Timestamp("2017-10-02")) & (days <= pd.Timestamp("2017-10-09"))
    kr[shut] = np.nan  # Chuseok: six US days without a Seoul close
    adjusted = pd.DataFrame({"AAA": us, pf.KOSPI: kr, pf.FX: 1_100.0}, index=days)

    rets = pf.asset_returns(adjusted, ["AAA", pf.KOSPI], "KRW")

    back = cast("float", rets.loc["2017-10-10", pf.KOSPI])
    assert back == pytest.approx(kr["2017-10-10"] / kr["2017-09-29"] - 1)
    assert rets.loc["2017-10-02":"2017-10-09", pf.KOSPI].tolist() == [0.0] * 6
    # USD/KRW missing for 24 calendar days: AAA in won keeps every day.
    gap = adjusted.copy()
    gap.loc["2017-09-27":"2017-10-19", pf.FX] = np.nan
    won = pf.asset_returns(gap, ["AAA"], "KRW")["AAA"]
    assert bool(won.notna().all())
    assert len(won) == len(days) - 1


def test_clones_rebalance_with_turnover_and_end_on_priced_weights() -> None:
    days = pd.bdate_range("2026-03-02", periods=12)
    a = [10.0, 20.0, *[20.0] * 10]
    c = [10.0] * 4 + [np.nan] * 8  # delisted: carried as cash, then left out
    prices = pd.DataFrame({"A": a, "B": [10.0] * 12, "C": c}, index=days)
    at = [d.date() for d in days]
    half = pd.Series({"A": 0.5, "B": 0.5})
    schedule = [(at[0], half), (at[2], half), (at[3], pd.Series({"B": 0.5, "C": 0.5}))]

    made = pf.clone(prices, schedule)

    assert [s.names for s in made.steps] == [2, 2, 2]
    assert made.steps[0].turnover is None  # bought from cash
    # A doubled: the drifted mix is 2/3 A, 1/3 B; back to half and half.
    assert made.steps[1].turnover == pytest.approx(1 / 6)
    assert made.steps[2].turnover == pytest.approx(0.5)
    assert made.steps[2].day == at[3]
    assert made.cumulative.equals(pf.rebalanced(prices, schedule))
    assert made.weights.to_dict() == {"B": 1.0}  # C's price stopped
    assert pf.clone(prices, []).cumulative.empty


def test_a_clone_does_not_buy_at_a_close_older_than_a_week() -> None:
    days = pd.bdate_range("2026-03-02", periods=20)  # Monday 2 March
    a = [10.0] * 20
    halted = [10.0] + [np.nan] * 10 + [12.0] * 9  # resumes 16 March
    paused = [10.0] * 3 + [np.nan] * 4 + [10.0] * 13  # off 5-10 March
    prices = pd.DataFrame({"A": a, "H": halted, "P": paused}, index=days)
    at = [d.date() for d in days]
    every = pd.Series({"A": 1.0, "H": 1.0, "P": 1.0})

    bought = pf.clone(prices, [(at[6], every)])  # 10 March

    assert bought.steps[0].names == 2  # H's close is 8 days old, P's 6
    assert bought.weights.to_dict() == {"A": 0.5, "P": 0.5}
    # In-base returns still span a holiday up to a month long.
    assert pf.carry_over(prices)["H"].iloc[6] == 10.0


def test_without_the_total_return_index_comparisons_degrade() -> None:
    full = market()
    prices = pf.Prices(
        full.close.drop(columns=pf.TOTAL_RETURN),
        full.adjusted.drop(columns=pf.TOTAL_RETURN),
    )
    p = folio(trade(date(2023, 1, 3), "buy", 10, 100))
    other = folio(trade(date(2023, 1, 3), "buy", 2, 50_000, ticker="BBB.KS"))
    book = pf.Book(
        portfolios=(p, other.model_copy(update={"id": "p2", "name": "국내"}))
    )

    frame_, flags = pf.lineup(book, prices, "USD", 12)
    found = pf.lagged_correlation(frame_, flags)
    blank = frame_.assign(**{pf.BENCHMARK_LABEL: np.nan})
    kept = pf.lagged_correlation(blank, [*flags, False])

    assert list(frame_.columns) == ["테스트", "국내"]  # no blank S&P 500 column
    assert len(flags) == 2
    assert found is not None
    assert kept is not None  # an all-blank column is left out, not every day
    assert kept.matrix.equals(found.matrix)
    rets = pf.window(pf.held_returns(p, prices.close, prices.adjusted), 12)
    days = pd.DatetimeIndex(rets.index)
    bench = pf.blended(p, prices.adjusted, days)
    stats = pf.ratios(rets, pd.Series(0.0, index=days), bench)
    assert bench.isna().all()  # all US: no benchmark until ^SP500TR is saved
    assert stats.te is None
    assert stats.ir is None
    assert stats.volatility is not None


def test_a_union_calendar_annualises_by_its_own_days_a_year() -> None:
    us = pd.bdate_range("2024-01-01", "2026-12-31")
    seoul = us + pd.Timedelta(days=1)  # Tuesday to Saturday: a different calendar
    union = us.union(seoul)
    rets = pd.Series(np.random.default_rng(8).normal(0, 0.01, len(union)), index=union)
    stats = pf.ratios(rets, pd.Series(0.0, index=union), rets)

    assert pf.per_year(us) == pytest.approx(261, abs=0.5)
    assert pf.per_year(union) == pytest.approx(313, abs=0.5)  # six days a week
    assert pf.per_year(us[:1]) == pf.TRADING_DAYS
    sold_out = us[(us < "2025-03-01") | (us >= "2025-07-01")]  # four months in cash
    assert pf.per_year(sold_out) == pytest.approx(261, abs=0.5)
    assert stats.volatility == pytest.approx(
        rets.std(ddof=1) * math.sqrt(pf.per_year(union))
    )


def test_the_risk_free_rate_lines_up_with_the_returns() -> None:
    days = pd.bdate_range("2025-01-01", periods=300)
    rets = pd.Series(
        np.random.default_rng(6).normal(0.001, 0.01, len(days)), index=days
    )
    rates = pd.Series([3.0, 6.0], index=pd.to_datetime(["2025-01-03", "2025-06-02"]))
    year = pf.per_year(days)  # 299 returns over 419 calendar days: about 260
    assert year == pytest.approx(299 * 365 / 419)

    def daily(rate: float) -> float:  # CD 91-day: simple on 365 days
        return math.pow(1 + rate * 91 / 365, 365 / 91 / year) - 1

    rf = pf.risk_free(rates, pd.DatetimeIndex(days), "simple")
    assert rf.iloc[:2].isna().all()  # before the first rate
    assert rf.loc["2025-01-03"] == pytest.approx(daily(0.03))
    assert rf.loc["2025-05-30"] == pytest.approx(daily(0.03))
    assert rf.loc["2025-06-02"] == pytest.approx(daily(0.06))
    bill = pf.risk_free(rates, pd.DatetimeIndex(days), "discount")  # ^IRX
    price = 1 - 0.06 * 91 / 360
    assert bill.loc["2025-06-02"] == pytest.approx(
        math.pow(price, -365 / 91 / year) - 1
    )
    zero = pf.risk_free(pd.Series(dtype="float64"), pd.DatetimeIndex(days), "simple")
    assert zero.tolist() == [0.0] * len(days)

    stats = pf.ratios(rets, rf, rets * 0.5)
    assert stats.days == len(days) - 2
    kept = rets.iloc[2:]
    excess = kept - rf.iloc[2:]
    assert stats.sharpe == pytest.approx(
        excess.mean() / excess.std(ddof=1) * math.sqrt(pf.per_year(kept.index))
    )
    assert stats.ci is not None
    assert stats.sharpe is not None
    assert stats.ci.low < stats.sharpe < stats.ci.high
    assert stats.calmar is not None  # 300 trading days: over a year
    assert stats.under is not None
    assert stats.under.trough >= stats.under.peak
    short = pf.ratios(rets.head(100), rf, rets)
    assert short.sharpe is not None
    assert short.ci is None
    assert short.psr is None
    assert short.min_trl is None
    assert short.te is None
    assert short.calmar is None  # under a year


def test_the_currency_split_adds_up_to_the_held_return() -> None:
    days = [D1, D2, D3, D4, D5]
    close = frame(
        days,
        AAA=[100.0, 110.0, 110.0, 99.0, 108.9],
        **{"BBB.KS": [1_000.0, 1_000.0, 1_100.0, 1_100.0, 1_210.0]},
        **{pf.FX: [1_000.0, 1_050.0, 1_050.0, 1_100.0, 1_100.0]},
    )
    p = folio(
        trade(D1, "buy", 10, 100),
        trade(D1, "buy", 1, 1_000, ticker="BBB.KS"),
        trade(D3, "buy", 5, 110),
        base="KRW",
    )

    split = pf.currency_split(p, close)
    assert split is not None
    assert split.local + split.fx + split.cross == pytest.approx(split.base)
    held = pf.held_returns(p, close, close)
    chained = pf.standing(held)[0] - 1
    assert split.base == pytest.approx(chained)
    assert split.fx == pytest.approx(0.1 * 10_000 * 0 + split.fx)  # finite
    one = folio(trade(D1, "buy", 10, 100), base="KRW")
    only = pf.currency_split(one, close)
    assert only is not None
    # The dollar +10% and AAA +8.9% in dollars: 1.089 x 1.1.
    assert only.local == pytest.approx(0.089)
    assert only.fx == pytest.approx(0.1)
    assert only.cross == pytest.approx(0.0089)
    assert only.base == pytest.approx(1.089 * 1.1 - 1)
    assert pf.currency_split(folio(trade(D1, "buy", 1, 100)), close) is None


def test_the_blended_benchmark_drifts_with_the_korean_share() -> None:
    days = [D1, D2, D3]
    close = frame(
        days,
        AAA=[100.0, 100.0, 200.0],
        **{"BBB.KS": [100.0, 100.0, 100.0]},
        **{pf.TOTAL_RETURN: [10.0, 11.0, 11.0], pf.KOSPI: [10.0, 10.0, 12.0]},
        **{pf.FX: [1.0, 1.0, 1.0]},
    )
    p = folio(trade(D1, "buy", 1, 100), trade(D1, "buy", 1, 100, ticker="BBB.KS"))
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in days])

    mix = pf.blended(p, close, index)
    # Day 2: half Korean at day 1's close; day 3: a third Korean at day 2's.
    assert mix.iloc[1] == pytest.approx(0.5 * 0.0 + 0.5 * 0.1)
    assert mix.iloc[2] == pytest.approx(0.5 * 0.2 + 0.5 * 0.0)
    lines = pf.benchmark_lines(p, close, index)
    assert lines["혼합 벤치마크"].iloc[0] == 0
    assert lines["KOSPI"].iloc[-1] == pytest.approx(0.2)


def test_the_blended_benchmark_takes_one_day_moves_across_a_gap() -> None:
    days = [D1, D2, D3, D4, D5]
    close = frame(
        days,
        AAA=[100.0, 100.0, 100.0, 100.0, 110.0],
        **{pf.TOTAL_RETURN: [10.0, 11.0, 12.0, 13.0, 14.0], pf.KOSPI: [10.0] * 5},
        **{pf.FX: [1.0] * 5},
    )
    p = folio(
        trade(D1, "buy", 1, 100), trade(D2, "sell", 1, 100), trade(D4, "buy", 1, 100)
    )
    rets = pf.held_returns(p, close, close)
    assert list(rets.index) == [pd.Timestamp(D2), pd.Timestamp(D5)]  # sold out D3-4

    mix = pf.blended(p, close, pd.DatetimeIndex(rets.index))
    assert mix.iloc[-1] == pytest.approx(14 / 13 - 1)  # not 14 / 11 - 1


def what_if_prices() -> tuple[pd.DataFrame, pd.Series]:
    """Two stocks, a copy of their 60/40 mix, and a flat line (cash)."""
    days = pd.bdate_range("2023-01-02", periods=600)
    r = np.random.default_rng(3).normal(0, 0.01, size=(len(days), 2))
    r[0] = 0.0
    growth = np.cumprod(1 + r, axis=0)
    mix = np.cumprod(1 + r @ np.array([0.6, 0.4]))
    adjusted = pd.DataFrame(
        {
            "AAA": 100 * growth[:, 0],
            "BBB": 50 * growth[:, 1],
            "COPY": 10 * mix,
            "FLAT": 1.0,
            pf.BENCHMARK: 100 * growth[:, 0],
        },
        index=days,
    )
    return adjusted, pd.Series({"AAA": 0.6, "BBB": 0.4})


def test_what_if_cash_scales_risk_and_a_copy_changes_nothing() -> None:
    adjusted, w = what_if_prices()
    j = pf.joint(
        adjusted, pf.with_candidate(pf.with_candidate(w, "FLAT"), "COPY"), "USD", 36
    )
    x = j.window.to_numpy(dtype="float64")
    cov = np.cov(x.T, bias=True)  # cash has no variance: no Ledoit-Wolf target
    assert j.w.tolist() == pytest.approx([0.6, 0.4, 0.0, 0.0])
    assert len(x) >= quant.min_samples(quant.TAIL_LEVEL)

    cash = pf.added(j, cov, "FLAT", 0.1)
    assert cash.vol[1] == pytest.approx(0.9 * cash.vol[0])
    assert cash.beta[1] == pytest.approx(0.9 * cash.beta[0])
    assert cash.es is not None
    assert cash.es[1] == pytest.approx(0.9 * cash.es[0])
    assert cash.es_share == pytest.approx(0.0, abs=1e-12)

    copy = pf.added(j, cov, "COPY", 0.1)
    assert copy.vol[1] == pytest.approx(copy.vol[0])
    assert copy.es is not None
    assert copy.es[1] - copy.es[0] == pytest.approx(0.0, abs=1e-12)
    assert copy.correlation == pytest.approx(1.0)
    assert copy.threshold == pytest.approx(1.0)
    assert copy.es_share == pytest.approx(0.1)

    held = pf.mixed_in(j, "AAA", 0.1)  # already held: on top of its scaled weight
    assert held.tolist() == pytest.approx([0.9 * 0.6 + 0.1, 0.9 * 0.4, 0.0, 0.0])


def test_what_if_paths_reuse_the_portfolios_own_draws() -> None:
    adjusted, w = what_if_prices()
    own = pf.joint(adjusted, w, "USD", 36)
    j = pf.joint(adjusted, pf.with_candidate(w, "FLAT"), "USD", 36)
    block = pf.mean_block(j)
    assert len(j.window) == len(own.window)
    assert block == pf.mean_block(own)

    picks = quant.stationary_indices(len(j.window), 300, 21, block)
    assert np.array_equal(
        picks, quant.stationary_indices(len(own.window), 300, 21, block)
    )
    before = pf.growth(j, picks)
    after = pf.growth(j, picks, w=pf.mixed_in(j, "FLAT", 0.1))

    assert np.allclose(before, pf.growth(own, picks))
    # Same days drawn: each simulated day's move is 90% of the one before.
    step = before[:, 1:] / before[:, :-1] - 1
    assert np.allclose(after[:, 1:] / after[:, :-1] - 1, 0.9 * step)


def test_the_do_nothing_book_holds_the_first_mix() -> None:
    days = [date(2026, 1, 2), date(2026, 2, 2), date(2026, 3, 2), date(2026, 4, 1)]
    close = frame(
        days, AAA=[100.0, 100.0, 80.0, 80.0], BBB=[100.0, 100.0, 100.0, 120.0]
    )
    p = folio(
        trade(days[0], "buy", 10, 100),
        trade(days[1], "sell", 10, 100),
        trade(days[1], "buy", 10, 100, ticker="BBB"),
    )

    result = pf.trading(p, close)
    assert result is not None
    assert result.actual == pytest.approx(1_200)  # 10 BBB
    assert result.held == pytest.approx(800)  # still 10 AAA
    assert result.effect is not None
    assert not result.effect.annualised
    assert result.effect.value > 0


def test_correlation_beta_and_spread_on_a_mixed_book() -> None:
    prices = market()
    adjusted = prices.adjusted.assign(**{pf.KOSPI: prices.adjusted["BBB.KS"] / 50})
    w = pd.Series({"AAA": 0.6, "BBB.KS": 0.4})

    j = pf.joint(adjusted, w, "KRW", 36)
    assert j.mixed
    assert j.korean.tolist() == [False, True]
    cov, f, _ = pf.covariance(j)
    assert 0 <= f <= 1
    assert np.allclose(cov, cov.T)
    assert pf.correlation_frame(cov, j.names).loc["AAA", "AAA"] == pytest.approx(1.0)
    rows = {b.name: b for b in pf.betas(j)}
    assert math.isnan(rows["AAA"].dimson)
    assert rows["BBB.KS"].kospi == pytest.approx(1.0, abs=0.05)
    assert math.isfinite(rows[pf.PORTFOLIO_LABEL].welch)
    spread = pf.diversification(j, cov)
    assert spread.risk.sum() == pytest.approx(1.0)
    assert spread.effective == pytest.approx(1 / (0.6**2 + 0.4**2))


def test_simulation_and_tails_use_the_new_draws() -> None:
    prices = market()
    w = pd.Series({"AAA": 0.5, "BBB.KS": 0.5})
    j = pf.joint(prices.adjusted, w, "USD", 36)
    picks = quant.stationary_indices(len(j.window), 500, 63, pf.mean_block(j))

    f = pf.volatility(j)
    plain = pf.growth(j, picks)
    filtered = pf.growth(j, quant.stationary_indices(len(j.window), 500, 63, 1), f)
    assert plain.shape == filtered.shape == (500, 64)
    assert not np.array_equal(plain, filtered)
    sim = pf.simulation(plain, 100.0, 0.9, 1.0)
    falls = sim.paths.falls
    assert 0 <= falls[0.2].end <= falls[0.1].end <= falls[0.1].swing <= 1
    assert falls[0.3].swing <= falls[0.1].swing
    assert len(sim.fan) == 64

    t = pf.tails(j, f)
    assert t.days == len(j.window)
    assert pf.tail_of(t, "역사적", 0.99).var is None  # under 1,200 days
    assert pf.tail_of(t, "EVT", 0.99).var is not None  # 75 excesses of 750
    head = pf.tail_of(t, "역사적", 0.975)
    assert head.es is not None
    for method in ("역사적", "FHS"):  # bootstrap intervals, like EVT's
        row = pf.tail_of(t, method, 0.975)
        assert row.var_ci is not None
        assert row.es_ci is not None
        assert row.var_ci[0] <= cast("float", row.var) <= row.var_ci[1]
        assert row.es_ci[0] <= cast("float", row.es) <= row.es_ci[1]
        assert pf.tail_of(t, method, 0.99).var_ci is None  # no estimate, no interval
    assert set(t.backtests) == {"역사적", "FHS", "EVT"}  # 756 days: 75 excesses
    assert t.lookback == len(j.window)  # twenty years: the window's own length
    held = j.full.to_numpy(dtype="float64") @ j.w
    past = quant.past_windows(held, quant.BACKTEST_DAYS, len(j.window))
    assert past is not None
    assert t.backtests["역사적"] == quant.backtest(held[-250:], past)
    filtered = pf.refit(j, pf.Vol(), len(j.window))
    assert filtered is not None
    evt = quant.evt_windows(filtered)
    assert evt is not None
    assert t.backtests["EVT"] == quant.backtest(held[-250:], evt)
    assert t.component is not None
    assert t.component.sum() == pytest.approx(head.es)
    six = pf.joint(prices.adjusted, w, "USD", 6)
    short = pf.tails(six, pf.volatility(six))
    assert all(row.var is None for row in short.rows)
    assert short.lookback == 250
    assert set(short.backtests) == {"역사적", "FHS", "EVT"}  # a row each, as above
    assert short.backtests["FHS"] is not None
    assert short.backtests["EVT"] is None  # 25 excesses of 250


def test_backtests_look_back_as_far_as_the_window_else_a_year() -> None:
    assert pf.backtest_lookback(756, 2000) == 756
    assert pf.backtest_lookback(756, 1005) == 250  # 756 days before the tested 250
    assert pf.backtest_lookback(252, 502) == 252
    assert pf.backtest_lookback(126, 2000) == 250  # under 240 days: no 95% VaR
    assert pf.backtest_lookback(126, 499) is None


def test_the_resampled_density_check_passes_iid_returns_out_of_sample() -> None:
    j = pf.joint(market().adjusted, pd.Series({"AAA": 0.5, "BBB.KS": 0.5}), "USD", 12)
    _, starts = quant.blocks(j.full.to_numpy(dtype="float64") @ j.w)

    d = pf.density(j)

    assert d.blocks == int((starts >= len(j.window)).sum())  # a window before each
    assert d.test is not None
    assert d.test.p > 0.01


def test_the_filter_label_names_the_ewma_holdings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    j = pf.joint(market().adjusted, pd.Series({"AAA": 0.5, "BBB.KS": 0.5}), "USD", 36)
    monkeypatch.setitem(sys.modules, "arch", None)  # without the garch extra

    f = pf.volatility(j)
    assert pf.filter_label(j, f) == "EWMA(λ=0.94)"

    calls = iter(range(100))

    def second_fails(_: object) -> NDArray[np.float64] | None:
        return None if next(calls) == 1 else np.array([2e-6, 0.03, 0.1, 0.9])

    monkeypatch.setattr(quant, "gjr_fit", second_fails)
    f = pf.volatility(j)
    assert pf.filter_label(j, f) == "GJR-GARCH(1,1), EWMA(λ=0.94): BBB.KS"


def ranged(prices: pf.Prices, *names: str) -> pf.Prices:
    """``prices`` with made-up opens, highs and lows for ``names``."""
    close = prices.close[list(names)]
    return replace(prices, open=close * 0.995, high=close * 1.01, low=close * 0.985)


def test_the_range_proxy_is_garman_klass_in_base_and_nan_without_ranges() -> None:
    prices = ranged(market("2026-01-05", "2026-01-16"), "AAA")
    names = ["AAA", "BBB.KS"]
    index = pf.trading_days(prices.adjusted, names)[1:]

    proxy = pf.range_proxy(prices, names, "KRW", index)

    assert pf.no_ranges(prices, names) == ["BBB.KS"]
    assert proxy is not None
    assert proxy.shape == (len(index), 2)
    assert np.isnan(proxy[:, 1]).all()  # no ranges: the filter takes r^2
    fx = prices.adjusted[pf.FX].to_numpy()
    factor = (prices.adjusted["AAA"] / prices.close["AAA"]).to_numpy() * fx
    won = prices.adjusted["AAA"].to_numpy() * fx
    want = quant.garman_klass(
        prices.open["AAA"].to_numpy() * factor,
        prices.high["AAA"].to_numpy() * factor,
        prices.low["AAA"].to_numpy() * factor,
        won,
        np.roll(won, 1),
    )[1:]
    np.testing.assert_allclose(proxy[:, 0], want, rtol=1e-12)
    assert (
        pf.range_proxy(market("2026-01-05", "2026-01-16"), names, "KRW", index) is None
    )


def test_vix_is_read_on_each_holdings_own_day() -> None:
    days = pd.bdate_range("2026-01-05", "2026-01-09")  # Mon-Fri
    adjusted = pd.DataFrame({pf.VIX: [10.0, 20.0, np.nan, 40.0, 50.0]}, index=days)
    index = days[1:]

    x = pf.vix_regressor(adjusted, ["AAA", "BBB.KS"], index)

    assert x is not None
    us = [20.0, 20.0, 40.0, 50.0]  # Wednesday has none: Tuesday's
    seoul = [10.0, 20.0, 20.0, 40.0]  # the US close before the day
    np.testing.assert_allclose(x, quant.vix_variance(np.column_stack([us, seoul])))
    late = pd.DatetimeIndex([*index, pd.Timestamp("2026-01-20")])
    assert pf.vix_regressor(adjusted, ["AAA"], late) is None  # stale
    assert pf.vix_regressor(adjusted, ["AAA"], days.shift(-1)) is None  # before it
    assert pf.vix_regressor(adjusted.drop(columns=pf.VIX), ["AAA"], index) is None


def test_the_model_options_drive_the_filter_tails_and_checks() -> None:
    prices = market("2022-01-03")
    j = pf.joint(prices.adjusted, pd.Series({"AAA": 0.5, "BBB.KS": 0.5}), "USD", 36)
    vix = prices.adjusted[pf.BENCHMARK] / 50
    x = pf.vix_regressor(prices.adjusted.assign(**{pf.VIX: vix}), j.names, j.full.index)
    proxy = pf.range_proxy(ranged(prices, "AAA"), j.names, "USD", j.full.index)
    v = pf.Vol("har", proxy, x)

    plain = pf.volatility(j)
    assert np.array_equal(plain.sigma, pf.volatility(j, pf.Vol()).sigma)
    f = pf.volatility(j, v)
    assert f.model is not None
    assert pf.filter_label(j, f) == "HAR"
    assert not np.array_equal(f.sigma, pf.volatility(j, pf.Vol("har")).sigma)
    t = pf.tails(j, f, v)
    assert set(t.backtests) == {"역사적", "FHS", "EVT"}


def test_shocks_reverse_stress_and_bad_prints() -> None:
    values = pd.Series({"AAA": 600.0, "BBB.KS": 400.0})
    betas = np.array([1.0, 1.0])

    rows = {s.name: v for s, v in pf.shocked(values, betas, "KRW")}
    assert rows["한국발 위기"] == pytest.approx(0.6 * 0.15 + 0.4 * -0.30)
    assert rows["미국발 위기"] == pytest.approx(0.6 * (0.75 * 1.10 - 1))
    assert rows["환율 −15%"] == pytest.approx(0.6 * -0.15)
    usd = {s.name: v for s, v in pf.shocked(values, betas, "USD")}
    assert usd["환율 +15%"] == pytest.approx(0.4 * (1 / 1.15 - 1))
    assert pf.reverse(values, betas) == pytest.approx(-0.30)
    assert pf.reverse(values, np.array([0.1, 0.1])) is None

    days = pd.bdate_range("2025-01-01", periods=10)
    spiky = pd.DataFrame({"AAA": [10.0] * 4 + [100.0] + [10.0] * 5}, index=days)
    assert pf.bad_print_days(spiky, ["AAA", "ZZZ"]) == {"AAA": [date(2025, 1, 7)]}

    prices = market()
    betas_ = pf.local_betas(prices.adjusted, ["AAA", "NEW.KS"], 12)
    assert betas_[1] == 1.0  # no data: 1
    assert 0 < betas_[0] < 1  # AAA moves apart from the index: pulled toward 1
    pair = cast("pd.DataFrame", prices.adjusted[["AAA", pf.BENCHMARK]])
    rets = pf.window(pf.returns(pair), 12)
    raw = quant.welch_beta(rets["AAA"].to_numpy(), rets[pf.BENCHMARK].to_numpy())
    n = quant.effective_days(len(rets))  # about 193 of 252 days
    assert betas_[0] == pytest.approx(quant.shrunk_beta(raw, n))
    held = pf.local_betas(prices.adjusted, ["AAA", pf.BENCHMARK, pf.KOSPI], 12)
    assert held.tolist() == [betas_[0], 1.0, 1.0]  # an index held as itself


def test_turbulence_and_style_on_weekly_returns() -> None:
    prices = market()
    adjusted = prices.adjusted.assign(SPY=prices.adjusted[pf.BENCHMARK])
    names = ["AAA", "BBB.KS"]

    now = pf.turbulence(adjusted, names, "USD")
    assert now is not None
    assert 0 <= now.rank <= 1
    assert now.weeks > pf.WEEKS
    assert pf.turbulence(adjusted, ["AAA"], "USD") is None

    w = pd.Series({"AAA": 0.5, pf.BENCHMARK: 0.5})
    fit = pf.style(adjusted, w, "USD")
    assert fit is not None
    assert fit.names == ("SPY", pf.CASH)
    assert fit.weights.sum() == pytest.approx(1.0)
    assert fit.weeks == pf.WEEKS
    assert "GLD" in fit.dropped
    assert fit.r2 is not None
    assert 0 < fit.r2 <= 1


# ---------------------------------------------------------------- storage


def test_the_book_round_trips(tmp_path: Path) -> None:
    root = tmp_path / pf.DATA_DIR
    book, pid = pf.add_portfolio(pf.Book(), "연금", "KRW")
    first = trade(D1, "buy", 1.5, 70_000, ticker="005930.KS")
    book = pf.add_trade(book, pid, first.model_copy(update={"memo": "실적 발표 전"}))

    assert pf.load_book(root) == pf.Book()
    pf.save_book(root, book)

    assert pf.load_book(root) == book
    assert pf.load_book(root).portfolios[0].trades[0].memo == "실적 발표 전"
    saved = (root / pf.BOOK_FILE).read_text("utf-8")
    assert f'"version": {pf.SCHEMA_VERSION}' in saved
    assert f'"day": "{D1.isoformat()}"' in saved
    assert not list(root.glob("*.tmp"))
    _ = (root / pf.BOOK_FILE).write_text("{", encoding="utf-8")
    with pytest.raises(pf.BookError):
        _ = pf.load_book(root)


def test_prices_merge_and_round_trip(tmp_path: Path) -> None:
    old = pf.Prices(
        frame([D1], AAA=[1.0], BBB=[2.0]), frame([D1], AAA=[1.0], BBB=[2.0])
    )
    new = pf.Prices(frame([D2], AAA=[3.0]), frame([D2], AAA=[3.0]))
    merged = pf.merge(old, new)

    assert merged.close["AAA"].dropna().tolist() == [3.0]
    assert merged.close["BBB"].dropna().tolist() == [2.0]
    pf.save_prices(tmp_path, merged)
    assert pf.load_prices(tmp_path).tickers == {"AAA", "BBB"}
    assert pf.load_prices(tmp_path / "missing").tickers == frozenset()


def test_index_history_back_to_1987_survives_a_refresh(tmp_path: Path) -> None:
    old_days = pd.bdate_range("1987-01-02", periods=3)
    new_days = pd.bdate_range("2007-01-02", periods=3)
    deep = pd.DataFrame(
        {pf.BENCHMARK: [1.0, 2.0, 3.0], "AAA": [1.0, 1.0, 1.0]}, index=old_days
    )
    fresh = pd.DataFrame(
        {pf.BENCHMARK: [4.0, 5.0, 6.0], "AAA": [2.0, 2.0, 2.0]}, index=new_days
    )

    merged = pf.merge(pf.Prices(deep, deep), pf.Prices(fresh, fresh))

    assert merged.adjusted[pf.BENCHMARK].dropna().tolist() == [1, 2, 3, 4, 5, 6]
    assert merged.adjusted["AAA"].dropna().tolist() == [2, 2, 2]  # holdings: replaced
    assert not pf.shallow(merged)
    assert pf.shallow(pf.Prices(fresh, fresh))

    starts: list[date] = []

    class Deep:
        def history(self, tickers: Sequence[str], start: date) -> pf.Prices:
            starts.append(start)
            frame_ = deep.reindex(columns=list(tickers)).dropna(axis=1, how="all")
            return pf.Prices(frame_, frame_)

    fetcher = pf.Fetcher(tmp_path, source=lambda _: Deep())
    assert fetcher.deepen(pf.Prices(fresh, fresh))
    pause = threading.Event()
    for _ in range(100):
        if not fetcher.progress.running:
            break
        _ = pause.wait(0.05)
    assert starts == [pf.LONG_START]
    assert not fetcher.deepen(pf.Prices(fresh, fresh))  # once per session


class FakeSource:
    def history(self, tickers: Sequence[str], start: date) -> pf.Prices:
        assert start == pf.HISTORY_START
        known = [t for t in tickers if t != "ZZZ"]
        values = {t: [1.0, 2.0] for t in known}
        return pf.Prices(frame([D1, D2], **values), frame([D1, D2], **values))


def test_the_fetcher_saves_prices_and_does_not_retry_on_its_own(
    tmp_path: Path,
) -> None:
    def fake(_: Path) -> pf.Source:
        return FakeSource()

    fetcher = pf.Fetcher(tmp_path, source=fake)
    wanted = ["AAA", "ZZZ"]

    assert fetcher.untried(wanted, set()) == wanted
    assert fetcher.start(fetcher.untried(wanted, set()))
    pause = threading.Event()
    for _ in range(100):
        if not fetcher.progress.running:
            break
        _ = pause.wait(0.05)

    assert fetcher.progress.failed
    assert "ZZZ" in fetcher.progress.message
    assert pf.load_prices(tmp_path).tickers == {"AAA"}
    assert fetcher.untried(wanted, {"AAA"}) == []
    assert not fetcher.start([])


def test_ranges_are_saved_only_once_asked_then_refreshed(tmp_path: Path) -> None:
    level = iter([1.0, 2.0, 3.0, 4.0])

    class Ranged:
        def history(self, tickers: Sequence[str], start: date) -> pf.Prices:
            del start
            values = {t: [next(level)] * 2 for t in tickers}
            return ranged(
                pf.Prices(frame([D1, D2], **values), frame([D1, D2], **values)),
                *tickers,
            )

    fetcher = pf.Fetcher(tmp_path, source=lambda _: Ranged())
    pause = threading.Event()

    def fetch(*, ranges: bool, ticker: str = "AAA") -> pf.Prices:
        assert fetcher.start([ticker], ranges=ranges)
        for _ in range(100):
            if not fetcher.progress.running:
                break
            _ = pause.wait(0.05)
        return pf.load_prices(tmp_path)

    assert fetch(ranges=False).high.empty
    assert not any((tmp_path / name).exists() for name in pf.RANGE_FILES)
    assert fetch(ranges=True).high["AAA"].tolist() == pytest.approx([2.02, 2.02])
    assert fetch(ranges=False).high["AAA"].tolist() == pytest.approx([3.03, 3.03])
    saved = fetch(ranges=False, ticker="BBB")  # none asked for BBB
    assert saved.high.columns.tolist() == ["AAA"]
    assert isinstance(saved.high.index, pd.DatetimeIndex)
    assert "BBB" in saved.tickers


# ---------------------------------------------------------------- page


def page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The portfolio page in demo mode, with files under ``tmp_path``."""
    settings = tmp_path / "finagent.settings.json"
    monkeypatch.setattr(sys, "argv", ["app.py", "--demo", f"--settings={settings}"])
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at._page_hash = calc_hash("portfolio")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    return at.run()


def test_the_page_with_a_portfolio(  # noqa: PLR0915 - one walk through every section
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / pf.DATA_DIR
    book, pid = pf.add_portfolio(pf.Book(), "연금", "KRW")
    book = pf.add_trade(book, pid, trade(date(2025, 1, 6), "buy", 10, 100))
    book = pf.add_trade(book, pid, trade(date(2026, 2, 2), "sell", 4, 120))
    pf.save_book(root, book)
    prices = market()
    extra = {
        pf.KOSPI: prices.adjusted["BBB.KS"] / 50,
        "SPY": prices.adjusted[pf.BENCHMARK],
    }
    pf.save_prices(
        root, pf.Prices(prices.close.assign(**extra), prices.adjusted.assign(**extra))
    )
    at = page(tmp_path, monkeypatch)

    assert not at.exception
    assert at.dataframe  # holdings and the trade log
    charts = [str(cast("object", c.proto)) for c in at.get("vega_lite_chart")]
    assert any("innerRadius" in chart for chart in charts)  # 현황's weight donut

    _ = at.button(key="pf_trade_open").click().run()
    _ = at.date_input(key="pf_day").set_value(date(2025, 3, 4))
    _ = at.text_input(key="pf_ticker").set_value("bbb.ks")
    _ = at.number_input(key="pf_quantity").set_value(3)
    _ = at.number_input(key="pf_price").set_value(52_000)
    _ = at.text_input(key="pf_memo").set_value("분할 매수 1차")
    _ = at.button(key="pf_buy").click().run()

    assert not at.exception
    trades = pf.load_book(root).portfolios[0].trades
    assert [(t.ticker, t.side, t.memo) for t in trades][-1] == (
        "BBB.KS",
        "buy",
        "분할 매수 1차",
    )
    assert at.text_input(key="pf_memo").value == ""
    log = next(d.value for d in at.dataframe if "메모" in d.value.columns)
    assert "분할 매수 1차" in list(log["메모"])

    def sell(quantity: float) -> None:
        _ = at.date_input(key="pf_day").set_value(date(2025, 3, 5))
        _ = at.text_input(key="pf_ticker").set_value("BBB.KS")
        _ = at.number_input(key="pf_quantity").set_value(quantity)
        _ = at.number_input(key="pf_price").set_value(53_000)
        _ = at.button(key="pf_sell").click().run()

    sell(10)  # more than the 3 held
    assert any("보유 3주보다" in e.value for e in at.error)
    sell(1)
    assert not at.exception
    trades = pf.load_book(root).portfolios[0].trades
    assert [(t.ticker, t.side, t.quantity) for t in trades][-1] == ("BBB.KS", "sell", 1)

    for section in ("returns", "correlation", "outlook"):
        _ = at.segmented_control(key="pf_section").set_value(section).run()
        assert not at.exception, section
        assert at.get("vega_lite_chart"), section
    # The opt-in extras: advanced ratios, style analysis, weekly turbulence.
    _ = at.segmented_control(key="pf_section").set_value("analysis").run()
    at.session_state["pf_advanced"] = True
    at.session_state["pf_style"] = True
    _ = at.run()
    assert not at.exception
    _ = at.segmented_control(key="pf_section").set_value("stress").run()
    at.session_state["pf_turbulence"] = True
    _ = at.run()
    assert not at.exception
    assert any(
        "1997 IMF" in list(d.value["시나리오"])
        for d in at.dataframe
        if "시나리오" in d.value.columns
    )

    # Remove the newest trade (the first row), standing in for a row click.
    key = f"pf_trades_{at.session_state.get('pf_trades_gen', 0)}"
    at.session_state[key] = {"selection": {"rows": [0], "columns": [], "cells": []}}
    _ = at.button(key="pf_remove").click().run()
    assert not at.exception
    assert len(pf.load_book(root).portfolios[0].trades) == 3  # four, less the newest

    _ = at.button(key="pf_new").click().run()
    _ = at.text_input(key="pf_new_name").set_value("국내")
    _ = at.button(key="pf_new_save").click().run()
    assert not at.exception
    assert [p.name for p in pf.load_book(root).portfolios] == ["연금", "국내"]

    _ = at.button(key="pf_delete").click().run()
    _ = at.button(key="pf_delete_ok").click().run()
    assert not at.exception
    assert len(pf.load_book(root).portfolios) == 1


def prefill_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """A portfolio with saved prices; fetching would raise."""

    def no_network(*_: object) -> None:
        raise AssertionError("fetched in demo mode")

    monkeypatch.setattr(pf.Yahoo, "__init__", no_network)
    root = tmp_path / pf.DATA_DIR
    book, _ = pf.add_portfolio(pf.Book(), "연금", "USD")
    pf.save_book(root, book)
    pf.save_prices(root, market("2026-01-02"))
    return page(tmp_path, monkeypatch)


def test_the_price_is_prefilled_from_saved_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = prefill_page(tmp_path, monkeypatch)
    closes = cast("pd.Series", market("2026-01-02").close["AAA"])

    def close(day: str) -> float:
        return round(float(cast("float", closes[day])), 2)

    # Demo mode never fetches a ticker without saved prices (it would raise).
    _ = at.button(key="pf_trade_open").click().run()
    _ = at.text_input(key="pf_ticker").set_value("ZZZ").run()
    assert not at.exception
    assert at.number_input(key="pf_price").value is None

    _ = at.date_input(key="pf_day").set_value(date(2026, 10, 21))
    _ = at.text_input(key="pf_ticker").set_value("aaa").run()
    assert at.number_input(key="pf_price").value == close("2026-10-21")

    # A Saturday falls back to Friday's close.
    _ = at.date_input(key="pf_day").set_value(date(2026, 10, 17)).run()
    assert at.number_input(key="pf_price").value == close("2026-10-16")

    # A typed price stays through unrelated reruns.
    _ = at.number_input(key="pf_price").set_value(123.0).run()
    _ = at.number_input(key="pf_quantity").set_value(2.0).run()
    _ = at.run()
    assert at.number_input(key="pf_price").value == 123.0
    assert at.number_input(key="pf_quantity").value == 2.0


def test_a_file_without_memos_loads(tmp_path: Path) -> None:
    old = (
        '{"version": 1, "portfolios": [{"id": "p", "name": "연금", "base": "KRW",'
        ' "trades": [{"id": "t", "day": "2026-03-02", "ticker": "AAA",'
        ' "side": "buy", "quantity": 1, "price": 10, "fee": 0}]}]}'
    )
    tmp_path.mkdir(exist_ok=True)
    _ = (tmp_path / pf.BOOK_FILE).write_text(old, encoding="utf-8")

    (loaded,) = pf.load_book(tmp_path).portfolios[0].trades
    assert loaded.memo == ""


def stock_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[int]
) -> tuple[AppTest, str]:
    """종목 분석 over a KRW book holding AAA and BBB.KS; ``calls`` counts paths."""
    real = pf.growth

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(pf, "growth", counted)
    root = tmp_path / pf.DATA_DIR
    book, pid = pf.add_portfolio(pf.Book(), "연금", "KRW")
    for item in (
        trade(date(2021, 1, 4), "buy", 10, 100),
        trade(date(2021, 1, 4), "buy", 2, 50_000, ticker="BBB.KS"),
    ):
        book = pf.add_trade(book, pid, item)
    pf.save_book(root, book)
    prices = market()
    kospi = prices.adjusted["BBB.KS"] / 50
    pf.save_prices(
        root,
        pf.Prices(
            prices.close.assign(**{pf.KOSPI: kospi}),
            prices.adjusted.assign(**{pf.KOSPI: kospi}),
        ),
    )
    at = page(tmp_path, monkeypatch)
    _ = at.segmented_control(key="pf_mode").set_value("stock").run()
    assert not at.exception
    assert "stock_section" not in [w.key for w in at.get("button_group")]
    return at, pid


def test_stock_mode_analyses_one_holding_and_reuses_the_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    at, pid = stock_page(tmp_path, monkeypatch, calls)

    _ = at.pills(key="stock_held").set_value("BBB.KS").run()
    assert not at.exception
    assert "stock_base" not in [w.key for w in at.get("button_group")]
    betas = next(d.value for d in at.dataframe if "Dimson" in d.value.columns)
    assert len(betas) == 1  # the stock alone, no 구분 column
    assert "구분" not in betas.columns

    _ = at.segmented_control(key="stock_section").set_value("tail").run()
    assert not at.exception
    assert not any(m.value.startswith("**종목별 ES") for m in at.markdown)
    _ = at.segmented_control(key="stock_section").set_value("stress").run()
    assert not at.exception
    assert any("이 종목 1주" in d.value.columns for d in at.dataframe)  # not the book
    assert "주간 난기류" not in [e.label for e in at.expander]

    # What if the stock were added to a portfolio: the draws are reused.
    _ = at.text_input(key="stock_input").set_value("aaa")
    _ = at.button(key="stock_go").click().run()
    assert at.session_state["stock_ticker"] == "AAA"
    _ = at.segmented_control(key="stock_section").set_value("whatif").run()
    assert not at.exception
    assert at.selectbox(key="whatif_pf").value == pid
    assert at.segmented_control(key="whatif_horizon").value == "1M"
    assert any("시나리오" in d.value.columns for d in at.dataframe)
    drawn = len(calls)
    _ = at.slider(key="whatif_percent").set_value(20).run()
    assert not at.exception
    assert len(calls) == drawn + 1  # only the new mix; the draws and "before" stay

    _ = at.text_input(key="stock_input").set_value("ZZZ")
    _ = at.button(key="stock_go").click().run()
    assert not at.exception
    assert any("저장된 가격이 없습니다" in i.value for i in at.info)
    # Saved prices are read once per file stamp: new files are picked up.
    root = tmp_path / pf.DATA_DIR
    saved = pf.load_prices(root)
    zzz = saved.adjusted["AAA"] * 2
    pf.save_prices(
        root,
        pf.Prices(saved.close.assign(ZZZ=zzz), saved.adjusted.assign(ZZZ=zzz)),
    )
    _ = at.run()
    assert not at.exception
    assert not any("저장된 가격이 없습니다" in i.value for i in at.info)


def risk_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prices: pf.Prices
) -> AppTest:
    """꼬리위험 of a USD book holding AAA and BBB.KS."""
    root = tmp_path / pf.DATA_DIR
    book, pid = pf.add_portfolio(pf.Book(), "연금", "USD")
    for item in (
        trade(date(2023, 1, 3), "buy", 10, 100),
        trade(date(2023, 1, 3), "buy", 2, 50_000, ticker="BBB.KS"),
    ):
        book = pf.add_trade(book, pid, item)
    pf.save_book(root, book)
    pf.save_prices(root, prices)
    at = page(tmp_path, monkeypatch)
    _ = at.segmented_control(key="pf_section").set_value("tail").run()
    _ = at.segmented_control(key="pf_window").set_value("1Y").run()
    return at


def var_help(at: AppTest) -> str:
    return next(m.proto.help for m in at.markdown if m.value.startswith("**VaR·ES**"))


def tables(at: AppTest) -> list[pd.DataFrame]:
    return [d.value for d in at.dataframe]


PF_PATH_KEYS = ("pf_gen", "pf_fhs", "pf_vol_model")


def path_controls(at: AppTest) -> list[str]:
    """전망's path controls shown, in page order."""
    keys = [cast("str | None", getattr(node, "key", None)) for node in at.main]
    return [k for k in keys if k in PF_PATH_KEYS]


def test_the_risk_options_render_and_the_risk_is_cached_until_inputs_change(  # noqa: PLR0915 - one walk through both tabs
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = pf.growth

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(pf, "growth", counted)
    labels: list[int] = []  # one per 변동성 필터 the page needs
    label = pf.filter_label

    def labelled(*args: object, **kwargs: object) -> str:
        labels.append(1)
        return label(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(pf, "filter_label", labelled)
    at = risk_page(tmp_path, monkeypatch, market("2022-01-03"))

    assert not at.exception
    assert "변동성 필터 GJR-GARCH(1,1)" in var_help(at)
    assert "pf_horizon" not in [w.key for w in at.get("button_group")]
    keys = {t.key for t in at.toggle}
    assert not {"pf_ranges", "pf_vix"} & keys
    first = tables(at)

    _ = at.segmented_control(key="pf_vol_model").set_value("har").run()
    _ = at.toggle(key="pf_ranges").set_value(True).run()
    _ = at.toggle(key="pf_vix").set_value(True).run()
    assert not at.exception
    assert "변동성 필터 HAR(" in var_help(at)  # demo: no ranges or VIX to use
    assert not [b for b in at.button if b.label.endswith("받기")]

    _ = at.segmented_control(key="pf_vol_model").set_value("pdv").run()
    assert "변동성 필터 PDV(" in var_help(at)
    assert "pf_vix" not in [t.key for t in at.toggle]

    # 전망: 경로 생성, then FHS, then the 변동성 모델 only under FHS. Hidden, it
    # keeps the 꼬리위험 choice, and the other way round; no filter without FHS.
    fitted = len(labels)
    _ = at.segmented_control(key="pf_section").set_value("outlook").run()
    assert path_controls(at) == ["pf_gen", "pf_fhs"]
    assert at.segmented_control(key="pf_horizon").options == ["1M", "3M", "6M", "1Y"]
    assert any(m.value.startswith("**하락·상승 경로 비율**") for m in at.markdown)
    assert "LR" in next(t for t in tables(at) if "p" in t.columns).columns
    _ = at.run()
    _ = at.run()  # hidden over more than one rerun
    assert len(labels) == fitted
    _ = at.toggle(key="pf_fhs").set_value(True).run()
    assert not at.exception
    assert path_controls(at) == ["pf_gen", "pf_fhs", "pf_vol_model"]
    assert at.segmented_control(key="pf_vol_model").value == "pdv"
    assert not at.warning
    assert len(labels) == fitted + 1
    assert "LR" in next(t for t in tables(at) if "p" in t.columns).columns
    _ = at.segmented_control(key="pf_vol_model").set_value("har").run()
    _ = at.toggle(key="pf_fhs").set_value(False).run()
    assert path_controls(at) == ["pf_gen", "pf_fhs"]
    drawn = len(calls)
    _ = at.run()  # an unrelated rerun
    assert len(calls) == drawn
    _ = at.segmented_control(key="pf_section").set_value("tail").run()
    assert at.segmented_control(key="pf_vol_model").value == "har"
    _ = at.segmented_control(key="pf_vol_model").set_value("gjr").run()
    assert not [t for t in tables(at) if "LR" in t.columns]
    assert [t.to_dict() for t in tables(at)] == [t.to_dict() for t in first]
    assert len(calls) == drawn  # 꼬리위험 draws no paths
    _ = at.segmented_control(key="pf_section").set_value("outlook").run()
    assert len(calls) == drawn  # the paths are still cached

    # New trades and new prices are picked up; nothing else recomputes.
    root = tmp_path / pf.DATA_DIR
    shown = [str(m.value) for m in at.metric]
    before = len(calls)
    book = pf.load_book(root)
    more = trade(date(2024, 1, 2), "buy", 10, 52_000, ticker="BBB.KS")
    pf.save_book(root, pf.add_trade(book, book.portfolios[0].id, more))
    _ = at.run()
    assert not at.exception
    assert len(calls) == before + 1
    assert [str(m.value) for m in at.metric] != shown
    moved = market("2022-01-03")
    pf.save_prices(root, pf.Prices(moved.close * 1.1, moved.adjusted * 1.1))
    _ = at.run()
    assert not at.exception
    assert len(calls) == before + 2


def test_a_trained_generator_draws_the_fan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "data" / "models"
    real = Path(__file__).resolve().parents[1] / "data" / "models" / "sigcwgan"
    _ = shutil.copytree(
        real, models / "sigcwgan", ignore=shutil.ignore_patterns("*.txt")
    )
    prices = market("2022-01-03")
    names = {"AAA": "AAPL", "NEW": "MSFT"}
    root = tmp_path / pf.DATA_DIR
    book, held = pf.add_portfolio(pf.Book(), "생성", "USD")
    book, other = pf.add_portfolio(book, "기타", "USD")
    for pid, ticker in ((held, "AAPL"), (held, "MSFT"), (other, "BBB.KS")):
        book = pf.add_trade(
            book, pid, trade(date(2023, 1, 3), "buy", 2, 100, ticker=ticker)
        )
    pf.save_book(root, book)
    pf.save_prices(
        root,
        pf.Prices(
            prices.close.rename(columns=names), prices.adjusted.rename(columns=names)
        ),
    )
    at = page(tmp_path, monkeypatch)
    _ = at.selectbox(key="pf_selected").set_value(held).run()
    _ = at.segmented_control(key="pf_section").set_value("outlook").run()
    _ = at.segmented_control(key="pf_window").set_value("1Y").run()

    def fan() -> str:
        """The fan chart, after 현황's donut and value lines and 하락·상승."""
        return str(cast("object", at.get("vega_lite_chart")[3].proto))

    assert not at.exception
    assert at.segmented_control(key="pf_gen").options == ["재표본", "SigCWGAN"]
    off = at.segmented_control(key="pf_gen_off")
    assert off.disabled
    assert "Kronos: 구현 중" in cast("str", off.help)
    assert "Tail-GAN: 가중치 없음" in cast("str", off.help)
    before = fan()
    _ = at.run()
    assert fan() == before  # the same draws
    _ = at.segmented_control(key="pf_gen").set_value("sigcwgan").run()
    assert not at.exception
    assert path_controls(at) == ["pf_gen"]  # no FHS, so no 변동성 모델
    assert fan() != before
    assert not [m for m in at.markdown if m.value.startswith("**분포 검증**")]

    # FHS is hidden, not reset, while a generator draws the paths.
    _ = at.segmented_control(key="pf_gen").set_value("resample").run()
    assert fan() == before
    _ = at.toggle(key="pf_fhs").set_value(True).run()
    assert path_controls(at) == ["pf_gen", "pf_fhs", "pf_vol_model"]
    _ = at.segmented_control(key="pf_gen").set_value("sigcwgan").run()
    assert path_controls(at) == ["pf_gen"]
    _ = at.run()
    _ = at.segmented_control(key="pf_gen").set_value("resample").run()
    assert not at.exception
    assert at.toggle(key="pf_fhs").value
    assert path_controls(at) == ["pf_gen", "pf_fhs", "pf_vol_model"]

    _ = at.selectbox(key="pf_selected").set_value(other).run()
    assert not at.exception
    assert at.segmented_control(key="pf_gen").options == ["재표본"]
    off = at.segmented_control(key="pf_gen_off")
    assert "SigCWGAN: BBB.KS 없음" in cast("str", off.help)


def test_the_fetch_buttons_appear_only_for_missing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not in demo mode, but on the demo's dashboard data and without fetching.
    def demo_data(
        *, demo: bool, env: Mapping[str, str], settings_path: Path | None = None
    ) -> DashboardService:
        del demo
        return build_service(demo=True, env=env, settings_path=settings_path)

    monkeypatch.setattr("finagent.cli.build_service", demo_data)
    started: list[tuple[list[str], bool]] = []

    def start(
        _: pf.Fetcher,
        tickers: Sequence[str],
        since: object = None,
        *,
        ranges: bool = False,
    ) -> bool:
        del since
        started.append((list(tickers), ranges))
        return False

    monkeypatch.setattr(pf.Fetcher, "start", start)
    monkeypatch.setattr(sys, "argv", ["app.py", f"--settings={tmp_path / 's.json'}"])
    prices = market("2022-01-03")
    root = tmp_path / pf.DATA_DIR
    book, pid = pf.add_portfolio(pf.Book(), "연금", "USD")
    for item in (
        trade(date(2023, 1, 3), "buy", 10, 100),
        trade(date(2023, 1, 3), "buy", 2, 50_000, ticker="BBB.KS"),
    ):
        book = pf.add_trade(book, pid, item)
    pf.save_book(root, book)
    pf.save_prices(root, ranged(prices, "AAA"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at._page_hash = calc_hash("portfolio")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    _ = at.run()
    _ = at.segmented_control(key="pf_section").set_value("tail").run()
    _ = at.segmented_control(key="pf_window").set_value("1Y").run()

    def shown() -> list[str]:
        return [b.label for b in at.button if b.label.endswith("받기")]

    assert not at.exception
    assert shown() == []
    _ = at.segmented_control(key="pf_vol_model").set_value("har").run()
    _ = at.toggle(key="pf_ranges").set_value(True).run()
    _ = at.toggle(key="pf_vix").set_value(True).run()
    assert shown() == ["고가·저가 받기", "VIX 받기"]
    assert at.button(key="pf_ranges_fetch").help == "BBB.KS"
    _ = at.button(key="pf_ranges_fetch").click().run()
    _ = at.button(key="pf_vix_fetch").click().run()
    assert (["BBB.KS"], True) in started
    assert ([pf.VIX], False) in started
    before = len(started)
    _ = at.button(key="pf_refresh").click().run()
    refreshed = [t for tickers, _ in started[before:] for t in tickers]
    assert pf.BENCHMARK in refreshed
    assert pf.VIX not in refreshed  # 가격 갱신: not opted in

    vix = prices.adjusted[pf.BENCHMARK] / 50
    pf.save_prices(
        root,
        ranged(
            pf.Prices(
                prices.close.assign(**{pf.VIX: vix}),
                prices.adjusted.assign(**{pf.VIX: vix}),
            ),
            "AAA",
            "BBB.KS",
        ),
    )
    _ = at.run()
    assert not at.exception
    assert shown() == []
    assert "변동성 필터 HAR (고가·저가, VIX)(" in var_help(at)
    before = len(started)
    _ = at.button(key="pf_refresh").click().run()
    assert pf.VIX in [t for tickers, _ in started[before:] for t in tickers]
