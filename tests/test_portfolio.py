"""Portfolios without the network: trade-log math, analytics, risk, the page."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import math
import re
import sys
import threading
from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

from finagent import perf, quant
from finagent import portfolio as pf
from finagent.cli import APP

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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
    assert pf.holdings_table([held], "KRW")["평가액 (KRW)"] == ["미확인"]
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
        **{pf.BENCHMARK: [10.0, 11.0, 12.0, 13.0, 14.0]},
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


def test_realized_pnl_by_year_and_month() -> None:
    ledger = pf.replay(
        [
            trade(date(2024, 1, 2), "buy", 10, 100),
            trade(date(2024, 5, 2), "sell", 2, 110),
            trade(date(2024, 5, 20), "sell", 2, 90),
            trade(date(2025, 2, 3), "sell", 1, 150),
        ],
        "USD",
    )
    years = pf.realized_by(ledger.sales)
    months = pf.realized_by(ledger.sales, monthly=True)

    assert dict(years) == {"2024": pytest.approx(0), "2025": pytest.approx(50)}
    assert [month for month, _ in months] == ["2024-05", "2025-02"]


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


def test_under_a_year_returns_stay_cumulative() -> None:
    days = pd.bdate_range("2025-01-02", periods=200)
    short = pd.Series(np.linspace(0, 0.3, len(days)), index=days)
    long = pd.Series(
        np.linspace(0, 0.3, 400), index=pd.bdate_range("2024-01-02", periods=400)
    )

    first = pf.time_weighted(short)
    second = pf.time_weighted(long)
    assert first is not None
    assert second is not None
    assert first.annual is None
    assert first.cumulative == pytest.approx(0.3)
    assert second.annual is not None
    assert second.annual < 0.3
    assert pf.gap(first, second) == perf.Gap(0.0, annualised=False)
    assert pf.gap(second, second) == perf.Gap(0.0, annualised=True)
    assert pf.time_weighted(pd.Series(dtype="float64")) is None


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
    trials = np.array([s_ / math.sqrt(252) for s_ in sharpes])
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


def test_held_returns_use_yesterdays_positions() -> None:
    p = folio(trade(D1, "buy", 1, 100), trade(D3, "buy", 1, 50))
    adjusted = frame([D1, D2, D3, D4], AAA=[100.0, 110.0, 99.0, 108.9])

    assert pf.held_returns(p, adjusted).tolist() == pytest.approx([0.1, -0.1, 0.1])


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


def test_the_us_sleeve_and_the_korean_share() -> None:
    w = pd.Series({"AAA": 0.3, "BBB.KS": 0.6, "CCC": 0.1})

    assert pf.korean_weight(w) == pytest.approx(0.6)
    assert pf.us_sleeve(w).to_dict() == pytest.approx({"AAA": 0.75, "CCC": 0.25})
    assert pf.us_sleeve(pd.Series({"BBB.KS": 1.0})).empty
    theirs = pd.Series({"AAA": 0.5, "DDD": 0.5})
    assert perf.active_share(theirs, pf.us_sleeve(w)) == pytest.approx(0.5)


def test_stress_uses_the_sp500_times_a_beta_for_holdings_without_history() -> None:
    prices = market()
    p = folio(
        trade(date(2026, 1, 5), "buy", 10, 100),
        trade(date(2026, 1, 5), "buy", 10, 100, ticker="NEW"),
    )
    items = pf.holdings(p, prices.close)
    values = {h.ticker: h.value for h in items}
    fx = pf.long_fx(prices.adjusted, pd.Series(dtype="float64"))
    rows = pf.scenario_replays(values, prices.adjusted, "USD", fx)
    gfc = next(r for r in rows if r.scenario.name == "금융위기")
    covid = next(r for r in rows if r.scenario.name == "코로나")

    assert gfc.proxied == ("NEW",)
    assert covid.proxied == ("NEW",)
    rets = pf.asset_returns(prices.adjusted, ["AAA", "NEW"], "USD")
    assert pf.late_starts(rets) == "NEW 2021-01-05"
    # NEW (a copy of AAA) barely moves with the S&P 500: its beta is near 0,
    # pulled toward 1 by its ~1,500 days.
    beta = pf.proxy_beta(prices.adjusted, "NEW", pf.BENCHMARK)
    assert gfc.proxy_betas == (beta,)
    assert 0 < beta < 0.5
    assert gfc.window is not None
    start, end = (pd.Timestamp(d) for d in gfc.window)
    inside = prices.adjusted.loc[start:end, ["AAA", pf.BENCHMARK]]
    own = (inside["AAA"] / inside["AAA"].iloc[0]).to_numpy()
    index = inside[pf.BENCHMARK].to_numpy()
    proxy = np.concatenate([[1.0], np.cumprod(1 + beta * (index[1:] / index[:-1] - 1))])
    growth = own * values["AAA"] + proxy * values["NEW"]
    path = growth / sum(values.values())
    assert gfc.change == pytest.approx(path[-1] - 1)
    assert gfc.worst == pytest.approx((path / np.maximum.accumulate(path) - 1).min())
    assert gfc.worst <= min(gfc.change, 0.0)
    assert gfc.pnl == pytest.approx(gfc.change * sum(values.values()))
    assert gfc.fx == pytest.approx(0.0)  # all in dollars
    old = next(r for r in rows if r.scenario.name == "1987 블랙먼데이")
    assert old.window is None  # no index history back then
    assert math.isnan(old.change)


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

    rets = pf.held_returns(p, adjusted)

    assert len(rets) < len(days) - 100  # the sold-out stretch is left out
    assert pf.market_beta(rets, adjusted, "USD") == pytest.approx(1.0, abs=0.05)
    assert math.isnan(pf.market_beta(rets.head(2), adjusted, "USD"))


def test_monte_carlo_does_not_extrapolate_a_strong_year() -> None:
    days = pd.bdate_range("2023-01-02", periods=760)
    drift = np.exp(np.cumsum(np.random.default_rng(11).normal(0.003, 0.01, len(days))))
    adjusted = pd.DataFrame({"AAA": 100 * drift, pf.BENCHMARK: 1_000.0}, index=days)
    j = pf.joint(adjusted, pd.Series({"AAA": 1.0}), "USD", 36)  # +75% a year
    picks = quant.stationary_indices(len(j.window), 2_000, 252, pf.mean_block(j))

    ends = pf.growth(j, picks, fhs=False)[:, -1]

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
    # NEW lists in 2021, inside a 6-year window: left out, AAA stands alone.
    six, _ = pf.lineup(book, prices, "KRW", 72)
    rets = pf.window(pf.asset_returns(prices.adjusted, ["AAA"], "KRW"), 72)
    assert np.allclose(six["미국"].dropna(), rets["AAA"].dropna().loc[six.index])
    found = pf.lagged_correlation(frame_, flags)
    assert found is not None
    assert found.matrix.loc["테스트", "테스트"] == pytest.approx(1.0)
    assert 0 <= found.shrink <= 1


def test_the_old_risk_functions_are_gone() -> None:
    gone = (
        "metrics", "Metrics", "correlation", "monthly", "var", "var_table", "Risk",
        "draws", "block_draws", "simulate", "outcome", "Outcome", "stress",
        "stress_values", "Stress", "window_change", "window_changes",
        "proxy_change", "benchmark_since", "benchmark_returns", "compare",
        "INDEX_LABELS", "THIN_DAYS", "BLOCK_DAYS", "LEVELS",
    )  # fmt: skip
    assert [name for name in gone if hasattr(pf, name)] == []
    source = APP.read_text(encoding="utf-8")
    app_gone = (
        "pf_simulation", "pf_history_var", "pf_stress", "pf_stress_table",
        "cached_simulation", "cached_var", "cached_stress", "cached_metrics",
        "cached_correlation", "cached_compare", "inv_metrics",
    )  # fmt: skip
    assert [n for n in app_gone if re.search(rf"\bdef {n}\(", source)] == []
    assert "이전 방식" not in (APP.parents[2] / "README.md").read_text("utf-8")


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
        beta = quant.dimson_beta(r, m) if lagged else quant.welch_beta(r, m)
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
    assert covid.pnl == pytest.approx(1_000 * korean_part + 500 * covid.sp500, abs=0.5)
    assert covid.fx == pytest.approx(
        (1 + covid.change) / (1 + (1_000 * -0.25 + 500 * covid.sp500) / 1_500) - 1,
        abs=1e-3,
    )
    assert pf.KOSPI in pf.wanted(pf.Book())
    assert pf.RF_USD in pf.wanted(pf.Book())


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


def test_analytics_on_a_long_history() -> None:
    prices = market()
    p = folio(
        trade(date(2023, 1, 3), "buy", 10, 100),
        trade(date(2023, 1, 3), "buy", 2, 50_000, ticker="BBB.KS"),
        base="KRW",
    )
    rets = pf.window(pf.held_returns(p, prices.adjusted), pf.WINDOWS["1Y"])
    days = pd.DatetimeIndex(rets.index)
    rf = pf.risk_free(pd.Series(dtype="float64"), days)
    stats = pf.ratios(rets, rf, pf.blended(p, prices.adjusted, days))
    book = pf.Book(portfolios=(p,))
    frame_, flags = pf.lineup(book, prices, "KRW", 12)
    found = pf.lagged_correlation(frame_, flags)

    assert 240 < stats.days < 270
    assert stats.volatility is not None
    assert stats.volatility > 0
    assert found is not None
    assert list(found.matrix.columns) == ["테스트", pf.BENCHMARK_LABEL]
    assert pf.matrix_points(found.matrix)[0]["value"] == pytest.approx(1.0)


def test_the_risk_free_rate_lines_up_with_the_returns() -> None:
    days = pd.bdate_range("2025-01-01", periods=300)
    rets = pd.Series(
        np.random.default_rng(6).normal(0.001, 0.01, len(days)), index=days
    )
    rates = pd.Series([3.0, 6.0], index=pd.to_datetime(["2025-01-03", "2025-06-02"]))

    rf = pf.risk_free(rates, pd.DatetimeIndex(days))
    assert rf.iloc[:2].isna().all()  # before the first rate
    assert rf.loc["2025-01-03"] == pytest.approx(1.03 ** (1 / 252) - 1)
    assert rf.loc["2025-05-30"] == pytest.approx(1.03 ** (1 / 252) - 1)
    assert rf.loc["2025-06-02"] == pytest.approx(1.06 ** (1 / 252) - 1)
    zero = pf.risk_free(pd.Series(dtype="float64"), pd.DatetimeIndex(days))
    assert zero.tolist() == [0.0] * len(days)

    stats = pf.ratios(rets, rf, rets * 0.5)
    assert stats.days == len(days) - 2
    kept = rets.iloc[2:]
    excess = kept - rf.iloc[2:]
    assert stats.sharpe == pytest.approx(
        excess.mean() / excess.std(ddof=1) * math.sqrt(252)
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
    held = pf.held_returns(p, close)
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
        **{pf.BENCHMARK: [10.0, 11.0, 11.0], pf.KOSPI: [10.0, 10.0, 12.0]},
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
        **{pf.BENCHMARK: [10.0, 11.0, 12.0, 13.0, 14.0], pf.KOSPI: [10.0] * 5},
        **{pf.FX: [1.0] * 5},
    )
    p = folio(
        trade(D1, "buy", 1, 100), trade(D2, "sell", 1, 100), trade(D4, "buy", 1, 100)
    )
    rets = pf.held_returns(p, close)
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
    before = pf.growth(j, picks, fhs=False)
    after = pf.growth(j, picks, fhs=False, w=pf.mixed_in(j, "FLAT", 0.1))

    assert np.allclose(before, pf.growth(own, picks, fhs=False))
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

    plain = pf.growth(j, picks, fhs=False)
    filtered = pf.growth(j, picks, fhs=True)
    assert plain.shape == filtered.shape == (500, 64)
    assert not np.array_equal(plain, filtered)
    sim = pf.simulation(plain, 100.0, 0.9, 1.0)
    assert 0 <= sim.losses[0.2] <= sim.losses[0.1] <= 1
    assert sim.paths.drawdown_odds[0.3] <= sim.paths.drawdown_odds[0.1]
    assert len(sim.fan) == 64

    t = pf.tails(j)
    assert t.days == len(j.window)
    assert pf.tail_of(t, "역사적", 0.99).var is None  # under 1,000 days
    head = pf.tail_of(t, "역사적", 0.975)
    assert head.es is not None
    assert t.band is not None
    assert t.band[0] <= head.es <= t.band[1] * 1.5
    assert t.backtest is not None
    assert t.component is not None
    assert t.component.sum() == pytest.approx(head.es)
    short = pf.tails(pf.joint(prices.adjusted, w, "USD", 6))
    assert all(row.var is None for row in short.rows)
    assert short.month is None


def test_long_lines_are_thinned_for_charts_only() -> None:
    days = pd.bdate_range("2007-01-02", periods=5_000)
    walk = np.cumsum(np.random.default_rng(3).normal(0, 1, len(days)))
    series = pd.Series(walk, index=days)
    low, high = int(np.argmin(walk)), int(np.argmax(walk))

    thin = pf.thinned(series)
    points = pf.line_points({"a": series, "b": series.head(300)})

    assert len(thin) <= pf.CHART_POINTS
    assert thin.index[0] == days[0]
    assert thin.index[-1] == days[-1]
    assert days[low] in thin.index
    assert days[high] in thin.index
    assert thin.equals(series.loc[thin.index])  # real values, not averages
    assert pf.thinned(series.head(300)).equals(series.head(300))
    assert sum(r["series"] == "a" for r in points) == len(thin)
    assert sum(r["series"] == "b" for r in points) == 300
    assert len(series) == 5_000  # the series itself is left whole


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


# ---------------------------------------------------------------- page


def page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The portfolio page in demo mode, with files under ``tmp_path``."""
    settings = tmp_path / "finagent.settings.json"
    monkeypatch.setattr(sys, "argv", ["app.py", "--demo", f"--settings={settings}"])
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at._page_hash = calc_hash("portfolio")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    return at.run()


def test_the_page_without_portfolios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = page(tmp_path, monkeypatch)

    assert not at.exception
    assert "포트폴리오" in [h.value for h in at.header]
    assert any(i.value.startswith("포트폴리오가 없습니다") for i in at.info)

    _ = at.button(key="pf_new").click().run()
    _ = at.text_input(key="pf_new_name").set_value("연금")
    _ = at.button(key="pf_new_save").click().run()

    assert not at.exception
    saved = pf.load_book(tmp_path / pf.DATA_DIR)
    assert [p.name for p in saved.portfolios] == ["연금"]
    assert any(i.value.startswith("거래를 추가하세요") for i in at.info)


def test_the_page_with_a_portfolio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / pf.DATA_DIR
    book, pid = pf.add_portfolio(pf.Book(), "연금", "KRW")
    book = pf.add_trade(book, pid, trade(date(2025, 1, 6), "buy", 10, 100))
    book = pf.add_trade(book, pid, trade(date(2026, 2, 2), "sell", 4, 120))
    book, _ = pf.add_portfolio(book, "국내", "KRW")
    pf.save_book(root, book)
    pf.save_prices(root, market())
    at = page(tmp_path, monkeypatch)

    assert not at.exception
    assert at.dataframe  # holdings and the trade log

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

    for section in ("returns", "analysis", "correlation", "risk"):
        _ = at.segmented_control(key="pf_section").set_value(section).run()
        assert not at.exception, section
        assert at.get("vega_lite_chart"), section
    _ = at.segmented_control(key="pf_section").set_value("stress").run()
    assert not at.exception
    assert any(
        "1997 IMF" in list(d.value["시나리오"])
        for d in at.dataframe
        if "시나리오" in d.value.columns
    )

    # Remove the newest trade (the first row), standing in for a row click.
    _ = at.segmented_control(key="pf_section").set_value("holdings").run()
    key = f"pf_trades_{at.session_state.get('pf_gen', 0)}"
    at.session_state[key] = {"selection": {"rows": [0], "columns": [], "cells": []}}
    _ = at.button(key="pf_remove").click().run()
    assert not at.exception
    assert len(pf.load_book(root).portfolios[0].trades) == 3  # four, less the newest

    _ = at.button(key="pf_delete").click().run()
    _ = at.button(key="pf_delete_ok").click().run()
    assert not at.exception
    assert [p.name for p in pf.load_book(root).portfolios] == ["국내"]


def test_every_new_section_and_expander_renders_in_demo_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / pf.DATA_DIR
    book, pid = pf.add_portfolio(pf.Book(), "연금", "KRW")
    for item in (
        trade(date(2022, 1, 4), "buy", 10, 100),
        trade(date(2022, 1, 4), "buy", 2, 50_000, ticker="BBB.KS"),
        trade(date(2025, 6, 2), "sell", 3, 120),
    ):
        book = pf.add_trade(book, pid, item)
    pf.save_book(root, book)
    prices = market()
    kospi = prices.adjusted["BBB.KS"] / 50
    spy = prices.adjusted[pf.BENCHMARK]
    pf.save_prices(
        root,
        pf.Prices(
            prices.close.assign(**{pf.KOSPI: kospi, "SPY": spy}),
            prices.adjusted.assign(**{pf.KOSPI: kospi, "SPY": spy}),
        ),
    )
    at = page(tmp_path, monkeypatch)

    _ = at.segmented_control(key="pf_section").set_value("returns").run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"TWR 누적", "MWR", "현지 수익률", "환율", "교차항", "타이밍 차이"} <= set(
        labels
    )

    _ = at.segmented_control(key="pf_section").set_value("analysis").run()
    assert not at.exception
    assert any("0으로 계산" in c.value for c in at.caption)  # demo: no rate
    at.session_state["pf_advanced"] = True
    at.session_state["pf_style"] = True
    _ = at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "소르티노" in labels
    assert any(c.value.startswith(pf_app_note()) for c in at.caption)

    _ = at.segmented_control(key="pf_window").set_value("3Y").run()
    _ = at.segmented_control(key="pf_section").set_value("risk").run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"ES 97.5% (1일)", "초과 횟수", "Kupiec p"} <= set(labels)

    _ = at.segmented_control(key="pf_section").set_value("stress").run()
    at.session_state["pf_turbulence"] = True
    _ = at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "−30% 손실에 필요한 지수 하락폭" in labels
    assert "이번 주 이례 정도" in labels
    assert "**가상 충격** · 지금 보유분" in [m.value for m in at.markdown]


def pf_app_note() -> str:
    return "과거 움직임이 비슷했던 조합이며 실제 보유나 추천이 아닙니다"


def prefill_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """A portfolio with saved prices; fetching would raise."""

    def no_network(*_: object) -> None:
        raise AssertionError("fetched in demo mode")

    monkeypatch.setattr(pf.Yahoo, "__init__", no_network)
    root = tmp_path / pf.DATA_DIR
    book, _ = pf.add_portfolio(pf.Book(), "연금", "USD")
    pf.save_book(root, book)
    pf.save_prices(root, market())
    return page(tmp_path, monkeypatch)


def test_the_price_is_prefilled_from_saved_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = prefill_page(tmp_path, monkeypatch)
    closes = cast("pd.Series", market().close["AAA"])

    def close(day: str) -> float:
        return round(float(cast("float", closes[day])), 2)

    _ = at.date_input(key="pf_day").set_value(date(2026, 10, 21))
    _ = at.text_input(key="pf_ticker").set_value("aaa").run()
    assert at.number_input(key="pf_price").value == close("2026-10-21")
    assert any(c.value == "10/21 종가 기준 · 지연" for c in at.caption)

    # A Saturday falls back to Friday's close.
    _ = at.date_input(key="pf_day").set_value(date(2026, 10, 17)).run()
    assert at.number_input(key="pf_price").value == close("2026-10-16")
    assert any(c.value.startswith("10/16 종가") for c in at.caption)

    # A typed price stays through unrelated reruns.
    _ = at.number_input(key="pf_price").set_value(123.0).run()
    _ = at.number_input(key="pf_quantity").set_value(2.0).run()
    _ = at.run()
    assert at.number_input(key="pf_price").value == 123.0
    assert at.number_input(key="pf_quantity").value == 2.0


def test_demo_mode_does_not_fetch_a_new_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = prefill_page(tmp_path, monkeypatch)

    _ = at.text_input(key="pf_ticker").set_value("ZZZ").run()

    assert not at.exception
    assert at.number_input(key="pf_price").value is None
    assert any(c.value == "저장된 가격 없음" for c in at.caption)


def test_a_ticker_fetched_on_demand_is_saved(tmp_path: Path) -> None:
    def fake(_: Path) -> pf.Source:
        return FakeSource()

    prices = pf.Fetcher(tmp_path, source=fake).fetch_now(["AAA"])

    assert prices.tickers == {"AAA"}
    assert pf.close_on(prices.close, "AAA", D5) == (2.0, D2)
    assert pf.load_prices(tmp_path).tickers == {"AAA"}


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


def test_risk_is_cached_until_the_trades_or_prices_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = pf.growth

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(pf, "growth", counted)
    root = tmp_path / pf.DATA_DIR
    book, pid = pf.add_portfolio(pf.Book(), "연금", "USD")
    pf.save_book(root, pf.add_trade(book, pid, trade(date(2024, 1, 2), "buy", 10, 100)))
    pf.save_prices(root, market())
    at = page(tmp_path, monkeypatch)

    def shown() -> list[str]:
        return [str(m.value) for m in at.metric]

    _ = at.segmented_control(key="pf_section").set_value("risk").run()
    assert not at.exception
    assert "10% 이상 손실 경로 비율" in [m.label for m in at.metric]
    first = shown()
    _ = at.run()  # an unrelated rerun
    assert len(calls) == 1

    korean = trade(date(2024, 1, 2), "buy", 10, 52_000, ticker="BBB.KS")
    pf.save_book(root, pf.add_trade(pf.load_book(root), pid, korean))
    _ = at.run()
    assert not at.exception
    assert len(calls) == 2
    assert shown() != first

    _ = at.toggle(key="pf_fhs").set_value(True).run()
    assert not at.exception
    assert len(calls) == 3

    moved = market()
    pf.save_prices(root, pf.Prices(moved.close * 1.1, moved.adjusted * 1.1))
    _ = at.run()
    assert len(calls) == 4


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
    assert any(i.value.startswith("티커를 넣거나") for i in at.info)
    return at, pid


def test_one_held_stock_renders_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at, _ = stock_page(tmp_path, monkeypatch, [])

    _ = at.pills(key="stock_held").set_value("BBB.KS").run()
    assert not at.exception
    assert at.segmented_control(key="stock_base").value == "KRW"
    labels = [m.label for m in at.metric]
    assert {"샤프 지수", "PSR", "최대 낙폭"} <= set(labels)
    betas = next(d.value for d in at.dataframe if "Dimson" in d.value.columns)
    assert list(betas["구분"]) == ["BBB.KS"]

    _ = at.segmented_control(key="stock_base").set_value("USD").run()
    _ = at.segmented_control(key="pf_window").set_value("3Y").run()
    _ = at.segmented_control(key="stock_section").set_value("risk").run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"ES 97.5% (1일)", "10% 이상 손실 경로 비율"} <= set(labels)
    assert not any(m.value.startswith("**종목별 ES") for m in at.markdown)

    _ = at.segmented_control(key="stock_section").set_value("stress").run()
    assert not at.exception
    assert "−30% 손실에 필요한 지수 하락폭" in [m.label for m in at.metric]
    heads = [m.value for m in at.markdown]
    assert "**가상 충격** · 이 종목 1주" in heads
    assert not any("지금 보유분" in h for h in heads)
    assert any("이 종목 1주" in d.value.columns for d in at.dataframe)
    assert "주간 난기류" not in [e.label for e in at.expander]

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


def test_adding_a_stock_to_a_portfolio_reuses_the_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    at, pid = stock_page(tmp_path, monkeypatch, calls)

    _ = at.text_input(key="stock_input").set_value("aaa")
    _ = at.button(key="stock_go").click().run()
    assert at.session_state["stock_ticker"] == "AAA"
    _ = at.segmented_control(key="stock_section").set_value("whatif").run()
    assert not at.exception
    assert at.selectbox(key="whatif_pf").value == pid
    labels = [m.label for m in at.metric]
    assert {"연 변동성", "베타", "분산 비율", "ES 97.5% (1일)"} <= set(labels)
    assert any("지금 비중" in c.value for c in at.caption)  # AAA is held
    assert any("시나리오" in d.value.columns for d in at.dataframe)
    drawn = len(calls)

    _ = at.slider(key="whatif_percent").set_value(20).run()
    assert not at.exception
    assert len(calls) == drawn + 1  # only the new mix; the draws and "before" stay
