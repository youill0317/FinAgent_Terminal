"""Portfolios without the network: trade-log math, analytics, risk, the page."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from finagent import portfolio as pf

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray


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


# ---------------------------------------------------------------- method review


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


def test_an_expected_return_moves_the_fan_and_the_density_check() -> None:
    prices = market()
    t = np.arange(len(prices.adjusted), dtype="float64")
    adjusted = prices.adjusted.copy()
    # The S&P 500's moves and its own, rising 0.1% a day: beta about 1.
    adjusted["AAA"] = (
        adjusted["AAA"] * adjusted[pf.BENCHMARK] / 1_000 * np.exp(1e-3 * t)
    )
    j = pf.joint(adjusted, pd.Series({"AAA": 1.0}), "USD", 12)

    beta = float(j.w @ pf.asset_betas(j))
    assert beta == pytest.approx(1.0, abs=0.1)

    ones = np.ones((3, 4))
    fan = pf.simulation(ones, 100.0, 1.0, 1.0, 0.01).fan
    assert [row["p50"] for row in fan] == pytest.approx(
        [100 * 1.01**d for d in range(4)]
    )

    # Zero-drift forecasts sit below a rising holding; its own drift recentres them.
    held = j.full.to_numpy(dtype="float64") @ j.w
    zero = pf.density(j).test
    drift = pf.density(j, drift=np.full(len(held), held.mean())).test
    assert zero is not None
    assert drift is not None
    assert zero.mean > 0.2
    assert zero.p < 0.01
    assert abs(drift.mean) < 0.1
    assert drift.p > 0.01
    assert drift.sd == pytest.approx(zero.sd, abs=0.1)  # the spread is not touched


def test_past_means_forecast_each_day_from_the_days_before_it() -> None:
    prices = market()
    j = pf.joint(prices.adjusted, pd.Series({"AAA": 0.5, "BBB.KS": 0.5}), "USD", 12)
    x = j.full.to_numpy(dtype="float64")
    n = len(j.window)
    mean, var = pf.trailing_means(j)
    assert mean.shape == (len(x) + 1, 2)
    assert np.isnan(mean[n - 1 : n]).all()
    assert mean[n] == pytest.approx(x[:n].mean(axis=0))
    assert mean[-1] == pytest.approx(j.window.to_numpy().mean(axis=0))  # today's
    assert var[-1] == pytest.approx(x[-n:].var(axis=0, ddof=1) / n)


def test_james_stein_and_bayes_stein_shrink_sample_means() -> None:
    prices = market()
    two = pf.joint(prices.adjusted, pd.Series({"AAA": 0.5, "BBB.KS": 0.5}), "USD", 12)
    assert pf.james_stein(two) is None
    assert pf.bayes_stein(two) is None

    adjusted = prices.adjusted.copy()
    rng = np.random.default_rng(42)
    adjusted["CCC"] = 80.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, len(adjusted))))
    w = pd.Series({"AAA": 0.4, "BBB.KS": 0.3, "CCC": 0.3})
    j = pf.joint(adjusted, w, "USD", 12)
    x = j.window.to_numpy(dtype="float64")
    n = len(j.window)
    k = len(j.names)
    y = x.mean(axis=0)
    s = np.cov(x, rowvar=False, ddof=1)

    # James & Stein (1961): X = sqrt(n) y, A = (n - 1) S ~ Wishart(m = n - 1),
    # (1 - (k - 2) / (m - k + 3) / (X' A^-1 X)) X, toward 0.
    js = pf.james_stein(j)
    assert js is not None
    m = n - 1
    big_x = math.sqrt(n) * y
    quad = float(big_x @ np.linalg.inv(m * s) @ big_x)
    c_js = 1.0 - (k - 2) / (m - k + 3) / quad
    assert js[-1] == pytest.approx(c_js * y)

    # Jorion (1986): shrink toward minimum-variance mean y0.
    bs = pf.bayes_stein(j)
    assert bs is not None
    bs_mean, bs_se = bs
    sigma: NDArray[np.float64] = s * ((n - 1) / (n - k - 2))
    sigma_inv = np.linalg.inv(sigma)
    ones = np.ones(k, dtype="float64")
    denom = float(ones @ sigma_inv @ ones)
    y0 = float(ones @ sigma_inv @ y) / denom
    diff = y - y0 * ones
    v = (k + 2) / ((k + 2) + n * float(diff @ sigma_inv @ diff))
    assert bs_mean[-1] == pytest.approx((1.0 - v) * y + v * y0 * ones)
    # Eq. 15 less Sigma, lambda from v = lambda / (n + lambda).
    lam = v * n / (1.0 - v)
    v_mu: NDArray[np.float64] = sigma / (n + lam) + lam / (
        n * (n + 1 + lam) * denom
    ) * np.outer(ones, ones)
    assert bs_se == pytest.approx(math.sqrt(float(j.w @ v_mu @ j.w)))


@pytest.mark.parametrize("months", list(pf.MOP_LAGS.values()))
def test_momentum_follows_moskowitz_ooi_pedersen(months: int) -> None:
    prices = market()
    j = pf.joint(prices.adjusted, pd.Series({"AAA": 0.5, "BBB.KS": 0.5}), "USD", 6)
    x = j.full.to_numpy(dtype="float64")
    rf = pd.Series(1e-4, index=j.full.index)
    got, error, seen = pf.momentum_returns(j, rf, months)
    assert got.shape == (len(x) + 1, 2)
    assert math.isfinite(error)
    assert error > 0
    assert seen > 0

    # Eq. (1) ex-ante volatility at the first day its weights sum to 99%.
    excess = x - rf.to_numpy()[:, None]
    sigma = pf.ewma_volatility(excess)
    burn = pf.MOP_BURN
    assert np.isnan(sigma[:burn]).all()
    delta = pf.MOP_COM / (pf.MOP_COM + 1.0)
    weights = (1.0 - delta) * delta ** np.arange(burn - 1, -1, -1)
    assert weights.sum() >= pf.MOP_COVERAGE > weights[1:].sum()
    bar = weights @ excess[:burn]
    var0 = pf.MOP_DAYS * (weights @ (excess[:burn] - bar) ** 2)
    assert sigma[burn] == pytest.approx(np.sqrt(var0))

    # Eq. 3 at lag h, refitted by pooled OLS on calendar months' compounded
    # returns; the first and last months may be partial and are left out.
    rfs = rf.to_numpy(dtype="float64")
    periods = pd.DatetimeIndex(j.full.index).to_period("M")
    rows = [np.flatnonzero(periods == p) for p in periods.unique()]

    def compounded(r: NDArray[np.int64]) -> NDArray[np.float64]:
        stock: NDArray[np.float64] = np.prod(1 + x[r], axis=0)
        cash: NDArray[np.float64] = np.prod(1 + rfs[r], keepdims=True)
        return stock - cash

    signs: list[float] = []
    scaled: list[float] = []
    for m in range(1 + months, len(rows) - 1):
        lagged, now = compounded(rows[m - months]), compounded(rows[m])
        vols = sigma[rows[m][:1]].ravel()
        keep = np.isfinite(vols)
        signs.extend(cast("list[float]", np.sign(lagged[keep]).tolist()))
        scaled.extend(cast("list[float]", (now[keep] / vols[keep]).tolist()))
    design = np.column_stack([np.ones(len(signs)), signs])
    coef: NDArray[np.float64] = np.linalg.lstsq(design, np.array(scaled), rcond=None)[0]
    sign_now = np.sign(compounded(rows[len(rows) - 1 - months]))
    fitted = coef @ np.vstack([np.ones(len(sign_now)), sign_now])
    today = sigma[-1:].ravel()
    days = pf.per_year(j.full.index) / pf.MONTHS_PER_YEAR
    total = (1 + rfs[-1:]) ** days + today * fitted
    assert got[-1] == pytest.approx(total ** (1 / days) - 1)


def test_quote_figures_give_market_cap_weights(tmp_path: Path) -> None:
    day = date(2026, 1, 2)
    us = pf.facts_of({"currency": "USD", "marketCap": 1_000_000.0}, day)
    kr = pf.facts_of({"currency": "KRW", "marketCap": 1_300_000_000.0}, day)
    assert us.market_cap == pytest.approx(1_000_000.0)
    assert math.isnan(pf.facts_of({}, day).market_cap)

    pf.save_facts(tmp_path, {"AAA": us})
    pf.save_facts(tmp_path, {"BBB.KS": kr, "NONE": pf.facts_of({}, day)})
    saved = pf.load_facts(tmp_path)
    assert saved["AAA"].market_cap == pytest.approx(1_000_000.0)
    assert math.isnan(saved["NONE"].market_cap)

    prices = market()
    fx = float(pf.fx_series(prices.adjusted).dropna().iloc[-1])
    weights = pf.market_weights(saved, ["AAA", "BBB.KS"], prices.adjusted)
    assert weights is not None
    total_cap = 1_000_000.0 * fx + 1_300_000_000.0
    expected = [1_000_000.0 * fx / total_cap, 1_300_000_000.0 / total_cap]
    assert weights == pytest.approx(expected)
    assert pf.market_weights(saved, ["AAA", "NONE"], prices.adjusted) is None
    assert pf.market_weights(saved, ["AAA", "MISSING"], prices.adjusted) is None


def test_french_csv_parses_monthly_factors_from_july_1963() -> None:
    text = (
        "This file was created by using the 202609 CRSP database.\n"
        "\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "196306,   -0.50,    0.10,    0.20,    0.23\n"
        "196307,   -0.39,   -0.48,   -0.81,    0.27\n"
        "196308,    5.08,   -0.80,    1.70,    0.25\n"
        "\n"
        " Annual Factors: January-December \n"
        "1963,   10.00,    1.00,    2.00,    3.00\n"
    )
    frame_ = pf.parse_french(text)
    assert list(frame_.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert list(frame_.index) == [
        pd.Timestamp("1963-07-01"),
        pd.Timestamp("1963-08-01"),
    ]
    assert frame_.loc["1963-07-01", "Mkt-RF"] == pytest.approx(-0.0039)


def test_factor_returns_follow_fama_french_1997() -> None:
    days = pd.bdate_range("2020-01-01", "2025-12-31")
    months = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
    rng = np.random.default_rng(0)
    factors = pd.DataFrame(
        rng.normal(0.005, 0.03, (len(months), len(pf.FACTORS))),
        index=months,
        columns=list(pf.FACTORS),
    ).assign(RF=0.002)
    loads = np.array([1.1, 0.4, -0.3])
    # Build monthly stock growth matching the 3-factor model, then interpolate daily.
    m_ret = 0.002 + factors[list(pf.FACTORS)].to_numpy() @ loads
    m_end = pd.DatetimeIndex([m + pd.offsets.MonthEnd(1) for m in months])
    m_prices = pd.Series(np.cumprod(1.0 + m_ret), index=m_end)
    daily_price = m_prices.reindex(days.union(m_end)).interpolate("time").reindex(days)
    adjusted = pd.DataFrame({"A": daily_price.to_numpy()}, index=days)

    w = np.array([1.0], dtype="float64")
    fit = pf.factor_returns(adjusted, ["A"], w, factors)
    assert fit is not None
    got, se = fit
    premia = factors[list(pf.FACTORS)].mean().to_numpy()
    assert got == pytest.approx([0.002 + float(loads @ premia)], abs=5e-4)
    # Table 8: Prem' var(u_g) Prem + g' var(u_Prem) g + 1'[var(u_Prem) o var(u_g)]1
    month_end = adjusted.groupby(pd.DatetimeIndex(adjusted.index).to_period("M")).last()
    rets = month_end["A"].pct_change().dropna()
    fac = factors.set_axis(pd.DatetimeIndex(factors.index).to_period("M"))
    both = fac.join(rets, how="inner")
    x = np.column_stack([np.ones(len(both)), both[list(pf.FACTORS)].to_numpy()])
    y = (both["A"] - both["RF"]).to_numpy(dtype="float64")
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    var_g = float(resid @ resid) / (len(y) - 4) * np.linalg.inv(x.T @ x)[1:, 1:]
    hist = factors[list(pf.FACTORS)].to_numpy(dtype="float64")
    var_prem = np.cov(hist, rowvar=False) / len(hist)
    g = coef[1:]
    total = premia @ var_g @ premia + g @ var_prem @ g + float((var_prem * var_g).sum())
    assert se == pytest.approx(math.sqrt(float(total)))

    # No factors, a Korean holding, or too few months: no model.
    none = factors.iloc[:0]
    assert pf.factor_returns(adjusted, ["A"], w, none) is None
    assert pf.factor_returns(adjusted.iloc[-100:], ["A"], w, factors) is None
    # 36 monthly returns are the floor; 35 are not enough.
    assert pf.factor_returns(adjusted.loc["2022-12-01":], ["A"], w, factors)
    assert pf.factor_returns(adjusted.loc["2023-01-01":], ["A"], w, factors) is None
    korean = adjusted.rename(columns={"A": "005930.KS"})
    assert pf.factor_returns(korean, ["005930.KS"], w, factors) is None


def test_a_tradeoff_shifts_the_same_paths_by_the_drift() -> None:
    ends = np.linspace(0.8, 1.2, 1001)
    flat = pf.tradeoff(ends, 21, 0.0)
    up = pf.tradeoff(ends, 21, 0.001)
    assert flat.mean == pytest.approx(0.0, abs=1e-12)
    assert flat.median == pytest.approx(0.0, abs=1e-12)
    assert flat.gain == pytest.approx(0.5, abs=0.001)
    assert flat.low == pytest.approx(-0.18)
    assert flat.high == pytest.approx(0.18)
    assert flat.es is not None
    assert up.es is not None
    assert up.mean == pytest.approx(1.001**21 - 1, abs=1e-12)
    assert up.gain > flat.gain
    assert up.es < flat.es


def test_the_density_check_skips_months_without_a_drift() -> None:
    prices = market()
    j = pf.joint(prices.adjusted, pd.Series({"AAA": 1.0}), "USD", 12)
    held = len(j.full)
    some = np.full(held, 1e-4)
    some[: held // 2] = np.nan
    every = pf.density(j, drift=np.full(held, 1e-4))
    fewer = pf.density(j, drift=some)
    assert 0 < fewer.blocks < every.blocks


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


def test_black_litterman_combines_equilibrium_prior_and_views() -> None:
    prices = market()
    j = pf.joint(prices.adjusted, pd.Series({"AAA": 0.5, "BBB.KS": 0.5}), "USD", 6)
    rf = pd.Series(1e-4, index=j.full.index)
    mkt_w = np.array([0.6, 0.4], dtype="float64")
    x = j.window.to_numpy(dtype="float64")
    rf_window = rf.reindex(j.window.index).ffill().fillna(0.0).to_numpy(dtype="float64")
    excess = x - rf_window[:, None]
    sigma = np.cov(excess, rowvar=False, ddof=1)
    tau_sigma: NDArray[np.float64] = pf.BL_TAU * sigma
    # Formula 1: lambda = the specified premium (here 4% a year, a day at the
    # views' compounding) over the market portfolio's variance.
    premium = math.pow(1.04, 1 / pf.per_year(j.full.index)) - 1
    lam = premium / float(mkt_w @ sigma @ mkt_w)
    prior: NDArray[np.float64] = 1e-4 + lam * (sigma @ mkt_w)
    variance = cast("list[float]", np.diag(tau_sigma).tolist())[0]
    prior_aaa = cast("list[float]", prior.tolist())[0]
    q = math.pow(1.2, 1 / pf.per_year(j.full.index)) - 1

    def posterior(c: float) -> NDArray[np.float64]:
        got = pf.black_litterman(
            j, rf, mkt_w, [pf.View(ticker="AAA", annual=0.2, confidence=c)], 0.04
        )
        assert got is not None
        return got

    # By hand, one view on AAA: the gap shared through AAA's covariances.
    for c in (1.0, 0.3):
        shrink = variance / c  # its variance plus Omega, (1 - c) / c of it
        gap = q - prior_aaa
        assert posterior(c) == pytest.approx(prior + tau_sigma[:, 0] / shrink * gap)
    assert posterior(1.0)[:1] == pytest.approx([q])  # full confidence: the view
    assert posterior(1e-6) == pytest.approx(prior, rel=1e-4)  # none: the prior
    # No view on a holding or missing market weights: no model.
    other = [pf.View(ticker="ZZZ", annual=0.2, confidence=0.5)]
    view = [pf.View(ticker="AAA", annual=0.2, confidence=0.5)]
    assert pf.black_litterman(j, rf, mkt_w, other) is None
    assert pf.black_litterman(j, rf, None, view) is None


def test_views_are_saved_with_the_portfolio(tmp_path: Path) -> None:
    book, pid = pf.add_portfolio(pf.Book(), "내 계좌", "USD")
    views = (pf.View(ticker="AAA", annual=0.1, confidence=0.6),)
    pf.save_book(tmp_path, pf.set_views(book, pid, views, 0.05))
    saved = pf.load_book(tmp_path).get(pid)
    assert saved is not None
    assert saved.views == views
    assert saved.premium == pytest.approx(0.05)
    # Files from before views: none.
    old = '{"version": 1, "portfolios": [{"id": "x", "name": "a", "trades": []}]}'
    _ = (tmp_path / pf.BOOK_FILE).write_text(old, encoding="utf-8")
    loaded = pf.load_book(tmp_path).get("x")
    assert loaded is not None
    assert loaded.views == ()
    assert loaded.premium == pf.BL_PREMIUM
    with pytest.raises(ValidationError):
        _ = pf.View(ticker="AAA", annual=0.1, confidence=0.0)
