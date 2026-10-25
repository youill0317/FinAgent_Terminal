"""Performance math: known answers from the literature and properties."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from finagent import perf

SORTINO_CASE = [0.01, -0.02, 0.03, -0.01]


def rng(seed: int = 3) -> np.random.Generator:
    return np.random.default_rng(seed)


def day_offsets(*days: str) -> list[float]:
    stamps = pd.DatetimeIndex(list(days))
    return [float((d - stamps[0]).days) for d in stamps]


# ---------------------------------------------------------------- known answers


def test_lo_standard_errors() -> None:
    assert perf.sharpe_se(1.5, 60) == pytest.approx(0.188, abs=5e-4)
    assert perf.sharpe_se(3.0, 60) == pytest.approx(0.303, abs=5e-4)


def test_expected_max_sharpe_and_deflated_sharpe() -> None:
    assert perf.expected_max_sr(100, 1.0) == pytest.approx(2.531, abs=5e-4)
    bar = perf.expected_max_sr(100, 0.15**2)
    assert bar is not None
    dsr = perf.psr_from(1.5 / math.sqrt(12), 60, -1.0, 6.0, bar)
    assert dsr == pytest.approx(0.6247, abs=5e-4)
    assert perf.expected_max_sr(1, 1.0) is None


def test_min_track_record_length() -> None:
    assert perf.min_trl(0.1, 0.0, 3.0) == pytest.approx(272.9, abs=0.05)
    assert perf.min_trl(0.0) is None


def test_xirr_known_answer() -> None:
    days = day_offsets("2020-01-01", "2021-01-01", "2022-01-01")

    assert perf.xirr(days, [-100, -50, 300]) == pytest.approx(0.4993, abs=5e-5)


def test_xirr_of_a_few_days_is_a_large_annual_rate() -> None:
    # 105/100 over 7 days is 1.05^(365/7) - 1; 125/100 over 30 days likewise.
    week = perf.xirr([0, 7], [-100, 105])
    month = perf.mwr([0, 30], [-100, 125])

    assert week == pytest.approx(1.05 ** (365 / 7) - 1, rel=1e-9)  # about 1,173%
    assert month is not None
    assert month.cumulative == pytest.approx(0.25, rel=1e-9)
    assert perf.xirr([0, 30], [-100, 125]) == pytest.approx(
        1.25 ** (365 / 30) - 1, rel=1e-9
    )  # about 1,410%


def test_xirr_is_none_when_two_rates_fit() -> None:
    # -100 + 230/(1+r) - 132/(1+r)^2 = 0 at r = 10% and at r = 20%.
    assert perf.xirr([0, 365, 730], [-100, 230, -132]) is None
    assert perf.xirr([0, 0], [-100, 100]) is None  # no span
    assert perf.xirr([0, 400], [-100, -50]) is None  # no sign change
    assert perf.mwr([0, 400], [100, 50]) is None


def test_ratios_known_answers() -> None:
    assert perf.sortino(SORTINO_CASE) == pytest.approx(0.22361, abs=1e-5)
    assert perf.omega(SORTINO_CASE) == pytest.approx(4 / 3)
    assert perf.ulcer_index([-0.10, 0.0]) == pytest.approx(10.0)
    # Drawdowns count from the starting capital; Calmar needs a year.
    assert perf.drawdown_series([-0.1, 0.05]) == pytest.approx([-0.1, -0.055])
    assert perf.calmar([0.1, -0.2, 0.15], 364) is None
    assert perf.calmar([0.1, -0.2, 0.15], 365) == pytest.approx(
        (1.1 * 0.8 * 1.15 - 1) / 0.2
    )


def test_fx_split_known_answer() -> None:
    split = perf.fx_split([[1.0]], [[0.10]], [[0.05]])

    assert split.base == pytest.approx(0.155)
    assert split.local == pytest.approx(0.10)
    assert split.fx == pytest.approx(0.05)
    assert split.cross == pytest.approx(0.005)


# ---------------------------------------------------------------- Sharpe


def test_mertens_reduces_to_lo() -> None:
    for sr, n in [(0.1, 252), (0.05, 1000), (-0.2, 60)]:
        lo = math.sqrt((1 + sr**2 / 2) / n)
        assert perf.sharpe_se(sr, n, 0.0, 3.0) == pytest.approx(lo)


def test_psr_is_half_at_its_own_sharpe() -> None:
    assert perf.psr_from(0.08, 500, -0.5, 5.0, 0.08) == pytest.approx(0.5)


def test_sharpe_interval_needs_a_year() -> None:
    rets = rng().normal(0.0005, 0.01, 1000)
    ci = perf.sharpe_ci(rets)

    assert ci is not None
    assert ci.low < ci.sharpe < ci.high
    sr = perf.sharpe(rets)
    assert sr is not None
    assert ci.sharpe == pytest.approx(sr * math.sqrt(252))
    kospi = perf.sharpe_ci(rets, periods=243)
    assert kospi is not None
    assert kospi.sharpe == pytest.approx(sr * math.sqrt(243))
    assert perf.sharpe_ci(rets[:251]) is None
    assert perf.psr(rets[:251]) is None
    assert perf.psr(rets) is not None


def test_deflated_sharpe_grows_with_fewer_trials() -> None:
    rets = rng().normal(0.001, 0.01, 500)
    few = perf.deflated_sharpe(rets, [0.0, 0.05])
    many = perf.deflated_sharpe(rets, rng(4).normal(0.0, 0.05, 200))

    assert few is not None
    assert many is not None
    assert 0 < many < few < 1
    assert perf.deflated_sharpe(rets[:100], [0.0, 0.05]) is None


# ---------------------------------------------------------------- drawdowns


def test_rising_path_has_no_drawdown() -> None:
    rets = [0.01, 0.02, 0.005, 0.03]
    wealth = pd.Series(
        np.cumprod([1.0, *[1 + r for r in rets]]),
        index=pd.date_range("2024-01-01", periods=5),
    )

    assert np.all(perf.drawdown_series(rets) == 0)
    assert perf.max_drawdown(rets) == 0
    assert perf.average_drawdown(rets) == 0
    assert perf.cdar(rets) == 0
    assert perf.ulcer_index(rets) == 0
    assert perf.calmar(rets, 400) is None
    assert perf.ulcer_performance(rets, 400) is None
    under = perf.underwater(wealth)
    assert under.max_drawdown == 0
    assert under.longest_days == 0
    assert under.current_days == 0
    assert under.recovery_days == 0


def test_cdar_limits() -> None:
    rets = rng().normal(0.0, 0.02, 300)
    mdd = perf.max_drawdown(rets)
    avg = perf.average_drawdown(rets)

    assert perf.cdar(rets, 1.0) == pytest.approx(mdd)
    assert perf.cdar(rets, 0.0) == pytest.approx(avg)
    assert mdd <= perf.cdar(rets, 0.95) <= perf.cdar(rets, 0.5) <= avg


def test_cdar_counts_the_boundary_drawdown_by_its_fraction() -> None:
    # 30 drawdowns at 95%: m = 1.5, so the deepest and half the next.
    rets = [-0.01] * 30
    dd = np.sort(perf.drawdown_series(rets))

    assert perf.cdar(rets, 0.95) == pytest.approx((dd[0] + 0.5 * dd[1]) / 1.5)
    # 5% of 100 is 5 observations, not 6 from floating-point noise.
    hundred = [-0.01] * 100
    assert perf.cdar(hundred, 0.95) == pytest.approx(
        np.sort(perf.drawdown_series(hundred))[:5].mean()
    )


def test_underwater_dates_and_durations() -> None:
    wealth = pd.Series(
        [1.0, 1.1, 0.99, 1.0, 1.1, 1.2, 1.14],
        index=pd.date_range("2024-01-01", periods=7),
    )
    under = perf.underwater(wealth)

    assert under.max_drawdown == pytest.approx(-0.1)
    assert under.peak == pd.Timestamp("2024-01-02")
    assert under.trough == pd.Timestamp("2024-01-03")
    assert under.recovery == pd.Timestamp("2024-01-05")
    assert under.recovery_days == 2
    assert under.longest_days == 3
    assert under.current_days == 1
    falling = pd.Series(
        [1.0, 2.0, 1.5, 1.0], index=pd.date_range("2024-01-01", periods=4)
    )
    lost = perf.underwater(falling)
    assert lost.max_drawdown == pytest.approx(-0.5)
    assert lost.recovery is None
    assert lost.recovery_days is None
    assert lost.longest_days == lost.current_days == 2


# ---------------------------------------------------------------- money


def test_under_a_year_is_not_annualised() -> None:
    short = perf.annualise(0.05, 364)
    long = perf.annualise(0.21, 730)

    assert short.annual is None
    assert not short.annualised
    assert short.cumulative == 0.05
    assert long.annualised
    assert long.annual == pytest.approx(0.1)


def test_91_day_rates_are_made_effective_by_how_they_are_quoted() -> None:
    five = pd.Series([5.0])
    # ^IRX: bank discount, price 1 - 0.05 x 91/360 = 0.987361; (1/price)^(365/91).
    bill = float(perf.effective_annual(five, "discount").iloc[0])
    # CD 91-day: simple on 365 days, (1 + 0.05 x 91/365)^(365/91).
    cd = float(perf.effective_annual(five, "simple").iloc[0])

    assert bill == pytest.approx(5.2341, abs=1e-4)
    assert cd == pytest.approx(5.0946, abs=1e-4)
    assert bill > cd > 5.0


def test_fx_parts_sum_to_base() -> None:
    gen = rng()
    w = gen.dirichlet([1, 1, 1], 60)
    local = gen.normal(0, 0.01, (60, 3))
    fx = np.column_stack([np.zeros(60), gen.normal(0, 0.005, (60, 2))])
    split = perf.fx_split(w, local, fx)

    assert split.local + split.fx + split.cross == pytest.approx(split.base, abs=1e-15)
    chained = np.prod(1 + (w * perf.to_base(local, fx)).sum(axis=1)) - 1
    assert split.base == pytest.approx(chained)


def test_tracking_error_and_information_ratio() -> None:
    gen = rng()
    bench = gen.normal(0, 0.01, 300)
    rets = bench + gen.normal(0.0002, 0.002, 300)
    te = perf.tracking_error(rets, bench)
    ir = perf.information_ratio(rets, bench)

    assert te is not None
    assert ir is not None
    active = rets - bench
    assert te == pytest.approx(active.std(ddof=1) * math.sqrt(252))
    assert ir == pytest.approx(active.mean() * 252 / te)
    assert perf.tracking_error(rets[:251], bench[:251]) is None
    assert perf.information_ratio(bench, bench) is None
    # A Korea/US union calendar has about 258 returns a year.
    assert perf.tracking_error(rets, bench, 258) == pytest.approx(
        active.std(ddof=1) * math.sqrt(258)
    )
    assert perf.information_ratio(rets, bench, 258) == pytest.approx(
        active.mean() * math.sqrt(258) / active.std(ddof=1)
    )


def test_active_share_is_half_the_absolute_gap() -> None:
    a = pd.Series({"A": 0.5, "B": 0.5})

    assert perf.active_share(a, a) == 0
    assert perf.active_share(a, pd.Series({"C": 1.0})) == pytest.approx(1.0)
    assert perf.active_share(a, pd.Series({"A": 1.0})) == pytest.approx(0.5)


def test_timing_gap_is_zero_without_flows() -> None:
    for span in (200, 400):
        money = perf.mwr([0, span], [-100, 130])
        time = perf.annualise(0.30, span)
        assert money is not None
        gap = perf.timing_gap(money, time)
        assert gap.value == pytest.approx(0.0, abs=1e-9)
        assert gap.annualised == (span >= 365)


def test_counterfactual_buys_the_starting_mix() -> None:
    prices = np.array([[10.0, 20.0], [11.0, 18.0], [12.0, 22.0]])
    shares = np.array([3.0, 1.0])
    flows = np.array([0.0, 100.0, 0.0])
    value = perf.counterfactual(prices, shares, flows)

    assert value[0] == pytest.approx(50.0)
    assert value[1] == pytest.approx(51.0 + 100.0)
    bought = shares * (1 + 100.0 / 51.0)
    assert value[2] == pytest.approx(float(prices[2] @ bought))
    assert perf.counterfactual(prices, shares, np.zeros(3)) == pytest.approx(
        prices @ shares
    )


def test_trading_effect_is_zero_without_trades() -> None:
    prices = np.array([[10.0, 20.0], [11.0, 18.0], [12.0, 22.0], [13.0, 21.0]])
    shares = np.array([3.0, 1.0])
    flows = np.array([0.0, 100.0, -40.0, 0.0])
    held = perf.counterfactual(prices, shares, flows)
    # The real book does the same: each flow buys or sells the held mix.
    count = shares.copy()
    for t in range(1, 4):
        count *= 1 + flows[t] / float(prices[t] @ count)
    actual = float(prices[-1] @ count)
    days = [0.0, 100.0, 300.0, 500.0]
    cash = [-float(prices[0] @ shares), -100.0, 40.0, 0.0]
    effect = perf.trading_effect(days, cash, actual, float(held[-1]))

    assert effect is not None
    assert effect.annualised
    assert effect.value == pytest.approx(0.0, abs=1e-9)
    worse = perf.trading_effect(days, cash, actual * 0.9, float(held[-1]))
    assert worse is not None
    assert worse.value < 0


# ---------------------------------------------------------------- style


def test_simplex_projection() -> None:
    w = perf.project_simplex([0.9, 0.4, -0.3])

    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    assert w == pytest.approx([0.75, 0.25, 0.0])
    assert perf.project_simplex([0.2, 0.3, 0.5]) == pytest.approx([0.2, 0.3, 0.5])


def test_style_recovers_planted_weights() -> None:
    gen = rng()
    cov = np.full((4, 4), 0.5) + 0.5 * np.eye(4)
    proxies = gen.multivariate_normal(np.zeros(4), cov * 0.02**2, 300)
    planted = np.array([0.5, 0.3, 0.2, 0.0])
    fund = proxies @ planted + gen.normal(0, 0.001, 300)
    fit = perf.style_fit(fund, proxies)

    assert np.abs(fit.weights - planted).max() < 0.02
    assert np.all(fit.weights >= 0)
    assert fit.weights.sum() == pytest.approx(1.0)
    assert fit.r2 is not None
    assert fit.r2 > 0.95
    selection = perf.selection_returns(fund, proxies, 52)
    assert len(selection) == 300 - 52
    assert np.abs(selection).mean() < 0.005


# ---------------------------------------------------------------- simulation


def test_path_risk_measures_falls_and_rises_three_ways() -> None:
    paths = np.array(
        [
            [0.95, 0.88],  # falls 12%: at the end, on a day, from the high
            [0.88, 0.95],  # down 12% on a day, 5% at the end
            [1.15, 1.02],  # 11% off its high, never below today; up 15% on a day
            [0.92, 1.03],  # up 12% from its low, never 10% above today
            [1.12, 1.13],  # up 13% at the end
        ]
    )
    risk = perf.path_risk(paths, current=1.0, peak=1.0)

    assert risk.falls[0.10] == perf.Reach(end=0.2, touched=0.4, swing=0.6)
    assert risk.rises[0.10] == perf.Reach(end=0.2, touched=0.4, swing=0.6)
    assert risk.falls[0.20] == risk.rises[0.20] == perf.Reach(0.0, 0.0, 0.0)
    assert risk.recovery_median is None  # at the peak already


def test_path_risk_shares_are_nested() -> None:
    rng = np.random.default_rng(7)
    steps = np.exp(rng.normal(0.0, 0.03, (2000, 63)))
    paths = np.cumprod(steps, axis=1)
    paths[:5, 30:] = 0.0  # wiped out after rising or falling
    risk = perf.path_risk(paths, current=1.0, peak=1.0, levels=(0.05, 0.1, 0.2, 0.3))

    for side in (risk.falls, risk.rises):
        for r in side.values():
            assert 0 <= r.end <= r.touched <= r.swing <= 1
    assert risk.falls[0.1].touched < risk.falls[0.1].swing  # the measures differ
    assert risk.rises[0.1].end < risk.rises[0.1].touched


def test_path_risk_counts_drawdowns_and_recoveries() -> None:
    paths = np.array(
        [
            [1.00, 1.10, 1.20, 1.25],  # back at the peak in period 4, the last
            [0.95, 0.85, 0.80, 0.90],  # falls 20%, never back
            [1.25, 1.00, 1.30, 1.30],  # back in period 1, then down 20%
            [0.85, 1.25, 1.25, 1.25],  # down 15%, back in period 2
        ]
    )
    risk = perf.path_risk(paths, current=0.8, peak=1.0)

    assert risk.falls[0.10].swing == 0.75
    assert risk.falls[0.30].swing == 0.0
    assert risk.never == 0.25
    assert risk.recovery_low == 1
    assert risk.recovery_median == 2
    assert risk.recovery_high is None  # past the horizon


# ---------------------------------------------------------------- clones


def test_shrinkage_limits() -> None:
    estimates = np.array([0.01, 0.03, 0.02])

    pooled = perf.shrink(estimates, [0.1, 0.1, 0.1])  # noise swamps the spread
    assert pooled == pytest.approx([0.02] * 3)
    assert perf.shrink(estimates, [0.0, 0.0, 0.0]) == pytest.approx(estimates)
    assert perf.shrink([0.02, 0.02], [0.0, 0.0]) == pytest.approx([0.02, 0.02])
    partial = perf.shrink(estimates, [0.005, 0.005, 0.005])
    assert np.all(np.abs(partial - 0.02) < np.abs(estimates - 0.02) + 1e-15)
