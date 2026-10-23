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


def test_ratios_known_answers() -> None:
    assert perf.sortino(SORTINO_CASE) == pytest.approx(0.22361, abs=1e-5)
    assert perf.omega(SORTINO_CASE) == pytest.approx(4 / 3)
    assert perf.ulcer_index([-0.10, 0.0]) == pytest.approx(10.0)


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
    assert perf.sharpe_ci(rets[:251]) is None
    assert perf.psr(rets[:251]) is None
    assert perf.psr(rets) is not None


def test_sharpe_none_without_spread() -> None:
    assert perf.sharpe([0.01] * 5) is None
    assert perf.sortino([0.01, 0.02]) is None
    assert perf.omega([0.01, 0.02]) is None


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


def test_drawdown_includes_starting_capital() -> None:
    assert perf.drawdown_series([-0.1, 0.05]) == pytest.approx([-0.1, -0.055])


def test_cdar_limits() -> None:
    rets = rng().normal(0.0, 0.02, 300)
    mdd = perf.max_drawdown(rets)
    avg = perf.average_drawdown(rets)

    assert perf.cdar(rets, 1.0) == pytest.approx(mdd)
    assert perf.cdar(rets, 0.0) == pytest.approx(avg)
    assert mdd <= perf.cdar(rets, 0.95) <= perf.cdar(rets, 0.5) <= avg


def test_cdar_takes_exactly_the_tail_share() -> None:
    # 5% of 100 is 5 observations, not 6 from floating-point noise.
    rets = [-0.01] * 100
    dd = np.sort(perf.drawdown_series(rets))

    assert perf.cdar(rets, 0.95) == pytest.approx(dd[:5].mean())


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


def test_underwater_not_recovered() -> None:
    wealth = pd.Series(
        [1.0, 2.0, 1.5, 1.0], index=pd.date_range("2024-01-01", periods=4)
    )
    under = perf.underwater(wealth)

    assert under.max_drawdown == pytest.approx(-0.5)
    assert under.recovery is None
    assert under.recovery_days is None
    assert under.longest_days == under.current_days == 2


def test_calmar_needs_a_year() -> None:
    rets = [0.1, -0.2, 0.15]

    assert perf.calmar(rets, 364) is None
    calmar = perf.calmar(rets, 365)
    assert calmar == pytest.approx((1.1 * 0.8 * 1.15 - 1) / 0.2)


# ---------------------------------------------------------------- money


def test_under_a_year_is_not_annualised() -> None:
    short = perf.annualise(0.05, 364)
    long = perf.annualise(0.21, 730)

    assert short.annual is None
    assert not short.annualised
    assert short.cumulative == 0.05
    assert long.annualised
    assert long.annual == pytest.approx(0.1)


def test_mwr_under_a_year_is_holding_period() -> None:
    result = perf.mwr([0, 182], [-100, 105])

    assert result is not None
    assert result.annual is None
    assert result.cumulative == pytest.approx(0.05, abs=1e-9)


def test_xirr_needs_a_sign_change() -> None:
    assert perf.xirr([0, 400], [-100, -50]) is None
    assert perf.mwr([0, 400], [100, 50]) is None


def test_daily_risk_free_carries_rates_forward() -> None:
    rates = pd.Series([3.65, 5.0], index=pd.DatetimeIndex(["2024-01-02", "2024-01-05"]))
    index = pd.date_range("2024-01-01", "2024-01-06")
    rf = perf.daily_rf(rates, index)

    assert math.isnan(float(rf.iloc[0]))
    daily = 1.0365 ** (1 / 252) - 1
    assert rf.iloc[1:4].to_numpy() == pytest.approx([daily] * 3)
    assert rf.iloc[4:].to_numpy() == pytest.approx([1.05 ** (1 / 252) - 1] * 2)


def test_fx_parts_sum_to_base() -> None:
    gen = rng()
    w = gen.dirichlet([1, 1, 1], 60)
    local = gen.normal(0, 0.01, (60, 3))
    fx = np.column_stack([np.zeros(60), gen.normal(0, 0.005, (60, 2))])
    split = perf.fx_split(w, local, fx)

    assert split.local + split.fx + split.cross == pytest.approx(split.base, abs=1e-15)
    chained = np.prod(1 + (w * perf.to_base(local, fx)).sum(axis=1)) - 1
    assert split.base == pytest.approx(chained)


def test_blend_with_pure_weights_is_one_leg() -> None:
    kr = np.array([0.01, -0.02, 0.005])
    us = np.array([0.002, 0.01, -0.01])

    assert perf.blend(1.0, kr, us) == pytest.approx(kr)
    assert perf.blend(0.0, kr, us) == pytest.approx(us)
    assert perf.blend([1.0, 0.0, 0.5], kr, us) == pytest.approx([0.01, 0.01, -0.0025])


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


def test_style_with_one_proxy_is_all_in() -> None:
    gen = rng()
    proxy = gen.normal(0, 0.02, (100, 1))
    fit = perf.style_fit(gen.normal(0, 0.02, 100), proxy)

    assert fit.weights == pytest.approx([1.0])


# ---------------------------------------------------------------- simulation


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

    assert risk.drawdown_odds[0.10] == 0.75
    assert risk.drawdown_odds[0.20] == 0.0
    assert risk.drawdown_odds[0.30] == 0.0
    assert risk.never == 0.25
    assert risk.recovery_low == 1
    assert risk.recovery_median == 2
    assert risk.recovery_high is None  # past the horizon


def test_path_risk_at_the_peak_has_no_recovery() -> None:
    paths = rng().lognormal(0, 0.05, (1000, 20)).cumprod(axis=1)
    risk = perf.path_risk(paths, current=1.0, peak=1.0)

    assert risk.recovery_median is None
    assert risk.never == 0.0
    assert 0 <= risk.drawdown_odds[0.30] <= risk.drawdown_odds[0.10] <= 1


# ---------------------------------------------------------------- clones


def test_shrinkage_limits() -> None:
    estimates = np.array([0.01, 0.03, 0.02])

    pooled = perf.shrink(estimates, [0.1, 0.1, 0.1])  # noise swamps the spread
    assert pooled == pytest.approx([0.02] * 3)
    assert perf.shrink(estimates, [0.0, 0.0, 0.0]) == pytest.approx(estimates)
    assert perf.shrink([0.02, 0.02], [0.0, 0.0]) == pytest.approx([0.02, 0.02])
    partial = perf.shrink(estimates, [0.005, 0.005, 0.005])
    assert np.all(np.abs(partial - 0.02) < np.abs(estimates - 0.02) + 1e-15)


def test_top_weights() -> None:
    values = pd.Series([50.0, 30.0, 20.0, 10.0], index=["A", "B", "A", "C"])

    value = perf.top_weights(values, 2)
    assert value.to_dict() == pytest.approx({"A": 70 / 100, "B": 30 / 100})
    equal = perf.top_weights(values, 2, equal=True)
    assert equal.to_dict() == {"A": 0.5, "B": 0.5}


def test_turnover_after_drift() -> None:
    old = pd.Series({"A": 0.5, "B": 0.5})
    drifted = perf.drift(old, pd.Series({"A": 1.5, "B": 0.5}))

    assert drifted.to_dict() == pytest.approx({"A": 0.75, "B": 0.25})
    assert perf.turnover(old, drifted) == pytest.approx(0.25)
    assert perf.turnover(drifted, drifted) == 0
