"""Risk math with known answers and properties on synthetic returns."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest

from finagent import quant

if TYPE_CHECKING:
    from numpy.typing import NDArray


def normal(days: int, cov: NDArray[np.float64], seed: int = 1) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(len(cov)), cov, size=days)


COV = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]) * 1e-4
W = np.array([0.5, 0.3, 0.2])


# ---------------------------------------------------------------- Monte Carlo


def test_block_length_follows_the_window_and_the_market_mix() -> None:
    assert quant.block_length(126, mixed=False) == 5
    assert quant.block_length(252, mixed=False) == 6
    assert quant.block_length(756, mixed=False) == 9
    assert quant.block_length(126, mixed=True) == 10
    assert quant.block_length(60, mixed=True) == 6  # at most n / 10


def test_stationary_blocks_average_the_mean_length_and_wrap() -> None:
    n, length = 1000, 8.0
    picks = quant.stationary_indices(n, 400, 250, length, seed=3)
    runs_on = picks[:, 1:] == (picks[:, :-1] + 1) % n
    starts = runs_on.size - int(runs_on.sum())  # after the first day

    assert picks.shape == (400, 250)
    assert runs_on.size / starts == pytest.approx(length, rel=0.03)
    assert picks.min() >= 0
    assert picks.max() < n
    small = quant.stationary_indices(10, 50, 40, 20.0, seed=3)
    assert ((small[:, :-1] == 9) & (small[:, 1:] == 0)).any()


def test_draws_are_fixed_by_the_seed() -> None:
    first = quant.stationary_indices(300, 20, 30, 5.0, seed=11)

    assert np.array_equal(first, quant.stationary_indices(300, 20, 30, 5.0, seed=11))
    assert not np.array_equal(
        first, quant.stationary_indices(300, 20, 30, 5.0, seed=12)
    )


def test_growth_paths_equal_a_per_asset_replay_with_daily_rebalancing() -> None:
    rets = normal(60, COV) + 0.002
    picks = quant.stationary_indices(60, 50, 30, 5.0)
    growth = quant.growth_paths(rets, W, picks)

    demeaned = rets @ W - (rets @ W).mean()
    by_hand = np.cumprod(1 + demeaned[picks], axis=1)
    assert growth.shape == (50, 31)
    assert np.allclose(growth[:, 0], 1.0)
    assert np.allclose(growth[:, 1:], by_hand)


def test_fhs_paths_start_at_todays_volatility_and_revert() -> None:
    rets = normal(800, COV, seed=2)
    rets[-40:] *= 3  # a volatile last two months
    f = quant.devolatilise(rets)
    scale = quant.revolatilise(f.now, f.long_run, 2000)

    assert np.allclose(scale[0], f.now)
    assert np.allclose(scale[-1], f.long_run, rtol=1e-6)
    assert (f.now > f.long_run).all()
    picks = quant.stationary_indices(800, 300, 21, 5.0)
    calm = quant.growth_paths(rets, W, picks)
    stormy = quant.growth_paths(f.z, W, picks, scale[:21])
    assert stormy[:, -1].std() > 1.5 * calm[:, -1].std()


# ---------------------------------------------------------------- covariance


def test_lag_adjustment_recovers_a_planted_lead_and_stays_psd() -> None:
    rng = np.random.default_rng(4)
    us = rng.standard_normal(3000) * 0.01
    kr = 0.5 * np.concatenate([[0.0], us[:-1]]) + rng.standard_normal(3000) * 0.01
    cov, f = quant.lagged_covariance(np.column_stack([kr, us]), np.array([True, False]))

    assert f == 1.0
    assert cov[0, 1] == pytest.approx(0.5e-4, rel=0.1)
    wide = rng.standard_normal((20, 30)) * 0.01  # fewer days than assets
    korean = np.arange(30) < 15
    cov, f = quant.lagged_covariance(wide, korean)
    assert 0.0 <= f < 1.0
    assert np.linalg.eigvalsh(cov).min() >= -1e-12


def test_ledoit_wolf_keeps_variances_and_pulls_toward_the_mean_correlation() -> None:
    rets = normal(120, COV, seed=5)
    shrunk, weight = quant.ledoit_wolf(rets)
    y = rets - rets.mean(axis=0)
    sample = y.T @ y / len(y)

    assert 0.0 < weight <= 1.0
    assert np.allclose(np.diag(shrunk), np.diag(sample))
    corr, raw = quant.correlation(shrunk), quant.correlation(sample)
    mean = (raw.sum() - 3) / 6
    assert np.all(np.abs(corr - mean) <= np.abs(raw - mean) + 1e-12)
    assert np.linalg.eigvalsh(shrunk).min() > 0


# ---------------------------------------------------------------- beta


def test_dimson_recovers_a_planted_lagged_beta() -> None:
    rng = np.random.default_rng(6)
    m = rng.standard_normal(2000) * 0.01
    r = 0.6 * m + 0.4 * np.concatenate([[0.0], m[:-1]])
    r += rng.standard_normal(2000) * 0.005

    assert quant.dimson_beta(r, m) == pytest.approx(1.0, abs=0.05)
    assert float(np.polyfit(m, r, 1)[0]) == pytest.approx(0.6, abs=0.05)


def test_welch_beta_shrugs_off_an_outlier_and_shrinkage_pulls_to_one() -> None:
    rng = np.random.default_rng(7)
    m = rng.standard_normal(500) * 0.01
    r = 1.2 * m + rng.standard_normal(500) * 0.002
    r[-1] = 0.5  # one wild print on the most heavily weighted day

    assert quant.welch_beta(r, m) == pytest.approx(1.2, abs=0.1)
    assert quant.shrunk_beta(1.5, 250) == pytest.approx(1.25)
    assert quant.shrunk_beta(1.5, 0) == pytest.approx(1.0)


# ---------------------------------------------------------------- diversification


def test_euler_contributions_add_up_and_a_single_asset_is_undiversified() -> None:
    shares = quant.risk_contributions(W, COV)
    var = float(W @ COV @ W)

    assert float((shares * var).sum()) == pytest.approx(var)
    assert shares @ np.ones(3) == pytest.approx(1.0)
    single = np.array([[4e-4]])
    assert quant.diversification_ratio(np.array([1.0]), single) == pytest.approx(1.0)
    assert quant.diversification_ratio(W, COV) > 1.0
    assert quant.effective_holdings(np.full(4, 0.25)) == pytest.approx(4.0)
    assert quant.effective_holdings(np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_what_if_lowers_volatility_only_below_the_correlation_threshold() -> None:
    sd = np.array([0.01, 0.02, 0.01])
    w = np.array([1.0, 0.0, 0.0])
    betas = np.array([1.2, 0.2, 1.0])
    for rho, lower in ((0.3, True), (0.9, False)):
        corr = np.array([[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cov = corr * np.outer(sd, sd)
        change = quant.what_if(w, cov, betas, 1, 0.01)

        assert change.threshold == pytest.approx(0.5)
        assert change.correlation == pytest.approx(rho)
        assert (change.vol < 0) is lower
        assert change.beta == pytest.approx(0.01 * (0.2 - 1.2))
        assert change.ratio > 0


# ---------------------------------------------------------------- VaR and ES


def test_historical_var_and_es_of_a_known_sample() -> None:
    r = -np.arange(1, 401) / 1000

    result = quant.var_es(r, 0.975)

    assert result is not None
    assert result[0] == pytest.approx(0.391)  # 10th worst of 400
    assert result[1] == pytest.approx(0.3955)  # mean of the worst 10


def test_es_is_never_below_var_and_short_samples_are_withheld() -> None:
    rng = np.random.default_rng(8)
    r = rng.standard_t(4, 1000) * 0.01
    for level in (0.95, 0.975, 0.99):
        result = quant.var_es(r, level)
        assert result is not None
        assert result[1] >= result[0] > 0

    assert quant.var_es(r[:249], 0.95) is None
    assert quant.var_es(r[:250], 0.95) is not None
    assert quant.var_es(r[:399], 0.975) is None
    assert quant.var_es(r[:999], 0.99) is None


def test_fhs_equals_hs_when_volatility_is_constant() -> None:
    rng = np.random.default_rng(9)
    r = 0.01 * rng.choice([-1.0, 1.0], 500)

    sigma, now = quant.ewma_sigma(r)
    hs = quant.var_es(r, 0.95)
    fhs = quant.var_es(quant.filtered_returns(r), 0.95)

    assert np.allclose(sigma, 0.01)
    assert float(now) == pytest.approx(0.01)
    assert hs is not None
    assert fhs is not None
    assert fhs == pytest.approx(hs, rel=1e-9)


def test_monthly_var_overlapping_and_square_root_scaled() -> None:
    r = np.full(300, -0.01)

    result = quant.month_var(r, 0.95)

    assert result is not None
    assert result[0] == pytest.approx(1 - 0.99**21)
    assert result[1] == pytest.approx(math.sqrt(21) * 0.01)
    assert quant.month_var(r[:200], 0.95) is None


def test_component_es_sums_to_total_es() -> None:
    rets = normal(1000, COV, seed=10)

    parts = quant.component_es(rets, W)
    total = quant.var_es(rets @ W, 0.975)

    assert parts is not None
    assert total is not None
    assert float(parts.sum()) == pytest.approx(total[1])
    assert (parts > 0).all()


def test_incremental_es_of_cash_and_of_the_portfolio_itself() -> None:
    rets = normal(500, COV, seed=11)
    total = quant.var_es(rets @ W, 0.975)

    cash = quant.incremental_es(rets, W, np.zeros(500), 0.1)
    itself = quant.incremental_es(rets, W, rets @ W, 0.1)

    assert total is not None
    assert cash == pytest.approx(-0.1 * total[1])
    assert itself == pytest.approx(0.0, abs=1e-15)


def test_es_band_brackets_the_estimate_and_repeats() -> None:
    r = np.random.default_rng(12).standard_normal(500) * 0.01

    band = quant.es_band(r, paths=300)
    point = quant.var_es(r, 0.975)

    assert band is not None
    assert point is not None
    assert band[0] < point[1] < band[1]
    assert quant.es_band(r, paths=300) == band
    assert quant.es_band(r[:399]) is None


# ---------------------------------------------------------------- backtest


def test_kupiec_with_no_exceptions_in_250_days() -> None:
    stat, p = quant.kupiec(0, 250, 0.05)

    assert stat == pytest.approx(-2 * 250 * math.log(0.95))
    assert stat == pytest.approx(25.65, abs=0.01)
    assert p == pytest.approx(math.erfc(math.sqrt(stat / 2)))
    assert p < 1e-5
    assert quant.kupiec(12, 240, 0.05) == pytest.approx((0.0, 1.0))


def test_traffic_light_follows_the_binomial_zones() -> None:
    assert quant.traffic_light(4, 250, 0.01) == "green"
    assert quant.traffic_light(5, 250, 0.01) == "yellow"
    assert quant.traffic_light(9, 250, 0.01) == "yellow"
    assert quant.traffic_light(10, 250, 0.01) == "red"


def test_christoffersen_flags_clustered_exceptions() -> None:
    clustered = np.zeros(250, dtype=np.bool_)
    clustered[100:110] = True
    spread = np.zeros(250, dtype=np.bool_)
    spread[::25] = True

    assert quant.christoffersen(clustered)[1] < 0.001
    assert quant.christoffersen(spread)[1] > 0.1
    assert quant.christoffersen(np.zeros(250, dtype=np.bool_)) == (0.0, 1.0)


def test_backtest_counts_days_beyond_the_previous_days_var() -> None:
    r = 0.01 * np.where(np.arange(500) % 2 == 0, 1.0, -1.0)
    shocks = [300, 340, 380, 420, 460]
    r[shocks] = -0.05

    result = quant.backtest(r)

    assert result is not None
    assert result.exceptions == len(shocks)
    assert result.expected == pytest.approx(12.5)
    assert result.light == "green"
    assert quant.backtest(r[:499]) is None


# ---------------------------------------------------------------- stress


def test_union_window_spans_both_peaks_and_troughs() -> None:
    days = pd.bdate_range("2020-01-01", "2020-12-31")
    t = np.arange(len(days), dtype=np.float64)
    us = pd.Series(100 + 20 * np.exp(-(((t - 40) / 30) ** 2)) - 0.1 * t, index=days)
    kr = cast("pd.Series", us.shift(10))  # Korea peaks and bottoms later
    us.loc["2020-05-01":] += 50  # recovery
    kr.loc["2020-05-15":] += 50
    peak = cast("pd.Timestamp", pd.Timestamp("2020-02-01"))
    trough = cast("pd.Timestamp", pd.Timestamp("2020-04-15"))

    window = quant.union_window(us, kr, peak, trough)

    assert window is not None
    start, end = window
    for prices in (us, kr):
        inside = prices.loc[start:end]
        assert float(inside.min()) == pytest.approx(
            float(prices.loc[:"2020-06-30"].min())
        )
    assert start == us.loc["2020-01-01":"2020-04-01"].idxmax()
    assert end == kr.loc[:"2020-05-14"].idxmin()


def test_replay_reports_the_end_change_and_the_worst_point() -> None:
    prices = np.array([[100.0, 10.0], [120.0, 10.0], [60.0, 10.0], [90.0, 10.0]])

    result = quant.replay(prices, np.array([1.0, 0.0]))

    assert result.change == pytest.approx(-0.1)
    assert result.worst == pytest.approx(-0.5)


def test_hypothetical_shock_converts_currency_multiplicatively() -> None:
    w = np.array([0.5, 0.5])
    betas = np.array([1.0, 2.0])
    moves = np.array([-0.1, -0.1])
    fx = np.array([0.05, 0.0])

    change = quant.hypothetical(w, betas, moves, fx)

    assert change == pytest.approx(0.5 * (0.9 * 1.05 - 1) + 0.5 * -0.2)


def test_reverse_stress_round_trips() -> None:
    w = np.array([0.4, 0.6])
    betas = np.array([0.8, 1.3])
    fx = np.array([0.02, 0.0])

    shock = quant.reverse_stress(w, betas, 0.2, fx)

    assert shock is not None
    moves = np.full(2, shock)
    assert quant.hypothetical(w, betas, moves, fx) == pytest.approx(-0.2)
    assert quant.reverse_stress(w, np.array([0.1, 0.1]), 0.5, np.zeros(2)) is None


def test_bad_prints_flag_only_jumps_that_revert() -> None:
    prints = [100.0, 101.0, 10.0, 100.0, 180.0, 110.0, 40.0, 55.0, 56.0]

    flagged = quant.bad_prints(np.array(prints))

    assert np.flatnonzero(flagged).tolist() == [2, 4]  # not the crash to 40


# ---------------------------------------------------------------- turbulence


def test_turbulence_of_iid_normal_returns_averages_one() -> None:
    cov = np.array(
        [
            [1.0, 0.6, 0.3, 0.1],
            [0.6, 1.0, 0.4, 0.2],
            [0.3, 0.4, 1.0, 0.5],
            [0.1, 0.2, 0.5, 1.0],
        ]
    )
    rets = normal(1750, cov * 1e-4, seed=13)

    result = quant.turbulence(rets)
    strided = quant.turbulence(rets, refit=21)

    assert len(result.index) == 1500
    assert float(result.index.mean()) == pytest.approx(1.0, abs=0.1)
    assert float(strided.index.mean()) == pytest.approx(1.0, abs=0.1)
    assert float(np.median(result.surprise)) == pytest.approx(1.0, abs=0.2)
    assert quant.percentile_rank(result.index, float(np.median(result.index))) == (
        pytest.approx(0.5, abs=0.01)
    )


def test_turbulence_rises_when_correlations_break() -> None:
    cov = np.array([[1.0, 0.9], [0.9, 1.0]]) * 1e-4
    rets = normal(300, cov, seed=14)
    rets[-1] = [0.01, -0.01]  # an ordinary size, an unusual direction

    result = quant.turbulence(rets)

    assert result.index[-1] > 5
    assert result.surprise[-1] > 5
    assert quant.percentile_rank(result.index, float(result.index[-1])) == 1.0
