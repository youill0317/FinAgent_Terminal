"""Risk math with known answers and properties on synthetic returns."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass, replace
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
GJR = np.array([2e-6, 0.03, 0.12, 0.9])  # omega, alpha, gamma, beta


def gjr(days: int, seed: int = 1) -> NDArray[np.float64]:
    """A GJR-GARCH(1,1) series with Student-t(6) shocks, from its long-run level."""
    omega, alpha, gamma, beta = GJR
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(6, days) / math.sqrt(1.5)
    var = omega / (1 - alpha - gamma / 2 - beta)
    out = np.empty(days)
    for t in range(days):
        out[t] = math.sqrt(var) * shocks[t]
        var = omega + (alpha + gamma * (out[t] < 0)) * out[t] ** 2 + beta * var
    return out


@pytest.fixture
def no_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import arch`` fails, as without the garch extra: every filter is EWMA."""
    monkeypatch.setitem(sys.modules, "arch", None)


# ---------------------------------------------------------------- Monte Carlo


def ar1(days: int, phi: float, seed: int = 1) -> NDArray[np.float64]:
    shocks = np.random.default_rng(seed).standard_normal(days)
    out = np.empty(days)
    out[0] = shocks[0]
    for t in range(1, days):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def test_block_length_grows_with_dependence_within_its_bounds() -> None:
    noise = np.random.default_rng(1).standard_normal(750)

    assert quant.block_length(noise) < 3
    assert quant.block_length(ar1(750, 0.3)) > quant.block_length(noise)
    assert quant.block_length(gjr(750)) > quant.politis_white(gjr(750))  # squares
    assert quant.block_length(ar1(2000, 0.9)) == quant.MAX_BLOCK
    assert quant.block_length(ar1(100, 0.98)) == 10  # n / 10
    assert quant.block_length(np.zeros(100)) == 1


def test_politis_white_matches_arch() -> None:
    bootstrap = pytest.importorskip("arch.bootstrap")
    for phi in (-0.3, 0.0, 0.5, 0.9):
        x = ar1(1000, phi, seed=2)
        for series in (x, x**2):
            expected = bootstrap.optimal_block_length(series).iloc[0, 0]
            assert quant.politis_white(series) == pytest.approx(expected, rel=1e-9)


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


def test_growth_paths_equal_a_per_asset_replay_with_daily_rebalancing() -> None:
    rets = normal(60, COV) + 0.002
    picks = quant.stationary_indices(60, 50, 30, 5.0)
    growth = quant.growth_paths(rets, W, picks)

    demeaned = rets @ W - (rets @ W).mean()
    by_hand = np.cumprod(1 + demeaned[picks], axis=1)
    assert growth.shape == (50, 31)
    assert np.allclose(growth[:, 0], 1.0)
    assert np.allclose(growth[:, 1:], by_hand)


def signs(days: int, assets: int = 3) -> NDArray[np.float64]:
    """Residuals of +1 and -1 by turns: every shock is one volatility."""
    return np.tile([[1.0], [-1.0]], (days // 2, assets))


@pytest.mark.usefixtures("no_arch")
def test_ewma_paths_start_at_todays_volatility_and_revert() -> None:
    rets = normal(800, COV, seed=2)
    rets[-40:] *= 3  # a volatile last two months
    f = quant.devolatilise(rets)
    long_run = np.sqrt(np.mean(rets**2, axis=0))

    assert not f.garch.any()
    assert (f.now > long_run).all()
    picks = quant.stationary_indices(800, 300, 252, 5.0)
    first = np.eye(3)[0]
    flat = replace(f, z=signs(800))
    sd = np.abs(quant.filtered_paths(flat, first, picks, f.now**2))
    k = np.arange(252)
    phi = quant.REVERSION
    path = np.sqrt(long_run[0] ** 2 + phi**k * (f.now[0] ** 2 - long_run[0] ** 2))
    assert np.allclose(sd, path)  # the same on every path
    calm = quant.growth_paths(rets, W, picks[:, :21])
    stormy = quant.growth_paths(f.z, W, picks[:, :21], f)
    assert stormy[:, -1].std() > 1.5 * calm[:, -1].std()


def test_gjr_paths_follow_their_own_shocks_then_the_expectation() -> None:
    step = np.column_stack([GJR, GJR])
    f = quant.Devolatilised(
        signs(2, 2), np.ones((2, 2)), np.ones(2), step, np.ones(2, dtype=np.bool_)
    )
    picks = np.array([[0, 1, 0, 0, 1], [1, 1, 0, 1, 1]])  # +1, -1, +1, ...
    start = np.array([1e-4, 4e-4])
    out = quant.filtered_paths(f, np.array([1.0, 0.0]), picks, start, feedback=3)

    omega, alpha, gamma, beta = GJR
    for path in range(2):
        var = start[0]
        for k in range(5):
            sign = 1.0 if picks[path, k] == 0 else -1.0
            assert out[path, k] == pytest.approx(sign * math.sqrt(var))
            moved = gamma * (sign < 0) if k < 3 else gamma / 2
            var = omega + (alpha + moved + beta) * var


def test_a_year_of_fhs_paths_stays_near_the_historical_years(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ten years of a persistent GJR process (0.99) with a two-year storm and
    # a 1987-sized day: from the average variance, a simulated year's 95% and
    # 99% losses stay near the rolling 252-day ones. A year of shock
    # feedback overshoots the 99% by half.
    r = gjr(2520, seed=2)
    r[500:1000] *= 2.0
    r[1500] = -8 * float(r.std())

    def true_process(_: object) -> NDArray[np.float64]:
        return GJR

    monkeypatch.setattr(quant, "gjr_fit", true_process)
    f = quant.devolatilise(r[:, None])
    picks = quant.stationary_indices(len(r), 5000, 252, 1)
    start = np.mean(r**2, keepdims=True)
    rolling = np.expm1(np.convolve(np.log1p(r - r.mean()), np.ones(252), "valid"))
    history = -np.quantile(rolling, [0.05, 0.01])

    def year(feedback: int | None) -> NDArray[np.float64]:
        daily = quant.filtered_paths(f, np.ones(1), picks, start, feedback)
        return -np.quantile(
            np.prod(1 + np.maximum(daily, -1), axis=1) - 1, [0.05, 0.01]
        )

    capped = year(quant.FEEDBACK_DAYS) / history
    assert ((capped > 2 / 3) & (capped < 1.4)).all()
    assert year(None)[1] / history[1] > 1.4
    month = quant.filtered_paths(f, np.ones(1), picks[:, :21], start)
    assert np.array_equal(
        month, quant.filtered_paths(f, np.ones(1), picks[:, :21], start, 21)
    )


def test_gjr_fit_recovers_a_planted_process() -> None:
    _ = pytest.importorskip("arch")
    r = gjr(5000)

    params = quant.gjr_fit(r)
    f = quant.devolatilise(r[:, None])

    assert params is not None
    assert params[0] == pytest.approx(GJR[0], rel=0.5)
    assert params[1:] == pytest.approx(GJR[1:], abs=0.04)
    assert f.garch.tolist() == [True]
    assert np.allclose(f.step[1:, 0], params[1:])
    persistence = params[1] + params[2] / 2 + params[3]
    assert f.step[0, 0] == pytest.approx((1 - persistence) * np.mean(r**2))
    assert float(np.sqrt(np.mean(f.z[100:] ** 2))) == pytest.approx(1.0, abs=0.05)
    assert quant.gjr_fit(r[: quant.MIN_GARCH - 1]) is None  # too short


@pytest.mark.usefixtures("no_arch")
def test_a_filter_fitted_on_early_rows_never_sees_later_ones() -> None:
    r = gjr(1000, seed=4)[:, None]
    fit = 750
    changed = r.copy()
    changed[fit + 10 :] *= 3

    before = quant.devolatilise(r, fit=fit)
    after = quant.devolatilise(changed, fit=fit)

    # Each day's volatility uses only earlier returns, seed included.
    assert np.array_equal(before.sigma[: fit + 11], after.sigma[: fit + 11])
    assert np.array_equal(before.step, after.step)


@pytest.mark.usefixtures("no_arch")
def test_without_arch_every_filter_is_ewma() -> None:
    r = gjr(1000)[:, None]

    f = quant.devolatilise(r)
    sigma, now = quant.ewma_sigma(r)

    assert quant.gjr_fit(r[:, 0]) is None
    assert f.garch.tolist() == [False]
    assert np.allclose(f.sigma, sigma)
    assert np.allclose(f.now, now)


@dataclass(frozen=True, slots=True)
class _Fit:
    params: list[float]  # omega (percent squared), alpha, gamma, beta
    convergence_flag: int


def fake_arch(monkeypatch: pytest.MonkeyPatch, params: list[float], flag: int) -> None:
    class Model:
        def fit(self, **_: object) -> _Fit:
            return _Fit(params, flag)

    def arch_model(_: object, **options: object) -> Model:
        del options
        return Model()

    fake = types.ModuleType("arch")
    fake.arch_model = arch_model  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "arch", fake)


@pytest.mark.parametrize(
    ("params", "flag"),
    [
        ([0.02, 0.05, 0.1, 0.9], 0),  # alpha + gamma / 2 + beta = 1
        ([0.02, 0.05, 0.1, 0.899], 0),  # 0.999: too close to 1
        ([0.02, 0.05, 0.1, 0.8], 4),  # did not converge
        ([math.nan, 0.05, 0.1, 0.8], 0),
    ],
)
def test_a_nonstationary_or_failed_fit_falls_back_to_ewma(
    monkeypatch: pytest.MonkeyPatch, params: list[float], flag: int
) -> None:
    r = gjr(600)
    fake_arch(monkeypatch, params, flag)

    assert quant.gjr_fit(r) is None
    assert quant.devolatilise(r[:, None]).garch.tolist() == [False]
    fake_arch(monkeypatch, [0.02, 0.05, 0.1, 0.8], 0)
    assert quant.gjr_fit(r) == pytest.approx([2e-6, 0.05, 0.1, 0.8])  # percent^2


# ---------------------------------------------------------------- variance models


def test_the_gjr_filter_and_its_paths_are_unchanged_by_the_new_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Values the GJR path gave before HAR and PDV were added, to 1e-12.
    def true_process(_: object) -> NDArray[np.float64]:
        return GJR.copy()

    monkeypatch.setattr(quant, "gjr_fit", true_process)
    r = np.column_stack([gjr(600, seed=3), gjr(600, seed=4)])
    w = np.array([0.7, 0.3])
    f = quant.devolatilise(r, fit=500)
    picks = quant.stationary_indices(300, 50, 30, 2.0)
    g = quant.growth_paths(r[-300:], w, picks, f)

    got = [f.sigma[-1, 0], f.now[1], f.step[0, 0], g[7, -1], g[0, 25]]
    pinned = [
        "0x1.aebadd6d19171p-7",
        "0x1.d040fd63c0672p-8",
        "0x1.23f5b5e991733p-20",
        "0x1.0f05c94401404p+0",
        "0x1.f2ed24a9977ecp-1",
    ]
    np.testing.assert_allclose(
        np.array(got, dtype="float64"),
        [float.fromhex(v) for v in pinned],
        rtol=1e-12,
        atol=0,
    )
    same = quant.devolatilise(r, fit=500, model="gjr")
    assert np.array_equal(same.sigma, f.sigma)
    assert f.model is None


def har_process(
    days: int, coef: tuple[float, ...], seed: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A level HAR-RV-IV variance with gamma noise of mean 1, and its regressor."""
    c, day, week, month, iv = coef
    rng = np.random.default_rng(seed)
    x = np.empty(days)
    x[0] = 1e-4
    for t in range(1, days):  # a persistent positive regressor
        x[t] = 1e-5 + 0.9 * x[t - 1] + 3e-5 * (rng.random() - 0.5)
    level = (c + iv * 1e-4) / (1 - day - week - month)
    rv = np.full(days, level)
    for t in range(quant.HAR_MONTH, days):
        mean = (
            c
            + day * rv[t - 1]
            + week * rv[t - 5 : t].mean()
            + month * rv[t - 22 : t].mean()
            + iv * x[t - 1]
        )
        rv[t] = mean * rng.gamma(4.0, 0.25)
    return rv, x


def test_har_fit_recovers_a_planted_process() -> None:
    coef = (1e-6, 0.3, 0.3, 0.2, 0.5)
    rv, x = har_process(3000, coef)

    got = quant.har_fit(rv, x)
    plain = quant.har_fit(rv)

    assert got[1:] == pytest.approx(coef[1:], abs=0.1)
    assert plain[-1] == 0.0
    # the extra regressor carries what the RV lags alone cannot
    assert got[1:4].sum() < plain[1:4].sum()


def test_a_har_filter_has_unit_residuals_and_never_sees_later_rows() -> None:
    rv, x = har_process(1200, (1e-6, 0.3, 0.3, 0.2, 0.5), seed=2)
    r = (np.sqrt(rv) * np.random.default_rng(3).standard_normal(1200))[:, None]
    fit = 1000
    changed = r.copy()
    changed[fit + 10 :] *= 3
    moved = x.copy()
    moved[-1] *= 5

    f = quant.devolatilise(r, fit=fit, model="har", proxy=rv[:, None], x=x)
    after = quant.devolatilise(changed, fit=fit, model="har", proxy=rv[:, None], x=x)
    news = quant.devolatilise(r, fit=fit, model="har", proxy=rv[:, None], x=moved)

    assert f.model is not None
    assert f.model.kind == "har"
    assert f.garch.tolist() == [True]
    assert np.mean(f.z[:fit] ** 2) == pytest.approx(1.0)
    assert np.array_equal(f.sigma[: fit + 11], after.sigma[: fit + 11])
    assert np.array_equal(f.sigma, news.sigma)  # day t's x enters day t + 1
    assert news.now[0] > f.now[0]
    assert quant.devolatilise(r[:400], model="har").model is None  # EWMA


def pdv_process(days: int, seed: int = 1) -> tuple[NDArray[np.float64], ...]:
    """Returns under a planted 4-factor PDV, and their sigma."""
    b0, b10, b11, b2, theta = PDV[:5]
    decay = PDV[5:]
    rng = np.random.default_rng(seed)
    state = np.array([0.0, 0.0, 1e-4, 1e-4])
    r, sigma = np.empty(days), np.empty(days)
    for t in range(days):
        trend = (1 - theta) * state[2] + theta * state[3]
        sigma[t] = max(b0 + b10 * state[0] + b11 * state[1] + b2 * trend**0.5, 1e-3)
        r[t] = sigma[t] * rng.standard_normal()
        day = np.array([r[t], r[t], r[t] ** 2, r[t] ** 2])
        state = decay * state + (1 - decay) * day
    return r, sigma


# b0, b10, b11, b2, theta2 and decays of half-lives 2, 32 (R1) and 4, 64 (R2)
PDV = np.array([0.002, -1.0, -0.5, 0.7, 0.5, *(0.5 ** (1 / np.array([2, 32, 4, 64])))])


def test_pdv_fit_recovers_a_planted_process() -> None:
    r, sigma = pdv_process(3000)

    got = quant.pdv_fit(r)
    f = quant.devolatilise(r[:, None], model="pdv")

    assert got is not None
    assert got[0] == pytest.approx(PDV[0], abs=1e-3)
    assert got[1] + got[2] == pytest.approx(PDV[1] + PDV[2], rel=0.5)  # leverage
    assert got[3] == pytest.approx(PDV[3], abs=0.15)
    assert np.corrcoef(f.sigma[:, 0], sigma)[0, 1] > 0.95
    assert np.mean(f.z**2) == pytest.approx(1.0)
    assert quant.pdv_fit(np.zeros(600)) is None


def test_pdv_paths_with_an_ewma_kernel_follow_the_ewma_recursion() -> None:
    # sigma = sqrt(R20) with decay 0.94 is EWMA: model_paths then matches
    # filtered_paths under the EWMA recursion, shocks fed back or expected.
    rng = np.random.default_rng(4)
    z = rng.standard_normal((40, 2))
    step = np.column_stack([[0.0, 0.06, 0.0, 0.94]] * 2)
    ewma = quant.Devolatilised(z, np.ones((40, 2)), np.ones(2), step, np.ones(2, bool))
    params = np.column_stack([[0, 0, 0, 1, 0, 0, 0, 0.94, 0.94]] * 2).astype(float)
    model = quant.Recursion("pdv", params, np.zeros((1, 2, 4)), np.zeros(2))
    start = np.array([1e-4, 4e-4])
    state = np.column_stack([np.zeros((2, 2)), start, start])
    picks = rng.integers(0, 40, size=(30, 15))
    w = np.array([0.6, 0.4])

    for feedback in (None, 5):
        expected = quant.filtered_paths(ewma, w, picks, start, feedback)
        got = quant.model_paths(replace(ewma, model=model), w, picks, state, feedback)
        assert np.allclose(got, expected, rtol=1e-12, atol=0)


def test_har_paths_start_today_and_follow_the_forecast_after_feedback() -> None:
    rv, _ = har_process(1500, (1e-6, 0.3, 0.3, 0.2, 0.0), seed=5)
    r = (np.sqrt(rv) * np.random.default_rng(6).standard_normal(1500))[:, None]
    f = quant.devolatilise(r, model="har")
    assert f.model is not None
    flat = replace(f, z=signs(1500, 1))  # every shock one volatility
    picks = quant.stationary_indices(1500, 20, 10, 1)

    paths = quant.model_paths(flat, np.ones(1), picks, f.model.states[-1], 0)

    state = f.model.states[-1]
    for k in range(10):  # no feedback: the HAR forecast, the same on every path
        var = f.model.variance(state)
        assert np.allclose(np.abs(paths[:, k]), np.sqrt(var[0]))
        state = f.model.update(state, np.zeros(1), var)
    assert np.abs(paths[0, 0]) == pytest.approx(f.now[0])
    growth = quant.growth_paths(r, np.ones(1), picks, f)
    assert growth.shape == (20, 11)


# ---------------------------------------------------------------- range-based variance


def gbm_ohlc(
    days: int, steps: int = 390, seed: int = 1
) -> tuple[NDArray[np.float64], ...]:
    """Daily open, high, low, close of a GBM: intraday var 3e-4, overnight 1e-4."""
    rng = np.random.default_rng(seed)
    night = math.sqrt(1e-4) * rng.standard_normal(days)
    moves = math.sqrt(3e-4 / steps) * rng.standard_normal((days, steps))
    logs = np.empty((days, steps + 1))
    level = 0.0
    for d in range(days):
        logs[d] = level + night[d] + np.concatenate([[0.0], np.cumsum(moves[d])])
        level = logs[d, -1]
    prices = 100 * np.exp(logs)
    return prices[:, 0], prices.max(axis=1), prices.min(axis=1), prices[:, -1]


def test_range_estimators_recover_a_gbm_variance() -> None:
    o, h, lo, c = gbm_ohlc(300)
    previous = np.concatenate([[np.nan], c[:-1]])

    assert np.mean(quant.parkinson(h, lo)) == pytest.approx(3e-4, rel=0.1)
    assert np.mean(quant.rogers_satchell(o, h, lo, c)) == pytest.approx(3e-4, rel=0.1)
    assert np.mean(quant.garman_klass(o, h, lo, c)) == pytest.approx(3e-4, rel=0.1)
    full = quant.garman_klass(o, h, lo, c, previous)
    assert np.isnan(full[0])
    assert np.mean(full[1:]) == pytest.approx(4e-4, rel=0.1)
    yz = quant.yang_zhang(o, h, lo, c, 21)
    assert np.isnan(yz[:21]).all()
    assert np.mean(yz[21:]) == pytest.approx(4e-4, rel=0.1)


def test_range_estimators_take_adjusted_prices_and_flag_bad_ones() -> None:
    o, h, lo, c = (np.array([10.0, 10.0, 5.0]) * k for k in (1.0, 1.02, 0.99, 1.01))
    adjusted = np.array([5.05, 5.05, 5.05])  # a 2-for-1 split before day 2
    ao, ah, al, ac = quant.adjusted_ohlc(o, h, lo, c, adjusted)
    previous = np.concatenate([[np.nan], ac[:-1]])

    assert np.allclose(ao, [5.0, 5.0, 5.0])
    assert np.array_equal(ac, adjusted)
    night = quant.garman_klass(ao, ah, al, ac, previous) - quant.garman_klass(
        ao, ah, al, ac
    )
    assert np.allclose(night[1:], math.log(5.0 / 5.05) ** 2)  # no split jump
    flat = np.full(2, 7.0)
    assert np.array_equal(quant.garman_klass(flat, flat, flat, flat), [0.0, 0.0])
    bad = np.array([10.0, 0.0])
    assert np.isnan(quant.parkinson(bad, np.array([9.0, 9.0]))).tolist() == [
        False,
        True,
    ]


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
    # Kish: (sum w)^2 / sum w^2; no decay counts every day, Welch's about a year.
    w = np.exp(-quant.WELCH_DECAY * np.arange(756))
    assert quant.effective_days(756) == pytest.approx(w.sum() ** 2 / (w**2).sum())
    assert quant.effective_days(756) == pytest.approx(254.6, abs=0.1)
    assert quant.effective_days(10_000) < 2 / quant.WELCH_DECAY + 0.01
    assert quant.effective_days(500, decay=0.0) == pytest.approx(500)


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
    r = -np.arange(1, 481) / 1000

    result = quant.var_es(r, 0.975)

    assert result is not None
    assert result[0] == pytest.approx(0.469)  # 12th worst of 480
    assert result[1] == pytest.approx(0.4745)  # mean of the worst 12


def test_es_is_never_below_var_and_short_samples_are_withheld() -> None:
    rng = np.random.default_rng(8)
    r = rng.standard_t(4, 1200) * 0.01
    for level in (0.95, 0.975, 0.99):
        result = quant.var_es(r, level)
        assert result is not None
        assert result[1] >= result[0] > 0

    # Twelve days in the tail: 240, 480 and 1,200 days.
    assert [quant.min_samples(x) for x in (0.95, 0.975, 0.99)] == [240, 480, 1200]
    assert quant.var_es(r[:239], 0.95) is None
    assert quant.var_es(r[:240], 0.95) is not None  # a Korea-only year, ~245 days
    assert quant.var_es(r[:479], 0.975) is None
    assert quant.var_es(r[:1199], 0.99) is None


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


# ---------------------------------------------------------------- EVT


def test_gpd_fit_recovers_known_tails() -> None:
    u = np.random.default_rng(16).random(20_000)
    for xi in (0.3, -0.2):
        y = 2.0 * ((1 - u) ** -xi - 1) / xi  # GPD by inversion
        fitted = quant.gpd_fit(y)
        assert fitted == pytest.approx((xi, 2.0), abs=0.05)
    assert quant.gpd_fit(-np.log1p(-u)) == pytest.approx((0.0, 1.0), abs=0.03)
    # Uniform excesses have xi = -1, beyond the MLE: probability-weighted moments.
    assert quant.gpd_fit(u) == pytest.approx((-1.0, 1.0), abs=0.05)


def test_evt_matches_exponential_and_student_t_tails() -> None:
    rng = np.random.default_rng(17)
    gains = -rng.exponential(0.01, 20_000)  # losses Exp(1%): xi = 0
    got = quant.evt_var_es(gains, 0.99)
    assert got is not None
    assert got[0] == pytest.approx(0.01 * math.log(100), rel=0.03)
    assert got[1] == pytest.approx(got[0] + 0.01, rel=0.03)
    # Student-t(4): 99% quantile 3.7469 and ES 5.2212.
    got = quant.evt_var_es(rng.standard_t(4, 20_000), 0.99)
    assert got is not None
    assert got == pytest.approx((3.7469, 5.2212), rel=0.05)


def test_evt_needs_fifty_excesses_and_a_level_beyond_them() -> None:
    r = np.random.default_rng(18).standard_t(4, 500)

    assert quant.evt_var_es(r, 0.99) is not None  # 50 excesses
    assert quant.evt_var_es(r[:499], 0.99) is None
    assert quant.evt_var_es(r, 0.9) is None  # inside the threshold
    assert quant.evt_intervals(r[:499], 0.99) is None


def test_bootstrap_intervals_cover_the_normal_tail_and_narrow_with_more_days() -> None:
    rng = np.random.default_rng(29)
    levels = (0.95, 0.99)
    # Normal 95%: quantile 1.6449, ES 2.0627.
    short = quant.bootstrap_intervals(rng.standard_normal(600), levels)
    long = quant.bootstrap_intervals(rng.standard_normal(4000), levels)

    assert short[1] is None  # under 1,200 days for 99%
    assert short[0] is not None
    assert long[0] is not None
    assert long[1] is not None
    for var_ci, es_ci in (short[0], long[0]):
        assert var_ci[0] < 1.6449 < var_ci[1]
        assert es_ci[0] < 2.0627 < es_ci[1]
    assert long[0][1][1] - long[0][1][0] < short[0][1][1] - short[0][1][0]
    assert quant.bootstrap_intervals(rng.standard_normal(239), levels) == [None, None]


def test_evt_windows_put_a_gpd_on_each_forecasts_largest_losses() -> None:
    rng = np.random.default_rng(31)
    right = rng.standard_t(4, (250, 1000))  # each day's forecast: its true law
    r = rng.standard_t(4, 250)

    evt = quant.evt_windows(right)

    assert evt is not None
    assert evt.shape == right.shape
    k = 100  # 10% of 1,000 scenarios
    body = np.sort(-right[0])[: 1000 - k]
    assert np.array_equal(np.sort(-evt[0])[: 1000 - k], body)  # body kept
    got = quant.evt_var_es(right[0], 0.95)
    assert got is not None
    var, es = quant.var_es(evt[0], 0.95) or (0.0, 0.0)
    assert (var, es) == pytest.approx(got, rel=0.03)
    fair = quant.backtest(r, evt)
    narrower = quant.evt_windows(0.6 * right)
    assert narrower is not None
    narrow = quant.backtest(r, narrower)
    assert fair.es > 0.05
    assert narrow.es < 0.01
    assert quant.evt_windows(right[:, :499]) is None  # under 50 excesses


def gpd_loglik(y: NDArray[np.float64], xi: float, scale: float) -> float:
    z = 1 + xi * y / scale
    if scale <= 0 or z.min() <= 0:
        return -math.inf
    return float(-len(y) * math.log(scale) - (1 + 1 / xi) * np.log(z).sum())


def test_evt_intervals_end_where_the_profile_likelihood_drops_by_1_92() -> None:
    r = np.random.default_rng(21).standard_t(4, 750)
    found = quant.evt_intervals(r, 0.99)
    point = quant.evt_var_es(r, 0.99)
    assert found is not None
    assert point is not None
    (var_low, var_high), (es_low, es_high) = found
    assert var_low < point[0] < var_high
    assert es_low < point[1] < es_high
    # Recomputed on fine grids of xi and the tail share zeta (binomial, k of n),
    # as in McNeil-Frey-Embrechts (2015, 5.2).
    loss = np.sort(-r)
    u = float(loss[-76])
    y = loss[-75:] - u
    mle = quant.gpd_mle(y)
    assert mle is not None
    top = gpd_loglik(y, *mle)
    xi = np.linspace(-0.4999, 0.9999, 1_501)[None, :]
    share = 75 / 750
    zeta = share + math.sqrt(share * (1 - share) / 750) * np.linspace(-3, 3, 81)
    binomial = 75 * np.log(zeta / share) + 675 * np.log((1 - zeta) / (1 - share))
    g = (((1 - 0.99) / zeta[:, None]) ** -xi - 1) / xi

    def profile(d: float, per: NDArray[np.float64]) -> float:
        scale = d * per
        z = 1 + np.multiply.outer(xi / scale, y)
        with np.errstate(invalid="ignore"):
            loglik = -75 * np.log(scale) - (1 + 1 / xi) * np.log(z).sum(axis=-1)
        loglik = np.where((z > 0).all(axis=-1), loglik, -np.inf)
        return float((loglik + binomial[:, None]).max())

    for end in (var_low, var_high):
        assert profile(end - u, 1 / g) == pytest.approx(top - 1.9207, abs=0.01)
    for end in (es_low, es_high):
        assert profile(end - u, (1 - xi) / (g + 1)) == pytest.approx(
            top - 1.9207, abs=0.01
        )


def test_evt_intervals_cover_a_long_t_tail_and_open_up_for_a_heavy_one() -> None:
    rng = np.random.default_rng(22)
    found = quant.evt_intervals(rng.standard_t(4, 20_000), 0.99)
    assert found is not None
    (var_low, var_high), (es_low, es_high) = found
    assert var_low < 3.7469 < var_high < 1.1 * var_low  # t(4): VaR 3.7469, ES 5.2212
    assert es_low < 5.2212 < es_high < 1.2 * es_low
    # Pareto losses with xi = 0.7 and 50 excesses: the data cannot cap the ES.
    u = np.random.default_rng(2).random(500)
    heavy = quant.evt_intervals(-((1 - u) ** -0.7 - 1) / 0.7, 0.99)
    assert heavy is not None
    assert math.isfinite(heavy[0][1])
    assert heavy[1][1] == math.inf
    # Uniform losses: xi = -1, beyond a regular MLE, so no interval.
    assert quant.evt_intervals(-rng.random(1000), 0.99) is None


# ---------------------------------------------------------------- backtest


def test_kupiec_with_no_exceptions_in_250_days() -> None:
    stat, p = quant.kupiec(0, 250, 0.05)

    assert stat == pytest.approx(-2 * 250 * math.log(0.95))
    assert stat == pytest.approx(25.65, abs=0.01)
    assert p == pytest.approx(math.erfc(math.sqrt(stat / 2)))
    assert p < 1e-5
    assert quant.kupiec(12, 240, 0.05) == pytest.approx((0.0, 1.0))


def test_traffic_light_follows_the_binomial_zones() -> None:
    # Basel (1996): 250 days at 99%, green 0-4, yellow 5-9, red 10 or more.
    basel = ["green"] * 5 + ["yellow"] * 5 + ["red"] * 21
    assert [quant.traffic_light(x, 250, 0.01) for x in range(31)] == basel
    # Elsewhere the zones end where the binomial CDF reaches 95% and 99.99%.
    for days, p in ((250, 0.05), (500, 0.01), (1000, 0.025)):
        cdf = np.cumsum(
            [math.comb(days, i) * p**i * (1 - p) ** (days - i) for i in range(100)]
        )
        yellow = int(np.argmax(cdf >= 0.95))
        red = int(np.argmax(cdf >= 0.9999))
        assert quant.traffic_light(yellow - 1, days, p) == "green"
        assert quant.traffic_light(yellow, days, p) == "yellow"
        assert quant.traffic_light(red - 1, days, p) == "yellow"
        assert quant.traffic_light(red, days, p) == "red"


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

    past = quant.past_windows(r)
    assert past is not None
    result = quant.backtest(r[-250:], past)

    assert result.exceptions == len(shocks)
    assert result.expected == pytest.approx(12.5)
    assert result.light == "green"
    # The k-th shock's window holds k earlier ones: ES = (0.05 k + 0.01 (12.5 - k))
    # / 12.5, and Z2 = 1 - sum(0.05 / ES) / 12.5.
    es = [(0.05 * k + 0.01 * (12.5 - k)) / 12.5 for k in range(5)]
    assert result.z2 == pytest.approx(1 - sum(0.05 / x for x in es) / 12.5)
    assert result.es < 0.01
    assert quant.past_windows(r[:499]) is None


def test_z2_rejects_only_forecasts_that_are_too_narrow() -> None:
    rng = np.random.default_rng(19)
    r = rng.standard_normal(250)
    right = rng.standard_normal((250, 250))  # each day's forecast: its true law

    fair = quant.backtest(r, right)
    narrow = quant.backtest(r, 0.6 * right)
    wide = quant.backtest(r, 1.5 * right)

    assert fair.es > 0.05
    assert narrow.es < 0.01
    assert narrow.z2 < 0 < wide.z2
    assert wide.es > 0.5
    assert fair.de_u > 0.05
    assert narrow.de_u < 0.01


def testdu_escanciano_statistics_of_a_known_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quant, "DE_LAGS", 2)
    # p = 5%: cumulative violations H = (0.8, 0, 0.4, 0), mean 0.3.
    u, c = quant.du_escanciano(np.array([0.01, 0.5, 0.03, 0.9]), 0.05)

    assert u == pytest.approx(2 * (0.3 - 0.025) / math.sqrt(0.05 * (1 / 3 - 0.0125)))
    # h = H - 0.025: gamma_0 0.185625, gamma_1 -0.0127083, gamma_2 0.145625.
    assert c == pytest.approx(
        4 * ((0.0127083 / 0.185625) ** 2 + (0.145625 / 0.185625) ** 2), rel=1e-5
    )


def testdu_escanciano_conditional_test_sees_clustered_tail_days() -> None:
    rng = np.random.default_rng(23)
    past = rng.standard_normal((250, 250))
    r = rng.standard_normal(250)  # each day's return from its forecast's law
    worst = np.argsort(r)[:15]
    clustered = np.concatenate([np.delete(r, worst)[:100], r[worst]])
    clustered = np.concatenate([clustered, np.delete(r, worst)[100:]])

    spread = quant.backtest(r, past)
    bunched = quant.backtest(clustered, past)

    assert spread.de_c > 0.05
    assert bunched.de_c < 0.01  # the same days, two weeks in a row
    assert abs(bunched.de_u - spread.de_u) < 0.2  # as many violations as before


@pytest.mark.usefixtures("no_arch")
def test_filtered_forecasts_equal_historical_ones_at_constant_volatility() -> None:
    r = 0.01 * np.random.default_rng(20).choice([-1.0, 1.0], 600)
    f = quant.devolatilise(r[:, None], fit=350)

    filtered = quant.filtered_windows(f, np.ones(1))
    past = quant.past_windows(r)

    assert filtered is not None
    assert past is not None
    assert np.allclose(filtered, past)
    assert quant.filtered_windows(quant.devolatilise(r[:499, None]), np.ones(1)) is None


# ---------------------------------------------------------------- density forecasts


def test_blocks_are_non_overlapping_and_end_on_the_last_day() -> None:
    r = np.random.default_rng(8).normal(0.0, 0.01, 100)

    outcomes, starts = quant.blocks(r, 21)

    assert starts.tolist() == [16, 37, 58, 79]
    assert outcomes[-1] == pytest.approx(np.prod(1 + r[79:]) - 1)


def test_pits_never_reach_0_or_1() -> None:
    samples = np.array([[1.0, 2.0, 3.0]] * 3)

    got = quant.pit(np.array([0.0, 2.0, 9.0]), samples)

    assert got.tolist() == [0.5 / 4, 2.0 / 4, 3.5 / 4]


def test_berkowitz_passes_the_true_forecasts_and_rejects_too_wide_ones() -> None:
    rng = np.random.default_rng(10)
    outcomes = rng.standard_normal(120)
    draws = rng.standard_normal((120, 1000))

    right = quant.berkowitz(quant.pit(outcomes, draws))
    wide = quant.berkowitz(quant.pit(outcomes, 2 * draws))

    assert right is not None
    assert wide is not None
    assert right.p > 0.01
    assert wide.p < 0.001
    assert wide.sd < 0.7
    assert quant.berkowitz(np.full(quant.MIN_MONTHS - 1, 0.5)) is None


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
    # log(59/100) = -0.53 jumps; the +3% next day takes back only 6% of it.
    assert not quant.bad_prints(np.array([100.0, 59.0, 61.0, 62.0])).any()


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
