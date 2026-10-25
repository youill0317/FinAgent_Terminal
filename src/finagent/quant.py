"""Portfolio risk math: numpy and pandas in, numbers out (no files, no network).

GJR-GARCH fits use the optional ``arch`` package; without it the volatility
filter is EWMA. The HAR and path-dependent volatility filters, range-based
variances and forecast tests are numpy only.

Conventions: returns are simple daily returns, rows are days and columns are
assets; losses are positive fractions (a VaR of 0.02 is a 2% loss). Every
function describes risk; none proposes weights.
"""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

SEED: Final = 42
TRADING_DAYS: Final = 252
MONTH_DAYS: Final = 21
EWMA_LAMBDA: Final = 0.94  # RiskMetrics daily
REVERSION: Final = 0.985  # daily pull of EWMA variance back to its long-run level
MIN_GARCH: Final = 500  # days a GJR-GARCH fit needs (Hwang-Valls Pereira 2006)
FILTER_DAYS: Final = 2520  # ten years: the filter's fit and its residuals
MAX_PERSISTENCE: Final = 0.999  # alpha + gamma / 2 + beta; a half-life of 693 days
FEEDBACK_DAYS: Final = MONTH_DAYS  # simulated days whose shocks move volatility
PERCENT: Final = 100.0  # arch fits returns in percent
MAX_BLOCK: Final = 20
MIN_BLOCKS: Final = 10  # a window holds at least this many mean blocks
QUIET_LAGS: Final = 5  # Politis-White: this many small autocorrelations in a row
TAIL_LEVEL: Final = 0.975  # the ES level shown as the tail figure
MIN_TAIL: Final = 12  # days beyond the VaR a VaR or ES needs: 240 at 95%
MIN_MONTHS: Final = 24  # non-overlapping 21-day months a density test needs
BOOT_PATHS: Final = 2000  # bootstrap intervals: 50 draws beyond each 2.5% end
GPD_SHARE: Final = 0.10  # largest losses the EVT tail is fitted to (McNeil-Frey)
MIN_EXCESSES: Final = 50  # losses over the threshold a GPD needs: 500 days at 10%
MIN_XI: Final = -0.5  # below this the GPD MLE is not regular (Smith 1985)
GPD_GRID: Final = 400
XI_EDGE: Final = 1e-6  # an MLE this close to MIN_XI sits on the boundary
XI_TOP: Final = 1.5  # profile intervals search xi up to here; equity tails ~0.2-0.4
PROFILE_GRID: Final = 100
LIMIT_GRID: Final = 1000  # scales, 1e-4 to 1e4 x the largest excess, tried at xi = 1
PROFILE_STEPS: Final = 30  # bisections: an interval's ends to 2^-30 of the estimate
REACH: Final = 10  # doublings of the estimate an interval's upper end is sought in
SHARE_GRID: Final = 15  # tail shares P(loss > u) the intervals search, k / n +- 2.5 sd
SHARE_REACH: Final = 2.5  # beyond 1.96 sd the binomial alone drops 1.92
SUM_GRID: Final = 4000  # points the excesses' log-sum is tabled on for the intervals
CI_LEVEL: Final = 0.95
BACKTEST_LEVEL: Final = 0.95
BACKTEST_DAYS: Final = 250
ES_SIMS: Final = 20_000  # ES p-value draws: Monte Carlo SD <= 0.0036, 0.0015 near 0.05
DE_LAGS: Final = 5  # autocorrelations in Du-Escanciano's conditional test: a week
GREEN: Final = 0.95  # binomial CDF of the exception count below which: green
YELLOW: Final = 0.9999  # ... below which: yellow; else red
BETA_PRIOR_DAYS: Final = 250  # beta keeps n / (n + this); the rest is 1
WELCH_DELTA: Final = 3.0
WELCH_DECAY: Final = 2 / 256
PSD_TOLERANCE: Final = 1e-10  # smallest eigenvalue >= -this x the largest
BISECTIONS: Final = 60
SEARCH_DAYS: Final = 90  # calendar days around a stated peak or trough
JUMP: Final = 0.5  # |log return| of a bad print, reverted the next day
REVERSAL: Final = 0.5  # the next day leaves less than this share of the jump
TURBULENCE_WINDOW: Final = 250
DIGITS: Final = 9  # n x (1 - level) is rounded to this before ceil and floor
HAR_WEEK: Final = 5  # Corsi (2009): the weekly and monthly averages' days
HAR_MONTH: Final = 22
VAR_FLOOR: Final = 0.01  # HAR and PDV variances: at least this x the fit's mean r^2
PDV_HALF_LIVES: Final = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)  # days
PDV_THETAS: Final = (0.0, 0.25, 0.5, 0.75, 1.0)  # weights of the slow R2 kernel
SCALE_REACH: Final = 30.0  # HAR and PDV scales are sought within e^-30..e^30
YZ_ALPHA: Final = 0.34  # Yang-Zhang (2000): k = 0.34 / (1.34 + (n + 1) / (n - 1))

type Light = Literal["green", "yellow", "red"]


# ---------------------------------------------------------------- Monte Carlo


def politis_white(x: NDArray[np.float64]) -> float:
    """Politis-White (2004) stationary-bootstrap block for the mean of ``x``.

    With the Patton-Politis-White (2009) correction, as
    ``arch.bootstrap.optimal_block_length``: b = (G / g)^(2/3) n^(1/3), G
    and g the flat-top-weighted sums of k R(k) and R(k) to lag M = 2m, m
    the first of max(5, log10 n) lags in a row with |autocorrelation| under
    2 sqrt(log10(n) / n); at most min(3 sqrt(n), n / 3). 1 for a series
    with no movement.
    """
    n = len(x)
    e = x - x.mean()
    run = max(QUIET_LAGS, int(math.log10(n)))
    most = math.ceil(math.sqrt(n)) + run
    lags = np.arange(most + 1)
    cross = np.array([e[k:] @ e[: n - k] for k in lags])
    if cross[0] <= 0:
        return 1.0
    squares = np.concatenate([[0.0], np.cumsum(e**2)])
    rho = np.abs(cross) / np.sqrt((squares[n] - squares[lags]) * squares[n - lags])
    small = rho < 2 * math.sqrt(math.log10(n) / n)
    quiet = np.convolve(small, np.ones(run), "valid") == run  # lags j..j+run-1
    found = np.flatnonzero(quiet[: most - run + 1])
    m = min(2 * max(int(found[0]), 1), most) if len(found) else most
    k = np.arange(1, m + 1)
    flat = np.minimum(1.0, 2 * (1 - k / m))
    acv = cross[: m + 1] / n
    g = 2 * (flat * k) @ acv[1:]
    spectrum = acv[0] + 2 * flat @ acv[1:]
    block = abs(g / spectrum) ** (2 / 3) * n ** (1 / 3)
    return min(block, math.ceil(min(3 * math.sqrt(n), n / 3)))


def block_length(r: NDArray[np.float64]) -> float:
    """Mean stationary-bootstrap block for 1-D returns ``r`` (``politis_white``).

    The larger of the blocks for r and for (r - mean)^2: tail losses over a
    horizon depend on how volatility clusters, which r alone (1 to 3 days
    on real returns) does not show. A portfolio's series carries the
    Korea/US close lag as its own autocorrelation, so a mix needs no floor
    of its own. Clipped to 1..``MAX_BLOCK`` and to n / ``MIN_BLOCKS``, so
    the window holds ten blocks.
    """
    d = r - r.mean()
    block = max(politis_white(r), politis_white(d**2))
    return max(1.0, min(block, MAX_BLOCK, len(r) / MIN_BLOCKS))


def stationary_indices(
    n: int,
    paths: int,
    horizon: int,
    mean_block: float,
    seed: int | np.random.Generator = SEED,
) -> NDArray[np.int64]:
    """Politis-Romano draws of historical days, shape (paths, horizon).

    Blocks start at uniform days and run for geometric lengths with mean
    ``mean_block``, wrapping from the last day to the first. Draw once and
    reuse: any weight vector then replays the same days. A generator
    ``seed`` is drawn from as it stands.
    """
    rng = np.random.default_rng(seed)
    if mean_block <= 1:
        return rng.integers(0, n, size=(paths, horizon))
    fresh = rng.random((paths, horizon)) < 1 / mean_block
    fresh[:, 0] = True
    starts = rng.integers(0, n, size=(paths, horizon))
    step = np.arange(horizon)
    began = np.where(fresh, step, 0)
    del fresh
    _ = np.maximum.accumulate(began, axis=1, out=began)
    picks = np.take_along_axis(starts, began, axis=1)
    del starts
    picks -= began  # in place: fewer (paths, horizon) arrays at once
    picks += step
    picks %= n
    return picks


def _variance(
    r: NDArray[np.float64], params: NDArray[np.float64], fit: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GJR variance of each column: day t's uses returns before t only.

    ``params`` are omega, alpha, gamma and beta (one each, or one row of
    four per column) in s2' = omega + (alpha + gamma [r < 0]) r^2 + beta s2,
    seeded at the mean of r^2 over the first ``fit`` rows (all by default).
    Returns (variance per day; the next day's).
    """
    omega, alpha, gamma, beta = params
    level = np.mean(r[:fit] ** 2, axis=0)
    var = np.empty_like(r)
    for t in range(len(r)):  # days; vectorised over assets
        var[t] = level
        level = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * level
    return var, level


def _ewma(lam: float = EWMA_LAMBDA) -> NDArray[np.float64]:
    return np.array([0.0, 1 - lam, 0.0, lam])


def ewma_sigma(
    r: NDArray[np.float64], lam: float = EWMA_LAMBDA
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """EWMA volatility of each column: day t's uses returns before t only.

    Zero-mean RiskMetrics form, seeded at the sample mean of r^2. Returns
    (sigma per day, shaped like ``r``; the forecast for the next day).
    """
    var, level = _variance(r, _ewma(lam))
    return np.sqrt(var), np.sqrt(level)


def gjr_fit(r: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """GJR-GARCH(1,1) of 1-D returns: (omega, alpha, gamma, beta), or None.

    Zero mean, normal quasi-maximum likelihood with ``arch``: consistent
    whatever the shocks' distribution, which filtered HS takes from the data
    anyway. None without ``arch``, under ``MIN_GARCH`` days, when the fit
    fails or does not converge, or when alpha + gamma / 2 + beta reaches
    ``MAX_PERSISTENCE``: that close to 1 a fit cannot tell lasting
    volatility from a break or one outlier (Lamoureux-Lastrapes 1990).
    """
    if len(r) < MIN_GARCH:
        return None
    try:
        from arch import arch_model  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None
    model = arch_model(PERCENT * r, mean="Zero", p=1, o=1, q=1, rescale=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # arch re-enables its ConvergenceWarning unless told not to warn;
            # non-convergence is checked below.
            fit = model.fit(disp="off", show_warning=False)
    except (ValueError, np.linalg.LinAlgError):
        return None
    params = np.array(fit.params, dtype=np.float64)  # omega, alpha, gamma, beta
    _, alpha, gamma, beta = params
    if fit.convergence_flag != 0 or not np.isfinite(params).all():
        return None
    if alpha + gamma / 2 + beta >= MAX_PERSISTENCE:
        return None
    params[0] /= PERCENT**2
    return params


type Filter = Literal["gjr", "har", "pdv"]


def _smooth(
    x: NDArray[np.float64], decay: NDArray[np.float64], seed: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Exponential averages s' = decay s + (1 - decay) x_t, before each row.

    Row t of the result (one more than ``x``) uses rows before t only,
    starting from ``seed``; the last row includes every row.
    """
    out = np.empty((len(x) + 1, *x.shape[1:]))
    level = np.broadcast_to(seed, x.shape[1:])
    for t in range(len(x)):  # days; vectorised over the rest
        out[t] = level
        level = decay * level + (1 - decay) * x[t]
    out[-1] = level
    return out


def _unit_scale(
    square: NDArray[np.float64], raw: NDArray[np.float64], floor: float
) -> float:
    """The s with mean(square / max(s raw, floor)) = 1, bisected in log s.

    Residuals r / sigma then have unit variance, floored days included. 1
    when nothing moves (``floor`` 0).
    """
    if floor <= 0:
        return 1.0
    low, high = -SCALE_REACH, SCALE_REACH
    for _ in range(BISECTIONS):
        mid = (low + high) / 2
        if float(np.mean(square / np.maximum(math.exp(mid) * raw, floor))) > 1:
            low = mid
        else:
            high = mid
    return math.exp((low + high) / 2)


def har_features(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """HAR regressors (1, day, week, month, x) of states (..., ``HAR_MONTH`` + 1).

    A state holds the last ``HAR_MONTH`` variance proxies, oldest first, and
    the extra regressor: day is the last proxy, week the mean of the last
    ``HAR_WEEK`` and month the mean of all of them.
    """
    lags = state[..., :HAR_MONTH]
    return np.stack(
        [
            np.ones(lags.shape[:-1]),
            lags[..., -1],
            lags[..., -HAR_WEEK:].mean(axis=-1),
            lags.mean(axis=-1),
            state[..., HAR_MONTH],
        ],
        axis=-1,
    )


def _har_states(
    proxy: NDArray[np.float64], x: NDArray[np.float64], fit: int | None
) -> NDArray[np.float64]:
    """HAR states before each day of (days, assets) ``proxy`` and ``x``.

    Day t's state holds the proxies of days t - ``HAR_MONTH`` .. t - 1 and x of
    day t - 1; days before the first are the first ``fit`` rows' means.
    """
    assets = proxy.shape[1]
    seed = np.broadcast_to(np.mean(proxy[:fit], axis=0), (HAR_MONTH, assets))
    lags = np.lib.stride_tricks.sliding_window_view(
        np.vstack([seed, proxy]), HAR_MONTH, axis=0
    )  # (days + 1, assets, HAR_MONTH)
    extra = np.vstack([np.mean(x[:fit], axis=0), x])
    return np.concatenate([lags, extra[..., None]], axis=-1)


def har_fit(
    proxy: NDArray[np.float64], x: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """HAR-RV of a 1-D daily variance proxy by OLS: (c, b_d, b_w, b_m, b_x).

    Corsi (2009) in levels: RV_t = c + b_d RV_{t-1} + b_w RV^w_{t-1} + b_m
    RV^m_{t-1} + b_x x_{t-1} + e_t, RV^w and RV^m the means of the last
    ``HAR_WEEK`` and ``HAR_MONTH`` days, fitted on the days that have all of
    them. Levels, not logs: OLS then estimates E[RV_t | past] itself, the
    variance filtered HS divides by, and a simulated day's squared return
    is an unbiased RV in the same recursion; a log-HAR needs a
    distribution-specific correction for both (E log z^2 = -1.27 for
    normal z). ``x`` is an extra regressor read on day t - 1 (HAR-RV-IV with
    ``vix_variance``); b_x is 0 without it.
    """
    extra = np.zeros_like(proxy) if x is None else x
    states = _har_states(proxy[:, None], extra[:, None], None)[HAR_MONTH:-1, 0]
    design = har_features(states)
    if x is None:
        design = design[:, :-1]
    coef = np.linalg.lstsq(design, proxy[HAR_MONTH:], rcond=None)[0]
    return coef if x is not None else np.append(coef, 0.0)


def vix_variance(vix: NDArray[np.float64]) -> NDArray[np.float64]:
    """An annualised vol index in points (VIX) as daily variance: (v / 100)^2 / 252."""
    return (vix / PERCENT) ** 2 / TRADING_DAYS


def pdv_sigma(
    params: NDArray[np.float64], state: NDArray[np.float64]
) -> NDArray[np.float64]:
    """PDV sigma of states (..., assets, 4) under ``pdv_fit`` params (9, assets).

    sigma = b0 + b10 R10 + b11 R11 + b2 sqrt((1 - theta2) R20 + theta2 R21),
    the state holding (R10, R11, R20, R21); not floored.
    """
    b0, b10, b11, b2, theta = params[:5]
    r10, r11, r20, r21 = np.moveaxis(state, -1, 0)
    trend = np.maximum((1 - theta) * r20 + theta * r21, 0.0)
    return b0 + b10 * r10 + b11 * r11 + b2 * np.sqrt(trend)


def pdv_fit(
    r: NDArray[np.float64], target: NDArray[np.float64] | None = None
) -> NDArray[np.float64] | None:
    """Path-dependent volatility of 1-D daily returns (Guyon-Lekeufack 2023).

    sigma_t = b0 + b1 R1 + b2 sqrt(R2), R1 a weighted sum of past returns
    and R2 of past squared returns, each with the paper's Markovian
    two-exponential kernel: R1 = (1 - theta1) R10 + theta1 R11, R2 = (1 -
    theta2) R20 + theta2 R21, and R_j' = a_j R_j + (1 - a_j) r (r^2 for R2)
    after each day, so day t's uses returns before t. Daily units (sigma
    of r, kernels summing to 1): the paper's annualised b0 is b0 here times
    sqrt(252), its b1 (b10 + b11) over sqrt(252), b2 the same; theta1 =
    b11 / (b10 + b11). The decays come from ``PDV_HALF_LIVES``, a fast one
    and one at least four times slower per kernel, and theta2 from
    ``PDV_THETAS``; for each, the b are OLS of ``target`` (a daily
    volatility proxy; |r| by default, or sqrt of a range-based variance) on
    (1, R10, R11, sqrt(R2)), and the grid point with the least squared
    error wins. The b are then scaled so that mean(r^2 / sigma^2) = 1 over
    the days, sigma floored at sqrt(``VAR_FLOOR`` x mean r^2): E|z| =
    sqrt(2 / pi) for normal z, and a range proxy measures its own scale.
    Kernels start at the days' mean return and mean r^2. Returns (b0, b10,
    b11, b2, theta2, a10, a11, a20, a21); None for returns that never move.
    """
    square = r**2
    level = float(square.mean())
    if level <= 0:
        return None
    y = np.abs(r) if target is None else target
    decay = 0.5 ** (1 / np.array(PDV_HALF_LIVES))
    count = len(decay)
    one = _smooth(np.repeat(r[:, None], count, 1), decay, np.full(count, r.mean()))
    two = _smooth(np.repeat(square[:, None], count, 1), decay, np.full(count, level))
    fast, slow = np.triu_indices(count, 2)  # half-lives 4x apart or more
    theta = np.array(PDV_THETAS)
    mixed = (1 - theta) * two[:-1, fast, None] + theta * two[:-1, slow, None]
    roots = np.sqrt(mixed.reshape(len(r), -1))  # (days, pairs x thetas)
    columns = np.column_stack([np.ones(len(r)), one[:-1], roots, y])
    gram = columns.T @ columns
    pick = np.array(
        [
            [0, 1 + i, 1 + j, 1 + count + k]
            for i, j in zip(fast, slow, strict=True)
            for k in range(roots.shape[1])
        ]
    )
    xtx = gram[pick[:, :, None], pick[:, None, :]]
    xty = gram[pick, -1]
    beta = np.einsum("cij,cj->ci", np.linalg.pinv(xtx), xty)
    sse = -2 * np.einsum("ci,ci->c", beta, xty)
    sse += np.einsum("ci,cij,cj->c", beta, xtx, beta)
    best = int(np.argmin(sse))
    second = int(pick[best, 3]) - 1 - count
    pair, mix = divmod(second, len(theta))
    params = np.array(
        [
            *beta[best],
            theta[mix],
            decay[pick[best, 1] - 1],
            decay[pick[best, 2] - 1],
            decay[fast[pair]],
            decay[slow[pair]],
        ]
    )
    state = np.stack(
        [
            one[:-1, pick[best, 1] - 1],
            one[:-1, pick[best, 2] - 1],
            two[:-1, fast[pair]],
            two[:-1, slow[pair]],
        ],
        axis=-1,
    )
    sigma = pdv_sigma(params[:, None], state[:, None, :])[:, 0]
    params[:4] *= math.sqrt(
        _unit_scale(square, sigma * np.abs(sigma), VAR_FLOOR * level)
    )
    return params


@dataclass(frozen=True, slots=True)
class Recursion:
    """A HAR or PDV variance filter per asset, and its state before each day.

    ``states[t]`` (assets, size) holds data before day t only, as
    ``_variance`` does, and ``states[-1]`` gives the next day's variance.
    HAR (``har_fit``): ``HAR_MONTH`` past squared-return-scale proxies,
    oldest first, then the extra regressor, with params (c, b_d, b_w, b_m,
    b_x) in units of r^2. PDV (``pdv_fit``): (R10, R11, R20, R21) and its
    nine params. Variances are at least ``floor``.
    """

    kind: Literal["har", "pdv"]
    params: NDArray[np.float64]  # (5 or 9, assets)
    states: NDArray[np.float64]  # (days + 1, assets, HAR_MONTH + 1 or 4)
    floor: NDArray[np.float64]  # (assets,)

    def variance(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Variance per asset of states shaped (..., assets, size)."""
        if self.kind == "har":
            var = np.einsum("...ak,ka->...a", har_features(state), self.params)
            return np.maximum(var, self.floor)
        return np.maximum(pdv_sigma(self.params, state), np.sqrt(self.floor)) ** 2

    def update(
        self,
        state: NDArray[np.float64],
        ret: NDArray[np.float64],
        square: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """The state after a day with returns ``ret`` and squares ``square``.

        HAR takes ``square`` as the day's proxy and keeps the extra
        regressor (it is not simulated); PDV moves its kernels by both.
        """
        if self.kind == "har":
            return np.concatenate(
                [state[..., 1:HAR_MONTH], square[..., None], state[..., HAR_MONTH:]],
                axis=-1,
            )
        decay = self.params[5:].T
        day = np.stack([ret, ret, square, square], axis=-1)
        return decay * state + (1 - decay) * day


def _har_recursion(
    r: NDArray[np.float64],
    fit: int | None,
    proxy: NDArray[np.float64],
    x: NDArray[np.float64] | None,
) -> Recursion:
    """``devolatilise``'s HAR: ``har_fit`` per column, rescaled to units of r^2."""
    extra = (
        np.zeros_like(r)
        if x is None
        else np.broadcast_to(x.reshape(len(r), -1), r.shape)
    )
    params = np.column_stack(
        [
            har_fit(proxy[:fit, i], None if x is None else extra[:fit, i])
            for i in range(r.shape[1])
        ]
    )
    states = _har_states(proxy, extra, fit)
    raw = np.einsum("dak,ka->da", har_features(states[:-1][:fit]), params)
    square = r[:fit] ** 2
    floor = VAR_FLOOR * np.mean(square, axis=0)
    scale = np.array(
        [
            _unit_scale(square[:, i], raw[:, i], float(floor[i]))
            for i in range(r.shape[1])
        ]
    )
    states[..., :HAR_MONTH] *= scale[:, None]
    params[[0, 4]] *= scale
    return Recursion("har", params, states, floor)


def _pdv_recursion(
    r: NDArray[np.float64], fit: int | None, proxy: NDArray[np.float64]
) -> Recursion:
    """``devolatilise``'s PDV: ``pdv_fit`` per column, to sqrt(``proxy``)."""
    target = np.sqrt(proxy)
    fits = [pdv_fit(r[:fit, i], target[:fit, i]) for i in range(r.shape[1])]
    params = np.column_stack([np.zeros(9) if p is None else p for p in fits])
    square = r**2
    days = np.stack([r, r, square, square], axis=-1)
    mean, level = np.mean(r[:fit], axis=0), np.mean(square[:fit], axis=0)
    seed = np.stack([mean, mean, level, level], axis=-1)
    states = _smooth(days, params[5:].T, seed)
    return Recursion("pdv", params, states, VAR_FLOOR * level)


@dataclass(frozen=True, slots=True)
class Devolatilised:
    z: NDArray[np.float64]  # r_t / sigma_t, shaped like the returns
    sigma: NDArray[np.float64]  # sigma_t, shaped like the returns
    now: NDArray[np.float64]  # sigma for the next day, per asset
    step: NDArray[
        np.float64
    ]  # (4, assets): the variance recursion simulated days follow
    garch: NDArray[np.bool_]  # per asset: GJR-GARCH (or ``model``), else EWMA
    model: Recursion | None = None  # a HAR or PDV filter; None: ``step``'s


def devolatilise(
    r: NDArray[np.float64],
    fit: int | None = None,
    model: Filter = "gjr",
    proxy: NDArray[np.float64] | None = None,
    x: NDArray[np.float64] | None = None,
) -> Devolatilised:
    """Returns (days, assets) divided by their own volatility (filtered HS).

    Each column's filter is GJR-GARCH(1,1) estimated on its first ``fit``
    rows (all by default; ``gjr_fit``), else EWMA; it then runs over every
    row. Pass the last ``FILTER_DAYS``: enough for a stable fit, yet clear
    of old breaks such as 1987 and 1997-98 that push GARCH persistence
    toward 1 (Lamoureux-Lastrapes 1990). Their residuals ``z`` are the
    filtered-HS scenarios. A column with no movement gets residuals of 0.
    Simulated days follow the GJR recursion (under EWMA, which has no
    long-run level, they return by ``REVERSION`` a day) toward the fit rows'
    mean of r^2: omega is set to (1 - persistence) times it (variance
    targeting, Engle-Mezrich 1996), as a near-1 persistence makes the
    fitted omega's own level unreliable.

    ``model`` "har" or "pdv" fits every column's filter that way instead
    (``har_fit``, ``pdv_fit``) when the first ``fit`` rows reach
    ``MIN_GARCH``, else EWMA as above. ``proxy`` (shaped like r) is each
    day's variance proxy, r^2 where absent or nan: pass a range-based one
    (``garman_klass`` with the previous close) for a sharper fit. HAR's are
    rescaled so mean(r^2 / sigma^2) = 1 over the fit rows, as ``pdv_fit``'s
    are; ``x`` (days,) or (days, assets) is HAR's extra regressor, day t's
    value entering day t + 1 (PDV takes none). Simulated days then follow
    ``model_paths``; ``step`` is nan and ``garch`` all True.
    """
    if model != "gjr" and len(r[:fit]) >= MIN_GARCH:
        square = r**2
        given = square if proxy is None else proxy.reshape(r.shape)
        filled = np.where(np.isfinite(given), given, square)
        found = (
            _har_recursion(r, fit, filled, x)
            if model == "har"
            else _pdv_recursion(r, fit, filled)
        )
        var = found.variance(found.states)
        sigma = np.sqrt(var[:-1])
        z = np.divide(r, sigma, out=np.zeros_like(r), where=sigma > 0)
        assets = r.shape[1]
        nan = np.full((4, assets), np.nan)
        every = np.ones(assets, dtype=np.bool_)
        return Devolatilised(z, sigma, np.sqrt(var[-1]), nan, every, found)
    fits = (
        [gjr_fit(r[:fit, i]) for i in range(r.shape[1])]
        if model == "gjr"
        else [None] * r.shape[1]
    )
    garch = np.array([p is not None for p in fits], dtype=np.bool_)
    params = np.column_stack([_ewma() if p is None else p for p in fits])
    var, level = _variance(r, params, fit)
    sigma = np.sqrt(var)
    z = np.divide(r, sigma, out=np.zeros_like(r), where=sigma > 0)
    revert = np.array([0.0, 0.0, 0.0, REVERSION])[:, None]
    step = np.where(garch, params, revert)
    _, alpha, gamma, beta = step
    step[0] = (1 - alpha - gamma / 2 - beta) * np.mean(r[:fit] ** 2, axis=0)
    return Devolatilised(z, sigma, np.sqrt(level), step, garch)


def filtered_returns(f: Devolatilised) -> NDArray[np.float64]:
    """Filtered-HS scenarios: each day's residual at today's volatility.

    Shaped like the returns; for a portfolio, multiply the asset matrix by
    the weights afterwards so each holding keeps its own volatility.
    """
    return f.z * f.now


def filtered_paths(
    f: Devolatilised,
    w: NDArray[np.float64],
    picks: NDArray[np.int64],
    var: NDArray[np.float64],
    feedback: int | None = None,
) -> NDArray[np.float64]:
    """Portfolio returns along drawn days of ``f``'s residuals, (paths, horizon).

    Barone-Adesi et al. (1999): each asset's variance starts at ``var``
    ((assets,) or (paths, assets)) and follows ``f.step`` on its own
    simulated shocks for the first ``feedback`` days (all by
    default), then its expectation s2' = omega + (alpha + gamma / 2 + beta)
    s2: volatility is forecastable for weeks, not a year (Christoffersen-
    Diebold 2000), and a year of feedback compounds every error in the fit.
    The residuals are demeaned first (zero drift); where shocks feed the
    variance they are also scaled to unit variance, which the recursion's
    persistence assumes. One day at a time, so no paths x days x assets
    array is built.
    """
    omega, alpha, gamma, beta = f.step
    z = f.z - f.z.mean(axis=0)
    rms = np.sqrt(np.mean(z**2, axis=0))
    z = z / np.where((alpha + gamma > 0) & (rms > 0), rms, 1.0)
    var = np.broadcast_to(var, (len(picks), z.shape[1]))
    out = np.empty(picks.shape)
    for k in range(picks.shape[1]):
        shock = np.sqrt(var) * z[picks[:, k]]
        out[:, k] = shock @ w
        if feedback is None or k < feedback:
            var = omega + (alpha + gamma * (shock < 0)) * shock**2 + beta * var
        else:
            var = omega + (alpha + gamma / 2 + beta) * var
    return out


def model_paths(
    f: Devolatilised,
    w: NDArray[np.float64],
    picks: NDArray[np.int64],
    state: NDArray[np.float64],
    feedback: int | None = None,
) -> NDArray[np.float64]:
    """``filtered_paths`` for a HAR or PDV filter (``f.model``), from ``state``.

    ``state`` is (assets, size) or (paths, assets, size), e.g.
    ``f.model.states[-1]`` for today's. Each day's variance comes from the
    state, which the day's simulated returns and squares then move
    (``Recursion.update``) for the first ``feedback`` days (all by
    default); after them, their expectation does: a return of 0 and a
    square equal to the variance, which for HAR is the exact multi-step
    forecast. The residuals are demeaned and scaled to unit variance.
    """
    found = f.model
    if found is None:
        raise ValueError("model_paths needs a HAR or PDV filter")
    z = f.z - f.z.mean(axis=0)
    rms = np.sqrt(np.mean(z**2, axis=0))
    z = z / np.where(rms > 0, rms, 1.0)
    state = np.broadcast_to(state, (len(picks), *found.states.shape[1:]))
    out = np.empty(picks.shape)
    for k in range(picks.shape[1]):
        var = found.variance(state)
        shock = np.sqrt(var) * z[picks[:, k]]
        out[:, k] = shock @ w
        if feedback is None or k < feedback:
            state = found.update(state, shock, shock**2)
        else:
            state = found.update(state, np.zeros_like(var), var)
    return out


def growth_paths(
    rets: NDArray[np.float64],
    w: NDArray[np.float64],
    picks: NDArray[np.int64],
    f: Devolatilised | None = None,
) -> NDArray[np.float64]:
    """Growth of 1 along drawn paths, shape (paths, horizon + 1).

    ``rets`` (days, assets) are the days ``picks`` index into; weights are
    reset to ``w`` daily. With ``f``, ``picks`` index into its residuals on
    those days (its last rows) instead and each asset's volatility starts at
    today's, moved by the shocks of the first ``FEEDBACK_DAYS``
    (``filtered_paths``). Daily returns
    are demeaned (zero drift) and floored at -100%.
    """
    if f is None:
        steps = rets @ w
        daily = (steps - steps.mean())[picks]
    else:
        days = replace(f, z=f.z[-len(rets) :])
        if f.model is None:
            daily = filtered_paths(days, w, picks, f.now**2, FEEDBACK_DAYS)
        else:
            daily = model_paths(days, w, picks, f.model.states[-1], FEEDBACK_DAYS)
    _ = np.maximum(daily, -1.0, out=daily)
    daily += 1.0
    growth = np.ones((len(picks), picks.shape[1] + 1))
    _ = np.cumprod(daily, axis=1, out=growth[:, 1:])
    return growth


# ---------------------------------------------------------------- range-based variance


def adjusted_ohlc(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    adjusted: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Raw open, high, low and close times adjusted / close, the day's factor.

    The range estimators below take prices on one split- and
    dividend-consistent scale, so an overnight return from the previous
    adjusted close to the adjusted open spans an ex-date correctly.
    """
    factor = adjusted / close
    return open_ * factor, high * factor, low * factor, adjusted


def _logs(*prices: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    """Natural logs; nan for a missing or non-positive price."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return tuple(
            np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), np.nan) for p in prices
        )


def parkinson(
    high: NDArray[np.float64], low: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Parkinson (1980) daily variance (log(H / L))^2 / (4 log 2); 0 for no range."""
    h, lo = _logs(high, low)
    return (h - lo) ** 2 / (4 * math.log(2))


def rogers_satchell(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rogers-Satchell (1991) daily variance, unbiased under any drift.

    log(H / C) log(H / O) + log(L / C) log(L / O), floored at 0 (a high
    below the open or close, or a low above them, is a bad print).
    """
    o, h, lo, c = _logs(open_, high, low, close)
    return np.maximum((h - c) * (h - o) + (lo - c) * (lo - o), 0.0)


def garman_klass(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    previous: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Garman-Klass (1980) daily variance, open to close.

    0.5 (log(H / L))^2 - (2 log 2 - 1) (log(C / O))^2, floored at 0. With
    the previous close, the overnight (log(O / C_prev))^2 is added (Yang-
    Zhang's extension): the close-to-close variance, as r^2 measures it,
    for ``devolatilise``'s proxy. Prices as ``adjusted_ohlc`` gives them;
    nan where one is missing or not positive.
    """
    o, h, lo, c = _logs(open_, high, low, close)
    day = np.maximum(0.5 * (h - lo) ** 2 - (2 * math.log(2) - 1) * (c - o) ** 2, 0.0)
    if previous is None:
        return day
    return day + (o - _logs(previous)[0]) ** 2


def yang_zhang(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    window: int = MONTH_DAYS,
) -> NDArray[np.float64]:
    """Yang-Zhang (2000) daily close-to-close variance over each ``window`` days.

    s_o^2 + k s_c^2 + (1 - k) mean RS: the sample variances of the
    overnight log(O_t / C_{t-1}) and open-to-close log(C / O) returns and
    the mean ``rogers_satchell``, k = 0.34 / (1.34 + (n + 1) / (n - 1)),
    so neither drift nor opening jumps bias it. Row t covers days t -
    window + 1 .. t, along axis 0; nan for the first ``window`` rows (day 0
    has no overnight return) and for windows with a missing price.
    """
    o, c = _logs(open_, close)
    night = np.concatenate([np.full((1, *o.shape[1:]), np.nan), o[1:] - c[:-1]])
    k = YZ_ALPHA / (1 + YZ_ALPHA + (window + 1) / (window - 1))
    parts = [night, c - o, rogers_satchell(open_, high, low, close)]
    views = [np.lib.stride_tricks.sliding_window_view(p, window, axis=0) for p in parts]
    var = (
        views[0].var(axis=-1, ddof=1)
        + k * views[1].var(axis=-1, ddof=1)
        + (1 - k) * views[2].mean(axis=-1)
    )
    return np.concatenate([np.full((window - 1, *var.shape[1:]), np.nan), var])


# ---------------------------------------------------------------- covariance


def _psd(matrix: NDArray[np.float64]) -> bool:
    eig = np.linalg.eigvalsh(matrix)
    return bool(eig[0] >= -PSD_TOLERANCE * max(float(np.abs(eig).max()), 1e-300))


def lagged_covariance(
    rets: NDArray[np.float64], korean: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], float]:
    """Daily covariance with Korea/US asynchronous closes put back.

    A Korean close comes before the same day's US open, so Korea's day t
    reacts to the US day t-1: S = S0 + f (S1 + S1'), with S1 kept only for
    Korean rows and non-Korean columns, and f the largest value in 0..1
    (bisected) that leaves S positive semidefinite. Returns (S, f).
    """
    x = rets - rets.mean(axis=0)
    n = len(x)
    s0 = x.T @ x / n
    s1 = x[1:].T @ x[:-1] / n  # s1[i, j] = cov(x_i at t, x_j at t-1)
    lag = np.where(np.outer(korean, ~korean), s1, 0.0)
    lag = lag + lag.T
    if _psd(s0 + lag):
        return s0 + lag, 1.0
    low, high = 0.0, 1.0
    for _ in range(BISECTIONS):
        mid = (low + high) / 2
        if _psd(s0 + mid * lag):
            low = mid
        else:
            high = mid
    return s0 + low * lag, low


def correlation(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Correlation matrix of a covariance matrix; 0 off a zero-variance row."""
    sd = np.sqrt(np.diag(cov))
    outer = np.outer(sd, sd)
    corr = np.divide(cov, outer, out=np.zeros_like(cov), where=outer > 0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _mean_correlation(sample: NDArray[np.float64]) -> float:
    n = len(sample)
    return float((correlation(sample).sum() - n) / (n * (n - 1)))


def _target(sample: NDArray[np.float64]) -> NDArray[np.float64]:
    """The constant-correlation matrix with ``sample``'s variances."""
    sd = np.sqrt(np.diag(sample))
    target = _mean_correlation(sample) * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(sample))
    return target


def _intensity(x: NDArray[np.float64]) -> float:
    """Ledoit-Wolf (2004) optimal shrinkage toward constant correlation."""
    t = len(x)
    y = x - x.mean(axis=0)
    s = y.T @ y / t
    sd = np.sqrt(np.diag(s))
    target = _target(s)
    gamma = float(((target - s) ** 2).sum())
    if gamma <= 0 or not (sd > 0).all():
        return 0.0
    mean = _mean_correlation(s)
    sq = y**2
    pi = sq.T @ sq / t - s**2  # pi_ij = mean((y_i y_j - s_ij)^2)
    theta = (y**3).T @ y / t - np.diag(s)[:, None] * s
    ratio = sd[None, :] / sd[:, None]
    off = mean / 2 * (ratio * theta + ratio.T * theta.T)
    np.fill_diagonal(off, 0.0)
    rho = float(np.trace(pi) + off.sum())
    kappa = (float(pi.sum()) - rho) / gamma
    return min(max(kappa / t, 0.0), 1.0)


def ledoit_wolf(
    rets: NDArray[np.float64], sample: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], float]:
    """Covariance shrunk toward constant correlation. Returns (cov, weight).

    The weight comes from ``rets``' moments (1/T); it is applied to
    ``sample`` when given (e.g. ``lagged_covariance``'s) and its own target.
    """
    t, n = rets.shape
    if sample is None:
        y = rets - rets.mean(axis=0)
        cov = y.T @ y / t
    else:
        cov = sample
    if n < 2:  # noqa: PLR2004 - correlation needs two assets
        return cov, 0.0
    weight = _intensity(rets)
    return weight * _target(cov) + (1 - weight) * cov, weight


# ---------------------------------------------------------------- beta


def dimson_beta(r: NDArray[np.float64], m: NDArray[np.float64]) -> float:
    """Sum of slopes on the market's same day and previous day.

    For a Korean series against the S&P 500, whose day t-1 closes after
    Korea's day t-1 and lands in Korea's day t. Rows with a nan are dropped.
    """
    lagged = np.concatenate([[np.nan], m[:-1]])
    ok = np.isfinite(r) & np.isfinite(m) & np.isfinite(lagged)
    if ok.sum() < 3:  # noqa: PLR2004 - three slopes need three rows
        return math.nan
    design = np.column_stack([np.ones(int(ok.sum())), m[ok], lagged[ok]])
    coef = np.linalg.lstsq(design, r[ok], rcond=None)[0]
    return float(coef[1] + coef[2])


def welch_beta(
    r: NDArray[np.float64],
    m: NDArray[np.float64],
    delta: float = WELCH_DELTA,
    decay: float = WELCH_DECAY,
) -> float:
    """Welch's slope-winsorised, age-weighted beta (robust).

    Each return is clipped to between (1 - delta) and (1 + delta) times the
    market's that day, and day weights fall by exp(-decay) per day of age.
    """
    bounds = np.stack([(1 - delta) * m, (1 + delta) * m])
    clipped = np.clip(r, bounds.min(axis=0), bounds.max(axis=0))
    weight = np.exp(-decay * np.arange(len(r) - 1, -1, -1, dtype=np.float64))
    weight /= weight.sum()
    dm = m - weight @ m
    spread = float((weight * dm) @ dm)
    if spread <= 0:
        return math.nan
    return float((weight * dm) @ (clipped - weight @ clipped)) / spread


def effective_days(n: int, decay: float = WELCH_DECAY) -> float:
    """Kish's effective sample size (sum w)^2 / sum w^2 of ``welch_beta``'s weights.

    n with no decay; at the default about 117 for 126 days, 193 for 252 and
    255 for 756 (2 / decay at most).
    """
    weight = np.exp(-decay * np.arange(n, dtype=np.float64))
    return float(weight.sum() ** 2 / (weight @ weight))


def shrunk_beta(beta: float, n: float, prior: int = BETA_PRIOR_DAYS) -> float:
    """``beta`` from ``n`` (effective) days pulled toward 1 by prior / (n + prior)."""
    weight = n / (n + prior)
    return weight * beta + (1 - weight)


# ---------------------------------------------------------------- diversification


def risk_contributions(
    w: NDArray[np.float64], cov: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Euler shares of portfolio variance, w_i (cov w)_i / var; they sum to 1."""
    marginal = cov @ w
    var = float(w @ marginal)
    if var <= 0:
        return np.full(len(w), np.nan)
    return w * marginal / var


def diversification_ratio(w: NDArray[np.float64], cov: NDArray[np.float64]) -> float:
    """Weighted average volatility over portfolio volatility (1 = none)."""
    var = float(w @ cov @ w)
    if var <= 0:
        return math.nan
    return float(w @ np.sqrt(np.diag(cov))) / math.sqrt(var)


def effective_holdings(w: NDArray[np.float64]) -> float:
    """1 / Herfindahl of the weights: N for N equal ones."""
    share = w / w.sum()
    return 1 / float(share @ share)


@dataclass(frozen=True, slots=True)
class WhatIf:
    vol: float  # change in volatility, in the covariance's units
    beta: float  # change in beta
    ratio: float  # change in diversification ratio
    correlation: float  # of the candidate with the current portfolio
    threshold: float  # sigma_p / sigma_c: a lower correlation lowers volatility


def what_if(
    w: NDArray[np.float64],
    cov: NDArray[np.float64],
    betas: NDArray[np.float64],
    candidate: int,
    fraction: float,
) -> WhatIf:
    """Effect of putting ``fraction`` of the portfolio into ``candidate``.

    ``w``, ``cov`` and ``betas`` cover the holdings and the candidate (its
    weight in ``w`` usually 0); the rest are scaled down pro rata. A small
    addition lowers volatility exactly when the correlation is below the
    threshold.
    """
    after = (1 - fraction) * w
    after[candidate] += fraction
    sigma_p = math.sqrt(float(w @ cov @ w))
    sigma_c = math.sqrt(float(cov[candidate, candidate]))
    joint = float(cov[candidate] @ w)
    corr = joint / (sigma_p * sigma_c) if sigma_p * sigma_c > 0 else math.nan
    return WhatIf(
        vol=math.sqrt(float(after @ cov @ after)) - sigma_p,
        beta=float((after - w) @ betas),
        ratio=diversification_ratio(after, cov) - diversification_ratio(w, cov),
        correlation=corr,
        threshold=sigma_p / sigma_c if sigma_c > 0 else math.inf,
    )


# ---------------------------------------------------------------- VaR and ES


def min_samples(level: float) -> int:
    """Days needed before a VaR or ES at ``level`` is shown (else 표본 부족).

    Enough for ``MIN_TAIL`` days in the tail: 240 at 95%, 480 at 97.5%,
    1,200 at 99%.
    """
    return math.ceil(round(MIN_TAIL / (1 - level), DIGITS))


def _tail(
    loss: NDArray[np.float64], level: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """VaR and ES of losses along the last axis.

    With m = n (1 - level): VaR is the ceil(m)-th largest loss and ES the
    mean of the largest m, the boundary loss counted by its fraction, so
    ES >= VaR always and component ES adds up to ES exactly.
    """
    n = loss.shape[-1]
    m = round(n * (1 - level), DIGITS)
    top = math.ceil(m)
    part = np.partition(loss, n - top, axis=-1)
    var = part[..., n - top]
    es = (part[..., n - top + 1 :].sum(axis=-1) + (m - top + 1) * var) / m
    return var, es


def _tail_weights(loss: NDArray[np.float64], level: float) -> NDArray[np.float64]:
    """Each scenario's share in ``_tail``'s ES; they sum to 1."""
    n = len(loss)
    m = round(n * (1 - level), DIGITS)
    top = math.ceil(m)
    order = np.argpartition(loss, n - top)
    weight = np.zeros(n)
    weight[order[n - top + 1 :]] = 1.0
    weight[order[n - top]] = m - top + 1
    return weight / m


def var_es(r: NDArray[np.float64], level: float) -> tuple[float, float] | None:
    """Historical VaR and ES of 1-D returns; None below ``min_samples``.

    For filtered HS pass ``filtered_returns(devolatilise(r))`` (times the weights).
    """
    if len(r) < min_samples(level):
        return None
    var, es = _tail(-r, level)
    return float(var), float(es)


def bootstrap_intervals(
    r: NDArray[np.float64],
    levels: Sequence[float],
    paths: int = BOOT_PATHS,
    seed: int = SEED,
) -> list[tuple[tuple[float, float], tuple[float, float]] | None]:
    """``CI_LEVEL`` percentile intervals ((VaR low, high), (ES ...)) per level.

    From a stationary bootstrap of ``r`` (mean block ``block_length``); one
    draw serves every level. None at a level below
    ``min_samples``, where ``var_es`` gives none.
    """
    n = len(r)
    if n < min(min_samples(level) for level in levels):
        return [None] * len(levels)
    loss = (-r)[stationary_indices(n, paths, n, block_length(r), seed)]
    ends = (50 * (1 - CI_LEVEL), 50 * (1 + CI_LEVEL))
    out: list[tuple[tuple[float, float], tuple[float, float]] | None] = []
    for level in levels:
        if n < min_samples(level):
            out.append(None)
            continue
        var, es = _tail(loss, level)
        var_low, var_high = np.percentile(var, ends)
        es_low, es_high = np.percentile(es, ends)
        out.append(((float(var_low), float(var_high)), (float(es_low), float(es_high))))
    return out


def _pwm(y: NDArray[np.float64]) -> tuple[float, float]:
    """Hosking-Wallis (1987) probability-weighted moments: (xi, scale)."""
    k = len(y)
    a0 = float(y.mean())
    a1 = float((1 - (np.arange(1, k + 1) - 0.35) / k) @ np.sort(y)) / k
    return 2 - a0 / (a0 - 2 * a1), 2 * a0 * a1 / (a0 - 2 * a1)


def gpd_fit(y: NDArray[np.float64]) -> tuple[float, float]:
    """Generalised Pareto (xi, scale) of excesses ``y`` >= 0.

    Maximum likelihood (``gpd_mle``), else probability-weighted moments.
    """
    if float(y.max()) <= 0:  # no movement
        return 0.0, 0.0
    return gpd_mle(y) or _pwm(y)


def gpd_mle(y: NDArray[np.float64]) -> tuple[float, float] | None:
    """Maximum-likelihood GPD (xi, scale) of excesses ``y`` with some above 0.

    Profiled in theta = xi / scale (Grimshaw 1993): at a given theta, xi =
    mean(log(1 + theta y)) and the log-likelihood is -k (log(xi / theta) +
    xi + 1). It is searched over s = log(1 + theta max y), on a grid, then
    by golden section next to the best point. None for a maximum on the
    grid's edge or at ``MIN_XI``, where the MLE is not regular.
    """
    top = float(y.max())

    def profile(s: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        theta = np.expm1(s) / top
        xi = np.log1p(np.multiply.outer(theta, y)).mean(axis=-1)
        exponential = np.full_like(xi, y.mean())  # the theta -> 0 limit
        scale = np.divide(xi, theta, out=exponential, where=theta != 0)
        loglik = -(np.log(scale) + xi + 1)
        return np.where(xi > MIN_XI, loglik, -np.inf), xi, scale

    grid = np.linspace(-6.0, 8.0, GPD_GRID)  # theta max y from -0.998 to 3,000
    loglik = profile(grid)[0]
    best = int(np.argmax(loglik))
    if not 0 < best < GPD_GRID - 1:
        return None
    low, high = float(grid[best - 1]), float(grid[best + 1])
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(BISECTIONS):
        a, b = high - ratio * (high - low), low + ratio * (high - low)
        if profile(np.array(a))[0] > profile(np.array(b))[0]:
            high = b
        else:
            low = a
    _, xi, scale = profile(np.array((low + high) / 2))
    if xi < MIN_XI + XI_EDGE:  # still rising toward MIN_XI
        return None
    return float(xi), float(scale)


def _excesses(
    r: NDArray[np.float64], level: float
) -> tuple[float, NDArray[np.float64], float] | None:
    """``evt_var_es``'s u, the k losses over it less u, and log(n (1 - level) / k)."""
    loss = np.sort(-r)
    n = len(loss)
    k = math.floor(round(n * GPD_SHARE, DIGITS))
    if k < MIN_EXCESSES or level <= 1 - k / n:
        return None
    u = float(loss[n - k - 1])
    return u, loss[n - k :] - u, math.log(n * (1 - level) / k)


def _evt(u: float, xi: float, scale: float, log_ratio: float) -> tuple[float, float]:
    grow = -log_ratio if xi == 0 else math.expm1(-xi * log_ratio) / xi  # xi -> 0
    var = u + scale * grow
    return var, (var + scale - xi * u) / (1 - xi)


def evt_var_es(r: NDArray[np.float64], level: float) -> tuple[float, float] | None:
    """VaR and ES of 1-D returns from a GPD over the largest losses.

    McNeil-Frey (2000) on filtered HS: pass ``filtered_returns`` (times the
    weights). As a GPD scales with its data, fitting these equals fitting
    the residuals and multiplying by today's volatility. With u the
    (k + 1)-th largest of n losses, k = 10% of n:
    VaR = u + scale ((n (1 - level) / k)^-xi - 1) / xi and
    ES = (VaR + scale - xi u) / (1 - xi). None under ``MIN_EXCESSES``
    excesses, at a level inside the threshold, or for xi >= 1 (no finite
    ES).
    """
    found = _excesses(r, level)
    if found is None:
        return None
    u, y, log_ratio = found
    xi, scale = gpd_fit(y)
    if xi >= 1:
        return None
    return _evt(u, xi, scale, log_ratio)


def _gpd_loglik(
    y: NDArray[np.float64], xi: NDArray[np.float64], scale: NDArray[np.float64]
) -> NDArray[np.float64]:
    """GPD log-likelihood of excesses ``y`` at each (xi != 0, scale); -inf off it."""
    ok = (scale > 0) & (xi * float(y.max()) > -scale)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = np.log1p(np.multiply.outer(xi / scale, y)).sum(axis=-1)
        loglik = -len(y) * np.log(scale) - (1 + 1 / xi) * z
    return np.where(ok, loglik, -np.inf)


def _summed(
    y: NDArray[np.float64],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Sum of log1p(t y) at each t, interpolated in log1p(t max y) from a table."""
    top = float(y.max())
    grid = np.linspace(-10, 10, SUM_GRID)
    table = np.log1p(np.multiply.outer(np.expm1(grid) / top, y)).sum(axis=1)

    def at(t: NDArray[np.float64]) -> NDArray[np.float64]:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.interp(np.log1p(t * top), grid, table, right=np.inf)  # off it

    return at


def _profile_interval(
    loglik: Callable[[float], NDArray[np.float64]],
    excess: float,
    drop: float,
    limit: float = -math.inf,
) -> tuple[float, float]:
    """The d around ``excess`` whose ``loglik(d)`` grid's top is within ``drop``.

    The grid is (tail shares, xi); bisected below, and above after doubling
    until it falls short. The upper end is inf when the likelihood's
    ``limit`` as d grows stays within ``drop``, or it never falls short
    within 2^``REACH``.
    """

    def lift(line: NDArray[np.float64], i: int) -> float:
        """The parabola's rise above ``line[i]`` through it and its neighbours."""
        if not 0 < i < len(line) - 1:
            return 0.0
        left, best, right = (float(v) for v in line[i - 1 : i + 2])
        bend = 2 * best - left - right
        ok = math.isfinite(bend) and bend > 0
        return (left - right) ** 2 / (8 * bend) if ok else 0.0

    def profile(d: float) -> float:
        grid = loglik(d)
        row, i = (int(v) for v in np.unravel_index(int(np.argmax(grid)), grid.shape))
        best = float(grid[row, i])
        if math.isfinite(best):  # refined in xi and in the tail share
            best += lift(grid[row], i) + lift(grid[:, i], row)
        return best

    cut = profile(excess) - drop
    low, high = 0.0, excess
    for _ in range(PROFILE_STEPS):
        mid = (low + high) / 2
        low, high = (mid, high) if profile(mid) < cut else (low, mid)
    lower = high
    if limit >= cut:
        return lower, math.inf
    top = excess
    for _ in range(REACH):
        top *= 2
        if profile(top) < cut:
            break
    else:
        return lower, math.inf
    low, high = top / 2, top
    for _ in range(PROFILE_STEPS):
        mid = (low + high) / 2
        low, high = (low, mid) if profile(mid) < cut else (mid, high)
    return lower, low


def evt_intervals(
    r: NDArray[np.float64], level: float, conf: float = CI_LEVEL
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Profile-likelihood intervals of ``evt_var_es``: ((VaR low, high), (ES ...)).

    McNeil-Frey-Embrechts (2015, 5.2): the GPD's scale is written through
    the VaR's excess d over u, scale = d / g(xi) with g = (c^-xi - 1) / xi
    and c = (1 - level) / zeta, or the ES's, d (1 - xi) / (g + 1). The tail
    share zeta = P(loss > u), estimated k / n, is a parameter too, with
    the binomial likelihood of k of n losses over u, so an interval near
    the threshold widens for it. The interval holds each d whose
    log-likelihood, maximised over xi from ``MIN_XI`` to ``XI_TOP`` and
    zeta within ``SHARE_REACH`` sds, is within chi2_1(conf) / 2 of the
    estimate's. It takes today's
    volatility as given, as ``evt_var_es`` does. An upper end the data
    cannot bound is inf: for ES, when the likelihood at xi = 1 (where ES
    grows without end) is within the drop. None as ``evt_var_es``, or when
    the MLE is not regular (``gpd_mle``).
    """
    found = _excesses(r, level)
    if found is None or float(found[1].max()) <= 0:
        return None
    u, y, log_ratio = found
    mle = gpd_mle(y)
    if mle is None or mle[0] >= 1:
        return None
    var, es = _evt(u, *mle, log_ratio)
    n, k = len(r), len(y)
    share = k / n
    zeta = share + math.sqrt(share * (1 - share) / n) * np.linspace(
        -SHARE_REACH, SHARE_REACH, SHARE_GRID
    )
    binomial = k * np.log(zeta / share) + (n - k) * np.log((1 - zeta) / (1 - share))
    step = (XI_TOP - MIN_XI) / PROFILE_GRID
    xi = np.linspace(MIN_XI + step / 2, XI_TOP - step / 2, PROFILE_GRID)  # never 0
    grow = np.expm1(-xi * np.log((1 - level) / zeta)[:, None]) / xi  # (zeta, xi)
    summed = _summed(y)
    top = float(y.max())

    def loglik(per: NDArray[np.float64]) -> Callable[[float], NDArray[np.float64]]:
        def at(d: float) -> NDArray[np.float64]:
            scale = d * per
            t = xi / scale
            with np.errstate(divide="ignore", invalid="ignore"):
                grid = -k * np.log(scale) - (1 + 1 / xi) * summed(t)
            ok = (scale > 0) & (t * top > -1)
            return np.where(ok, grid, -np.inf) + binomial[:, None]

        return at

    drop = NormalDist().inv_cdf((1 + conf) / 2) ** 2 / 2
    var_low, var_high = _profile_interval(loglik(1 / grow), var - u, drop)
    scales = top * np.logspace(-4, 4, LIMIT_GRID)
    limit = float(_gpd_loglik(y, np.ones(LIMIT_GRID), scales).max())  # ES: xi -> 1
    es_per = (1 - xi) / (grow + 1)
    es_low, es_high = _profile_interval(loglik(es_per), es - u, drop, limit)
    return (u + var_low, u + var_high), (u + es_low, u + es_high)


def component_es(
    rets: NDArray[np.float64], w: NDArray[np.float64], level: float = TAIL_LEVEL
) -> NDArray[np.float64] | None:
    """Tasche component ES: each holding's average loss on the tail days.

    The tail days are the portfolio's worst (as in ``var_es``); the parts
    sum to the portfolio ES. None below ``min_samples``.
    """
    if len(rets) < min_samples(level):
        return None
    weight = _tail_weights(-(rets @ w), level)
    return -(weight @ rets) * w


def incremental_es(
    rets: NDArray[np.float64],
    w: NDArray[np.float64],
    candidate: NDArray[np.float64],
    fraction: float,
    level: float = TAIL_LEVEL,
) -> float | None:
    """Change in ES from putting ``fraction`` into ``candidate``'s returns.

    The holdings are scaled down pro rata; full revaluation on the same days.
    """
    if len(rets) < min_samples(level):
        return None
    before = rets @ w
    after = (1 - fraction) * before + fraction * candidate
    return float(_tail(-after, level)[1] - _tail(-before, level)[1])


# ---------------------------------------------------------------- backtest


def _xlogy(x: float, y: float) -> float:
    return x * math.log(y) if x > 0 else 0.0


def _chi2_p(stat: float) -> float:
    """Upper tail of chi-squared with 1 degree of freedom."""
    return math.erfc(math.sqrt(max(stat, 0.0) / 2))


def kupiec(exceptions: int, days: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test: (LR, p-value)."""
    x, n = exceptions, days
    null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    fit = _xlogy(n - x, 1 - x / n) + _xlogy(x, x / n)
    stat = -2 * (null - fit)
    return stat, _chi2_p(stat)


def christoffersen(hits: NDArray[np.bool_]) -> tuple[float, float]:
    """Christoffersen independence test of exception clustering: (LR, p)."""
    before, after = hits[:-1], hits[1:]
    n00 = int((~before & ~after).sum())
    n01 = int((~before & after).sum())
    n10 = int((before & ~after).sum())
    n11 = int((before & after).sum())
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    null = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    fit = (
        _xlogy(n00, 1 - p01)
        + _xlogy(n01, p01)
        + _xlogy(n10, 1 - p11)
        + _xlogy(n11, p11)
    )
    stat = -2 * (null - fit)
    return stat, _chi2_p(stat)


def traffic_light(exceptions: int, days: int, p: float) -> Light:
    """Basel-style zone from the binomial CDF of the exception count."""
    cdf = sum(
        math.comb(days, i) * p**i * (1 - p) ** (days - i) for i in range(exceptions + 1)
    )
    if cdf < GREEN:
        return "green"
    return "yellow" if cdf < YELLOW else "red"


@dataclass(frozen=True, slots=True)
class Backtest:
    days: int
    exceptions: int  # days whose loss beat that day's VaR
    expected: float
    kupiec: float  # p-value
    christoffersen: float  # p-value
    light: Light
    z2: float  # Acerbi-Székely; below 0 when ES is too small
    es: float  # one-sided p-value of z2
    de_u: float  # Du-Escanciano unconditional, two-sided p-value
    de_c: float  # Du-Escanciano conditional, p-value


def past_windows(
    r: NDArray[np.float64], days: int = BACKTEST_DAYS, lookback: int = BACKTEST_DAYS
) -> NDArray[np.float64] | None:
    """Historical forecasts for each of the last ``days``, (days, lookback).

    Each is the ``lookback`` returns before it; None when there are fewer
    than days + lookback.
    """
    if len(r) < days + lookback:
        return None
    return np.lib.stride_tricks.sliding_window_view(r[-days - lookback : -1], lookback)


def filtered_windows(
    f: Devolatilised,
    w: NDArray[np.float64],
    days: int = BACKTEST_DAYS,
    lookback: int = BACKTEST_DAYS,
) -> NDArray[np.float64] | None:
    """Filtered-HS forecasts for each of the last ``days``, as ``past_windows``.

    Each is the ``lookback`` residual days before it at that day's
    volatility, weighted by ``w``. Fit ``f`` on the days before them
    (``devolatilise(r, fit=len(r) - days)``) to keep them out of sample.
    """
    if len(f.z) < days + lookback:
        return None
    z = np.lib.stride_tricks.sliding_window_view(
        f.z[-days - lookback : -1], lookback, axis=0
    )  # (days, assets, lookback)
    return np.einsum("dal,da->dl", z, f.sigma[-days:] * w)


def evt_windows(
    past: NDArray[np.float64], level: float = BACKTEST_LEVEL
) -> NDArray[np.float64] | None:
    """EVT forecasts from filtered-HS ones (``filtered_windows``), for ``backtest``.

    McNeil-Frey's forecast as scenarios: in each day's row the k = 10%
    largest losses (``_excesses``) become u plus the fitted GPD's quantiles
    at tail shares (i - 0.5) / k, i = 1..k; the rest stay. A row whose fit
    has xi >= 1 (no finite ES) keeps its scenarios. None under
    ``MIN_EXCESSES`` excesses a row.
    """
    out = past.copy()
    for row in out:
        found = _excesses(row, level)
        if found is None:
            return None
        u, y, _ = found
        xi, scale = gpd_fit(y)
        if xi >= 1:
            continue
        k = len(y)
        log_share = np.log((np.arange(k, 0, -1) - 0.5) / k)  # y ascends
        grow = -log_share if xi == 0 else np.expm1(-xi * log_share) / xi
        row[np.argsort(-row)[-k:]] = -(u + scale * grow)
    return out


def _z2(
    loss: NDArray[np.float64],
    var: NDArray[np.float64],
    es: NDArray[np.float64],
    p: float,
) -> NDArray[np.float64]:
    beyond = np.where(loss > var, loss, 0.0) / es
    return 1 - beyond.sum(axis=-1) / (loss.shape[-1] * p)


def du_escanciano(
    pit: NDArray[np.float64], p: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Du-Escanciano (2017) U and C(``DE_LAGS``) of PITs along the last axis.

    Cumulative violations H = (p - u) / p where u <= p, else 0, have mean
    p / 2 and variance p (1/3 - p/4) under a right forecast: U = sqrt(n)
    (mean H - p/2) / sqrt(p (1/3 - p/4)), and C = n sum rho_j^2, rho_j the
    lag-j autocorrelation of H - p/2 (Box-Pierce).
    """
    h = np.where(pit <= p, (p - pit) / p, 0.0) - p / 2
    n = h.shape[-1]
    u = math.sqrt(n) * h.mean(axis=-1) / math.sqrt(p * (1 / 3 - p / 4))
    lags = range(1, DE_LAGS + 1)
    gamma = np.stack([(h[..., j:] * h[..., :-j]).mean(axis=-1) for j in lags], -1)
    rho = gamma / (h**2).mean(axis=-1)[..., None]
    return u, n * (rho**2).sum(axis=-1)


def backtest(
    r: NDArray[np.float64],
    past: NDArray[np.float64],
    level: float = BACKTEST_LEVEL,
    sims: int = ES_SIMS,
    seed: int = SEED,
) -> Backtest:
    """Each day's VaR and ES at ``level`` against its realised return ``r``.

    ``past`` (days, scenarios) are each day's forecast returns
    (``past_windows``, ``filtered_windows``). ES by Acerbi-Székely (2014)
    Z2 = 1 - sum(L_t [L_t > VaR_t] / ES_t) / (T (1 - level)): its p-value is
    the share of ``sims`` draws, one scenario a day from the forecasts
    themselves, with a Z2 at or below it, so the discrete forecasts need no
    asymptotics or fixed thresholds. Du-Escanciano's tests take each day's
    PIT, the mid-rank share of its scenarios below the return, and get
    their p-values from the same draws: U's from |U|, C's from C.
    """
    days, count_scenarios = past.shape
    p = 1 - level
    var, es = _tail(-past, level)
    hits = -r > var
    count = int(hits.sum())
    z2 = float(_z2(-r, var, es, p))
    picks = np.random.default_rng(seed).integers(0, count_scenarios, size=(sims, days))
    null = _z2(-past[np.arange(days), picks], var, es, p)
    below = (past < r[:, None]).sum(axis=1) + 0.5 * (past == r[:, None]).sum(axis=1)
    u, c = du_escanciano(below / count_scenarios, p)
    rank = np.empty((days, count_scenarios), dtype=np.int64)
    order = np.argsort(past, axis=1)
    np.put_along_axis(rank, order, np.arange(count_scenarios)[None, :], axis=1)
    u_null, c_null = du_escanciano(
        (rank[np.arange(days), picks] + 0.5) / count_scenarios, p
    )
    return Backtest(
        days=days,
        exceptions=count,
        expected=days * p,
        kupiec=kupiec(count, days, p)[1],
        christoffersen=christoffersen(hits)[1],
        light=traffic_light(count, days, p),
        z2=z2,
        es=float((1 + (null <= z2).sum()) / (1 + sims)),
        de_u=float((1 + (np.abs(u_null) >= abs(float(u))).sum()) / (1 + sims)),
        de_c=float((1 + (c_null >= c).sum()) / (1 + sims)),
    )


# ---------------------------------------------------------------- density forecasts


def blocks(
    r: NDArray[np.float64], days: int = MONTH_DAYS
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Compounded returns of non-overlapping runs of ``days``, ending on the last day.

    Returns (outcomes, first rows): a forecast for a block uses rows before
    its first only. The earliest len(r) mod days rows are left out.
    """
    count = len(r) // days
    starts = len(r) - count * days + days * np.arange(count, dtype=np.int64)
    total = np.concatenate([[0.0], np.cumsum(np.log1p(r))])
    return np.expm1(total[starts + days] - total[starts]), starts


def pit(
    outcomes: NDArray[np.float64], samples: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Each outcome's PIT under its forecast's draws ``samples`` (n, draws).

    The outcome's mid-rank among itself and the draws, (below + ties / 2 +
    1/2) / (draws + 1): never 0 or 1, so the normal transform stays finite.
    """
    below = (samples < outcomes[:, None]).sum(axis=1)
    ties = (samples == outcomes[:, None]).sum(axis=1)
    return (below + 0.5 * ties + 0.5) / (samples.shape[1] + 1)


@dataclass(frozen=True, slots=True)
class Berkowitz:
    lr: float  # chi-squared, 3 degrees of freedom
    p: float
    mean: float  # of the normal-transformed PITs: 0 under the forecasts
    sd: float  # 1; below 1 when the forecasts are too wide
    rho: float  # first-order autocorrelation: 0


def _chi2_3(stat: float) -> float:
    """Upper tail of chi-squared with 3 degrees of freedom."""
    x = max(stat, 0.0)
    return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)


def berkowitz(pits: NDArray[np.float64]) -> Berkowitz | None:
    """Berkowitz (2001) LR test of PITs (``pit``) as independent uniforms.

    With x = Phi^-1(PIT), the AR(1) x_t = c + rho x_{t-1} + s e_t, fitted
    by conditional maximum likelihood (OLS), against c = rho = 0 and s = 1:
    LR = 2 (L1 - L0), chi-squared(3). Pass non-overlapping outcomes
    (``blocks``). None under ``MIN_MONTHS`` PITs.
    """
    n = len(pits)
    if n < MIN_MONTHS:
        return None
    normal = NormalDist()
    x = np.array([normal.inv_cdf(float(u)) for u in pits])
    design = np.column_stack([np.ones(n - 1), x[:-1]])
    coef = np.linalg.lstsq(design, x[1:], rcond=None)[0]
    resid = x[1:] - design @ coef
    s2 = float(resid @ resid) / (n - 1)
    fitted = -(n - 1) / 2 * (math.log(2 * math.pi * s2) + 1)
    null = -(n - 1) / 2 * math.log(2 * math.pi) - float(x[1:] @ x[1:]) / 2
    lr = 2 * (fitted - null)
    return Berkowitz(
        lr, _chi2_3(lr), float(x.mean()), float(x.std(ddof=1)), float(coef[1])
    )


# ---------------------------------------------------------------- stress


def _extreme(
    prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp, *, low: bool
) -> pd.Timestamp | None:
    inside = prices.dropna().loc[start:end]
    if inside.empty:
        return None
    return cast("pd.Timestamp", inside.idxmin() if low else inside.idxmax())


def union_window(
    first: pd.Series,
    second: pd.Series,
    peak: pd.Timestamp,
    trough: pd.Timestamp,
    search: int = SEARCH_DAYS,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Earlier of two indices' peaks to the later of their troughs.

    Each peak is the highest close within ``search`` days of ``peak`` (and
    not after ``trough``); each trough the lowest close after that peak,
    from ``search`` days before ``trough`` to ``search`` days after it.
    None when a series has no closes there.
    """
    reach = pd.Timedelta(days=search)
    early = cast("pd.Timestamp", peak - reach)
    late = min(cast("pd.Timestamp", peak + reach), trough)
    floor = cast("pd.Timestamp", trough - reach)
    end = cast("pd.Timestamp", trough + reach)
    peaks: list[pd.Timestamp] = []
    troughs: list[pd.Timestamp] = []
    for prices in (first, second):
        top = _extreme(prices, early, late, low=False)
        if top is None:
            return None
        bottom = _extreme(prices, max(top, floor), end, low=True)
        if bottom is None:
            return None
        peaks.append(top)
        troughs.append(bottom)
    return min(peaks), max(troughs)


@dataclass(frozen=True, slots=True)
class Replay:
    change: float  # first to last day of the window
    worst: float  # deepest fall from a running peak inside it, e.g. -0.3


def replay(prices: NDArray[np.float64], w: NDArray[np.float64]) -> Replay:
    """Buy-and-hold of weights ``w`` through a window's prices (days, assets)."""
    value = (prices / prices[0]) @ w / w.sum()
    peak = np.maximum.accumulate(value)
    return Replay(float(value[-1] / value[0] - 1), float((value / peak - 1).min()))


def hypothetical(
    w: NDArray[np.float64],
    betas: NDArray[np.float64],
    index_moves: NDArray[np.float64],
    fx_moves: NDArray[np.float64],
) -> float:
    """Portfolio change when each holding's index moves and its currency does.

    A holding moves beta x its own index's move (floored at -100%) in its
    own currency, then (1 + local)(1 + fx) - 1 in the base currency.
    """
    local = np.maximum(betas * index_moves, -1.0)
    return float(w @ ((1 + local) * (1 + fx_moves) - 1) / w.sum())


def reverse_stress(
    w: NDArray[np.float64],
    betas: NDArray[np.float64],
    loss: float,
    fx_moves: NDArray[np.float64],
) -> float | None:
    """The same fall in every index that costs the portfolio ``loss``.

    Bisection on -100%..0; None when even -100% falls short of it.
    """

    def gap(shock: float) -> float:
        moves = np.full(len(w), shock)
        return hypothetical(w, betas, moves, fx_moves) + loss

    if gap(0.0) <= 0:
        return 0.0
    if gap(-1.0) > 0:
        return None
    low, high = -1.0, 0.0
    for _ in range(BISECTIONS):
        mid = (low + high) / 2
        if gap(mid) > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def bad_prints(prices: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Closes that jump by more than 0.5 in log and come back the next day.

    Back means the next day's log move goes the other way and takes back
    more than half the jump (|l1 + l2| < 0.5 |l1|), so a crash followed by a
    partial rebound is kept.
    """
    logs = np.diff(np.log(prices))
    jump = np.abs(logs[:-1]) > JUMP
    back = np.abs(logs[:-1] + logs[1:]) < REVERSAL * np.abs(logs[:-1])
    flagged = np.zeros(len(prices), dtype=np.bool_)
    flagged[1:-1] = jump & back
    return flagged


# ---------------------------------------------------------------- turbulence


@dataclass(frozen=True, slots=True)
class Turbulence:
    index: NDArray[np.float64]  # d_t for rows window..n-1; about 1 when calm
    surprise: NDArray[np.float64]  # d_t over its correlation-blind part


def turbulence(
    rets: NDArray[np.float64], window: int = TURBULENCE_WINDOW, refit: int = 1
) -> Turbulence:
    """Mahalanobis turbulence d_t = z' inv(S) z / n with z = r_t - mu.

    mu and S (Ledoit-Wolf) come from the ``window`` days before t, refitted
    every ``refit`` days. The correlation surprise divides d_t by the mean
    of z_i^2 / s_ii, the part a diagonal S would give.
    """
    n, assets = rets.shape
    index = np.empty(max(n - window, 0))
    surprise = np.empty_like(index)
    for start in range(window, n, refit):
        past = rets[start - window : start]
        cov = ledoit_wolf(past)[0]
        z = rets[start : start + refit] - past.mean(axis=0)
        d = np.einsum("ij,ij->i", z @ np.linalg.inv(cov), z) / assets
        size = (z**2 / np.diag(cov)).mean(axis=1)
        rows = slice(start - window, start - window + len(z))
        index[rows] = d
        surprise[rows] = np.divide(d, size, out=np.full_like(d, np.nan), where=size > 0)
    return Turbulence(index, surprise)


def percentile_rank(history: NDArray[np.float64], value: float) -> float:
    """Share of ``history`` at or below ``value``."""
    return float((history <= value).mean())
