"""Portfolio risk math: numpy and pandas in, numbers out (no files, no network).

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

SEED: Final = 7
TRADING_DAYS: Final = 252
MONTH_DAYS: Final = 21
EWMA_LAMBDA: Final = 0.94  # RiskMetrics daily
REVERSION: Final = 0.985  # daily pull of FHS variance back to its long-run level
MIN_BLOCK: Final = 5
MAX_BLOCK: Final = 20
MIXED_BLOCK: Final = 10  # Korea/US mixes: keeps about 90% of the lag-1 co-movement
MIN_BLOCKS: Final = 10  # a window holds at least this many mean blocks
TAIL_LEVEL: Final = 0.975  # the ES level shown as the tail figure
MIN_SAMPLES: Final = ((0.95, 250), (0.975, 400), (0.99, 1000))  # level, days
BAND: Final = (5.0, 95.0)  # percentiles of the bootstrap ES band
BAND_PATHS: Final = 1000
BACKTEST_LEVEL: Final = 0.95
BACKTEST_DAYS: Final = 250
GREEN: Final = 0.95  # binomial CDF of the exception count below which: green
YELLOW: Final = 0.9999  # ... below which: yellow; else red
BETA_PRIOR_DAYS: Final = 250  # beta is shrunk to 1 by n / (n + this)
WELCH_DELTA: Final = 3.0
WELCH_DECAY: Final = 2 / 256
PSD_TOLERANCE: Final = 1e-10  # smallest eigenvalue >= -this x the largest
BISECTIONS: Final = 60
SEARCH_DAYS: Final = 90  # calendar days around a stated peak or trough
JUMP: Final = 0.5  # |log return| of a bad print, reverted the next day
TURBULENCE_WINDOW: Final = 250
DIGITS: Final = 9  # n x (1 - level) is rounded to this before ceil and floor

type Light = Literal["green", "yellow", "red"]


# ---------------------------------------------------------------- Monte Carlo


def block_length(n: int, *, mixed: bool) -> int:
    """Mean stationary-bootstrap block for an ``n``-day window.

    round(n^(1/3)) clipped to 5..20, at least 10 for a Korea/US mix, and at
    most n / 10 so the window still holds ten blocks.
    """
    length = min(max(round(n ** (1 / 3)), MIN_BLOCK), MAX_BLOCK)
    if mixed:
        length = max(length, MIXED_BLOCK)
    return max(1, min(length, n // MIN_BLOCKS))


def stationary_indices(
    n: int, paths: int, horizon: int, mean_block: float, seed: int = SEED
) -> NDArray[np.int64]:
    """Politis-Romano draws of historical days, shape (paths, horizon).

    Blocks start at uniform days and run for geometric lengths with mean
    ``mean_block``, wrapping from the last day to the first. Draw once and
    reuse: any weight vector then replays the same days.
    """
    rng = np.random.default_rng(seed)
    if mean_block <= 1:
        return rng.integers(0, n, size=(paths, horizon))
    fresh = rng.random((paths, horizon)) < 1 / mean_block
    fresh[:, 0] = True
    starts = rng.integers(0, n, size=(paths, horizon))
    step = np.arange(horizon)
    began = np.where(fresh, step, 0)
    _ = np.maximum.accumulate(began, axis=1, out=began)
    first = np.take_along_axis(starts, began, axis=1)
    return (first + step - began) % n


def ewma_sigma(
    r: NDArray[np.float64], lam: float = EWMA_LAMBDA
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """EWMA volatility of each column: day t's uses returns before t only.

    Zero-mean RiskMetrics form, seeded at the sample mean of r^2. Returns
    (sigma per day, shaped like ``r``; the forecast for the next day).
    """
    level = np.mean(r**2, axis=0)
    var = np.empty_like(r)
    for t in range(len(r)):  # days; vectorised over assets
        var[t] = level
        level = lam * level + (1 - lam) * r[t] ** 2
    return np.sqrt(var), np.sqrt(level)


@dataclass(frozen=True, slots=True)
class Devolatilised:
    z: NDArray[np.float64]  # r_t / sigma_t, shaped like the returns
    now: NDArray[np.float64]  # sigma for the next day, per asset
    long_run: NDArray[np.float64]  # root mean square return, per asset


def devolatilise(r: NDArray[np.float64], lam: float = EWMA_LAMBDA) -> Devolatilised:
    """Returns divided by their own EWMA volatility (filtered HS).

    Run it on the full history so the filter is warm, then take the window's
    rows of ``z``. A column with no movement gets residuals of 0.
    """
    sigma, now = ewma_sigma(r, lam)
    z = np.divide(r, sigma, out=np.zeros_like(r), where=sigma > 0)
    return Devolatilised(z, now, np.sqrt(np.mean(r**2, axis=0)))


def revolatilise(
    now: NDArray[np.float64],
    long_run: NDArray[np.float64],
    horizon: int,
    phi: float = REVERSION,
) -> NDArray[np.float64]:
    """Volatility of each asset on each simulated day, shape (horizon, assets).

    sigma_k^2 = long^2 + phi^k (now^2 - long^2): today's level at first,
    the long-run level a year out.
    """
    k = np.arange(horizon, dtype=np.float64)[:, None]
    var = long_run**2 + phi**k * (now**2 - long_run**2)
    return np.sqrt(np.maximum(var, 0.0))


def filtered_returns(
    r: NDArray[np.float64], lam: float = EWMA_LAMBDA
) -> NDArray[np.float64]:
    """Filtered-HS scenarios: each day's residual at today's volatility.

    Same shape as ``r``; for a portfolio, multiply the asset matrix by the
    weights afterwards so each holding keeps its own volatility.
    """
    f = devolatilise(r, lam)
    return f.z * f.now


def growth_paths(
    rets: NDArray[np.float64],
    w: NDArray[np.float64],
    picks: NDArray[np.int64],
    scale: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Growth of 1 along drawn paths, shape (paths, horizon + 1).

    ``rets`` (days, assets) are the days ``picks`` index into; weights are
    reset to ``w`` daily. ``scale`` (horizon, assets), from ``revolatilise``,
    multiplies each asset on each simulated day; ``rets`` are then the
    devolatilised residuals. Portfolio returns are formed first, one column
    per simulated day at most, so no paths x days x assets array is built.
    Each column is demeaned (zero drift) and floored at -100%.
    """
    mix = w[None, :] if scale is None else scale * w
    steps = rets @ mix.T  # (days, 1) or (days, horizon)
    steps -= steps.mean(axis=0)
    _ = np.maximum(steps, -1.0, out=steps)
    paths, horizon = picks.shape
    columns = 0 if steps.shape[1] == 1 else np.arange(horizon)
    daily = steps[picks, columns]
    daily += 1.0
    growth = np.ones((paths, horizon + 1))
    _ = np.cumprod(daily, axis=1, out=growth[:, 1:])
    return growth


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


def shrunk_beta(beta: float, n: int, prior: int = BETA_PRIOR_DAYS) -> float:
    """``beta`` from ``n`` days pulled toward 1 with weight n / (n + prior)."""
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
    """Days needed before a VaR or ES at ``level`` is shown (else 표본 부족)."""
    for top, days in MIN_SAMPLES:
        if level <= top:
            return days
    return MIN_SAMPLES[-1][1]


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

    For filtered HS pass ``filtered_returns(r)`` (times the weights).
    """
    if len(r) < min_samples(level):
        return None
    var, es = _tail(-r, level)
    return float(var), float(es)


def overlapping(r: NDArray[np.float64], days: int = MONTH_DAYS) -> NDArray[np.float64]:
    """Compounded returns over every run of ``days`` consecutive days."""
    total = np.concatenate([[0.0], np.cumsum(np.log1p(r))])
    return np.expm1(total[days:] - total[:-days])


def month_var(r: NDArray[np.float64], level: float) -> tuple[float, float] | None:
    """1-month VaR two ways: (overlapping 21-day, sqrt(21) x 1-day).

    The overlapping figure rests on only n / 21 independent months.
    """
    if len(r) < min_samples(level):
        return None
    month = float(_tail(-overlapping(r), level)[0])
    day = float(_tail(-r, level)[0])
    return month, math.sqrt(MONTH_DAYS) * day


def es_band(
    r: NDArray[np.float64],
    level: float = TAIL_LEVEL,
    mean_block: float | None = None,
    paths: int = BAND_PATHS,
    seed: int = SEED,
) -> tuple[float, float] | None:
    """5th-95th percentile band of ES from a stationary bootstrap of ``r``.

    The mean block defaults to ``block_length``; None below ``min_samples``.
    """
    n = len(r)
    if n < min_samples(level):
        return None
    block = block_length(n, mixed=False) if mean_block is None else mean_block
    picks = stationary_indices(n, paths, n, block, seed)
    es = _tail(-r[picks], level)[1]
    low, high = np.percentile(es, BAND)
    return float(low), float(high)


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


def backtest(
    r: NDArray[np.float64],
    level: float = BACKTEST_LEVEL,
    days: int = BACKTEST_DAYS,
    lookback: int = BACKTEST_DAYS,
) -> Backtest | None:
    """Historical VaR on each of the last ``days``, out of sample.

    Each day's VaR uses only the ``lookback`` days before it; None when
    there are fewer than days + lookback returns.
    """
    if len(r) < days + lookback:
        return None
    past = np.lib.stride_tricks.sliding_window_view(r[-days - lookback : -1], lookback)
    var = _tail(-past, level)[0]
    hits = -r[-days:] > var
    count = int(hits.sum())
    p = 1 - level
    return Backtest(
        days=days,
        exceptions=count,
        expected=days * p,
        kupiec=kupiec(count, days, p)[1],
        christoffersen=christoffersen(hits)[1],
        light=traffic_light(count, days, p),
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

    Back means the two days together move less than 0.5 in log, so a crash
    followed by a partial rebound is kept.
    """
    logs = np.diff(np.log(prices))
    jump = np.abs(logs[:-1]) > JUMP
    back = np.abs(logs[:-1] + logs[1:]) < JUMP
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
