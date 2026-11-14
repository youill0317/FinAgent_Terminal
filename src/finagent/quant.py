# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""정상 블록 부트스트랩·Ledoit-Wolf 수축 공분산·Welch 베타·VaR/ES 백테스트 수리 모듈."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

SEED: Final = 42
MONTH_DAYS: Final = 21
QUIET_LAGS: Final = 5
TAIL_LEVEL: Final = 0.95
MIN_TAIL: Final = 12
MIN_MONTHS: Final = 24
BACKTEST_LEVEL: Final = 0.95
BACKTEST_DAYS: Final = 250
ES_SIMS: Final = 5000
DE_LAGS: Final = 5
GREEN: Final = 0.95
YELLOW: Final = 0.9999
WELCH_DELTA: Final = 3.0
WELCH_DECAY: Final = 2 / 256
DIGITS: Final = 9

type Light = Literal["green", "yellow", "red"]


def politis_white(x: NDArray[np.float64]) -> float:
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
    quiet = np.convolve(small, np.ones(run), "valid") == run
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
    return max(1.0, politis_white(r))


def stationary_indices(
    n: int,
    paths: int,
    horizon: int,
    mean_block: float,
    seed: int | np.random.Generator = SEED,
) -> NDArray[np.int64]:
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
    picks -= began
    picks += step
    picks %= n
    return picks


def growth_paths(
    rets: NDArray[np.float64],
    w: NDArray[np.float64],
    picks: NDArray[np.int64],
    *,
    demean: bool = True,
) -> NDArray[np.float64]:
    steps = rets @ w
    daily = (steps - steps.mean() if demean else steps)[picks]
    _ = np.maximum(daily, -1.0, out=daily)
    daily += 1.0
    growth = np.ones((len(picks), picks.shape[1] + 1))
    _ = np.cumprod(daily, axis=1, out=growth[:, 1:])
    return growth


def correlation(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    sd = np.sqrt(np.diag(cov))
    outer = np.outer(sd, sd)
    corr = np.divide(cov, outer, out=np.zeros_like(cov), where=outer > 0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _mean_correlation(sample: NDArray[np.float64]) -> float:
    n = len(sample)
    return float((correlation(sample).sum() - n) / (n * (n - 1)))


def _target(sample: NDArray[np.float64]) -> NDArray[np.float64]:
    sd = np.sqrt(np.diag(sample))
    target = _mean_correlation(sample) * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(sample))
    return target


def _intensity(x: NDArray[np.float64]) -> float:
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
    pi = sq.T @ sq / t - s**2
    theta = (y**3).T @ y / t - np.diag(s)[:, None] * s
    ratio = sd[None, :] / sd[:, None]
    off = mean / 2 * (ratio * theta + ratio.T * theta.T)
    np.fill_diagonal(off, 0.0)
    rho = float(np.trace(pi) + off.sum())
    kappa = (float(pi.sum()) - rho) / gamma
    return min(max(kappa / t, 0.0), 1.0)


def ledoit_wolf(rets: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    t, n = rets.shape
    y = rets - rets.mean(axis=0)
    cov = y.T @ y / t
    if n < 2:  # noqa: PLR2004
        return cov, 0.0
    weight = _intensity(rets)
    return weight * _target(cov) + (1 - weight) * cov, weight


def welch_beta(
    r: NDArray[np.float64],
    m: NDArray[np.float64],
    delta: float = WELCH_DELTA,
    decay: float = WELCH_DECAY,
) -> float:
    bounds = np.stack([(1 - delta) * m, (1 + delta) * m])
    clipped = np.clip(r, bounds.min(axis=0), bounds.max(axis=0))
    weight = np.exp(-decay * np.arange(len(r) - 1, -1, -1, dtype=np.float64))
    weight /= weight.sum()
    dm = m - weight @ m
    spread = float((weight * dm) @ dm)
    if spread <= 0:
        return math.nan
    return float((weight * dm) @ (clipped - weight @ clipped)) / spread


def risk_contributions(
    w: NDArray[np.float64], cov: NDArray[np.float64]
) -> NDArray[np.float64]:
    marginal = cov @ w
    var = float(w @ marginal)
    if var <= 0:
        return np.full(len(w), np.nan)
    return w * marginal / var


def diversification_ratio(w: NDArray[np.float64], cov: NDArray[np.float64]) -> float:
    var = float(w @ cov @ w)
    if var <= 0:
        return math.nan
    return float(w @ np.sqrt(np.diag(cov))) / math.sqrt(var)


def effective_holdings(w: NDArray[np.float64]) -> float:
    share = w / w.sum()
    return 1 / float(share @ share)


def min_samples(level: float) -> int:
    return math.ceil(round(MIN_TAIL / (1 - level), DIGITS))


def _tail(
    loss: NDArray[np.float64], level: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = loss.shape[-1]
    m = round(n * (1 - level), DIGITS)
    top = math.ceil(m)
    part = np.partition(loss, n - top, axis=-1)
    var = part[..., n - top]
    es = (part[..., n - top + 1 :].sum(axis=-1) + (m - top + 1) * var) / m
    return var, es


def _tail_weights(loss: NDArray[np.float64], level: float) -> NDArray[np.float64]:
    n = len(loss)
    m = round(n * (1 - level), DIGITS)
    top = math.ceil(m)
    order = np.argpartition(loss, n - top)
    weight = np.zeros(n)
    weight[order[n - top + 1 :]] = 1.0
    weight[order[n - top]] = m - top + 1
    return weight / m


def var_es(r: NDArray[np.float64], level: float) -> tuple[float, float] | None:
    if len(r) < min_samples(level):
        return None
    var, es = _tail(-r, level)
    return float(var), float(es)


def component_es(
    rets: NDArray[np.float64], w: NDArray[np.float64], level: float = TAIL_LEVEL
) -> NDArray[np.float64] | None:
    if len(rets) < min_samples(level):
        return None
    weight = _tail_weights(-(rets @ w), level)
    return -(weight @ rets) * w


def _xlogy(x: float, y: float) -> float:
    return x * math.log(y) if x > 0 else 0.0


def _chi2_p(stat: float) -> float:
    return math.erfc(math.sqrt(max(stat, 0.0) / 2))


def kupiec(exceptions: int, days: int, p: float) -> tuple[float, float]:
    x, n = exceptions, days
    null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    fit = _xlogy(n - x, 1 - x / n) + _xlogy(x, x / n)
    stat = -2 * (null - fit)
    return stat, _chi2_p(stat)


def christoffersen(hits: NDArray[np.bool_]) -> tuple[float, float]:
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
    cdf = sum(
        math.comb(days, i) * p**i * (1 - p) ** (days - i) for i in range(exceptions + 1)
    )
    if cdf < GREEN:
        return "green"
    return "yellow" if cdf < YELLOW else "red"


@dataclass(frozen=True, slots=True)
class Backtest:
    days: int
    exceptions: int
    expected: float
    kupiec: float
    christoffersen: float
    light: Light
    z2: float
    es: float
    de_u: float
    de_c: float


def past_windows(
    r: NDArray[np.float64], days: int = BACKTEST_DAYS, lookback: int = BACKTEST_DAYS
) -> NDArray[np.float64] | None:
    if len(r) < days + lookback:
        return None
    return np.lib.stride_tricks.sliding_window_view(r[-days - lookback : -1], lookback)


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


def blocks(
    r: NDArray[np.float64], days: int = MONTH_DAYS
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    count = len(r) // days
    starts = len(r) - count * days + days * np.arange(count, dtype=np.int64)
    total = np.concatenate([[0.0], np.cumsum(np.log1p(r))])
    return np.expm1(total[starts + days] - total[starts]), starts


def pit(
    outcomes: NDArray[np.float64], samples: NDArray[np.float64]
) -> NDArray[np.float64]:
    below = (samples < outcomes[:, None]).sum(axis=1)
    ties = (samples == outcomes[:, None]).sum(axis=1)
    return (below + 0.5 * ties + 0.5) / (samples.shape[1] + 1)


@dataclass(frozen=True, slots=True)
class Berkowitz:
    lr: float
    p: float
    mean: float
    sd: float
    rho: float


def _chi2_3(stat: float) -> float:
    x = max(stat, 0.0)
    return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)


def berkowitz(pits: NDArray[np.float64]) -> Berkowitz | None:
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
