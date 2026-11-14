# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""포트폴리오 위험조정 성과(Sharpe·PSR·DSR·Sortino·Calmar·CDaR·MWR) 및 스타일 회귀."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

type Floats = npt.NDArray[np.float64]

TRADING_DAYS: Final = 252
YEAR_DAYS: Final = 365
MIN_OBS: Final = 252
CONFIDENCE: Final = 0.95
NORMAL_KURT: Final = 3.0
CDAR_LEVEL: Final = 0.95
PERCENT: Final = 100.0
PATH_LEVELS: Final = (0.10, 0.20, 0.30)
EULER_GAMMA: Final = 0.5772156649015329
TOLERANCE: Final = 1e-12
XIRR_REACH: Final = 100.0
XIRR_GRID: Final = 4001
XIRR_STEPS: Final = 200
BILL_DAYS: Final = 91
STYLE_STEPS: Final = 5000
STYLE_TOLERANCE: Final = 1e-10

_NORMAL: Final = NormalDist()


def _floats(values: npt.ArrayLike) -> Floats:
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SharpeCI:
    sharpe: float
    low: float
    high: float
    n: int


def sharpe(rets: npt.ArrayLike, rf: npt.ArrayLike = 0.0) -> float | None:
    x = _floats(rets) - _floats(rf)
    if len(x) < 2:  # noqa: PLR2004
        return None
    std = float(x.std(ddof=1))
    return float(x.mean()) / std if std > 0 else None


def skew_kurt(rets: npt.ArrayLike) -> tuple[float, float] | None:
    x = _floats(rets)
    d = x - x.mean()
    var = float((d**2).mean())
    if var <= 0:
        return None
    return float((d**3).mean()) / var**1.5, float((d**4).mean()) / var**2


def _spread(sr: float, skew: float, kurt: float) -> float:
    return 1 - skew * sr + (kurt - 1) / 4 * sr**2


def sharpe_se(sr: float, n: int, skew: float = 0.0, kurt: float = NORMAL_KURT) -> float:
    return math.sqrt(_spread(sr, skew, kurt) / n)


def sharpe_ci(
    rets: npt.ArrayLike,
    rf: npt.ArrayLike = 0.0,
    conf: float = CONFIDENCE,
    periods: float = TRADING_DAYS,
) -> SharpeCI | None:
    x = _floats(rets) - _floats(rf)
    sr = sharpe(x)
    moments = skew_kurt(x)
    if len(x) < MIN_OBS or sr is None or moments is None:
        return None
    se = sharpe_se(sr, len(x), *moments)
    z = _NORMAL.inv_cdf(0.5 + conf / 2)
    k = math.sqrt(periods)
    return SharpeCI(sr * k, (sr - z * se) * k, (sr + z * se) * k, len(x))


def psr_from(sr: float, n: int, skew: float, kurt: float, sr_star: float) -> float:
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(_spread(sr, skew, kurt))
    return _NORMAL.cdf(z)


def expected_max_sr(n_trials: int, var_trials: float) -> float | None:
    if n_trials < 2:  # noqa: PLR2004
        return None
    return math.sqrt(var_trials) * (
        (1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def drawdown_series(rets: npt.ArrayLike) -> Floats:
    growth = np.cumprod(1 + _floats(rets))
    peak = np.maximum.accumulate(np.concatenate([[1.0], growth]))[1:]
    return growth / peak - 1


def max_drawdown(rets: npt.ArrayLike) -> float:
    return float(drawdown_series(rets).min(initial=0.0))


def cdar(rets: npt.ArrayLike, alpha: float = CDAR_LEVEL) -> float:
    dd = np.sort(drawdown_series(rets))
    if not len(dd):
        return 0.0
    m = round((1 - alpha) * len(dd), 9)
    if m <= 0:
        return float(dd[0])
    top = math.ceil(m)
    return float((dd[: top - 1].sum() + (m - top + 1) * dd[top - 1]) / m)


def sortino(rets: npt.ArrayLike, mar: npt.ArrayLike = 0.0) -> float | None:
    x = _floats(rets) - _floats(mar)
    if not len(x):
        return None
    down = math.sqrt(float((np.minimum(x, 0.0) ** 2).mean()))
    return float(x.mean()) / down if down > 0 else None


def _cagr(rets: npt.ArrayLike, span_days: float) -> float | None:
    return annualise(float(np.prod(1 + _floats(rets))) - 1, span_days).annual


def calmar(rets: npt.ArrayLike, span_days: float) -> float | None:
    rate = _cagr(rets, span_days)
    mdd = max_drawdown(rets)
    return rate / -mdd if rate is not None and mdd < 0 else None


@dataclass(frozen=True, slots=True)
class Underwater:
    max_drawdown: float
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovery: pd.Timestamp | None
    recovery_days: int | None
    longest_days: int
    current_days: int


def underwater(wealth: pd.Series) -> Underwater:
    level = wealth.to_numpy(dtype="float64")
    dates = pd.DatetimeIndex(wealth.index)
    dd = level / np.maximum.accumulate(level) - 1
    under = dd < -TOLERANCE
    steps = np.arange(len(level))
    last_peak = np.maximum.accumulate(np.where(under, 0, steps))

    def at(i: int) -> pd.Timestamp:
        return cast("pd.Timestamp", dates[i])

    def days(end: int, start: int) -> int:
        return (at(end) - at(start)).days

    trough = int(np.argmin(dd))
    peak = int(last_peak[trough])
    back = steps[(steps > trough) & ~under]
    recovered = int(back[0]) if len(back) else None
    if not under[trough]:
        recovered = trough
    ends = steps[1:][under[:-1] & ~under[1:]]
    spans = [days(int(e), int(last_peak[e - 1])) for e in ends]
    current = days(len(level) - 1, int(last_peak[-1])) if under[-1] else 0
    return Underwater(
        max_drawdown=float(min(dd.min(), 0.0)),
        peak=at(peak),
        trough=at(trough),
        recovery=None if recovered is None else at(recovered),
        recovery_days=None if recovered is None else days(recovered, trough),
        longest_days=max([*spans, current]),
        current_days=current,
    )


def effective_annual(rates: pd.Series) -> pd.Series:
    gross = 1 + rates / PERCENT * BILL_DAYS / YEAR_DAYS
    return (gross ** (YEAR_DAYS / BILL_DAYS) - 1) * PERCENT


def daily_rf(
    rates: pd.Series,
    index: pd.DatetimeIndex,
    periods: float = TRADING_DAYS,
) -> pd.Series:
    known = effective_annual(rates.dropna().sort_index())
    carried = known.reindex(known.index.union(index)).ffill().reindex(index)
    return ((1 + carried / PERCENT) ** (1 / periods) - 1).astype("float64")


@dataclass(frozen=True, slots=True)
class Annualised:
    cumulative: float
    annual: float | None
    annualised: bool


def annualise(cumulative: float, span_days: float) -> Annualised:
    if span_days < YEAR_DAYS or cumulative <= -1:
        return Annualised(cumulative, None, annualised=False)
    annual = (1 + cumulative) ** (YEAR_DAYS / span_days) - 1
    return Annualised(cumulative, annual, annualised=True)


def _log_growth(days: npt.ArrayLike, cash: npt.ArrayLike) -> tuple[float, float] | None:
    t = _floats(days)
    c = _floats(cash)
    span = float(t.max() - t.min()) if len(t) else 0.0
    if span <= 0 or not ((c > 0).any() and (c < 0).any()):
        return None
    u = (t - t.min()) / span

    def npv(h: npt.ArrayLike) -> Floats:
        return np.exp(-np.multiply.outer(_floats(h), u)) @ c

    grid = np.linspace(-XIRR_REACH, XIRR_REACH, XIRR_GRID)
    sign = np.sign(npv(grid))
    zeros = np.flatnonzero(sign == 0)
    flips = np.flatnonzero(sign[:-1] * sign[1:] < 0)
    if len(zeros) + len(flips) != 1:
        return None
    if len(zeros):
        return float(grid[zeros[0]]), span
    low, high = float(grid[flips[0]]), float(grid[flips[0] + 1])
    at_low = float(npv(low))
    for _ in range(XIRR_STEPS):
        mid = (low + high) / 2
        if at_low * float(npv(mid)) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2, span


def mwr(days: npt.ArrayLike, cash: npt.ArrayLike) -> Annualised | None:
    found = _log_growth(days, cash)
    if found is None:
        return None
    h, span = found
    return annualise(math.expm1(h), span)


def _active(rets: npt.ArrayLike, bench: npt.ArrayLike) -> Floats | None:
    active = _floats(rets) - _floats(bench)
    return active if len(active) >= MIN_OBS else None


def tracking_error(
    rets: npt.ArrayLike, bench: npt.ArrayLike, periods: float = TRADING_DAYS
) -> float | None:
    active = _active(rets, bench)
    if active is None:
        return None
    return float(active.std(ddof=1)) * math.sqrt(periods)


def information_ratio(
    rets: npt.ArrayLike, bench: npt.ArrayLike, periods: float = TRADING_DAYS
) -> float | None:
    active = _active(rets, bench)
    te = tracking_error(rets, bench, periods)
    if active is None or te is None or te <= 0:
        return None
    return float(active.mean()) * periods / te


def active_share(w_p: pd.Series, w_b: pd.Series) -> float:
    names = w_p.index.union(w_b.index)
    gap = w_p.reindex(names).fillna(0.0) - w_b.reindex(names).fillna(0.0)
    return float(gap.abs().sum()) / 2


@dataclass(frozen=True, slots=True)
class Gap:
    value: float
    annualised: bool


def gap(first: Annualised, second: Annualised) -> Gap:
    if first.annual is not None and second.annual is not None:
        return Gap(first.annual - second.annual, annualised=True)
    return Gap(first.cumulative - second.cumulative, annualised=False)


def counterfactual(
    prices: npt.ArrayLike, shares: npt.ArrayLike, flows: npt.ArrayLike
) -> Floats:
    basket = _floats(prices) @ _floats(shares)
    units = 1 + np.cumsum(_floats(flows) / basket)
    return units * basket


def trading_effect(
    days: npt.ArrayLike, real: npt.ArrayLike, still: npt.ArrayLike
) -> Gap | None:
    first = mwr(days, real)
    second = mwr(days, still)
    return None if first is None or second is None else gap(first, second)


def project_simplex(v: npt.ArrayLike) -> Floats:
    x = _floats(v)
    u = np.sort(x)[::-1]
    css = np.cumsum(u)
    j = np.arange(1, len(x) + 1)
    rho = int(np.nonzero(u - (css - 1) / j > 0)[0][-1])
    theta = (css[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0.0)


def _style(gram: Floats, cross: Floats, start: Floats) -> Floats:
    top = float(np.linalg.eigvalsh(gram)[-1])
    if top <= 0:
        return start
    step = 1 / (2 * top)
    w = start
    y = start
    t = 1.0
    for _ in range(STYLE_STEPS):
        nxt = project_simplex(y - step * 2 * (gram @ y - cross))
        if float(np.abs(nxt - w).max()) < STYLE_TOLERANCE:
            return nxt
        if float((y - nxt) @ (nxt - w)) > 0:
            t = 1.0
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = nxt + (t - 1) / t_next * (nxt - w)
        w, t = nxt, t_next
    return w


def _moments(y: Floats, x: Floats) -> tuple[Floats, Floats]:
    yc = y - y.mean()
    xc = x - x.mean(axis=0)
    return xc.T @ xc / len(y), xc.T @ yc / len(y)


@dataclass(frozen=True, slots=True)
class StyleFit:
    weights: Floats
    r2: float | None


def style_fit(y: npt.ArrayLike, x: npt.ArrayLike) -> StyleFit:
    target = _floats(y)
    proxies = _floats(x)
    k = proxies.shape[1]
    w = _style(*_moments(target, proxies), np.full(k, 1 / k))
    total = float(target.var())
    left = float((target - proxies @ w).var())
    return StyleFit(w, 1 - left / total if total > 0 else None)


def selection_returns(y: npt.ArrayLike, x: npt.ArrayLike, window: int) -> Floats:
    target = _floats(y)
    proxies = _floats(x)
    k = proxies.shape[1]
    w = np.full(k, 1 / k)
    out = np.empty(max(len(target) - window, 0))
    for i, t in enumerate(range(window, len(target))):
        w = _style(*_moments(target[t - window : t], proxies[t - window : t]), w)
        out[i] = target[t] - float(proxies[t] @ w)
    return out


@dataclass(frozen=True, slots=True)
class Reach:
    end: float
    touched: float
    swing: float


@dataclass(frozen=True, slots=True)
class PathRisk:
    falls: dict[float, Reach]
    rises: dict[float, Reach]


def _reach(
    end: Floats, touched: Floats, swing: Floats, levels: tuple[float, ...]
) -> dict[float, Reach]:
    return {
        level: Reach(
            float((end >= level).mean()),
            float((touched >= level).mean()),
            float((swing >= level).mean()),
        )
        for level in levels
    }


def path_risk(
    paths: npt.ArrayLike, levels: tuple[float, ...] = PATH_LEVELS
) -> PathRisk:
    growth = _floats(paths)
    end = growth[:, -1]
    ratio = np.maximum.accumulate(growth, axis=1)
    _ = np.maximum(ratio, 1.0, out=ratio)
    _ = np.divide(growth, ratio, out=ratio)
    from_high = ratio.min(axis=1)
    _ = np.minimum.accumulate(growth, axis=1, out=ratio)
    _ = np.minimum(ratio, 1.0, out=ratio)
    _ = np.divide(growth, ratio, out=ratio, where=ratio > 0)
    from_low = ratio.max(axis=1)
    falls = _reach(1 - end, 1 - growth.min(axis=1), 1 - from_high, levels)
    rises = _reach(end - 1, growth.max(axis=1) - 1, from_low - 1, levels)
    return PathRisk(falls, rises)


def shrink(estimates: npt.ArrayLike, se: npt.ArrayLike) -> Floats:
    a = _floats(estimates)
    s2 = _floats(se) ** 2
    mean = float(a.mean())
    tau2 = max(0.0, float(a.var(ddof=1)) - float(s2.mean())) if len(a) > 1 else 0.0
    weight = np.divide(tau2, tau2 + s2, out=np.ones_like(s2), where=tau2 + s2 > 0)
    return mean + weight * (a - mean)


def top_weights(values: pd.Series, n: int, *, equal: bool = False) -> pd.Series:
    top = cast("pd.Series", values.groupby(level=0).sum()).nlargest(n)
    if equal:
        return pd.Series(1 / len(top), index=top.index, dtype="float64")
    return top / float(top.sum())


def drift(w: pd.Series, growth: pd.Series) -> pd.Series:
    grown = w * growth.reindex(w.index)
    return grown / float(grown.sum())
