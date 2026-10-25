"""Performance math: numpy and pandas in, numbers out (no files, no network).

Descriptive only: every function measures the past or a simulated spread of
outcomes; nothing ranks, picks or suggests weights. "Not computable" (too
few observations, zero spread, no sign change) is ``None``.

Conventions: Sharpe is per period, mean excess ÷ std (ddof=1), annualised by
√(periods a year) only for display, 252 unless the caller counts its own.
Kurtosis is raw (normal = 3). Drawdowns compound from a starting capital of 1
and are negative (-0.25 is a 25% fall). Nothing is annualised over a span
shorter than 365 calendar days.
"""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

type Floats = npt.NDArray[np.float64]
type Quote = Literal["discount", "simple"]  # how a 91-day rate is stated

TRADING_DAYS: Final = 252
YEAR_DAYS: Final = 365
MIN_OBS: Final = 252  # fewer returns: Sharpe CI, PSR, DSR, TE and IR are None
CONFIDENCE: Final = 0.95
NORMAL_KURT: Final = 3.0
CDAR_LEVEL: Final = 0.95
PERCENT: Final = 100.0
PATH_LEVELS: Final = (0.10, 0.20, 0.30)  # simulated falls and rises
RECOVERY_RANGE: Final = (0.10, 0.90)
EULER_GAMMA: Final = 0.5772156649015329
TOLERANCE: Final = 1e-12  # a value this close below its peak has recovered
XIRR_REACH: Final = 100.0  # most |log growth| over the whole span tried
XIRR_GRID: Final = 4001  # points scanned for sign changes of the NPV
XIRR_STEPS: Final = 200
MAX_LOG: Final = 700.0  # exp() of more overflows: the annual rate is None
BILL_DAYS: Final = 91  # CD 91-day and the 13-week T-bill
DISCOUNT_YEAR: Final = 360  # days in a bank-discount year
STYLE_STEPS: Final = 5000
STYLE_TOLERANCE: Final = 1e-10

_NORMAL: Final = NormalDist()


def _floats(values: npt.ArrayLike) -> Floats:
    return np.asarray(values, dtype=np.float64)


def _cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------- Sharpe


@dataclass(frozen=True, slots=True)
class SharpeCI:
    sharpe: float  # annualised
    low: float
    high: float
    se: float  # annualised Mertens standard error
    n: int


def sharpe(rets: npt.ArrayLike, rf: npt.ArrayLike = 0.0) -> float | None:
    """Per-period Sharpe ratio: mean excess return ÷ its std (ddof=1)."""
    x = _floats(rets) - _floats(rf)
    if len(x) < 2:  # noqa: PLR2004 - a std needs two points
        return None
    std = float(x.std(ddof=1))
    return float(x.mean()) / std if std > 0 else None


def skew_kurt(rets: npt.ArrayLike) -> tuple[float, float] | None:
    """Population skewness and raw kurtosis (normal = 3)."""
    x = _floats(rets)
    d = x - x.mean()
    var = float((d**2).mean())
    if var <= 0:
        return None
    return float((d**3).mean()) / var**1.5, float((d**4).mean()) / var**2


def _spread(sr: float, skew: float, kurt: float) -> float:
    return 1 - skew * sr + (kurt - 1) / 4 * sr**2


def sharpe_se(sr: float, n: int, skew: float = 0.0, kurt: float = NORMAL_KURT) -> float:
    """Mertens standard error of a per-period Sharpe; Lo's at skew 0, kurt 3."""
    return math.sqrt(_spread(sr, skew, kurt) / n)


def sharpe_ci(
    rets: npt.ArrayLike,
    rf: npt.ArrayLike = 0.0,
    conf: float = CONFIDENCE,
    periods: float = TRADING_DAYS,
) -> SharpeCI | None:
    """Annualised Sharpe with a Mertens (non-normal, IID) confidence interval.

    ``periods`` returns make a year.
    """
    x = _floats(rets) - _floats(rf)
    sr = sharpe(x)
    moments = skew_kurt(x)
    if len(x) < MIN_OBS or sr is None or moments is None:
        return None
    se = sharpe_se(sr, len(x), *moments)
    z = _NORMAL.inv_cdf(0.5 + conf / 2)
    k = math.sqrt(periods)
    return SharpeCI(sr * k, (sr - z * se) * k, (sr + z * se) * k, se * k, len(x))


def psr_from(sr: float, n: int, skew: float, kurt: float, sr_star: float) -> float:
    """Probabilistic Sharpe ratio P(true SR > ``sr_star``), per-period, T-1."""
    return _cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(_spread(sr, skew, kurt)))


def psr(
    rets: npt.ArrayLike, sr_star: float = 0.0, rf: npt.ArrayLike = 0.0
) -> float | None:
    """PSR of ``rets`` against a per-period benchmark Sharpe ``sr_star``."""
    x = _floats(rets) - _floats(rf)
    sr = sharpe(x)
    moments = skew_kurt(x)
    if len(x) < MIN_OBS or sr is None or moments is None:
        return None
    return psr_from(sr, len(x), *moments, sr_star)


def min_trl(
    sr: float,
    skew: float = 0.0,
    kurt: float = NORMAL_KURT,
    sr_star: float = 0.0,
    conf: float = CONFIDENCE,
) -> float | None:
    """Minimum track record length, in observations, for PSR to reach ``conf``."""
    if sr <= sr_star:
        return None
    z = _NORMAL.inv_cdf(conf)
    return 1 + _spread(sr, skew, kurt) * (z / (sr - sr_star)) ** 2


def expected_max_sr(n_trials: int, var_trials: float) -> float | None:
    """Expected largest of ``n_trials`` Sharpes with zero true skill."""
    if n_trials < 2:  # noqa: PLR2004 - a maximum needs two trials
        return None
    return math.sqrt(var_trials) * (
        (1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(rets: npt.ArrayLike, trial_sharpes: npt.ArrayLike) -> float | None:
    """Deflated Sharpe: PSR against the best Sharpe expected by luck.

    ``trial_sharpes`` are the per-period Sharpes of every series looked at
    (e.g. all N 13F clones), so the bar rises with how many were compared.
    """
    trials = _floats(trial_sharpes)
    if len(trials) < 2:  # noqa: PLR2004 - a variance needs two trials
        return None
    bar = expected_max_sr(len(trials), float(trials.var(ddof=1)))
    return None if bar is None else psr(rets, bar)


# ---------------------------------------------------------------- ratios


def drawdown_series(rets: npt.ArrayLike) -> Floats:
    """Fall from the running peak after each return, from a capital of 1."""
    growth = np.cumprod(1 + _floats(rets))
    peak = np.maximum.accumulate(np.concatenate([[1.0], growth]))[1:]
    return growth / peak - 1


def max_drawdown(rets: npt.ArrayLike) -> float:
    return float(drawdown_series(rets).min(initial=0.0))


def average_drawdown(rets: npt.ArrayLike) -> float:
    """Mean drawdown over every observation (0 on days at a peak)."""
    dd = drawdown_series(rets)
    return float(dd.mean()) if len(dd) else 0.0


def cdar(rets: npt.ArrayLike, alpha: float = CDAR_LEVEL) -> float:
    """Conditional drawdown at risk: mean of the worst (1 - alpha) share of drawdowns.

    With m = N (1 - alpha), the deepest ceil(m) count, the shallowest of them
    by its fraction (Rockafellar-Uryasev, as ``quant._tail``). alpha -> 1
    gives the maximum drawdown, alpha = 0 the average drawdown.
    """
    dd = np.sort(drawdown_series(rets))
    if not len(dd):
        return 0.0
    m = round((1 - alpha) * len(dd), 9)
    if m <= 0:
        return float(dd[0])
    top = math.ceil(m)
    return float((dd[: top - 1].sum() + (m - top + 1) * dd[top - 1]) / m)


def ulcer_index(rets: npt.ArrayLike) -> float:
    """Root mean square drawdown in percent (Martin; divides by N)."""
    dd = drawdown_series(rets) * PERCENT
    return math.sqrt(float((dd**2).mean())) if len(dd) else 0.0


def sortino(rets: npt.ArrayLike, mar: npt.ArrayLike = 0.0) -> float | None:
    """Per-period Sortino; downside deviation over all observations."""
    x = _floats(rets) - _floats(mar)
    if not len(x):
        return None
    down = math.sqrt(float((np.minimum(x, 0.0) ** 2).mean()))
    return float(x.mean()) / down if down > 0 else None


def omega(rets: npt.ArrayLike, threshold: float = 0.0) -> float | None:
    """Gains above ``threshold`` ÷ losses below it."""
    x = _floats(rets) - threshold
    losses = float(-x[x < 0].sum())
    return float(x[x > 0].sum()) / losses if losses > 0 else None


def _cagr(rets: npt.ArrayLike, span_days: float) -> float | None:
    return annualise(float(np.prod(1 + _floats(rets))) - 1, span_days).annual


def calmar(rets: npt.ArrayLike, span_days: float) -> float | None:
    """Annual compound return ÷ |max drawdown|; None under a year."""
    rate = _cagr(rets, span_days)
    mdd = max_drawdown(rets)
    return rate / -mdd if rate is not None and mdd < 0 else None


def ulcer_performance(
    rets: npt.ArrayLike, span_days: float, rf: float = 0.0
) -> float | None:
    """Martin ratio: annual return minus annual ``rf``, in percent, ÷ Ulcer index.

    None under a year (the numerator is annualised) or with no drawdown.
    """
    rate = _cagr(rets, span_days)
    ulcer = ulcer_index(rets)
    return (rate - rf) * PERCENT / ulcer if rate is not None and ulcer > 0 else None


@dataclass(frozen=True, slots=True)
class Underwater:
    max_drawdown: float  # e.g. -0.25
    peak: pd.Timestamp  # last high before the deepest trough
    trough: pd.Timestamp
    recovery: pd.Timestamp | None  # first day back at the peak; None if not yet
    recovery_days: int | None  # calendar days from trough to recovery
    longest_days: int  # longest calendar span from a peak back to it (or to today)
    current_days: int  # calendar days since the last peak; 0 at a peak


def underwater(wealth: pd.Series) -> Underwater:
    """Drawdown dates and durations of a wealth series (e.g. ``1 + twr``).

    The first value is the starting capital. Durations are calendar days.
    """
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
    ends = steps[1:][under[:-1] & ~under[1:]]  # first day back at each peak
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


# ---------------------------------------------------------------- money and currencies


def effective_annual(rates: pd.Series, quote: Quote) -> pd.Series:
    """Effective annual percent from annual percent 91-day ``rates``.

    ``discount``: a bank-discount yield on a 360-day year (^IRX), bought at
    1 - d 91/360; ``simple``: a simple yield on a 365-day year (CD 91-day),
    growing by 1 + r 91/365. Either compounded over 365/91 periods a year.
    """
    d = rates / PERCENT
    if quote == "discount":
        gross = 1 / (1 - d * BILL_DAYS / DISCOUNT_YEAR)
    else:
        gross = 1 + d * BILL_DAYS / YEAR_DAYS
    return (gross ** (YEAR_DAYS / BILL_DAYS) - 1) * PERCENT


def daily_rf(
    rates: pd.Series,
    index: pd.DatetimeIndex,
    quote: Quote,
    periods: float = TRADING_DAYS,
) -> pd.Series:
    """Per-period risk-free returns on ``index`` from annual percent ``rates``.

    E.g. CD 91-day (``simple``) or ^IRX (``discount``) levels, made effective
    annual (``effective_annual``), then (1 + annual)^(1/periods) - 1, the
    last known rate carried forward. Days before the first rate stay NaN.
    """
    known = effective_annual(rates.dropna().sort_index(), quote)
    carried = known.reindex(known.index.union(index)).ffill().reindex(index)
    return ((1 + carried / PERCENT) ** (1 / periods) - 1).astype("float64")


@dataclass(frozen=True, slots=True)
class Annualised:
    cumulative: float  # over the whole span
    annual: float | None  # None under a year
    annualised: bool


def annualise(cumulative: float, span_days: float) -> Annualised:
    """A cumulative return and, for a span of a year or more, its annual rate."""
    if span_days < YEAR_DAYS or cumulative <= -1:
        return Annualised(cumulative, None, annualised=False)
    annual = (1 + cumulative) ** (YEAR_DAYS / span_days) - 1
    return Annualised(cumulative, annual, annualised=True)


def _log_growth(days: npt.ArrayLike, cash: npt.ArrayLike) -> tuple[float, float] | None:
    """The IRR as log growth h over the whole span, and the span in days.

    Solved in h, where NPV(h) = sum(c exp(-h t / span)), so a few-day span
    needs no huge annual rate. The NPV is scanned for sign changes on
    ``XIRR_GRID`` points of |h| <= ``XIRR_REACH``: exactly one is bisected;
    none or several (more than one rate fits) give None.
    """
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


def xirr(days: npt.ArrayLike, cash: npt.ArrayLike) -> float | None:
    """Annual internal rate of return (actual/365) of dated cash flows.

    ``days`` are calendar-day offsets; ``cash`` is from the investor's side
    (deposits negative, withdrawals and today's value positive). None with no
    sign change in ``cash``, all flows on one day, no rate within a growth of
    e^±100 over the span, several rates (the NPV changes sign more than
    once), or an annual rate too large for a float.
    """
    found = _log_growth(days, cash)
    if found is None:
        return None
    h, span = found
    annual = h * YEAR_DAYS / span
    return math.expm1(annual) if annual < MAX_LOG else None


def mwr(days: npt.ArrayLike, cash: npt.ArrayLike) -> Annualised | None:
    """Money-weighted return: XIRR, de-annualised over the span under a year."""
    found = _log_growth(days, cash)
    if found is None:
        return None
    h, span = found
    return annualise(math.expm1(h), span)


def to_base(local: npt.ArrayLike, fx: npt.ArrayLike) -> Floats:
    """Base-currency return from a local return and the currency's return."""
    return (1 + _floats(local)) * (1 + _floats(fx)) - 1


@dataclass(frozen=True, slots=True)
class FxSplit:
    local: float
    fx: float
    cross: float  # local x fx
    base: float  # local + fx + cross


def fx_split(w_prev: npt.ArrayLike, local: npt.ArrayLike, fx: npt.ArrayLike) -> FxSplit:
    """Split a chained base return into local, currency and cross parts.

    Each row is a day: base-currency weights at the prior close, each asset's
    local return and its currency's return against the base (0 for assets in
    the base currency). (1 + base) = (1 + local)(1 + fx).
    """
    w = _floats(w_prev)
    base = float(np.prod(1 + (w * to_base(local, fx)).sum(axis=-1))) - 1
    loc = float(np.prod(1 + (w * _floats(local)).sum(axis=-1))) - 1
    rate = (1 + base) / (1 + loc) - 1
    return FxSplit(loc, rate, loc * rate, base)


# ---------------------------------------------------------------- benchmark


def blend(w_kr: npt.ArrayLike, kr: npt.ArrayLike, us: npt.ArrayLike) -> Floats:
    """Daily returns of a two-index benchmark in base currency.

    ``kr`` and ``us`` are the two index legs already in base (see
    ``to_base``). ``w_kr`` per day is the portfolio's Korean share at the
    prior close, so the mix drifts with the portfolio; a single number is a
    fixed mix rebalanced daily.
    """
    w = _floats(w_kr)
    return w * _floats(kr) + (1 - w) * _floats(us)


def _active(rets: npt.ArrayLike, bench: npt.ArrayLike) -> Floats | None:
    active = _floats(rets) - _floats(bench)
    return active if len(active) >= MIN_OBS else None


def tracking_error(
    rets: npt.ArrayLike, bench: npt.ArrayLike, periods: float = TRADING_DAYS
) -> float | None:
    """Annualised std of daily active returns; None under 252 observations.

    ``periods`` returns make a year.
    """
    active = _active(rets, bench)
    if active is None:
        return None
    return float(active.std(ddof=1)) * math.sqrt(periods)


def information_ratio(
    rets: npt.ArrayLike, bench: npt.ArrayLike, periods: float = TRADING_DAYS
) -> float | None:
    """Annualised mean active return ÷ tracking error."""
    active = _active(rets, bench)
    te = tracking_error(rets, bench, periods)
    if active is None or te is None or te <= 0:
        return None
    return float(active.mean()) * periods / te


def active_share(w_p: pd.Series, w_b: pd.Series) -> float:
    """½Σ|w_p - w_b|: how far two weight sets differ (1 - overlap)."""
    names = w_p.index.union(w_b.index)
    gap = w_p.reindex(names).fillna(0.0) - w_b.reindex(names).fillna(0.0)
    return float(gap.abs().sum()) / 2


# ---------------------------------------------------------------- flows


@dataclass(frozen=True, slots=True)
class Gap:
    value: float  # annual rates' difference, or holding-period under a year
    annualised: bool


def _gap(first: Annualised, second: Annualised) -> Gap:
    if first.annual is not None and second.annual is not None:
        return Gap(first.annual - second.annual, annualised=True)
    return Gap(first.cumulative - second.cumulative, annualised=False)


def timing_gap(money: Annualised, time: Annualised) -> Gap:
    """Money-weighted minus time-weighted return over the same span.

    Annual rates from a year on; holding-period returns (flag off) before.
    """
    return _gap(money, time)


def counterfactual(
    prices: npt.ArrayLike, shares: npt.ArrayLike, flows: npt.ArrayLike
) -> Floats:
    """Daily value of holding the starting shares with no trades.

    ``prices`` is days x assets in base, ``shares`` the counts held at the
    first close, ``flows`` each day's external deposit (+) or withdrawal (-),
    which buys or sells the starting mix pro rata at that day's close.
    """
    basket = _floats(prices) @ _floats(shares)
    units = 1 + np.cumsum(_floats(flows) / basket)
    return units * basket


def trading_effect(
    days: npt.ArrayLike, cash: npt.ArrayLike, actual: float, held: float
) -> Gap | None:
    """XIRR of the real book minus XIRR of the do-nothing one.

    ``cash`` is the external flows on ``days`` from the investor's side,
    starting value included; ``actual`` and ``held`` are the two end values
    on the last day. Holding-period returns (flag off) under a year.
    """
    flows = _floats(cash)
    last = int(np.argmax(_floats(days)))
    real, still = flows.copy(), flows.copy()
    real[last] += actual
    still[last] += held
    first = mwr(days, real)
    second = mwr(days, still)
    return None if first is None or second is None else _gap(first, second)


# ---------------------------------------------------------------- style


def project_simplex(v: npt.ArrayLike) -> Floats:
    """Closest point with non-negative weights summing to 1 (Duchi et al.)."""
    x = _floats(v)
    u = np.sort(x)[::-1]
    css = np.cumsum(u)
    j = np.arange(1, len(x) + 1)
    rho = int(np.nonzero(u - (css - 1) / j > 0)[0][-1])
    theta = (css[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0.0)


def _style(gram: Floats, cross: Floats, start: Floats) -> Floats:
    """Minimise wᵀGw - 2bᵀw on the simplex by accelerated projected gradient.

    FISTA with a restart whenever the momentum points uphill.
    """
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
    weights: Floats  # ≥ 0, sum 1, one per proxy column
    r2: float | None  # share of the variance the mix explains


def style_fit(y: npt.ArrayLike, x: npt.ArrayLike) -> StyleFit:
    """Returns-based style analysis (Sharpe 1992) of ``y`` on proxy columns ``x``.

    The non-negative mix summing to 1 with the least tracking variance.
    """
    target = _floats(y)
    proxies = _floats(x)
    k = proxies.shape[1]
    w = _style(*_moments(target, proxies), np.full(k, 1 / k))
    total = float(target.var())
    left = float((target - proxies @ w).var())
    return StyleFit(w, 1 - left / total if total > 0 else None)


def selection_returns(y: npt.ArrayLike, x: npt.ArrayLike, window: int) -> Floats:
    """Out-of-sample selection returns: week t minus the mix fitted on t-W..t-1.

    One value per row from ``window`` on (length T - window).
    """
    target = _floats(y)
    proxies = _floats(x)
    k = proxies.shape[1]
    w = np.full(k, 1 / k)
    out = np.empty(max(len(target) - window, 0))
    for i, t in enumerate(range(window, len(target))):
        w = _style(*_moments(target[t - window : t], proxies[t - window : t]), w)
        out[i] = target[t] - float(proxies[t] @ w)
    return out


# ---------------------------------------------------------------- simulation


@dataclass(frozen=True, slots=True)
class Reach:
    """Shares of paths past a level, three ways: end <= touched <= swing."""

    end: float  # at the horizon, from today
    touched: float  # on any day, from today
    swing: float  # on any day, from the high (a fall) or low (a rise) so far


@dataclass(frozen=True, slots=True)
class PathRisk:
    falls: dict[float, Reach]  # level → paths down at least this far
    rises: dict[float, Reach]  # level → paths up at least this far
    recovery_median: int | None  # periods to regain the peak; None past horizon
    recovery_low: int | None  # 10th percentile
    recovery_high: int | None  # 90th percentile
    never: float  # share of paths still below the peak at the horizon


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
    paths: npt.ArrayLike,
    current: float,
    peak: float,
    levels: tuple[float, ...] = PATH_LEVELS,
) -> PathRisk:
    """Falls, rises and recovery odds from simulated growth paths.

    ``paths`` is paths x horizon cumulative growth from today (1.05 = +5%),
    column h being the growth after h + 1 periods: leave out a leading column
    of ones.
    A fall of X: 1 - V/ref >= X; a rise: V/ref - 1 >= X. ``Reach.end`` takes
    ref = today and the last period, ``touched`` today and any period,
    ``swing`` the high (a fall) or low (a rise) so far, today's included,
    so each holds on a path the one before it does.
    Recovery is the first period back at the historical ``peak``; its
    figures are None when already at the peak.
    """
    growth = _floats(paths)
    end = growth[:, -1]
    ratio = np.maximum.accumulate(growth, axis=1)  # one buffer, reused
    _ = np.maximum(ratio, 1.0, out=ratio)
    _ = np.divide(growth, ratio, out=ratio)
    from_high = ratio.min(axis=1)
    _ = np.minimum.accumulate(growth, axis=1, out=ratio)
    _ = np.minimum(ratio, 1.0, out=ratio)
    _ = np.divide(growth, ratio, out=ratio, where=ratio > 0)  # a wiped-out path: 0
    from_low = ratio.max(axis=1)
    falls = _reach(1 - end, 1 - growth.min(axis=1), 1 - from_high, levels)
    rises = _reach(end - 1, growth.max(axis=1) - 1, from_low - 1, levels)
    if current >= peak:
        return PathRisk(falls, rises, None, None, None, 0.0)
    hit = growth * current >= peak * (1 - TOLERANCE)
    reached = hit.any(axis=1)
    periods = np.where(reached, np.argmax(hit, axis=1) + 1, np.inf)

    def quantile(q: float) -> int | None:
        value = float(np.quantile(periods, q, method="inverted_cdf"))
        return int(value) if math.isfinite(value) else None

    low, high_q = RECOVERY_RANGE
    return PathRisk(
        falls,
        rises,
        quantile(0.5),
        quantile(low),
        quantile(high_q),
        float((~reached).mean()),
    )


# ---------------------------------------------------------------- clones


def shrink(estimates: npt.ArrayLike, se: npt.ArrayLike) -> Floats:
    """Empirical-Bayes shrinkage of N estimates toward their mean.

    tau² = max(0, var(est, ddof=1) - mean(se²)); each estimate moves to
    mean + tau²/(tau² + se²)·(est - mean).
    """
    a = _floats(estimates)
    s2 = _floats(se) ** 2
    mean = float(a.mean())
    tau2 = max(0.0, float(a.var(ddof=1)) - float(s2.mean())) if len(a) > 1 else 0.0
    weight = np.divide(tau2, tau2 + s2, out=np.ones_like(s2), where=tau2 + s2 > 0)
    return mean + weight * (a - mean)


def top_weights(values: pd.Series, n: int, *, equal: bool = False) -> pd.Series:
    """The ``n`` largest holdings by value, value- or equal-weighted.

    Rows sharing an index label (e.g. one CUSIP over several filing lines)
    are added up first.
    """
    top = cast("pd.Series", values.groupby(level=0).sum()).nlargest(n)
    if equal:
        return pd.Series(1 / len(top), index=top.index, dtype="float64")
    return top / float(top.sum())


def drift(w: pd.Series, growth: pd.Series) -> pd.Series:
    """Weights after each holding grew by ``growth`` (price relative, 1.1 = +10%)."""
    grown = w * growth.reindex(w.index)
    return grown / float(grown.sum())


def turnover(w_new: pd.Series, w_drifted: pd.Series) -> float:
    """One-way turnover of a rebalance: ½Σ|w_new - w_drifted|."""
    return active_share(w_new, w_drifted)
