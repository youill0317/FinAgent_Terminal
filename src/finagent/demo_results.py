"""Canned results the demo shows in place of the risk engines.

The demo checks the pages, not the numbers: nothing here simulates, fits or
loads a model. Each factory builds its result from closed forms of the
inputs that only label it (the holdings' value, the horizon).
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from finagent import generative, perf, portfolio, quant

DAILY_VOL: Final = 0.012  # the canned fan's spread per day
Z: Final = {5: -1.645, 25: -0.674, 50: 0.0, 75: 0.674, 95: 1.645}
# level: (end, touched, swing) at one month; longer horizons scale them.
FALLS: Final = {
    0.10: (0.08, 0.13, 0.18),
    0.20: (0.01, 0.02, 0.03),
    0.30: (0.0, 0.004, 0.006),
}
RISES: Final = {
    0.10: (0.10, 0.15, 0.26),
    0.20: (0.01, 0.02, 0.03),
    0.30: (0.0, 0.004, 0.006),
}


def simulation(value: float, days: int, drift: float = 0.0) -> portfolio.Simulation:
    """A fan widening with the square root of time, its end and path shares."""
    fan = [
        {
            "day": float(day),
            **{
                f"p{q}": value
                * (1 + drift) ** day
                * (1 + z * DAILY_VOL * math.sqrt(day))
                for q, z in Z.items()
            },
        }
        for day in range(days + 1)
    ]
    spread = DAILY_VOL * math.sqrt(days)
    centre = value * (1 + drift) ** days
    low, width = centre * (1 - 4 * spread), centre * 8 * spread / portfolio.BINS
    ends = [
        {
            "start": low + i * width,
            "end": low + (i + 1) * width,
            "count": round(
                portfolio.PATHS
                * width
                / (centre * spread * math.sqrt(2 * math.pi))
                * math.exp(
                    -(((low + (i + 0.5) * width) / centre - 1) ** 2) / (2 * spread**2)
                )
            ),
        }
        for i in range(portfolio.BINS)
    ]
    scale = min(math.sqrt(days / quant.MONTH_DAYS), 4.0)

    def reach(shares: tuple[float, float, float]) -> perf.Reach:
        end, touched, swing = (min(s * scale, 0.99) for s in shares)
        return perf.Reach(end, touched, swing)

    paths = perf.PathRisk(
        falls={level: reach(s) for level, s in FALLS.items()},
        rises={level: reach(s) for level, s in RISES.items()},
        recovery_median=days // 3,
        recovery_low=days // 10,
        recovery_high=None,
        never=0.4,
    )
    return portfolio.Simulation(paths, fan, ends)


def density() -> portfolio.Density:
    return portfolio.Density(quant.Berkowitz(2.1, 0.55, 0.05, 1.02, 0.03), 120)


def generator(
    method: generative.Method, tickers: tuple[str, ...]
) -> generative.Generator:
    """A generator to choose, without weights: the demo never draws from it."""
    meta = {"train_start": "2015-01-02", "train_end": "2026-09-30"}
    return generative.Generator(method, tickers, None, meta, {})


def volatility(assets: int, days: int) -> quant.Devolatilised:
    """A GJR-GARCH filter at a flat volatility; the demo never fits one."""
    flat = np.full((days, assets), DAILY_VOL)
    return quant.Devolatilised(
        z=np.zeros((days, assets)),
        sigma=flat,
        now=np.full(assets, DAILY_VOL),
        step=np.zeros((4, assets)),
        garch=np.ones(assets, dtype=np.bool_),
    )


VAR: Final = {0.95: 0.018, 0.975: 0.023, 0.99: 0.030}  # historical, a day's loss
METHODS: Final = {"역사적": 1.0, "FHS": 1.1, "EVT": 1.05}


def tails(assets: int, days: int) -> portfolio.Tails:
    """VaR and ES rows, green backtests and an even ES split by holding."""
    rows = tuple(
        portfolio.Tail(
            method,
            level,
            var * scale,
            var * scale * 1.3,
            (var * scale * 0.85, var * scale * 1.2),
            (var * scale * 1.1, var * scale * 1.6),
        )
        for method, scale in METHODS.items()
        for level, var in VAR.items()
    )
    test = quant.Backtest(250, 13, 12.5, 0.88, 0.6, "green", -0.1, 0.4, 0.7, 0.5)
    es = VAR[0.975] * 1.3
    return portfolio.Tails(
        rows,
        dict.fromkeys(METHODS, test),
        quant.BACKTEST_DAYS,
        np.full(assets, es / assets),
        days,
        days,
    )
