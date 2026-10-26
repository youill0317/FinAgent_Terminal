"""Performance math: known answers from the literature and properties."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finagent import perf

SORTINO_CASE = [0.01, -0.02, 0.03, -0.01]


def day_offsets(*days: str) -> list[float]:
    stamps = pd.DatetimeIndex(list(days))
    return [float((d - stamps[0]).days) for d in stamps]


# ---------------------------------------------------------------- known answers


def test_xirr_known_answer() -> None:
    days = day_offsets("2020-01-01", "2021-01-01", "2022-01-01")

    assert perf.xirr(days, [-100, -50, 300]) == pytest.approx(0.4993, abs=5e-5)


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


# ---------------------------------------------------------------- drawdowns


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
