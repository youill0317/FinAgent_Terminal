# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import pytest

from finagent import perf

SORTINO_CASE = [0.01, -0.02, 0.03, -0.01]


def test_mwr_is_cumulative_under_a_year_and_annual_from_one() -> None:
    short = perf.mwr([0.0, 182.0], [-100, 105])
    long = perf.mwr([0.0, 730.0], [-100, 121])

    assert short is not None
    assert short.cumulative == pytest.approx(0.05)
    assert short.annual is None
    assert not short.annualised
    assert long is not None
    assert long.cumulative == pytest.approx(0.21)
    assert long.annual == pytest.approx(0.1)
    assert long.annualised


def test_ratios_known_answers() -> None:
    assert perf.sortino(SORTINO_CASE) == pytest.approx(0.22361, abs=1e-5)
    assert perf.drawdown_series([-0.1, 0.05]) == pytest.approx([-0.1, -0.055])
    assert perf.calmar([0.1, -0.2, 0.15], 364) is None
    assert perf.calmar([0.1, -0.2, 0.15], 365) == pytest.approx(
        (1.1 * 0.8 * 1.15 - 1) / 0.2
    )
