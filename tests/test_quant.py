"""Risk math with known answers and properties on synthetic returns."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
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


def test_growth_paths_equal_a_per_asset_replay_with_daily_rebalancing() -> None:
    rets = normal(60, COV) + 0.002
    picks = quant.stationary_indices(60, 50, 30, 5.0)
    growth = quant.growth_paths(rets, W, picks)

    demeaned = rets @ W - (rets @ W).mean()
    by_hand = np.cumprod(1 + demeaned[picks], axis=1)
    assert growth.shape == (50, 31)
    assert np.allclose(growth[:, 0], 1.0)
    assert np.allclose(growth[:, 1:], by_hand)


# ---------------------------------------------------------------- beta


def test_dimson_recovers_a_planted_lagged_beta() -> None:
    rng = np.random.default_rng(6)
    m = rng.standard_normal(2000) * 0.01
    r = 0.6 * m + 0.4 * np.concatenate([[0.0], m[:-1]])
    r += rng.standard_normal(2000) * 0.005

    assert quant.dimson_beta(r, m) == pytest.approx(1.0, abs=0.05)
    assert float(np.polyfit(m, r, 1)[0]) == pytest.approx(0.6, abs=0.05)


# ---------------------------------------------------------------- VaR and ES


def test_historical_var_and_es_of_a_known_sample() -> None:
    r = -np.arange(1, 481) / 1000

    result = quant.var_es(r, 0.975)

    assert result is not None
    assert result[0] == pytest.approx(0.469)  # 12th worst of 480
    assert result[1] == pytest.approx(0.4745)  # mean of the worst 12


# ---------------------------------------------------------------- backtest


def test_kupiec_with_no_exceptions_in_250_days() -> None:
    stat, p = quant.kupiec(0, 250, 0.05)

    assert stat == pytest.approx(-2 * 250 * math.log(0.95))
    assert stat == pytest.approx(25.65, abs=0.01)
    assert p == pytest.approx(math.erfc(math.sqrt(stat / 2)))
    assert p < 1e-5
    assert quant.kupiec(12, 240, 0.05) == pytest.approx((0.0, 1.0))


# ---------------------------------------------------------------- density forecasts


def test_blocks_are_non_overlapping_and_end_on_the_last_day() -> None:
    r = np.random.default_rng(8).normal(0.0, 0.01, 100)

    outcomes, starts = quant.blocks(r, 21)

    assert starts.tolist() == [16, 37, 58, 79]
    assert outcomes[-1] == pytest.approx(np.prod(1 + r[79:]) - 1)


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
