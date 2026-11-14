# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

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


def test_growth_paths_equal_a_per_asset_replay_with_daily_rebalancing() -> None:
    rets = normal(60, COV) + 0.002
    picks = quant.stationary_indices(60, 50, 30, 5.0)
    growth = quant.growth_paths(rets, W, picks)

    demeaned = rets @ W - (rets @ W).mean()
    by_hand = np.cumprod(1 + demeaned[picks], axis=1)
    assert growth.shape == (50, 31)
    assert np.allclose(growth[:, 0], 1.0)
    assert np.allclose(growth[:, 1:], by_hand)
    kept = quant.growth_paths(rets, W, picks, demean=False)
    assert np.allclose(kept[:, 1:], np.cumprod(1 + (rets @ W)[picks], axis=1))


def lw_by_hand(
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    t, n = x.shape
    y = x - x.mean(axis=0)
    s = np.array([[(y[:, i] * y[:, j]).sum() / t for j in range(n)] for i in range(n)])
    sd = np.sqrt(np.diag(s))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    r_bar = sum(s[i, j] / (sd[i] * sd[j]) for i, j in pairs) / (n * (n - 1))
    f = np.array(
        [
            [s[i, i] if i == j else r_bar * sd[i] * sd[j] for j in range(n)]
            for i in range(n)
        ]
    )

    def pi(i: int, j: int) -> float:
        return float(((y[:, i] * y[:, j] - s[i, j]) ** 2).mean())

    def theta(i: int, j: int) -> float:
        return float(((y[:, i] ** 2 - s[i, i]) * (y[:, i] * y[:, j] - s[i, j])).mean())

    pi_hat = sum(pi(i, j) for i in range(n) for j in range(n))
    rho = sum(pi(i, i) for i in range(n)) + sum(
        r_bar / 2 * (sd[j] / sd[i] * theta(i, j) + sd[i] / sd[j] * theta(j, i))
        for i, j in pairs
    )
    gamma = float(((f - s) ** 2).sum())
    return f, s, min(max((pi_hat - rho) / gamma / t, 0.0), 1.0)


def test_ledoit_wolf_matches_the_paper_s_appendix_term_by_term() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_t(4, size=(40, 4)) * np.array([0.01, 0.02, 0.005, 0.03])
    x[:, 1] += x[:, 0]

    f, s, delta = lw_by_hand(x)
    cov, weight = quant.ledoit_wolf(x)

    assert 0 < delta < 1
    assert weight == pytest.approx(delta, rel=1e-9)
    assert np.allclose(cov, delta * f + (1 - delta) * s, rtol=1e-9, atol=0)


def test_historical_var_and_es_of_a_known_sample() -> None:
    r = -np.arange(1, 481) / 1000

    result = quant.var_es(r, 0.975)

    assert result is not None
    assert result[0] == pytest.approx(0.469)
    assert result[1] == pytest.approx(0.4745)
