from __future__ import annotations

import math

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import pytest

from finagent.analysis import technical


def bars(count: int = 250) -> pd.DataFrame:
    close = [100.0 + index for index in range(count)]
    return pd.DataFrame(
        {
            "date": [f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(count)],
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": [1000.0] * count,
        }
    )


def test_trend_values_match_hand_calculation() -> None:
    found = technical.indicators(bars())
    last = 349.0

    assert found["sma_5"] == pytest.approx(last - 2)
    assert found["sma_20"] == pytest.approx(last - 9.5)
    assert found["sma_200"] == pytest.approx(last - 99.5)
    assert found["rsi_14"] == pytest.approx(100.0)
    assert found["momentum_10"] == pytest.approx(10.0)
    assert found["volume_avg_20"] == pytest.approx(1000.0)
    assert found["obv"] == pytest.approx(249 * 1000.0)


def test_range_values_match_hand_calculation() -> None:
    found = technical.indicators(bars())

    assert found["stoch_k"] == pytest.approx(14 / 15 * 100)
    assert found["williams_r_14"] == pytest.approx(-1 / 15 * 100)
    assert found["atr_14"] == pytest.approx(2.0)
    deviation = 5.0
    assert found["cci_20"] == pytest.approx(9.5 / (0.015 * deviation))
    sd = math.sqrt(sum((i - 9.5) ** 2 for i in range(20)) / 20)
    assert found["bollinger_pct_b"] == pytest.approx((9.5 + 2 * sd) / (4 * sd))


def test_macd_on_a_steady_ramp_is_the_lag_gap_and_has_no_histogram() -> None:
    found = technical.indicators(bars())

    assert found["macd"] == pytest.approx((26 - 1) / 2 - (12 - 1) / 2)
    assert found["macd_hist"] == pytest.approx(0.0, abs=1e-6)
