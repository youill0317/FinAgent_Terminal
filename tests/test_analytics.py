from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK

from finagent_terminal.analytics import compute_indicators


def fixture_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": values},
        index=pd.date_range("2024-01-01", periods=len(values), freq="D"),
    )


def test_returns_and_indicators_when_price_frame_is_valid() -> None:
    # Given
    frame = fixture_frame([100.0, 110.0, 105.0, 120.0])

    # When
    result = compute_indicators(frame, moving_average_windows=(2,))

    # Then
    assert result["return"].round(4).tolist() == [0.0, 0.1, -0.0455, 0.1429]
    assert result["cumulative_return"].round(2).tolist() == [0.0, 0.1, 0.05, 0.2]
    assert result["normalized_return"].round(2).tolist() == [100.0, 110.0, 105.0, 120.0]
    assert result["moving_average_2"].round(2).tolist() == [100.0, 105.0, 107.5, 112.5]
    assert result["drawdown"].round(4).tolist() == [0.0, 0.0, -0.0455, 0.0]
    assert {"volatility_20", "rsi_14", "macd", "macd_signal"} <= set(result.columns)


def test_empty_frame_returns_empty_indicators() -> None:
    # Given
    frame = pd.DataFrame()

    # When
    result = compute_indicators(frame)

    # Then
    assert result.empty
