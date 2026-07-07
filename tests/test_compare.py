from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from finagent_terminal.compare import CompareSymbolLimitError, compare_symbols


def fixture_frame(values: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": values},
        index=pd.date_range(start, periods=len(values), freq="D"),
    )


def test_normalized_returns_for_aapl_msft_qqq() -> None:
    # Given
    frames = {
        "AAPL": fixture_frame([100.0, 110.0, 121.0]),
        "MSFT": fixture_frame([200.0, 210.0, 220.0]),
        "QQQ": fixture_frame([50.0, 55.0, 60.0]),
    }

    # When
    result = compare_symbols(frames)

    # Then
    assert result.skipped_symbols == ()
    assert result.normalized_returns.round(2).to_dict("list") == {
        "AAPL": [100.0, 110.0, 121.0],
        "MSFT": [100.0, 105.0, 110.0],
        "QQQ": [100.0, 110.0, 120.0],
    }
    assert result.relative_performance.round(2).to_dict("records") == [
        {"symbol": "AAPL", "total_return": 0.21},
        {"symbol": "QQQ", "total_return": 0.2},
        {"symbol": "MSFT", "total_return": 0.1},
    ]


def test_empty_frame_is_skipped_without_crash() -> None:
    # Given
    frames = {
        "AAPL": fixture_frame([100.0, 105.0]),
        "EMPTY": pd.DataFrame(),
        "MSFT": fixture_frame([200.0, 210.0]),
    }

    # When
    result = compare_symbols(frames)

    # Then
    assert result.skipped_symbols == ("EMPTY",)
    assert result.normalized_returns.columns.tolist() == ["AAPL", "MSFT"]


def test_missing_close_column_is_skipped_without_crash() -> None:
    # Given
    frames = {
        "AAPL": fixture_frame([100.0, 105.0]),
        "BAD": pd.DataFrame({"Open": [1.0, 2.0]}),
        "MSFT": fixture_frame([200.0, 210.0]),
    }

    # When
    result = compare_symbols(frames)

    # Then
    assert result.skipped_symbols == ("BAD",)
    assert result.normalized_returns.columns.tolist() == ["AAPL", "MSFT"]


def test_compare_starts_at_shared_first_valid_date() -> None:
    # Given
    frames = {
        "AAPL": fixture_frame([90.0, 100.0, 110.0], start="2024-01-01"),
        "MSFT": fixture_frame([200.0, 220.0], start="2024-01-02"),
    }

    # When
    result = compare_symbols(frames)

    # Then
    assert result.normalized_returns.index.min() == pd.Timestamp("2024-01-02")
    assert result.normalized_returns.iloc[0].to_dict() == {"AAPL": 100.0, "MSFT": 100.0}


def test_rejects_more_than_six_symbols() -> None:
    # Given
    frames = {f"SYM{i}": fixture_frame([100.0, 101.0]) for i in range(7)}

    # When / Then
    with pytest.raises(CompareSymbolLimitError, match="2 to 6 symbols"):
        _ = compare_symbols(frames)
