from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from finagent_terminal.analytics import (
    MissingCloseColumnError,
    MonteCarloConfig,
    TooFewReturnsError,
    compute_indicators,
    compute_risk_metrics,
    monte_carlo_simulation,
)


def fixture_frame(values: list[float | None]) -> pd.DataFrame:
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


def test_monte_carlo_success_is_deterministic_with_fixed_seed() -> None:
    # Given
    frame = fixture_frame([100.0, 102.0, 99.0, 105.0, 108.0, 106.0])

    # When
    result = monte_carlo_simulation(
        frame,
        MonteCarloConfig(days=5, paths=20, seed=7, bins=4),
    )

    # Then
    assert result.percentile_10 == pytest.approx(98.1492)
    assert result.percentile_50 == pytest.approx(104.3951)
    assert result.percentile_90 == pytest.approx(116.8044)
    assert result.histogram_counts == (3, 8, 6, 3)
    assert len(result.histogram_edges) == 5
    assert result.summary_line == (
        "20 paths over 5 days; median final value 104.40, "
        "10th-90th range 98.15-116.80."
    )


def test_monte_carlo_missing_close_raises_typed_error() -> None:
    # Given
    frame = pd.DataFrame(
        {"Open": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    # When / Then
    with pytest.raises(MissingCloseColumnError, match="missing close column"):
        _ = monte_carlo_simulation(frame, MonteCarloConfig(days=5, paths=20, seed=7))


def test_monte_carlo_too_few_returns_raises_typed_error() -> None:
    # Given
    frame = fixture_frame([100.0, None, 100.0])

    # When / Then
    with pytest.raises(TooFewReturnsError, match="need at least 2 returns"):
        _ = monte_carlo_simulation(frame, MonteCarloConfig(days=5, paths=20, seed=7))


def test_risk_formula_conventions_when_benchmark_returns_exist() -> None:
    # Given
    prices = (100.0, 105.0, 102.0, 108.0, 107.0)
    benchmark_prices = (100.0, 102.0, 101.0, 103.0, 104.0)
    frame = fixture_frame(list(prices))
    benchmark = fixture_frame(list(benchmark_prices))
    expected = {
        "volatility": 0.6859962532240664,
        "max_drawdown": -0.02857142857142858,
        "sharpe": 6.5197863670357075,
        "sortino": 20.631878724975117,
        "historical_var_95": 0.02567460317460319,
        "beta": 2.8050467684356732,
        "alpha": -2.5443545152954514,
    }

    # When
    result = compute_risk_metrics(frame, benchmark)

    # Then
    assert result.volatility == pytest.approx(expected["volatility"])
    assert result.max_drawdown == pytest.approx(expected["max_drawdown"])
    assert result.sharpe == pytest.approx(expected["sharpe"])
    assert result.sortino == pytest.approx(expected["sortino"])
    assert result.historical_var_95 == pytest.approx(expected["historical_var_95"])
    assert result.beta == pytest.approx(expected["beta"])
    assert result.alpha == pytest.approx(expected["alpha"])


def test_risk_metrics_success_when_normal_aapl_uses_spy_benchmark() -> None:
    # Given
    aapl = fixture_frame([500.0, 501.25, 498.0, 502.0, 501.25])
    spy = fixture_frame([100.0, 110.0, 111.0, 112.0, 113.0])

    # When
    result = compute_risk_metrics(aapl, spy)

    # Then
    assert result.volatility is not None
    assert result.max_drawdown == pytest.approx(-0.0064837905)
    assert result.sharpe is not None
    assert result.sortino is not None
    assert result.historical_var_95 is not None
    assert result.historical_var_95 > 0
    assert result.beta is not None
    assert result.alpha is not None


def test_risk_missing_benchmark_leaves_beta_alpha_unavailable() -> None:
    # Given / When
    result = compute_risk_metrics(fixture_frame([100.0, 101.0, 99.0, 102.0]))

    # Then
    assert result.volatility is not None
    assert result.beta is None
    assert result.alpha is None


def test_risk_missing_close_raises_typed_error() -> None:
    # Given
    frame = pd.DataFrame(
        {"Open": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    # When / Then
    with pytest.raises(MissingCloseColumnError, match="missing close column"):
        _ = compute_risk_metrics(frame)


def test_risk_empty_returns_are_unavailable() -> None:
    # Given / When
    result = compute_risk_metrics(fixture_frame([100.0]))

    # Then
    assert result.volatility is None
    assert result.max_drawdown == 0.0
    assert result.sharpe is None
    assert result.sortino is None
    assert result.historical_var_95 is None
    assert result.beta is None
    assert result.alpha is None


def test_risk_zero_close_malformed_input_returns_unavailable() -> None:
    # Given / When
    result = compute_risk_metrics(fixture_frame([0.0, 0.0, 1.0]))

    # Then
    assert result.volatility is None
    assert result.max_drawdown is None
    assert result.sharpe is None
    assert result.sortino is None
    assert result.historical_var_95 is None
    assert result.beta is None
    assert result.alpha is None


def test_risk_zero_variance_cases_are_unavailable() -> None:
    # Given / When
    result = compute_risk_metrics(
        fixture_frame([100.0, 100.0, 100.0, 100.0]),
        fixture_frame([200.0, 200.0, 200.0, 200.0]),
    )

    # Then
    assert result.volatility == 0.0
    assert result.max_drawdown == 0.0
    assert result.sharpe is None
    assert result.sortino is None
    assert result.historical_var_95 == 0.0
    assert result.beta is None
    assert result.alpha is None
