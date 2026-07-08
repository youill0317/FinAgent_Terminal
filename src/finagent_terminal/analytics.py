from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Final, override

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from finagent_terminal.analytics_risk import (
    MissingCloseColumnError,
    RiskMetrics,
    close_prices,
    compute_risk_metrics,
)

__all__ = (
    "MissingCloseColumnError",
    "MonteCarloConfig",
    "MonteCarloResult",
    "RiskMetrics",
    "TooFewReturnsError",
    "close_prices",
    "compute_indicators",
    "compute_risk_metrics",
    "macd_lite",
    "monte_carlo_simulation",
    "normalized_returns",
    "rsi_lite",
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_CLOSE_COLUMN: Final = "Close"
DEFAULT_MOVING_AVERAGE_WINDOWS: Final = (20, 50)
MIN_MONTE_CARLO_RETURNS: Final = 2
TRADING_DAYS_PER_YEAR: Final = 252
RSI_WINDOW: Final = 14
MACD_FAST_SPAN: Final = 12
MACD_SLOW_SPAN: Final = 26
MACD_SIGNAL_SPAN: Final = 9


@dataclass(frozen=True, slots=True)
class TooFewReturnsError(Exception):
    actual: int
    minimum: int = 2

    @override
    def __str__(self) -> str:
        return f"need at least {self.minimum} returns; got {self.actual}"


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    days: int = 30
    paths: int = 1000
    seed: int | None = None
    bins: int = 10


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    days: int
    paths: int
    start_price: float
    percentile_10: float
    percentile_50: float
    percentile_90: float
    histogram_counts: tuple[int, ...]
    histogram_edges: tuple[float, ...]

    @property
    def summary_line(self) -> str:
        return (
            f"{self.paths} paths over {self.days} days; "
            f"median final value {self.percentile_50:.2f}, "
            f"10th-90th range {self.percentile_10:.2f}-{self.percentile_90:.2f}."
        )


def compute_indicators(
    frame: pd.DataFrame,
    close_column: str = DEFAULT_CLOSE_COLUMN,
    moving_average_windows: tuple[int, ...] = DEFAULT_MOVING_AVERAGE_WINDOWS,
) -> pd.DataFrame:
    prices = close_prices(frame, close_column)
    if prices.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=prices.index)
    result["close"] = prices
    result["return"] = prices.pct_change().fillna(0.0)
    result["cumulative_return"] = (1.0 + result["return"]).cumprod() - 1.0
    result["normalized_return"] = normalized_returns(prices)

    for window in moving_average_windows:
        result[f"moving_average_{window}"] = prices.rolling(
            window=window,
            min_periods=1,
        ).mean()

    result["volatility_20"] = (
        result["return"].rolling(window=20, min_periods=2).std().fillna(0.0)
        * TRADING_DAYS_PER_YEAR**0.5
    )
    result["drawdown"] = prices / prices.cummax() - 1.0
    result["rsi_14"] = rsi_lite(prices)
    macd, signal = macd_lite(prices)
    result["macd"] = macd
    result["macd_signal"] = signal
    return result


def monte_carlo_simulation(
    frame: pd.DataFrame,
    config: MonteCarloConfig | None = None,
    close_column: str = DEFAULT_CLOSE_COLUMN,
) -> MonteCarloResult:
    simulation_config = MonteCarloConfig() if config is None else config
    prices = close_prices(frame, close_column)
    returns = prices.pct_change().dropna()
    return_values = tuple(
        float(value)
        for value in returns.to_numpy(dtype=np.float64)
        if np.isfinite(float(value))
    )
    if len(return_values) < MIN_MONTE_CARLO_RETURNS:
        raise TooFewReturnsError(actual=len(return_values))

    rng = Random(simulation_config.seed)  # noqa: S311
    start_price = 100.0
    terminal_values = tuple(
        _simulate_terminal_price(
            start_price,
            return_values,
            simulation_config.days,
            rng,
        )
        for _ in range(simulation_config.paths)
    )
    sorted_values = tuple(sorted(terminal_values))
    histogram_counts, histogram_edges = _histogram(
        terminal_values,
        simulation_config.bins,
    )
    return MonteCarloResult(
        days=simulation_config.days,
        paths=simulation_config.paths,
        start_price=start_price,
        percentile_10=_percentile(sorted_values, 10.0),
        percentile_50=_percentile(sorted_values, 50.0),
        percentile_90=_percentile(sorted_values, 90.0),
        histogram_counts=histogram_counts,
        histogram_edges=histogram_edges,
    )


def normalized_returns(prices: pd.Series) -> pd.Series:
    clean_prices = pd.to_numeric(prices, errors="coerce").dropna()
    if clean_prices.empty:
        return pd.Series(dtype="float64")

    price_values: NDArray[np.float64] = clean_prices.to_numpy(dtype=np.float64)
    first_values = price_values[:1]
    if first_values.size == 0 or first_values.all() == 0:
        return pd.Series(dtype="float64")
    baseline_values = np.repeat(first_values, price_values.size)
    normalized_values = price_values / baseline_values * 100.0
    return pd.Series(normalized_values, index=clean_prices.index)


def _simulate_terminal_price(
    start_price: float,
    returns: tuple[float, ...],
    days: int,
    rng: Random,
) -> float:
    terminal_price = start_price
    for _ in range(days):
        terminal_price *= 1.0 + rng.choice(returns)
    return terminal_price


def _percentile(sorted_values: tuple[float, ...], percentile: float) -> float:
    raw_index = (len(sorted_values) - 1) * percentile / 100.0
    lower_index = int(raw_index)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = raw_index - lower_index
    return sorted_values[lower_index] + (
        sorted_values[upper_index] - sorted_values[lower_index]
    ) * fraction


def _histogram(
    values: tuple[float, ...],
    bins: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    low = min(values)
    high = max(values)
    if high == low:
        return (len(values), *([0] * (bins - 1))), tuple(low for _ in range(bins + 1))

    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        counts[min(int((value - low) / width), bins - 1)] += 1
    return tuple(counts), tuple(low + width * index for index in range(bins + 1))


def rsi_lite(prices: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = prices.diff().fillna(0.0)
    gain = delta.clip(lower=0.0).rolling(window=window, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.mask(loss == 0)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(100.0)


def macd_lite(prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    fast = prices.ewm(span=MACD_FAST_SPAN, adjust=False).mean()
    slow = prices.ewm(span=MACD_SLOW_SPAN, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=MACD_SIGNAL_SPAN, adjust=False).mean()
    return macd, signal
