from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import fmean, stdev
from typing import Final, override

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

DEFAULT_CLOSE_COLUMN: Final = "Close"
MIN_RISK_RETURNS: Final = 2
TRADING_DAYS_PER_YEAR: Final = 252
ANNUALIZATION_FACTOR: Final[float] = sqrt(float(TRADING_DAYS_PER_YEAR))


@dataclass(frozen=True, slots=True)
class MissingCloseColumnError(Exception):
    column: str

    @override
    def __str__(self) -> str:
        return f"missing close column: {self.column}"


@dataclass(frozen=True, slots=True)
class RiskMetrics:
    volatility: float | None
    max_drawdown: float | None
    sharpe: float | None
    sortino: float | None
    historical_var_95: float | None
    beta: float | None
    alpha: float | None


def compute_risk_metrics(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
) -> RiskMetrics:
    prices = close_prices(frame)
    if _has_invalid_close(prices):
        return _unavailable_metrics()

    max_drawdown = _max_drawdown(prices)
    returns = prices.pct_change().dropna()
    return_values = _finite_values(returns)
    if len(return_values) < MIN_RISK_RETURNS:
        return _unavailable_metrics(max_drawdown)

    daily_risk_free_rate = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_values = tuple(value - daily_risk_free_rate for value in return_values)
    daily_std = stdev(return_values)
    volatility = daily_std * ANNUALIZATION_FACTOR
    annualized_excess_return = fmean(excess_values) * TRADING_DAYS_PER_YEAR
    downside = tuple(value for value in return_values if value < 0.0)
    downside_std = stdev(downside) if len(downside) >= MIN_RISK_RETURNS else 0.0
    beta, alpha = _benchmark_risk(returns, benchmark_frame)
    return RiskMetrics(
        volatility=volatility,
        max_drawdown=max_drawdown,
        sharpe=_ratio_or_none(annualized_excess_return, volatility),
        sortino=_ratio_or_none(
            annualized_excess_return,
            downside_std * ANNUALIZATION_FACTOR,
        ),
        historical_var_95=max(0.0, -_percentile(tuple(sorted(return_values)), 5.0)),
        beta=beta,
        alpha=alpha,
    )


def close_prices(
    frame: pd.DataFrame,
    close_column: str = DEFAULT_CLOSE_COLUMN,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    if close_column not in frame.columns:
        raise MissingCloseColumnError(column=close_column)

    prices = pd.to_numeric(frame[close_column], errors="coerce").dropna()
    if prices.empty:
        return pd.Series(dtype="float64")

    index = pd.to_datetime(prices.index, errors="coerce")
    prices.index = index
    prices = prices[prices.index.notna()].sort_index()
    return prices.groupby(level=0).last()


def _finite_values(values: pd.Series) -> tuple[float, ...]:
    value_array = array("d", values.to_numpy(dtype=np.float64))
    return tuple(value for value in value_array if isfinite(value))


def _has_invalid_close(prices: pd.Series) -> bool:
    value_array = array("d", prices.to_numpy(dtype=np.float64))
    return any(value <= 0.0 or not isfinite(value) for value in value_array)


def _unavailable_metrics(max_drawdown: float | None = None) -> RiskMetrics:
    return RiskMetrics(
        volatility=None,
        max_drawdown=max_drawdown,
        sharpe=None,
        sortino=None,
        historical_var_95=None,
        beta=None,
        alpha=None,
    )


def _max_drawdown(prices: pd.Series) -> float | None:
    if prices.empty:
        return None
    max_price = float("-inf")
    drawdowns: list[float] = []
    for price in _finite_values(prices):
        max_price = max(max_price, price)
        drawdowns.append(price / max_price - 1.0)
    if len(drawdowns) == 0:
        return None
    return min(drawdowns)


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    return numerator / denominator


def _benchmark_risk(
    returns: pd.Series,
    benchmark_frame: pd.DataFrame | None,
) -> tuple[float | None, float | None]:
    if benchmark_frame is None:
        return None, None
    try:
        benchmark_prices = close_prices(benchmark_frame)
    except MissingCloseColumnError:
        benchmark_prices = pd.Series(dtype="float64")
    if benchmark_prices.empty or _has_invalid_close(benchmark_prices):
        return None, None
    benchmark_returns = benchmark_prices.pct_change().dropna()
    joined = pd.concat(
        [returns.rename("asset"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if len(joined) < MIN_RISK_RETURNS:
        return None, None
    asset_values = _finite_values(joined["asset"])
    benchmark_values = _finite_values(joined["benchmark"])
    if len(asset_values) < MIN_RISK_RETURNS or len(asset_values) != len(
        benchmark_values,
    ):
        return None, None
    asset_mean = fmean(asset_values)
    benchmark_mean = fmean(benchmark_values)
    benchmark_variance = stdev(benchmark_values) ** 2
    if benchmark_variance == 0.0 or not np.isfinite(benchmark_variance):
        return None, None
    covariance = sum(
        (asset - asset_mean) * (benchmark - benchmark_mean)
        for asset, benchmark in zip(asset_values, benchmark_values, strict=True)
    ) / (len(asset_values) - 1)
    beta = covariance / benchmark_variance
    alpha = (asset_mean - beta * benchmark_mean) * TRADING_DAYS_PER_YEAR
    return beta, alpha


def _percentile(sorted_values: tuple[float, ...], percentile: float) -> float:
    raw_index = (len(sorted_values) - 1) * percentile / 100.0
    lower_index = int(raw_index)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = raw_index - lower_index
    return sorted_values[lower_index] + (
        sorted_values[upper_index] - sorted_values[lower_index]
    ) * fraction
