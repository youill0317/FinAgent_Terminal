from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, override

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_CLOSE_COLUMN: Final = "Close"
DEFAULT_MOVING_AVERAGE_WINDOWS: Final = (20, 50)
TRADING_DAYS_PER_YEAR: Final = 252
RSI_WINDOW: Final = 14
MACD_FAST_SPAN: Final = 12
MACD_SLOW_SPAN: Final = 26
MACD_SIGNAL_SPAN: Final = 9


@dataclass(frozen=True, slots=True)
class MissingCloseColumnError(Exception):
    column: str

    @override
    def __str__(self) -> str:
        return f"missing close column: {self.column}"


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
