# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""일봉 시세 기반 기술적 보조지표(SMA·RSI·MACD·볼린저·스토캐스틱·ATR·CCI·OBV) 산출."""

from __future__ import annotations

import math
from typing import Final, cast

import numpy as np
import pandas as pd

SMA_WINDOWS: Final = (5, 20, 60, 200)
RSI_WINDOW: Final = 14
MACD_FAST: Final = 12
MACD_SLOW: Final = 26
MACD_SIGNAL: Final = 9
BAND_WINDOW: Final = 20
BAND_WIDTH: Final = 2.0
STOCH_WINDOW: Final = 14
STOCH_SMOOTH: Final = 3
ATR_WINDOW: Final = 14
CCI_WINDOW: Final = 20
CCI_FACTOR: Final = 0.015
MOMENTUM_WINDOW: Final = 10
WILLIAMS_WINDOW: Final = 14
VOLUME_WINDOW: Final = 20
PERCENT: Final = 100.0


def _wilder(values: pd.Series, window: int) -> pd.Series:
    smooth = values.ewm(alpha=1 / window, adjust=False, min_periods=window)
    return cast("pd.Series", smooth.mean())


def _ema(values: pd.Series, window: int) -> pd.Series:
    smooth = values.ewm(span=window, adjust=False, min_periods=window)
    return cast("pd.Series", smooth.mean())


def _rsi(close: pd.Series) -> pd.Series:
    change = close.diff()
    gain = _wilder(change.clip(lower=0), RSI_WINDOW)
    loss = _wilder(-change.clip(upper=0), RSI_WINDOW)
    return gain / (gain + loss) * PERCENT


def _true_range(frame: pd.DataFrame) -> pd.Series:
    before = cast("pd.Series", frame["close"]).shift(1)
    return pd.concat(
        [
            cast("pd.Series", frame["high"]) - cast("pd.Series", frame["low"]),
            (cast("pd.Series", frame["high"]) - before).abs(),
            (cast("pd.Series", frame["low"]) - before).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _mean_deviation(values: np.ndarray) -> float:
    return float(np.abs(values - values.mean()).mean())


def indicators(frame: pd.DataFrame) -> dict[str, float]:
    bars = frame.sort_values("date").reset_index(drop=True)
    close = cast("pd.Series", bars["close"])
    high = cast("pd.Series", bars["high"])
    low = cast("pd.Series", bars["low"])
    volume = cast("pd.Series", bars["volume"])
    typical = (high + low + close) / 3
    macd = _ema(close, MACD_FAST) - _ema(close, MACD_SLOW)
    signal = _ema(macd, MACD_SIGNAL)
    middle = close.rolling(BAND_WINDOW).mean()
    spread = close.rolling(BAND_WINDOW).std(ddof=0) * BAND_WIDTH
    highest = high.rolling(STOCH_WINDOW).max()
    lowest = low.rolling(STOCH_WINDOW).min()
    stoch = (close - lowest) / (highest - lowest) * PERCENT
    typical_mean = typical.rolling(CCI_WINDOW).mean()
    deviation = typical.rolling(CCI_WINDOW).apply(_mean_deviation, raw=True)
    highest_w = high.rolling(WILLIAMS_WINDOW).max()
    lowest_w = low.rolling(WILLIAMS_WINDOW).min()
    series: dict[str, pd.Series] = {
        **{f"sma_{n}": close.rolling(n).mean() for n in SMA_WINDOWS},
        "rsi_14": _rsi(close),
        "macd": macd,
        "macd_signal": signal,
        "macd_hist": macd - signal,
        "bollinger_pct_b": (close - (middle - spread)) / (2 * spread),
        "stoch_k": stoch,
        "stoch_d": stoch.rolling(STOCH_SMOOTH).mean(),
        "atr_14": _wilder(_true_range(bars), ATR_WINDOW),
        "cci_20": (typical - typical_mean) / (CCI_FACTOR * deviation),
        "momentum_10": close - close.shift(MOMENTUM_WINDOW),
        "williams_r_14": (highest_w - close) / (highest_w - lowest_w) * -PERCENT,
        "obv": (pd.Series(np.sign(close.diff())).fillna(0) * volume).cumsum(),
        "volume_avg_20": volume.rolling(VOLUME_WINDOW).mean(),
    }
    latest = {name: float(values.iloc[-1]) for name, values in series.items()}
    return {name: value for name, value in latest.items() if math.isfinite(value)}
