from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, override

import pandas as pd  # noqa: PANDAS_OK

from finagent_terminal.analytics import (
    DEFAULT_CLOSE_COLUMN,
    MissingCloseColumnError,
    close_prices,
    normalized_returns,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

MIN_COMPARE_SYMBOLS: Final = 2
MAX_COMPARE_SYMBOLS: Final = 6


@dataclass(frozen=True, slots=True)
class CompareSymbolLimitError(Exception):
    count: int

    @override
    def __str__(self) -> str:
        return f"compare mode requires 2 to 6 symbols, got {self.count}"


@dataclass(frozen=True, slots=True)
class CompareResult:
    normalized_returns: pd.DataFrame
    relative_performance: pd.DataFrame
    skipped_symbols: tuple[str, ...]


def compare_symbols(
    frames: Mapping[str, pd.DataFrame],
    close_column: str = DEFAULT_CLOSE_COLUMN,
) -> CompareResult:
    symbol_count = len(frames)
    if symbol_count < MIN_COMPARE_SYMBOLS or symbol_count > MAX_COMPARE_SYMBOLS:
        raise CompareSymbolLimitError(count=symbol_count)

    valid_prices: dict[str, pd.Series] = {}
    skipped_symbols: list[str] = []
    for symbol, frame in frames.items():
        try:
            prices = close_prices(frame, close_column)
        except MissingCloseColumnError:
            skipped_symbols.append(symbol)
            continue
        if prices.empty:
            skipped_symbols.append(symbol)
            continue
        valid_prices[symbol] = prices

    aligned = pd.DataFrame(valid_prices).dropna()
    if aligned.empty:
        return CompareResult(
            normalized_returns=pd.DataFrame(),
            relative_performance=pd.DataFrame(columns=["symbol", "total_return"]),
            skipped_symbols=tuple(skipped_symbols),
        )

    normalized = aligned.apply(normalized_returns)
    total_returns = normalized.iloc[-1] / normalized.iloc[0] - 1.0
    relative_performance = (
        total_returns.rename("total_return")
        .rename_axis("symbol")
        .reset_index()
        .sort_values("total_return", ascending=False, ignore_index=True)
    )
    return CompareResult(
        normalized_returns=normalized,
        relative_performance=relative_performance,
        skipped_symbols=tuple(skipped_symbols),
    )
