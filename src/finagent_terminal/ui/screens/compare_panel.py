from __future__ import annotations

from typing import TYPE_CHECKING

from finagent_terminal.compare import (
    MAX_COMPARE_SYMBOLS,
    MIN_COMPARE_SYMBOLS,
    CompareResult,
    CompareSymbolLimitError,
    compare_symbols,
)
from finagent_terminal.ui.widgets.charts import render_frame_chart
from finagent_terminal.ui.widgets.tables import (
    render_performance_table,
    render_tail_table,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pandas import DataFrame  # noqa: RUF100  # noqa: PANDAS_OK

    from finagent_terminal.data.models import MarketDataSnapshot


def render_compare_panel(
    title: str,
    symbols: tuple[str, ...],
    snapshots: Mapping[str, MarketDataSnapshot],
) -> tuple[str, ...]:
    try:
        _require_symbol_limit(symbols)
    except CompareSymbolLimitError as error:
        return (title, str(error))

    skipped, frames = _loaded_frames(symbols, snapshots)
    if len(frames) < MIN_COMPARE_SYMBOLS:
        return (
            title,
            f"Selected: {', '.join(symbols)}",
            f"Skipped: {', '.join(skipped) if skipped else 'none'}",
            "compare mode needs at least 2 loaded symbols",
        )

    result = compare_symbols(frames)
    all_skipped = (*skipped, *result.skipped_symbols)
    return (
        title,
        f"Selected: {', '.join(symbols)}",
        f"Skipped: {', '.join(all_skipped) if all_skipped else 'none'}",
        *_result_lines(result),
    )


def _loaded_frames(
    symbols: tuple[str, ...],
    snapshots: Mapping[str, MarketDataSnapshot],
) -> tuple[tuple[str, ...], dict[str, DataFrame]]:
    skipped: list[str] = []
    frames: dict[str, DataFrame] = {}
    for symbol in symbols:
        snapshot = snapshots.get(symbol)
        if snapshot is None or snapshot.error is not None or snapshot.history is None:
            skipped.append(symbol)
            continue
        frames[symbol] = snapshot.history.frame
    return tuple(skipped), frames


def _result_lines(result: CompareResult) -> tuple[str, ...]:
    return (
        "Normalized returns:",
        *render_tail_table(result.normalized_returns),
        "Relative performance:",
        *render_performance_table(result.relative_performance),
        "Chart:",
        *render_frame_chart(result.normalized_returns),
    )


def _require_symbol_limit(symbols: tuple[str, ...]) -> None:
    symbol_count = len(symbols)
    if symbol_count < MIN_COMPARE_SYMBOLS or symbol_count > MAX_COMPARE_SYMBOLS:
        raise CompareSymbolLimitError(count=symbol_count)
