from __future__ import annotations

from typing import TYPE_CHECKING, Final

from finagent_terminal.ui.widgets.freshness import freshness_badge

if TYPE_CHECKING:
    from finagent_terminal.data.models import FreshnessState, MarketDataSnapshot

CHART_LEVELS: Final = ".@"
MIN_CHART_POINTS: Final = 2


def market_row(
    symbol: str,
    snapshot: MarketDataSnapshot | None,
    freshness: FreshnessState,
) -> str:
    price = _format_price(snapshot)
    change = _format_change(snapshot)
    chart = mini_chart(snapshot)
    error = _format_error(snapshot)
    badge = freshness_badge(freshness)
    return (
        f"  {symbol:<8} {price:>10} {change:>17} "
        f"{badge:<12} chart: {chart}{error}"
    )


def mini_chart(snapshot: MarketDataSnapshot | None) -> str:
    if snapshot is None or snapshot.history is None:
        return "--"
    if snapshot.history.rows < MIN_CHART_POINTS:
        return "--"
    quote = snapshot.quote
    if (
        quote is None
        or quote.last_price is None
        or quote.previous_close is None
        or quote.last_price == quote.previous_close
    ):
        return "--"
    if quote.last_price > quote.previous_close:
        return CHART_LEVELS
    return CHART_LEVELS[::-1]


def _format_price(snapshot: MarketDataSnapshot | None) -> str:
    if snapshot is None or snapshot.quote is None or snapshot.quote.last_price is None:
        return "--"
    return f"{snapshot.quote.last_price:.2f}"


def _format_change(snapshot: MarketDataSnapshot | None) -> str:
    if snapshot is None or snapshot.quote is None:
        return "--"
    change = snapshot.quote.change
    change_percent = snapshot.quote.change_percent
    if change is None or change_percent is None:
        return "--"
    return f"{change:+.2f} ({change_percent:+.2f}%)"


def _format_error(snapshot: MarketDataSnapshot | None) -> str:
    if snapshot is None or snapshot.error is None:
        return ""
    return f" error: {snapshot.error.message}"
