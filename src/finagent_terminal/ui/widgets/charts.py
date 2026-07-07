from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

CHART_CHARS: Final = " .:-=+*#%@"


class ChartSeries(Protocol):
    @property
    def empty(self) -> bool: ...

    def dropna(self) -> ChartSeries: ...
    def tail(self, n: int) -> ChartSeries: ...
    def tolist(self) -> list[float]: ...


class ChartFrame(Protocol):
    @property
    def empty(self) -> bool: ...

    def items(self) -> Iterable[tuple[Hashable, ChartSeries]]: ...


def render_series_chart(values: ChartSeries, width: int = 48) -> str:
    clean_values = values.dropna()
    if clean_values.empty:
        return "(no data)"

    points = tuple(float(value) for value in clean_values.tail(width).tolist())
    low = min(points)
    high = max(points)
    if high == low:
        return "-" * len(points)

    span = high - low
    scale = len(CHART_CHARS) - 1
    return "".join(CHART_CHARS[int((value - low) / span * scale)] for value in points)


def render_frame_chart(frame: ChartFrame, width: int = 32) -> tuple[str, ...]:
    if frame.empty:
        return ("(no data)",)
    return tuple(
        f"{column!s:<8} {render_series_chart(series, width)}"
        for column, series in frame.items()
    )
