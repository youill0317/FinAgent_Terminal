from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


class NumericColumns(Protocol):
    @property
    def columns(self) -> Iterable[str]: ...


class TableFrame(Protocol):
    @property
    def empty(self) -> bool: ...

    def tail(self, rows: int) -> TableFrame: ...
    def round(self, decimals: int) -> TableFrame: ...
    def select_dtypes(self, *, include: str) -> NumericColumns: ...

    def to_string(
        self,
        *,
        formatters: Mapping[str, Callable[[float], str]] | None = None,
        index: bool = True,
    ) -> str: ...


def render_key_value_rows(rows: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    if len(rows) == 0:
        return ("(none)",)
    return tuple(f"{key}: {value}" for key, value in rows)


def render_tail_table(frame: TableFrame, rows: int = 5) -> tuple[str, ...]:
    if frame.empty:
        return ("(no data)",)
    table = frame.tail(rows).round(2)
    numeric_columns = table.select_dtypes(include="number").columns
    formatters = dict.fromkeys(numeric_columns, "{:.2f}".format)
    return tuple(table.to_string(formatters=formatters).splitlines())


def render_performance_table(frame: TableFrame) -> tuple[str, ...]:
    if frame.empty:
        return ("(no data)",)
    formatters = {"total_return": "{:.2%}".format}
    return tuple(frame.to_string(formatters=formatters, index=False).splitlines())
