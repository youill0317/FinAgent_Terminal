"""Values a cited id stands for. The verifier compares claimed numbers to these."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

NUMBER_IN_TEXT: Final = re.compile(r"[-+−]?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class Evidence:
    """``cite`` is what the model writes inside ``[[...]]``.

    ``points`` are the values in order (oldest first); ``extra`` holds loose
    numbers such as rule thresholds that a sentence may quote.
    """

    cite: str
    label: str
    points: tuple[float, ...] = ()
    extra: tuple[float, ...] = ()

    def candidates(self) -> tuple[float, ...]:
        """Every number a correct sentence about this id could contain."""
        out: list[float] = [*self.points, *self.extra]
        if len(self.points) >= 2:  # noqa: PLR2004 - latest and previous
            prev, last = self.points[-2], self.points[-1]
            out.append(last - prev)
            if prev != 0:
                out.append((last / prev - 1.0) * 100.0)
        for prev, last in pairwise(self.points[-13:]):
            if prev != 0:
                out.append((last / prev - 1.0) * 100.0)
        if self.points:
            out.extend((max(self.points), min(self.points)))
        return tuple(out)


def numbers_in(text: str) -> tuple[float, ...]:
    return tuple(
        parse_number(match.group(0)) for match in NUMBER_IN_TEXT.finditer(text)
    )


def parse_number(text: str) -> float:
    return float(text.replace(",", "").replace("−", "-").replace("+", ""))


def merge(items: Iterable[Evidence]) -> dict[str, Evidence]:
    return {item.cite: item for item in items}
