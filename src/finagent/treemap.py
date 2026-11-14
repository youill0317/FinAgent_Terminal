"""비중 시각화를 위한 Squarified Treemap 사각형 분할 알고리즘."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def inset(self, gap: float, top: float = 0.0) -> Rect:
        return Rect(
            self.x + gap / 2,
            self.y + gap / 2 + top,
            max(self.w - gap, 0.0),
            max(self.h - gap - top, 0.0),
        )


def _worst(row: Sequence[float], side: float) -> float:
    total = sum(row)
    return max(
        max(side * side * area / (total * total), total * total / (side * side * area))
        for area in row
    )


def squarify(sizes: Sequence[float], box: Rect) -> list[Rect]:
    total = sum(sizes)
    out = [Rect(box.x, box.y, 0.0, 0.0)] * len(sizes)
    if total <= 0 or box.w <= 0 or box.h <= 0:
        return out
    scale = box.w * box.h / total
    order = sorted((i for i, s in enumerate(sizes) if s > 0), key=lambda i: -sizes[i])
    areas = [sizes[i] * scale for i in order]
    x, y, w, h = box.x, box.y, box.w, box.h
    start = 0
    while start < len(areas):
        side = min(w, h)
        end = start + 1
        while end < len(areas) and _worst(areas[start : end + 1], side) <= _worst(
            areas[start:end], side
        ):
            end += 1
        thick = sum(areas[start:end]) / side
        offset = 0.0
        for k in range(start, end):
            length = areas[k] / thick
            out[order[k]] = (
                Rect(x, y + offset, thick, length)
                if w >= h
                else Rect(x + offset, y, length, thick)
            )
            offset += length
        if w >= h:
            x, w = x + thick, w - thick
        else:
            y, h = y + thick, h - thick
        start = end
    return out
