from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from finagent.sources import Series


@dataclass(frozen=True, slots=True)
class UiContext:
    full_series: Callable[[str], Series]
