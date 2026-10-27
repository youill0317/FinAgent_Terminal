"""One pace for every SEC EDGAR request in this process.

The screener's 10-K margins (``screen``) and the 13F page (``investors``) can
fetch at the same time; SEC allows a user 10 requests a second, and the two
together stay at ``PER_SECOND``.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

PER_SECOND: Final = 8
INTERVAL: Final = 1 / PER_SECOND

_LOCK: Final = threading.Lock()
_last = float("-inf")  # monotonic time the latest request was let through


def wait(sleep: Callable[[float], None] = time.sleep) -> None:
    """Return when the next SEC request may start: starts are ``INTERVAL`` apart."""
    global _last  # noqa: PLW0603 - the one shared pace
    with _LOCK:
        gap = _last + INTERVAL - time.monotonic()
        if gap > 0:
            sleep(gap)
        _last = time.monotonic()
