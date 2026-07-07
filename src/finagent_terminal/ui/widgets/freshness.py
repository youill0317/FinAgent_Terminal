from __future__ import annotations

# pyright: reportUnnecessaryComparison=false
from typing import assert_never

from finagent_terminal.data.models import FreshnessState


def freshness_badge(state: FreshnessState) -> str:
    match state:
        case FreshnessState.NOT_LOADED:
            return "[not_loaded]"
        case FreshnessState.FRESH:
            return "[fresh]"
        case FreshnessState.STALE:
            return "[stale]"
        case FreshnessState.FAILED:
            return "[failed]"
        case unreachable:
            assert_never(unreachable)
