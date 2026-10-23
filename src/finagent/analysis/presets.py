"""Starting points for the analysis box.

A preset starts a run with its text. ``panels`` are deep panels fetched for
that run only (the saved view does not change).
The texts live in ``text/preset_<id>.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Final

TEXT: Final = files("finagent.analysis") / "text"


@dataclass(frozen=True, slots=True)
class Preset:
    id: str
    title: str
    panels: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return (TEXT / f"preset_{self.id}.txt").read_text(encoding="utf-8").strip()


PRESETS: Final = (
    Preset(
        "macro",
        "매크로 분석",
        (
            "growth",
            "labor",
            "cycle",
            "inflation",
            "rates",
            "fincon",
            "liquidity",
            "household",
            "korea",
            "korea_real",
            "euro",
            "global",
            "markets",
            "commodities",
            "credit",
            "housing",
        ),
    ),
)
PRESET_BY_ID: Final = {preset.id: preset for preset in PRESETS}
