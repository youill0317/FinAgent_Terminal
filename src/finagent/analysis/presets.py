"""Starting points for the analysis box.

A preset only fills the input; the user edits it and presses run. ``panels``
are deep panels fetched for that run only (the saved view does not change).
The texts live in ``text/preset_<id>.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Final

TEXT: Final = files("finagent.analysis") / "text"
# The deep-dive preset opens with this placeholder selected in the input.
PLACEHOLDER: Final = "{지표}"


@dataclass(frozen=True, slots=True)
class Preset:
    id: str
    title: str
    panels: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return (TEXT / f"preset_{self.id}.txt").read_text(encoding="utf-8").strip()


PRESETS: Final = (
    Preset("briefing", "오늘의 브리핑"),
    Preset("cycle", "경기 국면", ("growth", "labor", "cycle")),
    Preset("inflation_rates", "물가·금리 경로", ("inflation", "rates")),
    Preset("recession", "침체 신호", ("fincon", "labor")),
    Preset("liquidity", "유동성·신용", ("liquidity", "rates", "fincon")),
    Preset("korea_us", "한국 vs 미국", ("korea", "korea_real", "rates", "credit")),
    Preset("counter", "반대 근거 점검"),
    Preset("deep_dive", "지표 하나 깊게"),
)
PRESET_BY_ID: Final = {preset.id: preset for preset in PRESETS}
