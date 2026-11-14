from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import streamlit as st

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finagent.models import Area


@dataclass(frozen=True)
class Palette:
    area: dict[Area, str]
    up: str
    down: str
    ink: str
    guide: str
    muted: str
    paper: str

    def signed(self, up: bool) -> str:
        return self.up if up else self.down

    def tones(self, count: int) -> list[str]:
        if count <= 1:
            return [self.ink] * count
        return [mix(self.ink, self.guide, 0.82 * i / (count - 1)) for i in range(count)]

    def series(self, names: Sequence[str]) -> list[str]:
        fixed = {
            "포트폴리오": self.ink,
            "평가액": self.ink,
            "총손익": self.ink,
            "평가손익": self.ink,
            "종가": self.ink,
            "S&P 500": self.area["US"],
            "순투입": self.muted,
            "실현손익": self.muted,
        }
        extra = [self.area[a] for a in ("JP", "CN", "EA", "US", "KR")]
        out: list[str] = []
        spare = 0
        for name in names:
            if name in fixed:
                out.append(fixed[name])
            else:
                out.append(extra[spare % len(extra)])
                spare += 1
        return out


LIGHT: Final = Palette(
    area={
        "US": "#2563EB",
        "KR": "#E11D48",
        "EA": "#D97706",
        "JP": "#7C3AED",
        "CN": "#0D9488",
        "GL": "#475569",
    },
    up="#DC2626",
    down="#1D4ED8",
    ink="#0F172A",
    guide="#94A3B8",
    muted="#64748B",
    paper="#FFFFFF",
)
DARK: Final = Palette(
    area={
        "US": "#70A6E9",
        "KR": "#EC7E8B",
        "EA": "#F1C25B",
        "JP": "#A591E9",
        "CN": "#5FCFBE",
        "GL": "#94A3B8",
    },
    up="#E97E7B",
    down="#70A6E9",
    ink="#F1F5F9",
    guide="#64748B",
    muted="#94A3B8",
    paper="#0F172A",
)


def kind() -> str:
    return "dark" if st.context.theme.type == "dark" else "light"


def current() -> Palette:
    return DARK if kind() == "dark" else LIGHT


TINTS: Final = (0.0, 0.45, 0.7, 0.82)


def mix(color: str, target: str, share: float) -> str:
    if not share:
        return color
    base = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    end = [int(target[i : i + 2], 16) for i in (1, 3, 5)]
    blended = [round(a + (b - a) * share) for a, b in zip(base, end, strict=True)]
    return "#" + "".join(f"{value:02X}" for value in blended)


def tint(color: str, step: int) -> str:
    share = TINTS[min(step, len(TINTS) - 1)]
    return mix(color, current().guide, share)


def scheme_css(theme: str) -> str:
    return f"<style>:root {{color-scheme: {theme}}}</style>"
