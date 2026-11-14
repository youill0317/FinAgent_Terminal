"""매크로·종목·ETF 정밀 분석 프리셋 정의 및 프롬프트 로더."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from finagent.analysis import prompt


@dataclass(frozen=True, slots=True)
class Preset:
    id: Literal["macro", "stock", "etf"]
    title: str

    @property
    def text(self) -> str:
        return prompt.text(f"preset_{self.id}.txt")


MACRO: Final = Preset("macro", "매크로 분석")
STOCK: Final = Preset("stock", "종목 분석")
ETF: Final = Preset("etf", "ETF 분석")
