"""매크로·종목·ETF 정밀 분석 프리셋(슬래시 명령·스레드 이름) 정의."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


@dataclass(frozen=True, slots=True)
class Preset:
    id: Literal["macro", "stock", "etf"]
    title: str


MACRO: Final = Preset("macro", "매크로 분석")
STOCK: Final = Preset("stock", "종목 분석")
ETF: Final = Preset("etf", "ETF 분석")
