from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

SYMBOL_PATTERN: Final = re.compile(r"^[A-Z0-9._\-^=]{1,32}$")


@dataclass(frozen=True, slots=True)
class InvalidSymbolError(ValueError):
    symbol: str


@dataclass(frozen=True, slots=True)
class EmptySymbolGroupError(ValueError):
    name: str


@dataclass(frozen=True, slots=True)
class EmptySymbolGroupNameError(ValueError):
    name: str


@dataclass(frozen=True, slots=True)
class ScalarSymbolCollectionError(ValueError):
    value_type: str


def normalize_symbol(raw: str) -> str:
    symbol = raw.strip().upper()
    if SYMBOL_PATTERN.fullmatch(symbol) is None:
        raise InvalidSymbolError(raw)
    return symbol


class SymbolEntry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    label: str | None = None

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)


type SymbolPayload = dict[str, str | None]
type SymbolInput = SymbolEntry | SymbolPayload | str


class SymbolGroup(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    symbols: tuple[SymbolEntry, ...]
    label: str | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        name = value.strip()
        if name == "":
            raise EmptySymbolGroupNameError(value)
        return name

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(
        cls,
        value: tuple[SymbolInput, ...] | list[SymbolInput] | str | bytes,
    ) -> tuple[SymbolEntry, ...]:
        if isinstance(value, (str, bytes)):
            raise ScalarSymbolCollectionError(type(value).__name__)

        entries: list[SymbolEntry] = []
        for entry in value:
            if isinstance(entry, SymbolEntry):
                entries.append(entry)
                continue
            if isinstance(entry, str):
                entries.append(SymbolEntry(symbol=entry))
                continue
            entries.append(SymbolEntry.model_validate(entry))
        return tuple(entries)

    @field_validator("symbols")
    @classmethod
    def require_symbols(cls, value: tuple[SymbolEntry, ...]) -> tuple[SymbolEntry, ...]:
        if len(value) == 0:
            raise EmptySymbolGroupError(cls.__name__)
        return value


def _group(
    name: str,
    symbols: tuple[str, ...],
    label: str | None = None,
) -> SymbolGroup:
    return SymbolGroup.model_validate(
        {"name": name, "symbols": list(symbols), "label": label},
    )


DEFAULT_PRESETS: Final[tuple[SymbolGroup, ...]] = (
    _group("Major Indices", ("^GSPC", "^IXIC", "^DJI", "SPY", "QQQ", "DIA")),
    _group("Mega Cap Tech", ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA")),
    _group("Semiconductors", ("NVDA", "AMD", "AVGO", "ASML", "TSM", "INTC", "MU")),
    _group("Banks", ("JPM", "BAC", "WFC", "C", "GS", "MS")),
    _group("Energy", ("XOM", "CVX", "COP", "SLB", "EOG", "XLE")),
    _group("Crypto ETFs", ("IBIT", "FBTC", "ARKB", "BITB", "GBTC", "ETHE")),
)
