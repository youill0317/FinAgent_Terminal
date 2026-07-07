from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd  # noqa: PANDAS_OK

type MarketScalar = str | int | float | bool | None


class FreshnessState(StrEnum):
    NOT_LOADED = "not_loaded"
    FRESH = "fresh"
    STALE = "stale"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProviderError:
    symbol: str
    message: str
    error_type: str

    @classmethod
    def from_exception(cls, symbol: str, error: Exception) -> ProviderError:
        return cls(
            symbol=symbol,
            message=str(error),
            error_type=type(error).__name__,
        )


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    last_price: float | None
    previous_close: float | None
    change: float | None
    change_percent: float | None
    currency: str | None
    market_cap: int | None
    volume: int | None
    day_high: float | None
    day_low: float | None
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class OhlcvFrame:
    symbol: str
    period: str
    interval: str
    frame: pd.DataFrame
    rows: int
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class Profile:
    symbol: str
    name: str | None
    sector: str | None
    industry: str | None
    website: str | None
    market_cap: int | None
    trailing_pe: float | None
    forward_pe: float | None
    dividend_yield: float | None
    beta: float | None
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class MarketDataSnapshot:
    symbol: str
    period: str
    interval: str
    fetched_at: datetime
    freshness: FreshnessState
    quote: Quote | None = None
    history: OhlcvFrame | None = None
    profile: Profile | None = None
    error: ProviderError | None = None

    def with_freshness(self, freshness: FreshnessState) -> MarketDataSnapshot:
        if freshness == self.freshness:
            return self
        return replace(self, freshness=freshness)
