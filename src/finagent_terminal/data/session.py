from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol

from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    ProviderError,
)
from finagent_terminal.symbols import normalize_symbol

DEFAULT_STALE_AFTER: Final = timedelta(minutes=15)
type Clock = Callable[[], datetime]


class MarketDataProvider(Protocol):
    def fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot: ...


@dataclass(frozen=True, slots=True)
class CacheKey:
    symbol: str
    period: str
    interval: str


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)  # noqa: MUTABLE_OK
class DataSession:
    provider: MarketDataProvider
    clock: Clock = utc_now
    stale_after: timedelta = DEFAULT_STALE_AFTER
    _cache: dict[CacheKey, MarketDataSnapshot] = field(default_factory=dict, init=False)

    def get_or_fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        key = _cache_key(symbol, period, interval)
        cached = self.get_cached(symbol, period=period, interval=interval)
        if cached is not None and cached.freshness == FreshnessState.FRESH:
            return cached

        try:
            snapshot = self.provider.fetch(
                key.symbol,
                period=period,
                interval=interval,
            )
        except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
            snapshot = MarketDataSnapshot(
                symbol=key.symbol,
                period=period,
                interval=interval,
                fetched_at=self.clock(),
                freshness=FreshnessState.FAILED,
                error=ProviderError.from_exception(key.symbol, error),
            )
        self._cache[key] = snapshot
        return snapshot

    def get_cached(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot | None:
        snapshot = self._cache.get(_cache_key(symbol, period, interval))
        if snapshot is None:
            return None
        return snapshot.with_freshness(self._freshness_for(snapshot))

    def freshness(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> FreshnessState:
        cached = self.get_cached(symbol, period=period, interval=interval)
        if cached is None:
            return FreshnessState.NOT_LOADED
        return cached.freshness

    def _freshness_for(self, snapshot: MarketDataSnapshot) -> FreshnessState:
        if snapshot.error is not None:
            return FreshnessState.FAILED
        if self.clock() - snapshot.fetched_at >= self.stale_after:
            return FreshnessState.STALE
        return FreshnessState.FRESH


def _cache_key(symbol: str, period: str, interval: str) -> CacheKey:
    return CacheKey(symbol=normalize_symbol(symbol), period=period, interval=interval)
