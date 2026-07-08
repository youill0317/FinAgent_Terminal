from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, final

from finagent_terminal.data.source_models import (
    SourceAvailability,
    SourceBundle,
    SourceItem,
    SourceRequest,
    SourceStatus,
    SourceStatusValue,
)

if TYPE_CHECKING:
    from types import TracebackType

    import httpx

__all__ = (
    "SourceAvailability",
    "SourceBundle",
    "SourceItem",
    "SourceRequest",
    "SourceSession",
    "SourceStatus",
    "SourceStatusValue",
    "source_availability",
    "source_availability_lines",
)

SOURCE_ENV_SETTINGS: Final = (
    ("FRED", "FRED_API_KEY", True),
    ("Finnhub", "FINNHUB_API_KEY", True),
    ("FMP", "FMP_API_KEY", True),
    ("SEC EDGAR", "FINAGENT_SEC_USER_AGENT", False),
)

type SourceCacheKey = str | tuple[str, ...]
type SourceFetch = Callable[[], SourceBundle]


@final
class SourceSession:
    __slots__ = ("_cache", "_finnhub", "_fmp", "_public_sources")

    def __init__(  # noqa: D107
        self,
        *,
        env: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        from finagent_terminal.data.public_sources import (  # noqa: PLC0415
            PublicSourceClient,
        )
        from finagent_terminal.data.vendor_clients import (  # noqa: PLC0415
            FinnhubClient,
            FmpClient,
        )

        self._cache: dict[tuple[str, ...], SourceBundle] = {}
        self._public_sources = PublicSourceClient(env=env, http_client=http_client)
        self._fmp = FmpClient.from_env(env, http_client=http_client)
        self._finnhub = FinnhubClient.from_env(env, http_client=http_client)

    def __enter__(self) -> SourceSession:
        """Return this session for context-manager cleanup."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close owned HTTP resources when leaving a context."""
        self.close()

    def close(self) -> None:
        self._public_sources.close()

    def get_or_fetch(
        self,
        key: SourceCacheKey,
        fetcher: SourceFetch,
    ) -> SourceBundle:
        cache_key = normalize_source_cache_key(key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        bundle = fetcher()
        self._cache[cache_key] = bundle
        return bundle

    def get_cached(self, key: SourceCacheKey) -> SourceBundle | None:
        return self._cache.get(normalize_source_cache_key(key))

    def clear(self) -> None:
        self._cache.clear()
        self._public_sources.clear()

    def cache_size(self) -> int:
        return len(self._cache) + self._public_sources.cache_size()

    def sec_companyfacts(self, ticker: str) -> SourceBundle:
        return self._public_sources.sec_companyfacts(ticker)

    def sec_submissions(self, ticker: str) -> SourceBundle:
        return self._public_sources.sec_submissions(ticker)

    def fred_series_observations(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> SourceBundle:
        return self._public_sources.fred_series_observations(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
        )

    def treasury_fiscal_data(
        self,
        endpoint: str,
        *,
        page_size: int = 100,
    ) -> SourceBundle:
        return self._public_sources.treasury_fiscal_data(
            endpoint,
            page_size=page_size,
        )

    def gdelt_articles(self, query: str) -> SourceBundle:
        return self._public_sources.gdelt_articles(query)

    def gdelt_events(self, query: str) -> SourceBundle:
        return self._public_sources.gdelt_events(query)

    def fmp_symbol(self, ticker: str) -> SourceBundle:
        ticker_key = ticker.strip().upper()
        return self.get_or_fetch(
            ("fmp", ticker_key),
            lambda: self._fmp.fetch_symbol(ticker_key),
        )

    def fmp_news(self, ticker: str) -> SourceBundle:
        ticker_key = ticker.strip().upper()
        return self.get_or_fetch(
            ("fmp", "news", ticker_key),
            lambda: self._fmp.fetch_news(ticker_key),
        )

    def fmp_fundamentals(self, ticker: str) -> SourceBundle:
        ticker_key = ticker.strip().upper()
        return self.get_or_fetch(
            ("fmp", "fundamentals", ticker_key),
            lambda: self._fmp.fetch_fundamentals(ticker_key),
        )

    def fmp_estimates(self, ticker: str) -> SourceBundle:
        ticker_key = ticker.strip().upper()
        return self.get_or_fetch(
            ("fmp", "estimates", ticker_key),
            lambda: self._fmp.fetch_estimates(ticker_key),
        )

    def finnhub_symbol(self, ticker: str) -> SourceBundle:
        ticker_key = ticker.strip().upper()
        return self.get_or_fetch(
            ("finnhub", ticker_key),
            lambda: self._finnhub.fetch_symbol(ticker_key),
        )

    def finnhub_news(self, ticker: str) -> SourceBundle:
        ticker_key = ticker.strip().upper()
        return self.get_or_fetch(
            ("finnhub", "news", ticker_key),
            lambda: self._finnhub.fetch_news(ticker_key),
        )

    def finnhub_sentiment(self, ticker: str) -> SourceBundle:
        ticker_key = ticker.strip().upper()
        return self.get_or_fetch(
            ("finnhub", "sentiment", ticker_key),
            lambda: self._finnhub.fetch_sentiment(ticker_key),
        )

    def finnhub_estimates(self, ticker: str) -> SourceBundle:
        ticker_key = ticker.strip().upper()
        return self.get_or_fetch(
            ("finnhub", "estimates", ticker_key),
            lambda: self._finnhub.fetch_estimates(ticker_key),
        )


def source_availability(
    env: Mapping[str, str] | None = None,
) -> tuple[SourceAvailability, ...]:
    source = os.environ if env is None else env
    return tuple(
        SourceAvailability(
            source=source_name,
            env_var=env_var,
            configured=_configured(source.get(env_var)),
            required=required,
        )
        for source_name, env_var, required in SOURCE_ENV_SETTINGS
    )


def source_availability_lines(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    return (
        "Data sources:",
        *(_availability_line(item) for item in source_availability(env)),
    )


def normalize_source_cache_key(key: SourceCacheKey) -> tuple[str, ...]:
    if isinstance(key, str):
        return (_normalize_cache_part(key),)
    return tuple(_normalize_cache_part(part) for part in key)


def _availability_line(availability: SourceAvailability) -> str:
    if availability.configured:
        return (
            f"{availability.source} ({availability.env_var}): "
            f"{availability.status.value}"
        )
    if availability.required:
        return (
            f"{availability.source} ({availability.env_var}): "
            f"{availability.status.value} (missing {availability.env_var})"
        )
    return (
        f"{availability.source} ({availability.env_var}): "
        f"{availability.status.value} (optional)"
    )


def _configured(value: str | None) -> bool:
    return value is not None and value.strip() != ""


def _normalize_cache_part(value: str) -> str:
    return value.strip()
