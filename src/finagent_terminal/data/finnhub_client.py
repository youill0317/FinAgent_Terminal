from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, final

from finagent_terminal.data.finnhub_items import (
    eps_items,
    news_items,
    quote_item,
    recommendation_items,
    sentiment_items,
)
from finagent_terminal.data.source_models import (
    SourceBundle,
    SourceItem,
    SourceRequest,
    SourceStatus,
)
from finagent_terminal.data.vendor_common import (
    VendorCall,
    VendorResponse,
    available_status,
    collect_vendor,
    default_http_client,
    env_value,
    get_vendor,
    missing_key_bundle,
    normalized_symbol,
)
from finagent_terminal.data.vendor_payloads import (
    FINNHUB_EPS_ADAPTER,
    FINNHUB_NEWS_ADAPTER,
    FINNHUB_QUOTE_ADAPTER,
    FINNHUB_RECOMMENDATION_ADAPTER,
    FINNHUB_SENTIMENT_ADAPTER,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx

FINNHUB_SOURCE: Final = "Finnhub"
FINNHUB_API_KEY_ENV: Final = "FINNHUB_API_KEY"
FINNHUB_QUOTE_URL: Final = "https://finnhub.io/api/v1/quote"
FINNHUB_NEWS_URL: Final = "https://finnhub.io/api/v1/company-news"
FINNHUB_SENTIMENT_URL: Final = "https://finnhub.io/api/v1/stock/social-sentiment"
FINNHUB_RECOMMENDATION_URL: Final = "https://finnhub.io/api/v1/stock/recommendation"
FINNHUB_EPS_URL: Final = "https://finnhub.io/api/v1/stock/eps-estimate"


@final
@dataclass(frozen=True, slots=True)
class FinnhubClient:
    _api_key: str | None
    _http_client: httpx.Client | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> FinnhubClient:
        return cls(env_value(FINNHUB_API_KEY_ENV, env), http_client)

    def fetch_symbol(self, symbol: str) -> SourceBundle:
        return self._fetch(symbol, self._fetch_symbol)

    def fetch_quote(self, symbol: str) -> SourceBundle:
        return self._fetch(symbol, self._fetch_quote)

    def fetch_news(self, symbol: str) -> SourceBundle:
        return self._fetch(symbol, self._fetch_news)

    def fetch_sentiment(self, symbol: str) -> SourceBundle:
        return self._fetch(symbol, self._fetch_sentiment)

    def fetch_estimates(self, symbol: str) -> SourceBundle:
        return self._fetch(symbol, self._fetch_estimates)

    def _fetch(
        self,
        symbol: str,
        fetcher: Callable[[httpx.Client, str], SourceBundle],
    ) -> SourceBundle:
        symbol_key = normalized_symbol(symbol, FINNHUB_SOURCE)
        if isinstance(symbol_key, SourceBundle):
            return symbol_key
        if self._api_key is None:
            return missing_key_bundle(FINNHUB_SOURCE, FINNHUB_API_KEY_ENV)
        if self._http_client is None:
            with default_http_client() as client:
                return fetcher(client, symbol_key)
        return fetcher(self._http_client, symbol_key)

    def _headers(self) -> dict[str, str]:
        return {"X-Finnhub-Token": self._api_key or ""}

    def _fetch_symbol(self, client: httpx.Client, symbol: str) -> SourceBundle:
        return _merge(
            (
                self._fetch_quote(client, symbol),
                self._fetch_news(client, symbol),
                self._fetch_sentiment(client, symbol),
                self._fetch_estimates(client, symbol),
            ),
        )

    def _fetch_quote(self, client: httpx.Client, symbol: str) -> SourceBundle:
        headers = self._headers()
        items: list[SourceItem] = []
        statuses: list[SourceStatus] = []
        warnings: list[str] = []
        requests: list[SourceRequest] = []

        quote = get_vendor(
            client,
            VendorCall(
                FINNHUB_SOURCE,
                FINNHUB_QUOTE_URL,
                {"symbol": symbol},
                headers,
                FINNHUB_QUOTE_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(quote, statuses, warnings, requests)
        if isinstance(quote, VendorResponse):
            items.append(quote_item(symbol, quote.data, FINNHUB_SOURCE))

        return _bundle(items, statuses, warnings, requests)

    def _fetch_news(self, client: httpx.Client, symbol: str) -> SourceBundle:
        headers = self._headers()
        items: list[SourceItem] = []
        statuses: list[SourceStatus] = []
        warnings: list[str] = []
        requests: list[SourceRequest] = []

        today = datetime.now(UTC).date()
        news_params = {
            "symbol": symbol,
            "from": str(today - timedelta(days=30)),
            "to": str(today),
        }
        news = get_vendor(
            client,
            VendorCall(
                FINNHUB_SOURCE,
                FINNHUB_NEWS_URL,
                news_params,
                headers,
                FINNHUB_NEWS_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(news, statuses, warnings, requests)
        if isinstance(news, VendorResponse):
            items.extend(news_items(news.data, FINNHUB_SOURCE))

        return _bundle(items, statuses, warnings, requests)

    def _fetch_sentiment(self, client: httpx.Client, symbol: str) -> SourceBundle:
        headers = self._headers()
        items: list[SourceItem] = []
        statuses: list[SourceStatus] = []
        warnings: list[str] = []
        requests: list[SourceRequest] = []

        sentiment = get_vendor(
            client,
            VendorCall(
                FINNHUB_SOURCE,
                FINNHUB_SENTIMENT_URL,
                {"symbol": symbol},
                headers,
                FINNHUB_SENTIMENT_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(sentiment, statuses, warnings, requests)
        if isinstance(sentiment, VendorResponse):
            items.extend(sentiment_items(symbol, sentiment.data, FINNHUB_SOURCE))

        return _bundle(items, statuses, warnings, requests)

    def _fetch_estimates(self, client: httpx.Client, symbol: str) -> SourceBundle:
        headers = self._headers()
        items: list[SourceItem] = []
        statuses: list[SourceStatus] = []
        warnings: list[str] = []
        requests: list[SourceRequest] = []

        recommendation = get_vendor(
            client,
            VendorCall(
                FINNHUB_SOURCE,
                FINNHUB_RECOMMENDATION_URL,
                {"symbol": symbol},
                headers,
                FINNHUB_RECOMMENDATION_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(recommendation, statuses, warnings, requests)
        if isinstance(recommendation, VendorResponse):
            items.extend(
                recommendation_items(symbol, recommendation.data, FINNHUB_SOURCE),
            )

        eps = get_vendor(
            client,
            VendorCall(
                FINNHUB_SOURCE,
                FINNHUB_EPS_URL,
                {"symbol": symbol},
                headers,
                FINNHUB_EPS_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(eps, statuses, warnings, requests)
        if isinstance(eps, VendorResponse):
            items.extend(eps_items(symbol, eps.data, FINNHUB_SOURCE))

        return _bundle(items, statuses, warnings, requests)


def _bundle(
    items: list[SourceItem],
    statuses: list[SourceStatus],
    warnings: list[str],
    requests: list[SourceRequest],
) -> SourceBundle:
    return SourceBundle(
        source=FINNHUB_SOURCE,
        items=tuple(items),
        statuses=tuple(statuses) or (available_status(FINNHUB_SOURCE),),
        warnings=tuple(warnings),
        requests=tuple(requests),
    )


def _merge(bundles: tuple[SourceBundle, ...]) -> SourceBundle:
    return SourceBundle(
        source=FINNHUB_SOURCE,
        items=tuple(item for bundle in bundles for item in bundle.items),
        statuses=tuple(status for bundle in bundles for status in bundle.statuses),
        warnings=tuple(warning for bundle in bundles for warning in bundle.warnings),
        requests=tuple(request for bundle in bundles for request in bundle.requests),
    )
