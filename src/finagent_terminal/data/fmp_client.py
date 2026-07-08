from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, final

from finagent_terminal.data.fmp_items import (
    balance_item,
    cash_item,
    earnings_item,
    income_item,
    news_items,
    profile_items,
)
from finagent_terminal.data.fmp_payloads import (
    FMP_BALANCE_ADAPTER,
    FMP_CASH_FLOW_ADAPTER,
    FMP_EARNINGS_CALENDAR_ADAPTER,
    FMP_INCOME_ADAPTER,
    FMP_NEWS_ADAPTER,
    FMP_PROFILE_ADAPTER,
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

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx

FMP_SOURCE: Final = "FMP"
FMP_API_KEY_ENV: Final = "FMP_API_KEY"
FMP_PROFILE_URL: Final = "https://financialmodelingprep.com/stable/profile"
FMP_INCOME_URL: Final = "https://financialmodelingprep.com/stable/income-statement"
FMP_BALANCE_URL: Final = (
    "https://financialmodelingprep.com/stable/balance-sheet-statement"
)
FMP_CASH_URL: Final = "https://financialmodelingprep.com/stable/cash-flow-statement"
FMP_EARNINGS_CALENDAR_URL: Final = (
    "https://financialmodelingprep.com/stable/earnings-calendar"
)
FMP_NEWS_URL: Final = "https://financialmodelingprep.com/stable/news/stock"


@final
@dataclass(frozen=True, slots=True)
class FmpClient:
    _api_key: str | None
    _http_client: httpx.Client | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> FmpClient:
        return cls(env_value(FMP_API_KEY_ENV, env), http_client)

    def fetch_symbol(self, symbol: str) -> SourceBundle:
        return self._fetch(symbol, self._fetch_symbol)

    def fetch_news(self, symbol: str) -> SourceBundle:
        return self._fetch(symbol, self._fetch_news)

    def fetch_fundamentals(self, symbol: str) -> SourceBundle:
        return self._fetch(symbol, self._fetch_fundamentals)

    def fetch_estimates(self, symbol: str) -> SourceBundle:
        return self._fetch(symbol, self._fetch_estimates)

    def _fetch(
        self,
        symbol: str,
        fetcher: Callable[[httpx.Client, str], SourceBundle],
    ) -> SourceBundle:
        symbol_key = normalized_symbol(symbol, FMP_SOURCE)
        if isinstance(symbol_key, SourceBundle):
            return symbol_key
        if self._api_key is None:
            return missing_key_bundle(FMP_SOURCE, FMP_API_KEY_ENV)
        if self._http_client is None:
            with default_http_client() as client:
                return fetcher(client, symbol_key)
        return fetcher(self._http_client, symbol_key)

    def _headers(self) -> dict[str, str]:
        return {"apikey": self._api_key or ""}

    def _fetch_symbol(self, client: httpx.Client, symbol: str) -> SourceBundle:
        return _merge(
            (
                self._fetch_fundamentals(client, symbol),
                self._fetch_estimates(client, symbol),
                self._fetch_news(client, symbol),
            ),
        )

    def _fetch_fundamentals(self, client: httpx.Client, symbol: str) -> SourceBundle:
        headers = self._headers()
        items: list[SourceItem] = []
        statuses: list[SourceStatus] = []
        warnings: list[str] = []
        requests: list[SourceRequest] = []

        profile = get_vendor(
            client,
            VendorCall(
                FMP_SOURCE,
                FMP_PROFILE_URL,
                {"symbol": symbol},
                headers,
                FMP_PROFILE_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(profile, statuses, warnings, requests)
        if isinstance(profile, VendorResponse):
            items.extend(profile_items(profile.data, FMP_SOURCE))

        income = get_vendor(
            client,
            VendorCall(
                FMP_SOURCE,
                FMP_INCOME_URL,
                {"symbol": symbol},
                headers,
                FMP_INCOME_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(income, statuses, warnings, requests)
        if isinstance(income, VendorResponse) and len(income.data) > 0:
            items.append(income_item(income.data[0], symbol, FMP_SOURCE))

        balance = get_vendor(
            client,
            VendorCall(
                FMP_SOURCE,
                FMP_BALANCE_URL,
                {"symbol": symbol},
                headers,
                FMP_BALANCE_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(balance, statuses, warnings, requests)
        if isinstance(balance, VendorResponse) and len(balance.data) > 0:
            items.append(balance_item(balance.data[0], symbol, FMP_SOURCE))

        cash_flow = get_vendor(
            client,
            VendorCall(
                FMP_SOURCE,
                FMP_CASH_URL,
                {"symbol": symbol},
                headers,
                FMP_CASH_FLOW_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(cash_flow, statuses, warnings, requests)
        if isinstance(cash_flow, VendorResponse) and len(cash_flow.data) > 0:
            items.append(cash_item(cash_flow.data[0], symbol, FMP_SOURCE))

        return _bundle(items, statuses, warnings, requests)

    def _fetch_estimates(self, client: httpx.Client, symbol: str) -> SourceBundle:
        headers = self._headers()
        items: list[SourceItem] = []
        statuses: list[SourceStatus] = []
        warnings: list[str] = []
        requests: list[SourceRequest] = []

        earnings = get_vendor(
            client,
            VendorCall(
                FMP_SOURCE,
                FMP_EARNINGS_CALENDAR_URL,
                {"symbol": symbol},
                headers,
                FMP_EARNINGS_CALENDAR_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(earnings, statuses, warnings, requests)
        if isinstance(earnings, VendorResponse) and len(earnings.data) > 0:
            items.append(earnings_item(earnings.data[0], symbol, FMP_SOURCE))

        return _bundle(items, statuses, warnings, requests)

    def _fetch_news(self, client: httpx.Client, symbol: str) -> SourceBundle:
        headers = self._headers()
        items: list[SourceItem] = []
        statuses: list[SourceStatus] = []
        warnings: list[str] = []
        requests: list[SourceRequest] = []

        news = get_vendor(
            client,
            VendorCall(
                FMP_SOURCE,
                FMP_NEWS_URL,
                {"symbols": symbol},
                headers,
                FMP_NEWS_ADAPTER,
                self._api_key or "",
            ),
        )
        collect_vendor(news, statuses, warnings, requests)
        if isinstance(news, VendorResponse):
            items.extend(news_items(news.data, FMP_SOURCE))

        return _bundle(items, statuses, warnings, requests)


def _bundle(
    items: list[SourceItem],
    statuses: list[SourceStatus],
    warnings: list[str],
    requests: list[SourceRequest],
) -> SourceBundle:
    return SourceBundle(
        source=FMP_SOURCE,
        items=tuple(items),
        statuses=tuple(statuses) or (available_status(FMP_SOURCE),),
        warnings=tuple(warnings),
        requests=tuple(requests),
    )


def _merge(bundles: tuple[SourceBundle, ...]) -> SourceBundle:
    return SourceBundle(
        source=FMP_SOURCE,
        items=tuple(item for bundle in bundles for item in bundle.items),
        statuses=tuple(status for bundle in bundles for status in bundle.statuses),
        warnings=tuple(warning for bundle in bundles for warning in bundle.warnings),
        requests=tuple(request for bundle in bundles for request in bundle.requests),
    )
