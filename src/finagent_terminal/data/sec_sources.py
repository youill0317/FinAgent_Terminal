from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from finagent_terminal.data.source_http import (
    SEC_USER_AGENT_ENV,
    PublicSourceCall,
    SourceHttpClient,
    available_status,
    join_requests,
    warning_bundle,
    with_optional_request,
)
from finagent_terminal.data.source_models import SourceBundle, SourceItem, SourceRequest

SEC_SOURCE: Final = "SEC EDGAR"
DEFAULT_SEC_USER_AGENT: Final = "FinAgentTerminal/0.1 public-source-client"
SEC_TICKERS_URL: Final = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL: Final = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL: Final = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_THROTTLE: Final = "SEC fair access: identify User-Agent; max 10 requests/second"


class SecTickerRow(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    ticker: str
    cik_str: str | int


class SecFactsResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    entity_name: str | None = Field(default=None, alias="entityName")
    facts: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SecCikLookup:
    cik: str | None
    request: SourceRequest | None
    warning: str | None = None


SEC_TICKERS_ADAPTER: Final[TypeAdapter[dict[str, SecTickerRow]]] = TypeAdapter(
    dict[str, SecTickerRow],
)


@final
class SecSourceClient:
    __slots__ = ("_cik_cache", "_http")

    def __init__(self, http: SourceHttpClient) -> None:  # noqa: D107
        self._http = http
        self._cik_cache: dict[str, str | None] = {}

    def clear(self) -> None:
        self._cik_cache.clear()

    def cache_size(self) -> int:
        return len(self._cik_cache)

    def companyfacts(self, ticker: str) -> SourceBundle:
        ticker_key = _normalized_ticker(ticker)
        if ticker_key is None:
            return warning_bundle(SEC_SOURCE, "missing SEC ticker")
        lookup = self._lookup_cik(ticker_key)
        if isinstance(lookup, SourceBundle):
            return lookup
        if lookup.cik is None:
            return warning_bundle(
                SEC_SOURCE,
                lookup.warning or f"missing SEC CIK for {ticker_key}",
                lookup.request,
            )
        fetched = self._http.get(
            PublicSourceCall(
                source=SEC_SOURCE,
                url=SEC_COMPANYFACTS_URL.format(cik=lookup.cik),
                params={},
                headers=self._headers(),
                throttle=SEC_THROTTLE,
            ),
        )
        if isinstance(fetched, SourceBundle):
            return with_optional_request(fetched, lookup.request)
        return self._facts_bundle(
            ticker_key,
            fetched.content,
            fetched.request,
            lookup.request,
        )

    def submissions(self, ticker: str) -> SourceBundle:
        ticker_key = _normalized_ticker(ticker)
        if ticker_key is None:
            return warning_bundle(SEC_SOURCE, "missing SEC ticker")
        lookup = self._lookup_cik(ticker_key)
        if isinstance(lookup, SourceBundle):
            return lookup
        if lookup.cik is None:
            return warning_bundle(
                SEC_SOURCE,
                lookup.warning or f"missing SEC CIK for {ticker_key}",
                lookup.request,
            )
        fetched = self._http.get(
            PublicSourceCall(
                source=SEC_SOURCE,
                url=SEC_SUBMISSIONS_URL.format(cik=lookup.cik),
                params={},
                headers=self._headers(),
                throttle=SEC_THROTTLE,
            ),
        )
        if isinstance(fetched, SourceBundle):
            return with_optional_request(fetched, lookup.request)
        return SourceBundle(
            source=SEC_SOURCE,
            items=(
                SourceItem(
                    source=SEC_SOURCE,
                    title=f"{ticker_key} submissions",
                ),
            ),
            statuses=(available_status(SEC_SOURCE, fetched.request),),
            requests=join_requests(lookup.request, fetched.request),
        )

    def _lookup_cik(self, ticker: str) -> SecCikLookup | SourceBundle:
        if ticker in self._cik_cache:
            return SecCikLookup(cik=self._cik_cache[ticker], request=None)
        fetched = self._http.get(
            PublicSourceCall(
                source=SEC_SOURCE,
                url=SEC_TICKERS_URL,
                params={},
                headers=self._headers(),
                throttle=SEC_THROTTLE,
            ),
        )
        if isinstance(fetched, SourceBundle):
            return fetched
        try:
            rows = SEC_TICKERS_ADAPTER.validate_json(fetched.content)
        except ValidationError:
            return warning_bundle(
                SEC_SOURCE,
                "SEC ticker lookup payload invalid",
                fetched.request,
            )
        for row in rows.values():
            if row.ticker.strip().upper() == ticker:
                cik = str(row.cik_str).strip().zfill(10)
                self._cik_cache[ticker] = cik
                return SecCikLookup(cik=cik, request=fetched.request)
        self._cik_cache[ticker] = None
        return SecCikLookup(
            cik=None,
            request=fetched.request,
            warning=f"missing SEC CIK for {ticker}",
        )

    def _headers(self) -> dict[str, str]:
        user_agent = self._http.env.get(SEC_USER_AGENT_ENV, "").strip()
        if user_agent == "":
            user_agent = DEFAULT_SEC_USER_AGENT
        return {"User-Agent": user_agent}

    def _facts_bundle(
        self,
        ticker: str,
        content: bytes,
        request: SourceRequest,
        lookup_request: SourceRequest | None,
    ) -> SourceBundle:
        try:
            facts = SecFactsResponse.model_validate_json(content)
        except ValidationError:
            return warning_bundle(
                SEC_SOURCE,
                "SEC companyfacts payload invalid",
                request,
            )
        entity_name = facts.entity_name or ticker
        fact_names = ", ".join(facts.facts) if facts.facts else "none"
        return SourceBundle(
            source=SEC_SOURCE,
            items=(
                SourceItem(
                    source=SEC_SOURCE,
                    title=f"{entity_name} company facts",
                    summary=f"facts: {fact_names}",
                ),
            ),
            statuses=(available_status(SEC_SOURCE, request),),
            requests=join_requests(lookup_request, request),
        )


def _normalized_ticker(ticker: str) -> str | None:
    normalized = ticker.strip().upper()
    if normalized == "":
        return None
    return normalized
