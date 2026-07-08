from __future__ import annotations

from typing import TYPE_CHECKING, final

from finagent_terminal.data.fred_sources import FredSourceClient
from finagent_terminal.data.gdelt_sources import GdeltSourceClient
from finagent_terminal.data.sec_sources import SecSourceClient
from finagent_terminal.data.source_http import SourceHttpClient
from finagent_terminal.data.treasury_sources import TreasurySourceClient

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    import httpx

    from finagent_terminal.data.source_models import SourceBundle


@final
class PublicSourceClient:
    __slots__ = ("_fred", "_gdelt", "_http", "_sec", "_treasury")

    def __init__(  # noqa: D107
        self,
        *,
        env: Mapping[str, str] | None,
        http_client: httpx.Client | None,
    ) -> None:
        http = SourceHttpClient(env=env, http_client=http_client)
        self._http = http
        self._sec = SecSourceClient(self._http)
        self._fred = FredSourceClient(self._http)
        self._treasury = TreasurySourceClient(self._http)
        self._gdelt = GdeltSourceClient(self._http)

    def __enter__(self) -> PublicSourceClient:
        """Return this client for context-manager cleanup."""
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
        self._http.close()

    def clear(self) -> None:
        self._sec.clear()

    def cache_size(self) -> int:
        return self._sec.cache_size()

    def sec_companyfacts(self, ticker: str) -> SourceBundle:
        return self._sec.companyfacts(ticker)

    def sec_submissions(self, ticker: str) -> SourceBundle:
        return self._sec.submissions(ticker)

    def fred_series_observations(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> SourceBundle:
        return self._fred.observations(
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
        return self._treasury.query(endpoint, page_size=page_size)

    def gdelt_articles(self, query: str) -> SourceBundle:
        return self._gdelt.articles(query)

    def gdelt_events(self, query: str) -> SourceBundle:
        return self._gdelt.events(query)
