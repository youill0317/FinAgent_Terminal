from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar, Final, final

from pydantic import BaseModel, ConfigDict, ValidationError

from finagent_terminal.data.source_http import (
    PublicSourceCall,
    SourceHttpClient,
    available_status,
    warning_bundle,
)
from finagent_terminal.data.source_models import SourceBundle, SourceItem

GDELT_SOURCE: Final = "GDELT"
GDELT_DOC_URL: Final = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_THROTTLE: Final = "GDELT DOC 2.0 API: bounded maxrecords"


class GdeltArticle(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    title: str
    url: str | None = None
    seendate: str | None = None
    domain: str | None = None


class GdeltDocResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    articles: tuple[GdeltArticle, ...] = ()


@final
class GdeltSourceClient:
    __slots__ = ("_http",)

    def __init__(self, http: SourceHttpClient) -> None:  # noqa: D107
        self._http = http

    def articles(self, query: str) -> SourceBundle:
        query_text = query.strip()
        if query_text == "":
            return warning_bundle(GDELT_SOURCE, "missing GDELT query")
        fetched = self._http.get(
            PublicSourceCall(
                source=GDELT_SOURCE,
                url=GDELT_DOC_URL,
                params={
                    "query": query_text,
                    "mode": "ArtList",
                    "format": "json",
                    "maxrecords": 10,
                },
                headers={},
                throttle=GDELT_THROTTLE,
            ),
        )
        if isinstance(fetched, SourceBundle):
            return fetched
        try:
            data = GdeltDocResponse.model_validate_json(fetched.content)
        except ValidationError:
            return warning_bundle(
                GDELT_SOURCE,
                "GDELT article payload invalid",
                fetched.request,
            )
        return SourceBundle(
            source=GDELT_SOURCE,
            items=tuple(
                SourceItem(
                    source=GDELT_SOURCE,
                    title=article.title,
                    summary=article.domain,
                    url=article.url,
                    published_at=_parse_gdelt_datetime(article.seendate),
                )
                for article in data.articles
            ),
            statuses=(available_status(GDELT_SOURCE, fetched.request),),
            requests=(fetched.request,),
        )

    def events(self, query: str) -> SourceBundle:
        query_text = query.strip()
        if query_text == "":
            return warning_bundle(GDELT_SOURCE, "missing GDELT query")
        fetched = self._http.get(
            PublicSourceCall(
                source=GDELT_SOURCE,
                url=GDELT_DOC_URL,
                params={
                    "query": query_text,
                    "mode": "ArtList",
                    "format": "json",
                    "maxrecords": 10,
                },
                headers={},
                throttle=GDELT_THROTTLE,
            ),
        )
        if isinstance(fetched, SourceBundle):
            return fetched
        try:
            data = GdeltDocResponse.model_validate_json(fetched.content)
        except ValidationError:
            return warning_bundle(
                GDELT_SOURCE,
                "GDELT event-context payload invalid",
                fetched.request,
            )
        return SourceBundle(
            source=GDELT_SOURCE,
            items=tuple(
                SourceItem(
                    source=GDELT_SOURCE,
                    title=f"Event context: {article.title}",
                    summary=article.domain,
                    url=article.url,
                    published_at=_parse_gdelt_datetime(article.seendate),
                )
                for article in data.articles
            ),
            statuses=(available_status(GDELT_SOURCE, fetched.request),),
            requests=(fetched.request,),
        )


def _parse_gdelt_datetime(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None
