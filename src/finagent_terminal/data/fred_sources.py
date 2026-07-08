from __future__ import annotations

from datetime import date
from typing import ClassVar, Final, final

from pydantic import BaseModel, ConfigDict, ValidationError

from finagent_terminal.data.source_http import (
    FRED_API_KEY_ENV,
    PublicSourceCall,
    SourceHttpClient,
    available_status,
    warning_bundle,
)
from finagent_terminal.data.source_models import (
    SourceBundle,
    SourceItem,
    SourceRequest,
    SourceStatus,
    SourceStatusValue,
)

FRED_SOURCE: Final = "FRED"
FRED_OBSERVATIONS_URL: Final = "https://api.stlouisfed.org/fred/series/observations"
FRED_THROTTLE: Final = "FRED public API: API-key quota applies"


class FredObservation(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    date: str
    value: str


class FredResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    observations: tuple[FredObservation, ...] = ()


@final
class FredSourceClient:
    __slots__ = ("_http",)

    def __init__(self, http: SourceHttpClient) -> None:  # noqa: D107
        self._http = http

    def observations(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> SourceBundle:
        series_key = series_id.strip().upper()
        if series_key == "":
            return warning_bundle(FRED_SOURCE, "missing FRED series id")
        if not _valid_optional_date(observation_start):
            return warning_bundle(FRED_SOURCE, "invalid FRED observation_start")
        if not _valid_optional_date(observation_end):
            return warning_bundle(FRED_SOURCE, "invalid FRED observation_end")

        api_key = self._http.env.get(FRED_API_KEY_ENV, "").strip()
        if api_key == "":
            return SourceBundle(
                source=FRED_SOURCE,
                statuses=(
                    SourceStatus(
                        source=FRED_SOURCE,
                        status=SourceStatusValue.UNAVAILABLE,
                        message=f"missing {FRED_API_KEY_ENV}",
                        throttle=FRED_THROTTLE,
                    ),
                ),
                warnings=(f"missing {FRED_API_KEY_ENV}",),
            )

        fetched = self._http.get(
            PublicSourceCall(
                source=FRED_SOURCE,
                url=FRED_OBSERVATIONS_URL,
                params=_without_none(
                    {
                        "series_id": series_key,
                        "api_key": api_key,
                        "file_type": "json",
                        "observation_start": observation_start,
                        "observation_end": observation_end,
                    },
                ),
                headers={},
                throttle=FRED_THROTTLE,
            ),
        )
        if isinstance(fetched, SourceBundle):
            return fetched
        return self._bundle(series_key, fetched.content, fetched.request)

    def _bundle(
        self,
        series: str,
        content: bytes,
        request: SourceRequest,
    ) -> SourceBundle:
        try:
            data = FredResponse.model_validate_json(content)
        except ValidationError:
            return warning_bundle(
                FRED_SOURCE,
                "FRED observations payload invalid",
                request,
            )
        summary = next(
            (
                f"{observation.date}: {observation.value}"
                for observation in data.observations
                if observation.value != "."
            ),
            "no observations",
        )
        return SourceBundle(
            source=FRED_SOURCE,
            items=(
                SourceItem(
                    source=FRED_SOURCE,
                    title=f"{series} observations",
                    summary=summary,
                ),
            ),
            statuses=(available_status(FRED_SOURCE, request),),
            requests=(request,),
        )


def _valid_optional_date(value: str | None) -> bool:
    if value is None or value == "":
        return True
    try:
        _ = date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _without_none(values: dict[str, str | None]) -> dict[str, str]:
    return {key: value for key, value in values.items() if value is not None}
