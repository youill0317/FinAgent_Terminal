from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, final

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from finagent_terminal.data.source_http import (
    PublicSourceCall,
    SourceHttpClient,
    available_status,
    warning_bundle,
)
from finagent_terminal.data.source_models import SourceBundle, SourceItem

if TYPE_CHECKING:
    from collections.abc import Mapping

TREASURY_SOURCE: Final = "Treasury Fiscal Data"
TREASURY_BASE_URL: Final = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
)
TREASURY_THROTTLE: Final = "Treasury Fiscal Data: bounded page size"


class TreasuryResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    data: tuple[dict[str, JsonValue], ...] = ()


@final
class TreasurySourceClient:
    __slots__ = ("_http",)

    def __init__(self, http: SourceHttpClient) -> None:  # noqa: D107
        self._http = http

    def query(self, endpoint: str, *, page_size: int = 100) -> SourceBundle:
        normalized_endpoint = endpoint.strip().strip("/")
        if normalized_endpoint == "":
            return warning_bundle(TREASURY_SOURCE, "missing Treasury endpoint")
        if page_size <= 0:
            return warning_bundle(TREASURY_SOURCE, "invalid Treasury page size")

        fetched = self._http.get(
            PublicSourceCall(
                source=TREASURY_SOURCE,
                url=f"{TREASURY_BASE_URL}/{normalized_endpoint}",
                params={"format": "json", "page[size]": page_size},
                headers={},
                throttle=TREASURY_THROTTLE,
            ),
        )
        if isinstance(fetched, SourceBundle):
            return fetched
        try:
            data = TreasuryResponse.model_validate_json(fetched.content)
        except ValidationError:
            return warning_bundle(
                TREASURY_SOURCE,
                "Treasury payload invalid",
                fetched.request,
            )
        return SourceBundle(
            source=TREASURY_SOURCE,
            items=tuple(_item(row) for row in data.data),
            statuses=(available_status(TREASURY_SOURCE, fetched.request),),
            requests=(fetched.request,),
        )


def _item(row: Mapping[str, JsonValue]) -> SourceItem:
    title = _row_text(row, "country") or _row_text(row, "record_date")
    summary = "; ".join(
        f"{key}={_json_text(value)}"
        for key, value in row.items()
        if key != "country" and _json_text(value) != ""
    )
    return SourceItem(
        source=TREASURY_SOURCE,
        title=title or "Fiscal record",
        summary=summary or None,
    )


def _row_text(row: Mapping[str, JsonValue], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = _json_text(value)
    if text == "":
        return None
    return text


def _json_text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float):
        return str(value)
    return ""
