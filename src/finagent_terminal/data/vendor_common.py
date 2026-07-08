from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final
from urllib.parse import quote, quote_plus

import httpx
from pydantic import TypeAdapter, ValidationError

from finagent_terminal.data.source_models import (
    SourceBundle,
    SourceRequest,
    SourceStatus,
    SourceStatusValue,
)
from finagent_terminal.symbols import InvalidSymbolError, normalize_symbol

if TYPE_CHECKING:
    from collections.abc import Mapping

REDACTED: Final = "<redacted>"
HTTP_ERROR_MIN_STATUS: Final = 400


@dataclass(frozen=True, slots=True)
class VendorCall[T]:
    source: str
    url: str
    params: Mapping[str, str]
    headers: Mapping[str, str]
    adapter: TypeAdapter[T]
    secret: str


@dataclass(frozen=True, slots=True)
class VendorResponse[T]:
    data: T
    request: SourceRequest


def get_vendor[T](
    client: httpx.Client,
    call: VendorCall[T],
) -> VendorResponse[T] | SourceBundle:
    try:
        response = client.get(call.url, params=call.params, headers=call.headers)
    except httpx.HTTPError as exc:
        return warning_bundle(
            call.source,
            f"{call.source} request failed: {redact(str(exc), call.secret)}",
        )

    request = SourceRequest(
        source=call.source,
        method=response.request.method,
        url=redact(str(response.request.url), call.secret),
        status_code=response.status_code,
    )
    if response.status_code in {401, 403}:
        return status_bundle(
            call.source,
            SourceStatusValue.UNAVAILABLE,
            f"{call.source} HTTP {response.status_code} unauthorized",
            request,
        )
    if response.status_code >= HTTP_ERROR_MIN_STATUS:
        return warning_bundle(
            call.source,
            f"{call.source} HTTP {response.status_code}",
            request,
        )
    try:
        return VendorResponse(
            data=call.adapter.validate_json(response.content),
            request=request,
        )
    except ValidationError as exc:
        return warning_bundle(
            call.source,
            f"{call.source} malformed payload: {redact(str(exc), call.secret)}",
            request,
        )


def collect_vendor[T](
    result: VendorResponse[T] | SourceBundle,
    statuses: list[SourceStatus],
    warnings: list[str],
    requests: list[SourceRequest],
) -> None:
    if isinstance(result, SourceBundle):
        statuses.extend(result.statuses)
        warnings.extend(result.warnings)
        requests.extend(result.requests)
        return
    requests.append(result.request)


def normalized_symbol(symbol: str, source: str) -> str | SourceBundle:
    try:
        return normalize_symbol(symbol)
    except InvalidSymbolError:
        return warning_bundle(source, f"invalid {source} symbol")


def env_value(name: str, env: Mapping[str, str] | None) -> str | None:
    value = (os.environ if env is None else env).get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def default_http_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        follow_redirects=True,
    )


def missing_key_bundle(source: str, env_name: str) -> SourceBundle:
    return status_bundle(
        source,
        SourceStatusValue.UNAVAILABLE,
        f"{env_name} missing; source unavailable",
    )


def warning_bundle(
    source: str,
    message: str,
    request: SourceRequest | None = None,
) -> SourceBundle:
    return status_bundle(source, SourceStatusValue.WARNING, message, request)


def status_bundle(
    source: str,
    status: SourceStatusValue,
    message: str,
    request: SourceRequest | None = None,
) -> SourceBundle:
    return SourceBundle(
        source=source,
        statuses=(SourceStatus(source=source, status=status, message=message),),
        warnings=(message,) if status is SourceStatusValue.WARNING else (),
        requests=() if request is None else (request,),
    )


def available_status(source: str) -> SourceStatus:
    return SourceStatus(
        source=source,
        status=SourceStatusValue.AVAILABLE,
        message="HTTP 200",
    )


def summary(*parts: str | None) -> str | None:
    text = "; ".join(part for part in parts if part is not None and part != "")
    return text or None


def number(label: str, value: float | None) -> str | None:
    if value is None:
        return None
    return f"{label}={value:g}"


def parse_datetime(value: str | int | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, int):
        return datetime.fromtimestamp(value, UTC)
    cleaned = value.strip()
    if cleaned == "":
        return None
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def redact(text: str, secret: str) -> str:
    if secret == "":
        return text
    redacted = text
    for variant in (secret, quote(secret, safe=""), quote_plus(secret, safe="")):
        redacted = redacted.replace(variant, REDACTED)
    return redacted
