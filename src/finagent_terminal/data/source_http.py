from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, final
from urllib.parse import quote, quote_plus

import httpx

from finagent_terminal.data.source_models import (
    SourceBundle,
    SourceRequest,
    SourceStatus,
    SourceStatusValue,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

FRED_API_KEY_ENV: Final = "FRED_API_KEY"
SEC_USER_AGENT_ENV: Final = "FINAGENT_SEC_USER_AGENT"
REDACTED: Final = "<redacted>"
HTTP_ERROR_MIN_STATUS: Final = 400


@dataclass(frozen=True, slots=True)
class PublicSourceCall:
    source: str
    url: str
    params: Mapping[str, str | int]
    headers: Mapping[str, str]
    throttle: str


@dataclass(frozen=True, slots=True)
class PublicSourceResponse:
    content: bytes
    request: SourceRequest


@final
class SourceHttpClient:
    __slots__ = ("_http_client", "_owns_http_client", "env")

    def __init__(  # noqa: D107
        self,
        *,
        env: Mapping[str, str] | None,
        http_client: httpx.Client | None,
    ) -> None:
        self.env = os.environ if env is None else env
        if http_client is None:
            self._owns_http_client = True
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                follow_redirects=True,
            )
            return
        self._owns_http_client = False
        self._http_client = http_client

    def __enter__(self) -> SourceHttpClient:
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
        if self._owns_http_client:
            self._http_client.close()

    def get(self, call: PublicSourceCall) -> PublicSourceResponse | SourceBundle:
        try:
            response = self._http_client.get(
                call.url,
                params={key: str(value) for key, value in call.params.items()},
                headers=call.headers,
            )
        except httpx.HTTPError as exc:
            message = f"{call.source} request failed: {exc.__class__.__name__}"
            return SourceBundle(
                source=call.source,
                statuses=(
                    SourceStatus(
                        source=call.source,
                        status=SourceStatusValue.WARNING,
                        message=message,
                        throttle=call.throttle,
                    ),
                ),
                warnings=(message,),
            )

        retry_after_header = None
        for header_name, header_value in response.headers.multi_items():
            if header_name.lower() == "retry-after":
                retry_after_header = header_value
                break
        request = SourceRequest(
            source=call.source,
            method=response.request.method,
            url=self.redacted_url(response.request.url),
            status_code=response.status_code,
            throttle=call.throttle,
            retry_after=_safe_retry_after(retry_after_header),
        )
        if response.status_code >= HTTP_ERROR_MIN_STATUS:
            return warning_bundle(
                call.source,
                f"{call.source} HTTP {response.status_code}",
                request,
            )
        return PublicSourceResponse(content=response.content, request=request)

    def redacted_url(self, url: httpx.URL) -> str:
        return self.redacted_text(str(url))

    def redacted_text(self, text: str) -> str:
        redacted = text
        for secret in (
            self.env.get(FRED_API_KEY_ENV, ""),
            self.env.get(SEC_USER_AGENT_ENV, ""),
        ):
            value = secret.strip()
            for variant in _secret_variants(value):
                redacted = redacted.replace(variant, REDACTED)
        return redacted


def warning_bundle(
    source: str,
    message: str,
    request: SourceRequest | None = None,
) -> SourceBundle:
    return SourceBundle(
        source=source,
        statuses=(
            SourceStatus(
                source=source,
                status=SourceStatusValue.WARNING,
                message=message,
                throttle=None if request is None else request.throttle,
                retry_after=None if request is None else request.retry_after,
            ),
        ),
        warnings=(message,),
        requests=() if request is None else (request,),
    )


def available_status(source: str, request: SourceRequest | None = None) -> SourceStatus:
    return SourceStatus(
        source=source,
        status=SourceStatusValue.AVAILABLE,
        message="HTTP 200",
        throttle=None if request is None else request.throttle,
        retry_after=None if request is None else request.retry_after,
    )


def join_requests(
    first: SourceRequest | None,
    second: SourceRequest,
) -> tuple[SourceRequest, ...]:
    if first is None:
        return (second,)
    return (first, second)


def with_optional_request(
    bundle: SourceBundle,
    request: SourceRequest | None,
) -> SourceBundle:
    if request is None:
        return bundle
    return bundle.model_copy(update={"requests": (request, *bundle.requests)})


def _secret_variants(secret: str) -> tuple[str, ...]:
    if secret == "":
        return ()
    return (secret, quote(secret, safe=""), quote_plus(secret, safe=""))


def _safe_retry_after(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return stripped
