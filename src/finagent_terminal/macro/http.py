from __future__ import annotations

import asyncio
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final

import httpx

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent_terminal.macro.models import SourceId

FRED_KEY_ENV: Final = "FRED_API_KEY"
ECOS_KEY_ENV: Final = "ECOS_API_KEY"
REQUEST_TIMEOUT_SECONDS: Final = 20.0
USER_AGENT: Final = "finagent-macro/0.1 (personal dashboard)"

# Concurrent requests allowed per source. OECD allows 60 downloads an hour,
# so it gets one lane; FRED allows about 120 a minute.
SOURCE_LANES: Final[Mapping[SourceId, int]] = MappingProxyType(
    {
        "FRED": 6,
        "TREASURY": 2,
        "NYFED": 2,
        "ECOS": 4,
        "ECB": 2,
        "BIS": 2,
        "OECD": 1,
    }
)


class SourceError(Exception):
    """A source could not deliver a usable series. The message is user-facing."""


class MissingKeyError(SourceError):
    """The source needs an API key that is not configured."""


def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


@final
class FetchContext:
    """One refresh pass: shared client, keys, and per-request de-duplication.

    Responses live only for the duration of the pass. Nothing is written to disk.
    """

    __slots__ = ("_client", "_env", "_inflight", "_lanes")

    def __init__(self, client: httpx.AsyncClient, env: Mapping[str, str]) -> None:  # noqa: D107
        self._client = client
        self._env = env
        self._lanes = {
            source: asyncio.Semaphore(lanes) for source, lanes in SOURCE_LANES.items()
        }
        self._inflight: dict[
            tuple[str, tuple[tuple[str, str], ...]], asyncio.Task[str]
        ] = {}

    def key(self, env_name: str) -> str:
        value = self._env.get(env_name, "").strip()
        if value == "":
            raise MissingKeyError(f"{env_name} 환경변수가 없습니다")
        return value

    def has_key(self, env_name: str) -> bool:
        return self._env.get(env_name, "").strip() != ""

    async def text(
        self,
        source: SourceId,
        url: str,
        params: Mapping[str, str] | None = None,
    ) -> str:
        """GET a URL once per pass; concurrent callers share the same request."""
        query = tuple(sorted((params or {}).items()))
        cache_key = (url, query)
        task = self._inflight.get(cache_key)
        if task is None:
            task = asyncio.ensure_future(self._get(source, url, dict(query)))
            self._inflight[cache_key] = task
        return await asyncio.shield(task)

    async def _get(self, source: SourceId, url: str, params: dict[str, str]) -> str:
        async with self._lanes[source]:
            try:
                response = await self._client.get(url, params=params)
            except httpx.TimeoutException as error:
                raise SourceError(f"{source} 응답 시간 초과") from error
            except httpx.HTTPError as error:
                # The exception text can contain the URL and with it an API key.
                raise SourceError(
                    f"{source} 연결 실패 ({type(error).__name__})"
                ) from error
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise SourceError(f"{source} HTTP {response.status_code}")
        return response.text
