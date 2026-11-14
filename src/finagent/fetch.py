"""비동기 HTTP 데이터 수집 컨텍스트 및 중복 요청 병합 제어."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING, Final, cast, final

import httpx

from finagent import limits
from finagent.limits import BACKGROUND_WAIT, LIMITS
from finagent.limits import (
    SourceError as SourceError,  # noqa: PLC0414  # 하위 호환 명시적 재노출
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from finagent.models import SourceId

FRED_KEY_ENV: Final = "FRED_API_KEY"
ECOS_KEY_ENV: Final = "ECOS_API_KEY"
REQUEST_TIMEOUT_SECONDS: Final = 20.0
USER_AGENT: Final = "finagent/0.2 (local dashboard)"
RETRY_DELAY_SECONDS: Final = 1.0
HTTP_CLIENT_ERROR: Final = 400
HTTP_SERVER_ERROR: Final = 500


class MissingKeyError(SourceError): ...


def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


@final
class SharedWork:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: dict[str, Future[object]] = {}

    async def run[T](self, key: str, make: Callable[[], Awaitable[T]]) -> T:
        with self._lock:
            running = self._running.get(key)
            mine: Future[object] = Future()
            if running is None:
                self._running[key] = mine
        if running is not None:
            return cast("T", await asyncio.wrap_future(running))
        try:
            result = await make()
        except BaseException as error:
            mine.set_exception(error)
            raise
        else:
            mine.set_result(result)
            return result
        finally:
            with self._lock:
                del self._running[key]


@final
class FetchContext:
    __slots__ = ("_client", "_env", "_inflight", "_retry_delay", "_shared", "max_wait")

    def __init__(
        self,
        client: httpx.AsyncClient,
        env: Mapping[str, str],
        *,
        retry_delay: float = RETRY_DELAY_SECONDS,
        max_wait: float = BACKGROUND_WAIT,
        shared: SharedWork | None = None,
    ) -> None:
        self._shared = shared or SharedWork()
        self._client = client
        self._env = env
        self._retry_delay = retry_delay
        self.max_wait = max_wait
        self._inflight: dict[
            tuple[str, tuple[tuple[str, str], ...]], asyncio.Task[str]
        ] = {}

    async def shared[T](self, key: str, make: Callable[[], Awaitable[T]]) -> T:
        return await self._shared.run(key, make)

    def key(self, env_name: str) -> str:
        value = self._env.get(env_name, "").strip()
        if value == "":
            raise MissingKeyError(f"{env_name} 환경변수 미설정")
        return value

    def has_key(self, env_name: str) -> bool:
        return self._env.get(env_name, "").strip() != ""

    async def text(
        self,
        source: SourceId,
        url: str,
        params: Mapping[str, str] | None = None,
    ) -> str:
        query = tuple(sorted((params or {}).items()))
        cache_key = (url, query)
        task = self._inflight.get(cache_key)
        if task is None:
            task = asyncio.ensure_future(self._get(source, url, dict(query)))
            self._inflight[cache_key] = task
        return await asyncio.shield(task)

    async def _get(self, source: SourceId, url: str, params: dict[str, str]) -> str:
        limiter = limits.current()
        left = LIMITS[source].retries
        async with limiter.aslot(source, self.max_wait):
            while True:
                await limiter.aacquire(source, max_wait=self.max_wait)
                try:
                    return await self._attempt(source, url, params)
                except _RetryableError as error:
                    if left == 0:
                        raise SourceError(str(error)) from error
                    if not error.limited:
                        await asyncio.sleep(self._retry_delay)
                left -= 1

    async def _attempt(self, source: SourceId, url: str, params: dict[str, str]) -> str:
        try:
            response = await self._client.get(url, params=params)
        except httpx.TimeoutException as error:
            raise _RetryableError(f"{source} 응답 시간 초과") from error
        except httpx.HTTPError as error:
            raise _RetryableError(
                f"{source} 연결 실패 ({type(error).__name__})"
            ) from error
        status = response.status_code
        if limits.limited(source, response):
            limits.current().report_limited(source, limits.retry_after(response))
            raise _RetryableError(f"{source} 요청 한도 초과", limited=True)
        if status >= HTTP_SERVER_ERROR:
            raise _RetryableError(f"{source} HTTP {status}")
        if status >= HTTP_CLIENT_ERROR:
            raise SourceError(f"{source} HTTP {status}")
        return response.text


class _RetryableError(Exception):
    def __init__(self, message: str, *, limited: bool = False) -> None:
        super().__init__(message)
        self.limited: bool = limited
