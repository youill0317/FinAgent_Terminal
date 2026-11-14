"""데이터 공급자별 호출 빈도 제한(Rate Limit) 및 동시성 슬롯 관리."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast, final

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping

    from finagent.models import SourceId

LimitStatus = Literal["ok", "waiting", "reached", "blocked"]

INTERACTIVE_WAIT: Final = 20.0
BACKGROUND_WAIT: Final = 120.0
SLOT_POLL: Final = 0.02
RETRY_SECONDS: Final = 2.0
HTTP_TOO_MANY_REQUESTS: Final = 429
HTTP_BAD_REQUEST: Final = 400
HTTP_SERVER_ERROR: Final = 500
MINUTE: Final = 60.0
HOUR: Final = 3600.0


class SourceError(Exception): ...


class LimitReached(SourceError):  # noqa: N818  # 도메인 상태 표현 예외 명칭 유지
    def __init__(self, source: SourceId, next_at: float) -> None:
        super().__init__(f"{source} 호출 제한: 다음 가능 시각까지 대기 상한 초과 예상")
        self.source: SourceId = source
        self.next_at: float = next_at


@dataclass(frozen=True, slots=True)
class SourceLimit:
    count: int
    seconds: float
    concurrent: int
    retries: int
    default_wait: float


LIMITS: Final[Mapping[SourceId, SourceLimit]] = MappingProxyType(
    {
        "FRED": SourceLimit(110, MINUTE, 4, 1, MINUTE),
        "ALPACA": SourceLimit(150, MINUTE, 4, 2, MINUTE),
        "SEC": SourceLimit(8, 1.0, 8, 2, 5.0),
        "OPENFIGI": SourceLimit(20, MINUTE, 4, 4, MINUTE),
        "OECD": SourceLimit(20, HOUR, 4, 1, HOUR),
        "ECOS": SourceLimit(60, MINUTE, 4, 1, MINUTE),
        "ECB": SourceLimit(60, MINUTE, 4, 1, MINUTE),
        "BIS": SourceLimit(60, MINUTE, 4, 1, MINUTE),
        "YAHOO": SourceLimit(200, MINUTE, 4, 0, MINUTE),
    }
)
FIGI_KEYED_LIMITS: Final[Mapping[SourceId, SourceLimit]] = MappingProxyType(
    {
        **LIMITS,
        "OPENFIGI": SourceLimit(200, MINUTE, 4, 4, MINUTE),
    }
)


@dataclass(frozen=True, slots=True)
class LimitState:
    source: SourceId
    status: LimitStatus
    next_at: float = 0.0


@dataclass(slots=True)
class _Usage:
    starts: deque[float] = field(default_factory=deque)
    blocked_until: float = 0.0
    waiting_until: float = 0.0
    reached_until: float = 0.0


@final
class Limiter:
    def __init__(
        self,
        *,
        limits: Mapping[SourceId, SourceLimit] = LIMITS,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._limits = limits
        self._clock = clock
        self._sleep = sleep
        self._asleep = asleep
        self._lock = threading.Lock()
        self._usage: dict[SourceId, _Usage] = {s: _Usage() for s in limits}
        self._slots = {
            source: threading.BoundedSemaphore(limit.concurrent)
            for source, limit in limits.items()
        }

    def acquire(
        self, source: SourceId, cost: int = 1, max_wait: float = BACKGROUND_WAIT
    ) -> None:
        delay = self._reserve(source, cost, max_wait)
        if delay > 0:
            self._sleep(delay)

    async def aacquire(
        self, source: SourceId, cost: int = 1, max_wait: float = BACKGROUND_WAIT
    ) -> None:
        delay = self._reserve(source, cost, max_wait)
        if delay > 0:
            await self._asleep(delay)

    @contextlib.contextmanager
    def slot(
        self, source: SourceId, max_wait: float = BACKGROUND_WAIT
    ) -> Generator[None]:
        semaphore = self._slots[source]
        if not semaphore.acquire(timeout=max_wait):
            raise self._busy(source)
        try:
            yield
        finally:
            semaphore.release()

    @contextlib.asynccontextmanager
    async def aslot(
        self, source: SourceId, max_wait: float = BACKGROUND_WAIT
    ) -> AsyncGenerator[None]:
        semaphore = self._slots[source]
        deadline = time.monotonic() + max_wait
        while not semaphore.acquire(blocking=False):
            if time.monotonic() >= deadline:
                raise self._busy(source)
            await asyncio.sleep(SLOT_POLL)
        try:
            yield
        finally:
            semaphore.release()

    def _busy(self, source: SourceId) -> LimitReached:
        with self._lock:
            usage = self._usage[source]
            at = max(usage.blocked_until, usage.waiting_until, self._clock())
            return LimitReached(source, at)

    def report_limited(
        self, source: SourceId, retry_after: float | None = None
    ) -> None:
        wait = retry_after
        if wait is None:
            wait = self._limits[source].default_wait
        with self._lock:
            usage = self._usage[source]
            usage.blocked_until = max(usage.blocked_until, self._clock() + wait)

    def reached(self, source: SourceId) -> LimitReached:
        with self._lock:
            return LimitReached(source, self._usage[source].blocked_until)

    def pause(self, seconds: float) -> None:
        self._sleep(seconds)

    def state(self) -> list[LimitState]:
        now = self._clock()
        with self._lock:
            return [
                self._state_of(source, usage, now)
                for source, usage in self._usage.items()
            ]

    @staticmethod
    def _state_of(source: SourceId, usage: _Usage, now: float) -> LimitState:
        if usage.blocked_until > now:
            return LimitState(source, "blocked", usage.blocked_until)
        if usage.reached_until > now:
            return LimitState(source, "reached", usage.reached_until)
        if usage.waiting_until > now:
            return LimitState(source, "waiting", usage.waiting_until)
        return LimitState(source, "ok")

    def _reserve(self, source: SourceId, cost: int, max_wait: float) -> float:
        limit = self._limits[source]
        keep = limit.seconds
        with self._lock:
            now = self._clock()
            usage = self._usage[source]
            while usage.starts and usage.starts[0] <= now - keep:
                _ = usage.starts.popleft()
            planned = list(usage.starts)
            at = max(now, usage.blocked_until)
            for _ in range(max(cost, 1)):
                at = _next_start(planned, limit, at)
                planned.append(at)
            delay = at - now
            if delay > max_wait:
                usage.reached_until = at
                raise LimitReached(source, at)
            usage.starts = deque(planned)
            if delay > 0:
                usage.waiting_until = at
            return delay


def _next_start(planned: list[float], limit: SourceLimit, earliest: float) -> float:
    at = max(earliest, planned[-1]) if planned else earliest
    if len(planned) >= limit.count:
        at = max(at, planned[-limit.count] + limit.seconds)
    return at


def limited(source: SourceId, response: httpx.Response) -> bool:
    status = response.status_code
    if status == HTTP_TOO_MANY_REQUESTS:
        return True
    return (
        source == "ECB"
        and status == HTTP_BAD_REQUEST
        and "html" in response.text[:200].lower()
    )


def retry_after(response: httpx.Response) -> float | None:
    after = cast("str", response.headers.get("retry-after", ""))
    return float(after) if after.isdigit() else None


_current = Limiter()


def current() -> Limiter:
    return _current


def configure(limiter: Limiter) -> Limiter:
    global _current  # noqa: PLW0603
    _current = limiter
    return limiter


def send(
    source: SourceId,
    call: Callable[[], httpx.Response],
    max_wait: float = BACKGROUND_WAIT,
) -> httpx.Response:
    limiter = current()
    left = LIMITS[source].retries
    with limiter.slot(source, max_wait):
        while True:
            limiter.acquire(source, max_wait=max_wait)
            try:
                response = call()
            except httpx.TransportError:
                if left == 0:
                    raise
                limiter.pause(RETRY_SECONDS)
                left -= 1
                continue
            if limited(source, response):
                limiter.report_limited(source, retry_after(response))
                if left == 0:
                    raise limiter.reached(source)
            elif response.status_code < HTTP_SERVER_ERROR or left == 0:
                return response
            else:
                limiter.pause(RETRY_SECONDS)
            left -= 1
