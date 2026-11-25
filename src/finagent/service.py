"""대시보드 스냅샷 빌드·부분 갱신 및 추가 시계열 인메모리 캐시 서비스."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Final, final

from finagent import catalog, memory
from finagent.dashboard import (
    EXTRAS,
    PAGE_EXTRAS,
    BuildProgress,
    build_dashboard,
    page_fetches,
)
from finagent.demo import dashboard as demo_dashboard
from finagent.fetch import (
    RETRY_DELAY_SECONDS,
    FetchContext,
    SharedWork,
    new_client,
)
from finagent.limits import BACKGROUND_WAIT
from finagent.sources import Difference, EtfSeries, series_code, years_before
from finagent.transforms import apply

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Collection, Mapping

    import httpx

    from finagent.dashboard import Extra
    from finagent.fetch import SourceError
    from finagent.models import Dashboard
    from finagent.sources import Fetch, MacroFetch, Series


@dataclass(frozen=True, slots=True)
class SeriesRecord:
    start: date
    fetched_at: datetime
    points: Series


def local_now() -> datetime:
    return datetime.now().astimezone()


SERIES_TTL: Final = timedelta(hours=1)
SNAPSHOT_TTL: Final = timedelta(hours=1)


def series_key(fetch: Fetch) -> str:
    return f"{fetch.source}:{series_code(fetch)}"


@final
class SeriesStore:
    def __init__(self, clock: Callable[[], datetime] = local_now) -> None:
        self._clock = clock

    def _is_fresh(self, fetched_at: datetime) -> bool:
        now = self._clock()
        return (
            fetched_at.date() == now.date()
            and timedelta(0) <= now - fetched_at <= SERIES_TTL
        )

    def covering(self, source: str, code: str, start: date) -> Series | None:
        found = memory.current()
        key = f"{source}:{code}"
        macro_at = found.macro_at
        extra = found.extra.get(key)
        if (
            extra is not None
            and extra.start <= start
            and self._is_fresh(extra.fetched_at)
            and (macro_at is None or extra.fetched_at >= macro_at)
        ):
            return extra.points
        if macro_at is not None and self._is_fresh(macro_at):
            day = macro_at.date()
            for fetch, points in list(found.macro.items()):
                if (
                    not isinstance(fetch, EtfSeries)
                    and series_key(fetch) == key
                    and years_before(day, fetch.years) <= start
                ):
                    return points
        return None

    def put_extra(self, source: str, code: str, start: date, points: Series) -> None:
        record = SeriesRecord(start, self._clock(), points)
        memory.current().extra.put(f"{source}:{code}", record)

    async def through(
        self,
        fetch: MacroFetch,
        ctx: FetchContext,
        today: date,
        *,
        fresh: bool = False,
    ) -> Series:
        start = years_before(today, fetch.years)
        code = series_code(fetch)
        found = None if fresh else self.covering(fetch.source, code, start)
        if found is not None:
            return found
        series = await ctx.shared(repr(fetch), partial(fetch.fetch, ctx, today))
        self.put_extra(fetch.source, code, start, series)
        return series


@final
class DashboardService:
    __slots__ = (
        "_built_at",
        "_client_factory",
        "_clock",
        "_demo",
        "_env",
        "_pending",
        "_refreshes",
        "_retry_delay",
        "_shared",
        "_snapshot",
        "_stale",
        "progress",
        "series_store",
    )

    def __init__(
        self,
        env: Mapping[str, str],
        *,
        client_factory: Callable[[], httpx.AsyncClient] = new_client,
        clock: Callable[[], datetime] = local_now,
        demo: bool = False,
        retry_delay: float = RETRY_DELAY_SECONDS,
    ) -> None:
        self._retry_delay = retry_delay
        self._shared = SharedWork()
        self.series_store = SeriesStore(clock)
        self._env = env
        self._client_factory = client_factory
        self._clock = clock
        self._demo = demo
        self._built_at: datetime | None = None
        self._stale: dict[Fetch, SourceError] = {}
        self._refreshes = 0
        self._snapshot: Dashboard | None = None
        self._pending: asyncio.Task[Dashboard] | None = None
        self.progress: BuildProgress | None = None

    def full_series(self, indicator_id: str) -> Series:
        indicator = catalog.BY_ID.get(indicator_id)
        if indicator is None:
            return ()
        fetch = indicator.fetch
        if isinstance(fetch, Difference):
            left, right = (
                memory.current().macro.get(fetch.left),
                memory.current().macro.get(fetch.right),
            )
            raw = fetch.combine(left, right) if left and right else None
        else:
            raw = memory.current().macro.get(fetch)
        if not raw:
            return ()
        return apply(raw, indicator.transform, indicator.freq, indicator.scale)

    @property
    def demo(self) -> bool:
        return self._demo

    @property
    def refreshes(self) -> int:
        return self._refreshes

    def today(self) -> date:
        return self._clock().date()

    @contextlib.asynccontextmanager
    async def fetch_context(
        self, *, max_wait: float = BACKGROUND_WAIT
    ) -> AsyncGenerator[FetchContext]:
        async with self._client_factory() as client:
            yield FetchContext(
                client,
                self._env,
                retry_delay=self._retry_delay,
                max_wait=max_wait,
                shared=self._shared,
            )

    async def get(self, *, refresh: bool = False, page: str | None = None) -> Dashboard:
        if page is None:
            return await self._get(refresh=refresh, only=None, extras=EXTRAS)
        return await self._get(
            refresh=refresh,
            only=page_fetches(page),
            extras=PAGE_EXTRAS.get(page, frozenset()),
        )

    async def retry_failed(self) -> Dashboard:
        return await self._get(refresh=True, only=set(self._stale), extras=frozenset())

    def _is_stale(self) -> bool:
        if self._demo or self._built_at is None:
            return False
        now = self._clock()
        return (
            self._built_at.date() != now.date()
            or now - self._built_at > SNAPSHOT_TTL
            or now < self._built_at
        )

    async def _get(
        self,
        *,
        refresh: bool,
        only: Collection[Fetch] | None,
        extras: Collection[Extra],
    ) -> Dashboard:
        pending = self._pending
        if pending is not None and not pending.done():
            return await asyncio.shield(pending)
        snapshot = self._snapshot
        stale = not refresh and snapshot is not None and self._is_stale()
        if not refresh and not stale and snapshot is not None:
            return snapshot
        renew = refresh or stale
        if renew:
            self._refreshes += 1
        self._pending = asyncio.ensure_future(
            self._build(refresh=renew, only=only, extras=extras)
        )
        return await asyncio.shield(self._pending)

    async def _build(
        self,
        *,
        refresh: bool,
        only: Collection[Fetch] | None,
        extras: Collection[Extra],
    ) -> Dashboard:
        if self._demo:
            self._snapshot = demo_dashboard.dashboard(self._clock())
            return self._snapshot
        found = memory.current()
        now = self._clock()
        self.progress = BuildProgress()
        try:
            async with self.fetch_context() as ctx:
                self._snapshot = await build_dashboard(
                    ctx,
                    now,
                    cache=found.macro,
                    refresh=refresh,
                    stale=self._stale,
                    progress=self.progress,
                    only=only,
                    extras=extras,
                    previous=self._snapshot,
                )
        finally:
            self.progress = None
        found.macro_at = now
        self._built_at = now
        return self._snapshot
