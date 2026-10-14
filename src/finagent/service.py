from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from typing import TYPE_CHECKING, final

from finagent import catalog
from finagent.dashboard import build_dashboard
from finagent.fetch import RETRY_DELAY_SECONDS, FetchContext, new_client

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx

    from finagent.models import Dashboard
    from finagent.sources import Fetch, Series


def local_now() -> datetime:
    return datetime.now().astimezone()


@final
class DashboardService:
    """Holds the latest dashboard and the series behind it in process memory.

    A pass runs when the app starts and whenever the user asks for a refresh.
    Changing the view fetches only the series not yet in memory; a refresh
    fetches everything the view shows again. There is no timer and nothing is
    written to disk.
    """

    __slots__ = (
        "_cache",
        "_client_factory",
        "_clock",
        "_demo",
        "_env",
        "_pending",
        "_pending_deep",
        "_pending_refresh",
        "_retry_delay",
        "_snapshot",
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
        """Create a service; inject a client factory and clock for tests or demo."""
        self._retry_delay = retry_delay
        self._env = env
        self._client_factory = client_factory
        self._clock = clock
        self._demo = demo
        self._cache: dict[Fetch, Series] = {}
        self._snapshot: Dashboard | None = None
        self._pending: asyncio.Task[Dashboard] | None = None
        self._pending_deep: frozenset[str] = catalog.CORE_VIEW
        self._pending_refresh = False

    async def get(
        self, *, refresh: bool = False, deep: frozenset[str] = catalog.CORE_VIEW
    ) -> Dashboard:
        """Return the dashboard for ``deep`` (see ``catalog.VIEWS``)."""
        deep &= catalog.ALL_VIEWS
        pending = self._pending
        if pending is not None and not pending.done():
            if self._pending_deep == deep:
                return await asyncio.shield(pending)
            # Another view is being built; let it finish, then build this one
            # from what it fetched.
            if self._pending_refresh:
                refresh = False
            with contextlib.suppress(Exception):
                _ = await asyncio.shield(pending)
            return await self.get(refresh=refresh, deep=deep)
        snapshot = self._snapshot
        if not refresh and snapshot is not None and frozenset(snapshot.deep) == deep:
            return snapshot
        if refresh:
            self._cache.clear()
        self._pending_deep = deep
        self._pending_refresh = refresh
        self._pending = asyncio.ensure_future(
            self._build(deep, None if refresh else snapshot)
        )
        return await asyncio.shield(self._pending)

    async def _build(
        self, deep: frozenset[str], previous: Dashboard | None
    ) -> Dashboard:
        async with self._client_factory() as client:
            ctx = FetchContext(client, self._env, retry_delay=self._retry_delay)
            dashboard = await build_dashboard(
                ctx,
                self._clock(),
                demo=self._demo,
                deep=deep,
                cache=self._cache,
                previous=previous,
            )
        self._snapshot = dashboard
        return dashboard
