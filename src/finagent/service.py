from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, final

from finagent.dashboard import build_dashboard
from finagent.fetch import RETRY_DELAY_SECONDS, FetchContext, new_client

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx

    from finagent.models import Dashboard


def local_now() -> datetime:
    return datetime.now().astimezone()


@final
class DashboardService:
    """Holds the latest dashboard in process memory.

    A pass runs when the app starts and whenever the user asks for a refresh.
    There is no timer and nothing is written to disk.
    """

    __slots__ = (
        "_client_factory",
        "_clock",
        "_demo",
        "_env",
        "_pending",
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
        self._snapshot: Dashboard | None = None
        self._pending: asyncio.Task[Dashboard] | None = None

    async def get(self, *, refresh: bool = False) -> Dashboard:
        if self._pending is not None and not self._pending.done():
            return await asyncio.shield(self._pending)
        if self._snapshot is not None and not refresh:
            return self._snapshot
        self._pending = asyncio.ensure_future(self._build())
        return await asyncio.shield(self._pending)

    async def _build(self) -> Dashboard:
        async with self._client_factory() as client:
            ctx = FetchContext(client, self._env, retry_delay=self._retry_delay)
            dashboard = await build_dashboard(ctx, self._clock(), demo=self._demo)
        self._snapshot = dashboard
        return dashboard
