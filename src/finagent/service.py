from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from typing import TYPE_CHECKING, final

from finagent import catalog
from finagent.dashboard import build_dashboard
from finagent.fetch import RETRY_DELAY_SECONDS, FetchContext, new_client
from finagent.settings import load_view, save_view
from finagent.sources import Ratio, combine_ratio
from finagent.transforms import apply

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping
    from datetime import date
    from pathlib import Path

    import httpx

    from finagent.fetch import Closes
    from finagent.models import Dashboard
    from finagent.sources import Fetch, Series


def local_now() -> datetime:
    return datetime.now().astimezone()


@final
class DashboardService:
    """Holds the latest dashboard and the series behind it in process memory.

    A pass runs when the app starts and whenever the user asks for a refresh.
    Changing the view fetches only the series not yet in memory; a refresh
    fetches everything the view shows again. There is no timer, and the only
    file written is the saved view at ``settings_path`` (none when it is
    ``None``), besides yfinance's caches when ``yahoo`` is ``YahooCloses``.
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
        "_refreshes",
        "_retry_delay",
        "_settings_error",
        "_settings_path",
        "_snapshot",
        "_view",
        "_yahoo",
    )

    def __init__(  # noqa: PLR0913
        self,
        env: Mapping[str, str],
        *,
        client_factory: Callable[[], httpx.AsyncClient] = new_client,
        clock: Callable[[], datetime] = local_now,
        demo: bool = False,
        retry_delay: float = RETRY_DELAY_SECONDS,
        settings_path: Path | None = None,
        yahoo: Closes | None = None,
    ) -> None:
        """Create a service; inject a client factory and clock for tests or demo.

        With ``settings_path`` the view is read from that file at start and
        saved there when it changes; without it the view lives in memory only.
        ``yahoo`` reads Yahoo closes; without it Yahoo rows fail.
        """
        self._retry_delay = retry_delay
        self._yahoo = yahoo
        self._env = env
        self._client_factory = client_factory
        self._clock = clock
        self._demo = demo
        self._cache: dict[Fetch, Series] = {}
        self._refreshes = 0
        self._snapshot: Dashboard | None = None
        self._pending: asyncio.Task[Dashboard] | None = None
        self._pending_deep: frozenset[str] = catalog.CORE_VIEW
        self._pending_refresh = False
        self._settings_path = settings_path
        self._view = catalog.CORE_VIEW
        self._settings_error: str | None = None
        if settings_path is not None:
            self._view, self._settings_error = load_view(settings_path)

    @property
    def view(self) -> frozenset[str]:
        return self._view

    @property
    def series_cache(self) -> dict[Fetch, Series]:
        """Series fetched so far (memory only); analysis tools add to it."""
        return self._cache

    def full_series(self, indicator_id: str) -> Series:
        """An indicator's whole fetched history with its transform applied.

        Reads memory only; empty when the series is not (or no longer) cached.
        """
        indicator = next(
            (
                i
                for spec in catalog.PANELS
                for i in spec.indicators
                if i.id == indicator_id
            ),
            None,
        )
        if indicator is None:
            return ()
        fetch = indicator.fetch
        if isinstance(fetch, Ratio):
            top, bottom = (
                self._cache.get(fetch.numerator),
                self._cache.get(fetch.denominator),
            )
            raw = combine_ratio(top, bottom) if top and bottom else None
        else:
            raw = self._cache.get(fetch)
        if not raw:
            return ()
        return apply(raw, indicator.transform, indicator.freq, indicator.scale)

    @property
    def demo(self) -> bool:
        return self._demo

    @property
    def refreshes(self) -> int:
        """How many refreshes have started; tells whether data moved on."""
        return self._refreshes

    def today(self) -> date:
        return self._clock().date()

    @contextlib.asynccontextmanager
    async def fetch_context(self) -> AsyncGenerator[FetchContext]:
        """A client and fetch context for one analysis run's tool calls."""
        async with self._client_factory() as client:
            yield FetchContext(
                client, self._env, retry_delay=self._retry_delay, yahoo=self._yahoo
            )

    async def set_view(self, deep: frozenset[str]) -> Dashboard:
        """Switch to ``deep``, save it, and return the dashboard for it."""
        deep &= catalog.ALL_VIEWS
        if deep != self._view:
            self._view = deep
            self._settings_error = None
            if self._settings_path is not None:
                try:
                    save_view(self._settings_path, deep)
                except OSError as error:
                    self._settings_error = f"설정을 저장하지 못했습니다: {error}"
        return await self.get()

    async def get(
        self, *, refresh: bool = False, deep: frozenset[str] | None = None
    ) -> Dashboard:
        """Return the dashboard for ``deep`` (default: the saved view)."""
        view = self._view if deep is None else deep & catalog.ALL_VIEWS
        return self._annotate(await self._get(refresh=refresh, deep=view))

    def _annotate(self, dashboard: Dashboard) -> Dashboard:
        path = self._settings_path
        return dashboard.model_copy(
            update={
                "settings_file": None if path is None else str(path),
                "settings_error": self._settings_error,
            }
        )

    async def _get(self, *, refresh: bool, deep: frozenset[str]) -> Dashboard:
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
            return await self._get(refresh=refresh, deep=deep)
        snapshot = self._snapshot
        if not refresh and snapshot is not None and frozenset(snapshot.deep) == deep:
            return snapshot
        if refresh:
            self._cache.clear()
            self._refreshes += 1
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
            ctx = FetchContext(
                client, self._env, retry_delay=self._retry_delay, yahoo=self._yahoo
            )
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
