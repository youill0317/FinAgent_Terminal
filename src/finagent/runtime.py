from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from concurrent.futures import Future

    from finagent.analysis.service import AnalysisService
    from finagent.service import DashboardService


@final
class Runtime:
    __slots__ = ("_loop", "analysis", "dashboard", "workspace")

    def __init__(self, dashboard: DashboardService, analysis: AnalysisService) -> None:
        self.dashboard = dashboard
        self.analysis = analysis
        self.workspace = analysis.open()
        self._loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._loop.run_forever, name="finagent-loop", daemon=True
        ).start()

    def renew(self) -> None:
        if not self.workspace.busy:
            self.workspace = self.analysis.open()

    def call[T](self, coro: Coroutine[object, object, T]) -> T:
        return self.submit(coro).result()

    def submit[T](self, coro: Coroutine[object, object, T]) -> Future[T]:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
