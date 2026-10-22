"""One background event loop shared by every page session.

Streamlit runs each page script in its own thread. The dashboard service
shares pending fetches between callers and the analysis service holds a run
lock; both belong to a single event loop, so every call goes through the loop
owned here.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from finagent.analysis.service import AnalysisService, Event, RunInput
    from finagent.service import DashboardService


@final
class Runtime:
    __slots__ = ("_loop", "analysis", "dashboard")

    def __init__(self, dashboard: DashboardService, analysis: AnalysisService) -> None:
        """Wrap both services and start the loop thread they run on."""
        self.dashboard = dashboard
        self.analysis = analysis
        self._loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._loop.run_forever, name="finagent-loop", daemon=True
        ).start()

    def call[T](self, coro: Coroutine[object, object, T]) -> T:
        """Run ``coro`` on the shared loop and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def start(self, given: RunInput) -> Run:
        """Start an analysis run in the background; the page drains its events."""
        events: queue.SimpleQueue[Event | None] = queue.SimpleQueue()

        async def pump() -> None:
            async for event in self.analysis.run(given):
                events.put(event)

        async def spawn() -> asyncio.Task[None]:
            task = asyncio.create_task(pump())
            # Also runs when the task is cancelled before it starts.
            task.add_done_callback(lambda _: events.put(None))
            return task

        return Run(self._loop, self.call(spawn()), events)


@final
class Run:
    __slots__ = ("_events", "_loop", "_task")

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        task: asyncio.Task[None],
        events: queue.SimpleQueue[Event | None],
    ) -> None:
        """Track a run task; ``None`` in ``events`` marks its end."""
        self._loop = loop
        self._task = task
        self._events = events

    def drain(self) -> tuple[list[Event], bool]:
        """Events that arrived since the last call, and whether the run ended."""
        out: list[Event] = []
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return out, False
            if event is None:
                return out, True
            out.append(event)

    def stop(self) -> None:
        """Cancel the run; it still ends with the remaining events."""
        _ = self._loop.call_soon_threadsafe(self._task.cancel)
