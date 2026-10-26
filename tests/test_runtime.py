from __future__ import annotations

import threading
from datetime import UTC, datetime

import httpx

from finagent.analysis.providers.demo import DemoProvider
from finagent.analysis.service import AnalysisService, Event, RunInput
from finagent.demo import DEMO_KEYS, demo_closes, demo_transport
from finagent.runtime import Run, Runtime
from finagent.service import DashboardService

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def runtime(delay: float = 0) -> Runtime:
    transport = demo_transport(NOW.date())
    service = DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: NOW,
        retry_delay=0,
        yahoo=demo_closes(NOW.date()),
    )
    return Runtime(
        service,
        AnalysisService(service, DEMO_KEYS, providers={"demo": DemoProvider(delay)}),
    )


def until_ended(run: Run) -> list[Event]:
    pause = threading.Event()
    seen: list[Event] = []
    for _ in range(500):
        events, ended = run.drain()
        seen.extend(events)
        if ended:
            return seen
        _ = pause.wait(0.01)
    raise AssertionError("run did not end")


def test_a_run_streams_to_done() -> None:
    rt = runtime()

    events = until_ended(rt.start(RunInput("요약")))

    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "done"
    assert not rt.analysis.busy


def first_events(run: Run) -> list[Event]:
    pause = threading.Event()
    for _ in range(500):
        events, _ = run.drain()
        if events:
            return events
        _ = pause.wait(0.01)
    raise AssertionError("run did not start")


def test_a_stopped_run_releases_the_lock() -> None:
    rt = runtime(delay=1)
    run = rt.start(RunInput("요약"))
    _ = first_events(run)

    run.stop()
    stopped = until_ended(run)
    again = rt.start(RunInput("다시"))
    started = first_events(again)
    again.stop()
    _ = until_ended(again)

    assert all(event["type"] != "done" for event in stopped)
    assert started[0]["type"] == "start"
