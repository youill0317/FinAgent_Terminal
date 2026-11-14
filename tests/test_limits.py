from __future__ import annotations

from typing import TYPE_CHECKING

from finagent.limits import LIMITS, Limiter
from tests.fake_sources import FakeClock

if TYPE_CHECKING:
    from finagent.models import SourceId


def limiter(clock: FakeClock) -> Limiter:
    return Limiter(clock=clock.time, sleep=clock.sleep, asleep=clock.asleep)


def within_windows(times: list[float], source: SourceId) -> bool:
    ordered = sorted(times)
    limit = LIMITS[source]
    return all(
        ordered[i + limit.count] - ordered[i] >= limit.seconds
        for i in range(len(ordered) - limit.count)
    )


def test_requests_beyond_the_window_wait_for_it() -> None:
    clock = FakeClock()
    limit = limiter(clock)
    log: list[float] = []

    for _ in range(61):
        limit.acquire("ECOS")
        log.append(clock.time())

    assert within_windows(log, "ECOS")
    assert log[60] - log[0] == 60.0


def test_a_limited_answer_holds_every_caller_until_retry_after() -> None:
    clock = FakeClock()
    limit = limiter(clock)

    limit.report_limited("ECB", 30.0)
    assert limit.state()[list(LIMITS).index("ECB")].status == "blocked"
    limit.acquire("ECB")

    assert clock.slept == [30.0]
