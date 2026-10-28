"""The demo's generator training: canned jobs on the settings dialog.

``Trainer`` has ``colab_jobs.Trainer``'s page interface with no CLI, thread
or file; ``prices`` gives the dialog its ticker and date choices.
"""

# pandas ships no complete type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Final, final

import pandas as pd

from finagent import generative
from finagent.colab_jobs import DEFAULT_GPU, Job, Request
from finagent.demo import COUNT, VALUE

# Dates in 2099 and a 999-minute job: no real training could have them.
REQUEST: Final = Request(
    tickers=("DEMOK.KS", "DEMOA", "DEMOB", "DEMOC", "KRW=X"),
    start="2099-01-01",
    end="2099-12-31",
    gpu=DEFAULT_GPU,
    models=generative.METHODS,
)
RUNNING: Final = timedelta(minutes=COUNT)  # a job's 경과
STARTED: Final = datetime(2099, 9, 9, 9, 9, tzinfo=UTC)
LAST: Final = Job(
    id=f"{STARTED:%Y%m%d-%H%M%S}",
    session=f"finagent-gen-{STARTED:%Y%m%d-%H%M%S}",
    request=REQUEST,
    stage="done",
    started=STARTED.isoformat(),
    finished=(STARTED + RUNNING).isoformat(),
    results=dict.fromkeys(generative.METHODS, "교체"),
)


def prices() -> pd.DataFrame:
    """Closes of ``REQUEST``'s tickers over its dates at ``VALUE``: the choices."""
    r = REQUEST
    days = pd.bdate_range(r.start, r.end)
    return pd.DataFrame(VALUE, index=days, columns=list(r.tickers))


@final
class Trainer:
    """``colab_jobs.Trainer``'s page interface on canned jobs."""

    __slots__ = ("job", "last", "running")

    def __init__(self) -> None:
        """Idle on ``LAST``; ``start`` shows a job in training, ``cancel`` idle."""
        self.job: Job | None = LAST
        self.last: Job | None = LAST
        self.running = False

    def elapsed(self, job: Job) -> timedelta:
        if not job.finished:
            return RUNNING
        return datetime.fromisoformat(job.finished) - datetime.fromisoformat(
            job.started
        )

    def start(self, request: Request, adjusted: pd.DataFrame) -> bool:
        del adjusted
        if self.running:
            return False
        self.job = replace(
            LAST,
            id="demo",
            session="finagent-gen-demo",
            request=request,
            stage="training",
            finished="",
            results={},
        )
        self.running = True
        return True

    def cancel(self) -> None:
        self.job, self.running = self.last, False
