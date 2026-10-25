"""Colab retraining jobs with the ``colab`` CLI stubbed: no VM, no network."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from finagent import colab_jobs, generative

TRAINED = Path(__file__).resolve().parents[1] / "data" / "models"
TICKERS = ("AAPL", "MSFT", "PGR", "^GSPC", "^KS11", "KRW=X")  # the trained ones


class FakeColab:
    """``subprocess.run`` for the CLI: records commands, serves trained files."""

    def __init__(self, fail: str = "") -> None:
        self.calls: list[str] = []
        self.argv: list[list[str]] = []
        self.fail: str = fail

    def __call__(
        self, argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        command = argv[1]
        self.calls.append(command)
        self.argv.append(argv)
        if command == self.fail:
            return subprocess.CompletedProcess(argv, 1, "", "quota")
        out = ""
        if command == "exec" and "status.txt" in str(kwargs["input"]):
            out = "sigcwgan exit=0 seconds=1\nnsde exit=0 seconds=1\ndone\n"
        if command == "download":
            method, name = argv[-2].split("/")[-2:]
            local = Path(argv[-1])
            _ = shutil.copy(TRAINED / method / name, local)
            if method == "nsde" and name == generative.META:  # fails verification
                meta = cast(
                    "dict[str, list[str]]", json.loads(local.read_text("utf-8"))
                )
                meta["tickers"] = ["AAPL", *meta["tickers"][1:-1], "XXX"]
                _ = local.write_text(json.dumps(meta), "utf-8")
        return subprocess.CompletedProcess(argv, 0, out, "")


def prices() -> pd.DataFrame:
    days = pd.bdate_range("2015-01-01", "2021-01-01")
    walk = np.exp(np.random.default_rng(0).normal(0, 0.01, (len(days), 6)).cumsum(0))
    return pd.DataFrame(walk, index=days, columns=list(TICKERS))


def request() -> colab_jobs.Request:
    return colab_jobs.Request(
        TICKERS, "2015-01-01", "2021-01-01", "T4", ("sigcwgan", "nsde")
    )


@pytest.fixture
def colab(monkeypatch: pytest.MonkeyPatch) -> FakeColab:
    fake = FakeColab()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(colab_jobs, "executable", lambda: "colab")
    return fake


def test_a_job_swaps_only_verified_weights_and_stops(
    tmp_path: Path, colab: FakeColab
) -> None:
    root = tmp_path / "data" / "models"
    _ = shutil.copytree(TRAINED / "nsde", root / "nsde")
    _ = shutil.copytree(TRAINED / "sigcwgan", root / "sigcwgan")
    old = (root / "sigcwgan" / generative.WEIGHTS).stat().st_mtime_ns
    jobs = colab_jobs.Trainer(root, poll=0)

    assert jobs.start(request(), prices())
    jobs.join(30)

    uploads = 1 + len(colab_jobs.SCRIPTS) + 1  # dataset, scripts, generative.py
    downloads = 2 * len(colab_jobs.FILES)
    assert colab.calls == [
        "new",
        *["upload"] * uploads,
        "exec",  # launch
        "exec",  # one poll: done
        *["download"] * downloads,
        "stop",
    ]
    job = jobs.job
    assert job is not None
    assert job.stage == "done", job.message
    assert job.stopped
    assert job.results == {"sigcwgan": "교체", "nsde": "검증 실패: 종목이 다름"}
    assert jobs.last == job
    assert (root / "sigcwgan.prev" / generative.WEIGHTS).stat().st_mtime_ns == old
    assert generative.load(root / "sigcwgan") is not None
    assert not (root / "nsde.prev").exists()  # the failed one is left alone
    assert not any((root / colab_jobs.STAGING).iterdir())
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert all(p.is_relative_to(root) for p in written)


def test_a_failed_step_still_stops_the_session(
    tmp_path: Path, colab: FakeColab
) -> None:
    colab.fail = "upload"
    root = tmp_path / "data" / "models"
    jobs = colab_jobs.Trainer(root, poll=0)

    assert jobs.start(request(), prices())
    jobs.join(30)

    assert colab.calls == ["new", "upload", "stop"]
    job = jobs.job
    assert job is not None
    assert job.stage == "failed"
    assert "colab upload 실패: quota" in job.message
    assert job.stopped
    assert jobs.last is None
    assert not (root / "sigcwgan").exists()


def test_a_restart_stops_a_job_past_its_limit(tmp_path: Path, colab: FakeColab) -> None:
    root = tmp_path / "data" / "models"
    root.mkdir(parents=True)
    started = datetime.now(UTC) - colab_jobs.MAX_RUNTIME - timedelta(minutes=1)
    stale = colab_jobs.Job(
        id="x",
        session="finagent-gen-x",
        request=request(),
        stage="training",
        started=started.isoformat(),
        created=True,
    )
    _ = (root / colab_jobs.STATE_FILE).write_text(json.dumps({"job": asdict(stale)}))

    jobs = colab_jobs.Trainer(root, poll=0)
    jobs.reconcile()
    jobs.join(30)

    assert colab.calls == ["stop"]
    job = jobs.job
    assert job is not None
    assert job.stage == "failed"
    assert "최대 실행 시간" in job.message
    assert job.stopped


def test_the_demo_trainer_switches_canned_states_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(subprocess, "run", refuse)
    jobs = colab_jobs.DemoTrainer()
    r = colab_jobs.DEMO_REQUEST
    prices = colab_jobs.demo_prices()
    assert colab_jobs.problem(r, prices) is None  # the canned choices can start
    assert not jobs.running
    assert jobs.job == jobs.last == colab_jobs.DEMO_LAST
    assert jobs.elapsed(colab_jobs.DEMO_LAST) == timedelta(hours=3, minutes=12)

    picked = colab_jobs.Request(r.tickers[:2], r.start, r.end, "L4", ("nsde",))
    assert jobs.start(picked, prices)
    assert jobs.running
    job = cast("colab_jobs.Job", jobs.job)
    assert (job.stage, job.request) == ("training", picked)
    assert jobs.elapsed(job) == colab_jobs.DEMO_RUNNING
    assert not jobs.start(r, prices)  # one at a time
    assert jobs.last == colab_jobs.DEMO_LAST

    jobs.cancel()
    assert not jobs.running
    assert jobs.job == colab_jobs.DEMO_LAST
