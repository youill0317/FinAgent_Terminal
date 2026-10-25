"""Retraining the scenario generators on Colab, started from the 설정 dialog.

Like the screener and the portfolio page, this module is an exception to the
dashboard's rules (``tests/test_guardrails.py``): it runs the ``colab`` CLI
as a subprocess and writes files, only under ``data/models/`` (the job state,
a staging folder, and the swapped weights). The CLI keeps its own login and
session records; this module only checks that its login token exists.

One job at a time per models folder, in a background thread:

1. ``export``: the chosen tickers' daily log returns over the chosen dates,
   on days all have a close (``returns_table``), to a staging ``returns.npz``;
2. ``new``: ``colab new`` with the chosen GPU;
3. ``upload``: the dataset, the ``training/`` scripts and ``generative.py``
   (the parity check imports that exact module);
4. ``launch``: one short ``colab exec`` that starts setup and the chosen
   trainings on the VM in the background and returns;
5. ``training``: every ``POLL_SECONDS`` one short ``colab exec`` reads the
   VM's ``status.txt``, until it says ``done`` or ``MAX_RUNTIME`` passes;
6. ``download``: each finished model's files to staging;
7. ``verify``: each must load with ``generative.load``, carry the chosen
   tickers and a recorded parity under ``PARITY``; then it replaces
   ``data/models/<method>/``, the previous set kept in ``<method>.prev/``.

``colab stop`` always runs at the end. The state is saved in ``STATE_FILE``
after every stage, so a restarted app shows it and ``reconcile`` picks the
job up again: polling resumes once the trainings were launched; a job
stopped before that, or past ``MAX_RUNTIME``, has its session stopped.
"""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, final

import numpy as np
import pandas as pd

from finagent import generative

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

STATE_FILE: Final = "training.json"
STAGING: Final = ".incoming"
PREVIOUS: Final = ".prev"  # data/models/<method>.prev/
GPUS: Final = ("T4", "L4", "G4", "A100", "H100")  # colab new --gpu
DEFAULT_GPU: Final = "T4"
POLL_SECONDS: Final = 300.0
MAX_RUNTIME: Final = timedelta(hours=8)  # the three on a T4 took 3.2 hours
CALL_SECONDS: Final = 120.0  # one CLI call; ``colab new`` gets NEW_SECONDS
NEW_SECONDS: Final = 900.0
EXEC_SECONDS: Final = 60  # colab exec --timeout for the launch and each poll
POLL_FAILURES: Final = 3  # polls in a row that fail before the job does
PARITY: Final = 1e-3  # torch vs numpy max abs diff a model may record
MIN_TICKERS: Final = 2  # Tail-GAN's long-short portfolios need two assets
MIN_DAYS: Final = 1300  # Tail-GAN: 1000 windows of 21 days in the 80% split
REMOTE: Final = "/content"
OUT: Final = "/content/out"
FILES: Final = ("weights.npz", "meta.json", "log.txt")
SCRIPTS: Final = (
    "common.py",
    "train_tailgan.py",
    "train_sigcwgan.py",
    "train_nsde.py",
    "setup_colab.sh",
    "run_all.sh",
    "requirements-colab.txt",
)
TRAINING_DIR: Final = Path(__file__).resolve().parents[2] / "training"
TOKEN: Final = Path.home() / ".config" / "colab-cli" / "token.json"
LOGIN: Final = "colab sessions"  # the CLI's first call opens its OAuth login
INSTALL: Final = "uv tool install google-colab-cli"
STAGES: Final = {
    "export": "자료 내보내기",
    "new": "Colab 세션 생성",
    "upload": "업로드",
    "launch": "학습 시작",
    "training": "학습 중",
    "download": "내려받기",
    "verify": "검증·교체",
    "done": "완료",
    "failed": "실패",
}
RESUMABLE: Final = ("training", "download", "verify")
FINISHED: Final = ("done", "failed")
STATUS_LINE: Final = re.compile(r"^(\w+) exit=(\d+)")
NOTICE: Final = re.compile(r"new version|colab update|enable_update_check")
POLL_CODE: Final = (
    "import pathlib\n"
    f"p = pathlib.Path('{OUT}/status.txt')\n"
    "print(p.read_text() if p.exists() else '', end='')\n"
)


class JobError(RuntimeError):
    """A stage failed; the message is shown in 설정."""


@dataclass(frozen=True, slots=True)
class Request:
    tickers: tuple[str, ...]
    start: str  # ISO dates
    end: str
    gpu: str
    models: tuple[str, ...]  # generative.METHODS


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    session: str
    request: Request
    stage: str  # STAGES
    started: str  # ISO, UTC
    finished: str = ""
    polled: str = ""
    created: bool = False  # colab new was run: the session may exist
    stopped: bool = False
    message: str = ""
    results: dict[str, str] = field(default_factory=dict)  # method: outcome


def returns_table(
    adjusted: pd.DataFrame,
    tickers: Sequence[str],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Daily log returns of ``tickers`` on the days all have a close.

    Within ``start``..``end``. Mixed calendars: a return spans each market's
    holidays instead of showing a carried-over zero.
    """
    prices = adjusted.reindex(columns=list(tickers)).sort_index()
    first = None if start is None else pd.Timestamp(start)
    last = None if end is None else pd.Timestamp(end)
    prices = prices.loc[first:last].dropna(how="any")
    logs = cast("pd.DataFrame", np.log(prices))
    return logs.diff().iloc[1:]


def export(
    out: Path,
    adjusted: pd.DataFrame,
    tickers: Sequence[str],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Write ``returns_table`` to ``out`` as the training scripts read it."""
    rets = returns_table(adjusted, tickers, start, end)
    np.savez(
        out,
        returns=rets.to_numpy(dtype="float64"),
        tickers=np.array(list(tickers)),
        dates=np.array([d.strftime("%Y-%m-%d") for d in rets.index]),
    )
    return rets


def executable() -> str | None:
    found = shutil.which("colab")
    if found is not None:
        return found
    local = Path.home() / ".local" / "bin" / "colab"
    return str(local) if local.is_file() else None


def blocker() -> str | None:
    """Why a job cannot start here, as page text; None when it can."""
    if executable() is None:
        return f"colab CLI가 없습니다: {INSTALL}"
    if not TOKEN.is_file():
        return f"Colab 로그인이 필요합니다: 터미널에서 {LOGIN}"
    missing = [n for n in SCRIPTS if not (TRAINING_DIR / n).is_file()]
    if missing:
        return f"학습 스크립트가 없습니다: {', '.join(missing)}"
    return None


def problem(request: Request, adjusted: pd.DataFrame) -> str | None:
    """What keeps ``request`` from training, as page text; None when nothing."""
    if len(request.tickers) < MIN_TICKERS:
        return f"종목을 {MIN_TICKERS}개 이상 고르세요"
    if not request.models:
        return "모델을 고르세요"
    if request.gpu not in GPUS:
        return f"GPU: {', '.join(GPUS)}"
    rows = returns_table(
        adjusted,
        request.tickers,
        date.fromisoformat(request.start),
        date.fromisoformat(request.end),
    )
    if len(rows) < MIN_DAYS:
        return f"겹치는 수익률이 {len(rows):,}일입니다 ({MIN_DAYS:,}일 이상 필요)"
    return None


def verify(folder: Path, method: str, tickers: Sequence[str]) -> str | None:
    """Why the model in ``folder`` cannot replace the live one; None when fine."""
    gen = generative.load(folder)
    if gen is None or gen.method != method:
        return "불러오지 못함"
    if gen.tickers != tuple(tickers):
        return "종목이 다름"
    parity = gen.meta.get("parity")
    diff = (
        cast("dict[str, object]", parity).get("max_abs_diff")
        if isinstance(parity, dict)
        else None
    )
    if not isinstance(diff, (int, float)) or not math.isfinite(diff) or diff > PARITY:
        return f"패리티 {diff}"
    drawn = gen.paths(8, 5, np.random.default_rng(0))
    if not np.isfinite(drawn).all():
        return "경로에 nan"
    return None


def swap(root: Path, method: str, staged: Path) -> None:
    """Make ``staged`` the live ``method``; the live set becomes ``.prev``."""
    live = root / method
    previous = root / f"{method}{PREVIOUS}"
    if previous.exists():
        shutil.rmtree(previous)
    if live.exists():
        _ = live.rename(previous)
    _ = staged.rename(live)


def _now() -> datetime:
    return datetime.now(UTC)


@final
class Trainer:
    """Runs one training job at a time for ``root`` (``data/models/``)."""

    __slots__ = ("_cancel", "_clock", "_lock", "_thread", "limit", "poll", "root")

    def __init__(
        self,
        root: Path,
        *,
        poll: float = POLL_SECONDS,
        limit: timedelta = MAX_RUNTIME,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        """``poll`` seconds between status reads; a job fails after ``limit``."""
        self.root = root
        self.poll = poll
        self.limit = limit
        self._clock = clock
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- state

    @property
    def job(self) -> Job | None:
        """The current or latest job."""
        return self._read("job")

    @property
    def last(self) -> Job | None:
        """The latest job that replaced a model."""
        return self._read("last")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def elapsed(self, job: Job) -> timedelta:
        end = datetime.fromisoformat(job.finished) if job.finished else self._clock()
        return end - datetime.fromisoformat(job.started)

    def _read(self, key: str) -> Job | None:
        path = self.root / STATE_FILE
        try:
            raw = cast("dict[str, object]", json.loads(path.read_text("utf-8")))
        except (OSError, ValueError):
            return None
        item = raw.get(key)
        if not isinstance(item, dict):
            return None
        data = cast("dict[str, object]", item)
        try:
            request = cast("dict[str, object]", data.pop("request"))
            return Job(
                **data,  # pyright: ignore[reportArgumentType]
                request=Request(
                    tickers=tuple(cast("list[str]", request["tickers"])),
                    start=str(request["start"]),
                    end=str(request["end"]),
                    gpu=str(request["gpu"]),
                    models=tuple(cast("list[str]", request["models"])),
                ),
            )
        except (KeyError, TypeError):
            return None

    def _save(self, job: Job, *, success: bool = False) -> Job:
        path = self.root / STATE_FILE
        with self._lock:
            try:
                body = cast("dict[str, object]", json.loads(path.read_text("utf-8")))
            except (OSError, ValueError):
                body = {}
            body["job"] = asdict(job)
            if success:
                body["last"] = asdict(job)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f"{path.name}.tmp")
            _ = temp.write_text(json.dumps(body, ensure_ascii=False, indent=2), "utf-8")
            _ = temp.replace(path)
        return job

    # ---- control

    def start(self, request: Request, adjusted: pd.DataFrame) -> bool:
        """Start a job unless one is running."""
        if self.running:
            return False
        now = self._clock()
        stamp = now.strftime("%Y%m%d-%H%M%S")
        job = Job(
            id=stamp,
            session=f"finagent-gen-{stamp}",
            request=request,
            stage="export",
            started=now.isoformat(),
        )
        self._launch_thread(job, adjusted, fresh=True)
        return True

    def cancel(self) -> None:
        """Stop the running job at its next step; its session is stopped."""
        self._cancel.set()

    def reconcile(self) -> None:
        """Pick up a job a previous app process left unfinished."""
        job = self.job
        if job is None or job.stage in FINISHED or self.running:
            return
        if job.stage in RESUMABLE:
            self._launch_thread(job, None, fresh=False)  # past its limit it fails
        else:
            stale = replace(job, message="앱이 다시 시작돼 중단")
            self._launch_thread(stale, None, fresh=False)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _launch_thread(
        self, job: Job, adjusted: pd.DataFrame | None, *, fresh: bool
    ) -> None:
        self._cancel.clear()
        _ = self._save(job)
        self._thread = threading.Thread(
            target=self._run,
            args=(job, adjusted, fresh),
            name="finagent-training",
            daemon=True,
        )
        self._thread.start()

    # ---- the job

    def _run(self, job: Job, adjusted: pd.DataFrame | None, fresh: bool) -> None:
        staging = self.root / STAGING / job.id
        error = ""
        try:
            if not fresh and job.stage not in RESUMABLE:
                error = job.message  # interrupted before the work ran
            else:
                if adjusted is not None:
                    job = self._prepare(job, adjusted, staging)
                if not self._collect(job, staging):
                    error = "교체한 모델이 없습니다"
        except Exception as caught:  # noqa: BLE001 - shown in 설정, never lost
            error = str(caught) if isinstance(caught, JobError) else repr(caught)
        finally:
            job = self.job or job  # as far as it got
            stopped = job.created and self._stop(job)
            shutil.rmtree(staging, ignore_errors=True)
            done = replace(
                job,
                stage="failed" if error else "done",
                finished=self._clock().isoformat(),
                stopped=stopped,
                message=error,
            )
            _ = self._save(done, success=not error)

    def _prepare(self, job: Job, adjusted: pd.DataFrame, staging: Path) -> Job:
        """Export, ``colab new``, upload and launch the trainings on the VM."""
        r = job.request
        job = self._stage(job, "export")
        staging.mkdir(parents=True, exist_ok=True)
        start, end = date.fromisoformat(r.start), date.fromisoformat(r.end)
        _ = export(staging / "returns.npz", adjusted, r.tickers, start, end)
        job = self._stage(replace(job, created=True), "new")
        _ = self._cli("new", "-s", job.session, "--gpu", r.gpu, timeout=NEW_SECONDS)
        job = self._stage(job, "upload")
        for local in self._uploads(staging):
            remote = f"{REMOTE}/{local.name}"
            _ = self._cli("upload", "-s", job.session, str(local), remote)
        job = self._stage(job, "launch")
        _ = self._exec(job, self._launch_code(job))
        return job

    def _collect(self, job: Job, staging: Path) -> bool:
        """Wait, download, verify and swap; whether any model was replaced."""
        r = job.request
        job = self._stage(job, "training")
        codes = self._wait(job)
        job = self._stage(job, "download")
        models = r.models
        finished = [m for m in models if codes.get(m) == 0]
        for m in finished:
            (staging / m).mkdir(parents=True, exist_ok=True)
            for name in FILES:
                remote, local = f"{OUT}/{m}/{name}", str(staging / m / name)
                _ = self._cli("download", "-s", job.session, remote, local)
        job = self._stage(job, "verify")
        results = {
            m: f"학습 실패 (exit {codes.get(m)})" for m in models if m not in finished
        }
        for m in finished:
            why = verify(staging / m, m, r.tickers)
            if why is None:
                swap(self.root, m, staging / m)
            results[m] = "교체" if why is None else f"검증 실패: {why}"
        _ = self._save(replace(job, results=results))
        return "교체" in results.values()

    def _stage(self, job: Job, stage: str) -> Job:
        if self._cancel.is_set():
            raise JobError("중지됨")
        return self._save(replace(job, stage=stage))

    def _uploads(self, staging: Path) -> list[Path]:
        return [
            staging / "returns.npz",
            *(TRAINING_DIR / n for n in SCRIPTS),
            Path(generative.__file__),
        ]

    def _launch_code(self, job: Job) -> str:
        names = ["returns.npz", *SCRIPTS, "generative.py"]
        models = " ".join(job.request.models)
        script = (
            f"if bash {REMOTE}/gen/setup_colab.sh > {OUT}/setup.log 2>&1; then"
            f" OUT={OUT} MODELS='{models}' bash {REMOTE}/gen/run_all.sh;"
            f' else echo "setup exit=$?" >> {OUT}/status.txt;'
            f" echo done >> {OUT}/status.txt; fi"
        )
        return (
            "import os, subprocess\n"
            f"os.makedirs('{REMOTE}/gen', exist_ok=True)\n"
            f"os.makedirs('{OUT}', exist_ok=True)\n"
            f"for name in {names!r}:\n"
            f"    os.replace('{REMOTE}/' + name, '{REMOTE}/gen/' + name)\n"
            f"if os.path.exists('{OUT}/status.txt'):\n"
            f"    os.remove('{OUT}/status.txt')\n"
            f"subprocess.Popen(['bash', '-c', {script!r}], start_new_session=True,\n"
            "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "print('launched')\n"
        )

    def _wait(self, job: Job) -> dict[str, int]:
        """Poll the VM's status.txt until ``done``: each run's exit code."""
        failures = 0
        while True:
            if self.elapsed(job) > self.limit:
                raise JobError(f"최대 실행 시간 {self.limit} 초과")
            try:
                text = self._exec(job, POLL_CODE)
            except JobError:
                failures += 1
                if failures >= POLL_FAILURES:
                    raise
            else:
                failures = 0
                lines = text.splitlines()
                codes = {
                    m.group(1): int(m.group(2))
                    for m in map(STATUS_LINE.match, lines)
                    if m is not None
                }
                if "setup" in codes:
                    raise JobError(f"Colab 설치 실패 (exit {codes['setup']})")
                if "done" in (line.strip() for line in lines):
                    return codes
            job = self._save(replace(job, polled=self._clock().isoformat()))
            if self._cancel.wait(self.poll):
                raise JobError("중지됨")

    def _exec(self, job: Job, code: str) -> str:
        return self._cli(
            "exec", "-s", job.session, "--timeout", str(EXEC_SECONDS), stdin=code
        )

    def _stop(self, job: Job) -> bool:
        try:
            _ = self._cli("stop", "-s", job.session)
        except (JobError, OSError, subprocess.SubprocessError):
            return False
        return True

    def _cli(self, *args: str, timeout: float = CALL_SECONDS, stdin: str = "") -> str:
        """Run one ``colab`` command; its stdout. Raises ``JobError`` on failure."""
        exe = executable()
        if exe is None:
            raise JobError(f"colab CLI가 없습니다: {INSTALL}")
        try:
            done = subprocess.run(  # noqa: S603 - our own argument list
                [exe, *args],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            late = f"colab {args[0]}: {timeout:.0f}초 안에 끝나지 않음"
            raise JobError(late) from error
        if done.returncode != 0:
            lines = [
                line
                for line in f"{done.stdout}\n{done.stderr}".splitlines()
                if line.strip() and not NOTICE.search(line)
            ]
            tail = lines[-1][:200] if lines else f"exit {done.returncode}"
            raise JobError(f"colab {args[0]} 실패: {tail}")
        return done.stdout


# ---------------------------------------------------------------- demo

DEMO_REQUEST: Final = Request(
    tickers=("005930.KS", "AAPL", "MSFT", "NVDA", "KRW=X"),
    start="2015-01-02",
    end="2026-09-30",
    gpu=DEFAULT_GPU,
    models=generative.METHODS,
)
DEMO_LAST: Final = Job(
    id="20261020-090000",
    session="finagent-gen-20261020-090000",
    request=DEMO_REQUEST,
    stage="done",
    started="2026-10-20T09:00:00+00:00",
    finished="2026-10-20T12:12:00+00:00",
    results=dict.fromkeys(generative.METHODS, "교체"),
)
DEMO_RUNNING: Final = timedelta(hours=1, minutes=12)  # a started job's 경과


def demo_prices() -> pd.DataFrame:
    """Flat closes of ``DEMO_REQUEST``'s tickers over its dates: the choices."""
    r = DEMO_REQUEST
    days = pd.bdate_range(r.start, r.end)
    return pd.DataFrame(1.0, index=days, columns=list(r.tickers))


@final
class DemoTrainer:
    """``Trainer``'s page interface on canned jobs: no CLI, thread or file."""

    __slots__ = ("job", "last", "running")

    def __init__(self) -> None:
        """Idle on ``DEMO_LAST``; ``start`` shows a job in training, ``cancel`` idle."""
        self.job: Job | None = DEMO_LAST
        self.last: Job | None = DEMO_LAST
        self.running = False

    def elapsed(self, job: Job) -> timedelta:
        if not job.finished:
            return DEMO_RUNNING
        return datetime.fromisoformat(job.finished) - datetime.fromisoformat(
            job.started
        )

    def start(self, request: Request, adjusted: pd.DataFrame) -> bool:
        del adjusted
        if self.running:
            return False
        self.job = replace(
            DEMO_LAST,
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
