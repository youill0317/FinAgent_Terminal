from __future__ import annotations

import argparse
import asyncio
import os
import threading
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx

from finagent import __version__
from finagent.demo import DEMO_KEYS, demo_transport
from finagent.service import DashboardService, local_now

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from finagent.models import Dashboard

HOST: Final = "127.0.0.1"  # local only; the dashboard is never exposed on the network
DEFAULT_PORT: Final = 8765
DEFAULT_ENV_FILE: Final = Path(".env")
BROWSER_DELAY_SECONDS: Final = 1.0


class Options(argparse.Namespace):
    port: int = DEFAULT_PORT
    no_browser: bool = False
    demo: bool = False
    check: bool = False
    env_file: Path | None = None


def read_env_file(path: Path) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines. Blank lines and ``#`` comments are skipped.

    The app only reads this file; it never writes it.
    """
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:  # noqa: PLR2004
            value = value[1:-1]
        values[key.strip()] = value
    return values


def resolve_env(env_file: Path | None, environ: Mapping[str, str]) -> dict[str, str]:
    """Keys from the env file, overridden by the real environment."""
    path = env_file if env_file is not None else DEFAULT_ENV_FILE
    if env_file is not None and not path.is_file():
        raise SystemExit(f"env 파일을 찾을 수 없습니다: {path}")
    from_file = read_env_file(path) if path.is_file() else {}
    return {**from_file, **environ}


def build_service(*, demo: bool, env: Mapping[str, str]) -> DashboardService:
    if not demo:
        return DashboardService(env)
    transport = demo_transport(local_now().date())
    return DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        demo=True,
    )


def parse_args(argv: Sequence[str] | None) -> Options:
    parser = argparse.ArgumentParser(
        prog="finagent",
        description="공공·공개 데이터 매크로 대시보드를 로컬 브라우저에 띄웁니다.",
    )
    _ = parser.add_argument("--version", action="version", version=__version__)
    _ = parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    _ = parser.add_argument(
        "--no-browser", action="store_true", help="브라우저를 열지 않음"
    )
    _ = parser.add_argument(
        "--demo", action="store_true", help="API 키 없이 예시 데이터로 실행"
    )
    _ = parser.add_argument(
        "--check",
        action="store_true",
        help="서버 없이 한 번 받아와서 출처별 상태만 출력",
    )
    _ = parser.add_argument(
        "--env-file",
        type=Path,
        help="FRED_API_KEY, ECOS_API_KEY를 읽을 파일 (기본: 현재 폴더의 .env)",
    )
    return parser.parse_args(argv, namespace=Options())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env = resolve_env(args.env_file, os.environ)
    service = build_service(demo=args.demo, env=env)

    if args.check:
        return run_check(service)

    import uvicorn  # noqa: PLC0415 - only needed when serving

    from finagent.server import create_app  # noqa: PLC0415

    url = f"http://{HOST}:{args.port}/"
    print(f"FinAgent {__version__}: {url}  (종료: Ctrl+C)")  # noqa: T201
    if not args.no_browser:
        threading.Timer(BROWSER_DELAY_SECONDS, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(service), host=HOST, port=args.port, log_level="warning")
    return 0


def run_check(service: DashboardService) -> int:
    dashboard = asyncio.run(service.get())
    print(check_report(dashboard))  # noqa: T201
    return 0 if all(source.state == "ok" for source in dashboard.sources) else 1


def check_report(dashboard: Dashboard) -> str:
    lines = [f"갱신 {dashboard.generated_at} · {dashboard.elapsed_seconds:.1f}s"]
    for source in dashboard.sources:
        note = f"  {source.message}" if source.message else ""
        lines.append(
            f"{source.name:<9}{source.state:<12}{source.ok}/{source.total}{note}"
        )
    problems = [
        f"  {row.code}: {row.message}"
        for panel in dashboard.panels
        for row in panel.rows
        if row.status != "ok"
    ]
    if problems:
        lines.append("받지 못한 지표:")
        lines.extend(problems)
    return "\n".join(lines)
