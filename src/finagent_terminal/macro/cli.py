from __future__ import annotations

import argparse
import asyncio
import os
import threading
import webbrowser
from typing import TYPE_CHECKING, Final

import httpx

from finagent_terminal.macro.demo import DEMO_KEYS, demo_transport
from finagent_terminal.macro.service import DashboardService, local_now

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finagent_terminal.macro.models import Dashboard

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8765
BROWSER_DELAY_SECONDS: Final = 1.0


def build_service(*, demo: bool) -> DashboardService:
    if not demo:
        return DashboardService(dict(os.environ))
    transport = demo_transport(local_now().date())
    return DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        demo=True,
    )


class Options(argparse.Namespace):
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    no_browser: bool = False
    demo: bool = False
    check: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="finagent-macro",
        description="공공·공개 데이터 매크로 대시보드를 로컬 브라우저에 띄웁니다.",
    )
    _ = parser.add_argument("--host", default=DEFAULT_HOST)
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
    args = parser.parse_args(argv, namespace=Options())
    service = build_service(demo=args.demo)

    if args.check:
        dashboard = asyncio.run(service.get())
        print(check_report(dashboard))  # noqa: T201
        return 0 if all(source.state == "ok" for source in dashboard.sources) else 1

    import uvicorn  # noqa: PLC0415 - only needed when serving

    from finagent_terminal.macro.server import create_app  # noqa: PLC0415

    url = f"http://{args.host}:{args.port}/"
    print(f"FinAgent Macro: {url}  (종료: Ctrl+C)")  # noqa: T201
    if not args.no_browser:
        threading.Timer(BROWSER_DELAY_SECONDS, webbrowser.open, args=(url,)).start()
    uvicorn.run(
        create_app(service), host=args.host, port=args.port, log_level="warning"
    )
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
