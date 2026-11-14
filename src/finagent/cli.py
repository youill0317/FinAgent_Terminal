"""FinAgent 로컬 터미널 CLI 실행 및 공급자 수신 점검 진입점."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from finagent import (
    __version__,
    investors,
    limits,
    markets,
    prices,
    settings,
)
from finagent.analysis.service import AnalysisService
from finagent.demo import offline
from finagent.service import DashboardService

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from finagent.analysis.providers.base import ProviderStatus
    from finagent.models import Dashboard

HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8765
DEFAULT_ENV_FILE: Final = Path(".env")
DEFAULT_SETTINGS_FILE: Final = Path("finagent.settings.json")
APP: Final = Path(__file__).with_name("app.py")


class Options(argparse.Namespace):
    port: int = DEFAULT_PORT
    no_browser: bool = False
    demo: bool = False
    check: bool = False
    env_file: Path | None = None
    settings: Path | None = None


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:  # noqa: PLR2004  # 따옴표 쌍 최소 길이(2자) 기준
            value = value[1:-1]
        values[key.strip()] = value
    return values


def resolve_env(env_file: Path | None, environ: Mapping[str, str]) -> dict[str, str]:
    path = env_file if env_file is not None else DEFAULT_ENV_FILE
    if env_file is not None and not path.is_file():
        raise SystemExit(f"환경설정 파일 미확인: {path}")
    from_file = read_env_file(path) if path.is_file() else {}
    return {**from_file, **environ}


def resolve_settings(settings: Path | None) -> Path:
    return (settings if settings is not None else DEFAULT_SETTINGS_FILE).absolute()


def configure_sources(env: Mapping[str, str]) -> None:
    figi = env.get(investors.FIGI_KEY, "").strip()
    table = limits.FIGI_KEYED_LIMITS if figi else limits.LIMITS
    _ = limits.configure(limits.Limiter(limits=table))
    investors.set_figi_key(figi)
    _ = prices.configure(prices.PriceStore(markets.keys(env)))


def build_service(*, demo: bool, env: Mapping[str, str]) -> DashboardService:
    if not demo:
        return DashboardService(env)
    return DashboardService({}, client_factory=offline, demo=True)


def parse_args(argv: Sequence[str] | None) -> Options:
    parser = argparse.ArgumentParser(
        prog="macro",
        description="거시 지표·ETF·포트폴리오 통합 로컬 투자 터미널 실행",
    )
    _ = parser.add_argument("--version", action="version", version=__version__)
    _ = parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    _ = parser.add_argument(
        "--no-browser", action="store_true", help="브라우저 자동 실행 생략"
    )
    _ = parser.add_argument(
        "--demo", action="store_true", help="API 키 없이 데모 데이터로 실행"
    )
    _ = parser.add_argument(
        "--check",
        action="store_true",
        help="서버 기동 없이 데이터 공급자별 수신 상태 점검 및 출력",
    )
    _ = parser.add_argument(
        "--env-file",
        type=Path,
        help="API 키 환경변수 파일 경로 (기본: 현재 폴더의 .env)",
    )
    _ = parser.add_argument(
        "--settings",
        type=Path,
        help="설정 파일 경로 (기본: 현재 폴더의 finagent.settings.json)",
    )
    return parser.parse_args(argv, namespace=Options())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env = resolve_env(args.env_file, os.environ)
    settings_path = resolve_settings(args.settings)
    if args.check:
        try:
            settings.check(settings_path)
        except settings.SettingsError as error:
            print(error)  # noqa: T201  # CLI 설정 점검 오류 표준 출력
            return 1
        configure_sources(env)
        service = build_service(demo=args.demo, env=env)
        analysis = AnalysisService(service, env, settings_path=settings_path)
        return run_check(service, analysis)

    # CLI 인자 검증 후 지연 임포트
    from streamlit.web import cli as streamlit_cli  # noqa: PLC0415

    url = f"http://{HOST}:{args.port}/"
    print(f"FinAgent {__version__}: {url}  (종료: Ctrl+C)")  # noqa: T201  # CLI 기동 주소 안내 출력
    if not args.demo or args.settings is not None:
        print(f"설정: {settings_path}")  # noqa: T201  # CLI 활성 설정 경로 출력
    command = streamlit_args(args, settings_path)
    streamlit_cli.main(command, prog_name="streamlit", standalone_mode=False)
    return 0


def streamlit_args(args: Options, settings_path: Path) -> list[str]:
    own = args.demo and args.settings is None
    script = [] if own else [f"--settings={settings_path}"]
    if args.demo:
        script.append("--demo")
    if args.env_file is not None:
        script.append(f"--env-file={args.env_file.absolute()}")
    return [
        "run",
        str(APP),
        f"--server.address={HOST}",
        f"--server.port={args.port}",
        f"--server.headless={str(args.no_browser).lower()}",
        "--server.showEmailPrompt=false",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=minimal",
        "--",
        *script,
    ]


def run_check(service: DashboardService, analysis: AnalysisService) -> int:
    dashboard = asyncio.run(service.get())
    print(check_report(dashboard))  # noqa: T201  # CLI 지표 수신 상태 출력
    print(provider_report(asyncio.run(analysis.statuses())))  # noqa: T201  # CLI AI 공급자 상태 출력
    return 0 if all(source.state == "ok" for source in dashboard.sources) else 1


def provider_report(statuses: Mapping[str, ProviderStatus]) -> str:
    lines = ["AI 공급자 (선택 기능):"]
    for provider_id, status in statuses.items():
        note = f"  {status.message}" if status.message else ""
        lines.append(f"  {provider_id:<11}{status.kind:<14}{note}")
    return "\n".join(lines)


def check_report(dashboard: Dashboard) -> str:
    lines = [f"갱신 {dashboard.generated_at} · {dashboard.elapsed_seconds:.1f}s"]
    for source in dashboard.sources:
        note = f"  {source.message}" if source.message else ""
        lines.append(
            f"{source.name:<14}{source.state:<12}{source.ok}/{source.total}{note}"
        )
    sep = dashboard.projections
    if sep is not None:
        state = f"{sep.published} 발표" if sep.rows else "error"
        note = f"  {sep.message}" if sep.message else ""
        lines.append(f"{'FOMC SEP':<14}{state}{note}")
    problems = [
        f"  {row.code}: {row.message}"
        for panel in dashboard.panels
        for row in panel.rows
        if row.status != "ok"
    ]
    if problems:
        lines.append("수신 실패 지표:")
        lines.extend(problems)
    return "\n".join(lines)
