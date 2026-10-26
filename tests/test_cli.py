from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from finagent.cli import (
    main,
    read_env_file,
    resolve_env,
    run_check,
)
from finagent.demo import DEMO_KEYS, demo_transport
from finagent.service import DashboardService, local_now

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_env_file_parsing(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    lines = [
        "# keys",
        "FRED_API_KEY=abc123",
        "export ECOS_API_KEY = 'quoted value'",
        "",
        "NOT A PAIR",
        'EMPTY=""',
    ]
    _ = path.write_text("\n".join(lines), encoding="utf-8")

    assert read_env_file(path) == {
        "FRED_API_KEY": "abc123",
        "ECOS_API_KEY": "quoted value",
        "EMPTY": "",
    }


def test_real_environment_wins_over_env_file(tmp_path: Path) -> None:
    path = tmp_path / "keys.env"
    _ = path.write_text("FRED_API_KEY=from-file\nECOS_API_KEY=file-only\n", "utf-8")

    env = resolve_env(path, {"FRED_API_KEY": "from-shell"})

    assert env["FRED_API_KEY"] == "from-shell"
    assert env["ECOS_API_KEY"] == "file-only"


def test_check_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--demo", "--check"]) == 0
    assert "FRED" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []  # --check never writes settings

    demo = demo_transport(local_now().date())

    def ecb_down(request: httpx.Request) -> httpx.Response:
        if request.url.host == "data-api.ecb.europa.eu":
            return httpx.Response(500)
        return demo.handle_request(request)

    transport = httpx.MockTransport(ecb_down)
    failing = DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        retry_delay=0,
    )
    assert run_check(failing) == 1
    assert "ECB:EXR/USD: ECB HTTP 500" in capsys.readouterr().out
