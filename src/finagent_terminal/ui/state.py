from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

WorkspaceId = Literal["markets", "ticker", "tools", "reports", "settings"]
ConfigStatus = Literal["defaults", "loaded"]
ReportJobStatus = Literal["not_started", "running", "complete", "failed"]

WORKSPACES: Final[tuple[WorkspaceId, ...]] = (
    "markets",
    "ticker",
    "tools",
    "reports",
    "settings",
)
WORKSPACE_LABELS: Final[tuple[str, ...]] = (
    "Markets",
    "Ticker",
    "Tools",
    "Reports",
    "Settings",
)
WORKSPACE_BY_TAB_ID: Final[Mapping[str, WorkspaceId]] = MappingProxyType(
    {
        "markets": "markets",
        "ticker": "ticker",
        "tools": "tools",
        "reports": "reports",
        "settings": "settings",
    }
)
DEFAULT_SYMBOLS: Final[tuple[str, ...]] = ("SPY", "QQQ", "DIA")


@dataclass(frozen=True, slots=True)
class ConfigState:
    path: Path
    status: ConfigStatus


@dataclass(frozen=True, slots=True)
class ReportProgress:
    status: ReportJobStatus
    label: str
    completed: int
    total: int


@dataclass(frozen=True, slots=True)
class SessionCache:
    market_data_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AppState:
    config: ConfigState
    session_cache: SessionCache
    selected_symbols: tuple[str, ...]
    active_report: ReportProgress


def default_config_path() -> Path:
    finagent_home = os.environ.get("FINAGENT_HOME")
    root = (
        Path(finagent_home).expanduser()
        if finagent_home
        else Path.home() / ".finagent_terminal"
    )
    return root / "config.json"


def create_app_state() -> AppState:
    return AppState(
        config=ConfigState(path=default_config_path(), status="defaults"),
        session_cache=SessionCache(market_data_keys=()),
        selected_symbols=DEFAULT_SYMBOLS,
        active_report=ReportProgress(
            status="not_started",
            label="No report running",
            completed=0,
            total=0,
        ),
    )
