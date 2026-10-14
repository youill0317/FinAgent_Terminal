"""The saved view: which panels are shown in full, and the chosen AI provider.

This is the only file the app writes. It holds view ids and, once an AI
provider is picked, that provider's id and model names. It never holds
fetched data, conversations or API keys. A missing or unreadable file means
the default view (core rows only) and the default provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast

from finagent import catalog

if TYPE_CHECKING:
    from pathlib import Path

SETTINGS_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class AnalysisChoice:
    provider: str
    models: dict[str, str] = field(default_factory=dict)


def load_view(path: Path) -> tuple[frozenset[str], str | None]:
    """Return the saved view, or the default view and why the file was skipped."""
    if not path.is_file():
        return catalog.CORE_VIEW, None
    try:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return catalog.CORE_VIEW, f"설정 파일을 읽지 못해 기본 보기를 씁니다: {error}"
    deep = cast("dict[str, object]", raw).get("deep") if isinstance(raw, dict) else None
    if not isinstance(deep, list):
        return catalog.CORE_VIEW, "설정 파일 형식이 맞지 않아 기본 보기를 씁니다"
    ids = {item for item in cast("list[object]", deep) if isinstance(item, str)}
    return frozenset(ids) & catalog.ALL_VIEWS, None


def load_analysis(path: Path) -> AnalysisChoice | None:
    """The saved AI provider choice, or ``None`` when there is none."""
    raw = _read(path).get("analysis")
    if not isinstance(raw, dict):
        return None
    section = cast("dict[str, object]", raw)
    provider = section.get("provider")
    models = section.get("models")
    if not isinstance(provider, str):
        return None
    clean: dict[str, str] = {}
    if isinstance(models, dict):
        clean = {
            key: value
            for key, value in cast("dict[object, object]", models).items()
            if isinstance(key, str) and isinstance(value, str)
        }
    return AnalysisChoice(provider, clean)


def save_view(path: Path, deep: frozenset[str]) -> None:
    """Write the view, keeping any saved provider choice.

    Raises ``OSError`` if the file cannot be written.
    """
    ordered = [view.id for view in catalog.VIEWS if view.id in deep]
    body: dict[str, object] = {"version": SETTINGS_VERSION, "deep": ordered}
    analysis = _read(path).get("analysis")
    if isinstance(analysis, dict):
        body["analysis"] = analysis
    _write(path, body)


def save_analysis(path: Path, choice: AnalysisChoice) -> None:
    """Write the provider choice, keeping the saved view."""
    view, _ = load_view(path)
    ordered = [item.id for item in catalog.VIEWS if item.id in view]
    _write(
        path,
        {
            "version": SETTINGS_VERSION,
            "deep": ordered,
            "analysis": {"provider": choice.provider, "models": dict(choice.models)},
        },
    )


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return cast("dict[str, object]", raw) if isinstance(raw, dict) else {}


def _write(path: Path, body: dict[str, object]) -> None:
    """Write atomically; raises ``OSError`` if the file cannot be written."""
    text = json.dumps(body, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    _ = temp.write_text(text + "\n", encoding="utf-8")
    _ = temp.replace(path)
