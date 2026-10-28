"""The saved view: which panels are shown in full, and the chosen AI provider.

This is the only file the app writes. It holds view ids and, once an AI
provider is picked, that provider's id and model names, the last
generator-training choices (tickers, dates, GPU, models) and the screening
criteria that differ from the defaults. It never holds
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
SCOPES: Final = ("view", "all")
DEFAULT_SCOPE: Final = "view"


@dataclass(frozen=True, slots=True)
class AnalysisChoice:
    provider: str
    models: dict[str, str] = field(default_factory=dict)
    scope: str = DEFAULT_SCOPE
    instructions: str = ""  # the user's system prompt


@dataclass(frozen=True, slots=True)
class TrainingChoice:
    """The 생성 모델 학습 choices; dates are ISO strings, empty for the default."""

    tickers: tuple[str, ...] = ()
    start: str = ""
    end: str = ""
    gpu: str = ""
    models: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScreenChoice:
    """The 스크리닝 기준 switches flipped from their defaults.

    ``off`` holds the ids; ``screen.criteria`` drops unknown ones.
    """

    off: tuple[str, ...] = ()


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
    scope = section.get("scope")
    if not isinstance(scope, str) or scope not in SCOPES:
        scope = DEFAULT_SCOPE
    instructions = section.get("instructions")
    if not isinstance(instructions, str):
        instructions = ""
    return AnalysisChoice(provider, clean, scope, instructions)


def load_training(path: Path) -> TrainingChoice | None:
    """The saved training choices, or ``None`` when there are none."""
    raw = _read(path).get("training")
    if not isinstance(raw, dict):
        return None
    section = cast("dict[str, object]", raw)

    def text(key: str) -> str:
        value = section.get(key)
        return value if isinstance(value, str) else ""

    def names(key: str) -> tuple[str, ...]:
        value = section.get(key)
        if not isinstance(value, list):
            return ()
        return tuple(v for v in cast("list[object]", value) if isinstance(v, str))

    return TrainingChoice(
        names("tickers"), text("start"), text("end"), text("gpu"), names("models")
    )


def load_screen(path: Path) -> ScreenChoice:
    """The saved screening switches; ids of other types are dropped."""
    raw = _read(path).get("screen")
    if not isinstance(raw, dict):
        return ScreenChoice()
    off = cast("dict[str, object]", raw).get("off")
    if not isinstance(off, list):
        return ScreenChoice()
    return ScreenChoice(
        tuple(v for v in cast("list[object]", off) if isinstance(v, str))
    )


def save_view(path: Path, deep: frozenset[str]) -> None:
    """Write the view, keeping any saved provider and training choices.

    Raises ``OSError`` if the file cannot be written.
    """
    ordered = [view.id for view in catalog.VIEWS if view.id in deep]
    body: dict[str, object] = {"version": SETTINGS_VERSION, "deep": ordered}
    _write(path, _kept(body, path, "analysis", "training", "screen"))


def save_analysis(path: Path, choice: AnalysisChoice) -> None:
    """Write the provider choice, keeping the saved view and training choices."""
    view, _ = load_view(path)
    ordered = [item.id for item in catalog.VIEWS if item.id in view]
    body: dict[str, object] = {
        "version": SETTINGS_VERSION,
        "deep": ordered,
        "analysis": {
            "provider": choice.provider,
            "models": dict(choice.models),
            "scope": choice.scope,
            "instructions": choice.instructions,
        },
    }
    _write(path, _kept(body, path, "training", "screen"))


def save_training(path: Path, choice: TrainingChoice) -> None:
    """Write the training choices, keeping the saved view and provider."""
    view, _ = load_view(path)
    ordered = [item.id for item in catalog.VIEWS if item.id in view]
    body: dict[str, object] = {"version": SETTINGS_VERSION, "deep": ordered}
    body = _kept(body, path, "analysis")
    body["training"] = {
        "tickers": list(choice.tickers),
        "start": choice.start,
        "end": choice.end,
        "gpu": choice.gpu,
        "models": list(choice.models),
    }
    _write(path, _kept(body, path, "screen"))


def save_screen(path: Path, choice: ScreenChoice) -> None:
    """Write the screening criteria, keeping the rest; none clears the section."""
    view, _ = load_view(path)
    ordered = [item.id for item in catalog.VIEWS if item.id in view]
    body: dict[str, object] = {"version": SETTINGS_VERSION, "deep": ordered}
    body = _kept(body, path, "analysis", "training")
    if choice.off:
        body["screen"] = {"off": list(choice.off)}
    _write(path, body)


def _kept(body: dict[str, object], path: Path, *keys: str) -> dict[str, object]:
    """``body`` plus the saved sections ``keys``, where the file has them."""
    saved = _read(path)
    for key in keys:
        section = saved.get(key)
        if isinstance(section, dict):
            body[key] = section
    return body


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
