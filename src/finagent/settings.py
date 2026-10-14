"""The saved view: which panels are shown in full.

This is the only file the app writes. It holds view ids, never fetched data.
A missing or unreadable file means the default view (core rows only).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

from finagent import catalog

if TYPE_CHECKING:
    from pathlib import Path

SETTINGS_VERSION: Final = 1


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


def save_view(path: Path, deep: frozenset[str]) -> None:
    """Write the view atomically; raises ``OSError`` if the file cannot be written."""
    ordered = [view.id for view in catalog.VIEWS if view.id in deep]
    body = json.dumps(
        {"version": SETTINGS_VERSION, "deep": ordered}, ensure_ascii=False, indent=2
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    _ = temp.write_text(body + "\n", encoding="utf-8")
    _ = temp.replace(path)
