from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from finagent import catalog
from finagent.limits import LIMITS

ROOT: Final = Path(__file__).resolve().parents[1]
SRC: Final = ROOT / "src" / "finagent"
KEY_NAME: Final = r"\b(?:fred|ecos|alpaca|apca)[a-z0-9_]*(?:key|token|secret)?\s*=\s*"
KEY_VALUE: Final = r"['\"][A-Za-z0-9+/=?:&._-]{20,}['\"]"
HARDCODED_KEY_PATTERNS: Final = (
    re.compile(KEY_NAME + KEY_VALUE, re.IGNORECASE),
    re.compile(r"\b(?:api_key|apikey|token)=[A-Za-z0-9+/=?:&._-]{20,}"),
    re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
)


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _hits(pattern: re.Pattern[str], paths: list[Path]) -> list[str]:
    return [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in paths
        for number, line in enumerate(path.read_text("utf-8").splitlines(), 1)
        if pattern.search(line)
    ]


def test_no_hardcoded_keys() -> None:
    for pattern in HARDCODED_KEY_PATTERNS:
        assert _hits(pattern, _python_files()) == []
    value = '"abcdefghijklmnopqrstuvwxyz0123"'
    for name in ("FRED_API_KEY", "ALPACA_API_SECRET_KEY"):
        line = f"{name} = {value}"
        assert any(p.search(line) for p in HARDCODED_KEY_PATTERNS), line


def test_catalog_integrity() -> None:
    indicators = [ind for spec in catalog.PANELS for ind in spec.indicators]
    ids = [indicator.id for indicator in indicators]

    assert len(ids) == len(set(ids))
    assert {i for _, group in catalog.TAPE_GROUPS for i in group} <= set(ids)
    assert {indicator.fetch.source for indicator in indicators} <= set(LIMITS)
