"""종목 분석 참고문헌(References) 로드·금칙어 필터링 및 가이드 XML 조립."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from finagent import layout

if TYPE_CHECKING:
    from pathlib import Path

DATA_DIR: Final = layout.REFERENCES_DIR / "company-scouting-manager" / "references"
FILES: Final = (
    "analysis-playbook",
    "scoring",
    "data-rules",
    "analysis-coverage",
    "fundamental-scouting",
    "advanced-modules",
)
BANNED: Final = (
    "매니저",
    "여고생",
    "야구",
    "캐릭터",
    "말투",
    "비유",
    "면책",
    "정보 제공 목적",
    "권유",
    "주문 지시",
    "사용자에게 질문",
    "다음 파트",
    "업로드",
    "ChatGPT",
    "MCP",
)
SENTENCE_END: Final = re.compile(r"(?<=[.!?。])\s+(?=\S)")


def available(root: Path) -> bool:
    return all((root / f"{name}.md").is_file() for name in FILES)


def _clean(text: str) -> str:
    kept = [
        part
        for part in SENTENCE_END.split(text)
        if not any(word in part for word in BANNED)
    ]
    return " ".join(kept)


def filter_text(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            out.append("|".join(_clean(cell) for cell in line.split("|")))
            continue
        cleaned = _clean(line)
        if cleaned or not line.strip():
            out.append(cleaned)
    return "\n".join(out)


def guide(root: Path) -> str:
    parts = [
        f'<reference name="{name}">\n'
        + filter_text((root / f"{name}.md").read_text(encoding="utf-8"))
        + "\n</reference>"
        for name in FILES
    ]
    return "<analysis_guide>\n" + "\n".join(parts) + "\n</analysis_guide>"
