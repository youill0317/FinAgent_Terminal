"""종목 분석 참고문헌(References) 로드 및 가이드 XML 조립."""

from __future__ import annotations

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


def available(root: Path) -> bool:
    return all((root / f"{name}.md").is_file() for name in FILES)


def guide(root: Path) -> str:
    parts = [
        f'<reference name="{name}">\n'
        + (root / f"{name}.md").read_text(encoding="utf-8").strip()
        + "\n</reference>"
        for name in FILES
    ]
    return "<analysis_guide>\n" + "\n".join(parts) + "\n</analysis_guide>"
