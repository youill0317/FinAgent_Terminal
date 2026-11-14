"""데이터·사용자 설정·참고문헌 디렉터리 경로 상수 및 원자적 파일 쓰기 유틸리티."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Final

DATA: Final = Path("data")
USER_DIR: Final = DATA / "user"
REFERENCES_DIR: Final = DATA / "references"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    _ = temp.write_text(text, encoding="utf-8")
    _ = temp.replace(path)
