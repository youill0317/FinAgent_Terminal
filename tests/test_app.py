from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

import finagent

if TYPE_CHECKING:
    import pytest

APP = Path(finagent.__file__).parent / "app.py"
PAGES = ("economy", "monetary", "finance", "schedule", "etf", "portfolio")


def test_demo_renders_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["app", "--demo"])
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    for path in PAGES:
        # Streamlit AppTest 페이지 전환용 내부 해시 직접 설정
        at._page_hash = calc_hash(path)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert not at.run().exception, path
    for section in at.button_group(key="pf_section").options:
        _ = at.button_group(key="pf_section").set_value(section)
        assert not at.run().exception, section
    _ = at.button_group(key="pf_mode").set_value("stock")
    held = at.run().button_group(key="stock_held")
    _ = held.set_value(held.options[0])
    for section in at.run().button_group(key="stock_section").options:
        _ = at.button_group(key="stock_section").set_value(section)
        assert not at.run().exception, section
