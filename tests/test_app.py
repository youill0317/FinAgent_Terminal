"""The page in demo mode, run headless with Streamlit's AppTest."""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, cast

from streamlit.testing.v1 import AppTest

from finagent.cli import APP

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from finagent.display import AnalysisLog


def page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    settings = tmp_path / "finagent.settings.json"
    monkeypatch.setattr(sys, "argv", ["app.py", "--demo", f"--settings={settings}"])
    return AppTest.from_file(str(APP), default_timeout=60).run()


def button(at: AppTest, label: str) -> None:
    _ = next(b for b in at.button if b.label == label).click().run()


def test_the_dashboard_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = page(tmp_path, monkeypatch)

    assert not at.exception
    titles = [header.value for header in at.subheader]
    assert "LABR · 미국 고용" in titles
    assert "CAL · 앞으로 7일 미국 발표" in titles
    assert at.metric[0].label == "연준 상단"
    assert at.info[0].value.startswith("데모 모드")


def test_a_panel_switches_to_its_deep_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = page(tmp_path, monkeypatch)
    before = sum(len(frame.value) for frame in at.dataframe)

    _ = at.button(key="more_growth").click().run()

    assert not at.exception
    assert at.button(key="more_growth").label == "핵심만"
    assert sum(len(frame.value) for frame in at.dataframe) > before
    assert '"growth"' in (tmp_path / "finagent.settings.json").read_text("utf-8")


def test_an_analysis_runs_to_a_verified_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = page(tmp_path, monkeypatch)
    button(at, "AI 분석")
    _ = at.text_area(key="ai_input").set_value("요약해 주세요").run()
    button(at, "실행")

    pause = threading.Event()
    for _ in range(100):
        if at.session_state["ai_run"] is None:
            break
        _ = pause.wait(0.1)
        _ = at.run()

    assert not at.exception
    log = cast("AnalysisLog", at.session_state["ai_log"])
    assert log.conversation
    assert log.turns[-1].lines is not None
    assert at.text_area(key="ai_input").value == ""
