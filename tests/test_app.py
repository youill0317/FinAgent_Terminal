"""The page in demo mode, run headless with Streamlit's AppTest."""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, cast

from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

from finagent.cli import APP

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from finagent.display import AnalysisLog


def page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    settings = tmp_path / "finagent.settings.json"
    monkeypatch.setattr(sys, "argv", ["app.py", "--demo", f"--settings={settings}"])
    return AppTest.from_file(str(APP), default_timeout=60).run()


def open_page(at: AppTest, url_path: str) -> AppTest:
    """Go to a page of the app's own menu.

    ``AppTest.switch_page`` only takes page files; these pages are functions,
    which Streamlit keys by the hash of their url path.
    """
    at._page_hash = calc_hash(url_path)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    return at.run()


def button(at: AppTest, label: str) -> None:
    _ = next(b for b in at.button if b.label == label).click().run()


def test_the_dashboard_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = page(tmp_path, monkeypatch)

    assert not at.exception
    labels = [metric.label for metric in at.metric]
    assert labels[:5] == ["물가", "고용", "금리곡선", "금융여건", "연준 상단"]
    assert at.info[0].value.startswith("데모 모드")
    headers = [header.value for header in at.header]
    assert headers[:3] == ["국면 요약", "시장", "금리곡선"]

    _ = open_page(at, "us")
    assert not at.exception
    assert "미국 고용" in [header.value for header in at.subheader]

    _ = open_page(at, "schedule")
    assert not at.exception
    assert "앞으로 7일 미국 발표" in [header.value for header in at.subheader]
    assert any(
        c.value.startswith("This product uses the FRED® API") for c in at.caption
    )
    assert any("-badge[FRED " in m.value for m in at.markdown)


def test_a_panel_switches_to_its_deep_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = open_page(page(tmp_path, monkeypatch), "us")
    before = sum(len(frame.value) for frame in at.dataframe)

    _ = at.button(key="more_growth").click().run()

    assert not at.exception
    assert at.button(key="more_growth").label == "핵심만 보기"
    assert sum(len(frame.value) for frame in at.dataframe) > before
    assert '"growth"' in (tmp_path / "finagent.settings.json").read_text("utf-8")


def click_row(at: AppTest, panel: str, index: int = 0, column: str = "지표") -> None:
    """Stand in for a click on a table cell, which AppTest cannot send."""
    key = f"table_{panel}_{at.session_state.get('table_gen', 0)}"
    at.session_state[key] = {
        "selection": {"rows": [], "columns": [], "cells": [[index, column]]}
    }
    _ = at.run()


def test_a_clicked_row_opens_its_detail_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = open_page(page(tmp_path, monkeypatch), "us")
    assert not at.metric

    click_row(at, "growth", column="최신")  # only the name cell opens a detail
    assert not at.exception
    assert not at.metric

    click_row(at, "growth")
    assert not at.exception
    assert [m.label for m in at.metric] == ["최신", "이전", "기준"]
    assert len(at.get("vega_lite_chart")) == 1
    assert any(c.value.startswith("FRED:") for c in at.caption)

    _ = at.run()  # an unrelated rerun does not bring the dialog back
    assert not at.metric
    _ = at.button(key="more_growth").click().run()
    assert not at.metric

    click_row(at, "growth")  # the same row again opens it again
    assert [m.label for m in at.metric] == ["최신", "이전", "기준"]


def finish_run(at: AppTest) -> None:
    pause = threading.Event()
    for _ in range(100):
        if at.session_state["ai_run"] is None:
            break
        _ = pause.wait(0.1)
        _ = at.run()
    _ = at.run()


def test_an_analysis_runs_to_a_verified_answer_and_takes_a_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = page(tmp_path, monkeypatch)
    _ = at.button(key="ai_toggle").click().run()
    _ = at.chat_input[0].set_value("요약해 주세요").run()
    finish_run(at)

    assert not at.exception
    log = cast("AnalysisLog", at.session_state["ai_log"])
    assert log.conversation
    assert log.turns[-1].lines is not None
    first = log.conversation
    assert not at.chat_input[0].disabled

    _ = at.chat_input[0].set_value("더 설명해 주세요").run()
    finish_run(at)

    assert not at.exception
    log = cast("AnalysisLog", at.session_state["ai_log"])
    assert [turn.question for turn in log.turns] == [
        "요약해 주세요",
        "더 설명해 주세요",
    ]
    assert log.conversation == first
    assert all(turn.lines is not None for turn in log.turns)


def test_the_macro_preset_button_runs_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = page(tmp_path, monkeypatch)
    _ = at.button(key="ai_toggle").click().run()
    _ = at.button(key="ai_preset_macro").click().run()
    finish_run(at)

    assert not at.exception
    log = cast("AnalysisLog", at.session_state["ai_log"])
    assert log.turns[-1].lines is not None
    assert log.turns[0].question.startswith("<task>")
    assert "ai_preset_macro" not in [b.key for b in at.button]  # the input takes over


def test_the_settings_dialog_groups_views_by_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = page(tmp_path, monkeypatch)
    button(at, "설정")

    assert not at.exception
    labels = [expander.label for expander in at.expander]
    assert labels[0] == "개요 · 0/1"
    assert labels[1].startswith("미국 경제 · 0/")
    assert at.checkbox(key="view_growth").value is False
    assert "AI 분析" not in labels
    assert "AI 분석" in labels
    assert "적용" in [b.label for b in at.button]


def test_the_settings_dialog_holds_the_ai_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AppTest reruns the whole script on each change and the dialog is only
    # drawn on the run of the 설정 click, so applying cannot be driven here.
    at = page(tmp_path, monkeypatch)
    button(at, "설정")

    assert not at.exception
    provider = at.selectbox(key="set_provider")
    assert provider.options
    assert at.text_input(key=f"set_model_{provider.value}")
    assert "set_scope" in [w.key for w in at.get("button_group")]
    _ = at.button(key="ai_toggle").click().run()
    assert "set_provider" not in [w.key for w in at.selectbox]
    assert "ai_scope" not in [w.key for w in at.get("button_group")]
