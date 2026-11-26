from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

import finagent

if TYPE_CHECKING:
    import pytest

APP = Path(finagent.__file__).parent / "app.py"
PAGES = ("economy", "monetary", "finance", "schedule", "etf", "agent", "portfolio")
WITHOUT_CHAT = {"agent", "portfolio"}
STEP = 2


def test_demo_renders_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["app", "--demo"])
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    for path in PAGES:
        # Streamlit AppTest 페이지 전환용 내부 해시 직접 설정
        at._page_hash = calc_hash(path)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert not at.run().exception, path
        toggles = [b for b in at.button if b.key == "ai_toggle"]
        assert len(toggles) == (path not in WITHOUT_CHAT), path
        if path == "agent":
            _ = at.chat_input(key="ai_input").set_value("/foo").run()
            assert "알 수 없는 명령: /foo" in at.error[0].value
    for section in at.button_group(key="pf_section").options:
        _ = at.button_group(key="pf_section").set_value(section)
        assert not at.run().exception, section
    _ = at.button_group(key="pf_mode").set_value("investors")
    assert not at.run().exception, "investors"
    _ = at.button_group(key="pf_mode").set_value("stock")
    held = at.run().button_group(key="stock_held")
    _ = held.set_value(held.options[0])
    assert "기업 정보" in at.run().button_group(key="stock_section").options
    for section in at.run().button_group(key="stock_section").options:
        _ = at.button_group(key="stock_section").set_value(section)
        assert not at.run().exception, section


def test_a_report_thread_shows_its_workflow_not_agent_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finagent import display
    from finagent.ui import shared
    from tests.test_analysis_workflow import MACRO_CALL, Team, run
    from tests.test_display_flow import board_of, upto

    events = asyncio.run(run(Team(MACRO_CALL)))
    monkeypatch.setattr(sys, "argv", ["app", "--demo"])
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at._page_hash = calc_hash("agent")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    _ = at.run()
    try:
        shared.app.rt.workspace.board = board_of(upto(events, "t1-3", "turn"))
        at.session_state["ai_thread"] = "t1"
        _ = at.run()
        assert not at.exception
        progress = at.get("progress")
        bars = [str(e.proto.text) for e in progress]  # pyright: ignore[reportAny]
        assert "수집 · 에이전트 3개 작업 중 · 1/11" in bars
        assert "" in bars
        steps = [e.label for e in at.expander if e.proto.type == STEP]
        assert steps == [
            "준비 · 완료",
            "수집 · 작업 중 0/3",
            "토론 1라운드 · 대기",
            "토론 2라운드 · 대기",
            "토론 3라운드 · 대기",
            "종합 · 대기",
            "보고서 · 대기",
        ]
        assert at.button(key="ai_card_list_t1").label == "매크로 분석 · 수집 · 1/11"
        assert not [b for b in at.button if (b.key or "").startswith("ai_card_team")]

        _ = at.button_group(key="ai_pick_t1_collect").set_value("기술").run()
        assert not at.exception
        assert "PART V" in [caption.value for caption in at.caption]

        at.session_state["ai_thread"] = "t1-2"
        _ = at.run()
        assert not at.exception
        assert at.session_state["ai_thread"] == "t1"

        shared.app.rt.workspace.board = board_of(events)
        _ = at.run()
        assert not at.exception
        assert any(e.label.startswith("작업 과정 · 에이전트 6개") for e in at.expander)
        assert not at.get("progress")
        steps = [e.label for e in at.expander if e.proto.type == STEP]
        assert all(label.endswith("· 완료") for label in steps)
        assert at.button(key="ai_back").proto.help == "메인 대화 복귀"
        assert display.DELIVERED in [caption.value for caption in at.caption]
        assert not [g for g in at.button_group if g.key == "ai_pick_t1_synth"]

        _ = at.button_group(key="ai_pick_t1_round1").set_value("강세").run()
        assert not at.exception
        said = " ".join(str(m.value) for m in at.markdown)
        assert "bull 1 발언" in said
        assert "bull 3 발언" not in said
    finally:
        shared.app.rt.renew()


def test_failed_and_stopped_reports_render_every_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finagent.ui import shared
    from tests.test_analysis_workflow import MACRO_CALL, Team, run
    from tests.test_display_flow import SIDES, board_of, stop_when

    def debating(role: str, number: int) -> bool:
        return role in SIDES and number >= 2

    runs = {
        "failed": asyncio.run(run(Team(MACRO_CALL, failing=("bull", "synth")))),
        "stopped": asyncio.run(stop_when(Team(MACRO_CALL, held=debating), "debate", 2)),
    }
    monkeypatch.setattr(sys, "argv", ["app", "--demo"])
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at._page_hash = calc_hash("agent")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    _ = at.run()
    try:
        for name, events in runs.items():
            shared.app.rt.workspace.board = board_of(events)
            at.session_state["ai_thread"] = "t1"
            _ = at.run()
            assert not at.exception, name
            captions = [caption.value for caption in at.caption]
            assert any(text.startswith(("실패 · ", "중지 · ")) for text in captions)
            keys = [g.key for g in at.button_group if g.key]
            for key in [key for key in keys if key.startswith("ai_pick_t1")]:
                for option in at.button_group(key=key).options:
                    _ = at.button_group(key=key).set_value(option).run()
                    assert not at.exception, (name, key, option)
    finally:
        shared.app.rt.renew()
