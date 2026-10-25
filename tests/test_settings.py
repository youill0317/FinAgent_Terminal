from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import httpx

from finagent import catalog
from finagent.demo import DEMO_KEYS, demo_closes, demo_transport
from finagent.service import DashboardService, local_now
from finagent.settings import (
    AnalysisChoice,
    ScreenChoice,
    TrainingChoice,
    load_analysis,
    load_screen,
    load_training,
    load_view,
    save_analysis,
    save_screen,
    save_training,
    save_view,
)

if TYPE_CHECKING:
    from pathlib import Path


def service(path: Path) -> DashboardService:
    transport = demo_transport(local_now().date())
    return DashboardService(
        DEMO_KEYS,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        retry_delay=0,
        yahoo=demo_closes(local_now().date()),
        settings_path=path,
    )


def test_missing_file_means_the_default_view(tmp_path: Path) -> None:
    path = tmp_path / "finagent" / "settings.json"

    assert load_view(path) == (catalog.CORE_VIEW, None)
    dashboard = asyncio.run(service(path).get())
    assert dashboard.deep == ()
    assert dashboard.settings_file == str(path)
    assert not path.exists()


def test_a_changed_view_is_saved_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "finagent" / "settings.json"

    dashboard = asyncio.run(service(path).set_view(frozenset({"curve", "labor"})))

    assert dashboard.deep == ("labor", "curve")
    assert json.loads(path.read_text("utf-8")) == {
        "version": 1,
        "deep": ["labor", "curve"],
    }
    assert sorted(p.name for p in path.parent.iterdir()) == ["settings.json"]
    restarted = service(path)
    assert restarted.view == frozenset({"labor", "curve"})
    assert asyncio.run(restarted.get()).deep == ("labor", "curve")


def test_unchanged_view_is_not_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"

    _ = asyncio.run(service(path).set_view(catalog.CORE_VIEW))

    assert not path.exists()


def test_unreadable_file_falls_back_to_the_default(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    for body in ("{not json", '{"deep": "labor"}', "[1, 2]"):
        _ = path.write_text(body, encoding="utf-8")
        view, error = load_view(path)
        assert view == catalog.CORE_VIEW
        assert error is not None
    assert asyncio.run(service(path).get()).settings_error == error
    # Unknown ids in a readable file are dropped, not an error.
    _ = path.write_text('{"deep": ["housing", "gone", 3]}', encoding="utf-8")
    assert load_view(path) == (frozenset({"housing"}), None)


def test_a_failed_save_keeps_the_view_and_says_so(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    _ = blocker.write_text("", encoding="utf-8")
    path = blocker / "settings.json"

    dashboard = asyncio.run(service(path).set_view(frozenset({"housing"})))

    assert dashboard.deep == ("housing",)
    assert dashboard.settings_error is not None
    assert dashboard.settings_error.startswith("설정을 저장하지 못했습니다")


def test_screening_criteria_are_saved_and_kept_by_other_saves(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    choice = ScreenChoice({"split": 0.6, "min_bars": 300}, ("q_roe", "o_rsi"))

    save_screen(path, choice)
    assert load_screen(path) == choice
    save_view(path, frozenset({"labor"}))
    save_analysis(path, AnalysisChoice("demo"))
    save_training(path, TrainingChoice(("SPY",)))
    assert load_screen(path) == choice
    assert load_view(path)[0] == frozenset({"labor"})
    assert load_analysis(path) == AnalysisChoice("demo")

    save_screen(path, ScreenChoice())  # the defaults again
    assert "screen" not in json.loads(path.read_text("utf-8"))
    assert load_training(path) == TrainingChoice(("SPY",))


def test_screening_criteria_of_the_wrong_type_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    body = {"screen": {"values": {"split": "0.6", "min_bars": 300, "x": True}}}
    _ = path.write_text(json.dumps(body), encoding="utf-8")
    assert load_screen(path) == ScreenChoice({"min_bars": 300})

    for section in ([1], {"values": [1], "off": "q_roe"}, {"off": ["q_roe", 3]}):
        _ = path.write_text(json.dumps({"screen": section}), encoding="utf-8")
        assert load_screen(path).values == {}
    assert load_screen(path).off == ("q_roe",)
    _ = path.write_text("{not json", encoding="utf-8")
    assert load_screen(path) == ScreenChoice()
