from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import httpx

from finagent import catalog
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
from tests.fake_sources import DEMO_KEYS, demo_closes, demo_transport

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


def test_screening_criteria_are_saved_and_kept_by_other_saves(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    choice = ScreenChoice(("q_roe", "dq_altman"))

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
