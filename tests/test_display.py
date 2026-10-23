from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from finagent import catalog, display
from finagent.models import Fmt, IndicatorRow, Panel, ProjectionCell

if TYPE_CHECKING:
    from finagent.analysis.service import Event


def row(
    value: float | None,
    prev: float | None,
    *,
    dec: int = 2,
    freq: str = "M",
    day: str = "2026-09",
) -> IndicatorRow:
    return IndicatorRow.model_validate(
        {
            "id": "x",
            "label": "x",
            "code": "FRED:X",
            "source": "FRED",
            "freq": freq,
            "fmt": {"dec": dec},
            "status": "ok",
            "value": value,
            "prev": prev,
            "date": day,
        }
    )


def test_numbers_use_the_typographic_minus() -> None:
    f = Fmt(suf="%", dec=2, signed=True)

    assert display.fmt(-1.234, f) == "−1.23%"
    assert display.fmt(1234.5, f) == "+1,234.50%"
    assert display.fmt(None, f) == "—"


def test_change_rounds_away_below_the_shown_decimals() -> None:
    assert display.change_text(row(4.1, 4.104)) == "— 0"
    assert display.change_text(row(4.2, 4.1)) == "▲ 0.10"
    assert display.change_text(row(4.0, 4.1)) == "▼ 0.10"
    assert display.change_text(row(4.0, None)) == ""
    assert display.metric_delta(row(4.0, 4.1)) == "-0.10"
    assert display.metric_delta(row(4.0, None)) is None


def test_copyrighted_rows_carry_their_source_on_the_label() -> None:
    plain = row(1, 1).model_copy(update={"label": "A"})
    owned = row(1, 1).model_copy(update={"label": "S&P 500", "copyright": "S&P DJI"})
    panel = Panel(id="p", key="p", title="p", note="", rows=(plain, owned))

    table = display.panel_table(panel)

    assert "코드" not in table
    assert table["지표"] == ["A", "S&P 500 © S&P DJI"]


def test_short_dates_follow_the_frequency() -> None:
    assert display.short_date(row(1, 1, freq="M")) == "26.09"
    assert display.short_date(row(1, 1, freq="Q", day="2026-Q3")) == "26-Q3"
    assert display.short_date(row(1, 1, freq="D", day="2026-09-01")) == "09/01"
    assert display.weekday("2026-10-12") == "10/12 월"


def test_sep_cells_show_the_move_since_the_prior_release() -> None:
    new = ProjectionCell(value=2.1, prev=None)
    flat = ProjectionCell(value=2.1, prev=2.12)
    up = ProjectionCell(value=2.4, prev=2.1)
    down = ProjectionCell(value=1.6, prev=1.8)
    assert [display.sep_value(c) for c in (new, flat, up, None)] == [
        "2.1",
        "2.1",
        "2.4",
        "—",
    ]
    assert [display.sep_change(c) for c in (new, flat, up, down, None)] == [
        "신규",
        "—",
        "▲0.3",
        "▼0.2",
        "",
    ]


def test_arrows_are_red_up_and_blue_down() -> None:
    assert display.arrow_color("▲0.10") == display.UP_COLOR
    assert display.arrow_color("▼0.3") == display.DOWN_COLOR
    assert display.arrow_color("—") is None
    assert display.arrow_color("") is None


def test_live_text_hides_citation_marks_and_dollar_math() -> None:
    raw = "금값은 $2,400이다 [[wti]] 그리고 [[us_"

    assert display.live_text(raw) == "금값은 \\$2,400이다 그리고"


def test_verified_answer_marks_numbers_and_links_citations() -> None:
    lines = [
        [
            {"t": "실업률 "},
            {"n": "4.1%", "s": "ok", "a": None},
            {"t": ", 금리 "},
            {"n": "5%", "s": "mismatch", "a": "4.25%"},
            {"c": ["us_unrate", "nope"], "k": [True, False]},
        ]
    ]

    text = display.answer_markdown(lines, {"us_unrate": "labor"})

    assert text == (
        "실업률 :green[4.1%], 금리 :red[5%] :gray[(실제 4.25%)]"
        " [`us_unrate`](#labor) :red[`nope?`]"
    )


def test_log_follows_a_run_and_says_why_it_ended() -> None:
    log = display.AnalysisLog()
    log.ask("요약")
    start: Event = {
        "type": "start",
        "conversation": "abc",
        "snapshot": "2026-10-12T09:00:00+00:00",
        "view": "지금 보기",
        "stale": False,
        "provider_title": "데모",
        "model": "demo",
    }

    assert not log.apply(start)
    assert not log.apply({"type": "text", "text": "답"})
    assert not log.apply({"type": "usage", "input": 10, "output": 5, "note": ""})
    log.stopped = True
    log.finish(done=False)

    turn = log.turns[-1]
    assert log.conversation == "abc"
    assert log.header == "데이터 2026-10-12 09:00에 고정 · 메모리에만"
    assert turn.raw == "답"
    assert turn.run_line.endswith("토큰 입력 10 · 출력 5")
    assert turn.notices == [(display.STOPPED, False)]

    log.ask("다시")
    log.finish(done=False)
    assert log.turns[-1].notices == [(display.CUT_SHORT, True)]


def test_every_panel_is_on_exactly_one_page() -> None:
    placed = [panel for *_, ids in display.PANEL_PAGES for panel in ids]

    assert sorted(placed) == sorted(spec.id for spec in catalog.PANELS)


def test_tape_groups_skip_ids_without_data() -> None:
    grouped = [i for _, ids in display.TAPE_GROUPS for i in ids]
    assert sorted(grouped) == sorted(catalog.TAPE)
    # Eight tiles in every group, so every row of the overview is full.
    assert [len(ids) for _, ids in display.TAPE_GROUPS] == [8, 8, 8]
    assert set(display.TAPE_LABELS) == set(catalog.TAPE)

    rows = {i: row(1, 1).model_copy(update={"id": i}) for i in ("us_10y", "wti")}
    groups = display.tape_groups(catalog.TAPE, rows)

    assert [(title, [r.id for r in found]) for title, found in groups] == [
        ("금리", ["us_10y"]),
        ("환율·원자재", ["wti"]),
    ]
    assert display.trend(rows["wti"]) is None


def test_history_labels_become_first_days() -> None:
    assert display.label_day("2025-03") == date(2025, 3, 1)
    assert display.label_day("2025Q2") == date(2025, 4, 1)
    assert display.label_day("2025-03-15") == date(2025, 3, 15)


def test_detail_points_keep_the_last_months() -> None:
    series = tuple((date(2024, month, 1), float(month)) for month in range(1, 13))

    assert [p["day"] for p in display.detail_points(series, 3)] == [
        "2024-10-01",
        "2024-11-01",
        "2024-12-01",
    ]
    assert len(display.detail_points(series, 0)) == 12
    assert display.detail_points((), 24) == []
