from __future__ import annotations

from typing import TYPE_CHECKING

from finagent import display
from finagent.models import Fmt, IndicatorRow, ProjectionCell

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


def test_short_dates_follow_the_frequency() -> None:
    assert display.short_date(row(1, 1, freq="M")) == "26.09"
    assert display.short_date(row(1, 1, freq="Q", day="2026-Q3")) == "26-Q3"
    assert display.short_date(row(1, 1, freq="D", day="2026-09-01")) == "09/01"
    assert display.weekday("2026-10-12") == "10/12 월"


def test_sep_cells_show_the_move_since_the_prior_release() -> None:
    assert display.sep_cell(ProjectionCell(value=2.1, prev=None)) == "2.1 신규"
    assert display.sep_cell(ProjectionCell(value=2.1, prev=2.12)) == "2.1 — 0"
    assert display.sep_cell(ProjectionCell(value=2.4, prev=2.1)) == "2.4 ▲ 0.3"
    assert display.sep_cell(None) == "—"


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
