from __future__ import annotations

from typing import TYPE_CHECKING

from finagent import catalog, display
from finagent.models import Fmt, IndicatorRow, Panel, ProjectionCell, RegimeCell

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


def test_share_never_rounds_a_possible_outcome_to_0_or_100() -> None:
    assert display.share(0) == "0%"
    assert display.share(0.004) == "<1%"
    assert display.share(0.005) == "<1%"  # 0.5% rounds half to even
    assert display.share(0.5) == "50%"
    assert display.share(0.996) == ">99%"
    assert display.share(1.0) == "100%"


def test_change_rounds_away_below_the_shown_decimals() -> None:
    assert display.change_text(row(4.1, 4.104)) == "—"
    assert display.change_text(row(4.2, 4.1)) == "▲ 0.10"
    assert display.change_text(row(4.0, 4.1)) == "▼ 0.10"
    assert display.change_text(row(4.0, None)) == ""
    assert display.metric_delta(row(4.0, 4.1)) == "-0.10"
    assert display.metric_delta(row(4.0, None)) is None


def test_metric_delta_carries_the_unit_after_its_ascii_sign() -> None:
    def delta(value: float, prev: float, f: Fmt) -> str | None:
        return display.metric_delta(row(value, prev).model_copy(update={"fmt": f}))

    assert delta(4.3, 4.25, catalog.PCT2) == "+0.05%p"  # a change in % is in %p
    assert delta(0.1, 0.3, catalog.PP2) == "-0.20%p"
    assert delta(1400.0, 1385.5, Fmt(suf="원", dec=1)) == "+14.5원"
    assert delta(7.1, 7.5, catalog.USD_T) == "-$0.40T"


def test_table_changes_put_a_space_after_the_arrow() -> None:
    assert display.sep_change(ProjectionCell(value=3.4, prev=3.1)) == "▲ 0.3"
    assert display.sep_change(ProjectionCell(value=3.1, prev=3.12)) == "—"


def test_regime_card_moves_a_percent_unit_into_the_value() -> None:
    def card(value: str, unit: str) -> tuple[str, str]:
        cell = RegimeCell(
            label="물가", value=value, unit=unit, tone="good", tag="", rule=""
        )
        return display.regime_card(cell)

    assert card("2.8", "% 근원 PCE") == ("물가 · 근원 PCE", "2.8%")
    assert card("—", "%p 삼 룰") == ("물가 · 삼 룰", "—")
    assert card("−0.52", "NFCI") == ("물가 · NFCI", "−0.52")


def test_copyrighted_rows_carry_their_source_on_the_label() -> None:
    plain = row(1, 1).model_copy(update={"label": "A"})
    owned = row(1, 1).model_copy(update={"label": "S&P 500", "copyright": "S&P DJI"})
    panel = Panel(id="p", key="p", title="p", note="", rows=(plain, owned))

    table = display.panel_table(panel)

    assert "코드" not in table
    assert table["지표"] == ["A", "S&P 500 © S&P DJI"]


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
    assert log.header == "데이터 2026-10-12 09:00에 고정"
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
    assert set(display.TAPE_LABELS) == set(catalog.TAPE)

    rows = {i: row(1, 1).model_copy(update={"id": i}) for i in ("us_10y", "wti")}
    groups = display.tape_groups(catalog.TAPE, rows)

    assert [(title, [r.id for r in found]) for title, found in groups] == [
        ("금리", ["us_10y"]),
        ("환율·원자재", ["wti"]),
    ]
    assert display.trend(rows["wti"]) is None
