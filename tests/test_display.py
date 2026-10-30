from __future__ import annotations

from finagent import catalog, display
from finagent.models import RegimeCell


def test_share_never_rounds_a_possible_outcome_to_0_or_100() -> None:
    assert display.share(0) == "0%"
    assert display.share(0.004) == "<1%"
    assert display.share(0.005) == "<1%"  # 0.5% rounds half to even
    assert display.share(0.5) == "50%"
    assert display.share(0.996) == ">99%"
    assert display.share(1.0) == "100%"


def test_every_panel_is_on_exactly_one_page() -> None:
    placed = [panel for *_, ids in display.PANEL_PAGES for panel in ids]

    assert sorted(placed) == sorted(spec.id for spec in catalog.PANELS)


def test_regime_card_without_a_unit_is_the_bare_label() -> None:
    cell = RegimeCell(
        label="물가", value="999", unit="", tone="neutral", tag="데모", rule=""
    )

    assert display.regime_card(cell) == ("물가", "999")
