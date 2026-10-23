from finagent.catalog import PANELS
from finagent.guides import GUIDES


def test_every_indicator_has_a_guide() -> None:
    ids = {indicator.id for panel in PANELS for indicator in panel.indicators}

    assert set(GUIDES) == ids
    assert all(text.strip() for text in GUIDES.values())
