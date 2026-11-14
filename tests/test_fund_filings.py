from __future__ import annotations

import pytest

from finagent import fund_filings
from tests import fake_sources


def test_nport_rows_are_classified_and_collateral_is_left_out() -> None:
    ewy = fund_filings.parse_nport(fake_sources.NPORT_EWY)
    tlt = fund_filings.parse_nport(fake_sources.NPORT_TLT)
    mixed = fund_filings.parse_nport(fake_sources.NPORT_FUND_OF_FUNDS)

    assert [row.asset for row in ewy.rows] == [
        "stock",
        "stock",
        "stock",
        "stock",
        "collateral",
        "other",
    ]
    assert {row.asset for row in tlt.rows} == {"bond", "collateral"}
    assert [row.asset for row in mixed.rows] == ["fund", "stock"]
    assert ewy.rows[0].cusip is None
    assert ewy.rows[0].isin == "KR7005930003"
    mix = fund_filings.asset_mix(ewy.rows)
    assert mix["stock"] == pytest.approx(99.1)
    assert mix["cash"] == pytest.approx(100 - 99.1 - 0.51)
    assert all(row.asset != "collateral" for row in fund_filings.top(ewy.rows))
    assert fund_filings.countries(ewy.rows) == {"KR": pytest.approx(100.0)}
    years = fund_filings.maturity_years(tlt)
    assert years == pytest.approx((49.0 * 28.96 + 49.15 * 18.96) / 98.15, abs=0.01)
