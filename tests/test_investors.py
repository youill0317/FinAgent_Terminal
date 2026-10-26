"""13F parsing, quarters, mapping cache, clone returns and the investor view.

No network: SEC, OpenFIGI and Yahoo answers are small fixtures.
"""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportExplicitAny=false

from __future__ import annotations

import math
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from finagent import investors as inv
from finagent import portfolio as pf

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


CIK = "0000000042"
NS = "http://www.sec.gov/edgar/document/thirteenf/informationtable"


def entry(  # noqa: PLR0913 - one infoTable row
    cusip: str,
    shares: float,
    value: float,
    *,
    name: str = "ALPHA INC",
    kind: str = "SH",
    put_call: str = "",
) -> str:
    option = f"<putCall>{put_call}</putCall>" if put_call else ""
    return (
        f"<infoTable><nameOfIssuer>{name}</nameOfIssuer>"
        f"<titleOfClass>COM</titleOfClass><cusip>{cusip}</cusip>"
        f"<value>{value:g}</value><shrsOrPrnAmt><sshPrnamt>{shares:g}</sshPrnamt>"
        f"<sshPrnamtType>{kind}</sshPrnamtType></shrsOrPrnAmt>{option}"
        "<investmentDiscretion>SOLE</investmentDiscretion></infoTable>"
    )


def table_xml(*entries: str) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><ns1:informationTable xmlns:ns1="{NS}">'
        + "".join(entries).replace("<", "<ns1:").replace("<ns1:/", "</ns1:")
        + "</ns1:informationTable>"
    )


def cover(kind: str) -> str:
    return (
        '<edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler">'
        "<formData><coverPage><amendmentInfo><isAmendment>true</isAmendment>"
        f"<amendmentType>{kind}</amendmentType></amendmentInfo></coverPage>"
        "</formData></edgarSubmission>"
    )


# ---------------------------------------------------------------- parsing


def test_rows_are_summed_per_cusip_and_options_kept_apart() -> None:
    text = table_xml(
        entry("AAA000001", 100, 1_000),
        entry("AAA000001", 50, 500),  # another manager's row
        entry("BBB000002", 10, 300, name="BETA CORP"),
        entry("AAA000001", 20, 90, put_call="Put"),
        entry("AAA000001", 30, 80, put_call="CALL"),
        entry("CCC000003", 5_000, 4_000, name="NOTE", kind="PRN"),
    )

    parsed = inv.parse_table(text, date(2024, 2, 14))

    assert [(r.cusip, r.shares, r.value) for r in parsed.rows] == [
        ("AAA000001", 150, 1_500),
        ("BBB000002", 10, 300),
    ]
    assert (parsed.options, parsed.options_value) == (2, 170)
    assert (parsed.principal, parsed.principal_value) == (1, 4_000)
    assert inv.is_table(text)
    assert not inv.is_table(cover("RESTATEMENT"))


# ---------------------------------------------------------------- quarters


def row(cusip: str, shares: float, value: float) -> inv.Row:
    return inv.Row(cusip=cusip, name=f"{cusip} INC", shares=shares, value=value)


def table(  # noqa: PLR0913 - one filing
    accession: str,
    filed: date,
    period: date,
    *rows: inv.Row,
    form: str = "13F-HR",
    amendment: inv.Amendment = "",
    has_table: bool = True,
) -> inv.Table:
    return inv.Table(
        accession=accession,
        form=form,
        filed=filed,
        period=period,
        amendment=amendment,
        has_table=has_table,
        rows=rows,
    )


Q1, Q2 = date(2026, 3, 31), date(2026, 6, 30)


def test_changes_by_shares_and_sold_out() -> None:
    before = inv.quarters(
        [
            table(
                "a",
                date(2026, 5, 15),
                Q1,
                row("UP", 10, 1),
                row("DOWN", 10, 1),
                row("SAME", 10, 1),
                row("GONE", 10, 1),
            )
        ]
    )[0]
    now = inv.quarters(
        [
            table(
                "b",
                date(2026, 8, 14),
                Q2,
                row("UP", 11, 1),
                row("DOWN", 9, 1),
                row("SAME", 10, 5),
                row("NEW", 1, 1),
            )
        ]
    )[0]

    assert inv.change_labels(now, before) == {
        "UP": "증가",
        "DOWN": "감소",
        "SAME": "유지",
        "NEW": "신규",
    }
    assert [r.cusip for r in inv.sold_out(now, before)] == ["GONE"]
    assert inv.change_labels(now, None) == {}


# ---------------------------------------------------------------- clone returns


def prices(**columns: Sequence[float]) -> pd.DataFrame:
    days = pd.bdate_range("2026-03-02", periods=len(next(iter(columns.values()))))
    return pd.DataFrame(dict(columns), index=days, dtype="float64")


def test_the_clone_holds_between_rebalances() -> None:
    frame = prices(A=[10, 11, 12, 12], B=[10, 10, 5, 10])
    days = [d.date() for d in frame.index]
    schedule = [
        (days[0], pd.Series({"A": 0.5, "B": 0.5})),
        (days[2], pd.Series({"A": 1.0})),
    ]

    clone = pf.rebalanced(frame, schedule)

    # Bought half and half, left to drift: day 2 is .5*1.2 + .5*.5 = .85.
    assert np.allclose(clone.to_numpy(), [0.0, 0.05, -0.15, -0.15])
    # Rebalanced daily it would be 1.05 * (.5*12/11 + .5*.5) - 1, not -0.15.
    daily = pf.weighted(pf.returns(frame), pd.Series({"A": 0.5, "B": 0.5}))
    assert not math.isclose(float((1 + daily).prod() - 1), -0.15)


def filing(accepted: str | None, filed: date | None = None) -> inv.Filing:
    """A filing accepted at ``accepted``, a New York wall-clock time."""
    at = datetime.fromisoformat(accepted) if accepted else None
    day = filed or (at.date() if at else date(2026, 8, 14))
    return inv.Filing(accession="acc", form="13F-HR", filed=day, period=Q2, accepted=at)


# ---------------------------------------------------------------- the update


def test_without_a_user_agent_nothing_is_fetched(tmp_path: Path) -> None:
    def explode(*_: object) -> inv.Source:
        raise AssertionError("built a source without a User-Agent")

    fetcher = inv.Fetcher(tmp_path, build=explode)

    assert not fetcher.start(CIK, "")
    assert not fetcher.start(CIK, "   ")
    with pytest.raises(inv.InvestorError, match=inv.AGENT_KEY):
        _ = fetcher.check(CIK, "")
    assert not any(tmp_path.iterdir())
