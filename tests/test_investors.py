# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportExplicitAny=false

from __future__ import annotations

from datetime import date

from finagent import investors as inv

NS = "http://www.sec.gov/edgar/document/thirteenf/informationtable"


def entry(  # noqa: PLR0913
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


def test_rows_are_summed_per_cusip_and_options_kept_apart() -> None:
    text = table_xml(
        entry("AAA000001", 100, 1_000),
        entry("AAA000001", 50, 500),
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
