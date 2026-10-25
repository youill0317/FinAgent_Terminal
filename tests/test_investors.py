"""13F parsing, quarters, mapping cache, clone returns and the investor view.

No network: SEC, OpenFIGI and Yahoo answers are small fixtures.
"""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportExplicitAny=false

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

from finagent import investors as inv
from finagent import portfolio as pf
from finagent.cli import APP

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from finagent.service import DashboardService

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


def test_values_before_2023_are_in_thousands() -> None:
    text = table_xml(entry("AAA000001", 1, 7))

    assert inv.value_unit(date(2023, 1, 2)) == 1000
    assert inv.value_unit(date(2023, 1, 3)) == 1
    assert inv.parse_table(text, date(2022, 11, 14)).rows[0].value == 7_000
    assert inv.parse_table(text, date(2023, 2, 14)).rows[0].value == 7


def test_ciks_tickers_and_index_files() -> None:
    assert inv.clean_cik(" 1067983 ") == "0001067983"
    assert inv.clean_cik("CIK0001067983") == "0001067983"
    for bad in ("", "abc", "12345678901", "0"):
        with pytest.raises(inv.InvestorError):
            _ = inv.clean_cik(bad)
    assert inv.yahoo_ticker("BRK/B") == "BRK-B"
    assert inv.yahoo_ticker("aapl") == "AAPL"
    assert inv.yahoo_ticker("BAD TICKER") is None
    index = {
        "directory": {
            "item": [
                {"name": "primary_doc.xml", "size": "2000"},
                {"name": "0001-index.html", "size": "100"},
                {"name": "46994.xml", "size": "90000"},
                {"name": "form13fInfoTable.xml", "size": "50000"},
            ]
        }
    }
    assert inv.table_files(index) == ["form13fInfoTable.xml", "46994.xml"]


def test_openfigi_answers() -> None:
    hit = {
        "data": [
            {"ticker": "BRK/B", "name": "BERKSHIRE HATH-B", "marketSector": "Equity"}
        ]
    }

    assert inv.pick_ticker(hit) == ("BRK-B", "BERKSHIRE HATH-B")
    assert inv.pick_ticker({"warning": "No identifier found."}) == (None, "")
    assert inv.pick_ticker({"error": "No identifier found."}) == (None, "")
    assert inv.pick_ticker({"error": "Unexpected error."}) is None  # retried


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


def test_restatements_replace_and_new_holdings_add() -> None:
    tables = [
        table("a1", date(2026, 5, 15), Q1, row("A", 10, 100), row("B", 5, 50)),
        table(
            "a2",
            date(2026, 5, 20),
            Q1,
            row("B", 1, 10),
            row("C", 2, 20),
            form="13F-HR/A",
            amendment="NEW HOLDINGS",
        ),
        table(
            "b1",
            date(2026, 8, 14),
            Q2,
            row("A", 1, 1),
            form="13F-HR",
        ),
        table(
            "b2",
            date(2026, 8, 20),
            Q2,
            row("A", 12, 120),
            row("D", 3, 30),
            form="13F-HR/A",
            amendment="RESTATEMENT",
        ),
        table(
            "b3",
            date(2026, 8, 21),
            Q2,
            row("Z", 1, 1),
            form="13F-HR/A",
            amendment="UNKNOWN",
        ),
        table("b4", date(2026, 8, 22), Q2, form="13F-HR/A", has_table=False),
    ]

    newest, older = inv.quarters(tables)

    assert {(r.cusip, r.shares, r.value) for r in older.rows} == {
        ("A", 10, 100),
        ("B", 6, 60),
        ("C", 2, 20),
    }
    assert older.filed == date(2026, 5, 15)
    assert older.amended == (date(2026, 5, 20),)
    assert {r.cusip for r in newest.rows} == {"A", "D"}
    assert newest.filed == date(2026, 8, 14)
    assert newest.ignored == 2
    assert len(newest.versions) == 2


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


def test_filings_keep_the_last_eight_periods() -> None:
    periods = pd.date_range("2023-12-31", periods=11, freq="QE")
    arrays = {
        "accessionNumber": [f"acc{i}" for i in range(len(periods))] + ["x"],
        "form": ["13F-HR"] * len(periods) + ["4"],
        "filingDate": [(p + pd.Timedelta(days=45)).date().isoformat() for p in periods]
        + ["2026-01-01"],
        "reportDate": [p.date().isoformat() for p in periods] + ["2025-12-31"],
    }

    found = inv.filings_from(arrays)
    kept = inv.kept(found)

    assert len(found) == len(periods)
    assert [f.period for f in kept] == [p.date() for p in periods[-8:]]


# ---------------------------------------------------------------- mapping cache


class FakeSource:
    """SEC, OpenFIGI and Yahoo answers from dictionaries."""

    pages: dict[str, str]
    figi_answers: dict[str, Mapping[str, Any]]
    sectors: dict[str, str | None]

    def __init__(
        self,
        pages: Mapping[str, str],
        figi: Mapping[str, Mapping[str, Any]],
        sectors: Mapping[str, str | None],
    ) -> None:
        self.pages = dict(pages)
        self.figi_answers = dict(figi)
        self.sectors = dict(sectors)
        self.asked: list[str] = []
        self.jobs: list[str] = []

    def sec(self, url: str) -> str:
        self.asked.append(url)
        return self.pages[url]

    def figi(self, jobs: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
        assert len(jobs) <= inv.FIGI_JOBS
        self.jobs += [job["idValue"] for job in jobs]
        return [
            dict(self.figi_answers.get(job["idValue"], {"warning": "No identifier"}))
            for job in jobs
        ]

    def sector(self, ticker: str) -> str | None:
        return self.sectors.get(ticker)


def quiet(*_: object) -> None:
    pass


def test_misses_are_cached_and_retried_only_after_a_while(tmp_path: Path) -> None:
    source = FakeSource(
        {}, {"AAA000001": {"data": [{"ticker": "AAA", "marketSector": "Equity"}]}}, {}
    )
    day = date(2026, 10, 1)
    cusips = [f"C{i:08d}" for i in range(12)] + ["AAA000001"]

    asked = inv.map_cusips(source, tmp_path, cusips, day, quiet, quiet)
    cache = inv.load_cusips(tmp_path)

    assert asked == 13
    assert cache["AAA000001"].ticker == "AAA"
    assert cache["C00000000"].ticker is None
    assert cache["C00000000"].at == day

    source.jobs.clear()
    later = date(2026, 10, 1 + inv.MISS_RETRY_DAYS - 1)
    assert inv.map_cusips(source, tmp_path, cusips, later, quiet, quiet) == 0
    retry = date(2026, 11, 15)
    assert inv.map_cusips(source, tmp_path, cusips, retry, quiet, quiet) == 12
    assert "AAA000001" not in source.jobs


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


def test_filing_date_and_quarter_end_rebalances_differ() -> None:
    frame = prices(A=[10, 20, 20, 20, 20, 20], B=[10, 10, 10, 10, 5, 5])
    days = [d.date() for d in frame.index]
    weights = pd.Series({"A": 1.0})
    start = pd.Series({"B": 1.0})

    # Held B until the new weights apply: at the quarter end (day 0)
    # the clone catches A's doubling; from the filing date (day 2) it does not.
    quarter_end = pf.rebalanced(frame, [(days[0], weights)])
    filed = pf.rebalanced(frame, [(days[0], start), (days[2], weights)])

    assert math.isclose(float(quarter_end.iloc[-1]), 1.0)
    assert math.isclose(float(filed.iloc[-1]), 0.0)
    # A Saturday filing is bought at Monday's close.
    saturday = date(2026, 3, 7)
    assert pf.rebalanced(frame, [(saturday, weights)]).index[0] == pd.Timestamp(
        "2026-03-09"
    )


def test_unmapped_and_unpriced_weight_is_renormalized() -> None:
    frame = prices(A=[10, 11], B=[10, 12], C=[math.nan, 5])
    days = [d.date() for d in frame.index]
    rows = (row("A1", 1, 30), row("B1", 1, 10), row("C1", 1, 40), row("X1", 1, 20))
    cusips = {
        "A1": inv.Link(ticker="A", at=days[0]),
        "B1": inv.Link(ticker="B", at=days[0]),
        "C1": inv.Link(ticker="C", at=days[0]),
        "X1": inv.Link(ticker=None, at=days[0]),
    }

    w = inv.weights(rows, cusips)
    clone = pf.rebalanced(frame, [(days[0], w)])

    assert math.isclose(inv.unmapped_share(rows, cusips), 0.2)
    assert math.isclose(float(w.sum()), 0.8)
    # C has no price on day 0: A and B are scaled to 3/4 and 1/4.
    assert math.isclose(float(clone.iloc[-1]), 0.75 * 0.1 + 0.25 * 0.2)


def test_the_filed_schedule_skips_late_amendments_of_older_quarters() -> None:
    tables = [
        table("a1", date(2026, 5, 15), Q1, row("A", 1, 1)),
        table("b1", date(2026, 8, 14), Q2, row("B", 1, 1)),
        table(
            "a2",
            date(2026, 8, 20),
            Q1,
            row("A", 2, 2),
            form="13F-HR/A",
            amendment="RESTATEMENT",
        ),
    ]
    cusips = {c: inv.Link(ticker=c, at=Q1) for c in "AB"}

    found = inv.quarters(tables)
    # b1 was accepted after the close: bought the next day. a1 has no time.
    filed = inv.filed_schedule(found, cusips, {"b1": date(2026, 8, 15)})
    ends = inv.period_schedule(found, cusips)

    assert [(d, list(w.index)) for d, w in filed] == [
        (date(2026, 5, 16), ["A"]),
        (date(2026, 8, 15), ["B"]),
    ]
    assert [d for d, _ in ends] == [Q1, Q2]


def filing(accepted: str | None, filed: date | None = None) -> inv.Filing:
    """A filing accepted at ``accepted``, a New York wall-clock time."""
    at = datetime.fromisoformat(accepted) if accepted else None
    day = filed or (at.date() if at else date(2026, 8, 14))
    return inv.Filing(accession="acc", form="13F-HR", filed=day, period=Q2, accepted=at)


def test_the_buy_day_follows_the_acceptance_time() -> None:
    friday = date(2026, 8, 14)

    assert inv.buy_day(filing("2026-08-14T15:59")) == friday
    assert inv.buy_day(filing("2026-08-14T16:00")) == date(2026, 8, 15)
    assert inv.buy_day(filing(None)) == date(2026, 8, 15)  # unknown: next day
    # EDGAR's trailing Z is New York time, not UTC.
    assert inv.acceptance("2026-08-14T17:30:05.000Z") == datetime.fromisoformat(
        "2026-08-14T17:30:05"
    )
    assert inv.acceptance("") is None
    assert inv.acceptance("not a time") is None


def test_after_close_filings_roll_past_weekends_and_holidays() -> None:
    # Fri 2026-07-03 is a holiday: the frame skips it.
    days = pd.DatetimeIndex(["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07"])
    frame = pd.DataFrame({"A": [10.0, 10, 11, 12]}, index=days)
    w = pd.Series({"A": 1.0})

    def start(accepted: str) -> pd.Series:
        return pf.rebalanced(frame, [(inv.buy_day(filing(accepted)), w)])

    # Thursday after the close: the holiday and the weekend roll to Monday.
    assert start("2026-07-02T17:00").index[0] == pd.Timestamp("2026-07-06")
    # Thursday before the close: that day's close.
    assert start("2026-07-02T10:00").index[0] == pd.Timestamp("2026-07-02")
    # Accepted on the holiday itself: the next trading day.
    assert start("2026-07-03T09:00").index[0] == pd.Timestamp("2026-07-06")


def test_a_newer_13f_nt_is_flagged() -> None:
    arrays = {
        "accessionNumber": ["n1", "h1"],
        "form": ["13F-NT", "13F-HR"],
        "filingDate": ["2026-08-14", "2026-05-15"],
        "reportDate": ["2026-06-30", "2026-03-31"],
    }
    hr = inv.filings_from(arrays)
    nt = inv.filings_from(arrays, inv.NOTICES)
    now = datetime(2026, 10, 1, 9).astimezone()
    filer = inv.Filer(cik=CIK, name="A", fetched=now, filings=tuple(hr), notice=nt[0])

    assert [f.form for f in hr] == ["13F-HR"]
    assert inv.notice(filer) == nt[0]
    # An older notice is not news once a 13F-HR followed it.
    old = filer.model_copy(update={"notice": nt[0].model_copy(update={"filed": Q1})})
    assert inv.notice(old) is None
    assert inv.notice(filer.model_copy(update={"notice": None})) is None


def test_a_filer_two_quarters_behind_is_stale() -> None:
    today = date(2026, 10, 23)
    now = datetime(2026, 10, 1, 9).astimezone()

    def last(period: date) -> inv.Filer:
        f = filing(None).model_copy(update={"period": period})
        return inv.Filer(cik=CIK, name="A", fetched=now, filings=(f,))

    # 2026-09-30 is due 2026-11-14, so 2026-06-30 is the newest expected.
    assert inv.expected_period(today) == Q2
    assert inv.expected_period(date(2026, 11, 14)) == date(2026, 9, 30)
    assert inv.expected_period(date(2026, 2, 13)) == date(2025, 9, 30)
    assert inv.stale(last(Q2), today) is None
    assert inv.stale(last(Q1), today) is None  # one behind
    assert inv.stale(last(date(2025, 12, 31)), today) == date(2025, 12, 31)
    assert inv.stale(last(date(2025, 9, 30)), today) == date(2025, 9, 30)
    assert inv.stale(None, today) is None


@pytest.mark.parametrize("capped", [False, True], ids=["reaches-90%", "hits-the-cap"])
def test_large_filers_get_prices_for_their_top_holdings_only(*, capped: bool) -> None:
    # Harmonic values need ~160 names for 90%; geometric ones ~76.
    values = [10_000.0 / (i + 1) if capped else 10_000.0 * 0.97**i for i in range(300)]
    rows = tuple(row(f"C{i:03}", 1, v) for i, v in enumerate(values))
    cusips = {r.cusip: inv.Link(ticker=r.cusip, at=Q1) for r in rows}
    cusips["C001"] = inv.Link(ticker=None, at=Q1)  # one unlinked
    total = sum(values)

    top = inv.covered(rows)
    held = sum(r.value for r in top)

    assert len(top) <= inv.COVER_MAX
    assert (len(top) == inv.COVER_MAX) is capped
    assert (held >= inv.COVER_SHARE * total) is not capped
    assert [r.value for r in top] == sorted(values, reverse=True)[: len(top)]
    excluded = inv.excluded_share(rows)
    priced = inv.analysis_weights(rows, cusips)
    unlinked = 1 - excluded - float(priced.sum())
    assert math.isclose(excluded, 1 - held / total)
    assert math.isclose(unlinked, values[1] / total)
    assert math.isclose(excluded + unlinked + float(priced.sum()), 1.0)
    found = inv.quarters([table("a", date(2026, 5, 15), Q1, *rows)])
    assert len(inv.tickers(found, cusips)) == len(top) - 1


def test_late_starts_and_early_stops_are_dropped_and_renormalized() -> None:
    rets = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.0],
            "B": [math.nan, 0.01, 0.01],
            "C": [0.0, 0.0, 0.0],
            "D": [0.01, 0.01, math.nan],  # delisted
        },
        index=pd.bdate_range("2026-03-02", periods=3),
    )

    w, dropped = pf.covered(rets, pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}))

    assert dropped == ["B", "D"]
    assert np.allclose(w.to_numpy(), [2 / 3, 1 / 3])


def test_the_top_weights_clone_buys_on_the_same_days() -> None:
    schedule = [
        (date(2026, 5, 16), pd.Series({"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.05})),
        (date(2026, 8, 15), pd.Series(dtype="float64")),
    ]

    top = inv.top_schedule(schedule, 2)
    equal = inv.top_schedule(schedule, 2, equal=True)

    assert [d for d, _ in top] == [d for d, _ in schedule]
    assert top[0][1].to_dict() == pytest.approx({"A": 0.625, "B": 0.375})
    assert equal[0][1].to_dict() == {"A": 0.5, "B": 0.5}
    assert top[1][1].empty
    assert len(inv.top_schedule(schedule)[0][1]) == 4  # CONVICTION is 10


# ---------------------------------------------------------------- the update


def submissions(name: str = "ALPHA FUND LP") -> str:
    return json.dumps(
        {
            "name": name,
            "filings": {
                "recent": {
                    "accessionNumber": [
                        "0000000042-26-000003",
                        "0000000042-26-000002",
                        "0000000042-26-000001",
                    ],
                    "form": ["13F-NT", "13F-HR", "13F-HR"],
                    "filingDate": ["2026-08-20", "2026-08-14", "2026-05-15"],
                    "reportDate": ["2026-06-30", "2026-06-30", "2026-03-31"],
                    "acceptanceDateTime": [
                        "2026-08-20T10:00:00.000Z",
                        "2026-08-14T16:05:23.000Z",
                        "2026-05-15T09:30:00.000Z",
                    ],
                },
                "files": [],
            },
        }
    )


def sec_pages(cik: str = CIK, name: str = "ALPHA FUND LP") -> dict[str, str]:
    """One filer's pages; a second ``cik`` holds mostly BBB instead of AAA."""
    first, second = (
        ("AAA000001", "BBB000002") if cik == CIK else ("BBB000002", "AAA000001")
    )
    pages = {inv.SUBMISSIONS_URL.format(name=f"CIK{cik}.json"): submissions(name)}
    tables = {
        "000000004226000001": table_xml(
            entry(first, 10, 6_000), entry("ZZZ000009", 1, 4_000, name="Z")
        ),
        "000000004226000002": table_xml(
            entry(first, 12, 7_000),
            entry(second, 5, 3_000, name="BETA CORP"),
        ),
    }
    for folder_name, xml in tables.items():
        url = inv.partial_url(cik, folder_name)
        pages[url("index.json")] = json.dumps(
            {"directory": {"item": [{"name": "infotable.xml", "size": "10"}]}}
        )
        pages[url("infotable.xml")] = xml
    return pages


FIGI = {
    "AAA000001": {"data": [{"ticker": "AAA", "marketSector": "Equity"}]},
    "BBB000002": {"data": [{"ticker": "BBB/B", "marketSector": "Equity"}]},
}


def fake_source() -> FakeSource:
    return FakeSource(sec_pages(), FIGI, {"AAA": "Technology", "BBB-B": None})


def test_update_saves_filings_tables_and_caches(tmp_path: Path) -> None:
    source = fake_source()
    now = datetime(2026, 10, 1, 9).astimezone()

    message = inv.update(source, tmp_path, CIK, now, quiet, quiet)

    filer = inv.load_filer(tmp_path, CIK)
    assert filer is not None
    assert filer.name == "ALPHA FUND LP"
    found = inv.quarters(inv.load_tables(tmp_path, filer))
    assert [q.period for q in found] == [Q2, Q1]
    cusips = inv.load_cusips(tmp_path)
    assert cusips["BBB000002"].ticker == "BBB-B"
    assert cusips["ZZZ000009"].ticker is None
    sectors = inv.load_sectors(tmp_path)
    assert sectors["AAA"].sector == "Technology"
    assert sectors["BBB-B"].sector is None
    assert "티커 연결 2/3" in message
    assert inv.buy_days(filer) == {
        "0000000042-26-000002": date(2026, 8, 15),  # after the close
        "0000000042-26-000001": date(2026, 5, 15),
    }
    assert filer.notice is not None
    assert filer.notice.form == "13F-NT"

    # Saved filings are not fetched again.
    source.asked.clear()
    _ = inv.update(source, tmp_path, CIK, now, quiet, quiet)
    assert source.asked == [inv.SUBMISSIONS_URL.format(name=f"CIK{CIK}.json")]


def test_without_a_user_agent_nothing_is_fetched(tmp_path: Path) -> None:
    def explode(*_: object) -> inv.Source:
        raise AssertionError("built a source without a User-Agent")

    fetcher = inv.Fetcher(tmp_path, build=explode)

    assert not fetcher.start(CIK, "")
    assert not fetcher.start(CIK, "   ")
    with pytest.raises(inv.InvestorError, match=inv.AGENT_KEY):
        _ = fetcher.check(CIK, "")
    assert not any(tmp_path.iterdir())


def test_a_cik_without_13f_filings_is_refused(tmp_path: Path) -> None:
    body = json.dumps(
        {
            "name": "NOT A MANAGER",
            "filings": {
                "recent": {
                    "accessionNumber": ["1"],
                    "form": ["10-K"],
                    "filingDate": ["2026-01-01"],
                    "reportDate": ["2025-12-31"],
                },
                "files": [],
            },
        }
    )
    pages = {inv.SUBMISSIONS_URL.format(name=f"CIK{CIK}.json"): body}

    def build(_agent: str, _root: Path) -> inv.Source:
        return FakeSource(pages, {}, {})

    fetcher = inv.Fetcher(tmp_path, build=build, sleep=quiet)

    with pytest.raises(inv.InvestorError, match="13F-HR"):
        _ = fetcher.check(CIK, "Name name@example.org")
    assert fetcher.check  # the fetcher stays usable


# ---------------------------------------------------------------- the page


def market() -> pf.Prices:
    days = pd.bdate_range("2006-12-01", "2026-10-21")
    noise = np.random.default_rng(3).normal(0, 0.01, size=(len(days), 3))
    walk = np.exp(np.cumsum(noise, axis=0))
    adjusted = pd.DataFrame(
        {
            "AAA": 100 * walk[:, 0],
            "BBB-B": 50 * walk[:, 1],
            pf.BENCHMARK: 1_000 * walk[:, 2],
            pf.TOTAL_RETURN: 1_000 * walk[:, 2],
            pf.FX: 1_300.0,
        },
        index=days,
    )
    return pf.Prices(adjusted * 1.0, adjusted)


def page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, demo: bool = True
) -> AppTest:
    """The portfolio page switched to investors, files under ``tmp_path``."""
    settings = tmp_path / "finagent.settings.json"
    env = tmp_path / ".env"
    _ = env.write_text("", encoding="utf-8")
    monkeypatch.delenv(inv.AGENT_KEY, raising=False)
    flags = ["--demo"] if demo else []
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.py", *flags, f"--settings={settings}", f"--env-file={env}"],
    )
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at._page_hash = calc_hash("portfolio")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    _ = at.run()
    _ = at.segmented_control(key="pf_mode").set_value("investors").run()
    return at


def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **_kw: object) -> None:
        raise AssertionError("contacted the network")

    monkeypatch.setattr(inv.Web, "__init__", refuse)
    monkeypatch.setattr(pf.Yahoo, "__init__", refuse)


def test_the_investor_view_asks_for_a_user_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --demo and without the key: one line, and nothing fetched."""
    from finagent import cli

    no_network(monkeypatch)
    real = cli.build_service

    def offline(
        *, demo: bool, env: Mapping[str, str], settings_path: Path | None = None
    ) -> DashboardService:
        del demo
        return real(demo=True, env=env, settings_path=settings_path)

    monkeypatch.setattr(cli, "build_service", offline)
    at = page(tmp_path, monkeypatch, demo=False)

    assert not at.exception
    assert any(inv.AGENT_KEY in i.value for i in at.info)
    assert not (tmp_path / inv.DATA_DIR).exists()


def saved_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """One investor with saved 13F, a KRW portfolio and prices; nothing fetched."""
    no_network(monkeypatch)
    root = tmp_path / inv.DATA_DIR
    now = datetime(2026, 10, 1, 9).astimezone()
    _ = inv.update(fake_source(), root, CIK, now, quiet, quiet)
    inv.add_investor(root, CIK, "ALPHA FUND LP")
    pf_root = tmp_path / pf.DATA_DIR
    book, pid = pf.add_portfolio(pf.Book(), "연금", "KRW")
    first = pf.Trade(
        id="t1", day=date(2025, 1, 6), ticker="AAA", side="buy", quantity=10, price=100
    )
    pf.save_book(pf_root, pf.add_trade(book, pid, first))
    pf.save_prices(pf_root, market())

    at = page(tmp_path, monkeypatch)
    return at.selectbox(key="inv_selected").set_value(CIK).run()


def test_the_investor_view_with_saved_filings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = saved_view(tmp_path, monkeypatch)

    assert not at.exception
    assert any("2026-06-30" in str(o) for o in at.selectbox(key="inv_selected").options)
    holdings = next(d.value for d in at.dataframe if "티커" in d.value.columns)
    assert list(holdings["티커"]) == ["AAA", "BBB-B"]
    charts = [str(cast("object", c.proto)) for c in at.get("vega_lite_chart")]
    assert any("innerRadius" in chart for chart in charts)  # 현황's weight donut

    for section in (
        "returns",
        "analysis",
        "correlation",
        "tail",
        "outlook",
        "stress",
        "compare",
    ):
        _ = at.segmented_control(key="inv_section").set_value(section).run()
        assert not at.exception, section
        assert at.metric, section
    _ = at.segmented_control(key="inv_section").set_value("analysis").run()
    at.session_state["inv_style"] = True
    _ = at.run()
    assert not at.exception
    _ = at.segmented_control(key="inv_section").set_value("correlation").run()
    league = next(d.value for d in at.dataframe if "투자자" in d.value.columns)
    assert list(league["투자자"]) == ["ALPHA FUND LP"]
    assert "축소 샤프" not in league.columns  # one investor: nothing to shrink toward

    _ = at.segmented_control(key="inv_section").set_value("returns").run()
    steps = next(d.value for d in at.dataframe if "회전율" in d.value.columns)
    assert list(steps["회전율"])[-1] == "—"  # the first purchase, from cash

    def total() -> str:
        return str(next(m.value for m in at.metric if m.label.startswith("누적")))

    filed = total()
    _ = at.segmented_control(key="inv_variant").set_value("top").run()
    assert not at.exception
    assert total() == filed  # two holdings: the top ten are all of them
    _ = at.toggle(key="inv_equal").set_value(True).run()
    assert not at.exception
    assert total() != filed  # 50/50 instead of the reported 70/30

    # Older quarter: ZZZ was sold, and its weight was not linked to a price.
    _ = at.selectbox(key="inv_quarter").set_value(Q1).run()
    assert not at.exception

    # A ticker cell opens its dialog.
    gen = at.session_state.get("inv_gen", 0)
    key = f"inv_table_{gen}"
    at.session_state[key] = {
        "selection": {"rows": [], "columns": [], "cells": [[0, "티커"]]}
    }
    _ = at.run()
    assert not at.exception
    assert at.session_state["inv_gen"] == gen + 1  # the dialog opened


def test_two_investors_are_ranked_with_shrunk_sharpes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_network(monkeypatch)
    root = tmp_path / inv.DATA_DIR
    now = datetime(2026, 10, 1, 9).astimezone()
    other = "0000000043"
    second = FakeSource(sec_pages(other, "BETA FUND LP"), FIGI, {"AAA": None})
    for cik, name, source in (
        (CIK, "ALPHA FUND LP", fake_source()),
        (other, "BETA FUND LP", second),
    ):
        _ = inv.update(source, root, cik, now, quiet, quiet)
        inv.add_investor(root, cik, name)
    pf.save_prices(tmp_path / pf.DATA_DIR, market())

    at = page(tmp_path, monkeypatch)
    _ = at.selectbox(key="inv_selected").set_value(CIK).run()
    _ = at.segmented_control(key="inv_section").set_value("correlation").run()

    assert not at.exception
    league = next(d.value for d in at.dataframe if "투자자" in d.value.columns)
    assert sorted(league["투자자"]) == ["ALPHA FUND LP", "BETA FUND LP"]
    assert "축소 샤프" in league.columns
    sharpes = [float(v.replace("−", "-")) for v in league["샤프 (연)"]]
    assert sharpes == sorted(sharpes, reverse=True)
    dsr = next(
        str(m.proto.help) for m in at.markdown if m.value.startswith("**투자자 비교**")
    )
    assert "DSR" in dsr
    assert "기간 부족" in dsr  # about 110 days of clone returns
    assert not any(word in dsr for word in ("최고 투자자", "추천"))
