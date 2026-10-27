"""Well-known investors' 13F holdings from SEC EDGAR, kept under ``data/investors/``.

Like the screener and the portfolio page, this module is an exception to the
dashboard's data rules (``tests/test_guardrails.py``): it contacts SEC EDGAR
(``data.sec.gov``, ``www.sec.gov``) and OpenFIGI (``api.openfigi.com``),
imports ``yfinance`` for sectors, and writes files, all under
``data/investors/`` in the folder the app runs from. Of the other finagent
modules it imports only the pure math in ``perf``; prices come from the
portfolio page's store (``app.py`` glues the two). Nothing is fetched until
a ``Fetcher`` thread is started.

Stored files:

- ``investors.json``: CIKs the user added (only once one is added).
- ``<CIK>/filings.json``: the filer's name and its 13F-HR(/A) filings in the
  last ``QUARTERS`` reporting periods.
- ``<CIK>/<accession>.json``: one parsed information table, rows summed per
  CUSIP. Filings never change, so a saved one is not fetched again.
- ``cusips.json``: CUSIP → ticker from OpenFIGI, misses included with the
  date they were tried, so they are retried only after ``MISS_RETRY_DAYS``.
- ``sectors.json``: ticker → yfinance sector, misses likewise.

Values: filings made on or after ``DOLLARS_FROM`` (the SEC's 2023 Form 13F
compliance date) report dollars, earlier ones thousands of dollars.
Amendments: a 13F-HR/A marked RESTATEMENT replaces its quarter, one marked
NEW HOLDINGS adds to it; one with neither (or no information table) is
counted but not applied.
"""

# pandas and yfinance ship no type stubs; JSON from the web is untyped.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitStringConcatenation=false

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as clock
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Protocol, cast, final
from xml.etree import ElementTree as ET

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

# The guardrail test reads this dotted form as the math module.
import finagent.perf as perf  # noqa: PLR0402

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

DATA_DIR: Final = Path("data") / "investors"
LIST_FILE: Final = "investors.json"
FILINGS_FILE: Final = "filings.json"
CUSIP_FILE: Final = "cusips.json"
SECTOR_FILE: Final = "sectors.json"
YFINANCE_CACHE: Final = ".yfinance"
AGENT_KEY: Final = "SEC_USER_AGENT"
SUBMISSIONS_URL: Final = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL: Final = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{name}"
OPENFIGI_URL: Final = "https://api.openfigi.com/v3/mapping"
FORMS: Final = ("13F-HR", "13F-HR/A")
NOTICES: Final = ("13F-NT", "13F-NT/A")  # holdings reported by another manager
QUARTERS: Final = 8
MARKET_CLOSE: Final = clock(16, 0)  # New York; acceptance at or after: next day
DEADLINE_DAYS: Final = 45  # a 13F is due this long after its quarter end
STALE_QUARTERS: Final = 2  # periods behind the expected one: 보고 중단
COVER_SHARE: Final = 0.90  # prices and sectors for the largest holdings up to this
COVER_MAX: Final = 100  # ... or this many positions per quarter, whichever first
DOLLARS_FROM: Final = date(2023, 1, 3)
SEC_PAUSE: Final = 0.15  # seconds after each SEC request: under 10 a second
FIGI_JOBS: Final = 10  # OpenFIGI without a key: 10 jobs a request
FIGI_PAUSE: Final = 2.5  # and 25 requests a minute
FIGI_RETRIES: Final = 3
FIGI_WAIT: Final = 60.0  # after a 429 without Retry-After
YAHOO_PAUSE: Final = 0.2
YAHOO_FAILURES: Final = 5  # sector misses in a row that stop the stage
MISS_RETRY_DAYS: Final = 30
TIMEOUT: Final = 30.0
NOT_FOUND: Final = 404
TOO_MANY: Final = 429
CONVICTION: Final = 10  # holdings in the top-weights clone
UNKNOWN: Final = "미확인"
EXCLUDED: Final = "미포함"  # outside the covered holdings
CHANGES: Final = ("신규", "증가", "감소", "유지")
SOLD: Final = "청산"
CIK_PATTERN: Final = re.compile(r"\d{1,10}")
TICKER_PATTERN: Final = re.compile(r"[A-Z0-9][A-Z0-9-]{0,9}")
OPTION_TYPES: Final = frozenset({"PUT", "CALL"})

Amendment = Literal["", "RESTATEMENT", "NEW HOLDINGS", "UNKNOWN"]


class InvestorError(ValueError):
    """A request the page cannot take; the message is shown on the page."""


@dataclass(frozen=True, slots=True)
class Investor:
    cik: str  # ten digits
    name: str
    manager: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.manager})" if self.manager else self.name


# Each CIK was checked on EDGAR (13F-HR filings) on 2026-10-22. Greenlight's
# 13F-HR moved to DME Capital Management in 2024 (same EDGAR business address
# and phone as Greenlight Capital Inc); Scion's last is for 2025-09.
PRESETS: Final = (
    Investor("0001067983", "Berkshire Hathaway", "버핏"),
    Investor("0001336528", "Pershing Square", "애크먼"),
    Investor("0001649339", "Scion Asset Management", "버리"),
    Investor("0001709323", "Himalaya Capital", "리루"),
    Investor("0001061768", "Baupost Group", "클라만"),
    Investor("0001536411", "Duquesne Family Office", "드러켄밀러"),
    Investor("0001656456", "Appaloosa", "테퍼"),
    Investor("0001350694", "Bridgewater Associates", "달리오"),
    Investor("0001040273", "Third Point", "로브"),
    Investor("0001489933", "DME Capital (Greenlight)", "아인혼"),
)


# ---------------------------------------------------------------- stored shapes


class _Frozen(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class Filing(_Frozen):
    accession: str  # 0001067983-26-000004
    form: str
    filed: date
    period: date  # periodOfReport
    accepted: datetime | None = None  # EDGAR acceptance, New York wall clock


class Filer(_Frozen):
    """``<CIK>/filings.json``."""

    cik: str
    name: str
    fetched: datetime
    filings: tuple[Filing, ...] = ()
    notice: Filing | None = None  # the newest 13F-NT, if any


class Row(_Frozen):
    """One CUSIP in a filing, its manager rows summed."""

    cusip: str
    name: str
    title: str = ""
    shares: float
    value: float  # dollars


class Table(_Frozen):
    """``<CIK>/<accession>.json``: one filing's equity rows and what was left out."""

    accession: str
    form: str
    filed: date
    period: date
    amendment: Amendment = ""
    has_table: bool = True
    rows: tuple[Row, ...] = ()
    options: int = 0
    options_value: float = 0.0
    principal: int = 0  # rows in principal amount (notes), not shares
    principal_value: float = 0.0


class Link(_Frozen):
    ticker: str | None  # Yahoo form; None for a miss
    name: str = ""
    at: date


class Sector(_Frozen):
    sector: str | None
    at: date


class Added(_Frozen):
    """``investors.json``."""

    version: int = 1
    added: tuple[tuple[str, str], ...] = ()  # (cik, name)


# ---------------------------------------------------------------- parsing


def clean_cik(text: str) -> str:
    """Ten digits; raises ``InvestorError`` otherwise."""
    digits = text.strip().upper().removeprefix("CIK").strip()
    if not CIK_PATTERN.fullmatch(digits) or int(digits) == 0:
        raise InvestorError("CIK는 숫자 10자리 이내입니다.")
    return digits.zfill(10)


def value_unit(filed: date) -> int:
    """Dollars per reported unit: thousands before ``DOLLARS_FROM``."""
    return 1 if filed >= DOLLARS_FROM else 1000


def yahoo_ticker(ticker: str) -> str | None:
    """OpenFIGI's ``BRK/B`` as Yahoo's ``BRK-B``; None for one Yahoo would not know."""
    text = ticker.strip().upper().replace("/", "-").replace(".", "-")
    return text if TICKER_PATTERN.fullmatch(text) else None


def filings_from(
    arrays: Mapping[str, Any], forms: Sequence[str] = FORMS
) -> list[Filing]:
    """Filings of ``forms`` in a submissions ``recent`` block or page."""
    out: list[Filing] = []
    kinds = cast("list[str]", arrays.get("form", []))
    accepted = cast("list[object]", arrays.get("acceptanceDateTime", []))
    for i, form in enumerate(kinds):
        if form not in forms:
            continue
        try:
            out.append(
                Filing(
                    accession=str(arrays["accessionNumber"][i]),
                    form=form,
                    filed=date.fromisoformat(str(arrays["filingDate"][i])),
                    period=date.fromisoformat(str(arrays["reportDate"][i])),
                    accepted=acceptance(accepted[i] if i < len(accepted) else None),
                )
            )
        except (KeyError, IndexError, ValueError):
            continue
    return out


def acceptance(value: object) -> datetime | None:
    """``2026-08-14T16:05:23.000Z`` as a naive New York time, None if unreadable.

    EDGAR stamps acceptance in Eastern time; the trailing Z is not UTC.
    """
    text = str(value or "").strip().removesuffix("Z")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None) if text else None
    except ValueError:
        return None


def buy_day(filing: Filing) -> date:
    """The day whose close a follower could first trade at; later days roll on.

    Accepted before the 16:00 New York close: that day. At or after it, or
    with no acceptance time: the next day. Weekends and holidays roll to the
    next trading day where the prices are (``rebalanced``).
    """
    if filing.accepted is None:
        return filing.filed + timedelta(days=1)
    day = filing.accepted.date()
    late = filing.accepted.time() >= MARKET_CLOSE
    return day + timedelta(days=1) if late else day


def buy_days(filer: Filer | None) -> dict[str, date]:
    return {f.accession: buy_day(f) for f in filer.filings} if filer else {}


def untimed(filer: Filer | None) -> int:
    """Kept filings without an acceptance time (bought the next day)."""
    return sum(f.accepted is None for f in filer.filings) if filer else 0


def notice(filer: Filer | None) -> Filing | None:
    """The newest 13F-NT when it is newer than the newest 13F-HR."""
    if filer is None or filer.notice is None:
        return None
    last = max((f.filed for f in filer.filings), default=None)
    return filer.notice if last is None or filer.notice.filed > last else None


def _quarter_end(day: date) -> date:
    """The last quarter end on or before ``day``."""
    ends = [date(day.year, m, d) for m, d in ((3, 31), (6, 30), (9, 30), (12, 31))]
    past = [end for end in ends if end <= day]
    return past[-1] if past else date(day.year - 1, 12, 31)


def expected_period(today: date) -> date:
    """The newest quarter end whose 13F deadline has passed."""
    end = _quarter_end(today)
    while end + timedelta(days=DEADLINE_DAYS) > today:
        end = _quarter_end(end - timedelta(days=1))
    return end


def _index(day: date) -> int:
    return day.year * 4 + (day.month - 1) // 3


def stale(filer: Filer | None, today: date) -> date | None:
    """The last reported period when it is ``STALE_QUARTERS`` or more behind."""
    if filer is None or not filer.filings:
        return None
    last = max(f.period for f in filer.filings)
    behind = _index(expected_period(today)) - _index(last)
    return last if behind >= STALE_QUARTERS else None


def recent_periods(filings: Iterable[Filing], quarters: int = QUARTERS) -> list[date]:
    return sorted({f.period for f in filings}, reverse=True)[:quarters]


def kept(filings: Iterable[Filing], quarters: int = QUARTERS) -> list[Filing]:
    """Every filing for the newest ``quarters`` periods, oldest first."""
    filings = list(filings)
    periods = set(recent_periods(filings, quarters))
    unique = {f.accession: f for f in filings if f.period in periods}
    return sorted(unique.values(), key=lambda f: (f.filed, f.accession))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse(text: str) -> ET.Element:
    # SEC documents; expat refuses entity expansion attacks since 2.4.1.
    return ET.fromstring(text.encode("utf-8"))  # noqa: S314


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in element if _local(c.tag) == name), None)


def _text(element: ET.Element | None) -> str:
    return (element.text or "").strip() if element is not None else ""


def is_table(text: str) -> bool:
    try:
        return _local(_parse(text).tag) == "informationTable"
    except ET.ParseError:
        return False


@dataclass(frozen=True, slots=True)
class Parsed:
    rows: tuple[Row, ...]
    options: int
    options_value: float
    principal: int
    principal_value: float


def parse_table(text: str, filed: date) -> Parsed:
    """Equity rows summed per CUSIP, in dollars; options and notes counted apart."""
    unit = value_unit(filed)
    sums: dict[str, dict[str, Any]] = {}
    options = principal = 0
    options_value = principal_value = 0.0
    for entry in _parse(text):
        if _local(entry.tag) != "infoTable":
            continue
        amount = _child(entry, "shrsOrPrnAmt")
        shares = 0.0
        kind = ""
        if amount is not None:
            shares = float(_text(_child(amount, "sshPrnamt")) or 0)
            kind = _text(_child(amount, "sshPrnamtType")).upper()
        value = float(_text(_child(entry, "value")) or 0) * unit
        if _text(_child(entry, "putCall")).upper() in OPTION_TYPES:
            options += 1
            options_value += value
            continue
        if kind == "PRN":
            principal += 1
            principal_value += value
            continue
        cusip = _text(_child(entry, "cusip")).upper()
        row = sums.setdefault(
            cusip,
            {
                "cusip": cusip,
                "name": _text(_child(entry, "nameOfIssuer")),
                "title": _text(_child(entry, "titleOfClass")),
                "shares": 0.0,
                "value": 0.0,
            },
        )
        row["shares"] += shares
        row["value"] += value
    rows = tuple(
        sorted((Row(**r) for r in sums.values()), key=lambda r: (-r.value, r.cusip))
    )
    return Parsed(rows, options, options_value, principal, principal_value)


def parse_amendment(text: str) -> Amendment:
    """RESTATEMENT or NEW HOLDINGS from a 13F-HR/A cover page, else UNKNOWN."""
    try:
        root = _parse(text)
    except ET.ParseError:
        return "UNKNOWN"
    found = next((e for e in root.iter() if _local(e.tag) == "amendmentType"), None)
    kind = _text(found).upper()
    if kind == "RESTATEMENT":
        return "RESTATEMENT"
    if kind == "NEW HOLDINGS":
        return "NEW HOLDINGS"
    return "UNKNOWN"


def table_files(index: Mapping[str, Any]) -> list[str]:
    """XML documents in a filing's ``index.json`` that may be the info table."""
    items = cast("list[dict[str, Any]]", index.get("directory", {}).get("item", []))
    names = [
        (str(i.get("name", "")), float(i.get("size") or 0))
        for i in items
        if str(i.get("name", "")).lower().endswith(".xml")
        and str(i.get("name", "")).lower() != "primary_doc.xml"
    ]
    names.sort(key=lambda it: ("info" not in it[0].lower(), -it[1]))
    return [name for name, _ in names]


def pick_ticker(result: Mapping[str, Any]) -> tuple[str | None, str] | None:
    """(ticker, name) from one OpenFIGI job; None when it errored (retry later)."""
    if "error" in result and "data" not in result:
        message = str(result["error"]).lower()
        return (None, "") if "no identifier" in message else None
    data = cast("list[dict[str, Any]]", result.get("data") or [])
    equity = [d for d in data if d.get("marketSector") == "Equity"] or data
    for item in equity:
        ticker = yahoo_ticker(str(item.get("ticker") or ""))
        if ticker:
            return ticker, str(item.get("name") or "")
    return None, ""


# ---------------------------------------------------------------- storage


def _write(path: Path, text: str) -> None:
    """Write next to ``path`` and rename, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    _ = temp.write_text(text, encoding="utf-8")
    _ = temp.replace(path)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_model(path: Path, model: BaseModel) -> None:
    _write(path, model.model_dump_json(indent=1) + "\n")


def load_added(root: Path) -> Added:
    path = root / LIST_FILE
    if not path.is_file():
        return Added()
    try:
        return Added.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return Added()


def roster(root: Path) -> list[Investor]:
    """The presets, then the CIKs the user added."""
    known = {i.cik for i in PRESETS}
    extra = [Investor(c, n) for c, n in load_added(root).added if c not in known]
    return [*PRESETS, *extra]


def add_investor(root: Path, cik: str, name: str) -> None:
    old = load_added(root)
    if any(c == cik for c, _ in old.added) or any(i.cik == cik for i in PRESETS):
        return
    save_model(root / LIST_FILE, Added(added=(*old.added, (cik, name))))


def folder(root: Path, cik: str) -> Path:
    return root / cik


def load_filer(root: Path, cik: str) -> Filer | None:
    path = folder(root, cik) / FILINGS_FILE
    if not path.is_file():
        return None
    try:
        return Filer.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None


def load_tables(root: Path, filer: Filer) -> list[Table]:
    """Saved tables of the filer's kept filings, oldest filing first."""
    out: list[Table] = []
    for filing in filer.filings:
        path = folder(root, filer.cik) / f"{filing.accession}.json"
        if not path.is_file():
            continue
        try:
            out.append(Table.model_validate_json(path.read_bytes()))
        except (OSError, ValidationError):
            continue
    return out


def load_cusips(root: Path) -> dict[str, Link]:
    raw = _read_json(root / CUSIP_FILE)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Link] = {}
    for key, value in cast("dict[str, object]", raw).items():
        try:
            out[key] = Link.model_validate(value)
        except ValidationError:
            continue
    return out


def save_cusips(root: Path, cache: Mapping[str, Link]) -> None:
    body = {k: v.model_dump(mode="json") for k, v in sorted(cache.items())}
    _write(root / CUSIP_FILE, json.dumps(body, ensure_ascii=False, indent=0) + "\n")


def load_sectors(root: Path) -> dict[str, Sector]:
    raw = _read_json(root / SECTOR_FILE)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Sector] = {}
    for key, value in cast("dict[str, object]", raw).items():
        try:
            out[key] = Sector.model_validate(value)
        except ValidationError:
            continue
    return out


def save_sectors(root: Path, cache: Mapping[str, Sector]) -> None:
    body = {k: v.model_dump(mode="json") for k, v in sorted(cache.items())}
    _write(root / SECTOR_FILE, json.dumps(body, ensure_ascii=False, indent=0) + "\n")


def due(at: date | None, hit: bool, today: date) -> bool:
    """A cache entry to fetch: none yet, or a miss older than ``MISS_RETRY_DAYS``."""
    if at is None:
        return True
    return not hit and (today - at).days >= MISS_RETRY_DAYS


def stamp(root: Path, cik: str) -> tuple[tuple[str, int], ...]:
    """Modification times of everything the investor view reads."""
    paths = [root / CUSIP_FILE, root / SECTOR_FILE]
    if folder(root, cik).is_dir():
        paths += sorted(folder(root, cik).glob("*.json"))
    return tuple((p.name, p.stat().st_mtime_ns) for p in paths if p.is_file())


# ---------------------------------------------------------------- quarters


@dataclass(frozen=True, slots=True)
class Quarter:
    period: date
    filed: date  # the first filing for the period
    rows: tuple[Row, ...]  # after the amendments
    amended: tuple[date, ...]  # amendments applied
    ignored: int  # amendments of unknown type or without a table
    options: int
    options_value: float
    principal: int
    principal_value: float
    # holdings after each filing: (accession, filed, rows)
    versions: tuple[tuple[str, date, tuple[Row, ...]], ...]

    @property
    def value(self) -> float:
        return sum(r.value for r in self.rows)


def _add(rows: Iterable[Row], more: Iterable[Row]) -> tuple[Row, ...]:
    sums: dict[str, Row] = {r.cusip: r for r in rows}
    for r in more:
        old = sums.get(r.cusip)
        sums[r.cusip] = (
            r
            if old is None
            else old.model_copy(
                update={"shares": old.shares + r.shares, "value": old.value + r.value}
            )
        )
    return tuple(sorted(sums.values(), key=lambda r: (-r.value, r.cusip)))


def quarters(tables: Iterable[Table]) -> list[Quarter]:
    """Each period's holdings after its amendments, newest period first."""
    by: dict[date, list[Table]] = {}
    for t in tables:
        by.setdefault(t.period, []).append(t)
    out: list[Quarter] = []
    for period, group in by.items():
        group.sort(key=lambda t: (t.filed, t.accession))
        rows: tuple[Row, ...] = ()
        versions: list[tuple[str, date, tuple[Row, ...]]] = []
        amended: list[date] = []
        ignored = 0
        counts = [0, 0.0, 0, 0.0]
        for t in group:
            original = t.form == "13F-HR"
            if not t.has_table or (not original and t.amendment == "UNKNOWN"):
                ignored += 0 if original else 1
                continue
            if original or t.amendment == "RESTATEMENT":
                rows = t.rows
                counts = [t.options, t.options_value, t.principal, t.principal_value]
            else:  # NEW HOLDINGS
                rows = _add(rows, t.rows)
                counts = [
                    counts[0] + t.options,
                    counts[1] + t.options_value,
                    counts[2] + t.principal,
                    counts[3] + t.principal_value,
                ]
            if not original:
                amended.append(t.filed)
            versions.append((t.accession, t.filed, rows))
        if not versions:
            continue
        out.append(
            Quarter(
                period=period,
                filed=versions[0][1],
                rows=rows,
                amended=tuple(amended),
                ignored=ignored,
                options=int(counts[0]),
                options_value=float(counts[1]),
                principal=int(counts[2]),
                principal_value=float(counts[3]),
                versions=tuple(versions),
            )
        )
    return sorted(out, key=lambda q: q.period, reverse=True)


def weights(
    rows: Iterable[Row], cusips: Mapping[str, Link], total: float | None = None
) -> pd.Series:
    """Ticker → share of ``total`` (by default the rows' value); unmapped left out."""
    rows = list(rows)
    total = sum(r.value for r in rows) if total is None else total
    out: dict[str, float] = {}
    if total <= 0:
        return pd.Series(dtype="float64")
    for r in rows:
        hit = cusips.get(r.cusip)
        if hit is not None and hit.ticker:
            out[hit.ticker] = out.get(hit.ticker, 0.0) + r.value / total
    return pd.Series(out, dtype="float64").sort_values(ascending=False)


def covered(rows: Sequence[Row]) -> tuple[Row, ...]:
    """The largest holdings up to ``COVER_SHARE`` of the value or ``COVER_MAX``.

    Only these get prices and sectors; the rest are 미포함.
    """
    total = sum(r.value for r in rows)
    out: list[Row] = []
    held = 0.0
    for r in sorted(rows, key=lambda r: (-r.value, r.cusip)):
        if len(out) >= COVER_MAX or (total > 0 and held >= COVER_SHARE * total):
            break
        out.append(r)
        held += r.value
    return tuple(out)


def excluded_share(rows: Sequence[Row]) -> float:
    """Share of the value outside ``covered``: 미포함 비중."""
    total = sum(r.value for r in rows)
    if total <= 0:
        return math.nan
    return 1 - sum(r.value for r in covered(rows)) / total


def analysis_weights(rows: Sequence[Row], cusips: Mapping[str, Link]) -> pd.Series:
    """Covered, mapped holdings as shares of the whole reported value."""
    return weights(covered(rows), cusips, total=sum(r.value for r in rows))


def filed_schedule(
    found: Sequence[Quarter],
    cusips: Mapping[str, Link],
    days: Mapping[str, date],
) -> list[tuple[date, pd.Series]]:
    """Weights as a follower knew them, from each filing's ``buy_day``.

    ``days`` maps accessions to buy days; one missing is bought the day after
    filing. A filing for an older period than one already public (a late
    amendment) is left out, so the newest quarter is never replaced.
    """
    events = sorted(
        (
            (days.get(accession, filed + timedelta(days=1)), q.period, rows)
            for q in found
            for accession, filed, rows in q.versions
        ),
        key=lambda it: (it[0], it[1]),
    )
    out: list[tuple[date, pd.Series]] = []
    newest: date | None = None
    for day, period, rows in events:
        if newest is not None and period < newest:
            continue
        newest = period
        out.append((day, analysis_weights(rows, cusips)))
    return out


def period_schedule(
    found: Sequence[Quarter], cusips: Mapping[str, Link]
) -> list[tuple[date, pd.Series]]:
    """Final weights from each quarter end: theoretical, no filing delay."""
    return [
        (q.period, analysis_weights(q.rows, cusips)) for q in sorted(found, key=_period)
    ]


def top_schedule(
    schedule: Sequence[tuple[date, pd.Series]],
    n: int = CONVICTION,
    *,
    equal: bool = False,
) -> list[tuple[date, pd.Series]]:
    """Each set cut to its ``n`` largest weights (``perf.top_weights``), same days."""
    return [
        (day, perf.top_weights(w, n, equal=equal) if not w.empty else w)
        for day, w in schedule
    ]


def _period(q: Quarter) -> date:
    return q.period


def change_labels(now: Quarter, before: Quarter | None) -> dict[str, str]:
    """CUSIP → 신규, 증가, 감소 or 유지 by shares against the previous quarter."""
    if before is None:
        return {}
    old = {r.cusip: r.shares for r in before.rows}
    out: dict[str, str] = {}
    for r in now.rows:
        if r.cusip not in old:
            out[r.cusip] = CHANGES[0]
        elif r.shares > old[r.cusip]:
            out[r.cusip] = CHANGES[1]
        elif r.shares < old[r.cusip]:
            out[r.cusip] = CHANGES[2]
        else:
            out[r.cusip] = CHANGES[3]
    return out


def sold_out(now: Quarter, before: Quarter | None) -> list[Row]:
    """Rows of the previous quarter that are gone now."""
    if before is None:
        return []
    held = {r.cusip for r in now.rows}
    return [r for r in before.rows if r.cusip not in held]


def unmapped_share(rows: Iterable[Row], cusips: Mapping[str, Link]) -> float:
    rows = list(rows)
    total = sum(r.value for r in rows)
    if total <= 0:
        return math.nan
    lost = sum(
        r.value for r in rows if not (r.cusip in cusips and cusips[r.cusip].ticker)
    )
    return lost / total


def top_share(rows: Sequence[Row], count: int = 10) -> float:
    total = sum(r.value for r in rows)
    if total <= 0:
        return math.nan
    return sum(r.value for r in sorted(rows, key=lambda r: -r.value)[:count]) / total


def holdings_frame(
    q: Quarter,
    before: Quarter | None,
    cusips: Mapping[str, Link],
    sectors: Mapping[str, Sector],
) -> pd.DataFrame:
    """One row per CUSIP: ticker (or empty), name, sector, shares, value, weight."""
    labels = change_labels(q, before)
    total = q.value
    inside = {r.cusip for r in covered(q.rows)}
    rows: list[dict[str, object]] = []
    for r in q.rows:
        hit = cusips.get(r.cusip)
        ticker = hit.ticker if hit is not None and hit.ticker else ""
        found = sectors.get(ticker) if ticker else None
        sector = found.sector if found and found.sector else None
        if sector is None:
            sector = UNKNOWN if r.cusip in inside else EXCLUDED
        rows.append(
            {
                "ticker": ticker,
                "name": r.name,
                "sector": sector,
                "shares": r.shares,
                "value": r.value,
                "weight": r.value / total if total > 0 else math.nan,
                "change": labels.get(r.cusip, "—"),
                "cusip": r.cusip,
            }
        )
    columns = ["ticker", "name", "sector", "shares", "value", "weight", "change"]
    return pd.DataFrame(rows, columns=[*columns, "cusip"])


def sector_weights(frame: pd.DataFrame) -> pd.Series:
    return (
        cast("pd.Series", frame.groupby("sector")["weight"].sum())
        .sort_values(ascending=False)
        .rename("weight")
    )


def weight_history(found: Sequence[Quarter], cusip: str) -> list[tuple[date, float]]:
    """The CUSIP's weight in each quarter, oldest first (0 when not held)."""
    out: list[tuple[date, float]] = []
    for q in sorted(found, key=_period):
        total = q.value
        value = sum(r.value for r in q.rows if r.cusip == cusip)
        out.append((q.period, value / total if total > 0 else math.nan))
    return out


def overlap(theirs: pd.Series, mine: pd.Series) -> list[tuple[str, float, float]]:
    """Tickers in both with each side's weight, largest combined first."""
    common = set(map(str, theirs.index)) & set(map(str, mine.index))
    rows = [(t, _at(theirs, t), _at(mine, t)) for t in common]
    return sorted(rows, key=lambda it: (-(it[1] + it[2]), it[0]))


def _at(series: pd.Series, key: str) -> float:
    return float(cast("float", series[key]))


def tickers(found: Iterable[Quarter], cusips: Mapping[str, Link]) -> list[str]:
    """Every mapped ticker among each quarter's ``covered`` holdings."""
    out = {
        cusips[r.cusip].ticker
        for q in found
        for r in covered(q.rows)
        if r.cusip in cusips and cusips[r.cusip].ticker
    }
    return sorted(t for t in out if t)


# ---------------------------------------------------------------- fetching


class Source(Protocol):
    def sec(self, url: str) -> str:
        """The body of an SEC URL; raises on an HTTP error."""
        ...

    def figi(self, jobs: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
        """One OpenFIGI mapping request."""
        ...

    def sector(self, ticker: str) -> str | None:
        """The yfinance sector ("ETF" for funds), None when there is none."""
        ...


type Report = Callable[[str, int, int], None]


def submissions(source: Source, cik: str) -> dict[str, Any]:
    raw = json.loads(source.sec(SUBMISSIONS_URL.format(name=f"CIK{cik}.json")))
    return cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}


def filer_filings(
    source: Source,
    cik: str,
    sleep: Callable[[float], None],
    quarters: int = QUARTERS,
) -> tuple[str, list[Filing], Filing | None]:
    """The name, 13F-HR(/A) filings (older pages if needed), newest 13F-NT."""
    body = submissions(source, cik)
    sleep(SEC_PAUSE)
    name = str(body.get("name") or cik)
    history = cast("dict[str, Any]", body.get("filings") or {})
    recent = cast("dict[str, Any]", history.get("recent") or {})
    found = filings_from(recent)
    notices = filings_from(recent, NOTICES)
    newest = max(notices, key=lambda f: (f.filed, f.accession), default=None)
    pages = cast("list[dict[str, Any]]", history.get("files") or [])
    for page in pages:
        if len(recent_periods(found, quarters + 1)) > quarters:
            break  # one more period than kept: no later amendment is missed
        raw = json.loads(source.sec(SUBMISSIONS_URL.format(name=page["name"])))
        sleep(SEC_PAUSE)
        found += filings_from(cast("dict[str, Any]", raw))
    return name, found, newest


def check(source: Source, cik: str, sleep: Callable[[float], None]) -> str:
    """The filer's name; raises ``InvestorError`` if it has filed no 13F-HR."""
    try:
        name, found, _ = filer_filings(source, cik, sleep, quarters=1)
    except httpx.HTTPStatusError as error:
        if error.response.status_code == NOT_FOUND:
            raise InvestorError("EDGAR에 없는 CIK입니다.") from error
        raise
    if not found:
        raise InvestorError(f"{name}: 13F-HR 제출 기록이 없습니다.")
    return name


def fetch_table(
    source: Source, cik: str, filing: Filing, sleep: Callable[[float], None]
) -> Table:
    """One filing's information table (and amendment type) from the archive."""
    folder_name = filing.accession.replace("-", "")
    url = partial_url(cik, folder_name)
    index = json.loads(source.sec(url("index.json")))
    sleep(SEC_PAUSE)
    amendment: Amendment = ""
    if filing.form != "13F-HR":
        amendment = parse_amendment(source.sec(url("primary_doc.xml")))
        sleep(SEC_PAUSE)
    head = Table(
        accession=filing.accession,
        form=filing.form,
        filed=filing.filed,
        period=filing.period,
        amendment=amendment,
        has_table=False,
    )
    for name in table_files(cast("dict[str, Any]", index)):
        text = source.sec(url(name))
        sleep(SEC_PAUSE)
        if not is_table(text):
            continue
        parsed = parse_table(text, filing.filed)
        return head.model_copy(
            update={
                "has_table": True,
                "rows": parsed.rows,
                "options": parsed.options,
                "options_value": parsed.options_value,
                "principal": parsed.principal,
                "principal_value": parsed.principal_value,
            }
        )
    return head


def partial_url(cik: str, folder_name: str) -> Callable[[str], str]:
    def url(name: str) -> str:
        return ARCHIVE_URL.format(cik=int(cik), folder=folder_name, name=name)

    return url


def map_cusips(  # noqa: PLR0913 - one stage's inputs
    source: Source,
    root: Path,
    cusips: Sequence[str],
    today: date,
    report: Report,
    sleep: Callable[[float], None],
) -> int:
    """Map the due CUSIPs, saving after each request; returns how many were asked."""
    cache = load_cusips(root)
    todo = [
        c
        for c in dict.fromkeys(cusips)
        if due(
            cache[c].at if c in cache else None,
            c in cache and cache[c].ticker is not None,
            today,
        )
    ]
    for i in range(0, len(todo), FIGI_JOBS):
        report("티커 연결", i, len(todo))
        batch = todo[i : i + FIGI_JOBS]
        jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in batch]
        results = source.figi(jobs)
        for cusip, result in zip(batch, results, strict=False):
            picked = pick_ticker(result)
            if picked is not None:
                cache[cusip] = Link(ticker=picked[0], name=picked[1], at=today)
        save_cusips(root, cache)
        if i + FIGI_JOBS < len(todo):
            sleep(FIGI_PAUSE)
    return len(todo)


def fetch_sectors(  # noqa: PLR0913 - one stage's inputs
    source: Source,
    root: Path,
    names: Sequence[str],
    today: date,
    report: Report,
    sleep: Callable[[float], None],
) -> int:
    """Fetch the due sectors, saving as it goes; returns the tickers not answered."""
    cache = load_sectors(root)
    todo = [
        t
        for t in dict.fromkeys(names)
        if due(
            cache[t].at if t in cache else None,
            t in cache and cache[t].sector is not None,
            today,
        )
    ]
    failed = streak = 0
    for i, ticker in enumerate(todo):
        report("섹터", i, len(todo))
        try:
            cache[ticker] = Sector(sector=source.sector(ticker), at=today)
            streak = 0
        except Exception:  # noqa: BLE001 - left for the next update
            failed += 1
            streak += 1
            if streak >= YAHOO_FAILURES:
                failed += len(todo) - i - 1
                break
        if i % 20 == 19 or i == len(todo) - 1:  # noqa: PLR2004 - save every 20
            save_sectors(root, cache)
        sleep(YAHOO_PAUSE)
    if todo:
        save_sectors(root, cache)
    return failed


def update(  # noqa: PLR0913 - one investor's inputs
    source: Source,
    root: Path,
    cik: str,
    now: datetime,
    report: Report,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Fetch the investor's new filings, their tickers and sectors; one-line summary."""
    report("제출 목록", 0, 0)
    name, found, newest = filer_filings(source, cik, sleep)
    keep = kept(found)
    save_model(
        folder(root, cik) / FILINGS_FILE,
        Filer(cik=cik, name=name, fetched=now, filings=tuple(keep), notice=newest),
    )
    new = [f for f in keep if not (folder(root, cik) / f"{f.accession}.json").is_file()]
    for i, filing in enumerate(new):
        report("13F 문서", i, len(new))
        table = fetch_table(source, cik, filing, sleep)
        save_model(folder(root, cik) / f"{filing.accession}.json", table)
    filer = load_filer(root, cik)
    tables = load_tables(root, filer) if filer is not None else []
    cusips = [r.cusip for t in tables for r in t.rows]
    asked = map_cusips(source, root, cusips, now.date(), report, sleep)
    mapping = load_cusips(root)
    names = tickers(quarters(tables), mapping)
    missed = fetch_sectors(source, root, names, now.date(), report, sleep)
    linked = len({c for c in cusips if c in mapping and mapping[c].ticker})
    note = f" · 섹터 받지 못함 {missed}" if missed else ""
    return (
        f"{name} · {len(recent_periods(keep))}개 분기 · 새 문서 {len(new)}"
        f" · CUSIP 조회 {asked} · 티커 연결 {linked}/{len(set(cusips))}{note}"
    )


@final
class Web:
    """The real source: SEC with the user's User-Agent, OpenFIGI, yfinance."""

    def __init__(
        self, agent: str, root: Path, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        """``agent`` goes to the SEC only; yfinance's caches go under ``root``."""
        self._sec = httpx.Client(
            headers={"User-Agent": agent, "Accept-Encoding": "gzip, deflate"},
            timeout=TIMEOUT,
            follow_redirects=True,
        )
        self._figi = httpx.Client(timeout=TIMEOUT)
        self._root = root
        self._sleep = sleep
        self._yahoo = False

    def sec(self, url: str) -> str:
        response = self._sec.get(url)
        _ = response.raise_for_status()
        return response.text

    def figi(self, jobs: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
        for _ in range(FIGI_RETRIES):
            response = self._figi.post(OPENFIGI_URL, json=list(jobs))
            if response.status_code != TOO_MANY:
                _ = response.raise_for_status()
                return cast("list[dict[str, Any]]", response.json())
            wait = response.headers.get("retry-after", "")
            self._sleep(float(wait) if wait.isdigit() else FIGI_WAIT)
        raise InvestorError("OpenFIGI가 요청을 제한했습니다. 잠시 뒤 다시 갱신하세요.")

    def sector(self, ticker: str) -> str | None:
        import yfinance as yf  # noqa: PLC0415 - optional dependency

        if not self._yahoo:
            self._root.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(self._root / YFINANCE_CACHE))
            self._yahoo = True
        info = cast("Mapping[str, object]", yf.Ticker(ticker).info or {})
        sector = info.get("sector")
        if isinstance(sector, str) and sector:
            return sector
        return "ETF" if info.get("quoteType") == "ETF" else None

    def close(self) -> None:
        self._sec.close()
        self._figi.close()


@dataclass(frozen=True, slots=True)
class Progress:
    running: bool = False
    cik: str = ""
    stage: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    failed: bool = False


type Build = Callable[[str, Path], Source]


def _web(agent: str, root: Path) -> Source:
    return Web(agent, root)


@final
class Fetcher:
    """Fetches one investor at a time in a background thread; pages poll ``progress``.

    A CIK started once is not started again on its own; the header's 갱신
    button fetches it again.
    """

    __slots__ = ("_build", "_lock", "_progress", "_sleep", "_tried", "root")

    def __init__(
        self,
        root: Path,
        build: Build = _web,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Save under ``root``; ``build`` makes the source from the User-Agent."""
        self.root = root
        self._build = build
        self._sleep = sleep
        self._lock = threading.Lock()
        self._progress = Progress()
        self._tried: set[str] = set()

    @property
    def progress(self) -> Progress:
        with self._lock:
            return self._progress

    def tried(self, cik: str) -> bool:
        with self._lock:
            return cik in self._tried

    def start(self, cik: str, agent: str) -> bool:
        """Fetch ``cik`` unless a fetch is running or there is no User-Agent."""
        with self._lock:
            if self._progress.running or not agent.strip():
                return False
            self._tried.add(cik)
            self._progress = Progress(running=True, cik=cik, stage="준비")
        threading.Thread(
            target=self._run,
            args=(cik, agent.strip()),
            name="finagent-investors",
            daemon=True,
        ).start()
        return True

    def check(self, cik: str, agent: str) -> str:
        """The filer's name, in the calling thread; raises ``InvestorError``."""
        if not agent.strip():
            raise InvestorError(f"`.env`에 {AGENT_KEY}를 넣으세요.")
        source = self._build(agent.strip(), self.root)
        try:
            return check(source, cik, self._sleep)
        except httpx.HTTPError as error:
            raise InvestorError(f"EDGAR에 묻지 못했습니다: {error}") from error
        finally:
            _close(source)

    def _report(self, cik: str, stage: str, done: int, total: int) -> None:
        with self._lock:
            self._progress = Progress(
                running=True, cik=cik, stage=stage, done=done, total=total
            )

    def _run(self, cik: str, agent: str) -> None:
        source = self._build(agent, self.root)
        try:
            message = update(
                source,
                self.root,
                cik,
                datetime.now().astimezone(),
                lambda s, d, t: self._report(cik, s, d, t),
                self._sleep,
            )
        except Exception as error:  # noqa: BLE001 - shown on the page, not lost
            self._finish(cik, f"받지 못했습니다: {type(error).__name__}: {error}")
        else:
            self._finish(cik, message, failed=False)
        finally:
            _close(source)

    def _finish(self, cik: str, message: str, *, failed: bool = True) -> None:
        with self._lock:
            self._progress = Progress(cik=cik, message=message, failed=failed)


def _close(source: Source) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        _ = close()


# ---------------------------------------------------------------- text


def dollars(value: float) -> str:
    """Dollars in B/M above a million."""
    if not math.isfinite(value):
        return UNKNOWN
    for size, unit in ((1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"${value / size:,.1f}{unit}"
    return f"${value:,.0f}"


def latest(filer: Filer | None) -> Filing | None:
    """The newest period's first filing."""
    if filer is None or not filer.filings:
        return None
    period = max(f.period for f in filer.filings)
    return min((f for f in filer.filings if f.period == period), key=lambda f: f.filed)


def quarter_label(q: Quarter) -> str:
    amended = f" · 정정 {len(q.amended)}" if q.amended else ""
    return f"{q.period} 기준 · 제출 {q.filed}{amended}"
