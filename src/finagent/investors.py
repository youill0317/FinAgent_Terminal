# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitStringConcatenation=false
"""SEC 13F 기관투자자 공시 수집·CUSIP-티커 매핑 및 복제 포트폴리오 백테스트."""

from __future__ import annotations

import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as clock
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Protocol, cast, final
from xml.etree import ElementTree as ET

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

import finagent.perf as perf  # noqa: PLR0402  # 도메인 네임스페이스 명시적 참조
from finagent import layout, limits, memory, sec, yahoo

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

LIST_FILE: Final = layout.USER_DIR / "investors.json"
AGENT_KEY: Final = "SEC_USER_AGENT"
FIGI_KEY: Final = "OPENFIGI_API_KEY"
FIGI_HEADER: Final = "X-OPENFIGI-APIKEY"
OPENFIGI_URL: Final = "https://api.openfigi.com/v3/mapping"
FORMS: Final = ("13F-HR", "13F-HR/A")
NOTICES: Final = ("13F-NT", "13F-NT/A")
QUARTERS: Final = 8
MARKET_CLOSE: Final = clock(16, 0)
DEADLINE_DAYS: Final = 45
STALE_QUARTERS: Final = 2
COVER_SHARE: Final = 0.90
COVER_MAX: Final = 100
DOLLARS_FROM: Final = date(2023, 1, 3)
FIGI_JOBS: Final = 10
FIGI_KEYED_JOBS: Final = 100
YAHOO_FAILURES: Final = 5
TIMEOUT: Final = 30.0
NOT_FOUND: Final = 404
CONVICTION: Final = 10
UNKNOWN: Final = "미확인"
EXCLUDED: Final = "미포함"
CHANGES: Final = ("신규", "증가", "감소", "유지")
CIK_PATTERN: Final = re.compile(r"\d{1,10}")
TICKER_PATTERN: Final = re.compile(r"[A-Z0-9][A-Z0-9-]{0,9}")
OPTION_TYPES: Final = frozenset({"PUT", "CALL"})
ISIN_LENGTH: Final = 12
HK_DIGITS: Final = 4
EXCHANGE_SUFFIX: Final[Mapping[str, tuple[str, str]]] = {
    "KS": ("KR", ".KS"),
    "KQ": ("KR", ".KQ"),
    "TT": ("TW", ".TW"),
    "JT": ("JP", ".T"),
    "HK": ("HK", ".HK"),
    "CG": ("CN", ".SS"),
    "CS": ("CN", ".SZ"),
    "IS": ("IN", ".NS"),
    "LN": ("GB", ".L"),
    "GY": ("DE", ".DE"),
    "FP": ("FR", ".PA"),
    "NA": ("NL", ".AS"),
    "SE": ("CH", ".SW"),
    "SM": ("ES", ".MC"),
    "IM": ("IT", ".MI"),
    "SS": ("SE", ".ST"),
    "DC": ("DK", ".CO"),
    "FH": ("FI", ".HE"),
    "NO": ("NO", ".OL"),
    "BB": ("BE", ".BR"),
    "ID": ("IE", ".IR"),
    "PL": ("PT", ".LS"),
    "AV": ("AT", ".VI"),
    "CT": ("CA", ".TO"),
    "AT": ("AU", ".AX"),
    "BS": ("BR", ".SA"),
    "MM": ("MX", ".MX"),
    "SP": ("SG", ".SI"),
    "TB": ("TH", ".BK"),
    "IJ": ("ID", ".JK"),
    "MK": ("MY", ".KL"),
    "PM": ("PH", ".PS"),
    "AB": ("SA", ".SR"),
    "IT": ("IL", ".TA"),
    "SJ": ("ZA", ".JO"),
}

Amendment = Literal["", "RESTATEMENT", "NEW HOLDINGS", "UNKNOWN"]


class InvestorError(ValueError): ...


@dataclass(frozen=True, slots=True)
class Investor:
    cik: str
    name: str
    manager: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.manager})" if self.manager else self.name


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


class _Frozen(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class Filing(_Frozen):
    accession: str
    form: str
    filed: date
    period: date
    accepted: datetime | None = None


class Filer(_Frozen):
    cik: str
    name: str
    fetched: datetime
    filings: tuple[Filing, ...] = ()
    notice: Filing | None = None


class Row(_Frozen):
    cusip: str
    name: str
    shares: float
    value: float


class Table(_Frozen):
    accession: str
    form: str
    filed: date
    period: date
    amendment: Amendment = ""
    has_table: bool = True
    rows: tuple[Row, ...] = ()
    options: int = 0
    options_value: float = 0.0
    principal: int = 0
    principal_value: float = 0.0


class Link(_Frozen):
    ticker: str | None


class Sector(_Frozen):
    sector: str | None


class Added(_Frozen):
    added: tuple[tuple[str, str], ...] = ()


def clean_cik(text: str) -> str:
    digits = text.strip().upper().removeprefix("CIK").strip()
    if not CIK_PATTERN.fullmatch(digits) or int(digits) == 0:
        raise InvestorError("CIK는 숫자 10자리 이내입니다")
    return digits.zfill(10)


def value_unit(filed: date) -> int:
    return 1 if filed >= DOLLARS_FROM else 1000


def yahoo_ticker(ticker: str) -> str | None:
    text = ticker.strip().upper().replace("/", "-").replace(".", "-")
    return text if TICKER_PATTERN.fullmatch(text) else None


def filings_from(
    arrays: Mapping[str, Any], forms: Sequence[str] = FORMS
) -> list[Filing]:
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
    text = str(value or "").strip().removesuffix("Z")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None) if text else None
    except ValueError:
        return None


def buy_day(filing: Filing) -> date:
    if filing.accepted is None:
        return filing.filed + timedelta(days=1)
    day = filing.accepted.date()
    late = filing.accepted.time() >= MARKET_CLOSE
    return day + timedelta(days=1) if late else day


def buy_days(filer: Filer | None) -> dict[str, date]:
    return {f.accession: buy_day(f) for f in filer.filings} if filer else {}


def untimed(filer: Filer | None) -> int:
    return sum(f.accepted is None for f in filer.filings) if filer else 0


def notice(filer: Filer | None) -> Filing | None:
    if filer is None or filer.notice is None:
        return None
    last = max((f.filed for f in filer.filings), default=None)
    return filer.notice if last is None or filer.notice.filed > last else None


def _quarter_end(day: date) -> date:
    ends = [date(day.year, m, d) for m, d in ((3, 31), (6, 30), (9, 30), (12, 31))]
    past = [end for end in ends if end <= day]
    return past[-1] if past else date(day.year - 1, 12, 31)


def expected_period(today: date) -> date:
    end = _quarter_end(today)
    while end + timedelta(days=DEADLINE_DAYS) > today:
        end = _quarter_end(end - timedelta(days=1))
    return end


def _index(day: date) -> int:
    return day.year * 4 + (day.month - 1) // 3


def stale(filer: Filer | None, today: date) -> date | None:
    if filer is None or not filer.filings:
        return None
    last = max(f.period for f in filer.filings)
    behind = _index(expected_period(today)) - _index(last)
    return last if behind >= STALE_QUARTERS else None


def recent_periods(filings: Iterable[Filing], quarters: int = QUARTERS) -> list[date]:
    return sorted({f.period for f in filings}, reverse=True)[:quarters]


def kept(filings: Iterable[Filing], quarters: int = QUARTERS) -> list[Filing]:
    filings = list(filings)
    periods = set(recent_periods(filings, quarters))
    unique = {f.accession: f for f in filings if f.period in periods}
    return sorted(unique.values(), key=lambda f: (f.filed, f.accession))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse(text: str) -> ET.Element:
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
    items = cast("list[dict[str, Any]]", index.get("directory", {}).get("item", []))
    names = [
        (str(i.get("name", "")), float(i.get("size") or 0))
        for i in items
        if str(i.get("name", "")).lower().endswith(".xml")
        and str(i.get("name", "")).lower() != "primary_doc.xml"
    ]
    names.sort(key=lambda it: ("info" not in it[0].lower(), -it[1]))
    return [name for name, _ in names]


def _equity(result: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    if "error" in result and "data" not in result:
        message = str(result["error"]).lower()
        return [] if "no identifier" in message else None
    data = cast("list[dict[str, Any]]", result.get("data") or [])
    return [d for d in data if d.get("marketSector") == "Equity"] or data


def pick_ticker(result: Mapping[str, Any]) -> str | None:
    equity = _equity(result)
    if equity is None:
        return None
    for item in equity:
        ticker = yahoo_ticker(str(item.get("ticker") or ""))
        if ticker:
            return ticker
    return ""


def pick_listing(isin: str, result: Mapping[str, Any]) -> str | None:
    equity = _equity(result)
    if equity is None:
        return None
    for item in equity:
        code = str(item.get("exchCode") or "").split(" ", 1)[0]
        found = EXCHANGE_SUFFIX.get(code)
        symbol = str(item.get("ticker") or "").strip().upper()
        if found is None or found[0] != isin[:2] or not symbol:
            continue
        if code == "HK":
            symbol = symbol.zfill(HK_DIGITS)
        return f"{symbol}{found[1]}"
    return ""


def figi_job(code: str) -> dict[str, str]:
    if len(code) != ISIN_LENGTH:
        return {"idType": "ID_CUSIP", "idValue": code, "exchCode": "US"}
    if code.startswith("US"):
        return {"idType": "ID_ISIN", "idValue": code, "exchCode": "US"}
    return {"idType": "ID_ISIN", "idValue": code}


def _picked(code: str, result: Mapping[str, Any]) -> str | None:
    if len(code) == ISIN_LENGTH and not code.startswith("US"):
        return pick_listing(code, result)
    return pick_ticker(result)


def save_model(path: Path, model: BaseModel) -> None:
    layout.write_text(path, model.model_dump_json(indent=1) + "\n")


def load_added(root: Path) -> Added:
    path = root / LIST_FILE
    if not path.is_file():
        return Added()
    try:
        return Added.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return Added()


def roster(root: Path) -> list[Investor]:
    known = {i.cik for i in PRESETS}
    extra = [Investor(c, n) for c, n in load_added(root).added if c not in known]
    return [*PRESETS, *extra]


def add_investor(root: Path, cik: str, name: str) -> None:
    old = load_added(root)
    if any(c == cik for c, _ in old.added) or any(i.cik == cik for i in PRESETS):
        return
    save_model(root / LIST_FILE, Added(added=(*old.added, (cik, name))))


def remove_investor(root: Path, cik: str) -> None:
    kept_rows = tuple(row for row in load_added(root).added if row[0] != cik)
    save_model(root / LIST_FILE, Added(added=kept_rows))


def load_filer(cik: str) -> Filer | None:
    return memory.current().filers.get(cik)


def load_tables(filer: Filer) -> list[Table]:
    saved = memory.current().filings
    return [t for f in filer.filings if (t := saved.get(f.accession)) is not None]


def load_cusips() -> dict[str, Link]:
    return memory.current().cusips.rows()


def save_cusips(cache: Mapping[str, Link]) -> None:
    memory.current().cusips.update(cache)


def load_sectors() -> dict[str, Sector]:
    return memory.current().sectors.rows()


def stamp() -> tuple[int, ...]:
    found = memory.current()
    tables = (found.filers, found.filings, found.cusips, found.sectors)
    return tuple(table.version for table in tables)


@dataclass(frozen=True, slots=True)
class Quarter:
    period: date
    filed: date
    rows: tuple[Row, ...]
    amended: tuple[date, ...]
    ignored: int
    options: int
    options_value: float
    principal: int
    principal_value: float
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
            else:
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


def cover[T](ranked: Sequence[T], size: Callable[[T], float]) -> tuple[T, ...]:
    total = sum(size(item) for item in ranked)
    out: list[T] = []
    held = 0.0
    for item in ranked:
        if len(out) >= COVER_MAX or (total > 0 and held >= COVER_SHARE * total):
            break
        out.append(item)
        held += size(item)
    return tuple(out)


def covered(rows: Sequence[Row]) -> tuple[Row, ...]:
    return cover(sorted(rows, key=lambda r: (-r.value, r.cusip)), lambda r: r.value)


def excluded_share(rows: Sequence[Row]) -> float:
    total = sum(r.value for r in rows)
    if total <= 0:
        return math.nan
    return 1 - sum(r.value for r in covered(rows)) / total


def analysis_weights(rows: Sequence[Row], cusips: Mapping[str, Link]) -> pd.Series:
    return weights(covered(rows), cusips, total=sum(r.value for r in rows))


def filed_schedule(
    found: Sequence[Quarter],
    cusips: Mapping[str, Link],
    days: Mapping[str, date],
) -> list[tuple[date, pd.Series]]:
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
    return [
        (q.period, analysis_weights(q.rows, cusips)) for q in sorted(found, key=_period)
    ]


def top_schedule(
    schedule: Sequence[tuple[date, pd.Series]],
    n: int = CONVICTION,
    *,
    equal: bool = False,
) -> list[tuple[date, pd.Series]]:
    return [
        (day, perf.top_weights(w, n, equal=equal) if not w.empty else w)
        for day, w in schedule
    ]


def _period(q: Quarter) -> date:
    return q.period


def change_labels(now: Quarter, before: Quarter | None) -> dict[str, str]:
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
    if before is None:
        return []
    held = {r.cusip for r in now.rows}
    return [r for r in before.rows if r.cusip not in held]


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
    labels = change_labels(q, before)
    total = q.value
    inside = {r.cusip for r in covered(q.rows)}
    rows: list[dict[str, object]] = []
    for r in q.rows:
        hit = cusips.get(r.cusip)
        ticker = hit.ticker if hit is not None and hit.ticker else ""
        found = sectors.get(ticker) if ticker else None
        if r.cusip not in inside:
            sector = EXCLUDED
        else:
            sector = found.sector if found and found.sector else UNKNOWN
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
    out: list[tuple[date, float]] = []
    for q in sorted(found, key=_period):
        total = q.value
        value = sum(r.value for r in q.rows if r.cusip == cusip)
        out.append((q.period, value / total if total > 0 else math.nan))
    return out


def overlap(theirs: pd.Series, mine: pd.Series) -> list[tuple[str, float, float]]:
    common = set(map(str, theirs.index)) & set(map(str, mine.index))
    rows = [(t, _at(theirs, t), _at(mine, t)) for t in common]
    return sorted(rows, key=lambda it: (-(it[1] + it[2]), it[0]))


def _at(series: pd.Series, key: str) -> float:
    return float(cast("float", series[key]))


def tickers(found: Iterable[Quarter], cusips: Mapping[str, Link]) -> list[str]:
    out = {
        cusips[r.cusip].ticker
        for q in found
        for r in covered(q.rows)
        if r.cusip in cusips and cusips[r.cusip].ticker
    }
    return sorted(t for t in out if t)


class Source(Protocol):
    def sec(self, url: str) -> str: ...

    def figi(self, jobs: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]: ...

    def sector(self, ticker: str) -> str | None: ...


type Report = Callable[[str, int, int], None]


def submissions(source: Source, cik: str) -> dict[str, Any]:
    raw = json.loads(source.sec(sec.submissions_url(f"CIK{cik}.json")))
    return cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}


def filer_filings(
    source: Source, cik: str, quarters: int = QUARTERS
) -> tuple[str, list[Filing], Filing | None]:
    body = submissions(source, cik)
    name = str(body.get("name") or cik)
    history = cast("dict[str, Any]", body.get("filings") or {})
    recent = cast("dict[str, Any]", history.get("recent") or {})
    found = filings_from(recent)
    notices = filings_from(recent, NOTICES)
    newest = max(notices, key=lambda f: (f.filed, f.accession), default=None)
    pages = cast("list[dict[str, Any]]", history.get("files") or [])
    for page in pages:
        if len(recent_periods(found, quarters + 1)) > quarters:
            break
        raw = json.loads(source.sec(sec.submissions_url(page["name"])))
        found += filings_from(cast("dict[str, Any]", raw))
    return name, found, newest


def check(source: Source, cik: str) -> str:
    try:
        name, found, _ = filer_filings(source, cik, quarters=1)
    except httpx.HTTPStatusError as error:
        if error.response.status_code == NOT_FOUND:
            raise InvestorError("EDGAR에 없는 CIK입니다") from error
        raise
    if not found:
        raise InvestorError(f"{name}: 13F-HR 제출 기록이 없습니다")
    return name


def fetch_table(source: Source, cik: str, filing: Filing) -> Table:
    url = partial_url(cik, filing.accession)
    index = json.loads(source.sec(url("index.json")))
    amendment: Amendment = ""
    if filing.form != "13F-HR":
        amendment = parse_amendment(source.sec(url("primary_doc.xml")))
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


def partial_url(cik: str, accession: str) -> Callable[[str], str]:
    def url(name: str) -> str:
        return sec.archive_url(cik, accession, name)

    return url


_figi_key = ""


def set_figi_key(key: str) -> None:
    global _figi_key  # noqa: PLW0603
    _figi_key = key


def map_cusips(source: Source, cusips: Sequence[str], report: Report) -> int:
    cache = load_cusips()
    todo = [c for c in dict.fromkeys(cusips) if c not in cache]
    size = FIGI_KEYED_JOBS if _figi_key else FIGI_JOBS
    for i in range(0, len(todo), size):
        report("티커 연결", i, len(todo))
        batch = todo[i : i + size]
        results = source.figi([figi_job(c) for c in batch])
        for cusip, result in zip(batch, results, strict=False):
            picked = _picked(cusip, result)
            if picked is not None:
                cache[cusip] = Link(ticker=picked or None)
        save_cusips(cache)
    return len(todo)


def fetch_sectors(source: Source, names: Sequence[str], report: Report) -> int:
    cache = load_sectors()
    todo = [t for t in dict.fromkeys(names) if t not in cache]
    table = memory.current().sectors
    fresh: dict[str, Sector] = {}
    failed = streak = 0
    with ThreadPoolExecutor(yahoo.WORKERS) as pool:
        jobs = {pool.submit(source.sector, ticker): ticker for ticker in todo}
        for i, job in enumerate(as_completed(jobs)):
            report("섹터", i, len(todo))
            try:
                fresh[jobs[job]] = Sector(sector=job.result())
                streak = 0
            except Exception:  # noqa: BLE001
                failed += 1
                streak += 1
                if streak == YAHOO_FAILURES:
                    for other in jobs:
                        _ = other.cancel()
            if i % 20 == 19:  # noqa: PLR2004
                table.update(fresh)
    if todo:
        table.update(fresh)
    return failed


def update(
    source: Source,
    cik: str,
    now: datetime,
    report: Report,
) -> str:
    report("제출 목록", 0, 0)
    name, found, newest = filer_filings(source, cik)
    keep = kept(found)
    memory.current().filers.put(
        cik, Filer(cik=cik, name=name, fetched=now, filings=tuple(keep), notice=newest)
    )
    new = [f for f in keep if memory.current().filings.get(f.accession) is None]
    for i, filing in enumerate(new):
        report("13F 문서", i, len(new))
        memory.current().filings.put(filing.accession, fetch_table(source, cik, filing))
    filer = load_filer(cik)
    tables = load_tables(filer) if filer is not None else []
    cusips = [r.cusip for t in tables for r in t.rows]
    asked = map_cusips(source, cusips, report)
    mapping = load_cusips()
    names = tickers(quarters(tables), mapping)
    missed = fetch_sectors(source, names, report)
    linked = len({c for c in cusips if c in mapping and mapping[c].ticker})
    note = f" · 섹터 미수신 {missed}" if missed else ""
    return (
        f"{name} · {len(recent_periods(keep))}개 분기 · 새 문서 {len(new)}"
        f" · CUSIP 조회 {asked} · 티커 연결 {linked}/{len(set(cusips))}{note}"
    )


@final
class Web:
    def __init__(self, agent: str) -> None:
        self._sec = sec.Client(agent)
        headers = {FIGI_HEADER: _figi_key} if _figi_key else {}
        self._figi = httpx.Client(timeout=TIMEOUT, headers=headers)

    def sec(self, url: str) -> str:
        return self._sec.text(url)

    def figi(self, jobs: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
        try:
            response = limits.send(
                "OPENFIGI", lambda: self._figi.post(OPENFIGI_URL, json=list(jobs))
            )
        except limits.LimitReached as error:
            message = "OpenFIGI 요청 한도 초과: 잠시 후 재시도 필요"
            raise InvestorError(message) from error
        _ = response.raise_for_status()
        return cast("list[dict[str, Any]]", response.json())

    def sector(self, ticker: str) -> str | None:
        info = yahoo.info(ticker)
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


@final
class Fetcher:
    __slots__ = ("_lock", "_progress", "_tried")

    def __init__(self) -> None:
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
        if not agent.strip():
            raise InvestorError(f"`.env` 내 {AGENT_KEY} 미설정")
        source: Web | None = None
        try:
            source = Web(agent.strip())
            return check(source, cik)
        except (httpx.HTTPError, UnicodeEncodeError) as error:
            raise InvestorError(f"EDGAR 공시 서버 조회 실패: {error}") from error
        finally:
            _close(source)

    def _report(self, cik: str, stage: str, done: int, total: int) -> None:
        with self._lock:
            self._progress = Progress(
                running=True, cik=cik, stage=stage, done=done, total=total
            )

    def _run(self, cik: str, agent: str) -> None:
        source: Web | None = None
        try:
            source = Web(agent)
            message = update(
                source,
                cik,
                datetime.now().astimezone(),
                lambda s, d, t: self._report(cik, s, d, t),
            )
        except Exception as error:  # noqa: BLE001  # 백그라운드 스레드 최상위 예외 포착
            self._finish(cik, f"수신 실패: {type(error).__name__}: {error}")
        else:
            self._finish(cik, message, failed=False)
        finally:
            _close(source)

    def _finish(self, cik: str, message: str, *, failed: bool = True) -> None:
        with self._lock:
            self._progress = Progress(cik=cik, message=message, failed=failed)


def _close(source: Web | None) -> None:
    if source is not None:
        source.close()


def dollars(value: float) -> str:
    if not math.isfinite(value):
        return UNKNOWN
    for size, unit in ((1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"${value / size:,.1f}{unit}"
    return f"${value:,.0f}"


def latest(filer: Filer | None) -> Filing | None:
    if filer is None or not filer.filings:
        return None
    period = max(f.period for f in filer.filings)
    return min((f for f in filer.filings if f.period == period), key=lambda f: f.filed)


def quarter_label(q: Quarter) -> str:
    amended = f" · 정정 {len(q.amended)}" if q.amended else ""
    return f"{q.period} 기준 · 제출 {q.filed}{amended}"
