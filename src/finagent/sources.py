"""Fetch specs for each public source. Each spec returns a dated series.

A spec is a frozen, hashable value so a refresh pass can fetch each distinct
series once even when several indicators or charts use it.

Yahoo (gold futures) is the one source read through ``yfinance``, an optional
package, instead of a URL here; its caches go under ``data/macro/.yfinance/``
(``tests/test_guardrails.py``).
"""

from __future__ import annotations

import csv
import io
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, cast, final

from pydantic import BaseModel, ConfigDict, ValidationError

from finagent.fetch import (
    ECOS_KEY_ENV,
    FRED_KEY_ENV,
    MissingKeyError,
    SourceError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from finagent.fetch import FetchContext
    from finagent.models import SourceId

type Point = tuple[date, float]
type Series = tuple[Point, ...]

FRED_OBSERVATIONS_URL: Final = "https://api.stlouisfed.org/fred/series/observations"
FRED_RELEASE_DATES_URL: Final = "https://api.stlouisfed.org/fred/releases/dates"
FRED_SEARCH_URL: Final = "https://api.stlouisfed.org/fred/series/search"
FISCAL_BASE_URL: Final = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
)
NYFED_BASE_URL: Final = "https://markets.newyorkfed.org/api/rates"
ECOS_BASE_URL: Final = "https://ecos.bok.or.kr/api/StatisticSearch"
ECOS_TABLE_LIST_URL: Final = "https://ecos.bok.or.kr/api/StatisticTableList"
ECOS_ITEM_LIST_URL: Final = "https://ecos.bok.or.kr/api/StatisticItemList"
ECB_BASE_URL: Final = "https://data-api.ecb.europa.eu/service/data"
BIS_BASE_URL: Final = "https://stats.bis.org/api/v1/data"
OECD_BASE_URL: Final = "https://sdmx.oecd.org/public/rest/data"
# SDMX structure queries: ``{root}/dataflow/{agency}`` and one flow with its codes.
SDMX_STRUCTURE_ROOTS: Final[dict[str, tuple[str, str]]] = {
    "ECB": ("https://data-api.ecb.europa.eu/service/dataflow", "ECB"),
    "BIS": ("https://stats.bis.org/api/v1/dataflow", "BIS"),
    "OECD": ("https://sdmx.oecd.org/public/rest/dataflow", "all"),
}

# yfinance's caches, under the folder the app runs from.
YAHOO_CACHE: Final = Path("data") / "macro" / ".yfinance"

ECOS_MAX_ROWS: Final = 10000
FISCAL_PAGE_SIZE: Final = 10000
NYFED_MAX_ROWS: Final = 1000


def years_before(today: date, years: int) -> date:
    return today - timedelta(days=round(365.25 * years))


# ---------------------------------------------------------------- period parsing

_PERIOD_PATTERNS: Final = (
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), "ymd"),
    (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), "ymd"),
    (re.compile(r"^(\d{4})-?Q([1-4])$"), "q"),
    (re.compile(r"^(\d{4})-(\d{2})$"), "ym"),
    (re.compile(r"^(\d{4})(\d{2})$"), "ym"),
    (re.compile(r"^(\d{4})$"), "y"),
)


def parse_period(text: str) -> date | None:
    """Parse the period formats used by the sources into the period's first day."""
    value = text.strip()
    for pattern, kind in _PERIOD_PATTERNS:
        match = pattern.match(value)
        if match is None:
            continue
        parts = [int(group) for group in match.groups()]
        try:
            if kind == "ymd":
                return date(parts[0], parts[1], parts[2])
            if kind == "ym":
                return date(parts[0], parts[1], 1)
            if kind == "q":
                return date(parts[0], 3 * parts[1] - 2, 1)
            return date(parts[0], 1, 1)
        except ValueError:
            return None
    return None


def parse_number(text: str | float | None) -> float | None:
    if text is None:
        return None
    if isinstance(text, int | float):
        number = float(text)
        return None if math.isnan(number) else number
    cleaned = text.strip().replace(",", "")
    if cleaned in {"", ".", "null", "NaN", "NA", "-"}:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return None if math.isnan(number) else number


def to_series(points: Sequence[tuple[str, str | float | None]]) -> Series:
    clean: dict[date, float] = {}
    for period, raw in points:
        day = parse_period(period)
        number = parse_number(raw)
        if day is not None and number is not None:
            clean[day] = number
    return tuple(sorted(clean.items()))


def _require(series: Series, source: SourceId, what: str) -> Series:
    if not series:
        raise SourceError(f"{source} {what}: 관측값 없음")
    return series


class _Model(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")


# ---------------------------------------------------------------- FRED


class _FredObservation(_Model):
    date: str
    value: str


class _FredObservations(_Model):
    observations: tuple[_FredObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class FredSeries:
    series_id: str
    years: int = 4
    source: ClassVar[SourceId] = "FRED"

    async def fetch(self, ctx: FetchContext, today: date) -> Series:
        params = {
            "series_id": self.series_id,
            "api_key": ctx.key(FRED_KEY_ENV),
            "file_type": "json",
            "observation_start": years_before(today, self.years).isoformat(),
        }
        text = await ctx.text(self.source, FRED_OBSERVATIONS_URL, params)
        try:
            payload = _FredObservations.model_validate_json(text)
        except ValidationError as error:
            raise SourceError(f"FRED {self.series_id}: 응답 형식 오류") from error
        return _require(
            to_series([(row.date, row.value) for row in payload.observations]),
            self.source,
            self.series_id,
        )


class _FredReleaseDate(_Model):
    release_name: str
    date: str


class _FredReleaseDates(_Model):
    release_dates: tuple[_FredReleaseDate, ...] = ()


async def fred_release_dates(
    ctx: FetchContext, start: date, end: date
) -> tuple[tuple[date, str], ...]:
    params = {
        "api_key": ctx.key(FRED_KEY_ENV),
        "file_type": "json",
        "realtime_start": start.isoformat(),
        "realtime_end": end.isoformat(),
        "include_release_dates_with_no_data": "true",
        "limit": "1000",
        "sort_order": "asc",
    }
    text = await ctx.text("FRED", FRED_RELEASE_DATES_URL, params)
    try:
        payload = _FredReleaseDates.model_validate_json(text)
    except ValidationError as error:
        raise SourceError("FRED 발표 일정: 응답 형식 오류") from error
    out: list[tuple[date, str]] = []
    for row in payload.release_dates:
        day = parse_period(row.date)
        if day is not None and start <= day <= end:
            out.append((day, row.release_name))
    return tuple(out)


class _FredVintageObservation(_Model):
    realtime_start: str
    realtime_end: str
    date: str
    value: str


class _FredVintageObservations(_Model):
    observations: tuple[_FredVintageObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class Vintage:
    """A series as FRED published it from ``published`` on."""

    published: date
    series: Series


async def fred_vintages(
    ctx: FetchContext, series_id: str, since: date
) -> tuple[Vintage, Vintage | None]:
    """The current vintage of a series and the one it replaced.

    One request with a real-time window returns every vintage published since
    ``since``. FOMC projections get a new vintage at each SEP release.
    """
    params = {
        "series_id": series_id,
        "api_key": ctx.key(FRED_KEY_ENV),
        "file_type": "json",
        "realtime_start": since.isoformat(),
        "realtime_end": "9999-12-31",
    }
    text = await ctx.text("FRED", FRED_OBSERVATIONS_URL, params)
    try:
        payload = _FredVintageObservations.model_validate_json(text)
    except ValidationError as error:
        raise SourceError(f"FRED {series_id}: 응답 형식 오류") from error
    rows = [
        (start, end, row.date, row.value)
        for row in payload.observations
        if (start := parse_period(row.realtime_start)) is not None
        and (end := parse_period(row.realtime_end)) is not None
    ]
    if not rows:
        raise SourceError(f"FRED {series_id}: 관측값 없음")

    def as_of(day: date) -> Vintage | None:
        valid = [row for row in rows if row[0] <= day <= row[1]]
        series = to_series([(obs_date, value) for _, _, obs_date, value in valid])
        if not series:
            return None
        return Vintage(max(start for start, _, _, _ in valid), series)

    latest = as_of(max(start for start, _, _, _ in rows))
    if latest is None:
        raise SourceError(f"FRED {series_id}: 관측값 없음")
    return latest, as_of(latest.published - timedelta(days=1))


# ---------------------------------------------------------------- Treasury Fiscal Data


class _FiscalPayload(_Model):
    data: tuple[dict[str, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class FiscalSeries:
    endpoint: str
    value_field: str
    extra_filter: str = ""
    years: int = 4
    source: ClassVar[SourceId] = "TREASURY"

    async def fetch(self, ctx: FetchContext, today: date) -> Series:
        start = years_before(today, self.years).isoformat()
        filters = [f"record_date:gte:{start}"]
        if self.extra_filter:
            filters.append(self.extra_filter)
        params = {
            "fields": f"record_date,{self.value_field}",
            "filter": ",".join(filters),
            "sort": "-record_date",
            "page[size]": str(FISCAL_PAGE_SIZE),
            "format": "json",
        }
        text = await ctx.text(self.source, f"{FISCAL_BASE_URL}/{self.endpoint}", params)
        try:
            payload = _FiscalPayload.model_validate_json(text)
        except ValidationError as error:
            raise SourceError(f"Treasury {self.endpoint}: 응답 형식 오류") from error
        return _require(
            to_series(
                [
                    (row.get("record_date") or "", row.get(self.value_field))
                    for row in payload.data
                ]
            ),
            self.source,
            self.endpoint,
        )


# ---------------------------------------------------------------- New York Fed


class _NyFedRate(_Model):
    effectiveDate: str  # noqa: N815 - NY Fed field name
    percentRate: float | None = None  # noqa: N815


class _NyFedPayload(_Model):
    refRates: tuple[_NyFedRate, ...] = ()  # noqa: N815


@dataclass(frozen=True, slots=True)
class NyFedRate:
    path: str  # e.g. "unsecured/effr" or "secured/sofr"
    count: int = NYFED_MAX_ROWS
    source: ClassVar[SourceId] = "NYFED"

    async def fetch(self, ctx: FetchContext, _today: date) -> Series:
        url = f"{NYFED_BASE_URL}/{self.path}/last/{self.count}.json"
        text = await ctx.text(self.source, url)
        try:
            payload = _NyFedPayload.model_validate_json(text)
        except ValidationError as error:
            raise SourceError(f"NY Fed {self.path}: 응답 형식 오류") from error
        return _require(
            to_series(
                [(row.effectiveDate, row.percentRate) for row in payload.refRates]
            ),
            self.source,
            self.path,
        )


# ---------------------------------------------------------------- 한국은행 ECOS


class _EcosRow(_Model):
    TIME: str
    DATA_VALUE: str | None = None


class _EcosSearch(_Model):
    row: tuple[_EcosRow, ...] = ()


class _EcosResult(_Model):
    CODE: str
    MESSAGE: str = ""


class _EcosPayload(_Model):
    StatisticSearch: _EcosSearch | None = None
    RESULT: _EcosResult | None = None


def ecos_period(day: date, cycle: str) -> str:
    if cycle == "D":
        return day.strftime("%Y%m%d")
    if cycle == "M":
        return day.strftime("%Y%m")
    if cycle == "Q":
        return f"{day.year}Q{(day.month - 1) // 3 + 1}"
    return str(day.year)


@dataclass(frozen=True, slots=True)
class EcosSeries:
    stat: str
    cycle: str  # D, M, Q or A
    item1: str
    item2: str = ""
    years: int = 4
    source: ClassVar[SourceId] = "ECOS"

    async def fetch(self, ctx: FetchContext, today: date) -> Series:
        key = ctx.key(ECOS_KEY_ENV)
        start = ecos_period(years_before(today, self.years), self.cycle)
        end = ecos_period(today, self.cycle)
        parts = [
            ECOS_BASE_URL,
            key,
            "json",
            "kr",
            "1",
            str(ECOS_MAX_ROWS),
            self.stat,
            self.cycle,
            start,
            end,
            self.item1,
        ]
        if self.item2:
            parts.append(self.item2)
        text = await ctx.text(self.source, "/".join(parts))
        label = f"{self.stat}/{self.item1}"
        try:
            payload = _EcosPayload.model_validate_json(text)
        except ValidationError as error:
            raise SourceError(f"ECOS {label}: 응답 형식 오류") from error
        if payload.RESULT is not None:
            raise SourceError(
                f"ECOS {label}: {payload.RESULT.CODE} {payload.RESULT.MESSAGE}"
            )
        rows = payload.StatisticSearch.row if payload.StatisticSearch else ()
        return _require(
            to_series([(row.TIME, row.DATA_VALUE) for row in rows]), self.source, label
        )


# ------------------------------------------------------- SDMX CSV (ECB, BIS, OECD)

SDMX_FORMATS: Final[dict[str, tuple[str, str]]] = {
    "ECB": (ECB_BASE_URL, "csvdata"),
    "BIS": (BIS_BASE_URL, "csv"),
    "OECD": (OECD_BASE_URL, "csvfile"),
}


@dataclass(frozen=True, slots=True)
class SdmxSeries:
    """One series out of an SDMX CSV response.

    ``flow_key`` is the path after the provider's ``/data/`` root. ``match``
    picks rows when one request returns several series (OECD multi-country).
    """

    provider: SourceId
    flow_key: str
    match: tuple[tuple[str, str], ...] = ()
    years: int = 4

    @property
    def source(self) -> SourceId:
        return self.provider

    async def fetch(self, ctx: FetchContext, today: date) -> Series:
        base, fmt = SDMX_FORMATS[self.provider]
        start = years_before(today, self.years)
        params = {"startPeriod": start.strftime("%Y-%m"), "format": fmt}
        text = await ctx.text(self.provider, f"{base}/{self.flow_key}", params)
        rows = parse_sdmx_csv(text)
        wanted = dict(self.match)
        picked = [
            (row.get("TIME_PERIOD", ""), row.get("OBS_VALUE"))
            for row in rows
            if all(row.get(column) == value for column, value in wanted.items())
        ]
        return _require(to_series(picked), self.provider, self.flow_key)


def parse_sdmx_csv(text: str) -> list[dict[str, str]]:
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    header = next(reader, None)
    if header is None or "OBS_VALUE" not in header or "TIME_PERIOD" not in header:
        raise SourceError("SDMX 응답에 TIME_PERIOD/OBS_VALUE 열이 없습니다")
    return [dict(zip(header, row, strict=False)) for row in reader]


# ---------------------------------------------------------------- Yahoo Finance


@final
class YahooCloses:
    """The real ``Closes``: yfinance, with its caches under ``cache``."""

    def __init__(self, cache: Path) -> None:  # noqa: D107
        self._cache = cache
        self._pointed = False

    def __call__(self, ticker: str, start: date) -> Series:
        try:
            # Optional dependency, without type stubs.
            import yfinance as yf  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        except ImportError as error:
            raise MissingKeyError(
                "yfinance가 설치되지 않았습니다 (uv sync --extra screen)"
            ) from error
        if not self._pointed:
            self._cache.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(self._cache))
            self._pointed = True
        try:
            frame = yf.Ticker(ticker).history(  # pyright: ignore[reportUnknownMemberType]
                start=start.isoformat(), auto_adjust=False, raise_errors=True
            )
            stamps = cast("list[datetime]", frame.index.to_list())  # pyright: ignore[reportUnknownMemberType]
            closes = cast("Sequence[float]", frame["Close"])
        except Exception as error:  # yfinance raises many kinds
            raise SourceError(f"YAHOO {ticker}: {type(error).__name__}") from error
        return _require(
            to_series(
                [
                    (stamp.date().isoformat(), close)
                    for stamp, close in zip(stamps, closes, strict=True)
                ]
            ),
            "YAHOO",
            ticker,
        )


@dataclass(frozen=True, slots=True)
class YahooSeries:
    ticker: str
    years: int = 4
    source: ClassVar[SourceId] = "YAHOO"

    async def fetch(self, ctx: FetchContext, today: date) -> Series:
        return await ctx.closes(self.ticker, years_before(today, self.years))


# ---------------------------------------------------------------- derived


type Fetch = (
    FredSeries | FiscalSeries | NyFedRate | EcosSeries | SdmxSeries | YahooSeries
)


@dataclass(frozen=True, slots=True)
class Ratio:
    """``numerator / denominator`` on shared dates, e.g. a cross exchange rate."""

    numerator: Fetch
    denominator: Fetch

    @property
    def source(self) -> SourceId:
        return self.numerator.source

    def parts(self) -> tuple[Fetch, Fetch]:
        return (self.numerator, self.denominator)


def combine_ratio(numerator: Series, denominator: Series) -> Series:
    lookup = dict(denominator)
    return tuple(
        (day, value / lookup[day])
        for day, value in numerator
        if day in lookup and lookup[day] != 0
    )


# ---------------------------------------------------------------- search for tools


@dataclass(frozen=True, slots=True)
class SeriesInfo:
    """One search hit: ``code`` is what ``get_series`` takes for that source."""

    code: str
    name: str
    freq: str = ""
    unit: str = ""
    start: str = ""
    end: str = ""
    note: str = ""


class _FredSearchHit(_Model):
    id: str
    title: str
    frequency_short: str = ""
    units_short: str = ""
    observation_start: str = ""
    observation_end: str = ""
    notes: str = ""


class _FredSearch(_Model):
    seriess: tuple[_FredSearchHit, ...] = ()


async def fred_search(ctx: FetchContext, text: str, limit: int) -> list[SeriesInfo]:
    """FRED's own series search, most popular first."""
    params = {
        "search_text": text,
        "api_key": ctx.key(FRED_KEY_ENV),
        "file_type": "json",
        "limit": str(limit),
        "order_by": "popularity",
        "sort_order": "desc",
    }
    raw = await ctx.text("FRED", FRED_SEARCH_URL, params)
    try:
        payload = _FredSearch.model_validate_json(raw)
    except ValidationError as error:
        raise SourceError("FRED 검색: 응답 형식 오류") from error
    return [
        SeriesInfo(
            code=hit.id,
            name=hit.title,
            freq=hit.frequency_short,
            unit=hit.units_short,
            start=hit.observation_start,
            end=hit.observation_end,
            note=_copyright_note(hit.notes),
        )
        for hit in payload.seriess
    ]


def _copyright_note(notes: str) -> str:
    match = re.search(r"(Copyright[^.\n]*|©[^.\n]*)", notes)
    return match.group(1).strip() if match else ""


class _EcosListRow(_Model):
    STAT_CODE: str = ""
    STAT_NAME: str = ""
    CYCLE: str | None = None
    SRCH_YN: str | None = None
    ITEM_CODE: str | None = None
    ITEM_NAME: str | None = None
    GRP_NAME: str | None = None
    START_TIME: str | None = None
    END_TIME: str | None = None
    UNIT_NAME: str | None = None


class _EcosList(_Model):
    row: tuple[_EcosListRow, ...] = ()


class _EcosListPayload(_Model):
    StatisticTableList: _EcosList | None = None
    StatisticItemList: _EcosList | None = None
    RESULT: _EcosResult | None = None


async def ecos_tables(ctx: FetchContext) -> list[SeriesInfo]:
    """Every searchable ECOS statistics table (code, name, cycle)."""
    url = "/".join(
        [ECOS_TABLE_LIST_URL, ctx.key(ECOS_KEY_ENV), "json", "kr", "1", "10000"]
    )
    rows = await _ecos_list(ctx, url, "통계표 목록")
    return [
        SeriesInfo(code=row.STAT_CODE, name=row.STAT_NAME, freq=row.CYCLE or "")
        for row in rows
        if row.SRCH_YN != "N" and row.STAT_CODE
    ]


async def ecos_items(ctx: FetchContext, stat: str) -> list[SeriesInfo]:
    """The items of one ECOS table. ``code`` is ``stat/cycle/item``."""
    url = "/".join(
        [
            ECOS_ITEM_LIST_URL,
            ctx.key(ECOS_KEY_ENV),
            "json",
            "kr",
            "1",
            "10000",
            stat,
        ]
    )
    rows = await _ecos_list(ctx, url, f"{stat} 항목 목록")
    return [
        SeriesInfo(
            code=f"{stat}/{row.CYCLE or ''}/{row.ITEM_CODE or ''}",
            name=" · ".join(part for part in (row.GRP_NAME, row.ITEM_NAME) if part),
            freq=row.CYCLE or "",
            unit=row.UNIT_NAME or "",
            start=row.START_TIME or "",
            end=row.END_TIME or "",
        )
        for row in rows
        if row.ITEM_CODE
    ]


async def _ecos_list(
    ctx: FetchContext, url: str, what: str
) -> tuple[_EcosListRow, ...]:
    raw = await ctx.text("ECOS", url)
    try:
        payload = _EcosListPayload.model_validate_json(raw)
    except ValidationError as error:
        raise SourceError(f"ECOS {what}: 응답 형식 오류") from error
    if payload.RESULT is not None:
        raise SourceError(
            f"ECOS {what}: {payload.RESULT.CODE} {payload.RESULT.MESSAGE}"
        )
    body = payload.StatisticTableList or payload.StatisticItemList
    return body.row if body is not None else ()


async def sdmx_dataflows(ctx: FetchContext, provider: SourceId) -> list[SeriesInfo]:
    """The datasets (dataflows) one SDMX provider publishes."""
    root, agency = SDMX_STRUCTURE_ROOTS[provider]
    raw = await ctx.text(provider, f"{root}/{agency}")
    tree = parse_structure_xml(raw, provider)
    flows: list[SeriesInfo] = []
    for node in tree.iter():
        if _local(node.tag) != "Dataflow" or "id" not in node.attrib:
            continue
        flow_id = node.attrib["id"]
        if provider == "OECD":
            flow_id = f"{node.attrib.get('agencyID', '')},{flow_id}"
        flows.append(SeriesInfo(code=flow_id, name=_english_name(node)))
    return flows


async def sdmx_dimensions(
    ctx: FetchContext, provider: SourceId, flow: str
) -> list[SeriesInfo]:
    """One dataflow's key dimensions in order, each with its allowed codes."""
    root, agency = SDMX_STRUCTURE_ROOTS[provider]
    if provider == "OECD" and "," in flow:
        agency, flow = flow.split(",", 1)
    raw = await ctx.text(
        provider, f"{root}/{agency}/{flow}/latest", {"references": "all"}
    )
    tree = parse_structure_xml(raw, provider)
    codelists: dict[str, list[tuple[str, str]]] = {}
    for node in tree.iter():
        if _local(node.tag) == "Codelist" and "id" in node.attrib:
            codelists[node.attrib["id"]] = [
                (code.attrib.get("id", ""), _english_name(code))
                for code in node
                if _local(code.tag) == "Code"
            ]
    out: list[SeriesInfo] = []
    for node in tree.iter():
        if _local(node.tag) != "Dimension" or "id" not in node.attrib:
            continue
        refs = [
            ref.attrib.get("id", "")
            for ref in node.iter()
            if _local(ref.tag) == "Ref" and ref.attrib.get("class") == "Codelist"
        ]
        codes = codelists.get(refs[0], []) if refs else []
        listed = ", ".join(f"{code}={name}" for code, name in codes[:60])
        more = f" … 외 {len(codes) - 60}개" if len(codes) > 60 else ""  # noqa: PLR2004
        out.append(
            SeriesInfo(
                code=node.attrib["id"],
                name=f"위치 {node.attrib.get('position', '?')}",
                note=listed + more,
            )
        )
    return out


def parse_structure_xml(text: str, provider: SourceId) -> ET.Element:
    """Parse an SDMX structure message. Documents with a DTD are refused."""
    if "<!DOCTYPE" in text[:2000].upper():
        raise SourceError(f"{provider} 구조 응답에 DTD가 있어 읽지 않습니다")
    try:
        return ET.fromstring(text)  # noqa: S314 - no DTD, from the seven public hosts
    except ET.ParseError as error:
        raise SourceError(f"{provider} 구조 응답 형식 오류") from error


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _english_name(node: ET.Element) -> str:
    names = [child for child in node if _local(child.tag) == "Name"]
    for child in names:
        if child.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") == "en":
            return (child.text or "").strip()
    return (names[0].text or "").strip() if names else ""
