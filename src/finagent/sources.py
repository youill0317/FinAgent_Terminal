# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
"""FRED·ECOS·SDMX(ECB·BIS·OECD) 거시경제 시계열 및 발표 일정 어댑터."""

from __future__ import annotations

import asyncio
import csv
import io
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import takewhile
from typing import TYPE_CHECKING, ClassVar, Final, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from finagent import prices
from finagent.fetch import (
    ECOS_KEY_ENV,
    FRED_KEY_ENV,
    SourceError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from finagent.fetch import FetchContext
    from finagent.models import SourceId

type Point = tuple[date, float]
type Series = tuple[Point, ...]

FRED_OBSERVATIONS_URL: Final = "https://api.stlouisfed.org/fred/series/observations"
FRED_RELEASE_DATES_URL: Final = "https://api.stlouisfed.org/fred/releases/dates"
FRED_SEARCH_URL: Final = "https://api.stlouisfed.org/fred/series/search"
FRED_SERIES_RELEASE_URL: Final = "https://api.stlouisfed.org/fred/series/release"
FRED_ONE_RELEASE_DATES_URL: Final = "https://api.stlouisfed.org/fred/release/dates"
ECOS_BASE_URL: Final = "https://ecos.bok.or.kr/api/StatisticSearch"
ECOS_TABLE_LIST_URL: Final = "https://ecos.bok.or.kr/api/StatisticTableList"
ECOS_ITEM_LIST_URL: Final = "https://ecos.bok.or.kr/api/StatisticItemList"
ECB_BASE_URL: Final = "https://data-api.ecb.europa.eu/service/data"
BIS_BASE_URL: Final = "https://stats.bis.org/api/v1/data"
OECD_BASE_URL: Final = "https://sdmx.oecd.org/public/rest/data"
ECB_STATS_CALENDAR_URL: Final = (
    "https://www.ecb.europa.eu/press/calendars/statscal/html/index.en.html"
)
ECB_MEETINGS_URL: Final = (
    "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
)
BIS_RELEASE_CALENDAR_URL: Final = "https://data.bis.org/release-calendar"
OECD_EXPLORER_URL: Final = "https://data-explorer.oecd.org/"
SDMX_STRUCTURE_ROOTS: Final[dict[str, tuple[str, str]]] = {
    "ECB": ("https://data-api.ecb.europa.eu/service/dataflow", "ECB"),
    "BIS": ("https://stats.bis.org/api/v1/dataflow", "BIS"),
    "OECD": ("https://sdmx.oecd.org/public/rest/dataflow", "all"),
}


ECOS_MAX_ROWS: Final = 10000
CHART_YEARS: Final = 10
WINDOW_YEARS: Final = CHART_YEARS + 1


def years_before(today: date, years: int) -> date:
    day = today - timedelta(days=round(365.25 * years))
    return date(day.year, day.month - (day.month - 1) % 3, 1)


_PERIOD_PATTERNS: Final = (
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), "ymd"),
    (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), "ymd"),
    (re.compile(r"^(\d{4})-?Q([1-4])$"), "q"),
    (re.compile(r"^(\d{4})-W(\d{2})$"), "w"),
    (re.compile(r"^(\d{4})-(\d{2})$"), "ym"),
    (re.compile(r"^(\d{4})(\d{2})$"), "ym"),
    (re.compile(r"^(\d{4})$"), "y"),
)


def parse_period(text: str) -> date | None:
    value = text.strip()
    for pattern, kind in _PERIOD_PATTERNS:
        match = pattern.match(value)
        if match is None:
            continue
        parts = [int(group) for group in match.groups()]
        try:
            return _PERIOD_DAYS[kind](parts)
        except ValueError:
            return None
    return None


_PERIOD_DAYS: Final[dict[str, Callable[[list[int]], date]]] = {
    "ymd": lambda p: date(p[0], p[1], p[2]),
    "ym": lambda p: date(p[0], p[1], 1),
    "q": lambda p: date(p[0], 3 * p[1] - 2, 1),
    "w": lambda p: date.fromisocalendar(p[0], p[1], 5),
    "y": lambda p: date(p[0], 1, 1),
}


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


class _FredObservation(_Model):
    date: str
    value: str


class _FredObservations(_Model):
    observations: tuple[_FredObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class FredSeries:
    series_id: str
    years: int = WINDOW_YEARS
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


class _FredRelease(_Model):
    id: int


class _FredSeriesReleases(_Model):
    releases: tuple[_FredRelease, ...] = ()


class _FredOneReleaseDate(_Model):
    date: str


class _FredOneReleaseDates(_Model):
    release_dates: tuple[_FredOneReleaseDate, ...] = ()


_release_ids: Final[dict[str, int]] = {}


async def _fred_release_id(ctx: FetchContext, series_id: str) -> int:
    known = _release_ids.get(series_id)
    if known is not None:
        return known
    text = await ctx.text(
        "FRED",
        FRED_SERIES_RELEASE_URL,
        {
            "series_id": series_id,
            "api_key": ctx.key(FRED_KEY_ENV),
            "file_type": "json",
        },
    )
    try:
        releases = _FredSeriesReleases.model_validate_json(text).releases
    except ValidationError as error:
        raise SourceError(f"FRED {series_id} 발표: 응답 형식 오류") from error
    if not releases:
        raise SourceError(f"FRED {series_id} 발표: 발표 정보 없음")
    _release_ids[series_id] = releases[0].id
    return releases[0].id


async def fred_series_release_dates(
    ctx: FetchContext, series_id: str, start: date, end: date
) -> tuple[date, ...]:
    key = ctx.key(FRED_KEY_ENV)
    params = {
        "release_id": str(await _fred_release_id(ctx, series_id)),
        "api_key": key,
        "file_type": "json",
        "realtime_start": start.isoformat(),
        "realtime_end": end.isoformat(),
        "include_release_dates_with_no_data": "true",
        "limit": "1000",
        "sort_order": "asc",
    }
    text = await ctx.text("FRED", FRED_ONE_RELEASE_DATES_URL, params)
    try:
        payload = _FredOneReleaseDates.model_validate_json(text)
    except ValidationError as error:
        raise SourceError(f"FRED {series_id} 발표일: 응답 형식 오류") from error
    days = {
        day
        for row in payload.release_dates
        if (day := parse_period(row.date)) is not None and start <= day <= end
    }
    return tuple(sorted(days))


class _FredVintageObservation(_Model):
    realtime_start: str
    realtime_end: str
    date: str
    value: str


class _FredVintageObservations(_Model):
    observations: tuple[_FredVintageObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class Vintage:
    published: date
    series: Series


async def fred_vintages(
    ctx: FetchContext, series_id: str, since: date
) -> tuple[Vintage, Vintage | None]:
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


class _EcosRow(_Model):
    TIME: str
    DATA_VALUE: str | None = None


class _EcosSearch(_Model):
    list_total_count: int = 0
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
    cycle: str
    item1: str
    item2: str = ""
    years: int = WINDOW_YEARS
    source: ClassVar[SourceId] = "ECOS"

    async def fetch(self, ctx: FetchContext, today: date) -> Series:
        key = ctx.key(ECOS_KEY_ENV)
        start = ecos_period(years_before(today, self.years), self.cycle)
        end = ecos_period(today, self.cycle)
        label = f"{self.stat}/{self.item1}"
        rows: list[_EcosRow] = []
        while True:
            first = len(rows) + 1
            parts = [
                ECOS_BASE_URL,
                key,
                "json",
                "kr",
                str(first),
                str(first + ECOS_MAX_ROWS - 1),
                self.stat,
                self.cycle,
                start,
                end,
                self.item1,
            ]
            if self.item2:
                parts.append(self.item2)
            text = await ctx.text(self.source, "/".join(parts))
            try:
                payload = _EcosPayload.model_validate_json(text)
            except ValidationError as error:
                raise SourceError(f"ECOS {label}: 응답 형식 오류") from error
            if payload.RESULT is not None:
                raise SourceError(
                    f"ECOS {label}: {payload.RESULT.CODE} {payload.RESULT.MESSAGE}"
                )
            page = payload.StatisticSearch
            if page is None or not page.row:
                break
            rows.extend(page.row)
            if len(rows) >= page.list_total_count:
                break
        return _require(
            to_series([(row.TIME, row.DATA_VALUE) for row in rows]), self.source, label
        )


SDMX_FORMATS: Final[dict[str, tuple[str, str]]] = {
    "ECB": (ECB_BASE_URL, "csvdata"),
    "BIS": (BIS_BASE_URL, "csv"),
    "OECD": (OECD_BASE_URL, "csvfile"),
}


@dataclass(frozen=True, slots=True)
class SdmxSeries:
    provider: SourceId
    flow_key: str
    match: tuple[tuple[str, str], ...] = ()
    years: int = WINDOW_YEARS

    @property
    def source(self) -> SourceId:
        return self.provider

    async def fetch(self, ctx: FetchContext, today: date) -> Series:
        base, fmt = SDMX_FORMATS[self.provider]
        url = f"{base}/{self.flow_key}"
        start = years_before(today, self.years).strftime("%Y-%m")
        params = {"startPeriod": start, "format": fmt}

        async def parsed() -> list[dict[str, str]]:
            return parse_sdmx_csv(await ctx.text(self.provider, url, params))

        rows = await ctx.shared(f"{url}?{start}", parsed)
        wanted = dict(self.match)
        picked = [
            row
            for row in rows
            if all(row.get(column) == value for column, value in wanted.items())
        ]
        if (count := len({_sdmx_series(row) for row in picked})) > 1:
            raise SourceError(
                f"{self.provider} {self.flow_key}: 시리즈 {count}개가 섞인 응답"
            )
        return _require(
            to_series(
                [(row.get("TIME_PERIOD", ""), row.get("OBS_VALUE")) for row in picked]
            ),
            self.provider,
            self.flow_key,
        )


def _sdmx_series(row: dict[str, str]) -> tuple[str, ...]:
    dimensions = takewhile(lambda cell: cell[0] != "TIME_PERIOD", row.items())
    return tuple(value for _, value in dimensions)


def parse_sdmx_csv(text: str) -> list[dict[str, str]]:
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    header = next(reader, None)
    if header is None or "OBS_VALUE" not in header or "TIME_PERIOD" not in header:
        raise SourceError("SDMX 응답에 TIME_PERIOD/OBS_VALUE 열이 없습니다")
    return [dict(zip(header, row, strict=False)) for row in reader]


@dataclass(frozen=True, slots=True)
class EtfSeries:
    ticker: str
    years: int = CHART_YEARS
    source: ClassVar[SourceId] = "ALPACA"


def etf_start(fetches: Sequence[EtfSeries], today: date) -> date:
    return min(years_before(today, fetch.years) for fetch in fetches)


async def etf_series(
    fetches: Sequence[EtfSeries], ctx: FetchContext, today: date, *, force: bool
) -> dict[EtfSeries, Series | SourceError]:
    if not fetches:
        return {}
    result = await asyncio.to_thread(
        prices.current().ensure,
        [fetch.ticker for fetch in fetches],
        etf_start(fetches, today),
        today,
        force=force,
        max_wait=ctx.max_wait,
    )
    found = etf_points(fetches, today)
    return {
        fetch: SourceError(f"{fetch.ticker}: {result.lost[fetch.ticker]}")
        if fetch.ticker in result.lost
        else found.get(fetch, ())
        for fetch in fetches
    }


def etf_points(fetches: Sequence[EtfSeries], today: date) -> dict[EtfSeries, Series]:
    rows = prices.current().rows([f.ticker for f in fetches])
    groups = {
        str(ticker): group.set_index("date")["adj_close"].dropna()
        for ticker, group in rows.groupby("ticker")
    }
    out: dict[EtfSeries, Series] = {}
    for fetch in fetches:
        values = groups.get(fetch.ticker)
        if values is not None:
            start = pd.Timestamp(years_before(today, fetch.years))
            out[fetch] = prices.points(cast("pd.Series", values[values.index >= start]))
    return out


type MacroFetch = FredSeries | EcosSeries | SdmxSeries
type Fetch = MacroFetch | EtfSeries


def series_code(fetch: Fetch) -> str:
    if isinstance(fetch, FredSeries):
        return fetch.series_id
    if isinstance(fetch, EcosSeries):
        parts = (fetch.stat, fetch.cycle, fetch.item1, fetch.item2)
        return "/".join(part for part in parts if part)
    if isinstance(fetch, SdmxSeries):
        match = "&".join(f"{key}={value}" for key, value in fetch.match)
        return f"{fetch.flow_key}#{match}" if match else fetch.flow_key
    return fetch.ticker


@dataclass(frozen=True, slots=True)
class Difference:
    left: Fetch
    right: Fetch

    @property
    def source(self) -> SourceId:
        return self.left.source

    def parts(self) -> tuple[Fetch, Fetch]:
        return (self.left, self.right)

    def combine(self, left: Series, right: Series) -> Series:
        lookup = dict(right)
        return tuple((day, value - lookup[day]) for day, value in left if day in lookup)


@dataclass(frozen=True, slots=True)
class SeriesInfo:
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
    if "<!DOCTYPE" in text[:2000].upper():
        raise SourceError(f"{provider} XML 응답 내 DTD 포함에 따른 처리 제한")
    try:
        return ET.fromstring(text)  # noqa: S314  # DOCTYPE/DTD 사전 차단 검증 완료 XML 파싱
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
