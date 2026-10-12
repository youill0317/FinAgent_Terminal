"""Fetch specs for each public source. Each spec returns a dated series.

A spec is a frozen, hashable value so a refresh pass can fetch each distinct
series once even when several indicators or charts use it.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from finagent.fetch import (
    ECOS_KEY_ENV,
    FRED_KEY_ENV,
    SourceError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finagent.fetch import FetchContext
    from finagent.models import SourceId

type Point = tuple[date, float]
type Series = tuple[Point, ...]

FRED_OBSERVATIONS_URL: Final = "https://api.stlouisfed.org/fred/series/observations"
FRED_RELEASE_DATES_URL: Final = "https://api.stlouisfed.org/fred/releases/dates"
FISCAL_BASE_URL: Final = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
)
NYFED_BASE_URL: Final = "https://markets.newyorkfed.org/api/rates"
ECOS_BASE_URL: Final = "https://ecos.bok.or.kr/api/StatisticSearch"
ECB_BASE_URL: Final = "https://data-api.ecb.europa.eu/service/data"
BIS_BASE_URL: Final = "https://stats.bis.org/api/v1/data"
OECD_BASE_URL: Final = "https://sdmx.oecd.org/public/rest/data"

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


# ---------------------------------------------------------------- derived


type Fetch = FredSeries | FiscalSeries | NyFedRate | EcosSeries | SdmxSeries


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
