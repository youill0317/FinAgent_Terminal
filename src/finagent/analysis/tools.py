"""Data tools the model can call in any request: find, fetch and read guides.

The tools wrap the fetch specs in ``sources.py``, so they reach exactly the
seven public hosts the dashboard uses. Fetched series go into the dashboard's
in-memory cache and nowhere else.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import date
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Final, Literal, cast

from finagent import catalog
from finagent.analysis.evidence import Evidence
from finagent.fetch import SourceError
from finagent.sources import (
    EcosSeries,
    FiscalSeries,
    FredSeries,
    NyFedRate,
    Ratio,
    SdmxSeries,
    combine_ratio,
    ecos_items,
    ecos_tables,
    fred_search,
    sdmx_dataflows,
    sdmx_dimensions,
)
from finagent.transforms import apply

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    from finagent.fetch import FetchContext
    from finagent.models import Freq, SourceId
    from finagent.sources import Fetch, Series, SeriesInfo
    from finagent.transforms import Transform

MAX_TOOL_CALLS: Final = 12
MAX_POINTS: Final = 400
DEFAULT_YEARS: Final = 10
MAX_YEARS: Final = 80
MAX_HITS: Final = 25
GUIDE_LIMIT: Final = 7000
RAW_DECIMALS: Final = 4
ECOS_PLAIN_PARTS: Final = 3
FISCAL_PLAIN_PARTS: Final = 2

type Json = dict[str, object]
Frequency = Literal["native", "M", "Q", "A"]
SOURCES: Final = ("FRED", "ECOS", "ECB", "BIS", "OECD", "TREASURY", "NYFED")
SEARCHABLE: Final = ("catalog", "FRED", "ECOS", "ECB", "BIS", "OECD")
TRANSFORMS: Final = ("level", "yoy", "pct", "diff")
SDMX: Final = ("ECB", "BIS", "OECD")
# The data-sources guide sits next to the package in a repository checkout.
GUIDE_FILE: Final = Path(__file__).resolve().parents[3] / "docs" / "data-sources.md"
GUIDE_SECTIONS: Final[Mapping[str, str]] = {
    "FRED": "## 1.",
    "TREASURY": "## 2.",
    "NYFED": "## 3.",
    "ECOS": "## 4.",
    "ECB": "## 5.",
    "BIS": "## 6.",
    "OECD": "## 7.",
}
TEXT: Final = files("finagent.analysis") / "text"


def _text_file(name: str) -> str:
    return (TEXT / name).read_text(encoding="utf-8").strip()


CODE_FORMATS: Final[Mapping[str, str]] = dict(
    cast("tuple[str, str]", tuple(line.split("\t", 1)))
    for line in _text_file("code_formats.txt").splitlines()
)

TOOL_SPECS: Final[tuple[tuple[str, str, Json], ...]] = (
    (
        "search_series",
        _text_file("tool_search_series.txt"),
        {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": list(SEARCHABLE)},
                "query": {"type": "string", "description": "이름·코드 검색어"},
                "dataset": {
                    "type": "string",
                    "description": "ECOS 통계표 코드 또는 SDMX 데이터셋(선택)",
                },
            },
            "required": ["source", "query"],
            "additionalProperties": False,
        },
    ),
    (
        "get_series",
        _text_file("tool_get_series.txt"),
        {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "대시보드 지표 id"},
                "source": {"type": "string", "enum": list(SOURCES)},
                "code": {"type": "string"},
                "start": {"type": "string", "description": "YYYY 또는 YYYY-MM"},
                "end": {"type": "string", "description": "YYYY 또는 YYYY-MM"},
                "frequency": {
                    "type": "string",
                    "enum": ["native", "M", "Q", "A"],
                    "description": "native(원래 주기) 또는 월·분기·연 평균",
                },
                "transform": {"type": "string", "enum": list(TRANSFORMS)},
            },
            "additionalProperties": False,
        },
    ),
    (
        "source_guide",
        _text_file("tool_source_guide.txt"),
        {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": list(SOURCES)},
            },
            "required": ["source"],
            "additionalProperties": False,
        },
    ),
)
TOOL_NAMES: Final = tuple(name for name, _, _ in TOOL_SPECS)


@dataclass(frozen=True, slots=True)
class ToolResult:
    text: str
    ok: bool
    summary: str
    evidence: Evidence | None = None


@dataclass(slots=True)
class Toolbox:
    """One run's tools. Counts calls and remembers what was fetched."""

    ctx: FetchContext
    cache: MutableMapping[Fetch, Series]
    today: date
    in_document: frozenset[str] = frozenset()
    calls: int = 0
    evidence: dict[str, Evidence] = field(default_factory=dict)
    received: dict[str, str] = field(default_factory=dict)
    _ecos_tables: list[SeriesInfo] | None = None

    async def run(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        self.calls += 1
        if self.calls > MAX_TOOL_CALLS:
            message = f"한 번 실행의 도구 호출 한도({MAX_TOOL_CALLS}회)를 넘었습니다"
            return _error(name, message)
        handlers = {
            "search_series": self._search,
            "get_series": self._get,
            "source_guide": self._guide,
        }
        handler = handlers.get(name)
        if handler is None:
            return _error(name, f"알 수 없는 도구: {name}")
        try:
            return await handler(arguments)
        except SourceError as error:
            return _error(name, str(error))
        except ValueError as error:
            return _error(name, f"입력 오류: {error}")

    # ------------------------------------------------------------ search_series

    async def _search(self, args: Mapping[str, object]) -> ToolResult:
        source = _text(args, "source")
        query = _text(args, "query")
        dataset = _text(args, "dataset")
        if source not in SEARCHABLE:
            raise ValueError(f"source는 {', '.join(SEARCHABLE)} 중 하나")
        if source == "catalog":
            hits = self._catalog_hits(query)
        elif source == "FRED":
            hits = [_info_json(hit) for hit in await fred_search(self.ctx, query, 15)]
        elif source == "ECOS":
            hits = await self._ecos_hits(query, dataset)
        else:
            provider = cast("SourceId", source)
            found = (
                await sdmx_dimensions(self.ctx, provider, dataset)
                if dataset
                else _filter(await sdmx_dataflows(self.ctx, provider), query)
            )
            hits = [_info_json(hit) for hit in found[:MAX_HITS]]
        label = f"{source}{f' {dataset}' if dataset else ''}"
        summary = f'search_series({label}, "{query}") · {len(hits)}건'
        body: Json = {"source": source, "query": query, "results": hits}
        if dataset:
            body["dataset"] = dataset
        return ToolResult(_dump(body), ok=True, summary=summary)

    def _catalog_hits(self, query: str) -> list[Json]:
        words = query.lower().split()
        hits: list[Json] = []
        for spec in catalog.PANELS:
            for indicator in spec.indicators:
                haystack = (
                    f"{indicator.id} {indicator.label} {indicator.code} {spec.title}"
                )
                if all(word in haystack.lower() for word in words):
                    hits.append(
                        {
                            "id": indicator.id,
                            "name": indicator.label,
                            "panel": spec.title,
                            "code": indicator.code,
                            "freq": indicator.freq,
                            "in_document": indicator.id in self.in_document,
                        }
                    )
        return hits[:MAX_HITS]

    async def _ecos_hits(self, query: str, dataset: str) -> list[Json]:
        if dataset:
            items = await ecos_items(self.ctx, dataset)
            return [_info_json(hit) for hit in _filter(items, query)[:MAX_HITS]]
        if self._ecos_tables is None:
            self._ecos_tables = await ecos_tables(self.ctx)
        return [_info_json(hit) for hit in _filter(self._ecos_tables, query)[:MAX_HITS]]

    # ------------------------------------------------------------ get_series

    async def _get(self, args: Mapping[str, object]) -> ToolResult:
        start = _period(_text(args, "start"), self.today, default_years=DEFAULT_YEARS)
        end = _period(_text(args, "end"), self.today, default_years=0, end=True)
        if start > end:
            raise ValueError("start가 end보다 늦습니다")
        frequency = _text(args, "frequency") or "native"
        if frequency not in {"native", "M", "Q", "A"}:
            raise ValueError("frequency는 native, M, Q, A 중 하나")
        requested = _text(args, "transform")
        if requested and requested not in TRANSFORMS:
            raise ValueError(f"transform은 {', '.join(TRANSFORMS)} 중 하나")
        years = min(MAX_YEARS, math.ceil((self.today - start).days / 365.25) + 1)
        got = await self._resolve(args, years, requested)
        series = tuple(point for point in got.series if start <= point[0] <= end)
        series, shown_freq, note = _shape(series, got.freq, frequency)
        if not series:
            raise SourceError(f"{got.cite}: 요청한 기간에 관측값이 없습니다")
        points = [
            [period_text(day, shown_freq), round(value, got.decimals)]
            for day, value in series
        ]
        body: Json = {
            "cite_as": got.cite,
            "name": got.name,
            "source": got.source,
            "frequency": shown_freq,
            "transform": got.transform,
            "unit": got.unit,
            "count": len(points),
            "points": points,
        }
        if got.copyright:
            body["copyright"] = got.copyright
        if note:
            body["note"] = note
        values = tuple(round(value, got.decimals) for _, value in series)
        evidence = self._remember(got.cite, got.name, values)
        self.received[got.cite] = (
            f"{got.name} · {got.source} · {shown_freq} {len(points)}점"
        )
        span = f"{points[0][0]}~{points[-1][0]}"
        summary = f"get_series({got.cite}, {span}) · {shown_freq} {len(points)}점"
        return ToolResult(_dump(body), ok=True, summary=summary, evidence=evidence)

    async def _resolve(
        self, args: Mapping[str, object], years: int, requested: str
    ) -> _Resolved:
        indicator_id = _text(args, "id")
        if indicator_id:
            indicator = catalog_indicator(indicator_id)
            raw = await self._fetch(indicator.fetch, years)
            transform = cast("Transform", requested or indicator.transform)
            return _Resolved(
                series=apply(raw, transform, indicator.freq, indicator.scale),
                freq=indicator.freq,
                transform=transform,
                cite=indicator.id,
                name=indicator.label,
                source=indicator.code,
                unit=indicator.fmt.suf,
                decimals=indicator.fmt.dec,
                copyright=indicator.copyright or "",
            )
        source_id = _text(args, "source")
        code = _text(args, "code")
        if not source_id or not code:
            raise ValueError("id 또는 source와 code를 주세요")
        raw = await self._fetch(build_fetch(source_id, code, years), years)
        freq = infer_freq(raw)
        transform = cast("Transform", requested or "level")
        return _Resolved(
            series=apply(raw, transform, freq),
            freq=freq,
            transform=transform,
            cite=f"{source_id}:{code}",
            name=code,
            source=f"{source_id}:{code}",
            unit="%" if transform in {"yoy", "pct"} else "",
            decimals=RAW_DECIMALS,
        )

    async def _fetch(self, fetch: Fetch | Ratio, years: int) -> Series:
        if isinstance(fetch, Ratio):
            numerator = await self._fetch(fetch.numerator, years)
            denominator = await self._fetch(fetch.denominator, years)
            return combine_ratio(numerator, denominator)
        spec = with_years(fetch, years)
        cached = self.cache.get(spec)
        if cached is not None:
            return cached
        series = await spec.fetch(self.ctx, self.today)
        self.cache[spec] = series
        return series

    def _remember(self, cite: str, name: str, values: tuple[float, ...]) -> Evidence:
        known = self.evidence.get(cite)
        extra = (*known.points, *known.extra) if known is not None else ()
        item = Evidence(cite, name, values, extra)
        self.evidence[cite] = item
        return item

    # ------------------------------------------------------------ source_guide

    async def _guide(self, args: Mapping[str, object]) -> ToolResult:
        source = _text(args, "source")
        if source not in SOURCES:
            raise ValueError(f"source는 {', '.join(SOURCES)} 중 하나")
        used = [
            f"{indicator.code} · {indicator.label} ({indicator.freq})"
            for spec in catalog.PANELS
            for indicator in spec.indicators
            if indicator.fetch.source == source
        ]
        body: Json = {
            "source": source,
            "code_format": CODE_FORMATS[source],
            "dashboard_codes": used,
        }
        notes = guide_section(source)
        if notes:
            body["data_sources_md"] = notes
        return ToolResult(
            _dump(body), ok=True, summary=f"source_guide({source}) · 코드 {len(used)}개"
        )


@dataclass(frozen=True, slots=True)
class _Resolved:
    series: Series
    freq: Freq
    transform: Transform
    cite: str
    name: str
    source: str
    unit: str
    decimals: int
    copyright: str = ""


def _shape(series: Series, freq: Freq, frequency: str) -> tuple[Series, str, str]:
    """Aggregate to the asked frequency, then coarser until it fits the cap."""
    shown: str = freq
    note = ""
    if frequency != "native":
        series, shown = aggregate(series, frequency), frequency
    while len(series) > MAX_POINTS and shown != "A":
        shown = {"D": "M", "W": "M", "M": "Q", "Q": "A"}[shown]
        series = aggregate(series, shown)
        note = f"{MAX_POINTS}점을 넘어서 {shown} 평균으로 줄였습니다"
    return series, shown, note


# ---------------------------------------------------------------- helpers


def catalog_indicator(indicator_id: str) -> catalog.Indicator:
    for spec in catalog.PANELS:
        for indicator in spec.indicators:
            if indicator.id == indicator_id:
                return indicator
    msg = f"대시보드에 없는 id: {indicator_id} (search_series로 찾으세요)"
    raise ValueError(msg)


def build_fetch(source: str, code: str, years: int) -> Fetch:
    """A fetch spec for a raw source code (formats in ``CODE_FORMATS``)."""
    if source in SDMX:
        flow_key = code if source != "BIS" or code.endswith("/all") else f"{code}/all"
        if "/" not in flow_key:
            raise ValueError(f"{source} 코드는 데이터셋/키")
        return SdmxSeries(cast("SourceId", source), flow_key, years=years)
    builders = {
        "FRED": _fred_fetch,
        "ECOS": _ecos_fetch,
        "TREASURY": _fiscal_fetch,
        "NYFED": _nyfed_fetch,
    }
    builder = builders.get(source)
    if builder is None:
        raise ValueError(f"source는 {', '.join(SOURCES)} 중 하나")
    return builder(code, years)


def _fred_fetch(code: str, years: int) -> Fetch:
    if not re.fullmatch(r"[A-Za-z0-9_.]+", code):
        raise ValueError("FRED 시리즈 ID 형식이 아닙니다")
    return FredSeries(code.upper(), years)


def _ecos_fetch(code: str, years: int) -> Fetch:
    parts = code.split("/")
    if len(parts) not in {3, 4} or parts[1] not in {"D", "M", "Q", "A"}:
        raise ValueError("ECOS 코드는 통계표/주기/항목1[/항목2]")
    item2 = parts[3] if len(parts) > ECOS_PLAIN_PARTS else ""
    return EcosSeries(parts[0], parts[1], parts[2], item2, years)


def _fiscal_fetch(code: str, years: int) -> Fetch:
    parts = code.split("|")
    if len(parts) not in {2, 3}:
        raise ValueError("TREASURY 코드는 엔드포인트|값 필드[|필터]")
    extra = parts[2] if len(parts) > FISCAL_PLAIN_PARTS else ""
    return FiscalSeries(parts[0], parts[1], extra, years)


def _nyfed_fetch(code: str, _years: int) -> Fetch:
    if not re.fullmatch(r"[a-z]+/[a-z0-9]+", code):
        raise ValueError("NYFED 경로 형식이 아닙니다 (예: secured/sofr)")
    return NyFedRate(code)


def with_years(fetch: Fetch, years: int) -> Fetch:
    """The same spec reaching back at least ``years`` years."""
    if isinstance(fetch, NyFedRate):
        return fetch
    return replace(fetch, years=max(years, fetch.years))


def infer_freq(series: Series) -> Freq:
    if len(series) < 3:  # noqa: PLR2004
        return "M"
    gaps = [(b[0] - a[0]).days for a, b in pairwise(series)]
    gap = median(gaps)
    if gap <= 4:  # noqa: PLR2004
        return "D"
    if gap <= 8:  # noqa: PLR2004
        return "W"
    if gap <= 40:  # noqa: PLR2004
        return "M"
    return "Q"


def aggregate(series: Series, frequency: str) -> Series:
    """Period means: ``M`` months, ``Q`` quarters, ``A`` years."""
    buckets: dict[date, list[float]] = {}
    for day, value in series:
        if frequency == "M":
            key = date(day.year, day.month, 1)
        elif frequency == "Q":
            key = date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
        else:
            key = date(day.year, 1, 1)
        buckets.setdefault(key, []).append(value)
    return tuple(
        (key, sum(values) / len(values)) for key, values in sorted(buckets.items())
    )


def period_text(day: date, freq: str) -> str:
    if freq == "A":
        return str(day.year)
    if freq == "Q":
        return f"{day.year}Q{(day.month - 1) // 3 + 1}"
    if freq == "M":
        return day.strftime("%Y-%m")
    return day.isoformat()


def guide_section(source: str) -> str:
    """The source's section of ``docs/data-sources.md`` when it is available."""
    if not GUIDE_FILE.is_file():
        return ""
    text = GUIDE_FILE.read_text(encoding="utf-8")
    start = text.find(GUIDE_SECTIONS[source])
    if start < 0:
        return ""
    end = text.find("\n## ", start + 4)
    section = text[start : end if end > 0 else len(text)].strip()
    return section[:GUIDE_LIMIT]


def _period(text: str, today: date, *, default_years: int, end: bool = False) -> date:
    if not text:
        return (
            date(today.year - default_years, today.month, 1) if default_years else today
        )
    match = re.fullmatch(r"(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", text.strip())
    if match is None:
        raise ValueError("기간은 YYYY 또는 YYYY-MM")
    year = int(match.group(1))
    month = int(match.group(2) or (12 if end else 1))
    if not 1 <= month <= 12:  # noqa: PLR2004
        raise ValueError("월은 1~12")
    if end:
        next_month = date(year + month // 12, month % 12 + 1, 1)
        return min(today, date.fromordinal(next_month.toordinal() - 1))
    return date(year, month, 1)


def _filter(items: list[SeriesInfo], query: str) -> list[SeriesInfo]:
    words = query.lower().split()
    return [
        item
        for item in items
        if all(word in f"{item.code} {item.name}".lower() for word in words)
    ]


def _info_json(info: SeriesInfo) -> Json:
    out: Json = {"code": info.code, "name": info.name}
    for key in ("freq", "unit", "start", "end", "note"):
        value = cast("str", getattr(info, key))
        if value:
            out[key] = value
    return out


def _text(args: Mapping[str, object], key: str) -> str:
    value = args.get(key)
    return value.strip() if isinstance(value, str) else ""


def _dump(body: Json) -> str:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def _error(name: str, message: str) -> ToolResult:
    return ToolResult(
        _dump({"error": message}), ok=False, summary=f"{name} · 실패: {message}"
    )
