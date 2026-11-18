"""AI 에이전트용 거시경제 시계열·일봉 시세·데이터셋 가이드 조회 도구함."""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from itertools import pairwise
from statistics import median
from typing import TYPE_CHECKING, Final, Literal, cast

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from finagent import catalog, prices
from finagent.analysis import documents, prompt
from finagent.analysis.company_tools import (
    DEFAULT_FILINGS,
    DEFAULT_HOLDINGS,
    DEFAULT_PERIODS,
    DEFAULT_ROWS,
    MAX_HOLDINGS,
    METRIC_TAGS,
    text_arg,
)
from finagent.demo import dashboard as demo_dashboard
from finagent.fetch import SourceError
from finagent.limits import LimitReached
from finagent.prices import NO_PRICE, PriceSeries
from finagent.sources import (
    Difference,
    EcosSeries,
    EtfSeries,
    FredSeries,
    SdmxSeries,
    ecos_items,
    ecos_tables,
    fred_search,
    sdmx_dataflows,
    sdmx_dimensions,
    years_before,
)
from finagent.transforms import apply

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from finagent.analysis.company_tools import Company
    from finagent.fetch import FetchContext
    from finagent.models import Dashboard, SourceId
    from finagent.prices import Basis
    from finagent.service import SeriesStore
    from finagent.sources import Fetch, Series, SeriesInfo
    from finagent.transforms import Transform

type Period = Literal["D", "W", "M", "Q", "A"]

MAX_TOOL_CALLS: Final = 12
MAX_POINTS: Final = 400
DEFAULT_YEARS: Final = 10
MAX_YEARS: Final = 80
MAX_HITS: Final = 25
RAW_DECIMALS: Final = 4
PRICE_DEFAULT_DAYS: Final = 90
PRICE_TICKER_LIMIT: Final = 20
BASES: Final[tuple[str, ...]] = ("split", "total")
STALE_FRED_DAYS: Final = 730
FRED_HITS: Final = 15
ECOS_PLAIN_PARTS: Final = 3

type Json = dict[str, object]
type PortfolioReader = Callable[[str], list[Json]]
SOURCES: Final = ("FRED", "ECOS", "ECB", "BIS", "OECD")
ETF_CODE: Final = re.compile(r"[A-Za-z0-9.^=-]+")
SEARCHABLE: Final = ("catalog", "FRED", "ECOS", "ECB", "BIS", "OECD")
TRANSFORMS: Final = ("level", "yoy", "pct", "diff")
SDMX: Final = ("ECB", "BIS", "OECD")
DASHBOARD_PAGES: Final[Mapping[str, str]] = {
    "overview": "개요",
    "economy": "경제 지표",
    "monetary": "통화정책·금리",
    "finance": "금융시장·심리",
    "schedule": "일정·출처",
}

CODE_FORMATS: Final[Mapping[str, str]] = dict(
    cast("tuple[str, str]", tuple(line.split("\t", 1)))
    for line in prompt.text("code_formats.txt").splitlines()
)

TOOL_SPECS: Final[tuple[tuple[str, str, Json], ...]] = (
    (
        "search_series",
        prompt.text("tool_search_series.txt"),
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
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    (
        "get_series",
        prompt.text("tool_get_series.txt"),
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
        "get_prices",
        prompt.text("tool_get_prices.txt"),
        {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": PRICE_TICKER_LIMIT,
                },
                "start": {
                    "type": "string",
                    "description": f"YYYY-MM-DD (기본 {PRICE_DEFAULT_DAYS}일 전)",
                },
                "end": {"type": "string", "description": "YYYY-MM-DD (기본 오늘)"},
                "basis": {"type": "string", "enum": list(BASES)},
            },
            "required": ["tickers"],
            "additionalProperties": False,
        },
    ),
    (
        "company_facts",
        prompt.text("tool_company_facts.txt"),
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"{', '.join(METRIC_TAGS)} 또는 us-gaap 태그 이름",
                },
                "periods": {
                    "type": "integer",
                    "description": f"항목별 기간 수 (기본 {DEFAULT_PERIODS})",
                },
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    ),
    (
        "company_filings",
        prompt.text("tool_company_filings.txt"),
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "forms": {"type": "array", "items": {"type": "string"}},
                "since": {"type": "string", "description": "YYYY-MM-DD"},
                "limit": {"type": "integer", "description": f"기본 {DEFAULT_FILINGS}"},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    ),
    (
        "read_filing",
        prompt.text("tool_read_filing.txt"),
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "accession": {"type": "string"},
                "document": {
                    "type": "string",
                    "description": "main, earnings 또는 문서 이름",
                },
                "offset": {"type": "integer"},
            },
            "required": ["ticker", "accession"],
            "additionalProperties": False,
        },
    ),
    (
        "etf_lookup",
        prompt.text("tool_etf_lookup.txt"),
        {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "group": {"type": "string"},
                "category": {"type": "string"},
                "limit": {"type": "integer", "description": f"기본 {DEFAULT_ROWS}"},
            },
            "additionalProperties": False,
        },
    ),
    (
        "etf_holdings",
        prompt.text("tool_etf_holdings.txt"),
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "description": f"기본 {DEFAULT_HOLDINGS}, 최대 {MAX_HOLDINGS}",
                },
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    ),
    (
        "company_profile",
        prompt.text("tool_company_profile.txt"),
        {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    ),
    (
        "read_dashboard",
        prompt.text("tool_read_dashboard.txt"),
        {
            "type": "object",
            "properties": {
                "page": {"type": "string", "enum": list(DASHBOARD_PAGES)},
            },
            "required": ["page"],
            "additionalProperties": False,
        },
    ),
    (
        "read_portfolio",
        prompt.text("tool_read_portfolio.txt"),
        {
            "type": "object",
            "properties": {
                "portfolio": {"type": "string", "description": "포트폴리오 이름"},
            },
            "additionalProperties": False,
        },
    ),
    (
        "source_guide",
        prompt.text("tool_source_guide.txt"),
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
COMPANY_TOOLS: Final = frozenset(
    {
        "company_facts",
        "company_filings",
        "read_filing",
        "etf_lookup",
        "etf_holdings",
        "company_profile",
    }
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    text: str
    ok: bool
    summary: str


@dataclass(slots=True)
class Toolbox:
    ctx: FetchContext
    series: SeriesStore
    today: date
    company: Company
    in_document: frozenset[str] = frozenset()
    demo: bool = False
    dashboard: Dashboard | None = None
    portfolios: PortfolioReader | None = None
    limit: int = MAX_TOOL_CALLS
    calls: int = 0
    received: dict[str, str] = field(default_factory=dict)
    _etf_names: dict[str, str] = field(default_factory=dict)
    _ecos_tables: list[SeriesInfo] | None = None

    async def run(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        self.calls += 1
        if self.calls > self.limit:
            message = f"단일 실행 도구 호출 한도({self.limit}회) 초과"
            return _error(name, message)
        handlers = {
            "search_series": self._search,
            "get_series": self._get,
            "get_prices": self._prices,
            "source_guide": self._guide,
            "read_dashboard": self._dashboard,
            "read_portfolio": self._portfolio,
        }
        handler = handlers.get(name)
        if name in COMPANY_TOOLS:
            call = self._company(name, arguments)
        elif handler is not None:
            call = handler(arguments)
        else:
            return _error(name, f"알 수 없는 도구: {name}")
        try:
            return await call
        except LimitReached as error:
            return _reached(name, error)
        except Exception as error:  # noqa: BLE001  # 도구 실행 최상위 예외를 JSON 오류로 변환
            return _error(name, _failure(error))

    async def _company(self, name: str, args: Mapping[str, object]) -> ToolResult:
        answer = await asyncio.to_thread(self.company.run, name, args)
        self.received.update(answer.received)
        return ToolResult(_dump(answer.body), ok=True, summary=answer.summary)

    async def _dashboard(self, args: Mapping[str, object]) -> ToolResult:
        page = text_arg(args, "page")
        if page not in DASHBOARD_PAGES:
            raise ValueError(f"page는 {', '.join(DASHBOARD_PAGES)} 중 하나")
        if self.dashboard is None:
            raise ValueError("현재 실행 환경 내 대시보드 조회 미지원")
        found = documents.page_document(self.dashboard, page)
        title = DASHBOARD_PAGES[page]
        self.received.update(dict.fromkeys(found.ids, f"대시보드 {title}"))
        head = f'<dashboard page="{page}" snapshot="{found.snapshot}">'
        text = f"{head}\n{found.text}\n</dashboard>"
        summary = f"read_dashboard({title}) · {len(found.ids)}항목"
        return ToolResult(text, ok=True, summary=summary)

    async def _portfolio(self, args: Mapping[str, object]) -> ToolResult:
        if self.portfolios is None:
            raise ValueError("현재 실행 환경 내 포트폴리오 조회 미지원")
        found = await asyncio.to_thread(self.portfolios, text_arg(args, "portfolio"))
        for item in found:
            self.received[str(item["cite_as"])] = f"{item['name']} 보유 내역"
        summary = f"read_portfolio · {len(found)}개"
        return ToolResult(_dump({"portfolios": found}), ok=True, summary=summary)

    async def _search(self, args: Mapping[str, object]) -> ToolResult:
        source = text_arg(args, "source")
        query = text_arg(args, "query")
        dataset = text_arg(args, "dataset")
        if source and source not in SEARCHABLE:
            raise ValueError(f"source는 {', '.join(SEARCHABLE)} 중 하나")
        if not source:
            return await self._search_all(query)
        hits = await self._hits(source, query, dataset)
        label = f"{source}{f' {dataset}' if dataset else ''}"
        summary = f'search_series({label}, "{query}") · {len(hits)}건'
        body: Json = {"source": source, "query": query, "results": hits}
        if dataset:
            body["dataset"] = dataset
        return ToolResult(_dump(body), ok=True, summary=summary)

    async def _search_all(self, query: str) -> ToolResult:
        sources = ("catalog", "FRED", "ECOS")
        results = await asyncio.gather(
            *(self._hits(source, query, "") for source in sources),
            return_exceptions=True,
        )
        hits: list[Json] = []
        failed: list[str] = []
        for source, found in zip(sources, results, strict=True):
            if isinstance(found, SourceError):
                failed.append(str(found))
            elif isinstance(found, BaseException):
                raise found
            else:
                hits.extend({"source": source, **hit} for hit in found)
        body: Json = {"query": query, "results": hits}
        if failed:
            body["failed"] = failed
        summary = f'search_series("{query}") · {len(hits)}건'
        return ToolResult(_dump(body), ok=True, summary=summary)

    async def _hits(self, source: str, query: str, dataset: str) -> list[Json]:
        if source == "catalog":
            return self._catalog_hits(query)
        if self.demo:
            return []
        if source == "FRED":
            found = await fred_search(self.ctx, query, FRED_HITS)
            return [_info_json(hit) for hit in _ended_last(found, self.today)]
        if source == "ECOS":
            return await self._ecos_hits(query, dataset)
        provider = cast("SourceId", source)
        listed = (
            await sdmx_dimensions(self.ctx, provider, dataset)
            if dataset
            else _filter(await sdmx_dataflows(self.ctx, provider), query)
        )
        return [_info_json(hit) for hit in listed[:MAX_HITS]]

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

    async def _get(self, args: Mapping[str, object]) -> ToolResult:
        given = text_arg(args, "start")
        start = _period(given, self.today, default_years=DEFAULT_YEARS)
        end = _period(text_arg(args, "end"), self.today, default_years=0, end=True)
        if start > end:
            raise ValueError("start가 end보다 이후 시점")
        frequency = text_arg(args, "frequency") or "native"
        if frequency not in {"native", "M", "Q", "A"}:
            raise ValueError("frequency는 native, M, Q, A 중 하나")
        requested = text_arg(args, "transform")
        if requested and requested not in TRANSFORMS:
            raise ValueError(f"transform은 {', '.join(TRANSFORMS)} 중 하나")
        span = (
            min(MAX_YEARS, math.ceil((self.today - start).days / 365.25))
            if given
            else DEFAULT_YEARS
        )
        got = await self._resolve(args, span, requested)
        series = tuple(point for point in got.series if start <= point[0] <= end)
        series, shown_freq, note = shape(series, got.freq, frequency)
        if not series:
            raise SourceError(f"{got.cite}: 요청 기간 내 관측값 없음")
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
        self.received[got.cite] = (
            f"{got.name} · {got.source} · {shown_freq} {len(points)}점"
        )
        span = f"{points[0][0]}~{points[-1][0]}"
        summary = f"get_series({got.cite}, {span}) · {shown_freq} {len(points)}점"
        return ToolResult(_dump(body), ok=True, summary=summary)

    async def _resolve(
        self, args: Mapping[str, object], span: int, requested: str
    ) -> _Resolved:
        indicator_id = text_arg(args, "id")
        if indicator_id:
            indicator = catalog_indicator(indicator_id)
            transform = cast("Transform", requested or indicator.transform)
            years = span + (transform != "level")
            shown = indicator.code
            if self.demo:
                series = demo_dashboard.series(indicator.freq, self.today, years)
            else:
                raw = await self._fetch(indicator.fetch, years)
                series = apply(raw, transform, indicator.freq, indicator.scale)
                if isinstance(indicator.fetch, EtfSeries):
                    shown = self._etf_names.get(indicator.fetch.ticker, indicator.code)
            return _Resolved(
                series=series,
                freq=indicator.freq,
                transform=transform,
                cite=indicator.id,
                name=indicator.label,
                source=shown,
                unit=indicator.fmt.suf,
                decimals=indicator.fmt.dec,
                copyright=indicator.copyright or "",
            )
        source_id = text_arg(args, "source")
        code = text_arg(args, "code")
        if not source_id or not code:
            raise ValueError("id 또는 source·code 필수 입력")
        transform = cast("Transform", requested or "level")
        years = span + (transform != "level")
        origin = f"{source_id}:{code}"
        if self.demo:
            series, freq = demo_dashboard.series("M", self.today, years), "M"
        else:
            raw = await self._fetch(build_fetch(source_id, code, years), years)
            freq = infer_freq(raw)
            series = apply(raw, transform, "Q" if freq == "A" else freq)
        return _Resolved(
            series=series,
            freq=freq,
            transform=transform,
            cite=f"{source_id}:{code}",
            name=code,
            source=origin,
            unit="%" if transform in {"yoy", "pct"} else "",
            decimals=RAW_DECIMALS,
        )

    async def _fetch(self, fetch: Fetch | Difference, years: int) -> Series:
        if isinstance(fetch, Difference):
            left, right = await asyncio.gather(
                self._fetch(fetch.left, years),
                self._fetch(fetch.right, years),
            )
            return fetch.combine(left, right)
        spec = with_years(fetch, years)
        if isinstance(spec, EtfSeries):
            return await self._etf(spec)
        return await self.series.through(spec, self.ctx, self.today)

    async def _etf(self, spec: EtfSeries) -> Series:
        start = years_before(self.today, spec.years)
        found = await self._stored_prices([spec.ticker], start, self.today, "total")
        got = found[1].get(spec.ticker)
        if got is None:
            raise SourceError(f"{spec.ticker}: {found[0].get(spec.ticker, NO_PRICE)}")
        self._etf_names[spec.ticker] = got.source
        return prices.points(got.series)

    async def _stored_prices(
        self, tickers: Sequence[str], start: date, end: date, basis: Basis
    ) -> tuple[Mapping[str, str], Mapping[str, PriceSeries]]:
        store = prices.current()
        result = await asyncio.to_thread(
            store.ensure, tickers, start, end, max_wait=self.ctx.max_wait
        )
        found = store.latest(tickers, start, end, basis)
        if result.reached is not None and not found:
            raise result.reached
        return result.lost, found

    async def _prices(self, args: Mapping[str, object]) -> ToolResult:
        tickers = _tickers(args)
        basis = text_arg(args, "basis") or "split"
        if basis not in BASES:
            raise ValueError(f"basis는 {', '.join(BASES)} 중 하나")
        fallback = self.today - timedelta(days=PRICE_DEFAULT_DAYS)
        start = _day(text_arg(args, "start"), fallback)
        end = _day(text_arg(args, "end"), self.today)
        if start > end:
            raise ValueError("start가 end보다 이후 시점")
        lost: Mapping[str, str] = {}
        if self.demo:
            found = self._demo_prices(tickers, start, end)
        else:
            lost, found = await self._stored_prices(tickers, start, end, basis)
        rows: list[Json] = []
        failed: list[Json] = []
        share = MAX_POINTS // len(tickers)
        for ticker in tickers:
            got = found.get(ticker)
            if got is not None:
                rows.append(self._price_row(ticker, got, basis, share))
            if got is None or ticker in lost:
                reason = lost.get(ticker, "요청 기간 내 시세 없음")
                failed.append({"ticker": ticker, "error": f"{ticker}: {reason}"})
        if not rows:
            reasons = "; ".join(cast("str", row["error"]) for row in failed)
            raise SourceError(reasons)
        body: Json = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "basis": basis,
            "prices": rows,
        }
        if failed:
            body["failed"] = failed
        names = f"{tickers[0]} 외 {len(tickers) - 1}종목" if tickers[1:] else tickers[0]
        tail = f" · 실패 {len(failed)}" if failed else ""
        summary = f"get_prices({names}, {start}~{end}) · {len(rows)}종목{tail}"
        return ToolResult(_dump(body), ok=True, summary=summary)

    def _demo_prices(
        self, tickers: Sequence[str], start: date, end: date
    ) -> dict[str, PriceSeries]:
        years = math.ceil((self.today - start).days / 365.25) + 1
        kept = [
            point
            for point in demo_dashboard.series("M", self.today, years)
            if start <= point[0] <= end
        ]
        values = pd.Series(
            [value for _, value in kept],
            index=pd.DatetimeIndex([pd.Timestamp(day) for day, _ in kept]),
            dtype="float64",
        )
        return dict.fromkeys(tickers, PriceSeries(values, "Demo"))

    def _price_row(self, ticker: str, got: PriceSeries, basis: str, share: int) -> Json:
        series, name = prices.points(got.series), got.source
        cite = f"{name}:{ticker}"
        first, last = series[0][1], series[-1][1]
        shown, freq, note = shape(series, infer_freq(series), "native", share)
        points = [
            [period_text(day, freq), round(value, RAW_DECIMALS)] for day, value in shown
        ]
        self.received[cite] = f"{ticker} · {name} · {basis} {len(points)}점"
        row: Json = {
            "ticker": ticker,
            "cite_as": cite,
            "source": name,
            "first": round(first, RAW_DECIMALS),
            "last": round(last, RAW_DECIMALS),
            "change_pct": round((last / first - 1) * 100, 2) if first else None,
            "points": points,
        }
        if note:
            row["note"] = note
        return row

    async def _guide(self, args: Mapping[str, object]) -> ToolResult:
        source = text_arg(args, "source")
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
        return ToolResult(
            _dump(body), ok=True, summary=f"source_guide({source}) · 코드 {len(used)}개"
        )


@dataclass(frozen=True, slots=True)
class _Resolved:
    series: Series
    freq: Period
    transform: Transform
    cite: str
    name: str
    source: str
    unit: str
    decimals: int
    copyright: str = ""


def shape(
    series: Series, freq: Period, frequency: str, limit: int = MAX_POINTS
) -> tuple[Series, str, str]:
    shown: str = freq
    note = ""
    if frequency != "native":
        series, shown = aggregate(series, frequency), frequency
    while len(series) > limit and shown != "A":
        shown = {"D": "W", "W": "M", "M": "Q", "Q": "A"}[shown]
        series = aggregate(series, shown)
        note = f"{limit}점 초과에 따른 {shown} 평균 축약"
    if len(series) > limit:
        series = series[-limit:]
        note = f"{limit}점 초과에 따른 {shown} 평균 최근 {limit}점 유지"
    return series, shown, note


def catalog_indicator(indicator_id: str) -> catalog.Indicator:
    indicator = catalog.BY_ID.get(indicator_id)
    if indicator is None:
        msg = f"대시보드 미등록 id: {indicator_id} (search_series로 찾을 것)"
        raise ValueError(msg)
    return indicator


def build_fetch(source: str, code: str, years: int) -> Fetch:
    if source in SDMX:
        flow_key = code if source != "BIS" or code.endswith("/all") else f"{code}/all"
        if "/" not in flow_key:
            raise ValueError(f"{source} 코드는 데이터셋/키")
        return SdmxSeries(cast("SourceId", source), flow_key, years=years)
    builders = {"FRED": _fred_fetch, "ECOS": _ecos_fetch}
    builder = builders.get(source)
    if builder is None:
        raise ValueError(f"source는 {', '.join(SOURCES)} 중 하나")
    return builder(code, years)


def _fred_fetch(code: str, years: int) -> Fetch:
    if not re.fullmatch(r"[A-Za-z0-9_.]+", code):
        raise ValueError("FRED 시리즈 ID 형식 오류")
    return FredSeries(code.upper(), years)


def _ecos_fetch(code: str, years: int) -> Fetch:
    parts = code.split("/")
    if len(parts) not in {3, 4} or parts[1] not in {"D", "M", "Q", "A"}:
        raise ValueError("ECOS 코드는 통계표/주기/항목1[/항목2]")
    item2 = parts[3] if len(parts) > ECOS_PLAIN_PARTS else ""
    return EcosSeries(parts[0], parts[1], parts[2], item2, years)


def with_years(fetch: Fetch, years: int) -> Fetch:
    return replace(fetch, years=max(years, fetch.years))


def infer_freq(series: Series) -> Period:
    if len(series) < 3:  # noqa: PLR2004  # 주기 추정을 위한 최소 관측점 수(3개)
        return "M"
    gaps = [(b[0] - a[0]).days for a, b in pairwise(series)]
    gap = median(gaps)
    if gap <= 4:  # noqa: PLR2004  # 일간(D) 판정 최대 중앙 간격(4일)
        return "D"
    if gap <= 8:  # noqa: PLR2004  # 주간(W) 판정 최대 중앙 간격(8일)
        return "W"
    if gap <= 40:  # noqa: PLR2004  # 월간(M) 판정 최대 중앙 간격(40일)
        return "M"
    if gap <= 100:  # noqa: PLR2004  # 분기(Q) 판정 최대 중앙 간격(100일)
        return "Q"
    return "A"


def aggregate(series: Series, frequency: str) -> Series:
    buckets: dict[date, list[float]] = {}
    for day, value in series:
        if frequency == "W":
            key = day - timedelta(days=day.weekday())
        elif frequency == "M":
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
    if not 1 <= month <= 12:  # noqa: PLR2004  # 달력 월 유효 범위(1~12월)
        raise ValueError("월은 1~12")
    if end:
        next_month = date(year + month // 12, month % 12 + 1, 1)
        return min(today, date.fromordinal(next_month.toordinal() - 1))
    return date(year, month, 1)


def _tickers(args: Mapping[str, object]) -> list[str]:
    raw = args.get("tickers")
    items = cast("list[object]", raw) if isinstance(raw, list) else []
    tickers = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    if not 1 <= len(tickers) <= PRICE_TICKER_LIMIT:
        raise ValueError(f"tickers는 1~{PRICE_TICKER_LIMIT}개")
    for ticker in tickers:
        if not ETF_CODE.fullmatch(ticker):
            raise ValueError(f"티커 형식 오류: {ticker}")
    return list(dict.fromkeys(tickers))


def _day(text: str, fallback: date) -> date:
    if not text:
        return fallback
    return date.fromisoformat(text)


def _ended_last(items: list[SeriesInfo], today: date) -> list[SeriesInfo]:
    cutoff = (today - timedelta(days=STALE_FRED_DAYS)).isoformat()
    return sorted(items, key=lambda item: bool(item.end) and item.end < cutoff)


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


def _dump(body: Json) -> str:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def _reached(name: str, error: LimitReached) -> ToolResult:
    at = datetime.fromtimestamp(error.next_at).astimezone()
    retry_at = at.isoformat(timespec="seconds")
    message = f"{error.source} 호출 제한"
    return ToolResult(
        _dump({"error": message, "retry_at": retry_at}),
        ok=False,
        summary=f"{name} · {error.source} 다음 가능 {at:%H:%M}",
    )


def _error(name: str, message: str) -> ToolResult:
    return ToolResult(
        _dump({"error": message}), ok=False, summary=f"{name} · 실패: {message}"
    )


def _failure(error: Exception) -> str:
    if isinstance(error, SourceError):
        return str(error)
    if isinstance(error, ValueError):
        return f"입력 오류: {error}"
    return f"{type(error).__name__}: {error}"
