# pyright: reportImplicitStringConcatenation=false

from __future__ import annotations

import asyncio
import time
from collections import Counter
from datetime import date, datetime, timedelta
from functools import cache, partial
from itertools import pairwise
from typing import TYPE_CHECKING, Final, Literal, final

from finagent import catalog
from finagent.fetch import FRED_KEY_ENV, MissingKeyError, SourceError
from finagent.models import (
    CalendarEntry,
    Curve,
    CurveLine,
    Dashboard,
    IndicatorRow,
    Panel,
    PolicyChange,
    ProjectionCell,
    ProjectionRow,
    Projections,
    RegimeCell,
    SourceSummary,
    Spread,
)
from finagent.sources import (
    Difference,
    EtfSeries,
    etf_series,
    fred_release_dates,
    fred_series_release_dates,
    fred_vintages,
)
from finagent.transforms import (
    apply,
    history,
    monthly_mean,
    period_label,
    recession_bands,
    value_on_or_before,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    from finagent.fetch import FetchContext
    from finagent.models import Area, SourceId, SourceState, Tone
    from finagent.sources import Fetch, MacroFetch, Series, Vintage

SOURCE_NAMES: Final[tuple[tuple[SourceId, str], ...]] = (
    ("FRED", "FRED"),
    ("ECOS", "ECOS"),
    ("ECB", "ECB"),
    ("BIS", "BIS"),
    ("OECD", "OECD"),
    ("ALPACA", "Alpaca·Yahoo"),
    ("YAHOO", "Yahoo"),
)

type Outcome = Series | SourceError

POLICY_BANKS: Final[tuple[tuple[str, Area, str], ...]] = (
    ("us_policy_rate_bis", "US", "Fed"),
    ("ea_policy_rate_bis", "EA", "ECB"),
    ("jp_policy_rate_bis", "JP", "BOJ"),
    ("cn_policy_rate_bis", "CN", "PBOC"),
    ("kr_policy_rate_bis", "KR", "한국은행"),
)
POLICY_CHANGE_WINDOW_DAYS: Final = 365
RECENT_RELEASE_DAYS: Final = 14
FOMC_RELEASE_SERIES: Final = "FEDTARMD"
FOMC_HORIZON_DAYS: Final = 365
CONTEXT_PAGE: Final = "overview"

type Extra = Literal["calendar", "projections", "fomc"]

EXTRAS: Final[frozenset[Extra]] = frozenset({"calendar", "projections", "fomc"})
PAGE_EXTRAS: Final[Mapping[str, frozenset[Extra]]] = {
    "overview": frozenset({"projections"}),
    "schedule": frozenset({"calendar", "fomc"}),
}


@cache
def _fetch_pages() -> dict[Fetch, tuple[str, ...]]:
    pages: dict[Fetch, dict[str, None]] = {}
    for spec in catalog.PANELS:
        for fetch in catalog.indicator_fetches(spec.indicators):
            pages.setdefault(fetch, {})[spec.page] = None
    for fetch in catalog.CONTEXT_FETCHES:
        pages.setdefault(fetch, {})[CONTEXT_PAGE] = None
    return {fetch: tuple(found) for fetch, found in pages.items()}


def fetch_pages(fetch: Fetch) -> tuple[str, ...]:
    return _fetch_pages().get(fetch, ())


@final
class BuildProgress:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.total: dict[tuple[str, str], int] = {}
        self.done: dict[tuple[str, str], int] = {}
        self.count = 0
        self.series = 0
        self.etfs = 0
        self.extra: Literal["wait", "run", "done"] = "wait"

    def plan(self, fetches: Sequence[Fetch]) -> None:
        total = Counter((f.source, page) for f in fetches for page in fetch_pages(f))
        self.done = dict.fromkeys(total, 0)
        self.total = dict(total)
        self.count = len(fetches)

    def finish(self, fetch: Fetch) -> None:
        for page in fetch_pages(fetch):
            self.done[fetch.source, page] += 1
        self.series += 1

    def finish_etfs(self, fetches: Sequence[EtfSeries]) -> None:
        for (source, page), n in Counter(
            (f.source, page) for f in fetches for page in fetch_pages(f)
        ).items():
            self.done[source, page] = n
        self.etfs = len(fetches)


async def build_dashboard(  # noqa: PLR0913
    ctx: FetchContext,
    now: datetime,
    *,
    cache: dict[Fetch, Series],
    refresh: bool,
    stale: dict[Fetch, SourceError],
    progress: BuildProgress,
    only: Collection[Fetch] | None = None,
    extras: Collection[Extra] = EXTRAS,
    previous: Dashboard | None = None,
) -> Dashboard:
    started = time.perf_counter()
    today = now.date()
    fetches = all_fetches()
    renew = set(fetches if only is None else only)
    todo = [f for f in fetches if (refresh and f in renew) or f not in cache]
    etfs = [fetch for fetch in todo if isinstance(fetch, EtfSeries)]
    rest = [fetch for fetch in todo if not isinstance(fetch, EtfSeries)]
    progress.plan(todo)

    async def run(fetch: MacroFetch) -> Outcome:
        result = await _run(fetch, ctx, today)
        progress.finish(fetch)
        return result

    async def run_etfs() -> dict[EtfSeries, Series | SourceError]:
        result = await etf_series(etfs, ctx, today, force=refresh)
        progress.finish_etfs(etfs)
        return result

    results, held = await asyncio.gather(
        asyncio.gather(*(run(fetch) for fetch in rest)), run_etfs()
    )
    fetched: dict[Fetch, Outcome] = {**dict(zip(rest, results, strict=True)), **held}
    for fetch, result in fetched.items():
        if isinstance(result, SourceError) and fetch in cache:
            stale[fetch] = result
        else:
            _ = stale.pop(fetch, None)
    got = {f: r for f, r in fetched.items() if not isinstance(r, SourceError)}
    cache.update(got)
    outcomes: dict[Fetch, Outcome] = {
        fetch: fetched[fetch] if fetch in fetched else cache[fetch] for fetch in fetches
    }
    kept = {fetch: stale[fetch] for fetch in fetches if fetch in stale}
    shown: dict[Fetch, Outcome] = {**outcomes, **{f: cache[f] for f in kept}}

    panels = tuple(
        Panel(
            id=spec.id,
            page=spec.page,
            title=spec.title,
            note=spec.note,
            rows=tuple(_row(indicator, shown, kept) for indicator in spec.indicators),
        )
        for spec in catalog.PANELS
    )
    rows = {row.id: row for panel in panels for row in panel.rows}
    progress.extra = "run"
    (
        (calendar, calendar_message),
        recent_releases,
        projections,
        (fomc_dates, fomc_message),
    ) = await asyncio.gather(
        _calendar(ctx, today)
        if previous is None or "calendar" in extras
        else _kept((previous.calendar, previous.calendar_message)),
        _recent_releases(ctx, today)
        if previous is None or "calendar" in extras
        else _kept(previous.recent_releases),
        _projections(ctx, today)
        if previous is None or "projections" in extras
        else _kept(previous.projections),
        _fomc_dates(ctx, today)
        if previous is None or "fomc" in extras
        else _kept((previous.fomc_dates, previous.fomc_message)),
    )
    progress.extra = "done"
    return Dashboard(
        generated_at=now.isoformat(timespec="seconds"),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        demo=False,
        sources=_summaries({**outcomes, **kept}),
        regime=regime(rows),
        curves=_curves(shown),
        spread=_spread(shown),
        projections=_with_current(projections, rows),
        panels=panels,
        calendar=calendar,
        policy_changes=_policy_changes(shown, today),
        fomc_dates=fomc_dates,
        recent_releases=recent_releases,
        calendar_message=calendar_message,
        fomc_message=fomc_message,
    )


async def _kept[T](value: T) -> T:
    return value


def page_fetches(page: str) -> tuple[Fetch, ...]:
    return tuple(fetch for fetch, pages in _fetch_pages().items() if page in pages)


def all_fetches() -> tuple[Fetch, ...]:
    return tuple(_fetch_pages())


async def _run(fetch: MacroFetch, ctx: FetchContext, today: date) -> Outcome:
    try:
        return await ctx.shared(repr(fetch), partial(fetch.fetch, ctx, today))
    except SourceError as error:
        return error


def _resolve(fetch: Fetch | Difference, outcomes: Mapping[Fetch, Outcome]) -> Outcome:
    if not isinstance(fetch, Difference):
        return outcomes[fetch]
    left = outcomes[fetch.left]
    right = outcomes[fetch.right]
    if isinstance(left, SourceError):
        return left
    if isinstance(right, SourceError):
        return right
    return fetch.combine(left, right)


def _row(
    indicator: catalog.Indicator,
    outcomes: Mapping[Fetch, Outcome],
    stale: Mapping[Fetch, SourceError],
) -> IndicatorRow:
    base = IndicatorRow(
        id=indicator.id,
        label=indicator.label,
        code=indicator.code,
        source=indicator.fetch.source,
        area=indicator.area,
        copyright=indicator.copyright,
        freq=indicator.freq,
        fmt=indicator.fmt,
        status="error",
    )
    outcome = _resolve(indicator.fetch, outcomes)
    if isinstance(outcome, MissingKeyError):
        return base.model_copy(
            update={"status": "missing_key", "message": str(outcome)}
        )
    if isinstance(outcome, SourceError):
        return base.model_copy(update={"message": str(outcome)})
    series = apply(outcome, indicator.transform, indicator.freq, indicator.scale)
    if not series:
        return base.model_copy(update={"message": "변환 대상 관측값 부족"})
    last_day, last_value = series[-1]
    parts = catalog.indicator_fetches([indicator])
    failed = next((stale[part] for part in parts if part in stale), None)
    return base.model_copy(
        update={
            "status": "ok",
            "message": None if failed is None else f"갱신 실패, 이전 값: {failed}",
            "value": last_value,
            "prev": series[-2][1] if len(series) > 1 else None,
            "date": period_label(last_day, indicator.freq),
            "history": history(series, indicator.freq),
        }
    )


def _summaries(outcomes: Mapping[Fetch, Outcome]) -> tuple[SourceSummary, ...]:
    summaries: list[SourceSummary] = []
    for source, name in SOURCE_NAMES:
        mine = [
            outcome for fetch, outcome in outcomes.items() if fetch.source == source
        ]
        if not mine:
            continue
        errors = [outcome for outcome in mine if isinstance(outcome, SourceError)]
        ok = len(mine) - len(errors)
        state: SourceState
        if not errors:
            state = "ok"
        elif ok > 0:
            state = "partial"
        elif all(isinstance(error, MissingKeyError) for error in errors):
            state = "missing_key"
        else:
            state = "error"
        summaries.append(
            SourceSummary(
                id=source,
                name=name,
                state=state,
                total=len(mine),
                ok=ok,
                message=str(errors[0]) if errors else None,
            )
        )
    return tuple(summaries)


CORE_PCE_HIGH: Final = 2.5
CORE_PCE_LOW: Final = 1.5
SAHM_TRIGGER: Final = 0.5
SAHM_WATCH: Final = 0.3
VIX_CALM: Final = 20.0
VIX_FEAR: Final = 30.0


def regime(rows: Mapping[str, IndicatorRow]) -> tuple[RegimeCell, ...]:
    return (
        _cell(
            rows,
            "us_core_pce",
            "물가",
            "근원 PCE",
            "%",
            f"PCEPILFE YoY: {CORE_PCE_LOW}~{CORE_PCE_HIGH}% 목표 부근 (연준 목표 2%)",
            lambda v: (
                ("warn", "목표 상회")
                if v > CORE_PCE_HIGH
                else ("warn", "목표 하회")
                if v < CORE_PCE_LOW
                else ("good", "목표 부근")
            ),
            dec=1,
        ),
        _cell(
            rows,
            "us_sahm",
            "고용",
            "삼의 법칙",
            "%p",
            f"SAHMREALTIME ≥ {SAHM_TRIGGER:.2f}: 경기 침체 신호",
            lambda v: (
                ("crit", "침체 신호")
                if v >= SAHM_TRIGGER
                else ("warn", "주의 구간")
                if v >= SAHM_WATCH
                else ("good", "침체 신호 없음")
            ),
        ),
        _cell(
            rows,
            "us_10y2y",
            "금리곡선",
            "10Y−2Y",
            "%p",
            "T10Y2Y < 0: 장단기 금리 역전",
            lambda v: ("warn", "역전") if v < 0 else ("neutral", "정상 기울기"),
            signed=True,
        ),
        _cell(
            rows,
            "us_vix",
            "변동성",
            "VIX",
            "",
            f"VIXCLS < {VIX_CALM:.0f} 안정 · {VIX_CALM:.0f}~{VIX_FEAR:.0f} 주의"
            f" · ≥ {VIX_FEAR:.0f} 공포",
            lambda v: (
                ("good", "안정")
                if v < VIX_CALM
                else ("warn", "주의")
                if v < VIX_FEAR
                else ("crit", "공포")
            ),
            dec=1,
        ),
    )


def _cell(  # noqa: PLR0913
    rows: Mapping[str, IndicatorRow],
    source: str,
    label: str,
    measure: str,
    unit: str,
    rule: str,
    judge: Callable[[float], tuple[Tone, str]],
    *,
    dec: int = 2,
    signed: bool = False,
) -> RegimeCell:
    row = rows.get(source)
    if row is None or row.value is None:
        return RegimeCell(
            source=source,
            label=label,
            measure=measure,
            value="—",
            tone="neutral",
            tag="데이터 없음",
            rule=rule,
        )
    tone, tag = judge(row.value)
    sign = "+" if signed and row.value > 0 else ""
    text = f"{sign}{row.value:.{dec}f}{unit}"
    return RegimeCell(
        source=source,
        label=label,
        measure=measure,
        value=text,
        tone=tone,
        tag=tag,
        rule=rule,
    )


def _curves(outcomes: Mapping[Fetch, Outcome]) -> tuple[Curve, ...]:
    return tuple(
        curve
        for spec in catalog.CURVES
        if (curve := _curve(spec, outcomes)) is not None
    )


def _curve(spec: catalog.CurveSpec, outcomes: Mapping[Fetch, Outcome]) -> Curve | None:
    tenor_series: list[Series] = []
    for _, fetch in spec.tenors:
        outcome = outcomes[fetch]
        if isinstance(outcome, SourceError):
            return None
        tenor_series.append(outcome)
    latest = min(series[-1][0] for series in tenor_series)
    targets = (
        ("최신", latest),
        ("1개월 전", latest - timedelta(days=30)),
        ("1년 전", latest - timedelta(days=365)),
    )
    lines: list[CurveLine] = []
    for label, target in targets:
        values = tuple(_value_at(series, target) for series in tenor_series)
        lines.append(CurveLine(label=label, date=target.isoformat(), values=values))
    return Curve(
        area=spec.area,
        label=spec.label,
        tenors=tuple(name for name, _ in spec.tenors),
        lines=tuple(lines),
    )


def _value_at(series: Series, target: date) -> float | None:
    point = value_on_or_before(series, target)
    return None if point is None else point[1]


def _spread(outcomes: Mapping[Fetch, Outcome]) -> Spread | None:
    spread = outcomes[catalog.SPREAD_SERIES]
    if isinstance(spread, SourceError):
        return None
    recession = outcomes[catalog.RECESSION_SERIES]
    bands = () if isinstance(recession, SourceError) else recession_bands(recession)
    return Spread(points=monthly_mean(spread), recessions=bands)


def _policy_changes(
    outcomes: Mapping[Fetch, Outcome], today: date
) -> tuple[PolicyChange, ...]:
    changes: list[PolicyChange] = []
    for indicator_id, area, bank in POLICY_BANKS:
        outcome = _resolve(catalog.BY_ID[indicator_id].fetch, outcomes)
        if isinstance(outcome, SourceError):
            continue
        change = _policy_change(area, bank, outcome, today)
        if change is not None:
            changes.append(change)
    return tuple(changes)


def _policy_change(
    area: Area, bank: str, series: Series, today: date
) -> PolicyChange | None:
    moves = [
        (day, before, after)
        for (_, before), (day, after) in pairwise(series)
        if after != before
    ]
    if not moves:
        return None
    day, before, after = moves[-1]
    since = today - timedelta(days=POLICY_CHANGE_WINDOW_DAYS)
    return PolicyChange(
        area=area,
        bank=bank,
        date=day.isoformat(),
        rate=after,
        prev=before,
        count_12m=sum(moved > since for moved, _, _ in moves),
    )


type VintagePair = tuple[Vintage, Vintage | None]


async def _vintages(
    ctx: FetchContext, series_id: str, since: date
) -> VintagePair | SourceError:
    try:
        return await fred_vintages(ctx, series_id, since)
    except SourceError as error:
        return error


async def _projections(ctx: FetchContext, today: date) -> Projections | None:
    if not ctx.has_key(FRED_KEY_ENV):
        return None
    since = today - timedelta(days=catalog.SEP_LOOKBACK_DAYS)
    ids = [row.series_id for row in catalog.SEP_ROWS] + [
        row.longer_run_id for row in catalog.SEP_ROWS if row.longer_run_id
    ]
    results = await asyncio.gather(*(_vintages(ctx, sid, since) for sid in ids))
    got: dict[str, VintagePair | SourceError] = dict(zip(ids, results, strict=True))
    errors = [result for result in results if isinstance(result, SourceError)]
    main = [
        pair
        for row in catalog.SEP_ROWS
        if not isinstance(pair := got[row.series_id], SourceError)
    ]
    if not main:
        return Projections(
            published="",
            prior_published=None,
            years=(),
            rows=(),
            message=str(errors[0]),
        )
    published = max(latest.published for latest, _ in main)
    priors = [prior.published for _, prior in main if prior is not None]
    years = sorted(
        {
            day.year
            for latest, _ in main
            for day, _ in latest.series
            if day.year >= published.year
        }
    )[: catalog.SEP_MAX_YEARS]
    return Projections(
        published=published.isoformat(),
        prior_published=max(priors).isoformat() if priors else None,
        years=tuple(str(year) for year in years),
        rows=tuple(_projection_row(spec, got, years) for spec in catalog.SEP_ROWS),
        message=str(errors[0]) if errors else None,
    )


def _with_current(
    projections: Projections | None, rows: Mapping[str, IndicatorRow]
) -> Projections | None:
    if projections is None:
        return None
    current = {
        f"FRED:{spec.series_id}": rows[spec.current].value for spec in catalog.SEP_ROWS
    }
    return projections.model_copy(
        update={
            "rows": tuple(
                row.model_copy(update={"current": current[row.code]})
                for row in projections.rows
            )
        }
    )


def _projection_row(
    spec: catalog.SepRow,
    got: Mapping[str, VintagePair | SourceError],
    years: list[int],
) -> ProjectionRow:
    pair = got[spec.series_id]
    if isinstance(pair, SourceError):
        cells = tuple(ProjectionCell() for _ in years)
    else:
        latest = _by_year(pair[0])
        prior = _by_year(pair[1]) if pair[1] is not None else {}
        cells = tuple(
            ProjectionCell(value=latest.get(year), prev=prior.get(year))
            for year in years
        )
    longer_run: ProjectionCell | None = None
    if spec.longer_run_id is not None:
        lr = got[spec.longer_run_id]
        longer_run = (
            ProjectionCell()
            if isinstance(lr, SourceError)
            else ProjectionCell(
                value=lr[0].series[-1][1],
                prev=lr[1].series[-1][1] if lr[1] is not None else None,
            )
        )
    return ProjectionRow(
        label=spec.label,
        code=f"FRED:{spec.series_id}",
        current=None,
        current_note=spec.current_note,
        current_dec=spec.current_dec,
        years=cells,
        longer_run=longer_run,
    )


def _by_year(vintage: Vintage) -> dict[int, float]:
    return {day.year: value for day, value in vintage.series}


async def _release_window(
    ctx: FetchContext, start: date, end: date
) -> tuple[tuple[CalendarEntry, ...], str | None]:
    if not ctx.has_key(FRED_KEY_ENV):
        return (), None
    try:
        releases = await fred_release_dates(ctx, start, end)
    except SourceError as error:
        return (), str(error)
    return tuple(_calendar_entries(releases)), None


async def _calendar(
    ctx: FetchContext, today: date
) -> tuple[tuple[CalendarEntry, ...], str | None]:
    return await _release_window(
        ctx, today, today + timedelta(days=catalog.CALENDAR_DAYS)
    )


async def _recent_releases(ctx: FetchContext, today: date) -> tuple[CalendarEntry, ...]:
    entries, _ = await _release_window(
        ctx, today - timedelta(days=RECENT_RELEASE_DAYS), today
    )
    return entries


def _calendar_entries(
    releases: Iterable[tuple[date, str]],
) -> Iterable[CalendarEntry]:
    seen: set[tuple[date, str]] = set()
    for day, name in releases:
        for fragment, label, agency in catalog.CALENDAR_RELEASES:
            if fragment.lower() in name.lower() and (day, label) not in seen:
                seen.add((day, label))
                yield CalendarEntry(date=day.isoformat(), name=label, source=agency)


async def _fomc_dates(
    ctx: FetchContext, today: date
) -> tuple[tuple[str, ...], str | None]:
    if not ctx.has_key(FRED_KEY_ENV):
        return (), None
    try:
        days = await fred_series_release_dates(
            ctx,
            FOMC_RELEASE_SERIES,
            today,
            today + timedelta(days=FOMC_HORIZON_DAYS),
        )
    except SourceError as error:
        return (), str(error)
    return tuple(day.isoformat() for day in days), None
