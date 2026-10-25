# Tooltip text is split across lines.
# pyright: reportImplicitStringConcatenation=false

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Final

from finagent import __version__, catalog
from finagent.fetch import FRED_KEY_ENV, MissingKeyError, SourceError
from finagent.models import (
    CalendarEntry,
    Curve,
    CurveLine,
    Dashboard,
    IndicatorRow,
    Panel,
    ProjectionCell,
    ProjectionRow,
    Projections,
    RegimeCell,
    SourceSummary,
    Spread,
    ViewOption,
)
from finagent.sources import (
    FredSeries,
    Ratio,
    combine_ratio,
    fred_release_dates,
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
    from collections.abc import Callable, Iterable, Mapping

    from finagent.fetch import FetchContext
    from finagent.models import SourceId, SourceState, Tone
    from finagent.sources import Fetch, Series, Vintage

SOURCE_NAMES: Final[tuple[tuple[SourceId, str], ...]] = (
    ("FRED", "FRED"),
    ("TREASURY", "Treasury"),
    ("NYFED", "NY Fed"),
    ("ECOS", "ECOS"),
    ("ECB", "ECB"),
    ("BIS", "BIS"),
    ("OECD", "OECD"),
    ("YAHOO", "Yahoo"),
)

type Outcome = Series | SourceError


async def build_dashboard(  # noqa: PLR0913
    ctx: FetchContext,
    now: datetime,
    *,
    demo: bool = False,
    deep: frozenset[str] = catalog.ALL_VIEWS,
    cache: dict[Fetch, Series] | None = None,
    previous: Dashboard | None = None,
) -> Dashboard:
    """Fetch what the view needs and assemble the dashboard.

    ``deep`` names the panels shown in full (plus ``curve`` for the yield
    curve chart); other panels show only their core rows. Series already in
    ``cache`` are not fetched again, and new ones are added to it. With
    ``previous`` its calendar and FOMC projections are reused instead of
    fetched, so changing the view only fetches the newly shown series.
    """
    started = time.perf_counter()
    today = now.date()
    specs = [(spec, catalog.shown(spec, deep)) for spec in catalog.PANELS]
    fetches = _all_fetches(
        [indicator for _, indicators in specs for indicator in indicators],
        curve=catalog.CURVE_VIEW in deep,
    )
    known = {} if cache is None else cache
    todo = [fetch for fetch in fetches if fetch not in known]
    results = await asyncio.gather(*(_run(fetch, ctx, today) for fetch in todo))
    fetched: dict[Fetch, Outcome] = dict(zip(todo, results, strict=True))
    if cache is not None:
        cache.update(
            {f: r for f, r in fetched.items() if not isinstance(r, SourceError)}
        )
    outcomes: dict[Fetch, Outcome] = {
        fetch: fetched[fetch] if fetch in fetched else known[fetch] for fetch in fetches
    }

    panels = tuple(
        Panel(
            id=spec.id,
            key=spec.key,
            title=spec.title,
            note=spec.note,
            rows=tuple(_row(indicator, outcomes) for indicator in indicators),
        )
        for spec, indicators in specs
        if indicators
    )
    rows = {row.id: row for panel in panels for row in panel.rows}
    if previous is None:
        calendar, projections = await asyncio.gather(
            _calendar(ctx, today), _projections(ctx, today, rows)
        )
    else:
        calendar, projections = previous.calendar, previous.projections
    return Dashboard(
        version=__version__,
        generated_at=now.isoformat(timespec="seconds"),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        demo=demo,
        sources=_summaries(outcomes),
        tape=catalog.TAPE,
        regime=_regime(rows),
        curve=_curve(outcomes) if catalog.CURVE_VIEW in deep else None,
        spread=_spread(outcomes),
        projections=projections,
        panels=panels,
        calendar=calendar,
        deep=tuple(view.id for view in catalog.VIEWS if view.id in deep),
        views=tuple(
            ViewOption(
                id=view.id,
                key=view.key,
                title=view.title,
                core=view.core,
                extra=view.extra,
            )
            for view in catalog.VIEWS
        ),
    )


def _all_fetches(
    indicators: Iterable[catalog.Indicator], *, curve: bool
) -> tuple[Fetch, ...]:
    seen: dict[Fetch, None] = {}
    for indicator in indicators:
        parts = (
            indicator.fetch.parts()
            if isinstance(indicator.fetch, Ratio)
            else (indicator.fetch,)
        )
        for part in parts:
            seen[part] = None
    if curve:
        for _, series_id in catalog.CURVE_TENORS:
            seen[FredSeries(series_id, catalog.CURVE_YEARS)] = None
    seen[catalog.SPREAD_SERIES] = None
    seen[catalog.RECESSION_SERIES] = None
    return tuple(seen)


async def _run(fetch: Fetch, ctx: FetchContext, today: date) -> Outcome:
    try:
        return await fetch.fetch(ctx, today)
    except SourceError as error:
        return error


def _resolve(fetch: Fetch | Ratio, outcomes: Mapping[Fetch, Outcome]) -> Outcome:
    if not isinstance(fetch, Ratio):
        return outcomes[fetch]
    numerator = outcomes[fetch.numerator]
    denominator = outcomes[fetch.denominator]
    if isinstance(numerator, SourceError):
        return numerator
    if isinstance(denominator, SourceError):
        return denominator
    return combine_ratio(numerator, denominator)


def _row(
    indicator: catalog.Indicator, outcomes: Mapping[Fetch, Outcome]
) -> IndicatorRow:
    base = IndicatorRow(
        id=indicator.id,
        label=indicator.label,
        code=indicator.code,
        source=indicator.fetch.source,
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
        return base.model_copy(update={"message": "변환할 관측값이 부족합니다"})
    last_day, last_value = series[-1]
    return base.model_copy(
        update={
            "status": "ok",
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


# ---------------------------------------------------------------- regime rules

CORE_PCE_HIGH: Final = 2.5
CORE_PCE_LOW: Final = 1.5
SAHM_TRIGGER: Final = 0.5
SAHM_WATCH: Final = 0.3


def _regime(rows: Mapping[str, IndicatorRow]) -> tuple[RegimeCell, ...]:
    return (
        _cell(
            rows.get("us_core_pce"),
            "물가",
            "% 근원 PCE",
            "PCEPILFE YoY: 1.5~2.5% 목표 부근 (연준 목표 2%)",
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
            rows.get("us_sahm"),
            "고용",
            "%p 삼 룰",
            "SAHMREALTIME ≥ 0.50 이면 침체 신호",
            lambda v: (
                ("crit", "침체 신호")
                if v >= SAHM_TRIGGER
                else ("warn", "주의 구간")
                if v >= SAHM_WATCH
                else ("good", "침체 신호 없음")
            ),
        ),
        _cell(
            rows.get("us_10y2y"),
            "금리곡선",
            "%p 10Y−2Y",
            "T10Y2Y < 0 이면 장단기 역전",
            lambda v: ("warn", "역전") if v < 0 else ("neutral", "정상 기울기"),
            signed=True,
        ),
        _cell(
            rows.get("us_nfci"),
            "금융여건",
            "NFCI",
            "NFCI > 0 이면 평균보다 긴축",
            lambda v: ("warn", "평균보다 긴축") if v > 0 else ("good", "평균보다 완화"),
            signed=True,
        ),
    )


# The 국면 요약 cards' tooltips, by label; ``rule`` stays the short form the
# analysis document quotes.
REGIME_HELP: Final = {
    "물가": "**정의** 근원 PCE 물가 전년 대비 상승률(PCEPILFE, %)\n\n**해석**"
    f" {CORE_PCE_LOW}~{CORE_PCE_HIGH}% 목표 부근 · {CORE_PCE_HIGH}% 초과 목표 상회"
    f" · {CORE_PCE_LOW}% 미만 목표 하회\n\n**기준** 연준 물가 목표 2%",
    "고용": "**정의** 실시간 삼 룰(SAHMREALTIME) · 실업률 3개월 평균 − 직전 12개월 중"
    f" 3개월 평균 최저치(%p)\n\n**해석** {SAHM_TRIGGER:.2f}%p 이상 침체 신호 ·"
    f" {SAHM_WATCH:.2f}%p 이상 주의 구간 · 그 아래 침체 신호 없음",
    "금리곡선": "**정의** 미 국채 10년물 − 2년물 수익률 차(T10Y2Y, %p)\n\n**해석**"
    " 0 미만 장단기 역전 · 0 이상 정상 기울기",
    "금융여건": "**정의** 시카고 연은 전국 금융여건지수(NFCI) · 평균 0, 표준편차 1로"
    " 표준화한 값\n\n**해석** 0 초과 평균보다 긴축 · 0 이하 평균보다 완화",
}


def _cell(  # noqa: PLR0913
    row: IndicatorRow | None,
    label: str,
    unit: str,
    rule: str,
    judge: Callable[[float], tuple[Tone, str]],
    *,
    dec: int = 2,
    signed: bool = False,
) -> RegimeCell:
    if row is None or row.value is None:
        return RegimeCell(
            label=label,
            value="—",
            unit=unit,
            tone="neutral",
            tag="데이터 없음",
            rule=rule,
        )
    tone, tag = judge(row.value)
    sign = "+" if signed and row.value > 0 else ""
    text = f"{sign}{row.value:.{dec}f}".replace("-", "−")
    return RegimeCell(label=label, value=text, unit=unit, tone=tone, tag=tag, rule=rule)


# ---------------------------------------------------------------- charts


def _curve(outcomes: Mapping[Fetch, Outcome]) -> Curve | None:
    tenor_series: list[Series] = []
    for _, series_id in catalog.CURVE_TENORS:
        outcome = outcomes[FredSeries(series_id, catalog.CURVE_YEARS)]
        if isinstance(outcome, SourceError):
            return None
        tenor_series.append(outcome)
    latest = max(series[-1][0] for series in tenor_series)
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
        tenors=tuple(name for name, _ in catalog.CURVE_TENORS), lines=tuple(lines)
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


# ---------------------------------------------------------------- FOMC projections

type VintagePair = tuple[Vintage, Vintage | None]


async def _vintages(
    ctx: FetchContext, series_id: str, since: date
) -> VintagePair | SourceError:
    try:
        return await fred_vintages(ctx, series_id, since)
    except SourceError as error:
        return error


async def _projections(
    ctx: FetchContext, today: date, rows: Mapping[str, IndicatorRow]
) -> Projections | None:
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
        rows=tuple(
            _projection_row(spec, got, years, rows) for spec in catalog.SEP_ROWS
        ),
        message=str(errors[0]) if errors else None,
    )


def _projection_row(
    spec: catalog.SepRow,
    got: Mapping[str, VintagePair | SourceError],
    years: list[int],
    rows: Mapping[str, IndicatorRow],
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
    values = [rows[id_].value for id_ in spec.current if id_ in rows]
    current = (
        sum(v for v in values if v is not None) / len(values)
        if values and None not in values
        else None
    )
    return ProjectionRow(
        label=spec.label,
        code=f"FRED:{spec.series_id}",
        current=current,
        current_note=spec.current_note,
        current_dec=spec.current_dec,
        years=cells,
        longer_run=longer_run,
    )


def _by_year(vintage: Vintage) -> dict[int, float]:
    return {day.year: value for day, value in vintage.series}


async def _calendar(ctx: FetchContext, today: date) -> tuple[CalendarEntry, ...]:
    if not ctx.has_key(FRED_KEY_ENV):
        return ()
    try:
        releases = await fred_release_dates(
            ctx, today, today + timedelta(days=catalog.CALENDAR_DAYS)
        )
    except SourceError:
        return ()
    return tuple(_calendar_entries(releases))


def _calendar_entries(
    releases: Iterable[tuple[date, str]],
) -> Iterable[CalendarEntry]:
    seen: set[tuple[date, str]] = set()
    for day, name in releases:
        for fragment, label, agency in catalog.CALENDAR_RELEASES:
            if fragment.lower() in name.lower() and (day, label) not in seen:
                seen.add((day, label))
                yield CalendarEntry(date=day.isoformat(), name=label, source=agency)
