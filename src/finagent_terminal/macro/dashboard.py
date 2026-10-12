from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Final

from finagent_terminal.macro import catalog
from finagent_terminal.macro.http import FRED_KEY_ENV, MissingKeyError, SourceError
from finagent_terminal.macro.models import (
    CalendarEntry,
    Curve,
    CurveLine,
    Dashboard,
    IndicatorRow,
    Panel,
    RegimeCell,
    SourceSummary,
    Spread,
)
from finagent_terminal.macro.sources import (
    FredSeries,
    Ratio,
    combine_ratio,
    fred_release_dates,
)
from finagent_terminal.macro.transforms import (
    apply,
    history,
    monthly_mean,
    period_label,
    recession_bands,
    value_on_or_before,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from finagent_terminal.macro.http import FetchContext
    from finagent_terminal.macro.models import SourceId, SourceState, Tone
    from finagent_terminal.macro.sources import Fetch, Series

SOURCE_NAMES: Final[tuple[tuple[SourceId, str], ...]] = (
    ("FRED", "FRED"),
    ("TREASURY", "Treasury"),
    ("NYFED", "NY Fed"),
    ("ECOS", "ECOS"),
    ("ECB", "ECB"),
    ("BIS", "BIS"),
    ("OECD", "OECD"),
)

type Outcome = Series | SourceError


async def build_dashboard(
    ctx: FetchContext,
    now: datetime,
    *,
    demo: bool = False,
) -> Dashboard:
    started = time.perf_counter()
    today = now.date()
    fetches = _all_fetches()
    results = await asyncio.gather(*(_run(fetch, ctx, today) for fetch in fetches))
    outcomes: dict[Fetch, Outcome] = dict(zip(fetches, results, strict=True))

    panels = tuple(
        Panel(
            id=spec.id,
            key=spec.key,
            title=spec.title,
            note=spec.note,
            rows=tuple(_row(indicator, outcomes) for indicator in spec.indicators),
        )
        for spec in catalog.PANELS
    )
    rows = {row.id: row for panel in panels for row in panel.rows}
    calendar = await _calendar(ctx, today)
    return Dashboard(
        generated_at=now.isoformat(timespec="seconds"),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        demo=demo,
        sources=_summaries(outcomes),
        tape=catalog.TAPE,
        regime=_regime(rows),
        curve=_curve(outcomes),
        spread=_spread(outcomes),
        panels=panels,
        calendar=calendar,
    )


def _all_fetches() -> tuple[Fetch, ...]:
    seen: dict[Fetch, None] = {}
    for spec in catalog.PANELS:
        for indicator in spec.indicators:
            parts = (
                indicator.fetch.parts()
                if isinstance(indicator.fetch, Ratio)
                else (indicator.fetch,)
            )
            for part in parts:
                seen[part] = None
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
