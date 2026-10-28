"""The demo's dashboard: rows, cards and charts straight from the catalog.

``VALUE`` stands in for each observation, dated back from today. The AI
analysis reads its series here too (``series``).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Final

from finagent import __version__, catalog
from finagent.dashboard import REGIME_HELP, SOURCE_NAMES
from finagent.demo import VALUE
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
from finagent.transforms import HISTORY_MONTHS, HISTORY_QUARTERS, period_label

if TYPE_CHECKING:
    from finagent.models import Freq
    from finagent.sources import Series

SPREAD_MONTHS: Final = 120


def dashboard(now: datetime, deep: frozenset[str]) -> Dashboard:
    today = now.date()
    specs = [(spec, catalog.shown(spec, deep)) for spec in catalog.PANELS]
    panels = tuple(
        Panel(
            id=spec.id,
            key=spec.key,
            title=spec.title,
            note=spec.note,
            rows=tuple(_row(indicator, today) for indicator in indicators),
        )
        for spec, indicators in specs
        if indicators
    )
    return Dashboard(
        version=__version__,
        generated_at=now.isoformat(timespec="seconds"),
        elapsed_seconds=0.0,
        demo=True,
        sources=_sources(panels),
        tape=catalog.TAPE,
        regime=tuple(
            RegimeCell(
                label=label,
                value=f"{VALUE:.0f}",
                unit="",
                tone="neutral",
                tag="데모",
                rule="데모 값",
            )
            for label in REGIME_HELP
        ),
        curve=_curve(today) if catalog.CURVE_VIEW in deep else None,
        spread=Spread(points=_months(today, SPREAD_MONTHS), recessions=()),
        projections=_projections(today),
        panels=panels,
        calendar=(
            CalendarEntry(
                date=today.isoformat(), name="데모컴퍼니 발표", source="데모"
            ),
        ),
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


def _row(indicator: catalog.Indicator, today: date) -> IndicatorRow:
    history = _history(indicator.freq, today)
    return IndicatorRow(
        id=indicator.id,
        label=indicator.label,
        code=indicator.code,
        source=indicator.fetch.source,
        copyright=indicator.copyright,
        freq=indicator.freq,
        fmt=indicator.fmt,
        status="ok",
        value=VALUE,
        prev=VALUE,
        date=period_label(today, indicator.freq),
        history=history,
    )


def _history(freq: Freq, today: date) -> tuple[tuple[str, float], ...]:
    if freq != "Q":
        return _months(today, HISTORY_MONTHS)
    quarters = [
        period_label(date(today.year, today.month, 1) - timedelta(days=91 * n), "Q")
        for n in range(HISTORY_QUARTERS)
    ]
    return tuple((label, VALUE) for label in reversed(quarters))


def _months(today: date, count: int) -> tuple[tuple[str, float], ...]:
    """``count`` months up to this one, oldest first, each at ``VALUE``."""
    stamp = today.year * 12 + today.month - 1
    return tuple(
        (f"{month // 12}-{month % 12 + 1:02d}", VALUE)
        for month in range(stamp - count + 1, stamp + 1)
    )


def series(freq: Freq, today: date, years: int) -> Series:
    """``years`` back to today at ``VALUE``: monthly, or quarterly for ``Q``."""
    step = 3 if freq == "Q" else 1
    stamp = today.year * 12 + (today.month - 1) // step * step
    return tuple(
        (date(month // 12, month % 12 + 1, 1), VALUE)
        for month in range(stamp - years * 12, stamp + 1, step)
    )


def _sources(panels: tuple[Panel, ...]) -> tuple[SourceSummary, ...]:
    totals = {
        source: sum(row.source == source for p in panels for row in p.rows)
        for source, _ in SOURCE_NAMES
    }
    return tuple(
        SourceSummary(id=source, name=name, state="ok", total=total, ok=total)
        for source, name in SOURCE_NAMES
        if (total := totals[source])
    )


def _curve(today: date) -> Curve:
    ago = (("최신", 0), ("1개월 전", 30), ("1년 전", 365))
    return Curve(
        tenors=tuple(name for name, _ in catalog.CURVE_TENORS),
        lines=tuple(
            CurveLine(
                label=label,
                date=(today - timedelta(days=days)).isoformat(),
                values=(VALUE,) * len(catalog.CURVE_TENORS),
            )
            for label, days in ago
        ),
    )


def _projections(today: date) -> Projections:
    years = tuple(str(today.year + n) for n in range(catalog.SEP_MAX_YEARS - 1))
    cell = ProjectionCell(value=VALUE, prev=VALUE)
    return Projections(
        published=today.isoformat(),
        prior_published=today.isoformat(),
        years=years,
        rows=tuple(
            ProjectionRow(
                label=spec.label,
                code=f"FRED:{spec.series_id}",
                current=VALUE,
                current_note=spec.current_note,
                current_dec=spec.current_dec,
                years=(cell,) * len(years),
                longer_run=cell if spec.longer_run_id else None,
            )
            for spec in catalog.SEP_ROWS
        ),
    )
