"""오프라인·데모 모드용 거시경제 대시보드 및 시계열 합성 데이터 생성기."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Final

from finagent import catalog
from finagent.dashboard import POLICY_BANKS, SOURCE_NAMES, all_fetches, regime
from finagent.demo import VALUE
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
from finagent.transforms import HISTORY_MONTHS, HISTORY_QUARTERS, period_label

if TYPE_CHECKING:
    from finagent.models import Freq
    from finagent.sources import Series

SPREAD_MONTHS: Final = 120
DEMO_FOMC_DAYS: Final = (45, 135, 225, 315)
DEMO_POLICY_CHANGE_DAYS: Final = 60
DEMO_POLICY_STEP: Final = 0.25


def dashboard(now: datetime) -> Dashboard:
    today = now.date()
    panels = tuple(
        Panel(
            id=spec.id,
            page=spec.page,
            title=spec.title,
            note=spec.note,
            rows=tuple(_row(indicator, today) for indicator in spec.indicators),
        )
        for spec in catalog.PANELS
    )
    return Dashboard(
        generated_at=now.isoformat(timespec="seconds"),
        elapsed_seconds=0.0,
        demo=True,
        sources=_sources(),
        regime=tuple(
            RegimeCell(
                source=cell.source,
                label=cell.label,
                measure=cell.measure,
                value=f"{VALUE:.0f}",
                tone="neutral",
                tag="데모",
                rule="데모 값",
            )
            for cell in regime({})
        ),
        curves=_curves(today),
        spread=Spread(points=_months(today, SPREAD_MONTHS), recessions=()),
        projections=_projections(today),
        panels=panels,
        calendar=(
            CalendarEntry(
                date=today.isoformat(), name="데모컴퍼니 발표", source="데모"
            ),
        ),
        policy_changes=tuple(
            PolicyChange(
                area=area,
                bank=bank,
                date=(today - timedelta(days=DEMO_POLICY_CHANGE_DAYS)).isoformat(),
                rate=VALUE,
                prev=VALUE + DEMO_POLICY_STEP,
                count_12m=1,
            )
            for _, area, bank in POLICY_BANKS
        ),
        fomc_dates=tuple(
            (today + timedelta(days=days)).isoformat() for days in DEMO_FOMC_DAYS
        ),
        recent_releases=(
            CalendarEntry(
                date=(today - timedelta(days=2)).isoformat(),
                name="데모컴퍼니 발표",
                source="데모",
            ),
        ),
    )


def _row(indicator: catalog.Indicator, today: date) -> IndicatorRow:
    history = _history(indicator.freq, today)
    return IndicatorRow(
        id=indicator.id,
        label=indicator.label,
        code=indicator.code,
        source=indicator.fetch.source,
        area=indicator.area,
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
    stamp = today.year * 12 + today.month - 1
    return tuple(
        (f"{month // 12}-{month % 12 + 1:02d}", VALUE)
        for month in range(stamp - count + 1, stamp + 1)
    )


def series(freq: Freq, today: date, years: int) -> Series:
    step = 3 if freq == "Q" else 1
    stamp = today.year * 12 + (today.month - 1) // step * step
    return tuple(
        (date(month // 12, month % 12 + 1, 1), VALUE)
        for month in range(stamp - years * 12, stamp + 1, step)
    )


def _sources() -> tuple[SourceSummary, ...]:
    fetches = all_fetches()
    totals = {
        source: sum(fetch.source == source for fetch in fetches)
        for source, _ in SOURCE_NAMES
    }
    return tuple(
        SourceSummary(id=source, name=name, state="ok", total=total, ok=total)
        for source, name in SOURCE_NAMES
        if (total := totals[source])
    )


def _curves(today: date) -> tuple[Curve, ...]:
    ago = (("최신", 0), ("1개월 전", 30), ("1년 전", 365))
    return tuple(
        Curve(
            area=spec.area,
            label=spec.label,
            tenors=tuple(name for name, _ in spec.tenors),
            lines=tuple(
                CurveLine(
                    label=label,
                    date=(today - timedelta(days=days)).isoformat(),
                    values=(VALUE,) * len(spec.tenors),
                )
                for label, days in ago
            ),
        )
        for spec in catalog.CURVES
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
