"""시계열 변환(전년동기대비·전기대비·차분·역수)·주기 레이블 및 경기침체 구간 산출."""

from __future__ import annotations

from bisect import bisect_right
from datetime import date, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from finagent.models import Freq

type Series = tuple[tuple[date, float], ...]

Transform = Literal["level", "yoy", "pct", "diff", "inverse"]

HISTORY_MONTHS: Final = 24
HISTORY_QUARTERS: Final = 12
MONTH_DAYS: Final = 31
YEAR_LOOKBACK_TOLERANCE_DAYS: Final = 10


def apply(
    series: Series, transform: Transform, freq: Freq, scale: float = 1.0
) -> Series:
    if transform == "level":
        out = series
    elif transform == "yoy":
        out = year_over_year(series, freq)
    elif transform == "pct":
        out = tuple(
            (day, (value / prior - 1.0) * 100.0)
            for (_, prior), (day, value) in pairwise(series)
            if prior != 0
        )
    elif transform == "diff":
        out = tuple(
            (day, value - prior) for (_, prior), (day, value) in pairwise(series)
        )
    else:
        out = tuple((day, 1.0 / value) for day, value in series if value != 0)
    if scale == 1.0:
        return out
    return tuple((day, value * scale) for day, value in out)


def year_over_year(series: Series, freq: Freq) -> Series:
    if freq in {"M", "Q"}:
        lookup = {(day.year, day.month): value for day, value in series}
        changes: list[tuple[date, float]] = []
        for day, value in series:
            prior = lookup.get((day.year - 1, day.month))
            if prior:
                changes.append((day, (value / prior - 1.0) * 100.0))
        return tuple(changes)
    days = [point[0] for point in series]
    out: list[tuple[date, float]] = []
    for day, value in series:
        target = day - timedelta(days=365)
        index = bisect_right(days, target)
        if index == 0:
            continue
        prior_day, prior = series[index - 1]
        if prior == 0 or (target - prior_day).days > YEAR_LOOKBACK_TOLERANCE_DAYS:
            continue
        out.append((day, (value / prior - 1.0) * 100.0))
    return tuple(out)


def value_on_or_before(series: Series, day: date) -> tuple[date, float] | None:
    index = bisect_right([point[0] for point in series], day)
    return series[index - 1] if index > 0 else None


def period_label(day: date, freq: Freq) -> str:
    if freq == "Q":
        return f"{day.year}Q{(day.month - 1) // 3 + 1}"
    if freq == "M":
        return day.strftime("%Y-%m")
    return day.isoformat()


def monthly_mean(series: Series) -> tuple[tuple[str, float], ...]:
    buckets: dict[str, list[float]] = {}
    for day, value in series:
        buckets.setdefault(day.strftime("%Y-%m"), []).append(value)
    return tuple(
        (month, sum(values) / len(values)) for month, values in buckets.items()
    )


def history(series: Series, freq: Freq) -> tuple[tuple[str, float], ...]:
    if freq in {"D", "W"}:
        return monthly_mean(series[-MONTH_DAYS * HISTORY_MONTHS :])[-HISTORY_MONTHS:]
    limit = HISTORY_QUARTERS if freq == "Q" else HISTORY_MONTHS
    return tuple((period_label(day, freq), value) for day, value in series[-limit:])


def recession_bands(series: Series) -> tuple[tuple[str, str], ...]:
    bands: list[tuple[str, str]] = []
    start: date | None = None
    last: date | None = None
    for day, value in series:
        if value >= 1 and start is None:
            start = day
        if value < 1 and start is not None and last is not None:
            bands.append((start.strftime("%Y-%m"), last.strftime("%Y-%m")))
            start = None
        last = day
    if start is not None and last is not None:
        bands.append((start.strftime("%Y-%m"), last.strftime("%Y-%m")))
    return tuple(bands)
