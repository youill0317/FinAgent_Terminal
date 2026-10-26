from __future__ import annotations

from datetime import date

import pytest

from finagent.sources import parse_period
from finagent.transforms import (
    year_over_year,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-09", date(2026, 10, 9)),
        ("20261009", date(2026, 10, 9)),
        ("2026-09", date(2026, 9, 1)),
        ("202609", date(2026, 9, 1)),
        ("2026Q2", date(2026, 4, 1)),
        ("2026-Q4", date(2026, 10, 1)),
        ("2026", date(2026, 1, 1)),
        ("2026-13", None),
        ("garbage", None),
    ],
)
def test_parse_period_accepts_every_source_format(
    text: str, expected: date | None
) -> None:
    assert parse_period(text) == expected


def _monthly(
    values: list[float], start_year: int = 2024
) -> tuple[tuple[date, float], ...]:
    return tuple(
        (date(start_year + index // 12, index % 12 + 1, 1), value)
        for index, value in enumerate(values)
    )


def test_monthly_year_over_year_matches_same_month_last_year() -> None:
    series = _monthly([100.0] * 12 + [103.0])

    result = year_over_year(series, "M")

    assert result == ((date(2025, 1, 1), pytest.approx(3.0)),)


def test_daily_year_over_year_skips_points_without_a_year_of_history() -> None:
    series = (
        (date(2025, 1, 2), 50.0),
        (date(2026, 1, 2), 55.0),
        (date(2026, 1, 5), 60.0),
    )

    result = year_over_year(series, "D")

    assert [day for day, _ in result] == [date(2026, 1, 2), date(2026, 1, 5)]
    assert result[0][1] == pytest.approx(10.0)
