from __future__ import annotations

from datetime import date

import pytest

from finagent_terminal.macro.sources import parse_number, parse_period, to_series
from finagent_terminal.macro.transforms import (
    apply,
    history,
    monthly_mean,
    recession_bands,
    rolling_sum,
    value_on_or_before,
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4.25", 4.25),
        ("1,385.10", 1385.1),
        (".", None),
        ("null", None),
        ("", None),
        ("NaN", None),
    ],
)
def test_parse_number_treats_source_placeholders_as_missing(
    raw: str, expected: float | None
) -> None:
    assert parse_number(raw) == expected


def test_to_series_sorts_and_drops_missing_values() -> None:
    series = to_series([("2026-03", "3"), ("2026-01", "1"), ("2026-02", ".")])

    assert series == ((date(2026, 1, 1), 1.0), (date(2026, 3, 1), 3.0))


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


def test_pct_diff_and_scale() -> None:
    series = _monthly([200.0, 202.0, 201.0])

    assert [v for _, v in apply(series, "pct", "M")] == [
        pytest.approx(1.0),
        pytest.approx(-0.4950495),
    ]
    assert [v for _, v in apply(series, "diff", "M", 1000.0)] == [2000.0, -1000.0]


def test_rolling_sum_needs_a_full_window() -> None:
    series = _monthly([1.0] * 13)

    result = rolling_sum(series, 12)

    assert len(result) == 2
    assert all(value == 12.0 for _, value in result)


def test_value_on_or_before() -> None:
    series = _monthly([1.0, 2.0, 3.0])

    assert value_on_or_before(series, date(2024, 2, 15)) == (date(2024, 2, 1), 2.0)
    assert value_on_or_before(series, date(2023, 12, 31)) is None


def test_daily_history_is_monthly_means_capped_at_24_months() -> None:
    series = tuple(
        (date(2023 + (m - 1) // 12, (m - 1) % 12 + 1, d), float(m))
        for m in range(1, 31)
        for d in (1, 15)
    )

    result = history(series, "D")

    assert len(result) == 24
    assert result[-1] == ("2025-06", 30.0)
    assert monthly_mean(((date(2026, 1, 1), 1.0), (date(2026, 1, 2), 3.0))) == (
        ("2026-01", 2.0),
    )


def test_quarterly_history_labels() -> None:
    series = ((date(2026, 4, 1), 2.1),)

    assert history(series, "Q") == (("2026Q2", 2.1),)


def test_recession_bands_find_each_run() -> None:
    series = _monthly([0, 1, 1, 0, 0, 1], start_year=2020)

    assert recession_bands(series) == (("2020-02", "2020-03"), ("2020-06", "2020-06"))
