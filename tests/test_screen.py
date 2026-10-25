"""The S&P 500 screener without the network: fake sources and synthetic prices."""

# pandas ships no complete type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import math
import sys
import threading
from datetime import UTC, date, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, cast, override

import httpx
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

from finagent import screen
from finagent.cli import APP

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# 2026-10-22 18:00 in New York: the day's bar is final.
NOW = datetime(2026, 10, 22, 22, 0, tzinfo=UTC)
DAY = date(2026, 10, 22)
HTML = """<table id="constituents">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
<th>GICS Sub-Industry</th><th>CIK</th></tr>
<tr><td>AAA</td><td>Alpha Inc.</td><td>Information Technology</td>
<td>Software</td><td>1</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td>
<td>Multi-Sector Holdings</td><td>2</td></tr>
<tr><td>CCC</td><td>Charlie Co</td><td>Utilities</td>
<td>Electric Utilities</td><td>3</td></tr>
</table>"""


def ignore(*_: object) -> None:
    pass


def series(values: Sequence[float]) -> pd.Series:
    return pd.Series(values, dtype="float64")


# ---------------------------------------------------------------- fixtures


def ohlcv(
    ticker: str, start: date, end: date, growth: float, volume: float = 1e6
) -> pd.DataFrame:
    """Business days in [start, end) on a path fixed by the calendar date.

    The same day always has the same bar, so refetching an overlap matches.
    """
    days = pd.bdate_range(start, end - timedelta(days=1))
    if days.empty:
        return screen.empty_prices()
    age = np.asarray((days - pd.Timestamp("2024-01-01")).days, dtype="float64")
    close = 50 * (1 + growth) ** age * (1 + 0.02 * np.sin(age / 3))
    swing = 1 + 0.02 * np.cos(age / 5)
    return pd.DataFrame(
        {
            "date": days,
            "ticker": ticker,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": volume * swing,
        }
    )


UNIVERSE = pd.DataFrame(
    {
        "ticker": ["UP1", "UP2", "UP3", "FLAT", "DOWN", "BANK", "OIL", "THIN"],
        "name": ["Up One", "Up Two", "Up Three", "Flat", "Down", "Bank", "Oil", "Thin"],
        "sector": [
            "Information Technology",
            "Information Technology",
            "Information Technology",
            "Information Technology",
            "Information Technology",
            "Financials",
            "Energy",
            "Energy",
        ],
        "industry": [
            "Software",
            "Software",
            "Software",
            "Software",
            "Software",
            "Diversified Banks",
            "Oil & Gas",
            "Oil & Gas",
        ],
    }
)
GROWTH = {
    "UP1": 0.002,
    "UP2": 0.0015,
    "UP3": 0.001,
    "FLAT": 0.0,
    "DOWN": -0.001,
    "BANK": 0.0005,
    "OIL": -0.0005,
    "THIN": 0.0,
}
FUNDAMENTALS: dict[str, dict[str, object]] = {
    "UP1": {"fcf": 5e9, "roe": 0.30, "book_value": 10.0, "fpe": 30.0, "tpe": 35.0},
    "UP2": {"fcf": 3e9, "roe": 0.20, "book_value": 10.0, "fpe": 20.0, "tpe": 22.0},
    "UP3": {"fcf": 1e9, "roe": 0.10, "book_value": 10.0, "fpe": 10.0, "tpe": 12.0},
    "FLAT": {"fcf": 1e9, "roe": 0.25, "book_value": 10.0, "fpe": 8.0, "tpe": 9.0},
    "DOWN": {"fcf": -1e9, "roe": 0.05, "book_value": 10.0, "fpe": 15.0, "tpe": 14.0},
    "BANK": {"roe": 0.12, "book_value": 50.0, "fpe": 11.0, "tpe": 12.0},
    "OIL": {"fcf": 2e9, "roe": 0.15, "book_value": 30.0, "fpe": 5.0, "tpe": 10.0},
    "THIN": {"fcf": 1e8, "roe": 0.05, "book_value": 5.0, "fpe": 9.0, "tpe": 9.0},
}


def fundamentals(ticker: str) -> dict[str, object]:
    base: dict[str, object] = dict.fromkeys(screen.FUNDAMENTAL_COLUMNS[1:-1], math.nan)
    base |= {
        "total_debt": 2e9,
        "total_cash": 1e9,
        "ebitda": 2e9,
        "nd_ebitda": 0.5,
        "mcap": 1e11,
        "ev_ebitda": 12.0,
        "short_pct": 0.02,
        "revenue": 110.0,
        "revenue_prior": 100.0,
        "rev_growth": 0.10,
        "rev_year": "2025-12-31",
        "currency": "USD",
    }
    base |= FUNDAMENTALS[ticker]
    if ticker == "BANK":
        base |= {
            "eq_ratio": 0.08,
            "tce_ratio": 0.06,
            "ebitda": math.nan,
            "nd_ebitda": math.nan,
        }
    return base


def estimates(_: str) -> dict[str, object]:
    return {
        "fy0_now": 5.0,
        "fy0_30d": 4.9,
        "fy0_90d": 4.8,
        "fy0_up30": 3.0,
        "fy0_down30": 1.0,
        "fy1_now": 6.0,
        "fy1_30d": 5.9,
        "fy1_90d": 5.5,
        "fy1_up30": 4.0,
        "fy1_down30": 1.0,
        "est_currency": "USD",
    }


class FakeSource:
    def __init__(
        self, failures: Mapping[str, Sequence[Exception]] | None = None
    ) -> None:
        self.failures: dict[str, list[Exception]] = {
            k: list(v) for k, v in (failures or {}).items()
        }
        self.constituent_calls: int = 0
        self.price_calls: list[tuple[tuple[str, ...], date, date]] = []
        self.fundamental_calls: list[tuple[str, bool]] = []
        self.estimate_calls: list[str] = []

    def constituents(self) -> pd.DataFrame:
        self.constituent_calls += 1
        return UNIVERSE.copy()

    def prices(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        self.price_calls.append((tuple(tickers), start, end))
        frames = [
            ohlcv(t, start, end, GROWTH[t], 1e5 if t == "THIN" else 1e6)
            for t in tickers
        ]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fail(self, key: str) -> None:
        pending = self.failures.get(key)
        if pending:
            raise pending.pop(0)

    def fundamentals(self, ticker: str, *, financial: bool) -> Mapping[str, object]:
        self.fundamental_calls.append((ticker, financial))
        self._fail(ticker)
        return fundamentals(ticker)

    def estimates(self, ticker: str) -> Mapping[str, object]:
        self.estimate_calls.append(ticker)
        self._fail(f"est:{ticker}")
        return estimates(ticker)


def run_update(source: FakeSource, root: Path, now: datetime = NOW) -> str:
    return screen.update(source, root, now, ignore, sleep=ignore)


def stored(tmp_path: Path) -> pd.DataFrame:
    root = tmp_path / "data" / "screen"
    _ = run_update(FakeSource(), root)
    inputs = screen.load(root)
    assert inputs is not None
    return screen.evaluate(inputs)


# ---------------------------------------------------------------- parsing


def test_constituents_are_parsed_from_the_wikipedia_table() -> None:
    parsed = screen.parse_constituents(HTML)

    assert list(parsed.columns) == ["ticker", "name", "sector", "industry"]
    assert list(parsed["ticker"]) == ["AAA", "BRK-B", "CCC"]
    assert parsed.loc[1, "sector"] == "Financials"


def test_downloads_become_one_row_per_ticker_and_day() -> None:
    days = pd.DatetimeIndex(["2026-10-20", "2026-10-21"], name="Date")
    columns = pd.MultiIndex.from_product(
        [list(screen.YAHOO_PRICES), ["AAA", "ZZZ"]], names=["Price", "Ticker"]
    )
    frame = pd.DataFrame(np.nan, index=days, columns=columns)
    for field, value in zip(screen.YAHOO_PRICES, range(1, 7), strict=True):
        frame[(field, "AAA")] = float(value)

    rows = screen.long_prices(frame)

    assert list(rows.columns) == list(screen.PRICE_COLUMNS)
    assert list(rows["ticker"]) == ["AAA", "AAA"]  # ZZZ had no data
    assert rows.iloc[0]["adj_close"] == 5.0
    assert rows.iloc[0]["volume"] == 6.0
    assert screen.long_prices(pd.DataFrame()).empty


def test_fundamentals_row_reads_info_and_statements() -> None:
    info = {
        "freeCashflow": 1e9,
        "returnOnEquity": 0.2,
        "totalDebt": 5e9,
        "totalCash": 1e9,
        "ebitda": 2e9,
        "forwardPE": "Infinity",
        "shortPercentOfFloat": 0.031,
        "financialCurrency": "USD",
    }
    income = pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [90.0], pd.Timestamp("2024-12-31"): [100.0]},
        index=["TotalRevenue"],
    )
    balance = pd.DataFrame(
        {pd.Timestamp("2026-06-30"): [10.0, 100.0, 7.0, 4.0]},
        index=[
            "StockholdersEquity",
            "TotalAssets",
            "TangibleBookValue",
            "GoodwillAndOtherIntangibleAssets",
        ],
    )

    row = screen.fundamentals_row(info, income, balance)

    assert row["nd_ebitda"] == 2.0
    assert row["rev_growth"] == pytest.approx(-0.1)
    assert row["rev_year"] == "2025-12-31"
    assert row["eq_ratio"] == 0.1
    assert row["tce_ratio"] == pytest.approx(7 / 96)
    assert row["short_pct"] == 0.031
    assert math.isnan(cast("float", row["fpe"]))
    assert row["currency"] == "USD"

    bare = screen.fundamentals_row({"ebitda": -1.0}, pd.DataFrame())
    assert math.isnan(cast("float", bare["nd_ebitda"]))
    assert math.isnan(cast("float", bare["rev_growth"]))
    assert math.isnan(cast("float", bare["eq_ratio"]))


def test_tce_ratio_formula_and_missing_inputs() -> None:
    def ratio(**rows: float) -> float:
        balance = pd.DataFrame({pd.Timestamp("2026-06-30"): list(rows.values())})
        balance.index = pd.Index(list(rows))
        return screen.capital_ratios(balance)[1]

    # JPM, 2026-06-30: TangibleBookValue = CommonStockEquity - intangibles.
    jpm = ratio(
        TangibleBookValue=289.254,
        CommonStockEquity=353.558,
        GoodwillAndOtherIntangibleAssets=64.304,
        TotalAssets=5015.069,
    )
    assert jpm == pytest.approx(289.254 / (5015.069 - 64.304))
    # Without TangibleBookValue: common equity (no preferred) less intangibles.
    assert ratio(
        CommonStockEquity=10.0, GoodwillAndOtherIntangibleAssets=2.0, TotalAssets=102.0
    ) == pytest.approx(0.08)
    assert math.isnan(ratio(TangibleBookValue=8.0, TotalAssets=100.0))
    assert math.isnan(ratio(CommonStockEquity=10.0, TotalAssets=100.0))


def test_margin_average_over_the_fiscal_years() -> None:
    years = [pd.Timestamp(f"{year}-12-31") for year in range(2025, 2020, -1)]
    income = pd.DataFrame(
        [[100.0, 80.0, -10.0, 50.0, math.nan], [30.0, 8.0, 5.0, math.nan, 1.0]],
        index=["TotalRevenue", "OperatingIncome"],
        columns=years,
    )

    row = screen.fundamentals_row({}, income)

    # 2023 has negative revenue, 2022 no operating income, 2021 no revenue.
    assert row["margin_avg"] == pytest.approx((0.30 + 0.10) / 2)
    assert row["margin_years"] == 2
    revenue_only = cast("pd.DataFrame", income.loc[["TotalRevenue"]])
    for bare in (
        screen.fundamentals_row({}, pd.DataFrame()),
        screen.fundamentals_row({}, revenue_only),
    ):
        assert math.isnan(cast("float", bare["margin_avg"]))
        assert bare["margin_years"] == 0
    assert {"margin_avg", "margin_years"} <= set(screen.FUNDAMENTAL_COLUMNS)


def test_the_yahoo_margin_is_the_median_so_one_bad_year_does_not_drag_it() -> None:
    years = [pd.Timestamp(f"{year}-12-31") for year in range(2025, 2020, -1)]
    income = pd.DataFrame(
        [[100.0] * 5, [20.0, 22.0, -50.0, 18.0, 21.0]],  # 2023: a one-off charge
        index=["TotalRevenue", "OperatingIncome"],
        columns=years,
    )

    row = screen.fundamentals_row({}, income)

    assert row["margin_avg"] == pytest.approx(0.20)  # the mean would be 0.062
    assert row["margin_years"] == 5


def quarters(**rows: list[float]) -> pd.DataFrame:
    dates = [
        pd.Timestamp(d)
        for d in ("2026-05-31", "2026-02-28", "2025-11-30", "2025-08-31", "2025-05-31")
    ]
    return pd.DataFrame(rows, index=dates[: len(next(iter(rows.values())))]).T


def test_op_margin_is_ttm_over_the_latest_four_quarters() -> None:
    # The fifth (oldest) quarter is left out; the info field is ignored.
    quarterly = quarters(
        TotalRevenue=[100.0, 80.0, 60.0, 60.0, 999.0],
        OperatingIncome=[50.0, 30.0, 10.0, 10.0, -999.0],
    )

    row = screen.fundamentals_row(
        {"operatingMargins": 0.5}, pd.DataFrame(), quarterly=quarterly
    )

    assert row["op_margin"] == pytest.approx(100 / 300)
    assert row["op_margin_quarters"] == 4
    assert row["ttm_end"] == "2026-05-31"  # the newest quarterly statement
    assert {"op_margin", "op_margin_quarters", "ttm_end"} <= set(
        screen.FUNDAMENTAL_COLUMNS
    )
    assert "op_margin" not in screen.INFO_FIELDS


def test_op_margin_is_nan_without_four_full_quarters() -> None:
    nan = math.nan
    broken = [
        quarters(TotalRevenue=[100.0, 80.0, nan, 60.0], OperatingIncome=[5.0] * 4),
        quarters(
            TotalRevenue=[100.0, 80.0, 60.0, 60.0], OperatingIncome=[5.0, nan, 5.0, 5.0]
        ),
        quarters(TotalRevenue=[100.0, -100.0, 0.0, 0.0], OperatingIncome=[5.0] * 4),
        quarters(TotalRevenue=[100.0, 80.0, 60.0], OperatingIncome=[5.0] * 3),
        quarters(TotalRevenue=[100.0] * 4),
        pd.DataFrame(),
    ]
    for quarterly in [*broken, None]:
        row = screen.fundamentals_row({}, pd.DataFrame(), quarterly=quarterly)
        assert math.isnan(cast("float", row["op_margin"]))
        assert row["op_margin_quarters"] == 0
        assert row["ttm_end"] == ""  # no quarter is used


def test_estimates_row_reads_fy0_and_fy1() -> None:
    trend = pd.DataFrame(
        {
            "current": [1.0, 2.0, 3.0, 4.0],
            "30daysAgo": [1.0, 2.0, 2.9, 3.9],
            "90daysAgo": [1.0, 2.0, 2.8, 3.5],
            "currency": "USD",
        },
        index=pd.Index(["0q", "+1q", "0y", "+1y"], name="period"),
    )
    revisions = pd.DataFrame(
        {"upLast30days": [0, 0, 2, 5], "downLast30days": [0, 0, 1, 3]},
        index=trend.index,
    )

    row = screen.estimates_row(trend, revisions)

    assert row["fy0_now"] == 3.0
    assert row["fy1_90d"] == 3.5
    assert row["fy1_up30"] == 5.0
    assert row["fy1_down30"] == 3.0
    assert row["est_currency"] == "USD"
    empty = screen.estimates_row(pd.DataFrame(), pd.DataFrame())
    assert math.isnan(cast("float", empty["fy1_now"]))


# ---------------------------------------------------------------- indicators


def test_moving_averages() -> None:
    assert screen.sma(series(range(1, 11)), 3).iloc[-1] == 9.0
    # span 3: alpha 0.5, seeded with the first value.
    assert screen.ema(series([1, 2, 3]), 3).tolist() == [1.0, 1.5, 2.25]


def test_rsi_uses_wilder_smoothing() -> None:
    # Gains 1, 0, 1 and losses 0, 1, 0 smoothed with alpha 1/2.
    values = screen.rsi(series([1, 2, 1, 2]), days=2).tolist()

    assert math.isnan(values[0])
    assert math.isnan(values[1])
    assert values[2] == pytest.approx(50.0)
    assert values[3] == pytest.approx(75.0)
    assert screen.rsi(series(range(30))).iloc[-1] == 100.0


def test_percent_b_uses_a_population_deviation() -> None:
    deviation = math.sqrt(2 / 3)
    expected = (3 - (2 - 2 * deviation)) / (4 * deviation)

    assert screen.percent_b(series([1, 2, 3]), days=3).iloc[-1] == pytest.approx(
        expected
    )


def test_atr_is_wilder_smoothed_true_range() -> None:
    high = series([10, 12, 13])
    low = series([8, 9, 11])
    close = series([9, 11, 12])
    # True ranges 2, 3 (12 - 9), 2; alpha 1/2: 2, 2.5, 2.25.
    values = screen.atr(high, low, close, days=2).tolist()

    assert math.isnan(values[0])
    assert values[1:] == pytest.approx([2.5, 2.25])


def test_slow_stochastic() -> None:
    high = series([10, 11, 12, 13])
    low = series([8, 9, 10, 11])
    close = series([9, 10, 11, 13])
    # Raw %K over 2 days: —, 66.7, 66.7, 100; then SMA 2 and SMA 2.
    k, d = screen.stochastic(high, low, close, days=2, smooth=2, signal=2)

    assert k.iloc[-1] == pytest.approx((200 / 3 + 100) / 2)
    assert d.iloc[-1] == pytest.approx((200 / 3 + (200 / 3 + 100) / 2) / 2)


def test_obv_and_its_slope() -> None:
    volume = series([100, 200, 300, 400, 500])
    balance = screen.obv(series([10, 11, 10, 10, 12]), volume)

    assert balance.tolist() == [0.0, 200.0, -100.0, -100.0, 400.0]
    assert screen.slope(series([1, 3, 5])) == pytest.approx(2.0)


def test_adjusted_prices_scale_open_high_low() -> None:
    frame = pd.DataFrame(
        {
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [10.0],
            "adj_close": [5.0],
            "volume": [1.0],
        }
    )
    adj = screen.adjusted(frame)

    assert adj.iloc[0].to_dict() == {
        "open": 5.0,
        "high": 6.0,
        "low": 4.5,
        "close": 5.0,
        "volume": 1.0,
    }


def test_tech_row_needs_the_minimum_history() -> None:
    short = ohlcv("UP1", date(2026, 1, 1), DAY, 0.002)
    row = screen.tech_row(short)

    assert row["bars"] < screen.MIN_BARS
    assert set(row) == {"bars", "price", "dollar_vol20"}

    full = screen.tech_row(ohlcv("UP1", date(2024, 9, 1), DAY, 0.004))
    assert full["close"] > full["sma20"] > full["sma60"] > full["sma200"]
    assert full["sma60"] > full["sma60_lag20"]
    assert full["from_high"] <= 0


# ---------------------------------------------------------------- tech score


PASSING: dict[str, float] = {
    "close": 110.0,
    "sma20": 105.0,
    "sma60": 100.0,
    "sma60_lag20": 95.0,
    "sma200": 90.0,
    "macd": 1.0,
    "macd_signal": 0.5,
    "rsi": 60.0,
    "obv": 1000.0,
    "obv_lag20": 900.0,
    "up_vol60": 2.0,
    "down_vol60": 1.0,
    "high52": 200.0,  # far from the high: the OBV slope decides
    "obv_max60": 1000.0,
    "obv_min60": 0.0,
    "obv_slope60": 5.0,
}
FAILING = {
    "t_above200": {"sma200": 120.0},
    "t_20over60": {"sma20": 99.0},
    "t_60rising": {"sma60_lag20": 101.0},
    "t_above20": {"sma20": 111.0},
    "m_macd_signal": {"macd_signal": 2.0},
    "m_macd_positive": {"macd": -0.1, "macd_signal": -0.5},
    "m_rsi": {"rsi": 70.5},
    "v_obv20": {"obv_lag20": 1000.0},
    "v_updown": {"down_vol60": 2.0},
    "v_obv_high": {"obv_slope60": -1.0},
}


def checks(**changes: float) -> dict[str, bool]:
    row = screen.tech_checks(pd.DataFrame([PASSING | changes])).iloc[0]
    return {str(key): bool(value) for key, value in row.items()}


def test_every_check_passes_on_the_base_row() -> None:
    assert all(checks().values())
    assert list(checks()) == list(screen.TECH_CHECKS)


@pytest.mark.parametrize("key", list(FAILING))
def test_each_check_fails_alone(key: str) -> None:
    result = checks(**FAILING[key])

    assert not result[key]
    assert all(ok for name, ok in result.items() if name != key)


def test_a_new_high_needs_obv_near_its_high() -> None:
    near = {"high52": 111.0, "obv_slope60": -1.0}  # 110 >= 0.98 x 111

    assert checks(**near)["v_obv_high"]  # OBV 1000 is the 60-day high
    assert not checks(**near, obv=940.0, obv_lag20=900.0)["v_obv_high"]  # below 950
    assert checks(**near, obv=960.0, obv_lag20=900.0)["v_obv_high"]


def test_rsi_band_and_missing_values() -> None:
    assert checks(rsi=50.0)["m_rsi"]
    assert checks(rsi=70.0)["m_rsi"]
    assert not checks(rsi=49.9)["m_rsi"]
    assert not checks(sma200=math.nan)["t_above200"]


def test_overheat_flags() -> None:
    flags = screen.overheat(
        pd.DataFrame(
            {
                "rsi": [71.0, 65.0],
                "stoch_k": [85.0, 85.0],
                "stoch_d": [90.0, 80.0],
                "pct_b": [1.1, 0.9],
            }
        )
    )

    assert flags.iloc[0].tolist() == [True, True, True]
    assert flags.iloc[1].tolist() == [False, False, False]


# ---------------------------------------------------------------- cutlines


def test_cutlines_include_ties() -> None:
    scores = series([9, 8, 8, 8, 5, 4, 3, 2, 2, 1])

    # Top 20% is 2 stocks, but three share the 2nd score.
    assert screen.top_share(scores, 0.2).tolist() == [True] * 4 + [False] * 6
    # Bottom 40% is 4 stocks: 1, 2, 2, 3.
    assert screen.bottom_share(scores, 0.4).sum() == 4
    assert screen.bottom_share(series([1, 1, 1, 2]), 0.25).sum() == 3
    assert screen.top_share(series([math.nan, 1.0]), 0.2).tolist() == [False, True]


def test_liquidity_filter() -> None:
    table = pd.DataFrame(
        {
            "bars": [300.0, 200.0, 300.0, *[300.0] * 10, math.nan],
            "price": [50.0, 50.0, 4.0, *[50.0] * 10, math.nan],
            "dollar_vol20": [1.0, 9.0, 9.0, *range(10, 20), math.nan],
        }
    )

    reasons = screen.liquidity(table).tolist()

    assert reasons[1] == "일봉 250개 미만"
    assert reasons[2] == "주가 $5 미만"
    assert reasons[-1] == "시세 없음"
    # Of the 11 left, the lowest ceil(1.1) = 2 dollar volumes go.
    assert reasons[0] == "거래대금 하위 10%"
    assert reasons[3] == "거래대금 하위 10%"
    assert reasons[4:-1] == [""] * 9


def test_gics_lists_each_sub_industry_once() -> None:
    subs = [
        sub
        for industries in screen.GICS_TREE.values()
        for names in industries.values()
        for sub in names
    ]

    assert len(subs) == len(set(subs)) == len(screen.GICS) == 163
    assert len(screen.GICS_TREE) == 25  # industry groups
    assert len({industry for industry, _ in screen.GICS.values()}) == 74
    # MSCI code order, 60108010 to 60108050.
    assert screen.GICS_TREE["Equity Real Estate Investment Trusts (REITs)"][
        "Specialized REITs"
    ] == (
        "Other Specialized REITs",
        "Self-Storage REITs",
        "Telecom Tower REITs",
        "Timber REITs",
        "Data Center REITs",
    )


def peer_table(*members: tuple[str, str, int]) -> pd.DataFrame:
    rows = [(sub, sector) for sub, sector, count in members for _ in range(count)]
    return pd.DataFrame(rows, columns=["industry", "sector"])


def peer_levels(table: pd.DataFrame) -> list[tuple[str, str]]:
    rows = screen.records(table.assign(peer=screen.peers(table)))
    return [(str(row["peer"]), screen.peer_level(row)) for row in rows]


def test_peers_widen_through_four_levels() -> None:
    it = "Information Technology"
    table = peer_table(
        ("Application Software", it, 10),
        ("Systems Software", it, 6),  # industry Software: 16
        ("IT Consulting & Other Services", it, 2),  # group: 18
        ("Semiconductors", it, 3),  # 3 at every level
        ("Something New", it, 2),  # unmapped
    )

    found = peer_levels(table)

    assert found[0] == ("Application Software", "세부 산업")
    assert found[10] == ("Software", "산업")
    assert found[16] == ("Software & Services", "산업 그룹")
    assert found[18] == (it, "섹터")
    assert found[21] == (it, "섹터")
    # An unmapped sub-industry with enough members is still its own group.
    many = peer_table(("Something New", it, 10))
    assert peer_levels(many)[0] == ("Something New", "세부 산업")


def test_a_group_named_like_its_industry_stays_apart() -> None:
    table = peer_table(
        ("Transaction & Payment Processing Services", "Financials", 9),
        ("Multi-Sector Holdings", "Financials", 1),  # industry: 10
        ("Consumer Finance", "Financials", 3),  # group: 13
        ("Diversified Banks", "Financials", 7),
        ("Regional Banks", "Financials", 6),  # industry Banks: 13
    )

    found = peer_levels(table)

    assert found[0] == ("Financial Services", "산업")
    assert found[10] == ("Financial Services 전체", "산업 그룹")
    assert found[13] == ("Banks", "산업")
    assert found[-1] == ("Banks", "산업")
    table = table.assign(peer=screen.peers(table))
    size = screen.within_peers(pd.Series(1, index=table.index), table, "size")
    assert size.tolist() == [10] * 10 + [13] * 3 + [13] * 13


def test_banks_are_found_from_the_industry_group() -> None:
    table = peer_table(
        ("Diversified Banks", "Financials", 1),
        ("Regional Banks", "Financials", 1),
        ("Asset Management & Custody Banks", "Financials", 1),
        ("Reinsurance", "Financials", 1),
        ("Something New", "Financials", 1),
        ("Diversified Banks", "Industrials", 1),  # not Financials
    )

    group = screen.financial_group(table)

    assert group.tolist()[:4] == [
        screen.BANKS,
        screen.BANKS,
        "Financial Services",
        "Insurance",
    ]
    assert group.iloc[4:].isna().all()


# ---------------------------------------------------------------- fundamentals


def frame(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": "X",
        "sector": "Information Technology",
        "industry": "Software",
        "peer": "Information Technology",
        "fcf": 1.0,
        "roe": 0.2,
        "book_value": 10.0,
        "eq_ratio": math.nan,
        "tce_ratio": math.nan,
        "revenue_prior": 100.0,
        "rev_growth": 0.1,
        "total_debt": 5.0,
        "total_cash": 1.0,
        "ebitda": 2.0,
        "nd_ebitda": 2.0,
        "fpe": 15.0,
        "tpe": 15.0,
        "ev_ebitda": 10.0,
        "op_margin": 0.2,
        "margin_avg": 0.2,
        "margin_years": 4,
        "margin_source": "야후",
        "op_margin_source": "야후",
        "revenue": 10.0,
        "price": 50.0,
    }
    return row | overrides


def status(*rows: dict[str, object]) -> pd.DataFrame:
    return screen.fundamentals_status(pd.DataFrame(list(rows)))


def test_cash_flow_disqualifier() -> None:
    # Capital ratios play no part outside financials.
    result = status(
        frame(tce_ratio=0.01, eq_ratio=0.01), frame(fcf=-1.0), frame(fcf=math.nan)
    )

    assert result["dq_fcf"].tolist() == ["통과", "결격", "미확인"]
    assert result["dq_state"].tolist() == ["통과", "결격", "미확인"]


FIN_SERVICES = {
    "sector": "Financials",
    "industry": "Financial Exchanges & Data",
    "peer": "Financial Services",
    "fcf": math.nan,
}
BANK = {
    "sector": "Financials",
    "industry": "Regional Banks",
    "peer": "Banks",
    "fcf": math.nan,
}


def test_non_bank_financials_use_the_group_median_equity_ratio() -> None:
    result = status(
        frame(**FIN_SERVICES, eq_ratio=0.10),
        frame(**FIN_SERVICES, eq_ratio=0.12),
        frame(**FIN_SERVICES, eq_ratio=0.04),  # median 0.10, limit 0.05
        frame(**FIN_SERVICES, eq_ratio=0.10, roe=-0.01),
        frame(**FIN_SERVICES, eq_ratio=math.nan),
        # Banks are in another peer group and never move this median.
        frame(**BANK, eq_ratio=0.01, tce_ratio=0.05),
        frame(**BANK, eq_ratio=0.01, tce_ratio=0.05),
    )

    assert result["eq_limit"].iloc[0] == pytest.approx(0.05)
    assert result["dq_fcf"].tolist()[:5] == ["통과", "통과", "결격", "결격", "미확인"]
    # Net debt/EBITDA and EV/EBITDA are not used for financials.
    assert set(result["dq_debt"]) == {"해당 없음"}


def test_banks_use_a_tangible_common_equity_floor() -> None:
    result = status(
        frame(**BANK, tce_ratio=0.04),
        frame(**BANK, tce_ratio=0.0399),
        frame(**BANK, tce_ratio=0.08, roe=-0.01),
        frame(**BANK, tce_ratio=math.nan),
        frame(**BANK, tce_ratio=0.08, roe=math.nan),
        frame(**BANK, tce_ratio=0.08, eq_ratio=0.001),  # equity ratio unused
    )

    assert screen.BANK_TCE_FLOOR == 0.04
    assert result["dq_fcf"].tolist() == [
        "통과",
        "결격",
        "결격",
        "미확인",
        "미확인",
        "통과",
    ]
    assert result["eq_limit"].isna().tolist() == [True] * 6


def test_revenue_disqualifier() -> None:
    result = status(
        frame(rev_growth=-0.05),
        frame(rev_growth=-0.049),
        frame(rev_growth=math.nan, revenue_prior=-1.0),
        frame(rev_growth=math.nan),
    )

    assert result["dq_rev"].tolist() == ["결격", "통과", "해석 부적절", "미확인"]
    assert result["dq_state"].tolist()[2:] == ["미확인", "미확인"]


def test_net_debt_disqualifier() -> None:
    result = status(
        frame(nd_ebitda=2.0),
        frame(nd_ebitda=3.5),
        frame(nd_ebitda=4.5),
        frame(total_cash=10.0, nd_ebitda=-2.5),
        frame(ebitda=-1.0, nd_ebitda=math.nan),
        frame(total_debt=math.nan, nd_ebitda=math.nan),
    )

    assert result["nd_limit"].iloc[0] == pytest.approx(5.5)  # median 2.75 x 2
    assert result["dq_debt"].tolist() == [
        "통과",
        "통과",
        "통과",
        "순현금",
        "미산출",
        "미확인",
    ]
    floor = status(frame(nd_ebitda=1.0), frame(nd_ebitda=1.0), frame(nd_ebitda=3.1))
    assert floor["nd_limit"].iloc[0] == 3.0  # max(1.0 x 2, 3.0)
    assert floor["dq_debt"].tolist() == ["통과", "통과", "결격"]


def test_cyclicals_flag_peak_earnings() -> None:
    oil = {"sector": "Energy", "peer": "Energy"}
    result = status(
        frame(**oil, fpe=5.0, tpe=10.0),
        frame(**oil, fpe=7.0, tpe=10.0),
        frame(fpe=5.0, tpe=10.0),  # not cyclical
        frame(**oil, fpe=5.0, tpe=-10.0),
    )

    assert result["peak"].tolist() == [True, False, False, False]


def test_tailwind_needs_both_margin_conditions() -> None:
    ratio_binds = 0.25  # x 1.5 = 0.375 is above + 5%p = 0.30
    gap_binds = 0.04  # + 5%p = 0.09 is above x 1.5 = 0.06
    at_ratio = ratio_binds * screen.TAILWIND_RATIO
    at_gap = gap_binds + screen.TAILWIND_GAP
    result = status(
        frame(margin_avg=ratio_binds, op_margin=at_ratio),
        frame(margin_avg=ratio_binds, op_margin=np.nextafter(at_ratio, 0)),
        frame(margin_avg=gap_binds, op_margin=at_gap),
        frame(margin_avg=gap_binds, op_margin=np.nextafter(at_gap, 0)),
        frame(margin_avg=0.25, op_margin=0.9, margin_years=2),  # too few years
        frame(margin_avg=-0.1, op_margin=0.9),  # no positive average
        frame(margin_avg=math.nan, op_margin=0.9),
        frame(margin_avg=0.25, op_margin=math.nan),
        frame(**FIN_SERVICES, margin_avg=0.25, op_margin=0.9),  # financials
    )

    assert (screen.TAILWIND_RATIO, screen.TAILWIND_GAP) == (1.5, 0.05)
    assert screen.MARGIN_YEARS == 3
    assert result["tailwind"].tolist() == [True, False, True, False] + [False] * 5


# ---------------------------------------------------------------- 퀄리티 and 밸류에이션


def measured(*rows: dict[str, object]) -> pd.DataFrame:
    return screen.axes(pd.DataFrame(list(rows)))


def test_axes_rank_each_measure_within_its_peer_group() -> None:
    result = measured(
        frame(roe=0.10, fpe=30.0),
        frame(roe=0.20, fpe=20.0),
        frame(roe=0.30, fpe=10.0),
        # No base: negative equity, revenue and multiples are left unranked.
        frame(roe=0.90, book_value=-1.0, revenue=-1.0, fpe=-5.0, ev_ebitda=0.0),
        frame(roe=0.01, sector="Utilities", peer="Utilities"),  # alone
    )

    # Hazen percentiles, (rank - 0.5) / n: the middle of three is the median.
    assert result["q_roe"].tolist()[:3] == pytest.approx([1 / 6, 1 / 2, 5 / 6])
    assert result["v_fpe"].tolist()[:3] == pytest.approx([5 / 6, 1 / 2, 1 / 6])
    assert result.loc[3, ["q_roe", "q_fcf", "v_fpe", "v_ev"]].isna().all()
    assert result.loc[4, "q_roe"] == 0.5


def test_the_split_halves_each_peer_group() -> None:
    result = measured(
        *(frame(roe=roe) for roe in (0.1, 0.2, 0.3, 0.4)),
        *(frame(roe=roe, sector="Utilities", peer="Utilities") for roe in (1, 1, 2, 2)),
    )

    roe = result["q_roe"]
    assert roe.tolist()[:4] == pytest.approx([1 / 8, 3 / 8, 5 / 8, 7 / 8])
    assert roe.tolist()[4:] == pytest.approx([1 / 4, 1 / 4, 3 / 4, 3 / 4])  # ties
    assert (roe[:4] >= screen.SPLIT).sum() == 2
    assert (roe[4:] >= screen.SPLIT).sum() == 2


def test_a_wider_peer_group_keeps_the_narrower_members() -> None:
    it = "Information Technology"
    table = peer_table(("Application Software", it, 10), ("Systems Software", it, 6))
    table = table.assign(peer=screen.peers(table))
    roe = pd.Series([float(i) for i in range(16)], index=table.index)

    rank = screen.within_peers(roe, table, "rank", pct=True)
    size = screen.within_peers(roe, table, "size")

    # Systems Software is compared within all of Software, Application
    # Software only within itself.
    assert rank.iloc[-1] == 1.0
    assert rank.iloc[10] == pytest.approx(11 / 16)
    assert rank.iloc[9] == 1.0
    assert size.tolist() == [10] * 10 + [16] * 6


def test_financials_use_pbr_in_place_of_margins_and_ev() -> None:
    result = measured(
        frame(**(BANK | {"fcf": 1.0}), price=20.0),
        frame(**(BANK | {"fcf": 1.0}), price=10.0),
        frame(price=10.0),
    )

    assert result.loc[:1, ["q_margin", "q_fcf", "v_ev"]].isna().all().all()
    assert result["v_pb"].tolist()[:2] == [0.75, 0.25]
    assert result.loc[:1, ["q_roe", "q_growth", "v_fpe"]].notna().all().all()
    assert math.isnan(result.loc[2, "v_pb"])  # PBR is for financials only
    assert result.loc[2, ["q_margin", "q_fcf", "v_ev"]].notna().all()


def test_margin_measure_is_the_multi_year_average() -> None:
    result = measured(
        frame(margin_avg=0.10, op_margin=0.50),
        frame(margin_avg=0.30, op_margin=0.05),
        frame(margin_avg=0.20, margin_years=3),
        frame(margin_avg=0.90, margin_years=2),  # too few years
        frame(margin_avg=0.90, margin_years=math.nan),  # an older file
    )

    # The TTM margin plays no part.
    assert result["q_margin"].tolist()[:3] == pytest.approx([1 / 6, 5 / 6, 1 / 2])
    assert result.loc[3:, "q_margin"].isna().all()


def test_quality_needs_two_measures_and_valuation_one() -> None:
    result = measured(
        frame(),
        frame(margin_avg=math.nan, fcf=math.nan, rev_growth=math.nan),  # ROE only
        frame(margin_avg=math.nan, fcf=math.nan),  # ROE and revenue growth
        frame(fpe=math.nan, ev_ebitda=-1.0),
    )

    assert screen.QUALITY_MIN == 2
    assert math.isnan(result.loc[1, "quality"])
    assert result.loc[1, "valuation"] == pytest.approx(
        result.loc[1, ["v_fpe", "v_ev"]].mean()
    )
    assert result.loc[2, "quality"] == pytest.approx(
        (result.loc[2, "q_roe"] + result.loc[2, "q_growth"]) / 2
    )
    assert math.isnan(result.loc[3, "valuation"])
    assert math.isnan(result.loc[3, "edge"])
    first = result.loc[0]
    assert first["quality"] == pytest.approx(first[list(screen.QUALITY)].mean())
    assert first["edge"] == pytest.approx(first["quality"] - first["valuation"])


def cell_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "dq_state": "통과",
        "peak": False,
        "tailwind": False,
        "quality": 0.8,
        "valuation": 0.2,
        "peer_n": 3,  # small peer groups are not set apart
    }
    return row | overrides


def cells(*rows: dict[str, object]) -> list[str]:
    return screen.groups(pd.DataFrame(list(rows))).tolist()


def test_cells_of_quality_by_valuation() -> None:
    assert cells(
        cell_row(),
        cell_row(valuation=0.5),
        cell_row(quality=0.49),
        cell_row(quality=0.3, valuation=0.9),
        cell_row(quality=0.5, valuation=0.49),
        cell_row(peak=True),
        cell_row(quality=math.nan),
        cell_row(valuation=math.nan),
        cell_row(dq_state="결격"),
        cell_row(dq_state="미확인", peak=True),
        cell_row(tailwind=True),
        cell_row(tailwind=True, dq_state="결격"),
    ) == [
        "prime",
        "pricey",
        "trap",
        "weak",
        "prime",  # 퀄리티 0.5 is the upper half
        "hold",  # peak earnings
        "hold",
        "hold",
        "",  # judgment withheld
        "",
        "hold",  # 업황 순풍
        "",
    ]


def test_estimate_direction() -> None:
    table = pd.DataFrame(
        [
            (6.0, 5.0, 3.0, 1.0),
            (5.0, 5.0, 2.0, 2.0),
            (4.0, 5.0, 1.0, 3.0),
            (6.0, 5.0, 1.0, 3.0),  # higher, but more cuts
            (4.0, 5.0, 3.0, 1.0),  # lower, but more raises
            (4.0, 5.0, 2.0, 2.0),
            (math.nan, 5.0, 3.0, 1.0),
            (6.0, 5.0, math.nan, 1.0),
        ],
        columns=["fy1_now", "fy1_90d", "fy1_up30", "fy1_down30"],
    )

    assert screen.estimate_direction(table).tolist() == [
        "상향",
        "상향",
        "하향",
        "보합",
        "보합",
        "보합",
        "미확인",
        "미확인",
    ]


def test_cells_are_sorted_by_edge_and_capped() -> None:
    table = pd.DataFrame(
        {
            "group": ["prime"] * 25 + ["hold"] * 3,
            "edge": [i % 5 / 10 for i in range(25)] + [math.nan] * 3,
            "mcap": [float(i) for i in range(28)],
        }
    )

    top = screen.group(table, "prime")

    assert len(top) == screen.TOP_N
    assert top["edge"].tolist() == sorted(top["edge"], reverse=True)
    first = top[top["edge"] == 0.4]["mcap"].tolist()
    assert first == sorted(first, reverse=True)
    assert screen.group(table, "hold")["mcap"].tolist() == [27.0, 26.0, 25.0]


# ---------------------------------------------------------------- whole run


def test_evaluate_on_stored_data(tmp_path: Path) -> None:
    table = stored(tmp_path)
    by = table.set_index("ticker")

    assert set(table["ticker"]) == set(UNIVERSE["ticker"])
    left_out = by[by["liquidity"] != ""]
    assert list(left_out.index) == ["THIN"]  # lowest dollar volume (ceil 0.8)
    kept = screen.liquid(table)
    assert kept["score"].between(0, 10).all()
    assert (kept["chart"] == "강").sum() >= 2  # ceil(20% of 7), ties included
    assert by.loc["UP1", "score"] > by.loc["DOWN", "score"]
    assert by.loc["DOWN", "dq_fcf"] == "결격"
    assert by.loc["BANK", "dq_debt"] == "해당 없음"
    assert by.loc["BANK", "peer"] == "Financials"  # one bank: the sector
    assert by.loc["BANK", "dq_fcf"] == "통과"  # TCE/TA 6%
    assert bool(by.loc["OIL", "peak"])
    assert by.loc["OIL", "group"] == "hold"
    assert by.loc["DOWN", "group"] == ""  # 결격
    assert by.loc["BANK", "v_pb"] == 0.5  # alone in its group: the median
    assert math.isnan(by.loc["BANK", "v_ev"])
    assert set(by.loc[list(kept["ticker"]), "est"]) == {"상향"}
    assert screen.failures(table)["ticker"].tolist() == ["THIN"]
    assert all(
        screen.group(kept, key).shape[0] <= screen.TOP_N for key in screen.GROUPS
    )
    points = screen.map_rows(kept)
    placed = sum(screen.counts(kept)[key] for key in screen.GROUPS if key != "hold")
    assert len(points) == placed
    assert {point["ticker"] for point in points}.isdisjoint({"OIL", "DOWN"})


# ---------------------------------------------------------------- criteria

# ``evaluate`` on the fixtures before the criteria became adjustable.
PINNED = {
    # ticker: score, chart, dq_fcf, dq_state, peak, quality, valuation, group
    "UP1": (7, "약", "통과", "통과", False, 23 / 30, 0.7, "pricey"),
    "UP2": (7, "약", "통과", "통과", False, 17 / 30, 0.6, "pricey"),
    "UP3": (9, "강", "통과", "통과", False, 0.4, 0.4, "trap"),
    "FLAT": (8, "강", "통과", "통과", False, 16 / 30, 0.3, "prime"),
    "DOWN": (1, "약", "결격", "결격", False, 7 / 30, 0.5, ""),
    "BANK": (8, "강", "통과", "통과", False, 0.5, 0.5, "pricey"),
    "OIL": (4, "약", "통과", "통과", True, 0.5, 0.5, "hold"),
}
PINNED_VIEW = {
    # ticker: weak_spot, tags, verdict, hot
    "UP1": ("선행 PER 백분위 90(비쌈)", "", "통과", "RSI"),
    "UP2": ("선행 PER 백분위 70(비쌈)", "", "통과", "RSI · 스토캐스틱"),
    "UP3": ("ROE 백분위 30", "", "통과", "스토캐스틱"),
    "FLAT": ("FCF 마진 백분위 40", "", "통과", ""),
    "DOWN": ("ROE 백분위 10", "", "결격: FCF", ""),
    "BANK": ("ROE 백분위 50", "", "통과", "스토캐스틱"),
    "OIL": ("ROE 백분위 50", "정점 이익", "통과", ""),
}


def prepared(tmp_path: Path) -> pd.DataFrame:
    root = tmp_path / "data" / "screen"
    _ = run_update(FakeSource(), root)
    inputs = screen.load(root)
    assert inputs is not None
    return screen.prepare(inputs)


def test_the_default_criteria_reproduce_the_constants(tmp_path: Path) -> None:
    table = stored(tmp_path)
    by = table.set_index("ticker")
    columns = ["score", "chart", "dq_fcf", "dq_state", "peak"]
    for ticker, (*fixed, quality, valuation, cell) in PINNED.items():
        assert list(by.loc[ticker, columns]) == fixed
        assert by.loc[ticker, "quality"] == pytest.approx(quality, rel=1e-12)
        assert by.loc[ticker, "valuation"] == pytest.approx(valuation, rel=1e-12)
        assert by.loc[ticker, "group"] == cell
    assert by.loc["THIN", "liquidity"] == "거래대금 하위 10%"
    shown = screen.view(screen.liquid(table), screen.Criteria()).set_index("ticker")
    for ticker, texts in PINNED_VIEW.items():
        assert tuple(shown.loc[ticker, ["weak_spot", "tags", "verdict", "hot"]]) == (
            texts
        )
    again = screen.judge(prepared(tmp_path / "again"), screen.Criteria())
    pd.testing.assert_frame_equal(again, table, check_exact=True)
    assert screen.changed(screen.DEFAULT) == []
    assert screen.saved(screen.DEFAULT) == ({}, ())


def judged(tmp_path: Path, off: Sequence[str] = (), **values: float) -> pd.DataFrame:
    table = screen.judge(prepared(tmp_path), screen.criteria(values, off))
    return table.set_index("ticker")


def test_a_measure_turned_off_leaves_the_mean(tmp_path: Path) -> None:
    by = judged(tmp_path, ["q_roe"])

    assert bool(by["q_roe"].isna().all())
    assert by.loc["UP1", "quality"] == pytest.approx(0.7)  # margin, FCF, growth
    # A financial has ROE and growth only: growth alone now meets the need.
    assert by.loc["BANK", "quality"] == pytest.approx(0.5)
    assert screen.criteria({}, ["q_roe"]).need(financial=True) == 1
    assert screen.criteria({}, ["q_roe"]).need(financial=False) == 2


def test_with_every_measure_off_the_axis_is_not_computed(tmp_path: Path) -> None:
    by = judged(tmp_path, list(screen.QUALITY))

    kept = cast("pd.DataFrame", by[by["liquidity"] == ""])
    assert bool(kept["quality"].isna().all())
    assert set(kept["group"]) == {"hold", ""}  # DOWN keeps its 결격
    assert "퀄리티 산출 불가" in screen.tags(by.loc["UP1"].to_dict())


def test_checks_turned_off_leave_the_score_and_flags(tmp_path: Path) -> None:
    by = judged(tmp_path, ["t_above200", "o_rsi", "strong"])
    base = stored(tmp_path / "base").set_index("ticker")

    kept = cast("pd.DataFrame", by[by["liquidity"] == ""])
    before = base.loc[kept.index]
    assert (kept["score"] == before["score"] - before["t_above200"].astype(int)).all()
    assert "강" not in set(kept["chart"])
    assert not bool(kept["o_rsi"].any())
    c = screen.criteria({}, ["t_above200", "o_rsi"])
    row = by.loc["UP1"].to_dict()
    assert len(screen.tech_lines(row, c)) == len(screen.TECH_CHECKS) - 1
    assert [label for _, label, _ in screen.overheat_lines(row, c)] == [
        screen.OVERHEAT["o_stoch"],
        screen.OVERHEAT["o_bb"],
    ]

    none = judged(tmp_path / "none", list(screen.TECH_CHECKS))
    assert set(none["chart"].dropna()) == {""}


def test_filters_and_disqualifiers_turned_off(tmp_path: Path) -> None:
    by = judged(tmp_path, ["illiquid", "dq_fcf", "peak"])

    assert (by["liquidity"] == "").all()  # THIN stays
    assert set(by["dq_fcf"]) == {screen.UNUSED}
    assert by.loc["DOWN", "dq_state"] == screen.CLEAR
    assert by.loc["DOWN", "group"] != ""
    assert not bool(by["peak"].any())
    c = screen.criteria({}, ["dq_fcf", "peak"])
    assert screen.dq_lines(by.loc["DOWN"].to_dict(), c)[0][0] == screen.UNUSED
    peak = screen.label_lines(by.loc["OIL"].to_dict(), c)[-2]
    assert peak[0] == screen.UNUSED


def test_numbers_move_the_cutlines(tmp_path: Path) -> None:
    by = judged(tmp_path, min_price=100)

    assert by.loc["FLAT", "liquidity"] == "주가 $100 미만"  # about $50
    assert by.loc["UP1", "liquidity"] == ""
    by = judged(tmp_path / "split", split=0.6)
    assert by.loc["UP2", "group"] == "weak"  # 퀄리티 57% < 60%
    assert by.loc["BANK", "group"] == "trap"  # 밸류에이션 50% < 60%
    help_text = screen.group_help(screen.criteria({"split": 0.6}, ()))
    assert "60% 이상" in help_text


def test_saved_criteria_are_checked() -> None:
    c = screen.criteria(
        {
            "split": 0.6,
            "min_bars": 300,
            "bank_tce_floor": 0.045,
            "quality_min": 2.5,  # not whole
            "illiquid_share": 0.9,  # above its range
            "min_price": True,  # a bool is not a number
            "weak_share": math.nan,
            "gone": 1.0,
        },
        ["q_roe", "split", "gone", "o_rsi"],
    )

    assert (c.split, c.min_bars, c.bank_tce_floor) == (0.6, 300, 0.045)
    assert (c.quality_min, c.illiquid_share, c.min_price, c.weak_share) == (
        screen.QUALITY_MIN,
        screen.ILLIQUID_SHARE,
        screen.MIN_PRICE,
        screen.WEAK_SHARE,
    )
    assert c.off == ("q_roe", "o_rsi")  # a boundary cannot be off
    assert screen.changed(c) == ["min_bars", "dq_fcf", "q_roe", "split", "o_rsi"]
    values, off = screen.saved(c)
    assert screen.criteria(values, off) == c
    assert {rule.key for rule in screen.RULES} >= set(screen.TECH_CHECKS)


# ---------------------------------------------------------------- storage


def test_quarter_and_universe_age(tmp_path: Path) -> None:
    assert screen.quarter(date(2026, 10, 1)) == "2026-Q4"
    assert screen.quarter(date(2026, 3, 31)) == "2026-Q1"
    assert screen.universe_due(tmp_path, DAY)
    UNIVERSE.to_csv(tmp_path / "universe-2026-10-01.csv", index=False)
    assert not screen.universe_due(tmp_path, date(2026, 10, 30))
    assert screen.universe_due(tmp_path, date(2026, 10, 31))


def test_the_last_final_bar_depends_on_new_york_time() -> None:
    # 16:59 and 17:00 in New York (EDT).
    assert screen.last_session(datetime(2026, 10, 22, 20, 59, tzinfo=UTC)) == date(
        2026, 10, 21
    )
    assert screen.last_session(datetime(2026, 10, 22, 21, 0, tzinfo=UTC)) == DAY


def test_price_plan_and_merge() -> None:
    store = pd.concat(
        [
            ohlcv("UP1", date(2026, 9, 1), date(2026, 10, 20), 0.001),
            ohlcv("OLD", date(2026, 9, 1), date(2026, 9, 10), 0.001),
        ],
        ignore_index=True,
    )
    start, full = screen.price_plan(store, ["UP1", "OLD", "NEW"])

    assert start == date(2026, 10, 19) - timedelta(days=screen.OVERLAP_DAYS)
    assert full == ["OLD", "NEW"]
    assert screen.price_plan(screen.empty_prices(), ["UP1"]) == (None, ["UP1"])

    assert start is not None
    fresh = ohlcv("UP1", start, date(2026, 10, 23), 0.001)
    assert screen.drifted(store, fresh) == []
    moved = fresh.assign(adj_close=fresh["adj_close"] * 0.99)
    assert screen.drifted(store, moved) == ["UP1"]

    merged = screen.merge_prices(store, [fresh], [], ["UP1"], date(2026, 10, 22))
    assert set(merged["ticker"]) == {"UP1"}  # OLD left the universe
    assert not merged.duplicated(["ticker", "date"]).any()
    assert merged["date"].max() == pd.Timestamp("2026-10-22")
    assert merged["date"].min() == pd.Timestamp("2026-09-01")


def test_first_update_writes_every_part(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    source = FakeSource()

    message = run_update(source, root)

    names = sorted(path.name for path in root.iterdir())
    assert names == [
        "estimates-2026-10-22.csv",
        "fundamentals-2026-Q4.csv",
        "prices.parquet",
        "universe-2026-10-22.csv",
    ]
    assert message.startswith("8종목 · 시세 2026-10-22")
    assert ("BANK", True) in source.fundamental_calls
    assert ("UP1", False) in source.fundamental_calls
    _, start, end = source.price_calls[0]
    assert start == DAY - timedelta(days=screen.HISTORY_DAYS)
    assert end == DAY + timedelta(days=1)
    assert screen.parts(root) == screen.Parts(DAY, DAY, "2026-Q4", DAY)


def test_later_updates_fetch_only_what_is_due(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    _ = run_update(FakeSource(), root)
    rows = len(screen.read_prices(root))

    later = FakeSource()
    _ = run_update(later, root, NOW + timedelta(days=7))

    assert later.constituent_calls == 0
    assert later.fundamental_calls == []  # same quarter
    assert len(later.estimate_calls) == len(UNIVERSE)  # a new day
    starts = {start for _, start, _ in later.price_calls}
    assert starts == {DAY - timedelta(days=screen.OVERLAP_DAYS)}  # incremental
    prices = screen.read_prices(root)
    assert prices["date"].max() == pd.Timestamp("2026-10-29")
    assert len(prices) == rows  # a week added, a week out of the window
    assert not prices.duplicated(["ticker", "date"]).any()

    month = FakeSource()
    _ = run_update(month, root, datetime(2026, 11, 21, 22, tzinfo=UTC))
    assert month.constituent_calls == 1
    assert month.fundamental_calls == []

    quarter = FakeSource()
    _ = run_update(quarter, root, datetime(2027, 1, 5, 22, tzinfo=UTC))
    assert len(quarter.fundamental_calls) == len(UNIVERSE)
    assert (root / "fundamentals-2027-Q1.csv").exists()


def test_a_finished_update_keeps_only_the_newest_dated_files(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    _ = run_update(FakeSource(), root)
    older = root / "universe-2026-09-01.csv"
    _ = older.write_text("ticker\n")
    sibling = root / "sp400"
    sibling.mkdir()
    kept = [sibling / "estimates-2026-10-01.csv", sibling / "estimates-2026-10-02.csv"]
    for path in kept:
        _ = path.write_text("ticker\n")

    refusals = [screen.RateLimitedError("429")] * (len(screen.RETRY_WAITS) + 1)
    with pytest.raises(screen.RateLimitedError):
        _ = run_update(FakeSource({"est:UP1": refusals}), root, NOW + timedelta(days=7))
    assert older.exists()  # a stopped update deletes nothing
    assert (root / "estimates-2026-10-22.csv").exists()

    _ = run_update(FakeSource(), root, NOW + timedelta(days=7))

    assert sorted(path.name for path in root.iterdir()) == [
        "estimates-2026-10-29.csv",
        "fundamentals-2026-Q4.csv",
        "prices.parquet",
        "sp400",
        "universe-2026-10-22.csv",
    ]
    assert all(path.exists() for path in kept)  # another root is left alone


def test_rate_limits_are_retried_and_failures_recorded(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    source = FakeSource(
        {"UP1": [screen.RateLimitedError("429")], "DOWN": [ValueError("no data")]}
    )
    waits: list[float] = []

    _ = screen.update(source, root, NOW, ignore, sleep=waits.append)

    assert [t for t, _ in source.fundamental_calls].count("UP1") == 2
    assert waits[0] == screen.RETRY_WAITS[0]
    saved = pd.read_csv(root / "fundamentals-2026-Q4.csv")
    errors = dict(zip(saved["ticker"], saved["error"].fillna(""), strict=True))
    assert errors["DOWN"] == "ValueError: no data"
    assert errors["UP1"] == ""

    again = FakeSource()
    _ = run_update(again, root, NOW + timedelta(hours=1))
    assert [t for t, _ in again.fundamental_calls] == ["DOWN"]  # only the failure
    assert again.estimate_calls == []  # today's estimates are complete


def test_an_update_stopped_by_yahoo_keeps_what_it_fetched(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    refusals = [screen.RateLimitedError("429")] * (len(screen.RETRY_WAITS) + 1)
    source = FakeSource({"FLAT": refusals})

    with pytest.raises(screen.RateLimitedError):
        _ = run_update(source, root)

    saved = pd.read_csv(root / "fundamentals-2026-Q4.csv")
    assert saved["ticker"].tolist() == ["UP1", "UP2", "UP3"]
    resumed = FakeSource()
    _ = run_update(resumed, root)
    assert resumed.fundamental_calls[0][0] == "FLAT"
    assert not list(root.glob("*.tmp"))


def test_an_older_fundamentals_file_is_fetched_again(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    _ = run_update(FakeSource(), root)
    path = root / "fundamentals-2026-Q4.csv"
    pd.read_csv(path).drop(columns=["margin_avg", "margin_years"]).to_csv(
        path, index=False
    )
    refusals = [screen.RateLimitedError("429")] * (len(screen.RETRY_WAITS) + 1)
    stopped = FakeSource({"UP3": refusals})

    with pytest.raises(screen.RateLimitedError):
        _ = run_update(stopped, root, NOW + timedelta(hours=1))

    assert [t for t, _ in stopped.fundamental_calls][:2] == ["UP1", "UP2"]
    saved = pd.read_csv(path)
    assert set(screen.FUNDAMENTAL_COLUMNS) <= set(saved.columns)
    errors = dict(zip(saved["ticker"], saved["error"].fillna(""), strict=True))
    assert errors["UP1"] == ""
    assert errors["FLAT"] == "옛 형식 (margin_avg, margin_years 없음)"
    # The old rows still count as failures, so the next update finishes them.
    resumed = FakeSource()
    _ = run_update(resumed, root, NOW + timedelta(hours=2))
    assert [t for t, _ in resumed.fundamental_calls] == [
        "UP3",
        "FLAT",
        "DOWN",
        "BANK",
        "OIL",
        "THIN",
    ]
    done = FakeSource()
    _ = run_update(done, root, NOW + timedelta(hours=3))
    assert done.fundamental_calls == []


def test_a_split_refetches_that_tickers_history(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    _ = run_update(FakeSource(), root)

    class Split(FakeSource):
        @override
        def prices(
            self, tickers: Sequence[str], start: date, end: date
        ) -> pd.DataFrame:
            frame = super().prices(tickers, start, end)
            frame.loc[frame["ticker"] == "UP2", "adj_close"] /= 2
            return frame

    source = Split()
    _ = run_update(source, root, NOW + timedelta(days=1))

    full = [
        tickers for tickers, start, _ in source.price_calls if start < date(2026, 1, 1)
    ]
    assert full == [("UP2",)]
    up2 = screen.history(root, "UP2")
    assert up2["date"].min() < pd.Timestamp("2025-01-01")


def no_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Updater's run without its real pauses between requests."""
    monkeypatch.setattr(screen, "update", partial(screen.update, sleep=ignore))


def test_one_update_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    no_pauses(monkeypatch)
    gate = threading.Event()

    class Slow(FakeSource):
        @override
        def constituents(self) -> pd.DataFrame:
            _ = gate.wait(5)
            return super().constituents()

    def slow(_: Path) -> screen.Source:
        return Slow()

    updater = screen.Updater(tmp_path, source=slow)

    assert updater.start()
    assert not updater.start()
    assert updater.progress.running
    gate.set()
    pause = threading.Event()
    for _ in range(300):
        if not updater.progress.running:
            break
        _ = pause.wait(0.1)

    progress = updater.progress
    assert not progress.running
    assert not progress.failed, progress.message
    assert progress.message.startswith("8종목")
    assert screen.load(tmp_path) is not None


def test_missing_packages_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    def nothing(_: str) -> None:
        return None

    monkeypatch.setattr("importlib.util.find_spec", nothing)

    assert screen.missing_packages() == ["yfinance", "lxml"]


# ---------------------------------------------------------------- SEC margins

AGENT = "finagent-test contact"  # no real address in tests
REVENUES, CONTRACT, SALES, GROSS = screen.REVENUE_CONCEPTS
INCOME = screen.OPERATING_INCOME


def fact(  # noqa: PLR0913 - one companyfacts entry
    end: str,
    value: float,
    *,
    filed: str = "",
    days: int = 364,
    form: str = "10-K",
    fp: str = "FY",
) -> dict[str, object]:
    stop = date.fromisoformat(end)
    return {
        "start": (stop - timedelta(days=days)).isoformat(),
        "end": end,
        "val": value,
        "form": form,
        "fp": fp,
        "filed": filed or f"{stop.year + 1}-02-15",
    }


def companyfacts(**concepts: list[dict[str, object]]) -> dict[str, object]:
    us_gaap = {name: {"units": {"USD": rows}} for name, rows in concepts.items()}
    return {"cik": 1, "entityName": "X", "facts": {"us-gaap": us_gaap}}


def years(first: int, last: int, value: float) -> list[dict[str, object]]:
    return [fact(f"{year}-12-31", value) for year in range(first, last + 1)]


def test_annual_values_are_10k_full_years_latest_filed() -> None:
    facts = companyfacts(
        **{
            INCOME: [
                fact("2024-12-31", 10.0, filed="2025-02-15"),
                fact("2024-12-31", 12.0, filed="2026-02-15"),  # restated
                fact("2025-12-31", 20.0),
                fact("2025-12-28", 99.0, days=91),  # the fourth quarter alone
                fact("2025-09-30", 99.0, days=273, form="10-Q", fp="Q3"),
                fact("2023-12-31", 99.0, form="8-K"),
                fact("2022-12-31", 99.0, fp="Q4"),
                fact("2021-12-31", 7.0, days=371),  # a 53-week year
            ]
        }
    )

    assert screen.annual(facts, INCOME) == {
        "2025-12-31": (20.0, "2026-02-15"),
        "2024-12-31": (12.0, "2026-02-15"),
        "2021-12-31": (7.0, "2022-02-15"),
    }
    assert screen.annual(facts, REVENUES) == {}
    assert screen.annual({}, INCOME) == {}


def test_revenue_concepts_fall_back_in_order() -> None:
    facts = companyfacts(
        **{
            INCOME: years(2015, 2025, 10.0),
            SALES: years(2015, 2018, 50.0),  # before ASC 606
            CONTRACT: years(2018, 2025, 100.0),
            REVENUES: years(2023, 2025, 200.0),  # fewer years in the window
        }
    )

    rows = screen.margin_history(facts)

    assert [str(row["end"])[:4] for row in rows] == [
        str(year) for year in range(2025, 2014, -1)
    ]
    concepts = {str(row["end"])[:4]: row["concept"] for row in rows}
    assert {concepts[str(y)] for y in range(2018, 2026)} == {CONTRACT}
    assert {concepts[str(y)] for y in range(2015, 2018)} == {SALES}
    assert rows[0] == {
        "period": "FY",
        "end": "2025-12-31",
        "revenue": 100.0,
        "op_income": 10.0,
        "concept": CONTRACT,
        "filed": "2026-02-15",
    }
    tie = companyfacts(
        **{INCOME: years(2021, 2025, 1.0), CONTRACT: years(2021, 2025, 4.0)}
        | {REVENUES: years(2021, 2025, 5.0)}
    )
    assert {row["concept"] for row in screen.margin_history(tie)} == {REVENUES}
    gross = companyfacts(
        **{INCOME: years(2021, 2025, 1.0), GROSS: years(2021, 2025, 4.0)}
    )
    assert {row["concept"] for row in screen.margin_history(gross)} == {GROSS}
    assert screen.margin_history(companyfacts()) == []


def history(ticker: str, facts: dict[str, object]) -> pd.DataFrame:
    rows = [{"ticker": ticker, **row} for row in screen.margin_history(facts)]
    return pd.DataFrame(rows, columns=list(screen.MARGIN_COLUMNS))


def test_the_sec_average_takes_the_newest_five_full_years() -> None:
    full = companyfacts(
        **{
            INCOME: [*years(2019, 2020, 50.0), *years(2021, 2025, 10.0)],
            CONTRACT: years(2019, 2025, 100.0),
        }
    )
    short = companyfacts(
        **{INCOME: years(2022, 2025, 10.0), CONTRACT: years(2022, 2025, 100.0)}
    )
    # Operating income stopped in 2019, so the newest five years lack it.
    stale = companyfacts(
        **{INCOME: years(2015, 2019, 10.0), CONTRACT: years(2015, 2025, 100.0)}
    )
    losses = companyfacts(
        **{
            INCOME: years(2021, 2025, 10.0),
            CONTRACT: [*years(2021, 2024, 100.0), fact("2025-12-31", 0.0)],
        }
    )
    table = pd.concat(
        [
            history("FULL", full),
            history("SHORT", short),
            history("STALE", stale),
            history("ZERO", losses),
        ]
    )

    result = screen.sec_margins(table).set_index("ticker")

    assert result.loc["FULL", "sec_avg"] == pytest.approx(0.10)  # not 2019-2020
    assert result["sec_years"].to_dict() == {
        "FULL": 5,
        "SHORT": 4,
        "STALE": 0,
        "ZERO": 4,
    }
    assert screen.SEC_YEARS == 5


def test_the_sec_margin_is_the_median_of_its_five_years() -> None:
    income = [20.0, 22.0, -50.0, 18.0, 21.0]  # 2023: a one-off charge
    facts = companyfacts(
        **{
            INCOME: [fact(f"{2021 + i}-12-31", op) for i, op in enumerate(income)],
            CONTRACT: years(2021, 2025, 100.0),
        }
    )

    result = screen.sec_margins(history("ONE", facts)).set_index("ticker")

    assert result.loc["ONE", "sec_avg"] == pytest.approx(0.20)  # mean 0.062
    assert result.loc["ONE", "sec_years"] == 5


def quarter_facts(year: int, value: float) -> list[dict[str, object]]:
    """Three 10-Q quarters and the 10-K's own fourth, each ``value``."""
    ends = ("03-31", "06-30", "09-30")
    filed = [fact(f"{year}-{end}", value, days=90, form="10-Q", fp="Q") for end in ends]
    return [*filed, fact(f"{year}-12-31", value, days=91)]


def test_quarters_are_three_month_values_latest_filed() -> None:
    facts = companyfacts(
        **{
            INCOME: [
                fact("2025-03-29", 1.0, days=90, form="10-Q", fp="Q1"),
                fact(
                    "2025-06-28", 2.0, days=91, form="10-Q", fp="Q2", filed="2025-08-01"
                ),
                fact("2025-06-28", 2.5, days=91, form="10-Q", fp="Q2"),  # restated
                fact("2025-06-28", 9.0, days=182, form="10-Q", fp="Q2"),  # half year
                fact("2025-10-04", 3.0, days=98, form="10-Q", fp="Q3"),  # 14 weeks
                fact("2025-10-04", 9.0, days=280, form="10-Q", fp="Q3"),  # nine months
                fact("2026-01-03", 4.0, days=91),  # the 10-K's fourth quarter
                fact("2025-12-31", 9.0, days=91, form="8-K", fp="Q4"),
                fact("2025-03-01", 9.0, days=60, form="10-Q", fp="Q1"),
            ]
        }
    )

    assert screen.quarterly(facts, INCOME) == {
        "2025-03-29": (1.0, "2026-02-15"),
        "2025-06-28": (2.5, "2026-02-15"),
        "2025-10-04": (3.0, "2026-02-15"),
        "2026-01-03": (4.0, "2027-02-15"),
    }


def test_a_missing_fourth_quarter_is_the_year_less_nine_months() -> None:
    year = fact("2025-12-31", 100.0, days=364)  # 2025-01-01 to 2025-12-31
    nine = fact("2025-09-30", 70.0, days=272, form="10-Q", fp="Q3", filed="2025-11-01")
    late = fact("2025-09-30", 72.0, days=272, form="10-Q", fp="Q3", filed="2026-11-01")
    other = fact("2025-09-30", 50.0, days=200, form="10-Q", fp="Q3")  # not the year's

    derived = screen.quarterly(companyfacts(**{INCOME: [year, nine, other]}), INCOME)
    restated = screen.quarterly(companyfacts(**{INCOME: [year, nine, late]}), INCOME)
    filed = screen.quarterly(
        companyfacts(**{INCOME: [year, nine, fact("2025-12-31", 31.0, days=91)]}),
        INCOME,
    )

    assert derived == {"2025-12-31": (30.0, "2026-02-15")}
    # The nine months the 10-K was filed against, not a later restatement.
    assert restated == {"2025-12-31": (30.0, "2026-02-15")}
    assert filed == {"2025-12-31": (31.0, "2026-02-15")}  # a filed one stands
    assert screen.quarterly(companyfacts(**{INCOME: [year]}), INCOME) == {}


def test_a_derived_quarter_takes_both_values_from_one_filing() -> None:
    year = fact("2025-12-31", 100.0, days=364)  # filed 2026-02-15
    nine = fact("2025-09-30", 70.0, days=272, form="10-Q", fp="Q3", filed="2025-11-01")
    late = fact("2025-09-30", 72.0, days=272, form="10-Q", fp="Q3", filed="2026-11-01")
    recast = fact("2025-12-31", 104.0, days=364, filed="2027-02-15")
    after = fact("2025-09-30", 71.0, days=272, form="10-Q", fp="Q3", filed="2026-03-01")

    def q4(*rows: dict[str, object]) -> tuple[float, str]:
        return screen.quarterly(companyfacts(**{INCOME: list(rows)}), INCOME)[
            "2025-12-31"
        ]

    assert q4(year, nine, late) == (30.0, "2026-02-15")
    assert q4(year, nine, late, recast) == (32.0, "2027-02-15")  # both recast
    assert q4(year, late, after) == (29.0, "2026-03-01")  # none before: the first


def test_quarters_filed_only_as_year_to_date_are_differences() -> None:
    # Fiscal year from 2025-01-01: Q1 alone, then only six and nine months.
    q1 = fact("2025-03-31", 10.0, days=89, form="10-Q", fp="Q1")
    half = fact("2025-06-30", 25.0, days=180, form="10-Q", fp="Q2")
    nine = fact("2025-09-30", 45.0, days=272, form="10-Q", fp="Q3")
    year = fact("2025-12-31", 70.0, days=364)
    restated = fact(
        "2025-06-30", 26.0, days=180, form="10-Q", fp="Q2", filed="2027-01-01"
    )
    filed_q2 = fact("2025-06-30", 14.0, days=91, form="10-Q", fp="Q2")

    derived = screen.quarterly(companyfacts(**{INCOME: [q1, half, nine, year]}), INCOME)
    later = screen.quarterly(companyfacts(**{INCOME: [q1, half, restated]}), INCOME)
    alone = screen.quarterly(companyfacts(**{INCOME: [q1, half, filed_q2]}), INCOME)

    assert {end: value for end, (value, _) in derived.items()} == {
        "2025-03-31": 10.0,
        "2025-06-30": 15.0,  # six months less the first three
        "2025-09-30": 20.0,  # nine less six
        "2025-12-31": 25.0,  # the year less nine
    }
    assert later["2025-06-30"] == (16.0, "2027-01-01")  # the restated half year
    assert alone["2025-06-30"] == (14.0, "2026-02-15")  # a filed quarter stands
    # Without the year to date before it, a half year gives no quarter.
    assert screen.quarterly(companyfacts(**{INCOME: [half]}), INCOME) == {}


def test_the_sec_ttm_needs_four_quarters_of_both_values() -> None:
    def ttm(income: list[dict[str, object]], revenue: list[dict[str, object]]) -> float:
        facts = companyfacts(
            **{
                INCOME: [*years(2021, 2025, 10.0), *income],
                REVENUES: [*years(2021, 2025, 100.0), *revenue],
            }
        )
        margins = screen.sec_margins(history("X", facts))
        return float(margins["sec_ttm"].iloc[0])

    q = quarter_facts(2025, 5.0)
    sales = quarter_facts(2025, 20.0)
    assert ttm(q, sales) == pytest.approx(0.25)
    assert math.isnan(ttm(q[:3], sales))  # no fourth quarter of income
    assert math.isnan(ttm(q, sales[:3]))  # no fourth quarter of revenue
    gap = [*quarter_facts(2024, 5.0)[:1], *q[1:]]  # 2024-03-31 then 2025-06-30
    assert math.isnan(ttm(gap, [*quarter_facts(2024, 20.0)[:1], *sales[1:]]))
    stale = quarter_facts(2024, 5.0)  # the 2025 10-K is newer than these
    assert math.isnan(ttm(stale, quarter_facts(2024, 20.0)))
    losses = [fact(str(row["end"]), -20.0, days=90) for row in sales]
    assert math.isnan(ttm(q, losses))  # revenue summing to zero or less


def test_retail_calendars_of_twelve_and_sixteen_weeks_count() -> None:
    def weeks(end: str, n: int, value: float, fp: str) -> dict[str, object]:
        return fact(end, value, days=7 * n - 1, form="10-Q", fp=fp)

    # Kroger-like: a 16-week first quarter, then 12 weeks each.
    kroger = companyfacts(
        **{
            INCOME: [
                weeks("2025-05-24", 16, 16.0, "Q1"),
                weeks("2025-08-16", 28, 28.0, "Q2"),  # six months: 28 weeks
                weeks("2025-11-08", 40, 40.0, "Q3"),  # nine months: 40 weeks
                fact("2026-01-31", 52.0, days=363),
            ]
        }
    )
    # PepsiCo-like: 12-week quarters, then a 16-week fourth.
    pepsi = companyfacts(
        **{
            INCOME: [
                weeks("2025-03-22", 12, 12.0, "Q1"),
                weeks("2025-06-14", 24, 24.0, "Q2"),  # six months: 24 weeks
                weeks("2025-09-06", 36, 36.0, "Q3"),  # nine months: 36 weeks
                fact("2025-12-27", 52.0, days=363),
            ]
        }
    )

    assert {e: v for e, (v, _) in screen.quarterly(kroger, INCOME).items()} == {
        "2025-05-24": 16.0,
        "2025-08-16": 12.0,
        "2025-11-08": 12.0,
        "2026-01-31": 12.0,
    }
    assert {e: v for e, (v, _) in screen.quarterly(pepsi, INCOME).items()} == {
        "2025-03-22": 12.0,
        "2025-06-14": 12.0,
        "2025-09-06": 12.0,
        "2025-12-27": 16.0,
    }

    def ttm(*ends: str) -> float:
        rows = pd.DataFrame({"end": list(ends), "revenue": 4.0, "op_income": 1.0})
        return screen.sec_ttm(rows, ends[0])

    assert ttm("2026-01-31", "2025-11-08", "2025-08-16", "2025-05-24") == 0.25
    assert ttm("2025-12-27", "2025-09-06", "2025-06-14", "2025-03-22") == 0.25
    # A four-month period among them spans more than a year.
    assert math.isnan(ttm("2026-04-29", "2025-12-31", "2025-09-30", "2025-06-30"))
    # Two values for one quarter under different ends are not two quarters.
    assert math.isnan(ttm("2025-12-31", "2025-12-27", "2025-09-27", "2025-06-28"))


def test_a_10q_quarter_beats_the_10k_and_a_mistag_is_dropped() -> None:
    # VICR's 10-K tagged its full year to the third quarter.
    q3 = fact("2025-09-30", 110.0, days=91, form="10-Q", fp="Q3", filed="2025-10-29")
    tagged = fact("2025-09-30", 400.0, days=91, filed="2026-03-02")
    assert screen.quarterly(companyfacts(**{CONTRACT: [q3, tagged]}), CONTRACT) == {
        "2025-09-30": (110.0, "2025-10-29")
    }

    def rows(fourth: float) -> pd.DataFrame:
        sales = [*quarter_facts(2025, 100.0)[:3], fact("2025-12-31", fourth, days=91)]
        facts = companyfacts(
            **{
                INCOME: [*years(2021, 2025, 10.0), *quarter_facts(2025, 5.0)],
                CONTRACT: [*years(2021, 2025, 400.0), *sales],
            }
        )
        return history("X", facts)

    mistag = rows(400.0)
    assert math.isnan(mistag.loc[mistag["end"] == "2025-12-31", "revenue"].iloc[-1])
    assert math.isnan(screen.sec_margins(mistag)["sec_ttm"].iloc[0])
    assert screen.sec_margins(rows(100.0))["sec_ttm"].iloc[0] == pytest.approx(0.05)
    season = screen.sec_margins(rows(244.0))  # H&R Block's 61% tax season
    assert season["sec_ttm"].iloc[0] == pytest.approx(20 / 544)
    assert screen.QUARTER_SHARE == 0.75


def test_quarter_revenue_picks_its_own_concept() -> None:
    def quarter_rows(**revenue: list[dict[str, object]]) -> pd.DataFrame:
        facts = companyfacts(
            **{
                INCOME: [*years(2021, 2025, 10.0), *quarter_facts(2025, 5.0)],
                REVENUES: years(2021, 2025, 100.0),  # the 10-K average's concept
            }
            | revenue
        )
        table = history("X", facts)
        assert screen.sec_margins(table)["sec_ttm"].iloc[0] == pytest.approx(0.25)
        return cast("pd.DataFrame", table[table["period"] == "Q"])

    sales = quarter_facts(2025, 20.0)
    # Quarters only in another concept still count, under that concept.
    assert set(quarter_rows(**{CONTRACT: sales})["concept"]) == {CONTRACT}
    # One concept with all four quarters beats a mix.
    both = quarter_rows(
        **{REVENUES: [*years(2021, 2025, 100.0), *sales[:3]], CONTRACT: sales}
    )
    assert set(both["concept"]) == {CONTRACT}
    # Without one, each quarter takes the first concept that has it.
    mixed = quarter_rows(
        **{REVENUES: [*years(2021, 2025, 100.0), *sales[:3]], CONTRACT: sales[3:]}
    )
    assert mixed["concept"].tolist() == [CONTRACT, REVENUES, REVENUES, REVENUES]


class FakeEdgar:
    def __init__(self, facts: Mapping[str, dict[str, object] | None]) -> None:
        self.data: dict[str, dict[str, object] | None] = dict(facts)
        self.ticker_calls: int = 0
        self.fact_calls: list[str] = []

    def tickers(self) -> pd.DataFrame:
        self.ticker_calls += 1
        names = list(self.data)
        ciks = [f"{i:010d}" for i in range(1, len(names) + 1)]
        return pd.DataFrame({"ticker": names, "cik": ciks})

    def facts(self, cik: str) -> Mapping[str, object] | None:
        self.fact_calls.append(cik)
        return self.data[list(self.data)[int(cik) - 1]]


class YahooMargins(FakeSource):
    """Yahoo's own four-year average, which SEC's replaces when it is full."""

    @override
    def fundamentals(self, ticker: str, *, financial: bool) -> Mapping[str, object]:
        row = dict(super().fundamentals(ticker, financial=financial))
        stale = {"ttm_end": "2026-06-30"} if ticker == "UP3" else {}
        return row | {
            "margin_avg": 0.30,
            "margin_years": 4.0,
            "op_margin": 0.40,
            "ttm_end": "2026-01-31",  # SEC's 2025-12-31 is the same quarter
            **stale,
        }


def test_sec_margins_replace_yahoo_only_with_five_years(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    edgar = FakeEdgar(
        {
            "UP1": companyfacts(
                **{
                    INCOME: [*years(2020, 2025, 10.0), *quarter_facts(2025, 5.0)],
                    REVENUES: [*years(2020, 2025, 100.0), *quarter_facts(2025, 20.0)],
                }
            ),
            "UP2": companyfacts(  # four years: Yahoo's average and TTM
                **{
                    INCOME: [*years(2022, 2025, 10.0), *quarter_facts(2025, 5.0)],
                    REVENUES: [*years(2022, 2025, 100.0), *quarter_facts(2025, 20.0)],
                }
            ),
            "UP3": companyfacts(  # quarters older than Yahoo's newest
                **{
                    INCOME: [*years(2020, 2025, 10.0), *quarter_facts(2025, 5.0)],
                    REVENUES: [*years(2020, 2025, 100.0), *quarter_facts(2025, 20.0)],
                }
            ),
            "DOWN": None,  # no companyfacts
            "FLAT": companyfacts(  # five years, no quarters: Yahoo's TTM
                **{INCOME: years(2021, 2025, 10.0), REVENUES: years(2021, 2025, 100.0)}
            ),
            "BANK": companyfacts(**{INCOME: years(2020, 2025, 10.0)}),
        }
    )

    message = screen.update(
        YahooMargins(), root, NOW, ignore, sleep=ignore, edgar=edgar
    )

    assert "SEC" not in message  # nothing failed
    assert edgar.ticker_calls == 1
    assert len(edgar.fact_calls) == 5  # financials and unknown tickers skip
    saved = pd.read_csv(root / "margins-2026-Q4.csv")
    assert set(saved["ticker"]) == {"UP1", "UP2", "UP3", "FLAT", "DOWN", "OIL", "THIN"}
    assert bool(saved["error"].isna().all())
    inputs = screen.load(root)
    assert inputs is not None
    table = screen.evaluate(inputs).set_index("ticker")
    assert table.loc["UP1", "margin_source"] == "SEC"
    assert table.loc["UP1", "margin_avg"] == pytest.approx(0.10)
    assert table.loc["UP1", "margin_years"] == 5
    assert table.loc["UP1", "op_margin"] == pytest.approx(0.25)
    assert table.loc["UP1", "op_margin_source"] == "SEC"
    assert table.loc["FLAT", "margin_source"] == "SEC"
    assert table.loc["UP3", "margin_source"] == "SEC"
    for ticker in ("UP2", "UP3", "FLAT"):
        assert table.loc[ticker, "op_margin"] == pytest.approx(0.40)
        assert table.loc[ticker, "op_margin_source"] == "야후"
    assert table.loc["UP1", "op_margin_end"] == "2025-12-31"  # SEC's quarters
    assert table.loc["UP2", "op_margin_end"] == "2026-01-31"  # Yahoo's
    assert table.loc["UP3", "op_margin_end"] == "2026-06-30"
    for ticker in ("UP2", "DOWN", "OIL"):
        assert table.loc[ticker, "margin_source"] == "야후"
        assert table.loc[ticker, "margin_avg"] == pytest.approx(0.30)
        assert table.loc[ticker, "margin_years"] == 4
    assert screen.parts(root).margins == "2026-Q4"

    again = FakeEdgar(edgar.data)
    _ = screen.update(
        YahooMargins(),
        root,
        NOW + timedelta(days=40),
        ignore,
        sleep=ignore,
        edgar=again,
    )
    assert again.fact_calls == []  # same quarter
    assert again.ticker_calls == 1  # the CIK list is a month old


def test_an_older_margins_file_is_fetched_again(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    data = {
        "UP1": companyfacts(
            **{INCOME: years(2020, 2025, 10.0), REVENUES: years(2020, 2025, 100.0)}
        )
    }
    _ = screen.update(
        FakeSource(), root, NOW, ignore, sleep=ignore, edgar=FakeEdgar(data)
    )
    path = root / "margins-2026-Q4.csv"
    pd.read_csv(path).drop(columns=["concept"]).to_csv(path, index=False)

    edgar = FakeEdgar(data)
    _ = screen.update(FakeSource(), root, NOW, ignore, sleep=ignore, edgar=edgar)

    assert edgar.fact_calls == ["0000000001"]
    assert edgar.ticker_calls == 0  # the stored CIK list is fresh
    assert "concept" in pd.read_csv(path).columns


def test_an_unreachable_sec_leaves_the_yahoo_average(tmp_path: Path) -> None:
    class Down(FakeEdgar):
        @override
        def tickers(self) -> pd.DataFrame:
            raise httpx.ConnectError("down")

    root = tmp_path / "data" / "screen"
    edgar = Down({})

    message = screen.update(FakeSource(), root, NOW, ignore, sleep=ignore, edgar=edgar)

    assert message.endswith(" · SEC 받지 못함 (ConnectError: down)")
    assert edgar.fact_calls == []
    assert screen.load(root) is not None
    assert screen.parts(root).margins is None


def test_a_refusing_sec_stops_the_stage_and_keeps_what_came(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusing(FakeEdgar):
        @override
        def facts(self, cik: str) -> Mapping[str, object] | None:
            if len(self.fact_calls) == 1:
                self.fact_calls.append(cik)
                raise screen.SecUnavailableError("HTTP 429")
            return super().facts(cik)

    root = tmp_path / "data" / "screen"
    sec = companyfacts(
        **{INCOME: years(2021, 2025, 10.0), REVENUES: years(2021, 2025, 100.0)}
    )
    edgar = Refusing(dict.fromkeys(("UP1", "UP2", "UP3", "FLAT"), sec))

    message = screen.update(FakeSource(), root, NOW, ignore, sleep=ignore, edgar=edgar)

    assert message.endswith(" · SEC 받지 못함 (SecUnavailableError: HTTP 429)")
    assert len(edgar.fact_calls) == 2  # no third request after the refusal
    stored = pd.read_csv(root / "margins-2026-Q4.csv")
    assert set(stored["ticker"]) == {"UP1"}

    statuses = iter([403, 429])

    def answer(request: httpx.Request) -> httpx.Response:
        if "CIK0000000003" in request.url.path:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(next(statuses))

    mocked = partial(httpx.Client, transport=httpx.MockTransport(answer))
    monkeypatch.setattr(httpx, "Client", mocked)
    client = screen.Sec(AGENT, sleep=ignore)
    for cik, reason in (("1", "HTTP 403"), ("2", "HTTP 429"), ("3", "ConnectError")):
        with pytest.raises(screen.SecUnavailableError, match=reason):
            _ = client.facts(f"000000000{cik}")
    client.close()


def test_without_a_user_agent_sec_is_never_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_pauses(monkeypatch)
    built: list[str] = []

    def edgar(agent: str) -> screen.Filings:
        built.append(agent)
        return FakeEdgar({"UP1": None})

    def source(_: Path) -> screen.Source:
        return FakeSource()

    def finish(updater: screen.Updater) -> screen.Progress:
        pause = threading.Event()
        for _ in range(300):
            if not updater.progress.running:
                break
            _ = pause.wait(0.1)
        return updater.progress

    updater = screen.Updater(tmp_path, source=source, edgar=edgar)
    assert updater.start("  ")
    assert not finish(updater).failed
    assert built == []
    assert not list(tmp_path.glob("margins-*"))
    assert not list(tmp_path.glob("ciks-*"))

    assert updater.start(f" {AGENT} ")
    assert not finish(updater).failed
    assert built == [AGENT]
    assert list(tmp_path.glob("margins-*"))
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        _ = screen.Sec(" ")


def test_sec_requests_carry_the_agent_and_pace_themselves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("company_tickers.json"):
            body = {"0": {"cik_str": 320193, "ticker": "BRK.B", "title": "X"}}
            return httpx.Response(200, json=body)
        if "CIK0000000002" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json=companyfacts())

    mocked = partial(httpx.Client, transport=httpx.MockTransport(answer))
    monkeypatch.setattr(httpx, "Client", mocked)
    waits: list[float] = []
    sec = screen.Sec(AGENT, sleep=waits.append)

    assert sec.tickers().to_dict(orient="records") == [
        {"ticker": "BRK-B", "cik": "0000320193"}
    ]
    assert sec.facts("0000000001") == companyfacts()
    assert sec.facts("0000000002") is None
    sec.close()

    assert [str(r.url) for r in seen] == [
        screen.SEC_TICKERS_URL,
        screen.COMPANYFACTS_URL.format(cik="0000000001"),
        screen.COMPANYFACTS_URL.format(cik="0000000002"),
    ]
    assert {r.headers["User-Agent"] for r in seen} == {AGENT}
    assert waits == [screen.SEC_PAUSE] * 3
    assert 1 / screen.SEC_PAUSE <= 10  # SEC's limit, with room for the 13F page


# ---------------------------------------------------------------- page


def page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The screening page in demo mode, with data under ``tmp_path``."""
    settings = tmp_path / "finagent.settings.json"
    monkeypatch.setattr(sys, "argv", ["app.py", "--demo", f"--settings={settings}"])
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at._page_hash = calc_hash("screening")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    return at.run()


def click(at: AppTest, column: str, index: int = 0) -> None:
    """Stand in for a click on a result cell, which AppTest cannot send."""
    key = f"screen_table_{at.session_state.get('screen_gen', 0)}"
    at.session_state[key] = {
        "selection": {"rows": [], "columns": [], "cells": [[index, column]]}
    }
    _ = at.run()


def test_the_page_shows_groups_opens_a_stock_and_switches_universes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = page(tmp_path, monkeypatch)
    assert not at.exception
    assert at.button(key="screen_refresh").disabled  # demo: nothing fetched
    assert not at.dataframe

    root = tmp_path / screen.SNAPSHOT_DIR
    _ = run_update(FakeSource(), root)
    inputs = screen.load(root)
    assert inputs is not None
    table = screen.evaluate(inputs)
    _ = at.run()  # saved files are picked up

    assert not at.exception
    assert "시세 2026-10-22" in " ".join(c.value for c in at.caption)
    assert len(at.dataframe[0].value) == 7  # the liquid stocks
    assert at.dataframe[0].proto.placeholder == "—"  # NaN cells show a dash
    group = at.segmented_control(key="screen_group")
    assert group.options[1:] == [
        f"{screen.GROUP_NAMES[key]} · {screen.counts(screen.liquid(table))[key]}"
        for key in screen.GROUPS
    ]
    assert len(at.get("vega_lite_chart")) == 1  # the map's scatter

    _ = group.set_value("pricey").run()
    assert not at.exception
    shown = at.dataframe[0].value["ticker"].tolist()
    assert shown == screen.pick(table, "pricey")["ticker"].tolist()
    assert len(shown) == 3

    _ = at.segmented_control(key="screen_group").set_value("all").run()
    _ = at.multiselect(key="screen_sectors").set_value(["Financials"]).run()
    assert at.dataframe[0].value["ticker"].tolist() == ["BANK"]

    click(at, "sector")  # only the ticker and name cells open a stock
    assert len(at.get("vega_lite_chart")) == 1

    click(at, "name")
    assert not at.exception
    assert len(at.get("vega_lite_chart")) == 2  # the map and the stock

    _ = at.run()  # an unrelated rerun does not bring the dialog back
    assert len(at.get("vega_lite_chart")) == 1

    click(at, "ticker")  # the same row again opens it again
    assert len(at.get("vega_lite_chart")) == 2

    _ = at.segmented_control(key="screen_universe").set_value("sp400").run()
    assert not at.exception
    assert not at.dataframe  # nothing saved for the S&P 400 yet
    midcap = root / screen.UNIVERSES["sp400"].folder
    _ = screen.update(
        FakeSource(), midcap, NOW, ignore, sleep=ignore, edgar=FakeEdgar({})
    )
    _ = at.run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 7
    assert "SEC 2026-Q4" in " ".join(c.value for c in at.caption)


def test_the_criteria_panel_saves_and_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = run_update(FakeSource(), tmp_path / screen.SNAPSHOT_DIR)
    at = page(tmp_path, monkeypatch)
    settings = tmp_path / "finagent.settings.json"
    assert not at.exception
    assert at.expander[0].label == "기준"
    assert at.button(key="crit_reset").disabled
    assert len(at.dataframe[0].value) == 7

    _ = at.checkbox(key="crit_on_illiquid").uncheck().run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 8  # THIN is in
    assert at.expander[0].label == "기준 · 변경 1"
    assert '"illiquid"' in settings.read_text("utf-8")
    assert at.number_input(key="crit_illiquid_share").disabled
    _ = at.checkbox(key="crit_on_t_above200").uncheck().run()
    click(at, "name")  # the stock dialog with a check off
    assert not at.exception
    assert any(m.value.endswith(("/9 강", "/9 약", "/9 ")) for m in at.metric)

    _ = at.number_input(key="crit_split").set_value(60.0).run()
    assert not at.exception
    saved = settings.read_text("utf-8")
    assert '"split": 0.6' in saved

    _ = at.button(key="crit_reset").click().run()
    assert not at.exception
    assert at.expander[0].label == "기준"
    assert len(at.dataframe[0].value) == 7
    assert '"screen"' not in settings.read_text("utf-8")
