"""The S&P 500 screener without the network: fake sources and synthetic prices."""

# pandas ships no complete type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import json
import math
import threading
from dataclasses import fields
from datetime import UTC, date, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest

from finagent import screen

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# 2026-10-22 18:00 in New York: the day's bar is final.
NOW = datetime(2026, 10, 22, 22, 0, tzinfo=UTC)
DAY = date(2026, 10, 22)


def ignore(*_: object) -> None:
    pass


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
    # fcf, roe, book_value, trailing_eps; gross profit and operating income
    # scale with roe.
    "UP1": {"fcf": 5e9, "roe": 0.30, "book_value": 10.0, "trailing_eps": 1.5},
    "UP2": {"fcf": 3e9, "roe": 0.20, "book_value": 10.0, "trailing_eps": 2.0},
    "UP3": {"fcf": 1e9, "roe": 0.10, "book_value": 10.0, "trailing_eps": 4.0},
    "FLAT": {"fcf": 1e9, "roe": 0.25, "book_value": 10.0, "trailing_eps": 5.0},
    "DOWN": {"fcf": -2e9, "roe": 0.05, "book_value": 10.0, "trailing_eps": 3.0},
    "BANK": {"roe": 0.12, "book_value": 50.0, "trailing_eps": 4.0},
    "OIL": {"fcf": 2e9, "roe": 0.15, "book_value": 30.0, "trailing_eps": 6.0},
    "THIN": {"fcf": 1e8, "roe": 0.05, "book_value": 5.0, "trailing_eps": 5.0},
}


def fundamentals(ticker: str) -> dict[str, object]:
    base: dict[str, object] = dict.fromkeys(screen.FUNDAMENTAL_COLUMNS[1:-1], math.nan)
    base |= FUNDAMENTALS[ticker]
    roe = cast("float", base["roe"])
    fcf = cast("float", base["fcf"])
    base |= {
        "ocf": fcf + 1e9,
        "total_debt": 2e9,
        "total_cash": 1e9,
        "ebitda": 2e9,
        "nd_ebitda": 0.5,
        "ev": 1.01e11,
        "mcap": 1e11,
        "ev_ebitda": 50.5,
        "short_pct": 0.02,
        "revenue": 110.0,
        "revenue_prior": 100.0,
        "rev_growth": 0.10,
        "rev_year": "2025-12-31",
        "margin_avg": 0.15,
        "margin_years": 5,
        "op_margin": 0.15,
        "revenue_ttm": 1e10,
        "gross_profit": roe * 2e10,
        "op_income_ttm": roe * 1e10,
        "ebit": roe * 1e10,
        "interest": 1e8,
        "net_income": roe * 5e9,
        "wc_change": 0.0,
        "total_assets": 5e10,
        **{f"fcf_y{year}": fcf for year in range(3)},
        **{f"gpa_y{year}": roe / 5 - 0.002 * year for year in range(4)},
        "last_fy_end": "2025-12-31",
        "next_fy_end": "2026-12-31",
        "currency": "USD",
    }
    if ticker == "OIL":
        base |= {"op_margin": 0.30}  # twice its 5-year median: peak earnings
    if ticker == "BANK":
        base |= {
            "fcf": math.nan,
            "ocf": math.nan,
            "eq_ratio": 0.08,
            "tce_ratio": 0.06,
            "tbv": 4e10,
            "shares": 1e9,
            "ebitda": math.nan,
            "nd_ebitda": math.nan,
            "ev": math.nan,
        }
    return base


def estimates(_: str) -> dict[str, object]:
    return {
        "fy1_now": 5.0,
        "fy1_30d": 4.9,
        "fy1_90d": 4.8,
        "fy1_up30": 3.0,
        "fy1_down30": 1.0,
        "fy1_n": 10.0,
        "fy2_now": 6.0,
        "fy2_30d": 5.9,
        "fy2_90d": 5.5,
        "fy2_up30": 4.0,
        "fy2_down30": 1.0,
        "fy2_n": 9.0,
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
        self.fundamental_calls: list[str] = []
        self.estimate_calls: list[str] = []

    def constituents(self) -> pd.DataFrame:
        self.constituent_calls += 1
        return UNIVERSE.copy()

    def prices(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        self.price_calls.append((tuple(tickers), start, end))
        frames = [
            ohlcv(t, start, end, GROWTH.get(t, 0.0), 1e5 if t == "THIN" else 1e6)
            for t in tickers
        ]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fail(self, key: str) -> None:
        pending = self.failures.get(key)
        if pending:
            raise pending.pop(0)

    def fundamentals(self, ticker: str) -> Mapping[str, object]:
        self.fundamental_calls.append(ticker)
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


def yearly(**rows: list[float]) -> pd.DataFrame:
    """A yearly statement, newest year (2025) first."""
    width = len(next(iter(rows.values())))
    years = [pd.Timestamp(f"{2025 - i}-12-31") for i in range(width)]
    return pd.DataFrame(rows, index=years).T


def ttm(**rows: float) -> pd.DataFrame:
    """A trailing (TTM) statement: one column, the newest quarter's end."""
    return pd.DataFrame(
        {pd.Timestamp("2026-06-30"): list(rows.values())}, index=list(rows)
    )


def test_fundamentals_row_reads_info_and_statements() -> None:
    info = {
        "freeCashflow": 1e9,
        "operatingCashflow": 3e9,
        "returnOnEquity": 0.2,
        "totalDebt": 5e9,
        "totalCash": 1e9,
        "ebitda": 2e9,
        "enterpriseValue": 4e10,
        "trailingEps": 4.5,
        "forwardPE": "Infinity",
        "shortPercentOfFloat": 0.031,
        "financialCurrency": "USD",
        "lastFiscalYearEnd": 1767139200,  # 2025-12-31
        "nextFiscalYearEnd": 1798675200,  # 2026-12-31
    }
    balance = pd.DataFrame(
        {pd.Timestamp("2026-06-30"): [10.0, 100.0, 7.0, 4.0, 2.0]},
        index=[
            "StockholdersEquity",
            "TotalAssets",
            "TangibleBookValue",
            "GoodwillAndOtherIntangibleAssets",
            "OrdinarySharesNumber",
        ],
    )
    statements = screen.Statements(
        income=yearly(
            TotalRevenue=[90.0, 100.0],
            OperatingIncome=[9.0, 20.0],
            GrossProfit=[36.0, math.nan],
            CostOfRevenue=[54.0, 60.0],
        ),
        trailing=ttm(GrossProfit=40.0, OperatingIncome=12.0, InterestExpense=3.0),
        cash=ttm(ChangeInWorkingCapital=-5.0),
        cash_yearly=yearly(FreeCashFlow=[1.0, -2.0, 3.0]),
        balance=balance,
        balance_yearly=yearly(TotalAssets=[90.0, 80.0]),
    )

    row = screen.fundamentals_row(info, statements)

    assert row["nd_ebitda"] == 2.0
    assert row["rev_growth"] == pytest.approx(-0.1)
    assert row["rev_year"] == "2025-12-31"
    assert row["eq_ratio"] == 0.1
    assert row["tce_ratio"] == pytest.approx(7 / 96)
    assert row["short_pct"] == 0.031
    assert math.isnan(cast("float", row["fpe"]))
    assert row["currency"] == "USD"
    assert (row["ocf"], row["ev"], row["trailing_eps"]) == (3e9, 4e10, 4.5)
    assert (row["gross_profit"], row["op_income_ttm"], row["interest"]) == (
        40.0,
        12.0,
        3.0,
    )
    assert row["wc_change"] == -5.0  # cash-flow signed: working capital grew
    assert (row["total_assets"], row["tbv"], row["shares"]) == (100.0, 7.0, 2.0)
    assert [row[f"fcf_y{i}"] for i in range(3)] == [1.0, -2.0, 3.0]
    assert math.isnan(cast("float", row["fcf_y3"]))
    assert row["gpa_y0"] == pytest.approx(36 / 90)  # the same year's assets
    assert row["gpa_y1"] == pytest.approx((100 - 60) / 80)  # no gross profit row
    assert (row["last_fy_end"], row["next_fy_end"]) == ("2025-12-31", "2026-12-31")
    assert set(row) == set(screen.FUNDAMENTAL_COLUMNS) - {"ticker", "error"}

    bare = screen.fundamentals_row({"ebitda": -1.0}, screen.Statements(pd.DataFrame()))
    assert math.isnan(cast("float", bare["nd_ebitda"]))
    assert math.isnan(cast("float", bare["rev_growth"]))
    assert math.isnan(cast("float", bare["eq_ratio"]))
    assert math.isnan(cast("float", bare["gpa_y0"]))
    revenue_only = screen.fundamentals_row(
        {},
        screen.Statements(
            income=yearly(TotalRevenue=[90.0]),
            balance_yearly=yearly(TotalAssets=[90.0]),
        ),
    )
    assert math.isnan(cast("float", revenue_only["gpa_y0"]))  # not revenue less 0
    assert bare["last_fy_end"] == ""


def test_sec_gross_profit_over_assets_and_its_change() -> None:
    years = pd.DataFrame(
        {
            "end": ["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"],
            "revenue": [100.0, 90.0, 80.0, 70.0],
            "op_income": [10.0, 9.0, 8.0, 7.0],
            "gross_profit": [50.0, math.nan, 32.0, 21.0],
            "cost_of_revenue": [math.nan, 54.0, math.nan, math.nan],
            "assets": [100.0, 100.0, 100.0, 100.0],
        }
    ).assign(**dict.fromkeys(("net_income", "ocf", "capex", "equity"), math.nan))

    row = screen.sec_row(years, pd.DataFrame(columns=["end"]))

    assert row["sec_gpa_y0"] == 0.5
    assert row["sec_gpa_y1"] == pytest.approx(0.36)  # revenue less cost of revenue
    assert row["sec_gpa_y3"] == pytest.approx(0.21)
    table = pd.DataFrame([row]).assign(
        **{f"gpa_y{year}": math.nan for year in range(screen.ANNUAL_YEARS)}
    )
    change = screen.improvement(table, 3).iloc[0]
    assert change["improve"] == pytest.approx(0.5 - 0.21)
    assert (change["improve_years"], change["improve_source"]) == (3, "SEC")


# ---------------------------------------------------------------- indicators


def test_tech_row_needs_the_minimum_history() -> None:
    short = ohlcv("UP1", date(2026, 1, 1), DAY, 0.002)
    row = screen.tech_row(short)

    assert row["bars"] < screen.MIN_BARS
    assert set(row) == {"bars", "price", "dollar_vol20"}

    full = screen.tech_row(ohlcv("UP1", date(2024, 9, 1), DAY, 0.004))
    assert full["close"] > full["sma21"] > full["sma200"]
    assert full["ma_dist"] == pytest.approx(full["sma21"] / full["sma200"])
    assert full["near_high"] == pytest.approx(full["from_high"] + 1)
    assert full["near_high"] <= 1
    assert full["mom12_1"] > 0


def test_tech_row_follows_the_criteria_windows() -> None:
    frame = ohlcv("UP1", date(2025, 9, 1), DAY, 0.002)
    close = frame["adj_close"].to_numpy()
    c = screen.criteria(
        {
            "mom_bars": 126,
            "skip_bars": 0,
            "high_bars": 63,
            "ma_short": 5,
            "ma_long": 300,
            "surge_bars": 10,
        },
        [],
    )
    row = screen.tech_row(frame, c)

    assert len(frame) < 300
    assert row["mom12_1"] == pytest.approx(close[-1] / close[-127] - 1)
    assert row["near_high"] == pytest.approx(close[-1] / (close[-63:] * 1.01).max())
    assert row["ma_short"] == pytest.approx(close[-5:].mean())
    assert math.isnan(row["ma_long"])  # fewer bars than the long average
    assert math.isnan(row["ma_dist"])
    assert row["surge_ret"] == pytest.approx(close[-1] / close[-11] - 1)
    longer = screen.tech_row(frame, screen.criteria({"mom_bars": 504}, []))
    assert math.isnan(longer["mom12_1"])  # short of 504 + 21 bars


def test_momentum_crash_follows_the_criteria_windows() -> None:
    prices = pd.concat(
        [
            ohlcv("AAA", date(2025, 6, 2), DAY, -0.001),
            ohlcv("BBB", date(2025, 6, 2), DAY, -0.002),
        ]
    )
    caps = pd.DataFrame({"ticker": ["AAA", "BBB"], "mcap": [1e10, 3e10]})

    assert screen.momentum_crash(prices, caps).state == screen.NOT_COMPUTED
    c = screen.criteria({"bear_bars": 252, "variance_bars": 21}, [])
    crash = screen.momentum_crash(prices, caps, c)
    assert crash.state in {screen.CRASH_WATCH, screen.CRASH_CLEAR}
    assert crash.market_return < 0


def test_every_criteria_number_has_one_input() -> None:
    inputs = [x.field for r in screen.RULES for x in r.limits]

    assert sorted(inputs) == sorted(
        f.name for f in fields(screen.Criteria) if f.name != "off"
    )
    assert screen.windows(screen.criteria({"surge_bars": 10, "split": 0.6}, [])) == (
        screen.criteria({"surge_bars": 10}, [])
    )


# ---------------------------------------------------------------- fundamentals


def frame(**overrides: object) -> dict[str, object]:
    """A liquid non-financial that passes every check and has every measure."""
    row: dict[str, object] = {
        "ticker": "X",
        "sector": "Information Technology",
        "industry": "Software",
        "peer": "Information Technology",
        "fcf": 1.0,
        "ocf": 2.0,
        "roe": 0.2,
        "book_value": 10.0,
        "eq_ratio": math.nan,
        "tce_ratio": math.nan,
        "tbv": math.nan,
        "shares": math.nan,
        "revenue_prior": 100.0,
        "rev_growth": 0.1,
        "total_debt": 5.0,
        "total_cash": 1.0,
        "ebitda": 2.0,
        "nd_ebitda": 2.0,
        "ev": 100.0,
        "ev_ebitda": 50.0,
        "trailing_eps": 2.0,
        "fwd_eps": 2.2,
        "op_margin": 0.2,
        "margin_avg": 0.2,
        "margin_years": 4,
        "margin_spread": 0.02,
        "long_med": 0.2,
        "long_mad": 0.02,
        "long_std": math.nan,  # no margin swings to rank
        "long_years": 10,
        "op_income_ttm": 2.0,
        "ebit": 2.0,
        "interest": 0.5,
        "sec_interest": math.nan,
        "sec_interest_paid": math.nan,
        "rnd": 0.5,
        "wc_change": -0.2,
        "gross_profit": 5.0,
        "revenue_ttm": 10.0,
        "cost_of_revenue": 5.0,
        "total_assets": 20.0,
        "net_income": 1.0,
        "improve": 0.01,
        "improve_years": 3,
        "sec_roe_std": math.nan,
        "revenue": 10.0,
        "price": 50.0,
        **{f"fcf_y{year}": 1.0 for year in range(screen.ANNUAL_YEARS)},
        **{f"sec_fcf_y{year}": math.nan for year in range(screen.ANNUAL_YEARS)},
    }
    return row | overrides


def status(
    *rows: dict[str, object], c: screen.Criteria = screen.DEFAULT
) -> pd.DataFrame:
    return screen.fundamentals_status(pd.DataFrame(list(rows)), c)


def years_of(values: Sequence[float], prefix: str = "fcf_y") -> dict[str, float]:
    padded = [*values, *[math.nan] * (screen.ANNUAL_YEARS - len(values))]
    return {f"{prefix}{year}": value for year, value in enumerate(padded)}


def test_cash_flow_disqualifier() -> None:
    nan = math.nan
    insurer = {"sector": "Financials", "industry": "Property & Casualty Insurance"}
    result = status(
        frame(tce_ratio=0.01, eq_ratio=0.01),  # capital ratios: financials only
        frame(ocf=-1.0),
        frame(**years_of([5.0, -1.0, -5.0])),  # 3-year FCF -1: not a disqualifier
        frame(**years_of([5.0, -1.0, -5.0, 100.0])),  # the fourth year is out
        frame(ocf=nan),
        frame(**years_of([]), fcf=nan),
        frame(**years_of([-9.0]), fcf=2.0),  # one year: the TTM FCF stands in
        frame(ocf=-1.0, sector="Real Estate", peer="Real Estate"),
        frame(**insurer, eq_ratio=0.01, ocf=nan),  # no capital ratio test
        frame(**insurer, roe=-0.01, ocf=nan),
    )

    assert screen.FCF_YEARS == 3
    assert result["dq_cash"].tolist() == [
        "통과",
        "결격",
        "통과",
        "통과",
        "미확인",
        "통과",
        "통과",
        "결격",
        "통과",
        "결격",
    ]
    assert result["dq_state"].tolist()[:2] == ["통과", "결격"]
    assert (result.loc[2, "fcf_cum"], result.loc[3, "fcf_cum"]) == (-1.0, -1.0)
    assert result.loc[6, "fcf_source"] == "TTM"


BANK = {
    "sector": "Financials",
    "industry": "Regional Banks",
    "peer": "Banks",
    "fcf": math.nan,
    "ocf": math.nan,
}


def test_net_debt_disqualifier() -> None:
    result = status(
        frame(nd_ebitda=2.0),
        frame(nd_ebitda=3.5),
        frame(nd_ebitda=4.5),
        frame(total_cash=10.0, nd_ebitda=-2.5),
        frame(ebitda=-1.0, nd_ebitda=math.nan),  # debt with no EBITDA
        frame(ebitda=-1.0, nd_ebitda=math.nan, total_cash=10.0),  # but net cash
        frame(total_debt=math.nan, nd_ebitda=math.nan),
    )

    assert result["nd_limit"].iloc[0] == pytest.approx(5.5)  # median 2.75 x 2
    assert result["dq_debt"].tolist() == [
        "통과",
        "통과",
        "통과",
        "순현금",
        "결격",
        "순현금",
        "미확인",
    ]
    floor = status(frame(nd_ebitda=1.0), frame(nd_ebitda=1.0), frame(nd_ebitda=4.1))
    assert floor["nd_limit"].iloc[0] == 4.0  # max(1.0 x 2, 4.0)
    assert floor["dq_debt"].tolist() == ["통과", "통과", "결격"]
    cap = status(frame(nd_ebitda=5.0), frame(nd_ebitda=5.0), frame(nd_ebitda=6.5))
    assert cap["nd_limit"].iloc[0] == 6.0  # min(5.0 x 2, 6.0)
    assert cap["dq_debt"].tolist() == ["통과", "통과", "결격"]


# ---------------------------------------------------------------- 보류


OIL = {"sector": "Energy", "industry": "Integrated Oil & Gas", "peer": "Energy"}


def test_peak_earnings_needs_the_margin_and_a_flat_forward_eps() -> None:
    margins = {
        "long_years": 10,
        "margin_years": 5,
        "margin_avg": 0.10,
        "long_med": 0.10,
        "long_mad": 0.01,
        "long_std": 0.015,
        "trailing_eps": 5.0,
    }
    rows = [
        OIL | margins | {"op_margin": 0.14, "fwd_eps": 5.0},  # z 2.7, +4%p
        OIL | margins | {"op_margin": 0.14, "fwd_eps": 6.0},  # EPS still growing
        OIL | margins | {"op_margin": 0.13, "fwd_eps": 5.0},  # z 2.0: below 2.5
        OIL
        | margins
        | {
            "op_margin": 0.14,
            "fwd_eps": 5.0,
            "sector": "Health Care",
            "industry": "Health Care Equipment",
            "long_std": 0.005,  # the steadiest: not cyclical by its margins
        },
    ]

    flags = screen.hold_flags(pd.DataFrame(rows))

    assert flags["peak"].tolist() == [True, False, False, False]
    assert flags["tailwind"].tolist() == [False, True, False, False]
    assert flags["margin_jump"].tolist() == [False, False, False, True]


# ---------------------------------------------------------------- 퀄리티 and 밸류에이션


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
        cell_row(tailwind=True),  # a label, not a hold
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
        "prime",
        "",
    ]


# ---------------------------------------------------------------- estimates


def year_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "as_of": "2026-10-22",
        "last_fy_end": "2025-12-31",
        "next_fy_end": "2026-12-31",
        "sec_fy_end": "",
        "sec_filed": "",
    }
    return row | overrides


def test_the_fiscal_year_state() -> None:
    table = pd.DataFrame(
        [
            year_row(),  # 70 days left of FY1
            year_row(last_fy_end="2025-09-27", next_fy_end="2026-09-26"),  # ended
            year_row(last_fy_end="2026-08-31", next_fy_end="2027-08-31"),  # no SEC
            year_row(  # ended 114 days ago, its 10-K 42
                last_fy_end="2026-06-30",
                next_fy_end="2027-06-30",
                sec_fy_end="2026-06-27",  # a 52-week year
                sec_filed="2026-09-10",
            ),
            year_row(  # its 10-K 155 days ago
                last_fy_end="2026-03-31",
                next_fy_end="2027-03-31",
                sec_fy_end="2026-03-31",
                sec_filed="2026-05-20",
            ),
            year_row(  # SEC still has the year before: the year end decides
                last_fy_end="2026-08-31",
                next_fy_end="2027-08-31",
                sec_fy_end="2025-08-31",
                sec_filed="2025-10-15",
            ),
            year_row(last_fy_end="", next_fy_end=""),
        ]
    )

    state = screen.year_state(table)

    assert state["weight"].tolist() == pytest.approx(
        [70 / 365.25, 0.0, 313 / 365.25, 251 / 365.25, 160 / 365.25, 313 / 365.25, 1.0]
    )
    assert state["pending"].tolist() == [False, True] + [False] * 5
    # Since that year's 10-K, else since the year end.
    assert state["turn_days"].tolist()[:6] == [295, 390, 52, 42, 155, 52]
    assert math.isnan(state["turn_days"].iloc[6])


def test_forward_eps_blends_the_two_years() -> None:
    table = pd.DataFrame(
        {
            "weight": [0.25, 0.0, 0.5, 0.5],
            "fy1_now": [4.0, 4.0, 4.0, math.nan],
            "fy2_now": [8.0, 8.0, math.nan, 8.0],
        }
    )

    assert screen.forward_eps(table).tolist() == [7.0, 8.0, 4.0, 8.0]


# ---------------------------------------------------------------- whole run


def test_evaluate_on_stored_data(tmp_path: Path) -> None:
    table = stored(tmp_path)
    by = table.set_index("ticker")

    assert set(table["ticker"]) == set(UNIVERSE["ticker"])
    assert (by["liquidity"] == "").all()  # no dollar volume cut by default
    cut = screen.liquidity(table, screen.criteria({"illiquid_share": 0.05}, []))
    assert table.loc[cut != "", "ticker"].tolist() == ["THIN"]
    kept = screen.liquid(table)
    assert kept["score"].between(0, 100).all()
    assert by.loc["UP1", "score"] > by.loc["DOWN", "score"]
    assert by.loc["DOWN", "dq_cash"] == "결격"  # negative operating cash flow
    assert by.loc["BANK", "dq_debt"] == "해당 없음"
    assert by.loc["BANK", "peer"] == "Financials"  # one bank: the sector
    assert by.loc["BANK", "dq_cash"] == "통과"  # TCE/TA 6%
    assert bool(by.loc["OIL", "peak"])  # cyclical, margin twice its median, EPS flat
    assert not bool(by.loc["OIL", "tailwind"])
    assert by.loc["OIL", "group"] == "hold"
    assert by.loc["DOWN", "group"] == ""  # 결격
    assert by.loc["BANK", "v_book"] == 0.5  # alone in its group: the median
    assert math.isnan(by.loc["BANK", "v_ebitda"])
    assert set(by.loc[list(kept["ticker"]), "est"]) == {"상향"}
    # 70 of 365 days left in FY1: forward EPS leans on FY2.
    w = 70 / 365.25
    assert by.loc["UP1", "fwd_eps"] == pytest.approx(w * 5.0 + (1 - w) * 6.0)
    assert screen.failures(table).empty
    assert all(
        screen.group(kept, key).shape[0] <= screen.TOP_N for key in screen.GROUPS
    )
    points = screen.map_rows(kept)
    both = kept[kept["quality"].notna() & kept["valuation"].notna()]
    assert len(points) == len(both[both["dq_state"] == screen.CLEAR])
    assert "DOWN" not in {point["ticker"] for point in points}
    missing = kept.assign(rev_strength=math.nan, score=math.nan)
    _ = json.dumps(screen.map_rows(missing), allow_nan=False)  # the chart's JSON


# ---------------------------------------------------------------- criteria


def test_saved_criteria_are_checked() -> None:
    c = screen.criteria(
        {
            "split": 0.6,
            "min_bars": 300,
            "bank_tce_floor": 0.045,
            "peer_min": 20,
            "quality_min": 2.5,  # not whole
            "illiquid_share": 0.9,  # above its range
            "min_price": True,  # a bool is not a number
            "weak_share": math.nan,
            "gone": 1.0,
        },
        ["q_roe", "split", "gone", "s_max"],
    )

    assert (c.split, c.min_bars, c.bank_tce_floor, c.peer_min) == (0.6, 300, 0.045, 20)
    assert (c.quality_min, c.illiquid_share, c.min_price, c.weak_share) == (
        screen.QUALITY_MIN,
        screen.ILLIQUID_SHARE,
        screen.MIN_PRICE,
        screen.WEAK_SHARE,
    )
    assert c.off == ("q_roe", "s_max")  # a boundary cannot be off
    assert screen.changed(c) == [
        "min_bars",
        "peer_min",
        "dq_cash",
        "q_roe",
        "split",
        "s_max",
    ]
    values, off = screen.saved(c)
    assert screen.criteria(values, off) == c
    share = next(
        x for r in screen.RULES for x in r.limits if x.field == "cyclical_share"
    )
    assert share.stored(share.shown(1 / 3)) == 1 / 3  # not 0.33333333
    assert share.stored(33.0) == 0.33
    assert {rule.key for rule in screen.RULES} >= set(screen.TECH_ITEMS)


# ---------------------------------------------------------------- storage


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
    assert "BANK" in source.fundamental_calls
    assert "UP1" in source.fundamental_calls
    _, start, end = source.price_calls[0]
    assert start == DAY - timedelta(days=screen.HISTORY_DAYS)
    assert end == DAY + timedelta(days=1)
    assert screen.parts(root) == screen.Parts(DAY, DAY, "2026-Q4", DAY)


def test_update_keeps_pricing_the_stocks_of_recent_snapshots(tmp_path: Path) -> None:
    root = tmp_path / "data" / "screen"
    folder = root / screen.SNAPSHOT_FOLDER
    folder.mkdir(parents=True)
    _ = (folder / "2026-09-01.csv").write_text("ticker,group\nGONE,prime\n")
    _ = (folder / "2025-01-02.csv").write_text("ticker,group\nOLD,prime\n")

    message = run_update(FakeSource(), root)

    priced = set(screen.read_prices(root)["ticker"])
    assert "GONE" in priced  # left the index after the snapshot
    assert "OLD" not in priced  # past SNAPSHOT_REACH_DAYS
    assert "시세 받은 종목 8" in message  # the index's own


def test_snapshot_keeps_every_criteria_number(tmp_path: Path) -> None:
    table = stored(tmp_path)

    path = screen.write_snapshot(tmp_path, table, screen.DEFAULT, DAY)

    meta = json.loads(path.with_suffix(".json").read_text("utf-8"))
    assert meta["values"]["weak_share"] == screen.WEAK_SHARE  # a default too
    assert meta["as_of"] == "2026-10-22"
    assert "sector" in pd.read_csv(path).columns


def snapshot(folder: Path, day: str, as_of: str, rows: str) -> None:
    _ = (folder / f"{day}.csv").write_text(f"ticker,sector,group\n{rows}")
    _ = (folder / f"{day}.json").write_text(f'{{"as_of": "{as_of}"}}')


def test_snapshot_check_against_the_stocks_screened(tmp_path: Path) -> None:
    days = pd.bdate_range("2025-01-01", periods=300)
    steps = np.arange(len(days), dtype="float64")
    paths = {"A": 100 * 1.001**steps, "B": np.full(len(days), 100.0)}
    prices = pd.concat(
        [
            *(
                pd.DataFrame({"date": days, "ticker": t, "adj_close": v})
                for t, v in paths.items()
            ),
            pd.DataFrame({"date": days[:10], "ticker": "C", "adj_close": 50.0}),
        ],
        ignore_index=True,
    )
    folder = tmp_path / screen.SNAPSHOT_FOLDER
    folder.mkdir()
    labels = cast("list[str]", days.strftime("%Y-%m-%d").tolist())
    first = labels[0]
    snapshot(folder, first, first, "A,Tech,prime\nB,Tech,weak\nC,Energy,prime\n")
    late = labels[290]
    snapshot(folder, late, late, "A,Tech,prime\n")  # no horizon reached yet
    snapshot(folder, "2024-06-03", "2024-06-03", "A,Tech,prime\n")  # not stored

    check = screen.snapshot_check(tmp_path, prices)

    assert set(check["day"]) == {first}
    assert len(check) == len(screen.HORIZONS) * len(screen.SNAPSHOT_GROUPS)
    month = check[check["horizon"] == "1개월"].set_index("group")
    gain = 1.001**21 - 1
    assert (month.loc["prime", "stocks"], month.loc["prime", "priced"]) == (2, 1)
    assert month.loc["prime", "excess"] == pytest.approx(gain / 2)  # C: no price
    assert month.loc["weak", "excess"] == pytest.approx(-gain / 2)
    assert month.loc["prime", "sector_excess"] == pytest.approx(gain / 2)
    assert (month.loc["prime", "beat"], month.loc["weak", "beat"]) == (1.0, 0.0)
    assert month.loc["trap", "stocks"] == 0

    summary = screen.check_summary(check)
    assert list(summary["horizon"].unique()) == list(screen.HORIZONS)
    row = summary.iloc[0]
    assert (row["horizon"], row["group"]) == ("1개월", "검토 우선")
    assert (row["snapshots"], row["stocks"], row["missing"]) == (1, 2.0, 0.5)
    assert "밸류 트랩 점검" not in set(summary["group"])  # no stock in it


def no_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Updater's run without its real pauses between requests."""
    monkeypatch.setattr(screen, "update", partial(screen.update, sleep=ignore))


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
