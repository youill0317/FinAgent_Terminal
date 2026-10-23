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
from typing import TYPE_CHECKING, cast, override

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


def test_wikipedia_symbols_become_yahoo_tickers() -> None:
    assert screen.yahoo_symbol("BRK.B") == "BRK-B"
    assert screen.yahoo_symbol(" bf.b ") == "BF-B"


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


def test_an_empty_yahoo_answer_is_a_failure_to_retry() -> None:
    table = pd.DataFrame({"x": [1.0]})

    assert screen.nonempty(table, "표") is table
    with pytest.raises(screen.EmptyAnswerError, match="표 빈 응답"):
        _ = screen.nonempty(pd.DataFrame(), "표")
    with pytest.raises(screen.EmptyAnswerError):
        _ = screen.nonempty(None, "표")


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


def test_macd_of_a_flat_line_is_zero() -> None:
    line, signal = screen.macd(series([10.0] * 40))

    assert line.iloc[-1] == 0.0
    assert signal.iloc[-1] == 0.0
    rising, _ = screen.macd(series(range(40)))
    assert rising.iloc[-1] > 0


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


# Every Financials sub-industry on the Wikipedia list (2026-10-22).
WIKIPEDIA_FINANCIALS = {
    "Asset Management & Custody Banks": "Financial Services",
    "Transaction & Payment Processing Services": "Financial Services",
    "Financial Exchanges & Data": "Financial Services",
    "Property & Casualty Insurance": "Insurance",
    "Diversified Banks": "Banks",
    "Insurance Brokers": "Insurance",
    "Investment Banking & Brokerage": "Financial Services",
    "Regional Banks": "Banks",
    "Life & Health Insurance": "Insurance",
    "Consumer Finance": "Financial Services",
    "Multi-line Insurance": "Insurance",
    "Multi-Sector Holdings": "Financial Services",
    "Reinsurance": "Insurance",
}


def test_financial_sub_industries_map_to_industry_groups() -> None:
    for industry, group in WIKIPEDIA_FINANCIALS.items():
        assert screen.FINANCIAL_GROUPS[industry] == group, industry


def test_peer_groups_fall_back_to_the_sector() -> None:
    table = pd.DataFrame(
        {
            "industry": ["Software"] * 10 + ["Semiconductors"] * 3,
            "sector": ["Information Technology"] * 13,
        }
    )

    assert (
        screen.peers(table).tolist()
        == ["Software"] * 10 + ["Information Technology"] * 3
    )


def test_financials_fall_back_to_their_industry_group() -> None:
    industries = [
        *["Asset Management & Custody Banks"] * 10,
        "Diversified Banks",
        "Regional Banks",
        "Transaction & Payment Processing Services",
        "Reinsurance",
        "Something New",  # unmapped: the sector
    ]
    table = pd.DataFrame({"industry": industries, "sector": "Financials"})

    assert screen.peers(table).tolist() == [
        *["Asset Management & Custody Banks"] * 10,
        "Banks",
        "Banks",
        "Financial Services",
        "Insurance",
        "Financials",
    ]
    rows = screen.records(table.assign(peer=screen.peers(table)))
    assert [screen.peer_level(r) for r in rows[9:11]] == ["세부 산업", "산업 그룹"]
    assert screen.peer_level(rows[-1]) == "섹터"


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
    }
    return row | overrides


def status(*rows: dict[str, object]) -> pd.DataFrame:
    return screen.fundamentals_status(pd.DataFrame(list(rows)))


def test_cash_flow_disqualifier() -> None:
    result = status(frame(), frame(fcf=-1.0), frame(fcf=math.nan))

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
    assert result["ev_cheap"].tolist() == [False] * 7


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


def test_non_financials_keep_the_cash_flow_test() -> None:
    result = status(frame(tce_ratio=0.01, eq_ratio=0.01), frame(fcf=-1.0))

    assert result["dq_fcf"].tolist() == ["통과", "결격"]


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


def test_roe_label_against_the_peer_median() -> None:
    result = status(
        frame(roe=0.10),
        frame(roe=0.20),
        frame(roe=0.30),
        frame(roe=-0.5, book_value=-1.0),
        frame(roe=math.nan),
    )

    assert result["roe_med"].iloc[0] == pytest.approx(0.20)
    assert result["roe_label"].tolist() == ["", "우수", "우수", "해석 부적절", "미확인"]


def test_valuation_labels_rank_positive_pers_in_the_peer_group() -> None:
    pers = [5.0, *range(10, 18), 40.0, -3.0, math.nan]
    result = status(*(frame(fpe=float(p)) for p in pers))

    labels = result["value_label"].tolist()
    assert labels[0] == "저렴"  # lowest 2 of 10 positive values
    assert labels[1] == "저렴"
    assert labels[2:8] == ["보통"] * 6
    assert labels[8:10] == ["비쌈", "비쌈"]
    assert labels[10:] == ["해석 부적절", "미확인"]


def test_cyclicals_flag_peak_earnings() -> None:
    oil = {"sector": "Energy", "peer": "Energy"}
    result = status(
        frame(**oil, fpe=5.0, tpe=10.0),
        frame(**oil, fpe=7.0, tpe=10.0),
        frame(fpe=5.0, tpe=10.0),  # not cyclical
        frame(**oil, fpe=5.0, tpe=-10.0),
    )

    assert result["peak"].tolist() == [True, False, False, False]


# ---------------------------------------------------------------- groups


def group_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "chart": "강",
        "dq_state": "통과",
        "roe_label": "우수",
        "value_label": "보통",
        "ev_cheap": False,
        "fy1_now": 6.0,
        "fy1_90d": 5.0,
        "fy1_up30": 3.0,
        "fy1_down30": 1.0,
        "score": 9,
    }
    return row | overrides


def members(*rows: dict[str, object]) -> list[list[str]]:
    result = screen.groups(pd.DataFrame(list(rows)))
    return [
        [k[3:] for k in result.columns if result.iloc[i][k]] for i in range(len(rows))
    ]


def test_group_membership() -> None:
    assert members(
        group_row(),
        group_row(value_label="비쌈"),
        group_row(value_label="미확인"),
        group_row(dq_state="미확인"),
        group_row(chart="약", value_label="저렴", score=4),
        group_row(chart="", roe_label="", value_label="저렴", score=5),
        group_row(chart="", roe_label="", ev_cheap=True, score=3),
        group_row(chart="", roe_label="", value_label="저렴", score=7),
        group_row(chart="", value_label="저렴", score=5, fy1_now=4.0),
        group_row(chart="", value_label="저렴", score=5, fy1_down30=4.0),
        group_row(chart="", value_label="저렴", score=5, fy1_now=math.nan),
        group_row(value_label="저렴", score=6, dq_state="결격"),
    ) == [
        ["A"],
        ["B"],
        [],  # 비쌈 아님 needs a valuation
        [],  # judgment withheld
        ["C", "D"],
        ["D"],
        ["D"],
        [],  # score above 6
        [],  # estimates cut
        [],  # more down than up revisions
        [],  # no estimates
        [],
    ]


def test_d_excludes_a_but_may_overlap_b() -> None:
    assert members(group_row(value_label="저렴", score=6)) == [["A"]]
    assert members(group_row(value_label="비쌈", ev_cheap=True, score=6)) == [
        ["B", "D"]
    ]


def test_groups_are_sorted_and_capped() -> None:
    table = pd.DataFrame(
        {
            "in_D": [True] * 25 + [False],
            "signals": [i % 4 for i in range(26)],
            "score": [3 + i % 4 for i in range(26)],
            "mcap": [float(i) for i in range(26)],
        }
    )

    top = screen.group(table, "D")

    assert len(top) == screen.TOP_N
    assert top["signals"].tolist() == sorted(top["signals"], reverse=True)
    first = top[top["signals"] == 3]["score"].tolist()
    assert first == sorted(first, reverse=True)


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
    assert by.loc["BANK", "peer"] == "Banks"
    assert by.loc["BANK", "dq_fcf"] == "통과"  # TCE/TA 6%
    assert bool(by.loc["OIL", "peak"])
    assert screen.tags(by.loc["OIL"].to_dict()).endswith("정점 이익")
    assert screen.failures(table)["ticker"].tolist() == ["THIN"]
    assert all(screen.group(kept, key).shape[0] <= screen.TOP_N for key in "ABCD")


def test_page_text_for_one_stock(tmp_path: Path) -> None:
    row = screen.liquid(stored(tmp_path)).set_index("ticker").loc["DOWN"].to_dict()

    assert len(screen.tech_lines(row)) == 10
    assert all(isinstance(ok, bool) for ok, _, _ in screen.tech_lines(row))
    assert screen.dq_lines(row)[0][0] == "결격"
    assert screen.verdict(row) == "결격: FCF"
    assert screen.estimate_table(row)["현재"].tolist() == ["5.00", "6.00"]
    assert len(screen.label_lines(row)) == 3
    assert len(screen.overheat_lines(row)) == 3


def test_values_are_formatted_for_the_page() -> None:
    assert screen.fmt("pct", 0.1234) == "12.3%"
    assert screen.fmt("signed", 0.05) == "+5.0%"
    assert screen.fmt("big", 2.5e12) == "$2,500.0B"
    assert screen.fmt("usd", 345.2) == "$345.20"
    assert screen.fmt("x", 2.25) == "2.2x"
    assert screen.fmt("num", math.nan) == "—"
    assert screen.fmt("num", None) == "—"


def test_chart_rows_cover_a_year_with_moving_averages(tmp_path: Path) -> None:
    _ = stored(tmp_path)
    rows = screen.chart_rows(screen.history(tmp_path / "data" / "screen", "UP1"))

    assert {row["line"] for row in rows} == {"종가", "20일선", "60일선", "200일선"}
    days = sorted({str(row["day"]) for row in rows})
    assert days[0] >= "2025-10-22"
    assert days[-1] == "2026-10-22"
    assert screen.chart_rows(screen.empty_prices()) == []


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


def test_pending_skips_fetched_rows_and_retries_failures() -> None:
    existing = pd.DataFrame({"ticker": ["A", "B"], "error": ["", "HTTPError"]})

    assert screen.pending(existing, ["A", "B", "C"]) == ["B", "C"]
    assert screen.pending(pd.DataFrame(), ["A"]) == ["A"]


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


def test_one_update_at_a_time(tmp_path: Path) -> None:
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


def test_the_page_without_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = page(tmp_path, monkeypatch)

    assert not at.exception
    assert "S&P 500 스크리닝" in [h.value for h in at.header]
    assert any(i.value.startswith("아직 받은 데이터가 없습니다") for i in at.info)
    assert at.button(key="screen_refresh").disabled
    assert not at.dataframe


def test_the_page_shows_groups_and_opens_a_stock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / screen.SNAPSHOT_DIR
    _ = run_update(FakeSource(), root)
    inputs = screen.load(root)
    assert inputs is not None
    table = screen.evaluate(inputs)
    at = page(tmp_path, monkeypatch)

    assert not at.exception
    bar = " ".join(c.value for c in at.caption)
    assert "시세 2026-10-22" in bar
    assert "재무 2026-Q4" in bar
    assert len(at.dataframe[0].value) == 7  # the liquid stocks

    selector = at.segmented_control(key="screen_group")
    labels = " ".join(selector.options)
    assert all(name in labels for name in screen.GROUP_NAMES.values())
    for key in "ABCD":  # the map's group cells, then the empty one
        assert not at.button(key=f"screen_cell_{key}").disabled
    assert at.button(key="screen_cell_none").disabled
    assert at.button(key="screen_cell_none").label == "—"
    captions = " ".join(c.value for c in at.caption)
    assert "비쌈 아님" in captions
    assert "차트 약" in captions
    assert screen.GROUPS["A"] not in captions  # conditions only in tooltips

    _ = at.button(key="screen_cell_B").click().run()
    assert not at.exception
    assert at.segmented_control(key="screen_group").value == "B"
    shown = at.dataframe[0].value["ticker"].tolist()
    assert shown == screen.pick(table, "B")["ticker"].tolist()

    _ = at.segmented_control(key="screen_group").set_value("all").run()
    _ = at.multiselect(key="screen_sectors").set_value(["Financials"]).run()
    assert at.dataframe[0].value["ticker"].tolist() == ["BANK"]

    click(at, "sector")  # only the ticker and name cells open a stock
    assert not at.get("vega_lite_chart")

    click(at, "name")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "TCE/TA" in text  # the bank test
    assert "해당 없음" in text
    assert len(at.get("vega_lite_chart")) == 1

    _ = at.run()  # an unrelated rerun does not bring the dialog back
    assert not at.get("vega_lite_chart")

    click(at, "ticker")  # the same row again opens it again
    assert len(at.get("vega_lite_chart")) == 1
