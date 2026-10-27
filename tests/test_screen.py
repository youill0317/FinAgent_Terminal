"""The S&P 500 screener without the network: fake sources and synthetic prices."""

# pandas ships no complete type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import json
import math
import threading
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
FUNDAMENTALS: dict[str, dict[str, float]] = {
    # operating cash flow and ROA; the newest year's net income is ROA x 50.
    "UP1": {"ocf": 6e9, "roa": 0.06},
    "UP2": {"ocf": 4e9, "roa": 0.04},
    "UP3": {"ocf": 2e9, "roa": 0.02},
    "FLAT": {"ocf": 2e9, "roa": 0.05},
    "DOWN": {"ocf": -1e9, "roa": -0.01},
    "BANK": {"ocf": math.nan, "roa": 0.012},
    "OIL": {"ocf": 3e9, "roa": 0.03},
    "THIN": {"ocf": 1e9, "roa": 0.01},
}


def fundamentals(ticker: str) -> dict[str, object]:
    base: dict[str, object] = dict.fromkeys(screen.FUNDAMENTAL_COLUMNS[1:-1], math.nan)
    given = FUNDAMENTALS[ticker]
    base |= {
        "mcap": 1e11,
        "margin_avg": 0.15,
        "margin_years": 5,
        "op_margin": 0.15,
        "fy_end": "2025-12-31",
        "last_fy_end": "2025-12-31",
        "next_fy_end": "2026-12-31",
        "currency": "USD",
    }
    for year in (0, 1):
        base |= {
            f"ni_{year}": given["roa"] * 5e10,
            f"sales_{year}": 1e10 * (1 + given["roa"] * (1 - year)),
            f"ocf_{year}": given["ocf"],
            f"ta_{year}": 5e10,
            f"ca_{year}": 1.5e10,
            f"cl_{year}": 1e10,
            f"cash_sti_{year}": 4e9,
            f"std_{year}": 5e8,
            f"ltd_{year}": 4.5e9,
            f"tl_{year}": 4e10,
        }
    if ticker == "BANK":
        base |= {f"{stem}_{y}": math.nan for stem in ("ca", "cl") for y in (0, 1)}
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


def test_fundamentals_row_reads_info_and_statements() -> None:
    info = {
        "marketCap": 4e10,
        "financialCurrency": "USD",
        "lastFiscalYearEnd": 1767139200,  # 2025-12-31
        "nextFiscalYearEnd": 1798675200,  # 2026-12-31
    }
    statements = screen.Statements(
        income=yearly(
            TotalRevenue=[90.0, 100.0],
            OperatingIncome=[9.0, 20.0],
            NetIncome=[5.0, 4.0],
        ),
        balance=yearly(TotalAssets=[200.0, 180.0], CurrentDebt=[3.0, math.nan]),
        cash=yearly(OperatingCashFlow=[12.0, 11.0]),
    )

    row = screen.fundamentals_row(info, statements)

    assert row["margin_avg"] == pytest.approx(0.15)  # median of 0.1 and 0.2
    assert row["mcap"] == 4e10
    assert row["currency"] == "USD"
    assert row["fy_end"] == "2025-12-31"
    assert (row["ni_0"], row["ni_1"], row["sales_0"]) == (5.0, 4.0, 90.0)
    assert (row["ta_0"], row["ta_1"], row["ocf_1"]) == (200.0, 180.0, 11.0)
    assert row["std_0"] == 3.0
    assert math.isnan(cast("float", row["std_1"]))
    assert math.isnan(cast("float", row["inv_adv_0"]))  # no such row
    assert (row["last_fy_end"], row["next_fy_end"]) == ("2025-12-31", "2026-12-31")
    assert set(row) == set(screen.FUNDAMENTAL_COLUMNS) - {"ticker", "error"}

    bare = screen.fundamentals_row({}, screen.Statements(pd.DataFrame()))
    assert math.isnan(cast("float", bare["ta_0"]))
    assert bare["fy_end"] == ""
    assert bare["last_fy_end"] == ""


# ---------------------------------------------------------------- indicators


def test_tech_row_uses_the_standard_definitions() -> None:
    frame = ohlcv("UP1", date(2024, 9, 1), DAY, 0.004)
    close = frame["adj_close"].to_numpy()

    row = screen.tech_row(frame)

    assert row["mom12_1"] == pytest.approx(close[-22] / close[-253] - 1)
    assert row["near_high"] == pytest.approx(close[-1] / close[-252:].max())
    assert row["ma_dist"] == pytest.approx(close[-1] / close[-200:].mean())
    assert row["ret_3m"] == pytest.approx(close[-1] / close[-64] - 1)

    short = screen.tech_row(ohlcv("UP1", date(2026, 1, 1), DAY, 0.002))
    assert short["bars"] < screen.YEAR_BARS
    assert math.isnan(short["mom12_1"])
    assert math.isnan(short["near_high"])
    assert not math.isnan(short["ret_1m"])


# ---------------------------------------------------------------- fundamentals


def frame(**overrides: object) -> dict[str, object]:
    """A non-financial with two fiscal years of every FTSE item."""
    row: dict[str, object] = {
        "ticker": "X",
        "sector": "Information Technology",
        "industry": "Software",
        "price": 50.0,
        "mcap": 200.0,
        "ni_0": 6.0,
        "ni_1": 5.0,
        "sales_0": 50.0,
        "sales_1": 40.0,
        "ocf_0": 8.0,
        "ocf_1": 7.0,
        "ta_0": 110.0,
        "ta_1": 90.0,
        "ca_0": 40.0,
        "ca_1": 30.0,
        "cl_0": 20.0,
        "cl_1": 15.0,
        "cash_sti_0": 10.0,
        "cash_sti_1": 8.0,
        "sti_0": 2.0,
        "sti_1": 1.0,
        "std_0": 5.0,
        "std_1": 4.0,
        "ltd_0": 15.0,
        "ltd_1": 12.0,
        "tl_0": 60.0,
        "tl_1": 50.0,
        "pref_0": math.nan,
        "pref_1": math.nan,
        "inv_adv_0": 6.0,
        "inv_adv_1": 5.0,
    }
    return row | overrides


def table_of(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


INSURER = {"sector": "Financials", "industry": "Property & Casualty Insurance"}


# ---------------------------------------------------------------- 퀄리티 and 밸류에이션


def test_measures_follow_the_ftse_rules() -> None:
    values = screen.measure_values(
        table_of(
            frame(),
            frame(**INSURER),
            frame(sector="Real Estate", industry="Retail REITs"),
            frame(ta_1=-1.0),
            frame(std_0=math.nan, ltd_0=math.nan),  # no debt rows: none
            frame(ocf_0=2.0),
            frame(ocf_0=math.nan),
        )
    )

    average = (110 + 90) / 2
    assert values["q_roa"][0] == pytest.approx(6 / average)
    assert values["q_turnover"][0] == pytest.approx(50 / 110 - 40 / 90)
    wc = (40 - 10) - (20 - 5), (30 - 8) - (15 - 4)
    nco = (110 - 40 - 6) - (60 - 20 - 15), (90 - 30 - 5) - (50 - 15 - 12)
    fin = (2 + 6) - (15 + 5), (1 + 5) - (12 + 4)  # no preferred row: zero
    change = (wc[0] - wc[1]) + (nco[0] - nco[1]) + (fin[0] - fin[1])
    assert values["q_accruals"][0] == pytest.approx(-change / average)
    # Financials and real estate: ROA alone for 퀄리티, every 밸류에이션 yield.
    for key in ("q_turnover", "q_accruals", "q_leverage"):
        assert values[key].iloc[1:3].isna().all()
    assert values["q_roa"].iloc[1:3].notna().all()
    assert values["v_sales"][1] == pytest.approx(50 / 200)
    assert bool(values["q_roa"].isna().iloc[3])  # an asset below zero: no value
    assert values["v_cash"][0] == pytest.approx(8 / 200)
    assert values["v_earnings"][0] == pytest.approx(6 / 200)

    ratio = screen.leverage_ratio(table_of(*(frame(ocf_0=v) for v in (8.0, 30, 2))))
    assert ratio.tolist() == pytest.approx([0.4, 1.0, 0.1])  # above the debt: 1
    # Leverage is less its sector's median when the sector has three or more.
    median = 0.4
    assert values["q_leverage"][0] == pytest.approx(0.4 - median)
    assert values["q_leverage"][4] == pytest.approx(1.0 - median)
    assert values["q_leverage"][5] == pytest.approx(0.1 - median)
    assert bool(values["q_leverage"].isna().iloc[6])
    alone = screen.measure_values(table_of(frame(), frame(ocf_0=2.0)))
    assert alone["q_leverage"].tolist() == pytest.approx([0.4, 0.1])


def test_normalise_truncates_at_three_and_renormalises() -> None:
    values = pd.Series([*range(20), 1000.0, math.nan], dtype="float")

    z = screen.normalise(values)

    assert z.abs().max() <= screen.Z_LIMIT + 1e-9
    assert pd.isna(z.iloc[-1])
    kept = z.dropna()
    assert float(kept.mean()) == pytest.approx(0.0, abs=1e-9)
    assert float(cast("float", kept.std(ddof=0))) == pytest.approx(1.0)
    assert screen.normalise(pd.Series([2.0, 2.0])).tolist() == [0.0, 0.0]


def test_axes_combine_what_each_stock_has() -> None:
    table = table_of(
        frame(ticker="A", ni_0=2.0, ocf_0=3.0, mcap=100.0),
        frame(ticker="B", ni_0=6.0, ocf_0=8.0, mcap=200.0),
        frame(ticker="C", ni_0=9.0, ocf_0=12.0, mcap=300.0),
        frame(ticker="D", **INSURER, ni_0=4.0, mcap=100.0),
        frame(ticker="E", **INSURER, ni_0=math.nan, ocf_0=math.nan, sales_0=math.nan),
    )

    found = screen.axes(table)

    assert found["q_count"].tolist() == [4, 4, 4, 1, 0]
    assert found["v_count"].tolist() == [3, 3, 3, 3, 0]
    # No 퀄리티 or 밸류에이션 value: a neutral zero (FTSE rule 5.2.3).
    assert (found.loc[4, "quality"], found.loc[4, "value"]) == (0.0, 0.0)
    assert found["quality"].iloc[2] > found["quality"].iloc[0]
    roa = screen.normalise(screen.measure_values(table)["q_roa"])
    assert found["q_roa"].tolist()[:4] == pytest.approx(roa.tolist()[:4])
    off = screen.axes(table, screen.criteria(["q_accruals", "v_sales"]))
    assert off["q_count"].tolist()[:3] == [3, 3, 3]
    assert off["v_count"].tolist()[:3] == [2, 2, 2]
    assert bool(off["q_accruals"].isna().all())


# ---------------------------------------------------------------- estimates


def year_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "as_of": "2026-10-22",
        "last_fy_end": "2025-12-31",
        "next_fy_end": "2026-12-31",
    }
    return row | overrides


def test_the_fiscal_year_state() -> None:
    table = pd.DataFrame(
        [
            year_row(),  # 70 days left of FY1
            year_row(last_fy_end="2025-09-27", next_fy_end="2026-09-26"),  # ended
            year_row(last_fy_end="2026-08-31", next_fy_end="2027-08-31"),
            year_row(last_fy_end="", next_fy_end=""),
        ]
    )

    state = screen.year_state(table)

    assert state["weight"].tolist() == pytest.approx(
        [70 / 365.25, 0.0, 313 / 365.25, 1.0]
    )
    assert state["pending"].tolist() == [False, True, False, False]


# ---------------------------------------------------------------- whole run


def test_evaluate_on_stored_data(tmp_path: Path) -> None:
    table = stored(tmp_path)
    by = table.set_index("ticker")

    assert set(table["ticker"]) == set(UNIVERSE["ticker"])
    assert (by["liquidity"] == "").all()
    kept = screen.liquid(table)
    assert by["quality"].idxmax() == "UP1"  # the most profitable
    assert by.loc["DOWN", "quality"] < 0
    assert by.loc["BANK", "q_count"] == 1  # ROA alone
    assert math.isnan(by.loc["BANK", "q_leverage"])
    assert by.loc["UP1", "roa"] == pytest.approx(0.06)
    assert screen.failures(table).empty
    assert len(screen.pick(table, ["Energy"])) == 2
    points = screen.map_rows(kept)
    assert len(points) == len(kept)
    _ = json.dumps(points, allow_nan=False)  # the chart's JSON
    assert list(screen.view(kept).columns) == [c for c, _, _ in screen.TABLE]


def load(tmp_path: Path) -> screen.Inputs:
    inputs = screen.load(tmp_path / "data" / "screen")
    assert inputs is not None
    return inputs


# ---------------------------------------------------------------- criteria


def test_saved_criteria_are_flips_of_the_defaults() -> None:
    assert screen.DEFAULT.off == ()
    c = screen.criteria(["q_roa", "gone", "v_sales"])

    assert c.off == ("q_roa", "v_sales")  # unknown ids dropped
    assert screen.changed(c) == ["q_roa", "v_sales"]
    assert screen.criteria(screen.changed(c)) == c
    assert screen.criteria([]) == screen.DEFAULT
    keys = [rule.key for rule in screen.RULES]
    assert len(keys) == len(set(keys))
    assert set(keys) == {"min_price"} | set(screen.QUALITY) | set(screen.VALUE)


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


def test_snapshot_keeps_every_criteria_number(tmp_path: Path) -> None:
    table = stored(tmp_path)

    path = screen.write_snapshot(tmp_path, table, screen.criteria(["v_sales"]), DAY)

    meta = json.loads(path.with_suffix(".json").read_text("utf-8"))
    assert meta["off"] == ["v_sales"]
    assert meta["definitions"] == screen.DEFINITIONS
    assert meta["as_of"] == "2026-10-22"
    assert "sector" in pd.read_csv(path).columns


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
