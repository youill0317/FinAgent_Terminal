"""The demo's own folder: a canned book, made-up prices and a small screen.

The demo checks the pages, not the numbers, so it never reads the user's
files: ``workspace`` makes a temporary folder once per process, holding the
settings file and every page's data, filled from closed-form series.
"""

# pandas ships no complete type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import math
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from finagent import portfolio, screen

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SETTINGS_FILE: Final = "finagent.settings.json"
YEARS: Final = 3  # of portfolio prices

# ticker: (start price, yearly growth, swing, swing period in days)
PRICES: Final = {
    "AAPL": (150.0, 0.18, 0.06, 40.0),
    "MSFT": (300.0, 0.14, 0.05, 55.0),
    "005930.KS": (60_000.0, 0.06, 0.08, 35.0),
    portfolio.BENCHMARK: (4_000.0, 0.10, 0.03, 60.0),
    portfolio.TOTAL_RETURN: (8_000.0, 0.12, 0.03, 60.0),
    portfolio.KOSPI: (2_400.0, 0.05, 0.04, 45.0),
    portfolio.FX: (1_300.0, 0.02, 0.02, 80.0),
}
RATE: Final = 4.0  # ^IRX, annual percent

# The canned book: (days before today, ticker, side, shares, price).
TRADES: Final[tuple[tuple[int, str, portfolio.Side, float, float], ...]] = (
    (700, "AAPL", "buy", 20, 160.0),
    (650, "MSFT", "buy", 10, 310.0),
    (600, "005930.KS", "buy", 100, 62_000.0),
    (300, "AAPL", "sell", 5, 190.0),
    (120, "MSFT", "buy", 5, 380.0),
)

# The screen: (ticker, name, sector, industry, daily growth, FCF, ROE).
STOCKS: Final = (
    ("ALFA", "Alfa Software", "Information Technology", "Software", 0.0015, 5e9, 0.30),
    ("BRVO", "Bravo Systems", "Information Technology", "Software", 0.0010, 3e9, 0.22),
    (
        "CHIP",
        "Chip Works",
        "Information Technology",
        "Semiconductors",
        0.0020,
        4e9,
        0.28,
    ),
    (
        "DLTA",
        "Delta Devices",
        "Information Technology",
        "Semiconductors",
        -0.0008,
        -1e9,
        0.04,
    ),
    ("ECHO", "Echo Health", "Health Care", "Pharmaceuticals", 0.0006, 2e9, 0.18),
    ("FOXT", "Foxtrot Bio", "Health Care", "Biotechnology", 0.0002, 5e8, 0.08),
    ("GOLF", "Golf Bank", "Financials", "Diversified Banks", 0.0004, math.nan, 0.12),
    (
        "HOTL",
        "Hotel Retail",
        "Consumer Discretionary",
        "Broadline Retail",
        0.0009,
        2e9,
        0.20,
    ),
    (
        "INDA",
        "India Motors",
        "Consumer Discretionary",
        "Automobile Manufacturers",
        -0.0003,
        1e9,
        0.10,
    ),
    ("JULI", "Juliet Oil", "Energy", "Integrated Oil & Gas", -0.0004, 3e9, 0.16),
    ("KILO", "Kilo Power", "Utilities", "Electric Utilities", 0.0003, 1e9, 0.09),
    (
        "LIMA",
        "Lima Foods",
        "Consumer Staples",
        "Packaged Foods & Meats",
        0.0001,
        1.5e9,
        0.14,
    ),
)


def workspace() -> Path:
    """A new temporary folder with the demo's data; returns its settings file."""
    root = Path(tempfile.mkdtemp(prefix="finagent-demo-"))
    today = datetime.now(UTC).date()
    pf_root = root / portfolio.DATA_DIR
    portfolio.save_book(pf_root, book(today))
    portfolio.save_prices(pf_root, prices(today))
    for universe in screen.UNIVERSES.values():
        folder = root / screen.SNAPSHOT_DIR / universe.folder
        _ = screen.update(DemoSource(), folder, datetime.now(UTC), _quiet, _quiet)
    return root / SETTINGS_FILE


def _quiet(*_: object) -> None:
    pass


def book(today: date) -> portfolio.Book:
    out, pid = portfolio.add_portfolio(portfolio.Book(), "데모 포트폴리오", "KRW")
    for ago, ticker, side, shares, price in TRADES:
        trade = portfolio.Trade(
            id=portfolio.new_id(),
            day=_business(today - timedelta(days=ago)),
            ticker=ticker,
            side=side,
            quantity=shares,
            price=price,
        )
        out = portfolio.add_trade(out, pid, trade)
    return out


def _business(day: date) -> date:
    while day.weekday() >= 5:  # noqa: PLR2004 - Saturday and Sunday
        day -= timedelta(days=1)
    return day


def prices(today: date) -> portfolio.Prices:
    days = pd.bdate_range(today - timedelta(days=365 * YEARS), today)
    t = np.arange(len(days), dtype="float64")
    frame = pd.DataFrame(
        {
            ticker: start
            * (1 + growth) ** (t / portfolio.TRADING_DAYS)
            * (1 + swing * np.sin(t / period))
            for ticker, (start, growth, swing, period) in PRICES.items()
        },
        index=days,
    )
    frame[portfolio.RF_USD] = RATE
    return portfolio.Prices(frame, frame.copy())


class DemoSource:
    """``screen.Source`` from ``STOCKS``: no network."""

    def constituents(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ticker": s[0], "name": s[1], "sector": s[2], "industry": s[3]}
                for s in STOCKS
            ]
        )

    def prices(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        days = pd.bdate_range(start, end - timedelta(days=1))
        if days.empty:
            return screen.empty_prices()
        growth = {s[0]: s[4] for s in STOCKS}
        age = np.asarray((days - pd.Timestamp("2020-01-01")).days, dtype="float64")
        frames = []
        for ticker in tickers:
            close = 50 * (1 + growth[ticker]) ** age * (1 + 0.03 * np.sin(age / 7))
            frames.append(
                pd.DataFrame(
                    {
                        "date": days,
                        "ticker": ticker,
                        "open": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "adj_close": close,
                        "volume": 2e6 * (1 + 0.2 * np.cos(age / 5)),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def fundamentals(
        self, ticker: str, *, financial: bool = False
    ) -> Mapping[str, object]:
        del financial
        _, _, sector, _, _, fcf, roe = next(s for s in STOCKS if s[0] == ticker)
        row: dict[str, object] = dict.fromkeys(
            screen.FUNDAMENTAL_COLUMNS[1:-1], math.nan
        )
        row |= {
            "fcf": fcf,
            "ocf": fcf + 1e9,
            "roe": roe,
            "book_value": 20.0,
            "trailing_eps": 2 + 10 * roe,
            "total_debt": 2e9,
            "total_cash": 1e9,
            "ebitda": 3e9,
            "nd_ebitda": 1 / 3,
            "ev": 6.1e10,
            "mcap": 6e10,
            "ev_ebitda": 20.3,
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
            **{f"opa_y{year}": roe / 5 - 0.002 * year for year in range(4)},
            "last_fy_end": "2025-12-31",
            "next_fy_end": "2026-12-31",
            "currency": "USD",
        }
        if sector == "Financials":
            row |= {"eq_ratio": 0.08, "tce_ratio": 0.06, "tbv": 4e10, "shares": 1e9}
            row |= {"ocf": math.nan, "ebitda": math.nan, "nd_ebitda": math.nan}
        return row

    def estimates(self, ticker: str) -> Mapping[str, object]:
        del ticker
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
