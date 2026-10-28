"""The demo's screen: the judged table for a few companies, every number ``VALUE``.

Nothing is screened; the page filters and draws this table as it is.
"""

# pandas ships no complete type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from datetime import date, timedelta
from typing import Final

import pandas as pd

from finagent import screen
from finagent.demo import VALUE

# (ticker, sector): enough sectors for the 섹터 filter.
COMPANIES: Final = (
    ("DEMOA", "Information Technology"),
    ("DEMOB", "Information Technology"),
    ("DEMOC", "Health Care"),
    ("DEMOD", "Health Care"),
    ("DEMOE", "Financials"),
    ("DEMOF", "Energy"),
)
# Where the 분포 map draws each company, its 999s being off the axes: evenly
# spaced on the diagonal, a layout no real screen gives.
MAP_SPOTS: Final = {
    ticker: (i - (len(COMPANIES) - 1) / 2,) * 2
    for i, (ticker, _) in enumerate(COMPANIES)
}
# Columns ``screen.judge`` adds to the stored ones.
JUDGED: Final = (
    "sec_avg",
    "sec_years",
    "sec_ttm",
    "weight",
    "rev_strength",
    "net_revision",
    "q_count",
    "v_count",
    "quality",
    "value",
    "roa",
    "leverage",
    "earnings_yield",
    "cash_yield",
    "sales_yield",
)
# Shares the page shows as percentages, and amounts it shows in billions.
PERCENTS: Final = (
    "roa",
    "earnings_yield",
    "cash_yield",
    "op_margin",
    "margin_avg",
    "near_high",
    "ret_1m",
    "ret_3m",
    "mom12_1",
    "rev_strength",
    "net_revision",
)
BILLIONS: Final = ("mcap", *screen.ANNUAL_COLUMNS)
DATES: Final = (
    "ttm_end",
    "fy_end",
    "last_fy_end",
    "next_fy_end",
    "sec_ttm_end",
    "op_margin_end",
    "as_of",
)


def table(today: date) -> pd.DataFrame:
    """The judged table's columns for ``COMPANIES``, every number shown as 999."""
    numbers = dict.fromkeys(
        (
            *screen.TECH_COLUMNS,
            *screen.FUNDAMENTAL_COLUMNS,
            *screen.ESTIMATE_COLUMNS,
            *screen.QUALITY,
            *screen.VALUE,
            *JUDGED,
        ),
        VALUE,
    )
    day = today.isoformat()
    return pd.DataFrame(
        [
            numbers
            | dict.fromkeys(PERCENTS, VALUE / 100)
            | dict.fromkeys(BILLIONS, VALUE * 1e9)
            | dict.fromkeys(DATES, day)
            | {
                "ticker": ticker,
                "name": f"데모컴퍼니 {ticker[-1]}",
                "sector": sector,
                "industry": "데모 업종",
                "currency": "USD",
                "est_currency": "USD",
                "margin_source": "데모",
                "op_margin_source": "데모",
                "error": "",
                "liquidity": "",
                "pending": False,
            }
            for ticker, sector in COMPANIES
        ]
    )


def parts(today: date) -> screen.Parts:
    return screen.Parts(
        today, today, screen.quarter(today), today, screen.quarter(today)
    )


def stock_chart(today: date) -> list[dict[str, str | float]]:
    """A year of the stock dialog's four lines, flat at ``VALUE``."""
    days = pd.bdate_range(today - timedelta(days=365), today)
    return [
        {"day": day.date().isoformat(), "line": line, "value": VALUE}
        for line in ("종가", "20일선", "60일선", "200일선")
        for day in days
    ]
