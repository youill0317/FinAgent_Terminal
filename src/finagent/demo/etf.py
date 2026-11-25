# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""오프라인·데모 모드용 ETF·종목 일봉 시세, 펀드 메타데이터 및 재무제표 생성기."""

from __future__ import annotations

import math
import zlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Final

import pandas as pd

from finagent import etf, fund_filings
from finagent.demo import SHARE, VALUE
from finagent.etf_universe import DEFAULTS
from finagent.fundamentals import ANNUAL_COLUMNS, Company, FilingRow

if TYPE_CHECKING:
    from collections.abc import Sequence

DAYS: Final = 400
SOURCE: Final = "데모"
STAMP: Final = "demo"
COMPANIES: Final = ("DEMOA", "DEMOB", "DEMOC", "DEMOD", "DEMOE", "DEMOF")
COMPANY_FIELDS: Final = (
    "market_cap",
    "per",
    "forward_per",
    "dividend_yield",
    "roe",
    "net_margin",
    "revenue_growth",
)


def _today() -> str:
    return datetime.now().astimezone().date().isoformat()


def _cell(value: object) -> etf.Cell:
    day = _today()
    return etf.Cell(value, SOURCE, day)


def rows(tickers: Sequence[str]) -> pd.DataFrame:
    end = datetime.now().astimezone().date()
    days = pd.bdate_range(end - timedelta(days=DAYS), end)
    step = list(range(len(days)))
    parts: list[pd.DataFrame] = []
    for ticker in tickers:
        seed = zlib.crc32(ticker.encode()) % 1000 / 1000
        line = [
            VALUE
            * (1 + (seed - 0.5) * i / len(days) + 0.05 * math.sin(i / 20 + seed * 6))
            for i in step
        ]
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": days,
                    "close": line,
                    "adj_close": line,
                    "volume": VALUE,
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["ticker", "date", "close", "adj_close", "volume"])
    return pd.concat(parts, ignore_index=True)


def funds() -> dict[str, dict[str, etf.Cell]]:
    weights = {"데모 섹터": SHARE * 10}
    return {
        p.ticker: {
            "name": _cell(f"{p.ticker} 데모 펀드"),
            "family": _cell(SOURCE),
            "inception": _cell(_today()),
            "aum": _cell(VALUE),
            "expense": _cell(SHARE),
            "yield": _cell(SHARE),
            "nav": _cell(VALUE),
            "previous_close": _cell(VALUE),
            "sector_weights": _cell(weights),
            "credit_weights": _cell(weights),
        }
        for p in DEFAULTS
    }


def detail(ticker: str) -> etf.Composition:
    holdings = tuple(
        fund_filings.Row(
            name=f"{name} 데모",
            cusip=None,
            isin=None,
            ticker=name,
            weight=SHARE,
            asset="stock",
            country="US",
            maturity=None,
        )
        for name in COMPANIES
    )
    return etf.Composition(ticker, _today(), SOURCE, None, full=True, rows=holdings)


def company(ticker: str) -> Company:
    cells = {name: _cell(VALUE) for name in COMPANY_FIELDS}
    cells |= {
        "name": _cell(f"{ticker} 데모"),
        "sector": _cell(SOURCE),
        "industry": _cell(SOURCE),
        "country": _cell(SOURCE),
        "currency": _cell("USD"),
    }
    return Company(ticker, ticker, _today(), cells)


def annual() -> dict[str, object]:
    return {
        "source": SOURCE,
        "currency": "USD",
        "fy_end": _today(),
        **dict.fromkeys(ANNUAL_COLUMNS, VALUE),
    }


def stock_rows(symbol: str) -> pd.DataFrame:
    return rows([symbol])


def filings(ticker: str) -> tuple[FilingRow, ...]:
    day = _today()
    return tuple(
        FilingRow(
            form=form,
            accession=f"0000000000-26-00000{index}",
            filed=day,
            items=items,
            primary=f"{ticker.lower()}-{form.lower()}.htm",
            url="https://www.sec.gov/",
        )
        for index, (form, items) in enumerate(
            (("10-K", ""), ("10-Q", ""), ("8-K", "2.02, 9.01")), 1
        )
    )
