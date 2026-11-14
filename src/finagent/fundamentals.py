# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false, reportExplicitAny=false
"""SEC XBRL(companyfacts)·Yahoo·Alpaca 기반 기업 기본 정보 및 연간 재무제표 추출."""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

from finagent import etf, limits, markets, memory, prices, sec, yahoo
from finagent.etf import FAILURES, Cell

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

HISTORY_DAYS: Final = 400
CALENDAR_YEAR: Final = 365
KOSDAQ: Final = ".KQ"
KOSPI: Final = ".KS"
USD: Final = "USD"
SEC_FACTS: Final = "SEC companyfacts"
ALPACA: Final = "Alpaca"
YAHOO: Final = "Yahoo"
YAHOO_FIELDS: Final[Mapping[str, str]] = {
    "name": "longName",
    "sector": "sector",
    "industry": "industry",
    "country": "country",
    "description": "longBusinessSummary",
    "market_cap": "marketCap",
    "currency": "currency",
    "per": "trailingPE",
    "forward_per": "forwardPE",
    "dividend_yield": "dividendYield",
    "roe": "returnOnEquity",
    "net_margin": "profitMargins",
    "revenue_growth": "revenueGrowth",
}
PERCENT_FIELDS: Final = frozenset({"roe", "net_margin", "revenue_growth"})


SEC_YEARS: Final = 5


YEAR_DAYS: Final = (350, 380)


REVENUE_CONCEPTS: Final = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)


OPERATING_INCOME: Final = "OperatingIncomeLoss"


ANNUAL_ROWS: Final = {
    "ni": ("income", "NetIncome"),
    "sales": ("income", "TotalRevenue"),
    "ocf": ("cash", "OperatingCashFlow"),
    "ta": ("balance", "TotalAssets"),
    "ca": ("balance", "CurrentAssets"),
    "cl": ("balance", "CurrentLiabilities"),
    "cash_sti": ("balance", "CashCashEquivalentsAndShortTermInvestments"),
    "sti": ("balance", "OtherShortTermInvestments"),
    "std": ("balance", "CurrentDebt"),
    "ltd": ("balance", "LongTermDebt"),
    "tl": ("balance", "TotalLiabilitiesNetMinorityInterest"),
    "pref": ("balance", "PreferredStock"),
    "inv_adv": ("balance", "InvestmentsAndAdvances"),
}


ANNUAL_COLUMNS: Final = tuple(
    f"{stem}_{year}" for year in (0, 1) for stem in ANNUAL_ROWS
)


SEC_ITEMS: Final = {
    "ni": ("NetIncomeLoss",),
    "ocf": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "ta": ("Assets",),
    "ca": ("AssetsCurrent",),
    "cl": ("LiabilitiesCurrent",),
    "cash_sti": ("CashCashEquivalentsAndShortTermInvestments",),
    "sti": (
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    ),
    "tl": ("Liabilities",),
    "pref": ("PreferredStockValue", "PreferredStockValueOutstanding"),
    "inv_adv": ("LongTermInvestments",),
    "std": ("DebtCurrent", "LongTermDebtCurrent"),
    "ltd": ("LongTermDebtNoncurrent",),
}


PROFIT: Final = "ProfitLoss"


MINORITY_PROFIT: Final = "NetIncomeLossAttributableToNoncontrollingInterest"


COMMON_PROFIT: Final = "NetIncomeLossAvailableToCommonStockholdersBasic"


CASH: Final = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
)


LIABILITIES_AND_EQUITY: Final = "LiabilitiesAndStockholdersEquity"


EQUITY: Final = (
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "StockholdersEquity",
)


REDEEMABLE: Final = "RedeemableNoncontrollingInterestEquityCarryingAmount"


MINORITY: Final = "MinorityInterest"


TOTAL_REVENUE: Final = (
    "Revenues",
    "RevenuesNetOfInterestExpense",
    "RegulatedAndUnregulatedOperatingRevenue",
)


NET_INTEREST: Final = "InterestIncomeExpenseNet"


NONINTEREST: Final = "NoninterestIncome"


SEC_STEMS: Final = (*SEC_ITEMS, "rev_total")


def finite(value: object) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        return math.nan
    number = float(value)
    return number if math.isfinite(number) else math.nan


@dataclass(frozen=True, slots=True)
class Statements:
    income: pd.DataFrame
    balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash: pd.DataFrame = field(default_factory=pd.DataFrame)


def _at(table: pd.DataFrame, row: str, end: object) -> float:
    if row not in table.index or end not in table.columns:
        return math.nan
    return finite(table.loc[row, end])


def fundamentals_row(info: Mapping[str, object], s: Statements) -> dict[str, object]:
    row: dict[str, object] = {}
    tables = {"income": s.income, "balance": s.balance, "cash": s.cash}
    ends = sorted(s.balance.columns, reverse=True)[:2]
    row["fy_end"] = str(cast("pd.Timestamp", ends[0]).date()) if ends else ""
    for year in (0, 1):
        end = ends[year] if year < len(ends) else None
        for stem, (statement, name) in ANNUAL_ROWS.items():
            row[f"{stem}_{year}"] = _at(tables[statement], name, end)
    currency = info.get("financialCurrency")
    row["currency"] = currency if isinstance(currency, str) else ""
    return row


def _get(value: object, key: str) -> object:
    return (
        cast("dict[str, object]", value).get(key) if isinstance(value, dict) else None
    )


type Values = dict[str, tuple[float, str]]


def _entries(
    facts: Mapping[str, object], concept: str
) -> list[tuple[str, str, int, float, str, str, str]]:
    units = _get(_get(_get(facts.get("facts"), "us-gaap"), concept), "units")
    entries = _get(units, "USD")
    out: list[tuple[str, str, int, float, str, str, str]] = []
    for item in cast("list[object]", entries if isinstance(entries, list) else []):
        start, end, filed = (_get(item, key) for key in ("start", "end", "filed"))
        value = finite(_get(item, "val"))
        if (
            isinstance(start, str)
            and isinstance(end, str)
            and isinstance(filed, str)
            and not math.isnan(value)
        ):
            days = (date.fromisoformat(end) - date.fromisoformat(start)).days
            form, fp = str(_get(item, "form")), str(_get(item, "fp"))
            out.append((start, end, days, value, filed, form, fp))
    return out


def _keep[K](out: dict[K, tuple[float, str]], key: K, value: float, filed: str) -> None:
    if filed >= out.get(key, (math.nan, ""))[1]:
        out[key] = (value, filed)


def _within(days: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= days <= bounds[1]


def annual(facts: Mapping[str, object], concept: str) -> Values:
    out: Values = {}
    for _, end, days, value, filed, form, fp in _entries(facts, concept):
        if form == "10-K" and fp == "FY" and _within(days, YEAR_DAYS):
            _keep(out, end, value, filed)
    return out


def balance(facts: Mapping[str, object], concept: str) -> Values:
    units = _get(_get(_get(facts.get("facts"), "us-gaap"), concept), "units")
    entries = _get(units, "USD")
    out: Values = {}
    for item in cast("list[object]", entries if isinstance(entries, list) else []):
        end, filed = _get(item, "end"), _get(item, "filed")
        value = finite(_get(item, "val"))
        if (
            _get(item, "start") is None
            and isinstance(end, str)
            and isinstance(filed, str)
            and _get(item, "form") == "10-K"
            and _get(item, "fp") == "FY"
            and not math.isnan(value)
        ):
            _keep(out, end, value, filed)
    return out


def _first(facts: Mapping[str, object], concepts: Sequence[str]) -> dict[str, float]:
    found: dict[str, float] = {}
    for concept in concepts:
        values = annual(facts, concept) | balance(facts, concept)
        for end, (value, _) in values.items():
            _ = found.setdefault(end, value)
    return found


def sec_items(facts: Mapping[str, object]) -> dict[str, dict[str, float]]:
    out = {stem: _first(facts, concepts) for stem, concepts in SEC_ITEMS.items()}
    minority_profit = _first(facts, (MINORITY_PROFIT,))
    for end, profit in _first(facts, (PROFIT,)).items():
        _ = out["ni"].setdefault(end, profit - minority_profit.get(end, 0.0))
    for end, profit in _first(facts, (COMMON_PROFIT,)).items():
        _ = out["ni"].setdefault(end, profit)
    for end, cash in _first(facts, CASH).items():
        _ = out["cash_sti"].setdefault(end, cash + out["sti"].get(end, 0.0))
    whole = _first(facts, EQUITY[:1])
    parent = _first(facts, EQUITY[1:])
    minority = _first(facts, (MINORITY,))
    redeemable = _first(facts, (REDEEMABLE,))
    for end, total in _first(facts, (LIABILITIES_AND_EQUITY,)).items():
        equity = whole.get(end, parent.get(end, math.nan) + minority.get(end, 0.0))
        if not math.isnan(equity):
            tl = total - equity - redeemable.get(end, 0.0)
            _ = out["tl"].setdefault(end, tl)
    out["rev_total"] = _first(facts, TOTAL_REVENUE)
    noninterest = _first(facts, (NONINTEREST,))
    for end, net in _first(facts, (NET_INTEREST,)).items():
        if end in noninterest:
            _ = out["rev_total"].setdefault(end, net + noninterest[end])
    return out


def sec_window(ends: Sequence[str], years: int = SEC_YEARS) -> list[str]:
    newest = sorted(ends, reverse=True)
    if not newest:
        return []
    cutoff = date.fromisoformat(newest[0]) - timedelta(days=365 * years - 180)
    return [end for end in newest if date.fromisoformat(end) > cutoff][:years]


def sec_years(facts: Mapping[str, object]) -> list[dict[str, object]]:
    income = annual(facts, OPERATING_INCOME)
    revenues = {concept: annual(facts, concept) for concept in REVENUE_CONCEPTS}
    items = sec_items(facts)
    ends = sorted(
        set(income).union(*revenues.values(), items["ni"]),
        reverse=True,
    )
    window = sec_window(ends)
    first = max(
        REVENUE_CONCEPTS,
        key=lambda c: sum(end in revenues[c] and end in income for end in window),
    )
    order = [first, *(c for c in REVENUE_CONCEPTS if c != first)]
    rows: list[dict[str, object]] = []
    for end in ends:
        concept = next((c for c in order if end in revenues[c]), "")
        revenue = revenues[concept][end][0] if concept else math.nan
        rows.append(
            {
                "end": end,
                "revenue": revenue,
                **{stem: items[stem].get(end, math.nan) for stem in SEC_STEMS},
            }
        )
    return rows


def sec_annual(years: pd.DataFrame) -> dict[str, object]:
    rows = years.sort_values("end", ascending=False)
    total = rows["rev_total"] if "rev_total" in rows.columns else rows["revenue"]
    rows = rows.assign(sales=total.where(total.notna(), rows["revenue"]))
    out: dict[str, object] = dict.fromkeys(ANNUAL_COLUMNS, math.nan)
    if rows.empty:
        return {"fy_end": "", **out}
    newest = rows.iloc[0]
    stop = date.fromisoformat(str(newest["end"]))
    before = [
        row
        for _, row in rows.iloc[1:].iterrows()
        if _within((stop - date.fromisoformat(str(row["end"]))).days, YEAR_DAYS)
    ]
    for year, row in enumerate([newest, *before[:1]]):
        for stem in ANNUAL_ROWS:
            out[f"{stem}_{year}"] = finite(row.get(stem))
    return {"fy_end": str(newest["end"]), **out}


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", frame.to_dict(orient="records"))


@dataclass(frozen=True, slots=True)
class Company:
    ticker: str
    symbol: str
    fetched_on: str
    cells: Mapping[str, Cell]


def _found(value: object) -> float | None:
    if isinstance(value, int | float) and not math.isnan(value):
        return float(value)
    return None


def _filled(info: Mapping[str, object]) -> bool:
    return bool(info.get("longName") or info.get("shortName")) and bool(
        info.get("marketCap")
    )


def _yahoo_cells(info: Mapping[str, object], day: str) -> dict[str, Cell]:
    out: dict[str, Cell] = {}
    for name, key in YAHOO_FIELDS.items():
        value = info.get(key)
        if name == "name" and not value:
            value = info.get("shortName")
        if name in PERCENT_FIELDS:
            found = _found(value)
            value = None if found is None else found * 100
        if value is not None and value != "":
            out[name] = Cell(value, YAHOO, day)
    return out


def _latest(values: Mapping[str, float]) -> tuple[str, float] | None:
    if not values:
        return None
    end = max(values)
    return end, values[end]


def _previous(values: Mapping[str, float], end: str) -> float | None:
    older = [e for e in values if e < end]
    return values[max(older)] if older else None


def sec_cells(facts: Mapping[str, object], market_cap: float | None) -> dict[str, Cell]:
    items = sec_items(facts)
    revenue: dict[str, float] = dict(items.get("rev_total", {}))
    for concept in REVENUE_CONCEPTS:
        for end, (value, _) in annual(facts, concept).items():
            _ = revenue.setdefault(end, value)
    income = _latest(items.get("ni", {}))
    if income is None:
        return {}
    end, profit = income
    out: dict[str, Cell] = {}
    assets, debts = items.get("ta", {}).get(end), items.get("tl", {}).get(end)
    if assets is not None and debts is not None and assets - debts > 0:
        out["roe"] = Cell(profit / (assets - debts) * 100, SEC_FACTS, end)
    sales = revenue.get(end)
    if sales:
        out["net_margin"] = Cell(profit / sales * 100, SEC_FACTS, end)
        before = _previous(revenue, end)
        if before:
            out["revenue_growth"] = Cell((sales / before - 1) * 100, SEC_FACTS, end)
    if market_cap is not None and profit > 0:
        out["per"] = Cell(market_cap / profit, SEC_FACTS, end)
    return out


def _alpaca_yield(
    keys: markets.Keys, symbol: str, today: date, store: prices.PriceStore
) -> Cell | None:
    client = markets.Alpaca(keys, max_wait=limits.INTERACTIVE_WAIT)
    start = today - timedelta(days=CALENDAR_YEAR + 30)
    try:
        dividends, ratios = client.payouts([symbol], start, today)
    except FAILURES:
        return None
    paid = etf.by_ticker(dividends, "rate")
    splits = etf.by_ticker(ratios, "ratio")
    series = store.series(symbol, "split").dropna()
    if series.empty:
        return None
    found = etf.distribution_yield(
        paid.get(symbol, []), splits.get(symbol, []), float(series.iloc[-1]), today
    )
    if found is None:
        return None
    return Cell(found, ALPACA, today.isoformat())


def _facts(symbol: str, agent: str, today: date) -> Mapping[str, object] | None:
    client = sec.Client(agent)
    try:
        archive = sec.Archive(client, today, limits.INTERACTIVE_WAIT)
        cik = archive.cik(symbol)
        return None if cik is None else archive.facts(cik)
    except FAILURES:
        return None
    finally:
        client.close()


def _table(value: object) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _sec_annual(symbol: str, agent: str, today: date) -> dict[str, object]:
    facts = _facts(symbol, agent, today)
    if facts is None:
        return {}
    years = pd.DataFrame(sec_years(facts))
    if years.empty:
        return {}
    return {"source": SEC_FACTS, "currency": USD, **sec_annual(years)}


def _yahoo_annual(symbol: str) -> dict[str, object]:
    try:
        with yahoo.ticker(symbol, limits.INTERACTIVE_WAIT) as stock:
            info = yahoo.stock_info(stock, symbol)
            statements = Statements(
                income=_table(stock.get_income_stmt(freq="yearly")),
                balance=_table(stock.get_balance_sheet(freq="yearly")),
                cash=_table(stock.get_cashflow(freq="yearly")),
            )
    except FAILURES:
        return {}
    row = fundamentals_row(info, statements)
    if not row["fy_end"]:
        return {}
    kept = ("currency", "fy_end", *ANNUAL_COLUMNS)
    return {"source": YAHOO, **{key: row[key] for key in kept}}


def annual_rows(symbol: str, agent: str, today: date) -> dict[str, object]:
    if markets.us_listed(symbol) and agent and agent.isascii():
        found = _sec_annual(symbol, agent, today)
        if found:
            return found
    return _yahoo_annual(symbol)


def _bars(symbol: str, today: date) -> prices.PriceStore:
    store = prices.current()
    with suppress(*FAILURES):
        _ = store.ensure(
            [symbol],
            today - timedelta(days=HISTORY_DAYS),
            today,
            max_wait=limits.INTERACTIVE_WAIT,
        )
    return store


def company_info(
    ticker: str,
    keys: markets.Keys,
    agent: str,
    today: date,
) -> Company:
    saved = memory.current().companies.get(ticker)
    if saved is not None and saved.fetched_on == today.isoformat():
        _ = _bars(saved.symbol, today)
        return saved
    day = today.isoformat()
    cells: dict[str, Cell] = dict(saved.cells) if saved is not None else {}
    symbol = ticker
    try:
        info = yahoo.info(symbol, limits.INTERACTIVE_WAIT)
        if not _filled(info) and ticker.endswith(KOSDAQ):
            symbol = ticker.removesuffix(KOSDAQ) + KOSPI
            info = yahoo.info(symbol, limits.INTERACTIVE_WAIT)
    except FAILURES:
        info, day = {}, saved.fetched_on if saved is not None else ""
    cells.update(_yahoo_cells(info, today.isoformat()))
    us = markets.us_listed(symbol)
    if us and agent and agent.isascii():
        facts = _facts(symbol, agent, today)
        cap = cells.get("market_cap")
        usd = cells.get("currency") is None or cells["currency"].value == USD
        if facts is not None:
            cells.update(sec_cells(facts, _found(cap.value) if cap and usd else None))
    store = _bars(symbol, today)
    if us and keys.alpaca:
        cell = _alpaca_yield(keys, symbol, today, store)
        if cell is not None:
            cells["dividend_yield"] = cell
    found = Company(ticker, symbol, day, cells)
    memory.current().companies.put(ticker, found)
    return found
