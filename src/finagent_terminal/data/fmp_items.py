from __future__ import annotations

from typing import TYPE_CHECKING

from finagent_terminal.data.source_models import SourceItem
from finagent_terminal.data.vendor_common import number, parse_datetime, summary

if TYPE_CHECKING:
    from finagent_terminal.data.fmp_payloads import (
        FmpBalancePayload,
        FmpCashFlowPayload,
        FmpEarningsCalendarPayload,
        FmpIncomePayload,
        FmpNewsPayload,
        FmpProfilePayload,
    )


def profile_items(
    rows: tuple[FmpProfilePayload, ...],
    source: str,
) -> tuple[SourceItem, ...]:
    return tuple(
        SourceItem(
            source=source,
            title=f"Profile: {row.company_name or row.symbol or 'company'}",
            summary=summary(
                row.sector,
                row.industry,
                number("market cap", row.market_cap),
            ),
        )
        for row in rows[:1]
    )


def income_item(row: FmpIncomePayload, symbol: str, source: str) -> SourceItem:
    return SourceItem(
        source=source,
        title=(
            f"Income statement {_period(row.calendar_year, row.date)}: "
            f"{row.symbol or symbol}"
        ),
        summary=summary(
            number("revenue", row.revenue),
            number("net income", row.net_income),
            number("eps", row.eps),
            number("diluted eps", row.eps_diluted),
        ),
    )


def balance_item(row: FmpBalancePayload, symbol: str, source: str) -> SourceItem:
    return SourceItem(
        source=source,
        title=(
            f"Balance sheet {_period(row.calendar_year, row.date)}: "
            f"{row.symbol or symbol}"
        ),
        summary=summary(
            number("assets", row.total_assets),
            number("liabilities", row.total_liabilities),
            number("equity", row.total_equity),
            number("cash", row.cash_and_cash_equivalents),
            number("debt", row.total_debt),
        ),
    )


def cash_item(row: FmpCashFlowPayload, symbol: str, source: str) -> SourceItem:
    return SourceItem(
        source=source,
        title=(
            f"Cash flow {_period(row.calendar_year, row.date)}: "
            f"{row.symbol or symbol}"
        ),
        summary=summary(
            number("operating cash flow", row.operating_cash_flow),
            number("free cash flow", row.free_cash_flow),
            number("capital expenditure", row.capital_expenditure),
            number(
                "net operating cash",
                row.net_cash_provided_by_operating_activities,
            ),
        ),
    )


def earnings_item(
    row: FmpEarningsCalendarPayload,
    symbol: str,
    source: str,
) -> SourceItem:
    return SourceItem(
        source=source,
        title=f"Earnings calendar {row.date or 'latest'}: {row.symbol or symbol}",
        summary=summary(
            number("eps", row.eps),
            number("estimated eps", row.eps_estimated),
            number("revenue", row.revenue),
            number("estimated revenue", row.revenue_estimated),
        ),
        published_at=parse_datetime(row.date),
    )


def news_items(rows: tuple[FmpNewsPayload, ...], source: str) -> tuple[SourceItem, ...]:
    return tuple(
        SourceItem(
            source=source,
            title=f"News: {row.title}",
            summary=row.text,
            url=row.url,
            published_at=parse_datetime(row.published_date),
        )
        for row in rows[:3]
    )


def _period(calendar_year: str | None, date: str | None) -> str:
    return calendar_year or date or "latest"
