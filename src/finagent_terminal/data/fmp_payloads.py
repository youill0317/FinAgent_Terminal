from __future__ import annotations

from typing import ClassVar, Final, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)
from pydantic_core import PydanticCustomError


def _has_any(*values: float | str | None) -> bool:
    return any(value is not None for value in values)


class FmpProfilePayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    symbol: str | None = None
    company_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("companyName", "company_name"),
    )
    sector: str | None = None
    industry: str | None = None
    market_cap: int | None = Field(
        default=None,
        validation_alias=AliasChoices("mktCap", "marketCap", "market_cap"),
    )

    @model_validator(mode="after")
    def _has_profile_identity(self) -> Self:
        if not _has_any(self.symbol, self.company_name):
            error_type = "fmp_profile_payload"
            message = "missing FMP profile identity"
            raise PydanticCustomError(error_type, message)
        return self


class FmpIncomePayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    symbol: str | None = None
    date: str | None = None
    calendar_year: str | None = Field(
        default=None,
        validation_alias=AliasChoices("calendarYear", "calendar_year"),
    )
    revenue: int | None = None
    net_income: int | None = Field(
        default=None,
        validation_alias=AliasChoices("netIncome", "net_income"),
    )
    eps: float | None = None
    eps_diluted: float | None = Field(
        default=None,
        validation_alias=AliasChoices("epsDiluted", "epsdiluted", "eps_diluted"),
    )

    @model_validator(mode="after")
    def _has_income_metrics(self) -> Self:
        if not _has_any(self.revenue, self.net_income, self.eps, self.eps_diluted):
            error_type = "fmp_income_payload"
            message = "missing FMP income statement metrics"
            raise PydanticCustomError(error_type, message)
        return self


class FmpBalancePayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    symbol: str | None = None
    date: str | None = None
    calendar_year: str | None = Field(
        default=None,
        validation_alias=AliasChoices("calendarYear", "calendar_year"),
    )
    total_assets: int | None = Field(
        default=None,
        validation_alias=AliasChoices("totalAssets", "total_assets"),
    )
    total_liabilities: int | None = Field(
        default=None,
        validation_alias=AliasChoices("totalLiabilities", "total_liabilities"),
    )
    total_equity: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "totalEquity",
            "totalStockholdersEquity",
            "total_equity",
            "total_stockholders_equity",
        ),
    )
    cash_and_cash_equivalents: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "cashAndCashEquivalents",
            "cash_and_cash_equivalents",
        ),
    )
    total_debt: int | None = Field(
        default=None,
        validation_alias=AliasChoices("totalDebt", "total_debt"),
    )

    @model_validator(mode="after")
    def _has_balance_metrics(self) -> Self:
        if not _has_any(
            self.total_assets,
            self.total_liabilities,
            self.total_equity,
            self.cash_and_cash_equivalents,
            self.total_debt,
        ):
            error_type = "fmp_balance_payload"
            message = "missing FMP balance sheet metrics"
            raise PydanticCustomError(error_type, message)
        return self


class FmpCashFlowPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    symbol: str | None = None
    date: str | None = None
    calendar_year: str | None = Field(
        default=None,
        validation_alias=AliasChoices("calendarYear", "calendar_year"),
    )
    operating_cash_flow: int | None = Field(
        default=None,
        validation_alias=AliasChoices("operatingCashFlow", "operating_cash_flow"),
    )
    free_cash_flow: int | None = Field(
        default=None,
        validation_alias=AliasChoices("freeCashFlow", "free_cash_flow"),
    )
    capital_expenditure: int | None = Field(
        default=None,
        validation_alias=AliasChoices("capitalExpenditure", "capital_expenditure"),
    )
    net_cash_provided_by_operating_activities: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "netCashProvidedByOperatingActivities",
            "net_cash_provided_by_operating_activities",
        ),
    )

    @model_validator(mode="after")
    def _has_cash_flow_metrics(self) -> Self:
        if not _has_any(
            self.operating_cash_flow,
            self.free_cash_flow,
            self.capital_expenditure,
            self.net_cash_provided_by_operating_activities,
        ):
            error_type = "fmp_cash_flow_payload"
            message = "missing FMP cash flow metrics"
            raise PydanticCustomError(error_type, message)
        return self


class FmpEarningsCalendarPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    symbol: str | None = None
    date: str | None = None
    eps: float | None = None
    eps_estimated: float | None = Field(
        default=None,
        validation_alias=AliasChoices("epsEstimated", "eps_estimated"),
    )
    revenue: float | None = None
    revenue_estimated: float | None = Field(
        default=None,
        validation_alias=AliasChoices("revenueEstimated", "revenue_estimated"),
    )

    @model_validator(mode="after")
    def _has_calendar_payload(self) -> Self:
        if (
            self.symbol is None
            or self.date is None
            or not _has_any(
                self.eps,
                self.eps_estimated,
                self.revenue,
                self.revenue_estimated,
            )
        ):
            error_type = "fmp_earnings_calendar_payload"
            message = "missing FMP earnings calendar fields"
            raise PydanticCustomError(error_type, message)
        return self


class FmpNewsPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    symbol: str | None = None
    title: str = Field(min_length=1)
    text: str | None = None
    url: str | None = None
    published_date: str | None = Field(
        default=None,
        validation_alias=AliasChoices("publishedDate", "published_date", "date"),
    )


FMP_PROFILE_ADAPTER: Final[TypeAdapter[tuple[FmpProfilePayload, ...]]] = TypeAdapter(
    tuple[FmpProfilePayload, ...],
)
FMP_INCOME_ADAPTER: Final[TypeAdapter[tuple[FmpIncomePayload, ...]]] = TypeAdapter(
    tuple[FmpIncomePayload, ...],
)
FMP_BALANCE_ADAPTER: Final[TypeAdapter[tuple[FmpBalancePayload, ...]]] = TypeAdapter(
    tuple[FmpBalancePayload, ...],
)
FMP_CASH_FLOW_ADAPTER: Final[TypeAdapter[tuple[FmpCashFlowPayload, ...]]] = TypeAdapter(
    tuple[FmpCashFlowPayload, ...],
)
FMP_EARNINGS_CALENDAR_ADAPTER: Final[
    TypeAdapter[tuple[FmpEarningsCalendarPayload, ...]]
] = TypeAdapter(tuple[FmpEarningsCalendarPayload, ...])
FMP_NEWS_ADAPTER: Final[TypeAdapter[tuple[FmpNewsPayload, ...]]] = TypeAdapter(
    tuple[FmpNewsPayload, ...],
)
