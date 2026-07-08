from __future__ import annotations

from typing import ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic_core import PydanticCustomError


class FinnhubQuotePayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    c: float | None = None
    d: float | None = None
    dp: float | None = None
    pc: float | None = None
    t: int | None = None

    @model_validator(mode="after")
    def _has_quote_payload(self) -> Self:
        if all(value is None for value in (self.c, self.d, self.dp, self.pc, self.t)):
            error_type = "finnhub_quote_payload"
            message = "missing Finnhub quote fields"
            raise PydanticCustomError(error_type, message)
        return self


class FinnhubNewsPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    headline: str
    summary: str | None = None
    url: str | None = None
    datetime: int | None = None


class FinnhubSentimentRow(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    mention: int | None = None
    positive_score: float | None = Field(default=None, alias="positiveScore")
    negative_score: float | None = Field(default=None, alias="negativeScore")

    @model_validator(mode="after")
    def _has_sentiment_metrics(self) -> Self:
        if all(
            value is None
            for value in (self.mention, self.positive_score, self.negative_score)
        ):
            error_type = "finnhub_sentiment_row"
            message = "missing Finnhub sentiment metrics"
            raise PydanticCustomError(error_type, message)
        return self


class FinnhubSentimentPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    symbol: str | None = None
    reddit: tuple[FinnhubSentimentRow, ...] = ()
    twitter: tuple[FinnhubSentimentRow, ...] = ()

    @model_validator(mode="after")
    def _has_sentiment_payload(self) -> Self:
        if len(self.reddit) == 0 and len(self.twitter) == 0:
            error_type = "finnhub_sentiment_payload"
            message = "missing Finnhub sentiment rows"
            raise PydanticCustomError(error_type, message)
        return self


class FinnhubRecommendationPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    symbol: str | None = None
    period: str
    strong_buy: int | None = Field(default=None, alias="strongBuy")
    buy: int | None = None
    hold: int | None = None
    sell: int | None = None
    strong_sell: int | None = Field(default=None, alias="strongSell")


class FinnhubEpsRow(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    period: str
    eps_avg: float | None = Field(default=None, alias="epsAvg")
    number_analysts: int | None = Field(default=None, alias="numberAnalysts")

    @model_validator(mode="after")
    def _has_eps_metrics(self) -> Self:
        if self.eps_avg is None and self.number_analysts is None:
            error_type = "finnhub_eps_row"
            message = "missing Finnhub EPS metrics"
            raise PydanticCustomError(error_type, message)
        return self


class FinnhubEpsPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    symbol: str | None = None
    data: tuple[FinnhubEpsRow, ...] = ()

    @model_validator(mode="after")
    def _has_eps_payload(self) -> Self:
        if len(self.data) == 0:
            error_type = "finnhub_eps_payload"
            message = "missing Finnhub EPS rows"
            raise PydanticCustomError(error_type, message)
        return self


FINNHUB_QUOTE_ADAPTER: Final[TypeAdapter[FinnhubQuotePayload]] = TypeAdapter(
    FinnhubQuotePayload,
)
FINNHUB_NEWS_ADAPTER: Final[TypeAdapter[tuple[FinnhubNewsPayload, ...]]] = TypeAdapter(
    tuple[FinnhubNewsPayload, ...],
)
FINNHUB_SENTIMENT_ADAPTER: Final[TypeAdapter[FinnhubSentimentPayload]] = TypeAdapter(
    FinnhubSentimentPayload,
)
FINNHUB_RECOMMENDATION_ADAPTER: Final[
    TypeAdapter[tuple[FinnhubRecommendationPayload, ...]]
] = TypeAdapter(tuple[FinnhubRecommendationPayload, ...])
FINNHUB_EPS_ADAPTER: Final[TypeAdapter[FinnhubEpsPayload]] = TypeAdapter(
    FinnhubEpsPayload,
)
