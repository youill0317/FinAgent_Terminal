from __future__ import annotations

from typing import TYPE_CHECKING

from finagent_terminal.data.source_models import SourceItem
from finagent_terminal.data.vendor_common import number, parse_datetime, summary

if TYPE_CHECKING:
    from finagent_terminal.data.vendor_payloads import (
        FinnhubEpsPayload,
        FinnhubNewsPayload,
        FinnhubQuotePayload,
        FinnhubRecommendationPayload,
        FinnhubSentimentPayload,
    )


def quote_item(
    symbol: str,
    quote: FinnhubQuotePayload,
    source: str,
) -> SourceItem:
    return SourceItem(
        source=source,
        title=f"Quote: {symbol}",
        summary=summary(
            number("last", quote.c),
            number("change", quote.d),
            number("change %", quote.dp),
        ),
        published_at=parse_datetime(quote.t),
    )


def news_items(
    rows: tuple[FinnhubNewsPayload, ...],
    source: str,
) -> tuple[SourceItem, ...]:
    return tuple(
        SourceItem(
            source=source,
            title=f"News: {row.headline}",
            summary=row.summary,
            url=row.url,
            published_at=parse_datetime(row.datetime),
        )
        for row in rows[:3]
    )


def sentiment_items(
    symbol: str,
    payload: FinnhubSentimentPayload,
    source: str,
) -> tuple[SourceItem, ...]:
    first = next(iter((*payload.reddit, *payload.twitter)), None)
    if first is None:
        return ()
    return (
        SourceItem(
            source=source,
            title=f"Sentiment: {payload.symbol or symbol}",
            summary=summary(
                number("mentions", first.mention),
                number("positive", first.positive_score),
                number("negative", first.negative_score),
            ),
        ),
    )


def recommendation_items(
    symbol: str,
    rows: tuple[FinnhubRecommendationPayload, ...],
    source: str,
) -> tuple[SourceItem, ...]:
    return tuple(
        SourceItem(
            source=source,
            title=f"Recommendations {row.period}: {row.symbol or symbol}",
            summary=summary(
                number("strong buy", row.strong_buy),
                number("buy", row.buy),
                number("hold", row.hold),
                number("sell", row.sell),
                number("strong sell", row.strong_sell),
            ),
        )
        for row in rows[:1]
    )


def eps_items(
    symbol: str,
    payload: FinnhubEpsPayload,
    source: str,
) -> tuple[SourceItem, ...]:
    return tuple(
        SourceItem(
            source=source,
            title=f"EPS estimate {row.period}: {payload.symbol or symbol}",
            summary=summary(
                number("avg", row.eps_avg),
                number("analysts", row.number_analysts),
            ),
        )
        for row in payload.data[:1]
    )
