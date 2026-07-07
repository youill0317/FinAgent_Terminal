from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol

import pandas as pd  # noqa: PANDAS_OK

from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    MarketScalar,
    OhlcvFrame,
    Profile,
    Quote,
)
from finagent_terminal.symbols import normalize_symbol

OHLCV_COLUMNS: Final = ("Open", "High", "Low", "Close", "Volume")
type Clock = Callable[[], datetime]


class TickerClient(Protocol):
    @property
    def fast_info(self) -> Mapping[str, MarketScalar]: ...

    @property
    def info(self) -> Mapping[str, MarketScalar]: ...

    def history(
        self,
        *,
        period: str,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame: ...


type TickerFactory = Callable[[str], TickerClient]


def utc_now() -> datetime:
    return datetime.now(UTC)


if TYPE_CHECKING:

    def default_ticker_factory(symbol: str) -> TickerClient:
        raise NotImplementedError(symbol)

else:
    import yfinance as yf

    def default_ticker_factory(symbol: str) -> TickerClient:
        return yf.Ticker(symbol)


@dataclass(frozen=True, slots=True)
class YFinanceProvider:
    ticker_factory: TickerFactory = default_ticker_factory
    clock: Clock = utc_now

    def fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        normalized_symbol = normalize_symbol(symbol)
        fetched_at = self.clock()
        ticker = self.ticker_factory(normalized_symbol)
        frame = ticker.history(period=period, interval=interval, auto_adjust=False)
        return MarketDataSnapshot(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
            fetched_at=fetched_at,
            freshness=FreshnessState.FRESH,
            quote=normalize_quote(
                normalized_symbol,
                ticker.fast_info,
                frame,
                fetched_at,
            ),
            history=normalize_history(
                normalized_symbol,
                period,
                interval,
                frame,
                fetched_at,
            ),
            profile=normalize_profile(
                normalized_symbol,
                ticker.info,
                ticker.fast_info,
                fetched_at,
            ),
        )


def normalize_quote(
    symbol: str,
    fast_info: Mapping[str, MarketScalar],
    frame: pd.DataFrame,
    fetched_at: datetime,
) -> Quote:
    last_price = _first_number(
        fast_info,
        ("lastPrice", "last_price", "regularMarketPrice", "currentPrice"),
    )
    if last_price is None:
        last_price = _last_frame_number(frame, "Close")

    previous_close = _first_number(
        fast_info,
        ("previousClose", "previous_close", "regularMarketPreviousClose"),
    )
    change = _change(last_price, previous_close)
    return Quote(
        symbol=symbol,
        last_price=last_price,
        previous_close=previous_close,
        change=change,
        change_percent=_change_percent(change, previous_close),
        currency=_first_string(fast_info, ("currency",)),
        market_cap=_first_int(fast_info, ("marketCap", "market_cap")),
        volume=_first_int(fast_info, ("volume", "lastVolume")),
        day_high=_first_number(
            fast_info,
            ("dayHigh", "day_high", "regularMarketDayHigh"),
        ),
        day_low=_first_number(fast_info, ("dayLow", "day_low", "regularMarketDayLow")),
        fetched_at=fetched_at,
    )


def normalize_history(
    symbol: str,
    period: str,
    interval: str,
    frame: pd.DataFrame,
    fetched_at: datetime,
) -> OhlcvFrame:
    if frame.empty:
        normalized = pd.DataFrame(columns=OHLCV_COLUMNS)
    else:
        source = frame.copy()
        source.index = pd.to_datetime(source.index, errors="coerce")
        source = source[source.index.notna()]
        columns = {
            column: pd.to_numeric(
                source[column]
                if column in source.columns
                else pd.Series(index=source.index),
                errors="coerce",
            )
            for column in OHLCV_COLUMNS
        }
        normalized = pd.DataFrame(columns, index=source.index).dropna(how="all")

    return OhlcvFrame(
        symbol=symbol,
        period=period,
        interval=interval,
        frame=normalized,
        rows=len(normalized),
        fetched_at=fetched_at,
    )


def normalize_profile(
    symbol: str,
    info: Mapping[str, MarketScalar],
    fast_info: Mapping[str, MarketScalar],
    fetched_at: datetime,
) -> Profile:
    return Profile(
        symbol=symbol,
        name=_first_string(info, ("longName", "shortName")),
        sector=_first_string(info, ("sector",)),
        industry=_first_string(info, ("industry",)),
        website=_first_string(info, ("website",)),
        market_cap=_first_int(info, ("marketCap", "market_cap"))
        or _first_int(fast_info, ("marketCap", "market_cap")),
        trailing_pe=_first_number(info, ("trailingPE", "trailingPe")),
        forward_pe=_first_number(info, ("forwardPE", "forwardPe")),
        dividend_yield=_first_number(info, ("dividendYield", "yield")),
        beta=_first_number(info, ("beta",)),
        fetched_at=fetched_at,
    )


def _first_number(
    values: Mapping[str, MarketScalar],
    keys: tuple[str, ...],
) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            return float(value)
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _first_int(
    values: Mapping[str, MarketScalar],
    keys: tuple[str, ...],
) -> int | None:
    value = _first_number(values, keys)
    if value is None:
        return None
    return int(value)


def _first_string(
    values: Mapping[str, MarketScalar],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value.strip() != "":
            return value
    return None


def _last_frame_number(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.astype("float64").iloc[-1])


def _change(last_price: float | None, previous_close: float | None) -> float | None:
    if last_price is None or previous_close is None:
        return None
    return last_price - previous_close


def _change_percent(change: float | None, previous_close: float | None) -> float | None:
    if change is None or previous_close is None or previous_close == 0.0:
        return None
    return change / previous_close
