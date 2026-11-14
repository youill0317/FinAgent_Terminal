# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""Alpaca Market Data API(미국 주식 일봉·액면분할·배당) 클라이언트."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Final, cast
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from finagent import limits
from finagent.limits import BACKGROUND_WAIT

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

ALPACA_KEY_ENV: Final = "ALPACA_API_KEY_ID"
ALPACA_SECRET_ENV: Final = "ALPACA_API_SECRET_KEY"  # noqa: S105  # 환경변수 키 이름 상수(시크릿 리터럴 아님)

ALPACA_BARS: Final = "https://data.alpaca.markets/v2/stocks/bars"
ALPACA_ACTIONS: Final = "https://data.alpaca.markets/v1/corporate-actions"

ALPACA_PAGE: Final = 10_000
ALPACA_ACTIONS_PAGE: Final = 1000
SPLIT_TYPES: Final = "forward_split,reverse_split,stock_dividend"
ALPACA_SYMBOLS: Final = 100
SIP_DELAY: Final = timedelta(minutes=30)
ALPACA_START: Final = date(2016, 1, 1)
TIMEOUT: Final = 30.0
NEW_YORK: Final = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class Keys:
    alpaca_id: str = ""
    alpaca_secret: str = ""

    @property
    def alpaca(self) -> bool:
        return bool(self.alpaca_id and self.alpaca_secret)


def keys(env: Mapping[str, str]) -> Keys:
    return Keys(
        env.get(ALPACA_KEY_ENV, "").strip(),
        env.get(ALPACA_SECRET_ENV, "").strip(),
    )


def provider_symbol(ticker: str) -> str:
    return ticker.replace("-", ".")


def app_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def us_listed(ticker: str) -> bool:
    return not (ticker.startswith("^") or "=" in ticker or "." in ticker)


def _get(
    client: httpx.Client,
    url: str,
    params: Mapping[str, str],
    headers: Mapping[str, str],
    max_wait: float,
) -> httpx.Response:
    response = limits.send(
        "ALPACA", lambda: client.get(url, params=params, headers=headers), max_wait
    )
    _ = response.raise_for_status()
    return response


def bar_end(last: date, now: datetime) -> str:
    close = datetime.combine(last + timedelta(days=1), datetime.min.time(), NEW_YORK)
    end = min(close - timedelta(seconds=1), now - SIP_DELAY)
    return end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_bars(payload: Mapping[str, object]) -> pd.DataFrame:
    bars = cast("Mapping[str, list[Mapping[str, object]]]", payload.get("bars") or {})
    rows = [
        {
            "ticker": app_symbol(symbol),
            "date": pd.Timestamp(str(bar["t"])).tz_convert(NEW_YORK).date(),
            "open": bar["o"],
            "high": bar["h"],
            "low": bar["l"],
            "close": bar["c"],
            "volume": bar["v"],
        }
        for symbol, series in bars.items()
        for bar in series
    ]
    frame = pd.DataFrame(
        rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"]
    )
    frame["date"] = pd.to_datetime(cast("pd.Series", frame["date"]))
    return frame


@dataclass(slots=True)
class Alpaca:
    keys: Keys
    client: httpx.Client | None = None
    max_wait: float = BACKGROUND_WAIT

    def bars(
        self,
        tickers: Sequence[str],
        start: date,
        last: date,
        adjustment: str,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        params = {
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": bar_end(last, now or datetime.now(UTC)),
            "adjustment": adjustment,
            "feed": "sip",
            "limit": str(ALPACA_PAGE),
        }
        pages = [parse_bars(page) for page in self._pages(ALPACA_BARS, params, tickers)]
        return pd.concat(pages, ignore_index=True) if pages else parse_bars({})

    def splits(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return _split_table(self._actions(tickers, SPLIT_TYPES, start, end))

    def payouts(
        self, tickers: Sequence[str], start: date, end: date
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pages = self._actions(tickers, f"cash_dividend,{SPLIT_TYPES}", start, end)
        paid = [row for page in pages for row in parse_cash_dividends(page)]
        return (
            pd.DataFrame(paid, columns=["ticker", "ex_date", "rate"]),
            _split_table(pages),
        )

    def _actions(
        self, tickers: Sequence[str], types: str, start: date, end: date
    ) -> list[Mapping[str, object]]:
        params = {
            "types": types,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": str(ALPACA_ACTIONS_PAGE),
        }
        return list(self._pages(ALPACA_ACTIONS, params, tickers))

    def _pages(
        self, url: str, params: Mapping[str, str], tickers: Sequence[str]
    ) -> Generator[Mapping[str, object]]:
        client = self.client or httpx.Client(timeout=TIMEOUT)
        headers = {
            "APCA-API-KEY-ID": self.keys.alpaca_id,
            "APCA-API-SECRET-KEY": self.keys.alpaca_secret,
        }
        try:
            for i in range(0, len(tickers), ALPACA_SYMBOLS):
                chunk = tickers[i : i + ALPACA_SYMBOLS]
                ask = {**params, "symbols": ",".join(provider_symbol(t) for t in chunk)}
                while True:
                    payload = cast(
                        "Mapping[str, object]",
                        _get(client, url, ask, headers, self.max_wait).json(),
                    )
                    yield payload
                    token = payload.get("next_page_token")
                    if not token:
                        break
                    ask = {**ask, "page_token": str(token)}
        finally:
            if self.client is None:
                client.close()


def _split_table(pages: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = [row for page in pages for row in parse_splits(page)]
    return pd.DataFrame(rows, columns=["ticker", "ex_date", "ratio"])


def parse_splits(payload: Mapping[str, object]) -> list[dict[str, object]]:
    actions = cast(
        "Mapping[str, list[Mapping[str, object]]]",
        payload.get("corporate_actions") or {},
    )
    return [
        {
            "ticker": app_symbol(str(item["symbol"])),
            "ex_date": str(item["ex_date"]),
            "ratio": float(cast("float", item["rate"]))
            if kind == "stock_dividends"
            else float(cast("float", item["new_rate"]))
            / float(cast("float", item["old_rate"])),
        }
        for kind in ("forward_splits", "reverse_splits", "stock_dividends")
        for item in actions.get(kind) or []
    ]


def parse_cash_dividends(payload: Mapping[str, object]) -> list[dict[str, object]]:
    actions = cast(
        "Mapping[str, list[Mapping[str, object]]]",
        payload.get("corporate_actions") or {},
    )
    return [
        {
            "ticker": app_symbol(str(item["symbol"])),
            "ex_date": str(item["ex_date"]),
            "rate": float(cast("float", item["rate"])),
        }
        for item in actions.get("cash_dividends") or []
    ]
