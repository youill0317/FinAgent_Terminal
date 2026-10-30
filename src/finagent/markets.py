"""Alpaca daily bars and Massive reference data, under their free plans' limits.

Alpaca's Basic plan answers 200 historical requests a minute and serves the
consolidated (SIP) tape for queries ending at least 15 minutes ago, from
2016; one request carries many symbols. Massive's Basic plan answers 5
requests a minute; its ticker overview has the market cap, currency and
type. Neither has Korean listings, indices or exchange rates: those stay
with Yahoo. Keys come from the env (``.env``); without them the callers
keep Yahoo for everything.
"""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Final, cast, final
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

ALPACA_KEY_ENV: Final = "ALPACA_API_KEY_ID"
ALPACA_SECRET_ENV: Final = "ALPACA_API_SECRET_KEY"  # noqa: S105 - its name
MASSIVE_KEY_ENV: Final = "MASSIVE_API_KEY"

ALPACA_BARS: Final = "https://data.alpaca.markets/v2/stocks/bars"
MASSIVE_TICKER: Final = "https://api.massive.com/v3/reference/tickers/"

ALPACA_PER_MINUTE: Final = 200  # Basic plan, historical requests
MASSIVE_PER_MINUTE: Final = 5  # Basic plan
ALPACA_PAGE: Final = 10_000  # bars per page, across all symbols (the maximum)
ALPACA_SYMBOLS: Final = 100  # symbols per request: none documented, kept short
SIP_DELAY: Final = timedelta(minutes=16)  # free SIP queries end 15+ minutes back
TIMEOUT: Final = 30.0
RETRIES: Final = 3  # 429s waited out before giving up
RETRY_SECONDS: Final = 60.0  # when a 429 names no wait
TOO_MANY_REQUESTS: Final = 429
NOT_FOUND: Final = 404
NEW_YORK: Final = ZoneInfo("America/New_York")


class RateLimitedError(RuntimeError):
    """A provider kept answering 429 after ``RETRIES`` waits."""


@dataclass(frozen=True, slots=True)
class Keys:
    """The keys found in the env; empty strings where missing."""

    alpaca_id: str = ""
    alpaca_secret: str = ""
    massive: str = ""

    @property
    def alpaca(self) -> bool:
        return bool(self.alpaca_id and self.alpaca_secret)


def keys(env: Mapping[str, str]) -> Keys:
    return Keys(
        env.get(ALPACA_KEY_ENV, "").strip(),
        env.get(ALPACA_SECRET_ENV, "").strip(),
        env.get(MASSIVE_KEY_ENV, "").strip(),
    )


@final
class Pace:
    """One pace per provider in this process: request starts ``interval`` apart."""

    def __init__(self, per_minute: int) -> None:
        """Requests at most ``per_minute``, evenly spaced."""
        self.interval = 60 / per_minute
        self._lock = threading.Lock()
        self._last = float("-inf")

    def wait(self, sleep: Callable[[float], None] = time.sleep) -> None:
        with self._lock:
            gap = self._last + self.interval - time.monotonic()
            if gap > 0:
                sleep(gap)
            self._last = time.monotonic()


ALPACA_PACE: Final = Pace(ALPACA_PER_MINUTE)
MASSIVE_PACE: Final = Pace(MASSIVE_PER_MINUTE)


def provider_symbol(ticker: str) -> str:
    """The app's (Yahoo's) ``BRK-B`` is Alpaca's and Massive's ``BRK.B``."""
    return ticker.replace("-", ".")


def app_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def us_listed(ticker: str) -> bool:
    """A US stock or ETF: not an index (``^``), rate (``=X``) or Korean listing."""
    return not (
        ticker.startswith("^") or ticker.endswith(("=X", ".KS", ".KQ")) or "=" in ticker
    )


def _get(
    client: httpx.Client,
    url: str,
    params: Mapping[str, str],
    pace: Pace,
    sleep: Callable[[float], None],
) -> httpx.Response:
    """GET at ``pace``; a 429 waits its ``Retry-After`` (or a minute) and retries."""
    for _ in range(RETRIES):
        pace.wait(sleep)
        response = client.get(url, params=params)
        if response.status_code != TOO_MANY_REQUESTS:
            _ = response.raise_for_status()
            return response
        after = str(response.headers.get("retry-after", ""))
        sleep(float(after) if after.isdigit() else RETRY_SECONDS)
    raise RateLimitedError(f"{httpx.URL(url).host}: 요청 한도 초과")


def bar_end(last: date, now: datetime) -> str:
    """Alpaca's inclusive ``end`` for daily bars through ``last``, SIP-delayed.

    A daily bar is stamped at midnight New York, so the day's end in New
    York keeps ``last``'s bar; free SIP queries end ``SIP_DELAY`` before now.
    """
    close = datetime.combine(last + timedelta(days=1), datetime.min.time(), NEW_YORK)
    end = min(close - timedelta(seconds=1), now - SIP_DELAY)
    return end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_bars(payload: Mapping[str, object]) -> pd.DataFrame:
    """One page of ``/v2/stocks/bars`` as rows of ticker, date, o, h, l, c, v."""
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
    """Daily bars for many US symbols per request (``/v2/stocks/bars``)."""

    keys: Keys
    client: httpx.Client | None = None
    sleep: Callable[[float], None] = time.sleep

    def bars(
        self,
        tickers: Sequence[str],
        start: date,
        last: date,
        adjustment: str,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        """Rows of ticker, date and OHLCV from ``start`` through ``last``.

        ``adjustment`` is Alpaca's: ``split`` (as Yahoo's Close) or ``all``
        (splits and dividends, as Yahoo's Adj Close).
        """
        client = self.client or httpx.Client(timeout=TIMEOUT)
        client.headers.update(
            {
                "APCA-API-KEY-ID": self.keys.alpaca_id,
                "APCA-API-SECRET-KEY": self.keys.alpaca_secret,
            }
        )
        end = bar_end(last, now or datetime.now(UTC))
        pages: list[pd.DataFrame] = []
        try:
            for i in range(0, len(tickers), ALPACA_SYMBOLS):
                chunk = tickers[i : i + ALPACA_SYMBOLS]
                params = {
                    "symbols": ",".join(provider_symbol(t) for t in chunk),
                    "timeframe": "1Day",
                    "start": start.isoformat(),
                    "end": end,
                    "adjustment": adjustment,
                    "feed": "sip",
                    "limit": str(ALPACA_PAGE),
                }
                while True:
                    payload = cast(
                        "Mapping[str, object]",
                        _get(
                            client, ALPACA_BARS, params, ALPACA_PACE, self.sleep
                        ).json(),
                    )
                    pages.append(parse_bars(payload))
                    token = payload.get("next_page_token")
                    if not token:
                        break
                    params = {**params, "page_token": str(token)}
        finally:
            if self.client is None:
                client.close()
        return pd.concat(pages, ignore_index=True) if pages else parse_bars({})


@dataclass(frozen=True, slots=True)
class Overview:
    """Massive's ticker overview: market cap in ``currency``, NaN where none."""

    market_cap: float
    currency: str
    kind: str  # Massive's type: CS (common stock), ETF, ADRC, ...


@dataclass(slots=True)
class Massive:
    """Reference data, one ticker per request (``/v3/reference/tickers/{t}``)."""

    keys: Keys
    client: httpx.Client | None = None
    sleep: Callable[[float], None] = time.sleep

    def overview(self, ticker: str) -> Overview | None:
        """``ticker``'s overview; None for a ticker Massive does not know."""
        client = self.client or httpx.Client(timeout=TIMEOUT)
        client.headers["Authorization"] = f"Bearer {self.keys.massive}"
        try:
            response = _get(
                client,
                MASSIVE_TICKER + provider_symbol(ticker),
                {},
                MASSIVE_PACE,
                self.sleep,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code == NOT_FOUND:
                return None
            raise
        finally:
            if self.client is None:
                client.close()
        payload = cast("Mapping[str, object]", response.json())
        result = cast("Mapping[str, object]", payload.get("results") or {})
        cap = result.get("market_cap")
        return Overview(
            float(cap) if isinstance(cap, int | float) and cap > 0 else float("nan"),
            str(result.get("currency_name") or "").upper(),
            str(result.get("type") or ""),
        )
