from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import pytest

from finagent import limits, memory, prices
from tests.fake_sources import FakeClock

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date


def market_rows(
    tickers: Sequence[str], start: date, days: int = 400, source: str = "Yahoo"
) -> pd.DataFrame:
    days_on = pd.date_range(start, periods=days)  # pyright: ignore[reportUnknownMemberType]
    level = np.tile(100.0 + np.arange(days), len(tickers))
    return pd.DataFrame(
        {
            "ticker": np.repeat(list(tickers), days),
            "date": np.tile(days_on, len(tickers)),
            "open": level,
            "high": level,
            "low": level,
            "close": level,
            "adj_close": level,
            "volume": 1.0,
            "source": source,
        }
    )


@pytest.fixture(autouse=True)
def fresh_prices(monkeypatch: pytest.MonkeyPatch) -> prices.PriceStore:
    def download(
        self: prices.PriceStore,
        tickers: Sequence[str],
        start: date,
        end: date,
        *,
        max_wait: float,
    ) -> pd.DataFrame:
        del max_wait
        days = (end - start).days
        return pd.concat(
            [
                market_rows(
                    [ticker],
                    start,
                    days,
                    prices.route("daily", ticker, self.keys)[0],
                )
                for ticker in tickers
            ],
            ignore_index=True,
        )

    monkeypatch.setattr(prices.PriceStore, "_download", download)
    memory.reset()
    return prices.configure(prices.PriceStore())


@pytest.fixture(autouse=True)
def fresh_limiter() -> FakeClock:
    clock = FakeClock()
    _ = limits.configure(
        limits.Limiter(clock=clock.time, sleep=clock.sleep, asleep=clock.asleep)
    )
    return clock
