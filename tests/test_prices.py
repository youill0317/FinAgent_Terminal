# pyright: reportUnknownLambdaType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
# pyright: reportUnknownVariableType=false, reportArgumentType=false

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, cast, final

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import pytest

from finagent import markets, portfolio, prices

if TYPE_CHECKING:
    from collections.abc import Sequence

KEYS = markets.Keys("id", "secret")
DAY = cast("pd.Timestamp", pd.Timestamp("2026-10-29"))


def rows(tickers: list[str], source: str, day: pd.Timestamp = DAY) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ticker": tickers,
            "date": [day] * len(tickers),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adj_close": 1.0,
            "volume": 1.0,
            "source": source,
        }
    )
    return frame.reindex(columns=list(prices.ROW_COLUMNS))


def test_daily_joins_yahoo_s_years_before_alpaca_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[tuple[list[str], date, date | None]] = []
    old = pd.to_datetime(["2015-12-30", "2015-12-31", "2016-01-04", "2016-01-05"])
    new = pd.to_datetime(["2016-01-04", "2016-01-05", "2026-10-29"])

    def alpaca(
        _alpaca: object, tickers: list[str], _start: date, _last: date
    ) -> pd.DataFrame:
        spy = pd.DataFrame({"ticker": "SPY", "date": new, "close": [20.0, 22.0, 30.0]})
        newco = pd.DataFrame({"ticker": "NEWCO", "date": new[2:], "close": [5.0]})
        frame = pd.concat([spy, newco], ignore_index=True)
        frame = cast("pd.DataFrame", frame[frame["ticker"].isin(tickers)])
        frame = frame.assign(adj_close=frame["close"] / 4, source="Alpaca")
        return frame.reindex(columns=list(prices.ROW_COLUMNS))

    def yahoo(
        tickers: list[str],
        start: date,
        end: date | None = None,
        _max_wait: float = 0.0,
    ) -> pd.DataFrame:
        asked.append((list(tickers), start, end))
        frames = [
            pd.DataFrame(
                {
                    "ticker": t,
                    "date": old,
                    "close": [8.0, 9.0, 10.0, 11.0],
                    "adj_close": [4.0, 4.5, 5.0, 5.5],
                    "source": "Yahoo",
                }
            )
            for t in tickers
        ]
        return pd.concat(frames, ignore_index=True).reindex(
            columns=list(prices.ROW_COLUMNS)
        )

    monkeypatch.setattr(prices, "alpaca_rows", alpaca)
    monkeypatch.setattr(prices, "yahoo_rows", yahoo)

    got = portfolio.wide_prices(
        prices.daily(["SPY", "NEWCO"], date(2007, 1, 1), None, KEYS)
    )

    assert asked == [(["SPY"], date(2007, 1, 1), date(2016, 1, 9))]
    assert list(got.close["SPY"].dropna()) == pytest.approx(
        [16.0, 18.0, 20.0, 22.0, 30.0]
    )
    assert list(got.adjusted["SPY"].dropna()) == pytest.approx(
        [4.0, 4.5, 5.0, 5.5, 7.5]
    )
    assert got.close["NEWCO"].dropna().tolist() == [5.0]

    asked.clear()
    _ = prices.daily(["SPY"], date(2016, 1, 1), None, KEYS)
    assert asked == []


def test_a_yahoo_outage_drops_the_truncated_history_instead_of_saving_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    new = pd.to_datetime(["2016-01-04", "2026-10-29"])

    def alpaca(
        _alpaca: object, _tickers: list[str], _start: date, _last: date
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": "SPY",
                "date": new,
                "close": [20.0, 30.0],
                "adj_close": [5.0, 7.5],
                "source": "Alpaca",
            }
        ).reindex(columns=list(prices.ROW_COLUMNS))

    monkeypatch.setattr(prices, "alpaca_rows", alpaca)
    monkeypatch.setattr(
        prices, "yahoo_rows", lambda *_args, **_kwargs: prices.empty_rows()
    )

    got = prices.daily(["SPY"], date(2007, 1, 1), None, KEYS)

    assert got.empty


NOON = datetime(2026, 11, 6, 22, 0, tzinfo=UTC)
FIRST = date(2026, 10, 1)
THROUGH = date(2026, 10, 20)


def bars(
    tickers: Sequence[str],
    start: date,
    end: date,
    *,
    adj: float = 1.0,
    close: float = 10.0,
) -> pd.DataFrame:
    days = pd.date_range(start, end - timedelta(days=1))
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": day,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adj_close": close * adj,
                "volume": 1.0,
                "source": "Alpaca",
            }
            for ticker in tickers
            for day in days
        ]
    )


@final
class Provider:
    def __init__(self, adj: float = 1.0, close: float = 10.0) -> None:
        self.adj = adj
        self.close = close
        self.calls: list[tuple[tuple[str, ...], date, date]] = []

    def __call__(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        self.calls.append((tuple(tickers), start, end))
        return bars(tickers, start, end, adj=self.adj, close=self.close)


@pytest.fixture
def store() -> prices.PriceStore:
    return prices.PriceStore(clock=lambda: NOON)


def test_a_changed_overlap_refetches_the_whole_ticker(
    store: prices.PriceStore,
) -> None:
    first = Provider()
    _ = store.ensure(["AAA", "BBB"], FIRST, THROUGH, fetch=first)

    def source(tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        calls.append((tuple(tickers), start))
        moved = [t for t in tickers if t == "AAA"]
        return pd.concat(
            [
                bars(moved, start, end, adj=0.9),
                bars([t for t in tickers if t != "AAA"], start, end),
            ],
            ignore_index=True,
        )

    calls: list[tuple[tuple[str, ...], date]] = []
    later = THROUGH + timedelta(days=2)
    _ = store.ensure(["AAA", "BBB"], FIRST, later, force=True, fetch=source)

    assert calls[0] == (("AAA", "BBB"), THROUGH - timedelta(days=prices.OVERLAP_DAYS))
    assert calls[1] == (("AAA",), FIRST)
    rows = store.rows(["AAA"])
    assert set(rows["adj_close"]) == {9.0}
    assert set(store.rows(["BBB"])["adj_close"]) == {10.0}


def test_split_and_total_are_never_mixed(store: prices.PriceStore) -> None:
    _ = store.ensure(["AAA"], FIRST, THROUGH, fetch=Provider(adj=0.5))

    assert set(store.latest(["AAA"], FIRST, THROUGH, "split")["AAA"].series) == {10.0}
    assert set(store.latest(["AAA"], FIRST, THROUGH, "total")["AAA"].series) == {5.0}
    assert set(store.series("AAA", "split")) == {10.0}
    assert set(store.series("AAA", "total")) == {5.0}
