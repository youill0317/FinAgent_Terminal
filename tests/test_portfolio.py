# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from finagent import markets, portfolio_fetch, prices
from finagent import portfolio as pf

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


D1, D2, D3, D4, D5 = (date(2026, 3, d) for d in (2, 3, 4, 5, 6))


def trade(  # noqa: PLR0913
    day: date,
    side: pf.Side,
    quantity: float,
    price: float,
    *,
    ticker: str = "AAA",
    fee: float = 0.0,
) -> pf.Trade:
    return pf.Trade(
        id=pf.new_id(),
        day=day,
        ticker=ticker,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
    )


def folio(*trades: pf.Trade) -> pf.Portfolio:
    return pf.Portfolio(id="p1", name="테스트", trades=trades)


def frame(days: Sequence[date], **columns: Sequence[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    return pd.DataFrame(dict(columns), index=index, dtype="float64")


def test_moving_average_cost_with_partial_sells_and_fees() -> None:
    ledger = pf.replay(
        [
            trade(D1, "buy", 10, 100, fee=10),
            trade(D2, "buy", 10, 120, fee=10),
            trade(D3, "sell", 5, 130, fee=5),
        ]
    )
    position = ledger.positions["AAA"]
    (sale,) = ledger.sales

    assert sale.average == pytest.approx(111)
    assert sale.basis == pytest.approx(555)
    assert sale.pnl == pytest.approx(650 - 5 - 555)
    assert sale.return_rate == pytest.approx(90 / 555)
    assert sale.held_days == 2
    assert position.quantity == 15
    assert position.average == pytest.approx(111)
    assert position.realized == pytest.approx(90)


def test_twr_ignores_buys_and_sells() -> None:
    p = folio(
        trade(D1, "buy", 10, 100),
        trade(D3, "buy", 10, 110),
        trade(D4, "sell", 10, 121),
    )
    close = frame(
        [D1, D2, D3, D4, D5],
        AAA=[100.0, 110.0, 121.0, 121.0, 133.1],
        **{pf.TOTAL_RETURN: [10.0, 11.0, 12.0, 13.0, 14.0]},
    )
    cumulative = pf.twr(p, close, close)

    assert cumulative.tolist() == pytest.approx([0, 0.1, 0.21, 0.21, 0.331])
    days = pd.DatetimeIndex(cumulative.index)
    sp = pf.market_returns(close, days)
    assert pf.cumulative(sp).iloc[-1] == pytest.approx(0.4)
    history = pf.history(p, close)
    assert history["invested"].tolist() == pytest.approx([1000, 1000, 2100, 890, 890])
    assert history["realized"].tolist() == pytest.approx([0, 0, 0, 1210 - 1050, 160])
    lines = pf.pnl_lines(history)
    assert lines["총손익"].iloc[-1] == pytest.approx(1331 - 890)
    assert lines["평가손익"].iloc[-1] == pytest.approx(1331 - 890 - 160)


def test_twr_and_mwr_count_dividends_by_the_adjusted_closes() -> None:
    p = folio(trade(D1, "buy", 10, 100))
    close = frame([D1, D2, D3], AAA=[100.0, 99.0, 99.0])
    adjusted = frame([D1, D2, D3], AAA=[99.0, 99.0, 99.0])

    assert pf.paid(p, close, adjusted, pd.DatetimeIndex(close.index)).tolist() == (
        pytest.approx([0, 10, 0])
    )
    assert pf.twr(p, close, adjusted).tolist() == pytest.approx([0, 0, 0])
    assert pf.twr(p, close, close).iloc[-1] == pytest.approx(-0.01)
    valued = pd.DatetimeIndex(pf.history(p, close).index)
    mwr = pf.money_weighted(p, close, adjusted, 990.0, valued)
    assert mwr is not None
    assert mwr.cumulative == pytest.approx(0.0, abs=1e-9)


def test_splits_scale_trades_entered_before_the_ex_date() -> None:
    p = folio(trade(D1, "buy", 10, 100), trade(D3, "buy", 5, 60))
    splits = {"AAA": (pf.Split(ex_date=D2, ratio=2.0),)}

    shown = pf.split_adjusted(p, splits)

    assert [(t.quantity, t.price) for t in shown.trades] == [(20, 50), (5, 60)]
    assert pf.split_adjusted(p, {}) == p
    assert pf.adjusted_book(pf.Book(portfolios=(p,)), splits).portfolios[0] == shown
    pf.save_splits(splits)
    pf.save_splits({"BBB": ()})
    assert pf.load_splits() == {"AAA": splits["AAA"], "BBB": ()}


def test_a_refresh_from_a_recent_day_keeps_earlier_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = prices.SplitRow(date(2021, 7, 20), 4.0)

    def splits(
        tickers: Sequence[str], start: date, _last: date, _keys: markets.Keys
    ) -> dict[str, tuple[prices.SplitRow, ...]]:
        return {t: tuple(r for r in (old,) if r.ex_date >= start) for t in tickers}

    monkeypatch.setattr(prices, "splits", splits)
    source = partial(portfolio_fetch.Official, keys=markets.Keys())
    fetcher = portfolio_fetch.Fetcher(source)

    recent = datetime.now(markets.NEW_YORK).date() - timedelta(days=30)
    assert fetcher.start(["AAA"], recent, force=True)
    while fetcher.progress.running:
        time.sleep(0.01)

    assert pf.load_splits()["AAA"] == (pf.Split(ex_date=old.ex_date, ratio=4.0),)


def test_the_book_round_trips(tmp_path: Path) -> None:
    root = tmp_path
    book, pid = pf.add_portfolio(pf.Book(), "연금")
    first = trade(D1, "buy", 1.5, 170, ticker="AAPL")
    book = pf.add_trade(
        book, pid, first.model_copy(update={"memo": "실적 발표 전"}), {}
    )

    assert pf.load_book(root) == pf.Book()
    pf.save_book(root, book)

    assert pf.load_book(root) == book
    assert pf.load_book(root).portfolios[0].trades[0].memo == "실적 발표 전"
    saved = (root / pf.BOOK_FILE).read_text("utf-8")
    assert f'"day": "{D1.isoformat()}"' in saved
    assert not list(root.glob("*.tmp"))
    legacy = pf.Portfolio.model_validate({"id": "p1", "name": "연금", "base": "USD"})
    assert "base" not in legacy.model_dump_json()
    _ = (root / pf.BOOK_FILE).write_text("{", encoding="utf-8")
    with pytest.raises(pf.BookError):
        _ = pf.load_book(root)
