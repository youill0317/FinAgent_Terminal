from __future__ import annotations

import atexit
import shutil
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

from finagent import portfolio
from finagent.demo import VALUE

SETTINGS_FILE: Final = "finagent.settings.json"

TRADES: Final[tuple[tuple[int, str, portfolio.Side], ...]] = (
    (700, "DEMOA", "buy"),
    (650, "DEMOB", "buy"),
    (600, "DEMOC", "buy"),
    (300, "DEMOA", "sell"),
    (120, "DEMOB", "buy"),
)


def settings() -> Path:
    root = Path(tempfile.mkdtemp(prefix="finagent-demo-"))
    _ = atexit.register(shutil.rmtree, root, ignore_errors=True)
    portfolio.save_book(root, book(datetime.now(UTC).date()))
    return root / SETTINGS_FILE


def book(today: date) -> portfolio.Book:
    out, pid = portfolio.add_portfolio(portfolio.Book(), "데모 포트폴리오")
    for ago, ticker, side in TRADES:
        trade = portfolio.Trade(
            id=portfolio.new_id(),
            day=today - timedelta(days=ago),
            ticker=ticker,
            side=side,
            quantity=VALUE,
            price=VALUE,
        )
        out = portfolio.add_trade(out, pid, trade, {})
    return out
