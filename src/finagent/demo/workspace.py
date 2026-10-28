"""The demo's own folder: its settings file and a trade book at ``VALUE``.

The demo never reads the user's files: ``workspace`` makes a temporary
folder once per process. No prices are saved; ``demo.portfolio`` stands in
for them.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

from finagent import portfolio
from finagent.demo import VALUE

SETTINGS_FILE: Final = "finagent.settings.json"

# The book's trades, every quantity and price at VALUE: (days ago, ticker, side).
# Made-up tickers, as the screen's: DEMOK.KS is Korean (in won), the others
# American, so the page shows both. DEMOA sells out, for sold-out tickers too.
TRADES: Final[tuple[tuple[int, str, portfolio.Side], ...]] = (
    (700, "DEMOA", "buy"),
    (650, "DEMOB", "buy"),
    (600, "DEMOK.KS", "buy"),
    (300, "DEMOA", "sell"),
    (120, "DEMOB", "buy"),
)


def workspace() -> Path:
    """A new temporary folder with the demo's book; returns its settings file."""
    root = Path(tempfile.mkdtemp(prefix="finagent-demo-"))
    portfolio.save_book(root / portfolio.DATA_DIR, book(datetime.now(UTC).date()))
    return root / SETTINGS_FILE


def book(today: date) -> portfolio.Book:
    out, pid = portfolio.add_portfolio(portfolio.Book(), "데모 포트폴리오", "KRW")
    for ago, ticker, side in TRADES:
        trade = portfolio.Trade(
            id=portfolio.new_id(),
            day=today - timedelta(days=ago),
            ticker=ticker,
            side=side,
            quantity=VALUE,
            price=VALUE,
        )
        out = portfolio.add_trade(out, pid, trade)
    return out
