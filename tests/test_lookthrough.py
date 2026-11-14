# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import pytest

from finagent import lookthrough
from finagent.etf import Composition
from finagent.fund_filings import Asset, Row


def row(ticker: str | None, weight: float, asset: Asset = "stock") -> Row:
    return Row(ticker or "", None, None, ticker, weight, asset, "US", None)


def held(ticker: str, *rows: Row, full: bool = True) -> Composition:
    return Composition(ticker, "2026-03-31", "N-PORT", None, full, rows)


QQQ = held("QQQ", row("NVDA", 9.0), row("MSFT", 8.0), row("AAPL", 7.5), row(None, 2.0))
SOXX = held("SOXX", row("NVDA", 8.5), row("AVGO", 8.0), row("TSY", 1.0, "bond"))
BOOK = {"GOOGL": 0.2, "MSFT": 0.2, "QQQ": 0.3, "SOXX": 0.2, "VOO": 0.1}


def test_direct_via_and_residual_add_up_to_the_whole_book() -> None:
    look = lookthrough.effective_holdings(BOOK, {"QQQ": QQQ, "SOXX": SOXX})

    total = sum(r.total for r in look.rows) + sum(look.residual.values())
    assert total == pytest.approx(1.0, abs=5e-5)
    nvda = next(r for r in look.rows if r.ticker == "NVDA")
    assert nvda.direct == 0.0
    assert nvda.total == pytest.approx(0.3 * 0.09 + 0.2 * 0.085)
    msft = next(r for r in look.rows if r.ticker == "MSFT")
    assert msft.total == pytest.approx(0.2 + 0.3 * 0.08)
    assert next(r for r in look.rows if r.ticker == "VOO").direct == 0.1
