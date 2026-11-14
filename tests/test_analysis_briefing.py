from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, NoReturn

import pytest

from finagent import lookthrough, portfolio
from finagent.analysis.briefing import Briefing
from finagent.analysis.company_tools import Company
from finagent.analysis.service import Brief
from finagent.cli import build_service
from finagent.demo import book as demo_book
from finagent.demo import portfolio as demo_portfolio

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

    from finagent import etf
    from finagent.analysis.context import Document


def unused() -> NoReturn:
    raise AssertionError


def look(w: pd.Series) -> tuple[lookthrough.Lookthrough, dict[str, etf.Composition]]:
    comps = demo_portfolio.compositions(portfolio.labels(w))
    weights = {name: 1 / len(comps) for name in comps}
    return lookthrough.effective_holdings(weights, comps), comps


def test_prepare_builds_the_macro_stock_and_etf_documents_in_demo(
    tmp_path: Path,
) -> None:
    service = build_service(demo=True, env={})
    portfolio.save_book(tmp_path, demo_book.book(service.today()))
    found = Briefing(
        root=tmp_path,
        settings_path=tmp_path / demo_book.SETTINGS_FILE,
        demo=True,
        env={},
        dashboard=service,
        held=lambda: ["DEMOA"],
        confirm=lambda _ticker: "확인 실패",
        options=lambda: ["QQQ"],
        prices=lambda: demo_portfolio.load_prices(tmp_path, ()),
        fetcher=unused,
        updater=unused,
        look=look,
    )
    data = asyncio.run(service.get())
    company = Company({}, service.today(), demo=True)

    def prepared(kind: str, ticker: str = "") -> Document:
        document = found.prepare(Brief(kind, ticker, "", data, company)).document
        assert document is not None
        return document

    macro, stock, fund = (
        prepared("macro"),
        prepared("stock", "DEMOA"),
        prepared("etf", "QQQ"),
    )

    assert "<macro_changes" in macro.text
    assert "<etf_changes>" in macro.text
    assert '<stock ticker="DEMOA">' in stock.text
    assert "데모 포트폴리오 거래" in stock.text.split('id="stk_holdings"')[1]
    assert '<etf ticker="QQQ">' in fund.text
    assert "포트폴리오: 데모 포트폴리오" in fund.text.split('id="etf_fit"')[1]
    assert "평가액의 10% 편입" in fund.text
    assert stock.base == fund.base != ""
    with pytest.raises(ValueError, match="ZZZZ: 티커를 확인하지 못했습니다"):
        _ = prepared("stock", "ZZZZ")
    with pytest.raises(ValueError, match="ZZZZ: 티커를 확인하지 못했습니다"):
        _ = prepared("etf", "ZZZZ")
