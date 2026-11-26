from __future__ import annotations

import asyncio
import json
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
    from finagent.service import DashboardService


def unused() -> NoReturn:
    raise AssertionError


def look(w: pd.Series) -> tuple[lookthrough.Lookthrough, dict[str, etf.Composition]]:
    comps = demo_portfolio.compositions(portfolio.labels(w))
    weights = {name: 1 / len(comps) for name in comps}
    return lookthrough.effective_holdings(weights, comps), comps


def demo_briefing(tmp_path: Path) -> tuple[Briefing, DashboardService]:
    service = build_service(demo=True, env={})
    portfolio.save_book(tmp_path, demo_book.book(service.today()))
    return Briefing(
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
    ), service


def test_prepare_builds_the_macro_stock_and_etf_documents_in_demo(
    tmp_path: Path,
) -> None:
    found, service = demo_briefing(tmp_path)
    data = asyncio.run(service.get())
    company = Company({}, service.today(), demo=True)

    guides: dict[str, str] = {}

    def prepared(kind: str, ticker: str = "") -> Document:
        given = found.prepare(Brief(kind, ticker, "", data, company))
        assert given.document is not None
        guides[kind] = given.guide
        return given.document

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
    assert guides["macro"].startswith("<macro_guide>")
    assert guides["etf"].startswith("<etf_guide>")
    assert guides["stock"] == ""
    assert "<etf_guide>" not in fund.text
    with pytest.raises(ValueError, match="ZZZZ: 티커를 확인하지 못했습니다"):
        _ = prepared("stock", "ZZZZ")
    with pytest.raises(ValueError, match="ZZZZ: 티커를 확인하지 못했습니다"):
        _ = prepared("etf", "ZZZZ")


def test_portfolios_lists_holdings_and_totals_as_plain_json(tmp_path: Path) -> None:
    found, _ = demo_briefing(tmp_path)

    every = found.portfolios("")
    one = found.portfolios("데모 포트폴리오")

    assert "데모 포트폴리오" in [item["name"] for item in every]
    assert [item["name"] for item in one] == ["데모 포트폴리오"]
    assert one[0]["holdings"]
    assert str(one[0]["cite_as"]).startswith("pf_")
    _ = json.dumps(every, allow_nan=False)
    with pytest.raises(ValueError, match="없는 포트폴리오: 없음"):
        _ = found.portfolios("없음")
