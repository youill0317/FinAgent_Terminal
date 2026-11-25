# pyright: reportAny=false, reportImplicitStringConcatenation=false
"""매크로·종목·ETF 정밀 분석 프리셋 브리핑 입력 데이터 준비 및 검증."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, cast

from finagent import (
    etf,
    fundamentals,
    investors,
    lookthrough,
    markets,
    portfolio,
    portfolio_fetch,
)
from finagent.analysis import documents, prompt, skill
from finagent.analysis.company_tools import universe
from finagent.analysis.context import build_document
from finagent.analysis.service import RunInput
from finagent.demo import etf as demo_etf
from finagent.demo import portfolio as demo_portfolio
from finagent.prices import current as price_store

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

    from finagent.analysis.service import Brief
    from finagent.service import DashboardService
    from finagent.sources import Series as Points

ETF_FRACTION: Final = 0.10
REFUSALS: Final[dict[etf.Verdict, str]] = {
    "찾을 수 없음": "미확인 티커",
    "개별 주식": "ETF 미해당 티커 (개별 주식)",
}


@dataclass(frozen=True, slots=True)
class Briefing:
    root: Path
    settings_path: Path
    demo: bool
    env: Mapping[str, str]
    dashboard: DashboardService
    held: Callable[[], list[str]]
    confirm: Callable[[str], etf.Verdict]
    options: Callable[[], list[str]]
    prices: Callable[[], portfolio.Prices]
    fetcher: Callable[[], portfolio_fetch.Fetcher]
    updater: Callable[[], etf.Updater]
    look: Callable[
        [pd.Series], tuple[lookthrough.Lookthrough, dict[str, etf.Composition]]
    ]

    def prepare(self, brief: Brief) -> RunInput:
        d, ticker = brief.dashboard, brief.ticker
        if brief.kind == "stock":
            folder = self.root / skill.DATA_DIR
            if not (self.demo or skill.available(folder)):
                raise ValueError(f"`{folder}` 경로 내 참고 자료 미존재")
            if ticker not in self.held():
                self.refuse(ticker, self.confirm(ticker), {"찾을 수 없음"})
            document = documents.stock_document(
                self.stock_inputs(ticker), build_document(d)
            )
            guide = "" if self.demo else skill.guide(folder)
            return RunInput("", document, brief.kind, guide)
        if brief.kind == "etf":
            if ticker not in self.options():
                self.refuse(ticker, self.confirm(ticker), {"찾을 수 없음", "개별 주식"})
            document = documents.etf_analysis_document(
                self.etf_inputs(brief), build_document(d)
            )
            return RunInput("", document, brief.kind, prompt.text("etf_guide.txt"))
        found = self.etf_points([ticker for _, _, ticker in documents.MARKET_ETFS])
        document = documents.macro_document(
            d,
            self.dashboard.full_series,
            self.dashboard.today(),
            lambda ticker: found.get(ticker, ()),
        )
        return RunInput("", document, brief.kind, prompt.text("macro_guide.txt"))

    @staticmethod
    def refuse(ticker: str, verdict: etf.Verdict, refused: set[etf.Verdict]) -> None:
        if verdict == "확인 실패":
            raise ValueError(f"{ticker}: 티커를 확인하지 못했습니다")
        if verdict in refused:
            raise ValueError(f"{ticker}: {REFUSALS[verdict]}")

    def market_keys(self) -> markets.Keys:
        return markets.keys(self.env)

    def sec_agent(self) -> str:
        return self.env.get(investors.AGENT_KEY, "").strip()

    def etf_points(self, tickers: Sequence[str]) -> dict[str, Points]:
        if self.demo:
            rows = demo_etf.rows(list(tickers))
        else:
            listed = {p.ticker for p in universe(self.settings_path)}
            rows = price_store().rows([t for t in tickers if t in listed])
        table = etf.wide(rows, "adj_close")
        return {
            str(name): tuple(
                (cast("pd.Timestamp", day).date(), float(value))
                for day, value in line.dropna().items()
            )
            for name, line in table.items()
        }

    def book(self) -> portfolio.Book:
        return portfolio.adjusted_book(
            portfolio.load_book(self.root),
            {} if self.demo else portfolio.load_splits(),
        )

    def portfolios(self, name: str) -> list[dict[str, object]]:
        book = self.book()
        chosen = [p for p in book.portfolios if not name or p.name == name]
        if name and not chosen:
            names = ", ".join(p.name for p in book.portfolios) or "없음"
            raise ValueError(f"없는 포트폴리오: {name} · 있는 포트폴리오: {names}")
        calc = demo_portfolio if self.demo else portfolio
        close = self.prices().close
        out: list[dict[str, object]] = []
        for p in chosen:
            head: dict[str, object] = {"name": p.name, "cite_as": f"pf_{p.id}"}
            try:
                items = calc.holdings(p, close)
            except portfolio.BookError as error:
                out.append({**head, "error": str(error)})
                continue
            out.append({**head, **_holdings_body(items)})
        return out

    def stock_inputs(self, ticker: str) -> documents.StockInputs:
        prices = self.prices()
        if self.demo:
            found = demo_etf.company(ticker)
            annual = demo_etf.annual()
            bars = demo_etf.stock_rows(ticker)
        else:
            day = datetime.now().astimezone().date()
            agent = self.sec_agent()
            found = fundamentals.company_info(ticker, self.market_keys(), agent, day)
            annual = fundamentals.annual_rows(found.symbol, agent, day)
            bars = price_store().rows([found.symbol])
        name = found.cells.get("name")
        return documents.StockInputs(
            ticker=ticker,
            name=ticker if name is None else str(name.value),
            company=found.cells,
            annual=annual,
            bars=bars,
            holdings=documents.holding_lines(self.book(), prices.close, ticker),
            investors=documents.investor_lines(self.root, ticker),
        )

    def etf_fit(
        self, brief: Brief, found: etf.Composition | None, book: portfolio.Book
    ) -> tuple[str, ...]:
        calc = demo_portfolio if self.demo else portfolio
        ticker = brief.ticker
        held = [p for p in book.portfolios if p.trades]
        if not held:
            return ("거래가 있는 포트폴리오 없음",)
        chosen = book.get(brief.portfolio)
        p = chosen if chosen is not None and chosen.trades else held[0]
        prices = self.prices()
        try:
            w = portfolio.weights(calc.holdings(p, prices.close))
        except portfolio.BookError as error:
            return (f"{p.name}: {error}",)
        if w.empty:
            return (f"{p.name}: 가격이 확인된 보유 종목 없음",)
        if not self.demo:
            self.fetcher().look_now(portfolio.labels(w))
        look, _ = self.look(w)
        totals = look.totals()
        lines = [f"포트폴리오: {p.name}"]
        rows = (
            lookthrough.expanded(found.rows, full=found.full)
            if found is not None
            else ()
        )
        overlap = [
            (r.ticker, r.weight) for r in rows if r.ticker and r.ticker in totals
        ]
        lines.extend(
            f"겹침 {name}: 포트폴리오 실질 비중 {portfolio.percent(totals[name])}"
            f" · ETF 내 {weight:.2f}%"
            for name, weight in overlap
        )
        if rows and not overlap:
            lines.append("겹치는 구성 종목 없음")
        adjusted = prices.adjusted
        if ticker not in adjusted.columns:
            more = calc.load_prices(self.root, [ticker]).adjusted
            if ticker in more.columns:
                adjusted = adjusted.join(more, how="outer")
        j = calc.joint(
            adjusted, portfolio.with_candidate(w, ticker), portfolio.WINDOWS["3Y"]
        )
        if len(j.window) < portfolio.MIN_DAYS or ticker not in j.names:
            lines.append(f"편입 효과: 함께 값이 있는 날 {len(j.window)}일로 계산 불가")
            return tuple(lines)
        cov = calc.covariance(j)
        a = calc.added(j, cov, ticker, ETF_FRACTION)
        head = f"평가액의 {ETF_FRACTION:.0%} 편입 · {len(j.window)}일"
        lines.extend(
            (
                f"{head} · 연 변동성 {portfolio.percent(a.vol[0])}"
                f" → {portfolio.percent(a.vol[1])}",
                f"{head} · 베타 {a.beta[0]:.2f} → {a.beta[1]:.2f}",
            )
        )
        if a.es is not None:
            lines.append(
                f"{head} · 일간 ES {portfolio.percent(a.es[0])}"
                f" → {portfolio.percent(a.es[1])}"
            )
        return tuple(lines)

    def etf_inputs(self, brief: Brief) -> documents.EtfInputs:
        ticker = brief.ticker
        if self.demo:
            funds = demo_etf.funds()
            found = demo_etf.detail(ticker)
            bars = demo_etf.rows([ticker])
        else:
            updater = self.updater()
            day = datetime.now().astimezone().date()
            found = etf.open_detail(ticker, self.sec_agent(), updater.store, day)
            funds = updater.store.funds()
            bars = price_store().rows([ticker])
        book = self.book()
        return documents.EtfInputs(
            ticker=ticker,
            funds=funds.get(ticker, {}),
            table=brief.company.etf_table(),
            composition=found,
            bars=bars,
            holdings=documents.holding_lines(book, self.prices().close, ticker),
            fit=self.etf_fit(brief, found, book),
        )


def _finite(value: float, digits: int = 2) -> float | None:
    return round(value, digits) if math.isfinite(value) else None


def _holdings_body(items: list[portfolio.Holding]) -> dict[str, object]:
    total = portfolio.totals(items)
    priced = sum(h.value for h in items if h.is_open and math.isfinite(h.value))
    held = [
        {
            "ticker": h.ticker,
            "quantity": _finite(h.position.quantity, 4),
            "average": _finite(h.position.average, 4),
            "price": _finite(h.price, 4),
            "price_date": h.day.isoformat() if h.day else None,
            "value": _finite(h.value),
            "weight_pct": _finite(h.value / priced * 100 if priced > 0 else math.nan),
            "unrealized": _finite(h.unrealized),
            "return_pct": _finite(h.return_rate * 100),
        }
        for h in items
        if h.is_open
    ]
    closed = [
        {"ticker": h.ticker, "realized": _finite(h.position.realized)}
        for h in items
        if not h.is_open
    ]
    return {
        "holdings": held,
        "closed": closed,
        "totals": {
            "value": _finite(total.value),
            "cost": _finite(total.cost),
            "unrealized": _finite(total.unrealized),
            "realized": _finite(total.realized),
            "total_return_pct": _finite(total.total_return * 100),
        },
    }
