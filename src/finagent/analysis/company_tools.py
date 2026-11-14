# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""AI 에이전트용 기업 재무(SEC XBRL)·공시 원문·기업 개요·ETF 조회 도구."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Final, cast

from finagent import (
    etf,
    etf_universe,
    fund_filings,
    fundamentals,
    investors,
    markets,
    prices,
    sec,
    yahoo,
)
from finagent.analysis.documents import holding_stats
from finagent.demo import etf as demo_etf
from finagent.limits import INTERACTIVE_WAIT, SourceError
from finagent.settings import load_etf

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import pandas as pd

type Json = dict[str, object]

DEFAULT_METRICS: Final = ("revenue", "net_income", "eps_diluted")
METRIC_TAGS: Final[Mapping[str, tuple[str, ...]]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss",),
    "operating_income": ("OperatingIncomeLoss",),
    "gross_profit": ("GrossProfit",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": ("StockholdersEquity",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
}
DEFAULT_PERIODS: Final = 8
MAX_PERIODS: Final = 40
DEFAULT_FILINGS: Final = 20
MAX_FILINGS: Final = 100
DEFAULT_ROWS: Final = 20
MAX_ROWS: Final = 50
DEFAULT_HOLDINGS: Final = 25
MAX_HOLDINGS: Final = 100
WEIGHTS: Final = ("sector_weights", "credit_weights")
DESCRIPTION_CHARS: Final = 1000
PROFILE_FIELDS: Final[Mapping[str, str]] = {
    "name": "name",
    "sector": "sector",
    "industry": "industry",
    "country": "country",
    "currency": "currency",
    "market_cap": "market_cap",
    "per": "per",
    "forward_per": "forward_per",
    "dividend_yield": "dividend_yield_pct",
    "roe": "roe_pct",
    "net_margin": "net_margin_pct",
    "revenue_growth": "revenue_growth_pct",
    "description": "description",
}
CHUNK: Final = 20_000
EARNINGS: Final = re.compile(r"(?:ex|exhibit)[-_]?99", re.IGNORECASE)
PAGE: Final = (".htm", ".html", ".txt")
DEMO_FACTS: Final = (
    ("2026-06-27", "2026 Q3", "10-Q", "2026-08-01", 94_036_000_000.0),
    ("2026-03-28", "2026 Q2", "10-Q", "2026-05-02", 95_359_000_000.0),
)
DEMO_FILING: Final = "0000320193-26-000071"
DEMO_TEXT: Final = "데모 공시 본문: 분기 매출 및 이익 요약"


@dataclass(frozen=True, slots=True)
class Answer:
    body: Json
    summary: str
    received: Mapping[str, str]


def text_arg(args: Mapping[str, object], key: str) -> str:
    value = args.get(key)
    return value.strip() if isinstance(value, str) else ""


def _strings(args: Mapping[str, object], key: str) -> list[str]:
    value = args.get(key)
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [str(v).strip() for v in items if str(v).strip()]
    return []


def _count(args: Mapping[str, object], key: str, default: int, top: int) -> int:
    value = args.get(key)
    return max(1, min(value, top)) if isinstance(value, int) else default


def _day(args: Mapping[str, object], key: str) -> date | None:
    value = text_arg(args, key)
    return date.fromisoformat(value) if value else None


def _ticker(args: Mapping[str, object]) -> str:
    ticker = text_arg(args, "ticker").upper()
    if not ticker:
        raise ValueError("ticker 필수 입력")
    return ticker


def _rows_of(tags: Mapping[str, object], names: Sequence[str]) -> list[Json]:
    found: list[list[Json]] = []
    for tag in names:
        units = cast("Mapping[str, object]", tags.get(tag) or {}).get("units")
        if isinstance(units, dict) and units:
            mapping = cast("Mapping[str, list[Json]]", units)
            found.append(next(iter(mapping.values())))
    return max(
        found,
        key=lambda rows: max((str(r.get("end")) for r in rows), default=""),
        default=[],
    )


def _periods(rows: Sequence[Json], count: int) -> list[Json]:
    grouped: dict[str, list[Json]] = {}
    for row in rows:
        frame = row.get("frame")
        if isinstance(frame, str):
            grouped.setdefault(frame, []).append(row)
    shown: list[Json] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: str(r.get("filed")))
        first, last = ordered[0], ordered[-1]
        shown.append(
            {
                "end": last.get("end"),
                "fiscal_period": f"{first.get('fy')} {first.get('fp')}",
                "form": last.get("form"),
                "filed": last.get("filed"),
                "value": last.get("val"),
            }
        )
    shown.sort(key=lambda r: str(r["end"]), reverse=True)
    return shown[:count]


def _matching(names: Sequence[str], wanted: str) -> set[str]:
    if wanted in names:
        return {wanted}
    key = wanted.casefold().replace(" ", "")
    return {n for n in names if key and key in n.casefold().replace(" ", "")}


def universe(settings: Path | None) -> tuple[etf_universe.Product, ...]:
    if settings is None:
        return etf_universe.DEFAULTS
    choice = load_etf(settings)
    return etf_universe.effective(choice.added, choice.removed)


@dataclass(slots=True)
class Company:
    env: Mapping[str, str]
    today: date
    demo: bool = False
    settings: Path | None = None
    _client: sec.Client | None = field(default=None, repr=False)
    _ciks: dict[str, str] = field(default_factory=dict, repr=False)
    _recents: dict[tuple[str, date], list[Json]] = field(
        default_factory=dict, repr=False
    )
    _table: tuple[object, pd.DataFrame] | None = field(default=None, repr=False)

    def run(self, name: str, args: Mapping[str, object]) -> Answer:
        return {
            "company_facts": self._facts,
            "company_filings": self._filings,
            "read_filing": self._read,
            "etf_lookup": self._etf,
            "etf_holdings": self._etf_holdings,
            "company_profile": self._profile,
        }[name](args)

    def _agent(self) -> str:
        agent = self.env.get(investors.AGENT_KEY, "").strip()
        if not agent:
            message = f"{investors.AGENT_KEY} 환경변수 미설정"
            raise SourceError(message)
        return agent

    def _sec(self) -> sec.Client:
        if self._client is None:
            self._client = sec.Client(self._agent())
        return self._client

    def _archive(self) -> sec.Archive:
        return sec.Archive(self._sec(), self.today, INTERACTIVE_WAIT)

    def _cik(self, ticker: str) -> str:
        if ticker not in self._ciks:
            cik = self._archive().cik(ticker)
            if cik is None:
                raise SourceError(f"SEC 미등록 티커: {ticker}")
            self._ciks[ticker] = cik
        return self._ciks[ticker]

    def _facts(self, args: Mapping[str, object]) -> Answer:
        ticker = _ticker(args)
        metrics = _strings(args, "metrics") or list(DEFAULT_METRICS)
        count = _count(args, "periods", DEFAULT_PERIODS, MAX_PERIODS)
        out: dict[str, list[Json]] = {}
        if self.demo:
            out = {
                metric: [
                    dict(
                        zip(
                            ("end", "fiscal_period", "form", "filed", "value"),
                            row,
                            strict=True,
                        )
                    )
                    for row in DEMO_FACTS[:count]
                ]
                for metric in metrics
            }
        else:
            raw = self._archive().facts(self._cik(ticker))
            if raw is None:
                raise SourceError(f"{ticker}: SEC 재무값 없음")
            facts = cast("Mapping[str, object]", raw.get("facts") or {})
            tags = cast("Mapping[str, object]", facts.get("us-gaap") or {})
            for metric in metrics:
                names = METRIC_TAGS.get(metric, (metric,))
                out[metric] = _periods(_rows_of(tags, names), count)
        cites = {metric: f"SEC:{ticker}:{metric}" for metric in out if out[metric]}
        if not cites:
            raise SourceError(f"{ticker}: 조회된 항목 없음 ({', '.join(metrics)})")
        items = [
            {"metric": metric, "cite_as": cites.get(metric), "periods": rows}
            for metric, rows in out.items()
        ]
        return Answer(
            {"ticker": ticker, "items": items},
            f"company_facts({ticker}) · {len(cites)}개 항목",
            {cite: f"{ticker} {metric} · SEC" for metric, cite in cites.items()},
        )

    def _recent(self, ticker: str) -> list[Json]:
        key = (ticker, self.today)
        if key not in self._recents:
            self._recents[key] = self._submissions(ticker)
        return self._recents[key]

    def _submissions(self, ticker: str) -> list[Json]:
        response = self._sec().submissions(self._cik(ticker), INTERACTIVE_WAIT)
        raw = sec.json_or_none(response)
        if raw is None:
            raise SourceError(f"{ticker}: SEC 제출 목록 없음")
        filings = cast("Mapping[str, Mapping[str, list[str]]]", raw.get("filings"))
        recent = filings["recent"]
        return [
            {
                "form": form,
                "accession": accession,
                "filed": filed,
                "items": items,
                "primary": primary,
            }
            for form, accession, filed, items, primary in zip(
                recent["form"],
                recent["accessionNumber"],
                recent["filingDate"],
                recent["items"],
                recent["primaryDocument"],
                strict=True,
            )
        ]

    def _filings(self, args: Mapping[str, object]) -> Answer:
        ticker = _ticker(args)
        forms = set(_strings(args, "forms"))
        since = _day(args, "since")
        count = _count(args, "limit", DEFAULT_FILINGS, MAX_FILINGS)
        rows = (
            [
                {
                    "form": "8-K",
                    "accession": DEMO_FILING,
                    "filed": "2026-08-01",
                    "items": "2.02,9.01",
                    "primary": "demo.htm",
                }
            ]
            if self.demo
            else self._recent(ticker)
        )
        kept = [
            row
            for row in rows
            if (not forms or row["form"] in forms)
            and (since is None or date.fromisoformat(str(row["filed"])) >= since)
        ][:count]
        shown = [
            {
                "form": row["form"],
                "accession": row["accession"],
                "filed": row["filed"],
                "items": row["items"],
                "cite_as": f"SEC:{row['accession']}",
            }
            for row in kept
        ]
        return Answer(
            {"ticker": ticker, "filings": shown},
            f"company_filings({ticker}) · {len(shown)}건",
            {str(r["cite_as"]): f"{ticker} {r['form']} {r['filed']}" for r in shown},
        )

    def _document(self, ticker: str, cik: str, accession: str, document: str) -> str:
        archive = self._archive()
        if document not in {"main", "earnings"}:
            return document
        kept = archive.main(accession)
        if document == "main" and kept is not None:
            return kept
        names = archive.documents(cik, accession)
        pages = [name for name in names if name.lower().endswith(PAGE)]
        if document == "earnings":
            found = next((name for name in pages if EARNINGS.search(name)), None)
            if found is None:
                raise SourceError(f"{accession}: 실적 보도자료 첨부 없음")
            return found
        primary = next(
            (
                str(row["primary"])
                for row in self._recent(ticker)
                if row["accession"] == accession
            ),
            None,
        )
        name = primary if primary in names else next(iter(pages), None)
        if name is None:
            raise SourceError(f"{accession}: 열람 가능 문서 없음")
        archive.keep_main(accession, name)
        return name

    def _read(self, args: Mapping[str, object]) -> Answer:
        ticker = _ticker(args)
        accession = text_arg(args, "accession")
        document = text_arg(args, "document") or "main"
        offset = args.get("offset")
        start = offset if isinstance(offset, int) and offset > 0 else 0
        if sec.ACCESSION.fullmatch(accession) is None:
            raise ValueError(f"접수번호 형식 오류: {accession}")
        if self.demo:
            name, text = "demo.htm", DEMO_TEXT
        else:
            cik = self._cik(ticker)
            name = self._document(ticker, cik, accession, document)
            text = self._archive().body(cik, accession, name)
        end = start + CHUNK
        cite = f"SEC:{accession}"
        return Answer(
            {
                "ticker": ticker,
                "accession": accession,
                "document": name,
                "cite_as": cite,
                "offset": start,
                "next_offset": end if end < len(text) else None,
                "text": text[start:end],
            },
            f"read_filing({accession}) · {name} {len(text[start:end])}자",
            {cite: f"{ticker} 공시 {accession}"},
        )

    def etf_table(self) -> pd.DataFrame:
        products = universe(self.settings)
        stamp = (
            products,
            demo_etf.STAMP
            if self.demo
            else (prices.current().stamp(), etf.FundStore().version),
        )
        if self._table is None or self._table[0] != stamp:
            self._table = (stamp, self._load_etf_table(products))
        return self._table[1]

    def _load_etf_table(self, products: Sequence[etf_universe.Product]) -> pd.DataFrame:
        if self.demo:
            rows = demo_etf.rows([p.ticker for p in products])
            funds = demo_etf.funds()
        else:
            rows = prices.current().rows([p.ticker for p in products])
            funds = etf.FundStore().funds()
        leaders = etf.representatives(products, funds)
        table = etf.table_frame(funds, products, etf.returns_frame(rows), leaders)
        cells = [funds.get(str(t), {}).get("yield") for t in table["티커"]]
        yields = [None if c is None else c.value for c in cells]
        return table.assign(**{"분배수익률(%)": yields})

    def _etf(self, args: Mapping[str, object]) -> Answer:
        tickers = {t.upper() for t in _strings(args, "tickers")}
        group = text_arg(args, "group")
        category = text_arg(args, "category")
        count = _count(args, "limit", DEFAULT_ROWS, MAX_ROWS)
        table = self.etf_table()
        groups = sorted({str(v) for v in table["분류"]})
        categories = sorted({str(v) for v in table["카테고리"]})
        group_names = _matching(groups, group)
        category_names = _matching(categories, category)
        columns = [str(c) for c in table.columns]
        rows: list[Json] = []
        cites: dict[str, str] = {}
        for values in table.itertuples(index=False):
            row = dict(zip(columns, cast("tuple[object, ...]", values), strict=True))
            ticker = str(row["티커"])
            if (
                (tickers and ticker not in tickers)
                or (group and row["분류"] not in group_names)
                or (category and row["카테고리"] not in category_names)
            ):
                continue
            if len(rows) < count:
                cite = f"etf_{ticker}"
                cites[cite] = f"{ticker} ETF"
                rows.append(
                    {
                        "ticker": ticker,
                        "cite_as": cite,
                        "values": {c: etf.cell_text(c, row[c]) for c in columns},
                    }
                )
        found = {str(row["ticker"]) for row in rows}
        missing = sorted(tickers - found)
        body: Json = {"rows": rows, "missing": missing}
        if group and not group_names:
            body["groups"] = groups
        if category and not category_names:
            body["categories"] = categories
        return Answer(
            body,
            f"etf_lookup · {len(rows)}개",
            cites,
        )

    def _profile(self, args: Mapping[str, object]) -> Answer:
        ticker = _ticker(args)
        if self.demo:
            found = demo_etf.company(ticker)
        else:
            agent = self.env.get(investors.AGENT_KEY, "").strip()
            found = fundamentals.company_info(
                ticker, markets.keys(self.env), agent, self.today
            )
        values: Json = {}
        for name, key in PROFILE_FIELDS.items():
            cell = found.cells.get(name)
            if cell is None:
                continue
            value = cell.value
            if name == "description":
                value = str(value)[:DESCRIPTION_CHARS]
            values[key] = {"value": value, "source": cell.source, "as_of": cell.as_of}
        if not values:
            raise SourceError(f"{ticker}: 기업 정보 미확인")
        cite = f"profile_{ticker}"
        return Answer(
            {
                "ticker": ticker,
                "symbol": found.symbol,
                "cite_as": cite,
                "values": values,
            },
            f"company_profile({ticker})",
            {cite: f"{ticker} 회사 정보"},
        )

    def _composition(self, ticker: str) -> etf.Composition | None:
        if self.demo:
            return demo_etf.detail(ticker)
        agent = self.env.get(investors.AGENT_KEY, "").strip()
        try:
            return etf.composition(ticker, agent, self.today)
        except etf.FAILURES:
            return etf.load_holdings(ticker)

    def _weights(self, ticker: str) -> Json:
        if self.demo:
            cells = demo_etf.funds().get(ticker, {})
        else:
            cells = etf.FundStore().funds().get(ticker, {})
        found: Json = {
            f"{name}_pct": cells[name].value for name in WEIGHTS if name in cells
        }
        if found or self.demo:
            return found
        try:
            info = yahoo.fund_info(ticker, INTERACTIVE_WAIT)
        except etf.FAILURES:
            return {}
        return {f"{name}_pct": info[name] for name in WEIGHTS if info.get(name)}

    def _etf_holdings(self, args: Mapping[str, object]) -> Answer:
        ticker = _ticker(args)
        count = _count(args, "limit", DEFAULT_HOLDINGS, MAX_HOLDINGS)
        found = self._composition(ticker)
        weights = self._weights(ticker)
        if found is None and not weights:
            raise SourceError(f"{ticker}: ETF 구성 미확인")
        body: Json = {"ticker": ticker, "cite_as": f"holdings_{ticker}"}
        if found is not None:
            rows = found.rows
            top, effective, lands, maturity = holding_stats(found)
            mix = found.assets or (fund_filings.asset_mix(rows) if rows else {})
            body |= {
                "source": found.source,
                "as_of": found.as_of,
                "full": found.full,
                "asset_mix_pct": {k: round(v, 2) for k, v in mix.items()},
                "holdings_count": sum(r.asset != "collateral" for r in rows),
                "top10_weight_pct": round(top, 2),
                "effective_holdings": None
                if effective is None
                else round(effective, 1),
                "countries_pct": {k: round(v, 2) for k, v in lands},
                "holdings": [
                    {
                        "ticker": r.ticker or r.isin or r.cusip,
                        "name": r.name,
                        "weight_pct": round(r.weight, 2),
                        "asset": r.asset,
                        "country": r.country,
                    }
                    for r in fund_filings.top(rows, count)
                ],
            }
            if maturity is not None:
                body["bond_maturity_years"] = maturity
        body |= weights
        cite = str(body["cite_as"])
        source = str(body.get("source", "Yahoo"))
        return Answer(
            body,
            f"etf_holdings({ticker}) · {source}",
            {cite: f"{ticker} 구성 · {source}"},
        )
