# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""매크로·종목·ETF·포트폴리오 AI 분석 컨텍스트 문서(XML) 조립기."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Final, cast
from xml.sax.saxutils import escape, quoteattr

import pandas as pd

from finagent import display, etf, fund_filings, fundamentals, investors, portfolio
from finagent.analysis import technical
from finagent.analysis.context import (
    Document,
    build_document,
    element,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import date
    from pathlib import Path

    from finagent.models import Dashboard, IndicatorRow
    from finagent.sources import Series

CHANGE_DAYS: Final = 30
WEEK_DAYS: Final = 7


@dataclass(frozen=True, slots=True)
class SeriesChange:
    day: date
    latest: float
    daily: float | None
    weekly: float | None
    since: float | None


def _last_on_or_before(series: Series, day: date) -> float | None:
    found = [value for point, value in series if point <= day]
    return found[-1] if found else None


def _difference(last: float, base: float | None, *, points: bool) -> float | None:
    if base is None:
        return None
    if points:
        return last - base
    return (last / base - 1.0) * 100.0 if base else None


def series_change(series: Series, since: date, *, points: bool) -> SeriesChange | None:
    if not series:
        return None
    day, latest = series[-1]
    before = series[-2][1] if len(series) > 1 else None
    week = _last_on_or_before(series, day - timedelta(days=WEEK_DAYS))
    start = _last_on_or_before(series, since)
    return SeriesChange(
        day,
        latest,
        _difference(latest, before, points=points),
        _difference(latest, week, points=points),
        _difference(latest, start, points=points),
    )


def _shown(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    return f"{value:+,.2f}{unit}"


def _daily_rows(dashboard: Dashboard) -> list[list[IndicatorRow]]:
    seen: dict[str, dict[str, IndicatorRow]] = {}
    for panel in dashboard.panels:
        for row in panel.rows:
            if row.freq == "D" and row.status == "ok":
                _ = seen.setdefault(row.code, {}).setdefault(row.id, row)
    return [list(rows.values()) for rows in seen.values()]


MARKET_ETFS: Final = (
    ("업종", "기술", "XLK"),
    ("업종", "금융", "XLF"),
    ("업종", "헬스케어", "XLV"),
    ("업종", "에너지", "XLE"),
    ("업종", "산업재", "XLI"),
    ("업종", "소재", "XLB"),
    ("업종", "유틸리티", "XLU"),
    ("업종", "부동산", "XLRE"),
    ("업종", "임의소비재", "XLY"),
    ("업종", "필수소비재", "XLP"),
    ("업종", "커뮤니케이션", "XLC"),
    ("자산군", "장기 국채", "TLT"),
    ("자산군", "하이일드", "HYG"),
    ("자산군", "금", "GLD"),
    ("자산군", "원유", "USO"),
    ("자산군", "달러", "UUP"),
    ("자산군", "신흥국", "EEM"),
)


def etf_changes(
    etf_series: Callable[[str], Series], since: date, span: str
) -> tuple[str, set[str]]:
    lines = ["<etf_changes>"]
    ids: set[str] = set()
    for group, name, ticker in MARKET_ETFS:
        change = series_change(etf_series(ticker), since, points=False)
        cite = f"tgt_{ticker}"
        cells = (
            cite,
            name,
            group,
            ticker,
            "—" if change is None else change.day.isoformat(),
            f"일간 {_shown(None if change is None else change.daily, '%')}",
            f"주간 {_shown(None if change is None else change.weekly, '%')}",
            f"{span} {_shown(None if change is None else change.since, '%')}",
        )
        lines.append(" · ".join(cells))
        ids.add(cite)
    lines.append("</etf_changes>")
    return "\n".join(lines), ids


def macro_document(
    dashboard: Dashboard,
    full_series: Callable[[str], Series],
    today: date,
    etf_series: Callable[[str], Series],
) -> Document:
    base = build_document(dashboard)
    since = today - timedelta(days=CHANGE_DAYS)
    ids = set(base.ids)
    lines = [f"<macro_changes since={quoteattr(since.isoformat())}>"]
    span = f"{CHANGE_DAYS}일 전 이후"
    for same in _daily_rows(dashboard):
        row = same[0]
        points = row.fmt.suf in {"%", "%p"}
        change = series_change(full_series(row.id), since, points=points)
        if change is None:
            continue
        unit = "%p" if points else "%"
        cells = (
            row.id,
            escape(row.label),
            change.day.isoformat(),
            f"최신 {display.fmt(change.latest, row.fmt)}",
            f"일간 {_shown(change.daily, unit)}",
            f"주간 {_shown(change.weekly, unit)}",
            f"{span} {_shown(change.since, unit)}",
        )
        lines.append(" · ".join(cells))
        ids.update(alias.id for alias in same)
    lines.append("</macro_changes>")
    targets, cited = etf_changes(etf_series, since, span)
    ids.update(cited)
    parts = ["\n".join(lines), targets]
    return Document("\n\n".join(parts), frozenset(ids), base.snapshot, base.text)


BAR_COLUMNS: Final = ("open", "high", "low", "close", "adj_close", "volume")
BASIC_FIELDS: Final = (
    ("sector", "섹터"),
    ("industry", "산업"),
    ("country", "국가"),
    ("currency", "통화"),
    ("market_cap", "시가총액"),
    ("per", "PER"),
    ("forward_per", "선행 PER"),
    ("dividend_yield", "배당수익률(%)"),
    ("roe", "ROE(%)"),
    ("net_margin", "순이익률(%)"),
    ("revenue_growth", "매출 성장률(%)"),
)


@dataclass(frozen=True, slots=True)
class StockInputs:
    ticker: str
    name: str
    guide: str
    company: Mapping[str, etf.Cell]
    annual: Mapping[str, object]
    bars: pd.DataFrame
    holdings: tuple[str, ...] = ()
    investors: tuple[str, ...] = ()


def _item(cite: str, name: str, source: str, rows: Sequence[str]) -> tuple[str, str]:
    body = "\n".join(rows) or "데이터 없음"
    return element(cite, name, source, body), cite


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _basic_lines(inputs: StockInputs) -> list[str]:
    lines = [f"이름: {inputs.name}"]
    for name, label in BASIC_FIELDS:
        cell = inputs.company.get(name)
        if cell is not None:
            lines.append(_cell_line(label, cell))
    if not inputs.bars.empty:
        last = inputs.bars.sort_values("date").iloc[-1]
        lines.append(
            f"최근 종가: {_bar_cell(last.get('close'))} ({str(last['date'])[:10]})"
        )
    return lines


def _annual_lines(annual: Mapping[str, object]) -> list[str]:
    if not annual:
        return []
    return [
        f"결산일: {annual.get('fy_end') or '—'}",
        f"통화: {annual.get('currency') or '—'}",
    ] + [
        f"{column}: {_bar_cell(annual.get(column)) or '—'}"
        for column in fundamentals.ANNUAL_COLUMNS
    ]


def _bar_lines(bars: pd.DataFrame) -> list[str]:
    columns = [column for column in BAR_COLUMNS if column in bars]
    rows = [",".join(("date", *columns))]
    ordered = bars.sort_values("date")
    for record in fundamentals.records(ordered):
        day = str(record["date"])[:10]
        cells = [_bar_cell(record.get(column)) for column in columns]
        rows.append(",".join((day, *cells)))
    return rows


def _bar_cell(value: object) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.4f}".rstrip("0").rstrip(".")


def _technical_input(bars: pd.DataFrame) -> pd.DataFrame:
    full = bars.reindex(columns=["date", *BAR_COLUMNS])
    if "adj_close" not in bars:
        return full
    factor = full["adj_close"] / full["close"]
    return full.assign(
        open=full["open"] * factor,
        high=full["high"] * factor,
        low=full["low"] * factor,
        close=full["adj_close"],
    )


def _bar_items(prefix: str, bars: pd.DataFrame) -> list[tuple[str, str]]:
    if bars.empty:
        return []
    values = technical.indicators(_technical_input(bars))
    return [
        _item(
            f"{prefix}_prices", "최근 1년 일봉", "앱 시세", _bar_lines(_last_year(bars))
        ),
        _item(
            f"{prefix}_technical",
            "기술 지표",
            "앱 시세로 계산",
            [f"{name}: {value:,.4f}" for name, value in values.items()],
        ),
    ]


def stock_document(inputs: StockInputs, base: Document) -> Document:
    sources = sorted({cell.source for cell in inputs.company.values()})
    items: list[tuple[str, str]] = [
        _item(
            "stk_basic", "기본 정보", " · ".join(sources) or "—", _basic_lines(inputs)
        ),
        _item(
            "stk_annual",
            "연간 재무값 (0: 최근 결산, 1: 그 전 결산)",
            str(inputs.annual.get("source") or "—"),
            _annual_lines(inputs.annual),
        ),
    ]
    items.extend(_bar_items("stk", inputs.bars))
    items.extend(
        (
            _item(
                "stk_holdings", "보유 내역과 거래 기록", "포트폴리오", inputs.holdings
            ),
            _item("stk_13f", "13F 투자자 보유 현황", "SEC 13F", inputs.investors),
        )
    )
    stock = "\n".join(text for text, _ in items)
    ticker = quoteattr(inputs.ticker)
    parts = [inputs.guide] if inputs.guide else []
    parts.append(f"<stock ticker={ticker}>\n{stock}\n</stock>")
    ids = base.ids | {cite for _, cite in items}
    return Document("\n\n".join(parts), ids, base.snapshot, base.text)


def _last_year(bars: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(cast("pd.Series", bars["date"]))
    return cast(
        "pd.DataFrame",
        bars[dates >= dates.max() - pd.DateOffset(years=1)],
    )


def holding_lines(
    book: portfolio.Book, close: pd.DataFrame, ticker: str
) -> tuple[str, ...]:
    lines: list[str] = []
    for held in book.portfolios:
        trades = [trade for trade in held.trades if trade.ticker == ticker]
        if not trades:
            continue
        lines.append(_holding_line(held, close, ticker))
        lines.extend(_trade_line(held.name, trade) for trade in trades)
    return tuple(lines)


def _holding_line(held: portfolio.Portfolio, close: pd.DataFrame, ticker: str) -> str:
    try:
        items = portfolio.holdings(held, close)
    except portfolio.BookError as error:
        return f"{held.name}: {error}"
    item = next((h for h in items if h.ticker == ticker), None)
    if item is None or not item.is_open:
        return f"{held.name}: 현재 보유 수량 없음"
    cells = (
        f"수량 {item.position.quantity:,.4f}",
        f"평균 단가 {item.position.average:,.4f}",
        f"평가액 {item.value:,.2f} USD",
        f"평가손익 {item.unrealized:,.2f} USD",
    )
    return f"{held.name}: " + " · ".join(cells)


def _trade_line(name: str, trade: portfolio.Trade) -> str:
    side = "매수" if trade.side == "buy" else "매도"
    cells = (
        f"수량 {trade.quantity:,.4f}",
        f"가격 {trade.price:,.4f}",
        f"수수료 {trade.fee:,.2f}",
    )
    return f"{name} 거래 {trade.day.isoformat()} {side} " + " · ".join(cells)


def investor_lines(root: Path, ticker: str) -> tuple[str, ...]:
    cusips = investors.load_cusips()
    wanted = {cusip for cusip, link in cusips.items() if link.ticker == ticker}
    lines: list[str] = []
    for person in investors.roster(root) if wanted else ():
        filer = investors.load_filer(person.cik)
        if filer is None:
            continue
        for quarter in investors.quarters(investors.load_tables(filer))[:2]:
            rows = [row for row in quarter.rows if row.cusip in wanted]
            if rows:
                shares = sum(row.shares for row in rows)
                value = sum(row.value for row in rows)
                cells = (
                    person.label,
                    f"분기말 {quarter.period.isoformat()}",
                    f"제출 {quarter.filed.isoformat()}",
                    f"{shares:,.0f}주",
                    f"보고 가치 {value:,.0f}",
                )
                lines.append(" · ".join(cells))
    return tuple(lines)


NO_DATA: Final = "데이터 없음"
NO_PAGE: Final = "화면 데이터 없이 보낸 요청"


def _empty(snapshot: str) -> Document:
    return Document(f"<page_document>{NO_DATA}</page_document>", frozenset(), snapshot)


def request_document(snapshot: str) -> Document:
    return Document(f"<page_document>{NO_PAGE}</page_document>", frozenset(), snapshot)


def _combined(items: Sequence[tuple[str, str]], snapshot: str) -> Document:
    if not items:
        return _empty(snapshot)
    body = "\n".join(text for text, _ in items)
    ids = frozenset(cite for _, cite in items)
    return Document(f"<page_document>\n{body}\n</page_document>", ids, snapshot)


def _policy_item(dashboard: Dashboard) -> tuple[str, str]:
    rows = [
        " · ".join(
            (
                f"{change.bank} {change.date}: {change.prev:g} → {change.rate:g}",
                f"최근 12개월 변경 {change.count_12m}회",
            )
        )
        for change in dashboard.policy_changes
    ]
    return _item("policy_changes", "정책금리 변경", "BIS", rows)


def _schedule_document(dashboard: Dashboard) -> Document:
    calendar = [
        f"{entry.date} {entry.name} ({entry.source})" for entry in dashboard.calendar
    ]
    sources = [
        f"{source.name}: {source.ok}/{source.total} · {source.state}"
        for source in dashboard.sources
    ]
    rows = [
        f"{row.label} · {row.code} · {row.status}"
        for panel in dashboard.panels
        for row in panel.rows
    ]
    return _combined(
        (
            _item("calendar", "발표 일정", "FRED releases", calendar),
            _item("fomc_dates", "FOMC 일정", "FRED", dashboard.fomc_dates),
            _item("sources", "출처 상태", "FinAgent", sources),
            _item("indicators", "지표 목록", "FinAgent", rows),
        ),
        dashboard.generated_at,
    )


def page_document(dashboard: Dashboard, page: str) -> Document:
    if page == "schedule":
        return _schedule_document(dashboard)
    panels = tuple(panel for panel in dashboard.panels if panel.page == page)
    overview = page == "overview"
    if not panels and not overview:
        return _empty(dashboard.generated_at)
    base = build_document(
        dashboard.model_copy(
            update={
                "panels": panels,
                "regime": dashboard.regime if overview else (),
                "curves": tuple(c for c in dashboard.curves if c.area == "US")
                if overview
                else (),
                "spread": dashboard.spread if overview else None,
                "projections": dashboard.projections if overview else None,
                "calendar": (),
            }
        )
    )
    if page not in {"overview", "monetary"}:
        return base
    text, cite = _policy_item(dashboard)
    return Document(f"{base.text}\n\n{text}", base.ids | {cite}, base.snapshot)


@dataclass(frozen=True, slots=True)
class EtfView:
    group: str | None
    category: str | None
    period: str
    rows: pd.DataFrame


def etf_document(view: EtfView, snapshot: str) -> Document:
    columns = [str(c) for c in view.rows.columns]
    items: list[tuple[str, str]] = []
    for values in view.rows.itertuples(index=False):
        row = dict(zip(columns, cast("tuple[object, ...]", values), strict=True))
        cells = [f"{c}: {etf.cell_text(c, row[c])}" for c in columns]
        name = f"{row['티커']} {etf.cell_text('이름', row['이름'])}"
        items.append(
            _item(f"etf_{row['티커']}", name, "FinAgent ETF", [" · ".join(cells)])
        )
    if not items:
        return _empty(snapshot)
    picked = _combined(items, snapshot)
    head = " · ".join(
        (
            f"분류: {view.group or '전체'}",
            f"카테고리: {view.category or '전체'}",
            f"차트 기간: {view.period}",
            f"{len(items)}개",
        )
    )
    return Document(f"{escape(head)}\n{picked.text}", picked.ids, snapshot)


def portfolio_document(values: Mapping[str, str], snapshot: str) -> Document:
    return _combined(
        [
            _item(f"pv_{index}", label, "포트폴리오", [text])
            for index, (label, text) in enumerate(values.items(), 1)
        ],
        snapshot,
    )


ETF_FIELDS: Final = (
    ("name", "이름"),
    ("kind", "상품 종류"),
    ("family", "운용사"),
    ("inception", "설정일"),
    ("aum", "운용자산(USD)"),
    ("expense", "총보수(%)"),
    ("yield", "분배수익률(%)"),
    ("nav", "NAV"),
    ("previous_close", "전일 종가"),
    ("sector_weights", "섹터 비중(%)"),
    ("credit_weights", "신용등급 비중(%)"),
)
RANKED: Final = (*etf.PERIODS, "총보수", "운용자산", "거래대금")
LOWER_IS_BETTER: Final = frozenset({"총보수"})
PEERS: Final = 5
YEAR_BARS: Final = 252
COUNTRIES: Final = 10


@dataclass(frozen=True, slots=True)
class EtfInputs:
    ticker: str
    guide: str
    funds: Mapping[str, etf.Cell]
    table: pd.DataFrame
    composition: etf.Composition | None
    bars: pd.DataFrame
    holdings: tuple[str, ...] = ()
    fit: tuple[str, ...] = ()


def _cell_line(label: str, cell: etf.Cell) -> str:
    value = cell.value
    if isinstance(value, dict):
        parts = cast("dict[str, object]", value)
        text = ", ".join(f"{k} {v}" for k, v in parts.items())
    else:
        number = _number(value)
        text = str(value) if number is None else f"{number:,.4g}"
    return f"{label}: {text} ({cell.source}, {cell.as_of})"


def _etf_basic(inputs: EtfInputs) -> list[str]:
    lines = [
        _cell_line(label, inputs.funds[name])
        for name, label in ETF_FIELDS
        if name in inputs.funds
    ]
    nav, close = inputs.funds.get("nav"), inputs.funds.get("previous_close")
    if nav is not None and close is not None:
        a, b = _number(nav.value), _number(close.value)
        if a and b is not None:
            lines.append(f"NAV 대비 괴리(%): {(b / a - 1) * 100:+.3f}")
    return lines


def _etf_row(inputs: EtfInputs) -> pd.Series | None:
    found = inputs.table[inputs.table["티커"] == inputs.ticker]
    return None if found.empty else cast("pd.Series", found.iloc[0])


def _risk_lines(bars: pd.DataFrame) -> list[str]:
    if bars.empty or "adj_close" not in bars:
        return []
    close = cast(
        "pd.Series", _last_year(bars).sort_values("date")["adj_close"]
    ).dropna()
    if len(close) < 2:  # noqa: PLR2004  # 일간 수익률 산출을 위한 최소 관측일 수(2일)
        return []
    daily = close.pct_change().dropna()
    vol = float(cast("float", daily.std())) * math.sqrt(YEAR_BARS) * 100
    drawdown = float(cast("float", (close / close.cummax() - 1).min())) * 100
    total = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100
    lines = [f"1년 변동성(%): {vol:,.2f}", f"1년 최대 낙폭(%): {drawdown:,.2f}"]
    if vol > 0:
        lines.append(f"1년 수익률 ÷ 변동성: {total / vol:,.2f}")
    return lines


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    found = cast("pd.Series", pd.to_numeric(frame[name], errors="coerce"))
    return found.dropna()


def _pool(table: pd.DataFrame, category: object) -> pd.DataFrame:
    return cast(
        "pd.DataFrame", table[(table["카테고리"] == category) & (table["배수"] == 1)]
    )


def _etf_measures(inputs: EtfInputs) -> list[str]:
    row = _etf_row(inputs)
    if row is None:
        return _risk_lines(inputs.bars)
    same = _pool(inputs.table, row["카테고리"])
    ranked = row["배수"] == 1
    lines = [f"카테고리: {row['카테고리']} ({len(same)}개)"]
    for column in RANKED:
        value = _number(row[column])
        if value is None:
            continue
        text = f"{column}: {etf.cell_text(column, value)}"
        if ranked:
            values = _column(same, column).to_numpy(dtype="float64")
            better = values < value if column in LOWER_IS_BETTER else values > value
            text += f" · 카테고리 {int(better.sum()) + 1}/{len(values)}위"
        lines.append(text)
    return lines + _risk_lines(inputs.bars)


def _etf_peers(inputs: EtfInputs) -> list[str]:
    row = _etf_row(inputs)
    if row is None:
        return []
    pool = _pool(inputs.table, row["카테고리"])
    same = cast("pd.DataFrame", pool[pool["티커"] != inputs.ticker])
    top = same.sort_values(by="운용자산", ascending=False, na_position="last")
    top = top.head(PEERS)
    columns = [c for c in RANKED if c in top]
    lines = [
        f"{r['티커']} {r['이름']}: "
        + " · ".join(f"{c} {etf.cell_text(c, r[c])}" for c in columns)
        for _, r in top.iterrows()
    ]
    if lines:
        medians = [
            f"{c} {etf.cell_text(c, float(cast('float', _column(top, c).median())))}"
            for c in columns
        ]
        lines.append("중앙값: " + " · ".join(medians))
    return lines


def holding_stats(
    found: etf.Composition,
) -> tuple[float, float | None, list[tuple[str, float]], float | None]:
    rows = found.rows
    squared = sum((r.weight / 100) ** 2 for r in rows if r.asset != "collateral")
    maturity = (
        fund_filings.maturity_years(fund_filings.Holdings(found.as_of, rows))
        if found.full and found.as_of and rows
        else None
    )
    return (
        sum(r.weight for r in fund_filings.top(rows)),
        1 / squared if squared > 0 else None,
        list(fund_filings.countries(rows).items())[:COUNTRIES],
        maturity,
    )


def _etf_holdings(found: etf.Composition | None) -> list[str]:
    if found is None:
        return []
    rows = found.rows
    lines = [f"출처: {found.source} · 기준일 {found.as_of}"]
    if found.assets:
        mix = ", ".join(f"{k} {v:.1f}%" for k, v in found.assets.items())
        lines.append(f"자산 구성: {mix}")
    if not rows:
        return lines
    lines.extend(
        f"{r.ticker or r.isin or r.cusip or '—'} {r.name}: {r.weight:.2f}%"
        for r in fund_filings.top(rows)
    )
    top, effective, lands, maturity = holding_stats(found)
    lines.append(f"상위 {fund_filings.TOP_ROWS} 비중(%): {top:.2f}")
    if effective is not None:
        lines.append(f"유효 종목 수: {effective:,.1f}")
    if lands:
        shown = ", ".join(f"{k} {v:.1f}%" for k, v in lands)
        lines.append(f"국가: {shown}")
    if maturity is not None:
        lines.append(f"채권 가중 잔존만기(년): {maturity:.2f}")
    return lines


def etf_analysis_document(inputs: EtfInputs, base: Document) -> Document:
    items = [
        _item("etf_basic", "기본 정보", "FinAgent ETF", _etf_basic(inputs)),
        _item(
            "etf_measures",
            "측정값과 카테고리 내 순위",
            "FinAgent ETF",
            _etf_measures(inputs),
        ),
        _item(
            "etf_peers",
            f"같은 카테고리 운용자산 상위 {PEERS}개",
            "FinAgent ETF",
            _etf_peers(inputs),
        ),
        _item(
            "etf_holdings",
            "구성",
            inputs.composition.source if inputs.composition else "—",
            _etf_holdings(inputs.composition),
        ),
    ]
    items.extend(_bar_items("etf", inputs.bars))
    items.extend(
        (
            _item(
                "etf_position", "보유 내역과 거래 기록", "포트폴리오", inputs.holdings
            ),
            _item("etf_fit", "포트폴리오 편입 효과", "포트폴리오", inputs.fit),
        )
    )
    body = "\n".join(text for text, _ in items)
    parts = [inputs.guide, f"<etf ticker={quoteattr(inputs.ticker)}>\n{body}\n</etf>"]
    ids = base.ids | {cite for _, cite in items}
    return Document("\n\n".join(parts), ids, base.snapshot, base.text)
