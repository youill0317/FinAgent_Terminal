"""SEC N-PORT(보유 종목·자산 배분·채권 만기) 및 N-CSR(총보수율) XML 파서."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

type Asset = Literal["stock", "bond", "collateral", "fund", "other"]

STOCK_CATEGORIES: Final = frozenset({"EC", "EP"})
COLLATERAL: Final = "STIV"
FUND_ISSUER: Final = "RF"
EXPAND_LIMIT: Final = 100.5
TOP_ROWS: Final = 10
DAYS_PER_YEAR: Final = 365.25
EMPTY_IDS: Final = frozenset({"", "N/A", "NONE", "000000000"})
CLASS_ID: Final = re.compile(r"C\d{9}")


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    cusip: str | None
    isin: str | None
    ticker: str | None
    weight: float
    asset: Asset
    country: str | None
    maturity: str | None


@dataclass(frozen=True, slots=True)
class Holdings:
    as_of: str
    rows: tuple[Row, ...]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(node: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in node if _local(item.tag) == name), None)


def _text(node: ET.Element, name: str) -> str | None:
    found = _child(node, name)
    text = None if found is None or found.text is None else found.text.strip()
    return text or None


def _identifier(value: str | None) -> str | None:
    if value is None or value.strip().upper() in EMPTY_IDS:
        return None
    return value.strip().upper()


def _asset(category: str | None, issuer: str | None, debt: bool) -> Asset:
    if category == COLLATERAL:
        return "collateral"
    if debt:
        return "bond"
    if issuer == FUND_ISSUER:
        return "fund"
    if category in STOCK_CATEGORIES:
        return "stock"
    return "other"


def _row(node: ET.Element) -> Row:
    ids = _child(node, "identifiers")
    isin = ticker = None
    if ids is not None:
        found = _child(ids, "isin")
        isin = None if found is None else found.get("value")
        found = _child(ids, "ticker")
        ticker = None if found is None else found.get("value")
    debt = _child(node, "debtSec")
    return Row(
        name=_text(node, "name") or "",
        cusip=_identifier(_text(node, "cusip")),
        isin=_identifier(isin),
        ticker=_identifier(ticker),
        weight=float(_text(node, "pctVal") or 0),
        asset=_asset(
            _text(node, "assetCat"), _text(node, "issuerCat"), debt is not None
        ),
        country=_text(node, "invCountry"),
        maturity=None if debt is None else _text(debt, "maturityDt"),
    )


def parse_nport(body: str) -> Holdings:
    root = ET.fromstring(body)  # noqa: S314  # SEC EDGAR 아카이브 수신 N-PORT XML 파싱
    as_of = next(
        (item.text or "" for item in root.iter() if _local(item.tag) == "repPdDate"),
        "",
    )
    rows = tuple(_row(item) for item in root.iter() if _local(item.tag) == "invstOrSec")
    return Holdings(as_of.strip(), rows)


def parse_ncsr(body: str) -> dict[str, float]:
    root = ET.fromstring(body)  # noqa: S314  # SEC EDGAR 아카이브 수신 N-CSR XBRL 파싱
    classes: dict[str, list[str]] = {
        item.get("id", ""): CLASS_ID.findall(ET.tostring(item, encoding="unicode"))
        for item in root
        if _local(item.tag) == "context"
    }
    found: dict[str, float] = {}
    for item in root:
        if _local(item.tag) != "ExpenseRatioPct" or not item.text:
            continue
        for class_id in classes.get(item.get("contextRef", ""), []):
            _ = found.setdefault(class_id, float(item.text))
    return found


def asset_mix(rows: tuple[Row, ...]) -> dict[str, float]:
    stock = sum(row.weight for row in rows if row.asset == "stock")
    bond = sum(row.weight for row in rows if row.asset == "bond")
    other = sum(row.weight for row in rows if row.asset in {"fund", "other"})
    return {
        "stock": stock,
        "bond": bond,
        "other": other,
        "cash": max(0.0, 100 - stock - bond - other),
    }


def top(rows: tuple[Row, ...], count: int = TOP_ROWS) -> list[Row]:
    kept = [row for row in rows if row.asset != "collateral"]
    return sorted(kept, key=lambda row: row.weight, reverse=True)[:count]


def countries(rows: tuple[Row, ...]) -> dict[str, float]:
    kept = [row for row in rows if row.asset != "collateral" and row.country]
    total = sum(row.weight for row in kept)
    if total <= 0:
        return {}
    found: dict[str, float] = {}
    for row in kept:
        key = row.country or ""
        found[key] = found.get(key, 0.0) + row.weight / total * 100
    return dict(sorted(found.items(), key=lambda item: item[1], reverse=True))


def maturity_years(holdings: Holdings) -> float | None:
    start = date.fromisoformat(holdings.as_of)
    pairs = [
        ((date.fromisoformat(row.maturity) - start).days / DAYS_PER_YEAR, row.weight)
        for row in holdings.rows
        if row.asset == "bond" and row.maturity
    ]
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return None
    return sum(years * weight for years, weight in pairs) / total


def expandable(rows: tuple[Row, ...]) -> bool:
    stock = sum(row.weight for row in rows if row.asset == "stock")
    return bool(rows) and stock <= EXPAND_LIMIT
