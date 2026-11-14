# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""Yahoo Finance(yfinance) 시세·기업 정보·펀드 메타데이터 조회 어댑터."""

from __future__ import annotations

import math
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Final, cast

import pandas as pd

from finagent import limits, memory
from finagent.limits import BACKGROUND_WAIT

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from datetime import date

    from yfinance import Ticker

type Json = dict[str, object]

BATCH: Final = 100
WORKERS: Final = 4
NOT_FOUND: Final = 404
FUND_KINDS: Final = frozenset({"ETF"})
STOCK_KINDS: Final = frozenset({"EQUITY"})

_lock: Final = threading.Lock()
_folder: tempfile.TemporaryDirectory[str] | None = None


def use_cache() -> None:
    global _folder  # noqa: PLW0603
    import yfinance as yf  # noqa: PLC0415

    with _lock:
        if _folder is None:
            _folder = tempfile.TemporaryDirectory(prefix="finagent-yfinance-")
            yf.set_tz_cache_location(_folder.name)
            yf.config.debug.hide_exceptions = False


@contextmanager
def _call(cost: int, max_wait: float) -> Generator[None]:
    from yfinance.exceptions import YFException, YFRateLimitError  # noqa: PLC0415

    limiter = limits.current()
    with limiter.slot("YAHOO", max_wait):
        limiter.acquire("YAHOO", cost, max_wait)
        try:
            yield
        except YFRateLimitError as error:
            limiter.report_limited("YAHOO")
            raise limiter.reached("YAHOO") from error
        except (OSError, YFException) as error:
            msg = f"Yahoo 요청 실패 ({error})"
            raise limits.SourceError(msg) from error


def _absent(error: Exception) -> bool:
    from yfinance.exceptions import YFDataException  # noqa: PLC0415

    if isinstance(error, YFDataException | KeyError):
        return True
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == NOT_FOUND


@contextmanager
def ticker(symbol: str, max_wait: float = BACKGROUND_WAIT) -> Generator[Ticker]:
    use_cache()
    import yfinance as yf  # noqa: PLC0415

    with _call(1, max_wait):
        yield yf.Ticker(symbol)


def download(
    tickers: Sequence[str],
    start: date,
    end: date | None = None,
    max_wait: float = BACKGROUND_WAIT,
) -> pd.DataFrame:
    use_cache()
    import yfinance as yf  # noqa: PLC0415

    frames: list[pd.DataFrame] = []
    names = list(tickers)
    for i in range(0, len(names), BATCH):
        batch = names[i : i + BATCH]
        with _call(len(batch), max_wait):
            frame = yf.download(
                batch,
                start=start.isoformat(),
                end=None if end is None else end.isoformat(),
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=True,
            )
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)


def _percents(raw: Mapping[str, object]) -> dict[str, float]:
    found = {
        name: round(float(value) * 100, 2)
        for name, value in raw.items()
        if isinstance(value, int | float)
    }
    ordered = sorted(found.items(), key=lambda item: item[1], reverse=True)
    return {name: value for name, value in ordered if value}


def _saved_info(symbol: str) -> Mapping[str, object] | None:
    saved = memory.current().yahoo_info.get(symbol)
    today = datetime.now().astimezone().date()
    return saved[1] if saved is not None and saved[0] == today else None


def stock_info(stock: Ticker, symbol: str) -> Mapping[str, object]:
    saved = _saved_info(symbol)
    if saved is not None:
        return saved
    try:
        found = cast("Mapping[str, object]", stock.info or {})
    except Exception as error:
        if not _absent(error):
            raise
        return {}
    if found.get("quoteType"):
        today = datetime.now().astimezone().date()
        table = memory.current().yahoo_info
        for old, (day, _) in table.rows().items():
            if day != today:
                table.drop(old)
        table.put(symbol, (today, found))
    return found


def info(symbol: str, max_wait: float = BACKGROUND_WAIT) -> Mapping[str, object]:
    found = _saved_info(symbol)
    if found is not None:
        return found
    with ticker(symbol, max_wait) as stock:
        return stock_info(stock, symbol)


def fund(symbol: str, max_wait: float) -> Json:
    with ticker(symbol, max_wait) as stock:
        try:
            data = stock.funds_data
            holdings = data.top_holdings
            assets = cast("Mapping[str, object]", data.asset_classes)
        except Exception as error:
            if not _absent(error):
                raise
            return {}
    return {
        "top_holdings": [
            {
                "ticker": str(name),
                "name": row.get("Name"),
                "weight_pct": round(
                    float(cast("float", row["Holding Percent"])) * 100, 2
                ),
            }
            for name, row in holdings.iterrows()
        ],
        "asset_classes_pct": _percents(
            {name.removesuffix("Position"): v for name, v in assets.items()}
        ),
    }


def _number(raw: Mapping[str, object], name: str) -> float | None:
    value = raw.get(name)
    return (
        float(value)
        if isinstance(value, int | float) and not math.isnan(value)
        else None
    )


def fund_info(symbol: str, max_wait: float) -> Json:
    sectors: Mapping[str, object] = {}
    ratings: Mapping[str, object] = {}
    with ticker(symbol, max_wait) as stock:
        found = stock_info(stock, symbol)
        if str(found.get("quoteType")).upper() in FUND_KINDS:
            try:
                data = stock.funds_data
                sectors = cast("Mapping[str, object]", data.sector_weightings)
                ratings = cast("Mapping[str, object]", data.bond_ratings)
            except Exception as error:
                if not _absent(error):
                    raise
    inception = _number(found, "fundInceptionDate")
    found_yield = _number(found, "yield")
    return {
        "kind": found.get("quoteType"),
        "name": found.get("longName"),
        "family": found.get("fundFamily"),
        "total_assets": _number(found, "totalAssets"),
        "expense": _number(found, "netExpenseRatio"),
        "yield": None if found_yield is None else found_yield * 100,
        "inception": None
        if inception is None
        else pd.Timestamp(inception, unit="s").date().isoformat(),
        "sector_weights": _percents(sectors) or None,
        "credit_weights": _percents(ratings) or None,
    }


def fund_quote(symbol: str, max_wait: float) -> Json:
    found = info(symbol, max_wait)
    return {
        "nav": _number(found, "navPrice"),
        "previous_close": _number(found, "previousClose"),
    }


def quote_type(symbol: str, max_wait: float) -> object:
    return info(symbol, max_wait).get("quoteType")
