from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import date
from typing import TYPE_CHECKING

import httpx
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import pytest

from finagent.fetch import FetchContext, MissingKeyError, SourceError
from finagent.sources import (
    EcosSeries,
    FiscalSeries,
    FredSeries,
    SdmxSeries,
    YahooCloses,
    YahooSeries,
    ecos_period,
    fred_release_dates,
    fred_vintages,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

TODAY = date(2026, 10, 12)
KEYS = {"FRED_API_KEY": "fred-secret-key", "ECOS_API_KEY": "ecos-secret-key"}


def run[T](
    handler: Callable[[httpx.Request], httpx.Response],
    call: Callable[[FetchContext], Coroutine[None, None, T]],
    env: dict[str, str] | None = None,
) -> T:
    async def main() -> T:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            ctx = FetchContext(client, KEYS if env is None else env, retry_delay=0)
            return await call(ctx)

    return asyncio.run(main())


def json_response(payload: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode())


def test_fred_sends_key_and_skips_missing_dots() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response(
            {
                "observations": [
                    {"date": "2026-10-08", "value": "4.10"},
                    {"date": "2026-10-09", "value": "."},
                ]
            }
        )

    series = run(handler, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))

    assert series == ((date(2026, 10, 8), 4.1),)
    params = seen[0].url.params
    assert params["series_id"] == "DGS10"
    assert params["api_key"] == "fred-secret-key"
    assert params["file_type"] == "json"
    assert params["observation_start"] == "2022-10-12"


def test_missing_key_is_reported_without_a_request() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError

    with pytest.raises(MissingKeyError, match="FRED_API_KEY"):
        _ = run(handler, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY), env={})


def test_http_errors_never_leak_the_key() -> None:
    def server_error(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    def network_error(request: httpx.Request) -> httpx.Response:
        message = f"cannot reach {request.url}"
        raise httpx.ConnectError(message, request=request)

    for handler in (server_error, network_error):
        with pytest.raises(SourceError) as caught:
            _ = run(handler, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))
        assert "fred-secret-key" not in str(caught.value)


def test_same_request_in_one_pass_is_sent_once() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return json_response({"observations": [{"date": "2026-10-08", "value": "1"}]})

    async def twice(ctx: FetchContext) -> None:
        spec = FredSeries("DGS10")
        _ = await asyncio.gather(spec.fetch(ctx, TODAY), spec.fetch(ctx, TODAY))

    run(handler, twice)

    assert len(calls) == 1


def test_fiscal_data_reads_string_values_and_filters() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response(
            {
                "data": [
                    {"record_date": "2026-09-30", "avg_interest_rate_amt": "3.390"},
                    {"record_date": "2026-08-31", "avg_interest_rate_amt": "null"},
                ]
            }
        )

    spec = FiscalSeries(
        "v2/accounting/od/avg_interest_rates",
        "avg_interest_rate_amt",
        "security_desc:eq:Total Marketable",
    )
    series = run(handler, lambda ctx: spec.fetch(ctx, TODAY))

    assert series == ((date(2026, 9, 30), 3.39),)
    params = seen[0].url.params
    assert params["filter"] == (
        "record_date:gte:2022-10-12,security_desc:eq:Total Marketable"
    )
    assert params["fields"] == "record_date,avg_interest_rate_amt"


def test_ecos_builds_path_and_parses_rows() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response(
            {
                "StatisticSearch": {
                    "list_total_count": 1,
                    "row": [{"TIME": "202609", "DATA_VALUE": "2.1"}],
                }
            }
        )

    spec = EcosSeries("511Y002", "M", "FME", "99988")
    series = run(handler, lambda ctx: spec.fetch(ctx, TODAY))

    assert series == ((date(2026, 9, 1), 2.1),)
    assert seen[0].url.path == (
        "/api/StatisticSearch/ecos-secret-key/json/kr/1/10000/"
        "511Y002/M/202210/202610/FME/99988"
    )


def test_ecos_result_block_is_an_error_without_the_key() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return json_response(
            {"RESULT": {"CODE": "INFO-100", "MESSAGE": "인증키가 유효하지 않습니다."}}
        )

    with pytest.raises(SourceError, match="INFO-100") as caught:
        _ = run(
            handler, lambda ctx: EcosSeries("722Y001", "D", "0101000").fetch(ctx, TODAY)
        )
    assert "ecos-secret-key" not in str(caught.value)


@pytest.mark.parametrize(
    ("cycle", "expected"),
    [("D", "20261012"), ("M", "202610"), ("Q", "2026Q4"), ("A", "2026")],
)
def test_ecos_period_formats(cycle: str, expected: str) -> None:
    assert ecos_period(TODAY, cycle) == expected


def test_sdmx_picks_matching_rows_from_a_multi_series_csv() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "csvfile"
        return httpx.Response(
            200,
            content=(
                "\ufeffDATAFLOW,REF_AREA,FREQ,TIME_PERIOD,OBS_VALUE\n"
                "X,USA,M,2026-08,100.3\n"
                "X,KOR,M,2026-08,100.6\n"
                "X,KOR,M,2026-09,\n"
            ).encode(),
        )

    spec = SdmxSeries("OECD", "flow/key", (("REF_AREA", "KOR"),))
    series = run(handler, lambda ctx: spec.fetch(ctx, TODAY))

    assert series == ((date(2026, 8, 1), 100.6),)


def test_sdmx_without_expected_columns_is_an_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>maintenance</html>")

    with pytest.raises(SourceError, match="TIME_PERIOD"):
        _ = run(
            handler,
            lambda ctx: SdmxSeries("ECB", "EXR/D.USD.EUR.SP00.A").fetch(ctx, TODAY),
        )


def test_release_dates_are_limited_to_the_window() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "release_dates": [
                    {
                        "release_id": 10,
                        "release_name": "Consumer Price Index",
                        "date": "2026-10-14",
                    },
                    {
                        "release_id": 10,
                        "release_name": "Consumer Price Index",
                        "date": "2026-11-13",
                    },
                ]
            }
        )

    releases = run(
        handler,
        lambda ctx: fred_release_dates(ctx, TODAY, date(2026, 10, 19)),
    )

    assert releases == ((date(2026, 10, 14), "Consumer Price Index"),)


def test_vintages_pick_the_current_and_the_replaced_projection() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        rows = [
            ("2026-06-17", "2026-09-16", "2026-01-01", "3.9"),
            ("2026-06-17", "2026-09-16", "2027-01-01", "3.6"),
            ("2026-09-17", "9999-12-31", "2026-01-01", "3.6"),
            ("2026-09-17", "9999-12-31", "2027-01-01", "3.4"),
            ("2026-09-17", "9999-12-31", "2028-01-01", "3.1"),
            ("2025-10-12", "9999-12-31", "2025-01-01", "4.4"),
        ]
        return json_response(
            {
                "observations": [
                    {"realtime_start": a, "realtime_end": b, "date": c, "value": v}
                    for a, b, c, v in rows
                ]
            }
        )

    latest, prior = run(
        handler, lambda ctx: fred_vintages(ctx, "FEDTARMD", date(2025, 10, 12))
    )

    assert latest.published == date(2026, 9, 17)
    assert latest.series == (
        (date(2025, 1, 1), 4.4),
        (date(2026, 1, 1), 3.6),
        (date(2027, 1, 1), 3.4),
        (date(2028, 1, 1), 3.1),
    )
    assert prior is not None
    assert prior.published == date(2026, 6, 17)
    assert dict(prior.series)[date(2027, 1, 1)] == 3.6
    assert date(2028, 1, 1) not in dict(prior.series)
    params = seen[0].url.params
    assert params["realtime_start"] == "2025-10-12"
    assert params["realtime_end"] == "9999-12-31"


def test_server_errors_are_retried_once_only() -> None:
    calls: list[int] = []

    def flaky(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(502)
        return json_response({"observations": [{"date": "2026-10-08", "value": "4"}]})

    series = run(flaky, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))

    assert series == ((date(2026, 10, 8), 4.0),)
    assert len(calls) == 2

    calls.clear()

    def down(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(SourceError, match="FRED HTTP 503"):
        _ = run(down, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))
    assert len(calls) == 2  # gives up after one retry


@pytest.mark.parametrize(
    ("status", "message"), [(429, "FRED 요청 한도 초과"), (404, "FRED HTTP 404")]
)
def test_client_errors_are_not_retried(status: int, message: str) -> None:
    calls: list[int] = []

    def refuse(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status)

    with pytest.raises(SourceError) as caught:
        _ = run(refuse, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))
    assert str(caught.value) == message
    assert "fred-secret-key" not in str(caught.value)
    assert len(calls) == 1


# ---------------------------------------------------------------- Yahoo


def fake_yfinance(
    monkeypatch: pytest.MonkeyPatch, answer: pd.DataFrame | Exception
) -> list[dict[str, object]]:
    """Stand in for the yfinance module; returns the ``history`` calls."""
    calls: list[dict[str, object]] = []

    class Ticker:
        def __init__(self, ticker: str) -> None:
            self.ticker: str = ticker

        def history(self, **kwargs: object) -> pd.DataFrame:
            calls.append({"ticker": self.ticker, **kwargs})
            if isinstance(answer, Exception):
                raise answer
            return answer

    def point(_: str) -> None:
        pass

    fake = types.ModuleType("yfinance")
    fake.set_tz_cache_location = point  # pyright: ignore[reportAttributeAccessIssue]
    fake.Ticker = Ticker  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return calls


def test_yahoo_closes_by_exchange_day_without_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = pd.DatetimeIndex(["2026-10-08", "2026-10-09"], tz="America/New_York")
    frame = pd.DataFrame({"Open": [1.0, 2.0], "Close": [4010.5, None]}, index=days)
    calls = fake_yfinance(monkeypatch, frame)

    series = YahooCloses(tmp_path)("GC=F", date(2022, 10, 12))

    assert series == ((date(2026, 10, 8), 4010.5),)
    assert calls == [
        {
            "ticker": "GC=F",
            "start": "2022-10-12",
            "auto_adjust": False,
            "raise_errors": True,
        }
    ]


def test_yahoo_failures_are_source_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = fake_yfinance(monkeypatch, ConnectionError("down"))
    with pytest.raises(SourceError, match="YAHOO GC=F: ConnectionError"):
        _ = YahooCloses(tmp_path)("GC=F", TODAY)

    _ = fake_yfinance(monkeypatch, pd.DataFrame({"Close": []}))
    with pytest.raises(SourceError, match="YAHOO GC=F: 관측값 없음"):
        _ = YahooCloses(tmp_path)("GC=F", TODAY)


def test_yahoo_without_yfinance_is_a_missing_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "yfinance", None)

    with pytest.raises(MissingKeyError, match="yfinance가 설치되지 않았습니다"):
        _ = YahooCloses(tmp_path / "cache")("GC=F", TODAY)
    assert not (tmp_path / "cache").exists()


def test_yahoo_series_reads_closes_through_the_context() -> None:
    asked: list[tuple[str, date]] = []

    def closes(ticker: str, start: date) -> tuple[tuple[date, float], ...]:
        asked.append((ticker, start))
        return ((TODAY, 4000.0),)

    async def main() -> tuple[tuple[date, float], ...]:
        async with httpx.AsyncClient() as client:
            ctx = FetchContext(client, {}, yahoo=closes)
            return await YahooSeries("GC=F", years=1).fetch(ctx, TODAY)

    assert asyncio.run(main()) == ((TODAY, 4000.0),)
    assert asked == [("GC=F", date(2025, 10, 12))]

    def unused(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(SourceError, match="YAHOO 시세를 받을 경로가 없습니다"):
        _ = run(unused, lambda ctx: YahooSeries("GC=F").fetch(ctx, TODAY))
