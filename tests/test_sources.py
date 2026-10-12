from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import TYPE_CHECKING

import httpx
import pytest

from finagent.fetch import FetchContext, MissingKeyError, SourceError
from finagent.sources import (
    EcosSeries,
    FiscalSeries,
    FredSeries,
    NyFedRate,
    SdmxSeries,
    ecos_period,
    fred_release_dates,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

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


def test_nyfed_reference_rates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/rates/secured/sofr/last/1000.json"
        return json_response(
            {
                "refRates": [
                    {"effectiveDate": "2026-10-09", "type": "SOFR", "percentRate": 3.61}
                ]
            }
        )

    series = run(handler, lambda ctx: NyFedRate("secured/sofr").fetch(ctx, TODAY))

    assert series == ((date(2026, 10, 9), 3.61),)


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
                "﻿DATAFLOW,REF_AREA,FREQ,TIME_PERIOD,OBS_VALUE\n"
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


def test_server_errors_are_retried_once() -> None:
    calls: list[int] = []

    def flaky(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(502)
        return json_response({"observations": [{"date": "2026-10-08", "value": "4"}]})

    series = run(flaky, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))

    assert series == ((date(2026, 10, 8), 4.0),)
    assert len(calls) == 2


def test_repeated_server_errors_give_up_after_one_retry() -> None:
    calls: list[int] = []

    def down(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(SourceError, match="FRED HTTP 503"):
        _ = run(down, lambda ctx: FredSeries("DGS10").fetch(ctx, TODAY))
    assert len(calls) == 2


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
