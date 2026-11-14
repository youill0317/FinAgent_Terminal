# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnknownLambdaType=false

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import TYPE_CHECKING, cast

import httpx

from finagent import (
    sec,
)
from finagent.analysis import cites, tools
from finagent.analysis.company_tools import Company
from finagent.service import SeriesStore

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from finagent.analysis.tools import ToolResult
    from finagent.fetch import FetchContext

TODAY = date(2026, 11, 6)
ACCESSION = "0000320193-26-000071"
TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
FACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"end": "2015-09-26", "val": 1.0, "fy": 2015, "fp": "FY",
                         "form": "10-K", "filed": "2015-10-28", "frame": "CY2015"}
                    ]
                }
            },
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        {"end": "2026-06-27", "val": 90.0, "fy": 2026, "fp": "Q3",
                         "form": "10-Q", "filed": "2026-08-01", "frame": "CY2026Q2"},
                        {"end": "2026-06-27", "val": 94.0, "fy": 2027, "fp": "Q3",
                         "form": "10-Q", "filed": "2027-08-01", "frame": "CY2026Q2"},
                        {"end": "2026-06-27", "val": 400.0, "fy": 2026, "fp": "Q3",
                         "form": "10-Q", "filed": "2026-08-01"},
                    ]
                }
            },
        }
    }
}  # fmt: skip
SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q"],
            "accessionNumber": [ACCESSION, "0000320193-26-000060"],
            "filingDate": ["2026-08-01", "2026-05-02"],
            "items": ["2.02,9.01", ""],
            "primaryDocument": ["a8k.htm", "a10q.htm"],
        }
    }
}
INDEX = {
    "directory": {
        "item": [
            {"name": "a8k.htm"},
            {"name": "ex99-1.htm"},
            {"name": "index.json"},
        ]
    }
}


def company(handler: Callable[[httpx.Request], httpx.Response]) -> Company:
    client = sec.Client("Test test@example.com")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))  # noqa: SLF001
    return Company({}, TODAY, _client=client)


def sec_handler(
    asked: list[str], page: str = "<html><p>본문</p></html>"
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        asked.append(path)
        if path.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "companyfacts" in path:
            return httpx.Response(200, json=FACTS)
        if "submissions" in path:
            return httpx.Response(200, json=SUBMISSIONS)
        if path.endswith("index.json"):
            return httpx.Response(200, json=INDEX)
        return httpx.Response(200, text=page)

    return handler


def run_tool(
    name: str,
    args: Mapping[str, object],
    found: Company,
    ids: dict[str, str] | None = None,
) -> ToolResult:
    async def run() -> ToolResult:
        box = tools.Toolbox(
            cast("FetchContext", cast("object", None)), SeriesStore(), TODAY, found
        )
        result = await box.run(name, args)
        if ids is not None:
            ids.update(box.received)
        return result

    return asyncio.run(run())


def parsed(result: ToolResult) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(result.text))


def test_company_facts_pick_the_live_tag_and_the_latest_filed_value() -> None:
    asked: list[str] = []
    ids: dict[str, str] = {}

    result = run_tool(
        "company_facts",
        {"ticker": "aapl", "metrics": ["revenue"]},
        company(sec_handler(asked)),
        ids,
    )

    assert result.ok
    item = cast("list[dict[str, object]]", parsed(result)["items"])[0]
    assert item["cite_as"] == "SEC:AAPL:revenue"
    assert item["periods"] == [
        {
            "end": "2026-06-27",
            "fiscal_period": "2026 Q3",
            "form": "10-Q",
            "filed": "2027-08-01",
            "value": 94.0,
        }
    ]
    assert cites.parse("[[SEC:AAPL:revenue]]", ids, {})[0][0]["k"] == [True]
