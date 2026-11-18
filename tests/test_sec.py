# pyright: reportPrivateUsage=false, reportAny=false
from __future__ import annotations

import httpx

from finagent import sec

ALLOWED_COUNTS = {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "500"}
ATOM = """<feed>
<entry><content><accession-number>0001100663-26-000037</accession-number>
<filing-date>2026-06-04</filing-date></content></entry>
<entry><content><accession-number>0001100663-25-000038</accession-number>
<filing-date>2025-12-03</filing-date></content></entry>
</feed>"""


def edgar(request: httpx.Request) -> httpx.Response:
    count = request.url.params.get("count")
    if count not in ALLOWED_COUNTS:
        return httpx.Response(400, text=f"Invalid count, Received: {count}")
    return httpx.Response(200, text=ATOM)


def test_latest_filing_uses_a_count_edgar_accepts_and_takes_the_newest() -> None:
    client = sec.Client("Test test@example.com")
    # SEC 응답 대체용 내부 HTTP 클라이언트 교체
    client._client = httpx.Client(transport=httpx.MockTransport(edgar))  # noqa: SLF001

    found = client.latest_filing("S000004310", "N-CSR")

    assert found == sec.Filing("0001100663-26-000037", "2026-06-04")
