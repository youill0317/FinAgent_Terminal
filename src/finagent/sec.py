"""SEC EDGAR 공시 아카이브·기업 팩트(Company Facts)·펀드 시리즈 클라이언트."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast, final

import httpx

from finagent import limits, memory

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date

TICKERS_URL: Final = "https://www.sec.gov/files/company_tickers.json"
FUNDS_URL: Final = "https://www.sec.gov/files/company_tickers_mf.json"
FILINGS_URL: Final = "https://www.sec.gov/cgi-bin/browse-edgar"
COMPANYFACTS_URL: Final = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL: Final = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL: Final = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{name}"
TIMEOUT_SECONDS: Final = 30.0
NOT_FOUND: Final = 404
INDEX_FILE: Final = "index.json"
FACTS_KEPT: Final = 4
ACCESSION: Final = re.compile(r"\d{10}-\d{2}-\d{6}")
DOCUMENT_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
HIDDEN: Final = re.compile(
    r"<(script|style|ix:header)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
BREAK: Final = re.compile(r"<(?:br|/p|/div|/tr|/table|/h[1-6])\b[^>]*>", re.IGNORECASE)
TAG: Final = re.compile(r"<[^>]+>")
ATOM_ACCESSION: Final = re.compile(r"<accession-number>([0-9-]+)</accession-number>")
ATOM_FILED: Final = re.compile(r"<filing-date>([0-9-]+)</filing-date>")
SPACES: Final = re.compile(r"[ \t\xa0]+")
LINES: Final = re.compile(r"\s*\n\s*")


class FilingError(ValueError): ...


@dataclass(frozen=True, slots=True)
class FundSeries:
    cik: str
    series: str
    class_id: str


@dataclass(frozen=True, slots=True)
class Filing:
    accession: str
    filed: str


def submissions_url(name: str) -> str:
    return SUBMISSIONS_URL.format(name=name)


def archive_url(cik: str, accession: str, name: str = "") -> str:
    folder = accession.replace("-", "")
    return ARCHIVE_URL.format(cik=int(cik), folder=folder, name=name)


@final
class Client:
    def __init__(self, agent: str) -> None:
        if not agent.strip():
            raise ValueError("SEC_USER_AGENT 미설정")
        self._client = httpx.Client(
            headers={"User-Agent": agent.strip(), "Accept-Encoding": "gzip, deflate"},
            timeout=TIMEOUT_SECONDS,
            follow_redirects=True,
        )

    def get(self, url: str, max_wait: float = limits.BACKGROUND_WAIT) -> httpx.Response:
        return limits.send("SEC", lambda: self._client.get(url), max_wait)

    def text(self, url: str, max_wait: float = limits.BACKGROUND_WAIT) -> str:
        response = self.get(url, max_wait)
        _ = response.raise_for_status()
        return response.text

    def cik_table(self, max_wait: float = limits.BACKGROUND_WAIT) -> httpx.Response:
        return self.get(TICKERS_URL, max_wait)

    def facts(
        self, cik: str, max_wait: float = limits.BACKGROUND_WAIT
    ) -> httpx.Response:
        return self.get(COMPANYFACTS_URL.format(cik=cik), max_wait)

    def submissions(
        self, cik: str, max_wait: float = limits.BACKGROUND_WAIT
    ) -> httpx.Response:
        return self.get(submissions_url(f"CIK{cik}.json"), max_wait)

    def fund_series(
        self, max_wait: float = limits.BACKGROUND_WAIT
    ) -> dict[str, FundSeries]:
        response = self.get(FUNDS_URL, max_wait)
        _ = response.raise_for_status()
        payload = cast("Mapping[str, list[object]]", response.json())
        fields = [str(name) for name in payload["fields"]]
        found: dict[str, FundSeries] = {}
        for row in cast("list[list[object]]", payload["data"]):
            item = dict(zip(fields, row, strict=True))
            symbol = str(item["symbol"]).upper().replace(".", "-")
            if symbol not in found:
                found[symbol] = FundSeries(
                    f"{int(str(item['cik'])):010d}",
                    str(item["seriesId"]),
                    str(item["classId"]),
                )
        return found

    def latest_filing(
        self, series: str, form: str, max_wait: float = limits.BACKGROUND_WAIT
    ) -> Filing | None:
        params = {
            "action": "getcompany",
            "CIK": series,
            "type": form,
            "dateb": "",
            "owner": "include",
            "count": "1",
            "output": "atom",
        }
        response = limits.send(
            "SEC", lambda: self._client.get(FILINGS_URL, params=params), max_wait
        )
        _ = response.raise_for_status()
        accession = ATOM_ACCESSION.search(response.text)
        filed = ATOM_FILED.search(response.text)
        if accession is None or filed is None:
            return None
        return Filing(accession.group(1), filed.group(1))

    def archive_names(
        self, cik: str, accession: str, max_wait: float = limits.BACKGROUND_WAIT
    ) -> list[str]:
        body = self.text(archive_url(cik, accession, INDEX_FILE), max_wait)
        raw = cast("dict[str, dict[str, list[dict[str, object]]]]", json.loads(body))
        return [str(item["name"]) for item in raw["directory"]["item"]]

    def archive(
        self,
        cik: str,
        accession: str,
        name: str,
        max_wait: float = limits.BACKGROUND_WAIT,
    ) -> str:
        return self.text(archive_url(cik, accession, name), max_wait)

    def close(self) -> None:
        self._client.close()


def json_or_none(response: httpx.Response) -> Mapping[str, object] | None:
    if response.status_code == NOT_FOUND:
        return None
    _ = response.raise_for_status()
    return cast("Mapping[str, object]", response.json())


def plain_text(body: str) -> str:
    kept = HIDDEN.sub(" ", body)
    kept = TAG.sub("", BREAK.sub("\n", kept))
    return LINES.sub("\n", SPACES.sub(" ", html.unescape(kept))).strip()


@final
class Archive:
    def __init__(
        self,
        client: Client,
        today: date,
        max_wait: float = limits.BACKGROUND_WAIT,
    ) -> None:
        self._client = client
        self._today = today
        self._max_wait = max_wait

    def cik(self, ticker: str) -> str | None:
        return self._ciks().get(ticker.strip().upper().replace(".", "-"))

    def _ciks(self) -> dict[str, str]:
        saved = memory.current().ciks
        if saved is not None and saved[0] == self._today:
            return saved[1]
        response = self._client.cik_table(self._max_wait)
        _ = response.raise_for_status()
        rows = cast("dict[str, dict[str, object]]", response.json()).values()
        found: dict[str, str] = {}
        for row in rows:
            ticker = str(row["ticker"]).upper().replace(".", "-")
            if ticker not in found:
                found[ticker] = f"{int(str(row['cik_str'])):010d}"
        memory.current().ciks = (self._today, found)
        return found

    def facts(self, cik: str) -> Mapping[str, object] | None:
        table = memory.current().company_facts
        saved = table.get(cik)
        if saved is not None and saved[0] == self._today:
            return saved[1]
        found = json_or_none(self._client.facts(cik, self._max_wait))
        table.drop(cik)
        table.put(cik, (self._today, found))
        for old in list(table.rows())[:-FACTS_KEPT]:
            table.drop(old)
        return found

    def documents(self, cik: str, accession: str) -> list[str]:
        if ACCESSION.fullmatch(accession) is None:
            raise FilingError(f"접수번호 형식 오류: {accession}")
        table = memory.current().documents
        saved = table.get(accession)
        if saved is not None:
            return saved
        names = self._client.archive_names(cik, accession, self._max_wait)
        table.put(accession, names)
        return names

    def main(self, accession: str) -> str | None:
        return memory.current().mains.get(accession)

    def keep_main(self, accession: str, name: str) -> None:
        memory.current().mains.put(accession, name)

    def body(self, cik: str, accession: str, name: str) -> str:
        if DOCUMENT_NAME.fullmatch(name) is None or name not in self.documents(
            cik, accession
        ):
            raise FilingError(f"공시 인덱스 내 미존재 문서: {name}")
        table = memory.current().bodies
        saved = table.get((accession, name))
        if saved is not None:
            return saved
        text = plain_text(
            self._client.text(archive_url(cik, accession, name), self._max_wait)
        )
        table.put((accession, name), text)
        return text
