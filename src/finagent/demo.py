"""Synthetic responses in each source's real wire format.

Used by ``finagent --demo`` (to see the dashboard without API keys) and by
the tests. Values are made up and plausible; they are not real observations.
"""

from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Final
from urllib.parse import unquote

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

    from finagent.models import Freq

DEMO_KEYS: Final = {"FRED_API_KEY": "demo", "ECOS_API_KEY": "demo"}

type Anchors = tuple[tuple[float, float], ...]  # (years ago, value), oldest first


@dataclass(frozen=True, slots=True)
class Walk:
    freq: Freq
    anchors: Anchors
    vol: float = 0.0
    step: bool = False
    floor: float | None = None


def _w(freq: Freq, *anchors: tuple[float, float], vol: float = 0.0) -> Walk:
    return Walk(freq, anchors, vol)


def _step(freq: Freq, *anchors: tuple[float, float]) -> Walk:
    return Walk(freq, anchors, step=True)


FRED: Final[dict[str, Walk]] = {
    "A191RL1Q225SBEA": _w(
        "Q", (4, 2.2), (3, 3.0), (2, 1.6), (1, 2.4), (0, 2.1), vol=0.6
    ),
    "INDPRO": _w("M", (4, 102.0), (0, 104.0), vol=0.3),
    "RSAFS": _w("M", (4, 680000), (0, 745000), vol=2500),
    "PCEC96": _w("M", (4, 15500), (0, 16600), vol=25),
    "HOUST": _w("M", (4, 1450), (2, 1350), (0, 1320), vol=40),
    "CFNAIMA3": _w("M", (4, 0.1), (2, -0.1), (0, -0.1), vol=0.08),
    "PAYEMS": _w("M", (4, 152000), (0, 160000), vol=40),
    "UNRATE": _w("M", (4, 3.6), (2, 4.0), (0, 4.3), vol=0.05),
    "SAHMREALTIME": _w("M", (4, 0.1), (2, 0.4), (1, 0.2), (0, 0.27), vol=0.02),
    "ICSA": _w("W", (4, 210000), (0, 226000), vol=6000),
    "CCSA": _w("W", (4, 1700000), (0, 1930000), vol=15000),
    "JTSJOL": _w("M", (4, 9500), (2, 7600), (0, 7100), vol=100),
    "CES0500000003": _w("M", (4, 33.0), (0, 37.6), vol=0.03),
    "CPIAUCSL": _w("M", (4, 290), (0, 327), vol=0.15),
    "CPILFESL": _w("M", (4, 295), (0, 333), vol=0.1),
    "PCEPILFE": _w("M", (4, 118), (0, 129), vol=0.05),
    "PPIFIS": _w("M", (4, 140), (0, 152), vol=0.2),
    "T5YIE": _w("D", (4, 2.5), (0, 2.3), vol=0.03),
    "T5YIFR": _w("D", (4, 2.3), (0, 2.25), vol=0.03),
    "DFEDTARU": _step(
        "D", (4, 5.5), (2.0, 5.0), (1.8, 4.5), (0.9, 4.25), (0.35, 4.0), (0.08, 3.75)
    ),
    "T10Y3M": _w("D", (4, -1.2), (2, -1.3), (1, 0.1), (0, 0.27), vol=0.04),
    "T10Y2Y": _w(
        "D",
        (10.8, 1.25),
        (9.3, 0.95),
        (8.3, 0.35),
        (7.2, -0.02),
        (6.6, 0.45),
        (5.6, 1.55),
        (4.8, 0.85),
        (4.3, -0.25),
        (3.3, -0.95),
        (2.6, -0.38),
        (2.1, 0.08),
        (1.3, 0.48),
        (0, 0.57),
        vol=0.04,
    ),
    "DFII10": _w("D", (4, 1.0), (2, 2.2), (1, 2.0), (0, 1.8), vol=0.03),
    "MORTGAGE30US": _w("W", (4, 6.5), (2, 7.0), (1, 6.8), (0, 6.18), vol=0.05),
    "NFCI": _w("W", (4, -0.3), (2, -0.5), (0, -0.52), vol=0.02),
    "BAMLH0A0HYM2": _w("D", (3, 4.5), (1.5, 3.2), (0.5, 3.6), (0, 3.05), vol=0.05),
    "BAA10Y": _w("D", (4, 2.0), (0, 1.62), vol=0.03),
    "VIXCLS": _w("D", (4, 20), (1.5, 15), (0.5, 24), (0, 16.8), vol=1.5),
    "DTWEXBGS": _w("D", (4, 121), (1.7, 128), (0.8, 121), (0, 120.4), vol=0.4),
    "WALCL": _w("W", (4, 8300000), (0, 6550000), vol=5000),
    "WRESBAL": _w("W", (4, 3300), (0, 3180), vol=40),
    "RRPONTSYD": Walk("D", ((4, 2200), (2, 500), (1, 100), (0, 28)), 10, floor=0),
    "WTREGEN": _w("W", (4, 700), (0, 820), vol=40),
    "M2SL": _w("M", (4, 20800), (0, 22600), vol=20),
    "MTSDS133FMS": _w("M", (4, -140000), (0, -150000), vol=90000),
    "DCOILWTICO": _w("D", (4, 80), (1, 70), (0, 64.2), vol=1.2),
    "DCOILBRENTEU": _w("D", (4, 85), (1, 74), (0, 67.9), vol=1.2),
    "DHHNGSP": _w("D", (4, 2.5), (1, 3.2), (0, 3.0), vol=0.15),
    "PCOPPUSDM": _w("M", (4, 8500), (0, 9840), vol=150),
}

# Treasury curve: (tenor series, 1 year ago, 1 month ago, latest).
_CURVE: Final = (
    ("DGS1MO", 4.20, 4.18, 3.92),
    ("DGS3MO", 4.02, 4.06, 3.85),
    ("DGS6MO", 3.85, 3.88, 3.72),
    ("DGS1", 3.62, 3.70, 3.58),
    ("DGS2", 3.50, 3.61, 3.55),
    ("DGS3", 3.52, 3.63, 3.58),
    ("DGS5", 3.62, 3.76, 3.70),
    ("DGS7", 3.82, 3.94, 3.88),
    ("DGS10", 4.03, 4.18, 4.12),
    ("DGS20", 4.60, 4.66, 4.62),
    ("DGS30", 4.62, 4.71, 4.68),
)
for _series_id, _year, _month, _now in _CURVE:
    FRED[_series_id] = _w(
        "D", (4, _year + 0.4), (1, _year), (1 / 12, _month), (0, _now), vol=0.02
    )

FISCAL: Final[dict[str, Walk]] = {
    "debt_to_penny": _w("D", (4, 31.5e12), (0, 38.12e12), vol=2e10),
    "avg_interest_rates": _w("M", (4, 2.2), (0, 3.39), vol=0.01),
}

NYFED: Final[dict[str, Walk]] = {
    "effr": _step(
        "D",
        (4, 5.33),
        (2.0, 4.83),
        (1.8, 4.33),
        (0.9, 4.08),
        (0.35, 3.88),
        (0.08, 3.63),
    ),
    "sofr": _w(
        "D",
        (4, 5.31),
        (2.0, 4.85),
        (1.8, 4.32),
        (0.9, 4.1),
        (0.35, 3.9),
        (0, 3.61),
        vol=0.02,
    ),
}

ECOS: Final[dict[str, Walk]] = {
    "722Y001/0101000": _step(
        "D", (4, 3.5), (1.6, 3.25), (1.3, 3.0), (0.9, 2.75), (0.6, 2.5)
    ),
    "817Y002/010200000": _w("D", (4, 3.3), (2, 3.4), (1, 2.6), (0, 2.55), vol=0.02),
    "817Y002/010210000": _w("D", (4, 3.4), (2, 3.5), (1, 2.8), (0, 2.86), vol=0.02),
    "731Y001/0000001": _w(
        "D", (4, 1300), (2, 1340), (1.5, 1470), (1, 1400), (0, 1385), vol=8
    ),
    "901Y009/0": _w("M", (4, 108), (0, 118.5), vol=0.1),
    "901Y118/T002": _w("M", (4, 55000), (0, 62000), vol=2500),
    "901Y027/I61BC": _w("M", (4, 2.8), (0, 2.7), vol=0.1),
    "901Y067/I16D": _w("M", (4, 100.2), (1, 98.6), (0, 98.9), vol=0.1),
    "511Y002/FME": _w("M", (4, 100), (1.8, 93), (0, 104), vol=1.2),
}

SDMX: Final[dict[str, Walk]] = {
    "FM/B.U2.EUR.4F.KR.DFR.LEV": _step(
        "D",
        (4, 4.0),
        (1.9, 3.75),
        (1.7, 3.5),
        (1.4, 3.25),
        (1.1, 3.0),
        (0.9, 2.75),
        (0.7, 2.5),
        (0.5, 2.25),
        (0.35, 2.0),
    ),
    "ICP/M.U2.N.000000.4.ANR": _w("M", (4, 5.0), (2, 2.4), (1, 2.2), (0, 2.1), vol=0.1),
    "EXR/D.USD.EUR.SP00.A": _w(
        "D", (4, 1.0), (1.5, 1.08), (0.8, 1.04), (0.3, 1.17), (0, 1.164), vol=0.004
    ),
    "EXR/D.JPY.EUR.SP00.A": _w(
        "D", (4, 140), (1.5, 162), (0.8, 160), (0, 173.7), vol=0.6
    ),
    "WS_CBPOL/M.JP/all": _step(
        "M", (4, -0.1), (2.5, 0.1), (2.2, 0.25), (1.7, 0.5), (0.8, 0.75)
    ),
    "WS_CBPOL/M.GB/all": _step(
        "M",
        (4, 5.25),
        (2.1, 5.0),
        (1.9, 4.75),
        (1.6, 4.5),
        (1.3, 4.25),
        (1.0, 4.0),
        (0.5, 3.75),
        (0.2, 3.5),
    ),
}
OECD_CLI: Final[dict[str, Walk]] = {
    "USA": _w("M", (4, 99.0), (0, 100.3), vol=0.05),
    "CHN": _w("M", (4, 99.5), (0, 100.1), vol=0.05),
    "KOR": _w("M", (4, 98.8), (0, 100.6), vol=0.05),
}

DEMO_RELEASES: Final = (
    (1, "Producer Price Index"),
    (2, "Consumer Price Index"),
    (2, "Unemployment Insurance Weekly Claims Report"),
    (3, "Advance Monthly Sales for Retail and Food Services"),
    (3, "Industrial Production and Capacity Utilization - G.17"),
    (3, "H.4.1 Factors Affecting Reserve Balances"),
    (4, "Kansas City Fed Manufacturing Survey"),
)


# ---------------------------------------------------------------- generation


def _dates(freq: Freq, start: date, today: date) -> list[date]:
    if freq == "D":
        end = today - timedelta(days=1)
        days = (end - start).days
        return [
            day
            for offset in range(days + 1)
            if (day := start + timedelta(days=offset)).weekday() < 5  # noqa: PLR2004
        ]
    if freq == "W":
        last = today - timedelta(days=4)
        out: list[date] = []
        while last >= start:
            out.append(last)
            last -= timedelta(days=7)
        return out[::-1]
    lag_days, step = (120, 3) if freq == "Q" else (40, 1)
    last_day = today - timedelta(days=lag_days)
    month = last_day.month - (last_day.month - 1) % step
    year = last_day.year
    out = []
    while date(year, month, 1) >= start.replace(day=1):
        out.append(date(year, month, 1))
        month -= step
        if month < 1:
            month += 12
            year -= 1
    return out[::-1]


def _level(walk: Walk, years_ago: float) -> float:
    anchors = walk.anchors
    if years_ago >= anchors[0][0]:
        return anchors[0][1]
    for (old_t, old_v), (new_t, new_v) in pairwise(anchors):
        if new_t <= years_ago <= old_t:
            if walk.step:
                return old_v
            span = old_t - new_t
            return new_v + (old_v - new_v) * (
                (years_ago - new_t) / span if span else 0.0
            )
    return anchors[-1][1]


def generate(
    key: str, walk: Walk, start: date, today: date
) -> list[tuple[date, float]]:
    rng = random.Random(zlib.crc32(key.encode()))  # noqa: S311 - demo data only
    points: list[tuple[date, float]] = []
    days = _dates(walk.freq, start, today)
    for index, day in enumerate(days):
        value = _level(walk, (today - day).days / 365.25)
        if index < len(days) - 1:
            value += rng.gauss(0.0, walk.vol)
        if walk.floor is not None:
            value = max(walk.floor, value)
        points.append((day, round(value, 4)))
    return points


# ---------------------------------------------------------------- wire formats


def demo_transport(today: date) -> httpx.MockTransport:
    """A transport that answers every source the dashboard calls."""
    routes: dict[str, Callable[[httpx.Request, date], httpx.Response]] = {
        "api.stlouisfed.org": _fred,
        "api.fiscaldata.treasury.gov": _fiscal,
        "markets.newyorkfed.org": _nyfed,
        "ecos.bok.or.kr": _ecos,
        "data-api.ecb.europa.eu": _sdmx,
        "stats.bis.org": _sdmx,
        "sdmx.oecd.org": _oecd,
    }

    def handle(request: httpx.Request) -> httpx.Response:
        route = routes.get(request.url.host)
        if route is None:
            return httpx.Response(404)
        return route(request, today)

    return httpx.MockTransport(handle)


def _param(request: httpx.Request, name: str, default: str = "") -> str:
    values = request.url.params.get_list(name)
    return values[0] if values else default


def _start(request: httpx.Request, param: str, default_years: int, today: date) -> date:
    raw = _param(request, param)
    if raw:
        parts = [int(part) for part in raw.split("-")]
        return date(parts[0], parts[1], parts[2] if len(parts) > 2 else 1)  # noqa: PLR2004
    return today - timedelta(days=365 * default_years)


def _json(payload: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode())


def _fred(request: httpx.Request, today: date) -> httpx.Response:
    if request.url.path.endswith("/releases/dates"):
        return _json(
            {
                "release_dates": [
                    {
                        "release_id": index,
                        "release_name": name,
                        "date": (today + timedelta(days=offset)).isoformat(),
                    }
                    for index, (offset, name) in enumerate(DEMO_RELEASES)
                ]
            }
        )
    series_id = _param(request, "series_id")
    start = _start(request, "observation_start", 4, today)
    if series_id == "USREC":
        points = [
            (day, 1.0 if date(2020, 2, 1) <= day <= date(2020, 4, 1) else 0.0)
            for day in _dates("M", start, today)
        ]
    elif series_id in FRED:
        points = generate(series_id, FRED[series_id], start, today)
    else:
        return httpx.Response(400, content=b'{"error_message":"Bad Request"}')
    return _json(
        {
            "observations": [
                {"date": day.isoformat(), "value": f"{value}"} for day, value in points
            ]
        }
    )


def _fiscal(request: httpx.Request, today: date) -> httpx.Response:
    name = request.url.path.rstrip("/").rsplit("/", 1)[-1]
    walk = FISCAL.get(name)
    if walk is None:
        return httpx.Response(404)
    field = _param(request, "fields", "record_date,value").split(",")[1]
    start = today - timedelta(days=365 * 4)
    rows = [
        {"record_date": day.isoformat(), field: f"{value:.2f}"}
        for day, value in reversed(generate(name, walk, start, today))
    ]
    return _json({"data": rows, "meta": {"count": len(rows)}})


def _nyfed(request: httpx.Request, today: date) -> httpx.Response:
    name = "effr" if "/effr/" in request.url.path else "sofr"
    start = today - timedelta(days=365 * 4)
    points = generate(name, NYFED[name], start, today)
    return _json(
        {
            "refRates": [
                {
                    "effectiveDate": day.isoformat(),
                    "type": name.upper(),
                    "percentRate": value,
                }
                for day, value in reversed(points)
            ]
        }
    )


def _ecos(request: httpx.Request, today: date) -> httpx.Response:
    parts = request.url.path.strip("/").split("/")
    # api/StatisticSearch/{key}/json/kr/{from}/{to}/{stat}/{cycle}/{start}/{end}/{item1}
    stat, cycle, start_text, item1 = parts[7], parts[8], parts[9], parts[11]
    walk = ECOS.get(f"{stat}/{item1}")
    if walk is None:
        return _json(
            {"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당하는 데이터가 없습니다."}}
        )
    start = date(int(start_text[:4]), int(start_text[4:6] or 1), 1)
    rows = [
        {
            "STAT_CODE": stat,
            "ITEM_CODE1": item1,
            "TIME": day.strftime("%Y%m%d" if cycle == "D" else "%Y%m"),
            "DATA_VALUE": f"{value}",
        }
        for day, value in generate(f"{stat}/{item1}", walk, start, today)
    ]
    return _json({"StatisticSearch": {"list_total_count": len(rows), "row": rows}})


def _csv_period(day: date, freq: Freq) -> str:
    return day.isoformat() if freq in {"D", "W"} else day.strftime("%Y-%m")


def _sdmx(request: httpx.Request, today: date) -> httpx.Response:
    path = unquote(request.url.path)
    marker = "/service/data/" if "/service/data/" in path else "/api/v1/data/"
    flow_key = path.split(marker, 1)[-1]
    walk = SDMX.get(flow_key)
    if walk is None:
        return httpx.Response(404)
    start = _start(request, "startPeriod", 4, today)
    lines = ["KEY,FREQ,TIME_PERIOD,OBS_VALUE"]
    lines += [
        f"{flow_key},{walk.freq},{_csv_period(day, walk.freq)},{value}"
        for day, value in generate(flow_key, walk, start, today)
    ]
    return httpx.Response(200, content="\n".join(lines).encode())


def _oecd(request: httpx.Request, today: date) -> httpx.Response:
    start = _start(request, "startPeriod", 4, today)
    lines = ["DATAFLOW,REF_AREA,FREQ,MEASURE,TIME_PERIOD,OBS_VALUE"]
    for area, walk in OECD_CLI.items():
        lines += [
            f"OECD.SDD.STES:DSD_STES@DF_CLI,{area},M,LI,{_csv_period(day, 'M')},{value}"
            for day, value in generate(f"CLI/{area}", walk, start, today)
        ]
    return httpx.Response(200, content="\n".join(lines).encode())
