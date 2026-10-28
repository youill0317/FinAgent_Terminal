"""Synthetic responses in each source's real wire format.

Used by the tests only: values are made up and plausible, not real
observations, so the app (``--demo`` included) never imports them.
Yahoo is read through yfinance, not HTTP, so ``demo_closes`` stands in for
``sources.YahooCloses`` instead.
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

from finagent.fetch import SourceError
from finagent.sources import ecos_period, parse_period

if TYPE_CHECKING:
    from collections.abc import Callable

    from finagent.fetch import Closes
    from finagent.models import Freq
    from finagent.sources import Series

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
    "A191RO1Q156NBEA": _w("Q", (4, 1.9), (2, 2.9), (1, 2.0), (0, 2.1), vol=0.2),
    "DFEDTARL": _step(
        "D", (4, 5.25), (2.0, 4.75), (1.8, 4.25), (0.9, 4.0), (0.35, 3.75), (0.08, 3.5)
    ),
    "GDPNOW": _w("Q", (4, 2.0), (2, 2.8), (1, 1.5), (0, 2.4), vol=0.5),
    "TCU": _w("M", (4, 79.8), (2, 78.5), (0, 77.6), vol=0.2),
    "NEWORDER": _w("M", (4, 72000), (0, 74500), vol=500),
    "DGORDER": _w("M", (4, 275000), (0, 300000), vol=6000),
    "ISRATIO": _w("M", (4, 1.33), (2, 1.38), (0, 1.37), vol=0.005),
    "TOTALSA": _w("M", (4, 14.8), (2, 15.6), (0, 16.1), vol=0.3),
    "GACDFSA066MSFRBPHI": _w("M", (4, -8), (2, -5), (1, 10), (0, 2), vol=6),
    "GACDINA066MNFRBNY": _w("M", (4, -5), (2, -10), (1, 3), (0, -2), vol=6),
    "W875RX1": _w("M", (4, 15600), (0, 16400), vol=15),
    "PSAVERT": _w("M", (4, 4.0), (2, 4.8), (0, 4.5), vol=0.2),
    "UMCSENT": _w("M", (4, 62), (2, 72), (1, 55), (0, 58), vol=2),
    "TOTALSL": _w("M", (4, 4800000), (0, 5150000), vol=3000),
    "DRCCLACBS": _w("Q", (4, 2.9), (2, 3.2), (0, 3.05), vol=0.03),
    "U6RATE": _w("M", (4, 6.8), (2, 7.5), (0, 8.0), vol=0.08),
    "CIVPART": _w("M", (4, 62.7), (2, 62.6), (0, 62.3), vol=0.05),
    "LNS11300060": _w("M", (4, 83.2), (2, 83.6), (0, 83.4), vol=0.1),
    "LNS12300060": _w("M", (4, 80.7), (2, 80.8), (0, 80.4), vol=0.1),
    "JTSQUR": _w("M", (4, 2.3), (2, 2.1), (0, 2.0), vol=0.03),
    "ECIALLCIV": _w("Q", (4, 158), (0, 178), vol=0.1),
    "ULCNFB": _w("Q", (4, 118), (0, 126), vol=0.2),
    "PCEPI": _w("M", (4, 121), (0, 131), vol=0.05),
    "MEDCPIM158SFRBCLE": _w("M", (4, 5.5), (2, 3.6), (0, 3.2), vol=0.4),
    "STICKCPIM159SFRBATL": _w("M", (4, 4.8), (2, 3.5), (0, 3.1), vol=0.05),
    "PCETRIM12M159SFRBDAL": _w("M", (4, 4.1), (2, 2.8), (0, 2.6), vol=0.03),
    "CUSR0000SAH1": _w("M", (4, 360), (0, 420), vol=0.2),
    "CUSR0000SASLE": _w("M", (4, 400), (0, 455), vol=0.2),
    "CUSR0000SACL1E": _w("M", (4, 161), (2, 160), (0, 163), vol=0.15),
    "T10YIE": _w("D", (4, 2.35), (0, 2.3), vol=0.02),
    "MICH": _w("M", (4, 4.0), (2, 3.0), (1, 4.6), (0, 4.0), vol=0.2),
    "IORB": _step(
        "D", (4, 5.4), (2.0, 4.9), (1.8, 4.4), (0.9, 4.15), (0.35, 3.9), (0.08, 3.65)
    ),
    "STLFSI4": _w("W", (4, 0.2), (2, -0.6), (0, -0.7), vol=0.1),
    "BAMLC0A0CM": _w("D", (3, 1.3), (1.5, 0.9), (0.5, 1.0), (0, 0.82), vol=0.02),
    "HQMCB10YR": _w("M", (4, 5.6), (2, 5.3), (0, 5.4), vol=0.08),
    "DRTSCILM": _w("Q", (4, 40), (2, 8), (1, -2), (0, 5), vol=3),
    "TOTBKCR": _w("W", (4, 17200), (0, 18300), vol=20),
    "BUSLOANS": _w("M", (4, 2780), (0, 2860), vol=5),
    "SP500": _w("D", (4, 4300), (2, 5200), (1, 5900), (0, 6650), vol=40),
    "NASDAQCOM": _w("D", (4, 13200), (2, 16500), (1, 19300), (0, 22400), vol=150),
    "NASDAQ100": _w("D", (4, 11500), (2, 17500), (1, 20500), (0, 24600), vol=170),
    "DJIA": _w("D", (4, 33000), (2, 38500), (1, 42500), (0, 46500), vol=250),
    "NIKKEI225": _w("D", (4, 27500), (2, 38500), (1, 39000), (0, 48500), vol=350),
    "TREAST": _w("W", (4, 5300000), (0, 4200000), vol=3000),
    "WSHOMCB": _w("W", (4, 2450000), (0, 2050000), vol=2000),
    "GFDEGDQ188S": _w("Q", (4, 118), (0, 121), vol=0.5),
    "PERMIT": _w("M", (4, 1500), (2, 1450), (0, 1330), vol=30),
    "HSN1F": _w("M", (4, 680), (2, 690), (0, 670), vol=25),
    "MSACSR": _w("M", (4, 7.8), (2, 8.6), (0, 9.2), vol=0.3),
    "USSTHPI": _w("Q", (4, 640), (0, 720), vol=1),
    "CSUSHPISA": _w("M", (4, 305), (0, 330), vol=0.3),
    "DRSFRMACBS": _w("Q", (4, 1.7), (2, 1.8), (0, 1.85), vol=0.02),
    "DEXCHUS": _w("D", (4, 7.2), (1.5, 7.28), (0, 7.12), vol=0.01),
    "DTWEXEMEGS": _w("D", (4, 128), (1.5, 131), (0, 127), vol=0.3),
    "PALUMUSDM": _w("M", (4, 2250), (0, 2600), vol=50),
    "PIORECRUSDM": _w("M", (4, 120), (1, 100), (0, 104), vol=4),
    "PWHEAMTUSDM": _w("M", (4, 330), (1, 240), (0, 230), vol=8),
    "PALLFNFINDEXM": _w("M", (4, 175), (1, 165), (0, 168), vol=2),
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
    "debt_to_penny/tot_pub_debt_out_amt": _w(
        "D", (4, 31.5e12), (0, 38.12e12), vol=2e10
    ),
    "debt_to_penny/debt_held_public_amt": _w("D", (4, 24.8e12), (0, 30.9e12), vol=2e10),
    "avg_interest_rates/avg_interest_rate_amt": _w("M", (4, 2.2), (0, 3.39), vol=0.01),
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
    "obfr": _w("D", (4, 5.31), (2.0, 4.82), (0.9, 4.07), (0, 3.62), vol=0.01),
    "tgcr": _w("D", (4, 5.30), (2.0, 4.83), (0.9, 4.08), (0, 3.58), vol=0.02),
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
    "817Y002/010101000": _w("D", (4, 3.5), (1, 2.8), (0, 2.52), vol=0.02),
    "817Y002/010502000": _w("D", (4, 3.7), (1, 2.9), (0, 2.6), vol=0.02),
    "817Y002/010300000": _w("D", (4, 4.3), (2, 3.9), (1, 3.1), (0, 2.95), vol=0.02),
    "731Y001/0000002": _w("D", (4, 960), (1.5, 870), (0, 940), vol=6),
    "732Y001/99": _w("M", (4, 420000000), (1, 410000000), (0, 422000000), vol=1e6),
    "161Y006/BBHA00": _w("M", (4, 3800000), (0, 4300000), vol=4000),
    "151Y001/1000000": _w("Q", (4, 1860000), (0, 1960000), vol=3000),
    "200Y102/10111": _w("Q", (4, 0.6), (2, 0.3), (1, -0.1), (0, 0.6), vol=0.3),
    "901Y010/QB": _w("M", (4, 105), (0, 113), vol=0.1),
    "404Y014/*AA": _w("M", (4, 118), (0, 121), vol=0.2),
    "901Y118/T004": _w("M", (4, 57000), (0, 53000), vol=2500),
    "301Y017/SA000": _w("M", (4, 2500), (1, 8000), (0, 9500), vol=1500),
    "901Y027/I61E": _w("M", (4, 62.5), (0, 63.0), vol=0.1),
    "901Y067/I16E": _w("M", (4, 99.5), (1, 100.2), (0, 100.8), vol=0.1),
    "513Y001/E2000": _w("M", (4, 95), (1.5, 88), (0, 95), vol=1),
    "802Y001/0001000": _w("D", (4, 2400), (1.5, 2650), (1, 2500), (0, 3500), vol=25),
    "802Y001/0089000": _w("D", (4, 800), (1.5, 850), (1, 690), (0, 860), vol=8),
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
    "HICP/M.U2.N.000000.4D0.ANR": _w(
        "M", (4, 5.0), (2, 2.4), (1, 2.2), (0, 2.1), vol=0.1
    ),
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
SDMX.update(
    {
        "FM/B.U2.EUR.4F.KR.MRR_FR.LEV": _step(
            "D", (4, 4.5), (1.9, 4.25), (1.1, 3.4), (0.9, 3.15), (0.35, 2.15)
        ),
        "EST/B.EU000A2X2A25.WT": _w(
            "D", (4, 3.9), (1.9, 3.65), (0.9, 2.66), (0.35, 1.92), (0, 1.93), vol=0.01
        ),
        "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y": _w(
            "D", (4, 2.6), (2, 2.4), (1, 2.5), (0, 2.7), vol=0.03
        ),
        "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SRS_10Y_3M": _w(
            "D", (4, -0.9), (2, -0.6), (1, 0.4), (0, 0.8), vol=0.03
        ),
        "HICP/M.U2.N.XEF000.4D0.ANR": _w("M", (4, 5.0), (2, 2.7), (0, 2.3), vol=0.1),
        "HICP/M.U2.N.SERV00.4D0.ANR": _w("M", (4, 4.5), (2, 4.0), (0, 3.1), vol=0.1),
        "BSI/M.U2.Y.V.M30.X.I.U2.2300.Z01.A": _w(
            "M", (4, -0.5), (2, 1.5), (1, 3.5), (0, 2.9), vol=0.2
        ),
        "BSI/M.U2.Y.U.A20T.A.I.U2.2240.Z01.A": _w(
            "M", (4, 2.0), (2, 0.5), (1, 1.5), (0, 2.8), vol=0.1
        ),
        "LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T": _w(
            "M", (4, 6.6), (2, 6.4), (0, 6.3), vol=0.05
        ),
        "STBS/M.I10.W.PROD.NS0020.4D0.GY.IX": _w(
            "M", (4, -3.0), (2, -2.0), (1, 1.0), (0, 1.2), vol=0.8
        ),
        "CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX": _w(
            "D", (4, 0.15), (1.5, 0.06), (0.5, 0.1), (0, 0.05), vol=0.01
        ),
        "EXR/M.E03.EUR.EN00.A": _w("M", (4, 120), (1, 123), (0, 128), vol=0.4),
        "WS_CBPOL/M.CN/all": _step("M", (4, 3.55), (1.3, 3.35), (1.1, 3.1), (0.4, 3.0)),
        "WS_CBPOL/M.CA/all": _step(
            "M", (4, 5.0), (1.4, 4.5), (1.0, 3.25), (0.6, 2.75), (0.1, 2.5)
        ),
        "WS_CBPOL/M.AU/all": _step("M", (4, 4.35), (0.7, 4.1), (0.4, 3.6)),
        "WS_CBPOL/M.CH/all": _step("M", (4, 1.75), (1.5, 1.0), (0.8, 0.25), (0.3, 0.0)),
        "WS_TC/Q.KR.H.A.M.770.A/all": _w("Q", (4, 102), (0, 90), vol=0.3),
        "WS_TC/Q.KR.P.A.M.770.A/all": _w("Q", (4, 222), (0, 205), vol=0.6),
        "WS_TC/Q.CN.P.A.M.770.A/all": _w("Q", (4, 215), (0, 205), vol=0.6),
        "WS_TC/Q.US.H.A.M.770.A/all": _w("Q", (4, 75), (0, 69), vol=0.2),
        "WS_CREDIT_GAP/Q.KR.P.A.C/all": _w("Q", (4, 8), (0, -8), vol=0.5),
        "WS_CREDIT_GAP/Q.CN.P.A.C/all": _w("Q", (4, 5), (0, 2), vol=0.5),
        "WS_SPP/Q.KR.R.628/all": _w("Q", (4, 110), (2, 98), (0, 99), vol=0.4),
        "WS_SPP/Q.US.R.628/all": _w("Q", (4, 160), (0, 163), vol=0.4),
        "WS_SPP/Q.CN.R.628/all": _w("Q", (4, 108), (0, 92), vol=0.4),
        "WS_EER/M.R.B.KR/all": _w("M", (4, 102), (1, 96), (0, 98), vol=0.4),
        "WS_EER/M.R.B.US/all": _w("M", (4, 115), (1, 118), (0, 113), vol=0.4),
    }
)

# (measure, area) for the OECD DF_CLI dataflow.
OECD_DEMO: Final[dict[tuple[str, str], Walk]] = {
    ("LI", "USA"): _w("M", (4, 99.0), (0, 100.3), vol=0.05),
    ("LI", "CHN"): _w("M", (4, 99.5), (0, 100.1), vol=0.05),
    ("LI", "KOR"): _w("M", (4, 98.8), (0, 100.6), vol=0.05),
    ("LI", "JPN"): _w("M", (4, 100.2), (0, 100.4), vol=0.05),
    ("LI", "DEU"): _w("M", (4, 98.5), (0, 100.0), vol=0.05),
    ("LI", "GBR"): _w("M", (4, 98.9), (0, 100.2), vol=0.05),
    ("BCICP", "USA"): _w("M", (4, 99.0), (0, 99.6), vol=0.1),
    ("BCICP", "CHN"): _w("M", (4, 99.4), (0, 99.0), vol=0.1),
    ("BCICP", "KOR"): _w("M", (4, 98.6), (0, 98.9), vol=0.1),
    ("BCICP", "DEU"): _w("M", (4, 97.5), (0, 98.2), vol=0.1),
    ("CCICP", "USA"): _w("M", (4, 97.5), (0, 97.9), vol=0.1),
    ("CCICP", "KOR"): _w("M", (4, 98.0), (0, 100.4), vol=0.1),
    ("CCICP", "DEU"): _w("M", (4, 97.0), (0, 98.0), vol=0.1),
}

# Yahoo tickers.
YAHOO: Final[dict[str, Walk]] = {
    "GC=F": _w("D", (4, 1750), (2, 2050), (1, 2700), (0, 4100), vol=25),
}

# FOMC SEP medians by years after the latest release year: (latest, prior).
# The September SEP adds a year, so the prior vintage is one year shorter.
SEP_DAYS_AGO: Final = (27, 118)
DEMO_SEP: Final[dict[str, tuple[tuple[float, ...], tuple[float, ...]]]] = {
    "FEDTARMD": ((3.6, 3.4, 3.1, 3.1), (3.9, 3.6, 3.4)),
    "GDPC1MD": ((1.6, 1.8, 1.9, 1.8), (1.4, 1.6, 1.8)),
    "UNRATEMD": ((4.5, 4.4, 4.3, 4.2), (4.5, 4.5, 4.4)),
    "PCECTPIMD": ((3.0, 2.6, 2.1, 2.0), (3.0, 2.4, 2.1)),
    "JCXFEMD": ((3.1, 2.6, 2.1, 2.0), (3.1, 2.4, 2.1)),
}
DEMO_SEP_LONGER_RUN: Final[dict[str, tuple[float, float]]] = {
    "FEDTARMDLR": (3.0, 3.0),
    "GDPC1MDLR": (1.8, 1.8),
    "UNRATEMDLR": (4.2, 4.2),
    "PCECTPIMDLR": (2.0, 2.0),
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
            if (day := start + timedelta(days=offset)).weekday() < 5
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


def demo_closes(today: date) -> Closes:
    """Yahoo daily closes, as ``sources.YahooCloses`` returns them."""

    def closes(ticker: str, start: date) -> Series:
        walk = YAHOO.get(ticker)
        if walk is None:
            raise SourceError(f"YAHOO {ticker}: 관측값 없음")
        return tuple(generate(ticker, walk, start, today))

    return closes


def _param(request: httpx.Request, name: str, default: str = "") -> str:
    values = request.url.params.get_list(name)
    return values[0] if values else default


def _start(request: httpx.Request, param: str, default_years: int, today: date) -> date:
    raw = _param(request, param)
    if raw:
        parts = [int(part) for part in raw.split("-")]
        return date(parts[0], parts[1], parts[2] if len(parts) > 2 else 1)
    return today - timedelta(days=365 * default_years)


def _json(payload: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode())


def _fred(request: httpx.Request, today: date) -> httpx.Response:
    if request.url.path.endswith("/series/search"):
        return _fred_search(_param(request, "search_text"))
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
    if _param(request, "realtime_start"):
        return _fred_vintages(series_id, today)
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


def _fred_search(text: str) -> httpx.Response:
    words = text.upper().split()
    hits = [sid for sid in FRED if any(word in sid for word in words)] or list(FRED)[:5]
    return _json(
        {
            "seriess": [
                {
                    "id": sid,
                    "title": f"{sid} (demo)",
                    "frequency_short": FRED[sid].freq,
                    "units_short": "demo",
                    "observation_start": "2000-01-01",
                    "observation_end": "2026-01-01",
                    "notes": "",
                }
                for sid in hits[:15]
            ]
        }
    )


def _fred_vintages(series_id: str, today: date) -> httpx.Response:
    latest, prior = (today - timedelta(days=days) for days in SEP_DAYS_AGO)
    prior_end = (latest - timedelta(days=1)).isoformat()
    rows: list[tuple[str, str, date, float]] = []
    if series_id in DEMO_SEP:
        new, old = DEMO_SEP[series_id]
        base = latest.year
        # A past target year stays in every vintage and must not become a column.
        rows.append((prior.isoformat(), "9999-12-31", date(base - 1, 1, 1), new[0]))
        rows += [
            (latest.isoformat(), "9999-12-31", date(base + offset, 1, 1), value)
            for offset, value in enumerate(new)
        ]
        rows += [
            (prior.isoformat(), prior_end, date(base + offset, 1, 1), value)
            for offset, value in enumerate(old)
        ]
    elif series_id in DEMO_SEP_LONGER_RUN:
        new_lr, old_lr = DEMO_SEP_LONGER_RUN[series_id]
        rows = [
            (prior.isoformat(), "9999-12-31", prior, old_lr),
            (latest.isoformat(), "9999-12-31", latest, new_lr),
        ]
    else:
        return httpx.Response(400, content=b'{"error_message":"Bad Request"}')
    return _json(
        {
            "observations": [
                {
                    "realtime_start": start,
                    "realtime_end": end,
                    "date": day.isoformat(),
                    "value": f"{value}",
                }
                for start, end, day, value in rows
            ]
        }
    )


def _fiscal(request: httpx.Request, today: date) -> httpx.Response:
    name = request.url.path.rstrip("/").rsplit("/", 1)[-1]
    field = _param(request, "fields", "record_date,value").split(",")[1]
    walk = FISCAL.get(f"{name}/{field}")
    if walk is None:
        return httpx.Response(404)
    start = today - timedelta(days=365 * 4)
    rows = [
        {"record_date": day.isoformat(), field: f"{value:.2f}"}
        for day, value in reversed(generate(name, walk, start, today))
    ]
    return _json({"data": rows, "meta": {"count": len(rows)}})


def _nyfed(request: httpx.Request, today: date) -> httpx.Response:
    name = request.url.path.split("/rates/", 1)[-1].split("/")[1]
    if name not in NYFED:
        return httpx.Response(404)
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
    if parts[1] == "StatisticTableList":
        stats = sorted({key.split("/")[0] for key in ECOS})
        rows = [
            {"STAT_CODE": stat, "STAT_NAME": f"{stat} (demo)", "CYCLE": "M"}
            for stat in stats
        ]
        return _json(
            {"StatisticTableList": {"list_total_count": len(rows), "row": rows}}
        )
    if parts[1] == "StatisticItemList":
        stat = parts[7]
        rows = [
            {
                "STAT_CODE": stat,
                "ITEM_CODE": key.split("/")[1],
                "ITEM_NAME": f"{key} (demo)",
                "CYCLE": walk.freq,
            }
            for key, walk in ECOS.items()
            if key.split("/")[0] == stat
        ]
        return _json(
            {"StatisticItemList": {"list_total_count": len(rows), "row": rows}}
        )
    # api/StatisticSearch/{key}/json/kr/{from}/{to}/{stat}/{cycle}/{start}/{end}/{item1}
    stat, cycle, start_text, item1 = parts[7], parts[8], parts[9], parts[11]
    walk = ECOS.get(f"{stat}/{item1}")
    if walk is None:
        return _json(
            {"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당하는 데이터가 없습니다."}}
        )
    start = parse_period(start_text) or today
    rows = [
        {
            "STAT_CODE": stat,
            "ITEM_CODE1": item1,
            "TIME": ecos_period(day, cycle),
            "DATA_VALUE": f"{value}",
        }
        for day, value in generate(f"{stat}/{item1}", walk, start, today)
    ]
    return _json({"StatisticSearch": {"list_total_count": len(rows), "row": rows}})


def _csv_period(day: date, freq: Freq) -> str:
    return day.isoformat() if freq in {"D", "W"} else day.strftime("%Y-%m")


SDMX_NS: Final = "http://www.sdmx.org/resources/sdmxml/schemas/v2_1"
STRUCTURE_NS: Final = " ".join(
    f'xmlns:{prefix}="{SDMX_NS}/{name}"'
    for prefix, name in (("m", "message"), ("s", "structure"), ("c", "common"))
)


def _structure(flows: list[str], keys: list[str]) -> httpx.Response:
    """A small SDMX-ML structure message: dataflows, or one flow's dimensions."""
    parts = [f"<m:Structure {STRUCTURE_NS}><m:Structures><s:Dataflows>"]
    for flow in flows:
        parts.append(f'<s:Dataflow id="{flow}" agencyID="DEMO" version="1.0">')
        parts.append(f'<c:Name xml:lang="en">{flow} (demo)</c:Name></s:Dataflow>')
    parts.append("</s:Dataflows>")
    if keys:
        split = [key.split(".") for key in keys]
        width = max(len(dims) for dims in split)
        parts.append("<s:Codelists>")
        for position in range(width):
            codes = sorted({dims[position] for dims in split if len(dims) > position})
            parts.append(f'<s:Codelist id="CL_D{position + 1}">')
            for code in codes:
                parts.append(f'<s:Code id="{code}"><c:Name xml:lang="en">{code}')
                parts.append("</c:Name></s:Code>")
            parts.append("</s:Codelist>")
        parts.append("</s:Codelists><s:DataStructures><s:DimensionList>")
        for position in range(1, width + 1):
            parts.append(f'<s:Dimension id="D{position}" position="{position}">')
            parts.append("<s:LocalRepresentation><s:Enumeration>")
            parts.append(f'<Ref id="CL_D{position}" class="Codelist"/>')
            parts.append("</s:Enumeration></s:LocalRepresentation></s:Dimension>")
        parts.append("</s:DimensionList></s:DataStructures>")
    parts.append("</m:Structures></m:Structure>")
    return httpx.Response(200, content="".join(parts).encode())


def _sdmx_structure(path: str) -> httpx.Response:
    tail = path.split("/dataflow/", 1)[-1].split("/")
    flows = sorted({key.split("/")[0] for key in SDMX})
    if len(tail) < 2:
        return _structure(flows, [])
    keys = [key.split("/")[1] for key in SDMX if key.split("/")[0] == tail[1]]
    return _structure([tail[1]], keys)


def _sdmx(request: httpx.Request, today: date) -> httpx.Response:
    path = unquote(request.url.path)
    if "/dataflow" in path:
        return _sdmx_structure(path)
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
    if "/dataflow" in request.url.path:
        tail = unquote(request.url.path).split("/dataflow/", 1)[-1].split("/")
        if len(tail) < 2:
            return _structure(["DSD_STES@DF_CLI"], [])
        return _structure([tail[1]], ["USA.M.LI", "KOR.M.BCICP"])
    start = _start(request, "startPeriod", 4, today)
    key = unquote(request.url.path).rsplit("/", 1)[-1].split(".")
    areas, measures = key[0].split("+"), key[2].split("+")
    flow = "OECD.SDD.STES:DSD_STES@DF_CLI"
    lines = ["DATAFLOW,REF_AREA,FREQ,MEASURE,TIME_PERIOD,OBS_VALUE"]
    for (measure, area), walk in OECD_DEMO.items():
        if area not in areas or measure not in measures:
            continue
        lines += [
            f"{flow},{area},M,{measure},{_csv_period(day, 'M')},{value}"
            for day, value in generate(f"{measure}/{area}", walk, start, today)
        ]
    return httpx.Response(200, content="\n".join(lines).encode())
