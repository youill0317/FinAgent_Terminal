from __future__ import annotations

import json
import random
import threading
import zlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import pairwise, product
from typing import TYPE_CHECKING, Final
from urllib.parse import unquote

import httpx

from finagent.sources import ecos_period, parse_period
from tests import fake_calendar

if TYPE_CHECKING:
    from collections.abc import Callable

    from finagent.models import Freq

DEMO_KEYS: Final = {"FRED_API_KEY": "demo", "ECOS_API_KEY": "demo"}

type Anchors = tuple[tuple[float, float], ...]


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
    "A191RO1Q156NBEA": _w("Q", (4, 1.9), (2, 2.9), (1, 2.0), (0, 2.1), vol=0.2),
    "GDPNOW": _w("Q", (4, 2.0), (2, 2.8), (1, 1.5), (0, 2.4), vol=0.5),
    "CPIAUCSL": _w("M", (4, 290), (0, 327), vol=0.15),
    "CPILFESL": _w("M", (4, 295), (0, 333), vol=0.1),
    "PCEPI": _w("M", (4, 117), (0, 128), vol=0.05),
    "PCEPILFE": _w("M", (4, 118), (0, 129), vol=0.05),
    "PPIFIS": _w("M", (4, 140), (0, 152), vol=0.2),
    "T5YIE": _w("D", (4, 2.5), (0, 2.3), vol=0.03),
    "UNRATE": _w("M", (4, 3.6), (2, 4.0), (0, 4.3), vol=0.05),
    "SAHMREALTIME": _w("M", (4, 0.1), (2, 0.4), (1, 0.2), (0, 0.27), vol=0.02),
    "PAYEMS": _w("M", (4, 152000), (0, 160000), vol=40),
    "ICSA": _w("W", (4, 210000), (0, 226000), vol=6000),
    "JTSJOL": _w("M", (4, 9500), (2, 7600), (0, 7100), vol=100),
    "RRSFS": _w("M", (4, 215000), (0, 221500), vol=600),
    "INDPRO": _w("M", (4, 102.0), (0, 104.0), vol=0.3),
    "PCEC96": _w("M", (4, 15500), (0, 16600), vol=25),
    "PNFIC1": _w("Q", (4, 3300), (0, 3600), vol=10),
    "WM2NS": _w("W", (4, 20800), (0, 22600), vol=20),
    "LCEAMN01JPM659S": _w("M", (4, 1.5), (2, 3.0), (0, 2.6), vol=0.3),
    "TOTLL": _w("W", (4, 11900), (0, 12800), vol=15),
    "A191RL1Q225SBEA": _w(
        "Q", (4, 2.2), (3, 3.0), (2, 1.6), (1, 2.4), (0, 2.1), vol=0.6
    ),
    "CES0500000003": _w("M", (4, 31.0), (0, 36.5), vol=0.03),
    "GACDFSA066MSFRBPHI": _w("M", (4, -10), (2, 5), (1, -8), (0, 3.5), vol=4),
    "GACDISA066MSFRBNY": _w("M", (4, -5), (2, 8), (1, -12), (0, 2.0), vol=4),
    "BOPGSTB": _w("M", (4, -70000), (1, -90000), (0, -65000), vol=3000),
    "RSAFS": _w("M", (4, 680000), (0, 725000), vol=2000),
    "TCU": _w("M", (4, 79.5), (0, 77.6), vol=0.2),
    "NEWORDER": _w("M", (4, 72000), (0, 76000), vol=400),
    "HOUST": _w("M", (4, 1550), (2, 1350), (0, 1320), vol=40),
    "TDSP": _w("Q", (4, 9.5), (0, 11.3), vol=0.05),
    "DRCCLACBS": _w("Q", (4, 1.7), (0, 3.1), vol=0.05),
    "DRBLACBS": _w("Q", (4, 1.0), (0, 1.3), vol=0.03),
    "DFII10": _w("D", (4, 1.5), (2, 2.2), (0, 1.8), vol=0.03),
    "BOPGEXP": _w("M", (4, 165000), (0, 180000), vol=2500),
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
    "VIXCLS": _w("D", (4, 20), (1.5, 15), (0.5, 24), (0, 16.8), vol=1.5),
    "NFCI": _w("W", (4, -0.3), (2, -0.5), (0, -0.52), vol=0.02),
    "SP500": _w("D", (4, 4300), (2, 5200), (1, 5900), (0, 6650), vol=40),
    "NIKKEI225": _w("D", (4, 27500), (2, 38500), (1, 39000), (0, 48500), vol=350),
    "WALCL": _w("W", (4, 8300000), (0, 6550000), vol=5000),
    "BAMLH0A0HYM2": _w("D", (3, 4.5), (1.5, 3.2), (0.5, 3.6), (0, 3.05), vol=0.05),
    "BAMLHE00EHYIOAS": _w("D", (3, 5.0), (1.5, 3.8), (0.5, 3.4), (0, 3.0), vol=0.05),
    "DRTSCILM": _w("Q", (4, 40), (2, 8), (1, -2), (0, 5), vol=3),
    "DTWEXBGS": _w("D", (4, 121), (1.7, 128), (0.8, 121), (0, 120.4), vol=0.4),
    "DEXJPUS": _w("D", (4, 145), (2, 155), (1, 150), (0, 152), vol=0.8),
    "DEXCHUS": _w("D", (4, 7.2), (1.5, 7.28), (0, 7.12), vol=0.01),
    "DCOILWTICO": _w("D", (4, 80), (1, 70), (0, 64.2), vol=1.2),
    "DCOILBRENTEU": _w("D", (4, 84), (1, 74), (0, 67.5), vol=1.2),
    "DHHNGSP": _w("D", (4, 2.5), (1, 3.2), (0, 3.0), vol=0.15),
    "PCOPPUSDM": _w("M", (4, 8500), (0, 9840), vol=150),
    "UMCSENT": _w("M", (4, 62), (2, 72), (1, 55), (0, 58), vol=2),
    "JPNGDPRQPSMEI": _w("Q", (4, 1.6), (2, 0.5), (0, 0.7), vol=0.3),
    "JPNSLRTTO01GYSAM": _w("M", (4, 1.0), (0, 1.5), vol=1.0),
    "JPNPRINTO01GYSAM": _w("M", (4, -2.0), (0, 2.2), vol=1.0),
    "JPNRGDPPC": _w("Q", (4, 300000), (0, 310000), vol=800),
    "NFIRSAXDCJPQ": _w("Q", (4, 38000000), (0, 39500000), vol=150000),
    "XTEXVA01JPM667S": _w("M", (4, 60e9), (0, 69e9), vol=1.5e9),
    "XTIMVA01JPM667S": _w("M", (4, 65e9), (0, 73e9), vol=1.5e9),
    "XTEXVA01CNM667S": _w("M", (4, 300e9), (0, 430e9), vol=8e9),
    "XTIMVA01CNM667S": _w("M", (4, 220e9), (0, 290e9), vol=6e9),
}

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


def _curve_walk(year: float, month: float, now: float) -> Walk:
    return _w("D", (4, year + 0.4), (1, year), (1 / 12, month), (0, now), vol=0.02)


for _series_id, _year, _month, _now in _CURVE:
    FRED[_series_id] = _curve_walk(_year, _month, _now)

ECOS: Final[dict[str, Walk]] = {
    "817Y002/010300000": _w("D", (4, 4.3), (2, 3.9), (1, 3.1), (0, 2.95), vol=0.02),
    "731Y001/0000001": _w(
        "D", (4, 1300), (2, 1340), (1.5, 1470), (1, 1400), (0, 1385), vol=8
    ),
    "802Y001/0001000": _w("D", (4, 2400), (1.5, 2650), (1, 2500), (0, 3500), vol=25),
    "200Y102/10211": _w("Q", (4, 3.0), (2, 1.0), (0, 3.4), vol=0.3),
    "200Y102/10222": _w("Q", (4, 2.5), (1, 1.0), (0, 1.8), vol=0.4),
    "200Y102/10223": _w("Q", (4, 3.0), (1, -2.0), (0, 2.5), vol=1.5),
    "901Y009/0": _w("M", (4, 108), (0, 118.5), vol=0.1),
    "901Y010/DB": _w("M", (4, 105), (0, 113), vol=0.1),
    "404Y014/*AA": _w("M", (4, 118), (0, 121), vol=0.2),
    "902Y008/JP": _w("M", (4, 102), (0, 114), vol=0.2),
    "902Y008/CN": _w("M", (4, 102), (0, 104), vol=0.2),
    "901Y027/I61BC/I28B": _w("M", (4, 2.8), (0, 2.7), vol=0.1),
    "902Y021/JPN": _w("M", (4, 2.7), (0, 2.5), vol=0.05),
    "901Y100/G0/T2": _w("M", (4, 102), (0, 104), vol=1.0),
    "901Y118/T002": _w("M", (4, 52000000), (0, 58000000), vol=2500000),
    "901Y118/T004": _w("M", (4, 50000000), (0, 53000000), vol=2000000),
    "200Y102/10111": _w("Q", (4, 0.8), (2, -0.2), (0, 1.0), vol=0.3),
    "200Y102/10224": _w("Q", (4, 2.0), (1, -3.0), (0, -0.7), vol=0.8),
    "511Y003/FMB": _w("M", (4, 3.5), (2, 3.0), (0, 2.7), vol=0.05),
    "901Y027/I61BA/I28B": _w("M", (4, 28300), (0, 28985), vol=15),
    "901Y027/I61E/I28B": _w("M", (4, 62.0), (0, 62.9), vol=0.1),
    "512Y013/99988/AA": _w("M", (4, 72), (2, 80), (0, 78), vol=1.5),
    "403Y001/3091AA": _w("M", (4, 95), (2, 140), (0, 185), vol=4),
    "901Y025/I31A": _w("M", (4, 76.0), (0, 71.2), vol=0.6),
    "151Y005/1110000": _w("M", (4, 1180000), (0, 1316613), vol=2000),
    "901Y067/I16E": _w("M", (4, 99.0), (2, 102.0), (0, 104.1), vol=0.1),
    "141Y004/1A0000/X00": _w("M", (4, 1300000), (0, 1512951), vol=3000),
    "104Y016/BDCA1": _w("M", (4, 2300000), (0, 2586900), vol=4000),
    "901Y033/AB00/2": _w("M", (4, 112.0), (0, 124.8), vol=1.5),
    "161Y005/BBHS00": _w("M", (4, 3800000), (0, 4350000), vol=4000),
    "902Y023/IRLT/JPN": _w("M", (4, 0.5), (1, 1.0), (0, 1.65), vol=0.05),
    "902Y023/IRLT/CHN": _w("M", (4, 2.8), (1, 1.9), (0, 1.8), vol=0.05),
    "514Y001/AA": _w("Q", (4, -15), (1, -5), (0, 3), vol=3),
    "902Y003/040101": _w("M", (4, 1800), (2, 1950), (1, 2600), (0, 4300), vol=40),
    "511Y002/FME/99988": _w("M", (4, 100), (1.8, 93), (0, 104), vol=1.2),
    "902Y015/JPN": _w("Q", (4, 0.3), (2, -0.4), (0, 0.4), vol=0.3),
}

_KTB: Final = (
    ("010190000", 2.95, 2.45, 2.40),
    ("010195000", 2.95, 2.50, 2.45),
    ("010200000", 2.95, 2.55, 2.55),
    ("010200001", 3.00, 2.70, 2.68),
    ("010210000", 3.05, 2.85, 2.86),
    ("010220000", 3.00, 2.80, 2.82),
    ("010230000", 2.95, 2.75, 2.76),
    ("010240000", 2.90, 2.70, 2.70),
)
for _item, _year, _month, _now in _KTB:
    ECOS[f"817Y002/{_item}"] = _curve_walk(_year, _month, _now)

SDMX: Final[dict[str, Walk]] = {
    "HICP/M.U2.N.000000.4D0.ANR": _w(
        "M", (4, 5.0), (2, 2.4), (1, 2.2), (0, 2.1), vol=0.1
    ),
    "HICP/M.U2.N.XEF000.4D0.ANR": _w("M", (4, 5.0), (2, 2.7), (0, 2.3), vol=0.1),
    "STBS/M.I10.N.PRIN.NS0020.4D0.GY.IX": _w(
        "M", (4, 30.0), (3, -5.0), (1, 1.0), (0, 2.0), vol=0.5
    ),
    "STBS/M.I10.W.PROD.NS0020.4D0.GY.IX": _w(
        "M", (4, -3.0), (2, -2.0), (1, 1.0), (0, 1.2), vol=0.8
    ),
    "STBS/M.I10.Y.TOVV.NS4701.4D0.N.IX": _w("M", (4, 105), (0, 107), vol=0.4),
    "MNA/Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY": _w(
        "Q", (4, 3.5), (2, 0.4), (0, 1.3), vol=0.2
    ),
    "MNA/Q.Y.I10.W0.S1M.S1.D.P31._Z._Z._T.EUR.LR.GY": _w(
        "Q", (4, 2.0), (2, 0.8), (0, 1.2), vol=0.2
    ),
    "MNA/Q.Y.I10.W0.S1.S1.D.P51G.N11G._T._Z.EUR.LR.GY": _w(
        "Q", (4, 1.5), (2, -1.0), (0, 1.8), vol=0.8
    ),
    "LFSI/M.I10.S.UNEHRT.TOTAL0.15_74.T": _w(
        "M", (4, 6.6), (2, 6.4), (0, 6.3), vol=0.05
    ),
    "BSI/M.U2.Y.V.M30.X.I.U2.2300.Z01.A": _w(
        "M", (4, -0.5), (2, 1.5), (1, 3.5), (0, 2.9), vol=0.2
    ),
    "ILM/W.U2.C.T000000.Z5.Z01": _w("W", (4, 8700000), (0, 5935000), vol=8000),
    "BLS/Q.U2.ALL.O.E.Z.B3.ST.S.WFNET": _w("Q", (4, 25), (2, 5), (0, 4), vol=3),
    "EXR/D.USD.EUR.SP00.A": _w(
        "D", (4, 1.0), (1.5, 1.08), (0.8, 1.04), (0.3, 1.17), (0, 1.164), vol=0.004
    ),
    "TRD/M.I10.N.X.TTT.J10.4.VAL": _w("M", (4, 235000000), (0, 258000000), vol=8000000),
    "CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX": _w(
        "D", (4, 0.15), (1.5, 0.06), (0.5, 0.1), (0, 0.05), vol=0.01
    ),
    "WS_CBPOL/D.US/all": _step(
        "D",
        (4, 5.375),
        (2.0, 4.875),
        (1.8, 4.375),
        (0.9, 4.125),
        (0.35, 3.875),
        (0.08, 3.625),
    ),
    "WS_CBPOL/D.XM/all": _step(
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
    "WS_CBPOL/D.JP/all": _step(
        "D", (4, -0.1), (2.5, 0.1), (2.2, 0.25), (1.7, 0.5), (0.8, 0.75)
    ),
    "WS_CBPOL/D.CN/all": _step("D", (4, 3.55), (1.3, 3.35), (1.1, 3.1), (0.4, 3.0)),
    "WS_CBPOL/D.KR/all": _step(
        "D", (4, 3.5), (1.6, 3.25), (1.3, 3.0), (0.9, 2.75), (0.6, 2.5)
    ),
    "WS_TC/Q.US.H.A.M.770.A/all": _w("Q", (4, 75), (0, 69), vol=0.2),
    "WS_DSR/Q.KR.H.367/all": _w("Q", (4, 12.5), (0, 11.5), vol=0.05),
    "WS_DSR/Q.JP.H.367/all": _w("Q", (4, 7.3), (0, 7.5), vol=0.03),
    "WS_CREDIT_GAP/Q.US.P.A.C.770/all": _w("Q", (4, -2.0), (0, -11.3), vol=0.3),
    "WS_CREDIT_GAP/Q.KR.P.A.C.770/all": _w("Q", (4, 5.0), (0, -15.1), vol=0.5),
    "WS_CREDIT_GAP/Q.XM.P.A.C.770/all": _w("Q", (4, -8.0), (0, -17.1), vol=0.3),
    "WS_CREDIT_GAP/Q.JP.P.A.C.770/all": _w("Q", (4, 10.0), (0, 5.7), vol=0.3),
    "WS_CREDIT_GAP/Q.CN.P.A.C.770/all": _w("Q", (4, 3.0), (0, -5.1), vol=0.4),
    "WS_SPP/Q.US.R.771/all": _w("Q", (4, 8.0), (2, 1.0), (0, -2.6), vol=0.4),
    "WS_SPP/Q.KR.R.771/all": _w("Q", (4, 5.0), (2, -6.0), (0, -0.5), vol=0.4),
    "WS_SPP/Q.XM.R.771/all": _w("Q", (4, 4.0), (2, -3.0), (0, 1.0), vol=0.3),
    "WS_SPP/Q.JP.R.771/all": _w("Q", (4, 3.0), (0, 2.4), vol=0.3),
    "WS_SPP/Q.CN.R.771/all": _w("Q", (4, 1.0), (2, -4.0), (0, -7.0), vol=0.4),
    "MNA/Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.G1": _w(
        "Q", (4, 0.3), (0, 0.4), vol=0.15
    ),
    "MNA/Q.Y.I10.W2.S1.S1._Z.EMP._Z._T._Z.PS._Z.N": _w(
        "Q", (4, 168000), (0, 176500), vol=100
    ),
    "TRD/M.I10.N.M.TTT.J10.4.VAL": _w("M", (4, 230000000), (0, 243000000), vol=4000000),
    "BSI/M.U2.Y.U.A20.A.I.U2.2250.Z01.A": _w("M", (4, 1.0), (0, 3.1), vol=0.1),
    "BSI/M.U2.Y.U.A20.A.I.U2.2240.Z01.A": _w("M", (4, 0.5), (0, 4.3), vol=0.15),
    "SPF/Q.U2.HICP.POINT.LT.Q.AVG": _w("Q", (4, 2.1), (0, 2.0), vol=0.02),
    "WS_TC/Q.KR.H.A.M.770.A/all": _w("Q", (4, 102), (0, 90), vol=0.3),
    "WS_TC/Q.XM.H.A.M.770.A/all": _w("Q", (4, 58), (0, 50), vol=0.2),
    "WS_TC/Q.JP.H.A.M.770.A/all": _w("Q", (4, 64), (0, 61), vol=0.2),
    "WS_TC/Q.CN.H.A.M.770.A/all": _w("Q", (4, 62), (0, 58), vol=0.2),
    "WS_TC/Q.US.N.A.M.770.A/all": _w("Q", (4, 78), (0, 72), vol=0.2),
    "WS_TC/Q.KR.N.A.M.770.A/all": _w("Q", (4, 118), (0, 107), vol=0.5),
    "WS_TC/Q.XM.N.A.M.770.A/all": _w("Q", (4, 110), (0, 103), vol=0.4),
    "WS_TC/Q.JP.N.A.M.770.A/all": _w("Q", (4, 118), (0, 113), vol=0.4),
    "WS_TC/Q.CN.N.A.M.770.A/all": _w("Q", (4, 160), (0, 146), vol=0.5),
}

_EA_CURVE: Final = (
    ("3M", 2.10, 1.95, 1.92),
    ("1Y", 2.05, 1.95, 1.95),
    ("2Y", 2.05, 2.00, 2.02),
    ("3Y", 2.10, 2.10, 2.12),
    ("5Y", 2.25, 2.30, 2.33),
    ("7Y", 2.40, 2.50, 2.52),
    ("10Y", 2.60, 2.75, 2.78),
    ("20Y", 2.95, 3.15, 3.18),
    ("30Y", 3.00, 3.20, 3.22),
)
for _tenor, _year, _month, _now in _EA_CURVE:
    SDMX[f"YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{_tenor}"] = _curve_walk(_year, _month, _now)

OECD_DEMO: Final[dict[tuple[str, str], Walk]] = {
    ("LI", "USA"): _w("M", (4, 99.0), (0, 100.3), vol=0.05),
    ("LI", "KOR"): _w("M", (4, 98.8), (0, 100.6), vol=0.05),
    ("LI", "JPN"): _w("M", (4, 100.2), (0, 100.4), vol=0.05),
    ("LI", "CHN"): _w("M", (4, 99.5), (0, 100.1), vol=0.05),
    ("BCICP", "USA"): _w("M", (4, 99.0), (0, 99.6), vol=0.1),
    ("BCICP", "KOR"): _w("M", (4, 98.6), (0, 98.9), vol=0.1),
    ("BCICP", "EA20"): _w("M", (4, 98.0), (0, 100.2), vol=0.1),
    ("BCICP", "JPN"): _w("M", (4, 100.5), (0, 101.7), vol=0.1),
    ("BCICP", "CHN"): _w("M", (4, 99.4), (0, 99.0), vol=0.1),
    ("CCICP", "EA20"): _w("M", (4, 96.5), (0, 98.3), vol=0.1),
    ("CCICP", "JPN"): _w("M", (4, 97.5), (0, 98.5), vol=0.1),
    ("CCICP", "CHN"): _w("M", (4, 99.0), (0, 97.0), vol=0.1),
    ("MABM", "JPN"): _w("M", (4, 3.0), (0, 1.4), vol=0.1),
    ("B1GQ_Q", "CHN"): _w("Q", (4, 5.4), (0, 4.5), vol=0.2),
    ("MABM", "CHN"): _w("M", (4, 260000000), (0, 356000000), vol=500000),
    ("CPI", "JPN"): _w("M", (4, 2.8), (2, 2.0), (0, 2.2), vol=0.1),
}

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
    lag_days, step = {"Q": (120, 3), "A": (300, 12)}.get(freq, (40, 1))
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
    rng = random.Random(zlib.crc32(key.encode()))  # noqa: S311
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


@dataclass(slots=True)
class FakeClock:
    now: float = 1_800_000_000.0
    slept: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.slept.append(seconds)
            self.now += max(seconds, 0.0)

    async def asleep(self, seconds: float) -> None:
        self.sleep(seconds)


_ANSWERS: Final[dict[tuple[httpx.URL, date], tuple[int, bytes]]] = {}


def demo_handler(request: httpx.Request, today: date) -> httpx.Response:
    key = (request.url, today)
    if key not in _ANSWERS:
        answer = _answer(request, today)
        _ANSWERS[key] = (answer.status_code, answer.content)
    status, content = _ANSWERS[key]
    return httpx.Response(status, content=content)


def _answer(request: httpx.Request, today: date) -> httpx.Response:
    routes: dict[str, Callable[[httpx.Request, date], httpx.Response]] = {
        "api.stlouisfed.org": _fred,
        "ecos.bok.or.kr": _ecos,
        "data-api.ecb.europa.eu": _sdmx,
        "stats.bis.org": _sdmx,
        "sdmx.oecd.org": _oecd,
    }
    special = fake_calendar.handle(request, today)
    if special is not None:
        return special
    route = routes.get(request.url.host)
    if route is None:
        return httpx.Response(404)
    return route(request, today)


def demo_transport(today: date) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        return demo_handler(request, today)

    return httpx.MockTransport(handle)


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
    stat, cycle, start_text, item1 = parts[7], parts[8], parts[9], parts[11]
    key = "/".join([stat, *parts[11:]])
    walk = ECOS.get(key)
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
        for day, value in generate(key, walk, start, today)
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
    flow, key = path.split(marker, 1)[-1].split("/", 1)
    bis = key.endswith("/all")
    start = _start(request, "startPeriod", 4, today)
    lines = ["KEY,REF_AREA,BORROWERS_CTY,TC_BORROWERS,FREQ,TIME_PERIOD,OBS_VALUE"]
    parts = key.removesuffix("/all").split(".")
    for dims in product(*(dim.split("+") for dim in parts)):
        single = ".".join(dims)
        walk = SDMX.get(f"{flow}/{single}/all" if bis else f"{flow}/{single}")
        if walk is None:
            continue
        area = dims[1] if len(dims) > 1 else ""
        sector = dims[2] if len(dims) > 2 else ""
        head = f"{flow}.{single},{area},{area},{sector},{walk.freq}"
        lines += [
            f"{head},{_csv_period(day, walk.freq)},{value}"
            for day, value in generate(f"{flow}/{single}", walk, start, today)
        ]
    if len(lines) == 1:
        return httpx.Response(404)
    return httpx.Response(200, content="\n".join(lines).encode())


def _oecd(request: httpx.Request, today: date) -> httpx.Response:
    path = unquote(request.url.path)
    if "/dataflow" in path:
        tail = path.split("/dataflow/", 1)[-1].split("/")
        if len(tail) < 2:
            return _structure(["DSD_STES@DF_CLI"], [])
        return _structure([tail[1]], ["USA.M.LI", "KOR.M.BCICP"])
    start = _start(request, "startPeriod", 4, today)
    flow, key = path.split("/data/", 1)[-1].replace(",", ":").rsplit("/", 1)
    dims = [dim.split("+") for dim in key.split(".")]
    areas = set(dims[0])
    tokens = {token for dim in dims[1:] for token in dim}
    lines = ["DATAFLOW,REF_AREA,FREQ,MEASURE,TIME_PERIOD,OBS_VALUE"]
    for (token, area), walk in OECD_DEMO.items():
        if area not in areas or token not in tokens:
            continue
        lines += [
            f"{flow},{area},{walk.freq},{token},{_csv_period(day, walk.freq)},{value}"
            for day, value in generate(f"{token}/{area}", walk, start, today)
        ]
    return httpx.Response(200, content="\n".join(lines).encode())


ALPACA_DIVIDENDS: Final = {
    "corporate_actions": {
        "cash_dividends": [
            {"symbol": "EWT", "ex_date": "2025-12-16", "rate": 1.802131},
            {"symbol": "EWT", "ex_date": "2025-12-16", "rate": 1.014019},
            {"symbol": "VGT", "ex_date": "2025-12-17", "rate": 0.757},
            {"symbol": "VGT", "ex_date": "2026-03-24", "rate": 0.7438},
            {"symbol": "VGT", "ex_date": "2026-06-24", "rate": 0.1384},
            {"symbol": "VGT", "ex_date": "2026-09-23", "rate": 0.1465},
        ]
    }
}
ALPACA_SPLITS: Final = {
    "corporate_actions": {
        "forward_splits": [
            {"symbol": "VGT", "ex_date": "2026-04-21", "new_rate": 8, "old_rate": 1},
            {"symbol": "SOXX", "ex_date": "2026-11-05", "new_rate": 3, "old_rate": 1},
        ]
    }
}

type NportRow = tuple[str, str, str, float, str | None, str | None, str, str | None]


def _nport_row(row: NportRow) -> str:
    name, cusip, isin, pct, asset, issuer, country, maturity = row
    debt = "" if maturity is None else f"<debtSec><maturityDt>{maturity}</maturityDt>"
    debt += "" if maturity is None else "<couponKind>Fixed</couponKind></debtSec>"
    ids = f'<identifiers><isin value="{isin}"/></identifiers>'
    head = f"<invstOrSec><name>{name}</name><cusip>{cusip}</cusip>{ids}"
    return (
        f"{head}<pctVal>{pct}</pctVal>"
        + ("" if asset is None else f"<assetCat>{asset}</assetCat>")
        + ("" if issuer is None else f"<issuerCat>{issuer}</issuerCat>")
        + f"<invCountry>{country}</invCountry>{debt}</invstOrSec>"
    )


def nport(as_of: str, rows: list[NportRow]) -> str:
    body = "".join(_nport_row(row) for row in rows)
    head = '<?xml version="1.0" encoding="UTF-8"?>'
    root = '<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">'
    info = (
        f"<formData><genInfo><repPdDate>{as_of}</repPdDate></genInfo>"
        "<fundInfo><netAssets>22300000000.00</netAssets></fundInfo>"
    )
    tail = "</invstOrSecs></formData></edgarSubmission>"
    return f"{head}{root}{info}<invstOrSecs>{body}{tail}"


NPORT_TLT: Final = nport(
    "2026-08-31",
    [
        (
            "United States of America",
            "912810UK2",
            "US912810UK24",
            49.0,
            "DBT",
            "UST",
            "US",
            "2055-08-15",
        ),
        (
            "United States of America",
            "912810TT5",
            "US912810TT51",
            49.15,
            "DBT",
            "UST",
            "US",
            "2045-08-15",
        ),
        (
            "BlackRock Cash Funds",
            "066922519",
            "US0669225197",
            1.09,
            "STIV",
            "RF",
            "US",
            None,
        ),
    ],
)
NPORT_EWY: Final = nport(
    "2026-08-31",
    [
        (
            "Samsung Electronics Co., Ltd.",
            "N/A",
            "KR7005930003",
            22.13,
            "EC",
            "CORP",
            "KR",
            None,
        ),
        ("SK hynix Inc.", "N/A", "KR7000660001", 20.96, "EC", "CORP", "KR", None),
        (
            "SK Square Co., Ltd.",
            "Y8T6X4107",
            "KR7402340004",
            55.23,
            "EC",
            "CORP",
            "KR",
            None,
        ),
        ("Hyundai Motor Pref", "N/A", "KR7005381009", 0.78, "EP", "CORP", "KR", None),
        (
            "BlackRock Cash Funds",
            "066922519",
            "US0669225197",
            5.48,
            "STIV",
            "RF",
            "US",
            None,
        ),
        ("KRW", "N/A", "N/A", 0.51, None, None, "KR", None),
    ],
)
NPORT_FUND_OF_FUNDS: Final = nport(
    "2026-07-31",
    [
        (
            "Vanguard Real Estate II Index Fund",
            "922031695",
            "US9220316957",
            14.5,
            "EC",
            "RF",
            "US",
            None,
        ),
        ("Welltower Inc", "95040Q104", "US95040Q1040", 87.0, "EC", "CORP", "US", None),
    ],
)
