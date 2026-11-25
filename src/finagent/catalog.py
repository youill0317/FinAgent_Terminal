from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from finagent.models import Fmt
from finagent.sources import (
    WINDOW_YEARS,
    Difference,
    EcosSeries,
    EtfSeries,
    FredSeries,
    SdmxSeries,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from finagent.models import Area, Freq
    from finagent.sources import Fetch
    from finagent.transforms import Transform


type Change = Literal["diff", "pct"]


@dataclass(frozen=True, slots=True)
class Indicator:
    id: str
    label: str
    code: str
    area: Area
    fetch: Fetch | Difference
    freq: Freq
    fmt: Fmt
    transform: Transform = "level"
    scale: float = 1.0
    copyright: str | None = None
    change: Change = "diff"


@dataclass(frozen=True, slots=True)
class PanelSpec:
    id: str
    page: str
    title: str
    note: str
    indicators: tuple[Indicator, ...]


@dataclass(frozen=True, slots=True)
class CurveSpec:
    area: Area
    label: str
    tenors: tuple[tuple[str, Fetch], ...]


PCT1: Final = Fmt(suf="%", dec=1)
PCT2: Final = Fmt(suf="%", dec=2)
PCT1_SIGNED: Final = Fmt(suf="%", dec=1, signed=True)
PP2: Final = Fmt(suf="%p", dec=2)
PP2_SIGNED: Final = Fmt(suf="%p", dec=2, signed=True)
PP1_SIGNED: Final = Fmt(suf="%p", dec=1, signed=True)
IDX0: Final = Fmt(dec=0)
IDX1: Final = Fmt(dec=1)
IDX2: Final = Fmt(dec=2)
IDX0_SIGNED: Final = Fmt(dec=0, signed=True)
IDX1_SIGNED: Final = Fmt(dec=1, signed=True)
IDX2_SIGNED: Final = Fmt(dec=2, signed=True)
FX4: Final = Fmt(dec=4)
USD_T: Final = Fmt(pre="$", suf="T", dec=2)
USD_B1_SIGNED: Final = Fmt(pre="$", suf="B", dec=1, signed=True)
USD2: Final = Fmt(pre="$", dec=2)
USD_BBL: Final = Fmt(pre="$", suf="/bbl", dec=2)
USD_MMBTU: Final = Fmt(pre="$", suf="/MMBtu", dec=2)
USD_T0: Final = Fmt(pre="$", suf="/t", dec=0)
USD_OZ: Final = Fmt(pre="$", suf="/oz", dec=0)
USD_LB: Final = Fmt(pre="$", suf="/lb", dec=2)
EUR_T: Final = Fmt(pre="€", suf="T", dec=2)
EUR_B1_SIGNED: Final = Fmt(pre="€", suf="B", dec=1, signed=True)
JPY_T: Final = Fmt(pre="¥", suf="T", dec=0)
KRW: Final = Fmt(suf="원", dec=1)
KRW_T: Final = Fmt(suf="조원", dec=1)
THOUSANDS_SIGNED: Final = Fmt(suf="K", dec=0, signed=True)
THOUSANDS: Final = Fmt(suf="K", dec=0)
MILLIONS: Final = Fmt(suf="M", dec=2)

AREAS: Final[dict[str, Area]] = {
    "us": "US",
    "kr": "KR",
    "ea": "EA",
    "jp": "JP",
    "cn": "CN",
    "gl": "GL",
}


def area_of(id_: str) -> Area:
    return AREAS[id_.split("_", 1)[0]]


def _indicator(  # noqa: PLR0913
    id_: str,
    label: str,
    code: str,
    fetch: Fetch | Difference,
    freq: Freq,
    fmt: Fmt,
    transform: Transform = "level",
    scale: float = 1.0,
    copyright_: str | None = None,
) -> Indicator:
    return Indicator(
        id_, label, code, area_of(id_), fetch, freq, fmt, transform, scale, copyright_
    )


ICE_YEARS: Final = 3


def fred(  # noqa: PLR0913
    id_: str,
    label: str,
    series_id: str,
    freq: Freq,
    fmt: Fmt,
    transform: Transform = "level",
    *,
    scale: float = 1.0,
    copyright_: str | None = None,
    years: int = WINDOW_YEARS,
) -> Indicator:
    return _indicator(
        id_,
        label,
        f"FRED:{series_id}",
        FredSeries(series_id, years),
        freq,
        fmt,
        transform,
        scale,
        copyright_,
    )


def ice_oas(id_: str, series_id: str) -> Indicator:
    return fred(
        id_,
        "하이일드 스프레드 (OAS)",
        series_id,
        "D",
        PP2,
        copyright_="ICE",
        years=ICE_YEARS,
    )


def ecos(  # noqa: PLR0913
    id_: str,
    label: str,
    stat: str,
    cycle: Freq,
    item1: str,
    fmt: Fmt,
    transform: Transform = "level",
    item2: str = "",
    *,
    scale: float = 1.0,
) -> Indicator:
    code = f"ECOS:{stat}/{item1}" + (f"/{item2}" if item2 else "")
    fetch = EcosSeries(stat, cycle, item1, item2)
    return _indicator(id_, label, code, fetch, cycle, fmt, transform, scale)


ECB_TENORS: Final = ("3M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y")
ECB_LONG_TENOR: Final = "10Y"
ECB_CURVE_TENORS: Final = tuple(t for t in ECB_TENORS if t != ECB_LONG_TENOR)
YC_FLOW: Final = "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_"
ECB_GROUPS: Final[dict[str, str]] = {
    key: request
    for request, keys in (
        (
            "MNA/Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY+G1",
            (
                "MNA/Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY",
                "MNA/Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.G1",
            ),
        ),
        (
            "HICP/M.U2.N.000000+XEF000.4D0.ANR",
            ("HICP/M.U2.N.000000.4D0.ANR", "HICP/M.U2.N.XEF000.4D0.ANR"),
        ),
        (
            "STBS/M.I10.N+Y+W.PRIN+TOVV+PROD.NS0020+NS4701.4D0.GY+N.IX",
            (
                "STBS/M.I10.N.PRIN.NS0020.4D0.GY.IX",
                "STBS/M.I10.Y.TOVV.NS4701.4D0.N.IX",
                "STBS/M.I10.W.PROD.NS0020.4D0.GY.IX",
            ),
        ),
        (
            "TRD/M.I10.N.X+M.TTT.J10.4.VAL",
            ("TRD/M.I10.N.X.TTT.J10.4.VAL", "TRD/M.I10.N.M.TTT.J10.4.VAL"),
        ),
        (
            "BSI/M.U2.Y.U+V.A20+M30.A+X.I.U2.2250+2240+2300.Z01.A",
            (
                "BSI/M.U2.Y.U.A20.A.I.U2.2250.Z01.A",
                "BSI/M.U2.Y.U.A20.A.I.U2.2240.Z01.A",
                "BSI/M.U2.Y.V.M30.X.I.U2.2300.Z01.A",
            ),
        ),
        (
            f"{YC_FLOW}{'+SR_'.join(ECB_CURVE_TENORS)}",
            tuple(f"{YC_FLOW}{tenor}" for tenor in ECB_CURVE_TENORS),
        ),
    )
    for key in keys
}
BIS_GROUPS: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "WS_CBPOL": ("D.US+XM+JP+CN+KR", ("REF_AREA",)),
    "WS_SPP": ("Q.US+KR+XM+JP+CN.R.771", ("REF_AREA",)),
    "WS_TC": ("Q.US+KR+XM+JP+CN.H+N.A.M.770.A", ("BORROWERS_CTY", "TC_BORROWERS")),
    "WS_CREDIT_GAP": ("Q.US+KR+XM+JP+CN.P.A.C.770", ("BORROWERS_CTY",)),
    "WS_DSR": ("Q.KR+JP.H.367", ("BORROWERS_CTY",)),
}


def ecb_fetch(flow_key: str) -> SdmxSeries:
    request = ECB_GROUPS.get(flow_key)
    if request is None:
        return SdmxSeries("ECB", flow_key)
    return SdmxSeries("ECB", request, (("KEY", flow_key.replace("/", ".", 1)),))


def bis_fetch(flow: str, key: str) -> SdmxSeries:
    request, columns = BIS_GROUPS[flow]
    match = tuple(zip(columns, key.split(".")[1:], strict=False))
    return SdmxSeries("BIS", f"{flow}/{request}/all", match)


def ecb(  # noqa: PLR0913
    id_: str,
    label: str,
    flow_key: str,
    freq: Freq,
    fmt: Fmt,
    transform: Transform = "level",
    *,
    scale: float = 1.0,
) -> Indicator:
    fetch = ecb_fetch(flow_key)
    return _indicator(id_, label, f"ECB:{flow_key}", fetch, freq, fmt, transform, scale)


def bis(  # noqa: PLR0913
    id_: str, label: str, flow: str, key: str, freq: Freq, fmt: Fmt
) -> Indicator:
    return _indicator(id_, label, f"BIS:{flow}/{key}", bis_fetch(flow, key), freq, fmt)


BIS_AREAS: Final[tuple[tuple[str, str], ...]] = (
    ("us", "US"),
    ("kr", "KR"),
    ("ea", "XM"),
    ("jp", "JP"),
    ("cn", "CN"),
)


def bis_by_area(  # noqa: PLR0913
    name: str,
    label: str,
    flow: str,
    key: str,
    freq: Freq,
    fmt: Fmt,
) -> tuple[Indicator, ...]:
    return tuple(
        bis(f"{prefix}_{name}", label, flow, key.format(area=code), freq, fmt)
        for prefix, code in BIS_AREAS
    )


def bis_policy_rate(id_: str, label: str, area: str) -> Indicator:
    return bis(id_, label, "WS_CBPOL", f"D.{area}", "D", PCT2)


def oecd(  # noqa: PLR0913
    id_: str,
    label: str,
    flow: str,
    key: str,
    freq: Freq,
    fmt: Fmt,
    transform: Transform = "level",
    *,
    match: tuple[tuple[str, str], ...] = (),
) -> Indicator:
    dsd = flow.split(",", 1)[-1]
    tail = ".".join(value for _, value in match) if match else key
    fetch = SdmxSeries("OECD", f"{flow}/{key}", match)
    return _indicator(id_, label, f"OECD:{dsd}/{tail}", fetch, freq, fmt, transform)


OECD_CLI: Final = "OECD.SDD.STES,DSD_STES@DF_CLI"
OECD_KEI: Final = "OECD.SDD.STES,DSD_KEI@DF_KEI"
OECD_MONAGG: Final = "OECD.SDD.STES,DSD_STES@DF_MONAGG"

CLI_KEY: Final = "USA+KOR+EA20+JPN+CHN.M.LI+BCICP+CCICP...AA...H"
KEI_KEY: Final = "CHN+JPN.Q+M.B1GQ_Q+MABM.GR._T+_Z.Y.GY"
OECD_CORE_CPI: Final = "OECD.SDD.TPS,DSD_PRICES_COICOP2018@DF_PRICES_C2018_N_TXCP01_NRG"
JP_CORE_CPI_KEY: Final = "JPN.M.N.CPI.PA._TXCP01_NRG.N.GY.."
TRADE_FRED: Final[dict[str, tuple[str, str]]] = {
    "JPN": ("XTEXVA01JPM667S", "XTIMVA01JPM667S"),
    "CHN": ("XTEXVA01CNM667S", "XTIMVA01CNM667S"),
}


def oecd_measure(  # noqa: PLR0913
    id_: str,
    label: str,
    flow: str,
    key: str,
    area: str,
    measure: str,
    freq: Freq,
    fmt: Fmt,
) -> Indicator:
    return oecd(
        id_,
        label,
        flow,
        key,
        freq,
        fmt,
        match=(("REF_AREA", area), ("MEASURE", measure)),
    )


def difference(  # noqa: PLR0913
    id_: str,
    label: str,
    code: str,
    left: Fetch,
    right: Fetch,
    freq: Freq,
    fmt: Fmt,
    *,
    scale: float = 1.0,
) -> Indicator:
    return _indicator(id_, label, code, Difference(left, right), freq, fmt, scale=scale)


def fred_trade_balance(id_: str, area: str) -> Indicator:
    exports_id, imports_id = TRADE_FRED[area]
    return difference(
        id_,
        "무역수지 (달러 기준)",
        f"FRED:{exports_id}−{imports_id}",
        FredSeries(exports_id),
        FredSeries(imports_id),
        "M",
        USD_B1_SIGNED,
        scale=1e-9,
    )


def cli(id_: str, area: str) -> Indicator:
    return oecd_measure(
        id_, "OECD 경기선행지수 CLI", OECD_CLI, CLI_KEY, area, "LI", "M", IDX1
    )


def exports(id_: str, area: str) -> Indicator:
    exports_id, _ = TRADE_FRED[area]
    return fred(id_, "수출 전년비 (달러 기준)", exports_id, "M", PCT1_SIGNED, "yoy")


def price(indicator: Indicator) -> Indicator:
    return replace(indicator, change="pct")


ETF_NOTE: Final = (
    "차트는 수정 종가를 고른 기간의 첫날 100으로 맞춰 그립니다"
    "\n\n선이 위에 있을수록 그 기간 수익률이 높습니다"
)


def etf(id_: str, label: str, ticker: str) -> Indicator:
    return price(
        _indicator(
            id_, f"{label} ({ticker})", f"ALPACA:{ticker}", EtfSeries(ticker), "D", USD2
        )
    )


def yahoo_price(id_: str, label: str, ticker: str, fmt: Fmt) -> Indicator:
    return price(
        _indicator(
            id_,
            label,
            f"YAHOO:{ticker}",
            EtfSeries(ticker, source="YAHOO"),
            "D",
            fmt,
        )
    )


PANELS: Final = (
    PanelSpec(
        "ov_growth",
        "overview",
        "경제 성장",
        (
            "5개 경제권 실질 GDP 전년비 및 OECD 경기선행지수"
            "\n\n선행지수 기준선 100(장기 추세)  \n경기 국면 전환 6~9개월 선행 경향"
        ),
        (
            fred(
                "us_gdp_yoy_ov",
                "실질 GDP 전년비",
                "A191RO1Q156NBEA",
                "Q",
                PCT1_SIGNED,
            ),
            ecos(
                "kr_gdp_yoy_ov", "실질 GDP 전년비", "200Y102", "Q", "10211", PCT1_SIGNED
            ),
            ecb(
                "ea_gdp_yoy_ov",
                "실질 GDP 전년비",
                "MNA/Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY",
                "Q",
                PCT1_SIGNED,
            ),
            fred("jp_gdp_ov", "실질 GDP 전년비", "JPNGDPRQPSMEI", "Q", PCT1_SIGNED),
            oecd_measure(
                "cn_gdp_ov",
                "실질 GDP 전년비",
                OECD_KEI,
                KEI_KEY,
                "CHN",
                "B1GQ_Q",
                "Q",
                PCT1_SIGNED,
            ),
            cli("us_cli", "USA"),
            cli("kr_cli", "KOR"),
            cli("jp_cli", "JPN"),
            cli("cn_cli", "CHN"),
        ),
    ),
    PanelSpec(
        "ov_inflation",
        "overview",
        "인플레이션",
        (
            "주요국 소비자물가(CPI) 전년비 추이"
            "\n\n주요 중앙은행 공식 물가 안정 목표치 2.0%"
        ),
        (
            fred("us_cpi", "CPI 전년비", "CPIAUCSL", "M", PCT1, "yoy"),
            ecos("kr_cpi_ov", "CPI 전년비", "901Y009", "M", "0", PCT1, "yoy"),
            ecb(
                "ea_hicp_ov",
                "CPI 전년비 (HICP)",
                "HICP/M.U2.N.000000.4D0.ANR",
                "M",
                PCT1,
            ),
            ecos("jp_cpi_ov", "CPI 전년비", "902Y008", "M", "JP", PCT1, "yoy"),
            ecos("cn_cpi_ov", "CPI 전년비", "902Y008", "M", "CN", PCT1, "yoy"),
            fred("us_core_pce", "근원 PCE 전년비", "PCEPILFE", "M", PCT1, "yoy"),
        ),
    ),
    PanelSpec(
        "ov_unemployment",
        "overview",
        "실업률",
        (
            "주요국 실업률 및 미국 삼의 법칙(Sahm Rule) 경기 침체 지표"
            "\n\n삼의 법칙 +0.50%p 이상 도달 시 경기 침체 신호"
        ),
        (
            fred("us_unrate", "실업률", "UNRATE", "M", PCT1),
            ecos(
                "kr_unrate_ov",
                "실업률 (SA)",
                "901Y027",
                "M",
                "I61BC",
                PCT1,
                item2="I28B",
            ),
            ecb(
                "ea_unrate_ov",
                "실업률 (15–74세)",
                "LFSI/M.I10.S.UNEHRT.TOTAL0.15_74.T",
                "M",
                PCT1,
            ),
            ecos("jp_unrate_ov", "실업률 (SA)", "902Y021", "M", "JPN", PCT1),
            fred("us_sahm", "삼의 법칙 지표 (실시간)", "SAHMREALTIME", "M", PP2),
        ),
    ),
    PanelSpec(
        "ov_rates",
        "overview",
        "금리",
        (
            "미국 장단기 금리차, 하이일드 스프레드, 기대인플레이션, 실질금리 종합"
            "\n\n장단기 금리차 0%p 하회 시 역전(침체 경고), "
            "스프레드 확대 시 신용 위험 가중"
        ),
        (
            fred("us_10y2y", "10년 − 2년", "T10Y2Y", "D", PP2_SIGNED),
            ice_oas("us_hy_oas_ov", "BAMLH0A0HYM2"),
            fred("us_t5yie", "5년 기대인플레이션 BEI", "T5YIE", "D", PCT2),
            fred("us_real_10y", "10년 실질금리 (TIPS)", "DFII10", "D", PCT2),
        ),
    ),
    PanelSpec(
        "ov_volatility",
        "overview",
        "시장 변동성",
        "",
        (fred("us_vix", "VIX", "VIXCLS", "D", IDX1, copyright_="Cboe"),),
    ),
    PanelSpec(
        "ov_equity",
        "overview",
        "주가지수",
        "",
        (
            price(
                fred("us_sp500", "S&P 500", "SP500", "D", IDX0, copyright_="S&P DJI")
            ),
            price(ecos("kr_kospi", "코스피", "802Y001", "D", "0001000", IDX2)),
            price(
                fred(
                    "jp_nikkei",
                    "닛케이 225",
                    "NIKKEI225",
                    "D",
                    IDX0,
                    copyright_="Nikkei",
                )
            ),
        ),
    ),
    PanelSpec(
        "ov_fx",
        "overview",
        "환율",
        "",
        (
            price(ecos("kr_usdkrw_ov", "원/달러", "731Y001", "D", "0000001", KRW)),
            price(fred("us_dollar_ov", "달러 광의지수", "DTWEXBGS", "D", IDX1)),
        ),
    ),
    PanelSpec(
        "ov_commodities",
        "overview",
        "원자재",
        "WTI 원유 및 금(COMEX 선물) 일간 가격 추이",
        (
            price(fred("gl_wti_ov", "WTI", "DCOILWTICO", "D", USD_BBL)),
            yahoo_price("gl_gold_ov", "금 (COMEX 선물)", "GC=F", USD_OZ),
        ),
    ),
    PanelSpec(
        "ov_policy",
        "overview",
        "정책 변화",
        "주요 5개 중앙은행(Fed, BOK, ECB, BOJ, PBOC) 기준 정책금리 및 최근 변동 내역",
        (
            bis_policy_rate("us_policy_rate_bis", "정책금리 (Fed)", "US"),
            bis_policy_rate("ea_policy_rate_bis", "정책금리 (ECB)", "XM"),
            bis_policy_rate("jp_policy_rate_bis", "정책금리 (BOJ)", "JP"),
            bis_policy_rate("cn_policy_rate_bis", "정책금리 (PBOC)", "CN"),
            bis_policy_rate("kr_policy_rate_bis", "정책금리 (한국은행)", "KR"),
        ),
    ),
    PanelSpec(
        "eco_gdp",
        "economy",
        "GDP 성장률",
        (
            "주요국 실질 GDP 성장률 및 경기선행지수"
            "\n\n전년비(YoY), 전기비 연율(SAAR), GDPNow 실시간 추정치는 "
            "산출 기준 상이로 상호 비교 불가"
        ),
        (
            fred("us_gdp_yoy", "실질 GDP 전년비", "A191RO1Q156NBEA", "Q", PCT1_SIGNED),
            fred(
                "us_gdp_saar",
                "실질 GDP 전기비 연율",
                "A191RL1Q225SBEA",
                "Q",
                PCT1_SIGNED,
            ),
            fred("us_gdpnow_eco", "GDPNow 나우캐스트", "GDPNOW", "Q", PCT1_SIGNED),
            ecos("kr_gdp_yoy", "실질 GDP 전년비", "200Y102", "Q", "10211", PCT1_SIGNED),
            ecos(
                "kr_gdp_qoq",
                "실질 GDP 전기비 (SA)",
                "200Y102",
                "Q",
                "10111",
                PCT1_SIGNED,
            ),
            ecos("kr_cli_cycle", "선행지수 순환변동치", "901Y067", "M", "I16E", IDX1),
            ecb(
                "ea_gdp_yoy",
                "실질 GDP 전년비",
                "MNA/Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY",
                "Q",
                PCT1_SIGNED,
            ),
            ecb(
                "ea_gdp_qoq",
                "실질 GDP 전기비 (SA)",
                "MNA/Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.G1",
                "Q",
                PCT1_SIGNED,
            ),
            fred("jp_gdp", "실질 GDP 전년비", "JPNGDPRQPSMEI", "Q", PCT1_SIGNED),
            ecos(
                "jp_gdp_qoq", "실질 GDP 전기비 (SA)", "902Y015", "Q", "JPN", PCT1_SIGNED
            ),
            oecd_measure(
                "cn_gdp",
                "실질 GDP 전년비",
                OECD_KEI,
                KEI_KEY,
                "CHN",
                "B1GQ_Q",
                "Q",
                PCT1_SIGNED,
            ),
            cli("us_cli_eco", "USA"),
            cli("kr_cli_eco", "KOR"),
            cli("jp_cli_eco", "JPN"),
            cli("cn_cli_eco", "CHN"),
        ),
    ),
    PanelSpec(
        "eco_inflation",
        "economy",
        "인플레이션율 (CPI, PPI)",
        (
            "소비자물가(CPI), 근원물가(Core), 생산자물가(PPI), 기대인플레이션 종합"
            "\n\n근원물가는 기저 인플레이션 추세, 생산자물가는 소비자물가 선행 지표"
        ),
        (
            fred("us_cpi_eco", "CPI 전년비", "CPIAUCSL", "M", PCT1, "yoy"),
            fred("us_core_cpi_eco", "근원 CPI 전년비", "CPILFESL", "M", PCT1, "yoy"),
            fred("us_pce", "PCE 전년비", "PCEPI", "M", PCT1, "yoy"),
            fred("us_core_pce_eco", "근원 PCE 전년비", "PCEPILFE", "M", PCT1, "yoy"),
            fred("us_ppi", "PPI 전년비 (최종수요)", "PPIFIS", "M", PCT1_SIGNED, "yoy"),
            ecos("kr_cpi", "CPI 전년비", "901Y009", "M", "0", PCT1, "yoy"),
            ecos(
                "kr_core_cpi",
                "근원 CPI 전년비 (식료품·에너지 제외)",
                "901Y010",
                "M",
                "DB",
                PCT1,
                "yoy",
            ),
            ecos("kr_ppi", "PPI 전년비", "404Y014", "M", "*AA", PCT1_SIGNED, "yoy"),
            ecos(
                "kr_inflation_expectations",
                "기대인플레이션율 (향후 1년)",
                "511Y003",
                "M",
                "FMB",
                PCT1,
            ),
            ecb(
                "ea_hicp", "CPI 전년비 (HICP)", "HICP/M.U2.N.000000.4D0.ANR", "M", PCT1
            ),
            ecb(
                "ea_core_hicp",
                "근원 CPI 전년비 (HICP, 에너지·식료품 제외)",
                "HICP/M.U2.N.XEF000.4D0.ANR",
                "M",
                PCT1,
            ),
            ecb(
                "ea_ppi",
                "PPI 전년비 (산업, 역내 판매)",
                "STBS/M.I10.N.PRIN.NS0020.4D0.GY.IX",
                "M",
                PCT1_SIGNED,
            ),
            ecb(
                "ea_inflation_expectations",
                "기대인플레이션 (SPF 장기)",
                "SPF/Q.U2.HICP.POINT.LT.Q.AVG",
                "Q",
                PCT1,
            ),
            ecos("jp_cpi", "CPI 전년비", "902Y008", "M", "JP", PCT1, "yoy"),
            oecd_measure(
                "jp_core_cpi",
                "근원 CPI 전년비 (식품·에너지 제외)",
                OECD_CORE_CPI,
                JP_CORE_CPI_KEY,
                "JPN",
                "CPI",
                "M",
                PCT1,
            ),
            ecos("cn_cpi", "CPI 전년비", "902Y008", "M", "CN", PCT1, "yoy"),
        ),
    ),
    PanelSpec(
        "eco_labor",
        "economy",
        "실업·고용",
        (
            "실업률, 비농업고용, 시간당 임금, JOLTS 구인 건수 종합"
            "\n\n실업률은 경기 후행, 실업수당 청구 건수 및 신규 고용은 경기 선행 지표"
        ),
        (
            fred("us_unrate_eco", "실업률", "UNRATE", "M", PCT1),
            fred(
                "us_payrolls",
                "비농업 고용 증감",
                "PAYEMS",
                "M",
                THOUSANDS_SIGNED,
                "diff",
            ),
            fred(
                "us_claims",
                "신규 실업수당 청구",
                "ICSA",
                "W",
                THOUSANDS,
                scale=0.001,
            ),
            fred("us_wages", "평균시급 전년비", "CES0500000003", "M", PCT1, "yoy"),
            fred("us_jolts", "JOLTS 구인", "JTSJOL", "M", MILLIONS, scale=0.001),
            ecos(
                "kr_unrate", "실업률 (SA)", "901Y027", "M", "I61BC", PCT1, item2="I28B"
            ),
            ecos(
                "kr_employment",
                "취업자 전년비 (SA)",
                "901Y027",
                "M",
                "I61BA",
                PCT1_SIGNED,
                "yoy",
                item2="I28B",
            ),
            ecos(
                "kr_employment_rate",
                "고용률 (SA)",
                "901Y027",
                "M",
                "I61E",
                PCT1,
                item2="I28B",
            ),
            ecb(
                "ea_unrate",
                "실업률 (15–74세)",
                "LFSI/M.I10.S.UNEHRT.TOTAL0.15_74.T",
                "M",
                PCT1,
            ),
            ecb(
                "ea_employment",
                "고용 전년비",
                "MNA/Q.Y.I10.W2.S1.S1._Z.EMP._Z._T._Z.PS._Z.N",
                "Q",
                PCT1_SIGNED,
                "yoy",
            ),
            ecos("jp_unrate", "실업률 (SA)", "902Y021", "M", "JPN", PCT1),
            fred(
                "jp_wages",
                "제조업 시간당 임금 전년비",
                "LCEAMN01JPM659S",
                "M",
                PCT1_SIGNED,
            ),
        ),
    ),
    PanelSpec(
        "eco_pmi",
        "economy",
        "기업 심리",
        (
            "제조업·서비스업 기업 체감 경기 설문 지표"
            "\n\nOECD BCI 및 한국 BSI 기준선 100, 미국 지역 연은 지수 기준선 0"
        ),
        (
            *(
                oecd_measure(
                    f"{prefix}_bci",
                    "OECD 기업신뢰지수 BCI",
                    OECD_CLI,
                    CLI_KEY,
                    code,
                    "BCICP",
                    "M",
                    IDX1,
                )
                for prefix, code in (
                    ("us", "USA"),
                    ("kr", "KOR"),
                    ("ea", "EA20"),
                    ("jp", "JPN"),
                    ("cn", "CHN"),
                )
            ),
            fred(
                "us_philly_fed",
                "필라델피아 연은 제조업 지수",
                "GACDFSA066MSFRBPHI",
                "M",
                IDX1_SIGNED,
            ),
            fred(
                "us_empire",
                "엠파이어스테이트 제조업 지수",
                "GACDISA066MSFRBNY",
                "M",
                IDX1_SIGNED,
            ),
            ecos(
                "kr_bsi", "전산업 업황 BSI", "512Y013", "M", "99988", IDX0, item2="AA"
            ),
        ),
    ),
    PanelSpec(
        "eco_exports",
        "economy",
        "수출",
        ("주요국 수출 증가율 및 무역수지\n\n한국 반도체 수출은 전체 수출 선행 지표"),
        (
            fred(
                "us_exports",
                "수출 전년비 (재화, 국제수지 기준)",
                "BOPGEXP",
                "M",
                PCT1_SIGNED,
                "yoy",
            ),
            fred(
                "us_trade_balance",
                "무역수지 (재화·서비스)",
                "BOPGSTB",
                "M",
                USD_B1_SIGNED,
                scale=0.001,
            ),
            ecos(
                "kr_exports",
                "수출 전년비 (통관 기준)",
                "901Y118",
                "M",
                "T002",
                PCT1_SIGNED,
                "yoy",
            ),
            difference(
                "kr_trade_balance",
                "무역수지 (통관 기준)",
                "ECOS:901Y118/T002−T004",
                EcosSeries("901Y118", "M", "T002"),
                EcosSeries("901Y118", "M", "T004"),
                "M",
                USD_B1_SIGNED,
                scale=1e-6,
            ),
            ecos(
                "kr_chip_exports",
                "반도체 수출금액지수 전년비",
                "403Y001",
                "M",
                "3091AA",
                PCT1_SIGNED,
                "yoy",
            ),
            ecb(
                "ea_exports",
                "수출 전년비 (역외, 총수출)",
                "TRD/M.I10.N.X.TTT.J10.4.VAL",
                "M",
                PCT1_SIGNED,
                "yoy",
            ),
            difference(
                "ea_trade_balance",
                "무역수지 (역외)",
                "ECB:TRD/M.I10.N.X−M.TTT.J10.4.VAL",
                ecb_fetch("TRD/M.I10.N.X.TTT.J10.4.VAL"),
                ecb_fetch("TRD/M.I10.N.M.TTT.J10.4.VAL"),
                "M",
                EUR_B1_SIGNED,
                scale=1e-6,
            ),
            exports("jp_exports", "JPN"),
            fred_trade_balance("jp_trade_balance", "JPN"),
            exports("cn_exports", "CHN"),
            fred(
                "cn_imports",
                "수입 전년비 (달러 기준)",
                TRADE_FRED["CHN"][1],
                "M",
                PCT1_SIGNED,
                "yoy",
            ),
            fred_trade_balance("cn_trade_balance", "CHN"),
        ),
    ),
    PanelSpec(
        "eco_retail",
        "economy",
        "소매판매",
        (
            "민간 소비 및 소매판매 증가율"
            "\n\n실질/명목 및 전년비(YoY)/전월비(MoM) 구분 비교"
        ),
        (
            fred(
                "us_retail", "소매판매 전년비 (실질)", "RRSFS", "M", PCT1_SIGNED, "yoy"
            ),
            fred(
                "us_retail_mom",
                "소매판매 전월비 (명목)",
                "RSAFS",
                "M",
                PCT1_SIGNED,
                "pct",
            ),
            ecos(
                "kr_retail",
                "소매판매 전년비 (실질)",
                "901Y100",
                "M",
                "G0",
                PCT1_SIGNED,
                "yoy",
                item2="T2",
            ),
            ecb(
                "ea_retail",
                "소매판매 전년비 (실질, 자동차·연료 제외)",
                "STBS/M.I10.Y.TOVV.NS4701.4D0.N.IX",
                "M",
                PCT1_SIGNED,
                "yoy",
            ),
            fred(
                "jp_retail",
                "소매판매 전년비 (실질)",
                "JPNSLRTTO01GYSAM",
                "M",
                PCT1_SIGNED,
            ),
        ),
    ),
    PanelSpec(
        "eco_production",
        "economy",
        "산업생산",
        ("산업생산 증가율 및 설비가동률\n\n설비가동률 상승 시 생산 여력 축소"),
        (
            fred("us_indpro", "산업생산 전년비", "INDPRO", "M", PCT1_SIGNED, "yoy"),
            fred("us_capacity", "설비가동률", "TCU", "M", PCT1),
            ecos(
                "kr_ip",
                "산업생산 전년비 (광공업, SA)",
                "901Y033",
                "M",
                "AB00",
                PCT1_SIGNED,
                "yoy",
                item2="2",
            ),
            ecos("kr_capacity", "제조업 평균가동률", "901Y025", "M", "I31A", PCT1),
            ecb(
                "ea_ip",
                "산업생산 전년비 (건설 제외)",
                "STBS/M.I10.W.PROD.NS0020.4D0.GY.IX",
                "M",
                PCT1_SIGNED,
            ),
            fred(
                "jp_ip",
                "산업생산 전년비 (건설 제외)",
                "JPNPRINTO01GYSAM",
                "M",
                PCT1_SIGNED,
            ),
        ),
    ),
    PanelSpec(
        "eco_capex",
        "economy",
        "소비·설비투자",
        (
            "실질 소비·설비투자 증가율 및 주택 지표"
            "\n\n핵심 자본재 수주는 설비투자, 주택착공 건수는 주거투자 선행 지표"
        ),
        (
            fred(
                "us_pce_real",
                "민간소비 실질 전년비 (PCE)",
                "PCEC96",
                "M",
                PCT1_SIGNED,
                "yoy",
            ),
            fred(
                "us_capex",
                "실질 비주거 고정투자 전년비",
                "PNFIC1",
                "Q",
                PCT1_SIGNED,
                "yoy",
            ),
            fred(
                "us_core_capex_orders",
                "핵심 자본재 수주 전년비",
                "NEWORDER",
                "M",
                PCT1_SIGNED,
                "yoy",
            ),
            fred("us_housing_starts", "주택착공 (연율)", "HOUST", "M", THOUSANDS),
            ecos(
                "kr_consumption",
                "민간소비 실질 전년비",
                "200Y102",
                "Q",
                "10222",
                PCT1_SIGNED,
            ),
            ecos(
                "kr_capex",
                "설비투자 실질 전년비",
                "200Y102",
                "Q",
                "10223",
                PCT1_SIGNED,
            ),
            ecos(
                "kr_construction",
                "건설투자 실질 전년비",
                "200Y102",
                "Q",
                "10224",
                PCT1_SIGNED,
            ),
            ecb(
                "ea_consumption",
                "민간소비 실질 전년비",
                "MNA/Q.Y.I10.W0.S1M.S1.D.P31._Z._Z._T.EUR.LR.GY",
                "Q",
                PCT1_SIGNED,
            ),
            ecb(
                "ea_gfcf",
                "총고정자본형성 실질 전년비",
                "MNA/Q.Y.I10.W0.S1.S1.D.P51G.N11G._T._Z.EUR.LR.GY",
                "Q",
                PCT1_SIGNED,
            ),
            fred(
                "jp_consumption",
                "민간소비 실질 전년비",
                "JPNRGDPPC",
                "Q",
                PCT1_SIGNED,
                "yoy",
            ),
            fred(
                "jp_gfcf",
                "총고정자본형성 실질 전년비",
                "NFIRSAXDCJPQ",
                "Q",
                PCT1_SIGNED,
                "yoy",
            ),
            *bis_by_area(
                "house_price",
                "실질 주택가격 전년비",
                "WS_SPP",
                "Q.{area}.R.771",
                "Q",
                PCT1_SIGNED,
            ),
        ),
    ),
    PanelSpec(
        "eco_household_debt",
        "economy",
        "가계부채",
        (
            "가계부채 레버리지 및 원리금 상환 부담 지표"
            "\n\nGDP 대비 가계신용 비율은 부채 규모, "
            "DSR 및 대출 연체율은 채무 상환 부담 척도"
        ),
        (
            *bis_by_area(
                "hh_debt",
                "가계신용/GDP",
                "WS_TC",
                "Q.{area}.H.A.M.770.A",
                "Q",
                PCT1,
            ),
            fred("us_hh_dsr", "가계 부채상환비율 DSR", "TDSP", "Q", PCT1),
            fred("us_card_delinquency", "신용카드 연체율", "DRCCLACBS", "Q", PCT2),
            bis(
                "kr_hh_dsr", "가계 부채상환비율 DSR", "WS_DSR", "Q.KR.H.367", "Q", PCT1
            ),
            bis(
                "jp_hh_dsr", "가계 부채상환비율 DSR", "WS_DSR", "Q.JP.H.367", "Q", PCT1
            ),
            *bis_by_area(
                "credit_gap",
                "민간 신용/GDP 갭",
                "WS_CREDIT_GAP",
                "Q.{area}.P.A.C.770",
                "Q",
                PP1_SIGNED,
            ),
            ecos(
                "kr_hh_credit",
                "예금취급기관 가계대출 전년비",
                "151Y005",
                "M",
                "1110000",
                PCT1_SIGNED,
                "yoy",
            ),
            ecb(
                "ea_hh_loans",
                "가계대출 전년비",
                "BSI/M.U2.Y.U.A20.A.I.U2.2250.Z01.A",
                "M",
                PCT1_SIGNED,
            ),
        ),
    ),
    PanelSpec(
        "eco_corporate_debt",
        "economy",
        "기업부채",
        (
            "비금융 민간기업 부채 레버리지 및 기업대출 추이"
            "\n\nGDP 대비 기업신용 비율은 부채 규모, "
            "상업 대출 연체율은 신용 리스크 척도"
        ),
        (
            *bis_by_area(
                "corp_debt",
                "비금융기업 신용/GDP",
                "WS_TC",
                "Q.{area}.N.A.M.770.A",
                "Q",
                PCT1,
            ),
            fred("us_ci_delinquency", "기업대출 연체율 (C&I)", "DRBLACBS", "Q", PCT2),
            ecos(
                "kr_corp_loans",
                "예금은행 기업대출 전년비",
                "141Y004",
                "M",
                "1A0000",
                PCT1_SIGNED,
                "yoy",
                item2="X00",
            ),
            ecb(
                "ea_corp_loans",
                "기업대출 전년비",
                "BSI/M.U2.Y.U.A20.A.I.U2.2240.Z01.A",
                "M",
                PCT1_SIGNED,
            ),
        ),
    ),
    PanelSpec(
        "eco_money",
        "economy",
        "통화량 (M2 등)",
        (
            "광의통화(M2) 및 민간 은행 대출 증가율"
            "\n\n통화 증가율 가속 시 유동성 확장, 음(-) 전환 시 유동성 축소"
        ),
        (
            fred("us_m2", "광의통화 전년비 (M2)", "WM2NS", "W", PCT1, "yoy"),
            fred(
                "us_bank_loans",
                "은행 대출·리스 전년비",
                "TOTLL",
                "W",
                PCT1_SIGNED,
                "yoy",
            ),
            ecos(
                "kr_m2",
                "광의통화 전년비 (M2 평잔, SA)",
                "161Y005",
                "M",
                "BBHS00",
                PCT1,
                "yoy",
            ),
            ecos(
                "kr_bank_loans",
                "예금은행 대출금 전년비",
                "104Y016",
                "M",
                "BDCA1",
                PCT1_SIGNED,
                "yoy",
            ),
            ecb(
                "ea_m3",
                "광의통화 전년비 (M3)",
                "BSI/M.U2.Y.V.M30.X.I.U2.2300.Z01.A",
                "M",
                PCT1,
            ),
            oecd(
                "jp_money",
                "광의통화 전년비",
                OECD_KEI,
                KEI_KEY,
                "M",
                PCT1,
                match=(("REF_AREA", "JPN"), ("FREQ", "M"), ("MEASURE", "MABM")),
            ),
            oecd(
                "cn_m2",
                "광의통화 전년비 (M2)",
                OECD_MONAGG,
                "CHN.M.MABM.XDC._Z.N._Z._Z.N",
                "M",
                PCT1,
                "yoy",
            ),
        ),
    ),
    PanelSpec(
        "mp_fed",
        "monetary",
        "Fed",
        "",
        (
            bis_policy_rate("us_policy_rate", "정책금리 (Fed)", "US"),
            fred("us_fed_assets_mp", "연준 총자산", "WALCL", "W", USD_T, scale=1e-6),
        ),
    ),
    PanelSpec(
        "mp_ecb",
        "monetary",
        "ECB",
        "",
        (
            bis_policy_rate("ea_policy_rate", "정책금리 (ECB)", "XM"),
            ecb(
                "ea_ecb_assets",
                "유로시스템 총자산",
                "ILM/W.U2.C.T000000.Z5.Z01",
                "W",
                EUR_T,
                scale=1e-6,
            ),
        ),
    ),
    PanelSpec(
        "mp_boj",
        "monetary",
        "BOJ",
        "",
        (
            bis_policy_rate("jp_policy_rate", "정책금리 (BOJ)", "JP"),
            fred(
                "jp_boj_assets", "일본은행 총자산", "JPNASSETS", "M", JPY_T, scale=1e-4
            ),
        ),
    ),
    PanelSpec(
        "mp_pboc",
        "monetary",
        "PBOC",
        "",
        (bis_policy_rate("cn_policy_rate", "정책금리 (PBOC)", "CN"),),
    ),
    PanelSpec(
        "mp_bok",
        "monetary",
        "한국은행",
        "",
        (
            bis_policy_rate("kr_policy_rate", "정책금리 (한국은행)", "KR"),
            ecos(
                "kr_bok_base",
                "본원통화 (평잔, SA)",
                "102Y004",
                "M",
                "ABA1",
                KRW_T,
                scale=1e-3,
            ),
        ),
    ),
    PanelSpec(
        "mp_rates",
        "monetary",
        "장기금리",
        (
            "주요국 10년 만기 국채 벤치마크 금리"
            "\n\n유로존은 최고등급(AAA) 중앙정부채 기준, 일본·중국은 월간 값"
        ),
        (
            fred("us_long_rate", "장기금리 (10년 국채)", "DGS10", "D", PCT2),
            ecos(
                "kr_long_rate",
                "장기금리 (10년 국고채)",
                "817Y002",
                "D",
                "010210000",
                PCT2,
            ),
            ecb(
                "ea_long_rate",
                "장기금리 (10년 AAA 국채)",
                f"{YC_FLOW}{ECB_LONG_TENOR}",
                "D",
                PCT2,
            ),
            ecos(
                "jp_long_rate",
                "장기금리 (10년 국채)",
                "902Y023",
                "M",
                "IRLT",
                PCT2,
                item2="JPN",
            ),
            ecos(
                "cn_long_rate",
                "장기금리 (10년 국채)",
                "902Y023",
                "M",
                "IRLT",
                PCT2,
                item2="CHN",
            ),
        ),
    ),
    PanelSpec(
        "fin_credit",
        "finance",
        "신용",
        (
            "은행 대출 태도 및 회사채 신용스프레드"
            "\n\n대출 태도 양수(+) 시 대출 기준 강화(신용 경색), "
            "신용스프레드 확대 시 부도 위험 가중"
        ),
        (
            fred(
                "us_sloos",
                "C&I 대출기준 강화 순비율",
                "DRTSCILM",
                "Q",
                PCT1_SIGNED,
            ),
            ice_oas("us_hy_oas", "BAMLH0A0HYM2"),
            ecos(
                "kr_lending_attitude",
                "대출태도지수 (국내은행 종합, +강화)",
                "514Y001",
                "Q",
                "AA",
                IDX0_SIGNED,
                scale=-1.0,
            ),
            difference(
                "kr_corp_spread",
                "회사채 스프레드 (AA− 3년 − 국고채 3년)",
                "ECOS:817Y002/010300000−010200000",
                EcosSeries("817Y002", "D", "010300000"),
                EcosSeries("817Y002", "D", "010200000"),
                "D",
                PP2,
            ),
            ecb(
                "ea_bls",
                "기업대출 신용기준 순비율",
                "BLS/Q.U2.ALL.O.E.Z.B3.ST.S.WFNET",
                "Q",
                PCT1_SIGNED,
            ),
            ice_oas("ea_hy_oas", "BAMLHE00EHYIOAS"),
        ),
    ),
    PanelSpec(
        "fin_fx",
        "finance",
        "환율",
        (
            "연준 명목 달러 광의지수 및 주요국 대미 환율"
            "\n\n지표값 상승 시 미 달러화 대비 해당 통화 가치 약세(달러 강세)"
        ),
        (
            price(fred("us_dollar", "달러 광의지수", "DTWEXBGS", "D", IDX1)),
            price(ecos("kr_usdkrw", "원/달러", "731Y001", "D", "0000001", KRW)),
            price(
                ecb(
                    "ea_usdeur",
                    "USD/EUR (역수)",
                    "EXR/D.USD.EUR.SP00.A",
                    "D",
                    FX4,
                    "inverse",
                )
            ),
            price(fred("jp_usdjpy", "USD/JPY", "DEXJPUS", "D", IDX2)),
            price(fred("cn_usdcny", "USD/CNY", "DEXCHUS", "D", FX4)),
        ),
    ),
    PanelSpec(
        "fin_commodities",
        "finance",
        "원자재",
        "글로벌 에너지(원유·천연가스) 및 주요 금속(구리·금) 일간 가격 추이",
        (
            price(fred("gl_wti", "WTI", "DCOILWTICO", "D", USD_BBL)),
            price(fred("gl_brent", "브렌트", "DCOILBRENTEU", "D", USD_BBL)),
            price(fred("us_henryhub", "헨리허브 천연가스", "DHHNGSP", "D", USD_MMBTU)),
            yahoo_price("gl_copper", "구리 (COMEX 선물)", "HG=F", USD_LB),
            yahoo_price("gl_gold", "금 (COMEX 선물)", "GC=F", USD_OZ),
        ),
    ),
    PanelSpec(
        "fin_volatility",
        "finance",
        "변동성·금융여건",
        (
            "금융시장 내재변동성 및 금융안정상황 지표"
            "\n\nVIX 변동성 경계선 20pt/30pt, "
            "시카고 연은 NFCI 0 초과 시 역사적 평균 대비 금융 긴축"
        ),
        (
            fred("us_vix_fin", "VIX", "VIXCLS", "D", IDX1, copyright_="Cboe"),
            fred("us_nfci", "시카고 연은 금융여건지수 NFCI", "NFCI", "W", IDX2_SIGNED),
            ecb(
                "ea_ciss",
                "금융 스트레스 지수 CISS",
                "CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX",
                "D",
                IDX2,
            ),
        ),
    ),
    PanelSpec(
        "fin_sentiment",
        "finance",
        "소비자심리",
        (
            "민간 가계 체감 경기 및 소비자심리지수(CCI)"
            "\n\n한국 CCSI 및 OECD 소비자신뢰지수 기준선 100(장기 평균선)"
        ),
        (
            fred(
                "us_umich",
                "미시간대 소비자심리",
                "UMCSENT",
                "M",
                IDX1,
                copyright_="UMich",
            ),
            ecos(
                "kr_ccsi",
                "소비자심리지수 CCSI",
                "511Y002",
                "M",
                "FME",
                IDX1,
                item2="99988",
            ),
            *(
                oecd_measure(
                    f"{prefix}_cci",
                    "OECD 소비자신뢰지수 CCI",
                    OECD_CLI,
                    CLI_KEY,
                    code,
                    "CCICP",
                    "M",
                    IDX1,
                )
                for prefix, code in (("ea", "EA20"), ("jp", "JPN"), ("cn", "CHN"))
            ),
        ),
    ),
    PanelSpec(
        "fin_sectors",
        "finance",
        "업종 ETF",
        ETF_NOTE,
        (
            etf("us_xlk", "기술", "XLK"),
            etf("us_xlf", "금융", "XLF"),
            etf("us_xlv", "헬스케어", "XLV"),
            etf("us_xly", "임의소비재", "XLY"),
            etf("us_xlp", "필수소비재", "XLP"),
            etf("us_xle", "에너지", "XLE"),
            etf("us_xli", "산업재", "XLI"),
            etf("us_xlb", "소재", "XLB"),
            etf("us_xlu", "유틸리티", "XLU"),
            etf("us_xlre", "부동산", "XLRE"),
            etf("us_xlc", "통신서비스", "XLC"),
        ),
    ),
    PanelSpec(
        "fin_etfs",
        "finance",
        "자산·국가·테마 ETF",
        ETF_NOTE,
        (
            etf("gl_gld", "금", "GLD"),
            etf("gl_slv", "은", "SLV"),
            etf("gl_cper", "구리", "CPER"),
            etf("gl_dba", "농산물", "DBA"),
            etf("gl_tlt", "미국 장기채", "TLT"),
            etf("gl_tip", "물가연동채", "TIP"),
            etf("gl_hyg", "하이일드 회사채", "HYG"),
            etf("gl_lqd", "투자등급 회사채", "LQD"),
            etf("gl_qqq", "나스닥 100", "QQQ"),
            etf("gl_iwm", "미국 소형주", "IWM"),
            etf("gl_rsp", "S&P 500 동일가중", "RSP"),
            etf("gl_iwf", "미국 대형 성장주", "IWF"),
            etf("gl_iwd", "미국 대형 가치주", "IWD"),
            etf("gl_eem", "신흥국", "EEM"),
            etf("gl_ewy", "한국", "EWY"),
            etf("gl_ewj", "일본", "EWJ"),
            etf("gl_ezu", "유로존", "EZU"),
            etf("gl_fxi", "중국 대형주", "FXI"),
            etf("gl_ita", "방산", "ITA"),
            etf("gl_smh", "반도체", "SMH"),
            etf("gl_xop", "석유·가스 생산", "XOP"),
        ),
    ),
)

BY_ID: Final = {i.id: i for spec in PANELS for i in spec.indicators}


def indicator_fetches(indicators: Iterable[Indicator]) -> tuple[Fetch, ...]:
    seen: dict[Fetch, None] = {}
    for indicator in indicators:
        fetch = indicator.fetch
        for part in fetch.parts() if isinstance(fetch, Difference) else (fetch,):
            seen[part] = None
    return tuple(seen)


TAPE_GROUPS: Final = (
    ("주가지수", ("us_sp500", "kr_kospi", "jp_nikkei")),
    ("변동성", ("us_vix",)),
    ("환율", ("kr_usdkrw_ov", "us_dollar_ov")),
)

CURVE_YEARS: Final = 2
RISK_FREE: Final = FredSeries("DGS3MO")
LONG_FETCHES: Final = frozenset({*indicator_fetches(BY_ID.values()), RISK_FREE})


def curve_fetch(fetch: Fetch) -> Fetch:
    return fetch if fetch in LONG_FETCHES else replace(fetch, years=CURVE_YEARS)


CURVES: Final = (
    CurveSpec(
        "US",
        "미국 국채",
        tuple(
            (tenor, curve_fetch(FredSeries(series_id)))
            for tenor, series_id in (
                ("1M", "DGS1MO"),
                ("3M", "DGS3MO"),
                ("6M", "DGS6MO"),
                ("1Y", "DGS1"),
                ("2Y", "DGS2"),
                ("3Y", "DGS3"),
                ("5Y", "DGS5"),
                ("7Y", "DGS7"),
                ("10Y", "DGS10"),
                ("20Y", "DGS20"),
                ("30Y", "DGS30"),
            )
        ),
    ),
    CurveSpec(
        "KR",
        "한국 국고채",
        tuple(
            (tenor, curve_fetch(EcosSeries("817Y002", "D", item)))
            for tenor, item in (
                ("1Y", "010190000"),
                ("2Y", "010195000"),
                ("3Y", "010200000"),
                ("5Y", "010200001"),
                ("10Y", "010210000"),
                ("20Y", "010220000"),
                ("30Y", "010230000"),
                ("50Y", "010240000"),
            )
        ),
    ),
    CurveSpec(
        "EA",
        "유로존 AAA 국채",
        tuple(
            (tenor, curve_fetch(ecb_fetch(f"{YC_FLOW}{tenor}"))) for tenor in ECB_TENORS
        ),
    ),
)
SPREAD_SERIES: Final = FredSeries("T10Y2Y")
RECESSION_SERIES: Final = FredSeries("USREC")
CONTEXT_FETCHES: Final = tuple(
    dict.fromkeys(
        [
            *(fetch for curve in CURVES for _, fetch in curve.tenors),
            SPREAD_SERIES,
            RECESSION_SERIES,
        ]
    )
)

CALENDAR_RELEASES: Final = (
    ("Gross Domestic Product", "GDP", "BEA"),
    ("GDPNow", "GDPNow", "Atlanta Fed"),
    ("Consumer Price Index", "소비자물가 CPI", "BLS"),
    ("Producer Price Index", "생산자물가 PPI", "BLS"),
    ("Personal Income and Outlays", "개인소득·지출 (PCE)", "BEA"),
    ("Employment Situation", "고용보고서", "BLS"),
    ("Job Openings and Labor Turnover", "JOLTS 구인", "BLS"),
    ("Unemployment Insurance Weekly Claims", "신규 실업수당 청구", "DOL"),
    ("Advance Monthly Sales for Retail", "소매판매", "Census"),
    ("U.S. International Trade in Goods and Services", "수출입", "Census·BEA"),
    ("Industrial Production and Capacity Utilization", "산업생산", "FRB"),
    ("H.4.1", "연준 대차대조표 H.4.1", "FRB"),
    ("Senior Loan Officer", "대출태도 서베이 SLOOS", "FRB"),
    ("Surveys of Consumers", "미시간대 소비자심리", "UMich"),
    ("Chicago Fed National Financial Conditions", "NFCI", "Chicago Fed"),
)
CALENDAR_DAYS: Final = 7


@dataclass(frozen=True, slots=True)
class SepRow:
    label: str
    series_id: str
    longer_run_id: str | None
    current: str
    current_note: str
    current_dec: int = 1


SEP_ROWS: Final = (
    SepRow(
        "연방기금금리",
        "FEDTARMD",
        "FEDTARMDLR",
        "us_policy_rate_bis",
        "BIS 정책금리",
        current_dec=2,
    ),
    SepRow("실질 GDP 성장률", "GDPC1MD", "GDPC1MDLR", "us_gdp_yoy", "최근 분기 YoY"),
    SepRow("실업률", "UNRATEMD", "UNRATEMDLR", "us_unrate", "최근 월"),
    SepRow("PCE 물가", "PCECTPIMD", "PCECTPIMDLR", "us_pce", "최근 월 YoY"),
    SepRow("근원 PCE 물가", "JCXFEMD", None, "us_core_pce", "최근 월 YoY"),
)
SEP_LOOKBACK_DAYS: Final = 400
SEP_MAX_YEARS: Final = 4
