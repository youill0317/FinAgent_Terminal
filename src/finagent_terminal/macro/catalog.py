"""What the dashboard shows. Edit this file to add, remove or re-order indicators.

Every entry is a public or open series. Analyst consensus, estimates,
investment opinions, sentiment scores and forecasts are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from finagent_terminal.macro.models import Fmt
from finagent_terminal.macro.sources import (
    EcosSeries,
    FiscalSeries,
    FredSeries,
    NyFedRate,
    Ratio,
    SdmxSeries,
)

if TYPE_CHECKING:
    from finagent_terminal.macro.models import Freq
    from finagent_terminal.macro.sources import Fetch
    from finagent_terminal.macro.transforms import Transform


@dataclass(frozen=True, slots=True)
class Indicator:
    id: str
    label: str
    code: str
    fetch: Fetch | Ratio
    freq: Freq
    fmt: Fmt
    transform: Transform = "level"
    scale: float = 1.0
    copyright: str | None = None


@dataclass(frozen=True, slots=True)
class PanelSpec:
    id: str
    key: str
    title: str
    note: str
    indicators: tuple[Indicator, ...]


PCT1: Final = Fmt(suf="%", dec=1)
PCT2: Final = Fmt(suf="%", dec=2)
PCT1_SIGNED: Final = Fmt(suf="%", dec=1, signed=True)
PP2_SIGNED: Final = Fmt(suf="%p", dec=2, signed=True)


def fred(  # noqa: PLR0913 - one helper call per catalog row
    id_: str,
    label: str,
    series_id: str,
    freq: Freq,
    fmt: Fmt,
    transform: Transform = "level",
    *,
    scale: float = 1.0,
    copyright_: str | None = None,
    years: int = 4,
) -> Indicator:
    return Indicator(
        id=id_,
        label=label,
        code=f"FRED:{series_id}",
        fetch=FredSeries(series_id, years),
        freq=freq,
        fmt=fmt,
        transform=transform,
        scale=scale,
        copyright=copyright_,
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
) -> Indicator:
    code = f"ECOS:{stat}/{item1}" + (f"/{item2}" if item2 else "")
    return Indicator(
        id=id_,
        label=label,
        code=code,
        fetch=EcosSeries(stat, cycle, item1, item2),
        freq=cycle,
        fmt=fmt,
        transform=transform,
    )


def bis_policy_rate(id_: str, label: str, area: str) -> Indicator:
    return Indicator(
        id=id_,
        label=label,
        code=f"BIS:WS_CBPOL/{area}",
        fetch=SdmxSeries("BIS", f"WS_CBPOL/M.{area}/all"),
        freq="M",
        fmt=PCT2,
    )


OECD_CLI_FLOW: Final = "OECD.SDD.STES,DSD_STES@DF_CLI/USA+CHN+KOR.M.LI...AA...H"


def oecd_cli(id_: str, label: str, area: str) -> Indicator:
    return Indicator(
        id=id_,
        label=label,
        code=f"OECD:CLI/{area}",
        fetch=SdmxSeries("OECD", OECD_CLI_FLOW, (("REF_AREA", area),)),
        freq="M",
        fmt=Fmt(dec=1),
    )


def ecb_fx(currency: str) -> SdmxSeries:
    return SdmxSeries("ECB", f"EXR/D.{currency}.EUR.SP00.A")


PANELS: Final = (
    PanelSpec(
        "growth",
        "GDP",
        "미국 성장",
        "BEA · Census · FRB · Chicago Fed",
        (
            fred("us_gdp", "실질 GDP (전기비 연율)", "A191RL1Q225SBEA", "Q", PCT1),
            fred("us_indpro", "산업생산 YoY", "INDPRO", "M", PCT1, "yoy"),
            fred("us_retail", "소매판매 MoM", "RSAFS", "M", PCT1_SIGNED, "pct"),
            fred("us_pce_real", "실질 개인소비 MoM", "PCEC96", "M", PCT1_SIGNED, "pct"),
            fred("us_houst", "주택착공 (연율, 천호)", "HOUST", "M", Fmt(dec=0)),
            fred(
                "us_cfnai",
                "시카고 연은 활동지수 3MA",
                "CFNAIMA3",
                "M",
                Fmt(dec=2, signed=True),
            ),
        ),
    ),
    PanelSpec(
        "labor",
        "LABR",
        "미국 고용",
        "BLS · DOL",
        (
            fred(
                "us_payrolls",
                "비농업 고용 증감",
                "PAYEMS",
                "M",
                Fmt(suf="K", dec=0, signed=True),
                "diff",
            ),
            fred("us_unrate", "실업률", "UNRATE", "M", PCT1),
            fred("us_sahm", "삼 룰 지표", "SAHMREALTIME", "M", Fmt(suf="%p", dec=2)),
            fred(
                "us_claims",
                "신규 실업수당 청구",
                "ICSA",
                "W",
                Fmt(suf="K", dec=0),
                scale=0.001,
            ),
            fred(
                "us_ccsa",
                "계속 실업수당 청구",
                "CCSA",
                "W",
                Fmt(suf="K", dec=0),
                scale=0.001,
            ),
            fred(
                "us_jolts",
                "JOLTS 구인건수",
                "JTSJOL",
                "M",
                Fmt(suf="M", dec=2),
                scale=0.001,
            ),
            fred("us_ahe", "시간당 임금 YoY", "CES0500000003", "M", PCT1, "yoy"),
        ),
    ),
    PanelSpec(
        "inflation",
        "INFL",
        "미국 물가",
        "BLS · BEA · FRB",
        (
            fred("us_cpi", "CPI YoY", "CPIAUCSL", "M", PCT1, "yoy"),
            fred("us_core_cpi", "근원 CPI YoY", "CPILFESL", "M", PCT1, "yoy"),
            fred("us_core_pce", "근원 PCE YoY", "PCEPILFE", "M", PCT1, "yoy"),
            fred("us_ppi", "PPI 최종수요 YoY", "PPIFIS", "M", PCT1, "yoy"),
            fred("us_bei5", "5년 BEI", "T5YIE", "D", PCT2),
            fred("us_5y5y", "5년 후 5년 기대인플레", "T5YIFR", "D", PCT2),
        ),
    ),
    PanelSpec(
        "rates",
        "RATE",
        "정책금리 · 국채",
        "FRB · NY Fed · Treasury",
        (
            fred("us_fftu", "연방기금 목표 상단", "DFEDTARU", "D", PCT2),
            Indicator(
                "us_effr",
                "실효 연방기금금리",
                "NYFED:EFFR",
                NyFedRate("unsecured/effr"),
                "D",
                PCT2,
            ),
            Indicator(
                "us_sofr", "SOFR", "NYFED:SOFR", NyFedRate("secured/sofr"), "D", PCT2
            ),
            fred("us_2y", "2년물 국채", "DGS2", "D", PCT2),
            fred("us_10y", "10년물 국채", "DGS10", "D", PCT2),
            fred("us_10y3m", "10년 − 3개월", "T10Y3M", "D", PP2_SIGNED),
            fred("us_10y2y", "10년 − 2년", "T10Y2Y", "D", PP2_SIGNED, years=11),
            fred("us_real10", "10년 실질금리 (TIPS)", "DFII10", "D", PCT2),
            fred(
                "us_mort30",
                "30년 모기지 금리",
                "MORTGAGE30US",
                "W",
                PCT2,
                copyright_="Freddie Mac",
            ),
        ),
    ),
    PanelSpec(
        "fincon",
        "FCI",
        "금융여건",
        "Chicago Fed · ICE · Moody's · Cboe · FRB",
        (
            fred(
                "us_nfci", "NFCI (0 미만 = 완화)", "NFCI", "W", Fmt(dec=2, signed=True)
            ),
            fred(
                "us_hy_oas",
                "하이일드 OAS",
                "BAMLH0A0HYM2",
                "D",
                Fmt(suf="%p", dec=2),
                copyright_="ICE · 3년치",
                years=3,
            ),
            fred(
                "us_baa10y",
                "Baa − 10년물",
                "BAA10Y",
                "D",
                Fmt(suf="%p", dec=2),
                copyright_="Moody's",
            ),
            fred("us_vix", "VIX", "VIXCLS", "D", Fmt(dec=1), copyright_="Cboe"),
            fred("us_dollar", "달러 지수 (광의)", "DTWEXBGS", "D", Fmt(dec=1)),
        ),
    ),
    PanelSpec(
        "liquidity",
        "LIQ",
        "유동성 · 재정",
        "FRB · Treasury Fiscal Data",
        (
            fred(
                "us_walcl",
                "연준 총자산",
                "WALCL",
                "W",
                Fmt(pre="$", suf="T", dec=2),
                scale=1e-6,
            ),
            fred(
                "us_reserves",
                "지급준비금",
                "WRESBAL",
                "W",
                Fmt(pre="$", suf="T", dec=2),
                scale=0.001,
            ),
            fred(
                "us_rrp", "역레포 잔액", "RRPONTSYD", "D", Fmt(pre="$", suf="B", dec=0)
            ),
            fred(
                "us_tga",
                "재무부 일반계정 TGA",
                "WTREGEN",
                "W",
                Fmt(pre="$", suf="B", dec=0),
            ),
            fred("us_m2", "M2 YoY", "M2SL", "M", PCT1, "yoy"),
            Indicator(
                "us_debt",
                "연방정부 총부채",
                "TREASURY:debt_to_penny",
                FiscalSeries("v2/accounting/od/debt_to_penny", "tot_pub_debt_out_amt"),
                "D",
                Fmt(pre="$", suf="T", dec=2),
                scale=1e-12,
            ),
            fred(
                "us_deficit12",
                "재정수지 12개월 합계",
                "MTSDS133FMS",
                "M",
                Fmt(pre="$", suf="T", dec=2),
                "sum12",
                scale=1e-6,
            ),
            Indicator(
                "us_avg_rate",
                "국채 평균 조달금리",
                "TREASURY:avg_interest_rates",
                FiscalSeries(
                    "v2/accounting/od/avg_interest_rates",
                    "avg_interest_rate_amt",
                    "security_desc:eq:Total Marketable",
                ),
                "M",
                PCT2,
            ),
        ),
    ),
    PanelSpec(
        "korea",
        "KR",
        "한국",
        "한국은행 ECOS · OECD",
        (
            ecos("kr_base", "한국은행 기준금리", "722Y001", "D", "0101000", PCT2),
            ecos("kr_ktb3", "국고채 3년", "817Y002", "D", "010200000", PCT2),
            ecos("kr_ktb10", "국고채 10년", "817Y002", "D", "010210000", PCT2),
            ecos(
                "kr_usdkrw",
                "원/달러 매매기준율",
                "731Y001",
                "D",
                "0000001",
                Fmt(suf="원", dec=1),
            ),
            ecos("kr_kospi", "KOSPI", "802Y001", "D", "0001000", Fmt(dec=2)),
            ecos("kr_cpi", "소비자물가 YoY", "901Y009", "M", "0", PCT1, "yoy"),
            ecos(
                "kr_exports",
                "수출 (통관) YoY",
                "901Y118",
                "M",
                "T002",
                PCT1_SIGNED,
                "yoy",
            ),
            ecos(
                "kr_unrate", "실업률 (SA)", "901Y027", "M", "I61BC", PCT1, item2="I28B"
            ),
            ecos(
                "kr_coincident",
                "동행지수 순환변동치",
                "901Y067",
                "M",
                "I16D",
                Fmt(dec=1),
            ),
            ecos(
                "kr_csi",
                "소비자심리지수 CSI",
                "511Y002",
                "M",
                "FME",
                Fmt(dec=1),
                item2="99988",
            ),
            oecd_cli("kr_cli", "OECD 경기선행지수 한국", "KOR"),
        ),
    ),
    PanelSpec(
        "global",
        "GLBL",
        "글로벌",
        "ECB · BIS · OECD · EIA · IMF",
        (
            Indicator(
                "ecb_dfr",
                "ECB 예금금리",
                "ECB:FM/DFR",
                SdmxSeries("ECB", "FM/B.U2.EUR.4F.KR.DFR.LEV"),
                "D",
                PCT2,
            ),
            bis_policy_rate("boj_rate", "일본은행 정책금리", "JP"),
            bis_policy_rate("boe_rate", "영란은행 정책금리", "GB"),
            Indicator(
                "ea_hicp",
                "유로존 HICP YoY",
                "ECB:ICP/U2/ANR",
                SdmxSeries("ECB", "ICP/M.U2.N.000000.4.ANR"),
                "M",
                PCT1,
            ),
            oecd_cli("us_cli", "OECD 경기선행지수 미국", "USA"),
            oecd_cli("cn_cli", "OECD 경기선행지수 중국", "CHN"),
            Indicator(
                "eurusd", "EUR/USD", "ECB:EXR/USD", ecb_fx("USD"), "D", Fmt(dec=4)
            ),
            Indicator(
                "usdjpy",
                "USD/JPY",
                "ECB:EXR/JPY÷USD",
                Ratio(ecb_fx("JPY"), ecb_fx("USD")),
                "D",
                Fmt(dec=2),
            ),
            fred("wti", "WTI ($/bbl)", "DCOILWTICO", "D", Fmt(pre="$", dec=2)),
            fred("brent", "브렌트 ($/bbl)", "DCOILBRENTEU", "D", Fmt(pre="$", dec=2)),
            fred(
                "henryhub",
                "헨리허브 가스 ($/MMBtu)",
                "DHHNGSP",
                "D",
                Fmt(pre="$", dec=2),
            ),
            fred("copper", "구리 ($/t)", "PCOPPUSDM", "M", Fmt(pre="$", dec=0)),
        ),
    ),
)

TAPE: Final = (
    "us_fftu",
    "us_10y",
    "us_10y3m",
    "us_dollar",
    "wti",
    "us_vix",
    "kr_usdkrw",
    "kr_kospi",
)

CURVE_TENORS: Final = (
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
CURVE_YEARS: Final = 4
SPREAD_SERIES: Final = FredSeries("T10Y2Y", 11)
RECESSION_SERIES: Final = FredSeries("USREC", 11)

# Release-name fragments (FRED releases/dates) shown in the weekly calendar.
CALENDAR_RELEASES: Final = (
    ("Consumer Price Index", "소비자물가 CPI", "BLS"),
    ("Producer Price Index", "생산자물가 PPI", "BLS"),
    ("Employment Situation", "고용보고서", "BLS"),
    ("Job Openings and Labor Turnover", "JOLTS 구인", "BLS"),
    ("Unemployment Insurance Weekly Claims", "신규 실업수당 청구", "DOL"),
    ("Gross Domestic Product", "GDP", "BEA"),
    ("Personal Income and Outlays", "개인소득·지출 (PCE)", "BEA"),
    ("Advance Monthly Sales for Retail", "소매판매", "Census"),
    ("New Residential Construction", "주택착공", "Census"),
    ("Industrial Production and Capacity Utilization", "산업생산", "FRB"),
    ("H.4.1", "연준 대차대조표 H.4.1", "FRB"),
    ("FOMC Press Release", "FOMC 결정", "FRB"),
    ("Chicago Fed National Financial Conditions", "NFCI", "Chicago Fed"),
)
CALENDAR_DAYS: Final = 7
