"""What the dashboard shows. Edit this file to add, remove or re-order indicators.

Every entry comes from one of the seven public sources in ``sources.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from finagent.models import Fmt
from finagent.sources import (
    EcosSeries,
    FiscalSeries,
    FredSeries,
    NyFedRate,
    Ratio,
    SdmxSeries,
)

if TYPE_CHECKING:
    from finagent.models import Freq
    from finagent.sources import Fetch
    from finagent.transforms import Transform


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
PP2: Final = Fmt(suf="%p", dec=2)
PP2_SIGNED: Final = Fmt(suf="%p", dec=2, signed=True)
IDX0: Final = Fmt(dec=0)
IDX1: Final = Fmt(dec=1)
IDX2: Final = Fmt(dec=2)
IDX1_SIGNED: Final = Fmt(dec=1, signed=True)
IDX2_SIGNED: Final = Fmt(dec=2, signed=True)
USD_T: Final = Fmt(pre="$", suf="T", dec=2)
USD_B: Final = Fmt(pre="$", suf="B", dec=0)
USD_B1: Final = Fmt(pre="$", suf="B", dec=1)
USD2: Final = Fmt(pre="$", dec=2)
USD0: Final = Fmt(pre="$", dec=0)


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
    *,
    scale: float = 1.0,
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
        scale=scale,
    )


def ecb(  # noqa: PLR0913
    id_: str,
    label: str,
    code: str,
    flow_key: str,
    freq: Freq,
    fmt: Fmt,
) -> Indicator:
    return Indicator(
        id=id_,
        label=label,
        code=f"ECB:{code}",
        fetch=SdmxSeries("ECB", flow_key),
        freq=freq,
        fmt=fmt,
    )


def bis(  # noqa: PLR0913
    id_: str,
    label: str,
    flow: str,
    key: str,
    freq: Freq,
    fmt: Fmt,
    transform: Transform = "level",
) -> Indicator:
    return Indicator(
        id=id_,
        label=label,
        code=f"BIS:{flow}/{key}",
        fetch=SdmxSeries("BIS", f"{flow}/{key}/all"),
        freq=freq,
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


# One OECD request returns every listed country (OECD allows 60 downloads an hour).
OECD_CLI_FLOW: Final = (
    "OECD.SDD.STES,DSD_STES@DF_CLI/USA+CHN+KOR+JPN+DEU+GBR.M.LI...AA...H"
)
OECD_CONFIDENCE_FLOW: Final = (
    "OECD.SDD.STES,DSD_STES@DF_CLI/USA+CHN+KOR+DEU.M.BCICP+CCICP...AA...H"
)


def oecd_cli(id_: str, label: str, area: str) -> Indicator:
    return Indicator(
        id=id_,
        label=label,
        code=f"OECD:CLI/{area}",
        fetch=SdmxSeries("OECD", OECD_CLI_FLOW, (("REF_AREA", area),)),
        freq="M",
        fmt=IDX1,
    )


def oecd_confidence(id_: str, label: str, area: str, measure: str) -> Indicator:
    return Indicator(
        id=id_,
        label=label,
        code=f"OECD:{measure}/{area}",
        fetch=SdmxSeries(
            "OECD",
            OECD_CONFIDENCE_FLOW,
            (("REF_AREA", area), ("MEASURE", measure)),
        ),
        freq="M",
        fmt=IDX1,
    )


def ecb_fx(currency: str) -> SdmxSeries:
    return SdmxSeries("ECB", f"EXR/D.{currency}.EUR.SP00.A")


def nyfed(id_: str, label: str, path: str) -> Indicator:
    name = path.rsplit("/", 1)[-1].upper()
    return Indicator(id_, label, f"NYFED:{name}", NyFedRate(path), "D", PCT2)


PANELS: Final = (
    PanelSpec(
        "growth",
        "GDP",
        "미국 성장",
        "BEA · Census · FRB · 지역 연은",
        (
            fred("us_gdp", "실질 GDP (전기비 연율)", "A191RL1Q225SBEA", "Q", PCT1),
            fred("us_gdp_yoy", "실질 GDP YoY", "A191RO1Q156NBEA", "Q", PCT1),
            fred("us_gdpnow", "GDPNow (애틀랜타 연은)", "GDPNOW", "Q", PCT1),
            fred("us_indpro", "산업생산 YoY", "INDPRO", "M", PCT1, "yoy"),
            fred("us_tcu", "설비가동률", "TCU", "M", PCT1),
            fred("us_retail", "소매판매 MoM", "RSAFS", "M", PCT1_SIGNED, "pct"),
            fred("us_pce_real", "실질 개인소비 MoM", "PCEC96", "M", PCT1_SIGNED, "pct"),
            fred(
                "us_core_capex",
                "핵심 자본재 수주 MoM",
                "NEWORDER",
                "M",
                PCT1_SIGNED,
                "pct",
            ),
            fred("us_durables", "내구재 수주 MoM", "DGORDER", "M", PCT1_SIGNED, "pct"),
            fred("us_isratio", "재고/판매 비율", "ISRATIO", "M", IDX2),
            fred("us_autos", "자동차 판매 (연율, 백만대)", "TOTALSA", "M", IDX1),
            fred("us_cfnai", "시카고 연은 활동지수 3MA", "CFNAIMA3", "M", IDX2_SIGNED),
            fred(
                "us_philly",
                "필라델피아 연은 제조업",
                "GACDFSA066MSFRBPHI",
                "M",
                IDX1_SIGNED,
            ),
            fred(
                "us_empire",
                "뉴욕 연은 제조업",
                "GACDINA066MNFRBNY",
                "M",
                IDX1_SIGNED,
            ),
        ),
    ),
    PanelSpec(
        "household",
        "HH",
        "미국 소득 · 가계",
        "BEA · FRB · 미시간대",
        (
            fred(
                "us_real_income",
                "이전소득 제외 실질소득 YoY",
                "W875RX1",
                "M",
                PCT1,
                "yoy",
            ),
            fred("us_saving", "개인 저축률", "PSAVERT", "M", PCT1),
            fred(
                "us_umich",
                "미시간대 소비자심리",
                "UMCSENT",
                "M",
                IDX1,
                copyright_="UMich",
            ),
            fred("us_consumer_credit", "소비자신용 YoY", "TOTALSL", "M", PCT1, "yoy"),
            fred("us_cc_delinq", "신용카드 연체율", "DRCCLACBS", "Q", PCT2),
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
            fred("us_u6", "U-6 실업률", "U6RATE", "M", PCT1),
            fred("us_sahm", "삼 룰 지표", "SAHMREALTIME", "M", PP2),
            fred("us_lfpr", "경제활동참가율", "CIVPART", "M", PCT1),
            fred(
                "us_prime_lfpr", "핵심연령 참가율 (25~54세)", "LNS11300060", "M", PCT1
            ),
            fred(
                "us_prime_epop", "핵심연령 고용률 (25~54세)", "LNS12300060", "M", PCT1
            ),
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
            fred("us_quits", "JOLTS 자발적 이직률", "JTSQUR", "M", PCT1),
            fred("us_ahe", "시간당 임금 YoY", "CES0500000003", "M", PCT1, "yoy"),
            fred("us_eci", "고용비용지수 ECI YoY", "ECIALLCIV", "Q", PCT1, "yoy"),
            fred("us_ulc", "단위노동비용 YoY", "ULCNFB", "Q", PCT1, "yoy"),
        ),
    ),
    PanelSpec(
        "inflation",
        "INFL",
        "미국 물가",
        "BLS · BEA · 클리블랜드·애틀랜타·댈러스 연은",
        (
            fred("us_cpi", "CPI YoY", "CPIAUCSL", "M", PCT1, "yoy"),
            fred("us_core_cpi", "근원 CPI YoY", "CPILFESL", "M", PCT1, "yoy"),
            fred("us_pce", "PCE YoY", "PCEPI", "M", PCT1, "yoy"),
            fred("us_core_pce", "근원 PCE YoY", "PCEPILFE", "M", PCT1, "yoy"),
            fred(
                "us_median_cpi",
                "중위 CPI (전월비 연율)",
                "MEDCPIM158SFRBCLE",
                "M",
                PCT1,
            ),
            fred(
                "us_sticky_cpi", "경직적 가격 CPI YoY", "STICKCPIM159SFRBATL", "M", PCT1
            ),
            fred("us_trim_pce", "절사평균 PCE YoY", "PCETRIM12M159SFRBDAL", "M", PCT1),
            fred("us_shelter", "CPI 주거비 YoY", "CUSR0000SAH1", "M", PCT1, "yoy"),
            fred(
                "us_services_cpi",
                "CPI 에너지 제외 서비스 YoY",
                "CUSR0000SASLE",
                "M",
                PCT1,
                "yoy",
            ),
            fred(
                "us_core_goods",
                "CPI 근원 상품 YoY",
                "CUSR0000SACL1E",
                "M",
                PCT1,
                "yoy",
            ),
            fred("us_ppi", "PPI 최종수요 YoY", "PPIFIS", "M", PCT1, "yoy"),
            fred("us_bei5", "5년 BEI", "T5YIE", "D", PCT2),
            fred("us_bei10", "10년 BEI", "T10YIE", "D", PCT2),
            fred("us_5y5y", "5년 후 5년 기대인플레", "T5YIFR", "D", PCT2),
            fred(
                "us_mich",
                "미시간대 1년 기대인플레",
                "MICH",
                "M",
                PCT1,
                copyright_="UMich",
            ),
        ),
    ),
    PanelSpec(
        "rates",
        "RATE",
        "정책금리 · 국채 · 단기자금",
        "FRB · NY Fed · Treasury",
        (
            fred("us_fftu", "연방기금 목표 상단", "DFEDTARU", "D", PCT2),
            fred("us_fftl", "연방기금 목표 하단", "DFEDTARL", "D", PCT2),
            fred("us_iorb", "지급준비금 이자율 IORB", "IORB", "D", PCT2),
            nyfed("us_effr", "실효 연방기금금리", "unsecured/effr"),
            nyfed("us_obfr", "오버나이트 은행 자금금리 OBFR", "unsecured/obfr"),
            nyfed("us_sofr", "SOFR", "secured/sofr"),
            nyfed("us_tgcr", "3자간 GC 레포 TGCR", "secured/tgcr"),
            fred("us_2y", "2년물 국채", "DGS2", "D", PCT2),
            fred("us_10y", "10년물 국채", "DGS10", "D", PCT2),
            fred("us_10y3m", "10년 − 3개월", "T10Y3M", "D", PP2_SIGNED),
            fred("us_10y2y", "10년 − 2년", "T10Y2Y", "D", PP2_SIGNED, years=11),
            fred("us_real10", "10년 실질금리 (TIPS)", "DFII10", "D", PCT2),
        ),
    ),
    PanelSpec(
        "fincon",
        "FCI",
        "신용 · 금융여건",
        "Chicago Fed · St. Louis Fed · ICE · Moody's · FRB",
        (
            fred("us_nfci", "NFCI (0 미만 = 완화)", "NFCI", "W", IDX2_SIGNED),
            fred(
                "us_stlfsi",
                "세인트루이스 연은 금융스트레스",
                "STLFSI4",
                "W",
                IDX2_SIGNED,
            ),
            fred(
                "us_hy_oas",
                "하이일드 OAS",
                "BAMLH0A0HYM2",
                "D",
                PP2,
                copyright_="ICE",
                years=3,
            ),
            fred(
                "us_ig_oas",
                "투자등급 OAS",
                "BAMLC0A0CM",
                "D",
                PP2,
                copyright_="ICE",
                years=3,
            ),
            fred("us_baa10y", "Baa − 10년물", "BAA10Y", "D", PP2, copyright_="Moody's"),
            fred("us_hqm10", "고등급 회사채 10년 (재무부 HQM)", "HQMCB10YR", "M", PCT2),
            fred(
                "us_sloos",
                "C&I 대출기준 강화 순비율 (SLOOS)",
                "DRTSCILM",
                "Q",
                PCT1_SIGNED,
            ),
            fred("us_bank_credit", "은행 총신용 YoY", "TOTBKCR", "W", PCT1, "yoy"),
            fred("us_ci_loans", "기업(C&I) 대출 YoY", "BUSLOANS", "M", PCT1, "yoy"),
            fred("us_dollar", "달러 지수 (광의)", "DTWEXBGS", "D", IDX1),
        ),
    ),
    PanelSpec(
        "markets",
        "EQ",
        "주식 · 변동성",
        "S&P DJI · Nasdaq · Cboe · 한국거래소(ECOS)",
        (
            fred("us_sp500", "S&P 500", "SP500", "D", IDX0, copyright_="S&P DJI"),
            fred(
                "us_nasdaq", "나스닥 종합", "NASDAQCOM", "D", IDX0, copyright_="Nasdaq"
            ),
            fred("us_vix", "VIX", "VIXCLS", "D", IDX1, copyright_="Cboe"),
            ecos("kr_kospi", "코스피", "802Y001", "D", "0001000", IDX2),
            ecos("kr_kosdaq", "코스닥", "802Y001", "D", "0089000", IDX2),
        ),
    ),
    PanelSpec(
        "liquidity",
        "LIQ",
        "연준 대차대조표 · 재정",
        "FRB · Treasury Fiscal Data",
        (
            fred("us_walcl", "연준 총자산", "WALCL", "W", USD_T, scale=1e-6),
            fred("us_fed_ust", "연준 보유 국채", "TREAST", "W", USD_T, scale=1e-6),
            fred("us_fed_mbs", "연준 보유 MBS", "WSHOMCB", "W", USD_T, scale=1e-6),
            fred("us_reserves", "지급준비금", "WRESBAL", "W", USD_T, scale=0.001),
            fred("us_rrp", "역레포 잔액", "RRPONTSYD", "D", USD_B),
            fred("us_tga", "재무부 일반계정 TGA", "WTREGEN", "W", USD_B),
            fred("us_m2", "M2 YoY", "M2SL", "M", PCT1, "yoy"),
            Indicator(
                "us_debt",
                "연방정부 총부채",
                "TREASURY:debt_to_penny",
                FiscalSeries("v2/accounting/od/debt_to_penny", "tot_pub_debt_out_amt"),
                "D",
                USD_T,
                scale=1e-12,
            ),
            Indicator(
                "us_debt_public",
                "민간 보유 연방부채",
                "TREASURY:debt_held_public",
                FiscalSeries("v2/accounting/od/debt_to_penny", "debt_held_public_amt"),
                "D",
                USD_T,
                scale=1e-12,
            ),
            fred("us_debt_gdp", "연방부채/GDP", "GFDEGDQ188S", "Q", PCT1),
            fred(
                "us_deficit12",
                "재정수지 12개월 합계",
                "MTSDS133FMS",
                "M",
                USD_T,
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
            Indicator(
                "us_avg_bill_rate",
                "T-bill 평균 조달금리",
                "TREASURY:avg_interest_rates/bills",
                FiscalSeries(
                    "v2/accounting/od/avg_interest_rates",
                    "avg_interest_rate_amt",
                    "security_desc:eq:Treasury Bills",
                ),
                "M",
                PCT2,
            ),
        ),
    ),
    PanelSpec(
        "housing",
        "HOUS",
        "미국 주택",
        "Census · FHFA · S&P DJI · Freddie Mac · FRB",
        (
            fred("us_houst", "주택착공 (연율, 천호)", "HOUST", "M", IDX0),
            fred("us_permits", "건축허가 (연율, 천호)", "PERMIT", "M", IDX0),
            fred(
                "us_new_home_sales",
                "신규 단독주택 판매 (연율, 천호)",
                "HSN1F",
                "M",
                IDX0,
            ),
            fred("us_home_supply", "신규주택 재고 (개월)", "MSACSR", "M", IDX1),
            fred("us_fhfa", "FHFA 주택가격 YoY", "USSTHPI", "Q", PCT1, "yoy"),
            fred(
                "us_case_shiller",
                "Case-Shiller 전국 YoY",
                "CSUSHPISA",
                "M",
                PCT1,
                "yoy",
                copyright_="S&P DJI",
            ),
            fred(
                "us_mort30",
                "30년 모기지 금리",
                "MORTGAGE30US",
                "W",
                PCT2,
                copyright_="Freddie Mac",
            ),
            fred("us_mort_delinq", "주택담보대출 연체율", "DRSFRMACBS", "Q", PCT2),
        ),
    ),
    PanelSpec(
        "korea",
        "KR",
        "한국 금리 · 금융",
        "한국은행 ECOS",
        (
            ecos("kr_base", "한국은행 기준금리", "722Y001", "D", "0101000", PCT2),
            ecos("kr_call", "콜금리 (익일물)", "817Y002", "D", "010101000", PCT2),
            ecos("kr_cd91", "CD 91일", "817Y002", "D", "010502000", PCT2),
            ecos("kr_ktb3", "국고채 3년", "817Y002", "D", "010200000", PCT2),
            ecos("kr_ktb10", "국고채 10년", "817Y002", "D", "010210000", PCT2),
            ecos("kr_corp_aa", "회사채 3년 AA−", "817Y002", "D", "010300000", PCT2),
            ecos(
                "kr_usdkrw",
                "원/달러 매매기준율",
                "731Y001",
                "D",
                "0000001",
                Fmt(suf="원", dec=1),
            ),
            ecos(
                "kr_jpykrw",
                "원/100엔",
                "731Y001",
                "D",
                "0000002",
                Fmt(suf="원", dec=1),
            ),
            ecos(
                "kr_fx_reserves",
                "외환보유액",
                "732Y001",
                "M",
                "99",
                USD_B1,
                scale=1e-6,
            ),
            ecos("kr_m2", "M2 YoY (평잔)", "161Y006", "M", "BBHA00", PCT1, "yoy"),
            ecos(
                "kr_household_credit",
                "가계신용 YoY",
                "151Y001",
                "Q",
                "1000000",
                PCT1,
                "yoy",
            ),
        ),
    ),
    PanelSpec(
        "korea_real",
        "KR2",
        "한국 성장 · 물가 · 경기",
        "한국은행 ECOS · OECD",
        (
            ecos(
                "kr_gdp",
                "실질 GDP 전기비",
                "200Y102",
                "Q",
                "10111",
                PCT1_SIGNED,
            ),
            ecos("kr_cpi", "소비자물가 YoY", "901Y009", "M", "0", PCT1, "yoy"),
            ecos(
                "kr_core_cpi",
                "근원물가 YoY (농산물·석유류 제외)",
                "901Y010",
                "M",
                "QB",
                PCT1,
                "yoy",
            ),
            ecos("kr_ppi", "생산자물가 YoY", "404Y014", "M", "*AA", PCT1, "yoy"),
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
                "kr_imports",
                "수입 (통관) YoY",
                "901Y118",
                "M",
                "T004",
                PCT1_SIGNED,
                "yoy",
            ),
            ecos(
                "kr_current_account",
                "경상수지 (계절조정)",
                "301Y017",
                "M",
                "SA000",
                Fmt(pre="$", suf="B", dec=1, signed=True),
                scale=0.001,
            ),
            ecos(
                "kr_unrate", "실업률 (SA)", "901Y027", "M", "I61BC", PCT1, item2="I28B"
            ),
            ecos(
                "kr_emp_rate", "고용률 (SA)", "901Y027", "M", "I61E", PCT1, item2="I28B"
            ),
            ecos("kr_coincident", "동행지수 순환변동치", "901Y067", "M", "I16D", IDX1),
            ecos("kr_leading", "선행지수 순환변동치", "901Y067", "M", "I16E", IDX1),
            ecos(
                "kr_csi",
                "소비자심리지수 CSI",
                "511Y002",
                "M",
                "FME",
                IDX1,
                item2="99988",
            ),
            ecos("kr_esi", "경제심리지수 ESI", "513Y001", "M", "E2000", IDX1),
            oecd_cli("kr_cli", "OECD 경기선행지수 한국", "KOR"),
        ),
    ),
    PanelSpec(
        "euro",
        "EA",
        "유로존",
        "ECB Data Portal",
        (
            ecb(
                "ecb_dfr",
                "ECB 예금금리",
                "FM/DFR",
                "FM/B.U2.EUR.4F.KR.DFR.LEV",
                "D",
                PCT2,
            ),
            ecb(
                "ecb_mro",
                "ECB 주요 리파이낸싱 금리",
                "FM/MRR_FR",
                "FM/B.U2.EUR.4F.KR.MRR_FR.LEV",
                "D",
                PCT2,
            ),
            ecb("ea_estr", "€STR", "EST/WT", "EST/B.EU000A2X2A25.WT", "D", PCT2),
            ecb(
                "ea_10y",
                "AAA 국채 10년",
                "YC/SR_10Y",
                "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                "D",
                PCT2,
            ),
            ecb(
                "ea_10y3m",
                "AAA 10년 − 3개월",
                "YC/SRS_10Y_3M",
                "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SRS_10Y_3M",
                "D",
                PP2_SIGNED,
            ),
            ecb(
                "ea_hicp",
                "HICP YoY",
                "HICP/U2/ANR",
                "HICP/M.U2.N.000000.4D0.ANR",
                "M",
                PCT1,
            ),
            ecb(
                "ea_core_hicp",
                "근원 HICP YoY",
                "HICP/XEF000",
                "HICP/M.U2.N.XEF000.4D0.ANR",
                "M",
                PCT1,
            ),
            ecb(
                "ea_services_hicp",
                "서비스 HICP YoY",
                "HICP/SERV00",
                "HICP/M.U2.N.SERV00.4D0.ANR",
                "M",
                PCT1,
            ),
            ecb(
                "ea_m3",
                "M3 YoY",
                "BSI/M30",
                "BSI/M.U2.Y.V.M30.X.I.U2.2300.Z01.A",
                "M",
                PCT1,
            ),
            ecb(
                "ea_nfc_loans",
                "기업 대출 YoY",
                "BSI/A20T/2240",
                "BSI/M.U2.Y.U.A20T.A.I.U2.2240.Z01.A",
                "M",
                PCT1,
            ),
            ecb(
                "ea_unrate",
                "실업률",
                "LFSI/UNEHRT",
                "LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T",
                "M",
                PCT1,
            ),
            ecb(
                "ea_ip",
                "산업생산 YoY",
                "STBS/PROD",
                "STBS/M.I10.W.PROD.NS0020.4D0.GY.IX",
                "M",
                PCT1_SIGNED,
            ),
            ecb(
                "ea_ciss",
                "시스템 스트레스 지수 CISS",
                "CISS/SS_CIN",
                "CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX",
                "D",
                IDX2,
            ),
            Indicator(
                "eurusd", "EUR/USD", "ECB:EXR/USD", ecb_fx("USD"), "D", Fmt(dec=4)
            ),
            ecb(
                "ea_neer",
                "유로 명목실효환율 EER-41",
                "EXR/E03",
                "EXR/M.E03.EUR.EN00.A",
                "M",
                IDX1,
            ),
        ),
    ),
    PanelSpec(
        "global",
        "GLBL",
        "글로벌 정책금리 · 환율",
        "BIS · ECB · FRB",
        (
            bis_policy_rate("boj_rate", "일본은행 정책금리", "JP"),
            bis_policy_rate("boe_rate", "영란은행 정책금리", "GB"),
            bis_policy_rate("pboc_rate", "중국인민은행 정책금리", "CN"),
            bis_policy_rate("boc_rate", "캐나다은행 정책금리", "CA"),
            bis_policy_rate("rba_rate", "호주중앙은행 정책금리", "AU"),
            bis_policy_rate("snb_rate", "스위스중앙은행 정책금리", "CH"),
            Indicator(
                "usdjpy",
                "USD/JPY",
                "ECB:EXR/JPY÷USD",
                Ratio(ecb_fx("JPY"), ecb_fx("USD")),
                "D",
                IDX2,
            ),
            fred("usdcny", "USD/CNY", "DEXCHUS", "D", Fmt(dec=4)),
            fred("us_dollar_em", "달러 지수 (신흥국 대비)", "DTWEXEMEGS", "D", IDX1),
        ),
    ),
    PanelSpec(
        "cycle",
        "CLI",
        "글로벌 경기 · 심리",
        "OECD",
        (
            oecd_cli("us_cli", "경기선행지수 미국", "USA"),
            oecd_cli("cn_cli", "경기선행지수 중국", "CHN"),
            oecd_cli("jp_cli", "경기선행지수 일본", "JPN"),
            oecd_cli("de_cli", "경기선행지수 독일", "DEU"),
            oecd_cli("gb_cli", "경기선행지수 영국", "GBR"),
            oecd_confidence("us_bci", "기업신뢰지수 미국", "USA", "BCICP"),
            oecd_confidence("cn_bci", "기업신뢰지수 중국", "CHN", "BCICP"),
            oecd_confidence("kr_bci", "기업신뢰지수 한국", "KOR", "BCICP"),
            oecd_confidence("de_bci", "기업신뢰지수 독일", "DEU", "BCICP"),
            oecd_confidence("us_cci", "소비자신뢰지수 미국", "USA", "CCICP"),
            oecd_confidence("kr_cci", "소비자신뢰지수 한국", "KOR", "CCICP"),
            oecd_confidence("de_cci", "소비자신뢰지수 독일", "DEU", "CCICP"),
        ),
    ),
    PanelSpec(
        "credit",
        "BIS",
        "글로벌 신용 · 주택 · 실효환율",
        "BIS Data Portal",
        (
            bis(
                "kr_hh_debt",
                "한국 가계부채/GDP",
                "WS_TC",
                "Q.KR.H.A.M.770.A",
                "Q",
                PCT1,
            ),
            bis(
                "kr_private_credit",
                "한국 민간신용/GDP",
                "WS_TC",
                "Q.KR.P.A.M.770.A",
                "Q",
                PCT1,
            ),
            bis(
                "cn_private_credit",
                "중국 민간신용/GDP",
                "WS_TC",
                "Q.CN.P.A.M.770.A",
                "Q",
                PCT1,
            ),
            bis(
                "us_hh_debt",
                "미국 가계부채/GDP",
                "WS_TC",
                "Q.US.H.A.M.770.A",
                "Q",
                PCT1,
            ),
            bis(
                "kr_credit_gap",
                "한국 신용갭",
                "WS_CREDIT_GAP",
                "Q.KR.P.A.C",
                "Q",
                Fmt(suf="%p", dec=1, signed=True),
            ),
            bis(
                "cn_credit_gap",
                "중국 신용갭",
                "WS_CREDIT_GAP",
                "Q.CN.P.A.C",
                "Q",
                Fmt(suf="%p", dec=1, signed=True),
            ),
            bis(
                "kr_house_real",
                "한국 실질 주택가격 YoY",
                "WS_SPP",
                "Q.KR.R.628",
                "Q",
                PCT1_SIGNED,
                "yoy",
            ),
            bis(
                "us_house_real",
                "미국 실질 주택가격 YoY",
                "WS_SPP",
                "Q.US.R.628",
                "Q",
                PCT1_SIGNED,
                "yoy",
            ),
            bis(
                "cn_house_real",
                "중국 실질 주택가격 YoY",
                "WS_SPP",
                "Q.CN.R.628",
                "Q",
                PCT1_SIGNED,
                "yoy",
            ),
            bis("kr_reer", "원화 실질실효환율", "WS_EER", "M.R.B.KR", "M", IDX1),
            bis("us_reer", "달러 실질실효환율", "WS_EER", "M.R.B.US", "M", IDX1),
        ),
    ),
    PanelSpec(
        "commodities",
        "CMDY",
        "원자재",
        "EIA · IMF",
        (
            fred("wti", "WTI ($/bbl)", "DCOILWTICO", "D", USD2),
            fred("brent", "브렌트 ($/bbl)", "DCOILBRENTEU", "D", USD2),
            fred("henryhub", "헨리허브 가스 ($/MMBtu)", "DHHNGSP", "D", USD2),
            fred("copper", "구리 ($/t)", "PCOPPUSDM", "M", USD0),
            fred("aluminum", "알루미늄 ($/t)", "PALUMUSDM", "M", USD0),
            fred("iron_ore", "철광석 ($/t)", "PIORECRUSDM", "M", USD2),
            fred("wheat", "밀 ($/t)", "PWHEAMTUSDM", "M", USD2),
            fred(
                "imf_commodities",
                "IMF 전체 원자재가격지수",
                "PALLFNFINDEXM",
                "M",
                IDX1,
            ),
        ),
    ),
)

TAPE: Final = (
    "us_fftu",
    "us_10y",
    "us_10y3m",
    "us_sp500",
    "us_dollar",
    "wti",
    "us_vix",
    "kr_usdkrw",
    "kr_ktb10",
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
    ("Employment Cost Index", "고용비용지수 ECI", "BLS"),
    ("Productivity and Costs", "생산성·단위노동비용", "BLS"),
    ("Unemployment Insurance Weekly Claims", "신규 실업수당 청구", "DOL"),
    ("Gross Domestic Product", "GDP", "BEA"),
    ("Personal Income and Outlays", "개인소득·지출 (PCE)", "BEA"),
    ("Advance Monthly Sales for Retail", "소매판매", "Census"),
    ("Advance Report on Durable Goods", "내구재 수주", "Census"),
    ("New Residential Construction", "주택착공", "Census"),
    ("New Residential Sales", "신규주택 판매", "Census"),
    ("Industrial Production and Capacity Utilization", "산업생산", "FRB"),
    ("H.4.1", "연준 대차대조표 H.4.1", "FRB"),
    ("FOMC Press Release", "FOMC 결정", "FRB"),
    ("Senior Loan Officer", "대출태도 서베이 SLOOS", "FRB"),
    ("Surveys of Consumers", "미시간대 소비자심리", "UMich"),
    ("Chicago Fed National Financial Conditions", "NFCI", "Chicago Fed"),
)
CALENDAR_DAYS: Final = 7


@dataclass(frozen=True, slots=True)
class SepRow:
    """One FOMC Summary of Economic Projections median and what to compare it to.

    ``series_id`` is dated by target year; ``longer_run_id`` holds the
    longer-run median. ``current`` lists dashboard row ids whose latest values
    are averaged (the federal funds target range is shown as its midpoint).
    """

    label: str
    series_id: str
    longer_run_id: str | None
    current: tuple[str, ...]
    current_note: str
    current_dec: int = 1


SEP_ROWS: Final = (
    SepRow(
        "연방기금금리",
        "FEDTARMD",
        "FEDTARMDLR",
        ("us_fftl", "us_fftu"),
        "목표 범위 중간값",
        current_dec=2,
    ),
    SepRow("실질 GDP 성장률", "GDPC1MD", "GDPC1MDLR", ("us_gdp_yoy",), "최근 분기 YoY"),
    SepRow("실업률", "UNRATEMD", "UNRATEMDLR", ("us_unrate",), "최근 월"),
    SepRow("PCE 물가", "PCECTPIMD", "PCECTPIMDLR", ("us_pce",), "최근 월 YoY"),
    SepRow("근원 PCE 물가", "JCXFEMD", None, ("us_core_pce",), "최근 월 YoY"),
)
# Vintages published within this window; SEP releases come every quarter.
SEP_LOOKBACK_DAYS: Final = 400
SEP_MAX_YEARS: Final = 4


# ---------------------------------------------------------------- default view

# Indicators shown by default: the macro inputs a single-company analysis needs
# (economic indicators, global monetary policy, economic cycle, market
# sentiment), plus every row the tape, regime row and SEP table read.
# Everything else in a panel appears only when that panel's deep view is on.
CORE: Final = frozenset(
    {
        # Economic indicators: GDP, CPI/PPI, jobs, retail sales, industrial
        # production; the Philadelphia Fed survey stands in for a PMI.
        "us_gdp",
        "us_gdp_yoy",
        "us_indpro",
        "us_retail",
        "us_philly",
        "us_payrolls",
        "us_unrate",
        "us_cpi",
        "us_core_cpi",
        "us_pce",
        "us_core_pce",
        "us_ppi",
        "kr_gdp",
        "kr_cpi",
        "kr_ppi",
        "kr_exports",
        "kr_unrate",
        # Monetary policy at the Fed, BOK, ECB, BOJ and PBOC; major economies.
        "us_fftu",
        "us_fftl",
        "us_10y",
        "kr_base",
        "kr_ktb10",
        "ecb_dfr",
        "ea_hicp",
        "ea_unrate",
        "boj_rate",
        "pboc_rate",
        "us_dollar",
        "kr_usdkrw",
        # Economic cycle: household and corporate debt, money supply, turning
        # points.
        "us_m2",
        "kr_m2",
        "kr_household_credit",
        "us_hh_debt",
        "kr_hh_debt",
        "kr_private_credit",
        "cn_private_credit",
        "us_ci_loans",
        "us_sahm",
        "us_10y2y",
        "us_10y3m",
        "kr_leading",
        "us_cli",
        "cn_cli",
        # Market sentiment.
        "us_sp500",
        "us_nasdaq",
        "kr_kospi",
        "kr_kosdaq",
        "us_vix",
        "us_hy_oas",
        "us_nfci",
        "us_umich",
        "kr_csi",
        "wti",
    }
)

CURVE_VIEW: Final = "curve"


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """One switch in the settings: a panel's deep rows, or the curve chart."""

    id: str
    key: str
    title: str
    core: int
    extra: int


VIEWS: Final = (
    *(
        ViewSpec(
            spec.id,
            spec.key,
            spec.title,
            core=sum(indicator.id in CORE for indicator in spec.indicators),
            extra=sum(indicator.id not in CORE for indicator in spec.indicators),
        )
        for spec in PANELS
    ),
    ViewSpec(CURVE_VIEW, "YC", "미 국채 수익률 곡선", core=0, extra=len(CURVE_TENORS)),
)
ALL_VIEWS: Final = frozenset(view.id for view in VIEWS)
# The default: every panel shows its core rows only.
CORE_VIEW: Final[frozenset[str]] = frozenset()


def shown(spec: PanelSpec, deep: frozenset[str]) -> tuple[Indicator, ...]:
    """The rows a panel shows: all of them with its deep view on, else the core."""
    if spec.id in deep:
        return spec.indicators
    return tuple(indicator for indicator in spec.indicators if indicator.id in CORE)
