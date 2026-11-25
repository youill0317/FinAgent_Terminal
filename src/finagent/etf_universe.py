"""ETF 기본 유니버스 및 사용자 정의 목록 관리."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class Product:
    ticker: str
    group: str
    category: str
    multiple: float = 1.0


DEFAULTS: Final[tuple[Product, ...]] = (
    Product("QQQ", "나스닥", "나스닥100", 1.0),
    Product("QQQM", "나스닥", "나스닥100", 1.0),
    Product("PSQ", "나스닥", "나스닥100", -1.0),
    Product("VOO", "S&P500", "S&P500", 1.0),
    Product("IVV", "S&P500", "S&P500", 1.0),
    Product("SPY", "S&P500", "S&P500", 1.0),
    Product("SH", "S&P500", "S&P500", -1.0),
    Product("SPDN", "S&P500", "S&P500", -1.0),
    Product("RSP", "S&P500", "S&P500 동일비중", 1.0),
    Product("VTI", "미국 지수추종", "미국 전체 시장", 1.0),
    Product("IWF", "미국 지수추종", "러셀1000 성장", 1.0),
    Product("VONG", "미국 지수추종", "러셀1000 성장", 1.0),
    Product("VTV", "미국 지수추종", "미국 대형 가치", 1.0),
    Product("IWD", "미국 지수추종", "미국 대형 가치", 1.0),
    Product("IJH", "미국 지수추종", "S&P 미드캡 400", 1.0),
    Product("SPMD", "미국 지수추종", "S&P 미드캡 400", 1.0),
    Product("IVOO", "미국 지수추종", "S&P 미드캡 400", 1.0),
    Product("MYY", "미국 지수추종", "S&P 미드캡 400", -1.0),
    Product("IJR", "미국 지수추종", "S&P 스몰캡 600", 1.0),
    Product("SPSM", "미국 지수추종", "S&P 스몰캡 600", 1.0),
    Product("VIOO", "미국 지수추종", "S&P 스몰캡 600", 1.0),
    Product("DIA", "미국 지수추종", "다우존스", 1.0),
    Product("DOG", "미국 지수추종", "다우존스", -1.0),
    Product("VTWO", "미국 지수추종", "러셀2000", 1.0),
    Product("IWM", "미국 지수추종", "러셀2000", 1.0),
    Product("RWM", "미국 지수추종", "러셀2000", -1.0),
    Product("SPHB", "미국 지수추종", "S&P500 고베타", 1.0),
    Product("SOXX", "반도체", "미국 반도체 대형주", 1.0),
    Product("SMH", "반도체", "미국 반도체 대형주", 1.0),
    Product("XLK", "테크", "미국 기술주", 1.0),
    Product("VGT", "테크", "미국 기술주", 1.0),
    Product("CIBR", "테크", "전세계 보안", 1.0),
    Product("HACK", "테크", "전세계 보안", 1.0),
    Product("BUG", "테크", "전세계 보안", 1.0),
    Product("IGV", "테크", "미국 소프트웨어", 1.0),
    Product("SKYY", "테크", "미국 클라우드 컴퓨팅", 1.0),
    Product("CLOU", "테크", "미국 클라우드 컴퓨팅", 1.0),
    Product("BOTZ", "테크", "전세계 로봇/인공지능", 1.0),
    Product("ICLN", "테크", "전세계 클린에너지", 1.0),
    Product("FNGS", "테크", "FANG 기술주", 1.0),
    Product("IYZ", "테크", "미국 통신", 1.0),
    Product("XLC", "테크", "미국 커뮤니케이션 서비스", 1.0),
    Product("DRIV", "테크", "자율주행", 1.0),
    Product("IDRV", "테크", "자율주행", 1.0),
    Product("KARS", "테크", "자율주행", 1.0),
    Product("XLE", "에너지", "미국 에너지 대형주", 1.0),
    Product("XOP", "에너지", "미국 원유/천연가스 탐사·생산", 1.0),
    Product("IYE", "에너지", "미국 에너지 전체", 1.0),
    Product("OIH", "에너지", "미국 정유 서비스", 1.0),
    Product("GLD", "원자재", "금", 1.0),
    Product("IAU", "원자재", "금", 1.0),
    Product("IAUM", "원자재", "금", 1.0),
    Product("SGOL", "원자재", "금", 1.0),
    Product("AAAU", "원자재", "금", 1.0),
    Product("SLV", "원자재", "은", 1.0),
    Product("GDX", "원자재", "전세계 금광", 1.0),
    Product("GDXJ", "원자재", "전세계 금광 소형주", 1.0),
    Product("XLB", "원자재", "미국 소재/원자재", 1.0),
    Product("VAW", "원자재", "미국 소재/원자재", 1.0),
    Product("IYM", "원자재", "미국 소재/원자재", 1.0),
    Product("FMAT", "원자재", "미국 소재/원자재", 1.0),
    Product("XME", "원자재", "미국 금속·마이닝", 1.0),
    Product("SILJ", "원자재", "은광 채굴·탐사", 1.0),
    Product("USO", "원자재", "원유", 1.0),
    Product("OILK", "원자재", "원유", 1.0),
    Product("UNG", "원자재", "천연가스", 1.0),
    Product("CPER", "원자재", "구리", 1.0),
    Product("DBC", "원자재", "원자재 바스켓", 1.0),
    Product("DBA", "원자재", "농산물", 1.0),
    Product("XLU", "유틸리티", "미국 유틸리티 대형주", 1.0),
    Product("IDU", "유틸리티", "미국 유틸리티 대형주", 1.0),
    Product("XLI", "산업재", "미국 산업재 대형주", 1.0),
    Product("ITA", "산업재", "미국 항공/방위산업", 1.0),
    Product("ITB", "산업재", "미국 건설", 1.0),
    Product("IYT", "산업재", "미국 운송", 1.0),
    Product("IYJ", "산업재", "미국 산업재 전체", 1.0),
    Product("XLV", "헬스케어", "미국 헬스케어 대형주", 1.0),
    Product("XBI", "헬스케어", "미국 바이오테크", 1.0),
    Product("IYH", "헬스케어", "미국 헬스케어 전체", 1.0),
    Product("XPH", "헬스케어", "미국 제약", 1.0),
    Product("IBBQ", "헬스케어", "나스닥 바이오테크", 1.0),
    Product("VNQ", "부동산", "미국 부동산 전체", 1.0),
    Product("SCHH", "부동산", "미국 부동산 전체", 1.0),
    Product("REK", "부동산", "미국 부동산 전체", -1.0),
    Product("XLRE", "부동산", "미국 부동산 대형주", 1.0),
    Product("XLY", "소비재", "미국 임의소비재", 1.0),
    Product("XLP", "소비재", "미국 필수소비재", 1.0),
    Product("IYK", "소비재", "미국 필수소비재", 1.0),
    Product("XRT", "소비재", "미국 유통", 1.0),
    Product("AWAY", "소비재", "전세계 여행 기술주", 1.0),
    Product("XLF", "금융", "미국 금융 대형주", 1.0),
    Product("IYF", "금융", "미국 금융 대형주", 1.0),
    Product("SEF", "금융", "미국 금융 대형주", -1.0),
    Product("KRE", "금융", "미국 지역은행", 1.0),
    Product("TLT", "미국 국채", "장기 국채 20년 이상", 1.0),
    Product("TBF", "미국 국채", "장기 국채 20년 이상", -1.0),
    Product("IEF", "미국 국채", "중기 국채 7-10년", 1.0),
    Product("TBX", "미국 국채", "중기 국채 7-10년", -1.0),
    Product("SHY", "미국 국채", "단기 국채 1-3년", 1.0),
    Product("SGOV", "미국 국채", "초단기 국채 0-3개월", 1.0),
    Product("TIP", "미국 국채", "물가연동 국채", 1.0),
    Product("AGG", "미국 채권", "종합 채권", 1.0),
    Product("BND", "미국 채권", "종합 채권", 1.0),
    Product("LQD", "미국 채권", "투자등급 회사채", 1.0),
    Product("HYG", "하이일드 채권", "하이일드 채권", 1.0),
    Product("SJB", "하이일드 채권", "하이일드 채권", -1.0),
    Product("KWEB", "중국", "중국 인터넷", 1.0),
    Product("FXI", "중국", "중국 대형주 50", 1.0),
    Product("YXI", "중국", "중국 대형주 50", -1.0),
    Product("ASHR", "중국", "중국 CSI 300", 1.0),
    Product("EFA", "기타 국가", "동아시아/유럽/호주", 1.0),
    Product("VGK", "기타 국가", "유럽 선진국", 1.0),
    Product("EZU", "기타 국가", "유로존", 1.0),
    Product("EEM", "기타 국가", "신흥시장", 1.0),
    Product("EUM", "기타 국가", "신흥시장", -1.0),
    Product("EWJ", "기타 국가", "일본", 1.0),
    Product("EWY", "기타 국가", "한국", 1.0),
    Product("EWT", "기타 국가", "대만", 1.0),
    Product("EWZ", "기타 국가", "브라질", 1.0),
    Product("EWW", "기타 국가", "멕시코", 1.0),
    Product("INDA", "인도", "인도", 1.0),
    Product("ARKK", "기타", "아크(ARK)", 1.0),
    Product("SARK", "기타", "아크(ARK)", -1.0),
    Product("BIZD", "기타", "미국 BDC", 1.0),
    Product("BDCZ", "기타", "미국 BDC", 1.0),
    Product("VXX", "변동성", "변동성 지수", 1.0),
    Product("VIXY", "변동성", "변동성 지수", 1.0),
    Product("UUP", "통화", "미국 달러", 1.0),
)


def effective(
    added: Iterable[Product] = (), removed: Iterable[str] = ()
) -> tuple[Product, ...]:
    gone = {t.upper() for t in removed}
    found: dict[str, Product] = {p.ticker: p for p in DEFAULTS if p.ticker not in gone}
    for item in added:
        ticker = item.ticker.upper()
        found[ticker] = replace(item, ticker=ticker)
    return tuple(found.values())


def groups(products: Iterable[Product]) -> list[str]:
    return list(dict.fromkeys(p.group for p in products))


def categories(products: Iterable[Product], group: str | None = None) -> list[str]:
    return list(dict.fromkeys(p.category for p in products if group in (None, p.group)))
