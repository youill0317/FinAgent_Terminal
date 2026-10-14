# 데이터 출처별 카탈로그: 사용중 / 미사용

FinAgent가 쓰는 7개 출처가 무엇을 제공하는지 정리하고, 지금 `catalog.py`에서
쓰는 것과 쓰지 않는 것을 나눈 참고 문서입니다. 지표를 추가할 때 여기서 고르면
됩니다.

**조사 방법과 한계 (2026-10 조사).** 데이터 API와 공식 웹 페이지에 직접 접속할
수 없는 환경에서 웹 검색, 공식 페이지 검색 결과, 공개 GitHub 코드로 정리했습니다.
API를 실제로 호출해 본 것은 아닙니다.

| 표기 | 뜻 |
|---|---|
| ✔ | 공식 페이지 검색 결과나 공개 코드 여러 곳에서 코드가 확인됨 |
| △ | 한 곳에서만 확인됐거나 확인된 패턴으로 조합한 코드 |
| ? | 기억에 의존한 코드. 붙이기 전에 반드시 확인 |
| © | 원 출처에 저작권이 있음 |

✔ 표시도 실제 응답으로 확인한 것은 아닙니다. 지표를 추가한 뒤에는
`uv run finagent --check`로 받아지는지 확인하세요.

---

## 한눈에 보기

| 출처 | 사용중 | 미사용 중 쓸 만한 것 (대표) |
|---|---|---|
| FRED | 94개 + 곡선·침체 12개 | H.8·G.19 세부, JOLTS 채용·해고, 생산성, 신규 사업자, 주별 지역 데이터 |
| 미 재무부 | 4개 | 일간 TGA·세수, 월간 재정수지 세부, 이자비용, 국채 입찰 결과 |
| 뉴욕 연은 | 4개 | ON RRP 결과, SOMA 보유, 레포·SRF, 프라이머리 딜러 |
| 한국은행 ECOS | 26개 | 국고채 곡선, 대출·수신금리, 전산업생산, 소매판매, 주택가격, 기대인플레 |
| ECB | 16개 | GDP, 경상수지, 정부부채, 주택가격, 은행 대출태도, 다른 통화 환율 |
| BIS | 17개 | 부채상환비율, 글로벌 유동성, 장기 CPI |
| OECD | 13개 | 국가별 CPI·실업률·금리, 분기 GDP, 상품무역 |

---

## 1. FRED (세인트루이스 연은)

### 사용중: 94개

| 패널 | 개수 | 코드 |
|---|---|---|
| 미국 성장 | 13 | A191RL1Q225SBEA, GDPNOW, INDPRO, TCU, RSAFS, PCEC96, NEWORDER, DGORDER, ISRATIO, TOTALSA, CFNAIMA3, GACDFSA066MSFRBPHI, GACDINA066MNFRBNY |
| 미국 소득 · 가계 | 5 | W875RX1, PSAVERT, UMCSENT ©, TOTALSL, DRCCLACBS |
| 미국 고용 | 14 | PAYEMS, UNRATE, U6RATE, SAHMREALTIME, CIVPART, LNS11300060, LNS12300060, ICSA, CCSA, JTSJOL, JTSQUR, CES0500000003, ECIALLCIV, ULCNFB |
| 미국 물가 | 15 | CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, MEDCPIM158SFRBCLE, STICKCPIM159SFRBATL, PCETRIM12M159SFRBDAL, CUSR0000SAH1, CUSR0000SASLE, CUSR0000SACL1E, PPIFIS, T5YIE, T10YIE, T5YIFR, MICH © |
| 정책금리 · 국채 · 단기자금 | 7 | DFEDTARU, IORB, DGS2, DGS10, T10Y3M, T10Y2Y, DFII10 |
| 신용 · 금융여건 | 10 | NFCI, STLFSI4, BAMLH0A0HYM2 ©, BAMLC0A0CM ©, BAA10Y ©, HQMCB10YR, DRTSCILM, TOTBKCR, BUSLOANS, DTWEXBGS |
| 주식 · 변동성 | 3 | SP500 ©, NASDAQCOM ©, VIXCLS © |
| 연준 대차대조표 · 재정 | 9 | WALCL, TREAST, WSHOMCB, WRESBAL, RRPONTSYD, WTREGEN, M2SL, GFDEGDQ188S, MTSDS133FMS |
| 미국 주택 | 8 | HOUST, PERMIT, HSN1F, MSACSR, USSTHPI, CSUSHPISA ©, MORTGAGE30US ©, DRSFRMACBS |
| 글로벌 정책금리 · 환율 | 2 | DEXCHUS, DTWEXEMEGS |
| 원자재 | 8 | DCOILWTICO, DCOILBRENTEU, DHHNGSP, PCOPPUSDM, PALUMUSDM, PIORECRUSDM, PWHEAMTUSDM, PALLFNFINDEXM |

수익률 곡선용 DGS1MO~DGS30(11개)과 경기침체 음영용 USREC도 받습니다.

FOMC 경제전망 패널은 FEDTARMD, GDPC1MD, UNRATEMD, PCECTPIMD, JCXFEMD(목표 연도별
중앙값)와 FEDTARMDLR, GDPC1MDLR, UNRATEMDLR, PCECTPIMDLR(장기 중앙값) 9개를
`realtime_start`/`realtime_end`로 받아 최신 발표와 직전 발표를 비교합니다. 엔드포인트는 `series/observations`, `releases/dates` 두 개를 씁니다.

- ICE 시계열(BAMLH0A0HYM2, BAMLC0A0CM)은 FRED가 최근 3년치만 제공합니다.
- 2025년 10월 CPS 계열(UNRATE 등)은 셧다운으로 결측(".")입니다. 파서가 "."을 결측으로 처리합니다(`sources.parse_number`).

### 미사용: 대시보드에 쓸 만한 것

**금리·단기자금**

| ID | 이름 | 주기 | 확인 |
|---|---|---|---|
| DFEDTARL | 연방기금 목표 하단 | D | ? |
| SOFR, EFFR | SOFR, 실효 연방기금금리 (지금은 뉴욕 연은에서 직접 받음) | D | ✔ |
| DPRIME | 프라임 대출금리 | D | ? |
| DTB3 | 3개월 T-bill | D | ? |
| DFII5 | 5년 TIPS 금리 | D | ? |

**연준 대차대조표 (H.4.1)**

| ID | 이름 | 주기 | 확인 |
|---|---|---|---|
| WLRRAL | 역레포 부채 | W | ? |
| WCURCIR | 화폐발행 | W | ? |
| WLCFLPCL | 재할인창구 대출 | W | ? |
| BOGMBASE | 본원통화 | M | ? |

**은행·신용 (H.8, G.19, 연체율, SLOOS)**

| ID | 이름 | 주기 | 확인 |
|---|---|---|---|
| TOTLL | 대출·리스 | W | ? |
| CONSUMER | 소비자 대출 | W | ? |
| DPSACBW027SBOG | 상업은행 예금 | W | ? |
| REVOLSL, NONREVSL | 소비자신용 총액·리볼빙·비리볼빙 | M | ? |
| DRBLACBS | 기업대출 연체율 | Q | ? |
| DRTSCIS | 소기업 C&I 대출기준 강화 순비율 | Q | ✔ |
| DRTSCLCC | 신용카드 대출기준 강화 순비율 | Q | ? |

**금융여건·통화**

| ID | 이름 | 주기 | 확인 |
|---|---|---|---|
| ANFCI, NFCICREDIT | 조정 NFCI, NFCI 신용 하위지수 | W | ? |
| KCFSI | 캔자스시티 연은 금융스트레스지수 | M | ? |
| M1SL, M2V | M1, M2 유통속도 | M/Q | ? |

**환율 (H.10)**

| ID | 이름 | 주기 | 확인 |
|---|---|---|---|
| DEXKOUS | 원/달러 (뉴욕 정오) | D | ✔ |
| DEXJPUS, DEXUSEU, DEXUSUK | 엔·유로·파운드 | D | ? |
| DTWEXAFEGS | 달러지수 (선진국 / 신흥국 대비) | D | ? |

**고용**

| ID | 이름 | 주기 | 확인 |
|---|---|---|---|
| IC4WSA | 신규 실업수당 4주 평균 | W | ? |
| JTSHIR, JTSLDR | JOLTS 자발적 이직률·채용률·해고율 | M | ? |
| AWHAETP | 평균 주간 근로시간 | M | ? |
| TEMPHELPS | 임시직 고용 | M | ? |
| MANEMP | 제조업 고용 | M | ? |
| OPHNFB | 단위노동비용, 노동생산성 | Q | ? |

**국민계정·소득·재정·대외**

| ID | 이름 | 주기 | 확인 |
|---|---|---|---|
| GDPC1 | 실질 GDP 수준 | Q | ? |
| DSPIC96 | 실질 가처분소득 | M | ? |
| CP | 세후 기업이익 | Q | ? |
| TDSP | 가계 원리금상환/가처분소득 | Q | ? |
| BOPGSTB | 상품·서비스 무역수지 | M | ✔ |
| IEABC | 경상수지 | Q | ? |

**생산·주문·재고·주택**

| ID | 이름 | 주기 | 확인 |
|---|---|---|---|
| IPMAN | 제조업 산업생산 | M | ? |
| RRSFS | 실질 소매판매 | M | ? |
| TTLCONS | 건설지출 | M | ? |
| BABATOTALSAUS | 신규 사업자 신청 | M | ? |
| HPIPONM226S | FHFA 주택가격지수 (Case-Shiller 공공 대안) | Q/M | ? |

**물가 세부**

| ID | 이름 | 주기 | 확인 |
|---|---|---|---|
| TRMMEANCPIM158SFRBCLE | 16% 절사평균 CPI | M | ? |
| IR | 수입물가 | M | ? |
| GASREGW | 휘발유 소매가 (EIA) | W | ? |

### 미사용: 전망 · 모델 · 민간 지수

| ID | 이름 | 참고 |
|---|---|---|
| STLENI | 세인트루이스 연은 경제뉴스지수 | 나우캐스트 |
| GDPPOT, NROU | CBO 잠재 GDP, 자연실업률 | 미래 구간 포함 |
| EXPINF1YR, RECPROUSM156N | 클리블랜드 연은 기대인플레, 경기침체 확률 | 모델 추정 |
| DJIA | 다우존스 | S&P DJI © |
| CSUSHPINSA | Case-Shiller (원계열) | 계절조정 CSUSHPISA를 사용중 |
| EXHOSLUSM495S | 기존주택 판매 | NAR ©, 최근 1년치만 |
| BAMLC0A4CBBB | ICE BBB 스프레드 | ICE ©, 3년치만 |
| AAA10Y, DAAA, DBAA | Moody's 회사채 | Moody's © |

### 쓰지 않는 엔드포인트

- `release/series`: SLOOS나 H.8처럼 한 발표에 속한 시계열 목록을 한 번에 조회합니다.
- `series`: 메타데이터를 받습니다. notes 안의 Copyright 문구로 ©를 자동 표시할 수 있습니다.
- `series/observations`의 `units`, `frequency` 파라미터: 변환과 집계를 서버에서 처리합니다.
- ALFRED: 발표 당시 값과 수정 이력을 받습니다.
- Maps API: 주별 실업률 같은 지역 데이터를 받습니다.

---

## 2. 미 재무부 Fiscal Data (키 없음)

### 사용중: 4개

| 패널 | 개수 | 코드 |
|---|---|---|
| 연준 대차대조표 · 재정 | 4 | debt_to_penny, debt_held_public, avg_interest_rates, avg_interest_rates/bills |

### 미사용: 쓸 만한 것

| 엔드포인트 | 내용 | 주기 | 확인 |
|---|---|---|---|
| `v1/accounting/dts/operating_cash_balance` | TGA 잔액, 일간 (지금은 FRED 주간 WTREGEN을 씀) | D | ✔ |
| `v1/accounting/dts/deposits_withdrawals_operating_cash` | 세수·지출 항목별 입출금 (원천징수세, 관세 등) | D | ✔ |
| `v1/accounting/dts/debt_subject_to_limit` | 부채한도 대상 부채와 한도 | D | ✔ |
| `v1/accounting/mts/mts_table_1` | 월간 세입·세출·재정수지 | M | ✔ |
| `v1/accounting/mts/mts_table_4` | 세입 원천별 (관세, 법인세 등) | M | ? |
| `v2/accounting/od/interest_expense` | 국채 이자비용 | M | ✔ |
| `v1/accounting/od/auctions_query` | 국채 입찰 결과 (낙찰금리, 응찰률, 간접 응찰 비중) | 입찰 시 | ✔ |
| `v1/accounting/od/buybacks_operations` | 국채 바이백 | 수시 | ✔ |
| `v1/debt/mspd/mspd_table_1` | 시장성·비시장성 부채 구성 (단기채 비중) | M | ? |

### 미사용: 나머지

- `rates_of_exchange`: 재무부 보고용 분기 환율이라 시장 환율이 아닙니다.
- `gold_reserve`: 장부가로 고정돼 있어 의미가 없습니다.
- 저축채권 계열: 개인 저축 상품 데이터라 매크로 가치가 낮습니다.

참고: 국채 수익률 곡선(Par Yield Curve)은 Fiscal Data API가 아니라
home.treasury.gov에서 XML로 제공합니다. 지금은 FRED의 DGS 시리즈로 같은 데이터를
받고 있어서 추가할 필요가 없습니다.

---

## 3. 뉴욕 연은 Markets API (키 없음)

### 사용중: 4개

| 패널 | 개수 | 코드 |
|---|---|---|
| 정책금리 · 국채 · 단기자금 | 4 | EFFR, OBFR, SOFR, TGCR |

### 미사용: 쓸 만한 것

| 엔드포인트 | 내용 | 주기 | 확인 |
|---|---|---|---|
| `/api/rates/all/latest.json` | EFFR, OBFR, SOFR, BGCR, TGCR를 한 번에 조회 | D | ✔ |
| `/api/rates/secured/tgcr`, `bgcr` | 3자간 레포 금리 (SOFR−TGCR = 레포시장 긴장) | D | △ |
| `/api/rates/unsecured/obfr` | 오버나이트 은행 자금금리 | D | △ |
| `/api/rates/secured/sofrai` | SOFR 30·90·180일 평균, SOFR Index | D | ? |
| `/api/rp/reverserepo/...`, `/api/rp/all/all/results/...` | ON RRP 결과 (낙찰액, 참가자 수) | D | △ |
| `/api/rp/repo/...` | 레포 운영, 상설 레포(SRF) 이용액 | D | ? |
| `/api/soma/summary.json` | SOMA 보유 요약 (국채·MBS, QT 진행) | W | △ |
| `/api/pd/...` | 프라이머리 딜러 국채 포지션 | W | △ |

### 미사용: 나머지

- 증권대차, 중앙은행 통화스왑(평시에는 거의 0), 가이드시트, 딜러 시장점유율은 가치가 낮습니다.
- newyorkfed.org에서 파일로만 제공하는 지표가 있습니다(Markets API 아님).
  - GSCPI(공급망 압력지수), ACM 기간프리미엄, 경기침체 확률, Staff Nowcast, R-star가 있습니다.
  - API가 아니라 엑셀·CSV 파일이라 파서를 따로 만들어야 합니다.

---

## 4. 한국은행 ECOS

### 사용중: 26개

| 패널 | 개수 | 코드 |
|---|---|---|
| 주식 · 변동성 | 2 | 802Y001/0001000, 802Y001/0089000 |
| 한국 금리 · 금융 | 11 | 722Y001/0101000, 817Y002/010101000, 817Y002/010502000, 817Y002/010200000, 817Y002/010210000, 817Y002/010300000, 731Y001/0000001, 731Y001/0000002, 732Y001/99, 161Y006/BBHA00, 151Y001/1000000 |
| 한국 성장 · 물가 · 경기 | 13 | 200Y102/10111, 901Y009/0, 901Y010/QB, 404Y014/*AA, 901Y118/T002, 901Y118/T004, 301Y017/SA000, 901Y027/I61BC/I28B, 901Y027/I61E/I28B, 901Y067/I16D, 901Y067/I16E, 511Y002/FME/99988, 513Y001/E2000 |

처음 쓰던 9개는 공개 저장소들의 실제 호출 사례와 코드가 일치했습니다. 이번에 추가한 코드는 아래 확인 표시를 기준으로 골랐습니다(고용률 `I61E/I28B`, 코스피·코스닥 `802Y001`은 검증이 약한 편).

### 미사용: 쓸 만한 것

**금리·통화·신용**

| 통계표 / 항목 | 주기 | 이름 | 확인 |
|---|---|---|---|
| 817Y002 / 010320000 | D | 회사채 3년 BBB− | ✔ |
| 817Y002 / 010190000, 010195000, 010200001, 010220000, 010230000 | D | 국고채 1·2·5·20·30년 (수익률 곡선) | ✔ |
| 121Y006 / BECBLA01 | M | 예금은행 대출금리 (신규) — 의미가 출처마다 다름 | △ |
| 161Y005 / BBHS00 | M | M2 평잔 (계절조정) | ✔ |
| 151Y005 / 1110000 | M | 예금취급기관 가계대출 | △ |
| 104Y016 / BDCA1 | M | 예금은행 총대출금 | ✔ |

**환율·대외**

| 통계표 / 항목 | 주기 | 이름 | 확인 |
|---|---|---|---|
| 731Y001 / 0000053 | D | 원/위안 | ? |
| 731Y004 / 0000001 / 0000100 | M | 원/달러 월평균 | ✔ |
| 301Y013 / 000000 | M | 경상수지 (원계열) | △ |

**물가**

| 통계표 / 항목 | 주기 | 이름 | 확인 |
|---|---|---|---|
| 901Y010 / DB | M | 근원 CPI (식료품·에너지 제외) | ✔ |
| 401Y015 / *AA / W | M | 수입물가지수 (원화) | ✔ |
| 402Y014 / *AA / W | M | 수출물가지수 (원화) | ✔ |
| 511Y003 / FMB | M | 기대인플레이션 (향후 1년) | △ |

**성장·산업활동**

| 통계표 / 항목 | 주기 | 이름 | 확인 |
|---|---|---|---|
| 200Y104 / 1400 | Q | 실질 GDP 계절조정 수준 | ✔ |
| 901Y033 / A00 | M | 전산업생산지수 | ✔ |
| 901Y100 / G0 | M | 소매판매액지수 | ✔ |
| 901Y104 / I48A | M | 건설기성액 | ✔ |
| 901Y032 / I11AC | M | 광공업생산지수 | △ |
| 901Y038 / I51A | M | 서비스업생산지수 | △ |
| 901Y066 | M | 설비투자지수 (항목코드 출처마다 다름) | ? |

**경기·고용·심리**

| 통계표 / 항목 | 주기 | 이름 | 확인 |
|---|---|---|---|
| 901Y027 / I61BA / I28A | M | 취업자수 (원계열) | ✔ |
| 511Y002 / FMFB / 99988 | M | 주택가격전망 CSI | △ |

**부동산·해외**

| 통계표 / 항목 | 주기 | 이름 | 확인 |
|---|---|---|---|
| 901Y062 | M | 주택매매가격지수 (항목코드 출처마다 다름) | ? |
| 901Y063 / P64A | M | 주택전세가격지수 | △ |
| 901Y074 | M | 미분양주택 | ? |
| 902Y006 / US, JP, CN, XM | M | 주요국 정책금리 | ✔ |
| 902Y008 / 국가코드 | M | 주요국 CPI | ✔ |

### 미사용: 나머지

- 901Y014, 902Y002: 월간 주가 통계와 OECD 주요국 주가지수입니다. 일별 코스피·코스닥(802Y001)을 이미 씁니다.
- 101Y003, 101Y004: 개편 전 M2 통계표입니다. 현행은 161Y005~161Y008입니다.
- 511Y001, 512Y017, 512Y018: 끝난 계열입니다.
- 512Y014 (BSI 전망): 갱신 중단과 무효 조합 사례가 보고됐습니다. BSI는 넣기 전에 StatisticItemList로 END_TIME을 확인하세요.

---

## 5. ECB Data Portal (키 없음)

**2026년에 바뀐 점**
- **ICP 데이터셋이 2026-02-04에 중단되고 HICP 데이터셋으로 바뀌었습니다.** ✔
- STS 데이터셋은 STBS로, SEC는 CSEC로 바뀌었습니다.
- 불가리아가 2026-01-01에 유로존에 가입해, 고정구성 유로존 코드에 I10(EA21)이 새로 생겼습니다.
  - 변동구성 코드 U2를 쓰는 키는 영향이 없습니다.
  - I9(EA20)를 쓰는 키는 갱신이 멈췄을 수 있습니다.

### 사용중: 16개

| 패널 | 개수 | 코드 |
|---|---|---|
| 유로존 | 15 | FM/DFR, FM/MRR_FR, EST/WT, YC/SR_10Y, YC/SRS_10Y_3M, HICP/U2/ANR, HICP/XEF000, HICP/SERV00, BSI/M30, BSI/A20T/2240, LFSI/UNEHRT, STBS/PROD, CISS/SS_CIN, EXR/USD, EXR/E03 |
| 글로벌 정책금리 · 환율 | 1 | EXR/JPY÷USD |

유로존 HICP는 중단된 `ICP` 대신 `HICP` 데이터셋으로 받습니다. USD/JPY는 EUR/JPY ÷ EUR/USD로 계산합니다.

### 미사용: 쓸 만한 것

| 키 | 이름 | 주기 | 확인 |
|---|---|---|---|
| `YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y` | AAA 국채 2년 | D | ? |
| `IRS/M.I10.L.L40.CI.0000.EUR.N.Z` | 유로존 장기금리 (국가별로 바꾸면 BTP−Bund) | M | ✔ |
| `HICP/M.U2.N.NRGY00.4D0.ANR` | 에너지 HICP YoY | M | ✔ |
| `BSI/M.U2.Y.U.A20T.A.I.U2.2250.Z01.A` | 가계 대출 증가율 | M | △ |
| `MNA/Q.N.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N` | 실질 GDP (원계열 수준) | Q | ✔ |
| `RESR/Q.I9._T.N._TR.TVAL.4F0.TB.N.IX` | 주택가격지수 | Q | ✔ |
| `BLS/Q.U2.ALL.LE.E.Z.B3.ST.S.WFNET` | 은행 대출태도 (대기업) | Q | ✔ |
| `ILM/W.U2.C.T000000.Z5.Z01` | 유로시스템 총자산 | W | ✔ |
| `EXR/D.{GBP,CNY,KRW,CHF}.EUR.SP00.A` | 다른 통화 참조환율 | D | △ |

### 미사용: 나머지

- SPF(전문가 전망), MPD(스태프 전망), AME(AMECO): 전망 데이터셋입니다. SPF는 키가 예측자 단위라 구조가 복잡하고, AMECO는 연간입니다.
- ICP, STS, SEC, CBD: 중단된 데이터셋입니다.
- CSEC, IVF, QSA, TRD, CBD2, 지급결제·은행권·감독 통계: 너무 세부적이거나 늦게 나옵니다.

---

## 6. BIS Data Portal (키 없음)

### 사용중: 17개

| 패널 | 개수 | 코드 |
|---|---|---|
| 글로벌 정책금리 · 환율 | 6 | WS_CBPOL/JP, WS_CBPOL/GB, WS_CBPOL/CN, WS_CBPOL/CA, WS_CBPOL/AU, WS_CBPOL/CH |
| 글로벌 신용 · 주택 · 실효환율 | 11 | WS_TC/Q.KR.H.A.M.770.A, WS_TC/Q.KR.P.A.M.770.A, WS_TC/Q.CN.P.A.M.770.A, WS_TC/Q.US.H.A.M.770.A, WS_CREDIT_GAP/Q.KR.P.A.C, WS_CREDIT_GAP/Q.CN.P.A.C, WS_SPP/Q.KR.R.628, WS_SPP/Q.US.R.628, WS_SPP/Q.CN.R.628, WS_EER/M.R.B.KR, WS_EER/M.R.B.US |

### 미사용: 쓸 만한 것

| 데이터셋 / 예시 키 | 이름 | 주기 | 확인 |
|---|---|---|---|
| `WS_EER/M.N.B.US` | 명목실효환율 | M | ? |
| `WS_DSR/Q.KR.H` | 가계 부채상환비율 | Q | ? |
| `WS_LONG_CPI/M.KR.771` | 장기 CPI YoY | M | ? |
| `WS_XRU/M.KR.KRW.A` | 달러 대비 환율 | M | ? |
| `WS_GLI` | 글로벌 유동성 지표 | Q | 데이터셋만 ✔ |

### 미사용: 나머지

- CPMI 지급결제 통계: 연간이고 늦게 나옵니다.
- 3년 주기 외환 서베이: 시계열 대시보드에 맞지 않습니다.
- 장외·거래소 파생상품: 니치 지표입니다.
- LBS·CBS 세부 시리즈: 키가 12차원이라 유지보수 비용이 큽니다.
- DPP: 국가별 정의가 제각각이라 SPP로 충분합니다.

---

## 7. OECD Data Explorer (키 없음)

### 사용중: 13개

| 패널 | 개수 | 코드 |
|---|---|---|
| 한국 성장 · 물가 · 경기 | 1 | CLI/KOR |
| 글로벌 경기 · 심리 | 12 | CLI/USA, CLI/CHN, CLI/JPN, CLI/DEU, CLI/GBR, BCICP/USA, BCICP/CHN, BCICP/KOR, BCICP/DEU, CCICP/USA, CCICP/KOR, CCICP/DEU |

OECD는 시간당 60회 제한이 있어 경기선행지수 6개국을 한 번에, 기업·소비자 신뢰지수를 한 번에 받습니다(갱신당 2회).

### 미사용: 쓸 만한 것

| 데이터셋 / 예시 키 | 이름 | 주기 | 확인 |
|---|---|---|---|
| `DSD_STES@DF_CLI` / `USA+CHN+KOR.M.BCICP...AA...H` | 기업신뢰지수 (PMI 대체 공공 지표) | M | △ |
| `DSD_STES@DF_CLI` / `….M.CCICP...AA...H` | 소비자신뢰지수 | M | △ |
| `DSD_STES@DF_CLI` / `JPN+DEU+GBR+IND+BRA+G7.M.LI...AA...H` | CLI 국가·권역 확장 | M | ✔ |
| `OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL` / `….M.N.CPI.PA._T.N.GY` | 국가별 CPI YoY | M | ✔ |
| 같은 데이터셋 / `….M.N.CPI.PA._TXCP01_NRG.N.GY` | 국가별 근원 CPI | M | ? |
| `OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M` | 국가별 월간 실업률 | M | △ |
| `OECD.SDD.STES,DSD_STES@DF_FINMARK` | 국가별 장기·단기 금리 | M | △ |
| `OECD.SDD.STES,DSD_KEI@DF_KEI` | 주요 단기지표 묶음 | M/Q | 데이터셋만 ✔ |
| `OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA` | 분기 GDP 성장률 | Q | 데이터셋만 ✔ |
| `OECD.SDD.TPS,DSD_IMTS@DF_IMTS` | 월간 상품무역 | M | 데이터셋만 ✔ |

### 미사용: 나머지

- Economic Outlook (`DSD_EO@DF_EO`) 계열: 반기 전망 데이터셋으로, 연간 전망치가 미래 연도에 찍혀 있습니다.
- 분석용 주택가격: 갱신 중단 보고가 있고 BIS와 겹칩니다.
- 연간 국민계정, 조세·임금·PPP: 연간 구조 지표입니다.
- BTS/CS 원자료: CLI 데이터셋의 BCI/CCI로 대체할 수 있습니다.
