# FinAgent

거시 경제 지표 모니터링, ETF 비교 분석, 포트폴리오 위험·수익 정량 분석, AI 리서치를 하나로 묶은 로컬 개인 투자 터미널입니다. 공개 데이터 API(FRED, 한국은행 ECOS, ECB, BIS, OECD, SEC EDGAR)와 시세 데이터를 로컬에서 직접 수집·가공하며, 모든 거래 내역과 설정은 외부 서버 전송 없이 로컬 환경에서만 관리됩니다.

## 주요 기능

- **거시 경제 대시보드**
  - 미국·한국·유로존·일본·중국·글로벌 주요 경제 지표(27개 패널, 162개 시계열)를 동일 기준(전년비·실질 등)으로 병렬 비교
  - 국면 요약(근원 PCE, 삼룰, 장단기 금리차, VIX), 미·한·유로 수익률 곡선, FOMC 경제전망(SEP), 5대 중앙은행 정책금리 추이 및 주요 지표 발표 일정 제공
- **ETF 탐색 및 비교**
  - 미국 상장 대표 ETF 125종(22개 분류·80개 카테고리) 기본 유니버스 제공 및 사용자 종목 추가·제거 지원
  - 운용자산(AUM) 및 기간별 수익률 트리맵 시각화, SEC 공시(N-PORT, N-CSR) 기반 보유 종목·자산군·섹터·신용등급 비중 및 실제 총보수 확인
- **포트폴리오 정량 분석**
  - 주식 분할을 반영한 거래 장부 재생(이동평균 원가, 실현·미실현 손익) 및 시간가중(TWR)·금액가중(MWR) 수익률 산출
  - 초과수익률 기반 위험 지표(샤프, 소르티노, 최대 낙폭, 베타, 상관계수, 분산효과) 및 보유 ETF 구성 종목을 펼친 실질 보유(Look-through) 위험 기여도 분석
  - Black–Litterman 기대수익률 추정, 블록 부트스트랩 몬테카를로 시뮬레이션, 신규 종목 편입 What-if 분석, SEC 13F 기관 투자자 포트폴리오 복제·비교
- **멀티 스레드 AI 분석**
  - Claude, OpenAI, Gemini 모델을 지원하는 조정자(Coordinator) 및 병렬 분석 스레드 아키텍처
  - 화면 컨텍스트와 전용 프리셋(매크로·종목·ETF)을 기반으로 내부 데이터 조회 도구(시계열·시세·SEC 재무 및 공시)와 웹 검색을 결합하고, 응답 내 지표·웹 출처 인용을 검증

## 시작하기

### 요구 사항

- Python 3.12 이상
- [uv](https://docs.astral.sh/uv/) 패키지 매니저

### 설치

```bash
# 전체 의존성 설치 (개발 도구, AI 공급자 SDK, yfinance 포함)
uv sync

# 최소 실행 환경 + 필요한 확장만 선택 설치할 경우 (선택지: claude, openai, gemini, llm, yahoo)
uv sync --no-dev --extra llm --extra yahoo
```

### 환경 변수 설정

프로젝트 루트에 `.env.example`을 복사하여 `.env` 파일을 생성하고 API 키를 입력합니다.

```bash
cp .env.example .env
```

| 환경 변수 | 필수 여부 | 용도 및 발급처 |
|---|---|---|
| `FRED_API_KEY` | **필수** | 미국 및 글로벌 거시 경제 지표 ([FRED](https://fred.stlouisfed.org/docs/api/api_key.html)) |
| `ECOS_API_KEY` | **필수** | 한국은행 경제통계시스템 지표 ([ECOS](https://ecos.bok.or.kr/api/)) |
| `ANTHROPIC_API_KEY` | 선택 | Claude 기반 AI 분석 |
| `OPENAI_API_KEY` | 선택 | OpenAI 기반 AI 분석 |
| `GEMINI_API_KEY` | 선택 | Gemini 기반 AI 분석 |
| `SEC_USER_AGENT` | 선택 | SEC EDGAR 공시(13F, N-PORT, N-CSR, 기업 재무) 조회 시 필요한 영문 이름·이메일 (예: `"Hong Gildong hong@example.com"`) |
| `ALPACA_API_KEY_ID`<br>`ALPACA_API_SECRET_KEY` | 선택 | 미국 상장 종목 일봉·분할·분배금 공식 수집 ([Alpaca](https://app.alpaca.markets/signup), 미설정 시 Yahoo Finance 대체) |
| `OPENFIGI_API_KEY` | 선택 | SEC 공시 내 CUSIP → 티커 변환 요청 한도 확장 ([OpenFIGI](https://www.openfigi.com/user/profile/api-key)) |

> API 키 없이 UI와 레이아웃만 확인하려면 `--demo` 모드로 실행할 수 있습니다.

### 실행

```bash
# 기본 실행 (http://127.0.0.1:8765)
uv run macro

# 데모 모드 (네트워크 요청 없이 전 지표 고정값으로 UI 확인)
uv run macro --demo

# 서버 기동 없이 데이터 출처별 수신 상태 및 AI 공급자 설정 점검
uv run macro --check
```

**주요 CLI 옵션**
- `--port <PORT>`: 실행 포트 지정 (기본값: `8765`)
- `--no-browser`: 실행 시 브라우저 자동 열기 생략
- `--env-file <PATH>`: 환경 변수 파일 경로 지정 (기본값: `./.env`)
- `--settings <PATH>`: 설정 파일 경로 지정 (기본값: `./finagent.settings.json`)
- `--version`: 버전 출력

## 프로젝트 구조

```text
FinAgent_Terminal/
├── src/finagent/
│   ├── cli.py                 # CLI 진입점 (macro 명령어 및 환경·수신 점검)
│   ├── app.py                 # Streamlit 앱 진입점 및 네비게이션 라우팅
│   ├── catalog.py             # 거시 경제 지표·패널·수익률 곡선 카탈로그 정의
│   ├── sources.py / fetch.py  # FRED·ECOS·SDMX(ECB·BIS·OECD) 수집 및 요청 제한 제어
│   ├── prices.py / markets.py # 통합 시세 저장소 (Alpaca·Yahoo 라우팅 및 증분 캐싱)
│   ├── sec.py / investors.py  # SEC EDGAR 공시 수집 및 13F 기관 투자자 분석
│   ├── etf.py / etf_universe.py # ETF 유니버스, 수익률 트리맵, N-PORT·N-CSR 분석
│   ├── portfolio.py / quant.py / perf.py # 거래 장부 재생, 성과·위험 정량 계산, 시뮬레이션
│   ├── memory.py              # 프로세스 인메모리 데이터 저장소
│   ├── analysis/              # 멀티 스레드 AI 분석 엔진, 도구(Toolbox), 공급자 어댑터
│   └── ui/                    # 페이지별 Streamlit 렌더링 컴포넌트 및 차트·팔레트
├── docs/                      # 도메인별 상세 설계 및 지표 규격 문서
├── data/user/                 # 사용자 로컬 데이터 (포트폴리오 장부, 관심 투자자 목록)
└── tests/                     # 단위 및 통합 테스트
```

### 데이터 관리 원칙

- **인메모리 캐시**: 수집된 거시 시계열, 일봉 시세, SEC 공시 원문, AI 대화 및 리서치 보고서는 프로세스 메모리(`memory.Store`)에만 유지되며 앱 종료 시 소멸합니다.
- **로컬 디스크 격리**: 디스크에는 사용자가 직접 입력한 포트폴리오 거래 내역(`data/user/portfolios.json`), 추적 투자자 목록(`data/user/investors.json`), 앱 설정(`finagent.settings.json`, API 키 미포함)만 저장됩니다.

## 개발 및 테스트

```bash
# 린트 및 코드 포맷 검사
uv run ruff check .
uv run ruff format --check .

# 정적 타입 검사
uv run basedpyright

# 테스트 실행
uv run pytest -q

# 코드 수정 시 즉시 반영되는 Streamlit 개발 모드 실행
uv run streamlit run src/finagent/app.py -- --demo
```