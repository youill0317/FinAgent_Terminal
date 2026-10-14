# FinAgent

공공·공개 데이터만으로 만든 매크로 대시보드입니다. 내 컴퓨터에서 작은 서버를
띄우고(127.0.0.1) 브라우저에서 미국·한국·글로벌 매크로 지표를 한 화면에
보여줍니다.

A local macro dashboard built only on public and open data (FRED, U.S.
Treasury, New York Fed, Bank of Korea ECOS, ECB, BIS, OECD). It serves a
terminal-style page on `127.0.0.1` and keeps everything it fetches in memory.

<!-- 스크린샷: uv run finagent --demo 화면 -->

## 설치와 실행

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync
```

무료 API 키 두 개를 발급받습니다.

- FRED: <https://fredaccount.stlouisfed.org/apikeys>
- 한국은행 ECOS: <https://ecos.bok.or.kr/api/#/AuthKeyApply>

키는 환경변수로 넣거나, 실행하는 폴더의 `.env` 파일에 적습니다.
`.env`는 `.gitignore`에 들어 있고 앱은 이 파일을 읽기만 합니다.
같은 키가 양쪽에 있으면 환경변수가 우선합니다.

```bash
# .env
FRED_API_KEY=...
ECOS_API_KEY=...
```

```bash
uv run finagent              # http://127.0.0.1:8765 를 띄우고 브라우저를 엽니다
uv run finagent --demo       # 키·네트워크 없이 예시 데이터로 화면 확인
uv run finagent --check      # 서버 없이 한 번 받아와서 출처별 상태 출력 (모두 정상이면 종료 코드 0)
```

옵션: `--port`, `--no-browser`, `--env-file PATH`, `--version`.
서버는 항상 `127.0.0.1`에만 열립니다.

## 데이터 원칙

- **공공·공개 출처.** 아래 7개 출처의 API에서 받는 시계열을 씁니다. 공식 통계뿐
  아니라 같은 출처가 제공하는 나우캐스트(GDPNow), 주가지수(S&P 500, 나스닥,
  코스피), 민간 지수(미시간대 소비자심리, Case-Shiller, ICE 스프레드)도 씁니다.
  제3자 저작권 시계열은 코드 옆에 © 출처를 적습니다.
- **저장하지 않음.** 받은 시계열은 앱이 켜져 있는 동안 메모리에만 있고 디스크에
  쓰지 않습니다.
- **수동 갱신.** 앱을 시작할 때 한 번, 그리고 화면의 새로고침 버튼이나 `R` 키를
  누를 때만 받아옵니다. 타이머로 자동 갱신하지 않습니다.
- 키가 없거나 응답하지 않는 출처는 화면에 따로 표시되고, 나머지 지표는 그대로
  나옵니다. 일시적 오류(시간 초과, 연결 실패, 5xx)는 한 번 다시 시도합니다.
- FRED는 키당 분당 약 120회까지 받습니다. 한 번 갱신에 약 105회를 부르므로
  새로고침은 1분에 한 번 정도로 하세요.

## 출처

| 출처 | 키 | 화면 |
|---|---|---|
| FRED (세인트루이스 연은) | `FRED_API_KEY` | 미국 성장·가계·고용·물가·금리·신용·주택·연준 대차대조표, 주가지수, 수익률 곡선, 10Y−2Y, 원자재, 달러, 발표 일정 |
| 미 재무부 Fiscal Data | 없음 | 연방정부 총부채·민간 보유분, 국채 평균 조달금리 |
| 뉴욕 연은 Markets API | 없음 | EFFR, OBFR, SOFR, TGCR |
| 한국은행 ECOS | `ECOS_API_KEY` | 기준금리, 시장금리, 환율, 외환보유액, M2, 가계신용, GDP, 물가, 무역, 경상수지, 고용, 경기·심리지수, 코스피·코스닥 |
| ECB Data Portal | 없음 | 정책금리, €STR, 국채금리, HICP, M3, 대출, 실업률, 산업생산, CISS, 환율 |
| BIS Data Portal | 없음 | 주요국 정책금리, 신용/GDP, 신용갭, 실질 주택가격, 실질실효환율 |
| OECD Data Explorer | 없음 | 경기선행지수, 기업·소비자 신뢰지수 |

출처마다 제공하는 데이터와 아직 쓰지 않는 시계열은
[`docs/data-sources.md`](docs/data-sources.md)에 정리돼 있습니다.

## 지표 추가

모든 지표는 `src/finagent/catalog.py` 한 파일에 있습니다. 패널의 튜플에 한 줄을
추가하면 됩니다.

```python
fred("us_mfg_tcu", "제조업 설비가동률", "MCUMFN", "M", PCT1),
```

`--demo`와 테스트가 새 지표도 받을 수 있도록 `src/finagent/demo.py`의 해당 출처
표에 가짜 시계열 한 줄을 함께 추가합니다. 변환은 `"level"`, `"yoy"`, `"pct"`(전기 대비 %), `"diff"`, `"sum12"`(12개월 합)
중에서 고릅니다. 상단 줄에 넣을 지표는 `TAPE`에 id를 추가합니다.

## 개발

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest -q
```

테스트는 실제 API를 부르지 않습니다. `src/finagent/demo.py`가 각 출처의 실제 응답
형식을 흉내 낸 가짜 응답을 만들고, `--demo`와 테스트가 이것을 씁니다. 이 저장소를
만든 환경에서는 데이터 API에 접속할 수 없어서 실제 응답으로는 확인하지 못했습니다.
처음 실행할 때 `uv run finagent --check`로 출처별 상태를 확인하세요.
