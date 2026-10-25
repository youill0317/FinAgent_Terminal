# FinAgent

- 로컬 Streamlit 앱(127.0.0.1): 미국·한국·글로벌 거시 지표 대시보드
- 선택 기능: S&P 500·400 스크리닝, 포트폴리오 성과·위험 분석, 13F 투자자 복제, 생성 모델 경로
- 화면·계산 모두 Python(numpy·pandas), 생성 모델 학습만 Colab

## 실행

```bash
uv sync                          # 개발 의존성 포함 전체 설치
uv sync --no-dev --extra claude  # 필요한 것만: claude·openai·gemini·llm·screen·garch·kronos
source .venv/bin/activate

macro                # http://127.0.0.1:8765
macro --demo         # 키·네트워크 없이 예시 데이터
macro --check        # 서버 없이 출처별 상태 출력
# 옵션: --port --no-browser --env-file PATH --settings PATH --version
```

- 필수 환경변수: `FRED_API_KEY`, `ECOS_API_KEY`
- 선택: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`(또는 `GOOGLE_API_KEY`), `SEC_USER_AGENT`
- 추가 의존성: `screen`(yfinance, lxml), `garch`(arch, GJR-GARCH), `kronos`(torch 외)
- 생성 모델 학습: `colab` CLI(`uv tool install google-colab-cli`)와 로그인
- 출처별 시계열: [`docs/data-sources.md`](docs/data-sources.md)

## 분석 구성

### 공통 규칙

- 수익률: 수정종가 단순 일간 수익률, 기준통화(KRW·USD)로 환산 후 계산
- 손실: 양수 비율(VaR 0.02 = 2% 손실)
- 추정 기간: 6M·1Y·3Y, 전망 기간(시뮬레이션) 1M·3M·6M·1Y = 21·63·126·252일
- 연환산 관측 수: 고정 252가 아닌 표본 자체의 연간 관측 수

$$
P=\frac{\#\text{간격}\times 365}{\sum \Delta d},\qquad \Delta d\le 31\text{일인 간격만}
$$

- 한·미 혼합 포트폴리오: 두 거래일 합집합이라 약 258
- 1년(365일) 미만 기간: 연환산 생략, 누적값만
- 무위험수익률: KRW 기준 CD 91일(단리), USD 기준 ^IRX(할인율), 없으면 0

$$
R^{\text{CD}}=\left(1+r\tfrac{91}{365}\right)^{365/91}-1,\qquad
R^{\text{IRX}}=\left(1-d\tfrac{91}{360}\right)^{-365/91}-1,\qquad
r^f_t=(1+R)^{1/P}-1
$$

- 난수: 모든 모의 `SEED = 42`

### 성과

- 원가: 이동평균법, 매수 수수료는 원가에, 매도 수수료는 대금에서 차감
- 외화 거래: 거래일 환율, 평가는 최근 환율
- 시간가중수익률(TWR): 순유입은 하루 시작, 순유출은 하루 끝 처리

$$
r_t=\frac{V_t-V_{t-1}-F_t}{V_{t-1}+\max(F_t,0)},\qquad \text{TWR}=\prod_t(1+r_t)-1
$$

- 금액가중수익률(MWR, XIRR): 전 구간 로그 성장 $h$ 로 풀이

$$
\text{NPV}(h)=\sum_i c_i\,e^{-h\,t_i/T}=0,\qquad \text{연율}=e^{365h/T}-1
$$

- $|h|\le 100$ 구간 4,001점에서 부호 변화 탐색, 정확히 하나일 때만 이분법 200회, 없거나 여럿이면 미산출
- 타이밍 차이: MWR − TWR(1년 미만은 누적 수익률 차)
- 무매매 비교: 첫날 보유 수량 고정, 이후 입출금은 첫날 구성 비례 매매, 실제 XIRR − 무매매 XIRR
- 보유 수익률: 전일 종가 기준 비중 × 수정종가 변화, 매매는 수익으로 미계산
- 환율 분해: 일별 전일 비중 $w$, 현지 수익률 $r^{\ell}$, 통화 수익률 $r^{x}$

$$
1+R=\prod_t\Big(1+\sum_i w_{t,i}\big[(1+r^{\ell}_{t,i})(1+r^{x}_{t,i})-1\big]\Big),\qquad
1+L=\prod_t\Big(1+\sum_i w_{t,i}\,r^{\ell}_{t,i}\Big)
$$

- 환율 효과 $X=(1+R)/(1+L)-1$, 교차항 $L\cdot X$
- 혼합 벤치마크: 전일 한국 비중 $w^{KR}_{t-1}$ 로 매일 섞음, S&P 500은 총수익지수 ^SP500TR, KOSPI는 가격지수

$$
b_t=w^{KR}_{t-1}\,r^{\text{KOSPI}}_t+(1-w^{KR}_{t-1})\,r^{\text{SP500TR}}_t
$$

### 위험조정 지표

- 초과수익 $x_t=r_t-r^f_t$, 일간 샤프 $SR=\bar x/s_x$ (ddof 1), 표시값 $SR\sqrt P$
- 왜도 $\gamma_3$, 원첨도 $\gamma_4$ (정규 3), 252개 미만이면 신뢰구간·PSR·MinTRL·DSR·추적오차·정보비율 생략
- Mertens(2002) 표준오차와 95% 신뢰구간

$$
\sigma_{SR}=\sqrt{\frac{1-\gamma_3 SR+\frac{\gamma_4-1}{4}SR^2}{n}},\qquad SR\pm z_{0.975}\,\sigma_{SR}
$$

- PSR·MinTRL(Bailey–López de Prado 2012), 기준 $SR_0=0$, 신뢰수준 95%

$$
\text{PSR}=\Phi\!\left(\frac{(SR-SR_0)\sqrt{n-1}}{\sqrt{1-\gamma_3 SR+\frac{\gamma_4-1}{4}SR^2}}\right),\qquad
\text{MinTRL}=1+\Big(1-\gamma_3 SR+\tfrac{\gamma_4-1}{4}SR^2\Big)\Big(\frac{z_{0.95}}{SR-SR_0}\Big)^2
$$

- 소르티노: $\bar x/\sqrt{\overline{\min(x,0)^2}}$ (전 관측 분모), $\times\sqrt P$
- 오메가: $\sum x^+/\sum x^-$, 임계값 0
- 낙폭: $D_t=W_t/\max_{s\le t}W_s-1$, 시작 자본 1, 최대 낙폭·고점·저점·회복일·최장 수중 기간(달력일)
- 칼마: CAGR / |MDD|(1년 이상)
- 얼서 지수: $\sqrt{\overline{(100D_t)^2}}$, UPI: (CAGR − 연 무위험수익률) × 100 / 얼서
- CDaR 95%: 가장 깊은 5% 낙폭의 평균, 경계 관측은 비율만큼 반영
- 추적오차 $s(r-b)\sqrt P$, 정보비율 $\overline{(r-b)}\,P/\text{TE}$
- 액티브 셰어·회전율: $\tfrac12\sum_i|w_i-w'_i|$

### 상관·베타·분산

- 한·미 종가 시차 보정 공분산: 한국 종가가 같은 날 미국 개장 전이라 한국 $t$ 가 미국 $t-1$ 에 반응

$$
S=S_0+f\,(S_1+S_1^\top),\qquad S_1[i,j]=\text{cov}(x_{i,t},x_{j,t-1})\ \ (i\ \text{한국},\ j\ \text{비한국})
$$

- $f\in[0,1]$: $S$ 가 양반정치인 최댓값, 이분법 60회
- Ledoit–Wolf(2004) 축소: 목표 $F$ 는 평균 상관 상수 행렬

$$
\hat\Sigma=\delta F+(1-\delta)S,\qquad \delta=\min\!\big(\max(\hat\kappa/T,0),1\big),\quad \hat\kappa=(\hat\pi-\hat\rho)/\hat\gamma
$$

- Welch 베타: 수익률을 $-2m_t$ 와 $4m_t$ 사이로 절단($\delta=3$), 나이별 가중 $e^{-(2/256)\,\text{age}}$, 가중 최소제곱 기울기
- Dimson 베타(한국 종목 vs S&P 500): $r_t=a+b_0m_t+b_1m_{t-1}$, $\beta=b_0+b_1$
- 한국 종목은 KOSPI 대비 Welch 베타도 표시, 벤치마크는 ^GSPC(가격지수)
- 베타 축소: Kish 유효 표본 $n=(\sum w)^2/\sum w^2$ (126일 ≈ 117, 252일 ≈ 193, 756일 ≈ 255)

$$
\tilde\beta=\frac{n}{n+250}\hat\beta+\frac{250}{n+250}
$$

- 위험 기여(오일러): $RC_i=w_i(\Sigma w)_i/(w^\top\Sigma w)$, 합 1
- 분산 비율: $w^\top\sigma/\sqrt{w^\top\Sigma w}$ (1 = 분산 효과 없음)
- 유효 종목 수: $1/\sum_i\tilde w_i^2$

### 변동성 필터

- 적합 구간: 종목별 최근 2,520거래일(10년), 500일 미만이면 EWMA
- GJR-GARCH(1,1)(Glosten–Jagannathan–Runkle 1993): 평균 0, 정규 준최우추정(`arch`, % 단위)

$$
\sigma^2_{t+1}=\omega+\big(\alpha+\gamma\,\mathbf 1[r_t<0]\big)r_t^2+\beta\sigma_t^2
$$

- 기각: 미수렴 또는 $\alpha+\gamma/2+\beta\ge 0.999$ (반감기 693일) → EWMA
- EWMA: $\sigma^2_{t+1}=0.94\,\sigma^2_t+0.06\,r_t^2$ (RiskMetrics), 초기값은 적합 구간 $\overline{r^2}$
- 모의 구간의 분산 목표화(Engle–Mezrich 1996): $\omega=(1-\alpha-\gamma/2-\beta)\,\overline{r^2}$, EWMA는 하루 0.985씩 $\overline{r^2}$ 로 회귀
- HAR(Corsi 2009), 수준 회귀 OLS

$$
RV_t=c+b_d RV_{t-1}+b_w\overline{RV}^{(5)}_{t-1}+b_m\overline{RV}^{(22)}_{t-1}+b_x x_{t-1}+e_t
$$

- 로그 HAR 대신 수준: $E[RV_t\mid\text{과거}]$ 직접 추정, 모의 일의 $r^2$ 를 같은 재귀에 투입 가능
- HAR-RV-IV: $x_t$ = 전일 VIX의 일간 분산 $(\text{VIX}/100)^2/252$, 한국 종목은 그 전 미국 거래일 값
- 경로의존 변동성 PDV(Guyon–Lekeufack 2023), 2-지수 마르코프 커널

$$
\sigma_t=\beta_0+\beta_{10}R_{10}+\beta_{11}R_{11}+\beta_2\sqrt{(1-\theta_2)R_{20}+\theta_2R_{21}},\qquad
R_{1j}'=a_jR_{1j}+(1-a_j)r_t,\quad R_{2j}'=a_jR_{2j}+(1-a_j)r_t^2
$$

- 감쇠 $a=0.5^{1/h}$, 반감기 $h\in$ {1, 2, 4, …, 128}일 중 느린 쪽이 빠른 쪽의 4배 이상인 쌍
- $\theta_2\in$ {0, 0.25, 0.5, 0.75, 1}, 격자마다 목표($|r|$ 또는 $\sqrt{GK}$)를 $(1,R_{10},R_{11},\sqrt{R_2})$ 에 OLS, 제곱오차 최소 격자점 선택
- HAR·PDV 공통 스케일: $\overline{r^2/\max(\hat\sigma^2,\,0.01\,\overline{r^2})}=1$ 이 되도록 로그 스케일 이분법, 분산 하한 $0.01\,\overline{r^2}$
- 고가·저가 사용 시 분산 대용치: Garman–Klass(1980) + 전일 종가 대비 갭(Yang–Zhang 확장), 시가·고가·저가는 수정종가/종가 비율로 조정, 외화 종목은 환율 반영

$$
\sigma^2_{GK}=\max\!\Big(\tfrac12\big(\ln\tfrac HL\big)^2-(2\ln2-1)\big(\ln\tfrac CO\big)^2,\,0\Big)+\Big(\ln\tfrac{O}{C_{t-1}}\Big)^2
$$

- 고가·저가 없는 종목·날: $r^2$

### 시뮬레이션

- 경로 수 `PATHS = 20,000`, 평균 수익 0(일간 수익률 평균 차감), 일간 −100% 하한, 매일 현재 비중으로 재조정
- 재표본: Politis–Romano(1994) 정상 부트스트랩, 균등 시작일·기하분포 길이(평균 $b$)·끝에서 처음으로 순환
- 평균 묶음 길이: Politis–White(2004) + Patton–Politis–White(2009) 보정

$$
\hat b=\Big(\frac{\hat G}{\hat g}\Big)^{2/3}n^{1/3},\qquad
\hat G=\sum_{|k|\le M}\lambda\!\big(\tfrac kM\big)|k|\hat R(k),\quad
\hat g=\sum_{|k|\le M}\lambda\!\big(\tfrac kM\big)\hat R(k),\quad
\lambda(s)=\min\big(1,2(1-|s|)\big)
$$

- $M=2\hat m$, $\hat m$: $|\hat\rho_k|<2\sqrt{\log_{10}n/n}$ 가 $\max(5,\log_{10}n)$ 번 연속되는 첫 시차, 상한 $\min(3\sqrt n,n/3)$
- 수익률 $r$ 과 $(r-\bar r)^2$ 의 $\hat b$ 중 큰 쪽(변동성 군집 반영), $[1,\ \min(20,\ n/10)]$ 로 절단
- 지금 변동성 반영(FHS 경로, Barone-Adesi 외 1999): 잔차 $z_t=r_t/\sigma_t$ 를 평균 차감·단위 분산화 후 같은 부트스트랩 날짜로 추출, 오늘 분산에서 시작
- 처음 21일: 경로별 모의 충격으로 분산 갱신, 이후 기댓값 $\sigma^2{}'=\omega+(\alpha+\gamma/2+\beta)\sigma^2$ (Christoffersen–Diebold 2000), HAR·PDV는 수익 0·제곱 = 분산으로 상태 갱신(HAR은 정확한 다기간 예측)
- 결과: 가치 분위 5·25·50·75·95%, 수준 $X$ = 10·20·30% 하락·상승 경로 비율 세 겹($V_0$ = 지금, 극값에 $V_0$ 포함): 기간 말 $V_T/V_0-1\le-X$ · 기간 중 도달 $\min_t V_t/V_0-1\le-X$ · 고점 대비 $\max_t(1-V_t/\max_{s\le t}V_s)\ge X$, 상승은 대칭(저점 대비 $\max_t(V_t/\min_{s\le t}V_s-1)\ge X$), 경로마다 앞 기준이 뒤 기준을 함의, 전고점 회복 기간 중앙값·10%·90%, 미회복 비율
- 생성 모델 경로: 아래 생성 모델 참고
- 분포 검증(Berkowitz 2001): 화면의 경로 방식 그대로 1개월 수익률 2,000개 모의, 생성 모델 경로는 생략
  - 재표본: 전체 기간을 겹치지 않는 21일 구간으로 나눔, 구간 직전 선택 기간 길이 날만 같은 부트스트랩(평균 차감, $\hat b$ 도 그 날들로 추정), 직전 날이 선택 기간보다 짧은 구간 제외
  - FHS: 필터 기간을 겹치지 않는 21일 구간으로 나눔, 구간 첫날 필터 상태에서 모의
- PIT $=(\#\text{아래}+\tfrac12\#\text{동률}+\tfrac12)/(2000+1)$, $x_t=\Phi^{-1}(\text{PIT}_t)$

$$
x_t=c+\rho x_{t-1}+s\,\varepsilon_t,\qquad LR=2(\ell_1-\ell_0)\sim\chi^2_3\ \ (H_0:\ c=\rho=0,\ s=1)
$$

- 24구간 미만이면 생략

### VaR·ES

- 신뢰수준 95·97.5·99%, 방법 역사적·FHS·EVT, 대상은 선택 기간
- 최소 표본: 꼬리 12일 이상, $\lceil 12/(1-\alpha)\rceil$ = 240·480·1,200일
- 역사적: $m=n(1-\alpha)$, VaR = $\lceil m\rceil$ 번째 큰 손실

$$
\text{ES}=\frac{1}{m}\Big(\sum_{i<\lceil m\rceil}L_{(i)}+(m-\lceil m\rceil+1)\,\text{VaR}\Big)
$$

- 경계 손실을 비율만큼 반영해 항상 ES ≥ VaR, 종목별 ES 기여 합 = ES
- FHS: 시나리오 $z_{t,i}\,\sigma_{T+1,i}$, 종목별 변동성 유지 후 비중 가중
- EVT(McNeil–Frey 2000): FHS 손실 상위 $k=\lfloor 0.1n\rfloor$ 개(50개 이상, 즉 500일 이상), 임계값 $u$ = $(k+1)$ 번째 큰 손실, 초과분에 GPD

$$
\text{VaR}=u+\frac{\beta}{\xi}\left[\Big(\frac{n(1-\alpha)}{k}\Big)^{-\xi}-1\right],\qquad
\text{ES}=\frac{\text{VaR}+\beta-\xi u}{1-\xi},\quad \xi<1
$$

- GPD 최우추정: Grimshaw(1993) 프로파일, $\theta=\xi/\beta$ 에서 $\xi(\theta)=\overline{\ln(1+\theta y)}$, $\ell=-k\big(\ln(\xi/\theta)+\xi+1\big)$
- 탐색: $s=\ln(1+\theta\max y)\in[-6,8]$ 400점 격자 후 황금분할 60회, 격자 경계나 $\xi\le-0.5$ 이면 Hosking–Wallis(1987) 확률가중적률(도시 위치 $(i-0.35)/k$)

$$
\hat\xi=2-\frac{a_0}{a_0-2a_1},\qquad \hat\beta=\frac{2a_0a_1}{a_0-2a_1}
$$

- EVT 95% 프로파일 우도 구간(McNeil–Frey–Embrechts 2015 §5.2): 척도를 VaR 초과분 $d$ 로 재매개화

$$
\beta=\frac{d}{g(\xi)},\quad g(\xi)=\frac{c^{-\xi}-1}{\xi},\quad c=\frac{1-\alpha}{\zeta}\qquad(\text{ES: }\beta=d(1-\xi)/(g+1))
$$

- 꼬리 비율 $\zeta=P(L>u)$ 도 모수, 이항 우도 포함, $k/n\pm2.5$ 표준편차 15점 격자
- $\xi\in(-0.5,1.5)$ 100점 격자에서 최대화, 우도 하락 $\chi^2_1(0.95)/2=1.92$ 이내의 $d$ 집합, 상한이 막히지 않으면 ∞(ES는 $\xi\to1$ 우도가 하락폭 이내일 때)
- ES 97.5%의 90% 범위(5–95 분위, 경로 10,000개)
  - 모든 종목 GJR: 적합 필터로 선택 기간 길이 경로 모의, 잔차일 균등 추출, 시작 분산은 임의 일의 분산, $[\text{ES}/R_{95},\ \text{ES}/R_{5}]$, $R$ = 경로 ES / 전 경로 합산 ES
  - 한 종목이라도 EWMA: 포트폴리오 수익률 자체의 GJR로 같은 방식(Berkowitz–O'Brien 2002)
  - 그것도 실패: 선택 기간 정상 부트스트랩
- 1개월 95% VaR 두 방식: 겹치는 21일 복리 수익률 분위, $\sqrt{21}\times$ 1일 VaR, 둘 다 평균 차감, 겹치지 않는 21일 24개 이상일 때만, 1.25배 이상 차이면 표시
- 종목별 ES 기여(Tasche): 포트폴리오 꼬리 날의 가중 평균 손실 × 비중

### 사후검정

- 사후검정: 최근 250일, 95%, 날마다 직전 $L$ 일로 예측
- $L$: 선택 기간이 240일 이상이고 이력이 250 + 선택 기간 이상이면 선택 기간, 아니면 250, 이력 500일 미만이면 생략
- FHS 예측: 검정 250일 이전 2,520일로 필터 재추정(표본 외)
- Kupiec(1995) 비율 검정, $x$ 예외 / $n$ 일

$$
LR_{uc}=-2\Big[\ln\big((1-p)^{n-x}p^{x}\big)-\ln\big((1-\hat p)^{n-x}\hat p^{x}\big)\Big]\sim\chi^2_1,\quad \hat p=x/n
$$

- Christoffersen(1998) 독립성: 예외 전이 $n_{00},n_{01},n_{10},n_{11}$ 의 1차 마르코프 대 독립 우도비, $\chi^2_1$
- 신호등: 예외 수의 이항 누적확률 < 0.95 초록, < 0.9999 노랑, 그 외 빨강
- Acerbi–Székely(2014) Z2

$$
Z_2=1-\frac{1}{Tp}\sum_t\frac{L_t\,\mathbf 1[L_t>\text{VaR}_t]}{\text{ES}_t}
$$

- Du–Escanciano(2017): PIT $u_t$ = 예측 시나리오 중 실현값 아래 비율(중간 순위), $H_t=\frac{p-u_t}{p}\mathbf 1[u_t\le p]$

$$
U=\frac{\sqrt n\,(\bar H-p/2)}{\sqrt{p(1/3-p/4)}},\qquad C=n\sum_{j=1}^{5}\hat\rho_j^2\ \ (H_t-p/2\ \text{의 자기상관})
$$

- p값(Z2·U·C): 날마다 그날 예측 시나리오에서 하나씩 뽑은 모의 20,000회(`ES_SIMS`), $(1+\#)/(1+20{,}000)$

### 스트레스·난기류·스타일

- 과거 구간 재현 11개: 1987 블랙먼데이, 1997 IMF, 닷컴, 금융위기, 2011 미국 신용등급, 2015 중국, 2018 4분기, 코로나, 2022, 2024 한국 블랙먼데이, 2025 관세
- 구간 확정: S&P 500(^GSPC)·KOSPI 각각 명시 고점 ±90일 내 최고 종가 → 명시 저점 ±90일 내 최저 종가, 가장 이른 고점 ~ 가장 늦은 저점
- 지금 평가액 매수 후 보유: 구간 변화, 구간 내 최대 하락 $\min(V_t/\max_{s\le t}V_s-1)$, 환율 배율 $(1+\text{기준통화})/(1+\text{현지})-1$
- 구간 자료 없는 종목: 자기 지수(KOSPI 또는 S&P 500, 미국 전용 구간은 S&P 500) 일간 수익률 × 대용 베타
- 대용 베타: 전 기간, 한국 종목 vs 미국 지수는 Dimson, 그 외 감쇠 없는 Welch, $n/(n+250)$ 축소, 60일 미만이면 1
- 가상 충격: 종목별 자기 지수 대비 현지 통화 Welch 베타(선택 기간, Kish $n$ 축소)

$$
\Delta V/V=\sum_i w_i\Big[\big(1+\max(\tilde\beta_i\,\Delta I_i,-1)\big)(1+\Delta x_i)-1\Big]
$$

| 충격 | S&P 500 | KOSPI | 원/달러 |
|---|---:|---:|---:|
| 한국발 위기 | 0 | −30% | +15% |
| 미국발 위기 | −25% | 0 | +10% |
| 환율 −15% | 0 | 0 | −15% |
| 환율 +15% | 0 | 0 | +15% |

- 역스트레스: 환율 고정, 모든 지수 같은 하락폭으로 −30% 손실이 되는 지수 하락, $[-100\%,0]$ 이분법
- 오류 시세 표시: $|\Delta\ln P_t|>0.5$ 이고 다음 날 $|\ell_1+\ell_2|<0.5|\ell_1|$
- 난기류(Kritzman–Li 2010): 주간(금요일) 수익률, 직전 156주 평균·Ledoit–Wolf 공분산, 4주마다 재추정

$$
d_t=\frac1n(r_t-\mu)^\top\Sigma^{-1}(r_t-\mu),\qquad \text{상관 놀람}=d_t\Big/\frac1n\sum_i\frac{(r_{t,i}-\mu_i)^2}{\Sigma_{ii}}
$$

- 이번 주 값의 과거 주 대비 백분위
- 스타일 분석(Sharpe 1992): 주간 156주, 비교 SPY·IWM·IVW·IVE·MTUM·QUAL·USMV·AGG·GLD·KOSPI·KOSDAQ + 현금(수익 0)

$$
\min_{w\ge0,\ \sum w=1}\ \text{Var}\big(y-Xw\big),\qquad R^2=1-\frac{\text{Var}(y-Xw)}{\text{Var}(y)}
$$

- 풀이: FISTA(가속 사영 경사, 상승 시 재시작), 심플렉스 사영(Duchi 외), 최대 5,000회, 허용오차 1e-10
- 선택 수익: 직전 156주로 맞춘 $w$ 의 표본 외 $y_t-x_t^\top w$ 평균 × 52

### 종목 분석·추가 효과

- 종목 분석: 포트폴리오와 같은 계산을 비중 100%로, 기준 통화는 종목의 거래 통화, 벤치마크는 한국 종목 KOSPI·그 외 S&P 500, 단일 종목에 무의미한 상관·분산 비율·ES 기여·난기류 제외
- 추가 효과: 평가액의 1–30%(기본 5%)를 후보에 넣고 기존 보유는 비례 축소

$$
w'=(1-\phi)\,w+\phi\,e_c
$$

- 변동성·베타(한국 종목 Dimson, 그 외 Welch)·분산 비율 전후 비교, 시차 보정 Ledoit–Wolf 공분산
- 변동성 감소 기준: 후보와 현재 포트폴리오 상관 $\rho<\sigma_p/\sigma_c$
- 증분 ES 97.5%: 같은 날들로 전면 재평가, 추가 후 후보의 ES 기여 비중
- 몬테카를로: 추가 전후 같은 재표본 날짜

### 스크리닝

- 0단계 유동성: 일봉 250개 이상, 주가 5달러 이상, 20일 평균 거래대금 하위 10% 제외(동점 포함)
- 비교 업종: 유동성 통과 종목 중 GICS 세부 산업 → 산업 → 산업 그룹 → 섹터 순, 10종목 이상인 가장 좁은 단계, 그 단계 라벨을 가진 전 종목과 비교(GICS 2023년 3월 개정)
- 업종 내 백분위(Hazen, 동점 평균 순위)

$$
p_i=\frac{\text{rank}_i-0.5}{n}
$$

- 1단계 결격(하나라도 결격, 또는 미확인·해석 부적절이면 칸 배정 제외)
  - 비금융: FCF < 0
  - 은행(GICS 산업 그룹 Banks): ROE < 0 또는 TCE/TA < 4%, TCE/TA = TangibleBookValue / (총자산 − 영업권·무형자산)
  - 기타 금융: ROE < 0 또는 자기자본비율 < 업종 중앙값 × 0.5
  - 매출 성장 ≤ −5%(직전 연도 매출 ≤ 0이면 해석 부적절)
  - 순부채/EBITDA > max(업종 중앙값 × 2, 3.0), 순현금 통과, EBITDA ≤ 0 미산출, 금융업 해당 없음
- 2단계 퀄리티·밸류에이션: 항목별 업종 백분위의 평균
  - 퀄리티(2개 이상): ROE(장부가치 > 0), 영업이익률 5년 중앙값(3년 이상), FCF 마진 = FCF / 연간 매출(매출 > 0), 매출 성장(직전 회계연도 대비)
  - 밸류에이션(1개 이상): 선행 PER(> 0), EV/EBITDA(> 0)
  - 금융업: 퀄리티 ROE·매출 성장, 밸류에이션 선행 PER·PBR(주가 / 주당 장부가치 > 0)
- 3단계 칸: `SPLIT = 0.5`

| | 밸류에이션 < 0.5 | 밸류에이션 ≥ 0.5 |
|---|---|---|
| 퀄리티 ≥ 0.5 | 검토 우선 | 밸류에이션 부담 |
| 퀄리티 < 0.5 | 밸류 트랩 점검 | 위험 우세 |

- 칸 안 정렬: 퀄리티 − 밸류에이션 내림차순, 같으면 시가총액, 칸별 상위 20종목(섹터 필터 후)
- 보류(시가총액 순)
  - 정점 이익: 에너지·소재·산업재·경기소비재, 후행 PER > 0, 선행 PER(> 0) < 후행 PER × 0.6
  - 업황 순풍: 비금융, 중앙값 > 0·3년 이상, TTM 영업이익률 ≥ 중앙값 × 1.5(`TAILWIND_RATIO`) 이면서 ≥ 중앙값 + 5%p(`TAILWIND_GAP`)
  - 퀄리티 또는 밸류에이션 미산출
- 영업이익률 5년 중앙값: SEC 10-K 최근 5개 회계연도(최신 연도에서 5년 − 180일 이내, 기간 350–380일, 영업이익 있고 매출 > 0)의 중앙값(MEDIAN), 5년 미만이면 Yahoo 연간 손익 중앙값(3년 이상)
- 매출 항목: Revenues, RevenueFromContractWithCustomerExcludingAssessedTax, SalesRevenueNet, RevenueFromContractWithCustomerIncludingAssessedTax 중 5년 창에 영업이익과 함께 가장 많은 것, 동률은 앞 순서
- 최근 제출본 우선: 같은 기간 값은 가장 늦게 제출된 것
- SEC 분기값: 10-Q 3개월(80–120일) 값, 없으면 누계 차이(6개월 − 3개월, 9개월 − 6개월, 4분기 = 10-K 연간 − 9개월 누계), 분기 매출 > 연 매출 × 0.75면 오태깅으로 제외
- SEC TTM: 연속 4분기, 간격 각 80–120일·합 245–290일, 최신 분기 ≥ 최신 10-K 기말, 매출 합 > 0, $\sum OI/\sum \text{매출}$
- TTM 선택: SEC 중앙값 종목은 SEC TTM, 단 최신 분기가 Yahoo 최신 분기보다 31일 넘게 이르면 Yahoo TTM(최근 4분기 합)
- 추정치 방향(색 표시만): FY1 EPS 컨센서스가 90일 전 이상·30일 하향 ≤ 상향이면 상향, 90일 전 미만·하향 > 상향이면 하향, 나머지 보합
- 기술 점수(10점, 표시만): 수정 OHLC(시가·고가·저가 × 수정종가/종가)
  - 추세 4: 종가 > 200일선, 20일선 > 60일선, 60일선 > 20거래일 전 60일선, 종가 > 20일선
  - 모멘텀 3: MACD > 시그널, MACD > 0, 50 ≤ RSI ≤ 70
  - 거래량 3: OBV > 20거래일 전, 60일 상승일 평균 거래량 > 하락일, 52주 신고가(종가 ≥ 252일 고가 × 0.98)면 OBV ≥ 60일 최고 − 0.05 × 60일 범위, 아니면 OBV 60일 OLS 기울기 > 0
  - 상위 20% 차트 강, 하위 40% 차트 약(동점 포함)
- 과열 표시: RSI > 70, 스토캐스틱 %K > 80이면서 %K < %D, 볼린저 %B > 1
- 지표 정의

$$
\text{RSI}=100-\frac{100}{1+\text{EMA}_{1/14}(\Delta C^+)/\text{EMA}_{1/14}(\Delta C^-)},\qquad
\text{MACD}=\text{EMA}_{12}-\text{EMA}_{26},\ \ \text{시그널}=\text{EMA}_9(\text{MACD})
$$

$$
\%B=\frac{C-(\text{SMA}_{20}-2\sigma_{20})}{4\sigma_{20}},\qquad
\%K=\text{SMA}_3\Big(100\frac{C-L_{14}}{H_{14}-L_{14}}\Big),\ \ \%D=\text{SMA}_3(\%K),\qquad
\text{OBV}_t=\sum_{s\le t}\text{sign}(\Delta C_s)\,V_s
$$

- RSI·ATR 14일 Wilder 평활($\alpha=1/14$), 볼린저 $\sigma$ 는 모표준편차, 1·3개월 수익률 $C_t/C_{t-21}-1$, $C_t/C_{t-63}-1$

### 유명 투자자 복제(13F)

- 보유: 13F-HR·13F-HR/A 정보표 최근 8개 분기, 같은 CUSIP 합산, 풋·콜·원금(PRN) 행은 별도 집계·보유에서 제외
- 금액 단위: 2023-01-03 이후 제출분 달러, 이전 천 달러
- 정정: RESTATEMENT는 분기 교체, NEW HOLDINGS는 합산
- 분석 대상: 가치 상위부터 누적 90% 또는 100종목까지, 나머지 미포함 비중, 비중은 전체 보고 가치 대비, 티커 미연결분은 가격 미연결 비중
- 매수일: 접수 시각이 뉴욕 16:00 이전이면 그날 종가, 이후·시각 없음이면 다음 날, 휴장은 다음 거래일
- 복제: 각 제출 비중을 매수일 종가에 사서 다음 제출까지 보유(가격 변동대로 비중 표류), 이미 공개된 분기보다 오래된 분기의 늦은 정정은 제외
- 마지막 종가가 7일보다 오래된 종목은 매수 제외, 나머지 비중 재정규화
- 변형: 보고 비중 상위 10종목(가치 비중 또는 동일 비중), 분기 말 매수 이론값(공개 전 정보)
- 회전율: $\tfrac12\sum|w^{\text{new}}-w^{\text{drift}}|$, 표류 비중 $w_i g_i/\sum_j w_j g_j$
- 위험: 복제 일간 수익률과 현재 표류 비중에 포트폴리오와 같은 계산, 무위험수익률 ^IRX, 벤치마크 S&P 500 총수익
- 여러 투자자 비교: 샤프 경험적 베이즈 축소와 DSR

$$
\tau^2=\max\big(0,\ \text{Var}(SR)-\overline{\sigma_{SR}^2}\big),\qquad
SR_i^{EB}=\overline{SR}+\frac{\tau^2}{\tau^2+\sigma_{SR,i}^2}\big(SR_i-\overline{SR}\big)
$$

$$
SR^{\max}_N=\sqrt{\text{Var}(SR)}\Big[(1-\gamma_E)\Phi^{-1}\big(1-\tfrac1N\big)+\gamma_E\Phi^{-1}\big(1-\tfrac1{Ne}\big)\Big],\qquad
\text{DSR}=\text{PSR}(SR_0=SR^{\max}_N)
$$

- $\gamma_E=0.5772$ (오일러–마스케로니), 최고 샤프 시리즈가 252개 이상일 때만
- 보유 겹침: 내 포트폴리오의 미국 주식 부분과 1 − 액티브 셰어

### 생성 모델

- 공통: 일간 로그 수익률, 학습 종목·기간은 설정에서 선택(2종목 이상, 1,300일 이상), 시간순 80% 학습 분할, 시드 42
- 추론: torch 가중치를 numpy 순전파로 재구현, torch 대비 고정 잡음 최대 절대오차 < 1e-3일 때만 교체(이전 모델 보관)
- 경로 사용: 20,000개(2,000개씩), 단순 수익률 $e^{x}-1$, 외화 종목은 같은 경로의 원/달러로 환산(KRW 기준 $(1+r)(1+r^{fx})-1$, USD 기준 $(1+r)/(1+r^{fx})-1$), 평균 0·−100% 하한
- 선택 조건: 보유 종목 전부(기준통화 외 종목이 있으면 KRW=X도) 학습 종목에 포함
- 기간 처리: 창 모델은 독립 창을 이어 붙여 $\lceil h/\text{창}\rceil$ 개 생성 후 절단(창 간 변동성 군집 없음), SigCWGAN은 하루씩 자기회귀

| | Tail-GAN | SigCWGAN | 신경망 SDE |
|---|---|---|---|
| 논문 | Cont·Cucuringu·Xu·Zhang | Liao 외 2020 | Issa·Horvath·Lemercier·Salvi 2023 |
| 생성기 | MLP 1000→128→256→512→1024→(종목×21), BatchNorm(eps 0.8), LeakyReLU 0.2, ±1 절단 | ArFNN: [잡음, 직전 3일] → 잔차 블록(50, 50, 50), PReLU | Euler–Maruyama, dt 1, 은닉 16, 잡음 8, 드리프트·확산 MLP 64×3 LipSwish + tanh, 시작 1, 선형 출력 |
| 잡음 | Student t(5), 1,000차원 | 표준정규, 종목 수 차원 | 브라운 증분 8차원 |
| 창 | 21일 | 제한 없음(p = q = 3) | 63일 |
| 손실 | 판별기가 매수 보유·롱숏 정적 포트폴리오 50개의 P&L을 NeuralSort(온도 0.01) 정렬해 VaR·ES(α 5%) 출력, FZ형 점수(W = 10) | 조건부 시그니처 W1: 깊이 2 시그니처, Scale(0.2)·Cumsum·AddLags(2)·LeadLag | 시그니처 커널 MMD(RBF σ 1, dyadic 1), 비적대적 |
| 학습 | 3,000 에폭, 배치 1,000, Adam lr G 1e-6·D 1e-7, β (0.5, 0.999) | 1,000 스텝, 배치 200, mc 1,000, Adam 1e-2, StepLR(100, 0.9), 클립 10 | 1,000 스텝, 배치 128, Adadelta lr 1e-3·감쇠 1e-2, 500 스텝 후 SWA, 초기 가중치 × 0.5 |
| 정규화 | 없음 | 종목별 전 기간 표준화 | 63일 누적 로그 수익률 표준편차 |

- SigCWGAN 조건: 최근 수익률 마지막 3일(없으면 학습 자료 마지막 3일)
- Kronos(구현 중): Kronos-small·Kronos-Tokenizer-base, 종목별 독립 표본(공동 움직임 없음), 종가로 OHLC 재구성(시가 = 고가 = 저가 = 종가, 거래량 없음), 문맥 512일, 온도 1.0, top-p 0.9, `kronos` 추가 의존성

### 거시 대시보드

- 변환: 수준, 전년 대비 %(월·분기는 같은 달, 일·주는 365일 전 값, 10일 초과 차이면 생략), 전기 대비 %, 차분, 12개월 합
- 교차 환율: 같은 날짜 값의 비율
- 수익률 곡선: 1M–30Y 11개 만기, 최신·1개월 전·1년 전(각 날짜 이전 마지막 값)
- 10Y−2Y: 월평균, NBER 침체(USREC = 1) 구간 음영
- 국면 판정
  - 물가: 근원 PCE 전년비 > 2.5% 목표 상회, < 1.5% 목표 하회
  - 고용: 삼 룰 ≥ 0.50 침체 신호, ≥ 0.30 주의
  - 금리곡선: 10Y−2Y < 0 역전
  - 금융여건: NFCI > 0 평균보다 긴축
- 스파크라인: 일·주간은 월평균 24개월, 월간 24개, 분기 12개
- FOMC 경제전망: 위원 중앙값, 직전 발표(ALFRED 시점별 값) 대비 변화

### AI 분석

- 모델 도구 호출: 시계열 검색·이력 조회·출처 코드 안내, 실행당 12회 이하
- 숫자 검증: 문장 끝 `[[지표 id]]` 근거의 실제 값과 문장 정밀도로 대조(bp는 ÷100), 일치·불일치·근거 없음 구분

## 개발

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest -q
uv run streamlit run src/finagent/app.py -- --demo   # 파일 수정 즉시 재실행
```

- 지표 추가: `src/finagent/catalog.py` 패널 튜플 한 줄 + `src/finagent/demo.py` 가짜 시계열 한 줄, 상단 줄은 `TAPE`, 기본 화면은 `CORE`
- 테스트: 실제 API 미호출, `demo.py`의 가짜 응답 사용
