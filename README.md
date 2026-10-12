# FinAgent Terminal

## Macro dashboard (`finagent-macro`)

A local web dashboard of US, Korean and global macro data, built only on
public and open sources. It runs a small server on `127.0.0.1` and opens
your browser.

```bash
uv sync
export FRED_API_KEY=...   # free: https://fredaccount.stlouisfed.org/apikeys
export ECOS_API_KEY=...   # free: https://ecos.bok.or.kr/api/#/AuthKeyApply
uv run finagent-macro            # serve on http://127.0.0.1:8765 and open a browser
uv run finagent-macro --demo     # made-up sample data, no keys or network needed
uv run finagent-macro --check    # fetch once and print per-source status
```

Options: `--host`, `--port`, `--no-browser`.

**Sources**

| Source | Key | Used for |
|---|---|---|
| FRED (St. Louis Fed) | `FRED_API_KEY` | US growth, labor, prices, rates, curve, liquidity, energy, release calendar |
| U.S. Treasury Fiscal Data | none | total public debt, average interest rate |
| New York Fed Markets API | none | EFFR, SOFR |
| 한국은행 ECOS | `ECOS_API_KEY` | base rate, KTB yields, USD/KRW, KOSPI, CPI, exports, jobs, CSI |
| ECB Data Portal | none | deposit rate, euro-area HICP, EUR FX |
| BIS Data Portal | none | BoJ and BoE policy rates |
| OECD Data Explorer | none | composite leading indicators (US, China, Korea) |

A source without its key, or one that is down, is shown as such; the rest of
the dashboard still loads.

**Data rules**

- Fetched on app start and when you press Refresh (or `R`). There is no timer.
- Kept in process memory only. Nothing fetched is written to disk.
- No analyst consensus, estimates, investment opinions, sentiment scores or
  forecasts. Proprietary PMIs are replaced by the OECD leading indicator.
- Series that carry third-party copyright on FRED (ICE, Moody's, Cboe,
  Freddie Mac) are marked with © and are for personal viewing. FRED serves
  only the last three years of ICE BofA spreads.

The indicator list lives in `src/finagent_terminal/macro/catalog.py`.

## Terminal workspace (`finagent`)

Python 3.12 terminal workspace for US market research. It has six Textual
workspaces: Markets, Ticker, News, Tools, Reports, and Settings.

FinAgent fetches market data on demand with yfinance and keeps fetched quotes
and OHLCV frames in memory for the session only. Local files are limited to
editable symbol groups, UI/provider settings, and exported Markdown reports.

## Install

```bash
uv sync
```

## Run

```bash
uv run finagent
uv run finagent --help
uv run python -m finagent_terminal --help
```

For offline verification:

```bash
FINAGENT_HOME=$(mktemp -d) \
FINAGENT_FAKE_DATA=1 \
FINAGENT_FAKE_LLM=1 \
uv run finagent --smoke
```

## Configuration

Set `FINAGENT_HOME` to choose the config/report directory. Without it, the app
uses `~/.finagent_terminal/`.

`config.json` stores:

- symbol groups and watchlists
- selected LLM provider/model
- small UI settings

It must not store fetched prices, history, OHLCV rows, or fundamentals.

## LLM Providers

Reports can use these optional environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`
- `FINAGENT_LLM_PROVIDER`
- `FINAGENT_LLM_MODEL`

Markets, Ticker, Tools, and Settings work without LLM API keys. Reports require
either a configured provider key or `FINAGENT_FAKE_LLM=1` for deterministic
tests and smoke runs.

## Market Data Sources

Set these optional environment variables for user-owned source access:

- `FRED_API_KEY`
- `FINNHUB_API_KEY`
- `FMP_API_KEY`
- `FINAGENT_SEC_USER_AGENT`

yfinance and public/user-key sources have availability, delay, adjustment, and
coverage limits. FinAgent is a personal research terminal: it does not place or
propose orders, connect to broker accounts, run background sync, bundle vendor
keys, or persist fetched source data as a database/cache. Source data stays in
memory for the session; exported reports contain generated report text,
citations, timestamps, and warnings only.

## Reports

Reports are research assistance only; not investment advice. They are generated
from supplied yfinance snapshots and computed indicators, include provenance,
and export as a single Markdown file.
