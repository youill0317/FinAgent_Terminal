# FinAgent Terminal

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
