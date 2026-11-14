from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, final

import streamlit as st

from finagent import display, etf, investors, limits, markets, portfolio, yahoo
from finagent.cli import resolve_env
from finagent.dashboard import all_fetches
from finagent.demo import portfolio as demo_portfolio
from finagent.sources import EtfSeries, etf_start

if TYPE_CHECKING:
    from datetime import date

    from finagent.runtime import Runtime


@final
class App:
    def __init__(
        self, demo: bool, env_file: str, rt: Runtime, settings_path: Path
    ) -> None:
        self.demo = demo
        self.env_file = env_file
        self.rt = rt
        self.settings_path = settings_path
        self.root = settings_path.parent
        self.calc = demo_portfolio if demo else portfolio


app: App


def configure(new: App) -> None:
    global app  # noqa: PLW0603
    app = new


def limit_note() -> str:
    line = display.limit_lines(limits.current().state(), time.time())
    return f" · {line}" if line else ""


def env_values(env_file: str) -> dict[str, str]:
    return resolve_env(Path(env_file) if env_file else None, os.environ)


def etf_charted(today: date) -> dict[str, date]:
    charted = [fetch for fetch in all_fetches() if isinstance(fetch, EtfSeries)]
    start = etf_start(charted, today)
    return dict.fromkeys((fetch.ticker for fetch in charted), start)


@st.cache_resource(show_spinner=False)
def etf_updater() -> etf.Updater:
    return etf.Updater(market_keys(), etf_charted, sec_agent())


def etf_held() -> list[str]:
    try:
        return portfolio.book_tickers(portfolio.load_book(app.root))
    except portfolio.BookError:
        return []


def etf_confirm(ticker: str) -> etf.Verdict:
    if app.demo:
        return "확인됨"
    return etf.confirm(
        ticker, partial(yahoo.quote_type, max_wait=limits.INTERACTIVE_WAIT)
    )


def market_keys() -> markets.Keys:
    return markets.keys(env_values(app.env_file))


def sec_agent() -> str:
    env = env_values(app.env_file)
    return env.get(investors.AGENT_KEY, "").strip()
