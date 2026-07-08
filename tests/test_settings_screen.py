from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import pandas as pd  # noqa: PANDAS_OK

from finagent_terminal.config import FinAgentConfig
from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    OhlcvFrame,
    Quote,
)
from finagent_terminal.data.session import DataSession
from finagent_terminal.llm.base import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEYS,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
)
from finagent_terminal.symbols import SymbolGroup
from finagent_terminal.ui.screens.markets import MarketsScreen
from finagent_terminal.ui.screens.settings import (
    SettingsScreen,
    data_source_availability_lines,
)
from finagent_terminal.ui.screens.ticker import TickerScreen
from finagent_terminal.ui.screens.tools import ToolsScreen
from finagent_terminal.ui.state import create_app_state

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

FETCHED_AT: Final = datetime(2024, 1, 5, 12, tzinfo=UTC)
LLM_KEYS: Final = (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    *GEMINI_API_KEYS,
    OPENROUTER_API_KEY,
)


@dataclass(frozen=True, slots=True)
class FakeProvider:
    def fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        frame = pd.DataFrame(
            {"Close": [100.0, 105.0, 110.0], "Volume": [1, 2, 3]},
            index=pd.date_range("2024-01-03", periods=3, freq="D"),
        )
        return MarketDataSnapshot(
            symbol=symbol,
            period=period,
            interval=interval,
            fetched_at=FETCHED_AT,
            freshness=FreshnessState.FRESH,
            quote=Quote(
                symbol=symbol,
                last_price=110.0,
                previous_close=105.0,
                change=5.0,
                change_percent=0.0476,
                currency="USD",
                market_cap=None,
                volume=3,
                day_high=None,
                day_low=None,
                fetched_at=FETCHED_AT,
            ),
            history=OhlcvFrame(symbol, period, interval, frame, len(frame), FETCHED_AT),
        )


def test_providers_unavailable_without_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: no optional LLM provider keys are present.
    for key in LLM_KEYS:
        monkeypatch.delenv(key, raising=False)
    state = create_app_state()
    session = DataSession(FakeProvider())

    # When: Settings is rendered and market workspaces use fake data.
    settings_text = "\n".join(SettingsScreen(state).lines())
    markets = MarketsScreen(
        state,
        config=FinAgentConfig(
            groups=(
                SymbolGroup.model_validate({"name": "Smoke", "symbols": ["AAPL"]}),
            ),
        ),
        config_path=tmp_path / "config.json",
        session=session,
    )
    markets.refresh_market_data()
    ticker = TickerScreen(state, session)
    ticker.load_symbol("AAPL")
    tools = ToolsScreen(state, session)
    tools_text = "\n".join(tools.lines())

    # Then: provider availability is explicit, and non-report screens still work.
    assert "openai: unavailable" in settings_text
    assert "anthropic: unavailable" in settings_text
    assert "gemini: unavailable" in settings_text
    assert "openrouter: unavailable" in settings_text
    assert "Markets/Ticker/Tools: available without LLM API keys" in settings_text
    assert "Data: 1 symbols loaded" in "\n".join(markets.lines())
    assert "Symbol: AAPL" in "\n".join(ticker.lines())
    assert "Analysis hub" in tools_text
    assert "Monte Carlo: simulate future close-return paths" in tools_text
    assert "Monte Carlo: waiting for a run" in tools_text
    assert "Normalized returns:" not in tools_text


def test_data_source_availability_lines_have_no_secret_values() -> None:
    env = {
        "FRED_API_KEY": "fred-secret",
        "FINNHUB_API_KEY": "finn-secret",
        "FMP_API_KEY": "fmp-secret",
        "FINAGENT_SEC_USER_AGENT": "sec-agent-secret",
    }

    text = "\n".join(data_source_availability_lines(env))

    assert "FRED_API_KEY" in text
    assert "FINNHUB_API_KEY" in text
    assert "FMP_API_KEY" in text
    assert "FINAGENT_SEC_USER_AGENT" in text
    assert "available" in text
    assert "fred-secret" not in text
    assert "finn-secret" not in text
    assert "fmp-secret" not in text
    assert "sec-agent-secret" not in text
