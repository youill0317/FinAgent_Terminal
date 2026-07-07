from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import pandas as pd  # noqa: PANDAS_OK

from finagent_terminal import __version__
from finagent_terminal.app import FinAgentApp
from finagent_terminal.config import FinAgentConfig, app_home, config_path
from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    OhlcvFrame,
    Quote,
)
from finagent_terminal.data.session import DataSession
from finagent_terminal.llm import ProviderName, create_llm_client
from finagent_terminal.reports.export import (
    ReportExport,
    write_markdown_report,
)
from finagent_terminal.reports.models import DepthMode
from finagent_terminal.reports.state import (
    ReportJob,
    ReportRequest,
    run_screen_report,
)
from finagent_terminal.symbols import SymbolGroup
from finagent_terminal.ui.screens.markets import MarketsScreen
from finagent_terminal.ui.screens.settings import SettingsScreen
from finagent_terminal.ui.screens.ticker import TickerScreen
from finagent_terminal.ui.screens.tools import ToolsScreen
from finagent_terminal.ui.state import create_app_state

if TYPE_CHECKING:
    from collections.abc import Sequence

HELP_TEXT: Final = (
    "finagent - terminal market research workspace\n\n"
    "Usage:\n"
    "  finagent [--help | --version | --smoke]\n"
    "  python -m finagent_terminal [--help | --version | --smoke]\n\n"
    "Without arguments, starts the Textual app shell.\n"
    "--smoke requires FINAGENT_FAKE_DATA=1 and FINAGENT_FAKE_LLM=1.\n"
)
SMOKE_SYMBOLS: Final = ("AAPL", "MSFT", "QQQ")
SMOKE_AT: Final = datetime(2024, 1, 5, 12, tzinfo=UTC)


def main(argv: Sequence[str] | None = None) -> int:
    args = tuple(sys.argv[1:] if argv is None else argv)

    if args == ():
        FinAgentApp().run()
        return 0

    if args in {("--help",), ("-h",)}:
        _ = sys.stdout.write(HELP_TEXT)
        return 0

    if args == ("--version",):
        _ = sys.stdout.write(f"finagent {__version__}\n")
        return 0

    if args == ("--smoke",):
        return _run_smoke()

    _ = sys.stderr.write(f"Unknown argument: {' '.join(args)}\n\n{HELP_TEXT}")
    return 2


@dataclass(frozen=True, slots=True)
class _SmokeProvider:
    def fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        offset = float(sum(ord(character) for character in symbol) % 11)
        close_prices = (100.0 + offset, 103.0 + offset, 108.0 + offset)
        frame = pd.DataFrame(
            {
                "Close": close_prices,
                "Volume": [1000, 1200, 1400],
            },
            index=pd.date_range("2024-01-03", periods=3, freq="D"),
        )
        last_price = close_prices[-1]
        previous_close = close_prices[-2]
        return MarketDataSnapshot(
            symbol=symbol,
            period=period,
            interval=interval,
            fetched_at=SMOKE_AT,
            freshness=FreshnessState.FRESH,
            quote=Quote(
                symbol=symbol,
                last_price=last_price,
                previous_close=previous_close,
                change=last_price - previous_close,
                change_percent=(last_price - previous_close) / previous_close,
                currency="USD",
                market_cap=None,
                volume=1400,
                day_high=None,
                day_low=None,
                fetched_at=SMOKE_AT,
            ),
            history=OhlcvFrame(symbol, period, interval, frame, len(frame), SMOKE_AT),
        )


def _run_smoke() -> int:
    if os.environ.get("FINAGENT_FAKE_DATA") is None:
        _ = sys.stderr.write("--smoke requires FINAGENT_FAKE_DATA=1\n")
        return 2
    if os.environ.get("FINAGENT_FAKE_LLM") is None:
        _ = sys.stderr.write("--smoke requires FINAGENT_FAKE_LLM=1\n")
        return 2

    state = create_app_state()
    session = DataSession(_SmokeProvider())
    config = FinAgentConfig(
        groups=(
            SymbolGroup.model_validate(
                {"name": "Smoke", "symbols": list(SMOKE_SYMBOLS)},
            ),
        ),
    )

    markets = MarketsScreen(
        state,
        config=config,
        config_path=config_path(),
        session=session,
    )
    markets.refresh_market_data()
    ticker = TickerScreen(state, session)
    ticker.load_symbol("AAPL")
    tools = ToolsScreen(state, session)
    tools.compare_manual("AAPL MSFT QQQ")
    report = run_screen_report(
        ReportJob(
            request=ReportRequest(
                symbol="AAPL",
                provider=ProviderName.OPENAI,
                model="fake-model",
                depth=DepthMode.STANDARD,
            ),
            session=session,
            client_factory=create_llm_client,
            started_at=0.0,
        ),
    )
    export_path = write_markdown_report(
        ReportExport(report.context, report.result, SMOKE_AT),
        app_home() / "reports",
    )
    settings = SettingsScreen(state)
    reports_line = (
        "SMOKE reports: "
        f"Completed: {report.result.progress.completed_count}/"
        f"{report.result.progress.total_count}; Exported: {export_path}"
    )

    _ = sys.stdout.write(
        "\n".join(
            (
                f"SMOKE markets: {markets.lines()[2]}",
                f"SMOKE ticker: {ticker.lines()[0]}",
                f"SMOKE tools: {tools.lines()[2]}",
                reports_line,
                f"SMOKE settings: {settings.lines()[4]}",
            ),
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
