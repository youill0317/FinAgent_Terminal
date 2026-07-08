from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event
from typing import TYPE_CHECKING

import httpx
import pandas as pd  # noqa: PANDAS_OK
from textual.app import App

from finagent_terminal.data.models import (
    FreshnessState,
    MarketDataSnapshot,
    OhlcvFrame,
    Profile,
    Quote,
)
from finagent_terminal.data.sources import SourceSession
from finagent_terminal.llm import FakeLLMClient, LLMSelection
from finagent_terminal.llm.base import ProviderCallError
from finagent_terminal.reports.models import ReportSectionPrompt, ReportSectionResult
from tests.source_routing_fixtures import SEC_SAMPLE, macro_response, success_response
from tests.vendor_source_fixtures import FINNHUB_SAMPLE, FMP_SAMPLE

if TYPE_CHECKING:
    from finagent_terminal.llm.base import StructuredLLMProvider
    from finagent_terminal.ui.screens.reports import ReportsScreen


@dataclass(frozen=True, slots=True)
class FakeSession:
    snapshot: MarketDataSnapshot

    def get_or_fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot:
        assert symbol == "AAPL"
        assert period == "1mo"
        assert interval == "1d"
        return self.snapshot


class ReportsTestApp(App[None]):
    def __init__(self, screen: ReportsScreen) -> None:
        super().__init__()
        self._screen: ReportsScreen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


class MutableClock:
    def __init__(self) -> None:
        self.current: float = 0.0

    def now(self) -> float:
        return self.current


def client_factory(
    selection: LLMSelection | None = None,
) -> StructuredLLMProvider:
    _ = selection
    return FakeLLMClient(mode="1")


def failing_client_factory(
    selection: LLMSelection | None = None,
) -> StructuredLLMProvider:
    _ = selection
    return FailingBullishClient()


@dataclass(frozen=True, slots=True)
class FailingBullishClient:
    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        if prompt.section.id == "bullish_researcher":
            raise ProviderCallError(
                provider="fake",
                message=f"fake failure for {prompt.section.title}",
            )
        return ReportSectionResult(
            section_id=prompt.section.id,
            title=prompt.section.title,
            summary=f"{prompt.context.symbol} {prompt.section.title} generated.",
            key_points=(prompt.context.market_summary,),
            risks=("No additional supplied-context risks.",),
        )


class StepwiseLLMClient:
    def __init__(self) -> None:
        self._started: dict[str, Event] = {}
        self._release: dict[str, Event] = {}

    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        section_id = prompt.section.id
        self._event(self._started, section_id).set()
        if not self._event(self._release, section_id).wait(timeout=5.0):
            raise ProviderCallError(provider="fake", message=f"{section_id} timeout")
        return ReportSectionResult(
            section_id=section_id,
            title=prompt.section.title,
            summary=f"{prompt.context.symbol} {prompt.section.title} partial",
            key_points=(prompt.context.market_summary,),
            risks=("No additional supplied-context risks.",),
        )

    def has_started(self, section_id: str) -> bool:
        return self._event(self._started, section_id).is_set()

    def release(self, section_id: str) -> None:
        self._event(self._release, section_id).set()

    def _event(self, bucket: dict[str, Event], section_id: str) -> Event:
        event = bucket.get(section_id)
        if event is None:
            event = Event()
            bucket[section_id] = event
        return event


def aapl_snapshot() -> MarketDataSnapshot:
    fetched_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    frame = pd.DataFrame(
        {"Close": [100.0, 101.0], "Volume": [1000, 1200]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    return MarketDataSnapshot(
        "AAPL",
        "1mo",
        "1d",
        fetched_at,
        FreshnessState.FRESH,
        Quote(
            "AAPL",
            101.0,
            100.0,
            1.0,
            0.01,
            "USD",
            3_000_000_000_000,
            12_000_000,
            102.0,
            99.5,
            fetched_at,
        ),
        OhlcvFrame("AAPL", "1mo", "1d", frame, 2, fetched_at),
        Profile(
            "AAPL",
            "Apple Inc.",
            "Technology",
            "Consumer Electronics",
            "https://example.test",
            3_000_000_000_000,
            30.0,
            28.0,
            None,
            None,
            fetched_at,
        ),
    )


def report_source_response(
    request: httpx.Request,
    requests: list[httpx.Request],
) -> httpx.Response:
    if request.url.host in {
        "api.fiscaldata.treasury.gov",
        "api.stlouisfed.org",
    }:
        return macro_response(request, requests)
    return success_response(request, requests, sec_has_aapl=True)


def report_source_session() -> SourceSession:
    return SourceSession(
        env={
            "FINAGENT_SEC_USER_AGENT": SEC_SAMPLE,
            "FMP_API_KEY": FMP_SAMPLE,
            "FINNHUB_API_KEY": FINNHUB_SAMPLE,
            "FRED_API_KEY": "fred-test-secret",
        },
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: report_source_response(request, []),
            ),
        ),
    )
