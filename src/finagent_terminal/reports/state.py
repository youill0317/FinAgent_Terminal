from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from finagent_terminal.analytics import MissingCloseColumnError, compute_indicators
from finagent_terminal.llm import LLMSelection, ProviderName, StructuredLLMProvider
from finagent_terminal.llm.base import ProviderError
from finagent_terminal.reports.models import (
    DepthMode,
    ReportContext,
    ReportSectionPrompt,
    ReportSectionResult,
)
from finagent_terminal.reports.pipeline import ReportPipeline
from finagent_terminal.reports.progress import ReportProgress, new_report_progress

if TYPE_CHECKING:
    from finagent_terminal.data.models import MarketDataSnapshot
    from finagent_terminal.reports.pipeline import ReportRunResult
    from finagent_terminal.ui.state import AppState


class ReportDataSession(Protocol):
    def get_or_fetch(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> MarketDataSnapshot: ...


class ReportClientFactory(Protocol):
    def __call__(
        self,
        selection: LLMSelection | None = None,
    ) -> StructuredLLMProvider: ...


@dataclass(frozen=True, slots=True)
class ReportRequest:
    symbol: str
    provider: ProviderName
    model: str
    depth: DepthMode


@dataclass(frozen=True, slots=True)
class ScreenReportResult:
    context: ReportContext
    result: ReportRunResult


@dataclass(frozen=True, slots=True)
class ReportScreenState:
    progress: ReportProgress
    latest: ScreenReportResult | None = None
    export_status: str = "No report exported."
    running: bool = False


type ReportUpdate = Callable[[ScreenReportResult], None]


@dataclass(frozen=True, slots=True)
class ReportJob:
    request: ReportRequest
    session: ReportDataSession
    client_factory: ReportClientFactory
    started_at: float = 0.0
    on_update: ReportUpdate | None = None


_REPORT_STATES: dict[int, ReportScreenState] = {}


def build_report_context(symbol: str, snapshot: MarketDataSnapshot) -> ReportContext:
    warnings: list[str] = []
    if snapshot.error is not None:
        warnings.append(snapshot.error.message)
    return ReportContext(
        symbol=symbol,
        market_summary=_market_summary(snapshot),
        indicator_summary=_indicator_summary(snapshot),
        warnings=tuple(warnings),
    )


def report_state_for(state: AppState) -> ReportScreenState:
    key = id(state)
    report_state = _REPORT_STATES.get(key)
    if report_state is not None:
        return report_state
    report_state = ReportScreenState(progress=_default_progress())
    _REPORT_STATES[key] = report_state
    return report_state


def set_report_state_for(
    state: AppState,
    report_state: ReportScreenState,
) -> ReportScreenState:
    _REPORT_STATES[id(state)] = report_state
    return report_state


def run_screen_report(job: ReportJob) -> ScreenReportResult:
    request = job.request
    snapshot = job.session.get_or_fetch(request.symbol)
    context = build_report_context(request.symbol, snapshot)
    result = ReportPipeline(
        client=_client_or_failure(request, job.client_factory),
        provider=request.provider.value,
        model=request.model,
        on_update=lambda update: _publish(job, context, update),
    ).run(
        context=context,
        depth=request.depth,
        started_at=job.started_at,
    )
    return ScreenReportResult(context=context, result=result)


def _default_progress() -> ReportProgress:
    return new_report_progress(
        provider=ProviderName.OPENAI.value,
        model="gpt-4o-mini",
        depth=DepthMode.STANDARD,
    )


def _publish(
    job: ReportJob,
    context: ReportContext,
    result: ReportRunResult,
) -> None:
    if job.on_update is None:
        return
    job.on_update(ScreenReportResult(context=context, result=result))


def _client_or_failure(
    request: ReportRequest,
    client_factory: ReportClientFactory,
) -> StructuredLLMProvider:
    try:
        return client_factory(
            LLMSelection(provider=request.provider, model=request.model),
        )
    except ProviderError as exc:
        return _FailingClient(str(exc))


@dataclass(frozen=True, slots=True)
class _FailingClient:
    message: str

    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        _ = prompt
        raise ProviderError(self.message)


def _market_summary(snapshot: MarketDataSnapshot) -> str:
    parts = [f"{snapshot.symbol} freshness {snapshot.freshness.value}"]
    if snapshot.quote is not None and snapshot.quote.last_price is not None:
        parts.append(f"last price {snapshot.quote.last_price:.2f}")
    if snapshot.quote is not None and snapshot.quote.change_percent is not None:
        parts.append(f"change {snapshot.quote.change_percent:.2%}")
    if snapshot.profile is not None and snapshot.profile.name is not None:
        parts.append(f"name {snapshot.profile.name}")
    return "; ".join(parts)


def _indicator_summary(snapshot: MarketDataSnapshot) -> str:
    if snapshot.history is None or snapshot.history.frame.empty:
        return ""
    try:
        indicators = compute_indicators(snapshot.history.frame)
    except MissingCloseColumnError:
        return ""
    if indicators.empty:
        return ""
    latest = indicators.iloc[-1]
    return (
        f"close {latest['close']:.2f}; "
        f"normalized_return {latest['normalized_return']:.2f}; "
        f"rsi_14 {latest['rsi_14']:.2f}"
    )
