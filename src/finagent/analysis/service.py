"""Runs one analysis at a time and keeps conversations in memory.

A conversation is pinned to the dashboard snapshot it started from. Each run
streams events: start, text, tool, usage, done (with the verification report)
or error. Nothing is written to disk except the provider choice, which goes
into the view settings file through ``settings.py``.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, final

from finagent import catalog
from finagent.analysis.context import build_document
from finagent.analysis.presets import PLACEHOLDER, PRESET_BY_ID, PRESETS
from finagent.analysis.prompt import Conversation, ToolRecord, Turn, system_prompt
from finagent.analysis.providers import DEFAULT_PROVIDER, registry
from finagent.analysis.providers.base import (
    McpEndpoint,
    ProviderError,
    RunRequest,
    TextDelta,
    ToolSpec,
)
from finagent.analysis.tools import TOOL_NAMES, TOOL_SPECS, Toolbox
from finagent.analysis.verify import verify
from finagent.settings import AnalysisChoice, load_analysis, save_analysis

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping
    from pathlib import Path

    from finagent.analysis.context import Document
    from finagent.analysis.evidence import Evidence
    from finagent.analysis.providers.base import Provider, ProviderStatus, RunTool
    from finagent.analysis.tools import Json, ToolResult
    from finagent.models import Dashboard
    from finagent.service import DashboardService

type Event = dict[str, object]
Scope = Literal["view", "all"]
ERROR_LIMIT: Final = 400
SPECS: Final = tuple(ToolSpec(name, text, schema) for name, text, schema in TOOL_SPECS)


@dataclass(slots=True)
class Thread:
    """One conversation: the pinned document plus every finished turn."""

    id: str
    document: Document
    deep: frozenset[str]
    epoch: int
    turns: list[Turn] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    received: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunInput:
    request: str
    provider: str = ""
    model: str = ""
    conversation: str = ""
    preset: str = ""
    scope: Scope = "view"


@final
class AnalysisService:
    __slots__ = (
        "_choice",
        "_dashboard",
        "_lock",
        "_mcp_run",
        "_mcp_token",
        "_mcp_url",
        "_providers",
        "_settings_path",
        "_threads",
        "_workspace",
    )

    def __init__(  # noqa: PLR0913
        self,
        dashboard: DashboardService,
        env: Mapping[str, str],
        *,
        workspace: Path,
        settings_path: Path | None = None,
        mcp_url: str | None = None,
        providers: Mapping[str, Provider] | None = None,
    ) -> None:
        """Wire the analysis to the dashboard service it reads snapshots from."""
        self._dashboard = dashboard
        self._providers = (
            dict(providers)
            if providers is not None
            else registry(env, demo=dashboard.demo)
        )
        self._workspace = workspace
        self._settings_path = settings_path
        self._mcp_url = mcp_url
        self._threads: dict[str, Thread] = {}
        self._lock = asyncio.Lock()
        self._mcp_run: RunTool | None = None
        self._mcp_token = ""
        saved = load_analysis(settings_path) if settings_path is not None else None
        first = DEFAULT_PROVIDER if DEFAULT_PROVIDER in self._providers else ""
        self._choice = saved or AnalysisChoice(first or next(iter(self._providers)))

    # ------------------------------------------------------------ choice and meta

    @property
    def choice(self) -> AnalysisChoice:
        return self._choice

    def model_for(self, provider_id: str) -> str:
        provider = self._providers[provider_id]
        return self._choice.models.get(provider_id) or provider.default_model

    def choose(self, provider_id: str, model: str = "") -> AnalysisChoice:
        if provider_id not in self._providers:
            msg = f"알 수 없는 공급자: {provider_id}"
            raise ProviderError(msg)
        models = dict(self._choice.models)
        if model.strip():
            models[provider_id] = model.strip()
        self._choice = AnalysisChoice(provider_id, models)
        if self._settings_path is not None:
            save_analysis(self._settings_path, self._choice)
        return self._choice

    async def statuses(self) -> dict[str, ProviderStatus]:
        ids = list(self._providers)
        found = await asyncio.gather(*(self._providers[i].status() for i in ids))
        return dict(zip(ids, found, strict=True))

    async def meta(self) -> Json:
        statuses = await self.statuses()
        return {
            "presets": [
                {"id": p.id, "title": p.title, "text": p.text, "panels": list(p.panels)}
                for p in PRESETS
            ],
            "placeholder": PLACEHOLDER,
            "tools": list(TOOL_NAMES),
            "choice": {"provider": self._choice.provider},
            "providers": [
                {
                    "id": provider.id,
                    "title": provider.title,
                    "kind": provider.kind,
                    "model": self.model_for(provider.id),
                    "default_model": provider.default_model,
                    "status": statuses[provider.id].kind,
                    "message": statuses[provider.id].message,
                }
                for provider in self._providers.values()
            ],
            "busy": self._lock.locked(),
        }

    # ------------------------------------------------------------ MCP bridge

    def mcp_token_ok(self, token: str) -> bool:
        return bool(self._mcp_token) and secrets.compare_digest(token, self._mcp_token)

    async def mcp_call(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        """A tool call that came in over ``/mcp`` during the current run."""
        run_tool = self._mcp_run
        if run_tool is None:
            msg = "실행 중인 분석이 없습니다"
            raise ProviderError(msg)
        return await run_tool(name, arguments)

    # ------------------------------------------------------------ run

    async def run(self, given: RunInput) -> AsyncGenerator[Event]:
        if self._lock.locked():
            yield {"type": "error", "message": "이미 분석이 실행 중입니다"}
            return
        async with self._lock:
            async for event in self._run(given):
                yield event

    async def _run(self, given: RunInput) -> AsyncGenerator[Event]:
        provider_id = given.provider or self._choice.provider
        provider = self._providers.get(provider_id)
        if provider is None:
            yield {"type": "error", "message": f"알 수 없는 공급자: {provider_id}"}
            return
        status = await provider.status()
        if not status.ready:
            yield {"type": "error", "message": status.message or status.kind}
            return
        if not given.request.strip():
            yield {"type": "error", "message": "요청을 입력해 주세요"}
            return
        model = given.model.strip() or self.model_for(provider_id)
        thread = await self._thread(given)
        yield {
            "type": "start",
            "conversation": thread.id,
            "snapshot": thread.document.snapshot,
            "view": thread.document.view,
            "stale": self._dashboard.refreshes != thread.epoch,
            "provider": provider_id,
            "provider_title": provider.title,
            "model": model,
        }
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        answer: list[str] = []
        records: list[ToolRecord] = []
        async with self._dashboard.fetch_context() as ctx:
            box = Toolbox(
                ctx,
                self._dashboard.series_cache,
                self._dashboard.today(),
                in_document=frozenset(thread.document.evidence),
            )

            async def run_tool(
                name: str, arguments: Mapping[str, object]
            ) -> ToolResult:
                result = await box.run(name, arguments)
                records.append(
                    ToolRecord(
                        name,
                        json.dumps(dict(arguments), ensure_ascii=False),
                        result.text,
                        result.ok,
                    )
                )
                await queue.put(
                    {"type": "tool", "summary": result.summary, "ok": result.ok}
                )
                return result

            request = RunRequest(
                conversation=Conversation(
                    system=system_prompt(thread.document.snapshot),
                    document=thread.document.text,
                    turns=tuple(thread.turns),
                    request=given.request.strip(),
                ),
                model=model,
                tools=SPECS,
                run_tool=run_tool,
                workspace=self._workspace,
                mcp=self._open_mcp(run_tool),
                evidence={**thread.evidence},
            )
            task = asyncio.create_task(self._pump(provider, request, queue, answer))
            try:
                while (event := await queue.get()) is not None:
                    yield event
            finally:
                if not task.done():
                    _ = task.cancel()
                _ = await asyncio.wait([task])
                self._mcp_run = None
                self._mcp_token = ""
            evidence = {**thread.evidence, **box.evidence}
        text = "".join(answer)
        if task.cancelled() or not task.result():
            return
        thread.turns.append(Turn(given.request.strip(), text, tuple(records)))
        thread.evidence = evidence
        thread.received.update(box.received)
        report = verify(text, evidence)
        yield {
            "type": "done",
            "verify": report.to_json(),
            "received": [
                {"cite": cite, "info": info} for cite, info in thread.received.items()
            ],
            "tool_calls": box.calls,
        }

    async def _pump(
        self,
        provider: Provider,
        request: RunRequest,
        queue: asyncio.Queue[Event | None],
        answer: list[str],
    ) -> bool:
        """Forward provider events to ``queue``; ``False`` if the run failed."""
        try:
            async for event in provider.stream(request):
                if isinstance(event, TextDelta):
                    answer.append(event.text)
                    await queue.put({"type": "text", "text": event.text})
                else:
                    await queue.put(
                        {
                            "type": "usage",
                            "input": event.input_tokens,
                            "output": event.output_tokens,
                            "note": event.note,
                        }
                    )
        except Exception as error:  # noqa: BLE001 - every provider failure is shown
            await queue.put({"type": "error", "message": _message(error)})
            await queue.put(None)
            return False
        await queue.put(None)
        return True

    def _open_mcp(self, run_tool: RunTool) -> McpEndpoint | None:
        """Let ``/mcp`` calls in for this run, recorded like any other call."""
        if self._mcp_url is None:
            return None
        self._mcp_run = run_tool
        self._mcp_token = secrets.token_urlsafe(24)
        return McpEndpoint(self._mcp_url, self._mcp_token)

    async def _thread(self, given: RunInput) -> Thread:
        known = self._threads.get(given.conversation)
        if known is not None:
            return known
        deep = self._dashboard.view
        extra: tuple[str, ...] = ()
        if given.scope == "all":
            deep = catalog.ALL_VIEWS
        elif (preset := PRESET_BY_ID.get(given.preset)) is not None:
            extra = tuple(panel for panel in preset.panels if panel not in deep)
            deep = deep | frozenset(extra)
        dashboard = await self._dashboard.get(deep=deep)
        document = build_document(
            dashboard, view=_view_label(dashboard, given.scope, extra)
        )
        thread = Thread(
            id=secrets.token_hex(6),
            document=document,
            deep=deep,
            epoch=self._dashboard.refreshes,
            evidence=dict(document.evidence),
        )
        self._threads[thread.id] = thread
        return thread


def _view_label(dashboard: Dashboard, scope: Scope, extra: tuple[str, ...]) -> str:
    if scope == "all":
        return "전체 매크로"
    titles = {view.id: view.title for view in dashboard.views}
    if not extra:
        return "지금 보기"
    return "지금 보기 + " + ", ".join(titles.get(panel, panel) for panel in extra)


def _message(error: BaseException) -> str:
    text = str(error).strip() or type(error).__name__
    if not isinstance(error, ProviderError):
        text = f"{type(error).__name__}: {text}"
    return text[:ERROR_LIMIT]
