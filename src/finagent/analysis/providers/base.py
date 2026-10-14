"""What every provider adapter implements, and the events it streams back.

An adapter turns the provider-neutral ``Conversation`` into its SDK's request
shape, streams text back, and calls ``run_tool`` when the model asks for one
of the data tools. Tool execution itself always happens in FinAgent.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from finagent.analysis.evidence import Evidence
    from finagent.analysis.prompt import Conversation
    from finagent.analysis.tools import Json, ToolResult

ProviderId = Literal[
    "claude_sub", "claude_api", "codex_sub", "openai_api", "gemini_api", "demo"
]
Kind = Literal["subscription", "api_key", "demo"]
StatusKind = Literal["ready", "missing_key", "login_needed", "not_installed", "error"]


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    kind: StatusKind
    message: str = ""

    @property
    def ready(self) -> bool:
        return self.kind == "ready"


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    note: str = ""


type ProviderEvent = TextDelta | Usage


class RunTool(Protocol):
    async def __call__(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        """Run one data tool and return its result (errors are results too)."""
        ...


@dataclass(frozen=True, slots=True)
class McpEndpoint:
    """FinAgent's ``/mcp`` URL and the bearer token valid for this run."""

    url: str
    token: str


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    schema: Json


def _no_evidence() -> dict[str, Evidence]:
    return {}


@dataclass(frozen=True, slots=True)
class RunRequest:
    conversation: Conversation
    model: str
    tools: tuple[ToolSpec, ...]
    run_tool: RunTool
    workspace: Path
    mcp: McpEndpoint | None = None
    evidence: Mapping[str, Evidence] = field(default_factory=_no_evidence)


class ProviderError(Exception):
    """A run failed in a way worth showing to the user as-is."""


class Provider(Protocol):
    id: ProviderId
    title: str
    kind: Kind
    default_model: str

    async def status(self) -> ProviderStatus:
        """Whether this provider can run now (installed, key or login present)."""
        ...

    def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        """Run one turn, calling ``request.run_tool`` for tool calls."""
        ...


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def env_value(env: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return value
    return ""
