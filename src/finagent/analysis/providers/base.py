"""AI 분석 공급자 공통 프로토콜·이벤트 타입 및 상태 점검 유틸리티."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from finagent.analysis.prompt import Conversation
    from finagent.analysis.tools import Json, ToolResult
    from finagent.analysis.web import WebSource

ProviderId = Literal["claude_api", "openai_api", "gemini_api", "demo"]
StatusKind = Literal["ready", "missing_key", "not_installed"]
StepKind = Literal["data", "web", "filter", "thread"]
WEB_STEP_LIMIT: Final = 20
TOO_MANY_ROUNDS: Final = "도구 왕복 한도 초과에 따른 실행 중단"
TRUNCATED: Final = "출력 길이 제한 도달에 따른 응답 절단"
NO_MODEL: Final = "미지원 모델: {model} (현재 키 사용 불가)"


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


@dataclass(frozen=True, slots=True)
class WebStep:
    summary: str
    ok: bool = True
    kind: StepKind = "web"


@dataclass(frozen=True, slots=True)
class WebFound:
    sources: tuple[WebSource, ...]


@dataclass(frozen=True, slots=True)
class ToolRound: ...


type ProviderEvent = TextDelta | Usage | WebStep | WebFound | ToolRound


class RunTool(Protocol):
    async def __call__(
        self, name: str, arguments: Mapping[str, object]
    ) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    schema: Json


@dataclass(frozen=True, slots=True)
class RunRequest:
    conversation: Conversation
    model: str
    tools: tuple[ToolSpec, ...]
    run_tool: RunTool
    max_rounds: int
    web: bool = False
    kind: str = "page"


class ProviderError(Exception): ...


class Provider(Protocol):
    id: ProviderId
    title: str
    default_model: str

    async def status(self) -> ProviderStatus: ...

    def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]: ...


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def key_status(module: str, extra: str, key: str, env_name: str) -> ProviderStatus:
    if not installed(module):
        return ProviderStatus("not_installed", f"uv sync --extra {extra}")
    if not key:
        return ProviderStatus("missing_key", f"{env_name} 미설정")
    return ProviderStatus("ready")


def env_value(env: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return value
    return ""
