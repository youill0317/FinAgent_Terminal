"""Claude through my own Claude subscription, via the Claude Agent SDK.

The SDK runs the Claude Code binary it ships with, logged in through
``claude auth login`` (Anthropic's own flow; this app never reads or stores
the token). Built-in tools, settings files and transcript writes are off.
The only tools are FinAgent's data tools, served in-process as an SDK MCP
server.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from finagent.analysis.providers.base import (
    ProviderError,
    ProviderStatus,
    TextDelta,
    Usage,
    installed,
)
from finagent.analysis.tools import MAX_TOOL_CALLS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

    from claude_agent_sdk import SdkMcpTool

    from finagent.analysis.providers.base import (
        Kind,
        ProviderEvent,
        ProviderId,
        RunRequest,
        ToolSpec,
    )

    type Handler = Callable[[object], Awaitable[dict[str, object]]]

SERVER: Final = "finagent"
STATUS_TTL_SECONDS: Final = 60.0
STATUS_TIMEOUT_SECONDS: Final = 15.0
RESULT_SIZE: Final = 200_000
# Keep the subscription login in charge and nothing of the session on disk.
CHILD_ENV: Final = {"CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1", "ANTHROPIC_API_KEY": ""}


class ClaudeSubscriptionProvider:
    id: ProviderId = "claude_sub"
    title: str = "Claude 구독"
    kind: Kind = "subscription"
    default_model: str = "claude-opus-5-5"

    def __init__(self, env: Mapping[str, str]) -> None:
        """Read keys from ``env`` (the merged .env file and environment)."""
        self._env: Mapping[str, str] = env
        self._cached: tuple[float, ProviderStatus] | None = None

    async def status(self) -> ProviderStatus:
        if not installed("claude_agent_sdk"):
            return ProviderStatus("not_installed", "uv sync --extra claude-sub")
        now = time.monotonic()
        if self._cached is not None and now - self._cached[0] < STATUS_TTL_SECONDS:
            return self._cached[1]
        found = await _login_status()
        self._cached = (now, found)
        return found

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        from claude_agent_sdk import (  # noqa: PLC0415 - optional dependency
            ClaudeAgentOptions,
            ResultMessage,
            StreamEvent,
            create_sdk_mcp_server,
            query,
        )

        server = create_sdk_mcp_server(
            SERVER, tools=[_sdk_tool(spec, request) for spec in request.tools]
        )
        options = ClaudeAgentOptions(
            system_prompt=request.conversation.system,
            tools=[],
            mcp_servers={SERVER: server},
            allowed_tools=[f"mcp__{SERVER}__{spec.name}" for spec in request.tools],
            strict_mcp_config=True,
            setting_sources=[],
            permission_mode="dontAsk",
            max_turns=MAX_TOOL_CALLS + 2,
            include_partial_messages=True,
            model=request.model,
            cwd=str(request.workspace),
            env=dict(CHILD_ENV),
        )
        conversation = request.conversation
        prompt = (
            conversation.transcript()
            if conversation.turns
            else conversation.messages()[0].text
        )
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, StreamEvent):
                text = _text_delta(message.event, message.parent_tool_use_id)
                if text:
                    yield TextDelta(text)
            elif isinstance(message, ResultMessage):
                if message.is_error:
                    detail = message.result or message.subtype
                    msg = f"Claude 구독 실행 실패: {detail}"
                    raise ProviderError(msg)
                usage = message.usage or {}
                yield Usage(
                    _int(usage.get("input_tokens")),
                    _int(usage.get("output_tokens")),
                    "구독 한도 사용",
                )


def _sdk_tool(spec: ToolSpec, request: RunRequest) -> SdkMcpTool[object]:
    from claude_agent_sdk import ToolAnnotations, tool  # noqa: PLC0415

    async def handler(args: object) -> dict[str, object]:
        arguments = cast("dict[str, object]", args) if isinstance(args, dict) else {}
        result = await request.run_tool(spec.name, arguments)
        return {
            "content": [{"type": "text", "text": result.text}],
            "is_error": not result.ok,
        }

    annotations = ToolAnnotations.model_validate(
        {"readOnlyHint": True, "maxResultSizeChars": RESULT_SIZE}
    )
    decorate = cast(
        "Callable[[Handler], SdkMcpTool[object]]",
        tool(spec.name, spec.description, dict(spec.schema), annotations=annotations),
    )
    return decorate(handler)


def _text_delta(event: Mapping[str, object], parent: str | None) -> str:
    if parent is not None or event.get("type") != "content_block_delta":
        return ""
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return ""
    body = cast("dict[str, object]", delta)
    text = body.get("text")
    return text if body.get("type") == "text_delta" and isinstance(text, str) else ""


def _int(value: object) -> int | None:
    return value if isinstance(value, int) else None


async def _login_status() -> ProviderStatus:
    """Ask the bundled Claude Code CLI whether a subscription login exists."""
    import claude_agent_sdk  # noqa: PLC0415

    cli = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    if not cli.is_file():
        return ProviderStatus("ready", "로그인 상태는 실행할 때 확인합니다")
    try:
        process = await asyncio.create_subprocess_exec(
            str(cli),
            "auth",
            "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(process.communicate(), STATUS_TIMEOUT_SECONDS)
    except (OSError, TimeoutError):
        return ProviderStatus("error", "Claude Code 로그인 상태를 확인하지 못했습니다")
    if process.returncode != 0:
        return ProviderStatus("login_needed", "터미널에서 claude auth login")
    method = ""
    try:
        parsed = cast("object", json.loads(out.decode() or "{}"))
        if isinstance(parsed, dict):
            method = str(cast("dict[str, object]", parsed).get("authMethod", ""))
    except (UnicodeDecodeError, json.JSONDecodeError):
        method = ""
    if method and method not in {"claude.ai", "oauth_token"}:
        return ProviderStatus("login_needed", f"구독 로그인이 아님({method})")
    return ProviderStatus("ready", "로그인됨")
