"""GPT through my own ChatGPT plan, via the Codex Python SDK (beta).

The SDK drives the Codex runtime it ships with, using the login from
``codex login``. Each run is an ephemeral thread (no session files) in a
read-only sandbox that never asks for approval. The shell and other built-in
features are switched off; FinAgent's data tools come in over its ``/mcp``
endpoint with a token that is only valid for this run.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Final

from finagent.analysis.providers.base import (
    ProviderError,
    ProviderStatus,
    TextDelta,
    Usage,
    installed,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from finagent.analysis.providers.base import (
        Kind,
        ProviderEvent,
        ProviderId,
        RunRequest,
    )

TOKEN_ENV: Final = "FINAGENT_MCP_TOKEN"  # noqa: S105 - a variable name, not a secret
STATUS_TTL_SECONDS: Final = 60.0
# Checked with openai-codex 0.162.1: these remove the shell, exec and the
# other agent features from what the model sees.
DISABLED_FEATURES: Final = (
    "shell_tool",
    "unified_exec",
    "multi_agent",
    "goals",
    "sleep_tool",
    "apps",
    "plugins",
    "image_generation",
    "browser_use",
    "computer_use",
    "hooks",
    "skill_search",
    "tool_suggest",
    "workspace_dependencies",
    "memories",
)
# Codex should use the ChatGPT login, not an API key from the environment.
DROPPED_ENV: Final = ("OPENAI_API_KEY", "CODEX_API_KEY")


class CodexSubscriptionProvider:
    id: ProviderId = "codex_sub"
    title: str = "ChatGPT 구독"
    kind: Kind = "subscription"
    default_model: str = "gpt-6-astra"

    def __init__(self, env: Mapping[str, str]) -> None:
        """Read keys from ``env`` (the merged .env file and environment)."""
        self._env: Mapping[str, str] = env
        self._cached: tuple[float, ProviderStatus] | None = None

    async def status(self) -> ProviderStatus:
        if not installed("openai_codex"):
            return ProviderStatus("not_installed", "uv sync --extra codex-sub")
        now = time.monotonic()
        if self._cached is not None and now - self._cached[0] < STATUS_TTL_SECONDS:
            return self._cached[1]
        found = await _login_status()
        self._cached = (now, found)
        return found

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        from openai_codex import (  # noqa: PLC0415 - optional dependency
            ApprovalMode,
            AsyncCodex,
            CodexConfig,
            Sandbox,
        )
        from openai_codex.generated.v2_all import (  # noqa: PLC0415
            AgentMessageDeltaNotification,
            ThreadTokenUsageUpdatedNotification,
            TurnCompletedNotification,
        )

        if request.mcp is None:
            msg = "도구 주소(/mcp)가 없어 ChatGPT 구독 경로를 쓸 수 없습니다"
            raise ProviderError(msg)
        overrides = (
            *(f"features.{name}=false" for name in DISABLED_FEATURES),
            f'mcp_servers.finagent.url="{request.mcp.url}"',
            f'mcp_servers.finagent.bearer_token_env_var="{TOKEN_ENV}"',
        )
        workspace = str(request.workspace)
        config = CodexConfig(
            config_overrides=overrides, env=_child_env(request.mcp.token), cwd=workspace
        )
        conversation = request.conversation
        prompt = (
            conversation.transcript()
            if conversation.turns
            else conversation.messages()[0].text
        )
        async with AsyncCodex(config=config) as codex:
            thread = await codex.thread_start(
                base_instructions=conversation.system,
                ephemeral=True,
                sandbox=Sandbox.read_only,
                approval_mode=ApprovalMode.deny_all,
                cwd=workspace,
                model=request.model,
            )
            turn = await thread.turn(prompt)
            async for note in turn.stream():
                payload = note.payload
                if isinstance(payload, AgentMessageDeltaNotification):
                    yield TextDelta(payload.delta)
                elif isinstance(payload, ThreadTokenUsageUpdatedNotification):
                    last = payload.token_usage.last
                    yield Usage(last.input_tokens, last.output_tokens, "구독 한도 사용")
                elif isinstance(payload, TurnCompletedNotification):
                    error = payload.turn.error
                    if error is not None:
                        msg = f"ChatGPT 구독 실행 실패: {error.message}"
                        raise ProviderError(msg)


def _child_env(token: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in DROPPED_ENV}
    env[TOKEN_ENV] = token
    return env


async def _login_status() -> ProviderStatus:
    from openai_codex import AsyncCodex, CodexConfig  # noqa: PLC0415

    env = {key: value for key, value in os.environ.items() if key not in DROPPED_ENV}
    try:
        async with AsyncCodex(config=CodexConfig(env=env)) as codex:
            account = await codex.account()
    except Exception as error:  # noqa: BLE001 - any failure means "not usable now"
        return ProviderStatus("error", f"Codex를 시작하지 못했습니다: {error}"[:200])
    if account.account is None:
        return ProviderStatus("login_needed", "터미널에서 codex login")
    return ProviderStatus("ready", "로그인됨")
