"""Claude through the Anthropic API with ``ANTHROPIC_API_KEY`` (pay per token).

Streams with the official ``anthropic`` SDK and runs a manual tool loop:
``tool_use`` blocks are executed by FinAgent and sent back as ``tool_result``.
The system prompt and the indicator document are cache points, so follow-up
questions reuse the cached prefix. Server-side refusal fallbacks are on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from finagent.analysis.providers.base import (
    ProviderError,
    ProviderStatus,
    TextDelta,
    Usage,
    env_value,
    installed,
)
from finagent.analysis.tools import MAX_TOOL_CALLS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from anthropic.types.beta import (
        BetaMessageParam,
        BetaToolParam,
        BetaToolResultBlockParam,
    )

    from finagent.analysis.providers.base import (
        Kind,
        ProviderEvent,
        ProviderId,
        RunRequest,
    )

KEY_ENV: Final = "ANTHROPIC_API_KEY"
MAX_TOKENS: Final = 64000
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
EFFORT: Final = "high"


class ClaudeApiProvider:
    id: ProviderId = "claude_api"
    title: str = "Claude API 키"
    kind: Kind = "api_key"
    default_model: str = "claude-opus-5-5"

    def __init__(self, env: Mapping[str, str]) -> None:
        """Read keys from ``env`` (the merged .env file and environment)."""
        self._key: str = env_value(env, KEY_ENV)

    async def status(self) -> ProviderStatus:
        if not installed("anthropic"):
            return ProviderStatus("not_installed", "uv sync --extra claude-api")
        if not self._key:
            return ProviderStatus("missing_key", f"{KEY_ENV}가 없습니다")
        return ProviderStatus("ready")

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        import anthropic  # noqa: PLC0415 - optional dependency

        client = anthropic.AsyncAnthropic(api_key=self._key)
        conversation = request.conversation
        messages = _messages(request)
        tools: list[BetaToolParam] = [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.schema,
                "eager_input_streaming": True,
            }
            for spec in request.tools
        ]
        for _ in range(MAX_TOOL_CALLS + 2):
            async with client.beta.messages.stream(
                model=request.model,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": conversation.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
                tools=tools,
                output_config={"effort": EFFORT},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                final = await stream.get_final_message()
            yield Usage(final.usage.input_tokens, final.usage.output_tokens)
            if final.stop_reason == "refusal":
                msg = "모델이 이 요청을 거절했습니다"
                raise ProviderError(msg)
            messages.append({"role": "assistant", "content": final.content})
            if final.stop_reason == "pause_turn":
                continue
            uses = [block for block in final.content if block.type == "tool_use"]
            if not uses:
                return
            if final.stop_reason == "max_tokens":
                msg = "답이 길이 제한에 걸려 도구 입력이 끊겼습니다"
                raise ProviderError(msg)
            results: list[BetaToolResultBlockParam] = []
            for block in uses:
                result = await request.run_tool(block.name, block.input)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.text,
                        "is_error": not result.ok,
                    }
                )
            messages.append({"role": "user", "content": results})
        msg = "도구 왕복이 너무 많아 멈췄습니다"
        raise ProviderError(msg)


def _messages(request: RunRequest) -> list[BetaMessageParam]:
    out: list[BetaMessageParam] = []
    for message in request.conversation.messages():
        if message.role == "user" and len(message.parts) == 2:  # noqa: PLR2004
            document, ask = message.parts
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": document,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": ask},
                    ],
                }
            )
        else:
            role = "user" if message.role == "user" else "assistant"
            out.append({"role": role, "content": message.text})
    return out
