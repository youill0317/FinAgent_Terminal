"""GPT through the OpenAI Responses API with ``OPENAI_API_KEY``.

Stateless: ``store`` is off and the encrypted reasoning items are passed back
with each tool round, so nothing of the conversation is kept at OpenAI.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

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

    from openai.types.responses import (
        FunctionToolParam,
        Response,
        ResponseInputParam,
        ResponseStreamEvent,
    )

    from finagent.analysis.providers.base import (
        ProviderEvent,
        ProviderId,
        RunRequest,
    )

KEY_ENV: Final = "OPENAI_API_KEY"


class OpenAiApiProvider:
    id: ProviderId = "openai_api"
    title: str = "OpenAI API 키"
    default_model: str = "gpt-6-astra"

    def __init__(self, env: Mapping[str, str]) -> None:
        """Read keys from ``env`` (the merged .env file and environment)."""
        self._key: str = env_value(env, KEY_ENV)

    async def status(self) -> ProviderStatus:
        if not installed("openai"):
            return ProviderStatus("not_installed", "uv sync --extra openai")
        if not self._key:
            return ProviderStatus("missing_key", f"{KEY_ENV}가 없습니다")
        return ProviderStatus("ready")

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        import openai  # noqa: PLC0415 - optional dependency

        client = openai.AsyncOpenAI(api_key=self._key)
        items: ResponseInputParam = [
            {
                "role": "user" if message.role == "user" else "assistant",
                "content": message.text,
            }
            for message in request.conversation.messages()
        ]
        tools: list[FunctionToolParam] = [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.schema,
                "strict": False,
            }
            for spec in request.tools
        ]
        for _ in range(MAX_TOOL_CALLS + 2):
            stream = await client.responses.create(
                model=request.model,
                instructions=request.conversation.system,
                input=items,
                tools=tools,
                stream=True,
                store=False,
                include=["reasoning.encrypted_content"],
            )
            final: Response | None = None
            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield TextDelta(event.delta)
                else:
                    final = _finished(event) or final
            if final is None:
                msg = "응답이 끝나지 않았습니다"
                raise ProviderError(msg)
            if final.usage is not None:
                yield Usage(final.usage.input_tokens, final.usage.output_tokens)
            calls = [item for item in final.output if item.type == "function_call"]
            if not calls:
                return
            items.extend(
                cast(
                    "ResponseInputParam",
                    [item.model_dump(exclude_none=True) for item in final.output],
                )
            )
            for item in calls:
                result = await request.run_tool(item.name, _arguments(item.arguments))
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result.text,
                    }
                )
        msg = "도구 왕복이 너무 많아 멈췄습니다"
        raise ProviderError(msg)


def _arguments(raw: str | None) -> dict[str, object]:
    try:
        parsed = cast("object", json.loads(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}


def _finished(event: ResponseStreamEvent) -> Response | None:
    """The final response from a terminal event; raises if the run failed."""
    if event.type in {"response.completed", "response.incomplete"}:
        return cast("Response", getattr(event, "response", None))
    if event.type == "response.failed":
        error = event.response.error
        msg = error.message if error is not None else "응답 실패"
        raise ProviderError(msg)
    return None
