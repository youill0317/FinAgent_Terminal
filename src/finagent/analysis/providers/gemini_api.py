"""Gemini through the Gemini API with ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).

Automatic function calling is off: FinAgent runs each tool call itself and
returns a function response. Model turns are sent back with every part they
streamed, so thought signatures on function calls are kept.
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

    from google import genai
    from google.genai import types

    from finagent.analysis.providers.base import (
        ProviderEvent,
        ProviderId,
        RunRequest,
    )

KEY_ENVS: Final = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
NOT_FOUND: Final = 404


class GeminiApiProvider:
    id: ProviderId = "gemini_api"
    title: str = "Gemini API 키"
    default_model: str = "gemini-4-argon"

    def __init__(self, env: Mapping[str, str]) -> None:
        """Read keys from ``env`` (the merged .env file and environment)."""
        self._key: str = env_value(env, *KEY_ENVS)

    async def status(self) -> ProviderStatus:
        if not installed("google.genai"):
            return ProviderStatus("not_installed", "uv sync --extra gemini")
        if not self._key:
            return ProviderStatus("missing_key", f"{KEY_ENVS[0]}가 없습니다")
        return ProviderStatus("ready")

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        from google import genai  # noqa: PLC0415 - optional dependency
        from google.genai import types  # noqa: PLC0415

        client = genai.Client(api_key=self._key)
        contents: list[types.Content] = [
            types.Content(
                role="user" if message.role == "user" else "model",
                parts=[types.Part.from_text(text=message.text)],
            )
            for message in request.conversation.messages()
        ]
        for _ in range(MAX_TOOL_CALLS + 2):
            parts: list[types.Part] = []
            usage: types.GenerateContentResponseUsageMetadata | None = None
            async for chunk in _chunks(client, request, contents):
                usage = chunk.usage_metadata or usage
                for part in _parts(chunk):
                    parts.append(part)
                    if part.function_call is None and part.text and not part.thought:
                        yield TextDelta(part.text)
            if usage is not None:
                yield Usage(usage.prompt_token_count, usage.candidates_token_count)
            calls = [part.function_call for part in parts if part.function_call]
            if not calls:
                return
            contents.append(types.Content(role="model", parts=parts))
            contents.append(await _respond(request, calls))
        msg = "도구 왕복이 너무 많아 멈췄습니다"
        raise ProviderError(msg)


async def _chunks(
    client: genai.Client, request: RunRequest, contents: list[types.Content]
) -> AsyncIterator[types.GenerateContentResponse]:
    """Stream one model turn; a 404 means the model is not open to this key."""
    from google.genai import errors, types  # noqa: PLC0415

    config = types.GenerateContentConfig(
        system_instruction=request.conversation.system,
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=spec.name,
                        description=spec.description,
                        parameters_json_schema=spec.schema,
                    )
                    for spec in request.tools
                ]
            )
        ],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    try:
        # The SDK's union of content types includes an untyped member.
        models = client.aio.models
        stream = await models.generate_content_stream(  # pyright: ignore[reportUnknownMemberType]
            model=request.model, contents=contents, config=config
        )
        async for chunk in stream:
            yield chunk
    except errors.ClientError as error:
        if error.code == NOT_FOUND:
            msg = f"모델 {request.model}을 쓸 수 없습니다 (이 키로는 없음)"
            raise ProviderError(msg) from error
        raise


async def _respond(
    request: RunRequest, calls: list[types.FunctionCall]
) -> types.Content:
    from google.genai import types  # noqa: PLC0415

    responses: list[types.Part] = []
    for call in calls:
        result = await request.run_tool(call.name or "", call.args or {})
        payload = cast("dict[str, object]", json.loads(result.text))
        responses.append(
            types.Part.from_function_response(name=call.name or "", response=payload)
        )
    return types.Content(role="user", parts=responses)


def _parts(chunk: types.GenerateContentResponse) -> list[types.Part]:
    out: list[types.Part] = []
    for candidate in chunk.candidates or []:
        if candidate.content is not None:
            out.extend(candidate.content.parts or [])
    return out
