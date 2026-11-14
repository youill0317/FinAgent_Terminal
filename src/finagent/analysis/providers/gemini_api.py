"""Google Gemini API 스트리밍·함수 호출 및 그라운딩 웹 검색 어댑터."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Final, cast

from finagent.analysis.providers.base import (
    NO_MODEL,
    TOO_MANY_ROUNDS,
    TRUNCATED,
    ProviderError,
    TextDelta,
    ToolRound,
    Usage,
    WebFound,
    WebStep,
    env_value,
    key_status,
)
from finagent.analysis.web import WebSource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from google.genai import types
    from google.genai.client import AsyncClient

    from finagent.analysis.providers.base import (
        ProviderEvent,
        ProviderId,
        ProviderStatus,
        RunRequest,
    )

KEY_ENVS: Final = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
NOT_FOUND: Final = 404
WEB_MAX_OUTPUT: Final = 65536


class GeminiApiProvider:
    id: ProviderId = "gemini_api"
    title: str = "Gemini API 키"
    default_model: str = "gemini-4-argon"

    def __init__(self, env: Mapping[str, str]) -> None:
        self._key: str = env_value(env, *KEY_ENVS)

    async def status(self) -> ProviderStatus:
        return key_status("google.genai", "gemini", self._key, KEY_ENVS[0])

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        # 선택적 의존성(google-genai) 지연 임포트
        from google import genai  # noqa: PLC0415
        from google.genai import errors  # noqa: PLC0415

        async with genai.Client(api_key=self._key).aio as client:
            try:
                async for event in _rounds(client, request):
                    yield event
            except errors.ClientError as error:
                if error.code == NOT_FOUND:
                    raise ProviderError(NO_MODEL.format(model=request.model)) from error
                raise


async def _rounds(
    client: AsyncClient, request: RunRequest
) -> AsyncIterator[ProviderEvent]:
    from google.genai import types  # noqa: PLC0415  # 선택적 의존성 지연 임포트

    contents: list[types.ContentUnion] = [
        types.Content(
            role="user" if message.role == "user" else "model",
            parts=[types.Part.from_text(text=message.text)],
        )
        for message in request.conversation.messages()
    ]
    for _ in range(request.max_rounds + 2):
        parts: list[types.Part] = []
        seen: set[str] = set()
        usage: types.GenerateContentResponseUsageMetadata | None = None
        async for chunk in _chunks(client, request, contents):
            usage = chunk.usage_metadata or usage
            steps, sources = _web_results(chunk, seen)
            for step in steps:
                yield step
            if sources:
                yield WebFound(sources)
            for part in _parts(chunk):
                parts.append(part)
                if part.function_call is None and part.text and not part.thought:
                    yield TextDelta(part.text)
            if _cut(chunk):
                raise ProviderError(TRUNCATED)
        if usage is not None:
            yield Usage(
                usage.prompt_token_count,
                (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
            )
        calls = [part.function_call for part in parts if part.function_call]
        if not calls:
            return
        contents.append(types.Content(role="model", parts=parts))
        contents.append(await _respond(request, calls))
        yield ToolRound()
    raise ProviderError(TOO_MANY_ROUNDS)


async def _chunks(
    client: AsyncClient, request: RunRequest, contents: list[types.ContentUnion]
) -> AsyncIterator[types.GenerateContentResponse]:
    from google.genai import types  # noqa: PLC0415  # 선택적 의존성 지연 임포트

    tools = [
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
    ]
    config = types.GenerateContentConfig(
        system_instruction=request.conversation.system,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if request.web:
        config.tools = [
            *tools,
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(url_context=types.UrlContext()),
        ]
        config.max_output_tokens = WEB_MAX_OUTPUT
        config.tool_config = types.ToolConfig(
            include_server_side_tool_invocations=True,
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.VALIDATED
            ),
        )
    stream = await client.models.generate_content_stream(
        model=request.model, contents=contents, config=config
    )
    async for chunk in stream:
        yield chunk


async def _respond(
    request: RunRequest, calls: list[types.FunctionCall]
) -> types.Content:
    from google.genai import types  # noqa: PLC0415  # 선택적 의존성 지연 임포트

    results = await asyncio.gather(
        *(request.run_tool(call.name or "", call.args or {}) for call in calls)
    )
    responses = [
        types.Part.from_function_response(
            name=call.name or "",
            response=cast("dict[str, object]", json.loads(result.text)),
        )
        for call, result in zip(calls, results, strict=True)
    ]
    return types.Content(role="user", parts=responses)


def _web_results(
    chunk: types.GenerateContentResponse, seen: set[str]
) -> tuple[list[WebStep], tuple[WebSource, ...]]:
    steps: list[WebStep] = []
    sources: list[WebSource] = []
    for candidate in chunk.candidates or []:
        grounding = candidate.grounding_metadata
        if grounding is not None:
            for query in grounding.web_search_queries or []:
                if query not in seen:
                    seen.add(query)
                    steps.append(WebStep(f"웹 검색: {query}"))
            sources.extend(
                WebSource(found.web.uri, found.web.title or "")
                for found in grounding.grounding_chunks or []
                if found.web is not None and found.web.uri
            )
        context = candidate.url_context_metadata
        for item in (context.url_metadata or []) if context is not None else []:
            url = item.retrieved_url or ""
            if url and url not in seen:
                seen.add(url)
                steps.append(WebStep(f"원문 읽기: {url}"))
                sources.append(WebSource(url))
    return steps, tuple(sources)


def _cut(chunk: types.GenerateContentResponse) -> bool:
    from google.genai import types  # noqa: PLC0415  # 선택적 의존성 지연 임포트

    return any(
        candidate.finish_reason == types.FinishReason.MAX_TOKENS
        for candidate in chunk.candidates or []
    )


def _parts(chunk: types.GenerateContentResponse) -> list[types.Part]:
    out: list[types.Part] = []
    for candidate in chunk.candidates or []:
        if candidate.content is not None:
            out.extend(candidate.content.parts or [])
    return out
