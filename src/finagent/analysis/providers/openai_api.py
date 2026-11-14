"""OpenAI Responses API 스트리밍·함수 호출 및 웹 검색 어댑터."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Final, cast

from finagent.analysis.providers.base import (
    NO_MODEL,
    TOO_MANY_ROUNDS,
    TRUNCATED,
    WEB_STEP_LIMIT,
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

    from openai import AsyncOpenAI
    from openai.types.responses import (
        Response,
        ResponseFunctionWebSearch,
        ResponseIncludable,
        ResponseInputParam,
        ResponseStreamEvent,
        ToolParam,
    )

    from finagent.analysis.providers.base import (
        ProviderEvent,
        ProviderId,
        ProviderStatus,
        RunRequest,
    )

KEY_ENV: Final = "OPENAI_API_KEY"


class OpenAiApiProvider:
    id: ProviderId = "openai_api"
    title: str = "OpenAI API 키"
    default_model: str = "gpt-6-astra"

    def __init__(self, env: Mapping[str, str]) -> None:
        self._key: str = env_value(env, KEY_ENV)

    async def status(self) -> ProviderStatus:
        return key_status("openai", "openai", self._key, KEY_ENV)

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        import openai  # noqa: PLC0415  # 선택적 의존성(openai) 지연 임포트

        async with openai.AsyncOpenAI(api_key=self._key) as client:
            try:
                async for event in _rounds(client, request):
                    yield event
            except openai.NotFoundError as error:
                raise ProviderError(NO_MODEL.format(model=request.model)) from error


async def _rounds(
    client: AsyncOpenAI, request: RunRequest
) -> AsyncIterator[ProviderEvent]:
    items: ResponseInputParam = [
        {
            "role": "user" if message.role == "user" else "assistant",
            "content": message.text,
        }
        for message in request.conversation.messages()
    ]
    include: list[ResponseIncludable] = ["reasoning.encrypted_content"]
    if request.web:
        include.append("web_search_call.action.sources")
    used = 0
    for _ in range(request.max_rounds + 2):
        web = request.web and used < WEB_STEP_LIMIT
        stream = await client.responses.create(
            model=request.model,
            instructions=request.conversation.system,
            input=items,
            tools=_tools(request, web=web),
            max_tool_calls=WEB_STEP_LIMIT - used if web else None,
            stream=True,
            store=False,
            include=include,
        )
        final: Response | None = None
        async for event in stream:
            for found in _stream_events(event):
                yield found
            final = _finished(event) or final
        if final is None:
            msg = "응답 미완료 종료"
            raise ProviderError(msg)
        if final.usage is not None:
            yield Usage(final.usage.input_tokens, final.usage.output_tokens)
        if request.web:
            yield WebFound(_cited(final))
        used += sum(item.type == "web_search_call" for item in final.output)
        calls = [item for item in final.output if item.type == "function_call"]
        if not calls:
            return
        items.extend(
            cast(
                "ResponseInputParam",
                [item.model_dump(exclude_none=True) for item in final.output],
            )
        )
        results = await asyncio.gather(
            *(
                request.run_tool(item.name, _arguments(item.name, item.arguments))
                for item in calls
            )
        )
        for item, result in zip(calls, results, strict=True):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result.text,
                }
            )
        yield ToolRound()
    raise ProviderError(TOO_MANY_ROUNDS)


def _tools(request: RunRequest, *, web: bool) -> list[ToolParam]:
    tools: list[ToolParam] = [
        {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.schema,
            "strict": False,
        }
        for spec in request.tools
    ]
    if web:
        tools.append({"type": "web_search"})
    return tools


def _stream_events(event: ResponseStreamEvent) -> tuple[ProviderEvent, ...]:
    if event.type == "response.output_text.delta":
        return (TextDelta(event.delta),)
    if event.type == "response.output_item.done" and (
        event.item.type == "web_search_call"
    ):
        return (
            WebStep(_step(event.item), ok=event.item.status != "failed"),
            WebFound(_action_sources(event.item)),
        )
    return ()


def _step(item: ResponseFunctionWebSearch) -> str:
    action = item.action
    if item.status == "failed":
        return "웹 검색 실패: 응답 실패"
    if action.type == "search":
        return f"웹 검색: {action.query or ', '.join(action.queries or [])}"
    if action.type == "open_page":
        return f"원문 읽기: {action.url or ''}"
    return f"원문 읽기: {action.url}"


def _action_sources(item: ResponseFunctionWebSearch) -> tuple[WebSource, ...]:
    action = item.action
    if action.type == "search":
        return tuple(WebSource(source.url) for source in action.sources or [])
    if action.url:
        return (WebSource(action.url),)
    return ()


def _cited(final: Response) -> tuple[WebSource, ...]:
    return tuple(
        WebSource(note.url, note.title)
        for item in final.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
        for note in part.annotations
        if note.type == "url_citation"
    )


def _arguments(name: str, raw: str) -> dict[str, object]:
    try:
        parsed = cast("object", json.loads(raw or "{}"))
    except json.JSONDecodeError as error:
        raise ProviderError(f"{name}: 도구 입력 JSON 파싱 오류") from error
    if not isinstance(parsed, dict):
        raise ProviderError(f"{name}: 도구 입력 JSON 객체 형식 오류")
    return cast("dict[str, object]", parsed)


def _finished(event: ResponseStreamEvent) -> Response | None:
    if event.type == "response.completed":
        return event.response
    if event.type == "response.incomplete":
        details = event.response.incomplete_details
        reason = details.reason if details is not None else None
        if reason == "max_output_tokens":
            raise ProviderError(TRUNCATED)
        raise ProviderError(f"응답 미완료 종료: {reason}")
    if event.type == "response.failed":
        error = event.response.error
        msg = error.message if error is not None else "응답 실패"
        raise ProviderError(msg)
    return None
