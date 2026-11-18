"""Anthropic Claude API 스트리밍·도구 호출 및 서버 측 웹 검색 어댑터."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final

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
    from collections.abc import AsyncIterator, Mapping, Sequence

    from anthropic import AsyncAnthropic
    from anthropic.types.beta import (
        BetaContentBlock,
        BetaMessageParam,
        BetaServerToolUseBlock,
        BetaStopReason,
        BetaTextBlockParam,
        BetaToolResultBlockParam,
        BetaToolUnionParam,
        BetaToolUseBlock,
    )

    from finagent.analysis.providers.base import (
        ProviderEvent,
        ProviderId,
        ProviderStatus,
        RunRequest,
    )

KEY_ENV: Final = "ANTHROPIC_API_KEY"
MAX_TOKENS: Final = 64000
WEB_MAX_TOKENS: Final = 128000
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
EFFORT: Final = "high"
SEARCH_USES: Final = 6
FETCH_USES: Final = 4
WEB_TOOLS: Final = ("web_search", "web_fetch")
WEB_CALLERS: Final = ("direct", "code_execution_20260120")
WEB_FAILED: Final = {"web_search": "웹 검색 실패", "web_fetch": "원문 읽기 실패"}
WEB_ERRORS: Final = {
    "invalid_tool_input": "입력 형식 오류",
    "query_too_long": "검색어 길이 초과",
    "max_uses_exceeded": "단일 요청 사용 한도 도달",
    "too_many_requests": "과다 요청 제한",
    "unavailable": "일시 장애",
    "url_not_accessible": "페이지 접근 불가",
    "url_not_allowed": "비허용 주소",
    "url_not_in_prior_context": "이전 대화 미등장 주소",
    "unsupported_content_type": "미지원 문서 형식",
    "content_too_large": "문서 크기 초과",
}
WEB_SPENT: Final = (
    f"웹 검색·원문 열람 한도({WEB_STEP_LIMIT}회) 소진."
    " 더 검색하지 말고 확보한 자료로 답을 끝낼 것"
)


class ClaudeApiProvider:
    id: ProviderId = "claude_api"
    title: str = "Claude API 키"
    default_model: str = "claude-opus-5-5"

    def __init__(self, env: Mapping[str, str]) -> None:
        self._key: str = env_value(env, KEY_ENV)

    async def status(self) -> ProviderStatus:
        return key_status("anthropic", "claude", self._key, KEY_ENV)

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        import anthropic  # noqa: PLC0415  # 선택적 의존성(anthropic) 지연 임포트

        async with anthropic.AsyncAnthropic(api_key=self._key) as client:
            try:
                async for event in _rounds(client, request):
                    yield event
            except anthropic.NotFoundError as error:
                raise ProviderError(NO_MODEL.format(model=request.model)) from error


async def _rounds(
    client: AsyncAnthropic, request: RunRequest
) -> AsyncIterator[ProviderEvent]:
    from anthropic import omit  # noqa: PLC0415  # 선택적 의존성 지연 임포트

    conversation = request.conversation
    messages = _messages(request)
    tools = _tools(request)
    used = 0
    told = False
    calls: dict[str, BetaServerToolUseBlock] = {}
    container: str | None = None
    for _ in range(request.max_rounds + 2):
        async with client.beta.messages.stream(
            model=request.model,
            max_tokens=WEB_MAX_TOKENS if request.web else MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": conversation.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
            tools=tools or omit,
            container=container,
            cache_control={"type": "ephemeral"},
            output_config={"effort": EFFORT},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    yield TextDelta(event.text)
                elif event.type == "content_block_stop":
                    for found in _block_events(event.content_block, calls):
                        used += (
                            isinstance(found, WebStep)
                            and found.ok
                            and (found.kind == "web")
                        )
                        yield found
            final = await stream.get_final_message()
        usage = final.usage
        yield Usage(
            usage.input_tokens
            + (usage.cache_read_input_tokens or 0)
            + (usage.cache_creation_input_tokens or 0),
            usage.output_tokens,
        )
        if request.web and (sources := _web_sources(final.content)):
            yield WebFound(sources)
        _check(final.stop_reason)
        container = final.container.id if final.container else container
        messages.append({"role": "assistant", "content": final.content})
        if final.stop_reason == "pause_turn":
            continue
        uses = [block for block in final.content if block.type == "tool_use"]
        if not uses:
            return
        content: list[BetaToolResultBlockParam | BetaTextBlockParam] = [
            *await _tool_results(request, uses)
        ]
        if request.web and used >= WEB_STEP_LIMIT and not told:
            content.append({"type": "text", "text": WEB_SPENT})
            told = True
        messages.append({"role": "user", "content": content})
        yield ToolRound()
    raise ProviderError(TOO_MANY_ROUNDS)


def _check(reason: BetaStopReason | None) -> None:
    if reason == "refusal":
        msg = "모델 요청 거절"
        raise ProviderError(msg)
    if reason in {"max_tokens", "model_context_window_exceeded"}:
        raise ProviderError(TRUNCATED)


def _tools(request: RunRequest) -> list[BetaToolUnionParam]:
    tools: list[BetaToolUnionParam] = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.schema,
            "eager_input_streaming": True,
        }
        for spec in request.tools
    ]
    if request.web:
        tools.extend(
            (
                {
                    "type": "web_search_20260318",
                    "name": "web_search",
                    "max_uses": SEARCH_USES,
                    "allowed_callers": list(WEB_CALLERS),
                },
                {
                    "type": "web_fetch_20260318",
                    "name": "web_fetch",
                    "max_uses": FETCH_USES,
                    "allowed_callers": list(WEB_CALLERS),
                },
            )
        )
    return tools


async def _tool_results(
    request: RunRequest, uses: Sequence[BetaToolUseBlock]
) -> list[BetaToolResultBlockParam]:
    results = await asyncio.gather(
        *(request.run_tool(block.name, block.input) for block in uses)
    )
    return [
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result.text,
            "is_error": not result.ok,
        }
        for block, result in zip(uses, results, strict=True)
    ]


def _step(block: BetaServerToolUseBlock) -> str:
    if block.name == "web_fetch":
        return f"원문 읽기: {block.input.get('url', '')}"
    if block.name == "web_search":
        return f"웹 검색: {block.input.get('query', '')}"
    return "웹 결과 거르기"


def _block_events(
    block: BetaContentBlock, calls: dict[str, BetaServerToolUseBlock]
) -> tuple[ProviderEvent, ...]:
    step = _web_progress(block, calls)
    steps = () if step is None else (step,)
    return (ToolRound(), *steps) if block.type == "server_tool_use" else steps


def _web_progress(
    block: BetaContentBlock, calls: dict[str, BetaServerToolUseBlock]
) -> WebStep | None:
    if block.type == "server_tool_use":
        if block.name in WEB_TOOLS:
            calls[block.id] = block
            return None
        return WebStep(_step(block), kind="filter")
    if (result := _web_result(block)) is not None:
        call, why = result
        return _web_step(calls.pop(call), why)
    return None


def _web_step(call: BetaServerToolUseBlock, why: str | None) -> WebStep:
    if why is None:
        return WebStep(_step(call))
    return WebStep(f"{WEB_FAILED[call.name]}: {WEB_ERRORS.get(why, why)}", ok=False)


def _web_result(block: BetaContentBlock) -> tuple[str, str | None] | None:
    if block.type == "web_search_tool_result":
        found = block.content
        return block.tool_use_id, None if isinstance(found, list) else found.error_code
    if block.type == "web_fetch_tool_result":
        read = block.content
        failed = read.type == "web_fetch_tool_result_error"
        return block.tool_use_id, read.error_code if failed else None
    return None


def _web_sources(content: Sequence[BetaContentBlock]) -> tuple[WebSource, ...]:
    sources: list[WebSource] = []
    for block in content:
        if block.type == "web_search_tool_result":
            if isinstance(block.content, list):
                sources.extend(
                    WebSource(item.url, item.title) for item in block.content
                )
        elif block.type == "web_fetch_tool_result":
            if block.content.type != "web_fetch_tool_result_error":
                document = block.content.content
                sources.append(WebSource(block.content.url, document.title or ""))
        elif block.type == "text":
            sources.extend(
                WebSource(citation.url, citation.title or "")
                for citation in block.citations or []
                if citation.type == "web_search_result_location"
            )
    return tuple(sources)


def _messages(request: RunRequest) -> list[BetaMessageParam]:
    out: list[BetaMessageParam] = []
    for message in request.conversation.messages():
        if message.role == "user" and len(message.parts) > 1:
            *documents, ask = message.parts
            out.append(
                {
                    "role": "user",
                    "content": [
                        *(
                            {
                                "type": "text",
                                "text": document,
                                "cache_control": {"type": "ephemeral"},
                            }
                            for document in documents
                        ),
                        {"type": "text", "text": ask},
                    ],
                }
            )
        else:
            role = "user" if message.role == "user" else "assistant"
            out.append({"role": role, "content": message.text})
    return out
