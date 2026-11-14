from __future__ import annotations

import asyncio
import copy
import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, cast

import pytest

from finagent.analysis.prompt import Conversation, Turn
from finagent.analysis.providers import claude_api
from finagent.analysis.providers.base import (
    RunRequest,
    ToolRound,
    ToolSpec,
    WebStep,
)
from finagent.analysis.tools import ToolResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from finagent.analysis.providers.base import ProviderEvent

type Request = dict[str, object]

SEARCHES = 10


@dataclass(frozen=True, slots=True)
class Block:
    type: str
    name: str = ""
    id: str = ""
    text: str = ""
    input: dict[str, object] = field(default_factory=dict)
    citations: None = None
    tool_use_id: str = ""
    content: object = None


@dataclass(frozen=True, slots=True)
class Failure:
    error_code: str


@dataclass(frozen=True, slots=True)
class Step:
    type: str
    text: str = ""
    content_block: Block | None = None


@dataclass(frozen=True, slots=True)
class Tokens:
    input_tokens: int = 1
    output_tokens: int = 1
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Reply:
    content: list[Block]
    stop_reason: str
    usage: Tokens = field(default_factory=Tokens)
    container: None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Step]:
        for block in self.content:
            if block.type == "text":
                yield Step("text", block.text)
            yield Step("content_block_stop", content_block=block)

    async def get_final_message(self) -> Self:
        return self


class Client:
    def __init__(self, script: list[Reply]) -> None:
        self.script: list[Reply] = script
        self.calls: list[Request] = []
        self.beta: Client = self
        self.messages: Client = self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def stream(self, **request: object) -> Reply:
        self.calls.append(copy.deepcopy(request))
        return self.script.pop(0)


def marks(found: object) -> int:
    if isinstance(found, dict):
        entries = cast("dict[str, object]", found)
        return ("cache_control" in entries) + sum(map(marks, entries.values()))
    if isinstance(found, list):
        return sum(map(marks, cast("list[object]", found)))
    return 0


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None, reason="anthropic 패키지 없음"
)
def test_a_run_keeps_one_tools_list_four_cache_marks_and_one_web_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searches = [
        block
        for n in range(SEARCHES + 1)
        for block in (
            Block("server_tool_use", "web_search", f"s{n}"),
            Block(
                "web_search_tool_result",
                tool_use_id=f"s{n}",
                content=Failure("max_uses_exceeded") if n == SEARCHES else [],
            ),
        )
    ]
    call = Block("tool_use", "get_series", "c1", input={"id": "us_unrate"})
    client = Client(
        [
            Reply([*searches, call], "tool_use"),
            Reply([*searches, call], "tool_use"),
            Reply([Block("text", text="확인"), call], "tool_use"),
            Reply([Block("text", text="끝")], "end_turn"),
        ]
    )

    def opened(api_key: str) -> Client:
        del api_key
        return client

    async def run_tool(name: str, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        return ToolResult("{}", ok=True, summary=name)

    monkeypatch.setattr("anthropic.AsyncAnthropic", opened)

    request = RunRequest(
        Conversation("지시", ("공통", "문서"), (Turn("앞 질문", "앞 답"),), "질문"),
        "claude-opus-5-5",
        (ToolSpec("get_series", "시계열", {"type": "object"}),),
        run_tool,
        40,
        web=True,
    )

    async def run() -> list[ProviderEvent]:
        provider = claude_api.ClaudeApiProvider({claude_api.KEY_ENV: "key"})
        return [event async for event in provider.stream(request)]

    events = asyncio.run(run())
    steps = [event for event in events if isinstance(event, WebStep)]

    assert len(client.calls) == len(("두 번 검색", "한도 넘김", "확인", "끝"))
    assert all(sent["tools"] == client.calls[0]["tools"] for sent in client.calls)
    assert max(marks(sent) for sent in client.calls) == len(
        ("system", "공통", "문서", "자동")
    )
    assert str(client.calls[-1]["messages"]).count(claude_api.WEB_SPENT) == 1
    assert [step.ok for step in steps].count(False) == len(("첫 응답", "둘째 응답"))
    assert [step.ok for step in steps].count(True) == 2 * SEARCHES
    rounds = sum(isinstance(event, ToolRound) for event in events)
    assert rounds == 2 * (SEARCHES + 1) + len(("검색", "한도", "확인"))
