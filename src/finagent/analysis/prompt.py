"""AI 분석 시스템 프롬프트 조립 및 대화 턴·도구 실행 이력 직렬화."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Collection

TEXT: Final = files("finagent.analysis") / "text"

DATA_MAP_HEAD: Final = "값이 필요할 때 먼저 쓸 곳:"
FIRST_REQUEST: Final = "위 문서를 바탕으로 다음 요청에 답할 것:"
REPLAY_LIMIT: Final = 4000
REPLAY_CUT: Final = "…(후속 내용 생략 · 필요 시 도구 재호출)"
REPORT_REQUEST: Final = (
    "스레드 보고·요청 수신. 검토 후 필요한 일을 하고 사용자에게 알릴 것"
)


def text(name: str) -> str:
    return (TEXT / name).read_text(encoding="utf-8").strip()


def system_prompt(  # noqa: PLR0913  # 조정자·스레드·팀 에이전트·챗봇별 지침 조합 옵션
    instructions: str = "",
    *,
    web_limit: int = 0,
    coordinator: bool = False,
    asking: bool = False,
    team: bool = False,
    chat: bool = False,
    tools: Collection[str] | None = None,
) -> str:
    parts = [text("system.txt")]
    if chat:
        parts.append(text("system_chat.txt"))
    elif tools is None or tools:
        parts.append(data_map(tools))
    if coordinator:
        parts.append(text("system_main.txt"))
    if asking:
        parts.append(text("system_thread.txt"))
    if team:
        parts.append(text("system_agent.txt"))
    if web_limit:
        parts.append(text("system_web.txt").format(limit=web_limit))
    if instructions.strip():
        parts.append(instructions.strip())
    return "\n\n".join(parts)


def data_map(tools: Collection[str] | None = None) -> str:
    rows = (line.split("\t", 1) for line in text("data_map.txt").splitlines())
    lines = [
        f"- {need}: {tool}"
        for need, tool in rows
        if tools is None or any(name in tool for name in tools)
    ]
    return "\n".join((DATA_MAP_HEAD, *lines))


@dataclass(frozen=True, slots=True)
class ToolRecord:
    name: str
    arguments: str
    result: str
    ok: bool


@dataclass(frozen=True, slots=True)
class Turn:
    request: str
    answer: str
    tools: tuple[ToolRecord, ...] = ()
    document: str = ""


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    parts: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)


@dataclass(frozen=True, slots=True)
class Conversation:
    system: str
    documents: tuple[str, ...]
    turns: tuple[Turn, ...] = field(default=())
    request: str = ""
    request_document: str = ""

    def messages(self) -> tuple[Message, ...]:
        out: list[Message] = []
        requests = [
            *((turn.request, turn.document) for turn in self.turns),
            (self.request, self.request_document),
        ]
        for index, (request, document) in enumerate(requests):
            wrapped = f"<request>\n{request}\n</request>" if request else REPORT_REQUEST
            body = f"{document}\n\n{wrapped}" if document else wrapped
            if index == 0 and self.documents:
                first = f"{FIRST_REQUEST}\n{body}"
                out.append(Message("user", (*self.documents, first)))
            else:
                out.append(Message("user", (body,)))
            if index < len(self.turns):
                out.append(
                    Message("assistant", (_answer_with_data(self.turns[index]),))
                )
        return tuple(out)


def _answer_with_data(turn: Turn) -> str:
    if not turn.tools:
        return turn.answer
    data = "\n".join(
        _tool_block(record.name, record.arguments, _head(record.result))
        for record in turn.tools
        if record.ok
    )
    if not data:
        return turn.answer
    return f"<earlier_tool_data>\n{data}\n</earlier_tool_data>\n\n{turn.answer}"


def _head(result: str) -> str:
    if len(result) <= REPLAY_LIMIT:
        return result
    return f"{result[:REPLAY_LIMIT]}{REPLAY_CUT}"


def _tool_block(name: str, arguments: str, result: str) -> str:
    head = f'<tool_result name="{name}" arguments={_attr(arguments)}>'
    return f"{head}\n{result}\n</tool_result>"


def _attr(value: str) -> str:
    return '"' + value.replace("&", "&amp;").replace('"', "&quot;") + '"'
