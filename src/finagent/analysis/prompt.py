"""The system prompt and how a conversation is laid out for the model.

The same text goes to every provider. Long data comes first and the request
last, which the Anthropic, OpenAI and Google prompting guides all recommend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import Final

TEXT: Final = files("finagent.analysis") / "text"

FIRST_REQUEST: Final = "위 지표 문서를 바탕으로 다음 요청에 답해 줘."


def system_prompt(instructions: str = "") -> str:
    """How to cite (``text/system.txt``), then the user's ``instructions``."""
    cite = (TEXT / "system.txt").read_text(encoding="utf-8").strip()
    return f"{cite}\n\n{instructions.strip()}" if instructions.strip() else cite


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


@dataclass(frozen=True, slots=True)
class Message:
    """One provider-neutral message.

    ``parts`` are joined unless an adapter wants the first part (the
    document) as its own cacheable block.
    """

    role: str
    parts: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)


@dataclass(frozen=True, slots=True)
class Conversation:
    system: str
    document: str
    turns: tuple[Turn, ...] = field(default=())
    request: str = ""

    def messages(self) -> tuple[Message, ...]:
        """Earlier turns, then the current request, as user/assistant messages.

        Earlier tool results ride along with the answer they fed, so a later
        provider (possibly a different one) can still cite that data.
        """
        out: list[Message] = []
        requests = [*(turn.request for turn in self.turns), self.request]
        for index, request in enumerate(requests):
            wrapped = f"<request>\n{request}\n</request>"
            if index == 0:
                out.append(
                    Message("user", (self.document, f"{FIRST_REQUEST}\n{wrapped}"))
                )
            else:
                out.append(Message("user", (wrapped,)))
            if index < len(self.turns):
                out.append(
                    Message("assistant", (_answer_with_data(self.turns[index]),))
                )
        return tuple(out)


def _answer_with_data(turn: Turn) -> str:
    if not turn.tools:
        return turn.answer
    data = "\n".join(
        _tool_block(record.name, record.arguments, record.result)
        for record in turn.tools
        if record.ok
    )
    if not data:
        return turn.answer
    return f"<earlier_tool_data>\n{data}\n</earlier_tool_data>\n\n{turn.answer}"


def _tool_block(name: str, arguments: str, result: str) -> str:
    head = f'<tool_result name="{name}" arguments={_attr(arguments)}>'
    return f"{head}\n{result}\n</tool_result>"


def _attr(value: str) -> str:
    return '"' + value.replace("&", "&amp;").replace('"', "&quot;") + '"'
