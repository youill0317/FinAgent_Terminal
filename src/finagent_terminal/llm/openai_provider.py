from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finagent_terminal.llm.base import (
    LLMProviderConfig,
    LLMTransport,
    ProviderCallError,
    ProviderName,
    ProviderUnavailableError,
    create_provider_http_client,
    parse_report_section_text,
)

if TYPE_CHECKING:
    from finagent_terminal.reports.models import (
        ReportSectionPrompt,
        ReportSectionResult,
    )


class ChatMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    content: str


class ChatChoice(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    choices: tuple[ChatChoice, ...] = Field(min_length=1)


def parse_chat_completion_text(provider: ProviderName, body: str) -> str:
    try:
        parsed = ChatCompletionResponse.model_validate_json(body)
    except ValidationError as exc:
        raise ProviderCallError(
            provider=provider.value,
            message="malformed response envelope",
        ) from exc
    return parsed.choices[0].message.content


@dataclass(frozen=True, slots=True)
class OpenAIProvider:
    config: LLMProviderConfig
    transport: LLMTransport | None = None

    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        transport = self.transport or OpenAITransport(config=self.config)
        text = transport.generate_text(prompt)
        return parse_report_section_text(
            provider=ProviderName.OPENAI.value,
            prompt=prompt,
            text=text,
        )


@dataclass(frozen=True, slots=True)
class OpenAITransport:
    config: LLMProviderConfig

    def generate_text(self, prompt: ReportSectionPrompt) -> str:
        if self.config.api_key is None:
            raise ProviderUnavailableError(
                provider=ProviderName.OPENAI,
                message="missing API key",
            )
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt.render()}],
            "response_format": {"type": "json_object"},
        }
        try:
            with create_provider_http_client() as client:
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                _ = response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderCallError(
                provider=ProviderName.OPENAI.value,
                message=str(exc),
            ) from exc

        return parse_chat_completion_text(ProviderName.OPENAI, response.text)
