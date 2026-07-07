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


class AnthropicContentBlock(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    text: str
    type: str


class AnthropicMessageResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    content: tuple[AnthropicContentBlock, ...] = Field(min_length=1)


def parse_anthropic_message_text(body: str) -> str:
    try:
        parsed = AnthropicMessageResponse.model_validate_json(body)
    except ValidationError as exc:
        raise ProviderCallError(
            provider=ProviderName.ANTHROPIC.value,
            message="malformed response envelope",
        ) from exc
    return parsed.content[0].text


@dataclass(frozen=True, slots=True)
class AnthropicProvider:
    config: LLMProviderConfig
    transport: LLMTransport | None = None

    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        transport = self.transport or AnthropicTransport(config=self.config)
        text = transport.generate_text(prompt)
        return parse_report_section_text(
            provider=ProviderName.ANTHROPIC.value,
            prompt=prompt,
            text=text,
        )


@dataclass(frozen=True, slots=True)
class AnthropicTransport:
    config: LLMProviderConfig

    def generate_text(self, prompt: ReportSectionPrompt) -> str:
        if self.config.api_key is None:
            raise ProviderUnavailableError(
                provider=ProviderName.ANTHROPIC,
                message="missing API key",
            )
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.config.model,
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt.render()}],
        }
        try:
            with create_provider_http_client() as client:
                response = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
                _ = response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderCallError(
                provider=ProviderName.ANTHROPIC.value,
                message=str(exc),
            ) from exc

        return parse_anthropic_message_text(response.text)
