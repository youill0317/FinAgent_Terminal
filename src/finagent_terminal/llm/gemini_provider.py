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


class GeminiPart(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    text: str


class GeminiContent(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    parts: tuple[GeminiPart, ...] = Field(min_length=1)


class GeminiCandidate(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    content: GeminiContent


class GeminiResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    candidates: tuple[GeminiCandidate, ...] = Field(min_length=1)


def parse_gemini_response_text(body: str) -> str:
    try:
        parsed = GeminiResponse.model_validate_json(body)
    except ValidationError as exc:
        raise ProviderCallError(
            provider=ProviderName.GEMINI.value,
            message="malformed response envelope",
        ) from exc
    return parsed.candidates[0].content.parts[0].text


@dataclass(frozen=True, slots=True)
class GeminiProvider:
    config: LLMProviderConfig
    transport: LLMTransport | None = None

    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        transport = self.transport or GeminiTransport(config=self.config)
        text = transport.generate_text(prompt)
        return parse_report_section_text(
            provider=ProviderName.GEMINI.value,
            prompt=prompt,
            text=text,
        )


@dataclass(frozen=True, slots=True)
class GeminiTransport:
    config: LLMProviderConfig

    def generate_text(self, prompt: ReportSectionPrompt) -> str:
        if self.config.api_key is None:
            raise ProviderUnavailableError(
                provider=ProviderName.GEMINI,
                message="missing API key",
            )
        payload = {
            "contents": [{"parts": [{"text": prompt.render()}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent"
        )
        try:
            with create_provider_http_client() as client:
                response = client.post(
                    url,
                    params={"key": self.config.api_key},
                    json=payload,
                )
                _ = response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderCallError(
                provider=ProviderName.GEMINI.value,
                message=str(exc),
            ) from exc

        return parse_gemini_response_text(response.text)
