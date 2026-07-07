from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from finagent_terminal.llm.base import (
    LLMProviderConfig,
    LLMTransport,
    ProviderCallError,
    ProviderName,
    ProviderUnavailableError,
    create_provider_http_client,
    parse_report_section_text,
)
from finagent_terminal.llm.openai_provider import parse_chat_completion_text

if TYPE_CHECKING:
    from finagent_terminal.reports.models import (
        ReportSectionPrompt,
        ReportSectionResult,
    )


@dataclass(frozen=True, slots=True)
class OpenRouterProvider:
    config: LLMProviderConfig
    transport: LLMTransport | None = None

    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        transport = self.transport or OpenRouterTransport(config=self.config)
        text = transport.generate_text(prompt)
        return parse_report_section_text(
            provider=ProviderName.OPENROUTER.value,
            prompt=prompt,
            text=text,
        )


@dataclass(frozen=True, slots=True)
class OpenRouterTransport:
    config: LLMProviderConfig

    def generate_text(self, prompt: ReportSectionPrompt) -> str:
        if self.config.api_key is None:
            raise ProviderUnavailableError(
                provider=ProviderName.OPENROUTER,
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
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                _ = response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderCallError(
                provider=ProviderName.OPENROUTER.value,
                message=str(exc),
            ) from exc

        return parse_chat_completion_text(ProviderName.OPENROUTER, response.text)
