from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from finagent_terminal.reports.models import (
    REPORT_DISCLAIMER,
    ReportSectionPrompt,
    ReportSectionResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

OPENAI_API_KEY: Final = "OPENAI_API_KEY"
ANTHROPIC_API_KEY: Final = "ANTHROPIC_API_KEY"
GEMINI_API_KEYS: Final = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
OPENROUTER_API_KEY: Final = "OPENROUTER_API_KEY"
HTTP_TIMEOUT: Final = httpx.Timeout(30.0, connect=5.0)
HTTP_LIMITS: Final = httpx.Limits(max_connections=20, max_keepalive_connections=5)


@unique
class ProviderName(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


class LLMSelection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    provider: ProviderName | None = None
    model: str | None = None
    api_key: str | None = None


class LLMProviderConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    api_key: str | None = None


class StructuredLLMProvider(Protocol):
    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult: ...


class LLMTransport(Protocol):
    def generate_text(self, prompt: ReportSectionPrompt) -> str: ...


class ProviderError(Exception):
    pass


class ProviderUnavailableError(ProviderError):
    provider: ProviderName | None
    message: str

    def __init__(self, *, provider: ProviderName | None, message: str) -> None:
        """Create a provider availability error."""
        self.provider = provider
        self.message = message
        if provider is None:
            super().__init__(message)
            return
        super().__init__(f"{provider.value} provider unavailable: {message}")


class ProviderCallError(ProviderError):
    provider: str
    message: str

    def __init__(self, *, provider: str, message: str) -> None:
        """Create a provider call error."""
        self.provider = provider
        self.message = message
        super().__init__(f"{provider} provider failed: {message}")


@dataclass(frozen=True, slots=True)
class FakeLLMClient:
    mode: str = "1"

    def generate_structured_report_section(
        self,
        prompt: ReportSectionPrompt,
    ) -> ReportSectionResult:
        if self.mode == "fail_section" and prompt.section.id == "bull_case":
            raise ProviderCallError(
                provider="fake",
                message=f"fake failure for {prompt.section.title}",
            )
        return ReportSectionResult(
            section_id=prompt.section.id,
            title=prompt.section.title,
            summary=(
                f"{prompt.context.symbol} {prompt.section.title} "
                "generated from supplied context."
            ),
            key_points=(prompt.context.market_summary,),
            risks=prompt.context.warnings
            or ("No additional supplied-context risks.",),
            disclaimer=REPORT_DISCLAIMER,
        )


def resolve_llm_config(
    selection: LLMSelection | None = None,
    env: Mapping[str, str] | None = None,
) -> LLMProviderConfig:
    source = os.environ if env is None else env
    chosen = selection if selection is not None else LLMSelection()
    provider = chosen.provider or _provider_from_env(source)
    model = chosen.model or source.get("FINAGENT_LLM_MODEL") or _default_model(provider)
    return LLMProviderConfig(
        provider=provider,
        model=model,
        api_key=chosen.api_key or _api_key_for(provider, source),
    )


def create_llm_client(
    selection: LLMSelection | None = None,
    env: Mapping[str, str] | None = None,
) -> StructuredLLMProvider:
    source = os.environ if env is None else env
    fake_mode = source.get("FINAGENT_FAKE_LLM")
    if fake_mode is not None:
        return FakeLLMClient(mode=fake_mode)

    config = resolve_llm_config(selection=selection, env=source)
    if config.api_key is None:
        raise ProviderUnavailableError(
            provider=config.provider,
            message="missing API key",
        )
    raise ProviderUnavailableError(
        provider=config.provider,
        message="provider factory must be called through finagent_terminal.llm",
    )


def parse_report_section_text(
    *,
    provider: str,
    prompt: ReportSectionPrompt,
    text: str,
) -> ReportSectionResult:
    try:
        result = TypeAdapter(ReportSectionResult).validate_json(text)
    except ValidationError as exc:
        raise ProviderCallError(
            provider=provider,
            message="response was not JSON or did not match report schema",
        ) from exc
    if result.section_id != prompt.section.id:
        raise ProviderCallError(
            provider=provider,
            message="response section id mismatch",
        )
    return result


def create_provider_http_client() -> httpx.Client:
    return httpx.Client(
        timeout=HTTP_TIMEOUT,
        limits=HTTP_LIMITS,
        follow_redirects=False,
    )


def _provider_from_env(env: Mapping[str, str]) -> ProviderName:
    raw = env.get("FINAGENT_LLM_PROVIDER")
    if raw is None:
        raise ProviderUnavailableError(
            provider=None,
            message="missing FINAGENT_LLM_PROVIDER",
        )
    try:
        return ProviderName(raw.lower())
    except ValueError as exc:
        raise ProviderUnavailableError(
            provider=None,
            message=f"unknown provider: {raw}",
        ) from exc


def _api_key_for(provider: ProviderName, env: Mapping[str, str]) -> str | None:
    key_names: Final = {
        ProviderName.OPENAI: (OPENAI_API_KEY,),
        ProviderName.ANTHROPIC: (ANTHROPIC_API_KEY,),
        ProviderName.GEMINI: GEMINI_API_KEYS,
        ProviderName.OPENROUTER: (OPENROUTER_API_KEY,),
    }
    for key in key_names[provider]:
        value = env.get(key)
        if value is not None:
            return value
    return None


def _default_model(provider: ProviderName) -> str:
    defaults: Final = {
        ProviderName.OPENAI: "gpt-4o-mini",
        ProviderName.ANTHROPIC: "claude-3-5-haiku-latest",
        ProviderName.GEMINI: "gemini-2.0-flash",
        ProviderName.OPENROUTER: "openai/gpt-4o-mini",
    }
    return defaults[provider]
