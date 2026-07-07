from __future__ import annotations

import os
from typing import TYPE_CHECKING

from finagent_terminal.llm.anthropic_provider import AnthropicProvider
from finagent_terminal.llm.base import (
    FakeLLMClient,
    LLMProviderConfig,
    LLMSelection,
    LLMTransport,
    ProviderCallError,
    ProviderName,
    ProviderUnavailableError,
    StructuredLLMProvider,
    resolve_llm_config,
)
from finagent_terminal.llm.gemini_provider import GeminiProvider
from finagent_terminal.llm.openai_provider import OpenAIProvider
from finagent_terminal.llm.openrouter_provider import OpenRouterProvider

if TYPE_CHECKING:
    from collections.abc import Mapping


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

    provider_classes = {
        ProviderName.OPENAI: OpenAIProvider,
        ProviderName.ANTHROPIC: AnthropicProvider,
        ProviderName.GEMINI: GeminiProvider,
        ProviderName.OPENROUTER: OpenRouterProvider,
    }
    return provider_classes[config.provider](config=config)


__all__ = (
    "AnthropicProvider",
    "FakeLLMClient",
    "GeminiProvider",
    "LLMProviderConfig",
    "LLMSelection",
    "LLMTransport",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderCallError",
    "ProviderName",
    "ProviderUnavailableError",
    "StructuredLLMProvider",
    "create_llm_client",
    "resolve_llm_config",
)
