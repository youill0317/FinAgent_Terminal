from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from finagent_terminal.llm import (
    AnthropicProvider,
    GeminiProvider,
    LLMSelection,
    OpenAIProvider,
    OpenRouterProvider,
    ProviderCallError,
    ProviderName,
    ProviderUnavailableError,
    create_llm_client,
    resolve_llm_config,
)
from finagent_terminal.llm.anthropic_provider import parse_anthropic_message_text
from finagent_terminal.llm.gemini_provider import parse_gemini_response_text
from finagent_terminal.llm.openai_provider import parse_chat_completion_text
from finagent_terminal.reports.models import (
    DepthMode,
    ReportContext,
    ReportSectionPrompt,
    ReportSectionResult,
    section_specs_for_depth,
)

if TYPE_CHECKING:
    from finagent_terminal.llm.base import StructuredLLMProvider


@dataclass(frozen=True, slots=True)
class JsonTransport:
    def generate_text(self, prompt: ReportSectionPrompt) -> str:
        return json.dumps(
            {
                "section_id": prompt.section.id,
                "title": prompt.section.title,
                "summary": f"{prompt.context.symbol} {prompt.section.title}",
                "key_points": ["point one"],
                "risks": ["risk one"],
                "disclaimer": "Research assistance only; not investment advice.",
            },
        )


def test_provider_selection_uses_env_when_config_is_absent() -> None:
    env = {
        "FINAGENT_LLM_PROVIDER": "anthropic",
        "FINAGENT_LLM_MODEL": "claude-test",
        "ANTHROPIC_API_KEY": "test-key",
    }

    config = resolve_llm_config(env=env)

    assert config.provider is ProviderName.ANTHROPIC
    assert config.model == "claude-test"
    assert config.api_key == "test-key"


def test_provider_selection_prefers_explicit_config_over_env() -> None:
    env = {
        "FINAGENT_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "env-key",
    }

    config = resolve_llm_config(
        selection=LLMSelection(
            provider=ProviderName.OPENROUTER,
            model="openrouter-test",
            api_key="config-key",
        ),
        env=env,
    )

    assert config.provider is ProviderName.OPENROUTER
    assert config.model == "openrouter-test"
    assert config.api_key == "config-key"


def test_fake_llm_env_returns_deterministic_client() -> None:
    env = {"FINAGENT_FAKE_LLM": "1"}
    client = create_llm_client(env=env)
    context = ReportContext(
        symbol="AAPL",
        market_summary="price is up",
        indicator_summary="rsi neutral",
    )
    prompt = ReportSectionPrompt(
        context=context,
        depth=DepthMode.QUICK,
        section=section_specs_for_depth(DepthMode.QUICK)[0],
    )

    result = client.generate_structured_report_section(prompt)

    assert result.section_id == prompt.section.id
    assert result.title == prompt.section.title
    assert "AAPL" in result.summary


def test_provider_adapter_parses_structured_json_from_transport() -> None:
    providers: tuple[StructuredLLMProvider, ...] = (
        OpenAIProvider(
            config=resolve_llm_config(
                selection=LLMSelection(
                    provider=ProviderName.OPENAI,
                    model="test-model",
                    api_key="test-key",
                ),
            ),
            transport=JsonTransport(),
        ),
        AnthropicProvider(
            config=resolve_llm_config(
                selection=LLMSelection(
                    provider=ProviderName.ANTHROPIC,
                    model="test-model",
                    api_key="test-key",
                ),
            ),
            transport=JsonTransport(),
        ),
        GeminiProvider(
            config=resolve_llm_config(
                selection=LLMSelection(
                    provider=ProviderName.GEMINI,
                    model="test-model",
                    api_key="test-key",
                ),
            ),
            transport=JsonTransport(),
        ),
        OpenRouterProvider(
            config=resolve_llm_config(
                selection=LLMSelection(
                    provider=ProviderName.OPENROUTER,
                    model="test-model",
                    api_key="test-key",
                ),
            ),
            transport=JsonTransport(),
        ),
    )
    prompt = ReportSectionPrompt(
        context=ReportContext(
            symbol="MSFT",
            market_summary="flat",
            indicator_summary="",
        ),
        depth=DepthMode.QUICK,
        section=section_specs_for_depth(DepthMode.QUICK)[0],
    )

    results = tuple(
        provider.generate_structured_report_section(prompt) for provider in providers
    )

    assert all(isinstance(result, ReportSectionResult) for result in results)
    assert {result.section_id for result in results} == {prompt.section.id}
    assert {result.key_points for result in results} == {("point one",)}


def test_missing_credentials_fail_before_real_provider_call() -> None:
    with pytest.raises(ProviderUnavailableError):
        _ = create_llm_client(
            selection=LLMSelection(provider=ProviderName.GEMINI),
            env={},
        )


@pytest.mark.parametrize(
    ("provider", "body"),
    [
        (ProviderName.OPENAI, "{}"),
        (ProviderName.OPENROUTER, '{"choices": []}'),
    ],
)
def test_chat_provider_malformed_200_envelope_raises_provider_error(
    provider: ProviderName,
    body: str,
) -> None:
    with pytest.raises(ProviderCallError) as exc_info:
        _ = parse_chat_completion_text(provider, body)

    assert exc_info.value.provider == provider.value
    assert exc_info.value.message == "malformed response envelope"


def test_anthropic_malformed_200_envelope_raises_provider_error() -> None:
    with pytest.raises(ProviderCallError) as exc_info:
        _ = parse_anthropic_message_text('{"content": []}')

    assert exc_info.value.provider == ProviderName.ANTHROPIC.value
    assert exc_info.value.message == "malformed response envelope"


def test_gemini_malformed_200_envelope_raises_provider_error() -> None:
    with pytest.raises(ProviderCallError) as exc_info:
        _ = parse_gemini_response_text('{"candidates": []}')

    assert exc_info.value.provider == ProviderName.GEMINI.value
    assert exc_info.value.message == "malformed response envelope"
