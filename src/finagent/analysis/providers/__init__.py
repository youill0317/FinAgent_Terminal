"""AI 분석 공급자(Claude·OpenAI·Gemini·데모) 레지스트리."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from finagent.analysis.providers.claude_api import ClaudeApiProvider
from finagent.analysis.providers.demo import DemoProvider
from finagent.analysis.providers.gemini_api import GeminiApiProvider
from finagent.analysis.providers.openai_api import OpenAiApiProvider

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent.analysis.providers.base import Provider

DEFAULT_PROVIDER: Final = "claude_api"


def registry(env: Mapping[str, str], *, demo: bool) -> dict[str, Provider]:
    if demo:
        return {"demo": DemoProvider()}
    providers: tuple[Provider, ...] = (
        ClaudeApiProvider(env),
        OpenAiApiProvider(env),
        GeminiApiProvider(env),
    )
    return {provider.id: provider for provider in providers}
