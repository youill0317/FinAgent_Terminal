"""The five AI providers (two subscriptions, three API keys) and the demo one."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from finagent.analysis.providers.claude_api import ClaudeApiProvider
from finagent.analysis.providers.claude_sub import ClaudeSubscriptionProvider
from finagent.analysis.providers.codex_sub import CodexSubscriptionProvider
from finagent.analysis.providers.demo import DemoProvider
from finagent.analysis.providers.gemini_api import GeminiApiProvider
from finagent.analysis.providers.openai_api import OpenAiApiProvider

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent.analysis.providers.base import Provider

DEFAULT_PROVIDER: Final = "claude_sub"


def registry(env: Mapping[str, str], *, demo: bool) -> dict[str, Provider]:
    """Providers in display order. ``--demo`` gets only the scripted one."""
    if demo:
        return {"demo": DemoProvider()}
    providers: tuple[Provider, ...] = (
        ClaudeSubscriptionProvider(env),
        ClaudeApiProvider(env),
        CodexSubscriptionProvider(env),
        OpenAiApiProvider(env),
        GeminiApiProvider(env),
    )
    return {provider.id: provider for provider in providers}
