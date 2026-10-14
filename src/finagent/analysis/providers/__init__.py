"""The three API-key AI providers and the demo one."""

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
    """Providers in display order. ``--demo`` gets only the scripted one."""
    if demo:
        return {"demo": DemoProvider()}
    providers: tuple[Provider, ...] = (
        ClaudeApiProvider(env),
        OpenAiApiProvider(env),
        GeminiApiProvider(env),
    )
    return {provider.id: provider for provider in providers}
