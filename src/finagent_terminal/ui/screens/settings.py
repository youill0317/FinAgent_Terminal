from __future__ import annotations

import os
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, override

from finagent_terminal.data.sources import source_availability_lines
from finagent_terminal.llm.base import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEYS,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    ProviderName,
)
from finagent_terminal.ui.screens import WorkspaceScreen

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent_terminal.ui.state import WorkspaceId

PROVIDER_KEYS: Final[Mapping[ProviderName, tuple[str, ...]]] = MappingProxyType(
    {
        ProviderName.OPENAI: (OPENAI_API_KEY,),
        ProviderName.ANTHROPIC: (ANTHROPIC_API_KEY,),
        ProviderName.GEMINI: GEMINI_API_KEYS,
        ProviderName.OPENROUTER: (OPENROUTER_API_KEY,),
    },
)


class SettingsScreen(WorkspaceScreen):
    workspace: ClassVar[WorkspaceId] = "settings"
    label: ClassVar[str] = "Settings"
    summary: ClassVar[str] = "Local settings are available."

    @override
    def lines(self) -> tuple[str, ...]:
        config = self.state.config
        return (
            self.summary,
            f"Config: {config.path}",
            f"Status: {config.status}",
            "Markets/Ticker/Tools: available without LLM API keys",
            *provider_availability_lines(),
            *data_source_availability_lines(),
        )


def provider_availability_lines(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    source = os.environ if env is None else env
    return (
        "Providers:",
        *(
            f"{provider.value}: {_availability_status(source, PROVIDER_KEYS[provider])}"
            for provider in ProviderName
        ),
    )


def _availability_status(env: Mapping[str, str], keys: tuple[str, ...]) -> str:
    if any(env.get(key) for key in keys):
        return "available"
    return f"unavailable (missing {', '.join(keys)})"


def data_source_availability_lines(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    return source_availability_lines(env)
