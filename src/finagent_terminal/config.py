from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

from finagent_terminal.symbols import DEFAULT_PRESETS, SymbolGroup

CONFIG_FILE_NAME: Final = "config.json"
APP_HOME_ENV: Final = "FINAGENT_HOME"
MARKET_DATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adjclose",
        "chart",
        "close",
        "fundamental",
        "fundamentals",
        "high",
        "history",
        "low",
        "market_data",
        "ohlcv",
        "open",
        "price",
        "prices",
        "quote",
        "quotes",
        "timestamp",
        "timestamps",
        "volume",
    },
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class ConfigMarketDataError(ValueError):
    field: str
    path: str


class ProviderSettings(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    provider: str = "openai"
    model: str | None = None


class UISettings(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    default_group: str = "Major Indices"
    theme: str = "system"


class FinAgentConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    groups: tuple[SymbolGroup, ...] = DEFAULT_PRESETS
    provider_settings: ProviderSettings = ProviderSettings()
    ui_settings: UISettings = UISettings()


class UserConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    groups: tuple[SymbolGroup, ...] = ()
    provider_settings: ProviderSettings | None = None
    ui_settings: UISettings | None = None


def app_home() -> Path:
    configured_home = os.environ.get(APP_HOME_ENV)
    if configured_home:
        return Path(configured_home).expanduser()
    return Path.home() / ".finagent_terminal"


def config_path() -> Path:
    return app_home() / CONFIG_FILE_NAME


def load_config(path: Path | None = None) -> FinAgentConfig:
    resolved_path = config_path() if path is None else path
    if not resolved_path.exists():
        return FinAgentConfig()

    raw_config = JSON_VALUE_ADAPTER.validate_json(
        resolved_path.read_text(encoding="utf-8"),
    )
    reject_market_data_fields(raw_config)
    user_config = UserConfig.model_validate(raw_config)
    return merge_config(FinAgentConfig(), user_config)


def save_config(config: FinAgentConfig, path: Path | None = None) -> Path:
    resolved_path = config_path() if path is None else path
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    _ = resolved_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return resolved_path


def merge_config(
    default_config: FinAgentConfig,
    user_config: UserConfig,
) -> FinAgentConfig:
    groups_by_name = {group.name: group for group in default_config.groups}
    for group in user_config.groups:
        groups_by_name[group.name] = group

    return FinAgentConfig(
        groups=tuple(groups_by_name.values()),
        provider_settings=(
            user_config.provider_settings or default_config.provider_settings
        ),
        ui_settings=user_config.ui_settings or default_config.ui_settings,
    )


def upsert_group(config: FinAgentConfig, group: SymbolGroup) -> FinAgentConfig:
    groups_by_name = {existing.name: existing for existing in config.groups}
    groups_by_name[group.name] = group
    return config.model_copy(update={"groups": tuple(groups_by_name.values())})


def reject_market_data_fields(value: JsonValue, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = key.strip().lower()
            if normalized_key in MARKET_DATA_FIELDS:
                raise ConfigMarketDataError(key, path)
            reject_market_data_fields(nested, f"{path}.{key}")
        return

    if isinstance(value, list):
        for index, nested in enumerate(value):
            reject_market_data_fields(nested, f"{path}[{index}]")
