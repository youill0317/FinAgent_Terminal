from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from finagent_terminal.config import (
    ConfigMarketDataError,
    JsonValue,
    config_path,
    load_config,
    save_config,
    upsert_group,
)
from finagent_terminal.symbols import SymbolGroup

if TYPE_CHECKING:
    from pathlib import Path


def test_load_config_uses_finagent_home_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a clean FINAGENT_HOME with no user config file.
    monkeypatch.setenv("FINAGENT_HOME", str(tmp_path))

    # When: configuration is loaded.
    config = load_config()

    # Then: bundled defaults are returned without requiring a file.
    assert config_path() == tmp_path / "config.json"
    assert config.groups[0].name == "Major Indices"
    assert tuple(entry.symbol for entry in config.groups[0].symbols) == (
        "^GSPC",
        "^IXIC",
        "^DJI",
        "SPY",
        "QQQ",
        "DIA",
    )


def test_user_config_merges_group_and_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a user override file that edits one default group and settings.
    monkeypatch.setenv("FINAGENT_HOME", str(tmp_path))
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "name": "Major Indices",
                        "symbols": ["SPY"],
                        "label": "My Index View",
                    },
                    {
                        "name": "Custom",
                        "symbols": [{"symbol": "nvda", "label": "Nvidia"}],
                    },
                ],
                "provider_settings": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                },
                "ui_settings": {"default_group": "Custom", "theme": "dark"},
            },
        ),
        encoding="utf-8",
    )

    # When: configuration is loaded.
    config = load_config()

    # Then: user groups override by name, custom groups append, and settings merge.
    groups = {group.name: group for group in config.groups}
    assert tuple(entry.symbol for entry in groups["Major Indices"].symbols) == ("SPY",)
    assert groups["Major Indices"].label == "My Index View"
    assert groups["Custom"].symbols[0].symbol == "NVDA"
    assert config.provider_settings.provider == "openrouter"
    assert config.ui_settings.default_group == "Custom"


def test_save_config_persists_custom_group_without_market_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a config with a custom editable group.
    monkeypatch.setenv("FINAGENT_HOME", str(tmp_path))
    config = upsert_group(
        load_config(),
        SymbolGroup.model_validate({"name": "Watchlist", "symbols": ["AAPL", "MSFT"]}),
    )

    # When: it is saved and loaded again.
    saved_path = save_config(config)
    reloaded = load_config()

    # Then: symbols persist, and the JSON contains only config fields.
    saved_json = saved_path.read_text(encoding="utf-8")
    assert reloaded.groups[-1].name == "Watchlist"
    assert tuple(entry.symbol for entry in reloaded.groups[-1].symbols) == (
        "AAPL",
        "MSFT",
    )
    assert "prices" not in saved_json.lower()
    assert "ohlcv" not in saved_json.lower()


@pytest.mark.parametrize(
    "payload",
    [
        {"prices": {"AAPL": 123.45}},
        {"groups": [{"name": "Bad", "symbols": ["AAPL"], "history": [1, 2, 3]}]},
        {
            "groups": [
                {
                    "name": "Bad",
                    "symbols": [{"symbol": "AAPL", "ohlcv": [[1, 2, 3, 4, 5]]}],
                },
            ],
        },
        {"provider_settings": {"provider": "openai", "fundamentals": {"pe": 20}}},
    ],
)
def test_rejects_market_data_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, JsonValue],
) -> None:
    # Given: a user config attempting to store fetched market data.
    monkeypatch.setenv("FINAGENT_HOME", str(tmp_path))
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload), encoding="utf-8")

    # When / Then: loading rejects the file before data can become app config.
    with pytest.raises(ConfigMarketDataError):
        _ = load_config()
