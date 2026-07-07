from __future__ import annotations

import pytest
from pydantic import ValidationError

from finagent_terminal.symbols import (
    DEFAULT_PRESETS,
    SymbolEntry,
    SymbolGroup,
    normalize_symbol,
)


def test_default_presets_include_expected_market_groups() -> None:
    # Given: the bundled symbol catalog.
    presets = {
        group.name: tuple(entry.symbol for entry in group.symbols)
        for group in DEFAULT_PRESETS
    }

    # When: defaults are inspected for first-run navigation.
    group_names = tuple(presets)

    # Then: the MVP groups exist, and Major Indices matches the plan exactly.
    assert group_names == (
        "Major Indices",
        "Mega Cap Tech",
        "Semiconductors",
        "Banks",
        "Energy",
        "Crypto ETFs",
    )
    assert presets["Major Indices"] == ("^GSPC", "^IXIC", "^DJI", "SPY", "QQQ", "DIA")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" spy ", "SPY"), ("brk-b", "BRK-B"), ("^gspc", "^GSPC")],
)
def test_normalize_symbol_accepts_yahoo_symbols(
    raw: str,
    expected: str,
) -> None:
    # Given: a Yahoo-compatible user-entered symbol.
    # When: it is normalized.
    symbol = normalize_symbol(raw)

    # Then: the stored form is stable.
    assert symbol == expected


@pytest.mark.parametrize("raw", ["", "AAPL!", "A" * 33, "AAPL USD"])
def test_symbol_entry_rejects_invalid_symbols(raw: str) -> None:
    # Given: an invalid user-entered symbol.
    # When / Then: parsing rejects it at the boundary.
    with pytest.raises(ValidationError):
        _ = SymbolEntry(symbol=raw)


def test_symbol_group_accepts_string_entries_for_editable_json() -> None:
    # Given: a human-editable group payload using simple strings.
    # When: it is parsed.
    group = SymbolGroup.model_validate(
        {
            "name": "Custom",
            "symbols": ["aapl", {"symbol": "msft", "label": "Microsoft"}],
        },
    )

    # Then: symbols are normalized and optional labels are preserved.
    assert tuple(entry.symbol for entry in group.symbols) == ("AAPL", "MSFT")
    assert group.symbols[1].label == "Microsoft"


def test_symbol_group_accepts_tuple_entries() -> None:
    group = SymbolGroup.model_validate({"name": "Custom", "symbols": ("AAPL", "MSFT")})

    assert tuple(entry.symbol for entry in group.symbols) == ("AAPL", "MSFT")


def test_symbol_group_rejects_blank_group_name() -> None:
    with pytest.raises(ValidationError):
        _ = SymbolGroup.model_validate({"name": "   ", "symbols": ["AAPL"]})


@pytest.mark.parametrize("symbols", ["AAPL", b"AAPL"])
def test_symbol_group_rejects_scalar_symbols(symbols: str | bytes) -> None:
    with pytest.raises(ValidationError):
        _ = SymbolGroup.model_validate({"name": "Custom", "symbols": symbols})


@pytest.mark.parametrize("symbols", [[], ()])
def test_symbol_group_rejects_empty_symbol_collections(
    symbols: list[str] | tuple[()],
) -> None:
    with pytest.raises(ValidationError):
        _ = SymbolGroup.model_validate({"name": "Custom", "symbols": symbols})
