from __future__ import annotations

from typing import TYPE_CHECKING

from finagent_terminal.symbols import SymbolEntry, SymbolGroup, normalize_symbol

if TYPE_CHECKING:
    from finagent_terminal.config import FinAgentConfig


def add_group(config: FinAgentConfig, name: str, symbol: str) -> FinAgentConfig:
    group_names = {group.name for group in config.groups}
    if name in group_names:
        return add_symbol(config, name, symbol)
    group = SymbolGroup.model_validate({"name": name, "symbols": [symbol]})
    return config.model_copy(update={"groups": (*config.groups, group)})


def remove_group(config: FinAgentConfig, name: str) -> FinAgentConfig:
    return config.model_copy(
        update={
            "groups": tuple(group for group in config.groups if group.name != name),
        },
    )


def move_group(config: FinAgentConfig, name: str, direction: int) -> FinAgentConfig:
    groups = list(config.groups)
    index = next((i for i, group in enumerate(groups) if group.name == name), None)
    if index is None:
        return config
    target = max(0, min(len(groups) - 1, index + direction))
    if target == index:
        return config
    group = groups.pop(index)
    groups.insert(target, group)
    return config.model_copy(update={"groups": tuple(groups)})


def add_symbol(config: FinAgentConfig, group_name: str, symbol: str) -> FinAgentConfig:
    normalized_symbol = normalize_symbol(symbol)
    groups: list[SymbolGroup] = []
    changed = False
    for group in config.groups:
        if group.name != group_name:
            groups.append(group)
            continue
        symbols = tuple(entry.symbol for entry in group.symbols)
        groups.append(
            group
            if normalized_symbol in symbols
            else group.model_copy(
                update={
                    "symbols": (*group.symbols, SymbolEntry(symbol=normalized_symbol)),
                },
            ),
        )
        changed = True
    if not changed:
        groups.append(
            SymbolGroup.model_validate(
                {"name": group_name, "symbols": [normalized_symbol]},
            ),
        )
    return config.model_copy(update={"groups": tuple(groups)})


def remove_symbol(
    config: FinAgentConfig,
    group_name: str,
    symbol: str,
) -> FinAgentConfig:
    normalized_symbol = normalize_symbol(symbol)
    groups: list[SymbolGroup] = []
    for group in config.groups:
        if group.name != group_name:
            groups.append(group)
            continue
        symbols = tuple(
            entry for entry in group.symbols if entry.symbol != normalized_symbol
        )
        if symbols:
            groups.append(group.model_copy(update={"symbols": symbols}))
    return config.model_copy(update={"groups": tuple(groups)})


def move_symbol(
    config: FinAgentConfig,
    group_name: str,
    symbol: str,
    direction: int,
) -> FinAgentConfig:
    normalized_symbol = normalize_symbol(symbol)
    groups: list[SymbolGroup] = []
    for group in config.groups:
        if group.name != group_name:
            groups.append(group)
            continue
        entries = list(group.symbols)
        index = next(
            (i for i, entry in enumerate(entries) if entry.symbol == normalized_symbol),
            None,
        )
        if index is None:
            groups.append(group)
            continue
        target = max(0, min(len(entries) - 1, index + direction))
        entry = entries.pop(index)
        entries.insert(target, entry)
        groups.append(group.model_copy(update={"symbols": tuple(entries)}))
    return config.model_copy(update={"groups": tuple(groups)})
