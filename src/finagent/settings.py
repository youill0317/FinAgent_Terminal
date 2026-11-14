"""터미널 환경설정(자동 갱신·테마·AI 분석 공급자) 로드 및 영속화."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast

from finagent import layout
from finagent.etf_universe import Product

if TYPE_CHECKING:
    from pathlib import Path

ETF_PERIOD: Final = "1개월"


@dataclass(frozen=True, slots=True)
class AnalysisChoice:
    provider: str
    models: dict[str, str] = field(default_factory=dict)
    instructions: str = ""
    thread_models: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EtfChoice:
    added: tuple[Product, ...] = ()
    removed: tuple[str, ...] = ()
    group: str | None = None
    category: str | None = None
    period: str = ETF_PERIOD


def load_analysis(path: Path) -> AnalysisChoice | None:
    raw = _read(path).get("analysis")
    if not isinstance(raw, dict):
        return None
    section = cast("dict[str, object]", raw)
    provider = section.get("provider")
    if not isinstance(provider, str):
        return None
    instructions = section.get("instructions")
    if not isinstance(instructions, str):
        instructions = ""
    return AnalysisChoice(
        provider,
        _models(section.get("models")),
        instructions,
        _models(section.get("thread_models")),
    )


def _models(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in cast("dict[object, object]", raw).items()
        if isinstance(key, str) and isinstance(value, str)
    }


def load_etf(path: Path) -> EtfChoice:
    raw = _read(path).get("etf")
    if not isinstance(raw, dict):
        return EtfChoice()
    section = cast("dict[str, object]", raw)
    added: list[Product] = []
    items = section.get("added")
    for item in cast("list[object]", items) if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        row = cast("dict[str, object]", item)
        ticker, group, category = (
            row.get("ticker"),
            row.get("group"),
            row.get("category"),
        )
        multiple = row.get("multiple", 1)
        if (
            isinstance(ticker, str)
            and isinstance(group, str)
            and isinstance(category, str)
            and isinstance(multiple, (int, float))
        ):
            added.append(Product(ticker.upper(), group, category, float(multiple)))
    removed = section.get("removed")
    group = section.get("group")
    category = section.get("category")
    period = section.get("period")
    return EtfChoice(
        tuple(added),
        tuple(v.upper() for v in cast("list[object]", removed) if isinstance(v, str))
        if isinstance(removed, list)
        else (),
        group if isinstance(group, str) else None,
        category if isinstance(category, str) else None,
        period if isinstance(period, str) else ETF_PERIOD,
    )


def save_etf(path: Path, choice: EtfChoice) -> None:
    body = _kept({}, path, "analysis")
    section: dict[str, object] = {}
    if choice.added:
        section["added"] = [
            {
                "ticker": p.ticker,
                "group": p.group,
                "category": p.category,
                "multiple": p.multiple,
            }
            for p in choice.added
        ]
    if choice.removed:
        section["removed"] = list(choice.removed)
    if choice.group is not None:
        section["group"] = choice.group
    if choice.category is not None:
        section["category"] = choice.category
    if choice.period != ETF_PERIOD:
        section["period"] = choice.period
    if section:
        body["etf"] = section
    _write(path, body)


def save_analysis(path: Path, choice: AnalysisChoice) -> None:
    body: dict[str, object] = {
        "analysis": {
            "provider": choice.provider,
            "models": dict(choice.models),
            "thread_models": dict(choice.thread_models),
            "instructions": choice.instructions,
        },
    }
    _write(path, _kept(body, path, "etf"))


def _kept(body: dict[str, object], path: Path, *keys: str) -> dict[str, object]:
    saved = _read(path)
    for key in keys:
        section = saved.get(key)
        if isinstance(section, dict):
            body[key] = section
    return body


class SettingsError(ValueError):
    pass


def check(path: Path) -> None:
    _ = _read(path)


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"설정 파일 읽기 실패: {path} ({error})"
        raise SettingsError(msg) from error
    if not isinstance(raw, dict):
        msg = f"설정 파일 JSON 객체 형식 오류: {path}"
        raise SettingsError(msg)
    return cast("dict[str, object]", raw)


def _write(path: Path, body: dict[str, object]) -> None:
    layout.write_text(path, json.dumps(body, ensure_ascii=False, indent=2) + "\n")
