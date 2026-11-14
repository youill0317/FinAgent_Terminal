"""AI 답변 내 인라인 인용 표기([[id]], [[web:...]]) 파싱 및 출처 링크 매핑."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from finagent.analysis import web

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from finagent.analysis.web import WebSource

MARK: Final = re.compile(r"\s?\[\[([^\[\]\n]+)\]\]")


def parse(
    answer: str, ids: Collection[str], sources: Mapping[str, WebSource]
) -> list[list[dict[str, object]]]:
    return [_line(line, ids, sources) for line in answer.split("\n")]


def _line(
    line: str, ids: Collection[str], sources: Mapping[str, WebSource]
) -> list[dict[str, object]]:
    pieces: list[dict[str, object]] = []
    position = 0
    for match in MARK.finditer(line):
        if match.start() > position:
            pieces.append({"t": line[position : match.start()]})
        cited = _cited(match.group(1))
        found = [web.resolve(c, sources) if web.is_web(c) else None for c in cited]
        pieces.append(
            {
                "c": list(cited),
                "k": [
                    hit is not None or cite in ids
                    for cite, hit in zip(cited, found, strict=True)
                ],
                "u": [hit.url if hit is not None else "" for hit in found],
            }
        )
        position = match.end()
    if position < len(line):
        pieces.append({"t": line[position:]})
    return pieces


def cited_sources(
    answer: str, sources: Mapping[str, WebSource]
) -> tuple[WebSource, ...]:
    found: dict[str, WebSource] = {}
    for match in MARK.finditer(answer):
        if web.is_web(match.group(1)):
            source = web.resolve(match.group(1), sources)
            if source is not None:
                found[source.url] = source
    return tuple(found.values())


def _cited(body: str) -> tuple[str, ...]:
    if web.is_web(body.strip()):
        return (body.strip(),)
    return tuple(part.strip() for part in body.split(",") if part.strip())
