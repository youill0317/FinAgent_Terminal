"""웹 검색·원문 열람 출처 URL 정규화 및 도메인·날짜 인용 레이블 포매터."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableMapping

CITE_PREFIX: Final = "web:"
SCHEME: Final = re.compile(r"^[a-z]+://", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class WebSource:
    url: str
    title: str = ""


def collect(
    found: MutableMapping[str, WebSource], sources: Iterable[WebSource]
) -> None:
    for source in sources:
        known = found.get(source.url)
        if known is None or not known.title:
            found[source.url] = source


def is_web(cite: str) -> bool:
    return cite.startswith(CITE_PREFIX)


def parse_cite(cite: str) -> tuple[str, str]:
    target, _, day = cite.removeprefix(CITE_PREFIX).replace("\\|", "|").partition("|")
    return target.strip(), day.strip()


def host(text: str) -> str:
    name = SCHEME.sub("", text.strip().lower())
    return name.partition("/")[0].removeprefix("www.")


def page(text: str) -> str:
    return SCHEME.sub("", text.strip()).removeprefix("www.").rstrip("/")


def resolve(cite: str, sources: Mapping[str, WebSource]) -> WebSource | None:
    wanted = page(parse_cite(cite)[0])
    if "/" in wanted:
        return next((s for s in sources.values() if page(s.url) == wanted), None)
    name = wanted.lower()
    if not name:
        return None
    return next(
        (
            source
            for source in sources.values()
            if name in {host(source.url), host(source.title)}
        ),
        None,
    )


def label(cite: str) -> str:
    target, day = parse_cite(cite)
    return f"{host(target)} · {day}" if day else host(target)
