"""Check the numbers in an answer against the values its ``[[id]]`` marks cite.

A mark at the end of a sentence cites that sentence. Every number in a cited
sentence is compared with the cited ids' values at the precision the sentence
uses. The result is a list of lines made of plain text, checked numbers and
citation marks, ready for the page to render.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from finagent.analysis.evidence import parse_number

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent.analysis.evidence import Evidence

Status = Literal["ok", "mismatch", "unsupported", "unknown_id"]

MARK: Final = re.compile(r"\s?\[\[([^\[\]\n]+)\]\]")
# A number not glued to letters, ids or other digits (DGS10, us_cpi, 2026Q2).
_DIGITS: Final = r"(?<![A-Za-z0-9_.,])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?"
_UNIT: Final = r"(?:\s?(?:%p|bp|%))?(?![A-Za-z0-9_])"
NUMBER: Final = re.compile(_DIGITS + _UNIT)
# Units that make a number a date, a count of periods or a tenor, not a value.
NOT_A_VALUE: Final = re.compile(r"\s?(?:년|월|일|분기|개월|주|시|분|차)")
DATE_LIKE: Final = re.compile(r"\d{4}[-./]\d{1,2}(?:[-./]\d{1,2})?|\d{4}\s?Q[1-4]")
LIST_MARK: Final = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+(?=\S)")
BASIS_POINTS: Final = 100.0


@dataclass(frozen=True, slots=True)
class Text:
    text: str


@dataclass(frozen=True, slots=True)
class Number:
    text: str
    status: Status
    actual: str | None = None


@dataclass(frozen=True, slots=True)
class Cite:
    ids: tuple[str, ...]
    known: tuple[bool, ...]


type Piece = Text | Number | Cite


@dataclass(slots=True)
class Report:
    lines: list[list[Piece]] = field(default_factory=list)
    counts: dict[Status, int] = field(
        default_factory=lambda: dict.fromkeys(
            ("ok", "mismatch", "unsupported", "unknown_id"), 0
        )
    )
    cited: set[str] = field(default_factory=set)

    def to_json(self) -> dict[str, object]:
        return {
            "lines": [[_piece_json(piece) for piece in line] for line in self.lines],
            "counts": dict(self.counts),
            "cited": sorted(self.cited),
        }


def verify(answer: str, evidence: Mapping[str, Evidence]) -> Report:
    report = Report()
    for line in answer.split("\n"):
        report.lines.append(_line(line, evidence, report))
    return report


def strip_marks(text: str) -> str:
    return MARK.sub("", text)


def _line(line: str, evidence: Mapping[str, Evidence], report: Report) -> list[Piece]:
    pieces: list[Piece] = []
    position = 0
    table_row = line.lstrip().startswith("|")
    for match in MARK.finditer(line):
        ids = tuple(part.strip() for part in match.group(1).split(",") if part.strip())
        segment = line[position : match.start()]
        if table_row:
            head, cited = "", segment
        else:
            head, cited = _last_sentence(segment)
        pieces.extend(_numbers(head, (), evidence, report))
        pieces.extend(_numbers(cited, ids, evidence, report))
        known = tuple(cite in evidence for cite in ids)
        pieces.append(Cite(ids, known))
        report.cited.update(cite for cite, ok in zip(ids, known, strict=True) if ok)
        position = match.end()
    pieces.extend(_numbers(line[position:], (), evidence, report))
    return _join_text(pieces)


def _last_sentence(segment: str) -> tuple[str, str]:
    parts = SENTENCE_END.split(segment)
    if len(parts) == 1:
        return "", segment
    cut = len(segment) - len(parts[-1])
    return segment[:cut], segment[cut:]


def _numbers(
    text: str,
    ids: tuple[str, ...],
    evidence: Mapping[str, Evidence],
    report: Report,
) -> list[Piece]:
    pieces: list[Piece] = []
    skip = _skipped_spans(text)
    position = 0
    for match in NUMBER.finditer(text):
        start, end = match.span()
        if any(a <= start < b for a, b in skip):
            continue
        if NOT_A_VALUE.match(text, end):
            continue
        pieces.append(Text(text[position:start]))
        number = _check(match.group(0), ids, evidence)
        report.counts[number.status] += 1
        pieces.append(number)
        position = end
    pieces.append(Text(text[position:]))
    return pieces


def _skipped_spans(text: str) -> list[tuple[int, int]]:
    spans = [match.span() for match in DATE_LIKE.finditer(text)]
    list_mark = LIST_MARK.match(text)
    if list_mark is not None:
        spans.append(list_mark.span())
    return spans


def _check(raw: str, ids: tuple[str, ...], evidence: Mapping[str, Evidence]) -> Number:
    if not ids:
        return Number(raw, "unsupported")
    known = [evidence[cite] for cite in ids if cite in evidence]
    if not known:
        return Number(raw, "unknown_id")
    core = raw.strip()
    scale = 1.0
    if core.endswith("bp"):
        scale = 1.0 / BASIS_POINTS
    digits = re.sub(r"\s?(?:%p|bp|%)$", "", core)
    claimed = parse_number(digits) * scale
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    tolerance = 0.5 * 10.0**-decimals * scale + 1e-9
    candidates = [value for item in known for value in item.candidates()]
    best = min(
        candidates, key=lambda value: abs(abs(value) - abs(claimed)), default=None
    )
    if best is None:
        return Number(raw, "mismatch")
    if abs(abs(best) - abs(claimed)) <= tolerance:
        return Number(raw, "ok")
    shown = best / scale
    return Number(raw, "mismatch", f"{shown:,.{decimals}f}".replace("-", "−"))


def _join_text(pieces: list[Piece]) -> list[Piece]:
    out: list[Piece] = []
    for piece in pieces:
        if isinstance(piece, Text):
            if not piece.text:
                continue
            last = out[-1] if out else None
            if isinstance(last, Text):
                out[-1] = Text(last.text + piece.text)
                continue
        out.append(piece)
    return out


def _piece_json(piece: Piece) -> dict[str, object]:
    if isinstance(piece, Text):
        return {"t": piece.text}
    if isinstance(piece, Number):
        return {"n": piece.text, "s": piece.status, "a": piece.actual}
    return {"c": list(piece.ids), "k": list(piece.known)}
