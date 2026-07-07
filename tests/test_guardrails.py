from __future__ import annotations

import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
SRC: Final = ROOT / "src"

FORBIDDEN_SCOPE_PATTERNS: Final = (
    r"\bsqlite\b",
    r"\bduckdb\b",
    r"\bparquet\b",
    r"\bsqlalchemy\b",
    r"\bpostgres\b",
    r"\bmysql\b",
    r"\blanggraph\b",
    r"\bwebsocket\b",
    r"\bfred\b",
    r"\bsec edgar\b",
    r"\balpha_vantage\b",
    r"\bpolygon\b",
    r"\bbroker login\b",
    r"\border placement\b",
    r"\border proposal\b",
    r"\bportfolio execution\b",
    r"\bsimulated exchange\b",
    r"\blive trading\b",
    r"\bhft\b",
    r"\bbacktesting\b",
    r"\bbackground sync\b",
)
REFERENCE_PROJECT_PATTERN: Final = re.compile(
    r"FinceptTerminal|Fincept Terminal|TradingAgents|Tauric",
    re.IGNORECASE,
)


def test_no_forbidden_runtime_scope_terms_in_source() -> None:
    # Given: source files are the executable app surface.
    pattern = re.compile("|".join(FORBIDDEN_SCOPE_PATTERNS), re.IGNORECASE)

    # When: scope guardrails are scanned.
    hits = _scan(SRC, pattern)

    # Then: no forbidden runtime integrations or storage surfaces are present.
    assert hits == ()


def test_reference_project_names_are_not_used_in_source() -> None:
    # Given: reference projects are attribution-only documentation material.
    # When: executable source is scanned.
    hits = _scan(SRC, REFERENCE_PROJECT_PATTERN)

    # Then: no reference-project names, branding, or screen text are embedded.
    assert hits == ()


def test_investment_advice_phrase_is_limited_to_disclaimers() -> None:
    # Given: report output may use the required non-advice disclaimer.
    pattern = re.compile(r"investment advice", re.IGNORECASE)

    # When: source files are scanned for advice wording.
    hits = _scan(SRC, pattern)

    # Then: every hit is in the report model disclaimer/validator path.
    hit_paths = {path for path, _line in hits}
    hit_text = "\n".join(line for _path, line in hits)
    assert hit_paths == {"src/finagent_terminal/reports/models.py"}
    assert (
        'REPORT_DISCLAIMER: Final = "Research assistance only; '
        'not investment advice."'
    ) in hit_text
    assert '"only and not investment advice."' in hit_text
    assert 'has_advice_warning = "not investment advice" in normalized' in hit_text
    assert '"not investment advice"' in hit_text


def _scan(root: Path, pattern: re.Pattern[str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(path.relative_to(ROOT)), line.strip())
        for path in sorted(root.rglob("*.py"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if pattern.search(line)
    )
