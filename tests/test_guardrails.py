from __future__ import annotations

import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
SRC: Final = ROOT / "src"
DEPENDENCY_METADATA: Final = (ROOT / "pyproject.toml",)

PERSISTENCE_PATTERNS: Final = (
    r"\bsqlite\b",
    r"\bduckdb\b",
    r"\bparquet\b",
    r"\bsqlalchemy\b",
    r"\bpostgres\b",
    r"\bmysql\b",
)
FORBIDDEN_DEPENDENCY_PATTERNS: Final = (
    r"\blanggraph\b",
    r"\btradingagents\b",
)
FORBIDDEN_RUNTIME_PATTERNS: Final = (
    r"\balpha_vantage\b",
    r"\bpolygon\b",
    r"\bbroker api\b",
    r"\bbroker login\b",
    r"\bbroker account\b",
    r"\bbrokerage\b",
    r"\border placement\b",
    r"\border proposal\b",
    r"\bplace orders?\b",
    r"\bportfolio execution\b",
    r"\bsimulated exchange\b",
    r"\blive trading\b",
    r"\bhft\b",
    r"\bbacktesting\b",
    r"\bbackground sync\b",
)
VENDOR_KEY_NAME_PATTERN: Final = (
    r"\b(?:fred|finnhub|fmp)[a-z0-9_]*(?:key|token|secret)?\s*=\s*"
)
VENDOR_KEY_VALUE_PATTERN: Final = r"['\"][A-Za-z0-9+/=?:&._-]{20,}['\"]"
HARDCODED_VENDOR_KEY_PATTERNS: Final = (
    f"{VENDOR_KEY_NAME_PATTERN}{VENDOR_KEY_VALUE_PATTERN}",
    r"\b(?:api_key|apikey|token)=[A-Za-z0-9+/=?:&._-]{20,}",
)
INTENDED_SOURCE_NAME_PATTERN: Final = re.compile(
    r'"(?:FRED|SEC EDGAR|Finnhub|FMP)"',
    re.IGNORECASE,
)
SOURCE_NAME_ALLOWLIST: Final = {
    "src/finagent_terminal/data/finnhub_client.py",
    "src/finagent_terminal/data/fmp_client.py",
    "src/finagent_terminal/data/fred_sources.py",
    "src/finagent_terminal/data/sec_sources.py",
    "src/finagent_terminal/data/source_routing.py",
    "src/finagent_terminal/data/sources.py",
    "src/finagent_terminal/reports/source_context.py",
    "src/finagent_terminal/ui/screens/news.py",
}
REFERENCE_PROJECT_PATTERN: Final = re.compile(
    r"FinceptTerminal|Fincept Terminal|TradingAgents|Tauric",
    re.IGNORECASE,
)


def test_no_forbidden_runtime_scope_terms_in_source() -> None:
    # Given: source files are the executable app surface.
    pattern = _compile((*PERSISTENCE_PATTERNS, *FORBIDDEN_RUNTIME_PATTERNS))

    # When: scope guardrails are scanned.
    hits = _scan(SRC, pattern)

    # Then: no forbidden runtime integrations or storage surfaces are present.
    assert hits == ()


def test_forbidden_dependencies_are_not_declared_or_imported() -> None:
    pattern = _compile(FORBIDDEN_DEPENDENCY_PATTERNS)

    hits = (*_scan_files(DEPENDENCY_METADATA, pattern), *_scan(SRC, pattern))

    assert hits == ()


def test_source_display_names_are_limited_to_source_modules() -> None:
    hits = _scan(SRC, INTENDED_SOURCE_NAME_PATTERN)

    assert {path for path, _line in hits} <= SOURCE_NAME_ALLOWLIST


def test_no_hardcoded_vendor_keys_or_key_bearing_urls() -> None:
    pattern = _compile(HARDCODED_VENDOR_KEY_PATTERNS)

    hits = _scan(SRC, pattern)

    assert hits == ()


def test_reference_project_names_are_not_used_in_product_or_docs() -> None:
    hits = (
        *_scan(SRC, REFERENCE_PROJECT_PATTERN),
        *_scan_files((ROOT / "README.md",), REFERENCE_PROJECT_PATTERN),
    )

    assert hits == ()


def test_guardrails_reject_injected_bad_examples() -> None:
    assert _matches(FORBIDDEN_DEPENDENCY_PATTERNS, "import langgraph")
    assert _matches(FORBIDDEN_DEPENDENCY_PATTERNS, "tradingagents = '*'")
    assert _matches(FORBIDDEN_RUNTIME_PATTERNS, "broker login endpoint")
    assert _matches(FORBIDDEN_RUNTIME_PATTERNS, "live trading order placement")
    assert _matches(PERSISTENCE_PATTERNS, "duckdb source cache")
    assert _matches(PERSISTENCE_PATTERNS, "parquet raw vendor payloads")
    assert _matches(
        HARDCODED_VENDOR_KEY_PATTERNS,
        'finnhub_key = "finnhub+/=?:&token-123456"',
    )
    assert _matches(
        HARDCODED_VENDOR_KEY_PATTERNS,
        "https://api.stlouisfed.org/fred?api_key=fred-secret-token-123456",
    )


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


def _scan_files(
    paths: tuple[Path, ...],
    pattern: re.Pattern[str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(path.relative_to(ROOT)), line.strip())
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if pattern.search(line)
    )


def _compile(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return _compile(patterns).search(text) is not None
