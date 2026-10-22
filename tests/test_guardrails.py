"""Scope guardrails: public sources only, nothing persisted, no keys in source."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from finagent import catalog
from finagent.fetch import SOURCE_LANES

ROOT: Final = Path(__file__).resolve().parents[1]
SRC: Final = ROOT / "src" / "finagent"
SOURCES_MODULE: Final = SRC / "sources.py"
# The one module allowed to write a file: the saved view (panel ids only).
SETTINGS_MODULE: Final = SRC / "settings.py"

PERSISTENCE_TERMS: Final = (
    r"\bsqlite\b",
    r"\bduckdb\b",
    r"\bparquet\b",
    r"\bsqlalchemy\b",
    r"\bpostgres\b",
    r"\bmysql\b",
    r"\bpickle\b",
    r"\bshelve\b",
    r"write_text\(",
    r"write_bytes\(",
    r"open\([^)]*[\"']w",
)
PERSISTENCE_PATTERN: Final = re.compile("|".join(PERSISTENCE_TERMS))
FORBIDDEN_PACKAGES: Final = (
    "yfinance",
    "textual",
    "finnhub",
    "fmp",
    "langgraph",
)
PROVIDERS: Final = SRC / "analysis" / "providers"
# Each optional AI package is imported by its own adapter and nowhere else.
PROVIDER_PACKAGES: Final = {
    "anthropic": PROVIDERS / "claude_api.py",
    "openai": PROVIDERS / "openai_api.py",
    "google": PROVIDERS / "gemini_api.py",
}
AI_PACKAGES: Final = ("anthropic", "openai", "google-genai")
KEY_NAME: Final = r"\b(?:fred|ecos)[a-z0-9_]*(?:key|token|secret)?\s*=\s*"
KEY_VALUE: Final = r"['\"][A-Za-z0-9+/=?:&._-]{20,}['\"]"
HARDCODED_KEY_PATTERNS: Final = (
    re.compile(KEY_NAME + KEY_VALUE, re.IGNORECASE),
    re.compile(r"\b(?:api_key|apikey|token)=[A-Za-z0-9+/=?:&._-]{20,}"),
    re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
)
PUBLIC_HOSTS: Final = frozenset(
    {
        "api.stlouisfed.org",
        "api.fiscaldata.treasury.gov",
        "markets.newyorkfed.org",
        "ecos.bok.or.kr",
        "data-api.ecb.europa.eu",
        "stats.bis.org",
        "sdmx.oecd.org",
    }
)
URL_HOST_PATTERN: Final = re.compile(r"https://([A-Za-z0-9.-]+)")
# Free-key sign-up pages the page links to; the app never fetches from them.
KEY_PAGES: Final = frozenset(
    {
        "https://fredaccount.stlouisfed.org/apikeys",
        "https://ecos.bok.or.kr/api/#/AuthKeyApply",
    }
)
URL_PATTERN: Final = re.compile(r"https://[^\s\"')]+")


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _hits(pattern: re.Pattern[str], paths: list[Path]) -> list[str]:
    return [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in paths
        for number, line in enumerate(path.read_text("utf-8").splitlines(), 1)
        if pattern.search(line)
    ]


def test_source_never_persists_fetched_data() -> None:
    others = [path for path in _python_files() if path != SETTINGS_MODULE]

    assert _hits(PERSISTENCE_PATTERN, others) == []


def test_settings_module_never_sees_fetched_data() -> None:
    imports = re.compile(r"^\s*(?:import|from)\s+finagent\.(?!catalog\b)")

    assert _hits(imports, [SETTINGS_MODULE]) == []


def test_forbidden_packages_are_not_declared_or_imported() -> None:
    names = "|".join(re.escape(name) for name in FORBIDDEN_PACKAGES)
    declared = re.compile(rf"^\s*\"(?:{names})\b", re.IGNORECASE)
    imported = re.compile(rf"^\s*(?:import|from)\s+(?:{names})\b")

    assert _hits(declared, [ROOT / "pyproject.toml"]) == []
    assert _hits(imported, _python_files()) == []


def test_ai_packages_stay_in_their_adapters() -> None:
    for package, home in PROVIDER_PACKAGES.items():
        imported = re.compile(rf"^\s*(?:import|from)\s+{re.escape(package)}\b")
        others = [path for path in _python_files() if path != home]

        assert _hits(imported, others) == [], package
        assert _hits(imported, [home]) != [], package


def test_ai_packages_are_optional_extras() -> None:
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    runtime = text.split("[project.optional-dependencies]")[0]
    runtime = runtime.split("dependencies = [", 1)[1].split("]", 1)[0]

    for package in AI_PACKAGES:
        assert package not in runtime
        assert f'"{package}>=' in text


def test_no_hardcoded_keys() -> None:
    for pattern in HARDCODED_KEY_PATTERNS:
        assert _hits(pattern, _python_files()) == []


def test_data_hosts_are_public_and_live_in_sources_module() -> None:
    hosts = set(URL_HOST_PATTERN.findall(SOURCES_MODULE.read_text("utf-8")))
    other_modules = [path for path in _python_files() if path != SOURCES_MODULE]

    links = {
        match.group(0)
        for path in other_modules
        for match in URL_PATTERN.finditer(path.read_text("utf-8"))
    }

    assert hosts == PUBLIC_HOSTS
    assert links <= KEY_PAGES


def test_catalog_integrity() -> None:
    indicators = [ind for spec in catalog.PANELS for ind in spec.indicators]
    ids = [indicator.id for indicator in indicators]

    assert len(ids) == len(set(ids))
    assert set(catalog.TAPE) <= set(ids)
    assert set(ids) >= catalog.CORE
    assert {indicator.fetch.source for indicator in indicators} <= set(SOURCE_LANES)
