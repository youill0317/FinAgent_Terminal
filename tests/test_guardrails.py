"""Scope guardrails: public sources only, nothing persisted, no keys in source.

The macro dashboard keeps all of these. The S&P 500/400 screener
(``src/finagent/screen.py``) is a documented exception: it may import
yfinance, contact Wikipedia, SEC EDGAR (with the user's ``SEC_USER_AGENT``)
and Yahoo through yfinance, and write its data files, only under
``data/screen/``. The portfolio page
(``src/finagent/portfolio.py``) is the other: it may import yfinance and write
its trade log and price files, only under ``data/portfolio/``. The investor
view (``src/finagent/investors.py``) is the third: it may contact SEC EDGAR
and OpenFIGI, import yfinance (sectors) and write only under
``data/investors/``; its prices go to the portfolio page's store via app.py.
The dashboard's gold row is the fourth: ``src/finagent/sources.py`` may import
yfinance for Yahoo closes, and yfinance's own caches go only under
``data/macro/.yfinance/``; the closes themselves stay in memory. Generator
retraining (``src/finagent/colab_jobs.py``) is the fifth: it may run the
``colab`` CLI as a subprocess and write only under ``data/models/`` (its job
state, a staging folder and the swapped weights; ``tests/test_colab_jobs.py``).
torch, for the optional Kronos generator, is imported only inside
``generative.load_kronos``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from finagent import catalog, generative
from finagent.fetch import SOURCE_LANES

ROOT: Final = Path(__file__).resolve().parents[1]
SRC: Final = ROOT / "src" / "finagent"
SOURCES_MODULE: Final = SRC / "sources.py"
# The one module allowed to write a file: the saved view (panel ids only).
SETTINGS_MODULE: Final = SRC / "settings.py"
# The exception: yfinance data saved under data/screen/.
SCREEN_MODULE: Final = SRC / "screen.py"
# The other exception: the trade log and prices under data/portfolio/.
PORTFOLIO_MODULE: Final = SRC / "portfolio.py"
# The third: 13F filings, CUSIP mappings and sectors under data/investors/.
INVESTORS_MODULE: Final = SRC / "investors.py"
# The fifth: the colab CLI as a subprocess, files under data/models/.
COLAB_MODULE: Final = SRC / "colab_jobs.py"
SCREEN_HOSTS: Final = frozenset({"en.wikipedia.org", "data.sec.gov", "www.sec.gov"})
INVESTORS_HOSTS: Final = frozenset({"data.sec.gov", "www.sec.gov", "api.openfigi.com"})

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
    "textual",
    "finnhub",
    "fmp",
    "langgraph",
)
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
    allowed = {
        SETTINGS_MODULE,
        SCREEN_MODULE,
        PORTFOLIO_MODULE,
        INVESTORS_MODULE,
        COLAB_MODULE,
    }
    others = [path for path in _python_files() if path not in allowed]

    assert _hits(PERSISTENCE_PATTERN, others) == []


def test_only_the_training_jobs_run_subprocesses() -> None:
    imported = re.compile(r"^\s*(?:import|from)\s+subprocess\b")
    others = [path for path in _python_files() if path != COLAB_MODULE]

    assert Path("data", "models") == generative.MODELS_DIR
    assert _hits(imported, others) == []
    finagent = re.compile(r"^\s*(?:import|from)\s+finagent\b")
    lines = [hit.split(": ", 1)[1] for hit in _hits(finagent, [COLAB_MODULE])]
    assert lines == ["from finagent import generative"]  # never portfolio


def test_forbidden_packages_are_not_declared_or_imported() -> None:
    names = "|".join(re.escape(name) for name in FORBIDDEN_PACKAGES)
    declared = re.compile(rf"^\s*\"(?:{names})\b", re.IGNORECASE)
    imported = re.compile(rf"^\s*(?:import|from)\s+(?:{names})\b")

    assert _hits(declared, [ROOT / "pyproject.toml"]) == []
    assert _hits(imported, _python_files()) == []


def test_no_hardcoded_keys() -> None:
    for pattern in HARDCODED_KEY_PATTERNS:
        assert _hits(pattern, _python_files()) == []


def test_data_hosts_are_public_and_live_in_sources_module() -> None:
    hosts = set(URL_HOST_PATTERN.findall(SOURCES_MODULE.read_text("utf-8")))
    screen_hosts = set(URL_HOST_PATTERN.findall(SCREEN_MODULE.read_text("utf-8")))
    investor_hosts = set(URL_HOST_PATTERN.findall(INVESTORS_MODULE.read_text("utf-8")))
    allowed = {SOURCES_MODULE, SCREEN_MODULE, INVESTORS_MODULE}
    other_modules = [path for path in _python_files() if path not in allowed]

    links = {
        match.group(0)
        for path in other_modules
        for match in URL_PATTERN.finditer(path.read_text("utf-8"))
    }

    assert hosts == PUBLIC_HOSTS
    assert screen_hosts == SCREEN_HOSTS
    assert investor_hosts == INVESTORS_HOSTS
    assert links <= KEY_PAGES


def test_catalog_integrity() -> None:
    indicators = [ind for spec in catalog.PANELS for ind in spec.indicators]
    ids = [indicator.id for indicator in indicators]

    assert len(ids) == len(set(ids))
    assert set(catalog.TAPE) <= set(ids)
    assert set(ids) >= catalog.CORE
    assert {indicator.fetch.source for indicator in indicators} <= set(SOURCE_LANES)
