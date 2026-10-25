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
import sys
import threading
import types
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from finagent import catalog, generative, investors, portfolio, screen
from finagent.fetch import SOURCE_LANES
from finagent.sources import YAHOO_CACHE, YahooCloses

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pytest

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
GENERATIVE_MODULE: Final = SRC / "generative.py"
# Pure numpy/pandas math the portfolio and investor modules may import; quant.py
# also fits GJR-GARCH with the optional arch package.
MATH_MODULES: Final = (SRC / "quant.py", SRC / "perf.py")
_OTHER_MODULE: Final = r"^\s*(?:import|from)\s+finagent\b(?!\.(?:quant|perf)\b)"
_OTHER_NAME: Final = r"^\s*from\s+finagent\s+import\s+(?!(?:quant|perf)\b)"
IMPORTS_BEYOND_MATH: Final = re.compile(f"{_OTHER_MODULE}|{_OTHER_NAME}")
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


def test_torch_is_an_optional_extra_imported_only_by_kronos() -> None:
    imported = re.compile(r"^\s*(?:import|from)\s+torch\b|import_module\(\"torch\"\)")
    text = GENERATIVE_MODULE.read_text("utf-8")
    kronos = text.split("def load_kronos(", 1)[1].split("\ndef ", 1)[0]
    pyproject = (ROOT / "pyproject.toml").read_text("utf-8")
    runtime = pyproject.split("[project.optional-dependencies]")[0]
    extras = pyproject.split("[project.optional-dependencies]")[1].split("\n[", 1)[0]

    assert [h.split(": ", 1)[1] for h in _hits(imported, _python_files())] == [
        'torch = importlib.import_module("torch")'
    ]
    assert 'importlib.import_module("torch")' in kronos
    assert "torch" not in runtime
    assert '"torch>=' in extras.split("kronos = [", 1)[1].split("]", 1)[0]


class _OneStock:
    """A screener source without the network: one stock, flat prices."""

    def constituents(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"ticker": ["AAA"], "name": ["A"], "sector": ["X"], "industry": ["Y"]}
        )

    def prices(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        span = (end - start).days
        days = [pd.Timestamp(start + timedelta(days=i)) for i in range(span)]
        bar = dict.fromkeys(("open", "high", "low", "close", "adj_close"), 10.0)
        return pd.DataFrame({"date": days, "ticker": tickers[0], **bar, "volume": 1.0})

    def fundamentals(self, ticker: str, *, financial: bool) -> Mapping[str, object]:
        return {"ticker_seen": ticker, "financial": financial}

    def estimates(self, ticker: str) -> Mapping[str, object]:
        return {"ticker_seen": ticker}


def _quiet(*_: object) -> None:
    pass


def test_screener_writes_only_under_its_folder(tmp_path: Path) -> None:
    root = tmp_path / screen.SNAPSHOT_DIR
    now = datetime(2026, 1, 2, 23, tzinfo=UTC)

    _ = screen.update(_OneStock(), root, now, _quiet, sleep=_quiet)

    assert Path("data", "screen") == screen.SNAPSHOT_DIR
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written
    assert all(path.is_relative_to(root) for path in written)


def test_screener_is_isolated_from_the_dashboard() -> None:
    imports_screen = re.compile(
        r"^\s*(?:import|from)\s+finagent(?:\.screen\b| import .*\bscreen\b)"
    )
    imports_finagent = re.compile(r"^\s*(?:import|from)\s+finagent\b")
    others = [path for path in _python_files() if path != SRC / "app.py"]

    assert _hits(imports_screen, others) == []
    assert _hits(imports_finagent, [SCREEN_MODULE]) == []


class _FlatPrices:
    """A portfolio price source without the network: flat prices."""

    def history(self, tickers: Sequence[str], start: date) -> portfolio.Prices:
        days = pd.DatetimeIndex([start])
        frame = pd.DataFrame(dict.fromkeys(tickers, 10.0), index=days)
        return portfolio.Prices(frame, frame)


def test_portfolio_writes_only_under_its_folder(tmp_path: Path) -> None:
    root = tmp_path / portfolio.DATA_DIR
    book, pid = portfolio.add_portfolio(portfolio.Book(), "연금", "KRW")
    trade = portfolio.Trade(
        id="t1", day=date(2026, 1, 2), ticker="AAA", side="buy", quantity=1, price=10
    )
    portfolio.save_book(root, portfolio.add_trade(book, pid, trade))
    fetcher = portfolio.Fetcher(root, source=lambda _: _FlatPrices())
    assert fetcher.start(["AAA", portfolio.FX])
    done = threading.Event()
    for _ in range(100):
        if not fetcher.progress.running:
            break
        _ = done.wait(0.05)

    assert Path("data", "portfolio") == portfolio.DATA_DIR
    assert not fetcher.progress.failed, fetcher.progress.message
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert len(written) == 3  # the log and two price files
    assert all(path.is_relative_to(root) for path in written)


def test_portfolio_is_isolated_from_the_dashboard() -> None:
    imports_portfolio = re.compile(
        r"^\s*(?:import|from)\s+finagent(?:\.portfolio\b| import .*\bportfolio\b)"
    )
    others = [path for path in _python_files() if path != SRC / "app.py"]

    assert _hits(imports_portfolio, others) == []
    assert _hits(IMPORTS_BEYOND_MATH, [PORTFOLIO_MODULE]) == []


class _OneFiler:
    """An investor source without the network: one filing, one holding."""

    def sec(self, url: str) -> str:
        if url.endswith(".json") and "submissions" in url:
            return (
                '{"name": "A", "filings": {"recent": {"accessionNumber": ["1-26-1"],'
                ' "form": ["13F-HR"], "filingDate": ["2026-02-14"],'
                ' "reportDate": ["2025-12-31"]}, "files": []}}'
            )
        if url.endswith("index.json"):
            return '{"directory": {"item": [{"name": "t.xml", "size": "1"}]}}'
        return (
            "<informationTable><infoTable><nameOfIssuer>A</nameOfIssuer>"
            "<cusip>AAA000001</cusip><value>10</value><shrsOrPrnAmt>"
            "<sshPrnamt>1</sshPrnamt><sshPrnamtType>SH</sshPrnamtType>"
            "</shrsOrPrnAmt></infoTable></informationTable>"
        )

    def figi(self, jobs: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
        return [{"data": [{"ticker": "AAA", "marketSector": "Equity"}]} for _ in jobs]

    def sector(self, ticker: str) -> str | None:
        return f"sector of {ticker}"


def test_investors_write_only_under_their_folder(tmp_path: Path) -> None:
    root = tmp_path / investors.DATA_DIR
    now = datetime(2026, 3, 2, 9, tzinfo=UTC)

    _ = investors.update(_OneFiler(), root, "0000000001", now, _quiet, sleep=_quiet)
    investors.add_investor(root, "0000000001", "A")

    assert Path("data", "investors") == investors.DATA_DIR
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert len(written) == 5  # filings, one table, CUSIPs, sectors, the list
    assert all(path.is_relative_to(root) for path in written)


def test_investors_are_isolated_from_the_dashboard() -> None:
    imports_investors = re.compile(
        r"^\s*(?:import|from)\s+finagent(?:\.investors\b| import .*\binvestors\b)"
    )
    others = [path for path in _python_files() if path != SRC / "app.py"]

    assert _hits(imports_investors, others) == []
    assert _hits(IMPORTS_BEYOND_MATH, [INVESTORS_MODULE]) == []


def test_math_modules_only_compute() -> None:
    """quant.py and perf.py take arrays and return numbers: no app, no I/O."""
    imports_finagent = re.compile(r"^\s*(?:import|from)\s+finagent\b")
    imports_io = re.compile(
        r"^\s*(?:import|from)\s+(?:httpx|yfinance|streamlit|pathlib|os|json)\b"
    )

    assert _hits(imports_finagent, list(MATH_MODULES)) == []
    assert _hits(imports_io, list(MATH_MODULES)) == []


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


def test_yahoo_caches_stay_under_their_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / YAHOO_CACHE
    pointed: list[str] = []
    day = pd.Timestamp("2026-10-09", tz="America/New_York")

    class Ticker:
        def __init__(self, _: str) -> None:
            pass

        def history(self, **_: object) -> pd.DataFrame:
            return pd.DataFrame({"Close": [4000.0]}, index=[day])

    fake = types.ModuleType("yfinance")
    fake.set_tz_cache_location = pointed.append  # pyright: ignore[reportAttributeAccessIssue]
    fake.Ticker = Ticker  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    closes = YahooCloses(root)
    _ = closes("GC=F", date(2026, 1, 1))
    _ = closes("GC=F", date(2026, 1, 1))

    assert Path("data", "macro", ".yfinance") == YAHOO_CACHE
    assert pointed == [str(root)]  # once, before the first request
    assert root.is_dir()


def test_yfinance_stays_in_its_homes() -> None:
    imported = re.compile(r"^\s*(?:import|from)\s+yfinance\b")
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    runtime = text.split("[project.optional-dependencies]")[0]
    extras = text.split("[project.optional-dependencies]")[1].split("\n[", 1)[0]
    screen_extra = extras.split("screen = [", 1)[1].split("]", 1)[0]

    homes = {SCREEN_MODULE, PORTFOLIO_MODULE, INVESTORS_MODULE, SOURCES_MODULE}
    assert _hits(imported, [p for p in _python_files() if p not in homes]) == []
    for home in homes:
        assert _hits(imported, [home]) != [], home
    assert "yfinance" not in runtime
    assert '"yfinance>=' in screen_extra


def test_arch_is_an_optional_extra_of_the_math_module() -> None:
    imported = re.compile(r"^\s*(?:import|from)\s+arch\b")
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    runtime = text.split("[project.optional-dependencies]")[0]
    extras = text.split("[project.optional-dependencies]")[1].split("\n[", 1)[0]
    garch_extra = extras.split("garch = [", 1)[1].split("]", 1)[0]
    home = SRC / "quant.py"

    assert _hits(imported, [p for p in _python_files() if p != home]) == []
    assert _hits(imported, [home]) != []
    assert "arch" not in runtime
    assert '"arch>=' in garch_extra


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
