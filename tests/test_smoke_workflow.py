from __future__ import annotations

from typing import TYPE_CHECKING

from finagent_terminal.__main__ import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_smoke_workflow_runs_with_fake_integrations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: smoke mode is isolated from real market-data and LLM providers.
    monkeypatch.setenv("FINAGENT_HOME", str(tmp_path))
    monkeypatch.setenv("FINAGENT_FAKE_DATA", "1")
    monkeypatch.setenv("FINAGENT_FAKE_LLM", "1")
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    # When: the CLI smoke workflow runs.
    status = main(["--smoke"])

    # Then: every workspace path reports observable output.
    output = capsys.readouterr().out
    assert status == 0
    assert "SMOKE markets: Data:" in output
    assert "SMOKE ticker: Symbol: AAPL" in output
    assert "SMOKE tools: Normalized returns:" in output
    assert "SMOKE reports: Completed: 5/5" in output
    assert "SMOKE settings: Providers:" in output
    report_paths = tuple((tmp_path / "reports").glob("*.md"))
    assert report_paths != ()
