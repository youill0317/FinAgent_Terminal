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
    assert "SMOKE markets compare: Compare: Smoke" in output
    assert "SMOKE ticker: Symbol: AAPL" in output
    assert "SMOKE ticker compare: Compare: AAPL vs SPY, QQQ, DIA" in output
    assert "SMOKE news: Market news" in output
    assert "SMOKE tools: Analysis hub" in output
    assert "SMOKE tools monte carlo: 20 paths over 5 days" in output
    assert "SMOKE tools: Compare:" not in output
    assert "SMOKE tools: Normalized returns:" not in output
    assert "SMOKE reports: Completed: 7/7" in output
    assert "Exported:" in output
    assert "SMOKE settings: Providers:" in output
    report_paths = tuple((tmp_path / "reports").glob("*.md"))
    assert report_paths != ()

def test_smoke_workflow_missing_fake_data_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("FINAGENT_FAKE_DATA", raising=False)
    monkeypatch.setenv("FINAGENT_FAKE_LLM", "1")

    status = main(["--smoke"])

    captured = capsys.readouterr()
    assert status == 2
    assert "--smoke requires FINAGENT_FAKE_DATA=1" in captured.err

def test_smoke_workflow_missing_fake_llm_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FINAGENT_FAKE_DATA", "1")
    monkeypatch.delenv("FINAGENT_FAKE_LLM", raising=False)

    status = main(["--smoke"])

    captured = capsys.readouterr()
    assert status == 2
    assert "--smoke requires FINAGENT_FAKE_LLM=1" in captured.err
