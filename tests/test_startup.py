from __future__ import annotations

import os
import subprocess
import sys

LLM_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)


def env_without_llm_keys() -> dict[str, str]:
    env = os.environ.copy()
    for key in LLM_KEYS:
        _ = env.pop(key, None)
    return env


def test_package_imports() -> None:
    # Given: the package has been installed in the active uv environment.
    # When: importing the top-level package.
    from finagent_terminal import __version__

    # Then: import succeeds and exposes a version string.
    assert isinstance(__version__, str)


def test_startup_without_llm_credentials() -> None:
    # Given: no optional LLM provider API keys are configured.
    env = env_without_llm_keys()

    # When: the module CLI help is requested.
    result = subprocess.run(
        [sys.executable, "-m", "finagent_terminal", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
    )

    # Then: startup does not require LLM credentials.
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, output
    assert "finagent" in output.lower()
