"""Scenario generators: numpy forward passes on tiny hand-made weights."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from finagent import generative

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

type Weights = dict[str, NDArray[np.float64]]

TICKERS = ("AAA", "BBB", "CCC")


def _linear(
    w: Weights, name: str, n_in: int, n_out: int, rng: np.random.Generator
) -> None:
    w[f"{name}.weight"] = 0.3 * rng.standard_normal((n_out, n_in))
    w[f"{name}.bias"] = 0.1 * rng.standard_normal(n_out)


def _tailgan(window: int = 4) -> Weights:
    rng = np.random.default_rng(0)
    w: Weights = {}
    sizes = ((5, 6), (6, 7), (7, 8), (8, 9), (9, len(TICKERS) * window))
    for name, norm, (n_in, n_out) in zip(
        generative.TAILGAN_LINEARS, generative.TAILGAN_NORMS, sizes, strict=True
    ):
        _linear(w, name, n_in, n_out, rng)
        if norm is not None:
            w[f"{norm}.weight"] = 1 + 0.1 * rng.standard_normal(n_out)
            w[f"{norm}.bias"] = 0.1 * rng.standard_normal(n_out)
            w[f"{norm}.running_mean"] = 0.1 * rng.standard_normal(n_out)
            w[f"{norm}.running_var"] = 1 + rng.random(n_out)
    return w


def _write(root: Path, method: str, weights: Weights, **meta: object) -> Path:
    folder = root / method
    folder.mkdir(parents=True)
    np.savez(folder / generative.WEIGHTS, allow_pickle=False, **weights)
    info = {
        "format": generative.FORMAT,
        "method": method,
        "tickers": list(TICKERS),
        "train_start": "2007-01-04",
        "train_end": "2022-10-26",
        **meta,
    }
    _ = (folder / generative.META).write_text(json.dumps(info), "utf-8")
    return folder


def test_missing_weights_are_unavailable(tmp_path: Path) -> None:
    rows = generative.status(tmp_path)

    assert generative.load(tmp_path / "tailgan") is None
    assert generative.available(tmp_path) == []
    assert [row.model for row in rows] == [*generative.METHODS, "kronos"]
    assert all(not row.ready for row in rows)
    assert {row.note for row in rows[:-1]} == {generative.NO_WEIGHTS}


def test_kronos_weights_do_not_load_as_a_numpy_generator(tmp_path: Path) -> None:
    folder = _write(tmp_path, "kronos", _tailgan(), window=4)

    assert generative.load(folder) is None
