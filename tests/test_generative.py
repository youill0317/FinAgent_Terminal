"""Scenario generators: numpy forward passes on tiny hand-made weights."""

# pandas ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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


def _sigcwgan(p: int = 2) -> Weights:
    rng = np.random.default_rng(1)
    d = len(TICKERS)
    w: Weights = {}
    _linear(w, "network.network.0.linear", d + p * d, 5, rng)
    _linear(w, "network.network.1.linear", 5, 5, rng)
    _linear(w, "network.network.2", 5, d, rng)
    w["network.network.0.activation.weight"] = np.array([0.25])
    w["network.network.1.activation.weight"] = np.array([0.25])
    w["past"] = 0.5 * rng.standard_normal((p, d))
    return w


def _nsde(hidden: int = 4, noise: int = 2) -> Weights:
    rng = np.random.default_rng(2)
    w: Weights = {}
    for prefix, n_out in (
        ("_func._drift._model", hidden),
        ("_func._diffusion._model", hidden * noise),
    ):
        _linear(w, f"{prefix}.0", 1 + hidden, 6, rng)
        _linear(w, f"{prefix}.2", 6, 6, rng)
        _linear(w, f"{prefix}.4", 6, n_out, rng)
    _linear(w, "_readout", hidden, len(TICKERS), rng)
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


def _models(root: Path) -> list[generative.Generator]:
    norm = {"mean": [0.001, 0.0, -0.001], "std": [0.02, 0.01, 0.015]}
    folders = [
        _write(root, "tailgan", _tailgan(), window=4),
        _write(root, "sigcwgan", _sigcwgan(), window=None, normalisation=norm),
        _write(root, "nsde", _nsde(), window=5, dt=1.0, tscale=1.0, normalisation=norm),
    ]
    loaded = [generative.load(folder) for folder in folders]
    assert all(model is not None for model in loaded)
    return [model for model in loaded if model is not None]


def test_paths_shape_dtype_and_simple_returns(tmp_path: Path) -> None:
    for model in _models(tmp_path):
        paths = model.paths(7, 11, np.random.default_rng(42))
        logs = model.log_paths(7, 11, np.random.default_rng(42))

        assert model.tickers == TICKERS
        assert paths.shape == (7, 11, len(TICKERS)), model.method
        assert paths.dtype == np.float64
        assert np.isfinite(paths).all()
        np.testing.assert_allclose(paths, np.expm1(logs))


def test_paths_are_deterministic_for_a_seed(tmp_path: Path) -> None:
    for model in _models(tmp_path):
        first = model.paths(5, 9, np.random.default_rng(42))
        again = model.paths(5, 9, np.random.default_rng(42))
        other = model.paths(5, 9, np.random.default_rng(43))

        np.testing.assert_array_equal(first, again)
        assert not np.allclose(first, other), model.method


def test_windowed_models_stack_independent_windows(tmp_path: Path) -> None:
    tailgan = _models(tmp_path)[0]
    long = tailgan.log_paths(3, 10, np.random.default_rng(42))
    z = np.random.default_rng(42).standard_t(generative.TAILGAN_DOF, (3 * 3, 5))
    windows = generative.tailgan_windows(
        tailgan.weights, z.astype(np.float32), len(TICKERS)
    )

    assert tailgan.window == 4
    assert windows.dtype == np.float32  # nothing upcasts the float32 network
    np.testing.assert_allclose(long, windows.reshape(3, 12, len(TICKERS))[:, :10])


def test_sigcwgan_conditions_on_history(tmp_path: Path) -> None:
    sigcwgan = _models(tmp_path)[1]
    days = pd.bdate_range("2026-01-01", periods=4)
    history = pd.DataFrame(0.01 * np.arange(12).reshape(4, 3), days, list(TICKERS))
    stored = sigcwgan.paths(4, 6, np.random.default_rng(42))
    recent = sigcwgan.paths(4, 6, np.random.default_rng(42), history)
    short = sigcwgan.paths(4, 6, np.random.default_rng(42), history.tail(1))

    assert sigcwgan.window is None
    assert not np.allclose(stored, recent)
    np.testing.assert_array_equal(stored, short)  # under p days: the stored past


def test_missing_weights_are_unavailable(tmp_path: Path) -> None:
    rows = generative.status(tmp_path)

    assert generative.load(tmp_path / "tailgan") is None
    assert generative.available(tmp_path) == []
    assert [row.model for row in rows] == [*generative.METHODS, "kronos"]
    assert all(not row.ready for row in rows)
    assert {row.note for row in rows[:-1]} == {generative.NO_WEIGHTS}


def test_status_names_each_trained_models_tickers(tmp_path: Path) -> None:
    _ = _models(tmp_path)

    rows = generative.status(tmp_path)

    assert all(row.ready for row in rows[:-1])
    assert {row.tickers for row in rows[:-1]} == {TICKERS}
    assert rows[-1].tickers == ()  # Kronos takes any


def test_tailgan_layers_by_hand() -> None:
    """latent 1 -> 1 -> 1 -> 1 -> 1 -> 2 outputs (1 asset, 2 days)."""
    one = np.array([[1.0]])
    w: Weights = {}
    for name in generative.TAILGAN_LINEARS[:-1]:
        w[f"{name}.weight"], w[f"{name}.bias"] = one, np.array([0.0])
    w["model.0.weight"] = np.array([[2.0]])
    w["model.11.weight"], w["model.11.bias"] = np.array([[1.0], [-3.0]]), np.zeros(2)
    for norm in ("model.3", "model.6", "model.9"):
        w[f"{norm}.weight"], w[f"{norm}.bias"] = np.array([1.0]), np.array([0.0])
        w[f"{norm}.running_mean"], w[f"{norm}.running_var"] = (
            np.zeros(1),
            np.full(1, 0.2),
        )
    z = np.array([[0.5], [-0.5]])
    out = generative.tailgan_windows(w, z, 1)

    # z = 0.5: 2 * 0.5 = 1, then three BatchNorms of 1 / sqrt(0.2 + 0.8) = 1:
    # 1, and the last layer gives (1, -3) clamped to (1, -1).
    # z = -0.5: -1 -> LeakyReLU -0.2 -> each BN keeps it, each LeakyReLU
    # scales by 0.2: -0.2^4 = -0.0016 -> (-0.0016, 0.0048).
    np.testing.assert_allclose(out[0, :, 0], [1.0, -1.0])
    np.testing.assert_allclose(out[1, :, 0], [-0.0016, 0.0048])


def test_sigcwgan_step_by_hand() -> None:
    """d = 1, p = 1: x = [z, past]; a residual PReLU(0.5) block, then Linear."""
    w: Weights = {
        "network.network.0.linear.weight": np.array([[1.0, 1.0], [1.0, -1.0]]),
        "network.network.0.linear.bias": np.array([0.0, 0.0]),
        "network.network.0.activation.weight": np.array([0.5]),
        "network.network.1.weight": np.array([[1.0, 2.0]]),
        "network.network.1.bias": np.array([0.1]),
    }
    z = np.array([[[1.0], [0.0]]])
    out = generative.sigcwgan_steps(w, z, np.array([[[2.0]]]))

    # step 1: [1, 2] -> (3, -1) -> PReLU (3, -0.5) + [1, 2] = (4, 1.5)
    #         -> 4 + 2 * 1.5 + 0.1 = 7.1, the new past
    # step 2: [0, 7.1] -> (7.1, -7.1) -> (7.1, -3.55) + [0, 7.1] = (7.1, 3.55)
    #         -> 7.1 + 7.1 + 0.1 = 14.3
    np.testing.assert_allclose(out[0, :, 0], [7.1, 14.3])


def test_nsde_euler_step_by_hand() -> None:
    """hidden 1, noise 1, one-layer fields: x1 = 1 + tanh(.) dt + tanh(.) dW."""
    w: Weights = {
        "_func._drift._model.0.weight": np.array([[0.0, 0.5]]),  # [t, x]
        "_func._drift._model.0.bias": np.array([0.0]),
        "_func._diffusion._model.0.weight": np.array([[1.0, 0.0]]),
        "_func._diffusion._model.0.bias": np.array([0.2]),
        "_readout.weight": np.array([[2.0]]),
        "_readout.bias": np.array([1.0]),
    }
    dw = np.array([[[0.3], [-0.1]]])
    out = generative.nsde_paths(w, dw, 1.0, 1.0)

    x1 = 1 + math.tanh(0.5) + math.tanh(0.2) * 0.3
    x2 = x1 + math.tanh(0.5 * x1) + math.tanh(1.2) * -0.1
    np.testing.assert_allclose(out[0, :, 0], [0.0, 2 * (x1 - 1), 2 * (x2 - 1)])


def test_lipswish_is_stable_for_large_inputs() -> None:
    x = np.array([-1e4, 0.0, 1e4])

    np.testing.assert_allclose(generative._lipswish(x), [0.0, 0.0, 0.909e4])  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_kronos_weights_do_not_load_as_a_numpy_generator(tmp_path: Path) -> None:
    folder = _write(tmp_path, "kronos", _tailgan(), window=4)

    assert generative.load(folder) is None
