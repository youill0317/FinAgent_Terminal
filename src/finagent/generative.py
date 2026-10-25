"""Scenario generators trained offline, run here as numpy forward passes.

Three generative models are trained once on Colab (``training/``) on the
daily log returns of the portfolio page's assets and exported, one folder
each under ``data/models/<method>/``, as ``weights.npz`` (the torch
``state_dict``, float32) and ``meta.json`` (method, tickers, training dates,
window, normalisation, architecture, versions, parity check):

- ``tailgan``: Tail-GAN (Cont, Cucuringu, Xu and Zhang, Management Science).
  An MLP maps Student-t noise to a fixed window of returns; its
  discriminator, used only in training, scores VaR and ES of portfolios.
- ``sigcwgan``: conditional Sig-Wasserstein GAN (Liao, Ni, Szpruch, Wiese,
  Sabate-Vidales and Xiao, 2020). An autoregressive residual MLP draws the
  next day from noise and the last ``p`` days, so any horizon.
- ``nsde``: neural SDE trained without a discriminator on the signature
  kernel score (Issa, Horvath, Lemercier and Salvi, NeurIPS 2023), solved
  with Euler-Maruyama over a fixed window.

Windowed models (``Generator.window``) reach longer horizons by stacking
independent windows, so nothing carries over between them (volatility
clustering across windows is lost).

Kronos (shiyu-coder/Kronos, a K-line foundation model) is not trained
here. It runs only when its files are in ``data/models/kronos/``
(``KRONOS_FILES``) and torch is installed; torch is imported only then.
Until both are there its status is "구현 중".

Conventions: ``paths`` returns simple daily returns, shape
(paths, horizon, assets), each asset in its own currency, columns in
``tickers`` order. Nothing here writes a file.
"""

# numpy's loaders and the lazily imported Kronos code are untyped.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, get_args

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

type Floats = NDArray[np.float64]
type Net = NDArray[np.floating]  # weights and activations: float32 from ``load``
type Method = Literal["tailgan", "sigcwgan", "nsde"]
type Model = Literal["tailgan", "sigcwgan", "nsde", "kronos"]

MODELS_DIR: Final = Path("data") / "models"
WEIGHTS: Final = "weights.npz"
META: Final = "meta.json"
FORMAT: Final = 1  # meta.json "format" this module reads
METHODS: Final[tuple[Method, ...]] = get_args(Method.__value__)
LABELS: Final[dict[Model, str]] = {
    "tailgan": "Tail-GAN",
    "sigcwgan": "SigCWGAN",
    "nsde": "Sig-kernel neural SDE",
    "kronos": "Kronos",
}
IN_PROGRESS: Final = "구현 중"
NO_WEIGHTS: Final = "가중치 없음"
NO_TORCH: Final = "torch 미설치"

TAILGAN_DOF: Final = 5  # Student-t noise, the repo's "t5"
TAILGAN_SLOPE: Final = 0.2  # LeakyReLU
TAILGAN_CHUNK: Final = 4096  # windows per noise draw: 4096 x 1000 latent, 33 MB
TAILGAN_LINEARS: Final = ("model.0", "model.2", "model.5", "model.8", "model.11")
TAILGAN_NORMS: Final = (None, "model.3", "model.6", "model.9", None)
SIGCWGAN_NET: Final = "network.network"
LIPSWISH: Final = 0.909

KRONOS_DIR: Final = "kronos"
# HF snapshots (config.json + model.safetensors) and the repo's model package.
KRONOS_FILES: Final = (
    "tokenizer/config.json",
    "tokenizer/model.safetensors",
    "model/config.json",
    "model/model.safetensors",
    "src/model/__init__.py",
    "src/model/kronos.py",
    "src/model/module.py",
)
KRONOS_CONTEXT: Final = 512  # Kronos-small's max_context
KRONOS_TEMPERATURE: Final = 1.0  # the README's sampling settings
KRONOS_TOP_P: Final = 0.9


# ---------------------------------------------------------------- layers


def _linear(x: Net, w: Mapping[str, Net], name: str) -> Net:
    return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]


def _sigmoid(x: Net) -> Net:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # no overflow for large |x|


def _lipswish(x: Net) -> Net:
    return LIPSWISH * x * _sigmoid(x)


def _mlp(x: Net, w: Mapping[str, Net], prefix: str, tscale: float) -> Net:
    """sigker-nsdes ``MLP``: Linear, LipSwish, ..., Linear, ``tscale`` tanh."""
    names = _linears(w, prefix)
    for name in names[:-1]:
        x = _lipswish(_linear(x, w, name))
    return tscale * np.tanh(_linear(x, w, names[-1]))


def _linears(w: Mapping[str, Net], prefix: str) -> list[str]:
    """Linear layers under ``prefix`` (``<prefix>.<i>.weight``), by index."""
    head = f"{prefix}."
    found = {
        int(key[len(head) :].split(".")[0])
        for key in w
        if key.startswith(head) and key.endswith(".weight")
    }
    return [f"{prefix}.{i}" for i in sorted(found)]


# ---------------------------------------------------------------- forward passes


def tailgan_windows(w: Mapping[str, Net], z: Net, assets: int) -> Net:
    """Tail-GAN generator in eval mode: (n, latent) noise to (n, T, assets).

    Linear-LeakyReLU, then three Linear-BatchNorm-LeakyReLU blocks (the
    repo's ``BatchNorm1d(n, 0.8)`` makes eps 0.8; running statistics), a
    Linear and a clamp to [-1, 1]. Torch reshapes to (n, assets, T).
    """
    x = z
    for name, norm in zip(TAILGAN_LINEARS[:-1], TAILGAN_NORMS[:-1], strict=True):
        x = _linear(x, w, name)
        if norm is not None:
            scale = w[f"{norm}.weight"] / np.sqrt(w[f"{norm}.running_var"] + 0.8)
            x = (x - w[f"{norm}.running_mean"]) * scale + w[f"{norm}.bias"]
        x = np.where(x > 0, x, TAILGAN_SLOPE * x)
    x = np.clip(_linear(x, w, TAILGAN_LINEARS[-1]), -1.0, 1.0)
    return x.reshape(len(z), assets, -1).transpose(0, 2, 1)


def sigcwgan_steps(w: Mapping[str, Net], z: Net, past: Net) -> Net:
    """SigCWGAN ``ArFNN``: noise (n, steps, d) and past (n, p, d) to (n, steps, d).

    Each step feeds [z_t, past flattened oldest first] through residual
    blocks (Linear then PReLU, plus the input when the sizes match) and a
    final Linear, then shifts the new day into the past.
    """
    *blocks, out = _linears(w, SIGCWGAN_NET)
    window = past.copy()
    days: list[Net] = []
    for t in range(z.shape[1]):
        x = np.concatenate([z[:, t], window.reshape(len(window), -1)], axis=1)
        for block in blocks:
            y = _linear(x, w, f"{block}.linear")
            y = np.where(y >= 0, y, w[f"{block}.activation.weight"] * y)
            x = x + y if y.shape == x.shape else y
        day = _linear(x, w, out)
        window = np.concatenate([window[:, 1:], day[:, None]], axis=1)
        days.append(day)
    return np.stack(days, axis=1)


def nsde_paths(w: Mapping[str, Net], dw: Net, dt: float, tscale: float) -> Net:
    """Neural SDE paths from Brownian increments ``dw`` (n, steps, noise).

    Euler-Maruyama (Ito) from the fixed start x0 = 1 on the grid t = k dt:
    x += drift(t, x) dt + diffusion(t, x) dW, then a linear readout and the
    start subtracted. Returns (n, steps + 1, assets), zero on day 0, in
    ``dw``'s dtype.
    """
    n, steps, noise = dw.shape
    hidden = w["_readout.weight"].shape[1]
    x = np.ones((n, hidden), dw.dtype)
    tx = np.empty((n, hidden + 1), dw.dtype)  # [t, x], refilled each step
    states = [x]
    for k in range(steps):
        tx[:, 0] = k * dt
        tx[:, 1:] = x
        drift = _mlp(tx, w, "_func._drift._model", tscale)
        diffusion = _mlp(tx, w, "_func._diffusion._model", tscale)
        shock = np.einsum("nhm,nm->nh", diffusion.reshape(n, hidden, noise), dw[:, k])
        x = x + drift * dt + shock
        states.append(x)
    ys = _linear(np.stack(states, axis=1), w, "_readout")
    return ys - ys[:, :1]


# ---------------------------------------------------------------- generators


@dataclass(frozen=True, slots=True)
class Generator:
    """A trained scenario generator; ``load`` reads one."""

    method: Method
    tickers: tuple[str, ...]
    window: int | None  # days per independent draw; None: any horizon
    meta: Mapping[str, object]
    weights: Mapping[str, Net]

    @property
    def mean(self) -> Floats:
        return _floats(self.meta, "mean", len(self.tickers), 0.0)

    @property
    def std(self) -> Floats:
        return _floats(self.meta, "std", len(self.tickers), 1.0)

    @property
    def label(self) -> str:
        return LABELS[self.method]

    def paths(
        self,
        n: int,
        horizon: int,
        rng: np.random.Generator,
        history: pd.DataFrame | None = None,
    ) -> Floats:
        """``n`` simulated paths of ``horizon`` days: simple returns.

        Shape (n, horizon, assets). ``history`` (simple daily returns, one
        column per ticker) conditions SIGCWGAN on its last ``p`` days;
        without it, on the last days of its training data. The other
        methods ignore it.
        """
        log = self.log_paths(n, horizon, rng, history)
        return np.expm1(log)

    def log_paths(
        self,
        n: int,
        horizon: int,
        rng: np.random.Generator,
        history: pd.DataFrame | None = None,
    ) -> Floats:
        """``paths`` as daily log returns."""
        d = len(self.tickers)
        if self.method == "sigcwgan":
            past = self._past(history).astype(np.float32)
            z = rng.standard_normal((n, horizon, d)).astype(np.float32)
            x = sigcwgan_steps(self.weights, z, np.repeat(past[None], n, axis=0))
            return x * self.std + self.mean
        window = cast("int", self.window)
        blocks = math.ceil(horizon / window)
        if self.method == "tailgan":
            latent = self.weights[f"{TAILGAN_LINEARS[0]}.weight"].shape[1]
            sizes = [
                min(TAILGAN_CHUNK, n * blocks - i)
                for i in range(0, n * blocks, TAILGAN_CHUNK)
            ]
            x = np.concatenate(
                [
                    tailgan_windows(
                        self.weights,
                        rng.standard_t(TAILGAN_DOF, (m, latent)).astype(np.float32),
                        d,
                    )
                    for m in sizes
                ]
            )
        else:
            dt = _number(self.meta, "dt", 1.0)
            hidden = self.weights["_readout.weight"].shape[1]
            noise = _linear_out(self.weights, "_func._diffusion._model") // hidden
            dw = rng.standard_normal((n * blocks, window, noise)) * math.sqrt(dt)
            dw = dw.astype(np.float32)
            cumulative = nsde_paths(
                self.weights, dw, dt, _number(self.meta, "tscale", 1.0)
            )
            x = np.diff(cumulative, axis=1)
        x = x * self.std + self.mean
        return x.reshape(n, blocks * window, d)[:, :horizon]

    def _past(self, history: pd.DataFrame | None) -> Net:
        """The standardised last ``p`` days SIGCWGAN conditions on."""
        stored = self.weights["past"]
        if history is None:
            return stored
        rows = history.reindex(columns=list(self.tickers)).dropna().tail(len(stored))
        if len(rows) < len(stored):
            return stored
        return (np.log1p(rows.to_numpy(dtype="float64")) - self.mean) / self.std


def _linear_out(w: Mapping[str, Net], prefix: str) -> int:
    return w[f"{_linears(w, prefix)[-1]}.weight"].shape[0]


def _number(meta: Mapping[str, object], key: str, default: float) -> float:
    value = meta.get(key, default)
    return float(cast("float", value))


def _floats(meta: Mapping[str, object], key: str, size: int, fill: float) -> Floats:
    norm = cast("Mapping[str, object]", meta.get("normalisation", {}))
    if key not in norm:
        return np.full(size, fill)
    return np.asarray(norm[key], dtype=np.float64)


def load(folder: Path) -> Generator | None:
    """The generator in ``folder`` (``WEIGHTS`` + ``META``); None without one."""
    weights, meta = folder / WEIGHTS, folder / META
    if not (weights.is_file() and meta.is_file()):
        return None
    info = cast("dict[str, object]", json.loads(meta.read_text("utf-8")))
    method = info.get("method")
    if info.get("format") != FORMAT or method not in METHODS:
        return None
    with np.load(weights) as archive:
        arrays = {key: archive[key].astype(np.float32) for key in archive.files}
    window = info.get("window")
    return Generator(
        method=method,
        tickers=tuple(cast("list[str]", info["tickers"])),
        window=None if method == "sigcwgan" else int(cast("int", window)),
        meta=info,
        weights=arrays,
    )


# ---------------------------------------------------------------- Kronos


def kronos_missing(folder: Path) -> list[str]:
    """``KRONOS_FILES`` not under ``folder``."""
    return [name for name in KRONOS_FILES if not (folder / name).is_file()]


class _Predictor(Protocol):
    """The part of the repo's ``KronosPredictor`` used here."""

    def predict_batch(self, *args: object, **kwargs: object) -> list[pd.DataFrame]: ...


@dataclass(frozen=True, slots=True)
class Kronos:
    """Kronos-small sampling each asset's next closes from its own history.

    Kronos reads one OHLCV series at a time, so each asset is drawn
    independently (no co-movement), from closes rebuilt out of
    ``history``: open = high = low = close, no volume.
    """

    predictor: _Predictor
    seed: Callable[[int], object]  # torch.manual_seed
    tickers: tuple[str, ...] = ()  # any: those in ``history``
    window: int | None = None
    label: str = LABELS["kronos"]

    def paths(
        self,
        n: int,
        horizon: int,
        rng: np.random.Generator,
        history: pd.DataFrame | None = None,
    ) -> Floats:
        """``n`` sampled paths per column of ``history``: simple returns."""
        if history is None or history.empty:
            raise ValueError("Kronos는 과거 수익률(history)이 있어야 합니다.")
        _ = self.seed(int(rng.integers(2**31 - 1)))
        columns = [
            self._one(cast("pd.Series", history[c]).dropna(), n, horizon)
            for c in history.columns
        ]
        return np.stack(columns, axis=2)

    def _one(self, rets: pd.Series, n: int, horizon: int) -> Floats:
        close = (1.0 + rets).cumprod().tail(KRONOS_CONTEXT)
        frame = pd.DataFrame({c: close.to_numpy() for c in ("open", "high", "low")})
        frame["close"] = close.to_numpy()
        past = pd.Series(pd.DatetimeIndex(close.index))
        last = cast("pd.Timestamp", close.index[-1])
        future = pd.Series(pd.bdate_range(last + pd.offsets.BDay(), periods=horizon))
        drawn = self.predictor.predict_batch(
            [frame] * n,
            [past] * n,
            [future] * n,
            pred_len=horizon,
            T=KRONOS_TEMPERATURE,
            top_p=KRONOS_TOP_P,
            sample_count=1,
            verbose=False,
        )
        level = np.stack([d["close"].to_numpy(dtype="float64") for d in drawn])
        start = np.full((n, 1), float(close.iloc[-1]))
        return level / np.concatenate([start, level[:, :-1]], axis=1) - 1.0


def load_kronos(folder: Path) -> Kronos | None:
    """Kronos from ``folder`` when all its files and torch are there; else None.

    Imports torch and the repo's ``model`` package (from ``folder/src``)
    only then.
    """
    if kronos_missing(folder) or importlib.util.find_spec("torch") is None:
        return None
    torch = importlib.import_module("torch")
    source = str(folder / "src")
    sys.path.insert(0, source)
    try:
        code = importlib.import_module("model")
    finally:
        sys.path.remove(source)
    tokenizer = code.KronosTokenizer.from_pretrained(str(folder / "tokenizer"))
    model = code.Kronos.from_pretrained(str(folder / "model"))
    predictor = code.KronosPredictor(
        model, tokenizer, device="cpu", max_context=KRONOS_CONTEXT
    )
    return Kronos(predictor=cast("_Predictor", predictor), seed=torch.manual_seed)


# ---------------------------------------------------------------- status


@dataclass(frozen=True, slots=True)
class Status:
    model: Model
    label: str
    ready: bool
    note: str  # page text: the training span, or why it is not ready
    tickers: tuple[str, ...] = ()  # trained on; () when not ready, and for Kronos


def status(root: Path = MODELS_DIR) -> list[Status]:
    """Each model's status under ``root``, Kronos last; reads only meta.json."""
    rows: list[Status] = []
    for method in METHODS:
        folder = root / method
        meta = folder / META
        if not ((folder / WEIGHTS).is_file() and meta.is_file()):
            rows.append(Status(method, LABELS[method], ready=False, note=NO_WEIGHTS))
            continue
        info = cast("dict[str, object]", json.loads(meta.read_text("utf-8")))
        span = f"{info.get('train_start', '?')}–{info.get('train_end', '?')} 학습"
        tickers = tuple(str(t) for t in cast("list[object]", info.get("tickers", [])))
        rows.append(
            Status(method, LABELS[method], ready=True, note=span, tickers=tickers)
        )
    folder = root / KRONOS_DIR
    if kronos_missing(folder):
        rows.append(Status("kronos", LABELS["kronos"], ready=False, note=IN_PROGRESS))
    elif importlib.util.find_spec("torch") is None:
        rows.append(Status("kronos", LABELS["kronos"], ready=False, note=NO_TORCH))
    else:
        rows.append(Status("kronos", LABELS["kronos"], ready=True, note="Kronos-small"))
    return rows


def available(root: Path = MODELS_DIR) -> list[Model]:
    """Models ready to draw paths."""
    return [s.model for s in status(root) if s.ready]
