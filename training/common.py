"""Shared helpers for the Colab training scripts (torch; never run locally).

Each ``train_*.py`` loads ``returns.npz`` (``export_dataset.py``), trains one
generator on a chronological 80% split with seed 42, and writes to its output
folder what ``src/finagent/generative.py`` reads:

- ``weights.npz``: the generator's ``state_dict`` as float32 arrays;
- ``meta.json``: method, tickers, dates, window, normalisation, architecture,
  training config, reference repo and commit, versions, the torch-vs-numpy
  parity check and a few diagnostics;
- ``log.txt`` (written by the launcher) with the training output.
"""

# ruff: noqa: T201, INP001, S101

from __future__ import annotations

import importlib.util
import json
import os
import platform
import random
import sys
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

SEED: Final = 42  # the app's seed (quant.SEED)
TRAIN_SHARE: Final = 0.8  # chronological train/test split, as the SigCWGAN repo
FORMAT: Final = 1
SMOKE: Final = os.environ.get("SMOKE") == "1"  # a few steps, to test the pipeline


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - the reference code uses the global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_returns(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Daily log returns (days, assets), tickers and ISO dates."""
    with np.load(path) as data:
        return (
            data["returns"].astype(np.float64),
            [str(t) for t in data["tickers"]],
            [str(d) for d in data["dates"]],
        )


def split(n: int) -> int:
    """Days in the training part of ``n``."""
    return int(n * TRAIN_SHARE)


def generative(path: Path) -> ModuleType:
    """``src/finagent/generative.py`` itself, uploaded next to the scripts."""
    spec = importlib.util.spec_from_file_location("generative", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generative"] = module
    spec.loader.exec_module(module)
    return module


def save(
    out: Path, state: dict[str, torch.Tensor], meta: dict[str, object]
) -> dict[str, np.ndarray]:
    """Write ``weights.npz`` (float32) and ``meta.json``; return the weights."""
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        k: v.detach().cpu().float().numpy()
        for k, v in state.items()
        if v.is_floating_point()
    }
    np.savez(out / "weights.npz", **arrays)
    full = {"format": FORMAT, "seed": SEED, **meta, "versions": versions()}
    (out / "meta.json").write_text(json.dumps(full, indent=2), "utf-8")
    print(f"saved {out}: {sum(a.size for a in arrays.values())} parameters")
    # numpy reads the float32 file back, so parity checks use those values
    return {k: v.astype(np.float64) for k, v in arrays.items()}


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda or "",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def update_meta(out: Path, **fields: object) -> None:
    path = out / "meta.json"
    meta = json.loads(path.read_text("utf-8"))
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2), "utf-8")


def tail_stats(log_rets: np.ndarray, horizon: int) -> dict[str, float]:
    """Equal-weight portfolio over ``horizon`` days: std, 95% VaR and ES.

    ``log_rets`` is (paths, days >= horizon, assets); losses are positive
    fractions of the simple portfolio return.
    """
    simple = np.expm1(log_rets[:, :horizon].sum(axis=1))  # per asset, per path
    port = simple.mean(axis=1)
    loss = -port
    var = float(np.quantile(loss, 0.95))
    return {
        "std": float(port.std()),
        "var95": var,
        "es95": float(loss[loss >= var].mean()),
    }


def windows(rets: np.ndarray, length: int) -> np.ndarray:
    """All overlapping windows of ``length`` days, (n, length, assets)."""
    idx = np.arange(len(rets) - length + 1)[:, None] + np.arange(length)
    return rets[idx]


def diagnostics(
    fake: np.ndarray, train: np.ndarray, test: np.ndarray, horizons: tuple[int, ...]
) -> dict[str, object]:
    """Generated vs real daily std per asset and portfolio tails per horizon."""
    out: dict[str, object] = {
        "daily_std": {
            "generated": fake.reshape(-1, fake.shape[-1]).std(axis=0).tolist(),
            "train": train.std(axis=0).tolist(),
            "test": test.std(axis=0).tolist(),
        }
    }
    for h in horizons:
        if fake.shape[1] < h:
            continue
        row: dict[str, object] = {"generated": tail_stats(fake, h)}
        for name, real in (("train", train), ("test", test)):
            if len(real) >= h + 50:
                row[name] = tail_stats(windows(real, h), h)
        out[f"{h}d"] = row
    return out
