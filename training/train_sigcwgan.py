"""Train SigCWGAN on Colab (one GPU) and export it for ``finagent.generative``.

    python train_sigcwgan.py RETURNS.npz OUT_DIR GENERATIVE.py REPO_DIR

Reference: Liao, Ni, Sabate-Vidales, Szpruch, Wiese and Xiao, "Conditional
Sig-Wasserstein GANs for Time Series Generation" (2020), official code
SigCGANs/Conditional-Sig-Wasserstein-GANs at commit
32ea56bd4abc6b934cf754279c3326ac3fae65e4, cloned to REPO_DIR and used as is
(``SigCWGAN``, ``SimpleGenerator``/``ArFNN``, the augmentations and
``StandardScalerTS``).

The repo needs ``signatory==1.2.1.1.4.0``, built for torch 1.4, which does
not install on Colab's torch 2. ``signatory.signature`` is its only use, so a
stand-in module computes the same truncated signature (depth 2, no
basepoint, levels concatenated) in plain torch; it is checked against
``iisignature`` before training.

Settings are the repo's ``train.py`` defaults and its multivariate stock
config ``STOCKS_SPX_DJI``: p = q = 3, hidden (50, 50, 50), batch 200,
1000 steps, mc_size 1000, Adam 1e-2 with StepLR(100, 0.9), gradient clip
10, depth-2 signatures of Scale(0.2), Cumsum, AddLags(2), LeadLag, and
per-asset standard scaling over the whole series. The stock config also
models realised variance channels, which this data has not; only returns.
Seed 42, chronological 80% train split (as ``train.py``).
"""

# ruff: noqa: T201, INP001, PLR0915, PLR2004, S101, PLC0415

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import common
import numpy as np
import torch

REPO = "SigCGANs/Conditional-Sig-Wasserstein-GANs"
COMMIT = "32ea56bd4abc6b934cf754279c3326ac3fae65e4"
P, Q = 3, 3
HIDDEN = (50, 50, 50)
BATCH = 200
STEPS = 5 if common.SMOKE else 1000
MC_SIZE = 1000
DEPTH = 2


def signature(path: torch.Tensor, depth: int, basepoint: bool = False) -> torch.Tensor:
    """signatory.signature for depth <= 2 of a piecewise linear path.

    path (batch, stream, channels) -> (batch, C + C^2): level 1 is the total
    increment, level 2 [i, j] = sum_k (X_{k-1} - X_0)^i dX_k^j + dX_k^i dX_k^j / 2,
    flattened row-major, as signatory and iisignature order words.
    """
    assert depth in (1, 2)
    assert not basepoint
    dx = path[:, 1:] - path[:, :-1]
    level1 = dx.sum(dim=1)
    if depth == 1:
        return level1
    before = path[:, :-1] - path[:, :1]
    level2 = torch.einsum("bki,bkj->bij", before, dx) + 0.5 * torch.einsum(
        "bki,bkj->bij", dx, dx
    )
    return torch.cat([level1, level2.flatten(1)], dim=1)


def check_signature() -> float:
    import iisignature

    x = np.random.default_rng(0).standard_normal((4, 7, 3))
    ours = signature(torch.tensor(x), 2).numpy()
    ref = np.stack([iisignature.sig(p, 2) for p in x])
    return float(np.abs(ours - ref).max())


def main(data: Path, out: Path, gen_module: Path, repo: Path) -> None:
    sys.modules["signatory"] = types.SimpleNamespace(signature=signature)  # type: ignore[assignment]
    sys.path.insert(0, str(repo))
    from lib.algos.base import BaseConfig
    from lib.algos.sigcwgan import SigCWGAN, SigCWGANConfig
    from lib.augmentations import AddLags, Cumsum, LeadLag, Scale, SignatureConfig
    from lib.data import Pipeline, StandardScalerTS, rolling_window

    sig_error = check_signature()
    print(f"torch signature vs iisignature max abs diff: {sig_error:.2e}", flush=True)
    assert sig_error < 1e-10

    common.seed_everything()
    dev = common.device()
    rets, tickers, dates = common.load_returns(data)
    d = rets.shape[1]

    # lib.data.get_equities_dataset: scale the whole series, then roll windows.
    raw = torch.from_numpy(rets[None]).float()
    pipeline = Pipeline(steps=[("standard_scale", StandardScalerTS(axis=(0, 1)))])
    scaled = pipeline.transform(raw)
    scaler = pipeline.steps[0][1]
    x_real = rolling_window(scaled[0], P + Q).to(dev)
    ind_train = int(x_real.shape[0] * common.TRAIN_SHARE)
    x_train = x_real[:ind_train]

    augmentations = (Scale(0.2), Cumsum(), AddLags(m=2), LeadLag())
    config = SigCWGANConfig(
        mc_size=MC_SIZE,
        sig_config_past=SignatureConfig(depth=DEPTH, augmentations=augmentations),
        sig_config_future=SignatureConfig(depth=DEPTH, augmentations=augmentations),
    )
    base = BaseConfig(
        device=str(dev),
        seed=common.SEED,
        batch_size=BATCH,
        hidden_dims=HIDDEN,
        p=P,
        q=Q,
        total_steps=STEPS,
        mc_samples=1000,
    )
    common.seed_everything()
    algo = SigCWGAN(base_config=base, config=config, x_real=x_train)
    start = time.time()
    algo.fit()
    seconds = time.time() - start
    loss = algo.training_loss["loss"]
    print(
        f"trained {STEPS} steps in {seconds:.0f}s, loss {loss[0]:.4f} -> {loss[-1]:.4f}"
    )

    generator = algo.G.eval()
    mean = scaler.mean.cpu().double().numpy().reshape(-1)
    std = scaler.std.cpu().double().numpy().reshape(-1)
    past = scaled[0, -P:].clone()  # the last p standardised days
    state = dict(generator.state_dict())
    state["past"] = past
    # rolling_window drops the last window; the train part ends at row
    # ind_train + P + Q - 2 of the returns.
    train_end = ind_train + P + Q - 2
    meta = {
        "method": "sigcwgan",
        "tickers": tickers,
        "train_start": dates[0],
        "train_end": dates[train_end],
        "test_end": dates[-1],
        "past_end": dates[-1],
        "window": None,
        "returns": "log",
        "normalisation": {"mean": mean.tolist(), "std": std.tolist()},
        "architecture": {
            "generator": "ArFNN (SimpleGenerator)",
            "p": P,
            "q": Q,
            "hidden": list(HIDDEN),
            "latent_dim": d,
            "activation": "PReLU (one slope per block)",
            "input": "[z_t, past flattened oldest first]",
        },
        "training": {
            "steps": STEPS,
            "batch_size": BATCH,
            "mc_size": MC_SIZE,
            "optimizer": "Adam(lr=1e-2), StepLR(100, 0.9), clip 10",
            "signature": "depth 2; Scale(0.2), Cumsum, AddLags(2), LeadLag",
            "signature_backend": "plain torch (signatory stand-in)",
            "signature_check_vs_iisignature": sig_error,
            "windows": int(ind_train),
            "seconds": round(seconds),
            "loss_first_last": [float(loss[0]), float(loss[-1])],
        },
        "source": {"repo": REPO, "commit": COMMIT, "paper": "Liao et al., SigCWGAN"},
    }
    weights = common.save(out, state, meta)

    # Parity: fixed noise and pasts through torch ArFNN and numpy.
    gen = common.generative(gen_module)
    rng = np.random.default_rng(common.SEED)
    z = rng.standard_normal((256, 63, d))
    pick = rng.integers(0, len(x_real), 256)
    x_past = x_real[pick, :P].double().cpu().numpy()
    with torch.no_grad():
        torch_out = generator(
            torch.tensor(z, dtype=torch.float32, device=dev),
            torch.tensor(x_past, dtype=torch.float32, device=dev),
        )
    numpy_out = gen.sigcwgan_steps(weights, z, x_past)
    diff = float(np.abs(torch_out.cpu().double().numpy() - numpy_out).max())
    print(f"parity max abs diff (63 steps): {diff:.3e}", flush=True)

    # Diagnostics: 252-day paths from 4000 random real pasts (unconditional mix).
    all_scaled = scaled[0].double().numpy()
    starts = rng.integers(P, len(all_scaled), 4000)
    pasts = np.stack([all_scaled[s - P : s] for s in starts])
    steps = gen.sigcwgan_steps(weights, rng.standard_normal((4000, 252, d)), pasts)
    fake = steps * std + mean
    model = gen.load(out)
    default = model.log_paths(4000, 252, np.random.default_rng(common.SEED))
    common.update_meta(
        out,
        parity={"max_abs_diff": diff, "samples": 256, "steps": 63},
        diagnostics=common.diagnostics(
            fake, rets[: train_end + 1], rets[train_end + 1 :], (21, 63, 252)
        ),
        diagnostics_last_past=common.diagnostics(
            default, rets[: train_end + 1], rets[train_end + 1 :], (21, 63, 252)
        ),
        loss_curve=[float(v) for v in loss[::20]],
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
