"""Train the signature-kernel neural SDE on Colab and export it for numpy.

    python train_nsde.py RETURNS.npz OUT_DIR GENERATIVE.py

Reference: Issa, Horvath, Lemercier and Salvi, "Non-adversarial training of
Neural SDEs with signature kernel scores" (NeurIPS 2023), code
issaz/sigker-nsdes at commit 3d40b2922db4a7446917bdfd225c9252dd7d01a2, with
``sigkernel`` (crispitagorico/sigkernel, commit
40a583155ea8d2194af0e90dddab37e2659cfcfd) for the signature-kernel MMD.
The repo's ``src.gan`` imports ``signatory`` at module level (built for
torch 1.9), so the generator classes are copied from ``src/gan/base.py`` and
``src/gan/generators.py`` (MLP, LipSwish, ScaledTanh, GeneratorFunc,
Generator) and the loss is ``sigkernel.SigKernel.compute_mmd`` as its
``SigKerMMDDiscriminator`` calls it (non-adversarial).

Settings are those of ``notebooks/unconditional_nsde.ipynb``: paths of 64
points (63 daily returns), batch 128, hidden 16, noise 8, MLPs of 3 layers
of 64 with LipSwish and a final tanh, fixed start x0 = 1, Ito, Euler with
dt = 1 on t = 0..63, Adadelta(lr=1e-3, weight decay 1e-2), 1000 steps,
stochastic weight averaging after step 500 (the averaged generator is the
one exported), drift and diffusion weights x0.5 at start, RBF signature
kernel with sigma 1 and dyadic order 1, max_batch 16, "mean_var"
normalisation of the terminal values, start subtracted, time as a channel.
The notebook's early stopping on marginals is not used (all 1000 steps
run). Data: cumulative log returns over overlapping 63-day windows of the
chronological 80% train split, seed 42. ``torchcde.linear_interpolation_coeffs``
on this regular grid returns the values themselves, so it is left out.
"""

# ruff: noqa: T201, INP001, D107, PLR0913, PLR0915, S101, SLF001

from __future__ import annotations

import sys
import time
from pathlib import Path

import common
import numpy as np
import sigkernel
import torch
import torchsde
from torch.optim import swa_utils

REPO = "issaz/sigker-nsdes"
COMMIT = "3d40b2922db4a7446917bdfd225c9252dd7d01a2"
SIGKERNEL = "crispitagorico/sigkernel@40a583155ea8d2194af0e90dddab37e2659cfcfd"
PATH_LENGTH = 64
BATCH = 128
CONFIG = {
    "initial_noise_size": 5,
    "noise_size": 8,
    "hidden_size": 16,
    "mlp_size": 64,
    "num_layers": 3,
    "activation": "LipSwish",
    "tanh": True,
    "tscale": 1,
    "fixed": True,
    "noise_type": "general",
    "sde_type": "ito",
    "dt_scale": 1e0,
    "integration_method": "euler",
}
SIGKER = {"dyadic_order": 1, "kernel_type": "rbf", "sigma": 1e0, "max_batch": 16}
LR = 1e-3
WEIGHT_DECAY = 1e-2
STEPS = 6 if common.SMOKE else 1000
SWA_START = STEPS // 2
INIT_MULT2 = 5e-1


# ------------------------------------------------------------ src/gan/base.py


class LipSwish(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.909 * torch.nn.functional.silu(x)


class ScaledTanh(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.nn.Tanh()(x)


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        mlp_size: int,
        num_layers: int,
        activation: str = "LipSwish",
        tanh: bool = True,
        tscale: float = 1,
    ) -> None:
        super().__init__()
        act = LipSwish if activation == "LipSwish" else getattr(torch.nn, activation)
        model: list[torch.nn.Module] = [torch.nn.Linear(in_size, mlp_size), act()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(act())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(ScaledTanh(tscale))
        self._model = torch.nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


# ------------------------------------------------------ src/gan/generators.py


class GeneratorFunc(torch.nn.Module):
    def __init__(
        self,
        noise_size: int,
        hidden_size: int,
        mlp_size: int,
        num_layers: int,
        activation: str,
        tanh: bool,
        sde_type: str,
        noise_type: str,
        tscale: float = 1.0,
    ) -> None:
        super().__init__()
        self.sde_type = sde_type
        self.noise_type = noise_type
        self._noise_size = 1 if noise_type == "diagonal" else noise_size
        self._hidden_size = hidden_size
        self._drift = MLP(
            1 + hidden_size, hidden_size, mlp_size, num_layers, activation, tanh, tscale
        )
        self._diffusion = MLP(
            1 + hidden_size,
            hidden_size * self._noise_size,
            mlp_size,
            num_layers,
            activation,
            tanh,
            tscale,
        )

    def _tx(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([t.expand(x.size(0), 1), x], dim=1)

    def f_and_g(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.f(t, x), self.g(t, x)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._drift(self._tx(t, x))

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        diffusion = self._diffusion(self._tx(t, x))
        if self.noise_type == "diagonal":
            return diffusion
        return diffusion.view(x.size(0), self._hidden_size, self._noise_size)


class Generator(torch.nn.Module):
    def __init__(
        self,
        data_size: int,
        initial_noise_size: int,
        noise_size: int,
        hidden_size: int,
        mlp_size: int,
        num_layers: int,
        activation: str,
        tanh: bool = True,
        tscale: float = 1.0,
        fixed: bool = True,
        noise_type: str = "general",
        sde_type: str = "stratonovich",
        integration_method: str = "reversible_heun",
        dt_scale: float = 1,
    ) -> None:
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self._dt_scale = dt_scale
        self._fixed = fixed
        self._initial = MLP(
            initial_noise_size,
            hidden_size,
            mlp_size,
            num_layers,
            activation,
            tanh=False,
        )
        self._func = GeneratorFunc(
            noise_size,
            hidden_size,
            mlp_size,
            num_layers,
            activation,
            tanh,
            sde_type,
            noise_type,
            tscale,
        )
        self._readout = torch.nn.Linear(hidden_size, data_size)
        self.sdeint_args: dict[str, object] = {"method": integration_method}
        self.sdeint_func = torchsde.sdeint  # euler: no adjoint in the notebook

    def forward(self, ts: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self._fixed:
            x0 = torch.full((batch_size, self._hidden_size), 1.0, device=ts.device)
        else:
            x0 = self._initial(
                torch.randn(batch_size, self._initial_noise_size, device=ts.device)
            )
        self.sdeint_args["dt"] = torch.diff(ts)[0] * self._dt_scale
        xs = self.sdeint_func(self._func, x0, ts, **self.sdeint_args).transpose(0, 1)
        ys = self._readout(xs)
        ts_ = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torch.cat([ts_, ys], dim=2)


def subtract_initial_point(paths: torch.Tensor) -> torch.Tensor:
    """src/utils/helper_functions/data_helper_functions.py."""
    _, length, _ = paths.size()
    res = paths.clone()
    start_points = torch.transpose(res[:, 0, 1:].unsqueeze(-1), -1, 1)
    res[..., 1:] -= torch.tile(start_points, (1, length, 1))
    return res


def main(data: Path, out: Path, gen_module: Path) -> None:
    common.seed_everything()
    dev = common.device()
    rets, tickers, dates = common.load_returns(data)
    cut = common.split(len(rets))
    train_rets, test_rets = rets[:cut], rets[cut:]
    d = rets.shape[1]

    # Path bank: time 0..63 and cumulative log returns from 0, stride 1.
    windows = common.windows(train_rets, PATH_LENGTH - 1)
    cumulative = np.concatenate(
        [np.zeros((len(windows), 1, d)), np.cumsum(windows, axis=1)], axis=1
    )
    stds = cumulative[:, -1].std(axis=0, ddof=1)  # torch.std, as get_ys_coeffs
    ts = torch.linspace(0, PATH_LENGTH - 1, PATH_LENGTH, device=dev).float()
    t_channel = np.broadcast_to(
        np.arange(PATH_LENGTH)[None, :, None], (len(windows), PATH_LENGTH, 1)
    )
    means = cumulative[:, -1].mean(axis=0)
    bank = np.concatenate([t_channel, (cumulative - means) / stds], axis=2)
    real = torch.tensor(bank, dtype=torch.float32, device=dev)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(real), batch_size=BATCH, shuffle=True
    )
    infinite = (elem for it in iter(lambda: loader, None) for elem in it)
    print(
        f"{len(real)} paths of {PATH_LENGTH} points x {d} assets on {dev}", flush=True
    )

    generator = Generator(data_size=d, **CONFIG).to(dev)
    with torch.no_grad():
        for _name, prm in generator._func.named_parameters():
            prm.data *= INIT_MULT2  # drift and diffusion alike (init_mult2_dr = _df)
    kernel = sigkernel.SigKernel(
        static_kernel=sigkernel.RBFKernel(sigma=SIGKER["sigma"]),
        dyadic_order=SIGKER["dyadic_order"],
    )
    averaged = swa_utils.AveragedModel(generator)
    optimiser = torch.optim.Adadelta(
        generator.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    optimiser.zero_grad()

    losses: list[float] = []
    start = time.time()
    for step in range(STEPS):
        (real_samples,) = next(infinite)
        generated = generator(ts, BATCH)
        x = subtract_initial_point(generated).double()
        y = subtract_initial_point(real_samples).double()
        loss = kernel.compute_mmd(x, y.detach(), max_batch=SIGKER["max_batch"])
        loss.backward()
        losses.append(float(loss.detach()))
        optimiser.step()
        optimiser.zero_grad()
        if step > SWA_START:
            averaged.update_parameters(generator)
        if step % 100 == 0 or step == STEPS - 1:
            print(
                f"step {step} mmd {losses[-1]:.5f} {time.time() - start:.0f}s",
                flush=True,
            )
    seconds = time.time() - start

    final = averaged.module.eval()
    meta = {
        "method": "nsde",
        "tickers": tickers,
        "train_start": dates[0],
        "train_end": dates[cut - 1],
        "test_end": dates[-1],
        "window": PATH_LENGTH - 1,
        "dt": 1.0,
        "tscale": float(CONFIG["tscale"]),
        "returns": "log",
        # Generated paths are in units of each asset's train std of 63-day
        # cumulative log returns; the start is subtracted, so no mean.
        "normalisation": {"mean": [0.0] * d, "std": stds.tolist()},
        "architecture": {**CONFIG, "path_length": PATH_LENGTH, "x0": "ones"},
        "training": {
            "steps": STEPS,
            "batch_size": BATCH,
            "optimizer": f"Adadelta(lr={LR}, weight_decay={WEIGHT_DECAY})",
            "swa_start": SWA_START,
            "init_mult2": INIT_MULT2,
            "loss": "signature-kernel MMD (non-adversarial)",
            "sigkernel": SIGKER,
            "sigkernel_source": SIGKERNEL,
            "paths": len(real),
            "seconds": round(seconds),
            "mmd_first_last": [losses[0], losses[-1]],
        },
        "source": {
            "repo": REPO,
            "commit": COMMIT,
            "paper": "Issa et al., NeurIPS 2023",
        },
    }
    weights = common.save(out, final.state_dict(), meta)

    # Parity: the same Brownian increments through torchsde and numpy Euler.
    gen = common.generative(gen_module)
    n = 256
    seen: list[torch.Tensor] = []

    class Recorded(torchsde.BrownianInterval):
        """Keeps each increment the solver draws (re-querying can differ)."""

        def __call__(self, ta, tb=None, return_U=False, return_A=False):  # noqa: ANN001, ANN204, N803
            out = super().__call__(ta, tb, return_U, return_A)
            seen.append(out.clone())
            return out

    final.sdeint_args["bm"] = Recorded(
        t0=float(ts[0]),
        t1=float(ts[-1]),
        size=(n, CONFIG["noise_size"]),
        device=dev,
        entropy=common.SEED,
    )
    with torch.no_grad():
        torch_out = subtract_initial_point(final(ts, n))[..., 1:]
    del final.sdeint_args["bm"]
    assert len(seen) == PATH_LENGTH - 1
    dw = torch.stack(seen, dim=1)
    numpy_out = gen.nsde_paths(
        weights, dw.double().cpu().numpy(), 1.0, float(CONFIG["tscale"])
    )
    diff = float(np.abs(torch_out.double().cpu().numpy() - numpy_out).max())
    print(f"parity max abs diff (63 steps): {diff:.3e}", flush=True)

    model = gen.load(out)
    fake = model.log_paths(10_000, 252, np.random.default_rng(common.SEED))
    common.update_meta(
        out,
        parity={"max_abs_diff": diff, "samples": n, "steps": PATH_LENGTH - 1},
        diagnostics=common.diagnostics(fake, train_rets, test_rets, (21, 63, 252)),
        loss_curve=losses[::20],
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
