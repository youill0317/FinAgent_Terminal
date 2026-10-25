"""Train Tail-GAN on Colab (one GPU) and export it for ``finagent.generative``.

    python train_tailgan.py RETURNS.npz OUT_DIR GENERATIVE.py

Reference: Cont, Cucuringu, Xu and Zhang, "Tail-GAN: Learning to Simulate
Tail Risk Scenarios" (Management Science), code chaozhang-ox/Tail-GAN at
commit bfbe9501a593d7797b2cfb80de1f20c78d99cda3. ``TailGAN.py`` parses
arguments and builds its paths at import, so the parts used here are copied
from it (Generator, Discriminator, the FZ "quant" score, the training step),
from ``util.py`` (deterministic NeuralSort), ``Transform.py`` (Inc2Price,
StaticPort, BuyHold) and ``gen_static_port.py`` (LShort portfolios).

Settings are the repo's defaults (argparse) except:

- ``n_rows`` = 6 assets and ``n_cols`` = 21 days (one month; the repo uses
  100 intraday steps); overlapping windows, stride 1, of the train split;
- strategies ``['Port']``: buy-and-hold of each asset and 50 random
  long-short static portfolios. The repo's mean-reversion and
  trend-following strategies need its CSV/threshold pipeline and 10/20-day
  moving averages, which barely fit a 21-day window, so they are left out;
- ``numNN`` = 1 model, not an ensemble of 10, and seed 42;
- 3000 epochs as in the repo, which is 3 batches of 1000 per epoch here
  (9000 steps; the repo's 50 000 samples give 150 000);
- ``drop_last``: the discriminator's first layer is Linear(batch_size, 256).
"""

# ruff: noqa: T201, INP001, D107, N802, N803, N806, PLR2004, S101

from __future__ import annotations

import sys
import time
from pathlib import Path

import common
import numpy as np
import torch
from scipy import sparse, stats
from torch import nn

REPO = "chaozhang-ox/Tail-GAN"
COMMIT = "bfbe9501a593d7797b2cfb80de1f20c78d99cda3"

N_EPOCHS = 2 if common.SMOKE else 3000
BATCH = 1000
LR_D = 1e-7
LR_G = 1e-6
TEMP = 0.01
B1, B2 = 0.5, 0.999
LATENT = 1000
N_COLS = 21
N_TRANS = 50
CAP = 10
ALPHAS = (0.05,)
W = 10.0
DOF = 5


# ------------------------------------------------- util.py / Transform.py


def deterministic_NeuralSort(s: torch.Tensor, tau: float) -> torch.Tensor:
    """s: batch_size x n x 1; tau: temperature (util.py)."""
    n = s.size()[1]
    one = torch.ones((n, 1), device=s.device)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
    scaling = (n + 1 - 2 * (torch.arange(n, device=s.device) + 1)).float()
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    return torch.softmax(P_max / tau, -1)


def Inc2Price(data: torch.Tensor) -> torch.Tensor:
    price0 = torch.ones(data.shape[0], data.shape[1], 1, device=data.device)
    return torch.cumsum(torch.cat((price0, data), dim=2), dim=2)


def StaticPort(prices_l: torch.Tensor, trans_mat: torch.Tensor) -> torch.Tensor:
    swap_prices = prices_l.permute(0, 2, 1)
    broad = trans_mat.reshape(1, *trans_mat.shape).repeat(swap_prices.size(0), 1, 1)
    return torch.bmm(swap_prices, broad).permute(0, 2, 1)


def BuyHold(prices_l: torch.Tensor, Cap: float) -> torch.Tensor:
    money = prices_l * Cap
    return money[:, :, -1] - money[:, :, 0]


def is_stock(vec: np.ndarray) -> bool:
    return bool(np.all((np.abs(vec) == 1.0) + (np.abs(vec) == 0.0)))


def gen_static_port(n_stocks: int, n_ports: int) -> np.ndarray:
    """gen_static_port.py, 'LShort', the in-sample matrix."""
    rvs = stats.norm(loc=0, scale=1).rvs
    unscaled = sparse.random(
        n_stocks, max(n_stocks**2, n_ports), density=0.9, data_rvs=rvs
    ).toarray()
    scaled = unscaled / np.abs(unscaled).sum(0)
    position = np.diag(1 / np.ones(n_stocks).dot(np.abs(scaled)))
    trans = np.dot(scaled, position)
    port = trans[:, ~np.apply_along_axis(is_stock, 0, trans)]
    assert port.shape[1] >= 10
    return port[:, :n_ports]


# ------------------------------------------------------------- TailGAN.py


class Generator(nn.Module):
    """MLP with 4 hidden layers (TailGAN.py)."""

    def __init__(self, R_shape: tuple[int, int]) -> None:
        super().__init__()
        self.R_shape = R_shape

        def block(in_feat: int, out_feat: int, normalize: bool = True) -> list:
            layers: list[nn.Module] = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(LATENT, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(R_shape))),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = torch.clamp(self.model(z), min=-1, max=1)
        return img.view(img.shape[0], *self.R_shape)


class Discriminator(nn.Module):
    """MLP on each strategy's sorted P&L to (VaR, ES) per alpha (TailGAN.py)."""

    def __init__(self, trans_mat: torch.Tensor) -> None:
        super().__init__()
        self.W = W
        self.alphas = ALPHAS
        self.trans_mat = trans_mat
        self.model = nn.Sequential(
            nn.Linear(BATCH, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2 * len(ALPHAS)),
        )

    def project_op(self, validity: torch.Tensor) -> torch.Tensor:
        for i, alpha in enumerate(self.alphas):
            v = validity[:, 2 * i].clone()
            e = validity[:, 2 * i + 1].clone()
            indicator = torch.sign(torch.as_tensor(0.5 - alpha))
            low = (self.W * v < e).float()
            high = (self.W * v >= e).float()
            mix = (v + self.W * e) / (1 + self.W**2)
            validity[:, 2 * i] = indicator * (low * v + high * mix)
            validity[:, 2 * i + 1] = indicator * (low * e + high * self.W * mix)
        return validity

    def compute_pnl(self, R: torch.Tensor) -> torch.Tensor:
        prices_l = Inc2Price(R)
        port_prices_l = StaticPort(prices_l, self.trans_mat)
        return torch.cat([BuyHold(prices_l, CAP), BuyHold(port_prices_l, CAP)], dim=1)

    def forward(self, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        PNL = self.compute_pnl(R)
        PNL_transpose = PNL.T
        PNL_s = PNL_transpose.reshape(*PNL_transpose.shape, 1)
        perm_matrix = deterministic_NeuralSort(PNL_s, TEMP)
        PNL_sort = torch.bmm(perm_matrix, PNL_s)
        validity = self.model(PNL_sort.reshape(*PNL_transpose.shape))
        return PNL, self.project_op(validity)


def G1_quant(v: torch.Tensor, W: float = W) -> torch.Tensor:
    return -W * v**2 / 2


def G2_quant(e: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * e


def G2in_quant(e: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * e**2 / 2


def S_quant(
    v: torch.Tensor, e: torch.Tensor, X: torch.Tensor, alpha: float, W: float = W
) -> torch.Tensor:
    """FZ-type joint score of (VaR, ES) for alpha < 0.5 (TailGAN.py)."""
    hit = (v >= X).float()
    rt = (
        (hit - alpha) * (G1_quant(v, W) - G1_quant(X, W))
        + 1.0 / alpha * G2_quant(e, alpha) * hit * (v - X)
        + G2_quant(e, alpha) * (e - v)
        - G2in_quant(e, alpha)
    )
    return torch.mean(rt)


def score(validity: torch.Tensor, PNL: torch.Tensor) -> torch.Tensor:
    loss = torch.zeros((), device=PNL.device)
    for i, alpha in enumerate(ALPHAS):
        loss = loss + S_quant(
            validity[:, [2 * i]], validity[:, [2 * i + 1]], PNL.T, alpha
        )
    return loss


def noise(n: int, dev: torch.device) -> torch.Tensor:
    z = np.random.standard_t(DOF, (n, LATENT))  # noqa: NPY002 - as the repo
    return torch.tensor(z, dtype=torch.float32, device=dev)


def train(real: torch.Tensor, trans_mat: torch.Tensor) -> tuple[Generator, list]:
    """Train_Single from TailGAN.py, without its per-epoch sample dumps."""
    dev = real.device
    generator = Generator((real.shape[1], real.shape[2])).to(dev)
    discriminator = Discriminator(trans_mat).to(dev)
    opt_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(B1, B2))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(B1, B2))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(real),
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
    )
    losses = []
    start = time.time()
    for epoch in range(N_EPOCHS):
        epoch_d, epoch_g = [], []
        for (real_R,) in loader:
            gen_R = generator(noise(real_R.shape[0], dev))

            opt_D.zero_grad()
            PNL, PNL_validity = discriminator(real_R)
            _, gen_validity = discriminator(gen_R)
            loss_D = score(PNL_validity, PNL) - score(gen_validity, PNL)
            loss_D.backward(retain_graph=True)
            opt_D.step()
            epoch_d.append(loss_D.item())

            opt_G.zero_grad()
            _, gen_validity = discriminator(gen_R)
            loss_G = score(gen_validity, PNL)
            loss_G.backward()
            opt_G.step()
            epoch_g.append(loss_G.item())
        losses.append([float(np.mean(epoch_d)), float(np.mean(epoch_g))])
        if epoch % 100 == 0 or epoch == N_EPOCHS - 1:
            print(
                f"[Epoch {epoch}] [D loss: {losses[-1][0]:.4f}] "
                f"[G loss: {losses[-1][1]:.4f}] {time.time() - start:.0f}s",
                flush=True,
            )
    return generator, losses


def main(data: Path, out: Path, gen_module: Path) -> None:
    common.seed_everything()
    dev = common.device()
    rets, tickers, dates = common.load_returns(data)
    cut = common.split(len(rets))
    train_rets, test_rets = rets[:cut], rets[cut:]
    d = rets.shape[1]

    trans_np = gen_static_port(d, N_TRANS)
    trans_mat = torch.tensor(trans_np, dtype=torch.float32, device=dev)
    real = torch.tensor(
        common.windows(train_rets, N_COLS).transpose(0, 2, 1),
        dtype=torch.float32,
        device=dev,
    )  # (windows, assets, days) as the repo's Dataset_IS
    print(f"{len(real)} windows of {N_COLS} days x {d} assets on {dev}", flush=True)

    start = time.time()
    generator, losses = train(real, trans_mat)
    seconds = time.time() - start
    generator.eval()

    meta = {
        "method": "tailgan",
        "tickers": tickers,
        "train_start": dates[0],
        "train_end": dates[cut - 1],
        "test_end": dates[-1],
        "window": N_COLS,
        "returns": "log",
        "architecture": {
            "latent_dim": LATENT,
            "hidden": [128, 256, 512, 1024],
            "batchnorm_eps": 0.8,
            "leaky_relu": 0.2,
            "clamp": 1.0,
            "noise": f"student_t({DOF})",
            "output": "(assets, days) reshaped; numpy transposes to (days, assets)",
        },
        "training": {
            "epochs": N_EPOCHS,
            "steps": N_EPOCHS * (len(real) // BATCH),
            "batch_size": BATCH,
            "lr_D": LR_D,
            "lr_G": LR_G,
            "betas": [B1, B2],
            "temp": TEMP,
            "strategies": ["BuyHold assets", f"{N_TRANS} LShort static portfolios"],
            "cap": CAP,
            "alphas": list(ALPHAS),
            "score": "quant (W=10)",
            "project": True,
            "numNN": 1,
            "windows": len(real),
            "seconds": round(seconds),
            "final_losses_DG": losses[-1],
        },
        "static_portfolios": trans_np.tolist(),
        "source": {"repo": REPO, "commit": COMMIT, "paper": "Cont et al., Tail-GAN"},
    }
    weights = common.save(out, generator.state_dict(), meta)

    # Parity: the same Student-t noise through torch (eval) and numpy.
    gen = common.generative(gen_module)
    z = np.random.default_rng(common.SEED).standard_t(DOF, (256, LATENT))
    with torch.no_grad():
        torch_out = generator(torch.tensor(z, dtype=torch.float32, device=dev))
    torch_np = torch_out.cpu().double().numpy().transpose(0, 2, 1)
    numpy_out = gen.tailgan_windows(weights, z, d)
    diff = float(np.abs(torch_np - numpy_out).max())
    print(f"parity max abs diff: {diff:.3e}", flush=True)

    model = gen.load(out)
    fake = model.log_paths(10_000, 252, np.random.default_rng(common.SEED))
    common.update_meta(
        out,
        parity={"max_abs_diff": diff, "samples": 256, "torch_dtype": "float32"},
        diagnostics=common.diagnostics(fake, train_rets, test_rets, (21, 63, 252)),
        loss_curve=losses[::50],
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
