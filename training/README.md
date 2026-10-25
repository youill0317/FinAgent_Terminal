# Scenario generators: training (Colab only)

The app never trains anything. These scripts train three generators on one
Colab T4, one after another, and export numpy-readable weights that
`src/finagent/generative.py` runs locally without torch.

| File | Runs on | What it does |
|---|---|---|
| `export_dataset.py` | local (`finagent.colab_jobs.export`) | `data/portfolio/adjusted.parquet` → `returns.npz`: daily log returns of AAPL, MSFT, PGR, ^GSPC, ^KS11, KRW=X on the days all six have a close |
| `setup_colab.sh`, `requirements-colab.txt` | Colab | packages and the pinned SigCWGAN repo |
| `run_all.sh` | Colab | `train_tailgan.py`, `train_sigcwgan.py`, `train_nsde.py` in turn (`MODELS` picks some); `SMOKE=1` runs a few steps of each |
| `common.py` | Colab | data split (chronological 80%), seed 42, export, parity, diagnostics |

Each training script documents its paper, reference repo and commit, and every
setting it changes from the reference defaults.

## Run

```sh
uv run python training/export_dataset.py /path/to/returns.npz
colab new -s finagent-gen --gpu T4
# upload returns.npz, training/*.py, training/*.sh, training/requirements-colab.txt
# and src/finagent/generative.py to /content/gen/
#   (the parity check imports that exact module)
bash /content/gen/setup_colab.sh
OUT=/content/out bash /content/gen/run_all.sh   # launch in the background
# download /content/out/<method>/{weights.npz,meta.json,log.txt}
# into data/models/<method>/
colab stop -s finagent-gen
```

## Weight contract (`data/models/<method>/`)

- `weights.npz`: the generator's torch `state_dict`, float32, keys unchanged
  (SigCWGAN adds `past`, its last `p` standardised days).
- `meta.json`: `format` (1), `method` (`tailgan`, `sigcwgan`, `nsde`),
  `tickers`, `train_start`/`train_end`, `window` (null: any horizon),
  `normalisation` (`mean`, `std` of log returns), `architecture`,
  `training`, `source` (repo, commit), `versions`, `parity` (torch vs numpy
  max abs diff on fixed noise), `diagnostics`.

## Kronos (not trained here)

Drop these into `data/models/kronos/` to enable it (status "구현 중" until
then):

- `tokenizer/config.json`, `tokenizer/model.safetensors`: a snapshot of
  `NeoQuasar/Kronos-Tokenizer-base`;
- `model/config.json`, `model/model.safetensors`: `NeoQuasar/Kronos-small`;
- `src/model/{__init__,kronos,module}.py`: the `model` package of
  shiyu-coder/Kronos at commit 67b630e67f6a18c9e9be918d9b4337c960db1e9a (the
  repo is not pip-installable).

It also needs torch, einops, huggingface_hub, safetensors and tqdm installed
in the app's environment; torch is imported only when all the files are
there.
