#!/usr/bin/env bash
# On the Colab VM: packages and the pinned reference repos. Idempotent.
set -euo pipefail
cd /content/gen
pip install -q -r requirements-colab.txt
pip install -q --no-build-isolation iisignature==0.24 \
  "git+https://github.com/crispitagorico/sigkernel@40a583155ea8d2194af0e90dddab37e2659cfcfd"
mkdir -p /content/refs
if [ ! -d /content/refs/sigcwgan ]; then
  git clone -q https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs /content/refs/sigcwgan
  git -C /content/refs/sigcwgan checkout -q 32ea56bd4abc6b934cf754279c3326ac3fae65e4
fi
python -c "import sigkernel, torchsde, iisignature; print('setup ok')"
