#!/usr/bin/env bash
# On the Colab VM: the trainings one after another, a log per model.
# MODELS picks some of them (default all three); SMOKE=1 runs a few steps of
# each to check the pipeline.
set -u
cd /content/gen
OUT=${OUT:-/content/out}
mkdir -p "$OUT"
for m in ${MODELS:-tailgan sigcwgan nsde}; do
  mkdir -p "$OUT/$m"
  extra=""
  [ "$m" = sigcwgan ] && extra=/content/refs/sigcwgan
  start=$(date +%s)
  python "train_$m.py" returns.npz "$OUT/$m" generative.py $extra > "$OUT/$m/log.txt" 2>&1
  code=$?
  echo "$m exit=$code seconds=$(( $(date +%s) - start ))" >> "$OUT/status.txt"
done
echo done >> "$OUT/status.txt"
