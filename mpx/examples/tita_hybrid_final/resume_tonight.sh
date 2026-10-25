#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jacopo/Desktop/repo_rl
PY=/home/jacopo/miniconda3/envs/mjpl/bin/python
ARCHIVE=tita_rl/test/mpx/mpx/examples/tita_hybrid_final
TODAY=$(TZ=Europe/Rome date +%F)
TARGET=$(TZ=Europe/Rome date -d "$TODAY 23:00:00" +%s)
while (( $(date +%s) < TARGET )); do sleep 15; done
cd "$ROOT"
"$PY" "$ARCHIVE/build_archive.py" --simulate --terrains rough,perlin,slope10 \
  >> "$ARCHIVE/run.log" 2>&1
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "$PY" "$ARCHIVE/build_archive.py" \
  --render --terrains flat,rough,perlin,slope10 \
  >> "$ARCHIVE/run.log" 2>&1
