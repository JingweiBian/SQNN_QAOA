#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
CONDA_ENV=${CONDA_ENV:-}
DEVICE=${DEVICE:-cuda}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/scale_v10_v14_maxcut3_cuda_upper}
CLASSICAL_TIME_LIMIT_SECONDS=${CLASSICAL_TIME_LIMIT_SECONDS:-3600}

if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=("$PYTHON")
fi

"${PYTHON_CMD[@]}" classical/scale_v10_v14_maxcut3.py \
  --min-n 512 \
  --max-n 16384 \
  --size-mode doubling \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --models v10 v14 \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR" \
  --cuda-high-throughput \
  --cuda-saturate \
  --auto-extend \
  --max-auto-n 65536 \
  --upper-bound-baselines b1 \
  --adaptive-refine \
  --refine-step-n 128 \
  --threshold-metric C_d \
  --classical-time-limit-seconds "$CLASSICAL_TIME_LIMIT_SECONDS" \
  --gw-rank 64 \
  --gw-steps 1200 \
  --gw-restarts 2 \
  --gw-rounding-samples 4096 \
  --gw-rounding-batch-size 2048 \
  --random-flip-samples 1024 \
  --random-flip-batch-size 256 \
  --greedy-restarts 32 \
  --greedy-passes 220 \
  --sample-count 256 \
  --v10-rounds 100 \
  --v10-epochs 200 \
  --v10-symmetry-trials 4 \
  --v14-rounds 280 \
  --v14-epochs 110 \
  --v14-head-count 1
