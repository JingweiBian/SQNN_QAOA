#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
CONDA_ENV=${CONDA_ENV:-}
GPUS=${GPUS:-0}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
MODELS=${MODELS:-"v10 v14"}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/scale_v10_v14_maxcut3_cuda_shards}
CLASSICAL_TIME_LIMIT_SECONDS=${CLASSICAL_TIME_LIMIT_SECONDS:-3600}
MIN_N=${MIN_N:-512}
MAX_N=${MAX_N:-16384}
MAX_AUTO_N=${MAX_AUTO_N:-65536}

# For upper-bound hunting on A100-class GPUs, prefer MIN_N=4096 or 8192.
# The n=512/1024 V10 trials are useful for continuity but too small to keep
# large GPUs busy.

if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=("$PYTHON")
fi

read -r -a GPU_LIST <<< "$GPUS"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a MODEL_LIST <<< "$MODELS"

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "GPUS is empty" >&2
  exit 1
fi
if (( WORKERS_PER_GPU < 1 )); then
  echo "WORKERS_PER_GPU must be >= 1" >&2
  exit 1
fi

EXPANDED_GPU_LIST=()
for ((replica = 0; replica < WORKERS_PER_GPU; replica++)); do
  for gpu in "${GPU_LIST[@]}"; do
    EXPANDED_GPU_LIST+=("$gpu")
  done
done

mkdir -p "$OUTPUT_ROOT/logs"

pids=()
for worker in "${!EXPANDED_GPU_LIST[@]}"; do
  gpu="${EXPANDED_GPU_LIST[$worker]}"
  shard_seeds=()
  for seed_index in "${!SEED_LIST[@]}"; do
    if (( seed_index % ${#EXPANDED_GPU_LIST[@]} == worker )); then
      shard_seeds+=("${SEED_LIST[$seed_index]}")
    fi
  done
  if [[ "${#shard_seeds[@]}" -eq 0 ]]; then
    continue
  fi

  shard_dir="$OUTPUT_ROOT/shard_${worker}_gpu_${gpu}"
  log_path="$OUTPUT_ROOT/logs/shard_${worker}_gpu_${gpu}.log"
  echo "starting shard $worker on physical GPU $gpu with seeds: ${shard_seeds[*]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "${PYTHON_CMD[@]}" classical/scale_v10_v14_maxcut3.py \
      --min-n "$MIN_N" \
      --max-n "$MAX_N" \
      --size-mode doubling \
      --seeds "${shard_seeds[@]}" \
      --models "${MODEL_LIST[@]}" \
      --device cuda \
      --output-dir "$shard_dir" \
      --cuda-high-throughput \
      --cuda-saturate \
      --auto-extend \
      --max-auto-n "$MAX_AUTO_N" \
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
  ) >"$log_path" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
