#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-outputs/v14_four_scheme_random50}"
TRAIN_DIR="${TRAIN_DIR:-outputs/v14_random50_training}"
RANDOM_COUNT="${RANDOM_COUNT:-50}"
MASTER_SEED="${MASTER_SEED:-20260626}"
EXCLUDE_SEEDS="${EXCLUDE_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
GPU_COUNT="${GPU_COUNT:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
TOTAL_SHARDS=$((GPU_COUNT * WORKERS_PER_GPU))
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mkdir -p "$OUT_DIR/logs"

echo "Output: $OUT_DIR"
echo "Training cache: $TRAIN_DIR"
echo "Random count: $RANDOM_COUNT"
echo "GPU count: $GPU_COUNT"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Total shards: $TOTAL_SHARDS"
echo "Starting shards..."

pids=()
cleanup() {
  echo "Stopping shards..."
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}
trap 'cleanup; exit 130' INT TERM

for shard in $(seq 0 $((TOTAL_SHARDS - 1))); do
  gpu=$((shard % GPU_COUNT))
  log="$OUT_DIR/logs/shard${shard}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
  python scripts/run_v14_four_scheme_seed_benchmark.py \
    --random-count "$RANDOM_COUNT" \
    --random-master-seed "$MASTER_SEED" \
    --exclude-seeds "$EXCLUDE_SEEDS" \
    --train-if-missing \
    --shard-count "$TOTAL_SHARDS" \
    --shard-index "$shard" \
    --device cuda:0 \
    --output-dir "$OUT_DIR" \
    --v14-training-dir "$TRAIN_DIR" \
    > "$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "shard ${shard}/${TOTAL_SHARDS}: gpu=${gpu}, pid=${pid}, log=${log}"
done

status=0
for index in "${!pids[@]}"; do
  pid="${pids[$index]}"
  if wait "$pid"; then
    echo "shard ${index} finished"
  else
    code=$?
    echo "shard ${index} failed with exit code ${code}; see $OUT_DIR/logs/shard${index}.log"
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one shard failed; not merging."
  exit "$status"
fi

python scripts/merge_v14_four_scheme_seed_benchmark.py --output-dir "$OUT_DIR"
echo "Done. Summary: $OUT_DIR/method_summary.csv"
