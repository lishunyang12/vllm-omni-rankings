#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=/home/zjy/code/lsy/results/pr6540-aligned-b300-e2e-20260829T072419Z
WORKTREE=/home/zjy/code/lsy/worktree/minimax-h3-super-acceleration
PYTHON_BIN="$WORKTREE/.venv/bin/python"

export CUDA_VISIBLE_DEVICES=2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400
export VLLM_OMNI_ENGINE_START_TIMEOUT_S=3600
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
    nvidia-smi --id=2,3 \
      --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits | paste -sd ';' -
    sleep 1
  done
}

cd "$WORKTREE"
sample_gpu > "$RUN_ROOT/metrics/gpu-samples.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

"$PYTHON_BIN" "$RUN_ROOT/run_e2e.py"
