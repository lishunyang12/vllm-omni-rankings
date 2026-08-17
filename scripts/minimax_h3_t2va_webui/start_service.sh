#!/usr/bin/env bash
set -euo pipefail

: "${H3_UI_TOKEN:?Set H3_UI_TOKEN before starting the service}"

script_dir=/home/zjy/code/lsy/worktree/vllm-omni-rankings-minimax-rtx5000/scripts/minimax_h3_t2va_webui
runtime_dir=/home/zjy/code/lsy/runtime/minimax_h3_t2va_webui
source_repo=/home/zjy/code/lsy/worktree/minimax-h3-svdquant
python=/home/zjy/code/lsy/vllm-omni/.venv/bin/python
vllm=/home/zjy/code/lsy/vllm-omni/.venv/bin/vllm
model=${H3_MODEL_PATH:-/home/zjy/code/lsy/bench_inputs/MiniMax-H3/FL2VA}
backend_port=${H3_BACKEND_PORT:-18091}
ui_port=${H3_UI_PORT:-8094}

mkdir -p "$runtime_dir/storage"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export VLLM_OMNI_SERVER_STORAGE__PATH="$runtime_dir/storage"
export PYTHONPATH="$source_repo"
export H3_BACKEND_URL="http://127.0.0.1:${backend_port}"

backend_pid=
ui_pid=
cleanup() {
  if [[ -n "$ui_pid" ]]; then kill "$ui_pid" 2>/dev/null || true; fi
  if [[ -n "$backend_pid" ]]; then kill "$backend_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

"$vllm" serve "$model" \
  --omni \
  --host 127.0.0.1 \
  --port "$backend_port" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend TRTLLM_ATTN \
  >"$runtime_dir/backend.log" 2>&1 &
backend_pid=$!

ready=0
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${backend_port}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    tail -100 "$runtime_dir/backend.log"
    exit 1
  fi
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  echo "MiniMax-H3 backend did not become ready" >&2
  exit 1
fi

"$python" -m uvicorn app:app \
  --app-dir "$script_dir" \
  --host 0.0.0.0 \
  --port "$ui_port" \
  >"$runtime_dir/ui.log" 2>&1 &
ui_pid=$!

# Exit as soon as either process stops; the EXIT trap terminates its peer.
set +e
wait -n "$backend_pid" "$ui_pid"
status=$?
set -e
exit "$status"
