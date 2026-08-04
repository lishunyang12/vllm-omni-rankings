#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the local MiniMax-H3 directory}"

MODEL="${MODEL:-${MODEL_ROOT}/FL2VA}"
PORT="${PORT:-8091}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRTLLM_ATTN}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend "${ATTENTION_BACKEND}"
