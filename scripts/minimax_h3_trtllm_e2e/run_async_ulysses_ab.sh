#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SESSION_ROOT=${SESSION_ROOT:-/home/scratch.lbo_other/vllm-omni-async-ulysses/b300_ab_$(date -u +%Y%m%dT%H%M%SZ)}
MODEL_DIR=${MODEL_DIR:-/home/scratch.lbo_other/MiniMax-H3/FL2VA}
PROMPT_FILE=${PROMPT_FILE:-${SCRIPT_DIR}/prompts/minimax_h3_official_starship.txt}
CONTAINER=${CONTAINER:-vllm-omni-h3-mixed-devel-lbo}
CONTAINER_SCRIPT=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/run_attention_ab.py
CONTAINER_PROMPT=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/prompts/minimax_h3_official_starship.txt

mkdir -p "${SESSION_ROOT}"
exec > >(tee -a "${SESSION_ROOT}/runner.log") 2>&1

telemetry_pid=
stop_telemetry() {
    if [[ -n ${telemetry_pid} ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
        kill "${telemetry_pid}" 2>/dev/null || true
        wait "${telemetry_pid}" 2>/dev/null || true
    fi
}
trap stop_telemetry EXIT

nvidia-smi \
    --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,clocks.sm,power.draw,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons_counters.sw_thermal_slowdown \
    --format=csv,noheader,nounits \
    --loop-ms=5000 >"${SESSION_ROOT}/gpu_telemetry.csv" 2>&1 &
telemetry_pid=$!

for label in dense async; do
    if [[ ${label} == async ]]; then
        async_ulysses=1
    else
        async_ulysses=0
    fi
    output=${SESSION_ROOT}/${label}
    mkdir -p "${output}"
    printf '[%s] START %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}"
    docker exec \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e PYTHONPATH=/workspace/vllm-omni-async:/workspace/vllm-omni \
        -e MODE=trtllm_dense \
        -e ASYNC_ULYSSES="${async_ulysses}" \
        -e MODEL_DIR="${MODEL_DIR}" \
        -e OUTPUT_ROOT="${output}" \
        -e PROMPT_FILE="${CONTAINER_PROMPT}" \
        -e HEIGHT=768 \
        -e WIDTH=1344 \
        -e DURATION_SECONDS=10 \
        -e NUM_INFERENCE_STEPS=50 \
        -e NUM_RUNS=6 \
        -e VIDEO_RUNS=2 \
        -e SEED=0 \
        "${CONTAINER}" python3 "${CONTAINER_SCRIPT}" >"${output}/run.log" 2>&1
    printf '[%s] DONE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}"
done

stop_telemetry
telemetry_pid=
printf '[%s] COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SESSION_ROOT}"
