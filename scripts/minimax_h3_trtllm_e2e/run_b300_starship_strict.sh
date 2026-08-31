#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SESSION_ROOT=${SESSION_ROOT:-/home/scratch.lbo_other/vllm-omni-5771/minimax_h3_trtllm_e2e/b300_starship_strict_$(date -u +%Y%m%dT%H%M%SZ)}
MODEL_DIR=${MODEL_DIR:-/home/scratch.lbo_other/MiniMax-H3/FL2VA}
PROMPT_FILE=${PROMPT_FILE:-${SCRIPT_DIR}/prompts/minimax_h3_official_starship.txt}
CONTAINER=${CONTAINER:-vllm-omni-h3-mixed-devel-lbo}
CONTAINER_SCRIPT=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/run_attention_ab.py
CONTAINER_PROMPT=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/prompts/minimax_h3_official_starship.txt
GPU_INDICES=${GPU_INDICES:-0,1,4,5}

modes=(
    trtllm_dense
    sage_fp8
    skip_softmax_005_gate097
    sage_fp8_skip_005_gate097
)

mkdir -p "${SESSION_ROOT}"
exec > >(tee -a "${SESSION_ROOT}/runner.log") 2>&1

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

telemetry_pid=
stop_telemetry() {
    if [[ -n ${telemetry_pid} ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
        kill "${telemetry_pid}" 2>/dev/null || true
        wait "${telemetry_pid}" 2>/dev/null || true
    fi
}
trap stop_telemetry EXIT

test -f "${PROMPT_FILE}"
docker inspect "${CONTAINER}" >/dev/null

nvidia-smi \
    --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,clocks.sm,power.draw,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons_counters.sw_thermal_slowdown \
    --format=csv,noheader,nounits \
    --loop-ms=5000 >>"${SESSION_ROOT}/gpu_telemetry.csv" 2>&1 &
telemetry_pid=$!

for mode in "${modes[@]}"; do
    output=${SESSION_ROOT}/${mode}
    mkdir -p "${output}"
    log "START mode=${mode} container=${CONTAINER}"
    set +e
    docker exec \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e MODE="${mode}" \
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
        -e PYTHONPATH=/workspace/vllm-omni \
        "${CONTAINER}" python3 "${CONTAINER_SCRIPT}" >"${output}/run.log" 2>&1
    code=$?
    set -e
    printf '%s\n' "${code}" >"${output}/exit_code"
    if ((code != 0)); then
        log "FAIL mode=${mode} exit=${code}"
        tail -100 "${output}/run.log"
        exit "${code}"
    fi
    log "DONE mode=${mode}"
done

stop_telemetry
telemetry_pid=

docker exec "${CONTAINER}" python3 \
    /workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/check_thermal_telemetry.py \
    "${SESSION_ROOT}/gpu_telemetry.csv" \
    --gpus "${GPU_INDICES}" \
    --output "${SESSION_ROOT}/thermal_audit.json"

log "COMPLETE session=${SESSION_ROOT}"
