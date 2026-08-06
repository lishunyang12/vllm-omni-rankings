#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SESSION_ROOT=${SESSION_ROOT:-/home/scratch.lbo_other/vllm-omni-5771/minimax_h3_trtllm_e2e/b300_overnight_$(date -u +%Y%m%dT%H%M%SZ)}
MODEL_DIR=${MODEL_DIR:-/home/scratch.lbo_other/MiniMax-H3/FL2VA}
NUM_RUNS=${NUM_RUNS:-6}
VIDEO_RUNS=${VIDEO_RUNS:-2}
G0_CONTAINER=${G0_CONTAINER:-vllm-omni-h3-sage-devel-lbo}
G1_CONTAINER=${G1_CONTAINER:-vllm-omni-h3-sage-skip-devel-lbo}
FA4_G0_CONTAINER=${FA4_G0_CONTAINER:-vllm-omni-h3-fa4-devel-lbo}
FA4_G1_CONTAINER=${FA4_G1_CONTAINER:-${G1_CONTAINER}}
SINGLE_CONTAINER=${SINGLE_CONTAINER:-}
SINGLE_FA4_CONTAINER=${SINGLE_FA4_CONTAINER:-${SINGLE_CONTAINER}}
THERMAL_GPU_INDICES=${THERMAL_GPU_INDICES:-}
CONTAINER_SCRIPT=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/run_attention_ab.py
FA4_SCRIPT=/tmp/minimax_h3_run_attention_ab.py

if [[ -n ${SINGLE_CONTAINER} ]]; then
    G1_CONTAINER=${SINGLE_CONTAINER}
    FA4_G1_CONTAINER=${SINGLE_FA4_CONTAINER}
    FA4_SCRIPT=${CONTAINER_SCRIPT}
fi

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

prepare_output() {
    local output=$1
    if [[ -f "${output}/summary.json" ]] && [[ $(cat "${output}/exit_code" 2>/dev/null) == 0 ]]; then
        return 1
    fi
    if [[ -d "${output}" ]]; then
        mv "${output}" "${output}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    mkdir -p "${output}"
}

run_mode() {
    local container=$1
    local label=$2
    local mode=$3
    local script_path=${4:-${CONTAINER_SCRIPT}}
    local requested_runs=${5:-${NUM_RUNS}}
    local requested_video_runs=${6:-${VIDEO_RUNS}}
    local output=${SESSION_ROOT}/${label}
    if ! prepare_output "${output}"; then
        log "SKIP complete ${label}"
        return 0
    fi
    log "START ${label} mode=${mode} container=${container}"
    set +e
    docker exec \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e MODE="${mode}" \
        -e MODEL_DIR="${MODEL_DIR}" \
        -e OUTPUT_ROOT="${output}" \
        -e NUM_RUNS="${requested_runs}" \
        -e VIDEO_RUNS="${requested_video_runs}" \
        -e PYTHONPATH=/workspace/vllm-omni \
        "${container}" python3 "${script_path}" >"${output}/run.log" 2>&1
    local code=$?
    set -e
    printf '%s\n' "${code}" >"${output}/exit_code"
    if ((code != 0)); then
        log "FAIL ${label} exit=${code}"
        tail -80 "${output}/run.log"
        return "${code}"
    fi
    log "DONE ${label}"
}

run_pair_queue() {
    local container=$1
    shift
    local suffix
    for suffix in "$@"; do
        run_mode "${container}" "final/skip_softmax_${suffix}" "skip_softmax_${suffix}"
        run_mode "${container}" "final/sage_fp8_skip_${suffix}" "sage_fp8_skip_${suffix}"
    done
}

if [[ -n ${SINGLE_CONTAINER} ]]; then
    containers=("${G1_CONTAINER}" "${FA4_G1_CONTAINER}")
else
    containers=("${G0_CONTAINER}" "${G1_CONTAINER}" "${FA4_G0_CONTAINER}" "${FA4_G1_CONTAINER}")
fi
for container in "${containers[@]}"; do
    docker inspect "${container}" >/dev/null
done
if [[ -z ${SINGLE_CONTAINER} ]]; then
    docker cp "${SCRIPT_DIR}/run_attention_ab.py" "${FA4_G0_CONTAINER}:${FA4_SCRIPT}"
    docker cp "${SCRIPT_DIR}/run_attention_ab.py" "${FA4_G1_CONTAINER}:${FA4_SCRIPT}"
fi

printf '%s\n' \
    '{' \
    "  \"model\": \"${MODEL_DIR}\"," \
    "  \"num_runs\": ${NUM_RUNS}," \
    "  \"video_runs\": \"${VIDEO_RUNS}\"," \
    "  \"single_container\": \"${SINGLE_CONTAINER}\"," \
    "  \"physical_gpu_indices\": \"${THERMAL_GPU_INDICES}\"," \
    '  "gates": [0.90, 0.95, 0.99],' \
    '  "thresholds": [0.05, 0.10, 0.30, 0.50],' \
    '  "sage": [false, true],' \
    '  "controls": ["recipe_flash", "trtllm_dense", "sage_fp8"]' \
    '}' >"${SESSION_ROOT}/run_config.json"

if [[ -n ${THERMAL_GPU_INDICES} ]]; then
    nvidia-smi \
        --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,clocks.sm,power.draw,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons_counters.sw_thermal_slowdown \
        --format=csv,noheader,nounits \
        --loop-ms=5000 >>"${SESSION_ROOT}/gpu_telemetry.csv" 2>&1 &
    telemetry_pid=$!
fi

if [[ -n ${SINGLE_CONTAINER} ]]; then
    log "PREFLIGHT isolated mixed GPU group"
    run_mode "${G1_CONTAINER}" preflight/mixed trtllm_dense "${CONTAINER_SCRIPT}" 2 none
    preflight=$(python3 "${SCRIPT_DIR}/select_single_preflight.py" \
        --group g1 \
        --summary "${SESSION_ROOT}/preflight/mixed/summary.json" \
        --output "${SESSION_ROOT}/preflight.json")
    log "PREFLIGHT selection=${preflight}"
    if [[ ${preflight} == none ]]; then
        log "ABORT mixed GPU group did not meet the dense preflight performance threshold"
        exit 2
    fi
    log "QUALIFY isolated mixed GPU group"
    run_mode "${G1_CONTAINER}" qualification/isolated_g1 trtllm_dense
    decision=$(python3 "${SCRIPT_DIR}/summarize_single_group.py" \
        --group g1 \
        --summary "${SESSION_ROOT}/qualification/isolated_g1/summary.json" \
        --output "${SESSION_ROOT}/qualification.json")
    if [[ -n ${THERMAL_GPU_INDICES} ]]; then
        thermal_decision=$(python3 "${SCRIPT_DIR}/check_thermal_telemetry.py" \
            "${SESSION_ROOT}/gpu_telemetry.csv" \
            --gpus "${THERMAL_GPU_INDICES}" \
            --output "${SESSION_ROOT}/qualification_thermal.json")
        log "QUALIFICATION thermal_decision=${thermal_decision}"
        if [[ ${thermal_decision} != pass ]]; then
            decision=abort
        fi
    fi
else
    log "PREFLIGHT isolated GPU group 0"
    run_mode "${G0_CONTAINER}" preflight/g0 trtllm_dense "${CONTAINER_SCRIPT}" 2 none
    log "PREFLIGHT isolated GPU group 1"
    run_mode "${G1_CONTAINER}" preflight/g1 trtllm_dense "${CONTAINER_SCRIPT}" 2 none
    preflight=$(python3 "${SCRIPT_DIR}/select_b300_preflight.py" \
        --g0 "${SESSION_ROOT}/preflight/g0/summary.json" \
        --g1 "${SESSION_ROOT}/preflight/g1/summary.json" \
        --output "${SESSION_ROOT}/preflight.json")
    log "PREFLIGHT selection=${preflight}"

    if [[ ${preflight} == none ]]; then
        log "ABORT no GPU group met the dense preflight performance threshold"
        exit 2
    elif [[ ${preflight} == both ]]; then
        log "QUALIFY isolated GPU group 0"
        run_mode "${G0_CONTAINER}" qualification/isolated_g0 trtllm_dense
        log "QUALIFY isolated GPU group 1"
        run_mode "${G1_CONTAINER}" qualification/isolated_g1 trtllm_dense
        log "QUALIFY concurrent A/A"
        run_mode "${G0_CONTAINER}" qualification/concurrent_g0 trtllm_dense &
        pid0=$!
        run_mode "${G1_CONTAINER}" qualification/concurrent_g1 trtllm_dense &
        pid1=$!
        wait "${pid0}"
        wait "${pid1}"
        decision=$(python3 "${SCRIPT_DIR}/summarize_timing.py" \
            --isolated-g0 "${SESSION_ROOT}/qualification/isolated_g0/summary.json" \
            --isolated-g1 "${SESSION_ROOT}/qualification/isolated_g1/summary.json" \
            --concurrent-g0 "${SESSION_ROOT}/qualification/concurrent_g0/summary.json" \
            --concurrent-g1 "${SESSION_ROOT}/qualification/concurrent_g1/summary.json" \
            --output "${SESSION_ROOT}/qualification.json")
    else
        if [[ ${preflight} == g0 ]]; then
            qualification_container=${G0_CONTAINER}
        else
            qualification_container=${G1_CONTAINER}
        fi
        log "QUALIFY isolated GPU group ${preflight#g}"
        run_mode "${qualification_container}" "qualification/isolated_${preflight}" trtllm_dense
        decision=$(python3 "${SCRIPT_DIR}/summarize_single_group.py" \
            --group "${preflight}" \
            --summary "${SESSION_ROOT}/qualification/isolated_${preflight}/summary.json" \
            --output "${SESSION_ROOT}/qualification.json")
    fi
fi
log "QUALIFICATION decision=${decision}"

if [[ ${decision} == abort ]]; then
    log "ABORT no isolated GPU group met timing stability criteria"
    exit 2
fi

queue0=(005_gate090 010_gate090 03_gate090 05_gate090 005_gate095 010_gate095)
queue1=(03_gate095 05_gate095 005_gate099 010_gate099 03_gate099 05_gate099)

if [[ ${decision} == parallel ]]; then
    (
        run_mode "${FA4_G0_CONTAINER}" final/recipe_flash recipe_flash "${FA4_SCRIPT}"
        run_mode "${G0_CONTAINER}" final/sage_fp8_g0 sage_fp8
        run_pair_queue "${G0_CONTAINER}" "${queue0[@]}"
    ) &
    final_pid0=$!
    (
        run_mode "${G1_CONTAINER}" final/sage_fp8_g1 sage_fp8
        run_pair_queue "${G1_CONTAINER}" "${queue1[@]}"
    ) &
    final_pid1=$!
    wait "${final_pid0}"
    wait "${final_pid1}"
else
    if [[ ${decision} == serial_g0 ]]; then
        serial_container=${G0_CONTAINER}
        serial_fa4_container=${FA4_G0_CONTAINER}
    else
        serial_container=${G1_CONTAINER}
        serial_fa4_container=${FA4_G1_CONTAINER}
    fi
    run_mode "${serial_fa4_container}" final/recipe_flash recipe_flash "${FA4_SCRIPT}"
    run_mode "${serial_container}" final/sage_fp8 sage_fp8
    run_pair_queue "${serial_container}" "${queue0[@]}" "${queue1[@]}"
fi

find "${SESSION_ROOT}" -name summary.json -print | sort >"${SESSION_ROOT}/summary_files.txt"
log "COMPLETE session=${SESSION_ROOT}"
