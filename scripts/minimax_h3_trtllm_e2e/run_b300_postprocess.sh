#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SESSION_ROOT=${SESSION_ROOT:?Set SESSION_ROOT to the completed B300 session}
MODEL_DIR=${MODEL_DIR:-/home/scratch.lbo_other/MiniMax-H3/FL2VA}
NUM_RUNS=${NUM_RUNS:-6}
G0_CONTAINER=${G0_CONTAINER:-vllm-omni-h3-sage-devel-lbo}
G1_CONTAINER=${G1_CONTAINER:-vllm-omni-h3-sage-skip-devel-lbo}
FA4_G0_CONTAINER=${FA4_G0_CONTAINER:-vllm-omni-h3-fa4-devel-lbo}
FA4_G1_CONTAINER=${FA4_G1_CONTAINER:-${G1_CONTAINER}}
CONTAINER_SCRIPT=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/run_attention_ab.py
FA4_SCRIPT=/tmp/minimax_h3_run_attention_ab.py
NSYS_HOST_DIR=/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64
NSYS_CONTAINER_DIR=/home/lbo/nsight-systems/2025.3.2
NSYS_IMPORTER=/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/QdstrmImporter

exec > >(tee -a "${SESSION_ROOT}/postprocess.log") 2>&1

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

runner_pid=${RUNNER_PID:-$(cat "${SESSION_ROOT}/runner.pid")}
while kill -0 "${runner_pid}" 2>/dev/null; do
    log "WAIT runner pid=${runner_pid}"
    sleep 60
done
if ! grep -q "COMPLETE session=${SESSION_ROOT}" "${SESSION_ROOT}/runner.log"; then
    log "STOP primary runner did not complete"
    exit 2
fi

decision=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
    "${SESSION_ROOT}/qualification.json")
case ${decision} in
    parallel|serial_g0)
        trtllm_container=${G0_CONTAINER}
        fa4_container=${FA4_G0_CONTAINER}
        ;;
    serial_g1)
        trtllm_container=${G1_CONTAINER}
        fa4_container=${FA4_G1_CONTAINER}
        ;;
    *) log "STOP invalid qualification decision=${decision}"; exit 2 ;;
esac

run_mode() {
    local mode=$1
    local attempt=$2
    local container=${trtllm_container}
    local script_path=${CONTAINER_SCRIPT}
    if [[ ${mode} == recipe_flash ]]; then
        container=${fa4_container}
        script_path=${FA4_SCRIPT}
    fi
    local output=${SESSION_ROOT}/reruns/${mode}/attempt${attempt}
    mkdir -p "${output}"
    log "RERUN mode=${mode} attempt=${attempt} container=${container}"
    set +e
    docker exec \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e MODE="${mode}" \
        -e MODEL_DIR="${MODEL_DIR}" \
        -e OUTPUT_ROOT="${output}" \
        -e NUM_RUNS="${NUM_RUNS}" \
        -e VIDEO_RUNS=2 \
        -e PYTHONPATH=/workspace/vllm-omni \
        "${container}" python3 "${script_path}" >"${output}/run.log" 2>&1
    local code=$?
    set -e
    printf '%s\n' "${code}" >"${output}/exit_code"
    if ((code != 0)); then
        log "RERUN FAIL mode=${mode} attempt=${attempt} exit=${code}"
    fi
}

audit() {
    python3 "${SCRIPT_DIR}/audit_b300_session.py" "${SESSION_ROOT}" \
        --output "${SESSION_ROOT}/timing_audit.json"
}

audit
for attempt in 1 2; do
    mapfile -t modes < <(python3 -c \
        'import json,sys; sys.stdout.write("\n".join(json.load(open(sys.argv[1]))["needs_rerun"]))' \
        "${SESSION_ROOT}/timing_audit.json")
    if ((${#modes[@]} == 0)); then
        break
    fi
    for mode in "${modes[@]}"; do
        run_mode "${mode}" "${attempt}"
    done
    audit
done

remaining=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["needs_rerun"]))' \
    "${SESSION_ROOT}/timing_audit.json")
if ((remaining != 0)); then
    log "WARNING ${remaining} mode(s) remain unstable after automatic reruns"
fi

log "PREPARE Nsight Systems target in ${trtllm_container}"
docker exec "${trtllm_container}" mkdir -p \
    "${NSYS_CONTAINER_DIR}/target-linux-x64" "${NSYS_CONTAINER_DIR}/bin"
docker cp "${NSYS_HOST_DIR}/." \
    "${trtllm_container}:${NSYS_CONTAINER_DIR}/target-linux-x64/"
docker exec "${trtllm_container}" ln -sfn ../target-linux-x64/nsys \
    "${NSYS_CONTAINER_DIR}/bin/nsys"

profile_mode() {
    local mode=$1
    local output=${SESSION_ROOT}/kernel_profiles/${mode}
    mkdir -p "${output}"
    log "PROFILE mode=${mode}"
    set +e
    docker exec \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e VLLM_OMNI_NSYS_STEADY_CAPTURE=1 \
        -e NSYS_CAPTURE_WARMUP_REQUESTS=1 \
        -e MODE="${mode}" \
        -e MODEL_DIR="${MODEL_DIR}" \
        -e OUTPUT_ROOT="${output}/benchmark" \
        -e NUM_RUNS=2 \
        -e VIDEO_RUNS=2 \
        -e PYTHONPATH=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/nsys_capture:/workspace/vllm-omni \
        "${trtllm_container}" "${NSYS_CONTAINER_DIR}/bin/nsys" profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        --output="${output}/trace" \
        python3 "${CONTAINER_SCRIPT}" >"${output}/run.log" 2>&1
    local code=$?
    set -e
    if ((code != 0)); then
        printf '%s\n' "${code}" >"${output}/exit_code"
        log "PROFILE FAIL mode=${mode} exit=${code}"
        return
    fi
    if [[ ! -f ${output}/trace.nsys-rep && -f ${output}/trace.qdstrm ]]; then
        "${NSYS_IMPORTER}" \
            --force-overwrite \
            --input-file="${output}/trace.qdstrm" \
            --output-file="${output}/trace.nsys-rep" >>"${output}/run.log" 2>&1
    fi
    if [[ ! -f ${output}/trace.nsys-rep ]]; then
        printf '%s\n' 2 >"${output}/exit_code"
        log "PROFILE FAIL mode=${mode} trace was not generated"
        return
    fi
    set +e
    nsys stats \
        --force-export=true \
        --force-overwrite=true \
        --report=cuda_gpu_kern_sum,nvtx_gpu_proj_sum \
        --format=csv \
        --output="${output}/stats" \
        "${output}/trace.nsys-rep" >>"${output}/run.log" 2>&1
    code=$?
    set -e
    printf '%s\n' "${code}" >"${output}/exit_code"
    if ((code != 0)); then
        log "PROFILE FAIL mode=${mode} stats exit=${code}"
        return
    fi
    log "PROFILE DONE mode=${mode}"
}

for mode in trtllm_dense sage_fp8 skip_softmax_05_gate099 sage_fp8_skip_05_gate099; do
    profile_mode "${mode}"
done

log "POSTPROCESS COMPLETE session=${SESSION_ROOT}"
