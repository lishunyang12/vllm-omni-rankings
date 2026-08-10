#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SESSION_ROOT=${SESSION_ROOT:-/home/scratch.lbo_other/vllm-omni-async-ulysses/b300_nsys_ab_$(date -u +%Y%m%dT%H%M%SZ)}
MODEL_DIR=${MODEL_DIR:-/home/scratch.lbo_other/MiniMax-H3/FL2VA}
PROMPT_FILE=${PROMPT_FILE:-${SCRIPT_DIR}/prompts/minimax_h3_official_starship.txt}
CONTAINER=${CONTAINER:-vllm-omni-h3-mixed-devel-lbo}

CONTAINER_SCRIPT=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/run_attention_ab.py
CONTAINER_PROMPT=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/prompts/minimax_h3_official_starship.txt
CAPTURE_PYTHONPATH=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/nsys_capture:/workspace/vllm-omni-async:/workspace/vllm-omni
NSYS_CONTAINER=/home/lbo/nsight-systems/2025.3.2/bin/nsys
NSYS_HOST=${NSYS_HOST:-/usr/local/bin/nsys}
NSYS_IMPORTER=${NSYS_IMPORTER:-/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/QdstrmImporter}

mkdir -p "${SESSION_ROOT}"
exec > >(tee -a "${SESSION_ROOT}/runner.log") 2>&1

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

test -f "${PROMPT_FILE}"
test -x "${NSYS_HOST}"
test -x "${NSYS_IMPORTER}"
docker inspect "${CONTAINER}" >/dev/null
docker exec "${CONTAINER}" test -x "${NSYS_CONTAINER}"
docker exec "${CONTAINER}" test -f \
    /workspace/vllm-omni-async/vllm_omni/diffusion/attention/parallel/async_ulysses.py

# Do not profile over a timing run in the shared stable container.
if docker top "${CONTAINER}" -eo pid,args | grep -q '[r]un_attention_ab.py'; then
    log "STOP ${CONTAINER} already has a MiniMax-H3 A/B run in progress"
    exit 2
fi

profile_mode() {
    local label=$1
    local async_ulysses=$2
    local output=${SESSION_ROOT}/${label}
    local trace=${output}/trace.nsys-rep
    mkdir -p "${output}/benchmark"

    log "PROFILE START label=${label} async_ulysses=${async_ulysses}"
    set +e
    docker exec \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e VLLM_OMNI_NSYS_STEADY_CAPTURE=1 \
        -e NSYS_CAPTURE_WARMUP_REQUESTS=1 \
        -e MODE=trtllm_dense \
        -e ASYNC_ULYSSES="${async_ulysses}" \
        -e MODEL_DIR="${MODEL_DIR}" \
        -e OUTPUT_ROOT="${output}/benchmark" \
        -e PROMPT_FILE="${CONTAINER_PROMPT}" \
        -e HEIGHT=768 \
        -e WIDTH=1344 \
        -e DURATION_SECONDS=10 \
        -e NUM_INFERENCE_STEPS=50 \
        -e NUM_RUNS=2 \
        -e VIDEO_RUNS=none \
        -e SEED=0 \
        -e PYTHONPATH="${CAPTURE_PYTHONPATH}" \
        "${CONTAINER}" "${NSYS_CONTAINER}" profile \
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
    printf '%s\n' "${code}" >"${output}/exit_code"
    if ((code != 0)); then
        log "PROFILE FAIL label=${label} exit=${code}"
        tail -100 "${output}/run.log"
        return "${code}"
    fi

    if [[ ! -f ${trace} && -f ${output}/trace.qdstrm ]]; then
        "${NSYS_IMPORTER}" \
            --force-overwrite \
            --input-file="${output}/trace.qdstrm" \
            --output-file="${trace}" >>"${output}/run.log" 2>&1
    fi
    test -f "${trace}"
    grep -q 'NSYS_CAPTURE_START .*request=1' "${output}/run.log"
    grep -q 'NSYS_CAPTURE_STOP .*request=1' "${output}/run.log"

    # The .nsys-rep is the source of truth for cross-stream overlap. The
    # summaries compare kernel/NCCL cost and CE-copy volume, while gpu_trace
    # preserves timestamp, device, and stream columns for scripted inspection.
    "${NSYS_HOST}" stats \
        --force-export=true \
        --force-overwrite=true \
        --report=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_gpu_trace,nvtx_gpu_proj_trace \
        --format=csv \
        --output="${output}/stats" \
        "${trace}" >>"${output}/run.log" 2>&1

    log "PROFILE DONE label=${label} trace=${trace}"
}

# Keep the baseline first by default. PROFILE_MODES can resume a partial
# session without repeating a completed capture, for example
# PROFILE_MODES=async_ulysses.
IFS=, read -r -a profile_modes <<<"${PROFILE_MODES:-sync_ulysses,async_ulysses}"
for label in "${profile_modes[@]}"; do
    case "${label}" in
        sync_ulysses) profile_mode "${label}" 0 ;;
        async_ulysses) profile_mode "${label}" 1 ;;
        *)
            log "STOP unknown profile mode: ${label}"
            exit 2
            ;;
    esac
done

log "COMPLETE session=${SESSION_ROOT}"
