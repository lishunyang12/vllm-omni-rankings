#!/usr/bin/env bash
set -euo pipefail

CONTAINER=${CONTAINER:-vllm-omni-blog-b300-pr6814-rebased}
SOURCE_ROOT=${SOURCE_ROOT:-/home/scratch.lbo_other/vllm-omni-pr6814-tp8-bench}
CACHE_ROOT=${CACHE_ROOT:-/home/scratch.lbo_other/cache-vllm-omni-blog-pr6814-sm103a-tp8-v1}
PORT=${PORT:-18084}
SESSION_NAME=${SESSION_NAME:-b300_vllm_pr6814_398b0a7e_8gpu_te_tp8_attention_serve_$(date -u +%Y%m%dT%H%M%SZ)}
HOST_ARTIFACT_ROOT=/home/scratch.lbo_gpu_1/projects/trtllm-gen/artifacts/minimax-h3-blog-b300
CONTAINER_ARTIFACT_ROOT=/artifacts
SESSION_ROOT=${HOST_ARTIFACT_ROOT}/${SESSION_NAME}
CONTAINER_SESSION_ROOT=${CONTAINER_ARTIFACT_ROOT}/${SESSION_NAME}
PROMPT_FILE=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/prompts/minimax_h3_official_starship.txt
MODEL=/home/scratch.lbo_other/MiniMax-H3/FL2VA
GPU_INDICES=0,1,2,3,4,5,6,7
EXPECTED_CPUSET=0-27,56-83,112-139,168-195
MEASURE=${MEASURE:-2}
MAX_START_TEMPERATURE_C=${MAX_START_TEMPERATURE_C:-50}
INTER_REQUEST_DELAY_S=${INTER_REQUEST_DELAY_S:-0}

if [[ -n ${MODES:-} ]]; then
    read -r -a modes <<<"${MODES}"
else
    modes=(
        trtllm_dense
        sage_fp8
        skip_softmax_005_gate097
        sage_fp8_skip_005_gate097
    )
fi

attention_config() {
    case "$1" in
        trtllm_dense)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN"}}'
            ;;
        sage_fp8)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN","quant":{"dtype_qk":"fp8_e4m3","q_block_size":1,"k_block_size":16}},"per_role":{"minimax_h3.token_refiner":{"backend":"TRTLLM_ATTN"}}}'
            ;;
        sage_fp8_k1)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN","quant":{"dtype_qk":"fp8_e4m3","q_block_size":1,"k_block_size":1}},"per_role":{"minimax_h3.token_refiner":{"backend":"TRTLLM_ATTN"}}}'
            ;;
        sage_fp8_k4)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN","quant":{"dtype_qk":"fp8_e4m3","q_block_size":1,"k_block_size":4}},"per_role":{"minimax_h3.token_refiner":{"backend":"TRTLLM_ATTN"}}}'
            ;;
        skip_softmax_005_gate097)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{"threshold":0.05,"disabled_until_timestep":0.97}},"per_role":{"minimax_h3.token_refiner":{"backend":"TRTLLM_ATTN"}}}'
            ;;
        skip_softmax_010_gate097)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{"threshold":0.1,"disabled_until_timestep":0.97}},"per_role":{"minimax_h3.token_refiner":{"backend":"TRTLLM_ATTN"}}}'
            ;;
        skip_softmax_020_gate097)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{"threshold":0.2,"disabled_until_timestep":0.97}},"per_role":{"minimax_h3.token_refiner":{"backend":"TRTLLM_ATTN"}}}'
            ;;
        skip_softmax_010_gate099)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{"threshold":0.1,"disabled_until_timestep":0.99}},"per_role":{"minimax_h3.token_refiner":{"backend":"TRTLLM_ATTN"}}}'
            ;;
        skip_softmax_005_gate099)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{"threshold":0.05,"disabled_until_timestep":0.99}},"per_role":{"minimax_h3.token_refiner":{"backend":"TRTLLM_ATTN"}}}'
            ;;
        sage_fp8_skip_005_gate097)
            printf '%s' '{"default":{"backend":"TRTLLM_ATTN","quant":{"dtype_qk":"fp8_e4m3","q_block_size":1,"k_block_size":4},"skip_softmax":{"threshold":0.05,"disabled_until_timestep":0.97}},"per_role":{"minimax_h3.token_refiner":{"backend":"TRTLLM_ATTN"}}}'
            ;;
        *)
            return 1
            ;;
    esac
}

server_pids() {
    docker exec "${CONTAINER}" bash -lc \
        "pgrep -f '^python3 -m vllm_omni.entrypoints.cli.main serve .* --port ${PORT}( |$)' || true"
}

stop_server() {
    local pids
    pids=$(server_pids)
    if [[ -n ${pids} ]]; then
        docker exec "${CONTAINER}" kill -TERM ${pids} || true
        for _ in $(seq 1 120); do
            [[ -z $(server_pids) ]] && return
            sleep 1
        done
        docker exec "${CONTAINER}" kill -KILL ${pids} || true
    fi
}

wait_for_health() {
    local mode_root=$1
    for _ in $(seq 1 600); do
        if docker exec "${CONTAINER}" curl -fsS --max-time 2 \
            "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return
        fi
        sleep 1
    done
    tail -n 200 "${mode_root}/server.log"
    return 1
}

mkdir -p "${SESSION_ROOT}"
printf '%s\n' "$$" >"${SESSION_ROOT}/runner.pid"
exec > >(tee -a "${SESSION_ROOT}/runner.log") 2>&1

actual_cpuset=$(docker inspect "${CONTAINER}" --format '{{.HostConfig.CpusetCpus}}')
if [[ ${actual_cpuset} != "${EXPECTED_CPUSET}" ]]; then
    printf 'Container CPU set is %q; expected %q\n' \
        "${actual_cpuset}" "${EXPECTED_CPUSET}" >&2
    exit 1
fi

telemetry_pid=
cleanup() {
    stop_server
    if [[ -n ${telemetry_pid} ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
        kill "${telemetry_pid}" 2>/dev/null || true
        wait "${telemetry_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

stop_server
nvidia-smi \
    --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,clocks.sm,power.draw,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons_counters.sw_thermal_slowdown \
    --format=csv,noheader,nounits \
    --loop-ms=5000 >"${SESSION_ROOT}/gpu_telemetry.csv" 2>&1 &
telemetry_pid=$!

for mode in "${modes[@]}"; do
    mode_root=${SESSION_ROOT}/${mode}
    container_mode_root=${CONTAINER_SESSION_ROOT}/${mode}
    mkdir -p "${mode_root}/client"
    config=$(attention_config "${mode}")
    printf '%s\n' "${config}" >"${mode_root}/attention_config.json"

    printf '[%s] starting %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${mode}"
    docker exec --detach \
        --user 47897:30 \
        --workdir "${SOURCE_ROOT}" \
        --env HOME=/home/scratch.lbo_other/vllm-omni-blog-home \
        --env HF_HOME=/home/scratch.lbo_other/hf-cache-vllm-omni-blog \
        --env XDG_CACHE_HOME="${CACHE_ROOT}" \
        --env PYTHONPATH="${SOURCE_ROOT}" \
        --env TORCH_CUDA_ARCH_LIST=10.3a \
        --env CUDA_VISIBLE_DEVICES=${GPU_INDICES} \
        --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
        --env VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
        --env PYTHONUNBUFFERED=1 \
        --env ATTENTION_CONFIG="${config}" \
        --env MODE_ROOT="${container_mode_root}" \
        "${CONTAINER}" bash -lc '
            exec python3 -m vllm_omni.entrypoints.cli.main serve '"${MODEL}"' \
                --omni \
                --host 0.0.0.0 \
                --port '"${PORT}"' \
                --trust-remote-code \
                --task-type fl2va \
                --num-gpus 8 \
                --usp 8 \
                --ring 1 \
                --ulysses-a2a-permute \
                --text-encoder-tp-size 8 \
                --vae-patch-parallel-size 8 \
                --vae-parallel-mode tile \
                --vae-use-tiling \
                --diffusion-attention-config "${ATTENTION_CONFIG}" \
                >"${MODE_ROOT}/server.log" 2>&1
        '
    wait_for_health "${mode_root}"

    docker exec \
        --user 47897:30 \
        --env HOME=/home/scratch.lbo_other/vllm-omni-blog-home \
        --env HF_HOME=/home/scratch.lbo_other/hf-cache-vllm-omni-blog \
        "${CONTAINER}" bash -lc \
        'python3 /artifacts/run_vllm_http_client.py \
            --server http://127.0.0.1:'"${PORT}"' \
            --prompt-file '"${PROMPT_FILE}"' \
            --output-dir '"${container_mode_root}"'/client \
            --warmup 1 \
            --measure '"${MEASURE}"' \
            --max-start-temperature-c '"${MAX_START_TEMPERATURE_C}"' \
            --inter-request-delay-s '"${INTER_REQUEST_DELAY_S}"' \
            --gpu-indices '"${GPU_INDICES}"' \
            2>&1 | tee '"${container_mode_root}"'/client.log'

    stop_server
    printf '[%s] completed %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${mode}"
done

kill "${telemetry_pid}" 2>/dev/null || true
wait "${telemetry_pid}" 2>/dev/null || true
telemetry_pid=

docker exec --user 47897:30 \
    --env HOME=/home/scratch.lbo_other/vllm-omni-blog-home \
    "${CONTAINER}" python3 \
    /workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e/check_thermal_telemetry.py \
    "${CONTAINER_SESSION_ROOT}/gpu_telemetry.csv" \
    --gpus "${GPU_INDICES}" \
    --output "${CONTAINER_SESSION_ROOT}/thermal_audit.json"

printf '[%s] complete: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SESSION_ROOT}"
