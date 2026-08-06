#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# Full MiniMax-H3 matrix for an eight-card, PCIe-only RTX PRO 5000 host.
# Every case runs in a fresh process and writes a self-contained directory.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="${TEST_ROOT:-/lustre/raplab/client/sylarl/minimax-h3-native}"
CODE_ROOT="${CODE_ROOT:-${TEST_ROOT}/vllm-omni-trtllm-sm120}"
MODEL_ROOT="${MODEL_ROOT:-${TEST_ROOT}/MiniMax-H3}"
RESULT_ROOT="${RESULT_ROOT:-${TEST_ROOT}/results/minimax-h3-rtx5000-matrix}"
PYTHON="${PYTHON:-${TEST_ROOT}/.venv/bin/python}"

GPU_COUNTS="${GPU_COUNTS:-1,2,4,8}"
MODES="${MODES:-bf16,fp8,fp8_sm120_attn}"
STEPS="${STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
REPEATS="${REPEATS:-1}"
RUN_REF2VA="${RUN_REF2VA:-1}"
RESUME="${RESUME:-1}"

GPU_IDS_1="${GPU_IDS_1:-0}"
GPU_IDS_2="${GPU_IDS_2:-0,1}"
# Logical Ulysses pairs 0-2 and 1-3 become physical PXB pairs 0-1 and 2-3.
GPU_IDS_4="${GPU_IDS_4:-0,2,1,3}"
# Keep TP2 pairs local; Ulysses4 then crosses the fewest unavoidable host links.
GPU_IDS_8="${GPU_IDS_8:-0,1,2,3,4,5,6,7}"
NUMA_NODE_1="${NUMA_NODE_1:-0}"
NUMA_NODE_2="${NUMA_NODE_2:-0}"
NUMA_NODE_4="${NUMA_NODE_4:-0}"

GPU_MODEL_LABEL="${GPU_MODEL_LABEL:-RTX PRO 5000 Blackwell}"
NODE_LABEL="${NODE_LABEL:-8x RTX PRO 5000 Blackwell (sm120, no NVLink, PCIe)}"
MAX_PREFLIGHT_MEMORY_MIB="${MAX_PREFLIGHT_MEMORY_MIB:-2048}"
MAX_PREFLIGHT_GPU_UTIL="${MAX_PREFLIGHT_GPU_UTIL:-10}"
DLO_RESIDENT_LAYERS="${DLO_RESIDENT_LAYERS:-20}"
DENSE_ATTENTION_BACKEND="${DENSE_ATTENTION_BACKEND:-CUDNN_ATTN}"

export PATH="${TEST_ROOT}/ffmpeg-tools/bin:${PATH}"
for required in \
  "${PYTHON}" \
  "${CODE_ROOT}/vllm_omni/__init__.py" \
  "${MODEL_ROOT}/FL2VA/model_index.json" \
  "${SCRIPT_DIR}/run_case.py" \
  "${SCRIPT_DIR}/summarize_matrix.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 1
  fi
done
if [[ "${RUN_REF2VA}" == "1" && ! -e "${MODEL_ROOT}/Ref2VA/model_index.json" ]]; then
  echo "RUN_REF2VA=1 requires ${MODEL_ROOT}/Ref2VA/model_index.json" >&2
  exit 1
fi
for command_name in nvidia-smi ffmpeg ffprobe numactl rg; do
  command -v "${command_name}" >/dev/null || {
    echo "Required command is missing: ${command_name}" >&2
    exit 1
  }
done
if [[ ! "${STEPS}" =~ ^[1-9][0-9]*$ || ! "${WARMUP_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "STEPS must be positive and WARMUP_STEPS must be non-negative." >&2
  exit 2
fi
if [[ ! "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be positive." >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${TEST_ROOT}/hf-cache"
export XDG_CACHE_HOME="${TEST_ROOT}/xdg-cache"
export UV_CACHE_DIR="${TEST_ROOT}/uv-cache"
export TORCHINDUCTOR_CACHE_DIR="${TEST_ROOT}/torchinductor-cache"
export TRITON_CACHE_DIR="${TEST_ROOT}/triton-cache"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export NCCL_RAS_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ACTIVE_MONITOR_PID=""
cleanup_active_monitor() {
  if [[ -n "${ACTIVE_MONITOR_PID}" ]]; then
    kill "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    wait "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    ACTIVE_MONITOR_PID=""
  fi
}
trap cleanup_active_monitor EXIT INT TERM

code_commit="$(git -C "${CODE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
printf '%s\n' \
  "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "code_root=${CODE_ROOT}" \
  "code_commit=${code_commit}" \
  "model_root=${MODEL_ROOT}" \
  "steps_requested=${STEPS}" \
  "warmup_steps=${WARMUP_STEPS}" \
  "gpu_counts=${GPU_COUNTS}" \
  "run_ref2va=${RUN_REF2VA}" \
  "repeats=${REPEATS}" \
  "modes=${MODES}" \
  > "${RESULT_ROOT}/matrix.env"

topology_for() {
  case "$1" in
    1) echo "1 1 1 1 1" ;;
    2) echo "2 1 2 2 0" ;;
    4) echo "2 2 4 4 0" ;;
    8) echo "2 4 8 8 0" ;;
    *) echo "Unsupported GPU count: $1" >&2; return 2 ;;
  esac
}

mode_for() {
  case "$1" in
    bf16) echo "bf16 dense" ;;
    fp8) echo "fp8 dense" ;;
    fp8_sm120_attn) echo "fp8 sm120_prims" ;;
    *) echo "Unsupported mode: $1" >&2; return 2 ;;
  esac
}

preflight_gpus() {
  local gpu_ids="$1"
  local state gpu used util
  state="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  IFS=',' read -r -a selected <<< "${gpu_ids}"
  for gpu in "${selected[@]}"; do
    used="$(awk -F',' -v wanted="${gpu}" '$1 + 0 == wanted {gsub(/ /,"",$2); print $2}' <<< "${state}")"
    util="$(awk -F',' -v wanted="${gpu}" '$1 + 0 == wanted {gsub(/ /,"",$3); print $3}' <<< "${state}")"
    if [[ -z "${used}" || -z "${util}" ]]; then
      echo "GPU ${gpu} is not visible to nvidia-smi." >&2
      return 1
    fi
    if (( used > MAX_PREFLIGHT_MEMORY_MIB || util > MAX_PREFLIGHT_GPU_UTIL )); then
      echo "GPU ${gpu} is busy: memory=${used} MiB, utilization=${util}%." >&2
      return 1
    fi
  done
}

run_case() {
  local gpu_count="$1" mode="$2" repeat="$3"
  local tp ulysses text_tp vae_parallel enable_dlo precision attention_mode
  read -r tp ulysses text_tp vae_parallel enable_dlo <<< "$(topology_for "${gpu_count}")"
  read -r precision attention_mode <<< "$(mode_for "${mode}")"

  local ids_var gpu_ids
  ids_var="GPU_IDS_${gpu_count}"
  gpu_ids="${!ids_var}"
  local case_name="g${gpu_count}-${mode}-tp${tp}-u${ulysses}-run${repeat}"
  local case_dir="${RESULT_ROOT}/${case_name}"
  mkdir -p "${case_dir}"
  if [[ "${RESUME}" == "1" && -f "${case_dir}/status.txt" ]] && grep -qx Passed "${case_dir}/status.txt"; then
    echo "[skip] ${case_name} already passed"
    return 0
  fi

  IFS=',' read -r -a selected <<< "${gpu_ids}"
  if [[ "${#selected[@]}" -ne "${gpu_count}" ]]; then
    echo "${ids_var}=${gpu_ids} contains ${#selected[@]} GPUs, expected ${gpu_count}." >&2
    return 2
  fi
  preflight_gpus "${gpu_ids}"

  local offload="none"
  local -a dlo_args=()
  if [[ "${enable_dlo}" == "1" ]]; then
    offload="dit DLO (resident-layers ${DLO_RESIDENT_LAYERS})"
    dlo_args=(--enable-distributed-layerwise-offload --dlo-resident-layers "${DLO_RESIDENT_LAYERS}")
  fi

  printf '%s\n' \
    "gpu_count=${gpu_count}" \
    "gpu_ids=${gpu_ids}" \
    "mode=${mode}" \
    "precision=${precision}" \
    "attention_mode=${attention_mode}" \
    "parallelism=TP${tp} x Ulysses${ulysses}" \
    "offload=${offload}" \
    "gpu_model_label=${GPU_MODEL_LABEL}" \
    "node_label=${NODE_LABEL}" \
    "repeat=${repeat}" \
    > "${case_dir}/case.env"

  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,clocks.current.sm \
    --format=csv,noheader,nounits -lms 500 > "${case_dir}/gpu_telemetry.csv" &
  ACTIVE_MONITOR_PID=$!

  local -a common_args=(
    --model-root "${MODEL_ROOT}"
    --output-dir "${case_dir}"
    --height 768 --width 1344 --duration 5.0
    --num-inference-steps "${STEPS}"
    --warmup-steps "${WARMUP_STEPS}"
    --seed-base 1101
    --num-gpus "${gpu_count}"
    --precision "${precision}"
    --attention-mode "${attention_mode}"
    --tensor-parallel-size "${tp}"
    --ulysses-degree "${ulysses}"
    --ring-degree 1
    --text-encoder-tp-size "${text_tp}"
    --vae-patch-parallel-size "${vae_parallel}"
    --attention-backend "${DENSE_ATTENTION_BACKEND}"
    "${dlo_args[@]}"
  )
  if [[ "${RUN_REF2VA}" == "1" ]]; then common_args+=(--expect-ref2va); fi

  local -a numa_prefix=()
  if [[ "${gpu_count}" != "8" ]]; then
    local numa_var numa_node
    numa_var="NUMA_NODE_${gpu_count}"
    numa_node="${!numa_var}"
    numa_prefix=(numactl --cpunodebind="${numa_node}" --membind="${numa_node}")
  fi

  echo "[run] ${case_name}; GPUs=${gpu_ids}; ${offload}"
  local rc=0
  set +e
  CUDA_VISIBLE_DEVICES="${gpu_ids}" "${numa_prefix[@]}" "${PYTHON}" "${SCRIPT_DIR}/run_case.py" \
    --partition fl2va "${common_args[@]}" 2>&1 | tee "${case_dir}/run.log"
  rc=${PIPESTATUS[0]}
  if [[ "${rc}" == "0" && "${RUN_REF2VA}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_ids}" "${numa_prefix[@]}" "${PYTHON}" "${SCRIPT_DIR}/run_case.py" \
      --partition ref2va "${common_args[@]}" 2>&1 | tee -a "${case_dir}/run.log"
    rc=${PIPESTATUS[0]}
  fi
  set -e

  cleanup_active_monitor
  if [[ "${rc}" == "0" ]]; then
    echo Passed > "${case_dir}/status.txt"
  elif rg -qi 'out of memory|OutOfMemoryError|CUDA error: out of memory' "${case_dir}/run.log"; then
    echo OOM > "${case_dir}/status.txt"
  else
    echo Failed > "${case_dir}/status.txt"
  fi
  echo "${rc}" > "${case_dir}/exit_code.txt"
  return 0
}

IFS=',' read -r -a requested_counts <<< "${GPU_COUNTS}"
IFS=',' read -r -a requested_modes <<< "${MODES}"
for gpu_count in "${requested_counts[@]}"; do
  for mode in "${requested_modes[@]}"; do
    for repeat in $(seq 1 "${REPEATS}"); do
      run_case "${gpu_count}" "${mode}" "${repeat}"
    done
  done
done

"${PYTHON}" "${SCRIPT_DIR}/summarize_matrix.py" \
  "${RESULT_ROOT}" \
  --csv-output "${RESULT_ROOT}/matrix.csv" \
  --json-output "${RESULT_ROOT}/matrix.json"

echo "Matrix CSV: ${RESULT_ROOT}/matrix.csv"
echo "Matrix JSON: ${RESULT_ROOT}/matrix.json"
