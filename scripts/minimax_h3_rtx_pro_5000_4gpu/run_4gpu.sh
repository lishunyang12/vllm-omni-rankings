#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# Reproduce the MiniMax-H3 four-GPU RTX PRO 5000 Blackwell measurements.
#
# Modes:
#   benchmark: normal runs for publishable latency/memory (default)
#   profile:   one Nsight Systems run plus SQLite/Markdown/JSON export

MODE="${MODE:-benchmark}"
TEST_ROOT="${TEST_ROOT:-/lustre/raplab/client/sylarl/minimax-h3-native}"
REPO_ROOT="${REPO_ROOT:-${TEST_ROOT}/vllm-omni-pr5852}"
MODEL_ROOT="${MODEL_ROOT:-${TEST_ROOT}/MiniMax-H3}"
RESULT_ROOT="${RESULT_ROOT:-${TEST_ROOT}/results/pr5852-rtx5000-final}"
GPU_IDS="${GPU_IDS:-4,6,5,7}"
GPU_ORDER_LABEL="${GPU_ORDER_LABEL:-ulysses-pxb}"
NUMA_NODE="${NUMA_NODE:-1}"
STEPS="${STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
REPEATS="${REPEATS:-3}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUDNN_ATTN}"
PYTHON="${PYTHON:-${TEST_ROOT}/.venv/bin/python}"

case "${MODE}" in
  benchmark|profile) ;;
  *) echo "MODE must be benchmark or profile, got: ${MODE}" >&2; exit 2 ;;
esac

for required in "${PYTHON}" "${MODEL_ROOT}/FL2VA/model_index.json" \
  "${REPO_ROOT}/examples/offline_inference/minimax_h3/run_all_tasks.sh"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 1
  fi
done

export PATH="${TEST_ROOT}/ffmpeg-tools/bin:${PATH}"
for command_name in nvidia-smi numactl ffmpeg ffprobe; do
  command -v "${command_name}" >/dev/null || {
    echo "Required command is missing: ${command_name}" >&2
    exit 1
  }
done
if [[ "${MODE}" == "profile" ]]; then
  command -v nsys >/dev/null || {
    echo "Nsight Systems (nsys) is required for MODE=profile" >&2
    exit 1
  }
fi

COMMON_ENV=(
  MODEL_ROOT="${MODEL_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  PYTHON="${PYTHON}"
  WORK_ROOT="${TEST_ROOT}"
  INSTALL_EDITABLE=0
  PYTHONPATH="${REPO_ROOT}"
  HF_HOME="${TEST_ROOT}/hf-cache"
  XDG_CACHE_HOME="${TEST_ROOT}/xdg-cache"
  UV_CACHE_DIR="${TEST_ROOT}/uv-cache"
  TORCHINDUCTOR_CACHE_DIR="${TEST_ROOT}/torchinductor-cache"
  NUM_INFERENCE_STEPS="${STEPS}"
  WARMUP_STEPS="${WARMUP_STEPS}"
  TP_SIZE=2
  ULYSSES_DEGREE=2
  RING_DEGREE=1
  TEXT_ENCODER_TP_SIZE=4
  VAE_PATCH_PARALLEL_SIZE=4
  ATTENTION_BACKEND="${ATTENTION_BACKEND}"
  RUN_REF2VA=0
  PROFILE_DIR=
  NCCL_RAS_ENABLE=0
)

RUNNER=(bash examples/offline_inference/minimax_h3/run_all_tasks.sh)
cd "${REPO_ROOT}"
mkdir -p "${RESULT_ROOT}"

if [[ "${MODE}" == "benchmark" ]]; then
  for run_index in $(seq 1 "${REPEATS}"); do
    output_dir="${RESULT_ROOT}/${GPU_ORDER_LABEL}-${STEPS}requested-normal-run${run_index}"
    mkdir -p "${output_dir}"
    echo "[benchmark] run=${run_index}/${REPEATS} output=${output_dir}"
    env "${COMMON_ENV[@]}" OUTPUT_DIR="${output_dir}" \
      numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
      "${RUNNER[@]}" 2>&1 | tee "${output_dir}/console.log"
  done
  exit 0
fi

if [[ "${REPEATS}" != "1" && "${REPEATS}" != "3" ]]; then
  echo "Ignoring REPEATS=${REPEATS}; MODE=profile always records one run." >&2
fi

profile_root="${RESULT_ROOT}/${GPU_ORDER_LABEL}-${STEPS}requested-nsys"
output_dir="${profile_root}/outputs"
mkdir -p "${output_dir}"

env "${COMMON_ENV[@]}" OUTPUT_DIR="${output_dir}" \
  numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --trace-fork-before-exec=true \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="${profile_root}/minimax-h3" \
    "${RUNNER[@]}" 2>&1 | tee "${profile_root}/nsys.log"

nsys export \
  --type=sqlite \
  --force-overwrite=true \
  --output="${profile_root}/minimax-h3.sqlite" \
  "${profile_root}/minimax-h3.nsys-rep"

"${PYTHON}" examples/offline_inference/minimax_h3/analyze_nsys.py \
  "${profile_root}/minimax-h3.sqlite" \
  --json-output "${profile_root}/kernel-breakdown.json" \
  | tee "${profile_root}/kernel-breakdown.md"

sha256sum \
  "${profile_root}/minimax-h3.nsys-rep" \
  "${profile_root}/minimax-h3.sqlite" \
  "${profile_root}/kernel-breakdown.json" \
  > "${profile_root}/artifact_sha256.txt"

echo "Profile report: ${profile_root}/kernel-breakdown.md"
