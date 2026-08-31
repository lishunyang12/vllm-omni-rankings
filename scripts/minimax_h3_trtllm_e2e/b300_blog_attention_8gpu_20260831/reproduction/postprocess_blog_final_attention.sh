#!/usr/bin/env bash
set -euo pipefail

RUN_TMUX=minimax_blog_final_attention_tail
CONTAINER=vllm-omni-blog-b300-pr6814-rebased
ARTIFACT_ROOT=/home/scratch.lbo_gpu_1/projects/trtllm-gen/artifacts/minimax-h3-blog-b300
CONTAINER_ARTIFACT_ROOT=/artifacts
BASE_SESSION=$(<"${ARTIFACT_ROOT}/current_blog_final_session.txt")
TAIL_SESSION=$(<"${ARTIFACT_ROOT}/current_blog_final_tail_session.txt")

while tmux has-session -t "${RUN_TMUX}" 2>/dev/null; do
    sleep 10
done

host_session_root=${ARTIFACT_ROOT}/${TAIL_SESSION}
session_root=${CONTAINER_ARTIFACT_ROOT}/${TAIL_SESSION}
mkdir -p \
    "${host_session_root}/trtllm_dense/client" \
    "${host_session_root}/sage_fp8_k4/client"
ln -sfn \
    "${CONTAINER_ARTIFACT_ROOT}/${BASE_SESSION}/trtllm_dense/client/response_2.mp4" \
    "${host_session_root}/trtllm_dense/client/response_2.mp4"
ln -sfn \
    "${CONTAINER_ARTIFACT_ROOT}/${BASE_SESSION}/sage_fp8_k4/client/response_2.mp4" \
    "${host_session_root}/sage_fp8_k4/client/response_2.mp4"
reference=${session_root}/trtllm_dense/client/response_2.mp4
compare_script=${CONTAINER_ARTIFACT_ROOT}/blog-branch-evidence-b300-snapshot/compare_video_lpips.py

for mode in sage_fp8_k4 skip_softmax_005_gate097 sage_fp8_skip_005_gate097; do
    docker exec \
        --user 47897:30 \
        --env HOME=/home/scratch.lbo_other/vllm-omni-blog-home \
        --env HF_HOME=/home/scratch.lbo_other/hf-cache-vllm-omni-blog \
        "${CONTAINER}" python3 "${compare_script}" \
        "${reference}" \
        "${session_root}/${mode}/client/response_2.mp4" \
        --output "${session_root}/dense_to_${mode}_video_quality.json"
done

docker exec \
    --user 47897:30 \
    --env HOME=/home/scratch.lbo_other/vllm-omni-blog-home \
    "${CONTAINER}" python3 \
    "${CONTAINER_ARTIFACT_ROOT}/compare_attention_audio.py" \
    "${session_root}" \
    --modes \
        trtllm_dense \
        sage_fp8_k4 \
        skip_softmax_005_gate097 \
        sage_fp8_skip_005_gate097 \
    --output "${session_root}/audio_quality.json"
