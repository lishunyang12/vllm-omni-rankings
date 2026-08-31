#!/usr/bin/env bash
set -euo pipefail

SESSION_ROOT=${SESSION_ROOT:?SESSION_ROOT is required}
CONTAINER=${CONTAINER:-vllm-omni-h3-mixed-devel-lbo}
RUNNER_TMUX=${RUNNER_TMUX:-h3_starship_strict_serial}
SCRIPTS=/workspace/vllm-omni-rankings/scripts/minimax_h3_trtllm_e2e

exec > >(tee -a "${SESSION_ROOT}/postprocess.log") 2>&1

while tmux has-session -t "${RUNNER_TMUX}" 2>/dev/null; do
    sleep 30
done

if ! grep -q 'COMPLETE session=' "${SESSION_ROOT}/runner.log"; then
    printf 'Runner did not complete successfully\n' >&2
    exit 1
fi

docker exec "${CONTAINER}" python3 "${SCRIPTS}/summarize_starship_strict.py" \
    "${SESSION_ROOT}" \
    --json "${SESSION_ROOT}/results.json" \
    --markdown "${SESSION_ROOT}/results.md"

for mode in \
    trtllm_dense \
    sage_fp8 \
    skip_softmax_005_gate097 \
    sage_fp8_skip_005_gate097; do
    docker exec "${CONTAINER}" ffmpeg -v error -xerror \
        -i "${SESSION_ROOT}/${mode}/t2va_${mode}_run2.mp4" \
        -map 0:v:0 -map 0:a:0 -f null -
done

docker exec "${CONTAINER}" python3 -c \
    'import json,sys; payload=json.load(open(sys.argv[1])); assert payload["accepted"]' \
    "${SESSION_ROOT}/results.json"

printf 'COMPLETE session=%s\n' "${SESSION_ROOT}"
