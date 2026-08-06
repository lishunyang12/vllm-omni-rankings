#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8091}"
API_URL="${API_URL:-http://127.0.0.1:${PORT}/v1/videos/sync}"
OUTPUT="${1:-t2va.mp4}"

curl -sS --fail-with-body -X POST "${API_URL}" \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1248' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","aspect_ratio":"16:9","duration":8.7,"audio_flow_shift":3.0}' \
  -o "${OUTPUT}"

echo "Wrote ${OUTPUT}"
