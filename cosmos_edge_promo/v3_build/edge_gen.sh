#!/bin/bash
# ============================================================================
#  edge_gen.sh — one-command Cosmos3-Edge generation at the OFFICIAL "perfect"
#  recipe. Everything defaults to NVIDIA's Cosmos3-Edge model-card config; you
#  only supply a mode + prompt (+ an image for i2v).
#
#  Usage:
#     ./edge_gen.sh t2v "a robot arm sorting colorful blocks on a table"
#     ./edge_gen.sh i2v "the car drives forward along the coastal road" --image in.jpg
#     ./edge_gen.sh t2i "a photorealistic red sports car at golden hour"
#
#  Optional overrides: -o OUT  --frames N  --steps N  --seed N  (rarely needed)
#
#  Official defaults (from nvidia/Cosmos3-Edge model card, i2v example):
#     832x480 · 121 frames · 24 fps · steps 50 · guidance 5.0 · flow_shift 3.0
#     seed 0 · max_sequence_length 4096 · official structured negative_prompt
#  The negative_prompt is THE quality lever: it suppresses the flicker /
#  compression-block / incoherent-motion artifacts small Edge otherwise shows.
# ============================================================================
set -euo pipefail

REPO="${REPO:-/home/zjy/code/lsy/vllm-omni}"
PY="${PY:-/home/zjy/code/lsy/cosmos-venv/bin/python}"
ASSETS="${ASSETS:-/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_official}"
MODEL="${MODEL:-nvidia/Cosmos3-Edge}"
GPU="${GPU:-0}"

# ---- official "perfect" defaults ----
STEPS=50; FRAMES=121; H=480; W=832; GS=5.0; FLOW=3.0; FPS=24; SEED=0; MAXSEQ=4096

MODE="${1:?usage: edge_gen.sh <t2i|t2v|i2v> \"prompt\" [--image img] [-o out]}"; shift
PROMPT="${1:?need a prompt}"; shift
IMAGE=""; OUT=""
while [ $# -gt 0 ]; do case "$1" in
  --image) IMAGE="$2"; shift 2;;
  -o|--output) OUT="$2"; shift 2;;
  --frames) FRAMES="$2"; shift 2;;
  --steps) STEPS="$2"; shift 2;;
  --seed) SEED="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done

# ---- fetch + compact the official negative_prompt (json.dumps, like the card) ----
NEG_JSON="$ASSETS/negative_prompt.json"
mkdir -p "$ASSETS"
[ -f "$NEG_JSON" ] || curl -sL "https://huggingface.co/${MODEL}/resolve/main/assets/negative_prompt.json" -o "$NEG_JSON"
NEG="$("$PY" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$NEG_JSON")"

EB="{\"flow_shift\": ${FLOW}, \"max_sequence_length\": ${MAXSEQ}, \"guardrails\": false}"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
common=(--model "$MODEL" --guidance-scale "$GS" --seed "$SEED" --negative-prompt "$NEG" --extra-body "$EB")

case "$MODE" in
  t2i)
    OUT="${OUT:-edge_t2i.png}"
    "$PY" examples/offline_inference/text_to_image/text_to_image.py \
      "${common[@]}" --height 640 --width 640 --num-inference-steps "$STEPS" \
      --prompt "$PROMPT" --output "$OUT" ;;
  t2v)
    OUT="${OUT:-edge_t2v.mp4}"
    "$PY" examples/offline_inference/text_to_video/text_to_video.py \
      "${common[@]}" --height "$H" --width "$W" --num-frames "$FRAMES" \
      --num-inference-steps "$STEPS" --fps "$FPS" --prompt "$PROMPT" --output "$OUT" ;;
  i2v)
    [ -n "$IMAGE" ] || { echo "i2v needs --image" >&2; exit 2; }
    OUT="${OUT:-edge_i2v.mp4}"
    "$PY" examples/offline_inference/image_to_video/image_to_video.py \
      "${common[@]}" --image "$IMAGE" --height "$H" --width "$W" --num-frames "$FRAMES" \
      --num-inference-steps "$STEPS" --fps "$FPS" --prompt "$PROMPT" --output "$OUT" ;;
  *) echo "mode must be t2i|t2v|i2v" >&2; exit 2;;
esac
echo "DONE -> $OUT  (steps $STEPS, ${W}x${H}, ${FRAMES}f, gs $GS, flow $FLOW, seed $SEED, +official negative_prompt)"
