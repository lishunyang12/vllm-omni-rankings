#!/bin/bash
# Cosmos3-Edge I2V (blue bird) via vLLM-Omni on B200 — dense + fp8, with peak-VRAM + timing.
# Run from ~/vllm-omni. Uses omni-env (vllm 0.26.0, has scale_out). All footage = real Edge output.
set -e
cd "${ROOT:-$HOME/vllm-omni}"
PY="${PY:-$HOME/omni-env/bin/python}"
GPU="${GPU:-0}"; export CUDA_VISIBLE_DEVICES="$GPU"
OUT="${OUT:-edge_i2v_out}"; mkdir -p "$OUT"
IMG="${IMG:-bird_input.png}"
IMG_URL="${IMG_URL:-https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png}"
PROMPT="${PROMPT:-CG animation style, a small blue bird takes off from a branch and lands on another branch.}"
[ -f "$IMG" ] || curl -L -o "$IMG" "$IMG_URL"
echo "image=$IMG  prompt=$PROMPT  gpu=$GPU"

sample(){ for i in $(seq 1 400); do nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits; sleep 3; done > "$OUT/$1_vram.log"; }

for Q in "" "--quantization fp8"; do
  tag=$([ -z "$Q" ] && echo dense || echo fp8)
  echo "########## EDGE I2V $tag ##########"
  sample "$tag" & S=$!
  $PY examples/offline_inference/image_to_video/image_to_video.py --model nvidia/Cosmos3-Edge \
    --image "$IMG" --prompt "$PROMPT" --num-frames 49 $Q \
    --extra-body '{"max_sequence_length": 4096, "guardrails": false}' \
    --output "$OUT/edge_i2v_$tag.mp4" 2>&1 | tee "$OUT/$tag.log"
  kill $S 2>/dev/null || true
  echo "$tag PEAK_VRAM_MiB=$(sort -n "$OUT/${tag}_vram.log" 2>/dev/null | tail -1)"
done
echo "########## SUMMARY ##########"
grep -hE "Total generation time|Worker peak GPU memory" "$OUT"/dense.log "$OUT"/fp8.log 2>/dev/null
echo "outputs in $OUT/"; ls -lh "$OUT"/*.mp4 2>/dev/null
