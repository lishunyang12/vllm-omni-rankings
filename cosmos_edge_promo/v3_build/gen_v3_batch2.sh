#!/bin/bash
# v3 batch2 — high-hit-rate content: simple INDOOR robotics T2V (the proven-clean category)
# at a safer 93 frames, plus a manipulation I2V. Correct Edge params 480x832/gs5/flow3.
set -u
REPO=/home/zjy/code/lsy/vllm-omni
PY=/home/zjy/code/lsy/cosmos-venv/bin/python
OUT=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_edge_gen
UMI=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_official/example_action_fd_umi_first_frame.png
EB='{"flow_shift": 3.0, "max_sequence_length": 4096, "guardrails": false}'
cd "$REPO"

CUDA_VISIBLE_DEVICES=5 $PY examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Edge --height 480 --width 832 --num-frames 93 --guidance-scale 5.0 \
  --prompt "A robotic arm carefully places a clear glass cup onto a wooden kitchen shelf, soft daylight from a window, slow smooth precise motion." \
  --extra-body "$EB" --output "$OUT/v3_t2v_kitchen.mp4" > "$OUT/v3_t2v_kitchen.log" 2>&1 &
P5=$!

CUDA_VISIBLE_DEVICES=6 $PY examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Edge --height 480 --width 832 --num-frames 93 --guidance-scale 5.0 \
  --prompt "A robotic arm slowly sorts colorful plastic blocks into small bins on a clean white table, bright even studio lighting, steady controlled motion." \
  --extra-body "$EB" --output "$OUT/v3_t2v_sort.mp4" > "$OUT/v3_t2v_sort.log" 2>&1 &
P6=$!

CUDA_VISIBLE_DEVICES=7 $PY examples/offline_inference/image_to_video/image_to_video.py \
  --model nvidia/Cosmos3-Edge --image "$UMI" --height 480 --width 832 --num-frames 93 --guidance-scale 5.0 \
  --prompt "The camera moves slowly forward over the desk toward the dark mat, objects on the table drifting closer, smooth continuous motion." \
  --extra-body "$EB" --output "$OUT/v3_i2v_umi.mp4" > "$OUT/v3_i2v_umi.log" 2>&1 &
P7=$!

wait $P5; echo "KITCHEN_EXIT=$?"
wait $P6; echo "SORT_EXIT=$?"
wait $P7; echo "UMI_EXIT=$?"
echo "BATCH2_DONE"
