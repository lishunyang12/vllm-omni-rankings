#!/bin/bash
# v3 footage batch — EXPLICIT correct Edge params (branch-independent): 480x832 / gs5 / flow3.
# 189 frames (~7.9s). Also validates that 189f is clean at correct params.
set -u
REPO=/home/zjy/code/lsy/vllm-omni
PY=/home/zjy/code/lsy/cosmos-venv/bin/python
OUT=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_edge_gen
IMG=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_official/example_i2v_input.jpg
EB='{"flow_shift": 3.0, "max_sequence_length": 4096, "guardrails": false}'
cd "$REPO"

CUDA_VISIBLE_DEVICES=5 $PY examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Edge --height 480 --width 832 --num-frames 189 --guidance-scale 5.0 \
  --prompt "A first-person dashcam view driving along a coastal mountain highway in daylight, the road curving between rocky cliffs on the left and the ocean on the right, guardrail passing by, smooth forward motion." \
  --extra-body "$EB" --output "$OUT/v3_t2v_driving.mp4" > "$OUT/v3_t2v_driving.log" 2>&1 &
P5=$!

CUDA_VISIBLE_DEVICES=6 $PY examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Edge --height 480 --width 832 --num-frames 189 --guidance-scale 5.0 \
  --prompt "A robotic arm on an industrial workbench picks up a red cube and carefully stacks it on top of a blue cube, precise controlled motion, bright factory lighting." \
  --extra-body "$EB" --output "$OUT/v3_t2v_robot.mp4" > "$OUT/v3_t2v_robot.log" 2>&1 &
P6=$!

CUDA_VISIBLE_DEVICES=7 $PY examples/offline_inference/image_to_video/image_to_video.py \
  --model nvidia/Cosmos3-Edge --image "$IMG" --height 480 --width 832 --num-frames 189 --guidance-scale 5.0 \
  --prompt "The car drives forward along the winding coastal highway, following the road as it curves to the left past the cliff, guardrail and ocean passing by on the right, smooth continuous motion." \
  --extra-body "$EB" --output "$OUT/v3_i2v_driving.mp4" > "$OUT/v3_i2v_driving.log" 2>&1 &
P7=$!

wait $P5; echo "T2V_DRIVING_EXIT=$?"
wait $P6; echo "T2V_ROBOT_EXIT=$?"
wait $P7; echo "I2V_DRIVING_EXIT=$?"
echo "BATCH_DONE"
