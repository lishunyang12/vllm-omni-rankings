#!/bin/bash
# Regenerate promo footage through the validated official recipe (edge_gen.sh defaults +
# official negative_prompt). Detailed prompts for free-T2V quality. fp8 pair for the speed story.
set -u
EDGE=/home/zjy/code/lsy/vllm-omni-rankings/cosmos_edge_promo/v3_build/edge_gen.sh
REPO=/home/zjy/code/lsy/vllm-omni
PY=/home/zjy/code/lsy/cosmos-venv/bin/python
OFF=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_official
OUT=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_edge_gen
NEG="$($PY -c "import json;print(json.dumps(json.load(open('$OFF/negative_prompt.json'))))")"
EB='{"flow_shift": 3.0, "max_sequence_length": 4096, "guardrails": false}'

ROBOT="A high-precision robotic arm on a clean industrial workbench under bright even studio lighting. The metallic arm smoothly reaches down, its parallel gripper closing gently around a bright red cube, lifting it and stacking it squarely on top of a blue cube. Sharp focus, realistic metal and plastic materials, steady controlled motion, professional product-demo cinematography, deep depth of field."
WAREHOUSE="A wheeled autonomous mobile robot glides smoothly across a polished concrete warehouse floor between tall metal shelving racks stacked with cardboard boxes. Bright industrial ceiling lights, crisp realistic textures, steady forward motion, wide-angle documentary style, sharp focus, deep depth of field."
PLATE="A robotic arm on a modern kitchen counter carefully picks up a clean white ceramic plate and holds it under gently running water, soft natural window light, realistic reflections and water, smooth steady motion, sharp focus, shallow depth of field."

# GPU5: T2V robot stack
GPU=5 "$EDGE" t2v "$ROBOT" -o "$OUT/v4_t2v_robot.mp4" > "$OUT/v4_t2v_robot.log" 2>&1 &
P5=$!
# GPU6: T2V warehouse AMR
GPU=6 "$EDGE" t2v "$WAREHOUSE" -o "$OUT/v4_t2v_warehouse.mp4" > "$OUT/v4_t2v_warehouse.log" 2>&1 &
P6=$!
# GPU7: fp8 DENSE plate (recipe params, direct call so we can add --quantization on the pair)
cd "$REPO"
CUDA_VISIBLE_DEVICES=7 $PY examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Edge --height 480 --width 832 --num-frames 121 --num-inference-steps 50 \
  --guidance-scale 5.0 --fps 24 --seed 0 --negative-prompt "$NEG" --extra-body "$EB" \
  --prompt "$PLATE" --output "$OUT/v4_plate_dense.mp4" > "$OUT/v4_plate_dense.log" 2>&1 &
P7=$!

wait $P5; echo "ROBOT_EXIT=$?"
wait $P6; echo "WAREHOUSE_EXIT=$?"
wait $P7; echo "PLATE_DENSE_EXIT=$?"

# fp8 counterpart of the plate (same seed/prompt/neg) on GPU5 (now free)
CUDA_VISIBLE_DEVICES=5 $PY examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Edge --height 480 --width 832 --num-frames 121 --num-inference-steps 50 \
  --guidance-scale 5.0 --fps 24 --seed 0 --quantization fp8 --negative-prompt "$NEG" --extra-body "$EB" \
  --prompt "$PLATE" --output "$OUT/v4_plate_fp8.mp4" > "$OUT/v4_plate_fp8.log" 2>&1
echo "PLATE_FP8_EXIT=$?"
echo "REGEN_V4_DONE"
