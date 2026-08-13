#!/usr/bin/env bash
set -euo pipefail

# Public offline examples for every LTX-2.5 pipeline/task/modality combination.
# Override these paths from the environment when needed.
VLLM_OMNI_ROOT="${VLLM_OMNI_ROOT:-/path/to/vllm-omni}"
LTX25_MODEL_ROOT="${LTX25_MODEL_ROOT:-/path/to/LTX-2.5-Diffusers}"
LTX25_IMAGE="${LTX25_IMAGE:-$PWD/scripts/ltx25_raw_b300_pipeline_parity/inputs/quickstart-seed42-frame0.png}"
LTX25_OUTPUT_DIR="${LTX25_OUTPUT_DIR:-$PWD/ltx25_outputs}"
PYTHON_BIN="${VLLM_OMNI_PYTHON:-python}"

PROMPT='A medium close-up shot features a Caucasian man with a beard, wearing a green and white baseball cap without any letters on the front, and a light blue shirt over a white t-shirt. He is positioned in the center of the frame, looking intently directly at the camera, his eyes focused on camera. His facial expression is one of deep concentration, with his brow slightly raised. As he looks straight at the camera, a quick sniff sound is heard, and then he speaks with a deep male voice and a satisfied tone, saying, '"'"'I think it'"'"'s so good.'"'"' The camera remains static throughout, maintaining a shallow depth of field, which keeps the man in sharp focus while the background is softly blurred, showing a beige wall behind him. After a brief pause, another short, audible sniff is heard. The man then continues to speak, his voice maintaining the same quality, as he states, '"'"'So good. So good.'"'"' He elaborates further, emphasizing his point with a final statement, '"'"'This got to be, it'"'"'s got to be the best tool I'"'"'ve ever seen.'"'"''
NEGATIVE_PROMPT='has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.'

mkdir -p "$LTX25_OUTPUT_DIR"
export PYTHONPATH="$VLLM_OMNI_ROOT"
export DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-CUDNN_ATTN}"

COMMON=(
  --model "$LTX25_MODEL_ROOT"
  --prompt "$PROMPT"
  --num-frames 481
  --frame-rate 24
  --fps 24
  --seed 42
  --audio-sample-rate 48000
  --enforce-eager
)

# Full/Dev one-stage T2V: direct 960x544 generation, official 30-step schedule.
"$PYTHON_BIN" "$VLLM_OMNI_ROOT/examples/offline_inference/text_to_video/text_to_video.py" \
  "${COMMON[@]}" \
  --model-class-name LTX2Pipeline \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --height 544 --width 960 --num-inference-steps 30 \
  --output "$LTX25_OUTPUT_DIR/full-one-stage-t2v.mp4"

# Full/Dev one-stage first-frame I2V. LTX-2.5's official conditioning default is CRF 18.
"$PYTHON_BIN" "$VLLM_OMNI_ROOT/examples/offline_inference/image_to_video/image_to_video.py" \
  "${COMMON[@]}" \
  --model-class-name LTX2Pipeline \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --image "$LTX25_IMAGE" --extra-body '{"image_crf":18}' \
  --height 544 --width 960 --num-inference-steps 30 \
  --output "$LTX25_OUTPUT_DIR/full-one-stage-i2v.mp4"


# Distilled one-stage T2V: official positive-only Stage 1 without latent upsampling.
"$PYTHON_BIN" "$VLLM_OMNI_ROOT/examples/offline_inference/text_to_video/text_to_video.py" \
  "${COMMON[@]}" \
  --model-class-name LTX2DistilledOneStagePipeline \
  --height 544 --width 960 --num-inference-steps 8 \
  --output "$LTX25_OUTPUT_DIR/distilled-one-stage-t2v.mp4"

# Distilled one-stage first-frame I2V.
"$PYTHON_BIN" "$VLLM_OMNI_ROOT/examples/offline_inference/image_to_video/image_to_video.py" \
  "${COMMON[@]}" \
  --model-class-name LTX2DistilledOneStagePipeline \
  --image "$LTX25_IMAGE" --extra-body '{"image_crf":18}' \
  --height 544 --width 960 --num-inference-steps 8 \
  --output "$LTX25_OUTPUT_DIR/distilled-one-stage-i2v.mp4"
# Distilled two-stage T2V: 960x544 Stage 1, latent 2x upsample, official 3-step refinement.
"$PYTHON_BIN" "$VLLM_OMNI_ROOT/examples/offline_inference/text_to_video/text_to_video.py" \
  "${COMMON[@]}" \
  --model-class-name LTX2DistilledTwoStagePipeline \
  --height 1088 --width 1920 --num-inference-steps 8 \
  --output "$LTX25_OUTPUT_DIR/distilled-two-stage-t2v.mp4"

# Distilled two-stage first-frame I2V; positive-only, so no negative prompt is accepted.
"$PYTHON_BIN" "$VLLM_OMNI_ROOT/examples/offline_inference/image_to_video/image_to_video.py" \
  "${COMMON[@]}" \
  --model-class-name LTX2DistilledTwoStagePipeline \
  --image "$LTX25_IMAGE" --extra-body '{"image_crf":18}' \
  --height 1088 --width 1920 --num-inference-steps 8 \
  --output "$LTX25_OUTPUT_DIR/distilled-two-stage-i2v.mp4"

# Full guided two-stage T2V: 30-step guided Stage 1 plus LoRA450 3-step refinement.
"$PYTHON_BIN" "$VLLM_OMNI_ROOT/examples/offline_inference/text_to_video/text_to_video.py" \
  "${COMMON[@]}" \
  --model-class-name LTX2TwoStagePipeline \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --height 1088 --width 1920 --num-inference-steps 30 \
  --output "$LTX25_OUTPUT_DIR/full-two-stage-t2v.mp4"

# Full guided two-stage first-frame I2V.
"$PYTHON_BIN" "$VLLM_OMNI_ROOT/examples/offline_inference/image_to_video/image_to_video.py" \
  "${COMMON[@]}" \
  --model-class-name LTX2TwoStagePipeline \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --image "$LTX25_IMAGE" --extra-body '{"image_crf":18}' \
  --height 1088 --width 1920 --num-inference-steps 30 \
  --output "$LTX25_OUTPUT_DIR/full-two-stage-i2v.mp4"
