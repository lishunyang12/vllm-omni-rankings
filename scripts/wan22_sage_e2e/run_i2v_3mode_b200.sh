#!/bin/bash
# B200 (SM100) I2V-only: dense + fp8 SAGE + int8 SAGE, same image+prompt+seed.
# Then LPIPS (all frames AND first 16) and side-by-side compare frames.
# int8 SAGE needs SM100 kernels -> B200 only (B300/SM103 has none).
#
# Prereqs on B200:
#   source ~/omni-env/bin/activate
#   export CUDA_HOME=$HOME/cuda13 PATH=$HOME/cuda13/bin:$PATH
#   curl -L -o i2v_input.png "https://picsum.photos/id/237/1280/720"
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

IMG="${IMG:-i2v_input.png}"
PROMPT="The puppy blinks and gently turns its head, breathing softly, warm natural light."
PY=python

[ -f "$IMG" ] || { echo "missing $IMG — run the curl download first"; exit 1; }
echo "image=$IMG  prompt=\"$PROMPT\""

echo "########## I2V DENSE ##########"
$PY wan22_i2v_bench.py --mode dense --image "$IMG" --prompt "$PROMPT" --save i2v_dense.npy
echo "########## I2V FP8 SAGE ##########"
$PY wan22_i2v_bench.py --mode fp8   --image "$IMG" --prompt "$PROMPT" --save i2v_fp8.npy
echo "########## I2V INT8 SAGE ##########"
$PY wan22_i2v_bench.py --mode int8  --image "$IMG" --prompt "$PROMPT" --save i2v_int8.npy

echo "########## QUALITY: fp8 vs dense ##########"
$PY lpips_bench.py i2v_dense.npy i2v_fp8.npy  --first 16
echo "########## QUALITY: int8 vs dense ##########"
$PY lpips_bench.py i2v_dense.npy i2v_int8.npy --first 16

echo "########## COMPARE FRAMES (ref | fp8 | int8) ##########"
$PY export_compare_frames.py i2v_dense.npy i2v_fp8.npy i2v_int8.npy --frames 0 4 8 15 --out b200_i2v_cmp

echo "########## DONE ##########"
