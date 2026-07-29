#!/bin/bash
# Wan2.2-I2V-A14B SAGE benchmark for B200/GB200 (SM100).
# Standard load: 1280x720 / 81f / 50 steps, torch.compile, CFG on both experts.
# Runs dense baseline + fp8 SAGE, reports fp8 latency and LPIPS vs dense.
#
# Prereqs (in the activated venv):
#   source ~/omni-env/bin/activate
#   export CUDA_HOME=$HOME/cuda13 PATH=$HOME/cuda13/bin:$PATH
#   uv pip install lpips
#
# Input image: set IMG=<path.jpg>, or drop an `i2v_input.png` next to this script,
# or let it derive one from an existing T2V `dense.npy` (frame 0).
#
#   IMG=my.jpg bash run_i2v_b200.sh          # your own image
#   bash run_i2v_b200.sh                      # auto: i2v_input.png or dense.npy frame0
set -e
GPU="${GPU:-0}"
IMG="${IMG:-i2v_input.png}"

if [ ! -f "$IMG" ]; then
  python - <<'PY'
import os, numpy as np
from PIL import Image
src = next((f for f in ("dense.npy", "fp8.npy", "int8.npy") if os.path.exists(f)), None)
if src is None:
    raise SystemExit("No input image and no dense.npy to derive one — set IMG=<path.jpg> and rerun.")
a = np.load(src).squeeze()
Image.fromarray((a[0] * 255).clip(0, 255).astype("uint8")).save("i2v_input.png")
print("Derived i2v_input.png from", src, "frame 0")
PY
  IMG="i2v_input.png"
fi
echo "Input image: $IMG"

echo "########## I2V DENSE (baseline) ##########"
CUDA_VISIBLE_DEVICES=$GPU python wan22_i2v_bench.py --mode dense --image "$IMG" --save i2v_dense.npy

echo "########## I2V FP8 SAGE ##########"
CUDA_VISIBLE_DEVICES=$GPU python wan22_i2v_bench.py --mode fp8 --image "$IMG" --save i2v_fp8.npy

echo ""
echo "########## QUALITY: fp8 vs dense ##########"
CUDA_VISIBLE_DEVICES=$GPU python lpips_bench.py i2v_dense.npy i2v_fp8.npy
echo ""
echo "Done. Latency per run above ([i2v] generation ...); LPIPS/PSNR just above."
