#!/bin/bash
# Clean SEQUENTIAL B300 re-run: T2V + I2V, dense then fp8, one at a time (no concurrency).
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="/home/zjy/.cache/uv/archive-v0/O8B9LwgX32nDlyC71rWvk:$PYTHONPATH"
PY=/home/zjy/code/lsy/vllm-omni/.venv/bin/python

echo "########## T2V DENSE ##########"
$PY wan22_bench.py --mode dense --save t2v_dense.npy
echo "########## T2V FP8 SAGE ##########"
$PY wan22_bench.py --mode fp8   --save t2v_fp8.npy
echo "########## T2V QUALITY ##########"
$PY lpips_bench.py t2v_dense.npy t2v_fp8.npy

# derive I2V first-frame condition from T2V dense frame 0 (same as prior B300 run)
$PY - <<'PY'
import numpy as np
from PIL import Image
a = np.load("t2v_dense.npy").squeeze()
Image.fromarray((a[0]).clip(0,255).astype("uint8")).save("i2v_input.png")
print("derived i2v_input.png from t2v_dense.npy frame 0", a.shape)
PY

echo "########## I2V DENSE ##########"
$PY wan22_i2v_bench.py --mode dense --image i2v_input.png --save i2v_dense.npy
echo "########## I2V FP8 SAGE ##########"
$PY wan22_i2v_bench.py --mode fp8   --image i2v_input.png --save i2v_fp8.npy
echo "########## I2V QUALITY ##########"
$PY lpips_bench.py i2v_dense.npy i2v_fp8.npy

echo "########## ALL DONE ##########"
