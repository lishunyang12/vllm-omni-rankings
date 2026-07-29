#!/bin/bash
# Re-run B300 I2V with the CORRECT (non-black) condition image, clean & sequential.
# Waits for GPU0 to go idle first so it never overlaps the in-flight run.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="/home/zjy/.cache/uv/archive-v0/O8B9LwgX32nDlyC71rWvk:$PYTHONPATH"
PY=/home/zjy/code/lsy/vllm-omni/.venv/bin/python

echo "########## WAIT FOR GPU0 IDLE ##########"
while true; do
  busy=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  [ "$busy" -lt 5 ] && sleep 20 && busy2=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits) && [ "$busy2" -lt 5 ] && break
  sleep 15
done
echo "GPU0 idle, starting clean I2V"

echo "########## I2V DENSE (correct image) ##########"
$PY wan22_i2v_bench.py --mode dense --image i2v_input.png --save i2v_dense.npy
echo "########## I2V FP8 SAGE (correct image) ##########"
$PY wan22_i2v_bench.py --mode fp8   --image i2v_input.png --save i2v_fp8.npy
echo "########## I2V QUALITY ##########"
$PY lpips_bench.py i2v_dense.npy i2v_fp8.npy
echo "########## I2V DONE ##########"
