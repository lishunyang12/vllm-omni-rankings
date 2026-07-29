#!/bin/bash
# Wan2.2-A14B SAGE benchmark for B200/GB200 (SM100).
# Standard load: 1280x720 / 81f / 50 steps, torch.compile, CFG on both experts.
# Runs dense baseline, then fp8 + int8, and reports LPIPS/PSNR of each vs dense.
#
# Prereqs (run once, in your activated venv):
#   source ~/omni-env/bin/activate
#   export CUDA_HOME=$HOME/cuda13 PATH=$HOME/cuda13/bin:$PATH
#   uv pip install lpips
#
# Then: bash run_bench_b200.sh
set -e
GPU="${GPU:-0}"
run() { CUDA_VISIBLE_DEVICES=$GPU python wan22_bench.py --mode "$1" --save "$2"; }

echo "########## DENSE (baseline) ##########"
run dense dense.npy

echo "########## FP8 SAGE ##########"
run fp8 fp8.npy

echo "########## INT8 SAGE (SM100 only) ##########"
run int8 int8.npy

echo ""
echo "########## QUALITY vs DENSE ##########"
echo "=== fp8 vs dense ===";  CUDA_VISIBLE_DEVICES=$GPU python lpips_bench.py dense.npy fp8.npy
echo "=== int8 vs dense ==="; CUDA_VISIBLE_DEVICES=$GPU python lpips_bench.py dense.npy int8.npy
echo ""
echo "Done. Latency printed per-run above ([bench] generation ...); LPIPS/PSNR just above."
