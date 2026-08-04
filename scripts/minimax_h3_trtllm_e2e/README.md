# MiniMax-H3 TRTLLM vs FA4 E2E

Matched MiniMax-H3 `FL2VA` T2VA outputs on four NVIDIA B300 GPUs. The clips
below are the steady-state second request after one regional-compile warmup.

## Test data

| Backend | Diffusion | End-to-end wall time | Video |
|---|---:|---:|---|
| `FLASH_ATTN` (FA4) | 83.854 s | 88.558 s | [MP4](./minimax_h3_t2va_fa4_4xb300_8p7s_50step.mp4) |
| `TRTLLM_ATTN` (dense BF16, packed) | **71.990 s** | **76.176 s** | [MP4](./minimax_h3_t2va_trtllm_dense_bf16_4xb300_8p7s_50step.mp4) |

Both outputs are 1248x768, 209 frames, 24 FPS, with 32 kHz stereo audio.
Dense BF16 uses neither Skip-Softmax nor attention quantization.

The raw records are available as [FA4 JSON](./fa4_summary.json) and
[TRTLLM JSON](./trtllm_dense_summary.json).

## Environment

| Item | Value |
|---|---|
| GPU | 4x NVIDIA B300 SXM6 AC, SM103, 267.7 GiB each |
| Driver | 610.43.02 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| FlashInfer | 0.6.16.post1 |
| Parallelism | TP1, Ulysses4, Ring1, text-encoder TP1, VAE tile4 |
| Compile | Regional compile enabled; `--enforce-eager` disabled |
| Model | MiniMax-H3 `FL2VA` |
| Prompt seed | 1101 |

## Serve

Set `MODEL_ROOT` to the directory containing the `FL2VA/` and `Ref2VA/`
partitions. The default is the dense BF16 TRTLLM path for datacenter
Blackwell. `TRTLLM_ATTN` is supported on SM100/SM103 only.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export GPU_IDS=0,1,2,3
export PORT=8091

./serve_b300.sh
```

For the FA4 baseline, start a separate server with:

```bash
ATTENTION_BACKEND=FLASH_ATTN ./serve_b300.sh
```

The server command is also stored in [`serve_b300.sh`](./serve_b300.sh).

## Curl request 1: T2VA

After the server is ready, run:

```bash
./curl_t2va.sh minimax_h3_t2va.mp4
```

The exact multipart request is stored in
[`curl_t2va.sh`](./curl_t2va.sh). It uses the same prompt, shape, duration,
steps, flow shifts, and seed as the recorded comparison.

## Reproduce the measured A/B run

[`run_attention_ab.py`](./run_attention_ab.py) runs the offline matched
comparison and writes a JSON summary plus MP4 output. It requires a local
vLLM-Omni checkout and a local `FL2VA` checkpoint:

```bash
export MODEL_DIR=/path/to/MiniMax-H3/FL2VA
export PYTHONPATH=/path/to/vllm-omni

CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODE=recipe_flash \
MODEL_DIR="${MODEL_DIR}" \
OUTPUT_ROOT=./outputs/recipe_flash \
NUM_RUNS=2 \
python run_attention_ab.py

CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODE=trtllm_dense \
MODEL_DIR="${MODEL_DIR}" \
OUTPUT_ROOT=./outputs/trtllm_dense \
NUM_RUNS=2 \
python run_attention_ab.py
```

Run 1 is the warmup and run 2 is the steady-state measurement. The uploaded
clips were generated from the current packed H3 optimization worktree, so
they are combined-change evidence rather than an isolated run of any single
upstream pull request.
