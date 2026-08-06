# MiniMax-H3 TRTLLM vs FA4 E2E

Matched MiniMax-H3 `FL2VA` T2VA outputs on four NVIDIA B300 or B200 GPUs. The
clips below are the steady-state second request after one regional-compile
warmup.

## B300 stable matrix

Timings are the median of five measured requests after one warmup; each mode
starts one server and sends all six requests to it. The linked clip is request
2. Every run uses the same prompt, seed, 1248x768 canvas, 209 frames, 24 FPS,
50 denoise steps, and 32 kHz stereo audio. The MiniMax-H3 token refiner remains
dense in every optimized run. Speedup is relative to dense `TRTLLM_ATTN`.

| Mode | Backend | SAGE | Skip-Softmax | Median diffuse | CV | Speedup vs dense | Video | Raw record |
|---|---|---|---|---:|---:|---:|---|---|
| `recipe_flash` | `FLASH_ATTN` | — | — | 74.972 s | 0.05% | 1.017× | [MP4](./b300_20260805_v6/recipe_flash.mp4) | [JSON](./b300_20260805_v6/recipe_flash.json) |
| `trtllm_dense` | `TRTLLM_ATTN` | — | — | 76.275 s | 0.33% | 1.000× | [MP4](./b300_20260805_v6/trtllm_dense.mp4) | [JSON](./b300_20260805_v6/trtllm_dense.json) |
| `sage_fp8` | `TRTLLM_ATTN` | FP8 | — | 63.578 s | 0.18% | 1.200× | [MP4](./b300_20260805_v6/sage_fp8.mp4) | [JSON](./b300_20260805_v6/sage_fp8.json) |
| `skip_softmax_005_gate090` | `TRTLLM_ATTN` | — | threshold 0.05, `disabled_until_timestep=0.90` (21/49 steps) | 73.069 s | 0.09% | 1.044× | [MP4](./b300_20260805_v6/skip_softmax_005_gate090.mp4) | [JSON](./b300_20260805_v6/skip_softmax_005_gate090.json) |
| `skip_softmax_010_gate090` | `TRTLLM_ATTN` | — | threshold 0.10, `disabled_until_timestep=0.90` (21/49 steps) | 72.808 s | 0.32% | 1.048× | [MP4](./b300_20260805_v6/skip_softmax_010_gate090.mp4) | [JSON](./b300_20260805_v6/skip_softmax_010_gate090.json) |
| `skip_softmax_03_gate090` | `TRTLLM_ATTN` | — | threshold 0.30, `disabled_until_timestep=0.90` (21/49 steps) | 71.505 s | 0.26% | 1.067× | [MP4](./b300_20260805_v6/skip_softmax_03_gate090.mp4) | [JSON](./b300_20260805_v6/skip_softmax_03_gate090.json) |
| `skip_softmax_05_gate090` | `TRTLLM_ATTN` | — | threshold 0.50, `disabled_until_timestep=0.90` (21/49 steps) | 71.200 s | 0.22% | 1.071× | [MP4](./b300_20260805_v6/skip_softmax_05_gate090.mp4) | [JSON](./b300_20260805_v6/skip_softmax_05_gate090.json) |
| `skip_softmax_005_gate095` | `TRTLLM_ATTN` | — | threshold 0.05, `disabled_until_timestep=0.95` (30/49 steps) | 72.111 s | 0.24% | 1.058× | [MP4](./b300_20260805_v6/skip_softmax_005_gate095.mp4) | [JSON](./b300_20260805_v6/skip_softmax_005_gate095.json) |
| `skip_softmax_010_gate095` | `TRTLLM_ATTN` | — | threshold 0.10, `disabled_until_timestep=0.95` (30/49 steps) | 71.343 s | 0.25% | 1.069× | [MP4](./b300_20260805_v6/skip_softmax_010_gate095.mp4) | [JSON](./b300_20260805_v6/skip_softmax_010_gate095.json) |
| `skip_softmax_03_gate095` | `TRTLLM_ATTN` | — | threshold 0.30, `disabled_until_timestep=0.95` (30/49 steps) | 69.698 s | 0.14% | 1.094× | [MP4](./b300_20260805_v6/skip_softmax_03_gate095.mp4) | [JSON](./b300_20260805_v6/skip_softmax_03_gate095.json) |
| `skip_softmax_05_gate095` | `TRTLLM_ATTN` | — | threshold 0.50, `disabled_until_timestep=0.95` (30/49 steps) | 69.193 s | 0.19% | 1.102× | [MP4](./b300_20260805_v6/skip_softmax_05_gate095.mp4) | [JSON](./b300_20260805_v6/skip_softmax_05_gate095.json) |
| `skip_softmax_005_gate099` | `TRTLLM_ATTN` | — | threshold 0.05, `disabled_until_timestep=0.99` (43/49 steps) | 70.446 s | 0.07% | 1.083× | [MP4](./b300_20260805_v6/skip_softmax_005_gate099.mp4) | [JSON](./b300_20260805_v6/skip_softmax_005_gate099.json) |
| `skip_softmax_010_gate099` | `TRTLLM_ATTN` | — | threshold 0.10, `disabled_until_timestep=0.99` (43/49 steps) | 69.524 s | 0.22% | 1.097× | [MP4](./b300_20260805_v6/skip_softmax_010_gate099.mp4) | [JSON](./b300_20260805_v6/skip_softmax_010_gate099.json) |
| `skip_softmax_03_gate099` | `TRTLLM_ATTN` | — | threshold 0.30, `disabled_until_timestep=0.99` (43/49 steps) | 67.214 s | 0.28% | 1.135× | [MP4](./b300_20260805_v6/skip_softmax_03_gate099.mp4) | [JSON](./b300_20260805_v6/skip_softmax_03_gate099.json) |
| `skip_softmax_05_gate099` | `TRTLLM_ATTN` | — | threshold 0.50, `disabled_until_timestep=0.99` (43/49 steps) | 65.975 s | 0.17% | 1.156× | [MP4](./b300_20260805_v6/skip_softmax_05_gate099.mp4) | [JSON](./b300_20260805_v6/skip_softmax_05_gate099.json) |
| `sage_fp8_skip_005_gate090` | `TRTLLM_ATTN` | FP8 | threshold 0.05, `disabled_until_timestep=0.90` (21/49 steps) | 63.045 s | 0.25% | 1.210× | [MP4](./b300_20260805_v6/sage_fp8_skip_005_gate090.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_005_gate090.json) |
| `sage_fp8_skip_010_gate090` | `TRTLLM_ATTN` | FP8 | threshold 0.10, `disabled_until_timestep=0.90` (21/49 steps) | 62.726 s | 0.27% | 1.216× | [MP4](./b300_20260805_v6/sage_fp8_skip_010_gate090.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_010_gate090.json) |
| `sage_fp8_skip_03_gate090` | `TRTLLM_ATTN` | FP8 | threshold 0.30, `disabled_until_timestep=0.90` (21/49 steps) | 61.496 s | 0.17% | 1.240× | [MP4](./b300_20260805_v6/sage_fp8_skip_03_gate090.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_03_gate090.json) |
| `sage_fp8_skip_05_gate090` | `TRTLLM_ATTN` | FP8 | threshold 0.50, `disabled_until_timestep=0.90` (21/49 steps) | 60.981 s | 0.18% | 1.251× | [MP4](./b300_20260805_v6/sage_fp8_skip_05_gate090.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_05_gate090.json) |
| `sage_fp8_skip_005_gate095` | `TRTLLM_ATTN` | FP8 | threshold 0.05, `disabled_until_timestep=0.95` (30/49 steps) | 62.783 s | 0.27% | 1.215× | [MP4](./b300_20260805_v6/sage_fp8_skip_005_gate095.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_005_gate095.json) |
| `sage_fp8_skip_010_gate095` | `TRTLLM_ATTN` | FP8 | threshold 0.10, `disabled_until_timestep=0.95` (30/49 steps) | 62.109 s | 0.10% | 1.228× | [MP4](./b300_20260805_v6/sage_fp8_skip_010_gate095.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_010_gate095.json) |
| `sage_fp8_skip_03_gate095` | `TRTLLM_ATTN` | FP8 | threshold 0.30, `disabled_until_timestep=0.95` (30/49 steps) | 60.583 s | 0.23% | 1.259× | [MP4](./b300_20260805_v6/sage_fp8_skip_03_gate095.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_03_gate095.json) |
| `sage_fp8_skip_05_gate095` | `TRTLLM_ATTN` | FP8 | threshold 0.50, `disabled_until_timestep=0.95` (30/49 steps) | 60.086 s | 0.43% | 1.269× | [MP4](./b300_20260805_v6/sage_fp8_skip_05_gate095.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_05_gate095.json) |
| `sage_fp8_skip_005_gate099` | `TRTLLM_ATTN` | FP8 | threshold 0.05, `disabled_until_timestep=0.99` (43/49 steps) | 62.284 s | 0.11% | 1.225× | [MP4](./b300_20260805_v6/sage_fp8_skip_005_gate099.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_005_gate099.json) |
| `sage_fp8_skip_010_gate099` | `TRTLLM_ATTN` | FP8 | threshold 0.10, `disabled_until_timestep=0.99` (43/49 steps) | 61.781 s | 0.28% | 1.235× | [MP4](./b300_20260805_v6/sage_fp8_skip_010_gate099.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_010_gate099.json) |
| `sage_fp8_skip_03_gate099` | `TRTLLM_ATTN` | FP8 | threshold 0.30, `disabled_until_timestep=0.99` (43/49 steps) | 59.741 s | 0.42% | 1.277× | [MP4](./b300_20260805_v6/sage_fp8_skip_03_gate099.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_03_gate099.json) |
| `sage_fp8_skip_05_gate099` | `TRTLLM_ATTN` | FP8 | threshold 0.50, `disabled_until_timestep=0.99` (43/49 steps) | 58.322 s | 0.22% | 1.308× | [MP4](./b300_20260805_v6/sage_fp8_skip_05_gate099.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_05_gate099.json) |

All 27 modes passed the timing gate: five measured requests, CV below 2%,
and span/median below 5%. The largest observed CV was 0.43% and the largest
span/median was 1.11%. `FLASH_ATTN` was 1.7% faster than dense
`TRTLLM_ATTN` in this controlled run. FP8 SAGE delivered 1.200×, and the most
aggressive measured combination delivered 1.308×.

`disabled_until_timestep` is compared against the shift-12 video sigma, not
the fraction of completed denoise steps. Values 0.99, 0.95, and 0.90 first
enable Skip-Softmax at steps 6, 19, and 28, leaving it enabled for 43, 30,
and 21 of the 49 steps respectively.

The stage timer synchronizes CUDA before and after `diffuse`. The run used
physical GPUs 0, 1, 4, and 5 with a balanced CPU set. Across 4,180 telemetry
samples per GPU, the maximum temperature was 77°C and neither the thermal
slowdown flag nor counter changed. See the [timing audit](./b300_20260805_v6/timing_audit.json),
[thermal audit](./b300_20260805_v6/thermal_audit.json), and
[kernel profile summary](./b300_20260805_v6/kernel_profiles.md). All 27 clips
also passed a [full ffmpeg decode audit](./b300_20260805_v6/media_audit.json).

### B300 kernel-level profiles

Nsight Systems profiled the second `diffuse` request in separate runs, so the
instrumentation does not affect the E2E table above.

| Mode | Rank-0 projected diffuse | Attention GPU time (4-GPU total) | Attention share | Attention speedup vs dense | Primary FMHA average |
|---|---:|---:|---:|---:|---:|
| `trtllm_dense` | 75.383 s | 161.747 s | 55.0% | 1.000× | 15.879 ms |
| `sage_fp8` | 63.054 s | 121.498 s | 50.0% | 1.331× | 12.062 ms |
| `skip_softmax_05_gate099` | 65.797 s | 132.495 s | 52.2% | 1.221× | 13.117 ms |
| `sage_fp8_skip_05_gate099` | 58.072 s | 105.466 s | 47.4% | 1.534× | 10.220 ms |

The SAGE-only trace contains 9,800 main-DiT SAGE FMHA calls and 386 short
dense token-refiner calls. The Skip-Softmax-only trace contains exactly 8,600
sparse calls (43 steps × 50 blocks × 4 GPUs) and 1,586 dense calls (six main
steps plus the refiner). The exact kernel names and raw CSV paths are available
with the [kernel profile artifacts](./b300_20260805_v6/kernel_profiles.json).

## B200 INT8 results

These runs use four NVIDIA B200 GPUs and the same prompt, seed, output shape,
denoise steps, and parallel configuration as the B300 runs. The table reports
request 2 after one warmup.

| Backend | SAGE | Skip-Softmax | Steady diffusion | Video | Raw record |
|---|---|---|---:|---|---|
| `TRTLLM_ATTN` | INT8 | — | 75.250 s | [MP4](./minimax_h3_t2va_trtllm_sage_int8_4xb200_8p7s_50step.mp4) | [JSON](./sage_int8_b200_summary.json) |
| `TRTLLM_ATTN` | INT8 | threshold 0.3, `disabled_until_timestep=0.90` (21/49 steps) | 72.332 s | [MP4](./minimax_h3_t2va_trtllm_sage_int8_skip_4xb200_8p7s_50step.mp4) | [JSON](./sage_int8_skip_b200_summary.json) |
| `TRTLLM_ATTN` | INT8 | threshold 0.3, `disabled_until_timestep=0.99` (43/49 steps) | 70.131 s | [MP4](./minimax_h3_t2va_trtllm_sage_int8_skip_t03_gate099_4xb200_8p7s_50step.mp4) | [JSON](./sage_int8_skip_03_gate099_b200_summary.json) |
| `TRTLLM_ATTN` | INT8 | threshold 0.5, `disabled_until_timestep=0.99` (43/49 steps) | 68.237 s | [MP4](./minimax_h3_t2va_trtllm_sage_int8_skip_t05_gate099_4xb200_8p7s_50step.mp4) | [JSON](./sage_int8_skip_05_gate099_b200_summary.json) |

The repeated steady-state diffusion times were 75.250 s and 75.228 s for
INT8 SAGE; 72.332 s and 72.401 s for threshold 0.3 at 0.90; 70.131 s and
70.084 s for threshold 0.3 at 0.99; and 68.237 s and 68.225 s for threshold
0.5 at 0.99. Each configuration produced identical frame and audio hashes
across its two steady-state runs.

Diffusion time is the comparable GPU metric; end-to-end wall time is distorted
by large inter-process video output transfers in this vLLM-Omni build.

[`results_manifest.json`](./results_manifest.json) is the machine-readable
index of the stable B300 configurations, timings, videos, and summaries. The
same rows are available as [CSV](./b300_20260805_v6/results.csv).

## B300 environment

| Item | Value |
|---|---|
| GPU | 4x NVIDIA B300 SXM6 AC, SM103, 267.7 GiB each |
| Physical GPU indices | 0, 1, 4, 5 |
| Container CPU set | 0-27, 56-83, 112-139, 168-195 |
| Driver | 610.43.02 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| FlashInfer | 0.6.16.post1 |
| Parallelism | TP1, Ulysses4, Ring1, text-encoder TP1, VAE tile4 |
| Compile | Regional compile enabled; `--enforce-eager` disabled |
| Model | MiniMax-H3 `FL2VA` |
| Prompt seed | 1101 |
| Timing | 1 warmup + 5 measured requests per mode; median reported |

## B200 environment

The B200 runs used 4x NVIDIA B200 (SM100, 178.3 GiB each), driver 595.58.03,
PyTorch 2.11.0+cu130, and FlashInfer 0.6.16.post1. All other model, compile,
and parallel settings match the B300 environment above.

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
