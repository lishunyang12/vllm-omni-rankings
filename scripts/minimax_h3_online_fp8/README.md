# MiniMax-H3 online FP8 ablation

This suite uses runtime/online FP8 only. Every output is generated from the
released BF16 checkpoints with seed 0, 50 denoising steps, 24 FPS, 32 kHz
stereo audio, and a requested duration of 5 seconds. There are no hard quality
gates: all successful outputs are retained for comparison.

The published measurements use eight NVIDIA B300 GPUs as two concurrent
four-GPU jobs. Each job uses USP 4, Qwen3-VL tensor parallel 4, VAE patch
parallel 4 with tile mode, and the cuDNN attention backend. The FL2VA and
Ref2VA checkpoints are served separately, matching their released task split.

## Cases

| Case | DiT | Qwen3-VL text decoder | FP8 linear modules/rank |
|---|---|---|---:|
| `bf16` | BF16 | BF16 | 0 |
| `dit_fp8` | FP8 | BF16 | 0 |
| `te_mlp_l0_9` | BF16 | MLP in layers 0-9 | 20 |
| `te_mlp_l0_24` | BF16 | MLP in layers 0-24 | 50 |
| `te_mlp_all` | BF16 | MLP in layers 0-49 | 100 |
| `te_mlp_o_all` | BF16 | MLP and attention output in layers 0-49 | 150 |
| `te_all_linear` | BF16 | QKV, attention output, and MLP in layers 0-49 | 200 |
| `dit_te_mlp_all` | FP8 | MLP in layers 0-49 | 100 |
| `all_linear_fp8` | FP8 | QKV, attention output, and MLP in layers 0-49 | 200 |

The vision tower, token embeddings, RMSNorms, RoPE, video/audio VAEs, and the
H3 mixed-precision input/output heads remain at checkpoint precision.

## Tasks

- `t2va.mp4`: official text-only H3 prompt.
- `i2va.mp4`: official first-frame H3 prompt and image.
- `ref2va.mp4`: official reference video including its embedded Audio 1.

The current vLLM path rejects a reference video plus an additional independent
audio condition. Therefore the separate official Audio 2 voice-timbre reference
is not present in these `ref2va.mp4` files; results are labeled accordingly in
the aggregate report.

## Run

Run one case on four visible GPUs:

```bash
MODEL_ROOT=/path/to/MiniMax-H3 \
OMNI_ROOT=/path/to/vllm-omni \
OUTPUT_ROOT=/path/to/results \
I2VA_IMAGE=/path/to/official-first-frame.png \
bash scripts/minimax_h3_online_fp8/run_case.sh te_mlp_l0_9 0,1,2,3 8091
```

`run_case.sh` performs a 2-step shape-matched warm-up and then saves one
50-step measured output for each task, together with server logs, wall-clock
JSON, GPU telemetry, and
`ffprobe` media metadata.

Set `BENCH_TASKS` to resume only selected tasks without overwriting completed
outputs, for example `BENCH_TASKS=ref2va` or `BENCH_TASKS=t2va,i2va`.

To keep eight GPUs occupied, run the matrix in pairs of four-GPU jobs. With no
arguments this runs the complete matrix; explicit case names run only that
subset:

```bash
bash scripts/minimax_h3_online_fp8/run_matrix.sh \
  te_mlp_all te_mlp_o_all te_all_linear dit_te_mlp_all
```

After all cases finish, aggregate performance and frame-level fidelity metrics:

```bash
/path/to/vllm-omni/.venv/bin/python \
  scripts/minimax_h3_online_fp8/collect_results.py /path/to/results
```

This writes `results.csv`, `results.json`, `summary.json`, and `RESULTS.md`.
PSNR/SSIM and waveform SNR/correlation are descriptive comparisons against
BF16 with the same seed; they are not pass/fail gates or substitutes for human
evaluation of video/audio quality.

## Distributed layerwise offload smoke

The published `dlo_smoke/` artifacts are a separate two-step, 5-second T2VA
functional smoke test for online FP8 with distributed layerwise offload. It
uses four GPUs and the no-AllGather path. The smoke video is not included in
the 50-step performance or fidelity aggregates.

## Distributed layerwise offload cold-start memory

`dlo_streaming_memory/` validates quantize-and-offload streaming in the DiT
loader with the same four-GPU, no-AllGather DLO configuration. It contains the
official 5-second T2VA output, 200 ms GPU telemetry, the direct-engine summary,
and a compact before/after report in `memory_summary.json`.

The original loader retained all 208 processed FP8 DiT linear layers on the
accelerator until loading finished. Its full-lifecycle peak was 58,696 MiB per
GPU. Streaming each completed layer to CPU and releasing the completed
quantization cache reduced that peak to 37,938 MiB per GPU: 20,758 MiB (35.4%)
lower. The serving-request peak remained 26,098 MiB per GPU. Three streaming
validation runs produced identical raw frame and audio SHA256 values.

Stage all 27 matrix videos, the DLO smoke and streaming-memory artifacts,
measurements, aggregate tables, and checksums into this directory before
committing:

```bash
bash scripts/minimax_h3_online_fp8/stage_results.sh /path/to/results
```
