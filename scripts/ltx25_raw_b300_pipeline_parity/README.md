# LTX-2.5 pipeline parity on NVIDIA B300

This directory compares the four public vLLM-Omni LTX-2.5 pipeline classes
with their official `Lightricks/LTX-2` references. The matrix covers both
text-to-video (T2V) and first-frame image-to-video (I2V), for eight cases in
total.

The two backends use different checkpoint packaging:

- vLLM-Omni loads a materialized `Lightricks/LTX-2.5-Diffusers` checkpoint.
- The official runtime loads the raw split artifacts from `Lightricks/LTX-2.5`.

## Current BF16 cuDNN release parity

The gallery now leads with the latest strict correctness run at vLLM-Omni PR
head `29a24da58`. Both implementations ran sequentially on the same NVIDIA
B300 with BF16, the same PyTorch SDPA cuDNN kernel selection, seed 42, official
schedules, and identical conditioning. These compact 25-frame pairs are under
[`results-release/`](results-release/); exact metrics and pinned revisions are
in [`results-release/metrics.json`](results-release/metrics.json).

| Public path | Task | SSIM mean / min | PSNR | LPIPS |
|---|---|---:|---:|---:|
| `LTX2Pipeline` | T2V | 0.999652 / 0.999564 | 58.80 dB | 0.0000367 |
| `LTX2Pipeline` | I2V | 0.999589 / 0.999480 | 62.37 dB | 0.0000576 |
| `LTX2TwoStagePipeline` | T2V | 0.999588 / 0.999485 | 56.40 dB | 0.0000687 |
| `LTX2TwoStagePipeline` | I2V | 0.999585 / 0.999475 | 62.10 dB | 0.0000627 |
| `LTX2DistilledTwoStagePipeline` | T2V | 0.997186 / 0.996164 | 46.04 dB | 0.0034523 |
| `LTX2DistilledTwoStagePipeline` | I2V | 0.996489 / 0.995810 | 46.29 dB | 0.0037874 |

All six public native-reference pairs exceed mean SSIM 0.99. The Full
two-stage I2V pair was regenerated after matching the official two-step FP32
conditioned-noising order; the checked-in score is 0.999585.

## Long-form stress matrix

| Mode | vLLM-Omni class | Official reference | T2V | I2V | Final output and schedule |
|---|---|---|:---:|:---:|---|
| Full one-stage | `LTX2Pipeline` | `TI2VidOneStagePipeline` | ✓ | ✓ | 960×544, 481 frames, 30 steps |
| Distilled one-stage | `LTX2DistilledOneStagePipeline` | `DistilledPipeline` Stage 1 extraction via `official_distilled_stage1.py` | ✓ | ✓ | 960×544, 481 frames, official 8-step Stage 1 |
| Distilled two-stage | `LTX2DistilledTwoStagePipeline` | `DistilledPipeline` | ✓ | ✓ | 1920×1088, 481 frames, official 8+3 schedule |
| Full guided two-stage | `LTX2TwoStagePipeline` | `TI2VidTwoStagesPipeline` | ✓ | ✓ | 1920×1088, 481 frames, official 30+3 schedule with LoRA450 |

Every case uses the corresponding pinned official pipeline/configuration and
the selected 960x544 or 1920x1088 high-load release shape. The benchmark
extends generation to 481 frames at 24 FPS (about 20 seconds) as a stress
workload; neither the duration nor dimensions are presented as official hard
maximums. All cases use the official complex
quickstart prompt and seeds 42, 43, and 44. I2V uses the same first-frame input
with CRF 18. Full paths use the official negative prompt; distilled paths are
positive-only.

The distilled one-stage official reference is deliberately narrow: it runs
the prompt/image conditioning and Stage 1 code extracted from the pinned
official `DistilledPipeline`, including FP32 sigmas and its ancestral sampler,
then decodes the Stage 1 video and audio with the official decoders. It does
not run latent upsampling or Stage 2 refinement.

## Public vLLM-Omni usage

[`run_vllm_examples.sh`](run_vllm_examples.sh) contains runnable offline
commands for all four classes in both T2V and I2V form. The equivalent online
form selects the class directly:

```bash
vllm serve /path/to/LTX-2.5-Diffusers \
  --omni \
  --model-class-name LTX2DistilledTwoStagePipeline
```

Replace the class with `LTX2Pipeline`, `LTX2DistilledOneStagePipeline`,
`LTX2DistilledTwoStagePipeline`, or `LTX2TwoStagePipeline` to select the
desired final pipeline.

## Warm online E2E benchmark (controlled workload)

[`run_warm_e2e.py`](run_warm_e2e.py) is the serving-latency harness. It is
separate from the 1080p/481-frame quality showcase above. Its controlled
workload fixes the official quickstart prompt, seed 42, 768x512 output, 121
frames, 24 FPS, and the same I2V first frame with CRF 18 for every class. Full
pipelines keep their official 30-step schedule and distilled pipelines keep
their official 8-step schedule; two-stage refinement remains pipeline-defined.

The harness launches the four classes sequentially so they use the same GPU.
For each class, one resident server handles both tasks in this order:

1. One T2V warm-up request, excluded from all latency fields.
2. Configurable sequential T2V timed repeats.
3. One I2V warm-up request, excluded from all latency fields.
4. Configurable sequential I2V timed repeats.

Warm E2E is the client wall-clock time from `POST /v1/videos/sync` until the
complete MP4 response is received over a persistent loopback HTTP session.
Server startup, model loading, warm-up, response validation, and optional file
writes are excluded. The raw result file records all three warm samples plus
mean/min/max/population-stdev.

| Pipeline | T2V warm E2E | I2V warm E2E |
|---|---:|---:|
| `LTX2Pipeline` | 32.94 s | 33.13 s |
| `LTX2TwoStagePipeline` | 11.74 s | 12.18 s |
| `LTX2DistilledOneStagePipeline` | 4.74 s | 4.77 s |
| `LTX2DistilledTwoStagePipeline` | 4.05 s | 4.65 s |

These values are the mean of three timed warm requests on NVIDIA B300 with
BF16 cuDNN. Exact samples and validated 768x512, 121-frame MP4 responses are
under [`results-warm-e2e/`](results-warm-e2e/). The controlled table is the
cross-pipeline latency comparison; the 481-frame section remains the separate
maximum-quality stress gallery because its one-stage and two-stage output
resolutions intentionally differ.

Run all four public classes on one B300:

```bash
CUDA_VISIBLE_DEVICES=0 \
/path/to/vllm-omni/.venv/bin/python \
  scripts/ltx25_raw_b300_pipeline_parity/run_warm_e2e.py \
  --model /path/to/LTX-2.5-Diffusers \
  --vllm-omni-root /path/to/vllm-omni \
  --vllm-bin /path/to/vllm-omni/.venv/bin/vllm \
  --repeats 3 \
  --attention-backend CUDNN_ATTN \
  --output-dir scripts/ltx25_raw_b300_pipeline_parity/results-warm-e2e
```

The default pipeline order is `LTX2Pipeline`, `LTX2TwoStagePipeline`,
`LTX2DistilledOneStagePipeline`, then `LTX2DistilledTwoStagePipeline`. Narrow
the run with `--pipelines` and/or `--tasks`. For example, this keeps one
distilled two-stage runtime resident while benchmarking both tasks five times:

```bash
CUDA_VISIBLE_DEVICES=0 \
/path/to/vllm-omni/.venv/bin/python \
  scripts/ltx25_raw_b300_pipeline_parity/run_warm_e2e.py \
  --model /path/to/LTX-2.5-Diffusers \
  --vllm-omni-root /path/to/vllm-omni \
  --vllm-bin /path/to/vllm-omni/.venv/bin/vllm \
  --pipelines LTX2DistilledTwoStagePipeline \
  --tasks t2v i2v \
  --repeats 5
```

Add `--dry-run` to either command to print the four serve commands and exact
multipart request contract without starting a server or touching a GPU.

## Reproduce the B300 comparison

The official and vLLM-Omni runners have separate model roots. Run them on
separate free GPUs so both sides can proceed concurrently:

```bash
CUDA_VISIBLE_DEVICES=7 \
VLLM_OMNI_PYTHON=/path/to/vllm-omni/.venv/bin/python \
python scripts/ltx25_raw_b300_pipeline_parity/run_pipeline_parity.py \
  --backend official \
  --model-root /path/to/LTX-2.5 \
  --omni-model-root /path/to/LTX-2.5-Diffusers \
  --official-root /path/to/LTX-2 \
  --output-dir scripts/ltx25_raw_b300_pipeline_parity/results-final
```

```bash
CUDA_VISIBLE_DEVICES=5 \
VLLM_OMNI_PYTHON=/path/to/vllm-omni/.venv/bin/python \
python scripts/ltx25_raw_b300_pipeline_parity/run_pipeline_parity.py \
  --backend omni \
  --model-root /path/to/LTX-2.5 \
  --omni-model-root /path/to/LTX-2.5-Diffusers \
  --vllm-omni-root /path/to/vllm-omni \
  --output-dir scripts/ltx25_raw_b300_pipeline_parity/results-final
```

The runners write separate manifests, logs, and MP4s under `results-final/`.
They resume completed outputs by default. Narrow a run with `--mode`,
`--modality`, or `--seeds`.

After both sides complete, compute alignment metrics from that result path:

```bash
/path/to/vllm-omni/.venv/bin/python \
  scripts/ltx25_raw_b300_pipeline_parity/compute_metrics.py \
  --results-dir scripts/ltx25_raw_b300_pipeline_parity/results-final
```

SSIM covers every decoded frame. PSNR reports the mean over finite per-frame
values. LPIPS/AlexNet covers 32 uniformly sampled decoded frames and records
the exact indices in `results-final/metrics.json`. Audio is available in every
MP4 for playback, but this gallery does not assign an audio similarity score.

This gallery is a production-backend stress comparison: both runtimes select
their native cuDNN attention path on B300. The strict reduced-shape correctness
guard in vLLM-Omni separately forces the same PyTorch SDPA path and connector
weights on both sides. Consequently, the decoded-video metrics here measure
long-run trajectory similarity; they are not the CI acceptance threshold.

## Authority and provenance

- Official checkpoint packaging: raw split `Lightricks/LTX-2.5` artifacts
- vLLM-Omni checkpoint packaging: materialized `Lightricks/LTX-2.5-Diffusers`
- Official runtime: pinned commit recorded in `manifest-official.json`
- vLLM-Omni runtime: pinned commit recorded in `manifest-omni.json`
- Workload: official quickstart prompt and schedules, extended to 481 frames
- Hardware: NVIDIA B300 SXM6
- Attention: native cuDNN attention selected on B300
