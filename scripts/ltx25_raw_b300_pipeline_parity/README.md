# LTX-2.5 pipeline parity on NVIDIA B300

This directory compares the four public vLLM-Omni LTX-2.5 pipeline classes
with their official `Lightricks/LTX-2` references. The matrix covers both
text-to-video (T2V) and first-frame image-to-video (I2V), for eight cases in
total.

The two backends use different checkpoint packaging:

- vLLM-Omni loads a materialized `Lightricks/LTX-2.5-Diffusers` checkpoint.
- The official runtime loads the raw split artifacts from `Lightricks/LTX-2.5`.

## Matrix

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
