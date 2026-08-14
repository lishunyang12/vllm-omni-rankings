# LTX-2.5 official parity on NVIDIA B300

This page compares all four public vLLM-Omni LTX-2.5 pipelines with the
corresponding pinned Lightricks implementation for both text-to-video and
first-frame image-to-video.

Public page:

<https://lishunyang12.github.io/vllm-omni-rankings/scripts/ltx25_raw_b300_pipeline_parity/>

## Fixed comparison contract

- Prompt: the long official Lightricks quickstart dialogue prompt
- Output: 1920x1088, 121 frames, 24 FPS, 5.0417 seconds, 48 kHz stereo audio
- Seed: 42
- Attention: BF16 cuDNN on NVIDIA B300
- I2V: the official seed-42 first frame with CRF 18
- Precision: native decoded frames 0 through 23 only, exactly the first second
- Playback: every published Official and vLLM-Omni video remains the complete
  five-second result
- Latency: resident online vLLM-Omni server, one excluded warm-up, then timed
  loopback requests through complete MP4 receipt

The prompt is intentionally difficult. It combines long dialogue, lip sync,
repeated sniff sounds, subtle facial expression, fixed gaze, a static camera,
and shallow depth of field. The same prompt is used for every class and task,
so cross-pipeline differences remain visible.

## Pipeline matrix

| Public vLLM-Omni class | Official reference | T2V | I2V | Exact schedule |
|---|---|:---:|:---:|---|
| LTX2Pipeline | TI2VidOneStagePipeline | yes | yes | Full/SFT 30-step |
| LTX2TwoStagePipeline | TI2VidTwoStagesPipeline | yes | yes | Full 30+3 with LoRA450 |
| LTX2DistilledOneStagePipeline | DistilledPipeline Stage 1 extraction | yes | yes | distilled 8-step Stage 1 |
| LTX2DistilledTwoStagePipeline | DistilledPipeline | yes | yes | distilled 8+3 |

Distilled one-stage uses a narrow Stage 1 extraction from the pinned official
DistilledPipeline; Lightricks does not publish a standalone Stage 1 CLI.

## Published artifacts

results-v2/videos/ contains exactly 16 MP4 files: one Official and one exact
timed vLLM-Omni response for each of the eight pipeline/task cases.

- results-v2/metrics.json: first-second SSIM, PSNR, and LPIPS
- results-v2/warm-e2e.json: warm E2E samples and summaries
- results-v2/contract.json: prompt, dimensions, revisions, and benchmark scope
- results-v2/requests/: exact per-pipeline T2V and I2V request templates

The quality reference uses raw Lightricks/LTX-2.5 split artifacts with the
pinned official runtime. vLLM-Omni uses the equivalent materialized
Lightricks/LTX-2.5-Diffusers checkpoint. The connector weights are held
constant across both implementations.

## Reproduce warm online E2E

    CUDA_VISIBLE_DEVICES=0 \
    /path/to/vllm-omni/.venv/bin/python \
      scripts/ltx25_raw_b300_pipeline_parity/run_warm_e2e.py \
      --model /path/to/LTX-2.5-Diffusers \
      --vllm-omni-root /path/to/vllm-omni \
      --vllm-bin /path/to/vllm-omni/.venv/bin/vllm \
      --request-dir scripts/ltx25_raw_b300_pipeline_parity/results-v2/requests \
      --repeats 2 \
      --save-responses \
      --attention-backend CUDNN_ATTN \
      --output-dir /tmp/ltx25-warm-e2e

The --request-dir option is important: it sends the same negative prompt,
guidance parameters, sigma arrays, stage schedules, and CRF used by the strict
Official comparison. --dry-run prints every serve command and multipart form
without starting a GPU server.

## Recompute precision metrics

    CUDA_VISIBLE_DEVICES=0 \
    /path/to/vllm-omni/.venv/bin/python \
      scripts/ltx25_raw_b300_pipeline_parity/compute_metrics.py \
      --results-dir scripts/ltx25_raw_b300_pipeline_parity/results-v2/videos \
      --max-frames 24 \
      --lpips-frames 24 \
      --device cuda \
      --force

The published scores are production-backend similarity measurements, not
bit-exact SDPA unit-test gates. Both sides use the same BF16 cuDNN backend;
small implementation-order differences can accumulate over a denoising
trajectory, and the hard prompt is intended to expose rather than conceal
those differences.
