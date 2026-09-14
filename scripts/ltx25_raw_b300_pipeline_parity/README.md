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
- Latency: persistent Official process with a reused pipeline object through
  complete MP4 encoding and resident vLLM-Omni loopback requests through
  complete MP4 receipt; both use one excluded warm-up per task and two timed
  repetitions

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
- results-v2/warm-e2e.json: Official and vLLM-Omni warm E2E samples,
  summaries, provenance, and speedups
- results-v2/contract.json: prompt, dimensions, revisions, and benchmark scope
- results-v2/requests/: exact per-pipeline T2V and I2V request templates

The quality reference uses raw Lightricks/LTX-2.5 split artifacts with the
pinned official runtime. vLLM-Omni uses the equivalent materialized
Lightricks/LTX-2.5-Diffusers checkpoint. The connector weights are held
constant across both implementations.

## Warm E2E results

| Pipeline | T2V Official | T2V Omni | Speedup | I2V Official | I2V Omni | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| LTX2Pipeline | 392.55 s | 216.44 s | 1.81x | 228.80 s | 217.03 s | 1.05x |
| LTX2TwoStagePipeline | 116.05 s | 54.06 s | 2.15x | 135.88 s | 53.21 s | 2.55x |
| LTX2DistilledOneStagePipeline | 32.64 s | 21.65 s | 1.51x | 33.79 s | 22.26 s | 1.52x |
| LTX2DistilledTwoStagePipeline | 35.42 s | 14.64 s | 2.42x | 36.15 s | 14.73 s | 2.45x |

Speedup is Official mean divided by vLLM-Omni mean. Official uses a direct
Python pipeline call because Lightricks does not expose an equivalent resident
HTTP service; vLLM-Omni includes loopback HTTP response delivery. Both sides
include complete MP4 production and exclude initialization and warm-up.

## Reproduce warm E2E

### vLLM-Omni

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

### Official

    CUDA_VISIBLE_DEVICES=0 \
    /path/to/vllm-omni/.venv/bin/python \
      scripts/ltx25_raw_b300_pipeline_parity/run_official_warm_e2e.py \
      --official-root /path/to/LTX-2-source \
      --model-root /path/to/LTX-2.5 \
      --connector-model /path/to/LTX-2.5-Diffusers \
      --request-dir scripts/ltx25_raw_b300_pipeline_parity/results-v2/requests \
      --image scripts/ltx25_raw_b300_pipeline_parity/inputs/quickstart-seed42-frame0.png \
      --pipeline LTX2Pipeline \
      --repeats 2 \
      --output-dir /tmp/ltx25-official-warm

Repeat with the other three public class names. This measures a reused Official
pipeline-object call through complete MP4 encoding, with one excluded warm-up
for T2V and one for I2V.

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

## MiniMax-H3 appendix

The same public page includes MiniMax-H3 T2VA and first-frame FL2VA generated
from the exact difficult prompt, seed 42, one isolated NVIDIA B300, and
approximately five seconds of playback. MiniMax-H3 retains its native output
contract rather than being resampled after generation:

- Output: 1344x768, 124 frames, 24 FPS, 5.175 seconds
- Audio: 32 kHz stereo AAC
- Denoise: 50 requested steps, 49 updates
- Runtime: BF16 `TRTLLM_ATTN`, regional compile, TP1, Ulysses1, Ring1,
  text-encoder TP1, VAE patch1, no offload
- Timing: one excluded complete warm-up per task, then two sequential warm E2E
  requests through receipt of the complete MP4

| MiniMax-H3 task | Warm samples | Mean | Generation / playback |
|---|---:|---:|---:|
| T2VA | 133.546 / 133.751 s | 133.648 s | 25.83x realtime |
| first-frame FL2VA | 144.250 / 144.367 s | 144.309 s | 27.89x realtime |

This appendix aligns prompt content, seed, GPU count, FPS, and playback length;
it does not claim equal compute or a same-backend speedup against LTX. The LTX
showcase uses 1920x1088, 121 frames, and cuDNN attention. MiniMax-H3 uses
1344x768, 124 frames, and its recommended B300 `TRTLLM_ATTN` backend.

Published MiniMax artifacts:

- `minimax-appendix/videos/minimax-h3-t2va-seed-42.mp4`
- `minimax-appendix/videos/minimax-h3-fl2va-seed-42.mp4`
- `minimax-appendix/inputs/quickstart-seed42-1344x768.png`
- `minimax-appendix/warm-e2e.json`
- `minimax-appendix/contract.json`
- `minimax-appendix/checksums.sha256`

Reproduce with a local mirror of the FL2VA partition:

    /path/to/vllm-omni/.venv/bin/python \
      scripts/ltx25_raw_b300_pipeline_parity/run_minimax_appendix.py \
      --model /path/to/MiniMax-H3/FL2VA \
      --vllm-omni-root /path/to/vllm-omni \
      --vllm-bin /path/to/vllm-omni/.venv/bin/vllm \
      --gpu 0 \
      --repeats 2 \
      --output-dir /dev/shm/ltx25-minimax-appendix

The script's `--dry-run` mode prints the exact resident server command and both
multipart request forms without starting a GPU process.

## LTX-2.5 Stage-2 TDP resolution appendix

The final section of the same public page contains a resident-server,
four-GPU comparison of global Ulysses SP and Stage-2 tiled data parallelism.
It uses the exact official difficult prompt, seeds 42 through 51, 121 frames,
24 FPS, the distilled 8-step Stage 1, and the matched Stage-2 sigma schedule
`[0.625, 0.4, 0.0]`.

| Delivery output | Internal shape | Global SP mean | Stage-2 TDP mean | Mean per-seed speedup | Full-frame SSIM | PSNR |
|---|---|---:|---:|---:|---:|---:|
| 1920x1080 | 1920x1088 | 8.974 s | 8.754 s | 1.026x | 0.984624 | 41.096 dB |
| DCI 2K 2048x1080 | 2048x1088 | 10.329 s | 10.006 s | 1.032x | 0.985059 | 41.029 dB |
| QHD 2K 2560x1440 | 2560x1472 | 17.949 s | 16.112 s | 1.113x | 0.986302 | 40.512 dB |

Latency is client wall time for `POST /v1/videos/sync` through receipt of the
complete MP4. One request per resolution/mode is excluded as warm-up, then all
ten seeds are measured with alternating mode order. Server startup, warm-up,
file writes, ffprobe validation, scoring, and page transcodes are excluded.

The Global-SP request uses the 64-pixel-aligned internal shape. The TDP request
uses the delivery shape and lets the runtime align internally, then crop after
decode. Both paths therefore denoise the same internal token grid. TDP keeps
global attention in Stage 1, uses a 2x2 grid with five latent cells of overlap
in Stage 2, and retains the full-context Stage-1 audio.

Every one of the 60 source responses decoded to exactly 121 frames at 24 FPS.
The published directory contains 30 labeled side-by-side AV MP4s, 30 posters,
full-frame and horizontal/vertical overlap-band SSIM/PSNR for each seed,
provenance, and checksums:

- `tdp-resolution-study/results.json`
- `tdp-resolution-study/videos/<resolution>/seed-<seed>.mp4`
- `tdp-resolution-study/previews/<resolution>/seed-<seed>.jpg`
- `tdp-resolution-study/checksums.sha256`

Reproduce the complete user-facing serving path:

    /path/to/vllm-omni/.venv/bin/python \
      scripts/ltx25_raw_b300_pipeline_parity/run_tdp_resolution_study.py \
      --model /path/to/LTX-2.5-Diffusers \
      --vllm-omni-root /path/to/vllm-omni \
      --vllm-bin /path/to/vllm-omni/.venv/bin/vllm \
      --gpus 0,1,2,3 \
      --output-dir /dev/shm/ltx25-tdp-resolution-study

Then validate, score, and create the page assets:

    /path/to/vllm-omni/.venv/bin/python \
      scripts/ltx25_raw_b300_pipeline_parity/analyze_tdp_resolution_study.py \
      --results-dir /dev/shm/ltx25-tdp-resolution-study \
      --publish-dir scripts/ltx25_raw_b300_pipeline_parity/tdp-resolution-study

The implementation and the high-resolution async-output fix are tracked in
[vLLM-Omni Draft PR #6430](https://github.com/vllm-project/vllm-omni/pull/6430).
