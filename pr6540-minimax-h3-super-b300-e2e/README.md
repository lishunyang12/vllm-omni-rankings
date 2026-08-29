# PR #6540 MiniMax H3 Super B300 E2E evidence

Static GitHub Pages report for the two-stage MiniMax H3 Super draft → LTX-2.5 refiner E2E run on commit `cc202c4673d34b3fdafdca05d187f34970375f75`.

- Hardware: 2 × NVIDIA B300 SXM6 AC (physical GPUs 2 and 3)
- Protocol: seed 42, concurrency 1, 1 warmup + 3 measured runs for 5-second and 10-second outputs
- Validation: 8/8 MP4 files passed stream checks and full ffmpeg decode
- Page: <https://lishunyang12.github.io/vllm-omni-rankings/pr6540-minimax-h3-super-b300-e2e/>

The exact runner and launch script are published with the raw metadata, results, GPU samples, deploy configuration, and complete logs under `artifacts/`.
