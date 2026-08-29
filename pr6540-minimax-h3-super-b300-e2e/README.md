# PR #6540 MiniMax H3 Super alignment evidence

Static GitHub Pages evidence for [vllm-project/vllm-omni#6540](https://github.com/vllm-project/vllm-omni/pull/6540).

The page contains all 28 generated test videos from this investigation:

- 4 Sana official-source reproduction artifacts (three Stage 1 intermediates and one final output)
- 4 vLLM-Omni videos from the known blurred latent-contract failure
- 4 post-fix videos before rebase
- 4 post-rebase final-HEAD videos
- 4 native LTX-2.5 1920×1088 final-HEAD videos
- 8 original PR baseline videos retained from the previous page

It also includes per-stage timings, deterministic output hashes, media validation, first-frame fidelity, the blur-regression check, configs, raw logs, and benchmark JSON. “Official-source reproduction” means the NVLabs source path adapted to run on B300; it is not an NVIDIA-published B300 benchmark.

Run `sha256sum -c pr6540-minimax-h3-super-b300-e2e/SHA256SUMS.txt` from the repository root to verify every published MP4.
