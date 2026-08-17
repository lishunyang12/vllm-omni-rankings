# MiniMax-H3 BF16 parallel high-quality samples

Four independent text-to-video-with-audio samples generated in parallel with the official MiniMax-H3 FL2VA BF16/FP32 checkpoint.

## Configuration

- Resolution: 1344 x 768
- Output: 124 frames at 24 FPS, 5.175-second MP4
- Sampling: 50 requested diffusion steps
- Media: H.264 video with 32 kHz stereo AAC audio
- Runtime: four B300 GPUs, one independent seed per GPU
- Attention: TRTLLM_ATTN with regional compilation

## Results

| Seed | Video | Wall time | Denoise | Peak memory |
|---:|---|---:|---:|---:|
| 1101 | [MP4](sam-lisa-seed1101.mp4) | 146.459 s | 131.480 s | 133,164 MiB |
| 2027 | [MP4](sam-lisa-seed2027.mp4) | 146.708 s | 133.009 s | 133,164 MiB |
| 4099 | [MP4](sam-lisa-seed4099.mp4) | 142.399 s | 130.070 s | 133,164 MiB |
| 7727 | [MP4](sam-lisa-seed7727.mp4) | 145.572 s | 132.959 s | 133,164 MiB |

Open [index.html](index.html) for the side-by-side player. Exact hashes and stage timings are in [results.json](results.json); the complete input is in [prompt.txt](prompt.txt).
