# MiniMax-H3 SM103a VSA E2E outputs

These videos are the measured outputs from the paired 8x B300 FastH3 E2E run
used to validate native FastVideo SM103a VSA support. Both arms used the
official 1344x768, 24 FPS starship request, four inference steps, seed 1101,
USP 8, Ring 1, and VAE tile parallelism 8.

| Requested duration | Dense output | Native SM103a VSA output | Framework E2E speedup |
| --- | --- | --- | ---: |
| 10 s | [dense-10s.mp4](./dense-10s.mp4) | [vsa-sm103a-10s.mp4](./vsa-sm103a-10s.mp4) | 1.26x |
| 15 s | [dense-15s.mp4](./dense-15s.mp4) | [vsa-sm103a-15s.mp4](./vsa-sm103a-15s.mp4) | 1.51x |

The 10-second outputs contain 243 frames (10.125 s), and the 15-second outputs
contain 362 frames (15.083 s). All files are H.264 at 1344x768 and 24 FPS and
passed a complete FFmpeg decode.

## SHA256

```text
683a8a0bcaa1b9406b2a66310843cfae3f06c24d5d231e0ef343da5fa08cbb37  dense-10s.mp4
40548bb1bd60ad8201542e3824e5ad578fdffb61dfda7ed95961691e1a07cdd4  dense-15s.mp4
1a6e39b9ac9abf4bf587e6f45b92ab04a27d7640dc514eed09960ce71a6c8e7a  vsa-sm103a-10s.mp4
211db86a18d74268edf3086adcfa87610a33136e27284571363e33c2148d22b9  vsa-sm103a-15s.mp4
```

Kernel change: [FastVideo PR #1812](https://github.com/hao-ai-lab/FastVideo/pull/1812).
