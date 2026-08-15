# MiniMax-H3 BF16 vs SVDQuant

A self-contained page with five paired BF16/SVDQuant videos. Each case shows only:

- the BF16 and SVDQuant videos;
- PSNR, SSIM, and temporal RMSE;
- BF16 and SVDQuant end-to-end request latency.

The page also reports the separate strict native-fused benchmark: one full
50-step warmup followed by three measured rounds on NVIDIA B300. The strict
mean is 134.599 s for BF16 and 103.977 s for SVDQuant, or 1.2945x E2E.

- Live page: <https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_svdquant_b300_results/>
- Attached metrics: [results.json](results.json)
