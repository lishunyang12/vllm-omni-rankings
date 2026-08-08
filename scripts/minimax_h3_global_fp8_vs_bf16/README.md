# MiniMax-H3：BF16 vs default global FP8

本目录用于直接观看 MiniMax-H3 BF16 与默认 global FP8 的五秒输出。

▶ [打开 BF16 / global FP8 并排在线播放器](https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_global_fp8_vs_bf16/)


所有配对结果使用相同的官方输入、seed 0、50 denoising steps、24 FPS 和
32 kHz stereo audio。global FP8 对应用户 API `--quantization fp8`：
量化 eligible DiT 与 Qwen3-VL text-decoder linears；vision tower、VAE、
norm、embedding、RoPE 和 mixed-precision input/output heads 保留原精度。

如果 GitHub 文件预览无法播放，请使用表格中的 **Raw MP4** 链接直接在
浏览器打开或下载。

| Task | BF16 | Default global FP8 | PSNR | SSIM | Audio correlation |
|---|---|---|---:|---:|---:|
| T2VA | [GitHub](bf16/t2va.mp4) · [Raw MP4](https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/minimax_h3_global_fp8_vs_bf16/bf16/t2va.mp4) | [GitHub](global_fp8/t2va.mp4) · [Raw MP4](https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/minimax_h3_global_fp8_vs_bf16/global_fp8/t2va.mp4) | 17.943 dB | 0.7290 | 0.9351 |
| I2VA | [GitHub](bf16/i2va.mp4) · [Raw MP4](https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/minimax_h3_global_fp8_vs_bf16/bf16/i2va.mp4) | [GitHub](global_fp8/i2va.mp4) · [Raw MP4](https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/minimax_h3_global_fp8_vs_bf16/global_fp8/i2va.mp4) | 29.403 dB | 0.9511 | 0.9704 |
| Ref2VA | [GitHub](bf16/ref2va.mp4) · [Raw MP4](https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/minimax_h3_global_fp8_vs_bf16/bf16/ref2va.mp4) | [GitHub](global_fp8/ref2va.mp4) · [Raw MP4](https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/minimax_h3_global_fp8_vs_bf16/global_fp8/ref2va.mp4) | 44.153 dB | 0.9839 | 0.9250 |

这些指标是相同 seed 下对 BF16 decoded output 的描述性比较，不是硬性质量
门槛。扩散轨迹可能因很小的数值变化而分岔，请同时观察 motion、prompt
adherence、subject consistency、visual artifacts 和 audio。

## Media information

- T2VA / Ref2VA：1344×768
- I2VA：1376×768（adaptive aspect ratio）
- Duration：5.207 秒
- Video：H.264, 24 FPS
- Audio：AAC, 32 kHz, stereo

## Provenance

原始 benchmark、全部 ablation cases、metrics 与日志位于：

[../minimax_h3_online_fp8](../minimax_h3_online_fp8)

对应 source commit：
[`802ef8e`](https://github.com/lishunyang12/vllm-omni-rankings/tree/802ef8e4e61ada7ced8babf1366170d9df26f27d/scripts/minimax_h3_online_fp8)
