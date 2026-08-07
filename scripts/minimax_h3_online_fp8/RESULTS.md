# Measured results

All numbers are descriptive; no quality threshold is applied. PSNR and SSIM
compare decoded video frames with the matching BF16 output at seed 0.

## Interpretation

- `all_linear_fp8` is the throughput-oriented choice and the simplest global
  configuration. It has the highest mean speedup in this matrix.
- `dit_fp8` is the peak-memory-oriented choice. It has the largest mean memory
  reduction and the lowest maximum peak memory among the FP8 DiT cases.
- `te_mlp_l0_9` is the conservative encoder-only starting point. Encoder-only
  quantization preserves higher similarity here, but provides no stable
  end-to-end speed or peak-memory benefit on this workload.
- Quantization quality is not monotonic for a single diffusion seed. Inspect the
  linked videos rather than treating any one fidelity metric as a hard gate.

## Aggregate results

| Case | Tasks | Mean speedup | Max peak GPU MiB | Mean memory saved | Mean PSNR dB | Mean SSIM | Mean audio SNR dB | Mean audio corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bf16` | 3 | 1.000x | 105278 | 0.0% | - | 1.0000 | - | 1.0000 |
| `dit_fp8` | 3 | 1.054x | 73720 | 30.3% | 29.588 | 0.8851 | 8.653 | 0.9309 |
| `te_mlp_l0_9` | 3 | 0.999x | 107092 | -0.7% | 37.240 | 0.9513 | 17.074 | 0.9904 |
| `te_mlp_l0_24` | 3 | 1.000x | 107220 | -0.7% | 36.901 | 0.9541 | 15.961 | 0.9889 |
| `te_mlp_all` | 3 | 0.978x | 107092 | -0.6% | 36.543 | 0.9492 | 15.467 | 0.9867 |
| `te_mlp_o_all` | 3 | 0.977x | 107102 | -0.7% | 35.657 | 0.9434 | 17.276 | 0.9909 |
| `te_all_linear` | 3 | 0.988x | 107104 | -0.7% | 35.589 | 0.9519 | 16.286 | 0.9882 |
| `dit_te_mlp_all` | 3 | 1.035x | 75546 | 29.7% | 29.712 | 0.8888 | 8.651 | 0.9348 |
| `all_linear_fp8` | 3 | 1.066x | 77042 | 29.2% | 30.500 | 0.8880 | 8.779 | 0.9435 |

## Per-task results

| Case | Task | Video | Wall s | Speedup | Peak GPU MiB | Memory saved | PSNR dB | SSIM | Audio SNR dB | Audio corr. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bf16` | `t2va` | [MP4](videos/bf16/t2va.mp4) | 65.60 | 1.000x | 102916 | 0.0% | - | 1.0000 | - | 1.0000 |
| `bf16` | `i2va` | [MP4](videos/bf16/i2va.mp4) | 71.57 | 1.000x | 103744 | 0.0% | - | 1.0000 | - | 1.0000 |
| `bf16` | `ref2va` | [MP4](videos/bf16/ref2va.mp4) | 229.41 | 1.000x | 105278 | 0.0% | - | 1.0000 | - | 1.0000 |
| `dit_fp8` | `t2va` | [MP4](videos/dit_fp8/t2va.mp4) | 61.01 | 1.075x | 71304 | 30.7% | 17.920 | 0.7301 | 8.242 | 0.9236 |
| `dit_fp8` | `i2va` | [MP4](videos/dit_fp8/i2va.mp4) | 67.27 | 1.064x | 72288 | 30.3% | 26.707 | 0.9414 | 10.535 | 0.9550 |
| `dit_fp8` | `ref2va` | [MP4](videos/dit_fp8/ref2va.mp4) | 224.03 | 1.024x | 73720 | 30.0% | 44.136 | 0.9839 | 7.182 | 0.9141 |
| `te_mlp_l0_9` | `t2va` | [MP4](videos/te_mlp_l0_9/t2va.mp4) | 63.58 | 1.032x | 102586 | 0.3% | 25.698 | 0.8813 | 18.005 | 0.9921 |
| `te_mlp_l0_9` | `i2va` | [MP4](videos/te_mlp_l0_9/i2va.mp4) | 73.27 | 0.977x | 107092 | -3.2% | 39.782 | 0.9863 | 15.726 | 0.9881 |
| `te_mlp_l0_9` | `ref2va` | [MP4](videos/te_mlp_l0_9/ref2va.mp4) | 232.07 | 0.989x | 104462 | 0.8% | 46.241 | 0.9862 | 17.490 | 0.9911 |
| `te_mlp_l0_24` | `t2va` | [MP4](videos/te_mlp_l0_24/t2va.mp4) | 63.26 | 1.037x | 102714 | 0.2% | 26.211 | 0.8920 | 16.746 | 0.9897 |
| `te_mlp_l0_24` | `i2va` | [MP4](videos/te_mlp_l0_24/i2va.mp4) | 73.98 | 0.967x | 107220 | -3.4% | 38.278 | 0.9842 | 14.418 | 0.9877 |
| `te_mlp_l0_24` | `ref2va` | [MP4](videos/te_mlp_l0_24/ref2va.mp4) | 230.42 | 0.996x | 104262 | 1.0% | 46.213 | 0.9862 | 16.719 | 0.9893 |
| `te_mlp_all` | `t2va` | [MP4](videos/te_mlp_all/t2va.mp4) | 65.36 | 1.004x | 102586 | 0.3% | 25.600 | 0.8774 | 16.262 | 0.9881 |
| `te_mlp_all` | `i2va` | [MP4](videos/te_mlp_all/i2va.mp4) | 75.83 | 0.944x | 107092 | -3.2% | 37.845 | 0.9840 | 14.766 | 0.9865 |
| `te_mlp_all` | `ref2va` | [MP4](videos/te_mlp_all/ref2va.mp4) | 232.74 | 0.986x | 104262 | 1.0% | 46.185 | 0.9862 | 15.372 | 0.9855 |
| `te_mlp_o_all` | `t2va` | [MP4](videos/te_mlp_o_all/t2va.mp4) | 65.26 | 1.005x | 102596 | 0.3% | 24.419 | 0.8635 | 16.020 | 0.9890 |
| `te_mlp_o_all` | `i2va` | [MP4](videos/te_mlp_o_all/i2va.mp4) | 76.05 | 0.941x | 107102 | -3.2% | 36.357 | 0.9805 | 17.558 | 0.9913 |
| `te_mlp_o_all` | `ref2va` | [MP4](videos/te_mlp_o_all/ref2va.mp4) | 232.75 | 0.986x | 104272 | 1.0% | 46.196 | 0.9862 | 18.249 | 0.9925 |
| `te_all_linear` | `t2va` | [MP4](videos/te_all_linear/t2va.mp4) | 64.19 | 1.022x | 102600 | 0.3% | 25.827 | 0.8889 | 14.926 | 0.9848 |
| `te_all_linear` | `i2va` | [MP4](videos/te_all_linear/i2va.mp4) | 74.90 | 0.956x | 107104 | -3.2% | 34.761 | 0.9805 | 16.047 | 0.9878 |
| `te_all_linear` | `ref2va` | [MP4](videos/te_all_linear/ref2va.mp4) | 232.71 | 0.986x | 104276 | 1.0% | 46.179 | 0.9862 | 17.886 | 0.9919 |
| `dit_te_mlp_all` | `t2va` | [MP4](videos/dit_te_mlp_all/t2va.mp4) | 60.74 | 1.080x | 71040 | 31.0% | 18.380 | 0.7415 | 8.922 | 0.9424 |
| `dit_te_mlp_all` | `i2va` | [MP4](videos/dit_te_mlp_all/i2va.mp4) | 71.03 | 1.008x | 75546 | 27.2% | 26.640 | 0.9409 | 10.393 | 0.9628 |
| `dit_te_mlp_all` | `ref2va` | [MP4](videos/dit_te_mlp_all/ref2va.mp4) | 225.22 | 1.019x | 72708 | 30.9% | 44.117 | 0.9839 | 6.638 | 0.8993 |
| `all_linear_fp8` | `t2va` | [MP4](videos/all_linear_fp8/t2va.mp4) | 59.00 | 1.112x | 71038 | 31.0% | 17.943 | 0.7290 | 8.473 | 0.9351 |
| `all_linear_fp8` | `i2va` | [MP4](videos/all_linear_fp8/i2va.mp4) | 67.38 | 1.062x | 77042 | 25.7% | 29.403 | 0.9511 | 9.964 | 0.9704 |
| `all_linear_fp8` | `ref2va` | [MP4](videos/all_linear_fp8/ref2va.mp4) | 224.00 | 1.024x | 72706 | 30.9% | 44.153 | 0.9839 | 7.901 | 0.9250 |
