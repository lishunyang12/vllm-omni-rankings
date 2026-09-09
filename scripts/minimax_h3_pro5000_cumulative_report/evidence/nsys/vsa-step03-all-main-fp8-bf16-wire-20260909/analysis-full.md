# MiniMax-H3 Nsight analysis

- Source: `/lustre/raplab/client/sylarl/minimax-h3-native/results/vllm-omni-fasth3-vsa-all-main-fp8-bf16-wire-o-bundle-sm120-sp8-step03-nsys-20260909T062108Z/minimax-h3-vsa-all-main-fp8-bf16-wire-sp8-step03.nsys-rep`
- SQLite: `/lustre/raplab/client/sylarl/minimax-h3-native/results/vllm-omni-fasth3-vsa-all-main-fp8-bf16-wire-o-bundle-sm120-sp8-step03-nsys-20260909T062108Z/minimax-h3-vsa-all-main-fp8-bf16-wire-sp8-step03.sqlite`
- Scope: `nvtx_range` / `minimax_h3.denoise.step_03` @ `vllm_omni.minimax_h3`
- Window wall time: **3730.282 ms**
- GPU active-span median: **3712.981 ms**
- Slowest trace device by active span: **device 0 / rank 0 / physical GPU 0 / 3729.858 ms**
- Largest non-barrier kernel sum: **trace device 7 / rank 7 / physical GPU 7 / 3040.551 ms**
- Device identity: **kernel PID -> server NCCL rank/cudaDev/nvmlDev/busId**

> GPU-time below is summed across devices and streams; it is not additive wall time.

## Cross-rank decode NVTX timeline

No explicit nested `minimax_h3.decode.*` stage range was found in scope.

## Per-GPU timeline

| Trace dev | Rank | Physical GPU | PCI bus | PID(s) | Active span ms | Busy union ms | Kernel-sum ms | Non-barrier / barrier kernel ms | Memcpy ms / MiB | Window idle ms | Largest idle gap ms |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0000:06:00.0 | 3827358 | 3729.858 | 3071.243 | 3062.094 | 2617.275 / 444.819 | 8.848 / 4911.56 | 659.038 | 3.173 |
| 1 | 1 | 4 | 0000:86:00.0 | 3827363 | 3713.009 | 3055.911 | 3047.145 | 2568.641 / 478.504 | 8.491 / 4902.19 | 674.371 | 17.273 |
| 2 | 2 | 1 | 0000:09:00.0 | 3827364 | 3713.021 | 3055.938 | 3046.945 | 2588.813 / 458.131 | 8.704 / 4902.19 | 674.344 | 17.260 |
| 3 | 3 | 5 | 0000:89:00.0 | 3827365 | 3712.955 | 3053.583 | 3044.794 | 2532.784 / 512.010 | 8.520 / 4902.19 | 676.699 | 17.322 |
| 4 | 4 | 2 | 0000:76:00.0 | 3827366 | 3712.984 | 3051.607 | 3042.528 | 2589.535 / 452.993 | 8.789 / 4902.19 | 678.674 | 17.290 |
| 5 | 5 | 6 | 0000:f6:00.0 | 3827367 | 3712.915 | 3054.177 | 3045.909 | 2525.397 / 520.512 | 8.003 / 4855.38 | 676.105 | 17.330 |
| 6 | 6 | 3 | 0000:79:00.0 | 3827368 | 3712.978 | 3054.150 | 3046.149 | 2612.105 / 434.045 | 7.709 / 4902.19 | 676.132 | 17.303 |
| 7 | 7 | 7 | 0000:f9:00.0 | 3827369 | 3712.951 | 3052.811 | 3044.234 | 3040.551 / 3.683 | 8.232 / 4855.38 | 677.471 | 17.327 |

## Per-device wall decomposition (no double-counting)

Per-device disjoint interval sweep. Different simultaneously active phases are charged once to concurrent_mixed; no values are summed across devices. Reference device (largest traced busy union): 0.

| Trace dev | Linear GEMM ms | VSA ms | RDMA barrier ms | RDMA other ms | Pointwise ms | Other single-phase ms | Mixed-concurrent ms | Idle ms | Account error ns |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1104.377 | 1121.481 | 444.819 | 0.000 | 130.360 | 270.206 | 0.000 | 659.038 | 0 |
| 1 | 1079.311 | 1103.678 | 478.504 | 0.000 | 129.152 | 265.266 | 0.000 | 674.371 | 0 |
| 2 | 1075.433 | 1122.718 | 458.131 | 0.000 | 129.233 | 270.422 | 0.000 | 674.344 | 0 |
| 3 | 1062.362 | 1084.466 | 512.010 | 0.000 | 129.068 | 265.677 | 0.000 | 676.699 | 0 |
| 4 | 1077.686 | 1120.870 | 452.993 | 0.000 | 129.071 | 270.987 | 0.000 | 678.674 | 0 |
| 5 | 1062.566 | 1076.884 | 520.512 | 0.000 | 128.715 | 265.499 | 0.000 | 676.105 | 0 |
| 6 | 1097.940 | 1123.529 | 434.045 | 0.000 | 129.233 | 269.403 | 0.000 | 676.132 | 0 |
| 7 | 1269.097 | 1364.636 | 3.683 | 0.000 | 130.424 | 284.970 | 0.000 | 677.471 | 0 |

## Kernel categories

| Category | Calls | Total GPU-ms | % kernel GPU-time | Median GPU-ms | Slowest trace dev / ms | Evidence rule |
|---|---:|---:|---:|---:|---:|---|
| `vsa_attention` | 400 | 9118.263 | 37.40% | 1121.176 | 7 / 1364.636 | Explicit block-sparse/VSA attention kernel, including FlashInfer #4944 CuTeDSL SM120 BlockSparseAttnForward |
| `gemm_unknown` | 2815 | 8828.772 | 36.21% | 1078.499 | 7 / 1269.097 | Recognizable GEMM, but its H3 layer role is not encoded in the kernel name |
| `flashinfer_rdma_barrier` | 3193 | 3304.697 | 13.56% | 455.562 | 5 / 520.512 | Explicit FlashInfer Ulysses PCIe/RDMA barrier; duration is synchronization residency/wait |
| `unknown` | 16803 | 1076.044 | 4.41% | 134.292 | 7 / 147.887 | No conservative category matched |
| `vsa_layout` | 2400 | 793.241 | 3.25% | 99.240 | 7 / 101.208 | Explicit H3 VSA compact/tile layout kernel |
| `h3_pointwise` | 1987 | 615.454 | 2.52% | 76.810 | 0 / 77.672 | Explicitly named H3 fused modulation, gated-residual, Q/K norm+RoPE, or SwiGLU kernel |
| `triton_reduction` | 1597 | 419.803 | 1.72% | 52.401 | 7 / 53.284 | Triton reduction kernel |
| `copy_kernel` | 3238 | 194.957 | 0.80% | 24.279 | 7 / 25.791 | Device-side copy/cat/transpose-like kernel (not a DMA memcpy record) |
| `nccl` | 8 | 28.416 | 0.12% | 3.809 | 5 / 4.328 | NCCL device kernel |
| `cudnn_attention` | 2 | 0.121 | 0.00% | 0.000 | 0 / 0.121 | cuDNN kernel with explicit SDPA/attention/FMHA evidence |
| `triton_other` | 4 | 0.030 | 0.00% | 0.000 | 0 / 0.030 | Other Triton kernel |
| `attention_other` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Non-cuDNN kernel with explicit attention/FMHA/flash-attention evidence |
| `cudnn_other` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | cuDNN kernel without explicit attention evidence |
| `flashinfer_other` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | FlashInfer kernel without explicit PCIe/RDMA evidence |
| `flashinfer_rdma_other` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Explicit FlashInfer Ulysses PCIe/RDMA kernel whose role is not distinguishable by name |
| `flashinfer_rdma_transfer` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Explicit FlashInfer Ulysses PCIe/RDMA data-movement kernel |
| `gemm_fc1` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | GEMM with an explicit FC1/gate/up-projection token |
| `gemm_fc2` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | GEMM with an explicit FC2/down-projection token |
| `gemm_out` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | GEMM with an explicit attention output-projection token |
| `gemm_qkv` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | GEMM with an explicit Q/K/V/QKV-projection token |
| `triton_activation` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Triton kernel with an explicit activation token |
| `triton_pointwise` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Triton pointwise kernel without a more specific activation token |
| `vsa_indexing` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Explicit VSA block-map-to-index compaction kernel |

## FlashInfer Ulysses barrier/rank-skew analysis

- Observed 3193 calls reconstructed as 400 chronological groups: 399 full-device and 1 partial.
- Layout: 8 barrier slots/layer (B0..B7); the group count is consistent with 50 layer cycles.
- Layout basis: explicit analyzer override.
- Full-group arrival-skew sum: 521.303 ms (13.97% of the selected wall window).
- Dominant latest arrival: trace device 7 in 179/399 groups (44.86%).
- Latest-device work overlapping the skew windows: vsa_attention=256.561 GPU-ms, gemm_unknown=183.982 GPU-ms, h3_pointwise=30.205 GPU-ms, vsa_layout=29.713 GPU-ms, unknown=14.121 GPU-ms, copy_kernel=2.046 GPU-ms; covered union 516.627 ms, uncovered 4.676 ms.
- Arrival-skew sums cover sequential full-device barriers and are already inside step wall time; they indicate load-imbalance exposure, not additional time or guaranteed removable time.

### Per-device barrier residency and arrival lag

| Trace dev | Calls | Residency sum ms | Residency median / p95 / max µs | Arrival-lag median / p95 µs | Latest arrival groups / % |
|---:|---:|---:|---:|---:|---:|
| 0 | 400 | 444.819 | 9.920 / 5312.760 / 5471.351 | 4.456 / 1042.643 | 26 / 6.52% |
| 1 | 399 | 478.504 | 11.200 / 5777.784 / 5959.192 | 2.869 / 573.320 | 22 / 5.51% |
| 2 | 399 | 458.131 | 9.728 / 5277.211 / 5422.236 | 4.876 / 1062.997 | 36 / 9.02% |
| 3 | 399 | 512.010 | 8.704 / 6174.861 / 6355.883 | 3.771 / 179.412 | 48 / 12.03% |
| 4 | 399 | 452.993 | 8.544 / 5297.467 / 5481.340 | 6.075 / 1056.788 | 50 / 12.53% |
| 5 | 399 | 520.512 | 9.984 / 6351.510 / 6443.382 | 1.484 / 20.685 | 19 / 4.76% |
| 6 | 399 | 434.045 | 9.728 / 5260.177 / 5394.193 | 4.417 / 1089.756 | 19 / 4.76% |
| 7 | 399 | 3.683 | 5.024 / 20.800 / 291.264 | 5.373 / 6347.465 | 179 / 44.86% |

### Per-layer chronological barrier slots

| Slot | Groups (full) | Arrival-skew sum ms | Arrival skew median / p95 / max µs | Release skew median / p95 / max µs | Dominant latest GPU / count | Late-device kernel evidence (GPU-ms) |
|---:|---:|---:|---:|---:|---:|---|
| B0 | 50 (49) | 199.951 | 4083.330 / 4145.386 / 4164.760 | 1.023 / 1.155 / 1.182 | 7 / 49 | gemm_unknown=169.308, h3_pointwise=30.205 |
| B1 | 50 (50) | 0.316 | 5.657 / 10.352 / 22.759 | 1.185 / 1.665 / 1.805 | 0 / 8 | uncovered=0.316 |
| B2 | 50 (50) | 1.722 | 17.248 / 136.206 / 387.015 | 1.335 / 1.837 / 1.940 | 4 / 16 | uncovered=1.722 |
| B3 | 50 (50) | 0.551 | 5.074 / 7.121 / 290.012 | 1.226 / 1.759 / 1.869 | 2 / 11 | uncovered=0.551 |
| B4 | 50 (50) | 2.023 | 31.660 / 106.447 / 121.879 | 1.056 / 1.594 / 1.686 | 7 / 38 | copy_kernel=1.913 |
| B5 | 50 (50) | 0.265 | 4.825 / 8.367 / 11.758 | 1.175 / 1.622 / 1.837 | 7 / 10 | uncovered=0.265 |
| B6 | 50 (50) | 316.185 | 6327.512 / 6417.836 / 6439.414 | 0.784 / 0.928 / 0.995 | 7 / 50 | vsa_attention=256.561, vsa_layout=29.713, gemm_unknown=14.673, unknown=14.121, copy_kernel=0.134 |
| B7 | 50 (50) | 0.291 | 5.277 / 9.819 / 11.308 | 1.159 / 1.760 / 1.791 | 3 / 11 | uncovered=0.291 |

### Per-device × barrier slot

| Slot | Trace dev | Calls | Residency sum ms | Residency median / p95 µs | Arrival-lag median / p95 µs | Latest groups |
|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 50 | 174.198 | 3559.002 / 3622.395 | 527.773 / 577.736 | 0 |
| B0 | 1 | 49 | 184.887 | 3776.027 / 3826.715 | 315.230 / 347.067 | 0 |
| B0 | 2 | 49 | 189.306 | 3858.844 / 3937.853 | 222.819 / 281.258 | 0 |
| B0 | 3 | 49 | 199.533 | 4077.427 / 4115.283 | 0.000 / 71.952 | 0 |
| B0 | 4 | 49 | 185.932 | 3812.477 / 3869.308 | 279.205 / 436.539 | 0 |
| B0 | 5 | 49 | 199.331 | 4074.264 / 4149.497 | 4.303 / 69.941 | 0 |
| B0 | 6 | 49 | 166.639 | 3395.510 / 3485.845 | 684.360 / 738.966 | 0 |
| B0 | 7 | 49 | 0.195 | 3.968 / 4.320 | 4083.330 / 4145.386 | 49 |
| B1 | 0 | 50 | 0.361 | 7.360 / 12.544 | 2.183 / 6.230 | 8 |
| B1 | 1 | 50 | 0.381 | 7.200 / 12.384 | 1.680 / 5.628 | 3 |
| B1 | 2 | 50 | 0.348 | 6.784 / 12.864 | 2.790 / 6.105 | 8 |
| B1 | 3 | 50 | 0.318 | 6.384 / 11.040 | 3.144 / 6.951 | 7 |
| B1 | 4 | 50 | 0.319 | 6.160 / 9.920 | 2.636 / 10.352 | 7 |
| B1 | 5 | 50 | 0.369 | 7.520 / 10.912 | 1.901 / 6.034 | 6 |
| B1 | 6 | 50 | 0.368 | 7.440 / 12.896 | 1.917 / 5.415 | 6 |
| B1 | 7 | 50 | 0.355 | 7.200 / 11.840 | 2.469 / 7.459 | 5 |
| B2 | 0 | 50 | 1.598 | 14.064 / 128.864 | 5.709 / 16.250 | 2 |
| B2 | 1 | 50 | 1.617 | 18.320 / 140.224 | 1.736 / 8.882 | 1 |
| B2 | 2 | 50 | 1.387 | 14.064 / 131.232 | 7.443 / 16.362 | 4 |
| B2 | 3 | 50 | 1.036 | 9.872 / 104.960 | 7.524 / 40.167 | 10 |
| B2 | 4 | 50 | 0.632 | 7.088 / 38.624 | 9.665 / 136.206 | 16 |
| B2 | 5 | 50 | 1.154 | 13.856 / 94.944 | 5.954 / 23.013 | 6 |
| B2 | 6 | 50 | 1.220 | 15.888 / 109.535 | 6.181 / 14.131 | 4 |
| B2 | 7 | 50 | 1.162 | 12.432 / 107.103 | 7.075 / 32.335 | 7 |
| B3 | 0 | 50 | 0.598 | 6.496 / 10.304 | 2.530 / 5.832 | 7 |
| B3 | 1 | 50 | 0.615 | 6.624 / 10.240 | 1.771 / 5.064 | 6 |
| B3 | 2 | 50 | 0.599 | 6.064 / 10.784 | 2.418 / 6.086 | 11 |
| B3 | 3 | 50 | 0.610 | 7.248 / 10.688 | 1.429 / 5.822 | 8 |
| B3 | 4 | 50 | 0.648 | 7.808 / 10.976 | 0.772 / 5.813 | 4 |
| B3 | 5 | 50 | 0.656 | 7.712 / 9.984 | 1.099 / 4.512 | 0 |
| B3 | 6 | 50 | 0.315 | 6.416 / 8.672 | 2.829 / 6.889 | 4 |
| B3 | 7 | 50 | 0.591 | 6.384 / 8.640 | 3.064 / 5.961 | 10 |
| B4 | 0 | 50 | 1.958 | 29.920 / 104.928 | 6.144 / 10.713 | 0 |
| B4 | 1 | 50 | 2.054 | 32.704 / 88.032 | 2.136 / 8.502 | 0 |
| B4 | 2 | 50 | 1.806 | 28.063 / 88.448 | 7.440 / 16.757 | 0 |
| B4 | 3 | 50 | 1.811 | 30.960 / 93.888 | 3.735 / 37.689 | 5 |
| B4 | 4 | 50 | 1.184 | 24.016 / 57.568 | 11.252 / 103.508 | 7 |
| B4 | 5 | 50 | 1.866 | 32.608 / 83.776 | 4.766 / 17.336 | 0 |
| B4 | 6 | 50 | 1.782 | 29.072 / 90.943 | 7.303 / 16.170 | 0 |
| B4 | 7 | 50 | 0.529 | 3.696 / 61.824 | 29.877 / 62.709 | 38 |
| B5 | 0 | 50 | 0.296 | 5.728 / 9.792 | 2.720 / 6.492 | 8 |
| B5 | 1 | 50 | 0.310 | 6.304 / 10.368 | 2.067 / 6.848 | 9 |
| B5 | 2 | 50 | 0.333 | 6.768 / 10.656 | 2.139 / 4.876 | 5 |
| B5 | 3 | 50 | 0.309 | 5.904 / 10.144 | 2.466 / 6.222 | 7 |
| B5 | 4 | 50 | 0.317 | 6.384 / 10.240 | 2.220 / 6.363 | 6 |
| B5 | 5 | 50 | 0.355 | 7.360 / 9.664 | 1.336 / 4.813 | 3 |
| B5 | 6 | 50 | 0.333 | 6.912 / 9.920 | 1.974 / 5.341 | 2 |
| B5 | 7 | 50 | 0.291 | 5.664 / 8.736 | 3.824 / 6.468 | 10 |
| B6 | 0 | 50 | 265.456 | 5302.632 / 5405.688 | 1029.397 / 1103.834 | 0 |
| B6 | 1 | 50 | 288.285 | 5763.560 / 5930.168 | 564.058 / 626.083 | 0 |
| B6 | 2 | 50 | 264.020 | 5273.899 / 5373.915 | 1046.755 / 1152.978 | 0 |
| B6 | 3 | 50 | 308.101 | 6164.764 / 6295.053 | 163.913 / 267.755 | 0 |
| B6 | 4 | 50 | 263.642 | 5286.172 / 5404.061 | 1034.871 / 1318.655 | 0 |
| B6 | 5 | 50 | 316.404 | 6331.717 / 6421.972 | 0.000 / 0.000 | 0 |
| B6 | 6 | 50 | 263.005 | 5256.273 / 5346.034 | 1062.645 / 1146.279 | 0 |
| B6 | 7 | 50 | 0.220 | 4.448 / 4.928 | 6327.512 / 6417.836 | 50 |
| B7 | 0 | 50 | 0.354 | 7.104 / 10.176 | 1.610 / 4.812 | 1 |
| B7 | 1 | 50 | 0.354 | 7.168 / 11.200 | 1.128 / 6.108 | 3 |
| B7 | 2 | 50 | 0.332 | 6.336 / 10.752 | 2.308 / 6.779 | 8 |
| B7 | 3 | 50 | 0.292 | 5.696 / 9.376 | 3.261 / 6.534 | 11 |
| B7 | 4 | 50 | 0.319 | 6.240 / 10.464 | 2.396 / 7.488 | 10 |
| B7 | 5 | 50 | 0.377 | 7.696 / 10.688 | 1.558 / 6.173 | 4 |
| B7 | 6 | 50 | 0.383 | 8.000 / 10.944 | 1.629 / 5.093 | 3 |
| B7 | 7 | 50 | 0.341 | 7.136 / 10.688 | 2.236 / 7.770 | 10 |

Attribution limits:

- Barrier slots B0..B7 are chronological modulo-8 positions; the trace has no semantic exchange labels.
- Barrier kernel residency is synchronization/spin time, not RDMA byte-transfer time.
- The latest-arriving device constrains that barrier release, but the trace alone does not identify why its preceding work was late.
- Late-path kernel evidence is overlap on the latest-arriving device during the arrival-skew window; uncovered time may be CPU, transport, DMA, or untraced work.
- 1 boundary/partial group(s) omit devices, commonly because NVTX-triggered capture began mid-kernel.

## FlashInfer/RDMA evidence and launch gaps

- Explicitly named kernels: 3193
- Matched CUDA launches: 3186; unmatched: 7
- Launch→GPU gap median/p95/max: 3.576 / 25755.348 / 31056.527 µs
- The CUDA launch API return-to-GPU-start gap includes stream queueing and backpressure; it is neither pure CPU dispatch overhead nor network transfer duration.

## CUDA API categories

| Category | Calls | Summed thread-ms | Interpretation |
|---|---:|---:|---|
| `synchronization_wait` | 3482 | 23658.263 | CUDA synchronize calls; duration includes waiting and is not CPU overhead alone |
| `kernel_launch` | 32448 | 106.892 | CUDA kernel/graph launch APIs |
| `memory_management` | 44 | 62.599 | CUDA allocation, mapping, and free APIs |
| `memcpy` | 5944 | 30.045 | CUDA memcpy APIs |
| `memset` | 882 | 22.109 | CUDA memset APIs |
| `event_stream` | 10540 | 8.202 | CUDA event and stream bookkeeping APIs |
| `other` | 32420 | 3.568 | Other CUDA APIs |
| `external_memory_possible_rdma_setup` | 0 | 0.000 | CUDA external/shareable-memory APIs; possible registration setup, not measured RDMA transfer time |

## Memcpy operations

| Trace dev | Kind | Calls | MiB (overlapping ops) | GPU-ms |
|---:|---|---:|---:|---:|
| 0 | Device-to-Device | 310 | 4911.539 | 8.674 |
| 0 | Device-to-Host | 436 | 0.025 | 0.172 |
| 0 | Host-to-Device | 10 | 0.000 | 0.003 |
| 1 | Device-to-Device | 308 | 4902.167 | 8.331 |
| 1 | Device-to-Host | 428 | 0.024 | 0.158 |
| 1 | Host-to-Device | 9 | 0.000 | 0.003 |
| 2 | Device-to-Device | 308 | 4902.167 | 8.533 |
| 2 | Device-to-Host | 426 | 0.024 | 0.168 |
| 2 | Host-to-Device | 8 | 0.000 | 0.002 |
| 3 | Device-to-Device | 308 | 4902.167 | 8.363 |
| 3 | Device-to-Host | 425 | 0.024 | 0.156 |
| 3 | Host-to-Device | 7 | 0.000 | 0.002 |
| 4 | Device-to-Device | 308 | 4902.167 | 8.619 |
| 4 | Device-to-Host | 426 | 0.024 | 0.168 |
| 4 | Host-to-Device | 7 | 0.000 | 0.002 |
| 5 | Device-to-Device | 306 | 4855.354 | 7.849 |
| 5 | Device-to-Host | 424 | 0.024 | 0.152 |
| 5 | Host-to-Device | 6 | 0.000 | 0.002 |
| 6 | Device-to-Device | 308 | 4902.167 | 7.539 |
| 6 | Device-to-Host | 428 | 0.024 | 0.167 |
| 6 | Host-to-Device | 10 | 0.000 | 0.003 |
| 7 | Device-to-Device | 306 | 4855.354 | 8.027 |
| 7 | Device-to-Host | 424 | 0.024 | 0.203 |
| 7 | Host-to-Device | 6 | 0.000 | 0.002 |

## Communication NVTX evidence

These CPU-range totals can overlap because NCCL group/collective ranges may be nested.

| Evidence | Range | Calls | Summed CPU-range ms |
|---|---|---:|---:|
| `nccl` | `ncclAllGather` | 8 | 0.633 |

## Top kernels

| # | Category | Calls | Total GPU-ms | Avg µs | % kernel GPU-time | Kernel |
|---:|---|---:|---:|---:|---:|---|
| 1 | `vsa_attention` | 400 | 9118.263 | 22795.659 | 37.40% | `kernel_cutlass_kernel_flashinfercute_dslsparsesm120_blk64flash_fwd_sm120BlockSparseAttnForwardSm120Blk64_object_at__tensor0000o12811101213_tensor0000o12811101213_tensor0000o128101…` |
| 2 | `gemm_unknown` | 1986 | 8567.961 | 4314.180 | 35.14% | `void cutlass::device_kernel<enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::Mainl…` |
| 3 | `flashinfer_rdma_barrier` | 3193 | 3304.697 | 1034.982 | 13.56% | `flashinfer::comm::ulysses_pcie::UlyssesPcieBarrier(unsigned long *, flashinfer::comm::ulysses_pcie::PeerSignalPointers, int, int, unsigned long *)` |
| 4 | `vsa_layout` | 1600 | 500.294 | 312.684 | 2.05% | `_h3_vsa_tile_pack_kernel` |
| 5 | `unknown` | 402 | 339.699 | 845.021 | 1.39% | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vllm::packed_silu_kernel<__nv_bfloat162>, (bool)1, (bool)1, (bool)0, (bool)1>(T1 *…` |
| 6 | `h3_pointwise` | 786 | 241.607 | 307.388 | 0.99% | `_rms_norm_rope_kernel` |
| 7 | `unknown` | 1200 | 198.399 | 165.332 | 0.81% | `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<float, at::native::sum_functor<c10::BFloat16, float, float>::operat…` |
| 8 | `triton_reduction` | 400 | 181.092 | 452.729 | 0.74% | `triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_max_mul_reciprocal_unsqueeze_1` |
| 9 | `vsa_layout` | 400 | 174.492 | 436.230 | 0.72% | `_h3_vsa_o_bundle_local_gate_kernel` |
| 10 | `h3_pointwise` | 400 | 170.750 | 426.874 | 0.70% | `_indexed_gate_rms_norm_scale_shift_kernel` |
| 11 | `triton_reduction` | 793 | 158.709 | 200.137 | 0.65% | `triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_max_mul_reciprocal_unsqueeze_0` |
| 12 | `gemm_unknown` | 400 | 148.301 | 370.753 | 0.61% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)` |
| 13 | `copy_kernel` | 401 | 133.355 | 332.557 | 0.55% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 14 | `vsa_layout` | 400 | 118.455 | 296.138 | 0.49% | `_h3_vsa_o_bundle_kernel` |
| 15 | `h3_pointwise` | 400 | 113.422 | 283.554 | 0.47% | `_indexed_gate_kernel` |
| 16 | `unknown` | 1600 | 103.335 | 64.585 | 0.42% | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int,…` |
| 17 | `unknown` | 400 | 98.631 | 246.578 | 0.40% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, float, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6…` |
| 18 | `gemm_unknown` | 400 | 98.127 | 245.318 | 0.40% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(T1::Params)` |
| 19 | `unknown` | 400 | 97.633 | 244.083 | 0.40% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, int, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6, …` |
| 20 | `h3_pointwise` | 393 | 87.798 | 223.404 | 0.36% | `_rms_norm_indexed_scale_shift_kernel` |
| 21 | `triton_reduction` | 400 | 79.978 | 199.944 | 0.33% | `triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_max_mul_reciprocal_unsqueeze_view_0` |
| 22 | `unknown` | 400 | 49.421 | 123.551 | 0.20% | `void at::native::mbtopk::gatherTopK<float, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, unsigned int, T2, at::cuda::detail::TensorInfo<T1, T2>, …` |
| 23 | `copy_kernel` | 400 | 48.645 | 121.614 | 0.20% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 24 | `unknown` | 1600 | 41.685 | 26.053 | 0.17% | `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(T1 *, short *, unsigned int *, unsigned int *, unsigned int, int, bool, unsigned int *, T2 *, unsigned …` |
| 25 | `unknown` | 401 | 38.629 | 96.331 | 0.16% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned…` |
| 26 | `unknown` | 400 | 36.775 | 91.939 | 0.15% | `void <unnamed>::softmax_warp_forward<float, float, float, (int)11, (bool)0, (bool)0, (int)32>(T2 *, const T1 *, int, int, int, const bool *, int, bool)` |
| 27 | `nccl` | 8 | 28.416 | 3552.018 | 0.12% | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 28 | `unknown` | 1600 | 23.732 | 14.833 | 0.10% | `at::native::mbtopk::computeDigitCumSum(short *, unsigned int *, unsigned int)` |
| 29 | `unknown` | 800 | 14.456 | 18.070 | 0.06% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| 30 | `unknown` | 1200 | 10.275 | 8.562 | 0.04% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float…` |
| 31 | `gemm_unknown` | 8 | 5.079 | 634.835 | 0.02% | `void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(int, int, int, BatchedTensor, int, BatchedTensor, int, BatchedTensor, int,…` |
| 32 | `gemm_unknown` | 8 | 4.666 | 583.239 | 0.02% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_tn_align1>(T1::Params)` |
| 33 | `copy_kernel` | 1200 | 4.056 | 3.380 | 0.02% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 34 | `unknown` | 800 | 3.904 | 4.880 | 0.02% | `void at_cuda_detail::cub::detail::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::detail::scan_by_key::policy_hub<thrust::_V_300200_SM_750_800_860_900_1000_1200::transform…` |
| 35 | `copy_kernel` | 9 | 2.946 | 327.335 | 0.01% | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::…` |
| 36 | `gemm_unknown` | 4 | 2.737 | 684.231 | 0.01% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params)` |
| 37 | `unknown` | 400 | 2.691 | 6.728 | 0.01% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 38 | `copy_kernel` | 400 | 2.380 | 5.949 | 0.01% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 39 | `copy_kernel` | 400 | 2.024 | 5.061 | 0.01% | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::…` |
| 40 | `unknown` | 1200 | 1.949 | 1.624 | 0.01% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimi…` |
| 41 | `h3_pointwise` | 8 | 1.877 | 234.668 | 0.01% | `_indexed_scale_shift_kernel` |
| 42 | `unknown` | 9 | 1.819 | 202.133 | 0.01% | `void vllm::rms_norm_kernel<c10::BFloat16, (int)8, (int)2, (bool)1>(T1 *, const T1 *, long, long, long, long, long, const T1 *, long, float, int, int)` |
| 43 | `unknown` | 67 | 1.728 | 25.795 | 0.01% | `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, long, long, bool)` |
| 44 | `unknown` | 112 | 1.355 | 12.100 | 0.01% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 45 | `gemm_unknown` | 4 | 1.190 | 297.527 | 0.00% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>(T1::Params)` |
| 46 | `copy_kernel` | 403 | 1.170 | 2.902 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned …` |
| 47 | `unknown` | 50 | 1.152 | 23.041 | 0.00% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFuncto…` |
| 48 | `unknown` | 112 | 0.944 | 8.427 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, bool, at::native::<unnamed>::CompareEqFunctor<float>>, std::array<char *, (unsigned …` |
| 49 | `unknown` | 32 | 0.856 | 26.754 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` |
| 50 | `unknown` | 800 | 0.743 | 0.929 | 0.00% | `void at_cuda_detail::cub::detail::scan_by_key::DeviceScanByKeyInitKernel<at_cuda_detail::cub::ReduceByKeyScanTileState<unsigned int, int, (bool)1>, thrust::_V_300200_SM_750_800_86…` |
| 51 | `unknown` | 400 | 0.676 | 1.690 | 0.00% | `void at::native::mbtopk::fill<unsigned int, unsigned int>(T1 *, T1, T2)` |
| 52 | `unknown` | 400 | 0.617 | 1.542 | 0.00% | `void at::native::mbtopk::computeBlockwiseKthCounts<unsigned int>(T1 *, short *, unsigned int, unsigned int, unsigned int *)` |
| 53 | `unknown` | 3 | 0.600 | 199.882 | 0.00% | `void at::native::indexFuncLargeIndex<c10::BFloat16, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1, at::native::<unnamed>::ReduceAdd>(at::cuda::detail::TensorInfo<T1, T3>, a…` |
| 54 | `unknown` | 24 | 0.574 | 23.897 | 0.00% | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIter…` |
| 55 | `unknown` | 97 | 0.568 | 5.851 | 0.00% | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<bool, at::native::func_wrapper_t<bool, at::native::and_kernel_cuda(at::TensorIterator &)::[lambda() (instance…` |
| 56 | `unknown` | 112 | 0.477 | 4.263 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, bool, at::native::<unnamed>::CompareEqFunctor<float>>, std::array<char *, (unsigned …` |
| 57 | `gemm_unknown` | 1 | 0.423 | 423.232 | 0.00% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(T1::Params)` |
| 58 | `unknown` | 112 | 0.402 | 3.591 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<bool, bool, bool, at::native::binary_internal::MulFunctor<bool>>, std::array<char *, (unsigned lon…` |
| 59 | `unknown` | 400 | 0.395 | 0.988 | 0.00% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::FillFunctor<int>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (in…` |
| 60 | `unknown` | 400 | 0.372 | 0.930 | 0.00% | `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scalar &, at::Tensor &)::[lambda() (instance 1)…` |
| 61 | `unknown` | 402 | 0.338 | 0.842 | 0.00% | `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scalar &, at::Tensor &)::[lambda() (instance 1)…` |
| 62 | `copy_kernel` | 12 | 0.321 | 26.784 | 0.00% | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::index_copy_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIterator &, long, long, long)::[lambd…` |
| 63 | `unknown` | 8 | 0.301 | 37.644 | 0.00% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float…` |
| 64 | `unknown` | 83 | 0.232 | 2.799 | 0.00% | `void at_cuda_detail::cub::detail::select::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::select::policy_hub<long, bool, int, (bool)0, (at_cuda_detail::cub::SelectImpl)0>::Po…` |
| 65 | `gemm_unknown` | 1 | 0.182 | 182.207 | 0.00% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 66 | `unknown` | 44 | 0.123 | 2.794 | 0.00% | `void at_cuda_detail::cub::detail::reduce::DeviceReduceKernel<at_cuda_detail::cub::detail::reduce::policy_hub<int, unsigned long long, cuda::std::__4::plus<void>>::Policy1000, thru…` |
| 67 | `cudnn_attention` | 2 | 0.121 | 60.432 | 0.00% | `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_3_64x64x128_4x1x1_cga1x1x1_kernel0_0` |
| 68 | `unknown` | 42 | 0.088 | 2.088 | 0.00% | `void at_cuda_detail::cub::detail::reduce::DeviceReduceSingleTileKernel<at_cuda_detail::cub::detail::reduce::policy_hub<int, unsigned long long, cuda::std::__4::plus<void>>::Policy…` |
| 69 | `gemm_unknown` | 2 | 0.076 | 37.840 | 0.00% | `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)2, (int)2, (bool)1, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, …` |
| 70 | `unknown` | 83 | 0.072 | 0.872 | 0.00% | `void at_cuda_detail::cub::detail::scan::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<int, (bool)1>, int *>(T1, int, T2)` |
| 71 | `unknown` | 1 | 0.060 | 59.840 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<c10::BFloat16>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| 72 | `unknown` | 44 | 0.059 | 1.337 | 0.00% | `void at_cuda_detail::cub::detail::reduce::DeviceReduceSingleTileKernel<at_cuda_detail::cub::detail::reduce::policy_hub<int, unsigned long long, cuda::std::__4::plus<void>>::Policy…` |
| 73 | `unknown` | 12 | 0.055 | 4.619 | 0.00% | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)8>>(at::TensorIterator…` |
| 74 | `copy_kernel` | 11 | 0.054 | 4.911 | 0.00% | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)2, (int)128, (int)1, (int)16, (int)4>(char *, at:…` |
| 75 | `unknown` | 32 | 0.038 | 1.191 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnOther_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 76 | `unknown` | 32 | 0.037 | 1.164 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<float>(at::TensorIteratorBase &, at::native::<unnamed>::OpType, T1)::[lambda(float) (…` |
| 77 | `unknown` | 16 | 0.030 | 1.874 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>, std::array<char *, (unsigned…` |
| 78 | `gemm_unknown` | 1 | 0.029 | 29.056 | 0.00% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(T1::Params)` |
| 79 | `unknown` | 4 | 0.026 | 6.488 | 0.00% | `void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::detail::radix::policy_hub<float, at::cuda::cub::detail::OpaqueType<(int)8>, unsign…` |
| 80 | `unknown` | 16 | 0.019 | 1.178 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool>>, std::array<char *, (unsigned long)3>>(int, …` |

## Top OS runtime calls

OSRT time is summed across threads and is never assigned to RDMA without explicit naming evidence.

| Call | Count | Summed thread-ms |
|---|---:|---:|
| `pthread_cond_timedwait` | 3946 | 423125.187 |
| `poll` | 887 | 299198.358 |
| `epoll_wait` | 24231 | 212140.588 |
| `nanosleep` | 1000 | 14294.077 |
| `epoll_pwait` | 3518 | 7091.576 |
| `sem_clockwait` | 38 | 3705.783 |
| `read` | 130975 | 1374.151 |
| `ioctl` | 2208 | 1348.867 |
| `pthread_cond_wait` | 53 | 1019.701 |
| `getdelim` | 27 | 964.333 |
| `open` | 151168 | 281.100 |
| `fopen` | 225 | 214.303 |
| `stat` | 48643 | 82.282 |
| `waitpid` | 33 | 4.338 |
| `recv` | 129 | 1.610 |
| `send` | 160 | 1.232 |
| `close` | 154 | 1.172 |
| `fgets` | 31 | 0.480 |
| `pthread_cond_signal` | 251 | 0.404 |
| `fwrite_unlocked` | 20 | 0.388 |

## Warnings and attribution limits

- 4.41% of summed kernel GPU-time is unknown; it was not force-fit into a named H3 stage.
- GEMM kernel names do not expose FC1/FC2/QKV/out roles; all recognizable GEMMs remain gemm_unknown.
- FlashInfer Ulysses barrier GPU-time is synchronization/spin residency; it must not be read as RDMA wire time.
- `kernel_totals`: Summed event duration across GPUs/streams; concurrent work double-counts and is not wall time.
- `busy_union`: Union of kernel, memcpy, and memset intervals independently for each GPU.
- `phase_wall`: A disjoint interval sweep per GPU. Same-phase overlap is counted once; cross-phase overlap is counted once as concurrent_mixed; devices are never summed.
- `api_totals`: Summed CPU thread duration; calls on different ranks/threads and nested ranges can overlap.
- `classification`: Kernel/API names only. No temporal inference is used to assign ambiguous GEMMs or stalls.
- `rdma`: Only explicit Ulysses PCIe/RDMA names are attributed. Generic ioctl/poll/CUDA waits are not RDMA.
- `capture_range`: An NVTX-triggered capture can clip kernels already in flight on non-trigger ranks and can perturb the trigger rank; compare absolute step time with an unprofiled run.
- `device_identity`: CUPTI device IDs are trace-local. Physical GPU and PCI bus identity is joined through each kernel's PID and NCCL startup mapping; TARGET_INFO_GPU.id is not directly joined after CUDA_VISIBLE_DEVICES reordering.
- `decode_nvtx`: Explicit CPU NVTX intervals only. Cross-rank overlap is computed on the shared trace clock; it is not CUDA-kernel attribution or a claim of removable latency.

## Canonical `nsys stats` cross-check

- Return code: 0
- Output lines: 190
- Output SHA256: `a98529df9cd1ae81771384c61cf6c82c5460b712b9867bbb33510d48abf3e9ed`
- Reports detected: `{"cuda_api": true, "kernels": true, "memory": true, "nvtx": true}`
