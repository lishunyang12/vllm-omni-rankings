# MiniMax-H3 Nsight analysis

- Source: `/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-vsa-sage-pr4691-20260910/runs/pr4691-nsys/pr4691-warmed.nsys-rep`
- SQLite: `/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-vsa-sage-pr4691-20260910/runs/pr4691-nsys/pr4691-warmed.sqlite`
- Scope: `nvtx_range` / `minimax_h3.denoise.step_03` @ `vllm_omni.minimax_h3`
- Window wall time: **4622.017 ms**
- GPU active-span median: **4621.951 ms**
- Slowest trace device by active span: **device 4 / rank 4 / physical GPU 2 / 4622.014 ms**
- Largest non-barrier kernel sum: **trace device 7 / rank 7 / physical GPU 7 / 3952.321 ms**
- Device identity: **kernel PID -> server NCCL rank/cudaDev/nvmlDev/busId**

> GPU-time below is summed across devices and streams; it is not additive wall time.

## Cross-rank decode NVTX timeline

No explicit nested `minimax_h3.decode.*` stage range was found in scope.

## Per-GPU timeline

| Trace dev | Rank | Physical GPU | PCI bus | PID(s) | Active span ms | Busy union ms | Kernel-sum ms | Non-barrier / barrier kernel ms | Memcpy ms / MiB | Window idle ms | Largest idle gap ms |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0000:06:00.0 | 560046 | 4621.796 | 3964.448 | 3955.362 | 3628.231 / 327.131 | 8.794 / 4887.74 | 657.568 | 3.162 |
| 1 | 1 | 4 | 0000:86:00.0 | 560050 | 4622.010 | 3966.304 | 3957.504 | 3597.643 / 359.861 | 8.524 / 4887.74 | 655.713 | 3.161 |
| 2 | 2 | 1 | 0000:09:00.0 | 560051 | 4621.609 | 3964.483 | 3955.535 | 3591.423 / 364.112 | 8.652 / 4841.11 | 657.534 | 3.165 |
| 3 | 3 | 5 | 0000:89:00.0 | 560052 | 4621.679 | 3965.906 | 3957.114 | 3509.999 / 447.114 | 8.517 / 4841.11 | 656.111 | 3.163 |
| 4 | 4 | 2 | 0000:76:00.0 | 560053 | 4622.014 | 3967.217 | 3958.203 | 3596.125 / 362.078 | 8.717 / 4876.08 | 654.800 | 3.163 |
| 5 | 5 | 6 | 0000:f6:00.0 | 560054 | 4621.936 | 3966.359 | 3958.099 | 3579.245 / 378.854 | 7.989 / 4887.74 | 655.658 | 3.176 |
| 6 | 6 | 3 | 0000:79:00.0 | 560055 | 4622.012 | 3959.099 | 3951.229 | 3693.740 / 257.489 | 7.575 / 4876.08 | 662.917 | 3.179 |
| 7 | 7 | 7 | 0000:f9:00.0 | 560056 | 4621.965 | 3965.015 | 3956.509 | 3952.321 / 4.188 | 8.154 / 4887.74 | 657.002 | 3.179 |

## Per-device wall decomposition (no double-counting)

Per-device disjoint interval sweep. Different simultaneously active phases are charged once to concurrent_mixed; no values are summed across devices. Reference device (largest traced busy union): 4.

| Trace dev | Linear GEMM ms | VSA ms | RDMA barrier ms | RDMA other ms | Pointwise ms | Other single-phase ms | Mixed-concurrent ms | Idle ms | Account error ns |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2512.148 | 656.518 | 327.131 | 0.000 | 75.251 | 393.401 | 0.000 | 657.568 | 0 |
| 1 | 2503.251 | 639.754 | 359.861 | 0.000 | 75.066 | 388.371 | 0.000 | 655.713 | 0 |
| 2 | 2473.279 | 657.395 | 364.112 | 0.000 | 75.132 | 394.565 | 0.000 | 657.534 | 0 |
| 3 | 2424.715 | 629.176 | 447.114 | 0.000 | 75.043 | 389.858 | 0.000 | 656.111 | 0 |
| 4 | 2478.525 | 656.217 | 362.078 | 0.000 | 75.216 | 395.182 | 0.000 | 654.800 | 0 |
| 5 | 2495.074 | 629.298 | 378.854 | 0.000 | 75.199 | 387.934 | 0.000 | 655.658 | 0 |
| 6 | 2575.649 | 657.766 | 257.489 | 0.000 | 75.212 | 392.983 | 0.000 | 662.917 | 0 |
| 7 | 2669.673 | 798.687 | 4.188 | 0.000 | 75.533 | 416.933 | 0.000 | 657.002 | 0 |

## Kernel categories

| Category | Calls | Total GPU-ms | % kernel GPU-time | Median GPU-ms | Slowest trace dev / ms | Evidence rule |
|---|---:|---:|---:|---:|---:|---|
| `gemm_unknown` | 2913 | 20132.314 | 63.61% | 2499.163 | 7 / 2669.673 | Recognizable GEMM, but its H3 layer role is not encoded in the kernel name |
| `vsa_attention` | 400 | 5324.811 | 16.82% | 656.367 | 7 / 798.687 | Explicit block-sparse/VSA attention kernel, including FlashInfer CuTeDSL SM120 BlockSparseAttnForward |
| `flashinfer_rdma_barrier` | 3200 | 2500.827 | 7.90% | 360.969 | 3 / 447.114 | Explicit FlashInfer Ulysses PCIe/RDMA barrier; duration is synchronization residency/wait |
| `unknown` | 17247 | 1052.148 | 3.32% | 131.113 | 7 / 144.784 | No conservative category matched |
| `copy_kernel` | 4896 | 797.965 | 2.52% | 99.415 | 7 / 107.004 | Device-side copy/cat/transpose-like kernel (not a DMA memcpy record) |
| `vsa_layout` | 2400 | 790.460 | 2.50% | 98.849 | 7 / 101.071 | Explicit H3 VSA compact/tile layout kernel |
| `h3_pointwise` | 2008 | 601.544 | 1.90% | 75.191 | 7 / 75.518 | Explicitly named H3 fused modulation, gated-residual, Q/K norm+RoPE, or SwiGLU kernel |
| `sage_quantization` | 1600 | 422.070 | 1.33% | 52.625 | 7 / 54.042 | Explicit Sage Q/K/V quantization and statistics kernels |
| `nccl` | 8 | 27.118 | 0.09% | 3.672 | 3 / 4.638 | NCCL device kernel |
| `cudnn_attention` | 16 | 0.124 | 0.00% | 0.016 | 7 / 0.018 | cuDNN kernel with explicit SDPA/attention/FMHA evidence |
| `triton_reduction` | 32 | 0.109 | 0.00% | 0.013 | 7 / 0.015 | Triton reduction kernel |
| `triton_other` | 32 | 0.065 | 0.00% | 0.008 | 7 / 0.010 | Other Triton kernel |
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

- Observed 3200 calls reconstructed as 400 chronological groups: 400 full-device and 0 partial.
- Layout: 10 barrier slots/layer (B0..B9); the group count is consistent with 40 layer cycles.
- Layout basis: explicit VSA evidence: Q/K/V/gate scatters plus O gather, each with opening and closing barriers.
- Full-group arrival-skew sum: 449.100 ms (9.72% of the selected wall window).
- Dominant latest arrival: trace device 7 in 170/400 groups (42.50%).
- Latest-device work overlapping the skew windows: gemm_unknown=222.189 GPU-ms, vsa_attention=126.636 GPU-ms, vsa_layout=29.782 GPU-ms, h3_pointwise=29.407 GPU-ms, copy_kernel=21.723 GPU-ms, unknown=14.610 GPU-ms; covered union 444.347 ms, uncovered 4.752 ms.
- Arrival-skew sums cover sequential full-device barriers and are already inside step wall time; they indicate load-imbalance exposure, not additional time or guaranteed removable time.

### Per-device barrier residency and arrival lag

| Trace dev | Calls | Residency sum ms | Residency median / p95 / max µs | Arrival-lag median / p95 µs | Latest arrival groups / % |
|---:|---:|---:|---:|---:|---:|
| 0 | 400 | 327.131 | 9.792 / 3401.219 / 3600.194 | 5.040 / 1765.625 | 45 / 11.25% |
| 1 | 400 | 359.861 | 14.240 / 3934.182 / 4840.103 | 3.370 / 1662.875 | 15 / 3.75% |
| 2 | 400 | 364.112 | 9.856 / 3985.767 / 4408.135 | 4.911 / 979.487 | 40 / 10.00% |
| 3 | 400 | 447.114 | 12.192 / 4994.753 / 5604.481 | 2.637 / 53.682 | 24 / 6.00% |
| 4 | 400 | 362.078 | 13.056 / 3839.912 / 4117.992 | 2.459 / 1125.730 | 5 / 1.25% |
| 5 | 400 | 378.854 | 11.376 / 4155.267 / 4245.346 | 2.216 / 1465.382 | 14 / 3.50% |
| 6 | 400 | 257.489 | 7.776 / 3331.775 / 3363.423 | 8.414 / 3058.377 | 87 / 21.75% |
| 7 | 400 | 4.188 | 5.152 / 30.432 / 377.664 | 5.264 / 4989.629 | 170 / 42.50% |

### Per-layer chronological barrier slots

| Slot | Groups (full) | Arrival-skew sum ms | Arrival skew median / p95 / max µs | Release skew median / p95 / max µs | Dominant latest GPU / count | Late-device kernel evidence (GPU-ms) |
|---:|---:|---:|---:|---:|---:|---|
| B0 | 40 (40) | 88.369 | 1372.401 / 5226.238 / 5425.458 | 1.270 / 1.619 / 1.654 | 7 / 27 | gemm_unknown=43.159, vsa_attention=25.523, vsa_layout=5.961, h3_pointwise=5.875, copy_kernel=4.395, unknown=2.922 |
| B1 | 40 (40) | 0.249 | 5.361 / 8.965 / 33.153 | 1.348 / 1.718 / 1.779 | 3 / 8 | uncovered=0.249 |
| B2 | 40 (40) | 89.814 | 2218.733 / 5168.689 / 5463.388 | 1.240 / 1.566 / 1.627 | 7 / 30 | gemm_unknown=44.393, vsa_attention=25.394, vsa_layout=5.955, h3_pointwise=5.878, copy_kernel=4.298, unknown=2.918 |
| B3 | 40 (40) | 0.227 | 5.123 / 9.664 / 12.518 | 1.224 / 1.698 / 1.738 | 7 / 11 | uncovered=0.227 |
| B4 | 40 (40) | 89.706 | 2075.875 / 5052.071 / 5442.950 | 1.223 / 1.576 / 1.651 | 7 / 27 | gemm_unknown=44.772, vsa_attention=25.088, vsa_layout=5.953, h3_pointwise=5.878, copy_kernel=4.364, unknown=2.924 |
| B5 | 40 (40) | 0.201 | 4.835 / 6.610 / 8.124 | 1.083 / 1.584 / 1.630 | 7 / 11 | uncovered=0.201 |
| B6 | 40 (40) | 90.267 | 2077.581 / 5174.697 / 5599.569 | 1.277 / 1.549 / 1.624 | 7 / 25 | gemm_unknown=44.976, vsa_attention=25.557, vsa_layout=5.962, h3_pointwise=5.881, copy_kernel=4.334, unknown=2.921 |
| B7 | 40 (40) | 0.229 | 5.521 / 8.159 / 13.687 | 1.316 / 1.695 / 1.785 | 0 / 8 | uncovered=0.229 |
| B8 | 40 (40) | 89.806 | 2104.822 / 5211.443 / 5536.699 | 1.272 / 1.549 / 1.682 | 7 / 27 | gemm_unknown=44.889, vsa_attention=25.074, vsa_layout=5.951, h3_pointwise=5.895, copy_kernel=4.333, unknown=2.926 |
| B9 | 40 (40) | 0.232 | 5.769 / 7.760 / 10.600 | 1.321 / 1.753 / 1.794 | 6 / 10 | uncovered=0.232 |

### Per-device × barrier slot

| Slot | Trace dev | Calls | Residency sum ms | Residency median / p95 µs | Arrival-lag median / p95 µs | Latest groups |
|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 40 | 63.713 | 812.880 / 3401.698 | 383.005 / 1903.148 | 3 |
| B0 | 1 | 40 | 68.835 | 1206.675 / 3897.255 | 14.055 / 1953.697 | 0 |
| B0 | 2 | 40 | 71.071 | 1098.193 / 4127.655 | 302.421 / 1119.672 | 2 |
| B0 | 3 | 40 | 87.858 | 1376.880 / 5231.361 | 3.476 / 74.706 | 0 |
| B0 | 4 | 40 | 70.435 | 1209.314 / 3894.505 | 195.612 / 1308.652 | 0 |
| B0 | 5 | 40 | 74.811 | 1034.945 / 4181.026 | 2.546 / 1626.980 | 0 |
| B0 | 6 | 40 | 50.266 | 642.640 / 3343.295 | 435.285 / 3207.622 | 8 |
| B0 | 7 | 40 | 0.427 | 4.240 / 30.176 | 1347.063 / 5226.238 | 27 |
| B1 | 0 | 40 | 0.279 | 6.576 / 9.921 | 2.054 / 5.368 | 7 |
| B1 | 1 | 40 | 0.292 | 6.960 / 10.496 | 1.785 / 4.878 | 4 |
| B1 | 2 | 40 | 0.279 | 6.368 / 9.920 | 2.490 / 6.225 | 6 |
| B1 | 3 | 40 | 0.260 | 5.680 / 9.152 | 3.047 / 6.027 | 8 |
| B1 | 4 | 40 | 0.331 | 7.872 / 10.624 | 1.132 / 4.497 | 1 |
| B1 | 5 | 40 | 0.302 | 7.280 / 11.072 | 1.607 / 5.317 | 2 |
| B1 | 6 | 40 | 0.233 | 5.648 / 9.568 | 2.851 / 7.161 | 8 |
| B1 | 7 | 40 | 0.290 | 6.384 / 10.016 | 2.846 / 5.771 | 4 |
| B2 | 0 | 40 | 65.583 | 1565.345 / 3411.202 | 350.555 / 1856.059 | 1 |
| B2 | 1 | 40 | 73.717 | 1470.274 / 3985.478 | 88.692 / 1540.282 | 0 |
| B2 | 2 | 40 | 72.679 | 1752.211 / 4030.215 | 340.989 / 1093.014 | 1 |
| B2 | 3 | 40 | 89.281 | 2219.569 / 5173.825 | 2.774 / 111.744 | 0 |
| B2 | 4 | 40 | 72.461 | 1768.084 / 3913.096 | 311.320 / 1267.652 | 0 |
| B2 | 5 | 40 | 75.842 | 1657.345 / 4211.203 | 6.472 / 1496.135 | 1 |
| B2 | 6 | 40 | 51.568 | 738.256 / 3340.223 | 556.715 / 3146.746 | 7 |
| B2 | 7 | 40 | 0.747 | 4.224 / 37.568 | 2043.077 / 5168.689 | 30 |
| B3 | 0 | 40 | 0.232 | 5.648 / 9.280 | 3.075 / 6.310 | 9 |
| B3 | 1 | 40 | 0.277 | 7.168 / 10.112 | 2.213 / 4.987 | 1 |
| B3 | 2 | 40 | 0.257 | 6.176 / 10.752 | 2.720 / 5.709 | 6 |
| B3 | 3 | 40 | 0.281 | 7.136 / 10.432 | 2.342 / 5.109 | 2 |
| B3 | 4 | 40 | 0.333 | 8.336 / 12.512 | 0.728 / 3.164 | 0 |
| B3 | 5 | 40 | 0.271 | 6.336 / 11.840 | 2.110 / 4.755 | 4 |
| B3 | 6 | 40 | 0.257 | 6.320 / 10.304 | 2.937 / 6.222 | 7 |
| B3 | 7 | 40 | 0.258 | 6.896 / 9.696 | 3.413 / 5.703 | 11 |
| B4 | 0 | 40 | 65.207 | 1420.001 / 3393.859 | 343.739 / 1855.049 | 1 |
| B4 | 1 | 40 | 70.315 | 1176.674 / 3964.902 | 85.827 / 2038.501 | 0 |
| B4 | 2 | 40 | 73.139 | 1705.907 / 4150.791 | 349.023 / 961.735 | 0 |
| B4 | 3 | 40 | 89.501 | 2077.216 / 5057.217 | 2.154 / 31.246 | 0 |
| B4 | 4 | 40 | 72.627 | 1749.972 / 3979.337 | 327.974 / 1077.171 | 0 |
| B4 | 5 | 40 | 74.664 | 1570.450 / 4127.972 | 7.658 / 1690.248 | 0 |
| B4 | 6 | 40 | 51.263 | 638.112 / 3341.376 | 438.998 / 3041.341 | 12 |
| B4 | 7 | 40 | 0.594 | 4.368 / 65.857 | 2039.300 / 5052.071 | 27 |
| B5 | 0 | 40 | 0.217 | 5.344 / 8.320 | 3.038 / 5.677 | 7 |
| B5 | 1 | 40 | 0.263 | 6.816 / 9.312 | 1.327 / 5.232 | 1 |
| B5 | 2 | 40 | 0.238 | 5.392 / 8.800 | 2.708 / 4.947 | 8 |
| B5 | 3 | 40 | 0.238 | 5.792 / 8.384 | 2.451 / 5.633 | 6 |
| B5 | 4 | 40 | 0.295 | 7.488 / 9.920 | 0.637 / 4.980 | 1 |
| B5 | 5 | 40 | 0.281 | 7.264 / 8.992 | 1.184 / 5.388 | 1 |
| B5 | 6 | 40 | 0.246 | 6.560 / 8.800 | 2.346 / 5.470 | 5 |
| B5 | 7 | 40 | 0.230 | 5.472 / 9.248 | 3.263 / 5.694 | 11 |
| B6 | 0 | 40 | 65.910 | 1445.698 / 3409.667 | 378.923 / 1800.308 | 0 |
| B6 | 1 | 40 | 72.159 | 1359.187 / 3940.166 | 150.569 / 1682.983 | 1 |
| B6 | 2 | 40 | 72.721 | 1692.291 / 4171.879 | 379.248 / 1040.147 | 1 |
| B6 | 3 | 40 | 89.549 | 2051.857 / 5179.616 | 5.123 / 103.337 | 0 |
| B6 | 4 | 40 | 72.244 | 1726.980 / 3943.304 | 349.542 / 1235.953 | 0 |
| B6 | 5 | 40 | 77.023 | 1547.330 / 4197.572 | 2.966 / 1457.614 | 0 |
| B6 | 6 | 40 | 50.288 | 690.000 / 3331.008 | 414.884 / 3244.480 | 13 |
| B6 | 7 | 40 | 0.486 | 4.336 / 42.496 | 2068.469 / 5174.697 | 25 |
| B7 | 0 | 40 | 0.236 | 5.760 / 9.088 | 3.060 / 6.153 | 8 |
| B7 | 1 | 40 | 0.264 | 6.512 / 11.137 | 2.171 / 5.467 | 6 |
| B7 | 2 | 40 | 0.254 | 6.448 / 9.792 | 2.234 / 6.009 | 8 |
| B7 | 3 | 40 | 0.257 | 6.688 / 9.408 | 2.457 / 5.931 | 3 |
| B7 | 4 | 40 | 0.296 | 7.456 / 9.888 | 1.369 / 4.589 | 3 |
| B7 | 5 | 40 | 0.270 | 7.008 / 10.144 | 2.120 / 5.615 | 3 |
| B7 | 6 | 40 | 0.269 | 7.104 / 10.368 | 1.815 / 6.889 | 6 |
| B7 | 7 | 40 | 0.290 | 7.152 / 11.360 | 2.542 / 4.755 | 3 |
| B8 | 0 | 40 | 65.517 | 1446.097 / 3416.803 | 364.686 / 1906.118 | 1 |
| B8 | 1 | 40 | 73.459 | 1221.986 / 4155.494 | 71.556 / 1777.724 | 0 |
| B8 | 2 | 40 | 73.219 | 1714.723 / 4195.463 | 385.497 / 1020.919 | 1 |
| B8 | 3 | 40 | 89.608 | 2098.017 / 5216.384 | 0.417 / 53.682 | 0 |
| B8 | 4 | 40 | 72.745 | 1731.988 / 4001.833 | 357.261 / 1141.555 | 0 |
| B8 | 5 | 40 | 75.109 | 1538.322 / 4145.442 | 8.009 / 1578.038 | 0 |
| B8 | 6 | 40 | 52.858 | 733.040 / 3343.264 | 415.719 / 2968.295 | 11 |
| B8 | 7 | 40 | 0.589 | 4.288 / 39.616 | 2052.498 / 5211.443 | 27 |
| B9 | 0 | 40 | 0.236 | 6.000 / 9.120 | 2.990 / 6.284 | 8 |
| B9 | 1 | 40 | 0.279 | 7.184 / 9.760 | 2.034 / 5.760 | 2 |
| B9 | 2 | 40 | 0.255 | 6.480 / 9.249 | 2.498 / 6.265 | 7 |
| B9 | 3 | 40 | 0.280 | 7.344 / 9.824 | 1.922 / 5.530 | 5 |
| B9 | 4 | 40 | 0.313 | 8.032 / 10.209 | 0.992 / 4.286 | 0 |
| B9 | 5 | 40 | 0.279 | 7.232 / 9.312 | 1.564 / 6.104 | 3 |
| B9 | 6 | 40 | 0.241 | 6.544 / 8.480 | 2.937 / 7.355 | 10 |
| B9 | 7 | 40 | 0.275 | 7.360 / 9.504 | 2.620 / 6.298 | 5 |

Attribution limits:

- Barrier slots B0..B9 are chronological modulo-10 positions; the trace has no semantic exchange labels.
- Barrier kernel residency is synchronization/spin time, not RDMA byte-transfer time.
- The latest-arriving device constrains that barrier release, but the trace alone does not identify why its preceding work was late.
- Late-path kernel evidence is overlap on the latest-arriving device during the arrival-skew window; uncovered time may be CPU, transport, DMA, or untraced work.

## FlashInfer/RDMA evidence and launch gaps

- Explicitly named kernels: 3200
- Matched CUDA launches: 3199; unmatched: 1
- Launch→GPU gap median/p95/max: 3.559 / 51733.517 / 54913.861 µs
- The CUDA launch API return-to-GPU-start gap includes stream queueing and backpressure; it is neither pure CPU dispatch overhead nor network transfer duration.

## CUDA API categories

| Category | Calls | Summed thread-ms | Interpretation |
|---|---:|---:|---|
| `synchronization_wait` | 3551 | 30958.719 | CUDA synchronize calls; duration includes waiting and is not CPU overhead alone |
| `kernel_launch` | 34759 | 113.091 | CUDA kernel/graph launch APIs |
| `memcpy` | 6014 | 30.824 | CUDA memcpy APIs |
| `memory_management` | 10 | 13.667 | CUDA allocation, mapping, and free APIs |
| `event_stream` | 10623 | 8.236 | CUDA event and stream bookkeeping APIs |
| `memset` | 935 | 6.494 | CUDA memset APIs |
| `other` | 28700 | 3.942 | Other CUDA APIs |
| `external_memory_possible_rdma_setup` | 0 | 0.000 | CUDA external/shareable-memory APIs; possible registration setup, not measured RDMA transfer time |

## Memcpy operations

| Trace dev | Kind | Calls | MiB (overlapping ops) | GPU-ms |
|---:|---|---:|---:|---:|
| 0 | Device-to-Device | 310 | 4887.713 | 8.615 |
| 0 | Device-to-Host | 436 | 0.025 | 0.175 |
| 0 | Host-to-Device | 10 | 0.000 | 0.003 |
| 1 | Device-to-Device | 310 | 4887.713 | 8.355 |
| 1 | Device-to-Host | 432 | 0.024 | 0.167 |
| 1 | Host-to-Device | 6 | 0.000 | 0.002 |
| 2 | Device-to-Device | 308 | 4841.088 | 8.474 |
| 2 | Device-to-Host | 434 | 0.025 | 0.175 |
| 2 | Host-to-Device | 9 | 0.000 | 0.003 |
| 3 | Device-to-Device | 308 | 4841.088 | 8.351 |
| 3 | Device-to-Host | 435 | 0.025 | 0.164 |
| 3 | Host-to-Device | 9 | 0.000 | 0.002 |
| 4 | Device-to-Device | 309 | 4876.056 | 8.542 |
| 4 | Device-to-Host | 432 | 0.024 | 0.173 |
| 4 | Host-to-Device | 6 | 0.000 | 0.002 |
| 5 | Device-to-Device | 310 | 4887.713 | 7.824 |
| 5 | Device-to-Host | 436 | 0.025 | 0.163 |
| 5 | Host-to-Device | 10 | 0.000 | 0.003 |
| 6 | Device-to-Device | 309 | 4876.056 | 7.402 |
| 6 | Device-to-Host | 432 | 0.024 | 0.171 |
| 6 | Host-to-Device | 6 | 0.000 | 0.002 |
| 7 | Device-to-Device | 310 | 4887.713 | 7.936 |
| 7 | Device-to-Host | 436 | 0.025 | 0.214 |
| 7 | Host-to-Device | 10 | 0.000 | 0.003 |

## Communication NVTX evidence

These CPU-range totals can overlap because NCCL group/collective ranges may be nested.

| Evidence | Range | Calls | Summed CPU-range ms |
|---|---|---:|---:|
| `nccl` | `ncclAllGather` | 8 | 0.609 |

## Top kernels

| # | Category | Calls | Total GPU-ms | Avg µs | % kernel GPU-time | Kernel |
|---:|---|---:|---:|---:|---:|---|
| 1 | `gemm_unknown` | 1248 | 16354.951 | 13104.928 | 51.68% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params)` |
| 2 | `vsa_attention` | 400 | 5324.811 | 13312.027 | 16.82% | `kernel_cutlass_kernel_flashinfercute_dslsparsesm120_blk64flash_fwd_sm120_sageBlockSparseAttnForwardSageSm120Blk64_object_at__tensor0000o12811101213_tensor0000o12811101213_tensor00…` |
| 3 | `gemm_unknown` | 800 | 3511.228 | 4389.035 | 11.09% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8>(T1::Params)` |
| 4 | `flashinfer_rdma_barrier` | 3200 | 2500.827 | 781.508 | 7.90% | `flashinfer::comm::ulysses_pcie::UlyssesPcieBarrier(unsigned long *, flashinfer::comm::ulysses_pcie::PeerSignalPointers, int, int, unsigned long *)` |
| 5 | `copy_kernel` | 2008 | 733.646 | 365.361 | 2.32% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 6 | `vsa_layout` | 1600 | 499.223 | 312.014 | 1.58% | `_h3_vsa_tile_pack_kernel` |
| 7 | `unknown` | 400 | 340.782 | 851.955 | 1.08% | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vllm::packed_silu_kernel<__nv_bfloat162>, (bool)1, (bool)1, (bool)0, (bool)1>(T1 *…` |
| 8 | `h3_pointwise` | 800 | 234.285 | 292.857 | 0.74% | `_rms_norm_rope_kernel` |
| 9 | `unknown` | 1200 | 197.636 | 164.697 | 0.62% | `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<float, at::native::sum_functor<c10::BFloat16, float, float>::operat…` |
| 10 | `sage_quantization` | 400 | 182.742 | 456.856 | 0.58% | `_quantize_sage_kv_kernel` |
| 11 | `vsa_layout` | 400 | 173.818 | 434.544 | 0.55% | `_h3_vsa_o_bundle_local_gate_kernel` |
| 12 | `h3_pointwise` | 400 | 164.003 | 410.009 | 0.52% | `_indexed_gate_rms_norm_scale_shift_kernel` |
| 13 | `gemm_unknown` | 400 | 148.243 | 370.608 | 0.47% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)` |
| 14 | `sage_quantization` | 400 | 142.698 | 356.746 | 0.45% | `_sage_kv_stats_partial_kernel` |
| 15 | `vsa_layout` | 400 | 117.420 | 293.549 | 0.37% | `_h3_vsa_o_bundle_kernel` |
| 16 | `h3_pointwise` | 400 | 112.618 | 281.546 | 0.36% | `_indexed_gate_kernel` |
| 17 | `gemm_unknown` | 400 | 99.378 | 248.444 | 0.31% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(T1::Params)` |
| 18 | `unknown` | 400 | 98.797 | 246.994 | 0.31% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, float, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6…` |
| 19 | `unknown` | 400 | 97.690 | 244.225 | 0.31% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, int, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6, …` |
| 20 | `sage_quantization` | 400 | 90.247 | 225.618 | 0.29% | `_quantize_sage_q_kernel` |
| 21 | `h3_pointwise` | 400 | 88.778 | 221.945 | 0.28% | `_rms_norm_indexed_scale_shift_kernel` |
| 22 | `unknown` | 1600 | 82.175 | 51.359 | 0.26% | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int,…` |
| 23 | `copy_kernel` | 400 | 49.149 | 122.873 | 0.16% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 24 | `unknown` | 400 | 45.563 | 113.906 | 0.14% | `void at::native::mbtopk::gatherTopK<float, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, unsigned int, T2, at::cuda::detail::TensorInfo<T1, T2>, …` |
| 25 | `unknown` | 400 | 43.630 | 109.075 | 0.14% | `void <unnamed>::softmax_warp_forward<float, float, float, (int)11, (bool)0, (bool)0, (int)32>(T2 *, const T1 *, int, int, int, const bool *, int, bool)` |
| 26 | `unknown` | 1600 | 38.004 | 23.752 | 0.12% | `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(T1 *, short *, unsigned int *, unsigned int *, unsigned int, int, bool, unsigned int *, T2 *, unsigned …` |
| 27 | `unknown` | 408 | 33.403 | 81.870 | 0.11% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned…` |
| 28 | `nccl` | 8 | 27.118 | 3389.764 | 0.09% | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 29 | `unknown` | 1600 | 23.386 | 14.616 | 0.07% | `at::native::mbtopk::computeDigitCumSum(short *, unsigned int *, unsigned int)` |
| 30 | `unknown` | 800 | 13.715 | 17.144 | 0.04% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |

## Top OS runtime calls

OSRT time is summed across threads and is never assigned to RDMA without explicit naming evidence.

| Call | Count | Summed thread-ms |
|---|---:|---:|
| `pthread_cond_timedwait` | 5159 | 563796.192 |
| `poll` | 1226 | 406718.507 |
| `epoll_wait` | 30064 | 277080.513 |
| `sem_clockwait` | 76 | 78569.671 |
| `pthread_cond_wait` | 14 | 46202.725 |
| `epoll_pwait` | 4352 | 9165.822 |
| `sem_wait` | 1 | 4622.017 |
| `ioctl` | 108 | 63.363 |
| `waitpid` | 19 | 2.633 |
| `recv` | 180 | 2.439 |
| `send` | 193 | 1.539 |
| `pthread_cond_signal` | 426 | 0.628 |
| `write` | 39 | 0.407 |
| `read` | 30 | 0.076 |
| `pthread_mutex_lock` | 14 | 0.071 |
| `pthread_cond_broadcast` | 1 | 0.008 |

## Warnings and attribution limits

- NCCL logger rank 6 disagrees with communicator rank 2 for PID 560055; ignored that line.
- 3.32% of summed kernel GPU-time is unknown; it was not force-fit into a named H3 stage.
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
