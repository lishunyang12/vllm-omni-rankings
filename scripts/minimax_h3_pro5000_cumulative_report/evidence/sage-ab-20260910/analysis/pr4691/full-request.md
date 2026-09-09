# MiniMax-H3 Nsight analysis

- Source: `/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-vsa-sage-pr4691-20260910/runs/pr4691-nsys/pr4691-warmed.nsys-rep`
- SQLite: `/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-vsa-sage-pr4691-20260910/runs/pr4691-nsys/pr4691-warmed.sqlite`
- Scope: `full_cuda_trace` / `full CUDA trace`
- Window wall time: **26302.164 ms**
- GPU active-span median: **26163.477 ms**
- Slowest trace device by active span: **device 0 / rank 0 / physical GPU 0 / 26301.688 ms**
- Largest non-barrier kernel sum: **trace device 7 / rank 7 / physical GPU 7 / 21688.408 ms**
- Device identity: **kernel PID -> server NCCL rank/cudaDev/nvmlDev/busId**

> GPU-time below is summed across devices and streams; it is not additive wall time.

## Cross-rank decode NVTX timeline

Rank mapping: NVTX PID -> server NCCL rank mapping.

| Stage | Rank | PID | Start / end offset ms | Wall ms | Clipped |
|---|---:|---:|---:|---:|---|
| `mp4_finish` | 0 | 560046 | 26301.992 / 26302.164 | 0.172 | yes |

| Overlap comparison | Video ms | Audio ms | Overlap ms | Audio hidden | Video overlap |
|---|---:|---:|---:|---:|---:|
| TAEH3 video vs Audio VAE | n/a | n/a | n/a | n/a | n/a |
| TAEH3 video vs AAC encode | n/a | n/a | n/a | n/a | n/a |
| TAEH3 video vs rank-1 audio pipeline | n/a | n/a | n/a | n/a | n/a |

Attribution limits:

- Stage durations are CPU NVTX wall ranges, not summed CPU utilization or GPU kernel attribution.
- Ranges on different ranks share a timestamp axis; per-stage summed time can double-count overlap.
- The hidden percentages describe observed interval overlap, not guaranteed removable latency.

## Per-GPU timeline

| Trace dev | Rank | Physical GPU | PCI bus | PID(s) | Active span ms | Busy union ms | Kernel-sum ms | Non-barrier / barrier kernel ms | Memcpy ms / MiB | Window idle ms | Largest idle gap ms |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0000:06:00.0 | 560046 | 26301.688 | 22128.899 | 22045.463 | 20725.893 / 1319.571 | 90.270 / 39999.79 | 4173.265 | 414.202 |
| 1 | 1 | 4 | 0000:86:00.0 | 560050 | 26163.268 | 21817.545 | 21743.223 | 20227.888 / 1515.335 | 65.594 / 35503.83 | 4484.619 | 418.320 |
| 2 | 2 | 1 | 0000:09:00.0 | 560051 | 26163.639 | 21823.410 | 21748.004 | 20274.429 / 1473.574 | 66.415 / 35503.83 | 4478.754 | 419.385 |
| 3 | 3 | 5 | 0000:89:00.0 | 560052 | 26163.301 | 21779.109 | 21704.656 | 19895.412 / 1809.244 | 65.770 / 35503.83 | 4523.055 | 419.457 |
| 4 | 4 | 2 | 0000:76:00.0 | 560053 | 26163.423 | 21867.387 | 21792.821 | 20346.283 / 1446.538 | 67.657 / 36385.17 | 4434.777 | 420.353 |
| 5 | 5 | 6 | 0000:f6:00.0 | 560054 | 26163.043 | 21823.177 | 21751.375 | 20202.702 / 1548.673 | 65.066 / 36385.17 | 4478.986 | 418.613 |
| 6 | 6 | 3 | 0000:79:00.0 | 560055 | 26163.531 | 21826.435 | 21756.319 | 20691.161 / 1065.159 | 63.205 / 36385.17 | 4475.728 | 418.129 |
| 7 | 7 | 7 | 0000:f9:00.0 | 560056 | 26163.998 | 21799.260 | 21725.558 | 21688.408 / 37.150 | 66.019 / 36385.17 | 4502.903 | 420.012 |

## Per-device wall decomposition (no double-counting)

Per-device disjoint interval sweep. Different simultaneously active phases are charged once to concurrent_mixed; no values are summed across devices. Reference device (largest traced busy union): 0.

| Trace dev | Linear GEMM ms | VSA ms | RDMA barrier ms | RDMA other ms | Pointwise ms | Other single-phase ms | Mixed-concurrent ms | Idle ms | Account error ns |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14211.506 | 2626.328 | 1319.571 | 0.000 | 300.756 | 3655.456 | 15.281 | 4173.265 | 0 |
| 1 | 14058.492 | 2553.272 | 1515.335 | 0.000 | 300.447 | 3389.998 | 0.000 | 4484.619 | 0 |
| 2 | 13999.937 | 2629.578 | 1473.574 | 0.000 | 300.362 | 3419.958 | 0.000 | 4478.754 | 0 |
| 3 | 13718.857 | 2514.896 | 1809.244 | 0.000 | 300.018 | 3436.094 | 0.000 | 4523.055 | 0 |
| 4 | 12970.257 | 2625.379 | 1446.538 | 0.000 | 300.837 | 4524.376 | 0.000 | 4434.777 | 0 |
| 5 | 12959.194 | 2512.531 | 1548.673 | 0.000 | 300.809 | 4501.971 | 0.000 | 4478.986 | 0 |
| 6 | 13391.030 | 2630.465 | 1065.159 | 0.000 | 300.588 | 4439.193 | 0.000 | 4475.728 | 0 |
| 7 | 14126.564 | 3195.072 | 37.150 | 0.000 | 302.327 | 4138.148 | 0.000 | 4502.903 | 0 |

## Kernel categories

| Category | Calls | Total GPU-ms | % kernel GPU-time | Median GPU-ms | Slowest trace dev / ms | Evidence rule |
|---|---:|---:|---:|---:|---:|---|
| `gemm_unknown` | 102129 | 109444.263 | 62.80% | 13859.397 | 0 / 14219.931 | Recognizable GEMM, but its H3 layer role is not encoded in the kernel name |
| `vsa_attention` | 1600 | 21287.522 | 12.22% | 2625.854 | 7 / 3195.072 | Explicit block-sparse/VSA attention kernel, including FlashInfer CuTeDSL SM120 BlockSparseAttnForward |
| `nccl` | 1824 | 11711.621 | 6.72% | 1271.801 | 5 / 2214.062 | NCCL device kernel |
| `flashinfer_rdma_barrier` | 12800 | 10215.243 | 5.86% | 1460.056 | 3 / 1809.244 | Explicit FlashInfer Ulysses PCIe/RDMA barrier; duration is synchronization residency/wait |
| `unknown` | 317087 | 6767.624 | 3.88% | 848.694 | 0 / 972.978 | No conservative category matched |
| `copy_kernel` | 95371 | 3830.143 | 2.20% | 472.122 | 7 / 510.855 | Device-side copy/cat/transpose-like kernel (not a DMA memcpy record) |
| `attention_other` | 21568 | 3695.884 | 2.12% | 478.039 | 0 / 532.313 | Non-cuDNN kernel with explicit attention/FMHA/flash-attention evidence |
| `vsa_layout` | 9600 | 3163.134 | 1.82% | 395.614 | 7 / 404.429 | Explicit H3 VSA compact/tile layout kernel |
| `h3_pointwise` | 8032 | 2405.707 | 1.38% | 300.617 | 7 / 302.265 | Explicitly named H3 fused modulation, gated-residual, Q/K norm+RoPE, or SwiGLU kernel |
| `sage_quantization` | 6400 | 1683.445 | 0.97% | 209.699 | 7 / 216.716 | Explicit Sage Q/K/V quantization and statistics kernels |
| `cudnn_other` | 480 | 61.642 | 0.04% | 0.000 | 0 / 61.642 | cuDNN kernel without explicit attention evidence |
| `cudnn_attention` | 64 | 0.494 | 0.00% | 0.062 | 7 / 0.074 | cuDNN kernel with explicit SDPA/attention/FMHA evidence |
| `triton_reduction` | 128 | 0.435 | 0.00% | 0.053 | 7 / 0.061 | Triton reduction kernel |
| `triton_other` | 128 | 0.261 | 0.00% | 0.032 | 7 / 0.039 | Other Triton kernel |
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

- Observed 12800 calls reconstructed as 1600 chronological groups: 1600 full-device and 0 partial.
- Layout: 10 barrier slots/layer (B0..B9); the group count is consistent with 160 layer cycles.
- Layout basis: explicit VSA evidence: Q/K/V/gate scatters plus O gather, each with opening and closing barriers.
- Full-group arrival-skew sum: 1831.604 ms (6.96% of the selected wall window).
- Dominant latest arrival: trace device 7 in 692/1600 groups (43.25%).
- Latest-device work overlapping the skew windows: gemm_unknown=910.355 GPU-ms, vsa_attention=516.568 GPU-ms, vsa_layout=119.004 GPU-ms, h3_pointwise=116.998 GPU-ms, copy_kernel=86.332 GPU-ms, unknown=58.224 GPU-ms; covered union 1807.481 ms, uncovered 24.123 ms.
- Arrival-skew sums cover sequential full-device barriers and are already inside step wall time; they indicate load-imbalance exposure, not additional time or guaranteed removable time.

### Per-device barrier residency and arrival lag

| Trace dev | Calls | Residency sum ms | Residency median / p95 / max µs | Arrival-lag median / p95 µs | Latest arrival groups / % |
|---:|---:|---:|---:|---:|---:|
| 0 | 1600 | 1319.571 | 9.952 / 3423.458 / 3923.521 | 5.091 / 1767.951 | 150 / 9.38% |
| 1 | 1600 | 1515.335 | 12.416 / 4032.484 / 5672.773 | 3.324 / 1435.197 | 69 / 4.31% |
| 2 | 1600 | 1473.574 | 8.992 / 4030.565 / 10932.909 | 5.511 / 999.530 | 201 / 12.56% |
| 3 | 1600 | 1809.244 | 12.224 / 5070.946 / 9683.706 | 2.686 / 72.893 | 107 / 6.69% |
| 4 | 1600 | 1446.538 | 12.480 / 3859.334 / 4432.006 | 3.354 / 1190.259 | 42 / 2.62% |
| 5 | 1600 | 1548.673 | 10.608 / 4211.300 / 18016.226 | 2.681 / 1438.283 | 82 / 5.12% |
| 6 | 1600 | 1065.159 | 8.096 / 3355.105 / 9468.184 | 6.460 / 3036.243 | 257 / 16.06% |
| 7 | 1600 | 37.150 | 5.120 / 37.568 / 16824.478 | 5.396 / 5058.318 | 692 / 43.25% |

### Per-layer chronological barrier slots

| Slot | Groups (full) | Arrival-skew sum ms | Arrival skew median / p95 / max µs | Release skew median / p95 / max µs | Dominant latest GPU / count | Late-device kernel evidence (GPU-ms) |
|---:|---:|---:|---:|---:|---:|---|
| B0 | 160 (160) | 357.277 | 371.207 / 5258.842 / 5886.947 | 1.248 / 1.583 / 1.657 | 7 / 110 | gemm_unknown=175.453, vsa_attention=102.714, vsa_layout=23.820, h3_pointwise=22.965, copy_kernel=17.460, unknown=11.697 |
| B1 | 160 (160) | 1.287 | 5.263 / 9.102 / 249.553 | 1.200 / 1.725 / 2.140 | 7 / 29 | uncovered=1.287 |
| B2 | 160 (160) | 364.696 | 2200.966 / 5328.328 / 5975.702 | 1.246 / 1.596 / 2.115 | 7 / 116 | gemm_unknown=182.298, vsa_attention=102.723, vsa_layout=23.797, h3_pointwise=23.514, copy_kernel=17.340, unknown=11.707 |
| B3 | 160 (160) | 0.930 | 5.136 / 10.235 / 27.908 | 1.187 / 1.670 / 1.864 | 7 / 38 | uncovered=0.930 |
| B4 | 160 (160) | 361.898 | 810.016 / 5249.962 / 6002.410 | 1.244 / 1.600 / 1.760 | 7 / 106 | gemm_unknown=184.101, vsa_attention=98.311, vsa_layout=23.793, h3_pointwise=23.491, copy_kernel=17.102, unknown=11.698 |
| B5 | 160 (160) | 0.855 | 5.090 / 8.124 / 10.760 | 1.236 / 1.630 / 1.910 | 7 / 31 | uncovered=0.855 |
| B6 | 160 (160) | 377.024 | 452.996 / 5493.318 / 18009.064 | 1.264 / 1.585 / 1.887 | 7 / 108 | gemm_unknown=184.384, vsa_attention=110.976, vsa_layout=23.796, h3_pointwise=23.504, copy_kernel=17.042, unknown=11.406 |
| B7 | 160 (160) | 1.328 | 5.314 / 9.795 / 367.732 | 1.206 / 1.675 / 1.792 | 2 / 33 | uncovered=1.328 |
| B8 | 160 (160) | 365.211 | 2099.476 / 5342.031 / 6121.262 | 1.236 / 1.600 / 1.730 | 7 / 108 | gemm_unknown=184.118, vsa_attention=101.843, vsa_layout=23.799, h3_pointwise=23.523, copy_kernel=17.387, unknown=11.715 |
| B9 | 160 (160) | 1.099 | 5.438 / 8.144 / 106.382 | 1.179 / 1.720 / 1.794 | 2 / 33 | uncovered=1.099 |

### Per-device × barrier slot

| Slot | Trace dev | Calls | Residency sum ms | Residency median / p95 µs | Arrival-lag median / p95 µs | Latest groups |
|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 160 | 256.575 | 105.744 / 3437.828 | 210.287 / 1903.148 | 6 |
| B0 | 1 | 160 | 293.793 | 368.704 / 4384.614 | 26.090 / 1898.311 | 2 |
| B0 | 2 | 160 | 287.262 | 358.256 / 4256.837 | 88.737 / 1109.071 | 13 |
| B0 | 3 | 160 | 355.210 | 352.096 / 5263.837 | 4.340 / 84.175 | 1 |
| B0 | 4 | 160 | 282.907 | 373.040 / 3943.461 | 52.361 / 1361.966 | 0 |
| B0 | 5 | 160 | 302.448 | 256.017 / 4224.772 | 5.533 / 1626.980 | 4 |
| B0 | 6 | 160 | 208.126 | 363.600 / 3390.494 | 98.202 / 3211.994 | 24 |
| B0 | 7 | 160 | 3.458 | 4.400 / 46.304 | 46.297 / 5258.842 | 110 |
| B1 | 0 | 160 | 1.399 | 6.368 / 11.360 | 2.088 / 5.955 | 23 |
| B1 | 1 | 160 | 1.462 | 6.944 / 11.072 | 2.216 / 5.567 | 15 |
| B1 | 2 | 160 | 1.359 | 5.888 / 9.920 | 2.944 / 6.558 | 28 |
| B1 | 3 | 160 | 1.438 | 6.768 / 11.168 | 2.380 / 5.839 | 23 |
| B1 | 4 | 160 | 1.567 | 7.376 / 11.680 | 1.311 / 4.618 | 9 |
| B1 | 5 | 160 | 1.393 | 7.008 / 11.392 | 1.538 / 5.770 | 9 |
| B1 | 6 | 160 | 1.138 | 6.624 / 9.920 | 2.422 / 7.161 | 24 |
| B1 | 7 | 160 | 1.428 | 6.320 / 11.584 | 2.764 / 6.459 | 29 |
| B2 | 0 | 160 | 266.222 | 1501.585 / 3534.465 | 501.129 / 1843.096 | 6 |
| B2 | 1 | 160 | 308.189 | 1225.058 / 4546.789 | 82.901 / 1495.105 | 1 |
| B2 | 2 | 160 | 293.776 | 1752.211 / 4268.741 | 372.813 / 1104.431 | 11 |
| B2 | 3 | 160 | 362.077 | 2201.490 / 5333.473 | 3.171 / 123.400 | 0 |
| B2 | 4 | 160 | 290.405 | 1770.468 / 3993.894 | 330.062 / 1350.013 | 1 |
| B2 | 5 | 160 | 308.884 | 1571.569 / 4271.745 | 6.216 / 1444.927 | 3 |
| B2 | 6 | 160 | 212.394 | 812.928 / 3391.677 | 516.004 / 3121.784 | 22 |
| B2 | 7 | 160 | 2.656 | 4.240 / 53.824 | 2026.835 / 5328.328 | 116 |
| B3 | 0 | 160 | 1.046 | 6.208 / 10.176 | 2.345 / 6.185 | 22 |
| B3 | 1 | 160 | 1.145 | 7.072 / 11.136 | 1.467 / 5.457 | 12 |
| B3 | 2 | 160 | 1.015 | 6.144 / 9.824 | 2.756 / 6.671 | 23 |
| B3 | 3 | 160 | 1.094 | 6.496 / 11.200 | 2.413 / 5.760 | 17 |
| B3 | 4 | 160 | 1.258 | 7.680 / 12.512 | 1.155 / 4.530 | 6 |
| B3 | 5 | 160 | 1.081 | 6.864 / 11.264 | 1.825 / 5.676 | 16 |
| B3 | 6 | 160 | 1.024 | 6.400 / 9.408 | 2.918 / 7.787 | 26 |
| B3 | 7 | 160 | 1.059 | 6.624 / 11.008 | 2.768 / 6.119 | 38 |
| B4 | 0 | 160 | 262.315 | 347.489 / 3470.145 | 70.489 / 1855.049 | 7 |
| B4 | 1 | 160 | 293.817 | 328.097 / 4631.109 | 78.232 / 2062.803 | 1 |
| B4 | 2 | 160 | 291.077 | 524.897 / 4366.278 | 218.255 / 1017.927 | 9 |
| B4 | 3 | 160 | 356.872 | 323.440 / 5254.690 | 2.771 / 54.409 | 0 |
| B4 | 4 | 160 | 286.272 | 324.625 / 4007.558 | 339.550 / 1285.606 | 1 |
| B4 | 5 | 160 | 298.587 | 206.144 / 4219.459 | 8.405 / 1663.224 | 3 |
| B4 | 6 | 160 | 207.535 | 481.952 / 3377.121 | 383.402 / 3110.275 | 33 |
| B4 | 7 | 160 | 3.363 | 4.384 / 70.080 | 356.105 / 5249.962 | 106 |
| B5 | 0 | 160 | 0.940 | 5.776 / 9.280 | 3.011 / 6.056 | 29 |
| B5 | 1 | 160 | 1.073 | 6.816 / 10.304 | 1.697 / 5.232 | 6 |
| B5 | 2 | 160 | 0.973 | 6.224 / 9.376 | 2.601 / 6.381 | 28 |
| B5 | 3 | 160 | 1.020 | 6.384 / 9.376 | 2.477 / 5.821 | 22 |
| B5 | 4 | 160 | 1.179 | 7.488 / 10.528 | 1.152 / 4.983 | 7 |
| B5 | 5 | 160 | 1.076 | 7.008 / 9.664 | 1.473 / 6.006 | 13 |
| B5 | 6 | 160 | 1.020 | 6.688 / 9.344 | 2.458 / 5.941 | 24 |
| B5 | 7 | 160 | 1.026 | 6.576 / 10.304 | 2.742 / 6.079 | 31 |
| B6 | 0 | 160 | 261.213 | 97.936 / 3488.738 | 181.536 / 1831.891 | 4 |
| B6 | 1 | 160 | 303.761 | 454.577 / 4454.341 | 24.822 / 1595.789 | 1 |
| B6 | 2 | 160 | 299.511 | 80.416 / 4337.509 | 346.673 / 1060.506 | 10 |
| B6 | 3 | 160 | 365.323 | 114.144 / 5498.334 | 4.278 / 116.915 | 0 |
| B6 | 4 | 160 | 287.848 | 243.312 / 4001.254 | 225.997 / 1464.943 | 2 |
| B6 | 5 | 160 | 323.674 | 285.648 / 4252.770 | 6.080 / 1457.614 | 3 |
| B6 | 6 | 160 | 213.995 | 93.568 / 3384.765 | 355.815 / 3237.439 | 32 |
| B6 | 7 | 160 | 19.258 | 4.448 / 46.720 | 197.237 / 5277.978 | 108 |
| B7 | 0 | 160 | 1.409 | 6.000 / 10.720 | 2.801 / 5.947 | 29 |
| B7 | 1 | 160 | 1.117 | 6.912 / 11.392 | 1.890 / 5.949 | 17 |
| B7 | 2 | 160 | 1.378 | 5.808 / 10.784 | 2.600 / 6.631 | 33 |
| B7 | 3 | 160 | 1.373 | 6.176 / 10.592 | 2.625 / 6.322 | 22 |
| B7 | 4 | 160 | 1.597 | 7.520 / 12.256 | 1.278 / 5.315 | 11 |
| B7 | 5 | 160 | 1.473 | 6.656 / 10.848 | 2.147 / 5.820 | 12 |
| B7 | 6 | 160 | 1.494 | 7.104 / 10.720 | 1.859 / 6.889 | 17 |
| B7 | 7 | 160 | 1.500 | 6.944 / 11.168 | 2.712 / 5.750 | 19 |
| B8 | 0 | 160 | 267.293 | 1375.441 / 3522.850 | 358.416 / 1813.638 | 3 |
| B8 | 1 | 160 | 309.662 | 1079.427 / 4709.448 | 51.709 / 1777.724 | 1 |
| B8 | 2 | 160 | 296.040 | 1587.683 / 4343.431 | 364.190 / 1046.002 | 13 |
| B8 | 3 | 160 | 363.611 | 2092.290 / 5346.877 | 1.541 / 110.569 | 0 |
| B8 | 4 | 160 | 292.202 | 1744.484 / 4037.190 | 357.261 / 1293.083 | 0 |
| B8 | 5 | 160 | 308.752 | 1502.162 / 4278.306 | 10.215 / 1553.884 | 7 |
| B8 | 6 | 160 | 217.314 | 653.968 / 3388.925 | 381.333 / 3019.787 | 28 |
| B8 | 7 | 160 | 2.152 | 4.384 / 48.416 | 2058.039 / 5342.031 | 108 |
| B9 | 0 | 160 | 1.158 | 6.112 / 10.176 | 2.420 / 6.048 | 21 |
| B9 | 1 | 160 | 1.317 | 7.280 / 10.304 | 1.591 / 5.447 | 13 |
| B9 | 2 | 160 | 1.182 | 5.968 / 9.760 | 2.753 / 6.520 | 33 |
| B9 | 3 | 160 | 1.227 | 6.432 / 9.856 | 2.565 / 6.465 | 22 |
| B9 | 4 | 160 | 1.304 | 7.712 / 10.848 | 1.267 / 4.428 | 5 |
| B9 | 5 | 160 | 1.305 | 7.232 / 10.528 | 1.828 / 5.408 | 12 |
| B9 | 6 | 160 | 1.118 | 6.608 / 9.344 | 2.391 / 7.140 | 27 |
| B9 | 7 | 160 | 1.250 | 6.576 / 10.048 | 2.689 / 6.349 | 27 |

Attribution limits:

- Barrier slots B0..B9 are chronological modulo-10 positions; the trace has no semantic exchange labels.
- Barrier kernel residency is synchronization/spin time, not RDMA byte-transfer time.
- The latest-arriving device constrains that barrier release, but the trace alone does not identify why its preceding work was late.
- Late-path kernel evidence is overlap on the latest-arriving device during the arrival-skew window; uncovered time may be CPU, transport, DMA, or untraced work.

## FlashInfer/RDMA evidence and launch gaps

- Explicitly named kernels: 12800
- Matched CUDA launches: 12792; unmatched: 8
- Launch→GPU gap median/p95/max: 3.549 / 51601.002 / 54926.091 µs
- The CUDA launch API return-to-GPU-start gap includes stream queueing and backpressure; it is neither pure CPU dispatch overhead nor network transfer duration.

## CUDA API categories

| Category | Calls | Summed thread-ms | Interpretation |
|---|---:|---:|---|
| `synchronization_wait` | 21666 | 147760.864 | CUDA synchronize calls; duration includes waiting and is not CPU overhead alone |
| `kernel_launch` | 577210 | 19009.880 | CUDA kernel/graph launch APIs |
| `memory_management` | 6294 | 4363.538 | CUDA allocation, mapping, and free APIs |
| `other` | 469384 | 3340.118 | Other CUDA APIs |
| `memset` | 89402 | 2450.562 | CUDA memset APIs |
| `memcpy` | 38228 | 544.835 | CUDA memcpy APIs |
| `event_stream` | 81817 | 70.089 | CUDA event and stream bookkeeping APIs |
| `external_memory_possible_rdma_setup` | 0 | 0.000 | CUDA external/shareable-memory APIs; possible registration setup, not measured RDMA transfer time |

## Memcpy operations

| Trace dev | Kind | Calls | MiB (overlapping ops) | GPU-ms |
|---:|---|---:|---:|---:|
| 0 | Device-to-Device | 2156 | 39021.956 | 70.134 |
| 0 | Device-to-Host | 2557 | 938.278 | 18.582 |
| 0 | Host-to-Device | 188 | 39.552 | 1.554 |
| 1 | Device-to-Device | 2132 | 35464.143 | 62.433 |
| 1 | Host-to-Device | 129 | 38.368 | 2.123 |
| 1 | Device-to-Host | 2512 | 1.316 | 1.039 |
| 2 | Device-to-Device | 2132 | 35464.143 | 63.234 |
| 2 | Host-to-Device | 129 | 38.368 | 2.080 |
| 2 | Device-to-Host | 2512 | 1.316 | 1.101 |
| 3 | Device-to-Device | 2132 | 35464.143 | 62.684 |
| 3 | Host-to-Device | 129 | 38.368 | 2.047 |
| 3 | Device-to-Host | 2512 | 1.316 | 1.039 |
| 4 | Device-to-Device | 2111 | 36345.487 | 64.927 |
| 4 | Host-to-Device | 129 | 38.368 | 1.631 |
| 4 | Device-to-Host | 2512 | 1.316 | 1.099 |
| 5 | Device-to-Device | 2111 | 36345.487 | 61.988 |
| 5 | Host-to-Device | 129 | 38.368 | 2.081 |
| 5 | Device-to-Host | 2512 | 1.316 | 0.997 |
| 6 | Device-to-Device | 2111 | 36345.487 | 60.224 |
| 6 | Host-to-Device | 129 | 38.368 | 1.894 |
| 6 | Device-to-Host | 2512 | 1.316 | 1.086 |
| 7 | Device-to-Device | 2111 | 36345.487 | 62.627 |
| 7 | Host-to-Device | 129 | 38.368 | 2.083 |
| 7 | Device-to-Host | 2512 | 1.316 | 1.308 |

## Communication NVTX evidence

These CPU-range totals can overlap because NCCL group/collective ranges may be nested.

| Evidence | Range | Calls | Summed CPU-range ms |
|---|---|---:|---:|
| `nccl` | `ncclAllReduce` | 1016 | 21.347 |
| `nccl` | `ncclAllGather` | 752 | 19.454 |
| `nccl` | `ncclBroadcast` | 56 | 1.444 |

## Top kernels

| # | Category | Calls | Total GPU-ms | Avg µs | % kernel GPU-time | Kernel |
|---:|---|---:|---:|---:|---:|---|
| 1 | `gemm_unknown` | 5792 | 65272.595 | 11269.440 | 37.46% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params)` |
| 2 | `vsa_attention` | 1600 | 21287.522 | 13304.701 | 12.22% | `kernel_cutlass_kernel_flashinfercute_dslsparsesm120_blk64flash_fwd_sm120_sageBlockSparseAttnForwardSageSm120Blk64_object_at__tensor0000o12811101213_tensor0000o12811101213_tensor00…` |
| 3 | `gemm_unknown` | 3200 | 14021.346 | 4381.671 | 8.05% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8>(T1::Params)` |
| 4 | `gemm_unknown` | 21168 | 12562.665 | 593.474 | 7.21% | `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` |
| 5 | `gemm_unknown` | 42336 | 10457.768 | 247.018 | 6.00% | `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn_align8>(T1::Params)` |
| 6 | `flashinfer_rdma_barrier` | 12800 | 10215.243 | 798.066 | 5.86% | `flashinfer::comm::ulysses_pcie::UlyssesPcieBarrier(unsigned long *, flashinfer::comm::ulysses_pcie::PeerSignalPointers, int, int, unsigned long *)` |
| 7 | `nccl` | 168 | 5781.034 | 34410.919 | 3.32% | `ncclDevKernel_AllReduce_Sum_f32_TREE_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 8 | `gemm_unknown` | 21168 | 5526.548 | 261.080 | 3.17% | `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_tn_align8>(T1::Params)` |
| 9 | `nccl` | 752 | 5287.514 | 7031.268 | 3.03% | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 10 | `attention_other` | 21168 | 3691.070 | 174.370 | 2.12% | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)64, (int)128, (int)128, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)64, (int)128, (int)12…` |
| 11 | `copy_kernel` | 8832 | 2935.492 | 332.370 | 1.68% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 12 | `vsa_layout` | 6400 | 1997.597 | 312.125 | 1.15% | `_h3_vsa_tile_pack_kernel` |
| 13 | `unknown` | 1600 | 1362.805 | 851.753 | 0.78% | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vllm::packed_silu_kernel<__nv_bfloat162>, (bool)1, (bool)1, (bool)0, (bool)1>(T1 *…` |
| 14 | `h3_pointwise` | 3200 | 937.104 | 292.845 | 0.54% | `_rms_norm_rope_kernel` |
| 15 | `unknown` | 4800 | 790.484 | 164.684 | 0.45% | `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<float, at::native::sum_functor<c10::BFloat16, float, float>::operat…` |
| 16 | `sage_quantization` | 1600 | 732.981 | 458.113 | 0.42% | `_quantize_sage_kv_kernel` |
| 17 | `vsa_layout` | 1600 | 695.797 | 434.873 | 0.40% | `_h3_vsa_o_bundle_local_gate_kernel` |
| 18 | `h3_pointwise` | 1600 | 655.630 | 409.768 | 0.38% | `_indexed_gate_rms_norm_scale_shift_kernel` |
| 19 | `unknown` | 21168 | 652.969 | 30.847 | 0.37% | `void vllm::act_and_mul_kernel<c10::Half, __half2, &vllm::silu_kernel<c10::Half>, &vllm::packed_silu_kernel<__half2>, (bool)1, (bool)1, (bool)0, (bool)1>(T1 *, const T1 *, int, flo…` |
| 20 | `copy_kernel` | 24069 | 617.506 | 25.656 | 0.35% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 21 | `unknown` | 42336 | 607.229 | 14.343 | 0.35% | `_scaled_residual_exact_kernel` |
| 22 | `gemm_unknown` | 1600 | 591.266 | 369.541 | 0.34% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)` |
| 23 | `sage_quantization` | 1600 | 564.153 | 352.596 | 0.32% | `_sage_kv_stats_partial_kernel` |
| 24 | `unknown` | 42336 | 499.294 | 11.794 | 0.29% | `void at::native::<unnamed>::vectorized_layer_norm_kernel<float, float, (bool)1>(int, T2, const T1 *, const T1 *, const T1 *, T2 *, T2 *, T1 *)` |
| 25 | `gemm_unknown` | 592 | 487.882 | 824.124 | 0.28% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(T1::Params)` |
| 26 | `vsa_layout` | 1600 | 469.740 | 293.587 | 0.27% | `_h3_vsa_o_bundle_kernel` |
| 27 | `nccl` | 800 | 462.618 | 578.273 | 0.27% | `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 28 | `h3_pointwise` | 1600 | 450.689 | 281.681 | 0.26% | `_indexed_gate_kernel` |
| 29 | `gemm_unknown` | 1600 | 396.965 | 248.103 | 0.23% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(T1::Params)` |
| 30 | `unknown` | 1600 | 395.142 | 246.964 | 0.23% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, float, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6…` |

## Top OS runtime calls

OSRT time is summed across threads and is never assigned to RDMA without explicit naming evidence.

| Call | Count | Summed thread-ms |
|---|---:|---:|
| `pthread_cond_timedwait` | 31236 | 3204894.781 |
| `poll` | 6443 | 2314481.907 |
| `epoll_wait` | 170760 | 1576778.438 |
| `pthread_cond_wait` | 44249 | 406296.881 |
| `sem_clockwait` | 296 | 352597.196 |
| `epoll_pwait` | 24786 | 52163.098 |
| `sem_wait` | 24 | 31944.857 |
| `ioctl` | 16551 | 11841.043 |
| `fopen` | 8 | 169.012 |
| `open64` | 4096 | 115.635 |
| `pthread_mutex_lock` | 6850 | 101.684 |
| `pthread_cond_broadcast` | 8762 | 26.425 |
| `waitpid` | 140 | 23.524 |
| `recv` | 964 | 14.305 |
| `pthread_cond_signal` | 7761 | 11.757 |
| `send` | 1047 | 9.889 |
| `fgets` | 1794 | 8.028 |
| `pthread_rwlock_wrlock` | 54 | 3.277 |
| `write` | 340 | 3.203 |
| `recvmsg` | 72 | 2.908 |

## Warnings and attribution limits

- No usable NVTX range was selected; results cover the complete CUDA GPU trace and may include startup.
- NCCL logger rank 6 disagrees with communicator rank 2 for PID 560055; ignored that line.
- 3.88% of summed kernel GPU-time is unknown; it was not force-fit into a named H3 stage.
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
