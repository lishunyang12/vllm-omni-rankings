# MiniMax-H3 Nsight analysis

- Source: `/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-vsa-sage-pr4691-20260910/runs/bf16-nsys/bf16-warmed.nsys-rep`
- SQLite: `/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-vsa-sage-pr4691-20260910/runs/bf16-nsys/bf16-warmed.sqlite`
- Scope: `nvtx_range` / `minimax_h3.denoise.step_03` @ `vllm_omni.minimax_h3`
- Window wall time: **5008.456 ms**
- GPU active-span median: **5008.322 ms**
- Slowest trace device by active span: **device 1 / rank 1 / physical GPU 4 / 5008.456 ms**
- Largest non-barrier kernel sum: **trace device 7 / rank 7 / physical GPU 7 / 4337.558 ms**
- Device identity: **kernel PID -> server NCCL rank/cudaDev/nvmlDev/busId**

> GPU-time below is summed across devices and streams; it is not additive wall time.

## Cross-rank decode NVTX timeline

No explicit nested `minimax_h3.decode.*` stage range was found in scope.

## Per-GPU timeline

| Trace dev | Rank | Physical GPU | PCI bus | PID(s) | Active span ms | Busy union ms | Kernel-sum ms | Non-barrier / barrier kernel ms | Memcpy ms / MiB | Window idle ms | Largest idle gap ms |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0000:06:00.0 | 515661 | 5008.244 | 4352.505 | 4343.401 | 3982.067 / 361.334 | 8.809 / 4887.74 | 655.952 | 3.800 |
| 1 | 1 | 4 | 0000:86:00.0 | 515669 | 5008.456 | 4353.234 | 4344.439 | 3988.466 / 355.973 | 8.522 / 4876.08 | 655.222 | 3.324 |
| 2 | 2 | 1 | 0000:09:00.0 | 515670 | 5008.134 | 4352.423 | 4343.496 | 3913.630 / 429.866 | 8.632 / 4841.11 | 656.033 | 3.457 |
| 3 | 3 | 5 | 0000:89:00.0 | 515671 | 5008.272 | 4353.313 | 4344.589 | 3858.734 / 485.855 | 8.450 / 4841.11 | 655.143 | 3.333 |
| 4 | 4 | 2 | 0000:76:00.0 | 515672 | 5008.095 | 4344.437 | 4335.449 | 3912.269 / 423.180 | 8.688 / 4841.11 | 664.019 | 3.333 |
| 5 | 5 | 6 | 0000:f6:00.0 | 515673 | 5008.372 | 4353.686 | 4345.415 | 3929.539 / 415.876 | 8.000 / 4887.74 | 654.770 | 3.347 |
| 6 | 6 | 3 | 0000:79:00.0 | 515674 | 5008.450 | 4350.927 | 4343.018 | 4081.737 / 261.281 | 7.614 / 4887.74 | 657.530 | 3.447 |
| 7 | 7 | 7 | 0000:f9:00.0 | 515675 | 5008.381 | 4352.439 | 4343.931 | 4337.558 / 6.372 | 8.156 / 4887.74 | 656.018 | 3.354 |

## Per-device wall decomposition (no double-counting)

Per-device disjoint interval sweep. Different simultaneously active phases are charged once to concurrent_mixed; no values are summed across devices. Reference device (largest traced busy union): 5.

| Trace dev | Linear GEMM ms | VSA ms | RDMA barrier ms | RDMA other ms | Pointwise ms | Other single-phase ms | Mixed-concurrent ms | Idle ms | Account error ns |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2545.901 | 1105.784 | 361.334 | 0.000 | 75.304 | 264.180 | 0.000 | 655.952 | 0 |
| 1 | 2551.597 | 1110.275 | 355.973 | 0.000 | 75.291 | 260.098 | 0.000 | 655.222 | 0 |
| 2 | 2490.728 | 1091.160 | 429.866 | 0.000 | 75.086 | 265.583 | 0.000 | 656.033 | 0 |
| 3 | 2446.967 | 1083.437 | 485.855 | 0.000 | 75.078 | 261.975 | 0.000 | 655.143 | 0 |
| 4 | 2490.807 | 1089.234 | 423.180 | 0.000 | 75.327 | 265.889 | 0.000 | 664.019 | 0 |
| 5 | 2510.676 | 1091.691 | 415.876 | 0.000 | 75.292 | 260.151 | 0.000 | 654.770 | 0 |
| 6 | 2614.293 | 1135.714 | 261.281 | 0.000 | 75.255 | 264.383 | 0.000 | 657.530 | 0 |
| 7 | 2669.588 | 1319.705 | 6.372 | 0.000 | 75.729 | 281.044 | 0.000 | 656.018 | 0 |

## Kernel categories

| Category | Calls | Total GPU-ms | % kernel GPU-time | Median GPU-ms | Slowest trace dev / ms | Evidence rule |
|---|---:|---:|---:|---:|---:|---|
| `gemm_unknown` | 2913 | 20320.557 | 58.49% | 2528.288 | 7 / 2669.588 | Recognizable GEMM, but its H3 layer role is not encoded in the kernel name |
| `vsa_attention` | 400 | 9027.002 | 25.98% | 1098.738 | 7 / 1319.705 | Explicit block-sparse/VSA attention kernel, including FlashInfer CuTeDSL SM120 BlockSparseAttnForward |
| `flashinfer_rdma_barrier` | 3200 | 2739.738 | 7.89% | 388.605 | 3 / 485.855 | Explicit FlashInfer Ulysses PCIe/RDMA barrier; duration is synchronization residency/wait |
| `unknown` | 17273 | 1043.195 | 3.00% | 130.028 | 7 / 143.883 | No conservative category matched |
| `vsa_layout` | 2400 | 790.724 | 2.28% | 98.841 | 7 / 101.093 | Explicit H3 VSA compact/tile layout kernel |
| `h3_pointwise` | 2008 | 602.253 | 1.73% | 75.277 | 7 / 75.714 | Explicitly named H3 fused modulation, gated-residual, Q/K norm+RoPE, or SwiGLU kernel |
| `copy_kernel` | 3298 | 196.038 | 0.56% | 24.394 | 7 / 26.025 | Device-side copy/cat/transpose-like kernel (not a DMA memcpy record) |
| `nccl` | 8 | 23.930 | 0.07% | 3.197 | 3 / 4.199 | NCCL device kernel |
| `cudnn_attention` | 16 | 0.123 | 0.00% | 0.015 | 7 / 0.018 | cuDNN kernel with explicit SDPA/attention/FMHA evidence |
| `triton_reduction` | 32 | 0.110 | 0.00% | 0.014 | 7 / 0.015 | Triton reduction kernel |
| `triton_other` | 32 | 0.066 | 0.00% | 0.008 | 7 / 0.010 | Other Triton kernel |
| `attention_other` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Non-cuDNN kernel with explicit attention/FMHA/flash-attention evidence |
| `cudnn_other` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | cuDNN kernel without explicit attention evidence |
| `flashinfer_other` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | FlashInfer kernel without explicit PCIe/RDMA evidence |
| `flashinfer_rdma_other` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Explicit FlashInfer Ulysses PCIe/RDMA kernel whose role is not distinguishable by name |
| `flashinfer_rdma_transfer` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Explicit FlashInfer Ulysses PCIe/RDMA data-movement kernel |
| `gemm_fc1` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | GEMM with an explicit FC1/gate/up-projection token |
| `gemm_fc2` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | GEMM with an explicit FC2/down-projection token |
| `gemm_out` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | GEMM with an explicit attention output-projection token |
| `gemm_qkv` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | GEMM with an explicit Q/K/V/QKV-projection token |
| `sage_quantization` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Explicit Sage Q/K/V quantization and statistics kernels |
| `triton_activation` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Triton kernel with an explicit activation token |
| `triton_pointwise` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Triton pointwise kernel without a more specific activation token |
| `vsa_indexing` | 0 | 0.000 | 0.00% | 0.000 | 0 / 0.000 | Explicit VSA block-map-to-index compaction kernel |

## FlashInfer Ulysses barrier/rank-skew analysis

- Observed 3200 calls reconstructed as 400 chronological groups: 400 full-device and 0 partial.
- Layout: 10 barrier slots/layer (B0..B9); the group count is consistent with 40 layer cycles.
- Layout basis: explicit VSA evidence: Q/K/V/gate scatters plus O gather, each with opening and closing barriers.
- Full-group arrival-skew sum: 485.127 ms (9.69% of the selected wall window).
- Dominant latest arrival: trace device 7 in 151/400 groups (37.75%).
- Latest-device work overlapping the skew windows: vsa_attention=203.181 GPU-ms, gemm_unknown=200.334 GPU-ms, vsa_layout=29.740 GPU-ms, h3_pointwise=29.488 GPU-ms, unknown=14.041 GPU-ms, copy_kernel=1.923 GPU-ms; covered union 478.707 ms, uncovered 6.420 ms.
- Arrival-skew sums cover sequential full-device barriers and are already inside step wall time; they indicate load-imbalance exposure, not additional time or guaranteed removable time.

### Per-device barrier residency and arrival lag

| Trace dev | Calls | Residency sum ms | Residency median / p95 / max µs | Arrival-lag median / p95 µs | Latest arrival groups / % |
|---:|---:|---:|---:|---:|---:|
| 0 | 400 | 361.334 | 12.720 / 4680.901 / 4960.452 | 5.123 / 2059.321 | 25 / 6.25% |
| 1 | 400 | 355.973 | 16.544 / 4692.200 / 4972.616 | 3.729 / 2194.597 | 18 / 4.50% |
| 2 | 400 | 429.866 | 14.400 / 4981.482 / 5043.529 | 4.498 / 916.637 | 24 / 6.00% |
| 3 | 400 | 485.855 | 18.224 / 5270.085 / 5505.282 | 1.404 / 7.524 | 28 / 7.00% |
| 4 | 400 | 423.180 | 6.976 / 4956.557 / 5036.299 | 14.860 / 1043.516 | 118 / 29.50% |
| 5 | 400 | 415.876 | 16.416 / 5125.541 / 5491.780 | 2.908 / 1443.605 | 8 / 2.00% |
| 6 | 400 | 261.281 | 13.232 / 4084.672 / 4479.359 | 4.291 / 3498.813 | 28 / 7.00% |
| 7 | 400 | 6.372 | 5.952 / 48.960 / 700.897 | 4.666 / 5265.184 | 151 / 37.75% |

### Per-layer chronological barrier slots

| Slot | Groups (full) | Arrival-skew sum ms | Arrival skew median / p95 / max µs | Release skew median / p95 / max µs | Dominant latest GPU / count | Late-device kernel evidence (GPU-ms) |
|---:|---:|---:|---:|---:|---:|---|
| B0 | 40 (40) | 95.114 | 1279.600 / 5278.397 / 5447.221 | 1.256 / 1.456 / 1.745 | 7 / 25 | vsa_attention=40.630, gemm_unknown=38.777, vsa_layout=5.949, h3_pointwise=5.889, unknown=2.811, copy_kernel=0.352 |
| B1 | 40 (40) | 0.203 | 4.990 / 6.968 / 8.777 | 1.034 / 1.549 / 1.658 | 4 / 10 | uncovered=0.203 |
| B2 | 40 (40) | 96.445 | 1853.986 / 5292.058 / 5356.037 | 1.200 / 1.471 / 1.498 | 7 / 24 | vsa_attention=40.637, gemm_unknown=40.148, vsa_layout=5.947, h3_pointwise=5.892, unknown=2.803, copy_kernel=0.363 |
| B3 | 40 (40) | 0.228 | 5.348 / 12.332 / 14.384 | 1.220 / 1.579 / 1.738 | 4 / 10 | uncovered=0.228 |
| B4 | 40 (40) | 96.818 | 1878.959 / 5271.057 / 5424.694 | 1.160 / 1.396 / 1.724 | 7 / 24 | vsa_attention=40.554, gemm_unknown=40.214, vsa_layout=5.947, h3_pointwise=5.907, unknown=2.812, copy_kernel=0.353 |
| B5 | 40 (40) | 1.623 | 5.408 / 342.760 / 701.466 | 1.260 / 1.701 / 1.740 | 4 / 8 | uncovered=1.623 |
| B6 | 40 (40) | 96.996 | 1869.192 / 5286.896 / 5500.423 | 1.204 / 1.469 / 1.556 | 7 / 24 | vsa_attention=40.734, gemm_unknown=40.356, vsa_layout=5.947, h3_pointwise=5.906, unknown=2.802, copy_kernel=0.496 |
| B7 | 40 (40) | 0.208 | 4.826 / 8.177 / 11.181 | 1.239 / 1.621 / 1.768 | 3 / 9 | uncovered=0.208 |
| B8 | 40 (40) | 97.148 | 1835.336 / 5311.632 / 5399.285 | 1.175 / 1.465 / 1.488 | 7 / 24 | gemm_unknown=40.838, vsa_attention=40.626, vsa_layout=5.950, h3_pointwise=5.895, unknown=2.812, copy_kernel=0.360 |
| B9 | 40 (40) | 0.343 | 5.269 / 18.644 / 96.838 | 1.275 / 1.720 / 1.771 | 1 / 9 | uncovered=0.343 |

### Per-device × barrier slot

| Slot | Trace dev | Calls | Residency sum ms | Residency median / p95 µs | Arrival-lag median / p95 µs | Latest groups |
|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 40 | 70.781 | 839.568 / 4678.340 | 284.039 / 2232.506 | 0 |
| B0 | 1 | 40 | 70.093 | 967.905 / 4703.081 | 134.184 / 2332.677 | 0 |
| B0 | 2 | 40 | 84.235 | 1118.481 / 4980.618 | 132.207 / 984.845 | 0 |
| B0 | 3 | 40 | 95.222 | 1282.528 / 5284.290 | 0.000 / 7.878 | 0 |
| B0 | 4 | 40 | 82.856 | 1015.091 / 4963.340 | 189.851 / 1144.170 | 14 |
| B0 | 5 | 40 | 81.388 | 992.368 / 5142.725 | 38.663 / 1592.010 | 0 |
| B0 | 6 | 40 | 51.784 | 245.280 / 4148.064 | 539.298 / 3660.237 | 1 |
| B0 | 7 | 40 | 0.617 | 4.480 / 47.904 | 1234.427 / 5278.397 | 25 |
| B1 | 0 | 40 | 0.242 | 5.984 / 9.344 | 2.092 / 5.061 | 5 |
| B1 | 1 | 40 | 0.266 | 6.832 / 9.152 | 1.008 / 5.164 | 3 |
| B1 | 2 | 40 | 0.249 | 6.384 / 9.280 | 2.117 / 5.352 | 7 |
| B1 | 3 | 40 | 0.248 | 6.064 / 9.312 | 2.417 / 5.110 | 5 |
| B1 | 4 | 40 | 0.229 | 5.872 / 9.344 | 1.857 / 6.512 | 10 |
| B1 | 5 | 40 | 0.261 | 6.800 / 10.048 | 1.232 / 5.386 | 4 |
| B1 | 6 | 40 | 0.259 | 6.432 / 9.024 | 2.246 / 4.627 | 3 |
| B1 | 7 | 40 | 0.262 | 6.864 / 9.696 | 2.274 / 5.320 | 3 |
| B2 | 0 | 40 | 71.360 | 1035.457 / 4704.453 | 256.414 / 2302.154 | 0 |
| B2 | 1 | 40 | 69.941 | 924.673 / 4743.048 | 251.327 / 2365.057 | 0 |
| B2 | 2 | 40 | 85.305 | 1525.347 / 4997.514 | 108.535 / 1000.850 | 0 |
| B2 | 3 | 40 | 96.559 | 1857.441 / 5297.476 | 0.000 / 6.623 | 0 |
| B2 | 4 | 40 | 84.152 | 1520.228 / 4975.532 | 148.531 / 1138.572 | 15 |
| B2 | 5 | 40 | 82.548 | 1406.610 / 5196.486 | 30.607 / 1625.860 | 0 |
| B2 | 6 | 40 | 51.387 | 366.800 / 4129.633 | 532.954 / 3701.277 | 1 |
| B2 | 7 | 40 | 0.585 | 4.592 / 48.960 | 1843.426 / 5292.058 | 24 |
| B3 | 0 | 40 | 0.282 | 7.280 / 14.048 | 1.853 / 5.408 | 4 |
| B3 | 1 | 40 | 0.282 | 6.528 / 14.816 | 1.740 / 5.160 | 4 |
| B3 | 2 | 40 | 0.290 | 6.176 / 15.264 | 1.910 / 5.097 | 1 |
| B3 | 3 | 40 | 0.260 | 6.512 / 14.496 | 2.337 / 6.491 | 8 |
| B3 | 4 | 40 | 0.206 | 4.768 / 8.320 | 3.199 / 12.332 | 10 |
| B3 | 5 | 40 | 0.305 | 7.408 / 11.296 | 1.028 / 4.568 | 0 |
| B3 | 6 | 40 | 0.286 | 6.880 / 14.912 | 1.992 / 4.743 | 5 |
| B3 | 7 | 40 | 0.272 | 6.240 / 14.880 | 2.704 / 6.050 | 8 |
| B4 | 0 | 40 | 71.865 | 1083.858 / 4726.117 | 230.532 / 2252.729 | 0 |
| B4 | 1 | 40 | 70.592 | 991.858 / 4701.450 | 248.709 / 2342.844 | 0 |
| B4 | 2 | 40 | 85.737 | 1578.755 / 4994.731 | 101.345 / 981.858 | 0 |
| B4 | 3 | 40 | 96.925 | 1881.153 / 5276.290 | 0.000 / 6.729 | 0 |
| B4 | 4 | 40 | 84.372 | 1537.077 / 4978.893 | 212.879 / 1089.749 | 15 |
| B4 | 5 | 40 | 82.735 | 1458.178 / 5147.621 | 41.280 / 1623.486 | 0 |
| B4 | 6 | 40 | 51.435 | 416.752 / 4097.793 | 558.716 / 3618.827 | 1 |
| B4 | 7 | 40 | 0.950 | 4.528 / 172.128 | 1797.169 / 5271.057 | 24 |
| B5 | 0 | 40 | 0.965 | 6.096 / 14.080 | 2.238 / 5.351 | 5 |
| B5 | 1 | 40 | 1.691 | 7.440 / 345.024 | 1.383 / 4.349 | 1 |
| B5 | 2 | 40 | 1.302 | 6.672 / 13.312 | 2.392 / 6.132 | 7 |
| B5 | 3 | 40 | 1.660 | 6.944 / 346.432 | 2.713 / 5.292 | 4 |
| B5 | 4 | 40 | 1.610 | 5.632 / 345.793 | 3.218 / 8.081 | 8 |
| B5 | 5 | 40 | 1.699 | 7.776 / 346.016 | 1.401 / 3.590 | 1 |
| B5 | 6 | 40 | 1.338 | 7.488 / 17.248 | 1.647 / 6.739 | 7 |
| B5 | 7 | 40 | 1.682 | 7.408 / 342.720 | 1.970 / 5.122 | 7 |
| B6 | 0 | 40 | 72.574 | 1073.809 / 4741.412 | 249.068 / 2075.331 | 0 |
| B6 | 1 | 40 | 70.927 | 975.266 / 4723.945 | 275.785 / 2301.385 | 0 |
| B6 | 2 | 40 | 86.077 | 1579.875 / 5009.353 | 128.772 / 927.472 | 1 |
| B6 | 3 | 40 | 97.110 | 1873.505 / 5292.420 | 0.000 / 9.092 | 0 |
| B6 | 4 | 40 | 84.750 | 1532.980 / 4987.212 | 203.599 / 1080.039 | 15 |
| B6 | 5 | 40 | 83.235 | 1427.666 / 5143.173 | 11.050 / 1628.152 | 0 |
| B6 | 6 | 40 | 52.099 | 402.608 / 4110.049 | 524.447 / 3561.593 | 0 |
| B6 | 7 | 40 | 0.803 | 4.352 / 71.040 | 1812.183 / 5286.896 | 24 |
| B7 | 0 | 40 | 0.252 | 5.936 / 9.408 | 1.508 / 7.146 | 4 |
| B7 | 1 | 40 | 0.268 | 6.464 / 10.976 | 1.728 / 3.795 | 1 |
| B7 | 2 | 40 | 0.256 | 6.128 / 9.184 | 2.117 / 6.016 | 5 |
| B7 | 3 | 40 | 0.250 | 6.464 / 10.368 | 1.954 / 6.570 | 9 |
| B7 | 4 | 40 | 0.245 | 6.784 / 8.896 | 2.199 / 7.404 | 9 |
| B7 | 5 | 40 | 0.253 | 6.032 / 9.568 | 2.524 / 4.594 | 3 |
| B7 | 6 | 40 | 0.255 | 6.192 / 10.240 | 2.279 / 5.975 | 5 |
| B7 | 7 | 40 | 0.250 | 6.048 / 8.960 | 3.010 / 6.492 | 4 |
| B8 | 0 | 40 | 72.657 | 1070.577 / 4706.597 | 253.808 / 2125.098 | 0 |
| B8 | 1 | 40 | 71.543 | 935.538 / 4737.673 | 250.456 / 2328.006 | 0 |
| B8 | 2 | 40 | 86.048 | 1527.971 / 5014.091 | 107.380 / 938.159 | 0 |
| B8 | 3 | 40 | 97.247 | 1837.234 / 5316.545 | 0.000 / 9.291 | 0 |
| B8 | 4 | 40 | 84.421 | 1556.244 / 4981.133 | 141.344 / 1093.449 | 15 |
| B8 | 5 | 40 | 83.090 | 1406.290 / 5167.302 | 34.571 / 1633.449 | 0 |
| B8 | 6 | 40 | 52.181 | 363.264 / 4098.593 | 571.998 / 3544.157 | 1 |
| B8 | 7 | 40 | 0.594 | 4.528 / 44.448 | 1817.998 / 5311.632 | 24 |
| B9 | 0 | 40 | 0.357 | 6.416 / 13.344 | 2.075 / 6.205 | 7 |
| B9 | 1 | 40 | 0.372 | 6.016 / 18.048 | 3.109 / 5.747 | 9 |
| B9 | 2 | 40 | 0.368 | 6.160 / 14.912 | 2.405 / 7.495 | 3 |
| B9 | 3 | 40 | 0.374 | 6.976 / 11.136 | 2.069 / 6.073 | 2 |
| B9 | 4 | 40 | 0.340 | 7.072 / 9.280 | 2.644 / 11.588 | 7 |
| B9 | 5 | 40 | 0.362 | 6.656 / 11.776 | 2.071 / 5.444 | 0 |
| B9 | 6 | 40 | 0.257 | 6.576 / 10.144 | 2.663 / 18.644 | 4 |
| B9 | 7 | 40 | 0.359 | 6.624 / 12.032 | 2.569 / 6.388 | 8 |

Attribution limits:

- Barrier slots B0..B9 are chronological modulo-10 positions; the trace has no semantic exchange labels.
- Barrier kernel residency is synchronization/spin time, not RDMA byte-transfer time.
- The latest-arriving device constrains that barrier release, but the trace alone does not identify why its preceding work was late.
- Late-path kernel evidence is overlap on the latest-arriving device during the arrival-skew window; uncovered time may be CPU, transport, DMA, or untraced work.

## FlashInfer/RDMA evidence and launch gaps

- Explicitly named kernels: 3200
- Matched CUDA launches: 3199; unmatched: 1
- Launch→GPU gap median/p95/max: 3.494 / 52389.463 / 54949.046 µs
- The CUDA launch API return-to-GPU-start gap includes stream queueing and backpressure; it is neither pure CPU dispatch overhead nor network transfer duration.

## CUDA API categories

| Category | Calls | Summed thread-ms | Interpretation |
|---|---:|---:|---|
| `synchronization_wait` | 3560 | 34125.161 | CUDA synchronize calls; duration includes waiting and is not CPU overhead alone |
| `kernel_launch` | 31578 | 103.484 | CUDA kernel/graph launch APIs |
| `memcpy` | 6021 | 30.929 | CUDA memcpy APIs |
| `memory_management` | 24 | 15.228 | CUDA allocation, mapping, and free APIs |
| `event_stream` | 10622 | 8.134 | CUDA event and stream bookkeeping APIs |
| `memset` | 941 | 7.255 | CUDA memset APIs |
| `other` | 27121 | 3.755 | Other CUDA APIs |
| `external_memory_possible_rdma_setup` | 0 | 0.000 | CUDA external/shareable-memory APIs; possible registration setup, not measured RDMA transfer time |

## Memcpy operations

| Trace dev | Kind | Calls | MiB (overlapping ops) | GPU-ms |
|---:|---|---:|---:|---:|
| 0 | Device-to-Device | 310 | 4887.713 | 8.633 |
| 0 | Device-to-Host | 436 | 0.025 | 0.173 |
| 0 | Host-to-Device | 10 | 0.000 | 0.003 |
| 1 | Device-to-Device | 309 | 4876.056 | 8.356 |
| 1 | Device-to-Host | 433 | 0.025 | 0.163 |
| 1 | Host-to-Device | 8 | 0.000 | 0.002 |
| 2 | Device-to-Device | 308 | 4841.088 | 8.459 |
| 2 | Device-to-Host | 434 | 0.025 | 0.170 |
| 2 | Host-to-Device | 9 | 0.000 | 0.003 |
| 3 | Device-to-Device | 308 | 4841.088 | 8.287 |
| 3 | Device-to-Host | 436 | 0.025 | 0.160 |
| 3 | Host-to-Device | 10 | 0.000 | 0.003 |
| 4 | Device-to-Device | 308 | 4841.088 | 8.512 |
| 4 | Device-to-Host | 434 | 0.025 | 0.173 |
| 4 | Host-to-Device | 9 | 0.000 | 0.003 |
| 5 | Device-to-Device | 310 | 4887.713 | 7.839 |
| 5 | Device-to-Host | 436 | 0.025 | 0.158 |
| 5 | Host-to-Device | 10 | 0.000 | 0.003 |
| 6 | Device-to-Device | 310 | 4887.713 | 7.443 |
| 6 | Device-to-Host | 432 | 0.024 | 0.169 |
| 6 | Host-to-Device | 6 | 0.000 | 0.002 |
| 7 | Device-to-Device | 310 | 4887.713 | 7.941 |
| 7 | Device-to-Host | 435 | 0.025 | 0.212 |
| 7 | Host-to-Device | 10 | 0.000 | 0.003 |

## Communication NVTX evidence

These CPU-range totals can overlap because NCCL group/collective ranges may be nested.

| Evidence | Range | Calls | Summed CPU-range ms |
|---|---|---:|---:|
| `nccl` | `ncclAllGather` | 8 | 0.622 |

## Top kernels

| # | Category | Calls | Total GPU-ms | Avg µs | % kernel GPU-time | Kernel |
|---:|---|---:|---:|---:|---:|---|
| 1 | `gemm_unknown` | 1248 | 16517.739 | 13235.368 | 47.54% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params)` |
| 2 | `vsa_attention` | 400 | 9027.002 | 22567.504 | 25.98% | `kernel_cutlass_kernel_flashinfercute_dslsparsesm120_blk64flash_fwd_sm120BlockSparseAttnForwardSm120Blk64_object_at__tensor0000o12811101213_tensor0000o12811101213_tensor0000o128101…` |
| 3 | `gemm_unknown` | 800 | 3536.236 | 4420.295 | 10.18% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8>(T1::Params)` |
| 4 | `flashinfer_rdma_barrier` | 3200 | 2739.738 | 856.168 | 7.89% | `flashinfer::comm::ulysses_pcie::UlyssesPcieBarrier(unsigned long *, flashinfer::comm::ulysses_pcie::PeerSignalPointers, int, int, unsigned long *)` |
| 5 | `vsa_layout` | 1600 | 499.333 | 312.083 | 1.44% | `_h3_vsa_tile_pack_kernel` |
| 6 | `unknown` | 400 | 340.742 | 851.855 | 0.98% | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vllm::packed_silu_kernel<__nv_bfloat162>, (bool)1, (bool)1, (bool)0, (bool)1>(T1 *…` |
| 7 | `h3_pointwise` | 800 | 234.668 | 293.335 | 0.68% | `_rms_norm_rope_kernel` |
| 8 | `unknown` | 1200 | 196.183 | 163.486 | 0.56% | `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<float, at::native::sum_functor<c10::BFloat16, float, float>::operat…` |
| 9 | `vsa_layout` | 400 | 173.892 | 434.730 | 0.50% | `_h3_vsa_o_bundle_local_gate_kernel` |
| 10 | `h3_pointwise` | 400 | 164.304 | 410.759 | 0.47% | `_indexed_gate_rms_norm_scale_shift_kernel` |
| 11 | `gemm_unknown` | 400 | 148.239 | 370.597 | 0.43% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)` |
| 12 | `copy_kernel` | 408 | 132.479 | 324.703 | 0.38% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 13 | `vsa_layout` | 400 | 117.500 | 293.749 | 0.34% | `_h3_vsa_o_bundle_kernel` |
| 14 | `h3_pointwise` | 400 | 112.523 | 281.308 | 0.32% | `_indexed_gate_kernel` |
| 15 | `gemm_unknown` | 400 | 99.757 | 249.392 | 0.29% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(T1::Params)` |
| 16 | `unknown` | 400 | 98.814 | 247.036 | 0.28% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, float, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6…` |
| 17 | `unknown` | 400 | 97.680 | 244.201 | 0.28% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, int, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6, …` |
| 18 | `h3_pointwise` | 400 | 88.907 | 222.266 | 0.26% | `_rms_norm_indexed_scale_shift_kernel` |
| 19 | `unknown` | 1600 | 82.075 | 51.297 | 0.24% | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int,…` |
| 20 | `copy_kernel` | 400 | 48.406 | 121.015 | 0.14% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 21 | `unknown` | 400 | 45.484 | 113.710 | 0.13% | `void at::native::mbtopk::gatherTopK<float, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, unsigned int, T2, at::cuda::detail::TensorInfo<T1, T2>, …` |
| 22 | `unknown` | 1600 | 38.089 | 23.805 | 0.11% | `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(T1 *, short *, unsigned int *, unsigned int *, unsigned int, int, bool, unsigned int *, T2 *, unsigned …` |
| 23 | `unknown` | 400 | 36.383 | 90.957 | 0.10% | `void <unnamed>::softmax_warp_forward<float, float, float, (int)11, (bool)0, (bool)0, (int)32>(T2 *, const T1 *, int, int, int, const bool *, int, bool)` |
| 24 | `unknown` | 408 | 33.259 | 81.517 | 0.10% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned…` |
| 25 | `nccl` | 8 | 23.930 | 2991.284 | 0.07% | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 26 | `unknown` | 1600 | 23.427 | 14.642 | 0.07% | `at::native::mbtopk::computeDigitCumSum(short *, unsigned int *, unsigned int)` |
| 27 | `unknown` | 800 | 13.748 | 17.185 | 0.04% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| 28 | `unknown` | 1200 | 10.071 | 8.393 | 0.03% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float…` |
| 29 | `gemm_unknown` | 8 | 5.197 | 649.645 | 0.01% | `void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(int, int, int, BatchedTensor, int, BatchedTensor, int, BatchedTensor, int,…` |
| 30 | `unknown` | 10 | 4.785 | 478.474 | 0.01% | `void at::native::indexFuncLargeIndex<c10::BFloat16, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1, at::native::<unnamed>::ReduceAdd>(at::cuda::detail::TensorInfo<T1, T3>, a…` |

## Top OS runtime calls

OSRT time is summed across threads and is never assigned to RDMA without explicit naming evidence.

| Call | Count | Summed thread-ms |
|---|---:|---:|
| `pthread_cond_timedwait` | 5296 | 610856.700 |
| `poll` | 1318 | 440720.410 |
| `epoll_wait` | 32640 | 300315.415 |
| `sem_clockwait` | 77 | 85139.549 |
| `pthread_cond_wait` | 14 | 50068.717 |
| `epoll_pwait` | 4774 | 9981.750 |
| `sem_wait` | 1 | 5008.456 |
| `ioctl` | 135 | 143.025 |
| `recv` | 198 | 3.064 |
| `waitpid` | 50 | 2.680 |
| `send` | 202 | 2.090 |
| `write` | 47 | 0.593 |
| `read` | 41 | 0.130 |
| `pthread_mutex_lock` | 30 | 0.097 |
| `pthread_cond_signal` | 26 | 0.073 |
| `pthread_cond_broadcast` | 1 | 0.004 |

## Warnings and attribution limits

- NCCL logger rank 7 disagrees with communicator rank 2 for PID 515675; ignored that line.
- 3.00% of summed kernel GPU-time is unknown; it was not force-fit into a named H3 stage.
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
