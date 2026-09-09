# MiniMax-H3 Nsight analysis

- Source: `/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-vsa-sage-pr4691-20260910/runs/bf16-nsys/bf16-warmed.nsys-rep`
- SQLite: `/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-vsa-sage-pr4691-20260910/runs/bf16-nsys/bf16-warmed.sqlite`
- Scope: `full_cuda_trace` / `full CUDA trace`
- Window wall time: **27535.432 ms**
- GPU active-span median: **27396.731 ms**
- Slowest trace device by active span: **device 0 / rank 0 / physical GPU 0 / 27535.064 ms**
- Largest non-barrier kernel sum: **trace device 7 / rank 7 / physical GPU 7 / 23245.813 ms**
- Device identity: **kernel PID -> server NCCL rank/cudaDev/nvmlDev/busId**

> GPU-time below is summed across devices and streams; it is not additive wall time.

## Cross-rank decode NVTX timeline

Rank mapping: NVTX PID -> server NCCL rank mapping.

| Stage | Rank | PID | Start / end offset ms | Wall ms | Clipped |
|---|---:|---:|---:|---:|---|
| `mp4_finish` | 0 | 515661 | 27535.278 / 27535.432 | 0.154 | yes |

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
| 0 | 0 | 0 | 0000:06:00.0 | 515661 | 27535.064 | 23482.506 | 23399.225 | 21899.242 / 1499.983 | 90.322 / 39999.79 | 4052.926 | 256.665 |
| 1 | 1 | 4 | 0000:86:00.0 | 515669 | 27396.696 | 23408.801 | 23334.551 | 21767.557 / 1566.995 | 65.578 / 35503.83 | 4126.631 | 215.579 |
| 2 | 2 | 1 | 0000:09:00.0 | 515670 | 27396.887 | 23298.074 | 23222.972 | 21479.215 / 1743.757 | 66.281 / 35503.83 | 4237.358 | 233.351 |
| 3 | 3 | 5 | 0000:89:00.0 | 515671 | 27396.413 | 23378.489 | 23304.177 | 21255.836 / 2048.341 | 65.613 / 35503.83 | 4156.943 | 228.260 |
| 4 | 4 | 2 | 0000:76:00.0 | 515672 | 27397.169 | 23326.749 | 23251.838 | 21524.170 / 1727.668 | 67.925 / 36385.17 | 4208.683 | 229.896 |
| 5 | 5 | 6 | 0000:f6:00.0 | 515673 | 27396.356 | 23352.026 | 23280.070 | 21598.369 / 1681.701 | 65.246 / 36385.17 | 4183.406 | 222.401 |
| 6 | 6 | 3 | 0000:79:00.0 | 515674 | 27396.698 | 23358.046 | 23287.682 | 22175.305 / 1112.376 | 63.435 / 36385.17 | 4177.386 | 221.031 |
| 7 | 7 | 7 | 0000:f9:00.0 | 515675 | 27396.764 | 23398.628 | 23324.578 | 23245.813 / 78.765 | 66.388 / 36385.17 | 4136.804 | 223.693 |

## Per-device wall decomposition (no double-counting)

Per-device disjoint interval sweep. Different simultaneously active phases are charged once to concurrent_mixed; no values are summed across devices. Reference device (largest traced busy union): 0.

| Trace dev | Linear GEMM ms | VSA ms | RDMA barrier ms | RDMA other ms | Pointwise ms | Other single-phase ms | Mixed-concurrent ms | Idle ms | Account error ns |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14352.045 | 4414.317 | 1499.983 | 0.000 | 300.798 | 2900.028 | 15.336 | 4052.926 | 0 |
| 1 | 14250.400 | 4425.374 | 1566.995 | 0.000 | 300.921 | 2865.112 | 0.000 | 4126.631 | 0 |
| 2 | 14062.816 | 4362.044 | 1743.757 | 0.000 | 300.328 | 2829.130 | 0.000 | 4237.358 | 0 |
| 3 | 13774.536 | 4320.887 | 2048.341 | 0.000 | 300.160 | 2934.566 | 0.000 | 4156.943 | 0 |
| 4 | 13012.469 | 4354.899 | 1727.668 | 0.000 | 300.929 | 3930.784 | 0.000 | 4208.683 | 0 |
| 5 | 13027.566 | 4354.885 | 1681.701 | 0.000 | 300.864 | 3987.010 | 0.000 | 4183.406 | 0 |
| 6 | 13545.045 | 4530.938 | 1112.376 | 0.000 | 300.843 | 3868.844 | 0.000 | 4177.386 | 0 |
| 7 | 14125.249 | 5279.154 | 78.765 | 0.000 | 302.920 | 3612.539 | 0.000 | 4136.804 | 0 |

## Kernel categories

| Category | Calls | Total GPU-ms | % kernel GPU-time | Median GPU-ms | Slowest trace dev / ms | Evidence rule |
|---|---:|---:|---:|---:|---:|---|
| `gemm_unknown` | 102129 | 110158.542 | 59.10% | 13918.676 | 0 / 14360.462 | Recognizable GEMM, but its H3 layer role is not encoded in the kernel name |
| `vsa_attention` | 1600 | 36042.498 | 19.34% | 4388.180 | 7 / 5279.154 | Explicit block-sparse/VSA attention kernel, including FlashInfer CuTeDSL SM120 BlockSparseAttnForward |
| `flashinfer_rdma_barrier` | 12800 | 11459.587 | 6.15% | 1624.348 | 3 / 2048.341 | Explicit FlashInfer Ulysses PCIe/RDMA barrier; duration is synchronization residency/wait |
| `nccl` | 1824 | 11255.605 | 6.04% | 1285.162 | 5 / 2205.122 | NCCL device kernel |
| `unknown` | 317087 | 6730.930 | 3.61% | 843.712 | 0 / 969.028 | No conservative category matched |
| `attention_other` | 21568 | 3699.710 | 1.98% | 478.234 | 0 / 532.980 | Non-cuDNN kernel with explicit attention/FMHA/flash-attention evidence |
| `vsa_layout` | 9600 | 3163.403 | 1.70% | 395.531 | 7 / 404.400 | Explicit H3 VSA compact/tile layout kernel |
| `h3_pointwise` | 8032 | 2407.325 | 1.29% | 300.795 | 7 / 302.859 | Explicitly named H3 fused modulation, gated-residual, Q/K norm+RoPE, or SwiGLU kernel |
| `copy_kernel` | 88971 | 1424.638 | 0.76% | 176.180 | 0 / 194.670 | Device-side copy/cat/transpose-like kernel (not a DMA memcpy record) |
| `cudnn_other` | 480 | 61.663 | 0.03% | 0.000 | 0 / 61.663 | cuDNN kernel without explicit attention evidence |
| `cudnn_attention` | 64 | 0.493 | 0.00% | 0.062 | 7 / 0.074 | cuDNN kernel with explicit SDPA/attention/FMHA evidence |
| `triton_reduction` | 128 | 0.438 | 0.00% | 0.054 | 7 / 0.061 | Triton reduction kernel |
| `triton_other` | 128 | 0.262 | 0.00% | 0.032 | 7 / 0.039 | Other Triton kernel |
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

- Observed 12800 calls reconstructed as 1600 chronological groups: 1600 full-device and 0 partial.
- Layout: 10 barrier slots/layer (B0..B9); the group count is consistent with 160 layer cycles.
- Layout basis: explicit VSA evidence: Q/K/V/gate scatters plus O gather, each with opening and closing barriers.
- Full-group arrival-skew sum: 2072.386 ms (7.53% of the selected wall window).
- Dominant latest arrival: trace device 7 in 614/1600 groups (38.38%).
- Latest-device work overlapping the skew windows: gemm_unknown=854.819 GPU-ms, vsa_attention=828.966 GPU-ms, vsa_layout=118.911 GPU-ms, h3_pointwise=117.419 GPU-ms, unknown=55.465 GPU-ms, copy_kernel=8.124 GPU-ms; covered union 1983.704 ms, uncovered 88.682 ms.
- Arrival-skew sums cover sequential full-device barriers and are already inside step wall time; they indicate load-imbalance exposure, not additional time or guaranteed removable time.

### Per-device barrier residency and arrival lag

| Trace dev | Calls | Residency sum ms | Residency median / p95 / max µs | Arrival-lag median / p95 µs | Latest arrival groups / % |
|---:|---:|---:|---:|---:|---:|
| 0 | 1600 | 1499.983 | 10.384 / 4735.428 / 28176.685 | 5.492 / 2232.506 | 147 / 9.19% |
| 1 | 1600 | 1566.995 | 14.432 / 4812.488 / 28291.234 | 3.617 / 1997.834 | 79 / 4.94% |
| 2 | 1600 | 1743.757 | 11.904 / 5009.772 / 9569.133 | 4.969 / 1003.043 | 112 / 7.00% |
| 3 | 1600 | 2048.341 | 15.008 / 5429.921 / 29362.106 | 1.333 / 7.870 | 99 / 6.19% |
| 4 | 1600 | 1727.668 | 7.776 / 4995.023 / 28240.017 | 7.861 / 1146.258 | 337 / 21.06% |
| 5 | 1600 | 1681.701 | 13.680 / 5175.908 / 14596.102 | 3.291 / 1625.860 | 74 / 4.62% |
| 6 | 1600 | 1112.376 | 10.320 / 4167.233 / 14048.793 | 5.075 / 3599.729 | 138 / 8.62% |
| 7 | 1600 | 78.765 | 6.016 / 43.552 / 19016.961 | 4.987 / 5426.324 | 614 / 38.38% |

### Per-layer chronological barrier slots

| Slot | Groups (full) | Arrival-skew sum ms | Arrival skew median / p95 / max µs | Release skew median / p95 / max µs | Dominant latest GPU / count | Late-device kernel evidence (GPU-ms) |
|---:|---:|---:|---:|---:|---:|---|
| B0 | 160 (160) | 393.327 | 310.273 / 5487.068 / 5862.598 | 1.220 / 1.593 / 1.799 | 7 / 108 | vsa_attention=165.323, gemm_unknown=165.203, vsa_layout=23.787, h3_pointwise=23.094, unknown=11.259, copy_kernel=1.478 |
| B1 | 160 (160) | 1.160 | 5.095 / 8.777 / 244.849 | 1.202 / 1.658 / 1.823 | 4 / 35 | uncovered=1.160 |
| B2 | 160 (160) | 399.755 | 1948.433 / 5562.802 / 5947.198 | 1.197 / 1.571 / 1.722 | 7 / 105 | gemm_unknown=170.488, vsa_attention=166.132, vsa_layout=23.784, h3_pointwise=23.565, unknown=11.279, copy_kernel=1.576 |
| B3 | 160 (160) | 0.993 | 5.043 / 11.787 / 37.887 | 1.250 / 1.657 / 1.766 | 4 / 38 | uncovered=0.993 |
| B4 | 160 (160) | 439.972 | 1423.327 / 5515.955 / 29357.840 | 1.192 / 1.584 / 1.766 | 7 / 104 | gemm_unknown=171.228, vsa_attention=153.457, vsa_layout=23.773, h3_pointwise=23.605, unknown=10.702, copy_kernel=1.507 |
| B5 | 160 (160) | 2.527 | 5.337 / 10.566 / 701.466 | 1.256 / 1.728 / 2.029 | 0 / 29 | uncovered=2.527 |
| B6 | 160 (160) | 428.766 | 1887.053 / 5514.067 / 25237.508 | 1.233 / 1.667 / 1.933 | 7 / 100 | vsa_attention=178.762, gemm_unknown=173.243, vsa_layout=23.777, h3_pointwise=23.587, unknown=10.953, copy_kernel=1.903 |
| B7 | 160 (160) | 1.393 | 5.258 / 9.871 / 495.810 | 1.270 / 1.712 / 1.847 | 0 / 27 | uncovered=1.393 |
| B8 | 160 (160) | 403.019 | 1897.417 / 5535.266 / 6197.933 | 1.218 / 1.660 / 1.764 | 7 / 101 | gemm_unknown=174.657, vsa_attention=165.292, vsa_layout=23.790, h3_pointwise=23.568, unknown=11.273, copy_kernel=1.660 |
| B9 | 160 (160) | 1.475 | 5.178 / 13.492 / 321.196 | 1.238 / 1.734 / 1.897 | 4 / 31 | uncovered=1.475 |

### Per-device × barrier slot

| Slot | Trace dev | Calls | Residency sum ms | Residency median / p95 µs | Arrival-lag median / p95 µs | Latest groups |
|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 160 | 288.659 | 305.184 / 4779.457 | 75.862 / 2392.204 | 4 |
| B0 | 1 | 160 | 299.739 | 311.680 / 4920.677 | 52.115 / 2160.874 | 2 |
| B0 | 2 | 160 | 344.601 | 260.033 / 5037.575 | 101.711 / 1100.645 | 2 |
| B0 | 3 | 160 | 393.235 | 308.688 / 5478.142 | 0.000 / 9.240 | 0 |
| B0 | 4 | 160 | 337.876 | 61.568 / 5021.487 | 183.337 / 1270.347 | 38 |
| B0 | 5 | 160 | 330.269 | 310.640 / 5299.010 | 24.682 / 1832.497 | 0 |
| B0 | 6 | 160 | 216.887 | 295.648 / 4306.750 | 43.606 / 3791.177 | 6 |
| B0 | 7 | 160 | 2.688 | 4.256 / 80.576 | 47.624 / 5487.068 | 108 |
| B1 | 0 | 160 | 1.275 | 6.128 / 11.488 | 2.092 / 5.502 | 16 |
| B1 | 1 | 160 | 1.359 | 6.800 / 11.808 | 1.722 / 5.390 | 17 |
| B1 | 2 | 160 | 1.101 | 6.272 / 10.592 | 2.091 / 5.660 | 20 |
| B1 | 3 | 160 | 1.282 | 6.112 / 10.368 | 2.368 / 6.087 | 25 |
| B1 | 4 | 160 | 1.234 | 6.224 / 9.728 | 2.146 / 6.989 | 35 |
| B1 | 5 | 160 | 1.324 | 6.720 / 10.592 | 1.505 / 5.984 | 18 |
| B1 | 6 | 160 | 1.338 | 6.816 / 10.400 | 2.246 / 5.351 | 11 |
| B1 | 7 | 160 | 1.378 | 7.072 / 11.232 | 2.201 / 5.877 | 18 |
| B2 | 0 | 160 | 291.342 | 1110.672 / 4810.467 | 272.635 / 2367.982 | 4 |
| B2 | 1 | 160 | 301.274 | 1049.378 / 4960.902 | 238.067 / 2142.023 | 4 |
| B2 | 2 | 160 | 348.792 | 1571.043 / 5034.214 | 140.584 / 1069.068 | 0 |
| B2 | 3 | 160 | 399.555 | 1952.482 / 5503.294 | 0.000 / 8.465 | 1 |
| B2 | 4 | 160 | 342.871 | 1540.341 / 5035.465 | 186.524 / 1252.690 | 38 |
| B2 | 5 | 160 | 337.040 | 1501.170 / 5324.130 | 25.319 / 1767.135 | 2 |
| B2 | 6 | 160 | 214.373 | 499.168 / 4298.557 | 391.717 / 3772.967 | 6 |
| B2 | 7 | 160 | 2.505 | 4.368 / 49.376 | 1785.555 / 5562.802 | 105 |
| B3 | 0 | 160 | 1.168 | 6.896 / 12.320 | 1.920 / 6.064 | 18 |
| B3 | 1 | 160 | 1.191 | 6.864 / 14.432 | 1.252 / 5.520 | 17 |
| B3 | 2 | 160 | 1.109 | 6.320 / 11.008 | 2.124 / 6.954 | 15 |
| B3 | 3 | 160 | 1.125 | 6.528 / 12.672 | 2.293 / 6.491 | 20 |
| B3 | 4 | 160 | 0.951 | 5.792 / 9.953 | 2.462 / 10.237 | 38 |
| B3 | 5 | 160 | 1.192 | 7.136 / 11.264 | 1.532 / 6.485 | 7 |
| B3 | 6 | 160 | 1.107 | 6.656 / 12.064 | 2.589 / 6.382 | 23 |
| B3 | 7 | 160 | 1.164 | 6.848 / 13.408 | 2.375 / 6.278 | 22 |
| B4 | 0 | 160 | 330.706 | 586.544 / 4792.068 | 50.784 / 2369.099 | 4 |
| B4 | 1 | 160 | 320.131 | 477.216 / 4922.663 | 32.733 / 2156.224 | 1 |
| B4 | 2 | 160 | 345.628 | 460.177 / 5033.641 | 187.737 / 1146.002 | 4 |
| B4 | 3 | 160 | 425.307 | 862.000 / 5520.638 | 0.000 / 9.758 | 0 |
| B4 | 4 | 160 | 358.337 | 462.817 / 5028.395 | 211.798 / 1364.104 | 37 |
| B4 | 5 | 160 | 322.471 | 375.632 / 5216.770 | 53.129 / 1784.130 | 4 |
| B4 | 6 | 160 | 221.958 | 497.008 / 4228.829 | 558.716 / 3779.684 | 6 |
| B4 | 7 | 160 | 41.215 | 4.288 / 172.128 | 1262.059 / 5464.453 | 104 |
| B5 | 0 | 160 | 1.894 | 5.888 / 10.848 | 2.579 / 7.548 | 29 |
| B5 | 1 | 160 | 2.756 | 6.976 / 12.544 | 1.612 / 5.144 | 5 |
| B5 | 2 | 160 | 2.334 | 7.072 / 12.288 | 2.063 / 6.320 | 19 |
| B5 | 3 | 160 | 2.660 | 6.256 / 12.288 | 2.742 / 5.679 | 18 |
| B5 | 4 | 160 | 2.572 | 5.968 / 10.848 | 2.710 / 6.992 | 25 |
| B5 | 5 | 160 | 2.517 | 7.168 / 11.232 | 1.619 / 5.874 | 19 |
| B5 | 6 | 160 | 2.325 | 6.656 / 11.488 | 2.324 / 6.529 | 27 |
| B5 | 7 | 160 | 2.775 | 7.344 / 12.000 | 2.008 / 5.322 | 18 |
| B6 | 0 | 160 | 286.125 | 156.976 / 4839.266 | 293.901 / 2521.217 | 10 |
| B6 | 1 | 160 | 331.067 | 968.306 / 4972.616 | 58.439 / 2197.845 | 2 |
| B6 | 2 | 160 | 345.496 | 538.177 / 5038.151 | 164.986 / 1155.798 | 4 |
| B6 | 3 | 160 | 418.675 | 1856.947 / 5519.653 | 0.000 / 9.352 | 1 |
| B6 | 4 | 160 | 336.730 | 96.865 / 5025.354 | 170.073 / 1427.925 | 35 |
| B6 | 5 | 160 | 343.838 | 163.984 / 5309.121 | 45.481 / 1861.490 | 3 |
| B6 | 6 | 160 | 234.447 | 360.160 / 4348.541 | 390.230 / 3800.231 | 5 |
| B6 | 7 | 160 | 21.364 | 4.368 / 88.000 | 1808.583 / 5514.067 | 100 |
| B7 | 0 | 160 | 1.492 | 5.952 / 10.624 | 1.845 / 7.146 | 27 |
| B7 | 1 | 160 | 1.101 | 6.672 / 11.104 | 1.665 / 5.749 | 11 |
| B7 | 2 | 160 | 1.501 | 6.288 / 10.560 | 2.657 / 6.347 | 27 |
| B7 | 3 | 160 | 1.567 | 6.784 / 11.712 | 1.995 / 6.030 | 21 |
| B7 | 4 | 160 | 1.533 | 6.912 / 10.272 | 1.922 / 7.246 | 22 |
| B7 | 5 | 160 | 1.605 | 7.104 / 10.752 | 1.612 / 5.215 | 13 |
| B7 | 6 | 160 | 1.537 | 6.688 / 10.848 | 2.486 / 5.759 | 21 |
| B7 | 7 | 160 | 1.572 | 6.752 / 11.200 | 2.567 / 6.455 | 18 |
| B8 | 0 | 160 | 296.060 | 1099.553 / 4807.429 | 297.551 / 2402.348 | 9 |
| B8 | 1 | 160 | 306.724 | 1003.171 / 4953.096 | 168.946 / 2194.597 | 2 |
| B8 | 2 | 160 | 351.616 | 1479.156 / 5045.257 | 111.400 / 1134.600 | 0 |
| B8 | 3 | 160 | 403.321 | 1890.386 / 5515.262 | 0.000 / 10.216 | 0 |
| B8 | 4 | 160 | 344.056 | 1564.885 / 5038.123 | 194.981 / 1394.634 | 38 |
| B8 | 5 | 160 | 339.815 | 1447.874 / 5288.709 | 48.183 / 1771.082 | 2 |
| B8 | 6 | 160 | 217.024 | 362.913 / 4279.521 | 571.571 / 3784.444 | 8 |
| B8 | 7 | 160 | 2.447 | 4.432 / 44.448 | 1811.317 / 5535.266 | 101 |
| B9 | 0 | 160 | 1.262 | 6.192 / 13.344 | 2.458 / 6.205 | 26 |
| B9 | 1 | 160 | 1.653 | 6.864 / 13.984 | 1.622 / 5.540 | 18 |
| B9 | 2 | 160 | 1.579 | 6.208 / 13.120 | 2.681 / 6.063 | 21 |
| B9 | 3 | 160 | 1.613 | 6.464 / 13.312 | 2.339 / 5.974 | 13 |
| B9 | 4 | 160 | 1.508 | 6.480 / 10.656 | 2.535 / 8.014 | 31 |
| B9 | 5 | 160 | 1.630 | 6.720 / 11.296 | 1.907 / 6.125 | 6 |
| B9 | 6 | 160 | 1.380 | 6.736 / 10.944 | 2.534 / 6.555 | 25 |
| B9 | 7 | 160 | 1.658 | 6.960 / 12.032 | 2.417 / 6.235 | 20 |

Attribution limits:

- Barrier slots B0..B9 are chronological modulo-10 positions; the trace has no semantic exchange labels.
- Barrier kernel residency is synchronization/spin time, not RDMA byte-transfer time.
- The latest-arriving device constrains that barrier release, but the trace alone does not identify why its preceding work was late.
- Late-path kernel evidence is overlap on the latest-arriving device during the arrival-skew window; uncovered time may be CPU, transport, DMA, or untraced work.

## FlashInfer/RDMA evidence and launch gaps

- Explicitly named kernels: 12800
- Matched CUDA launches: 12792; unmatched: 8
- Launch→GPU gap median/p95/max: 3.513 / 52192.361 / 54949.046 µs
- The CUDA launch API return-to-GPU-start gap includes stream queueing and backpressure; it is neither pure CPU dispatch overhead nor network transfer duration.

## CUDA API categories

| Category | Calls | Summed thread-ms | Interpretation |
|---|---:|---:|---|
| `synchronization_wait` | 21667 | 159884.825 | CUDA synchronize calls; duration includes waiting and is not CPU overhead alone |
| `kernel_launch` | 564410 | 19051.185 | CUDA kernel/graph launch APIs |
| `memory_management` | 5386 | 4204.630 | CUDA allocation, mapping, and free APIs |
| `memset` | 89363 | 2634.546 | CUDA memset APIs |
| `other` | 462680 | 2546.572 | Other CUDA APIs |
| `memcpy` | 38228 | 551.646 | CUDA memcpy APIs |
| `event_stream` | 82118 | 69.826 | CUDA event and stream bookkeeping APIs |
| `external_memory_possible_rdma_setup` | 0 | 0.000 | CUDA external/shareable-memory APIs; possible registration setup, not measured RDMA transfer time |

## Memcpy operations

| Trace dev | Kind | Calls | MiB (overlapping ops) | GPU-ms |
|---:|---|---:|---:|---:|
| 0 | Device-to-Device | 2156 | 39021.956 | 70.150 |
| 0 | Device-to-Host | 2557 | 938.278 | 18.562 |
| 0 | Host-to-Device | 188 | 39.552 | 1.610 |
| 1 | Device-to-Device | 2132 | 35464.143 | 62.343 |
| 1 | Host-to-Device | 129 | 38.368 | 2.192 |
| 1 | Device-to-Host | 2512 | 1.316 | 1.043 |
| 2 | Device-to-Device | 2132 | 35464.143 | 63.118 |
| 2 | Host-to-Device | 129 | 38.368 | 2.066 |
| 2 | Device-to-Host | 2512 | 1.316 | 1.097 |
| 3 | Device-to-Device | 2132 | 35464.143 | 62.384 |
| 3 | Host-to-Device | 129 | 38.368 | 2.225 |
| 3 | Device-to-Host | 2512 | 1.316 | 1.004 |
| 4 | Device-to-Device | 2111 | 36345.487 | 64.891 |
| 4 | Host-to-Device | 129 | 38.368 | 1.930 |
| 4 | Device-to-Host | 2512 | 1.316 | 1.104 |
| 5 | Device-to-Device | 2111 | 36345.487 | 61.994 |
| 5 | Host-to-Device | 129 | 38.368 | 2.242 |
| 5 | Device-to-Host | 2512 | 1.316 | 1.009 |
| 6 | Device-to-Device | 2111 | 36345.487 | 60.359 |
| 6 | Host-to-Device | 129 | 38.368 | 1.999 |
| 6 | Device-to-Host | 2512 | 1.316 | 1.078 |
| 7 | Device-to-Device | 2111 | 36345.487 | 62.778 |
| 7 | Host-to-Device | 129 | 38.368 | 2.312 |
| 7 | Device-to-Host | 2512 | 1.316 | 1.298 |

## Communication NVTX evidence

These CPU-range totals can overlap because NCCL group/collective ranges may be nested.

| Evidence | Range | Calls | Summed CPU-range ms |
|---|---|---:|---:|
| `nccl` | `ncclAllReduce` | 1016 | 20.998 |
| `nccl` | `ncclAllGather` | 752 | 18.751 |
| `nccl` | `ncclBroadcast` | 56 | 1.475 |

## Top kernels

| # | Category | Calls | Total GPU-ms | Avg µs | % kernel GPU-time | Kernel |
|---:|---|---:|---:|---:|---:|---|
| 1 | `gemm_unknown` | 5792 | 65869.105 | 11372.428 | 35.34% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params)` |
| 2 | `vsa_attention` | 1600 | 36042.498 | 22526.561 | 19.34% | `kernel_cutlass_kernel_flashinfercute_dslsparsesm120_blk64flash_fwd_sm120BlockSparseAttnForwardSm120Blk64_object_at__tensor0000o12811101213_tensor0000o12811101213_tensor0000o128101…` |
| 3 | `gemm_unknown` | 3200 | 14113.027 | 4410.321 | 7.57% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8>(T1::Params)` |
| 4 | `gemm_unknown` | 21168 | 12572.394 | 593.934 | 6.74% | `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` |
| 5 | `flashinfer_rdma_barrier` | 12800 | 11459.587 | 895.280 | 6.15% | `flashinfer::comm::ulysses_pcie::UlyssesPcieBarrier(unsigned long *, flashinfer::comm::ulysses_pcie::PeerSignalPointers, int, int, unsigned long *)` |
| 6 | `gemm_unknown` | 42336 | 10465.888 | 247.210 | 5.61% | `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn_align8>(T1::Params)` |
| 7 | `nccl` | 168 | 5885.568 | 35033.144 | 3.16% | `ncclDevKernel_AllReduce_Sum_f32_TREE_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 8 | `gemm_unknown` | 21168 | 5532.024 | 261.339 | 2.97% | `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_tn_align8>(T1::Params)` |
| 9 | `nccl` | 752 | 4822.261 | 6412.581 | 2.59% | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 10 | `attention_other` | 21168 | 3694.899 | 174.551 | 1.98% | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)64, (int)128, (int)128, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)64, (int)128, (int)12…` |
| 11 | `vsa_layout` | 6400 | 1997.448 | 312.101 | 1.07% | `_h3_vsa_tile_pack_kernel` |
| 12 | `unknown` | 1600 | 1362.744 | 851.715 | 0.73% | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vllm::packed_silu_kernel<__nv_bfloat162>, (bool)1, (bool)1, (bool)0, (bool)1>(T1 *…` |
| 13 | `h3_pointwise` | 3200 | 938.295 | 293.217 | 0.50% | `_rms_norm_rope_kernel` |
| 14 | `unknown` | 4800 | 784.846 | 163.510 | 0.42% | `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<float, at::native::sum_functor<c10::BFloat16, float, float>::operat…` |
| 15 | `vsa_layout` | 1600 | 695.779 | 434.862 | 0.37% | `_h3_vsa_o_bundle_local_gate_kernel` |
| 16 | `h3_pointwise` | 1600 | 656.049 | 410.030 | 0.35% | `_indexed_gate_rms_norm_scale_shift_kernel` |
| 17 | `unknown` | 21168 | 652.702 | 30.834 | 0.35% | `void vllm::act_and_mul_kernel<c10::Half, __half2, &vllm::silu_kernel<c10::Half>, &vllm::packed_silu_kernel<__half2>, (bool)1, (bool)1, (bool)0, (bool)1>(T1 *, const T1 *, int, flo…` |
| 18 | `copy_kernel` | 24069 | 615.561 | 25.575 | 0.33% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 19 | `unknown` | 42336 | 604.453 | 14.278 | 0.32% | `_scaled_residual_exact_kernel` |
| 20 | `gemm_unknown` | 1600 | 591.188 | 369.492 | 0.32% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)` |
| 21 | `copy_kernel` | 2432 | 531.396 | 218.502 | 0.29% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 22 | `unknown` | 42336 | 499.907 | 11.808 | 0.27% | `void at::native::<unnamed>::vectorized_layer_norm_kernel<float, float, (bool)1>(int, T2, const T1 *, const T1 *, const T1 *, T2 *, T2 *, T1 *)` |
| 23 | `gemm_unknown` | 592 | 488.806 | 825.686 | 0.26% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(T1::Params)` |
| 24 | `vsa_layout` | 1600 | 470.177 | 293.861 | 0.25% | `_h3_vsa_o_bundle_kernel` |
| 25 | `h3_pointwise` | 1600 | 450.518 | 281.574 | 0.24% | `_indexed_gate_kernel` |
| 26 | `nccl` | 800 | 447.041 | 558.801 | 0.24% | `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 27 | `gemm_unknown` | 1600 | 398.799 | 249.249 | 0.21% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(T1::Params)` |
| 28 | `unknown` | 1600 | 395.141 | 246.963 | 0.21% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, float, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6…` |
| 29 | `unknown` | 1600 | 390.641 | 244.151 | 0.21% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, int, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6, …` |
| 30 | `h3_pointwise` | 1600 | 355.018 | 221.886 | 0.19% | `_rms_norm_indexed_scale_shift_kernel` |

## Top OS runtime calls

OSRT time is summed across threads and is never assigned to RDMA without explicit naming evidence.

| Call | Count | Summed thread-ms |
|---|---:|---:|
| `pthread_cond_timedwait` | 31533 | 3357177.741 |
| `poll` | 6883 | 2422987.375 |
| `epoll_wait` | 179091 | 1651056.737 |
| `pthread_cond_wait` | 43473 | 421095.270 |
| `sem_clockwait` | 313 | 384580.395 |
| `epoll_pwait` | 26159 | 54827.029 |
| `sem_wait` | 24 | 33293.760 |
| `ioctl` | 14528 | 10016.139 |
| `open64` | 4096 | 116.706 |
| `pthread_mutex_lock` | 6612 | 96.524 |
| `fopen` | 8 | 78.810 |
| `pthread_cond_broadcast` | 8291 | 23.682 |
| `recv` | 1091 | 14.679 |
| `send` | 1112 | 11.000 |
| `pthread_cond_signal` | 6085 | 9.505 |
| `waitpid` | 262 | 9.347 |
| `fgets` | 1793 | 7.953 |
| `write` | 340 | 3.026 |
| `recvmsg` | 72 | 2.892 |
| `pthread_rwlock_wrlock` | 42 | 2.597 |

## Warnings and attribution limits

- No usable NVTX range was selected; results cover the complete CUDA GPU trace and may include startup.
- NCCL logger rank 7 disagrees with communicator rank 2 for PID 515675; ignored that line.
- 3.61% of summed kernel GPU-time is unknown; it was not force-fit into a named H3 stage.
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
