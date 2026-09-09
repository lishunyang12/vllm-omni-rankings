# MiniMax-H3 Nsight analysis

- Source: `/lustre/raplab/client/sylarl/minimax-h3-native/results/vllm-omni-fasth3-original-cutedsl-mlp-fp8-bf16-wire-e2e-nsys-20260909T085332Z/minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.nsys-rep`
- SQLite: `/lustre/raplab/client/sylarl/minimax-h3-native/results/vllm-omni-fasth3-original-cutedsl-mlp-fp8-bf16-wire-e2e-nsys-20260909T085332Z/minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.sqlite`
- Scope: `full_cuda_trace` / `full CUDA trace`
- Window wall time: **25258.295 ms**
- GPU active-span median: **25255.878 ms**
- Slowest trace device by active span: **device 4 / rank 4 / physical GPU 2 / 25258.247 ms**
- Largest non-barrier kernel sum: **trace device 7 / rank 7 / physical GPU 7 / 15463.063 ms**
- Device identity: **kernel PID -> server NCCL rank/cudaDev/nvmlDev/busId**

> GPU-time below is summed across devices and streams; it is not additive wall time.

## Cross-rank decode NVTX timeline

No explicit nested `minimax_h3.decode.*` stage range was found in scope.

## Per-GPU timeline

| Trace dev | Rank | Physical GPU | PCI bus | PID(s) | Active span ms | Busy union ms | Kernel-sum ms | Non-barrier / barrier kernel ms | Memcpy ms / MiB | Window idle ms | Largest idle gap ms |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0000:06:00.0 | 731733 | 25241.650 | 15413.282 | 15365.222 | 13515.778 / 1849.444 | 63.434 / 26633.80 | 9845.012 | 1184.783 |
| 1 | 1 | 4 | 0000:86:00.0 | 731738 | 25255.683 | 14931.355 | 14893.864 | 12839.779 / 2054.085 | 36.185 / 19626.75 | 10326.939 | 1258.424 |
| 2 | 2 | 1 | 0000:09:00.0 | 731739 | 25257.550 | 14552.772 | 14514.511 | 12584.945 / 1929.566 | 36.917 / 19626.75 | 10705.523 | 1258.661 |
| 3 | 3 | 5 | 0000:89:00.0 | 731740 | 25255.648 | 14760.890 | 14723.249 | 12472.223 / 2251.026 | 36.387 / 19626.75 | 10497.405 | 1252.101 |
| 4 | 4 | 2 | 0000:76:00.0 | 731741 | 25258.247 | 14920.154 | 14881.519 | 13002.402 / 1879.117 | 37.330 / 19626.75 | 10338.140 | 1221.975 |
| 5 | 5 | 6 | 0000:f6:00.0 | 731742 | 25255.665 | 14626.763 | 14591.516 | 12388.203 / 2203.313 | 34.055 / 19626.75 | 10631.532 | 1259.210 |
| 6 | 6 | 3 | 0000:79:00.0 | 731743 | 25257.829 | 15449.702 | 15415.662 | 13666.456 / 1749.206 | 32.680 / 19626.75 | 9808.592 | 1262.244 |
| 7 | 7 | 7 | 0000:f9:00.0 | 731744 | 25256.072 | 15523.787 | 15487.174 | 15463.063 / 24.111 | 34.993 / 19626.75 | 9734.508 | 1258.056 |

## Per-device wall decomposition (no double-counting)

Per-device disjoint interval sweep. Different simultaneously active phases are charged once to concurrent_mixed; no values are summed across devices. Reference device (largest traced busy union): 7.

| Trace dev | Linear GEMM ms | VSA ms | RDMA barrier ms | RDMA other ms | Pointwise ms | Other single-phase ms | Mixed-concurrent ms | Idle ms | Account error ns |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6796.874 | 4340.714 | 1849.444 | 0.000 | 424.631 | 1987.020 | 14.599 | 9845.012 | 0 |
| 1 | 6670.708 | 4254.157 | 2054.085 | 0.000 | 423.305 | 1529.100 | 0.000 | 10326.939 | 0 |
| 2 | 6678.118 | 4344.764 | 1929.566 | 0.000 | 423.575 | 1176.748 | 0.000 | 10705.523 | 0 |
| 3 | 6539.311 | 4179.050 | 2251.026 | 0.000 | 423.814 | 1367.688 | 0.000 | 10497.405 | 0 |
| 4 | 6728.591 | 4338.487 | 1879.117 | 0.000 | 423.608 | 1550.352 | 0.000 | 10338.140 | 0 |
| 5 | 6602.744 | 4151.534 | 2203.313 | 0.000 | 423.269 | 1245.902 | 0.000 | 10631.532 | 0 |
| 6 | 6851.567 | 4345.550 | 1749.206 | 0.000 | 424.256 | 2079.123 | 0.000 | 9808.592 | 0 |
| 7 | 7591.927 | 5280.078 | 24.111 | 0.000 | 428.230 | 2199.441 | 0.000 | 9734.508 | 0 |

## Kernel categories

| Category | Calls | Total GPU-ms | % kernel GPU-time | Median GPU-ms | Slowest trace dev / ms | Evidence rule |
|---|---:|---:|---:|---:|---:|---|
| `gemm_unknown` | 15759 | 54459.840 | 45.43% | 6703.354 | 7 / 7591.927 | Recognizable GEMM, but its H3 layer role is not encoded in the kernel name |
| `vsa_attention` | 1600 | 35234.335 | 29.39% | 4339.600 | 7 / 5280.078 | Explicit block-sparse/VSA attention kernel, including FlashInfer CuTeDSL SM120 BlockSparseAttnForward |
| `flashinfer_rdma_barrier` | 12800 | 13939.868 | 11.63% | 1904.342 | 3 / 2251.026 | Explicit FlashInfer Ulysses PCIe/RDMA barrier; duration is synchronization residency/wait |
| `unknown` | 82958 | 4371.800 | 3.65% | 522.786 | 0 / 650.382 | No conservative category matched |
| `nccl` | 1224 | 4057.310 | 3.38% | 406.059 | 7 / 1072.367 | NCCL device kernel |
| `vsa_layout` | 9600 | 3164.213 | 2.64% | 395.709 | 7 / 404.080 | Explicit H3 VSA compact/tile layout kernel |
| `h3_pointwise` | 8032 | 2407.543 | 2.01% | 300.669 | 7 / 302.833 | Explicitly named H3 fused modulation, gated-residual, Q/K norm+RoPE, or SwiGLU kernel |
| `triton_reduction` | 3328 | 987.147 | 0.82% | 123.087 | 7 / 125.397 | Triton reduction kernel |
| `copy_kernel` | 31362 | 850.652 | 0.71% | 100.900 | 0 / 141.153 | Device-side copy/cat/transpose-like kernel (not a DMA memcpy record) |
| `cudnn_other` | 3296 | 394.411 | 0.33% | 0.000 | 0 / 337.131 | cuDNN kernel without explicit attention evidence |
| `attention_other` | 400 | 4.841 | 0.00% | 0.605 | 7 / 0.715 | Non-cuDNN kernel with explicit attention/FMHA/flash-attention evidence |
| `cudnn_attention` | 64 | 0.495 | 0.00% | 0.063 | 7 / 0.074 | cuDNN kernel with explicit SDPA/attention/FMHA evidence |
| `triton_other` | 128 | 0.262 | 0.00% | 0.033 | 7 / 0.040 | Other Triton kernel |
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
- Layout: 8 barrier slots/layer (B0..B7); the group count is consistent with 200 layer cycles.
- Layout basis: explicit analyzer override.
- Full-group arrival-skew sum: 2286.617 ms (9.05% of the selected wall window).
- Dominant latest arrival: trace device 7 in 728/1600 groups (45.50%).
- Latest-device work overlapping the skew windows: vsa_attention=991.569 GPU-ms, gemm_unknown=963.382 GPU-ms, vsa_layout=118.770 GPU-ms, h3_pointwise=117.328 GPU-ms, unknown=55.729 GPU-ms, copy_kernel=6.186 GPU-ms; covered union 2252.963 ms, uncovered 33.654 ms.
- Arrival-skew sums cover sequential full-device barriers and are already inside step wall time; they indicate load-imbalance exposure, not additional time or guaranteed removable time.

### Per-device barrier residency and arrival lag

| Trace dev | Calls | Residency sum ms | Residency median / p95 / max µs | Arrival-lag median / p95 µs | Latest arrival groups / % |
|---:|---:|---:|---:|---:|---:|
| 0 | 1600 | 1849.444 | 9.680 / 5133.787 / 15620.776 | 4.573 / 1138.454 | 122 / 7.62% |
| 1 | 1600 | 2054.085 | 10.464 / 5664.025 / 14223.748 | 3.442 / 612.321 | 73 / 4.56% |
| 2 | 1600 | 1929.566 | 9.888 / 5104.192 / 14481.491 | 4.579 / 1130.367 | 126 / 7.88% |
| 3 | 1600 | 2251.026 | 9.824 / 6066.352 / 12600.974 | 2.229 / 171.519 | 89 / 5.56% |
| 4 | 1600 | 1879.117 | 9.152 / 5125.566 / 10227.542 | 5.192 / 1105.213 | 146 / 9.12% |
| 5 | 1600 | 2203.313 | 9.008 / 6223.667 / 9651.020 | 2.513 / 351.583 | 135 / 8.44% |
| 6 | 1600 | 1749.206 | 8.432 / 5096.661 / 11483.422 | 5.535 / 1559.058 | 181 / 11.31% |
| 7 | 1600 | 24.111 | 4.768 / 23.072 / 7210.867 | 5.304 / 6219.344 | 728 / 45.50% |

### Per-layer chronological barrier slots

| Slot | Groups (full) | Arrival-skew sum ms | Arrival skew median / p95 / max µs | Release skew median / p95 / max µs | Dominant latest GPU / count | Late-device kernel evidence (GPU-ms) |
|---:|---:|---:|---:|---:|---:|---|
| B0 | 200 (200) | 1039.295 | 5133.373 / 5545.649 / 15616.871 | 0.992 / 1.158 / 1.267 | 7 / 198 | gemm_unknown=904.760, h3_pointwise=117.328 |
| B1 | 200 (200) | 1.192 | 5.409 / 8.210 / 37.527 | 1.169 / 1.645 / 1.836 | 6 / 35 | uncovered=1.192 |
| B2 | 200 (200) | 5.881 | 20.221 / 43.275 / 633.332 | 1.385 / 1.653 / 1.950 | 6 / 60 | uncovered=5.881 |
| B3 | 200 (200) | 1.048 | 5.063 / 7.499 / 10.046 | 1.171 / 1.590 / 1.924 | 7 / 47 | uncovered=1.048 |
| B4 | 200 (200) | 6.072 | 29.136 / 41.491 / 122.870 | 1.223 / 1.395 / 2.726 | 7 / 183 | copy_kernel=5.649 |
| B5 | 200 (200) | 1.027 | 4.925 / 7.613 / 11.596 | 1.144 / 1.614 / 1.767 | 7 / 36 | uncovered=1.027 |
| B6 | 200 (200) | 1230.942 | 6197.865 / 6332.042 / 6402.897 | 1.052 / 1.170 / 1.798 | 7 / 198 | vsa_attention=991.569, vsa_layout=118.770, gemm_unknown=58.621, unknown=55.729, copy_kernel=0.537 |
| B7 | 200 (200) | 1.159 | 5.341 / 7.982 / 31.357 | 1.204 / 1.648 / 1.843 | 6 / 33 | uncovered=1.159 |

### Per-device × barrier slot

| Slot | Trace dev | Calls | Residency sum ms | Residency median / p95 µs | Arrival-lag median / p95 µs | Latest groups |
|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 200 | 815.630 | 4073.659 / 4443.866 | 1078.486 / 1181.583 | 1 |
| B0 | 1 | 200 | 916.698 | 4552.942 / 4897.430 | 590.753 / 673.714 | 0 |
| B0 | 2 | 200 | 904.535 | 4524.110 / 4863.293 | 655.212 / 750.476 | 0 |
| B0 | 3 | 200 | 1036.243 | 5137.904 / 5549.802 | 0.000 / 0.000 | 0 |
| B0 | 4 | 200 | 849.566 | 4232.096 / 4681.627 | 912.254 / 1006.549 | 0 |
| B0 | 5 | 200 | 958.983 | 4792.587 / 5340.790 | 329.014 / 508.531 | 1 |
| B0 | 6 | 200 | 727.156 | 3630.730 / 4001.173 | 1535.476 / 1634.976 | 0 |
| B0 | 7 | 200 | 11.569 | 3.952 / 4.384 | 5133.373 / 5545.649 | 198 |
| B1 | 0 | 200 | 1.404 | 6.704 / 10.560 | 2.244 / 6.206 | 27 |
| B1 | 1 | 200 | 1.433 | 6.576 / 10.816 | 2.281 / 5.652 | 12 |
| B1 | 2 | 200 | 1.318 | 6.512 / 10.368 | 2.353 / 6.649 | 27 |
| B1 | 3 | 200 | 1.402 | 7.056 / 9.856 | 2.020 / 5.858 | 14 |
| B1 | 4 | 200 | 1.346 | 6.592 / 10.016 | 2.436 / 5.790 | 23 |
| B1 | 5 | 200 | 1.274 | 6.128 / 9.888 | 2.681 / 6.659 | 30 |
| B1 | 6 | 200 | 1.287 | 6.464 / 9.248 | 3.050 / 5.911 | 35 |
| B1 | 7 | 200 | 1.350 | 6.880 / 9.536 | 2.735 / 6.841 | 32 |
| B2 | 0 | 200 | 5.218 | 17.504 / 40.288 | 6.085 / 19.009 | 13 |
| B2 | 1 | 200 | 5.226 | 17.952 / 39.648 | 5.219 / 16.342 | 12 |
| B2 | 2 | 200 | 4.526 | 15.456 / 37.824 | 5.918 / 28.540 | 21 |
| B2 | 3 | 200 | 4.788 | 18.256 / 43.264 | 4.744 / 20.659 | 12 |
| B2 | 4 | 200 | 4.312 | 12.848 / 36.288 | 9.331 / 29.298 | 27 |
| B2 | 5 | 200 | 3.914 | 12.608 / 35.616 | 9.155 / 34.395 | 44 |
| B2 | 6 | 200 | 3.810 | 9.248 / 35.968 | 11.862 / 29.990 | 60 |
| B2 | 7 | 200 | 5.302 | 18.272 / 39.808 | 5.332 / 17.773 | 11 |
| B3 | 0 | 200 | 1.211 | 6.192 / 8.992 | 2.329 / 5.762 | 21 |
| B3 | 1 | 200 | 1.341 | 6.976 / 10.176 | 1.220 / 5.537 | 17 |
| B3 | 2 | 200 | 1.256 | 6.384 / 10.048 | 2.259 / 5.903 | 26 |
| B3 | 3 | 200 | 1.233 | 6.144 / 9.088 | 2.310 / 6.331 | 14 |
| B3 | 4 | 200 | 1.246 | 6.272 / 9.440 | 2.409 / 6.127 | 29 |
| B3 | 5 | 200 | 1.394 | 7.248 / 9.696 | 1.200 / 5.161 | 20 |
| B3 | 6 | 200 | 1.262 | 6.496 / 8.832 | 2.423 / 5.544 | 26 |
| B3 | 7 | 200 | 1.110 | 5.616 / 8.384 | 3.447 / 6.120 | 47 |
| B4 | 0 | 200 | 5.549 | 26.608 / 38.880 | 6.730 / 15.172 | 1 |
| B4 | 1 | 200 | 6.262 | 30.192 / 43.616 | 2.159 / 9.844 | 1 |
| B4 | 2 | 200 | 5.270 | 25.744 / 38.752 | 7.042 / 17.086 | 3 |
| B4 | 3 | 200 | 6.316 | 30.608 / 42.496 | 1.688 / 10.517 | 2 |
| B4 | 4 | 200 | 5.074 | 25.024 / 36.384 | 8.788 / 21.139 | 2 |
| B4 | 5 | 200 | 5.569 | 27.888 / 40.224 | 4.774 / 18.206 | 5 |
| B4 | 6 | 200 | 4.780 | 22.864 / 37.024 | 10.495 / 21.399 | 3 |
| B4 | 7 | 200 | 1.098 | 3.680 / 12.352 | 28.748 / 36.372 | 183 |
| B5 | 0 | 200 | 1.144 | 5.504 / 8.960 | 2.748 / 5.657 | 33 |
| B5 | 1 | 200 | 1.335 | 6.768 / 9.984 | 1.480 / 4.743 | 17 |
| B5 | 2 | 200 | 1.288 | 6.640 / 9.888 | 1.907 / 5.347 | 22 |
| B5 | 3 | 200 | 1.269 | 6.448 / 9.472 | 2.053 / 5.317 | 18 |
| B5 | 4 | 200 | 1.193 | 5.776 / 9.152 | 2.626 / 5.534 | 33 |
| B5 | 5 | 200 | 1.338 | 6.880 / 9.184 | 1.659 / 5.573 | 18 |
| B5 | 6 | 200 | 1.245 | 6.304 / 9.376 | 2.383 / 5.362 | 23 |
| B5 | 7 | 200 | 1.264 | 6.432 / 9.343 | 2.349 / 5.607 | 36 |
| B6 | 0 | 200 | 1017.984 | 5127.563 / 5165.209 | 1077.459 / 1194.124 | 1 |
| B6 | 1 | 200 | 1120.355 | 5649.307 / 5783.291 | 554.486 / 622.491 | 0 |
| B6 | 2 | 200 | 1010.030 | 5098.046 / 5143.072 | 1107.745 / 1250.125 | 0 |
| B6 | 3 | 200 | 1198.411 | 6045.421 / 6201.737 | 159.638 / 233.958 | 0 |
| B6 | 4 | 200 | 1015.090 | 5122.350 / 5158.845 | 1085.251 / 1213.662 | 0 |
| B6 | 5 | 200 | 1229.433 | 6202.104 / 6336.285 | 0.000 / 0.000 | 0 |
| B6 | 6 | 200 | 1008.325 | 5088.786 / 5134.673 | 1110.387 / 1242.667 | 1 |
| B6 | 7 | 200 | 0.978 | 4.192 / 4.768 | 6197.865 / 6332.042 | 198 |
| B7 | 0 | 200 | 1.304 | 6.464 / 9.696 | 2.482 / 5.699 | 25 |
| B7 | 1 | 200 | 1.434 | 7.328 / 10.560 | 1.330 / 5.311 | 14 |
| B7 | 2 | 200 | 1.343 | 6.576 / 10.624 | 2.080 / 6.118 | 27 |
| B7 | 3 | 200 | 1.364 | 7.040 / 10.048 | 2.016 / 5.839 | 29 |
| B7 | 4 | 200 | 1.290 | 6.480 / 9.920 | 2.624 / 6.178 | 32 |
| B7 | 5 | 200 | 1.408 | 7.264 / 9.984 | 2.024 / 6.550 | 17 |
| B7 | 6 | 200 | 1.340 | 6.736 / 9.984 | 2.671 / 6.212 | 33 |
| B7 | 7 | 200 | 1.440 | 7.456 / 10.112 | 2.173 / 5.652 | 23 |

Attribution limits:

- Barrier slots B0..B7 are chronological modulo-8 positions; the trace has no semantic exchange labels.
- Barrier kernel residency is synchronization/spin time, not RDMA byte-transfer time.
- The latest-arriving device constrains that barrier release, but the trace alone does not identify why its preceding work was late.
- Late-path kernel evidence is overlap on the latest-arriving device during the arrival-skew window; uncovered time may be CPU, transport, DMA, or untraced work.

## FlashInfer/RDMA evidence and launch gaps

- Explicitly named kernels: 12800
- Matched CUDA launches: 12790; unmatched: 10
- Launch→GPU gap median/p95/max: 3.708 / 35728.796 / 40008.598 µs
- The CUDA launch API return-to-GPU-start gap includes stream queueing and backpressure; it is neither pure CPU dispatch overhead nor network transfer duration.

## CUDA API categories

| Category | Calls | Summed thread-ms | Interpretation |
|---|---:|---:|---|
| `synchronization_wait` | 19109 | 115866.625 | CUDA synchronize calls; duration includes waiting and is not CPU overhead alone |
| `other` | 185898 | 5438.907 | Other CUDA APIs |
| `memory_management` | 5199 | 2897.582 | CUDA allocation, mapping, and free APIs |
| `kernel_launch` | 170551 | 2826.298 | CUDA kernel/graph launch APIs |
| `memset` | 6203 | 2266.159 | CUDA memset APIs |
| `external_memory_possible_rdma_setup` | 480 | 1873.607 | CUDA external/shareable-memory APIs; possible registration setup, not measured RDMA transfer time |
| `memcpy` | 33046 | 470.688 | CUDA memcpy APIs |
| `event_stream` | 71698 | 282.448 | CUDA event and stream bookkeeping APIs |

## Memcpy operations

| Trace dev | Kind | Calls | MiB (overlapping ops) | GPU-ms |
|---:|---|---:|---:|---:|
| 0 | Device-to-Device | 1949 | 25655.509 | 43.353 |
| 0 | Device-to-Host | 2256 | 938.270 | 18.399 |
| 0 | Host-to-Device | 325 | 40.024 | 1.682 |
| 1 | Device-to-Device | 1551 | 19586.603 | 33.649 |
| 1 | Host-to-Device | 294 | 38.837 | 1.665 |
| 1 | Device-to-Host | 2233 | 1.308 | 0.871 |
| 2 | Device-to-Device | 1550 | 19586.602 | 34.407 |
| 2 | Host-to-Device | 290 | 38.837 | 1.584 |
| 2 | Device-to-Host | 2233 | 1.308 | 0.926 |
| 3 | Device-to-Device | 1550 | 19586.602 | 33.827 |
| 3 | Host-to-Device | 290 | 38.837 | 1.700 |
| 3 | Device-to-Host | 2233 | 1.308 | 0.860 |
| 4 | Device-to-Device | 1550 | 19586.602 | 34.679 |
| 4 | Host-to-Device | 290 | 38.837 | 1.719 |
| 4 | Device-to-Host | 2233 | 1.308 | 0.931 |
| 5 | Device-to-Device | 1550 | 19586.602 | 31.703 |
| 5 | Host-to-Device | 290 | 38.837 | 1.505 |
| 5 | Device-to-Host | 2233 | 1.308 | 0.847 |
| 6 | Device-to-Device | 1550 | 19586.602 | 30.065 |
| 6 | Host-to-Device | 290 | 38.837 | 1.688 |
| 6 | Device-to-Host | 2233 | 1.308 | 0.926 |
| 7 | Device-to-Device | 1550 | 19586.602 | 32.176 |
| 7 | Host-to-Device | 290 | 38.837 | 1.701 |
| 7 | Device-to-Host | 2233 | 1.308 | 1.116 |

## Communication NVTX evidence

These CPU-range totals can overlap because NCCL group/collective ranges may be nested.

| Evidence | Range | Calls | Summed CPU-range ms |
|---|---|---:|---:|
| `nccl` | `ncclCommInitRankConfig` | 32 | 8994.879 |
| `nccl` | `ncclAllReduce` | 816 | 4380.589 |
| `nccl` | `ncclAllGather` | 336 | 2690.289 |
| `nccl` | `ncclBroadcast` | 72 | 2082.995 |

## Top kernels

| # | Category | Calls | Total GPU-ms | Avg µs | % kernel GPU-time | Kernel |
|---:|---|---:|---:|---:|---:|---|
| 1 | `vsa_attention` | 1600 | 35234.335 | 22021.459 | 29.39% | `kernel_cutlass_kernel_flashinfercute_dslsparsesm120_blk64flash_fwd_sm120BlockSparseAttnForwardSm120Blk64_object_at__tensor0000o12811101213_tensor0000o12811101213_tensor0000o128101…` |
| 2 | `gemm_unknown` | 2592 | 20863.163 | 8049.060 | 17.40% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params)` |
| 3 | `gemm_unknown` | 3200 | 18873.007 | 5897.815 | 15.74% | `void cutlass::device_kernel<enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::Mainl…` |
| 4 | `flashinfer_rdma_barrier` | 12800 | 13939.868 | 1089.052 | 11.63% | `flashinfer::comm::ulysses_pcie::UlyssesPcieBarrier(unsigned long *, flashinfer::comm::ulysses_pcie::PeerSignalPointers, int, int, unsigned long *)` |
| 5 | `gemm_unknown` | 3200 | 13588.982 | 4246.557 | 11.34% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8>(T1::Params)` |
| 6 | `vsa_layout` | 6400 | 1997.377 | 312.090 | 1.67% | `_h3_vsa_tile_pack_kernel` |
| 7 | `nccl` | 336 | 1761.179 | 5241.603 | 1.47% | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 8 | `nccl` | 8 | 1650.724 | 206340.488 | 1.38% | `ncclDevKernel_AllReduce_Sum_u32_TREE_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 9 | `unknown` | 1600 | 1350.061 | 843.788 | 1.13% | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vllm::packed_silu_kernel<__nv_bfloat162>, (bool)1, (bool)1, (bool)0, (bool)1>(T1 *…` |
| 10 | `h3_pointwise` | 3200 | 938.155 | 293.173 | 0.78% | `_rms_norm_rope_kernel` |
| 11 | `unknown` | 4800 | 785.960 | 163.742 | 0.66% | `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<float, at::native::sum_functor<c10::BFloat16, float, float>::operat…` |
| 12 | `triton_reduction` | 1600 | 723.893 | 452.433 | 0.60% | `triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_max_mul_reciprocal_unsqueeze_1` |
| 13 | `vsa_layout` | 1600 | 695.324 | 434.577 | 0.58% | `_h3_vsa_o_bundle_local_gate_kernel` |
| 14 | `h3_pointwise` | 1600 | 655.698 | 409.811 | 0.55% | `_indexed_gate_rms_norm_scale_shift_kernel` |
| 15 | `gemm_unknown` | 1600 | 591.367 | 369.604 | 0.49% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)` |
| 16 | `nccl` | 800 | 573.890 | 717.362 | 0.48% | `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 17 | `copy_kernel` | 2432 | 531.501 | 218.545 | 0.44% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 18 | `vsa_layout` | 1600 | 471.512 | 294.695 | 0.39% | `_h3_vsa_o_bundle_kernel` |
| 19 | `h3_pointwise` | 1600 | 450.463 | 281.540 | 0.38% | `_indexed_gate_kernel` |
| 20 | `unknown` | 1600 | 395.214 | 247.009 | 0.33% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, float, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6…` |
| 21 | `gemm_unknown` | 1600 | 395.101 | 246.938 | 0.33% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(T1::Params)` |
| 22 | `unknown` | 1600 | 390.266 | 243.916 | 0.33% | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)32, (int)32, int, long, unsigned int>(at::cuda::detail::TensorInfo<T5, T7>, T7, T7, T7, at::cuda::detail::TensorInfo<T6, …` |
| 23 | `h3_pointwise` | 1600 | 355.827 | 222.392 | 0.30% | `_rms_norm_indexed_scale_shift_kernel` |
| 24 | `unknown` | 6400 | 327.039 | 51.100 | 0.27% | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int,…` |
| 25 | `triton_reduction` | 1600 | 262.814 | 164.259 | 0.22% | `triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_max_mul_reciprocal_unsqueeze_0` |
| 26 | `copy_kernel` | 1909 | 201.772 | 105.695 | 0.17% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 27 | `unknown` | 1600 | 181.661 | 113.538 | 0.15% | `void at::native::mbtopk::gatherTopK<float, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, unsigned int, T2, at::cuda::detail::TensorInfo<T1, T2>, …` |
| 28 | `unknown` | 6400 | 150.841 | 23.569 | 0.13% | `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(T1 *, short *, unsigned int *, unsigned int *, unsigned int, int, bool, unsigned int *, T2 *, unsigned …` |
| 29 | `unknown` | 1600 | 145.616 | 91.010 | 0.12% | `void <unnamed>::softmax_warp_forward<float, float, float, (int)11, (bool)0, (bool)0, (int)32>(T2 *, const T1 *, int, int, int, const bool *, int, bool)` |
| 30 | `unknown` | 1781 | 145.278 | 81.571 | 0.12% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned…` |
| 31 | `cudnn_other` | 232 | 107.284 | 462.431 | 0.09% | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_execute_kernel__5x_cudnn` |
| 32 | `cudnn_other` | 1408 | 96.165 | 68.299 | 0.08% | `void cudnn::engines_precompiled::nchwToNhwcKernel<__half, __half, float, (bool)0, (bool)1, (cudnnKernelDataType_t)0>(cudnn::engines_precompiled::nchw2nhwc_params_t<T3>, const T1 *…` |
| 33 | `unknown` | 6400 | 93.283 | 14.575 | 0.08% | `at::native::mbtopk::computeDigitCumSum(short *, unsigned int *, unsigned int)` |
| 34 | `unknown` | 3248 | 53.806 | 16.566 | 0.04% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| 35 | `cudnn_other` | 704 | 53.367 | 75.806 | 0.04% | `void cudnn::engines_precompiled::nhwcToNchwKernel<__half, __half, float, (bool)1, (bool)0, (cudnnKernelDataType_t)0>(cudnn::engines_precompiled::nhwc2nchw_params_t<T3>, const T1 *…` |
| 36 | `nccl` | 8 | 43.499 | 5437.385 | 0.04% | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 37 | `unknown` | 4800 | 40.870 | 8.515 | 0.03% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float…` |
| 38 | `cudnn_other` | 127 | 38.923 | 306.479 | 0.03% | `void cudnn::cnn::dgrad2d_grouped_direct_kernel<float, int, float, float, (bool)0, (bool)1, (int)1, (int)0>(cudnn::cnn::DgradGroupedDirectParams, const T1 *, const T1 *, T1 *, T4, …` |
| 39 | `unknown` | 660 | 38.782 | 58.761 | 0.03% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimi…` |
| 40 | `unknown` | 638 | 37.540 | 58.840 | 0.03% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::Half>>(at::TensorIteratorBase &, const T1 &)::[lambd…` |
| 41 | `cudnn_other` | 199 | 28.808 | 144.765 | 0.02% | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_128x64_32x6_nhwc_align8>(T1::Params)` |
| 42 | `nccl` | 72 | 28.019 | 389.151 | 0.02% | `ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 43 | `gemm_unknown` | 800 | 27.983 | 34.979 | 0.02% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8>(T1::Params)` |
| 44 | `gemm_unknown` | 43 | 27.847 | 647.614 | 0.02% | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_32x1_nn_align8>(T1::Params)` |
| 45 | `cudnn_other` | 220 | 26.269 | 119.403 | 0.02% | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x3_nhwc_align8>(T1::Params)` |
| 46 | `copy_kernel` | 14560 | 23.408 | 1.608 | 0.02% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 47 | `cudnn_other` | 22 | 23.087 | 1049.427 | 0.02% | `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x16_alignc4_execute_kernel__5x_cudnn` |
| 48 | `copy_kernel` | 1694 | 21.188 | 12.508 | 0.02% | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::…` |
| 49 | `gemm_unknown` | 32 | 20.338 | 635.566 | 0.02% | `void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(int, int, int, BatchedTensor, int, BatchedTensor, int, BatchedTensor, int,…` |
| 50 | `unknown` | 40 | 19.116 | 477.889 | 0.02% | `void at::native::indexFuncLargeIndex<c10::BFloat16, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1, at::native::<unnamed>::ReduceAdd>(at::cuda::detail::TensorInfo<T1, T3>, a…` |
| 51 | `gemm_unknown` | 32 | 18.727 | 585.213 | 0.02% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_tn_align1>(T1::Params)` |
| 52 | `copy_kernel` | 198 | 18.228 | 92.061 | 0.02% | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsigned int)2>, unsigned int, (int)2, (int)128, (int)1, (int)16, (int)8>(char *, at:…` |
| 53 | `unknown` | 66 | 17.469 | 264.676 | 0.01% | `void at::native::<unnamed>::upsample_nearest2d_out_frame<c10::Half, &at::native::nearest_neighbor_compute_source_index>(const T1 *, T1 *, unsigned long, unsigned long, unsigned lo…` |
| 54 | `unknown` | 254 | 16.348 | 64.363 | 0.01% | `void at::native::<unnamed>::replication_pad_forward_kernel1d<float>(torch::headeronly::detail::GenericPackedTensorAccessor<torch::headeronly::detail::TensorAccessor<c10::ArrayRef<…` |
| 55 | `copy_kernel` | 2884 | 16.176 | 5.609 | 0.01% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned …` |
| 56 | `unknown` | 127 | 15.709 | 123.694 | 0.01% | `void at::native::<unnamed>::conv_depthwise2d_forward_kernel_generic<float, int>(torch::headeronly::detail::GenericPackedTensorAccessor<torch::headeronly::detail::TensorAccessor<c1…` |
| 57 | `copy_kernel` | 4800 | 15.579 | 3.246 | 0.01% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 58 | `unknown` | 3200 | 15.498 | 4.843 | 0.01% | `void at_cuda_detail::cub::detail::scan_by_key::DeviceScanByKeyKernel<at_cuda_detail::cub::detail::scan_by_key::policy_hub<thrust::_V_300200_SM_750_800_860_900_1000_1200::transform…` |
| 59 | `gemm_unknown` | 64 | 15.225 | 237.890 | 0.01% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>(T1::Params)` |
| 60 | `unknown` | 124 | 14.725 | 118.748 | 0.01% | `fused_add_reciprocal_mul_sin_pow_mul_add` |
| 61 | `gemm_unknown` | 28 | 13.139 | 469.248 | 0.01% | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(T1::Params)` |
| 62 | `unknown` | 342 | 10.615 | 31.038 | 0.01% | `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, long, long, bool)` |
| 63 | `unknown` | 198 | 10.505 | 53.056 | 0.01% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::Half>, std::array<char *, (unsigned long)3>>(int, T2, T3)` |
| 64 | `unknown` | 1264 | 10.201 | 8.070 | 0.01% | `void vllm::rms_norm_kernel<c10::BFloat16, (int)8, (int)2, (bool)1>(T1 *, const T1 *, long, long, long, long, long, const T1 *, long, float, int, int)` |
| 65 | `copy_kernel` | 1616 | 8.121 | 5.026 | 0.01% | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::…` |
| 66 | `unknown` | 4800 | 7.733 | 1.611 | 0.01% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimi…` |
| 67 | `h3_pointwise` | 32 | 7.400 | 231.244 | 0.01% | `_indexed_scale_shift_kernel` |
| 68 | `unknown` | 1063 | 7.350 | 6.914 | 0.01% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFuncto…` |
| 69 | `unknown` | 1600 | 7.124 | 4.453 | 0.01% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 70 | `unknown` | 22 | 6.919 | 314.497 | 0.01% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::round_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 2…` |
| 71 | `unknown` | 23 | 6.886 | 299.395 | 0.01% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimi…` |
| 72 | `unknown` | 205 | 5.781 | 28.200 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` |
| 73 | `gemm_unknown` | 832 | 5.596 | 6.725 | 0.00% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 74 | `unknown` | 448 | 5.429 | 12.119 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 75 | `copy_kernel` | 22 | 5.210 | 236.809 | 0.00% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)…` |
| 76 | `copy_kernel` | 22 | 5.054 | 229.745 | 0.00% | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 1)]::…` |
| 77 | `gemm_unknown` | 400 | 5.010 | 12.524 | 0.00% | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8>(T1::Params)` |
| 78 | `attention_other` | 400 | 4.841 | 12.104 | 0.00% | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)128, (int)128, (i…` |
| 79 | `gemm_unknown` | 28 | 4.219 | 150.686 | 0.00% | `void implicit_convolve_sgemm<float, float, (int)1024, (int)5, (int)5, (int)3, (int)3, (int)3, (int)1, (bool)0, (bool)0, (bool)1>(int, int, int, const T1 *, int, T2 *, const T1 *, …` |
| 80 | `cudnn_other` | 12 | 3.681 | 306.755 | 0.00% | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4>(T1::Params)` |
| 81 | `unknown` | 448 | 3.624 | 8.089 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, bool, at::native::<unnamed>::CompareEqFunctor<float>>, std::array<char *, (unsigned …` |
| 82 | `cudnn_other` | 24 | 3.622 | 150.928 | 0.00% | `sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel__5x_cudnn` |
| 83 | `unknown` | 400 | 3.563 | 8.908 | 0.00% | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, (int)4, (int)4>>(T3)` |
| 84 | `unknown` | 139 | 3.058 | 22.003 | 0.00% | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(in…` |
| 85 | `cudnn_other` | 24 | 2.974 | 123.927 | 0.00% | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_64x64_32x5_nhwc_align4>(T1::Params)` |
| 86 | `unknown` | 3200 | 2.964 | 0.926 | 0.00% | `void at_cuda_detail::cub::detail::scan_by_key::DeviceScanByKeyInitKernel<at_cuda_detail::cub::ReduceByKeyScanTileState<unsigned int, int, (bool)1>, thrust::_V_300200_SM_750_800_86…` |
| 87 | `cudnn_other` | 7 | 2.801 | 400.082 | 0.00% | `sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel__5x_cudnn` |
| 88 | `unknown` | 1600 | 2.743 | 1.714 | 0.00% | `void at::native::mbtopk::fill<unsigned int, unsigned int>(T1 *, T1, T2)` |
| 89 | `unknown` | 1600 | 2.447 | 1.529 | 0.00% | `void at::native::mbtopk::computeBlockwiseKthCounts<unsigned int>(T1 *, short *, unsigned int, unsigned int, unsigned int *)` |
| 90 | `gemm_unknown` | 1200 | 2.440 | 2.033 | 0.00% | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, (bool)1, (bool)…` |
| 91 | `gemm_unknown` | 64 | 2.423 | 37.855 | 0.00% | `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)2, (int)2, (bool)1, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, …` |
| 92 | `unknown` | 128 | 2.408 | 18.814 | 0.00% | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIter…` |
| 93 | `unknown` | 416 | 2.328 | 5.597 | 0.00% | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<bool, at::native::func_wrapper_t<bool, at::native::and_kernel_cuda(at::TensorIterator &)::[lambda() (instance…` |
| 94 | `unknown` | 32 | 1.953 | 61.040 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<c10::BFloat16>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| 95 | `copy_kernel` | 800 | 1.953 | 2.441 | 0.00% | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)2>, unsigned int, (int)4, (int)64, (int)64>(T1 *, at::native::<unnamed>::CatArrInpu…` |
| 96 | `unknown` | 1200 | 1.920 | 1.600 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(int, T2, T3)` |
| 97 | `unknown` | 448 | 1.889 | 4.216 | 0.00% | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, bool, at::native::<unnamed>::CompareEqFunctor<float>>, std::array<char *, (unsigned …` |
| 98 | `cudnn_other` | 184 | 1.841 | 10.008 | 0.00% | `void cudnn::engines_precompiled::nchwToNhwcKernel<float, float, float, (bool)0, (bool)1, (cudnnKernelDataType_t)2>(cudnn::engines_precompiled::nchw2nhwc_params_t<T3>, const T1 *, …` |
| 99 | `unknown` | 800 | 1.803 | 2.254 | 0.00% | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::bi…` |
| 100 | `gemm_unknown` | 23 | 1.729 | 75.161 | 0.00% | `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_nn_align8>(T1::Params)` |

## Top OS runtime calls

OSRT time is summed across threads and is never assigned to RDMA without explicit naming evidence.

| Call | Count | Summed thread-ms |
|---|---:|---:|
| `pthread_cond_timedwait` | 27924 | 3078193.500 |
| `poll` | 6703 | 1912005.055 |
| `epoll_wait` | 164051 | 1514222.883 |
| `sem_clockwait` | 285 | 345253.206 |
| `pthread_cond_wait` | 43003 | 148740.327 |
| `epoll_pwait` | 23853 | 50104.486 |
| `sem_wait` | 14 | 25562.464 |
| `ioctl` | 21043 | 10191.573 |
| `accept` | 2128016 | 4974.736 |
| `open64` | 27443 | 3124.348 |
| `close` | 23085 | 2247.144 |
| `read` | 43153 | 1553.393 |
| `clock_nanosleep` | 28 | 1401.617 |
| `fopen` | 4187 | 817.649 |
| `munmap` | 151208 | 573.357 |
| `recv` | 8799 | 562.355 |
| `send` | 10563 | 153.256 |
| `socket` | 1212 | 132.287 |
| `recvmsg` | 1169 | 105.485 |
| `fallocate` | 16 | 97.006 |

## Warnings and attribution limits

- No usable NVTX range was selected; results cover the complete CUDA GPU trace and may include startup.
- NCCL logger rank 7 disagrees with communicator rank 2 for PID 731744; ignored that line.
- 3.65% of summed kernel GPU-time is unknown; it was not force-fit into a named H3 stage.
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
- Output lines: 322
- Output SHA256: `975304402f4e605814a7abc4c9cbb3cc4708faeb7c6ab24d7d96b7e793096b74`
- Reports detected: `{"cuda_api": true, "kernels": true, "memory": true, "nvtx": true}`
