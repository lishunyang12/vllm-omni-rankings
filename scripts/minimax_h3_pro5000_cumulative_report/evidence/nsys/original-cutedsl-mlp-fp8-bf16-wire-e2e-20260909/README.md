# MiniMax-H3 full-request Nsight trace

This folder publishes the complete marker-free end-to-end Nsight Systems capture for one MiniMax-H3 request on eight RTX PRO 5000 Blackwell GPUs.

The requested lane is intentionally conservative:

- FastH3 VSA/Data-Free, exactly four transformer forwards.
- FlashInfer PR #4259 original SM120 CuTeDSL VSA kernel, tile 64, top-k 162.
- FP8 only for the 50 FC1 and 50 FC2 modules in each H3 model replica.
- QKV projection, QKV all-to-all transport, output projection, and reverse-O transport remain BF16.
- FlashInfer PR #4876 PCIe/RDMA producer-direct Q/K and reverse-O transport.
- No skip-softmax path and no E4M3 QKV-wire path.
- TAEH3 FP16 video decode, cross-rank audio decode, GPU uint8 conversion, pinned D2H, libx264, and MP4 mux.
- No generation warmup and no custom H3 step, decode, or VSA capture markers.

## Result

The request returned a complete 362-frame, 15.083333-second MP4 in **25.568 seconds**: RTF **1.695**. This is a first-request profiling artifact, not the accepted warm-serving latency. It includes lazy regional `torch.compile`, original CuTeDSL specialization/JIT, and profiler overhead. Collection began only after `/health` became ready and stopped after the MP4 response completed, so model loading is outside the report.

The CUDA trace window is 25.258 seconds. Across all eight GPUs it contains 1,600 fine-VSA calls, exactly `4 forwards × 50 layers × 8 ranks`. The 12,800 FlashInfer barriers reconstruct as 1,600 complete eight-rank groups and 200 layer cycles, with no partial group.

GPU-time in the table is summed across devices and streams. It is useful for composition, but must not be added to wall latency.

| Kernel category | Calls | Summed GPU time | Share |
|---|---:|---:|---:|
| Recognizable GEMM | 15,759 | 54,459.8 ms | 45.43% |
| Original CuTeDSL fine VSA | 1,600 | 35,234.3 ms | 29.39% |
| FlashInfer RDMA barrier residency | 12,800 | 13,939.9 ms | 11.63% |
| NCCL | 1,224 | 4,057.3 ms | 3.38% |
| VSA layout | 9,600 | 3,164.2 ms | 2.64% |
| Fused H3 pointwise | 8,032 | 2,407.5 ms | 2.01% |

The original VSA kernel sums to a median 4,339.6 ms per rank over all four forwards. Rank 7 is the outlier at 5,280.1 ms, 21.7% above the rank median. Barrier arrival-skew sums to 2,286.6 ms, or 9.05% of the selected wall window; B0 and B6 account for 99.3% of that exposure. Late-path evidence is split almost evenly between VSA and GEMM, so the trace does not support treating communication alone as the cause.

Rank 0 copies 983.85 MB device-to-host in 18.40 GPU-ms. This is the post-conversion uint8 payload; the former multi-gigabyte FP32 frame transfer is absent.

## Complete artifacts

- [`minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.nsys-rep`](minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.nsys-rep) — complete 54.75 MB Nsight Systems report; open with Nsight Systems 2025.3.2 or newer.
- [`minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.sqlite.zst`](minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.sqlite.zst) — losslessly compressed complete 269.71 MB SQLite export, 55.29 MB on GitHub.
- [`generated-output.mp4`](generated-output.mp4) — complete returned MP4.
- [`analysis-full.md`](analysis-full.md) and [`analysis-full.json`](analysis-full.json) — cross-rank analysis.
- [`nsys-stats.txt`](nsys-stats.txt) and [`nsys-stats-window.txt`](nsys-stats-window.txt) — canonical Nsight reports.
- [`capture-contract.txt`](capture-contract.txt), [`stack-validation.txt`](stack-validation.txt), and [`profile-summary.json`](profile-summary.json) — immutable configuration and validation evidence.
- [`server.log`](server.log), [`runtime-config.txt`](runtime-config.txt), and [`capture-runner.log`](capture-runner.log) — runtime evidence.
- [`capture-runner.sh`](capture-runner.sh) — exact capture runner.

Restore and validate the complete SQLite database:

```bash
zstd -d minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.sqlite.zst
printf '%s  %s\n' \
  '53e017aef441ea5e8e5fed39030d55faa92072f7ef653b62cc84e4883d09f144' \
  'minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.sqlite' | sha256sum -c -
```

Or run [`restore-sqlite.sh`](restore-sqlite.sh). Verify every committed artifact with:

```bash
sha256sum -c SHA256SUMS.txt
```

## Precision and approximation gates

All eight runtime audit records report:

```text
target_modules=100 fc1=50 fc2=50
weight_quant=static_per_tensor_symmetric
activation_quant=dynamic_per_token_symmetric
input_dtype=bfloat16 output_dtype=bfloat16
unexpected_fp8_modules=0 fallback_count=0 status=ok
```

The server log contains zero skip-softmax markers, zero E4M3-QKV markers, and zero custom `MiniMaxH3NVTX`/H3-VSA-domain markers. The three SM120 CuTeDSL source blobs recorded in `capture-contract.txt` are object-identical to PR #4259 commit `750dbfd5`; PR #4876 is used only for the RDMA transport layer.

TAEH3 remains a lossy decoder substitution. MLP-only FP8 is also a numerical approximation. This package isolates those choices from the rejected E4M3 QKV-wire and skip-softmax approximations; it is performance evidence, not a claim of BF16/full-VAE output equivalence.

## Reading the communication numbers

`flashinfer_rdma_barrier` duration is GPU synchronization residency. It includes waiting for the latest rank and is not a measurement of bytes divided by link bandwidth. The reconstructed arrival skew is already inside request wall time and is not automatically removable. Use the per-slot and late-path tables in `analysis-full.md` to distinguish producer imbalance from transport overhead.
