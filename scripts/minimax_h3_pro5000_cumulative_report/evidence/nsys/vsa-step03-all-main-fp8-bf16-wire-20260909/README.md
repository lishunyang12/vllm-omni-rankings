# MiniMax-H3 VSA step-3 Nsight Systems trace

This folder contains the complete raw Nsight Systems report and its complete SQLite export for one MiniMax-H3 DiT forward on eight RTX PRO 5000 Blackwell GPUs. The selected window is denoise step 3 of FastH3's four-step schedule.

The accepted precision lane is **all-main FP8 linear compute with BF16 QKV transport**. The rejected E4M3 QKV-wire optimization is not enabled. Runtime validation found eight BF16 Q/K producer-direct markers, eight BF16 reverse-O producer-direct markers, and zero E4M3-wire markers.

## Download and open

- [`minimax-h3-vsa-all-main-fp8-bf16-wire-sp8-step03.nsys-rep`](minimax-h3-vsa-all-main-fp8-bf16-wire-sp8-step03.nsys-rep) — complete 11.37 MB Nsight report; open with Nsight Systems 2025.3.2 or newer.
- [`minimax-h3-vsa-all-main-fp8-bf16-wire-sp8-step03.sqlite`](minimax-h3-vsa-all-main-fp8-bf16-wire-sp8-step03.sqlite) — complete 31.38 MB SQLite export.
- [`analysis-full.md`](analysis-full.md) — full cross-rank analyzer output.
- [`analysis-full.json`](analysis-full.json) — machine-readable cross-rank analysis.
- [`vsa-nvtx-all-ranks.txt`](vsa-nvtx-all-ranks.txt) — GPU projections for every nested VSA phase across all ranks.
- [`nsys-stats-step-filtered.txt`](nsys-stats-step-filtered.txt) — standard Nsight reports filtered to the rank-0 step range.
- [`capture-contract.txt`](capture-contract.txt) and [`stack-validation.txt`](stack-validation.txt) — exact stack contract and fail-closed validation.
- [`capture-runner.sh`](capture-runner.sh) — runner used for the capture.

Verify the package with:

```bash
sha256sum -c SHA256SUMS.txt
```

## What the trace says about VSA

The profiled step spans 3.730 seconds. This is a diagnostic trace with substantial instrumentation overhead and is **not** the serving E2E result.

Across the eight ranks, the fine CuTeDSL block-sparse kernel runs 400 times: 50 H3 layers × 8 ranks. It accounts for 9,118.3 summed GPU-ms, or 37.40% of summed kernel GPU time. When every explicitly marked VSA phase is included—fine attention, pooling, coarse QK/PV, softmax, top-k routing, tile packing, untile, and reverse-O bundling—the VSA stack averages 1,351.4 ms per rank. Fine attention is 84.34% of that VSA-stage sum; everything around it is 15.66%.

| VSA phase | Eight-rank GPU-ms | Share of VSA stage sum |
|---|---:|---:|
| Fine block-sparse attention | 9,118.3 | 84.34% |
| Top-k + direct q2k route | 533.1 | 4.93% |
| Tile pack | 384.6 | 3.56% |
| Coarse QK score GEMM | 187.8 | 1.74% |
| Coarse Q/K pooling | 146.0 | 1.35% |
| Reverse-O bundle | 118.5 | 1.10% |
| Untile | 117.5 | 1.09% |
| Coarse PV GEMM | 98.1 | 0.91% |
| Coarse V pooling | 70.8 | 0.66% |
| Coarse softmax | 36.8 | 0.34% |

Two observations set the next optimization order:

1. The #4944 fine kernel remains the largest single kernel family, so block-size/work partitioning and SM120 kernel work still matter. However, the non-fine VSA path is already 15.66% of VSA-stage GPU time; optimizing only the fine kernel has a hard local ceiling.
2. Rank 7 is the critical imbalance: its fine-VSA sum is 1,364.6 ms versus the cross-rank median of 1,121.2 ms. It also constrains two chronological barrier slots. B0 and B6 contribute 516.1 ms of the 521.3 ms reconstructed arrival-skew sum. The same rank is also slow in GEMM, so this trace supports a rank/device balance investigation before attributing all skew to the VSA algorithm.

Barrier residency is synchronization exposure, not network-copy duration. Arrival-skew is already inside the profiled step and is not guaranteed removable latency. The trace has one expected boundary-partial group because NVTX capture starts while the first group is in flight; the following 399 groups contain all eight devices and reconstruct 50 complete layer cycles at eight barrier slots per layer.

## Useful SQLite queries

The SQLite file is the direct `nsys export --type=sqlite` output. For example, list the dominant kernel families:

```sql
SELECT s.value AS kernel,
       COUNT(*) AS calls,
       ROUND(SUM(k.end-k.start)/1e6, 3) AS gpu_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.shortName
GROUP BY s.value
ORDER BY gpu_ms DESC
LIMIT 30;
```

List nested VSA ranges:

```sql
SELECT COALESCE(n.text, s.value) AS range_name,
       COUNT(*) AS instances,
       ROUND(SUM(n.end-n.start)/1e6, 3) AS cpu_range_ms
FROM NVTX_EVENTS AS n
LEFT JOIN StringIds AS s ON s.id = n.textId
WHERE COALESCE(n.text, s.value) LIKE 'vsa.%'
  AND n.end IS NOT NULL
GROUP BY range_name
ORDER BY range_name;
```

NVTX CPU range duration is launch-side duration. For asynchronous GPU work, use `nvtx_gpu_proj_sum` or correlate the range to CUDA kernels; do not treat the CPU range duration as kernel duration.

## Reproduction contract

- MiniMax-H3 FL2VA, 1,280×720 requested / 1,280×704 effective, 362 frames at 24 fps, seed 1101.
- FastH3 VSA/Data-Free adapter, four transformer forwards.
- vLLM-Omni TP1/SP8 on eight SM120 GPUs.
- FlashInfer PR #4944 CuTeDSL VSA, tile 64, top-k 162.
- All 250 main-block linear modules use FP8 compute; QKV projection output and QKV wire remain BF16.
- FlashInfer PCIe/RDMA Q/K producer-direct and reverse-O producer-direct; direct q2k and O-bundle kmax 279.
- Node-default dynamic clocks. The clocks were not locked, matching the accepted benchmark policy.

See `capture-contract.txt` for hashes, exact configuration, and the adapter-bound AdaLN contract.
