# r2 V-split results

Normal and Nsight schedule checks passed. Each trace contains 400 verified rank/block instances in step 3 of 4. Reproducibility across repeated/reversed cohorts is not established.

| Case | Formal E2E samples (s) | Mean (s) | Median (s) |
|---|---|---:|---:|
| Baseline: fused QKV | 24.109, 23.998, 23.979, 23.992, 24.056 | 24.0268 | 23.998 |
| Candidate: QK + delayed V | 23.497, 23.482, 23.586, 23.849, 23.913 | 23.6654 | 23.586 |

Observed mean reduction: 0.3614 s (1.504%). Five baseline requests preceded five candidate requests; this was not an alternating A/B. Each cohort had one separate excluded warmup.

The existing r1 actual-weight audit passed 16,000 byte comparisons and was reused only after exact source and original compiler-selection identity checks. All ten r2 formal MP4 files match `aa55fa19de94066e999f04275362f5477b3d74eb925d45fece8918d1cc8f596a`. The candidate emitted all 400 expected split activation records. Both normal controllers and the numerical/normal analyzer passed.

The candidate rises from 23.497/23.482 s to 23.849/23.913 s. Engine execution also rises (23.401/23.409 s to 23.776/23.835 s); HTTP residual stays 0.033–0.037 s. The normal data alone do not identify the cause.

Clocks use the node default and are unlocked. The independent GPU balance gate was disabled. The idle health check and request-level foreign GPU-process monitoring passed; no foreign workload was terminated. The r1 contaminated candidate is excluded.

Evidence: [numerical and normal qualification](r2-qualification-results.json), [normal timing figure](r2-figures/normal-e2e.png), [source dependency review](r2-source-dependencies.md).

## Completed Nsight review

Both baseline and candidate profile outputs match the formal MP4 hash. All 400 rank/block structural/dependency checks pass in each trace. GPU7 V averages 4.5626 ms, with 2.7565 ms (60.4%) inside K interior and 1.7297 ms of GPU activity after K close. This is not fully hidden. V copy is now a 171,917,312-byte D2D memcpy (400/400 blocks), about 0.3018 ms on GPU7; post-receive V tile packing remains.

QK/V use the original cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8 specialization with gridY 28/14 versus fused QKV gridY 42. GPU7 QK+V kernel time is only 0.0625 ms greater than fused QKV, but V readiness extends past K close. The observed normal E2E reduction remains 1.504%, not a multiplication of per-block profile deltas.

[GPU7 timeline](r2-figures/gpu7-block0-qkv-zoom.png), [full timeline](r2-figures/gpu7-block0-full.png), [overlap by rank](r2-figures/v-overlap-by-rank.png), [candidate profile JSON](profile-r2-candidate.json), [baseline profile JSON](profile-r2-baseline.json).

User requested a further schedule rearrangement and high SM utilization. The independent v11 experiment is tracked in ../lossless-v11-full-overlap/WORK_STATE.md; r2 model and benchmark evidence remain unchanged.
