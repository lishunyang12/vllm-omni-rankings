# Additional lossless Ulysses candidates

Scope: H3 teahouse15s, local11992 rows, TP1/SP8, BF16 wire, 56 heads × 128,
physical rank order0,4,1,5,2,6,3,7. Eight-GPU synthetic operator experiments use
real RDMA; full-model conclusions require separate normal requests and identical
outputs with fixed compiler choices.

| Candidate | Paired micro mean, ms | Numerical checks | Decision |
|---|---:|---|---|
| Original Q/K preparation during V → prepare Q during K |10.683 →10.746|All packed Q/K/V, pools, scores, ordered sparse metadata bitwise on8ranks ×3patterns|No additional speed demonstrated|
| v1 O4 chunks → coarse tail once,4 chunks |6.579 →6.551|Gate, O projection and immutable received tail bitwise on8ranks ×3patterns|10,160,640 fewer remote bytes/rank/layer; timing improvement small versus variation|
| v1 O4 chunks → coarse tail once,8 chunks |Not timed after rejection|Gate exact, O projection differed onall24checks; example max absolute error0.03125, relativeL2~0.00266|Reject as bitwise lossless candidate|
| Fused QKV → QK then V during Q/K RDMA |24.199 →20.949|Q/K/V outputs bitwise on8ranks ×3patterns|Promising synthetic result; changes GEMM N and is not integrated into full-model candidate|
| Gate GEMM before QKV RDMA → gate during RDMA |14.156 →10.959|Unchanged BF16 GEMM shape, same operands; output bitwise on8ranks ×3patterns|Selected for separate full-model A/B|

The gate experiment excludes QKV projection and QK norm/RoPE from its timing.
The projection experiment includes staging but excludes real QK norm/RoPE and
attention. Reverse experiments include the real BF16 gate arithmetic and an
H3-shaped synthetic O projection. These timings cannot be added to infer an
end-to-end speedup.

Full-model implementation uses the existing side stream for the original gate
projector. The communication stream continues Q/K/V exchanges. Before VSA
consumes prepared Q/K, the existing join waits for side-stream work; reverse gate
and O projection also use this side stream. The caller joins communication and
side streams before output reuse, including exceptional exits. Input lifetime is
recorded on the consuming side stream. The wire payload and every original RDMA
completion/visibility check remain in place.

Hardware inspection found PCIe PXB/NODE/SYS GPU paths and no NVLink. A rank
permutation alone does not remove any GPU pair from an all-to-all. Symmetric
registration and GPUDirect do not eliminate PCIe payload. Removing a barrier
without proving data visibility does not implement a lossless optimization.

The first new-cache control changed its output with gate overlap OFF. Six selected
Inductor RMSNorm reduction configurations differed. A subsequent control using a
copy of the original compiler cache restored the exact original BF16 MP4 SHA256.
Formal follow-on A/B pins and checks those existing selections; the cold-cache
control remains separate diagnostic evidence.

Eight selected mlx5 ports report400Gb/s (nominal50GB/s per port). The unchanged
sysfs selection assigns a distinct nearest port to each GPU, including
GPU7→mlx5_5. Baseline GPU7's logical QKV send rate47.444GB/s is94.9% of that nominal
rate; this ratio includes protocol/barrier interior overhead and is not a PCIe
counter average. It supports prioritizing overlap and redundant bytes over an
assumption of large unused single-port bandwidth. See transport-capacity.json.

Before forward Ulysses each rank owns11992tokens ×56heads ×128channels.
Afterward each rank owns95936alignedtokens ×7heads ×128channels. Per tensor,
each GPU sends7peer shards ×11992×7×128×2bytes =150,427,648remote bytes;
Q+K+V total451,282,944. Without changing precision, ownership or the attention
algorithm, those remote values are required. The local self-shard does not pass
through RDMA. The O return restores the local token shard and full56heads before
its projection to hidden5376.

The completed full-model gate A/B uses three normal requests per schedule:
25.117667 s for fresh v1 and24.340333 s for v2 (-3.095%). All six MP4s, plus
three stage-run MP4s and the Nsight-run MP4, match the original BF16 SHA256.
GPU7 step3/layer0 spans29.115437 ms versus25.997027 ms from QKV GEMM start to
the end of the last forward RDMA window. Gate/window overlap is4.473024 ms.
These windows run from opening-barrier end to closing-barrier start and include
completion/scheduling waits; they do not measure continuous PCIe payload activity.
The raw samples show both SM and PCIe active during part of the moved gate GEMM.

QKV splitting remains a candidate for a separate full-model A/B. It competes
with the moved gate for compute resources and the same overlap window, so its
synthetic gain cannot be added to the gate gain. Reducing repeated coarse-tail
bytes has a verified payload benefit but no convincing timing gain in this run.
