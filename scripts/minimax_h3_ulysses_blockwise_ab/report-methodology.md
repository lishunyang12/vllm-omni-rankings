# H3 Ulysses overlap × FP8 blockwise: measurement contract

The four full-model cases are `bf16-base`, `bf16-overlap`, `fp8-base`, and
`fp8-overlap`. All use the same Cake PR4951 fine attention, top-k 162,
BF16 QKV/O transport, BF16 gate, full H3 VAE, teahouse prompt, seed 1101,
four denoising steps and 50 main transformer blocks. TP1/SP8 maps ranks
to physical GPUs `[0,4,1,5,2,6,3,7]`. The request asks for 1280×720;
the model emits 362 frames at 1280×704 and 24 fps (15.083333 seconds).
“BF16” refers to the main-block linears; Cake fine attention retains its
common Q/K INT8 and P/V FP8 implementation in every case.

## Precision

Only `blocks.N.attn.qkv_proj`, `blocks.N.attn.out_proj`,
`blocks.N.mlp.fc1` and `blocks.N.mlp.fc2`, N=0…49, change precision.
There are 200 affected linears per rank. Fused FastH3 BF16 weights are
quantized after loading, using static FP32 scales per 128×128 weight block.
Activations use dynamic FP8 E4M3 scales for each row and each 128 elements
along K. Inputs and outputs are BF16. The runtime audit validates all eight
ranks, exact module membership, scale shape/dtype, quantization keys and
`CutlassFp8BlockScaledMMKernel`; silent fallback is rejected.
`VLLM_USE_DEEP_GEMM=0` pins this kernel choice in all four cases.

The new precision flag is `VLLM_OMNI_H3_EXPERIMENT_BLOCKWISE=1`.
Legacy `VLLM_OMNI_MINIMAX_H3_FP8_SCOPE=none` disables the older per-tensor
FP8 lane; it does not describe this new blockwise experiment. The unique
runtime mode is `qkv-o-mlp-fp8-e4m3fn-a1x128-w128x128-online-v1`.

## Communication schedule

The existing FlashInfer backend uses registered GPU landing/output buffers,
mlx5 interleaved MKeys for scatter-heads source or gather-heads destination
layout, and host-posted RDMA write-with-immediate work requests. The NIC
performs the remote payload movement. Host CQ polling verifies both send
and receive completion tags, then `FlushHybridWrites` enforces the required
GPU visibility before the closing barrier. `UlyssesPcieBarrier` itself is
a single 32-thread block publishing and checking peer epochs, not the bulk
payload-copy kernel. This is a host-driven GPUDirect RDMA path, not a new
GPU-initiated network transport introduced by this patch. See the recorded
FlashInfer snapshot's `csrc/ulysses_pcie.cu`, `csrc/ulysses_pcie_transport.cuh`
and `include/flashinfer/comm/ulysses_pcie.cuh` for these unchanged mechanics.

`VLLM_OMNI_H3_EXPERIMENT_OVERLAP=1` moves Q/K tile packing, pooling,
coarse-score calculation and top-k metadata construction onto a side CUDA
stream during the V exchange. The reverse path splits O into four chunks
of 2,998 local rows. Each retains a 270-row coarse tail; local gate and
the actual O projector run while later chunks transfer. The native RDMA
posting, polling, visibility and barrier protocol is unchanged.
A high-priority communication stream is necessary to avoid the next
opening barrier waiting behind the previous O GEMM.

Symmetric memory names and registers peer-visible GPU buffers; it does not
remove the GPU-to-NIC PCIe path. NVIDIA describes this direct peer-device path
as using standard PCI Express features in its [GPUDirect RDMA overview](https://docs.nvidia.com/cuda/gpudirect-rdma/).
The existing route uses RDMA for all seven
remote peers and a copy-engine self-copy. Each Q/K/V exchange sends
150,427,648 logical bytes to remote peers per rank. Reverse O sends
153,814,528 bytes in the original schedule and 163,975,168 in the four-chunk
schedule (+6.606%). Receive volume is equal and is reported separately.
The optimization aims to reduce exposed communication time by overlapping
independent work; it does not eliminate these payloads.

The implementation is intentionally bound to the recorded prompt geometry:
11,992 local rows, 95,936 aligned global rows, 95,924 valid rows, 105,472
tiled rows, prefix `(558,1206)`, video `(107,22,40)`, seven local heads,
head dimension 128, top-k 162, and an eight-rank trusted owner plan.
Unsupported geometries fail explicitly.

## Correctness and performance evidence

- Single-GPU integration checks call the actual `forward_cuda`, Cake kernel,
  gate and BF16/FP8 O projectors with simulated identical-rank exchanges.
  Three changing random inputs for each precision pass finite and bitwise
  comparison under ordinary asynchronous CUDA execution. They do not prove
  SP8 or full-model quality by themselves.
- Independent SP8 microbenchmarks use real RDMA and H3 shapes with synthetic
  tensors. Eight paired trials report max-rank wall time without Nsight.
  These results explain scheduling mechanics and are never substituted for
  full-model speedup.
- Full-model normal runs use one warmup and three measured requests per case.
  End-to-end wall time comes from these runs with profiling disabled.
- Separate stage runs use one warmup and three measured requests. The stage
  profiler synchronizes CUDA; its DiT/Encoder/Decode values therefore remain
  distinct from normal E2E samples. Decode includes video and audio VAE.
- Separate Nsight runs capture one warmed request per case, all eight GPUs,
  NVTX, CUDA activity and GPU Metrics. Profiled timings are diagnostic and
  are not substituted for unprofiled E2E.
- The dedicated `minimax_h3.denoise.step_03` range selects the third step,
  containing 50 fine attention calls per GPU. The complete request contains
  200 calls per GPU. Mutually exclusive phase wall time includes concurrency
  and idle intervals. The generic GEMM bucket also includes coarse-attention
  matrix products whose roles are not encoded in kernel names; it is not a
  per-projection breakdown. Summed kernel time is not request wall time.
  The `idle` bucket means no traced CUDA kernel, memcpy or memset; RDMA NIC
  traffic may still be active there. It does not mean zero PCIe activity or
  constitute an SM-utilization measurement. Raw GPU Metrics provide that
  separate evidence in the detailed timeline.
- The detailed GPU7 plot selects the first layer in that NVTX step. Q/K/V
  windows use the last six preceding barrier kernels; reverse O uses the
  next two or eight. Optimized endpoints come from NVTX/API/GPU correlation,
  including the final output-assembly copy. Compute overlap counts kernels
  only; CUDA copies are shown separately.
  The original endpoint is inferred as the first GEMM after the final O
  barrier, following model source order. This endpoint inference is labeled.
- SM active and PCIe RX/TX values are raw individual GPU Metrics samples,
  targeted at 1000 Hz. They are not interval averages. Effective send rates
  divide logical remote payload by the interval between barriers, which also
  includes protocol overhead; they do not establish a hardware link ceiling.
- GPU frequency is left dynamic. The cases run sequentially; three repeats
  are insufficient to establish small differences. Other users' work is
  neither terminated nor included intentionally in a shared-GPU comparison.
  Before each measured request the runner checks for GPU processes outside
  the server process tree, then polls for foreign processes during the
  request (one-second sleep between polls). A nonempty foreign-process log
  rejects the sample. The final source/hardware audit also requires all
  measured-request exclusivity preflights and empty monitor logs.

## Media comparison

All cross-case quality comparisons use the first measured normal request,
all 362 aligned decoded frames and full stereo audio. RGB/YUV PSNR and SSIM
measure numerical differences, not subjective quality. Four additional
same-configuration comparisons between formal repeats 1 and 2 establish
repeat variation, including an audio PCM comparison. A changed MP4 SHA does not
imply changed video pixels. Actual contact sheets sample 1, 4.5, 9 and 14 s;
the page also offers synchronized full videos. This is a single prompt/seed
experiment, not a general quality guarantee.
The legacy media utility has no scheduling-only axis in its CLI schema;
its raw reports use `precision` with exact reference/candidate contracts.
The derived quality summary records the actual `experiment_axis` separately
as schedule, precision, or precision_and_schedule. This metadata distinction
does not change the decoded-frame or audio calculations.

## Shared stability fix and scope limit

The Cake native binding initializes TMA descriptors using `cuMemcpyHtoD`.
Reusing ordinary allocator storage from another nonblocking stream exposed
an initialization race in asynchronous integration. Dedicated descriptor
pool storage, retained by binding, passed the asynchronous checks. This fix
is applied identically to all four cases; no host synchronization is added
to the hot path. Binding caches retain descriptors for the worker lifetime;
long-running service memory bounds are outside this experiment's validation.

## Reproduction

The isolated `vllm-omni` worktree is based on `d9a610e`. The unchanged
FlashInfer snapshot is `f841539c496788eb0221dec2736d0886d1c90ebd`, with the
existing Cake ragged-block fix from the source experiment. Each run records
its command, benchmark-script hash, source diff and untracked source hashes.
The candidate patch covers the model integration. Run scripts retain exact
paths to the installed environment, checkpoints, prompt geometry and caches;
these paths must be adjusted on another machine. Checkpoints are not bundled.

```sh
python run_full_campaign.py
python postprocess_full_campaign.py
# Inspect actual videos/contact sheets and write visual-assessment.txt.
python package_four_way.py
node check_page.cjs http://127.0.0.1:PORT/ /tmp/h3-page-check
```

Individual cases use `bash run_model_case.sh MODE normal|stages|nsys`.
The campaign waits for eight free GPUs before each case and preserves
the benchmark's route and GPU health validation. The optional independent
GPU-balance benchmark is disabled, as recorded in every run.

## Compiler selections and bitwise reproducibility

All four original modes reused one compiler cache. During a later lossless
experiment, a new cache selected six different RMSNorm reduction configurations
(e.g. R0_BLOCK 8192 versus 4096); even the gate-disabled control changed its video.
The initial cold-cache control is therefore excluded from that later A/B.
`compile-config-snapshot.json` records 80 selected configurations present in the
original cache, including configurations that may not be active in each mode.
The later gate experiment uses a copy and explicitly checks that these selections
do not change. Bitwise video reproducibility requires holding the selected
reduction kernels fixed as well as prompt, seed, weights, precision and schedule.
No bitwise guarantee is made across independent compiler autotuning sessions.
