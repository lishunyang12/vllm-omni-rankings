# MiniMax-H3 on 4x RTX PRO 5000 Blackwell

Reproducible four-GPU MiniMax-H3 measurements for the PCIe-only RTX PRO 5000
Blackwell target. The implementation is vLLM-Omni
[#5852](https://github.com/vllm-project/vllm-omni/pull/5852) at commit
`6f8e151443496b51abc9c6e051ae333bb1c71ef2`.

## Measurement policy

Publish latency and kernel composition from separate runs:

1. Three normal full-step runs provide steady-state E2E/stage latency and peak
   memory. Nsight wall time is never used as the performance number.
2. One full-step Nsight Systems run provides aggregate GPU-kernel shares and
   per-GPU load balance.
3. Raw `.nsys-rep`, SQLite, generated MP4, and model files stay on the test
   host. This repository stores the reproduction script, small JSON/CSV/MD
   summaries, and SHA256 values for the retained raw artifacts.

MiniMax-H3 performs one fewer progress-bar denoise update than the requested
scheduler-step value in this configuration. `STEPS=50` is therefore recorded
as **50 requested / 49 executed denoise updates**. If matching an existing
table that says 49 requested, use `STEPS=49` and record 48 executed updates.

## Topology selection before the full run

The host has no GPU NVLink. Within NUMA node 1, physical pairs 4-5 and 6-7 are
PXB-local. TP2 x Ulysses2 creates logical TP groups 0-1/2-3 and logical
Ulysses groups 0-2/1-3.

Run both five-step profile orders before committing to the full run:

```bash
# A: Ulysses groups use PXB-local links (the uploaded preliminary result).
MODE=profile STEPS=5 REPEATS=1 \
GPU_IDS=4,6,5,7 GPU_ORDER_LABEL=ulysses-pxb \
bash run_4gpu.sh

# B: TP groups use PXB-local links.
MODE=profile STEPS=5 REPEATS=1 \
GPU_IDS=4,5,6,7 GPU_ORDER_LABEL=tp-pxb \
bash run_4gpu.sh
```

Choose the order with the lower normal-run median, using the profile only to
explain the difference. The A profile currently has 14.63%/15.76% `NCCL
other`; this bucket includes collectives such as TP AllReduce, so B remains a
required control.

## Final full-step commands

After selecting `GPU_IDS` and `GPU_ORDER_LABEL`, run the normal benchmark:

```bash
MODE=benchmark STEPS=50 REPEATS=3 \
GPU_IDS=4,6,5,7 GPU_ORDER_LABEL=ulysses-pxb \
bash run_4gpu.sh
```

Then capture the matching full-step profile:

```bash
MODE=profile STEPS=50 REPEATS=1 \
GPU_IDS=4,6,5,7 GPU_ORDER_LABEL=ulysses-pxb \
bash run_4gpu.sh
```

Override the defaults when the checkout or model lives elsewhere:

```bash
TEST_ROOT=/path/to/test-root \
REPO_ROOT=/path/to/vllm-omni-pr5852 \
MODEL_ROOT=/path/to/MiniMax-H3 \
RESULT_ROOT=/path/to/results \
bash run_4gpu.sh
```

The selected GPUs must be idle. `run_4gpu.sh` leaves Ref2VA disabled, keeps
weights resident, uses BF16 `CUDNN_ATTN`, binds the process to NUMA node 1,
and disables only NCCL RAS diagnostics to avoid listener-port conflicts with
unrelated jobs. NCCL data-plane collectives remain enabled.

## Preliminary five-step result

This is a topology screen, not the final full-step latency result. See
[`result_5step.json`](./result_5step.json),
[`kernel_breakdown_5step.md`](./kernel_breakdown_5step.md), and
[`gpu_peak_memory_5step.csv`](./gpu_peak_memory_5step.csv).

| Workload | NCCL AllGather | NCCL SendRecv | NCCL other | NCCL total | Dense attention | Other | Max GPU deviation |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2V | 0.93% | 8.53% | 14.63% | 24.09% | 31.15% | 44.76% | 0.56% |
| First-frame I2V | 0.89% | 7.26% | 15.76% | 23.91% | 32.53% | 43.56% | 0.09% |

The maximum external peak was 69,219 MiB (67.60 GiB), leaving 4,196 MiB
(4.10 GiB) on a 73,415 MiB GPU.

## Supplied FSDP comparison

The supplied four-card FSDP profile reported 35.7% NCCL for T2V and 35.04%
for I2V, including 23.1% T2V AllGather. The resident TP2 x Ulysses2 screen
reduces T2V NCCL share by 11.61 percentage points (32.5% relative) and I2V by
11.13 points (31.8% relative). T2V AllGather falls from 23.1% to 0.93%.

The larger attention percentage does not prove attention became slower: the
percentage denominator changed when communication fell. Absolute speedup must
use matched non-profile E2E/stage timings from the same hardware, prompts,
shape, step count, and physical GPU order.
