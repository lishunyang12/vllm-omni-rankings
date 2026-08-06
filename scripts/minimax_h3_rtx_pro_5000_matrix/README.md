# MiniMax-H3 full matrix on RTX PRO 5000 Blackwell

This package runs and exports the complete MiniMax-H3 comparison matrix on an
eight-card, PCIe-only RTX PRO 5000 Blackwell (SM120) host. It records T2VA,
first-frame FL2VA, and image+audio Ref2VA for 1, 2, 4, and 8 GPUs.

The output CSV has exactly these columns:

```text
GPU Model, Node / Cluster, GPU Count, Workload, Precision, Parallelism,
Offload, Denoise Steps, E2E (s), Text Encode (s), Visual Encode (s),
Audio Encode (s), Latent (s), Denoise (s), VAE Decode (s), Per Step (ms),
Peak Memory (GiB), Peak Memory Scope, SM Clock (MHz), Status, Notes
```

## Modes

The default run keeps three technically distinct modes separate:

- `bf16`: checkpoint BF16/FP32 with dense `CUDNN_ATTN`;
- `fp8`: online FP8 DiT linear layers with dense `CUDNN_ATTN`—this matches the
  normal “FP8” precision row in the reference table;
- `fp8_sm120_attn`: online FP8 DiT linear layers plus FP8 E4M3 Q/K/V through
  `TRTLLM_ATTN` and FlashInfer `cute-dsl-prims`.

Do not merge `fp8` and `fp8_sm120_attn` into one result. The former isolates
linear quantization; the latter also changes the attention kernel.

## Default 72 GB topologies

| GPUs | Topology | Offload | Default physical order |
|---:|---|---|---|
| 1 | TP1 x Ulysses1 | DiT DLO, 20 resident layers | `0` |
| 2 | TP2 x Ulysses1 | none | `0,1` (PXB pair) |
| 4 | TP2 x Ulysses2 | none | `0,2,1,3` (Ulysses PXB pairs) |
| 8 | TP2 x Ulysses4 | none | `0,1,2,3,4,5,6,7` (TP PXB pairs) |

TP1 x Ulysses4 and TP1 x Ulysses8 from a 96 GB RTX PRO 6000 table are not
valid defaults on 72 GB RTX PRO 5000 cards: their reported 94.8/89.9 GiB BF16
peaks exceed capacity. The matrix therefore uses the fastest known
resident-capable candidates. A five-step topology screen should still compare
`GPU_IDS_4=0,2,1,3` with `GPU_IDS_4=0,1,2,3` before publishing the final result.

## Install the experimental attention dependency

The dense BF16/FP8-linear modes use the regular environment. The SM120 mode
requires this exact experimental FlashInfer revision:

```bash
cd /lustre/raplab/client/sylarl/minimax-h3-native
git clone --recursive https://github.com/Tom-Zheng/flashinfer.git flashinfer-sm120
cd flashinfer-sm120
git checkout 4a2345906256da0849d7e1e4681db514ab9b800e
source ../.venv/bin/activate
python -m pip install -v '.[cu13]'
python -m pytest -q tests/attention/test_sm120_prims_prefill_backend.py
python -c 'from flashinfer.attention.cute_dsl.sm120_fmha import SM120PrimsBatchPrefillBackend; print("SM120 prims OK")'
```

Use [vLLM-Omni Draft PR #5860](https://github.com/vllm-project/vllm-omni/pull/5860),
which integrates this wrapper below `TRTLLM_ATTN`:

```bash
cd /lustre/raplab/client/sylarl/minimax-h3-native
git clone https://github.com/lishunyang12/vllm-omni.git vllm-omni-trtllm-sm120
cd vllm-omni-trtllm-sm120
git fetch origin feat/trtllm-sm120-prims
git checkout feat/trtllm-sm120-prims
source ../.venv/bin/activate
python -m pip install --no-deps -e .
```

## Smoke test first

Run one five-step, four-GPU BF16 case before committing the whole host:

```bash
cd /path/to/vllm-omni-rankings/scripts/minimax_h3_rtx_pro_5000_matrix

TEST_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native \
CODE_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native/vllm-omni-trtllm-sm120 \
MODEL_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native/MiniMax-H3 \
GPU_COUNTS=4 MODES=bf16 STEPS=5 WARMUP_STEPS=2 \
RUN_REF2VA=0 REPEATS=1 \
bash run_matrix.sh
```

Then smoke-test only the new attention path:

```bash
GPU_COUNTS=4 MODES=fp8_sm120_attn STEPS=5 WARMUP_STEPS=2 \
RUN_REF2VA=0 REPEATS=1 \
bash run_matrix.sh
```

The second command must log `TRTLLM_ATTN` with `cute-dsl-prims`. If it reports
compute capability other than 12.0, or the custom module is absent, treat it as
a failed case—never relabel a dense fallback as the new kernel.

## Full matrix

The following runs all 12 configurations and all required workloads. With
`REPEATS=3`, latency columns are medians and external peak memory/SM clock are
maxima across successful repeats:

```bash
TEST_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native \
CODE_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native/vllm-omni-trtllm-sm120 \
MODEL_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native/MiniMax-H3 \
RESULT_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native/results/rtx5000-full-matrix \
GPU_COUNTS=1,2,4,8 \
MODES=bf16,fp8,fp8_sm120_attn \
STEPS=50 WARMUP_STEPS=2 RUN_REF2VA=1 REPEATS=3 RESUME=1 \
bash run_matrix.sh 2>&1 | tee full-matrix-console.log
```

`Denoise Steps` records the requested value (`STEPS=50`). H3 executes 49
denoise updates for that request, and `Per Step` divides denoise time by those
49 actual updates. `RESUME=1` skips only cases whose `status.txt`
contains `Passed`; OOM and failed cases remain rerunnable.

Useful host overrides:

```bash
GPU_IDS_1=4 \
GPU_IDS_2=4,5 \
GPU_IDS_4=4,6,5,7 \
GPU_IDS_8=0,1,2,3,4,5,6,7 \
NUMA_NODE_1=1 NUMA_NODE_2=1 NUMA_NODE_4=1 \
bash run_matrix.sh
```

## Artifacts

Each case directory contains `summary.json`, `run.log`, generated MP4 files,
`gpu_telemetry.csv`, `status.txt`, and exact case metadata. The root contains:

- `matrix.csv`: spreadsheet-ready table with the requested columns;
- `matrix.json`: the same rows plus source commit metadata;
- `matrix.env`: immutable run-level parameters.

Peak memory and SM clock come from external `nvidia-smi` sampling every 500 ms.
The exporter selects the maximum per-GPU memory and maximum observed SM clock.
Nsight Systems profiling is intentionally separate from this E2E matrix so its
overhead cannot contaminate latency numbers.
