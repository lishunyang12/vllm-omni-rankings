# Qwen3.8-Flash-Next NVFP4 TP4 validation

## Scope

This report validates the NVFP4 MoE padding fix on top of
[vLLM PR #53896](https://github.com/vllm-project/vllm/pull/53896).

- Base: `2a4cd640ff1a61b66124ddbaaf02a73781f7295a`
- Fix: [`947755647e3e62b48d770f5b70fa9a6ac8effe27`](https://github.com/lishunyang12/vllm/commit/947755647e3e62b48d770f5b70fa9a6ac8effe27)
- Model: `Inferact/Qwen3.8-Flash-Next-NVFP4`
- Date: 2026-08-27 UTC
- Hardware: 4x NVIDIA B300 SXM6 AC (compute capability 10.3)
- FlashInfer: 0.6.17

The run uses tensor parallelism without expert parallelism and explicitly selects
the FlashInfer TRTLLM NVFP4 MoE backend. Qwen3.8 has an MoE intermediate size of
640, so TP4 gives each rank 160 logical intermediate channels. The backend pads
each projection from 160 to 192 channels.

## Regression tests

Command:

```bash
.venv/bin/python -m pytest -q \
  tests/quantization/test_trtllm_nvfp4_hidden_dim_padding.py \
  tests/quantization/test_fp8.py::test_prepare_gated_trtllm_fp8_moe_weights_pads_each_projection
```

Result:

```text
.....                                                                    [100%]
5 passed, 14 warnings in 2.74s
```

The NVFP4 regression test checks that the two fused projections and their block
scales are independently padded from 160 to 192. It also checks that the padding
regions are zero and that the down-projection tensors retain the expected data.

All staged pre-commit hooks and the workspace static precheck passed.

## Checkpoint shard validation

Layer 0, expert 0 was loaded from the real checkpoint and sliced independently
for TP ranks 0 through 3. All four ranks preserved the expected projection and
block-scale boundaries after the 160-to-192 alignment.

```text
rank 0: PASS
rank 1: PASS
rank 2: PASS
rank 3: PASS
```

## End-to-end configuration

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Inferact/Qwen3.8-Flash-Next-NVFP4",
    tensor_parallel_size=4,
    enable_expert_parallel=False,
    moe_backend="flashinfer_trtllm",
    language_model_only=True,
    max_model_len=128,
    max_num_seqs=1,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
)

outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0, max_tokens=8),
)
```

Key runtime evidence:

```text
Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend
Padding intermediate size from 160 to 192
CONFIG tp=4 tep=False
TOKEN_IDS [11751, 11, 264, 3177, 34756, 364, 1141, 25438]
TEXT ' Paris, a city renowned for its iconic'
```

The process exited with status 0. All 21 checkpoint shards loaded successfully,
and each TP rank used approximately 47.6 GiB for model weights and runtime state.

## Result

Qwen3.8-Flash-Next NVFP4 completes inference with pure TP4 and the FlashInfer
TRTLLM backend after padding each fused MoE projection independently. Expert
parallelism is disabled for this validation.
