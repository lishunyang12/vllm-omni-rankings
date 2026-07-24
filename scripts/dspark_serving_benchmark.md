# DSpark audio head — vLLM serving benchmark (with vs without draft)

End-to-end serving benchmark of the block-7 DSpark audio draft head in front of the
**Qwen3-Omni-30B-A3B** Thinker on LibriSpeech transcription. Measured against a live vLLM
server; output is verified **identical token-for-token** with and without the draft.

- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct` Thinker + DSpark audio draft (block 7, aux layers `[2,13,24,35,46]`)
- Task / data: LibriSpeech `train-clean-100` transcription · greedy decoding
- Hardware: 1×B300, tensor-parallel 2 · vLLM 0.25.0
- Requires the speculators config-loader fix (see bottom).

## Table 1 — Throughput & latency at concurrency 8

24 requests per run, streamed. Reported as **mean ± std over 4 runs** (a warm-up run is dropped).

| Metric | Without DSpark | With DSpark | Delta |
|---|--:|--:|:--|
| Benchmark duration (s) | 6.32 ± 0.4 | 1.32 ± 0.0 | ↓ 79.1% |
| Request throughput (req/s) | 3.82 ± 0.3 | 18.22 ± 0.2 | ↑ 377% |
| Output token throughput (tok/s) | 217.6 ± 14.5 | 1038.5 ± 12.6 | ↑ 377% |
| Mean E2EL (ms) | 1879 ± 119 | 402.7 ± 5.9 | ↓ 78.6% |
| Mean TTFT (ms) | 95.1 ± 1.7 | 116.7 ± 2.5 | ↑ 22.7% (worse) |
| Mean TPOT (ms) | 31.84 ± 2.0 | 5.12 ± 0.1 | ↓ 83.9% |
| Mean ITL (ms) | 31.74 ± 1.9 | 36.08 ± 0.7 | ↑ 13.7% (worse) |
| Audio throughput (audio-s/s) | 52.2 ± 3.5 | 248.9 ± 3.0 | ↑ 377% |
| Mean AUDIO_RTF | 7.33 ± 0.4 | 33.87 ± 0.5 | ↑ 362% |
| Accepted length τ (/8) | 1.00 | 7.07 | — |

**~4.8× throughput, TPOT ↓84%, E2EL ↓79%.** TTFT (+23%) and ITL (+14%) are slightly worse — the
draft adds overhead to the first token, and speculative decoding emits tokens in bursts, so the
*visible* inter-token gap widens even though the true per-token cost (TPOT) drops sharply.

## Table 2 — Concurrency sweep

Single fixed batch of 32 requests replayed at each concurrency level.

| Concurrency | Baseline (tok/s) | DSpark (tok/s) | Speedup | Mean lat. (s) | p99 lat. (s) | τ (/8) |
|--:|--:|--:|:--|--:|--:|--:|
| 1 | 29.3 | 145.4 | 5.0× | 0.39 | 0.58 | 7.00 |
| 4 | 95.4 | 494.6 | 5.2× | 0.44 | 0.68 | 6.98 |
| 8 | 214.8 | 1057.7 | 4.9× | 0.39 | 0.51 | 6.98 |
| 16 | 388.7 | 1203.0 | 3.1× | 0.56 | 0.92 | 7.01 |

Speedup holds ~5× through concurrency 8, then eases to 3.1× at 16 as the draft saturates the GPU
(compute-bound) while the plain model is still scaling. Accepted length is essentially constant
(τ ≈ 7) across load, so the taper is GPU saturation, not degraded drafting.

## The fix

In serving the head first accepted only ~1.1 tokens/step (no speedup). The speculators config
loader hardcoded the 1+N block layout, ignoring the checkpoint's `sample_from_anchor`, which
shifted the drafted block by one position. In
`vllm/transformers_utils/configs/speculators/algos.py` (`update_dspark`):

```diff
- pre_trained_config["dspark_bonus_anchor"] = True
+ pre_trained_config["dspark_bonus_anchor"] = not config_dict.get("sample_from_anchor", True)
```

Affects every `sample_from_anchor=True` speculators DSpark head (incl. the image head). An earlier
MRoPE hypothesis was disproven — serving feeds correct `arange` positions, and offline replay
confirmed the draft weights predict correctly on the same inputs; the block-layout mismatch was
the sole cause.
