# MiniMax-H3 BF16 topology matrix on B300

This package records a complete resident-topology screen for MiniMax-H3
`FL2VA` on an eight-GPU NVIDIA B300 NVLink host, followed by a steady-state
verification of the best 1, 2, 4, and 8 GPU configurations.

## Test configuration

- Hardware: 8x NVIDIA B300 SXM6 AC (SM103, NVLink), 275040 MiB per GPU
- Model partition: MiniMax-H3 `FL2VA`
- Precision and attention: dense BF16, `TRTLLM_ATTN`
- Output: 1344x768, 124 frames, 24 FPS, 5 seconds
- Denoise: 50 requested steps, 49 executed updates
- Parallel constants: ring degree 1, text-encoder TP 1, VAE patch parallelism equal to GPU count
- Offload: disabled; all candidates are fully resident
- Steady verification: one complete 50-step warmup before each measured 50-step request
- Peak memory: maximum per-GPU value sampled externally by `nvidia-smi`

The measurements were produced from the current packed MiniMax-H3 optimization
worktree. They are combined-change performance evidence rather than an isolated
benchmark of one upstream pull request.

## Recommended steady-state configurations

| GPUs | Parallelism | T2VA E2E (s) | T2VA denoise (s) | I2VA E2E (s) | I2VA denoise (s) | Peak GiB/GPU |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TP1 x Ulysses1 | 138.92 | 132.66 | 150.48 | 143.92 | 132.88 |
| 2 | TP2 x Ulysses1 | 76.82 | 72.79 | 82.93 | 78.59 | 102.11 |
| 4 | TP1 x Ulysses4 | 41.18 | 38.48 | 44.36 | 41.39 | 135.14 |
| 8 | TP1 x Ulysses8 | 22.28 | 20.00 | 24.36 | 21.88 | 135.75 |

The spreadsheet-ready winner rows are in
[`recommended_steady_50step.csv`](./recommended_steady_50step.csv). The full
candidate screen is in [`measured_matrix_50step.csv`](./measured_matrix_50step.csv).

## Full topology screen

The first-pass T2VA request can include a delayed compile spike after its short
warmup. The following I2VA request runs after compilation and independently
confirms every selected winner. The recommended table above uses a full
50-step warmup before every measured request.

| GPUs | Parallelism | T2VA E2E (s) | I2VA E2E (s) | Peak GiB/GPU |
|---:|---|---:|---:|---:|
| 1 | TP1 x Ulysses1 | 140.22 | 152.60 | 157.24 |
| 2 | TP1 x Ulysses2 | 80.97 | 85.30 | 133.34 |
| 2 | TP2 x Ulysses1 | 78.52 | 82.83 | 101.61 |
| 4 | TP1 x Ulysses4 | 42.26 | 44.26 | 135.14 |
| 4 | TP2 x Ulysses2 | 49.08 | 46.42 | 104.45 |
| 4 | TP4 x Ulysses1 | 44.61 | 46.94 | 85.98 |
| 8 | TP1 x Ulysses8 | 23.66 | 24.20 | 135.75 |
| 8 | TP2 x Ulysses4 | 27.49 | 25.44 | 106.25 |
| 8 | TP4 x Ulysses2 | 28.91 | 27.10 | 88.44 |
| 8 | TP8 x Ulysses1 | 28.18 | 28.66 | 78.28 |

## Deployment decision

DLO is not recommended for this BF16 workload on B300. The highest recommended
peak is 135.75 GiB/GPU against roughly 268.59 GiB/GPU available. DLO would add
host-to-device weight movement without solving a memory constraint.

| GPUs | `NUM_GPUS` | `TP_SIZE` | `ULYSSES_DEGREE` | `VAE_PATCH_PARALLEL_SIZE` | `ENABLE_DLO` |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 1 | 0 |
| 2 | 2 | 2 | 1 | 2 | 0 |
| 4 | 4 | 1 | 4 | 4 | 0 |
| 8 | 8 | 1 | 8 | 8 | 0 |

For every row, use `ATTENTION_BACKEND=TRTLLM_ATTN`, `RING_DEGREE=1`, and
`TEXT_ENCODER_TP_SIZE=1`.
