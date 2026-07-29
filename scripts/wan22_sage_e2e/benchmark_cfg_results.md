# Wan2.2-A14B SAGE benchmark — CFG standard load (B200 & B300)

Standard load: **Wan2.2-T2V-A14B, 1280×720, 81 frames, 50 steps, `torch.compile`,
CFG enabled on both high-noise and low-noise experts** (guidance 4.0/4.0 + negative prompt).
Single GPU, seed 0. FlashInfer 0.6.16rc3. LPIPS/PSNR are per-frame mean vs the same-machine
BF16-dense output. Scripts: `wan22_bench.py`, `lpips_bench.py`, `run_bench_b200.sh`.

## B200 / GB200 (SM100)

| mode | generation | per step | vs dense | LPIPS ↓ | PSNR ↑ | SAGE kernel |
|---|---|---|---|---|---|---|
| dense (BF16) | 474.37 s | 9.487 s | — | — | — | `Sm100fKernel_QkvBfloat16` |
| **fp8 SAGE** | **398.43 s** | 7.969 s | **1.19×** | **0.120** | 24.80 dB | `Sm100fKernel_QkvE4m3…SageQ1K16` |
| int8 SAGE | 409.06 s | 8.181 s | 1.16× | 0.304 | 19.35 dB | `Sm100aKernel_QkInt8VE4m3…SageQ1K16` |

## B300 (SM103)

| mode | generation | per step | vs dense | LPIPS ↓ | PSNR ↑ | SAGE kernel |
|---|---|---|---|---|---|---|
| dense (BF16) | 474.94 s | 9.499 s | — | — | — | — |
| **fp8 SAGE** | **392.34 s** | 7.847 s | **1.21×** | **0.260** | 19.34 dB | `Sm103aKernel_QkvE4m3…SageQ1K16` |
| int8 SAGE | — | — | — | — | — | no SM103 kernel (SM100 only) |

## Image-to-video (Wan2.2-I2V-A14B)

Same load (1280×720 / 81f / 50 steps / compile / CFG 4.0/4.0), but a first-frame condition
image is supplied. LPIPS/PSNR vs the same-machine dense output. Scripts: `wan22_i2v_bench.py`,
`run_i2v_b200.sh`.

### B200 / GB200 (SM100) — clean, same input image + seed

| mode | generation | per step | vs dense | LPIPS ↓ | PSNR ↑ | MSE ↓ |
|---|---|---|---|---|---|---|
| dense (BF16) | 482.42 s | 9.648 s | — | — | — | — |
| **fp8 SAGE** | **401.24 s** | 8.025 s | **1.20×** | 0.0998 | 20.40 dB | 0.03811 |
| int8 SAGE | 416.98 s | 8.340 s | 1.16× | 0.0966 | 20.38 dB | 0.03896 |

### B300 (SM103)

| mode | generation | per step | vs dense | LPIPS ↓ | PSNR ↑ |
|---|---|---|---|---|---|
| dense (BF16) | 591.0 s | 11.82 s | — | — | — |
| **fp8 SAGE** | **507.6 s** | 10.15 s | **1.16×** | 0.036 | — |

> The B200 and B300 I2V LPIPS are **not** directly comparable: different condition images
> (B200 = a busy natural photo, B300 = a smooth sunrise) and the B300 pair was run concurrently.
> Each is a valid same-machine dense-vs-quant comparison; the cross-machine number is not.

## Findings

**Speedup — verified on both, ~1.2×.** fp8 SAGE is 1.19× (B200) / 1.21× (B300) faster than
dense. Consistent across archs. With CFG the DiT work is ~doubled and attention is only part of
it, so the quantization win is modest; it grows with sequence length / without CFG.

**fp8 quality is arch-dependent.** fp8 vs dense LPIPS is **0.120 on B200 (SM100)** but
**0.260 on B300 (SM103)** — same config, different result. The two archs use different fp8
SAGE kernels (`Sm100fKernel` vs `Sm103aKernel`); identical math, different tiling/accumulation,
so rounding differs and compounds over 50 diffusion steps. B200's kernel stays much closer to
dense. Both outputs are individually high-quality, artifact-free video.

**int8 SAGE verified on SM100 (B200).** The `Sm100aKernel_QkInt8VE4m3…SageQ1K16` cubin loaded and
the run finished; int8 kernels exist only for SM100 (SM103/B300 has none). int8 is always slightly
slower than fp8 (the fp8 `Sm100fKernel` pipeline is more mature), so it never wins on speed.

**int8's quality gap vs fp8 is content-dependent, not fixed.** On unconstrained **T2V**, int8 is
much worse than fp8 (LPIPS **0.304 vs 0.120**): int8's uniform grid quantizes attention Q/K
outliers poorly while fp8_e4m3's wider dynamic range fits activation distributions better, and that
error compounds over 50 free-running steps. On **I2V**, where the first-frame image anchors the
trajectory, both stay close to dense and the gap collapses to a tie (LPIPS **0.0966 vs 0.0998**,
PSNR/MSE each marginally favoring the other — all noise-level). So int8's penalty is *exposed* by
free generation and *masked* by strong conditioning. **fp8 is still the recommended mode
everywhere**: equal-or-better quality, faster, and it has kernels on both B200 and B300 (int8 is
SM100-only, so it has no portability upside either).

**I2V is easier to quantize than T2V.** fp8 vs dense is 0.036–0.10 (I2V) vs 0.12–0.26 (T2V) on the
same machines. The condition image pins the generation, leaving quantization error much less room
to diverge — the more constrained the task, the closer quant tracks dense.

**Correction on CFG.** An earlier note claimed CFG made fp8 near-lossless (0.28 → 0.12). That was
a cross-machine confound: the 0.28 was B300-without-CFG and the 0.12 was B200-with-CFG.
Apples-to-apples on B300, CFG barely moved fp8 (0.28 → 0.26). The 0.12 comes from the B200
kernel, **not** from CFG.

**Determinism.** The pipeline is bit-reproducible (dense-vs-dense LPIPS = 0.0 on B300), so these
LPIPS values are real signal, not sampling noise; re-running the same config reproduces them.
