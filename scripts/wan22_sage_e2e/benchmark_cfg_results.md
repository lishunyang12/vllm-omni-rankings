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
the run finished; int8 kernels exist only for SM100 (SM103/B300 has none). But int8 is both
**slower and much lower quality** than fp8 (LPIPS 0.304 vs 0.120): int8's uniform grid quantizes
attention Q/K outliers poorly, while fp8_e4m3's wider dynamic range fits activation distributions
better. **fp8 is the recommended mode; int8 is not worth using.**

**Correction on CFG.** An earlier note claimed CFG made fp8 near-lossless (0.28 → 0.12). That was
a cross-machine confound: the 0.28 was B300-without-CFG and the 0.12 was B200-with-CFG.
Apples-to-apples on B300, CFG barely moved fp8 (0.28 → 0.26). The 0.12 comes from the B200
kernel, **not** from CFG.

**Determinism.** The pipeline is bit-reproducible (dense-vs-dense LPIPS = 0.0 on B300), so these
LPIPS values are real signal, not sampling noise; re-running the same config reproduces them.
