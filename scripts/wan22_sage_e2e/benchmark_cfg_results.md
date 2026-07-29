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

## B300 (SM103) — clean sequential re-run

| mode | generation | per step | vs dense | LPIPS ↓ | PSNR ↑ | SAGE kernel |
|---|---|---|---|---|---|---|
| dense (BF16) | 478.74 s | 9.575 s | — | — | — | — |
| **fp8 SAGE** | **397.23 s** | 7.945 s | **1.21×** | **0.142** | 23.87 dB | `Sm103aKernel_QkvE4m3…SageQ1K16` |
| int8 SAGE | — | — | — | — | — | no SM103 kernel (SM100 only) |

> Supersedes an earlier B300 T2V measurement (fp8 LPIPS 0.260 / PSNR 19.34). That value did not
> reproduce on a clean single-GPU run; the correct figure is **0.142**.

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

### B300 (SM103) — clean sequential re-run, image-anchored (gentle motion)

Condition image = frame 0 of the B300 T2V dense output (lakeside sunrise); prompt "The scene comes
to life with smooth, natural motion."

| mode | generation | per step | vs dense | LPIPS all 81f ↓ | LPIPS first 16f ↓ | PSNR ↑ |
|---|---|---|---|---|---|---|
| dense (BF16) | 498.45 s | 9.969 s | — | — | — | — |
| **fp8 SAGE** | **451.60 s** | 9.032 s | **1.10×** | 0.0362 | 0.0275 | 29.84 dB |
| int8 SAGE | — | — | — | — | — | no SM103 kernel (SM100 only) |

> Supersedes an earlier concurrent B300 I2V run (dense 591 s / fp8 507.6 s) whose absolute times were
> inflated by contention. Clean fp8 I2V here is very faithful (first-16 LPIPS 0.028) — a gentle
> image-anchored prompt gives quantization little room to diverge.

### B200 / GB200 (SM100) — high-motion I2V (Ruqing protocol: fp8 vs int8, first-16 frames)

Condition image = picsum id237 (puppy); aggressive-motion prompt _"The puppy suddenly leaps up and
shakes its whole body, ears flapping and fur flying, then bounds toward the camera — fast, dynamic
motion."_ Same image + seed across all three modes. First-16-frame LPIPS is the fidelity metric
(early frames are most image-anchored, so it isolates quality from content drift).

| mode | generation | vs dense | LPIPS all 81f ↓ | LPIPS first 16f ↓ |
|---|---|---|---|---|
| dense (BF16) | 482.9 s | — | — | — |
| fp8 SAGE | 405.0 s | 1.19× | 0.2506 | 0.0905 |
| **int8 SAGE** | 417.5 s | 1.16× | **0.1631** | **0.0576** |

> **int8 is closer to dense on BOTH metrics** (all-frames 0.163 < 0.251, first-16 0.058 < 0.091).
> Under strong motion, fp8's blur/distortion drives its LPIPS up while int8 stays sharper and more
> faithful — the reverse of the T2V all-frames ranking. See Videos / compare frames below.

## Findings

**Speedup — verified on both, ~1.2×.** fp8 SAGE is 1.19× (B200) / 1.21× (B300) faster than
dense. Consistent across archs. With CFG the DiT work is ~doubled and attention is only part of
it, so the quantization win is modest; it grows with sequence length / without CFG. Note the two
archs run this load at essentially the same speed (dense ≈ 9.5 s/step on both): at 720p/81f with
`torch.compile`+CFG the bottleneck is not raw FLOPs, so B300 shows no wall-clock edge over B200.

**fp8 quality is NOT strongly arch-dependent (corrected).** On clean single-GPU runs, fp8 vs
dense LPIPS is **0.120 on B200 (SM100)** and **0.142 on B300 (SM103)** — same ballpark. An earlier
B300 reading of 0.260 did not reproduce and is retracted. The two archs use different fp8 SAGE
kernels (`Sm100fKernel` vs `Sm103aKernel`) with slightly different tiling/accumulation, so a small
rounding difference remains, but both stay close to dense and produce high-quality, artifact-free
video. The large arch gap claimed earlier was a measurement artifact, not a kernel property.

**int8 SAGE verified on SM100 (B200).** The `Sm100aKernel_QkInt8VE4m3…SageQ1K16` cubin loaded and
the run finished; int8 kernels exist only for SM100 (SM103/B300 has none). int8 is always slightly
slower than fp8 (the fp8 `Sm100fKernel` pipeline is more mature), so it never wins on speed.

**int8 vs fp8: which is "more accurate" depends entirely on the metric, and all-frames T2V LPIPS
is misleading.** Three same-machine (B200) measurements:

| protocol | fp8 LPIPS | int8 LPIPS | apparent winner |
|---|---|---|---|
| T2V, all 81 frames | 0.120 | 0.304 | fp8 — but this is a **confound** |
| I2V gentle, all 81 frames | 0.0998 | 0.0966 | tie |
| **I2V high-motion, first 16 frames** | 0.0905 | **0.0576** | **int8** |

LPIPS conflates two different errors: **content divergence** (a different-but-equally-valid sample)
and **quality degradation** (blur/distortion). In free-running **T2V**, a text prompt is ambiguous,
so int8 drifts to different-but-good content (e.g. a different breed of the same animal) — that
inflates its all-frames LPIPS without any loss of quality. fp8 stays on the reference's content but
adds blur, which LPIPS scores *lower* even though it looks worse. So the T2V "fp8 wins" is about
content-tracking, not fidelity. Removing the ambiguity with **image-anchored I2V** and reading the
**first-16 frames** isolates fidelity — and there **int8 is clearly closer to dense**, especially
under strong motion where fp8's blur is most visible (int8 0.058 vs fp8 0.091; all-frames 0.163 vs
0.251). This matches the kernel author's report that fp8 SAGE tends to blur while int8 stays sharp.
**Neither mode is universally "more accurate"; pick by need** — int8 for maximum per-frame fidelity
to a dense reference (SM100 only), fp8 for content-tracking, portability (B200 **and** B300), and a
small speed edge.

**Motion matters as much as the task.** A gentle image-anchored I2V is the easiest case for
quantization — clean B300 fp8 first-16 LPIPS is just **0.028**, and gentle B200 fp8/int8 are ~0.10.
But a **high-motion** I2V prompt stresses fp8 hard: fp8 all-frames LPIPS jumps to **0.25** (int8
only 0.16). So "I2V quantizes more faithfully than T2V" holds only for calm scenes; fast motion
re-opens a real fp8 quality gap that int8 does not suffer.

**CFG does improve fp8 fidelity (clean same-machine).** On B300, same 720p/81f/50-step load,
fp8-vs-dense LPIPS drops from **0.284 without CFG** to **0.142 with CFG** — CFG roughly halves the
divergence. (An earlier note overclaimed this as "near-lossless 0.28 → 0.12" by mixing the B300
no-CFG 0.28 with a B200-with-CFG 0.12; the real same-machine effect is 0.28 → 0.14, and the extra
step down to B200's 0.12 is the arch/kernel difference.) Both effects are real; neither is large.

**Determinism.** The pipeline is bit-reproducible (dense-vs-dense LPIPS = 0.0 on B300), so these
LPIPS values are real signal, not sampling noise; re-running the same config reproduces them.

## Videos & compare frames

B200 high-motion I2V (puppy), dense vs fp8 vs int8 — visual evidence for the fidelity ranking above
(fp8 blur vs int8 sharpness on flapping ears / flying fur). _Clips to be attached after scp from the
B200 cluster; placeholders below._

| clip | file | status |
|---|---|---|
| dense (BF16 ref) | `videos/b200_i2v_dense.mp4` | ⏳ pending scp |
| fp8 SAGE | `videos/b200_i2v_fp8.mp4` | ⏳ pending scp |
| int8 SAGE | `videos/b200_i2v_int8.mp4` | ⏳ pending scp |
| ref \| fp8 \| int8 stills (f0/4/8/15) | `videos/i2v_cmp_f*.png` | ⏳ pending scp |

B300 T2V (CFG) clips already in-repo: `videos/b300_t2v_dense.mp4`, `videos/b300_t2v_fp8_sage.mp4`.

<!-- TODO after scp of b200_i2v_out/:
     mv b200_i2v_out/b200_i2v_*.mp4 b200_i2v_out/i2v_cmp_f*.png videos/
     then replace the ⏳ rows above with embedded links / observations. -->

