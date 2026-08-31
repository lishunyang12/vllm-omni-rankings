# MiniMax-H3 TRTLLM vs FA4 E2E

Matched MiniMax-H3 `FL2VA` T2VA outputs on four or eight NVIDIA B300 or B200
GPUs. Unless a section says otherwise, the linked clip is the first measured
request after one full-shape warmup.

## B300 production-blog attention A/B

The final eight-GPU starship comparison uses TP1, Ulysses8, Ring1, Fast
Ulysses, text-encoder TP8, and VAE tile8. Each row has one excluded warmup and
two thermally qualified measured requests; speedup uses the server's model
execution timer.

| Attention policy | Mean model execution | Speedup vs dense | Full-video LPIPS | Video |
|---|---:|---:|---:|---|
| Dense TRTLLM | 54.246 s | 1.000x | 0 | [MP4](./b300_blog_attention_8gpu_20260831/media/trtllm_dense.mp4) |
| FP8 SAGE, Q1/K4 | 46.592 s | 1.164x | 0.4093 | [MP4](./b300_blog_attention_8gpu_20260831/media/sage_fp8_k4.mp4) |
| Skip-Softmax 0.05/0.97 | 50.029 s | 1.084x | 0.0917 | [MP4](./b300_blog_attention_8gpu_20260831/media/skip_softmax_005_gate097.mp4) |
| FP8 SAGE Q1/K4 + Skip-Softmax 0.05/0.97 | 46.073 s | 1.177x | 0.4103 | [MP4](./b300_blog_attention_8gpu_20260831/media/sage_fp8_skip_005_gate097.mp4) |

The [complete evidence bundle](./b300_blog_attention_8gpu_20260831/) contains
the exact source patch, environment, launch scripts, raw timings, telemetry,
thermal audits, full-frame video metrics, decoded-audio metrics, and media.

## B300 stable matrix

Timings are the median of five measured requests after one warmup; each mode
starts one server and sends all six requests to it. The linked clip is request
2. Every run uses the same prompt, seed, 1248x768 canvas, 209 frames, 24 FPS,
50 denoise steps, and 32 kHz stereo audio. The MiniMax-H3 token refiner remains
dense in every optimized run. Speedup is relative to dense `TRTLLM_ATTN`.

| Mode | Backend | SAGE | Skip-Softmax | Median diffuse | CV | Speedup vs dense | Video | Raw record |
|---|---|---|---|---:|---:|---:|---|---|
| `recipe_flash` | `FLASH_ATTN` | — | — | 74.972 s | 0.05% | 1.017× | [MP4](./b300_20260805_v6/recipe_flash.mp4) | [JSON](./b300_20260805_v6/recipe_flash.json) |
| `trtllm_dense` | `TRTLLM_ATTN` | — | — | 76.275 s | 0.33% | 1.000× | [MP4](./b300_20260805_v6/trtllm_dense.mp4) | [JSON](./b300_20260805_v6/trtllm_dense.json) |
| `sage_fp8` | `TRTLLM_ATTN` | FP8 | — | 63.578 s | 0.18% | 1.200× | [MP4](./b300_20260805_v6/sage_fp8.mp4) | [JSON](./b300_20260805_v6/sage_fp8.json) |
| `skip_softmax_005_gate090` | `TRTLLM_ATTN` | — | threshold 0.05, `disabled_until_timestep=0.90` (21/49 steps) | 73.069 s | 0.09% | 1.044× | [MP4](./b300_20260805_v6/skip_softmax_005_gate090.mp4) | [JSON](./b300_20260805_v6/skip_softmax_005_gate090.json) |
| `skip_softmax_010_gate090` | `TRTLLM_ATTN` | — | threshold 0.10, `disabled_until_timestep=0.90` (21/49 steps) | 72.808 s | 0.32% | 1.048× | [MP4](./b300_20260805_v6/skip_softmax_010_gate090.mp4) | [JSON](./b300_20260805_v6/skip_softmax_010_gate090.json) |
| `skip_softmax_03_gate090` | `TRTLLM_ATTN` | — | threshold 0.30, `disabled_until_timestep=0.90` (21/49 steps) | 71.505 s | 0.26% | 1.067× | [MP4](./b300_20260805_v6/skip_softmax_03_gate090.mp4) | [JSON](./b300_20260805_v6/skip_softmax_03_gate090.json) |
| `skip_softmax_05_gate090` | `TRTLLM_ATTN` | — | threshold 0.50, `disabled_until_timestep=0.90` (21/49 steps) | 71.200 s | 0.22% | 1.071× | [MP4](./b300_20260805_v6/skip_softmax_05_gate090.mp4) | [JSON](./b300_20260805_v6/skip_softmax_05_gate090.json) |
| `skip_softmax_005_gate095` | `TRTLLM_ATTN` | — | threshold 0.05, `disabled_until_timestep=0.95` (30/49 steps) | 72.111 s | 0.24% | 1.058× | [MP4](./b300_20260805_v6/skip_softmax_005_gate095.mp4) | [JSON](./b300_20260805_v6/skip_softmax_005_gate095.json) |
| `skip_softmax_010_gate095` | `TRTLLM_ATTN` | — | threshold 0.10, `disabled_until_timestep=0.95` (30/49 steps) | 71.343 s | 0.25% | 1.069× | [MP4](./b300_20260805_v6/skip_softmax_010_gate095.mp4) | [JSON](./b300_20260805_v6/skip_softmax_010_gate095.json) |
| `skip_softmax_03_gate095` | `TRTLLM_ATTN` | — | threshold 0.30, `disabled_until_timestep=0.95` (30/49 steps) | 69.698 s | 0.14% | 1.094× | [MP4](./b300_20260805_v6/skip_softmax_03_gate095.mp4) | [JSON](./b300_20260805_v6/skip_softmax_03_gate095.json) |
| `skip_softmax_05_gate095` | `TRTLLM_ATTN` | — | threshold 0.50, `disabled_until_timestep=0.95` (30/49 steps) | 69.193 s | 0.19% | 1.102× | [MP4](./b300_20260805_v6/skip_softmax_05_gate095.mp4) | [JSON](./b300_20260805_v6/skip_softmax_05_gate095.json) |
| `skip_softmax_005_gate099` | `TRTLLM_ATTN` | — | threshold 0.05, `disabled_until_timestep=0.99` (43/49 steps) | 70.446 s | 0.07% | 1.083× | [MP4](./b300_20260805_v6/skip_softmax_005_gate099.mp4) | [JSON](./b300_20260805_v6/skip_softmax_005_gate099.json) |
| `skip_softmax_010_gate099` | `TRTLLM_ATTN` | — | threshold 0.10, `disabled_until_timestep=0.99` (43/49 steps) | 69.524 s | 0.22% | 1.097× | [MP4](./b300_20260805_v6/skip_softmax_010_gate099.mp4) | [JSON](./b300_20260805_v6/skip_softmax_010_gate099.json) |
| `skip_softmax_03_gate099` | `TRTLLM_ATTN` | — | threshold 0.30, `disabled_until_timestep=0.99` (43/49 steps) | 67.214 s | 0.28% | 1.135× | [MP4](./b300_20260805_v6/skip_softmax_03_gate099.mp4) | [JSON](./b300_20260805_v6/skip_softmax_03_gate099.json) |
| `skip_softmax_05_gate099` | `TRTLLM_ATTN` | — | threshold 0.50, `disabled_until_timestep=0.99` (43/49 steps) | 65.975 s | 0.17% | 1.156× | [MP4](./b300_20260805_v6/skip_softmax_05_gate099.mp4) | [JSON](./b300_20260805_v6/skip_softmax_05_gate099.json) |
| `sage_fp8_skip_005_gate090` | `TRTLLM_ATTN` | FP8 | threshold 0.05, `disabled_until_timestep=0.90` (21/49 steps) | 63.045 s | 0.25% | 1.210× | [MP4](./b300_20260805_v6/sage_fp8_skip_005_gate090.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_005_gate090.json) |
| `sage_fp8_skip_010_gate090` | `TRTLLM_ATTN` | FP8 | threshold 0.10, `disabled_until_timestep=0.90` (21/49 steps) | 62.726 s | 0.27% | 1.216× | [MP4](./b300_20260805_v6/sage_fp8_skip_010_gate090.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_010_gate090.json) |
| `sage_fp8_skip_03_gate090` | `TRTLLM_ATTN` | FP8 | threshold 0.30, `disabled_until_timestep=0.90` (21/49 steps) | 61.496 s | 0.17% | 1.240× | [MP4](./b300_20260805_v6/sage_fp8_skip_03_gate090.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_03_gate090.json) |
| `sage_fp8_skip_05_gate090` | `TRTLLM_ATTN` | FP8 | threshold 0.50, `disabled_until_timestep=0.90` (21/49 steps) | 60.981 s | 0.18% | 1.251× | [MP4](./b300_20260805_v6/sage_fp8_skip_05_gate090.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_05_gate090.json) |
| `sage_fp8_skip_005_gate095` | `TRTLLM_ATTN` | FP8 | threshold 0.05, `disabled_until_timestep=0.95` (30/49 steps) | 62.783 s | 0.27% | 1.215× | [MP4](./b300_20260805_v6/sage_fp8_skip_005_gate095.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_005_gate095.json) |
| `sage_fp8_skip_010_gate095` | `TRTLLM_ATTN` | FP8 | threshold 0.10, `disabled_until_timestep=0.95` (30/49 steps) | 62.109 s | 0.10% | 1.228× | [MP4](./b300_20260805_v6/sage_fp8_skip_010_gate095.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_010_gate095.json) |
| `sage_fp8_skip_03_gate095` | `TRTLLM_ATTN` | FP8 | threshold 0.30, `disabled_until_timestep=0.95` (30/49 steps) | 60.583 s | 0.23% | 1.259× | [MP4](./b300_20260805_v6/sage_fp8_skip_03_gate095.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_03_gate095.json) |
| `sage_fp8_skip_05_gate095` | `TRTLLM_ATTN` | FP8 | threshold 0.50, `disabled_until_timestep=0.95` (30/49 steps) | 60.086 s | 0.43% | 1.269× | [MP4](./b300_20260805_v6/sage_fp8_skip_05_gate095.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_05_gate095.json) |
| `sage_fp8_skip_005_gate099` | `TRTLLM_ATTN` | FP8 | threshold 0.05, `disabled_until_timestep=0.99` (43/49 steps) | 62.284 s | 0.11% | 1.225× | [MP4](./b300_20260805_v6/sage_fp8_skip_005_gate099.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_005_gate099.json) |
| `sage_fp8_skip_010_gate099` | `TRTLLM_ATTN` | FP8 | threshold 0.10, `disabled_until_timestep=0.99` (43/49 steps) | 61.781 s | 0.28% | 1.235× | [MP4](./b300_20260805_v6/sage_fp8_skip_010_gate099.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_010_gate099.json) |
| `sage_fp8_skip_03_gate099` | `TRTLLM_ATTN` | FP8 | threshold 0.30, `disabled_until_timestep=0.99` (43/49 steps) | 59.741 s | 0.42% | 1.277× | [MP4](./b300_20260805_v6/sage_fp8_skip_03_gate099.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_03_gate099.json) |
| `sage_fp8_skip_05_gate099` | `TRTLLM_ATTN` | FP8 | threshold 0.50, `disabled_until_timestep=0.99` (43/49 steps) | 58.322 s | 0.22% | 1.308× | [MP4](./b300_20260805_v6/sage_fp8_skip_05_gate099.mp4) | [JSON](./b300_20260805_v6/sage_fp8_skip_05_gate099.json) |

All 27 modes passed the timing gate: five measured requests, CV below 2%,
and span/median below 5%. The largest observed CV was 0.43% and the largest
span/median was 1.11%. `FLASH_ATTN` was 1.7% faster than dense
`TRTLLM_ATTN` in this controlled run. FP8 SAGE delivered 1.200×, and the most
aggressive measured combination delivered 1.308×.

`disabled_until_timestep` is compared against the shift-12 video sigma, not
the fraction of completed denoise steps. Values 0.99, 0.95, and 0.90 first
enable Skip-Softmax at steps 6, 19, and 28, leaving it enabled for 43, 30,
and 21 of the 49 steps respectively.

The stage timer synchronizes CUDA before and after `diffuse`. The run used
physical GPUs 0, 1, 4, and 5 with a balanced CPU set. Across 4,180 telemetry
samples per GPU, the maximum temperature was 77°C and neither the thermal
slowdown flag nor counter changed. See the [timing audit](./b300_20260805_v6/timing_audit.json),
[thermal audit](./b300_20260805_v6/thermal_audit.json), and
[kernel profile summary](./b300_20260805_v6/kernel_profiles.md). All 27 clips
also passed a [full ffmpeg decode audit](./b300_20260805_v6/media_audit.json).

Earlier simultaneous runs were not comparable because they competed for host
CPU resources, while physical GPUs 2, 3, 6, and 7 had accumulated software
thermal-slowdown time. The final matrix therefore runs modes serially, pins the
container to the CPU set listed below, and uses GPUs 0, 1, 4, and 5 only.

### B300 kernel-level profiles

Nsight Systems profiled the second `diffuse` request in separate runs, so the
instrumentation does not affect the E2E table above.

| Mode | Rank-0 projected diffuse | Attention GPU time (4-GPU total) | Attention share | Attention speedup vs dense | Primary FMHA average |
|---|---:|---:|---:|---:|---:|
| `trtllm_dense` | 75.383 s | 161.747 s | 55.0% | 1.000× | 15.879 ms |
| `sage_fp8` | 63.054 s | 121.498 s | 50.0% | 1.331× | 12.062 ms |
| `skip_softmax_05_gate099` | 65.797 s | 132.495 s | 52.2% | 1.221× | 13.117 ms |
| `sage_fp8_skip_05_gate099` | 58.072 s | 105.466 s | 47.4% | 1.534× | 10.220 ms |

The SAGE-only trace contains 9,800 main-DiT SAGE FMHA calls and 386 short
dense token-refiner calls. The Skip-Softmax-only trace contains exactly 8,600
sparse calls (43 steps × 50 blocks × 4 GPUs) and 1,586 dense calls (six main
steps plus the refiner). The exact kernel names and raw CSV paths are available
with the [kernel profile artifacts](./b300_20260805_v6/kernel_profiles.json).

## B300 Skip-Softmax fidelity exploration

This follow-up compares small steps beyond threshold 0.05 with
`disabled_until_timestep=0.95`. LPIPS is measured between the last decoded RGB
frame of each clip and the dense `TRTLLM_ATTN` reference. Every row uses the
same prompt and seed, with SAGE disabled. This is a last-frame signal rather
than a full-video quality metric.

| Threshold | `disabled_until_timestep` | Enabled steps | Last-frame LPIPS | Video | Raw record |
|---:|---:|---:|---:|---|---|
| — | — | 0/49 | 0.0000 | [MP4](./b300_skip_softmax_fidelity_20260806/trtllm_dense.mp4) | [JSON](./b300_skip_softmax_fidelity_20260806/trtllm_dense.json) |
| 0.05 | 0.95 | 30/49 | 0.1224 | [MP4](./b300_skip_softmax_fidelity_20260806/skip_softmax_005_gate095.mp4) | [JSON](./b300_skip_softmax_fidelity_20260806/skip_softmax_005_gate095.json) |
| 0.05 | 0.96 | 32/49 | 0.1367 | [MP4](./b300_skip_softmax_fidelity_20260806/skip_softmax_005_gate096.mp4) | [JSON](./b300_skip_softmax_fidelity_20260806/skip_softmax_005_gate096.json) |
| 0.05 | 0.97 | 35/49 | 0.1681 | [MP4](./b300_skip_softmax_fidelity_20260806/skip_softmax_005_gate097.mp4) | [JSON](./b300_skip_softmax_fidelity_20260806/skip_softmax_005_gate097.json) |
| 0.06 | 0.95 | 30/49 | 0.1710 | [MP4](./b300_skip_softmax_fidelity_20260806/skip_softmax_006_gate095.mp4) | [JSON](./b300_skip_softmax_fidelity_20260806/skip_softmax_006_gate095.json) |
| 0.06 | 0.96 | 32/49 | 0.1872 | [MP4](./b300_skip_softmax_fidelity_20260806/skip_softmax_006_gate096.mp4) | [JSON](./b300_skip_softmax_fidelity_20260806/skip_softmax_006_gate096.json) |
| 0.075 | 0.95 | 30/49 | 0.2234 | [MP4](./b300_skip_softmax_fidelity_20260806/skip_softmax_0075_gate095.mp4) | [JSON](./b300_skip_softmax_fidelity_20260806/skip_softmax_0075_gate095.json) |
| 0.05 | 0.98 | 39/49 | 0.2445 | [MP4](./b300_skip_softmax_fidelity_20260806/skip_softmax_005_gate098.mp4) | [JSON](./b300_skip_softmax_fidelity_20260806/skip_softmax_005_gate098.json) |

Increasing the gate from 0.95 to 0.96 preserves the last frame better than
increasing the threshold from 0.05 to 0.06. The complete LPIPS output is
available as [JSON](./b300_skip_softmax_fidelity_20260806/last_frame_lpips.json).

## B300 official starship prompt

This session uses the official reproducible MiniMax H3
[starship prompt](./prompts/minimax_h3_official_starship.txt): 1344x768,
243 frames, 10 seconds, 24 FPS, 50 denoise steps, and seed 0. Each mode starts
one engine and runs serially on physical GPUs 0, 1, 4, and 5, with one regional
compile warmup followed by five measured requests. The token refiner remains
dense in every optimized mode.

| Backend | SAGE | Skip-Softmax | Median diffuse | Speedup vs dense | CV | Video | Raw record |
|---|---|---|---:|---:|---:|---|---|
| `TRTLLM_ATTN` | — | — | 109.347 s | 1.000× | 0.12% | [MP4](./b300_starship_strict_20260806/trtllm_dense.mp4) | [JSON](./b300_starship_strict_20260806/trtllm_dense.json) |
| `TRTLLM_ATTN` | FP8 | — | 90.014 s | 1.215× | 0.17% | [MP4](./b300_starship_strict_20260806/sage_fp8.mp4) | [JSON](./b300_starship_strict_20260806/sage_fp8.json) |
| `TRTLLM_ATTN` | — | threshold 0.05, `disabled_until_timestep=0.97` (35/49 steps) | 101.734 s | 1.075× | 0.10% | [MP4](./b300_starship_strict_20260806/skip_softmax_005_gate097.mp4) | [JSON](./b300_starship_strict_20260806/skip_softmax_005_gate097.json) |
| `TRTLLM_ATTN` | FP8 | threshold 0.05, `disabled_until_timestep=0.97` (35/49 steps) | 86.983 s | 1.257× | 0.13% | [MP4](./b300_starship_strict_20260806/sage_fp8_skip_005_gate097.mp4) | [JSON](./b300_starship_strict_20260806/sage_fp8_skip_005_gate097.json) |

All four modes passed the strict timing gate: five deterministic measured
outputs, CV below 2%, span/median below 5%, and no thermal slowdown. The maximum
observed CV was 0.17%, the maximum span/median was 0.46%, and the maximum GPU
temperature was 79°C. See the complete [timing results](./b300_starship_strict_20260806/results.json)
and [thermal audit](./b300_starship_strict_20260806/thermal_audit.json).

## MiniMax-H3 single-GPU profile on B300

This session runs MiniMax-H3 FL2VA with the official
[starship prompt](./prompts/minimax_h3_official_starship.txt): 1344x768,
243 frames, 10 seconds, 24 FPS, and seed 0. It uses one B300, dense BF16
`TRTLLM_ATTN`, regional compile, and the fused QK norm/RoPE path. SAGE and
Skip-Softmax are disabled. Sampling uses 20 sigma points, corresponding to 19
denoise updates. The first request warms compilation outside the capture; the
second same-prompt request is captured with Nsight Systems.

### Timing and memory

- Model pipeline: 161.076 s, including 0.050 s cached text encode, 151.141 s
  denoise (7.955 s/update), and 9.877 s VAE decode.
- Output path: `Omni.generate()` returned in 224.764 s. The 63.687 s after the
  model pipeline is dominated by CPU/shared-memory packing, copying, and NumPy
  post-processing for 2.806 GiB of raw float output, rather than model compute;
  the GPU-to-host DMA itself took only 52.7 ms. CPU video/audio encoding and MP4
  mux added 37.151 s, bringing the complete run through the saved MP4 to
  261.915 s.
- GPU memory: 138.51 GiB peak reserved according to PyTorch. Nsight's
  device-allocation high-water mark is 138.96 GiB; the scopes differ slightly.

### Profile analysis

The trace contains exactly 1,900 `_rms_norm_rope_kernel` launches (50 blocks ×
19 updates × Q/K), confirming that the fusion is active throughout denoising.
The dense TRTLLM attention kernel accounts for 69.2% of summed GPU kernel time,
while fused QK norm/RoPE accounts for 1.3%. No SAGE or quantization kernel
appears.

### Artifacts

- [Output video](./b300_single_gpu_qk_norm_rope_nsys_20260811/captured_request.mp4)
- [Nsight Systems report](./b300_single_gpu_qk_norm_rope_nsys_20260811/trace.nsys-rep)
- [Run configuration](./b300_single_gpu_qk_norm_rope_nsys_20260811/config.json)
  and [raw timings](./b300_single_gpu_qk_norm_rope_nsys_20260811/result.json)

## B300 async Ulysses overlap

This session isolates asynchronous Ulysses input exchange on the same official
starship workload. Both modes use dense BF16 `TRTLLM_ATTN`, TP1, Ulysses4,
Ring1, text-encoder TP1, VAE tile4, and regional compile. Each mode has one
warmup followed by five measured requests on physical GPUs 0, 1, 4, and 5.

| Ulysses input exchange | Median diffuse | Speedup | CV | Peak memory | Video | Raw record |
|---|---:|---:|---:|---:|---|---|
| Standard NCCL | 107.557 s | 1.000× | 0.11% | 141,796 MB | [MP4](./b300_async_ulysses_20260810/dense.mp4) | [JSON](./b300_async_ulysses_20260810/dense_summary.json) |
| Async CUDA Copy Engine | 106.151 s | 1.013× | 0.19% | 141,540 MB | [MP4](./b300_async_ulysses_20260810/async.mp4) | [JSON](./b300_async_ulysses_20260810/async_summary.json) |

The async path keeps MiniMax H3's fused QKV projection. It starts the V
exchange before Q normalization and RoPE, then starts the Q exchange before K
normalization and RoPE. The reverse Ulysses exchange remains NCCL. The run
saved 1.406 s of diffuse time and 256 MB of peak worker memory. All measured
outputs were deterministic within each mode, and the thermal audit passed with
no slowdown signal or counter change. See [results](./b300_async_ulysses_20260810/results.json)
and the [thermal audit](./b300_async_ulysses_20260810/thermal_audit.json).

The steady-request Nsight capture shows that the async input path removes
29,391 NCCL kernel launches and 60.6% of aggregate NCCL kernel time. It moves
the input exchange to 88,173 peer-copy operations, of which 68.4% of copy-engine
busy time overlaps non-NCCL GPU compute. Nsight instrumentation records each
peer copy and therefore perturbs end-to-end timing; the table above uses the
unprofiled five-run medians. See the complete
[kernel profile summary](./b300_async_ulysses_20260810/nsys_profile_summary.json).

A two-bank ping-pong variant was also tested. It removed the pre-overwrite
barrier but improved the median by only 0.068 s (106.151 s to 106.083 s) while
raising peak worker memory by 762 MB, so the final implementation keeps the
simpler single bank. The rejected run is retained in the
[raw summary](./b300_async_ulysses_20260810/rejected_pingpong_summary.json) and
[thermal audit](./b300_async_ulysses_20260810/rejected_pingpong_thermal_audit.json).

Changing the compiled execution schedule is not bitwise output preserving:
the two modes produce different deterministic frame and audio hashes despite
the exchange itself matching NCCL bit-for-bit in the distributed parity test.
The clips are provided for inspection; end-of-diffusion LPIPS is not used as a
correctness criterion for this mathematically equivalent scheduling change.

## B200 INT8 results

These runs use four NVIDIA B200 GPUs and the same prompt, seed, output shape,
denoise steps, and parallel configuration as the B300 runs. The table reports
request 2 after one warmup.

| Backend | SAGE | Skip-Softmax | Steady diffusion | Video | Raw record |
|---|---|---|---:|---|---|
| `TRTLLM_ATTN` | INT8 | — | 75.250 s | [MP4](./minimax_h3_t2va_trtllm_sage_int8_4xb200_8p7s_50step.mp4) | [JSON](./sage_int8_b200_summary.json) |
| `TRTLLM_ATTN` | INT8 | threshold 0.3, `disabled_until_timestep=0.90` (21/49 steps) | 72.332 s | [MP4](./minimax_h3_t2va_trtllm_sage_int8_skip_4xb200_8p7s_50step.mp4) | [JSON](./sage_int8_skip_b200_summary.json) |
| `TRTLLM_ATTN` | INT8 | threshold 0.3, `disabled_until_timestep=0.99` (43/49 steps) | 70.131 s | [MP4](./minimax_h3_t2va_trtllm_sage_int8_skip_t03_gate099_4xb200_8p7s_50step.mp4) | [JSON](./sage_int8_skip_03_gate099_b200_summary.json) |
| `TRTLLM_ATTN` | INT8 | threshold 0.5, `disabled_until_timestep=0.99` (43/49 steps) | 68.237 s | [MP4](./minimax_h3_t2va_trtllm_sage_int8_skip_t05_gate099_4xb200_8p7s_50step.mp4) | [JSON](./sage_int8_skip_05_gate099_b200_summary.json) |

The repeated steady-state diffusion times were 75.250 s and 75.228 s for
INT8 SAGE; 72.332 s and 72.401 s for threshold 0.3 at 0.90; 70.131 s and
70.084 s for threshold 0.3 at 0.99; and 68.237 s and 68.225 s for threshold
0.5 at 0.99. Each configuration produced identical frame and audio hashes
across its two steady-state runs.

Diffusion time is the comparable GPU metric; end-to-end wall time is distorted
by large inter-process video output transfers in this vLLM-Omni build.

[`results_manifest.json`](./results_manifest.json) is the machine-readable
index of the stable B300 configurations, timings, videos, and summaries. The
same rows are available as [CSV](./b300_20260805_v6/results.csv).

## B300 environment

| Item | Value |
|---|---|
| GPU | 4x NVIDIA B300 SXM6 AC, SM103, 267.7 GiB each |
| Physical GPU indices | 0, 1, 4, 5 |
| Container CPU set | 0-27, 56-83, 112-139, 168-195 |
| Driver | 610.43.02 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| FlashInfer | 0.6.16.post1 |
| Parallelism | TP1, Ulysses4, Ring1, text-encoder TP1, VAE tile4 |
| Compile | Regional compile enabled; `--enforce-eager` disabled |
| Model | MiniMax-H3 `FL2VA` |
| Prompt seed | 1101 |
| Timing | 1 warmup + 5 measured requests per mode; median reported |

## B200 environment

The B200 runs used 4x NVIDIA B200 (SM100, 178.3 GiB each), driver 595.58.03,
PyTorch 2.11.0+cu130, and FlashInfer 0.6.16.post1. All other model, compile,
and parallel settings match the B300 environment above.

## Serve

Set `MODEL_ROOT` to the directory containing the `FL2VA/` and `Ref2VA/`
partitions. The default is the dense BF16 TRTLLM path for datacenter
Blackwell. `TRTLLM_ATTN` is supported on SM100/SM103 only.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export GPU_IDS=0,1,2,3
export PORT=8091

./serve_b300.sh
```

For the FA4 baseline, start a separate server with:

```bash
ATTENTION_BACKEND=FLASH_ATTN ./serve_b300.sh
```

The server command is also stored in [`serve_b300.sh`](./serve_b300.sh).

## Curl request 1: T2VA

After the server is ready, run:

```bash
./curl_t2va.sh minimax_h3_t2va.mp4
```

The exact multipart request is stored in
[`curl_t2va.sh`](./curl_t2va.sh). It uses the same prompt, shape, duration,
steps, flow shifts, and seed as the recorded comparison.

## Reproduce the measured A/B run

[`run_attention_ab.py`](./run_attention_ab.py) runs the offline matched
comparison and writes a JSON summary plus MP4 output. It requires a local
vLLM-Omni checkout and a local `FL2VA` checkpoint:

```bash
export MODEL_DIR=/path/to/MiniMax-H3/FL2VA
export PYTHONPATH=/path/to/vllm-omni

CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODE=recipe_flash \
MODEL_DIR="${MODEL_DIR}" \
OUTPUT_ROOT=./outputs/recipe_flash \
NUM_RUNS=2 \
python run_attention_ab.py

CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODE=trtllm_dense \
MODEL_DIR="${MODEL_DIR}" \
OUTPUT_ROOT=./outputs/trtllm_dense \
NUM_RUNS=2 \
python run_attention_ab.py
```

Run 1 is the warmup and run 2 is the steady-state measurement. The uploaded
clips were generated from the current packed H3 optimization worktree, so
they are combined-change evidence rather than an isolated run of any single
upstream pull request.
