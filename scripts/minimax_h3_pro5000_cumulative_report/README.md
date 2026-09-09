# MiniMax-H3 cumulative optimization report

This directory contains the evidence-backed English report for the cumulative
MiniMax-H3 text-to-video-with-audio latency path from **653.838 seconds** to a
formal three-run mean of **14.561 seconds**.

- [71-page PDF](minimax_h3_653s_to_14s_cumulative_report.pdf)
- [Web landing page](index.html)
- [Machine-readable report data](report_data.json)
- [Artifact checksums](SHA256SUMS.txt)
- [Moon Teahouse complete-video A/B player](quality/moon-teahouse-seed1101/index.html)
- [Two-prompt shape/style generalization A/B](quality/two-prompt-generalization/index.html)
- [Full 362-frame lossy-boundary comparisons](evidence/quality/)
- [Representative video frames](screenshots/)
- [Nsight Systems and Nsight Compute summaries](evidence/)
- [Raw eight-GPU GPC/SYS and NVML frequency data](evidence/nsys/gpu-frequency-20260909T122900Z/)
- [VSA + 4-step BF16 vs Sage A/B: videos, Nsight, timings, and PR4951 assessment](evidence/sage-ab-20260910/)

The final samples are `14.576 / 14.555 / 14.551` seconds. The endpoint is an
HTTP POST through a validated 362-frame, 1280x704, 24 fps H.264 video with
stereo 32 kHz AAC. The deliverable duration is 15.083333 seconds, so the final
mean is faster than real time and has 0.439 seconds of margin against the strict
15-second target.

## Evidence policy

Distillation, VSA, FP8/E4M3, and TAEH3 are explicitly treated as lossy or
approximate boundaries. Each boundary has real MP4 screenshots and full-video
post-codec PSNR/SSIM evidence. The metrics diagnose trajectory differences;
they are not presented as a perceptual admission threshold.

The report distinguishes measured, exact, lossy, implemented-only, and
rejected work. D7 AAC pre-encoding was implemented but never received a clean
formal A/B. A final decoder-focused Nsight Systems run was blocked by an
external GPU workload before model load, so no D7 timing or synthetic final
timeline is claimed.

The benchmark used a hash-locked Chinese prompt. The official MiniMax-H3
prompt-writing guide at commit
[`d21241f0a4b3acbb34c97dae47fa417b7065e438`](https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing)
is used to organize the English documentation rendering; that rendering is not
misrepresented as the executed prompt.

## Regeneration

The checked-in PDF and figures are self-contained. Regenerating the contact
sheets requires the original local benchmark MP4 artifacts at the paths encoded
in `generate_report.py`.

```bash
python scripts/minimax_h3_pro5000_cumulative_report/generate_report.py
```

The generator requires `av`, `matplotlib`, `numpy`, `Pillow`, `pypdf`, and
`reportlab`. It asserts that the resulting PDF contains at least 50 pages.
