# MiniMax-H3 cumulative optimization — revised full-VAE edition

The updated **24-page PDF** follows the restored **27.4937s BF16 VSA baseline**
through profiling-driven pipelines and selected precision changes to a five-request
mean of **14.9692s** with the **full H3 VAE and BF16 communication**.
All five final requests are below 15 seconds.

- [Revised PDF — original download path](minimax_h3_653s_to_14s_cumulative_report.pdf)
- [Web report](index.html)
- [中文累计优化表与来源](update-20260914/adopted-cumulative.zh.md)
- [Current measurements, quality and source identities](update-20260914/data.json)
- [Cumulative chart, SVG](update-20260914/figures/final-cumulative.svg)
- [Every adopted measured comparison, SVG](update-20260914/figures/adopted-comparisons.svg)
- [Checksums](SHA256SUMS.txt)

The revision explicitly includes gate projection overlap, QK/V preparation,
QKV producer splitting, four-chunk O return/projection, dependency reordering,
VAE pairing/batching, allocator changes, RDMA retention, activation quantization
fusion and the final O producer lookahead. Only adopted additions with measured
complete-request savings greater than 0.100s are listed as gain rows. Each
round retains its own control; unpaired precision checkpoints are not assigned
invented single-factor gains.

## Correction and original archive

The old **14.561s** result inherited E4M3 QKV transport that subsequently failed
complete-video human quality review. It is superseded as a recommended endpoint.
TAEH3 was retired from the selected full-VAE profile, not declared to have failed
its own quality gate. See the [later quality record](quality/moon-teahouse-seed1101/README.md).

The [original 71-page PDF](archive-20260907/minimax_h3_653s_to_14s_cumulative_report.pdf)
is archived byte for byte with its original data and generator. The revised PDF
retains 12 original foundation pages and labels their measurements as historical.
The original 653.838s and current endpoint use different contracts; no matched
653.838 / 14.9692 speedup is claimed.

## Quality and timing scope

VAE MXFP8 versus the preceding full-VAE reference: SSIM **0.978920**, PSNR
**43.041443 dB**, all **362 decoded frames**, codec effects included; identical
audio PCM. Later scheduling changes preserve six corresponding complete MP4s
byte for byte (one warmup and five formal requests). This is not a six-prompt
quality study. FastH3, VSA, Sage/CAKE and MXFP8 remain approximate.

Final formal samples: `14.973 / 14.975 / 14.968 / 14.954 / 14.976` seconds.
Timing spans client POST to complete MP4 saved; loading, warmup and post-run
validation are excluded. Eight SM120 GPUs, four ConnectX-8 groups, full video
VAE parallel8, 362 frames, 1280×704, 24 fps, H.264/AAC. Clocks are unlocked;
cohorts are sequential. No new GPU testing was performed for this revision.

## Regenerate

```bash
python scripts/minimax_h3_pro5000_cumulative_report/generate_report.py
```

The default entrypoint now builds the revised PDF using the checked-in archived
PDF, evidence and figures. It needs `matplotlib`, `numpy`, `Pillow`, `pypdf`,
`reportlab`, and system DejaVu Sans fonts. No model, GPU or original video files
are needed. The preserved frame comparison and original Nsight image are inputs.
The generator checks page geometry, source-archive identity, required numerical
content and the greater-than-0.100s selection rule. SVG charts are regenerated.

Source evidence snapshots retain their original relative links and historical
wording. Their manifest records where they came from; they do not replace the
revised selection and qualification statements above.
