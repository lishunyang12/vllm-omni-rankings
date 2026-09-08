# Moon Teahouse full-video A/B

This directory contains two complete MiniMax-H3 MP4 outputs for direct visual
and audiovisual comparison. Both use the exact same prompt, seed, aligned
geometry, four-step FastH3 VSA/Data-Free student, and VSA top-k.

- [Open the synchronized A/B player](index.html)
- [Standard VSA + four-step video](videos/standard-vsa4-bf16-full-vae.mp4)
- [Optimized real-time-stack video](videos/optimized-d6-realtime-stack.mp4)
- [Executed prompt](prompt.txt)
- [Machine-readable manifest](manifest.json)
- [SHA-256 checksums](SHA256SUMS.txt)

## Fixed request contract

- Hardware/runtime: 8x RTX PRO 5000 Blackwell (SM120), vLLM-Omni
- Task: T2VA through the FL2VA transformer partition
- Seed: `1101`
- Media: 362 H.264 frames, 1280x704, 24 fps, 15.083333 seconds
- Audio: stereo AAC, 32 kHz, 15.075 seconds
- Prompt tokens observed by Qwen: 457
- Packed sequence: 95,823 valid rows, 105,408 aligned rows
- FastH3: VSA/Data-Free, four transformer calls, tile 64, top-k 162
- Prompt payload SHA-256: `5abb96b31333eea455c241aa17e61d1eaf75d7e78625c80f17302b34ce11fad3`

## Compared lanes

The **standard** lane keeps BF16 main-transformer linear execution and BF16
QKV transport, and uses the full H3 video VAE. Its artifact request completed
in 27.995 seconds (RTF 1.856).

The **optimized** lane adds all-main FP8 linear execution, calibrated E4M3 QKV
transport, reverse-O bundling, persistent RDMA, TAEH3 FP16, corrected per-rank
OMP placement, and cross-rank audio decode. The requested artifact was produced
with no warm-up, so its first request includes lazy compilation and took 22.367
seconds (RTF 1.483). This exact serving stack has a separate matched-geometry
steady-state qualification of 14.576 / 14.555 / 14.551 seconds, mean 14.561
seconds (RTF 0.965). That qualification used another hash-locked prompt; it is
reported here as a runtime-stack result, not falsely attributed to this cold
artifact request.

## Quality boundary

Both outputs already use the distilled, trainable-sparse FastH3 student. The
optimized lane introduces additional numerical approximation (FP8/E4M3) and a
decoder substitution (TAEH3). TAEH3 is a lossy preview/real-time decoder, not a
lossless implementation of the full H3 VAE. This pair is provided for human
quality review; latency alone does not establish quality parity.
