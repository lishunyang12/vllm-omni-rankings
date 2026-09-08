# Moon Teahouse FP8/E4M3 x TAEH3 factorial A/B

This directory contains complete MiniMax-H3 MP4 outputs for direct visual and
audiovisual comparison. Every admitted factorial cell uses the exact same
prompt, seed, aligned geometry, four-step FastH3 VSA/Data-Free student, and VSA
top-k. The machine-readable manifest is the authority for which cells and
contrasts are complete.

- [Open the synchronized A/B player](index.html)
- [Exact Q0D0 control: BF16/BF16 + full H3 VAE](videos/q0d0-bf16-full-h3-vae.mp4)
- [Exact Q0 decoder diagnostic: TAEH3 FP32](videos/q0d1-bf16-taeh3-fp32.mp4)
- [Exact Q0 decoder endpoint: TAEH3 FP16](videos/q0d2-bf16-taeh3-fp16.mp4)
- [Exact Q1D0 endpoint: FP8/E4M3 + full H3 VAE](videos/q1d0-all-main-fp8-e4m3-wire-full-h3-vae.mp4)
- [Exact Q1D1 endpoint: FP8/E4M3 + TAEH3 FP16](videos/q1d1-all-main-fp8-e4m3-wire-taeh3-fp16.mp4)
- [Exact QmidD0: FP8/BF16 wire + full H3 VAE](videos/qmid-d0-all-main-fp8-bf16-wire-full-h3-vae.mp4)
- [Exact QmidD1: FP8/BF16 wire + TAEH3 FP16](videos/qmid-d1-all-main-fp8-bf16-wire-taeh3-fp16.mp4)
- [Full H3 VAE vs TAEH3 FP16 evidence](metrics/taeh3-at-bf16.json)
- [Full H3 VAE vs TAEH3 FP16 at FP8/E4M3 evidence](metrics/taeh3-at-fp8.json)
- [BF16/BF16 vs FP8/E4M3 at full H3 VAE evidence](metrics/fp8-e4m3-at-full-vae.json)
- [BF16/BF16 vs FP8/E4M3 at TAEH3 evidence](metrics/fp8-e4m3-at-taeh3.json)
- [Exact factorial combined-diagonal evidence](metrics/combined-exact-factorial.json)
- [Full H3 VAE vs TAEH3 FP32 evidence](metrics/taeh3-fp32-at-bf16.json)
- [TAEH3 FP32 vs FP16 evidence](metrics/taeh3-fp16-vs-fp32-at-bf16.json)
- [FP8 compute at full H3 VAE evidence](metrics/fp8-compute-at-full-vae.json)
- [E4M3 wire at full H3 VAE evidence](metrics/e4m3-wire-at-full-vae.json)
- [FP8 compute at TAEH3 evidence](metrics/fp8-compute-at-taeh3.json)
- [E4M3 wire at TAEH3 evidence](metrics/e4m3-wire-at-taeh3.json)
- [TAEH3 at FP8/BF16 wire evidence](metrics/taeh3-at-fp8-bf16-wire.json)
- [Historical standard VSA + four-step video](videos/standard-vsa4-bf16-full-vae.mp4)
- [All-main FP8 + BF16-wire observer video](videos/all-main-fp8-bf16-wire-full-vae-observer.mp4)
- [Historical optimized real-time-stack video](videos/optimized-d6-realtime-stack.mp4)
- [Legacy mixed-contract endpoint evidence](metrics/combined-bf16-full-vae-vs-realtime-stack.json)
- [FP8-compute diagnostic evidence](metrics/precision-bf16-vs-all-main-fp8-bf16-wire.json)
- [Executed prompt](prompt.txt)
- [Machine-readable manifest](manifest.json)
- [SHA-256 checksums](SHA256SUMS.txt)

The current manifest uses schema v2. All four exact factorial cells, all four
primary single-factor edges, and the exact factorial diagonal are available.
Both exact FP8/BF16-wire qmid cells and all five qmid decomposition contrasts
are also complete. The TAEH3-FP32 decoder diagnostic and historical
FP8/BF16-wire observer rung remain available alongside this exact evidence.
The page never substitutes an artifact from another prompt, seed, geometry, or
runtime contract.

## 2x2 methodology

The two independent high-level axes are:

- Precision `q`: `q0` is BF16 main-DiT linear execution plus BF16 QKV
  transport; `q1` is all-main FP8 linear execution plus prompt-bound calibrated
  E4M3 QKV transport.
- Decoder `d`: `d0` is the full H3 video VAE; `d1` is TAEH3 FP16.

The complete matrix therefore contains `q0d0`, `q0d2`, `q1d0`, and `q1d1`.
The primary controls are its four edges and one non-attributable diagonal:

1. TAEH3 at BF16: `q0d0 -> q0d2` (`taeh3_at_bf16`)
2. TAEH3 at FP8/E4M3: `q1d0 -> q1d1` (`taeh3_at_fp8`)
3. FP8/E4M3 at full VAE: `q0d0 -> q1d0` (`fp8_e4m3_at_full_vae`)
4. FP8/E4M3 at TAEH3: `q0d2 -> q1d1` (`fp8_e4m3_at_taeh3`)
5. Combined diagonal: `q0d0 -> q1d1` (`combined`)

The `q0d1_taeh3_fp32` artifact is the optional decoder-dtype middle rung, not a
core matrix corner. The exact FP16 corner is `q0d2_taeh3_fp16`; the other
factorial corners use artifact IDs `q0d0_bf16_full_vae`,
`q1d0_fp8_e4m3_full_vae`, and `q1d1_fp8_e4m3_taeh3_fp16`.

Each high-level axis has an optional middle rung for finer attribution. These
rungs refine the mechanism within an axis; they do not replace the four primary
2x2 edges.

| Axis | Reference | Diagnostic middle rung | Factorial endpoint | Isolation meaning | Current state |
| --- | --- | --- | --- | --- | --- |
| Decoder | Full H3 VAE at Q0 | TAEH3 FP32 at Q0 | TAEH3 FP16 at Q0 | Full VAE -> TAEH3 FP32 diagnoses decoder architecture; TAEH3 FP32 -> FP16 diagnoses decoder dtype | Q0, Qmid, and Q1 decoder comparisons complete |
| Precision · full VAE | BF16 compute / BF16 wire (`q0d0`) | all-main FP8 / BF16 wire (`qmid_d0`) | all-main FP8 / E4M3 wire (`q1d0`) | First edge isolates FP8 linear compute; second isolates E4M3 QKV transport | Both exact qmid diagnostics complete |
| Precision · TAEH3 | BF16 compute / BF16 wire (`q0d2`) | all-main FP8 / BF16 wire (`qmid_d1`) | all-main FP8 / E4M3 wire (`q1d1`) | Repeats the same decomposition with TAEH3 FP16 fixed | Both exact qmid diagnostics complete |

`factorial_complete` refers to the exact four-corner 2x2 and its primary edges;
`diagnostics_complete` separately records the exact qmid decomposition. The
older FP8/BF16-wire observer remains useful historical output evidence, but its
differing scheduling contract and observer overhead exclude it from the exact
qmid ladder.

Both exact FP8 middle rungs use static symmetric **per-tensor weight** scales and
dynamic symmetric **per-token activation** scales, with linear outputs returned
to BF16. The E4M3 wire endpoint adds a prompt-bound static QKV scale table of
shape `[50, 3]`: one scale for each transformer layer and each Q/K/V component.
These quantization granularities are independent of VSA's 64-token attention
tile. The calibration sidecar is valid only for its bound prompt, seed,
geometry, model revision, and execution contract.

Both decoder choices are repeated under both precision states. The decoder
effect is measured by `q0d0 -> q0d2` and `q1d0 -> q1d1`; the precision effect is
measured by `q0d0 -> q1d0` and `q0d2 -> q1d1`. If the two decoder edges—or the
two precision edges—differ, that is a precision-by-decoder interaction. A
TAEH3 result measured only at BF16 cannot be assumed to hold after FP8/E4M3
changes the joint audio-video latent trajectory.

### Edge-admission rules

A pair is labeled as a primary single-factor edge only when:

1. prompt payload hash, seed, packed geometry, four-step student, VSA tile and
   top-k, TP1/SP8 placement, attention kernel, media shape, and codec contract
   match;
2. the artifact factor maps differ only on the declared high-level axis (the
   `q` axis deliberately bundles FP8 compute and E4M3 transport);
3. both MP4s decode to exactly 362 `yuv420p` frames and 482,400 stereo audio
   sample frames beginning at timestamp zero;
4. complete evidence records the input byte counts and SHA-256 digests and all
   structural checks pass; and
5. artifact-local latency uses the same client boundary and records the
   generation policy, but it is not used for pairwise factor attribution. The
   cells run sequentially as one-shot/no-warmup requests, can inherit host,
   disk, and JIT caches from earlier cells, and use node-default unlocked GPU
   clocks. Their E2E/RTF values remain useful provenance, but subtracting them
   or taking their ratio does **not** establish an edge speedup. Observer-
   instrumented and separately warmed timings are likewise excluded.

Each tab uses two complete MP4s with synchronized playback. It shows the two
factor-state maps, artifact-local E2E/RTF, decoded RGB and YUV diagnostics,
temporal diagnostics, decoded-audio diagnostics, and a link to the complete
pairwise evidence JSON. Missing evidence is displayed as pending; metrics from
another prompt or artifact pair are never substituted.

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

## Currently published artifacts

The exact `q0d0_bf16_full_vae` control uses BF16 main-DiT linears, BF16 QKV
transport, the common reverse-O contract, and full H3 VAE PP8. Its artifact-local
one-shot E2E was 39.366 seconds (RTF 2.610). “Exact” here means the intended
factorial execution contract; full-SP8 raw exact-op decoder parity remains the
open validation item described below.

The `q0d1_taeh3_fp32` diagnostic holds the Q0 latent path fixed and replaces
full H3 VAE PP8 with rank-0 TAEH3 FP32 chunk-5 decode. Its artifact-local
one-shot E2E was 28.783 seconds (RTF 1.908). Decoded audio is bitwise identical;
the complete 362-frame evidence reports RGB PSNR 27.646 dB / SSIM 0.840229 and
YUV PSNR 33.565 dB / SSIM 0.943884.

The `q0d2_taeh3_fp16` endpoint keeps the same Q0 latent path and TAEH3 chunk-5
route while changing decoder execution to FP16. Its artifact-local one-shot E2E
was 28.055 seconds (RTF 1.860). Against full H3 VAE, the complete 362-frame
evidence reports RGB PSNR 27.646 dB / SSIM 0.840219 and YUV PSNR 33.564 dB /
SSIM 0.943887, with bitwise-identical decoded audio. The FP32-to-FP16 diagnostic
reports RGB PSNR 39.518 dB / SSIM 0.957604 and YUV PSNR 45.602 dB / SSIM
0.985468, again with bitwise-identical decoded audio. Because these requests
were sequential, cache-exposed, no-warmup runs under unlocked clocks, their E2E
differences and ratios are not decoder speedup measurements.

The `q1d0_fp8_e4m3_full_vae` endpoint keeps the full H3 VAE PP8 decoder and
changes the high-level precision axis to all-main FP8 linear execution plus
prompt-bound calibrated E4M3 QKV transport. Its artifact-local one-shot E2E was
28.007 seconds (RTF 1.857). Against Q0D0, the complete 362-frame evidence
reports RGB PSNR 13.678 dB / SSIM 0.422281 and YUV PSNR 19.617 dB / SSIM
0.778271. Both sides decode 482,400 stereo sample frames; audio is not bitwise
identical and measures 4.376 dB SNR. These same-seed differences record joint
audio-video trajectory divergence, not human-perceived quality. This
sequential no-warmup timing under unlocked clocks is not a precision speedup
measurement.

The `q1d1_fp8_e4m3_taeh3_fp16` endpoint combines the Q1 precision state with
rank-0 TAEH3 FP16 chunk-5 decode while retaining the factorial cells' common
reverse-O and nonpersistent-RDMA contract. Its artifact-local one-shot E2E was
22.543 seconds (RTF 1.495). Against Q1D0, the decoder-only edge reports RGB
PSNR 27.127 dB / SSIM 0.823736 and YUV PSNR 33.072 dB / SSIM 0.936506, with
bitwise-identical decoded audio. Against Q0D2, the precision edge reports RGB
PSNR 13.917 dB / SSIM 0.441525 and YUV PSNR 19.857 dB / SSIM 0.794663; both
sides have 482,400 audio sample frames and measure 4.376 dB SNR. The exact
Q0D0-to-Q1D1 diagonal reports RGB PSNR 13.800 dB / SSIM 0.423268 and YUV PSNR
19.738 dB / SSIM 0.785409. These values diagnose same-seed trajectory
difference rather than human quality, and the one-shot cell timings do not
establish factor speedups.

The Q1D1 MP4 bytes match the historical optimized artifact's digest, but the
primary contrasts reference the clean Q1D1 artifact ID and its common
factorial execution contract. The historical mixed-contract comparison remains
a separate legacy diagnostic.

### Exact qmid decomposition

`qmid_d0_fp8_bf16_full_vae` and `qmid_d1_fp8_bf16_taeh3_fp16` hold the QKV
wire in BF16 while changing main-DiT linear execution to all-main FP8. Their
artifact-local one-shot E2E values were respectively 28.983 seconds (RTF
1.921525) and 23.494 seconds (RTF 1.557613). The resulting full-video
diagnostics are:

| Diagnostic edge | RGB PSNR / SSIM | YUV PSNR / SSIM | Decoded audio |
| --- | --- | --- | --- |
| FP8 compute at full VAE: `q0d0 -> qmid_d0` | 14.041 dB / 0.425039 | 19.962 dB / 0.778650 | 4.391 dB SNR |
| E4M3 wire at full VAE: `qmid_d0 -> q1d0` | 14.274 dB / 0.432799 | 20.207 dB / 0.778557 | -4.007 dB SNR |
| FP8 compute at TAEH3: `q0d2 -> qmid_d1` | 14.297 dB / 0.445751 | 20.221 dB / 0.794100 | 4.391 dB SNR |
| E4M3 wire at TAEH3: `qmid_d1 -> q1d1` | 14.602 dB / 0.451444 | 20.534 dB / 0.794961 | -4.007 dB SNR |
| TAEH3 at FP8/BF16 wire: `qmid_d0 -> qmid_d1` | 27.579 dB / 0.843021 | 33.497 dB / 0.943440 | bitwise identical |

The two E4M3-only edges produce substantially different same-seed stochastic
trajectories after QKV transport changes. Their PSNR/SSIM values—and the
negative decoded-audio SNR—measure distance between those trajectories; they
do not establish visual or audio quality degradation. Human review of the
complete videos remains required. All qmid timings are sequential one-shot,
no-warmup observations under shared caches and unlocked clocks, so their
differences and ratios are not factor speedup measurements.

The historical `legacy_q0d0` control keeps BF16 main-transformer linear
execution, BF16 QKV transport, and the full H3 video VAE. Its recorded artifact
request completed in 27.995 seconds (RTF 1.856) after one warm-up.

The `fp8_bf16_wire_observer` diagnostic changes main-DiT linear execution to
all-main FP8 while retaining BF16 QKV transport and the full H3 video VAE. Its
32.370-second timer includes validation-observer overhead and is not used as a
latency comparison. It is retained as historical evidence; the two exact qmid
artifacts above provide the common-contract decomposition.

The historical `legacy_q1d1_realtime` artifact adds all-main FP8, calibrated
E4M3 QKV transport, reverse-O bundling, persistent RDMA, TAEH3 FP16, corrected
per-rank OMP placement, and cross-rank audio decode. Its no-warmup first request
included lazy compilation and took 22.367 seconds (RTF 1.483). This stack has a
separate matched-geometry steady-state qualification of 14.576 / 14.555 /
14.551 seconds, mean 14.561 seconds (RTF 0.965), but that qualification used a
different hash-locked prompt.

These two historical endpoints also differ in scheduling and timing method.
Their `legacy_combined_endpoint_diagnostic` comparison is useful for reviewing
the endpoint outputs but must not be presented as a clean FP8/E4M3 or TAEH3
effect. The exact factorial-cell reruns now supply all four primary
single-factor comparisons.

## Quality boundary

All outputs already use the distilled, trainable-sparse FastH3 student.
FP8/E4M3 is numerically approximate and can alter the joint audio-video
denoising trajectory; audio is therefore allowed to differ on precision edges.
TAEH3 is a lossy preview/real-time decoder, not a lossless implementation of the
full H3 VAE. A decoder-only edge holds the final latent and audio branch fixed,
so decoded audio must be bitwise identical.

RGB and YUV PSNR/SSIM measure same-seed post-codec difference. The temporal
numbers are transparent luma-derivative error proxies, not learned perceptual
metrics. Audio is compared after decoding exactly 482,400 stereo sample frames
to interleaved float32 PCM at 32 kHz. None of these diagnostics by itself scores
prompt adherence, motion plausibility, speech intelligibility, lip sync, or
human preference; complete-video review remains required.

The full-H3-VAE path has one additional open validation item: SM120 exact-op
raw-output equivalence under full SP8 remains pending. The current checks prove
MP4 structure and quantify decoded, post-codec differences; they do not yet
prove that the full-SP8 raw decoder output is operation-for-operation equivalent
to the reference path. Until that raw comparison closes, “full H3 VAE” names
the selected decoder lane and must not be read as an exact-ops parity claim.

## Manifest schema v2 contract

Schema v2 keeps the existing request/media sections and makes artifacts and
pairwise contrasts explicit. Artifact IDs are stable internal names; the page
does not infer a single-factor claim from filenames.

```json
{
  "schema_version": 2,
  "artifacts": {
    "variant-id": {
      "label": "Human-readable variant",
      "file": "videos/variant.mp4",
      "bytes": 0,
      "sha256": "...",
      "timing": { "e2e_seconds": null, "rtf": null },
      "factors": {
        "precision_stack": "q0 - BF16/BF16 wire",
        "linear_compute": "BF16",
        "qkv_transport": "BF16",
        "video_decoder": "full H3 VAE",
        "decoder_dtype": "released precision"
      }
    }
  },
  "contrasts": [
    {
      "id": "fp8_e4m3_at_full_vae",
      "kind": "single-factor",
      "factor": "precision_stack",
      "changed_factors": ["linear_compute", "qkv_transport"],
      "baseline": "baseline-artifact-id",
      "candidate": "candidate-artifact-id",
      "evidence_file": "metrics/fp8-e4m3-at-full-vae.json",
      "metrics": {
        "rgb": { "psnr_db": null, "ssim": null },
        "yuv": { "psnr_db": null, "ssim": null },
        "temporal": {
          "derivative_mae": null,
          "p95_derivative_mae": null,
          "max_derivative_mae": null
        },
        "audio": { "bit_exact": null, "snr_db": null }
      }
    }
  ]
}
```

The four single-factor IDs expected by the primary controls are
`taeh3_at_bf16`, `taeh3_at_fp8`, `fp8_e4m3_at_full_vae`, and
`fp8_e4m3_at_taeh3`. The end-to-end entry uses `combined`. Other contrast IDs
are rendered separately as diagnostic rungs.
