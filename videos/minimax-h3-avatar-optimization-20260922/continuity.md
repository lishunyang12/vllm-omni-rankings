# Reference-Conditioned Streaming with H3 Turbo

**Model migration, reference construction, and temporal consistency in a five-second avatar system**
Technical report · Retrospective evaluation · 22 September 2026

[System report](report.md) · [Experiment history](history.md) · [Method provenance](evidence/continuity-design.json) · [Per-segment measurements](evidence/continuity-ablation.json)

## Abstract

The transition from a real-time video benchmark to an interactive avatar changes the generation problem: each new segment must preserve an established character and scene while responding to a new instruction. This report describes the project's migration from its FastH3 T2VA/VSA configuration to **MiniMax-H3 Ref2VA with the LightX2V four-step Turbo adapter**. It specifies three complementary inputs: a persistent clean scene reference, the preceding accepted audiovisual latent tail, and a segment-specific instruction. The continuation method combines a noise-conditioned visual guide with an exact video target prefix, followed by decoded-frame screening and rollback to the last accepted state. A retrospective prefix ablation contains three 60-segment trajectories, with 59 boundaries per trajectory. Median adjacent-frame RGB error is 7.824 without prefix protection, 3.785 with video-prefix protection, and 3.934 with audiovisual-prefix protection, measured on the 0–255 scale. The corresponding maximum sampled top black-band heights are 23, 0, and 0 pixels. These observations support improved boundary behavior in the evaluated sequences. A separate 180-segment extension nevertheless exhibits late appearance drift, so exact prefix preservation does not establish indefinite visual consistency. Previously documented kernel and serving optimizations are outside the scope of this supplement.

## 1. Motivation and model migration

The earlier FastH3 configuration addressed a throughput objective: complete an approximately 15-second audiovisual clip before its playback ended. The avatar introduces a different objective. A viewer can issue a new instruction while the character is already holding a book, speaking, or moving a hand. Generating another visually plausible clip is insufficient if it recreates the initial pose or changes the room.

The model migration therefore precedes the duration decision in the system's design. The application adopts H3's **Ref2VA task path**, where visual and optional audio references participate in generation, and combines it with the **Ref2VA Turbo four-step adapter**. The serving controller then applies that generator to bounded windows and supplies the preceding state explicitly. Turbo provides the selected few-step generator; the application and continuation implementation provide the streaming protocol. Shortening the old request alone does not reproduce this system.

**Table 1. Evaluated model routes in this project.** The comparison identifies a task and adapter migration, not a matched accuracy or speed ranking between model families.

| Dimension | Earlier FastH3 route | Selected H3 Turbo route | Consequence for the avatar |
|---|---|---|---|
| Task | T2VA long-clip benchmark | Ref2VA audiovisual generation | Reference conditions become explicit inputs |
| Few-step weights | FastH3 student/adapter path | H3 Ref2VA base plus Ref2VA Turbo adapter | The two four-step routes are not interchangeable |
| Attention coverage | VSA in the selected benchmark | Dense coverage in the selected Ref2VA runtime | VSA-specific assumptions do not describe current conditioning |
| Temporal unit | A completed long request | A 124-frame continuation window | New instructions can enter at the next segment decision |
| Persistent state | Not the objective of the standalone benchmark | Accepted raw AV tail and application state | A segment starts from the established scene state |
| Evidence required | Finite clip quality and throughput | Boundary behavior, appearance, objects, speech and long-run stability | Fast generation alone is insufficient |

The adapter file is `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`. Setup records identify `MiniMaxAI/MiniMax-H3` at revision `73372e6cf53e414edd3ab03e357717fb0602e758` and `lightx2v/Minimax-h3-Turbo` at `3ec17a324ced54151364f24f8b5fb6bf7e26414f`. The native loader verifies and merges this adapter into the Ref2VA projections. Accordingly, **H3 Ref2VA Turbo** is the model name used for the current avatar in this supplement. Legacy filenames containing “FastH3 optimized” describe inherited infrastructure, not the active adapter identity. Source and configuration provenance are recorded in [the method manifest](evidence/continuity-design.json).

## 2. Reference construction and state representation

### 2.1 Separate appearance, recent motion, and intended behavior

Let R denote the clean visual reference, Tₖ₋₁ the accepted audiovisual tail preceding segment k, and Pₖ the instruction compiled from the conversation and scene state. The logical generation contract is:

> **(1)** (Vₖ, Aₖ) = Gθ(Pₖ, R, Tₖ₋₁; εₖ).

Here θ denotes the selected H3 Ref2VA Turbo weights, εₖ the request noise, and Vₖ/Aₖ the candidate video/audio latents. This equation describes information flow; it is not a replacement for the implementation's packed reference blocks, separate AV schedules, and masks.

**Table 2. Inputs and their distinct responsibilities in the September 22 configuration.**

| Input or record | Construction | Intended responsibility | What it does not establish |
|---|---|---|---|
| Clean scene reference R | One authored image, supplied twice as ordered image-reference blocks | Face, hair, clothing, room composition, lighting and canonical props | Exact output pixels or successful object manipulation |
| Accepted visual tail | Raw generated latent suffix from the previous accepted segment | Recent pose, gaze, hand contact and visible scene state | A drift-free representation; inherited errors can also propagate |
| Accepted audio tail | Raw generated stereo latent suffix aligned with the overlap | Recent acoustic context and temporal continuation | Stable speaker identity or intelligibility by itself |
| Segment instruction Pₖ | Persona, known scene facts, current reply span and action phase | What should happen next, including behavior across boundaries | Proof that the requested action actually occurred |
| Quality anchor | Separately reviewed 960×544 RGB frame | A fixed baseline for CPU drift measurements | Model conditioning; it is not inserted into the generation request |

This separation resolves a practical conflict. The clean reference specifies **which person and scene should persist**; the accepted tail specifies **their most recent generated configuration**. Requiring the character to return to the reference pose after every segment would defeat motion continuity. Conversely, retaining only the generated tail would make the appearance reference progressively dependent on prior generated errors.

### 2.2 How the scene reference is actually used

The evaluated live profile uses `reference-interior-side-liaozhai-tea-v1.png`, an authored 1672×941 image. It depicts the selected interior side-view composition and provides the reference for the character, book, lotus-window setting and table arrangement. The Ref2VA input preparation downsizes references according to the output-area “match” policy and aligns dimensions to a 32-pixel grid; it does not upscale small inputs under this policy.

![Production scene reference, embedded from the exact input PNG. Zhiwei holds the open paper book beside a lotus window; the table contains incense, dragon-beard candy, bamboo slips, one celadon teapot and one teacup. This is the image supplied twice as Picture 1 and Picture 2, not a generated video frame or a three-view identity set.](figures/production-scene-reference.png)

The request contains **two copies of this same image**, named by order as Picture 1 and Picture 2. The prompt explicitly identifies Picture 2 as a repeat of Picture 1 and asks for stable appearance while continuing the incoming motion and pose. These are two conditioning blocks containing one distinct view; they are not a front/side/back identity dataset. The H3 semantic encoder receives the images and their condition labels, while the visual encoder produces the reference latents used by the joint generator.

Both blocks remain present on successive requests. Reuse of their encoded values preserves the same conditioning contract; it does not turn a reference into a forced first frame. The reference image is also distinct from `tea-quality-anchor.png`, the manually reviewed frame used only by the publication screen. Their separate hashes and dimensions are included in [the method manifest](evidence/continuity-design.json).

Current voice generation is prompt-only: no external audio-reference file is attached. The preceding generated audio tail is still present during continuation. Thus “no voice reference” means no external reference waveform, not absence of all audio conditioning.

### 2.3 Stable attributes and mutable scene state

The persistent reference is useful when the camera, room and outfit should remain stable. It can conflict with an intended permanent change: a reference showing an open window or an earlier outfit continues to condition subsequent generation after a request to change that state. The prompt and scene ledger can express the intended change, but they do not remove contradictory visual evidence automatically.

In the current public profile, the selected scene reference remains fixed. Automatic replacement with an action-specific image or a visually verified updated world state is not an established capability. A future scene-change protocol would need to coordinate the reference revision, intended state and accepted tail, then validate the transition. Merely recording “window closed” in the ledger does not demonstrate that the pixels show a closed window.

![Experimental closed-leaf reference from subsequent private action development. A built-in image-generation edit changes the near shutter while retaining the intended identity, camera and tabletop arrangement. This image is a requested state target; it is not evidence that H3 successfully performed the action. It has not replaced the production reference.](figures/experimental-window-reference.png)

The second image is an **experimental asset**, prepared after the report's fixed performance cohorts. It is shown here to make the reference inputs inspectable, not to add an action-success result to the evaluation. Its edit asks for only the near shutter to close, with the far leaf and visible pond opening preserved. Subsequent generated motion must still be reviewed for hand contact, unintended movement of other leaves, state retention and scene continuity. The [reference-image manifest](evidence/reference-images.json) records both source filenames, dimensions, byte hashes and the complete image-edit prompt. Both PNGs are embedded directly in this HTML at their original resolution.

## 3. Continuation method

### 3.1 Opening and continuation geometry

The opening has no preceding motion tail. It is generated from the selected reference and opening instruction, checked, and adopted as the first accepted state. Each subsequent request receives the persistent reference again, a new instruction, and the preceding accepted AV suffix.

**Table 3. Temporal geometry of the selected continuation protocol.** Latent temporal units are distinct from decoded video frames.

| Quantity | Value or rule | Interpretation |
|---|---|---|
| Output rate | 24 video frames/s | Playback time base |
| Generated window F | 124 frames | Full output geometry before continuation cropping |
| Overlap O | 5 frames | Context interval shared with the preceding segment |
| Newly published content | F − O = 119 frames | 4.958333 s after the opening |
| Opening content | 124 frames | 5.166667 s; no overlap removal |
| Saved video tail | 7 temporal latent slices | Bounded checkpoint storage; not all slices are used by five-frame continuation |
| Used video guide | Last 2 temporal latent slices | Native geometry corresponding to the selected five-frame overlap |
| Saved audio tail | 37 latent ticks per stereo channel | Bounded audio checkpoint storage |
| Audio overlap | 8 or 9 ticks at 40 Hz | Derived from absolute frame boundaries, rather than repeatedly rounding a nominal five seconds |
| Position policy | Window-relative target offset | Bounded local temporal coordinates with a separately advancing media clock |

For a previous window ending at frame e, the next window starts at e − 5. The audio overlap is computed as `round(e × 5/3) − round((e − 5) × 5/3)`. This accounts for the ratio between the 40 Hz audio latent clock and the 24 Hz video clock. After generation and decode, the repeated video/audio interval is removed before publication. The relay does not transmit the retained context as new footage.

The checkpoint stores raw generated AV state, not the previously encoded MP4. This avoids a codec round trip in the model's recurrence. It does not prevent generative errors from entering the next request through the latent tail.

### 3.2 A conditioned guide and an exact prefix serve different purposes

The preceding visual suffix is appended as a latent-guide reference block. Its conditioning uses `video_guide_timestep = 0.8`. In the implementation's rectified-flow convention, the selected guide is formed from clean latent values and dependent noise with the corresponding timestep label:

> **(2)** Gₖ = τTᵛₖ₋₁ + (1 − τ)εᵍₖ, τ = 0.8.

This adjustment changes the guide presented to the model; it does not modify the stored accepted tail or weaken the clean scene-image blocks. The value and its timestep label must change together. Controlled tail-replay experiments motivated reducing direct dependence on inherited fine detail: a degraded tail could carry abnormal texture into the next segment. The selected τ represents a continuity/appearance tradeoff, not a generic reference-image strength or a calibrated guarantee against drift.

Separately, `target_prefix = video` places the **unmodified accepted video suffix into the target overlap**. The sampler marks these target positions as clean and restores their values before the loop and after every denoising update. The controller then checks exact equality between the candidate's first two video latent slices and the accepted parent's last two slices.

The distinction matters: the guide supplies conditioning, whereas the protected target prefix constrains the inherited boundary. The current path enforces exact video-prefix equality. It carries audio context but does not enable the stronger `av` target-prefix mode used in a separate ablation. Temporal VAE decoding can depend on neighboring latent values, so exact latent equality alone does not imply identical decoded boundary pixels; a separate decoded-frame seam screen is required.

### 3.3 Acceptance, rollback, and publication

**Algorithm 1. One accepted-state continuation step.**

```text
Input: fixed reference R, accepted tail T, accepted last RGB frame,
       logical segment index k, scene state and pending reply/action
1. Compile the next instruction without ending the current action or answer.
2. Condition H3 Ref2VA Turbo on R, T and the new instruction.
3. Generate one window with the protected video prefix.
4. Validate AV geometry, exact video-prefix equality, decoded boundary,
   sampled appearance drift and expected speech audibility.
5. If accepted: save the raw AV checkpoint, publish new media, then advance
   the accepted generation state and corresponding application receipt.
6. If rejected: abandon the backend branch that already consumed the candidate.
   Restore the same accepted parent in a new branch and retry once with new noise.
7. If the retry also fails: stop publication; do not adopt the rejected tail.
```

The branch change is an implementation response to the backend's non-transactional tail update: by the time application screening runs, the backend has already advanced its own state. Restoring the accepted parent prevents that rejected candidate from becoming the starting point for the following segment. The logical media position and accepted AV parent remain unchanged during retry.

The selected controller has **no periodic clean-reference reset, crossfade, interpolation, or synthetic cutaway**. An earlier four-segment reset strategy reduced chain length but produced visible posture resets approximately every 20 seconds, contrary to the fixed-camera requirement. The current policy therefore preserves one logical chain and can stop when neither candidate passes. A failed-chain recovery that is both continuous and visually reliable remains unresolved.

### 3.4 Continuity at the application and playback levels

Physical continuity and conversational continuity require different records. A multi-segment answer retains its reply identity and continuation intent across video boundaries. Directed action phases carry entry and exit descriptions so that the next prompt does not automatically return both hands to the initial reading pose. The scene ledger tracks intended transitions and generated/playback receipts, with visual confirmation explicitly unverified.

The relay advances on accumulated frame and audio-sample counts. Model generation, accepted scene state and footage already shown to viewers are therefore separate clocks. This distinction prevents a future planned action from being described as already visible. It also explains why the existence of a valid reference image, a successful model request or a committed scene record cannot alone certify character consistency.

## 4. Experimental evaluation

### 4.1 Protocol and measurement definitions

We reanalyze the retained prefix-ablation records on CPU. No new GPU generation was performed for this supplement. The three arms use the same saved instruction/seed schedule and the same loaded backend, with 60 segments per arm. These historical trials use 960×544, 24 fps, four steps, a 124-frame window, a five-frame overlap, τ = 0.8, first-step BF16 projections/attention, and subsequent MXFP8/Sage steps. Each request contains one clean image reference and an external audio reference. The sequences primarily contain silent reading and page turning. They therefore differ from the later two-copy, prompt-only-voice, all-fast live configuration.

For each of 59 adjacent boundaries, seam error is the mean absolute RGB difference between the preceding segment's last decoded frame and the incoming segment's first published frame, in 8-bit units. This is a discontinuity diagnostic: real motion, exposure changes and different generated poses also affect it. It is not a face-identity or perceptual-quality score.

For each clip, the historical scan inspects the first frame, frame 60, and the last frame. Top black-band height counts contiguous dark rows from the upper edge under the retained pixel thresholds. Figure 2 reports the maximum of those three samples for each clip. No claim is made about unsampled frames.

Each arm is one temporally correlated trajectory, not 60 independent experimental replications. We report empirical distributions, medians and interquartile ranges. We do not attach independent-sample confidence intervals, significance tests, or population-level failure probabilities.

**Table 4. Descriptive boundary statistics from the prefix ablation.** IQR is the 25th–75th percentile interval over 59 boundaries. A dash denotes a prefix that was not constrained or equality-checked in that arm.

| Prefix policy | Clips / boundaries | Seam median | Seam IQR | Exact video prefix | Exact audio prefix |
|---|---:|---:|---:|---:|---:|
| Off | 60 / 59 | 7.824 | 5.781–9.864 | — | — |
| Video | 60 / 59 | 3.785 | 3.584–3.995 | 59/59 | — |
| Video + audio | 60 / 59 | 3.934 | 3.631–4.214 | 59/59 | 59/59 |

![Figure 1. Boundary-error distributions for three prefix policies. Each arm is one correlated trajectory with 59 boundaries. Boxes show the median and IQR; whiskers extend to observations within 1.5 IQR; all observations are overlaid. The empirical CDF retains every boundary.](figures/continuity-seams.svg)

Video-prefix protection has a 51.6% lower observed median seam error than the unprotected arm. This is a descriptive contrast for these trajectories, not a claim of 51.6% higher perceptual quality. Their trajectories diverge after the conditioning policy changes, so the experiment does not isolate every subsequent pose difference.

**Table 5. Sampled top-border behavior.** “Affected clips” means at least one of the three inspected frames contains more than three qualifying top dark rows.

| Prefix policy | Affected clips | Maximum sampled band height | Scope |
|---|---:|---:|---|
| Off | 21/60 | 23 pixels | One retained trajectory |
| Video | 0/60 | 0 pixels | Same scan and clip count |
| Video + audio | 0/60 | 0 pixels | Same scan and clip count |

![Figure 2. Top black-band height by clip index in the three 60-segment trajectories. Each point is the maximum over first/frame-60/last samples. Both protected arms remain at zero; zero sampled black bands is not an overall visual-quality verdict.](figures/continuity-black-bands.svg)

### 4.2 Long-run behavior and the role of clean references

Extending the earlier video-prefix chain to 180 segments preserved all 179 checked video prefixes and retained zero sampled black bands, yet visual review found late color/texture drift and ghosted fingers. The result separates **boundary preservation** from **long-run appearance fidelity**. A metric can remain favorable while another aspect of the scene deteriorates.

A later lotus-scene trial using two identical clean image-reference blocks completed 180 segments without periodic resets or retries, whereas its preceding single-reference trial failed screening at continuation 49. This supported retaining the two-block conditioning policy for the selected scene. The trials were not a randomized, replicated study, and they are distinct from the earlier failed 180-segment prefix extension. Their different configurations must not be pooled into one success rate. The later production session's brightness-drift stop also prevents interpreting the two-reference trial as an indefinite-consistency guarantee. These trial identities and limits are documented in the [method-source manifest](evidence/continuity-design.json) and the [main report's longer-run evidence](report.md).

**Table 6. Evidence supporting each consistency mechanism and its remaining limit.**

| Mechanism or diagnosis | Evidence | Supported inference | Remaining limit |
|---|---|---|---|
| Raw-tail propagation | Controlled clear/degraded-tail replays with other requested inputs fixed | Existing visual defects can propagate through the tail | Does not identify every source of the initial defect |
| Exact video prefix | 59/59 checked boundaries in the video arm | The inherited latent boundary is preserved under the tested contract | Decoded appearance and future content can still change |
| Clean reference repetition | Separate single-/two-block lotus-scene trials | Two blocks were useful in the observed scene | No universal optimal reference count or hours-long guarantee |
| Publication screen | Fixed-anchor drift checks, seam checks and a real rejected continuation | Some visibly degraded candidates are withheld | Three-frame sampling misses some errors; no semantic object verifier |
| Accepted-parent retry | Controller restores a hashed raw AV checkpoint before retry | Rejected candidate state is not reused as the accepted parent | An already marginal accepted tail may fail again |
| Application state | Reply/action continuity and explicit scene receipts | Prompts can remain consistent across segment boundaries | Intended state is not automatic visual confirmation |

## 5. Limitations and interpretation

The system preserves several invariants exactly: segment ordering, selected latent-prefix values, accepted-tail provenance, and the progression of frame/sample clocks. It maintains face, clothing, lighting, props and voice through conditioning and screening, whose guarantees are weaker. These two classes of consistency should be reported separately.

Motion context is a bounded summary of recent generated content, including its appearance errors. Clean references do not overwrite those errors pixel by pixel. Window-relative positional encoding bounds temporal coordinates but does not repair the content carried by the recurrence. Likewise, passing a global color or black-band check does not establish that a hand, book title, window state or speaker identity is correct.

The supported conclusion is therefore limited: **reference-conditioned H3 Turbo, protected video overlap and accepted-state rollback improve the evaluated continuation behavior and provide an auditable streaming protocol. Indefinite character/scene fidelity remains unproven.** Further claims would require replicated long runs, varied actions and reference scenes, human inspection, and direct evaluation of identity, object state, speech continuity and lip synchronization.

## 6. Reproducibility and relation to the system report

The supplement uses the already recorded experiments and the September 22 runtime configuration. It changes documentation and CPU analysis only. Source documents with older “current” or “live paused” statements are treated as dated historical records; the selected configuration is specified by the report's runtime evidence.

- [Method and reference manifest](evidence/continuity-design.json): model identity, reference roles, temporal geometry and hashes of the inspected implementation and historical documents.
- [Sanitized per-segment ablation data](evidence/continuity-ablation.json): all 180 segment records and 177 boundary values used in Figures 1–2, without dialogue, media files or latent tensors.
- [Figure generator](build_continuity.py): reproduces the statistical figures on CPU.
- [System report](report.md): the existing technology stack, latency measurements and optimization history; those results are not repeated here.

```bash
python build_continuity.py
python build_report.py
```
