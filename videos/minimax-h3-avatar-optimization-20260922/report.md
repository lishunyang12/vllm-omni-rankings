# From 15-second generation to a five-second interactive H3 avatar

**Current system, optimization history, Nsight evidence, and the decisions behind the demo**  
Engineering review · 22 September 2026 · Eight SM120 / GB202 GPUs

[Complete historical appendix](history.md) · [Evidence inventory](evidence/history-inventory.json)

## 1. What this report establishes

The project first demonstrated that a roughly 15-second audiovisual clip could be generated in less than its playback duration. It then changed the workload into an interactive avatar: a continuing character must answer newly arriving messages, preserve motion and scene state, maintain intelligible speech, and deliver a continuous Twitch stream. These are different optimization objectives.

The current production path generates a **124-frame window at 24 fps**, retaining five overlapping frames and publishing **119 new frames, or 4.958333 seconds**, after the opening. It uses native H3 joint audio/video generation, a four-step Ref2VA Turbo adapter, dense Sage attention, resident MXFP8 main projections, an NVFP4 video decoder, and an FP32 audio decoder. The answer planner is a separate local Qwen3-4B model. No per-answer TTS or external voice-reference waveform is required in the selected prompt-only voice mode.

At the report's fixed production snapshot, **102 accepted continuations** had a **3.972 s publication median, 4.205 s P95, 4.807 s maximum**, and no request exceeded its 4.958 s playback budget. The relay had transmitted 101 clips with **zero recorded buffer underruns**. This includes 50 spoken clips; their publication median was 3.977 s. It is a bounded observation of the current mixed live workload, with the text service sharing the host, not an isolated benchmark or an unlimited-duration guarantee. [Current receipts](evidence/current-live-observation.json)

The later checkpoint at **12:34:41 UTC** extends the same session to **456 accepted continuations**, including 163 spoken clips. It records **two producer deadline misses** and **two relay waiting intervals totaling 1.833 seconds** across 455 transmitted clips. Thus the early zero-underrun result does not hold for the entire longer session. Both observations are retained, with their different workload mixes. [Extended receipts](evidence/extended-live-observation.json)

The previous session stopped after clip 871: both attempts at the next continuation failed the luminance-drift screen. The controller withheld those candidates, and the stream went offline. The newly authorized restart uses a fresh opening and the original quality checks. This failure belongs in the engineering result: adequate throughput has been demonstrated, but indefinite scene stability and automatic recovery have not.

The report follows the architectural depth and evidence discipline of the supplied [MiniMax H3 analysis report](https://github.com/lishunyang12/vllm-omni-rankings/blob/main/videos/minimax_h3_analysis_report.html). Its measurements here come from this project's local experiments and receipts. Earlier model-level comparisons are background; they are not multiplied into a single advertised speedup.

### Three separate meanings of “real time”

| Question | Required measurement | Current conclusion |
|---|---|---|
| Can generation keep up with playback? | Producer work divided by newly published media duration; include retries and tail latency | Yes in the bounded current sample |
| How quickly does a viewer get a response? | Arrival → selection → generation → first response sample transmitted → viewer presentation | Local components are instrumented; full viewer-end onset distribution is not established |
| Can the avatar remain coherent indefinitely? | Long-run identity, lighting, objects, speech, motion and failure recovery | Not established; a real brightness-drift stop was reproduced |

A stream can have a generation ratio below one and still respond slowly because it has queued future footage. It can also be fast and visually wrong. These are independent acceptance dimensions.

## 2. The operator's decisions, reconstructed from requirements and evidence

The requirement column paraphrases explicit development requests. The engineering rationale is an interpretation of those requests together with the experiments; it is not a claim to know an unstated motivation.

| Requirement or observation | Decision made for the demo | Evidence and cost |
|---|---|---|
| The earlier FastH3 path already produced a 15-second clip in about 13 seconds | Preserve that as a throughput reference; adapt the serving system for interaction | Warm complete-MP4 measurements reached 13.549 s, then separately qualified 13.304 s and 13.261 s configurations; these are different campaigns |
| Viewers should be able to influence what happens next | Move from long completed clips to roughly five-second rolling Ref2VA windows | Shorter commitment horizon; fixed preparation, reference and output costs recur more frequently |
| The character's face and scene matter more than a headline resolution | Select 960×544 for the live baseline while testing larger shapes separately | BF16-first 1088×608 took 5.611 s per 4.958 s continuation; the then-requested 4.5 s target was not achieved |
| Protecting the first diffusion step visibly helped quality | Introduce original BF16 first-step projections and attention | Longer quality retention in one soak; 4.492 s versus 3.888 s median in a paired configuration family; drift was delayed, not eliminated |
| Real-time headroom remained tight; explicitly try Sage/MXFP8 in step one | Temporarily use all four low-precision steps in production | Live idle median 4.575 → 3.857 s in the historical operational comparison; this is a lossy tradeoff, not an infrastructure-only gain |
| “Keep weights resident instead of moving them” | Test original BF16 residency and on-demand MXFP8 weight packing; revert after measurement | Large weight copies disappeared, but complete request median increased 5.611 → 5.704 s |
| “Can we remove MXFP8 entirely?” | Test resident original BF16 projections separately from attention precision | 960×544 all-BF16 projections still took 5.060 s median; 1088×608 took 6.568 s with mixed attention or 7.343 s with BF16 attention |
| Do not interrupt a turn by looking back at the book | Keep one answer identity and viewer attention across its multiple generated segments | Whole-answer scheduling and prompt corrections; a video boundary is not a conversational ending |
| Voice references seemed to add complexity and unnatural speech | Test and select prompt-only native H3 voice after review | Removes the TTS/reference-input dependency; a three-clip trial took 4.049–4.161 s including sample preparation; voice stability remains a separate quality question |
| Avoid constant BGM; keep environmental and action sound | Preserve native generated ambient PCM; use accompaniment only for explicit singing | Ordinary reading/dialogue has no added music bed; singing uses H3-authored accompaniment instructions |
| Make the response faster without degrading the picture | Reduce lookahead from two clips to one, bypass the LLM for supported direct gestures, cache unchanged inputs, remove duplicate validation decode | Lower queue delay and exact work reduction; less reserve to absorb retries |
| Try approximately two-second generation | Implement and test 56/73-frame experimental shapes; retain 124 live | 56-frame mixed-workload mean exceeds its 2.125 s playback duration; speech fragmentation and unintended subtitles also appeared |
| Keep the camera and movement continuous | Carry raw AV latent state; preserve the accepted prefix; retry from the same accepted parent | Avoids periodic reference-image cuts; does not eliminate accumulated model error |
| Add mirror, cloth, water, clothing and window interactions | Reuse a stateful planner, but keep unqualified recipes in a private research profile | An eat → wipe → read sample exists; window control and mirror reflection still fail visual acceptance |

The resulting operating point is a deliberate compromise: preserve exact engineering optimizations, use explicitly accepted approximate model kernels where necessary, reduce queueing, and reject visibly deteriorating output. It is not an all-lossless model or a fully solved persistent world.

## 3. The 15-second → five-second migration is a workload change

| Dimension | Earlier throughput demonstration | Current interactive avatar |
|---|---|---|
| Main task | FastH3 T2VA / VSA long clip | H3 Ref2VA Turbo rolling continuation |
| Video geometry | 1280×704, 362 frames, 24 fps | 960×544, 124 source frames; 119 new continuation frames |
| Media duration | 15.0833 s for the 362-frame output | Opening 5.1667 s; each continuation 4.9583 s |
| Attention | VSA sparsity plus Sage in the selected fast path | Dense Sage; all eligible KV blocks retained |
| Conditioning | Authored long-clip prompt | Fixed character reference, new line/action prompt and previous raw AV tail |
| Scheduler | Complete a standalone request | One logical continuous session with bounded future footage |
| Deadline | Finish a long MP4 within its playback duration | Publish the next accepted continuation before its playback slot |
| Output consumer | Benchmark/player | Persistent relay, public chat, continuous encoder, Twitch player |
| Quality objective | Accepted finite sample | Face, voice, lighting, props and actions across many requests |

It would be misleading to divide a historical 13.3 s long-clip result by a current 4.0 s short-clip result and call that a 3.3× infrastructure speedup. Resolution, duration, conditioning, task partition, attention policy and timing boundaries all changed.

The practical gain is a smaller **commitment horizon**. Once a diffusion request has started, the system does not rewrite its middle with a newly received comment. A shorter window permits a new request sooner. That benefit must be paid for with enough throughput margin to cover repeated conditioning, decoding and publication.

For continuation window length `F`, overlap `O=5`, and `fps=24`:

```text
new_media_seconds D = (F - O) / fps
producer_ratio     = total producer work / total new media
mean headroom      = D - mean producer time
viewer latency     = admission + planning/scheduling + generation/checks
                     + ready-to-play backlog + within-clip response onset
                     + Twitch delivery/player buffering
```

Planning and generation can overlap, so their complete durations should not automatically be added as serial work. A turn's dispatch time is constrained by both input readiness and generator availability. Increasing the buffer hides occasional overruns; it does not fix average production slower than playback.

## 4. Current complete technology stack

### 4.1 Request-to-viewer path

![Current live architecture and the accepted-state feedback loop](figures/architecture.svg)

```text
Twitch public chat / !ask
    ↓
Listener → SQLite inbox (deduplication, timestamps, expiry)
    ↓
Direct supported-action route OR local Qwen3-4B text planner
    ↓
Reply queue + character/scene facts + answer segmentation
    ↓
Bounded next-window scheduler (one-clip lookahead)
    ↓
Timed H3 prompt + two clean image reference blocks + accepted raw AV tail
    ↓
vLLM-Omni diffusion service
    ├─ semantic/reference conditioning and exact caches
    ├─ four joint audiovisual DiT evaluations across eight GPUs
    ├─ parallel tiled video VAE + FP32 audio VAE
    └─ byte packing, host transfer, MP4/PCM outputs
    ↓
Full AV validity + drift samples + seam/prefix + speech-audibility checks
    ↓ accepted only
Atomic media receipt + raw-tail checkpoint + scene/queue commit
    ↓
Persistent FFmpeg relay: native PCM → resample → one final broadcast AAC
    ↓
Twitch ingest → HLS/player → viewer
```

### 4.2 Components and responsibilities

| Layer | Current implementation | Why it is present |
|---|---|---|
| Character and environment | Zhiwei lotus-study persona; authored scene facts; reference image and timed prompts | Stable identity, grounded object descriptions and repeatable behavior |
| Input transport | `avatar_chat_inbox.py` | Receive public messages without blocking GPU generation |
| Language planning | `avatar_text_worker.py`, `avatar_text.py`, `avatar_zhiwei_dialogue.py`; local Qwen3-4B-Instruct-2507 | Convert dialogue into a reply and supported action; avoid a heavyweight remote planning dependency |
| Direct action compiler | `avatar_zhiwei_actions.py` and public action routing | Familiar commands such as nod/wave/read can bypass a model call |
| Durable work queue | `avatar_agent_replies.py`, SQLite WAL | Deduplication, claim/generate/air timestamps, bounded queue and multi-segment replies |
| Production scene state | `avatar_behavior.py` scene ledger | Intended scene and generated/playback receipts; not automatic visual recognition |
| Extended interaction research | `avatar_world/*`, separate `interaction_lab` profile | Object occupancy, action prerequisites, cancellation and wardrobe/scenery versions; not all publicly deployed |
| Short-window producer | `generate_short_avatar.py` | Select the next turn/idle action, call H3, publish atomically and follow relay progress |
| Continuous camera controller | `avatar_continuous.py` | Save accepted latent tails, validate prefix identity, screen seams, restore same parent for one retry |
| Model API | vLLM-Omni local diffusion endpoint | Keep expensive weights and communicators resident across requests |
| H3 semantic encoder | Qwen3-VL-derived H3 conditioning encoder, TP8 | Model conditioning; distinct from the 4B conversational planner |
| H3 generator | Ref2VA partition plus fused four-step Turbo adapter | Jointly generate speech, face/body motion and scene evolution |
| Tensor kernels | PyTorch/Triton, FlashInfer/CAKE, CUTLASS SM120 block-scaled GEMMs | Attention, projection, fusion and communication execution |
| DiT communication | FlashInfer PCIe/RDMA Ulysses, TP1/SP8, BF16 transport | Split the token sequence across eight GPUs while preserving communication precision |
| Video reconstruction | Native tiled video VAE, PP8; NVFP4 decoder linear projections | Reconstruct all frames with parallel tile scheduling |
| Audio reconstruction | Native audio VAE, FP32; float PCM sidecars | Keep original generated waveform available through validation and relay |
| Packaging | GPU RGB byte conversion; pinned host transfer; chunked CPU/PyAV output | Overlap output handling with upstream reconstruction |
| Broadcast | `twitch_fresh_queue.py`, persistent FFmpeg, RTMP/HLS | Continuous audiovisual clock and fresh clips; no prerecorded loop |
| Observability | `avatar_metrics.py`, Prometheus, Grafana, receipt/log analysis | Current-session generation, queue, playback, failures and hardware status |
| GPU diagnostics | Nsight Systems CUDA/NVTX/selected hardware captures | Locate compute, transfers, synchronization and host-submission gaps |
| Reproducibility | Runtime lock, source manifest, selected environment, media/checkpoint hashes | Bind performance observations to an actual execution path |

### 4.3 Locked production choices

| Setting | Selected value | Interpretation |
|---|---|---|
| Hardware | Eight GB202 GPUs, SM120; roughly 71.1 GiB exposed device memory each in the retained trace | PCIe/RDMA system, not an NVLink B200/H200 result |
| DiT / encoder / video VAE | TP1/SP8 / TP8 / PP8 | Different components use different parallel decomposition |
| Canvas and video rate | 960×544 at 24 fps | Selected live budget, not a resolution enhancement claim |
| Diffusion evaluations | Four | Distilled Ref2VA adapter; not original dense H3's long schedule |
| Video/audio scheduler shifts | 12 / 3.0 | Two noise schedules inside joint AV generation |
| Window / overlap | 124 / 5 frames | 119 new frames after opening |
| RoPE position mode | Window-relative | Keeps the local temporal coordinate bounded while the media clock continues |
| Main projections | GPU-resident MXFP8; four low-precision steps | A8W8 block scaling; original BF16 preservation remains available for experiments |
| Attention | Dense Sage in every step | Low-precision attention, but no FastH3 VSA pruning in the current Ref2VA path |
| Wire precision | BF16 | E4M3 QKV transport is disabled |
| Video / audio VAE | NVFP4 / FP32 | Video-only decoder approximation; audio decoder not quantized to NVFP4 |
| Clean reference blocks | Two copies of the same reference | Retains the tested conditioning arrangement; cache avoids redundant encoding rather than removing a block |
| Visual-tail guide timestep τ | 0.8 | Applies to the generated visual guide; distinct from the clean image references |
| Target-prefix preservation | Video prefix enabled | Exact accepted video prefix plus raw AV continuation state |
| External audio reference | None in prompt-only mode | Voice is described in the prompt; H3 generates the audio |
| Future footage | One-clip lookahead | Lower ready-to-play delay, less retry reserve |
| Broadcast format | H.264, 960×544, 24 fps; AAC stereo 48 kHz | Source audio is 32 kHz PCM and resampled for transport |
| Relay target bitrate | Video 5000 kb/s, audio 192 kb/s | Encoding target, not guaranteed instantaneous measured bitrate |

The configuration field `quality="lossless"` refers to the runtime's approximate-cache policy. It does **not** turn MXFP8, Sage, NVFP4 or four-step distillation into lossless computation. Likewise, `--cache-backend none` does not mean AdaLN and exact input caches are absent: they are separate mechanisms.

The backend environment reports Python 3.12.13, PyTorch 2.13.0+cu132, Triton 3.7.1, vLLM 0.28.0, vLLM-Omni 0.28.0rc2.dev62+gad1bb0f97, FlashInfer 0.6.16.post3, Transformers 5.14.1, Diffusers 0.40.0 and PyAV 18.1.0. These are installed distribution versions; local native and FlashInfer overlays are also active. The selected backend source-manifest SHA256 is `a53e5edb008f3b0b2ffe39feafd6706121e7a8ac0b86f3bd1837c4e40b2c6873`, which is more specific than a package version. [Version metadata](evidence/software-versions.json)

[Selected configuration and measurements](evidence/measurements.json) · [Actual accumulated backend projection counters](evidence/backend-dispatch-snapshot.json)

## 5. What carries over, and what does not

### 5.1 Exact AdaLN caching

H3's modulation tables depend on model weights, the exact timestep schedule and modality/time mapping, rather than the evolving video hidden state. Repeatedly evaluating the large per-block modulation projections is unnecessary when those dependencies are fixed. The native path loads validated precomputed tables and removes the online modulation projections. This is useful for original H3 as well as distilled schedules; the cache contents are still model- and schedule-specific.

The current Ref2VA admission checks require a loaded exact AdaLN cache and cached final modulation. A valid cache must match the fused adapter, task partition, video/audio schedules, dtype and indexing contract. A FastH3 table cannot be assumed valid for Ref2VA just because both use four evaluations. Changing scheduler guidance inputs without regenerating or validating the cache would violate that contract.

This is an exact reuse optimization, distinct from approximate reuse of a denoising block across different steps. The latter is not part of the selected live baseline. The current local optimized kernel admission is SM120/TP1/SP8-specific; that implementation restriction should not be confused with the broader semantic applicability of AdaLN caching.

### 5.2 Adapter fusion and persistent model state

The Ref2VA Turbo factors are merged into native projection weights before steady-state generation. The loader validates the full target set, adapter rank/scale, native per-head QKV layout, and gate/value ordering of FC1. Quantization is applied to the fused original weights, not to an unadapted base and not by reconstructing a supposed BF16 reference from quantized values.

The current integration validates 208 fused native parameters, including the two context-refiner blocks. Its main-stack MXFP8 path covers 200 projection modules across 50 blocks: packed QKV, attention output, gate/up and down projections. Main-step dispatch counters verify each real projection and the fused FC2 calls. Counters accumulated over many requests demonstrate execution; they are not a fresh per-request latency experiment.

Weights, communication registration and reusable buffers stay alive between clips. Model load time is outside the warm serving numbers. The historical VDN loading phase took about 1,181 seconds on shared storage, illustrating why repeatedly starting a model is unsuitable for interactive serving. This is not a claim that the current warm request includes that delay.

### 5.3 MXFP8: distinguish weights from activations

In the selected path, fused BF16 weights are quantized to MXFP8 **at initialization** and the packed payload/scales remain on the GPU. Input activations are quantized during each projection. “Online quantization” is ambiguous unless these two operations are distinguished.

MXFP8 here uses E4M3 payload values with block-shared E8M0 scales over 32 elements. The GEMM path is approximate relative to the original BF16 model. Eliminating repeated weight packing while retaining the same packed values is an exact scheduling/storage improvement relative to that MXFP8 baseline.

The BF16-first experiment preserved original BF16 projections in pinned host memory with double-buffer GPU prefetch. Turning step one back to MXFP8 removes that step's BF16 projection work and staging from its execution path. The original-weight facility can remain initialized for other requests; its existence alone does not prove large H2D transfers occur in the current all-MXFP8 request.

### 5.4 Fused SwiGLU and activation packing

The historical fusion combines the SwiGLU producer with the MXFP8 activation packing consumed by FC2. It eliminates intermediate memory traffic and launches while preserving the installed activation implementation's rounding sequence. The accepted implementation explicitly preserves BF16 rounding after SiLU and after multiplication; replacing it with an algebraically similar sigmoid expression did not preserve quantized payload bytes.

A real-shape microbenchmark reduced producer time from 1.4015 to 0.7981 ms, and producer plus the same FC2 GEMM from 5.9671 to 5.3568 ms. The complete model improvement was about 130 ms in that campaign, not the sum of arbitrary microbenchmark speedups. An alternate lookup-table implementation passed numerical checks but did not improve performance further.

The current Ref2VA path retains this FC2 fusion. It does not enable VSA-only QK/V split schedules: the admission code explicitly rejects those flags for dense Ref2VA. A historical FastH3 optimization inventory is therefore not a list of switches that can all be turned on for the avatar.

### 5.5 Attention, communication and producer placement

Sage reduces attention arithmetic precision. It is a quality tradeoff. The FlashInfer/CAKE kernel name includes “block_sparse,” but the current Ref2VA adapter keeps the complete dense KV set. A sparse-sounding kernel symbol is not evidence of pruning.

BF16 Ulysses exchange is retained. Persistent registered RDMA buffers avoid repeated registration/teardown; the historical matched comparison improved 17.9214 → 17.5780 s mean, while increasing retained memory. Q/K and output producer-direct paths reduce layout/materialization work around communication. QK norm/RoPE fusion preserves the intended normalization and positional transformation while reducing intermediate work.

Earlier FastH3 VSA experiments also overlapped gate/coarse work and pipelined four output chunks. Some of these depend on VSA layout and gate semantics and are **not** current dense Ref2VA features. The report records their historical value without silently transferring their speedup to the new task.

## 6. The weight-residency decision: what Nsight actually showed

A reasonable hypothesis was that keeping original BF16 weights on the GPU should eliminate a large transfer penalty. The experiment did eliminate the transfers. It did not reduce the complete request.

| 1088×608, BF16-first four-step comparison | MXFP8 resident + BF16 prefetch | BF16 resident + online MXFP8 packing |
|---|---:|---:|
| Unprofiled HTTP→MP4 median, nine requests | 5.610958 s | 5.704375 s |
| Unprofiled maximum | 5.736889 s | 5.814680 s |
| Large original-weight H2D copies per rank | 200 | 0 |
| Original-weight bytes per rank/request | 38,535,168,000 | 0 |
| Sampled step-one CPU NVTX median across ranks | 1544.952 ms | 1464.517 ms |
| Sampled step-two CPU NVTX median | 913.879 ms | 952.357 ms |
| Sampled step-three CPU NVTX median | 910.386 ms | 950.036 ms |
| Sampled step-four CPU NVTX median | 911.156 ms | 951.626 ms |
| Sampled complete DiT CPU NVTX median | 4312.628 ms | 4349.858 ms |

All 13 compared MP4 hashes matched in the original residency experiment. The new candidate reduced step-one time by roughly 80 ms, while the next three steps gained about 119 ms of work from on-demand weight packing. Complete request median regressed by 93.4 ms, about 1.7%.

![Weight residency: paired unprofiled requests and separate diagnostic NVTX scopes](figures/weight-residency.svg)

The original copies totaled roughly 1.42–1.44 GPU seconds **per device**, but prefetch already overlapped much of their execution with other work. Eight-device aggregate memcpy time is not a serial critical path. The resident candidate still had input transfers; for example, rank/device zero retained a roughly 11.54 MB large input transfer. “No weight staging” is accurate; “no H2D at all” is not.

The follow-up all-BF16 tests also failed the live deadline:

| Follow-up | Median HTTP→MP4 | Formal requests beyond 4.958 s |
|---|---:|---:|
| 1088, original prefetch baseline rerun | 5.5833 s | 9/9 |
| 1088, BF16 resident + later online MXFP8 | 5.6996 s | 9/9 |
| 1088, all BF16 projections, mixed attention | 6.5682 s | 9/9 |
| 1088, all BF16 projections and attention | 7.3434 s | 9/9 |
| 960, all BF16 projections, mixed attention | 5.0597 s | 9/9 |
| 960, BF16 resident + later online MXFP8 | 4.4268 s | 0/9 |

These controls justify retaining resident MXFP8 in the demo. They do not prove that every future kernel or memory placement will favor the same choice. [Recomputed Nsight results](evidence/nsys-reanalysis.json) · [Unprofiled samples](evidence/measurements.json)

## 7. Video reconstruction and output: exact improvements around a lossy decoder

The selected native video decoder uses NVFP4 linear projections. That precision choice is approximate; it must be separated from the exact scheduling work around it.

| Mechanism | What it removes or overlaps | Evidence and present interpretation |
|---|---|---|
| PP8 tiled VAE | Distributes reconstruction across devices | Retained; actual geometry and call-count audits matter |
| Pair/mixed/batch4 scheduling | Groups compatible temporal/tile work without changing the accepted ordering contract | Retained where current shapes are admitted; bigger batches were not automatically faster |
| Video/audio decode overlap | Independent decoders can execute after joint denoising | Retained; separate CUDA streams do not remove all CPU launch serialization |
| Early/asynchronous tile gather | Transfer completed tiles while later tiles decode | Retained; early-gather-only benefit was small and not stable in every test |
| Cached normalization constants | Avoid repeated construction/transfer of identical constants | Byte-equivalent qualified paths; retained |
| GPU blend/RGB packing | Fuse spatial blending and compact byte output; reduce FP32 intermediate traffic | Historical full-model MP4s matched; retained |
| Pinned double-buffer output | D2H and CPU video encoding run as a bounded pipeline | Retained; event/lifetime/backpressure handling remains required |
| Early AAC | Encode native audio while the video decoder is still running | Historical matched 15-second campaign: 13.551 → 13.304 s median with cached constants |
| Shared decode for checks | One complete AV decode also supplies quality samples and endpoint frames | Matched short-clip campaign: 5.044 → 4.774 s median, unchanged request and MP4 hashes |

The historical GPU RGB postprocess benchmark improved 2.84225 → 0.19347 ms per representative frame block. The full 15-second request comparison improved only 13.334 → 13.261 s median. The component gain is real, but upstream overlap and other work constrain end-to-end benefit.

![Independent matched long-clip optimization campaigns; gains are not cumulative](figures/matched-campaigns.svg)

In the Config-A hardware trace, AAC ran at approximately 12,948.7–13,123.1 ms from the trace request origin while video-VAE GPU work continued until about 15,417.8 ms. That establishes overlap. It does not establish that all AAC CPU time is additional serial savings. Output-submit ranges had no `cudaStreamSynchronize` calls in the audited trace.

Changing video-VAE precision cannot repair a bad upstream video latent. In a later quality diagnosis, degraded generated latents remained visibly degraded under FP16 reconstruction. NVFP4 introduced measurable differences, but it was not the sole cause of long-run degradation. The audio VAE stayed FP32; attributing extra spoken words to the video decoder would be inconsistent with the pipeline.

## 8. Exact input reuse makes the five-second workload affordable

The avatar repeatedly uses the same clean picture and often the same idle instruction. Encoding those identical inputs every five seconds is avoidable. At the same time, newly authored dialogue must remain fresh.

The exact reference cache uses content/configuration identity, ordered RGB inputs, prompt/tag identity and component identity; distributed ranks agree on a hit. Returned tensors are isolated from mutation. It preserves the two reference blocks and their positional role, rather than deleting a block to save time. The reference-image latent cache can still hit for a new spoken line even when semantic text conditioning must be recomputed.

### Four retained profiles, one instrumented request per condition

These are **rank-zero CPU NVTX durations**, in seconds, from the BF16-first cache experiment. They are not a fresh trace of the current all-Sage production mode.

| Range | One reference, cache off | Two references, off | Two references, hit | Two references, new speech |
|---|---:|---:|---:|---:|
| Request preparation | 0.562 | 0.876 | 0.117 | 0.476 |
| ↳ semantic encoder | 0.258 | 0.368 | 0.003 | 0.373 |
| Four-step DiT | 3.639 | 3.803 | 3.658 | 3.946 |
| ↳ first step | 1.589 | 1.593 | 1.455 | 1.726 |
| Decode range | 0.506 | 0.507 | 0.509 | 0.509 |
| ↳ video VAE | 0.461 | 0.466 | 0.468 | 0.469 |
| ↳ audio VAE | 0.026 | 0.024 | 0.024 | 0.024 |

Nested rows must not be added twice. Differences in DiT timing across these single captures are not automatically a causal benefit of the input cache. Across-rank medians in the new SQLite reanalysis differ from the rank-zero values here because other ranks include different waiting intervals.

The cache validation compared video latents, audio latents and complete MP4 hashes. Four reading pairs matched exactly; a subsequent OFF/ON/OFF experiment for reading, speaking and page turning also matched in all nine generated clips. This establishes tested-input equivalence, not a formal proof over all possible inputs.

The historical unprofiled 60-clip cache run had 57 formal samples: median 4.554 s, mean 4.671 s, P95 5.336 s, eight deadline misses. Relative to a preceding two-reference run's 5.289 s median, the operational reduction was about 0.735 s. Because these were different runs, that complete difference is not a controlled single-factor speedup. The same candidate showed an unrequested drinking action and was not accepted merely because it was faster.

## 9. Current measured latency and response budget

### 9.1 Fixed production snapshot

The new session began on 2026-09-22 at 11:57:03 UTC according to Twitch. The statistics below freeze the first 102 accepted continuations at 12:05:24 UTC, exclude the opening, and include any retries. No retries occurred in that snapshot. Percentiles use linear interpolation; the small sample is not a production SLO study.

| Workload | Samples | Publication P50 | P95 | Maximum | Over playback budget |
|---|---:|---:|---:|---:|---:|
| All continuation actions | 102 | 3.972 s | 4.205 s | 4.807 s | 0 |
| Quiet rest | 46 | 3.667 s | 4.267 s | 4.807 s | 0 |
| Speaking | 50 | 3.977 s | 4.204 s | 4.290 s | 0 |
| Page turn | 5 | 3.703 s | 4.066 s | 4.067 s | 0 |
| Wave | 1 | 4.007 s | Not a distribution | 4.007 s | 0 |

Mean producer time was 3.901 s, giving a producer ratio of approximately **0.787** and mean headroom of **1.057 s** per continuation. Maximum observed headroom was not the acceptance criterion; the slowest request had only about 0.152 s spare.

![Current producer budget and observed latency by workload](figures/current-latency.svg)

### 9.2 A measured boundary breakdown, not an invented GPU waterfall

| Measured boundary | Mean | P50 | P95 | Meaning |
|---|---:|---:|---:|---|
| HTTP request | 3.782 s | 3.852 s | 4.083 s | Backend request, including its preparation, model, reconstruction and response work |
| Everything outside HTTP inside producer step | 0.119 s | 0.120 s | 0.126 s | App preparation, complete decode/checks, checkpoint/receipt/publication work |
| Total producer publication | 3.901 s | 3.972 s | 4.205 s | Accepted continuation ready for the relay |

Means of these complementary boundaries add by construction. Their independently computed medians and percentiles do not necessarily add. The backend's aggregate field named `denoise_step_latency_ms` is not a substitute for a measured DiT-only trace.

Eliminating all 0.119 s of outside-HTTP work would be an impossible zero-cost-check idealization and would save only about 3% of current mean producer time. The main compute opportunities remain inside the backend. Conversely, reducing future footage can improve perceived response more than a small kernel improvement without changing model throughput at all.

### 9.3 Viewer response is a different clock

The local ledger records message arrival, text/reply readiness, selection, generated media and local relay start. A historical 30-clip one-lookahead sample measured a 1.242 s median from ready media to local playback. That excludes the text call, time waiting for a generation slot, earlier queued turns, within-clip first syllable and Twitch/player buffering.

Direct actions bypass the text model when routing is unambiguous. New dialogue uses the local 4B planner. Nine private action-planning inputs took 0.265–0.744 s in one small test, but that does not imply the avatar visibly acts within a second. A request arriving just after a window has started still waits for the next available generation decision.

There is no measured universal viewer-end response number in this report. A proper test must correlate one actual viewer message with the first meaningful visual/audio response in captured playback. A generic idle blink is not a response to the question.

### 9.4 Extended observation during report assembly

At 12:34:41 UTC, the same session contained 456 accepted continuations: overall publication mean 3.841 s, median 3.672 s, P95 4.115 s, maximum 5.895 s. Speaking alone remained at 3.976 s median and 4.193 s P95 over 163 clips. The lower overall median primarily reflects the larger quiet-reading share; it is not a newly deployed optimization.

Two accepted quiet clips exceeded the 4.958 s deadline, at 5.895 and 5.622 s. The relay reported two waiting intervals totaling 1.833 s, with 455 clips and 2,256.25 seconds of generated media transmitted. Its sampled AV clock difference was 8.333 ms. Those are clock/queue observations, not a perceptual lip-sync assessment. No cause for the two backend spikes is assigned without a matching trace. The live service remained active at this checkpoint. [Extended observation](evidence/extended-live-observation.json)

## 10. Why five seconds remains the live choice

The project implemented shorter shapes rather than assuming duration could be changed with one API parameter. It adapted temporal geometry, raw-tail metadata, AV clocks, decoder dispatch audits, prompt timing and speech capacity in an isolated backend.

| Candidate | New media | Formal mixed-workload clips | Ready median | Mean | P95 | Deadline misses |
|---|---:|---:|---:|---:|---:|---:|
| 56-frame window | 2.125 s | 17 | 2.289 s | 2.204 s | 2.486 s | 10/17 |
| 73-frame window | 2.833 s | 14 | 2.425 s | 2.553 s | 2.856 s | 4/14 |
| 124-frame window | 4.958 s | 13 | 3.706 s | 3.837 s | 4.129 s | 0/13 |

The 56-frame reading median was only 1.925 s, but new dialogue took about 2.28–2.48 s. A demo spends real time speaking, so selecting only idle performance would conceal an unsustainable workload. The 73-frame mean had margin, but all four speech windows were slightly slower than their playback budget. Neither short candidate completed the longer playback and quality qualification needed for production.

![Short-window experiments normalized to their own playback deadline](figures/window-budgets.svg)

The shorter text budget also split one tea name across two clips; an energy-based boundary proxy measured a roughly 1.385 s gap. One inspected 56-frame speech frame contained unwanted subtitles despite passing coarse quality checks. These failures explain why a shorter generation window is not automatically a better conversational experience.

The geometry is quantized: the supported experimental windows follow `17n+5`, with latent temporal extent `5n+2`. An exact two-second crop is not equivalent to a valid shorter continuation unless the latent and audio clocks are advanced consistently. The current service keeps a fixed window length within a logical session.

## 11. Audio fidelity, continuity and removal of unnecessary dependencies

### 11.1 What the audio investigation ruled out

Extra speech appeared in original FP32 PCM before broadcast AAC. Seven recorded failures were replayed with the same prompt, seed, reference and preceding accepted raw AV tail; baseline PCM matched the live PCM bit for bit. Replacing quantized projections and Sage with original BF16 did not remove the three larger reproduced errors.

| Paired audio diagnosis | Median HTTP generation | Main extra-speech failures |
|---|---:|---|
| BF16 first step, then MXFP8; BF16 first attention, then Sage | 4.677 s | Four identified cases remained |
| All BF16 projections, mixed attention | 6.529 s | Three larger cases remained |
| Mixed projections, all BF16 attention | 4.872 s | Three larger cases remained |
| All BF16 projections and attention | 6.541 s | Three larger cases remained |

This justified investigating conditioning and prompt leakage instead of removing every performance optimization as an audio fix. It did not rule out all effects of distillation or precision on voice quality.

### 11.2 Why per-line TTS references were not kept as a requirement

Matched reference waveforms helped selected failure cases, and removing one leaked instruction helped a causal replay. A wider 50-spoken-clip regression found no comparable extra utterance under that experiment's combined reference/prompt policy. But a later online mixed run required 176.539 s of producer work for 158.875 s of media, a ratio of 1.111. Prefetch reduced direct waiting while synthesis could still contend for GPU execution.

The operator then requested trying voice description alone and preferred the observed result. In the selected prompt-only mode, there is no reference synthesis, manifest lookup or external audio block on the per-answer path. The native joint model still generates speech and lip motion together. This simplifies scheduling and avoids reference-content leakage, while transferring responsibility for stable timbre and delivery back to model conditioning.

The initial three-clip prompt-only sample had no recognized extra spoken material, but ASR, pitch and energy proxies did not establish stable speaker identity or perfect pronunciation. This is an accepted demo direction with known limits, not a proof of voice consistency over hours.

### 11.3 What is currently done to preserve the waveform

- Keep the audio VAE in FP32 and preserve the generated float PCM sidecar.
- Feed original PCM to the relay instead of decoding and re-encoding the backend AAC as the source.
- Use one final broadcast AAC encode at 192 kb/s after resampling for the 48 kHz relay clock.
- Keep video and audio on an accumulated sample/frame clock; do not add an assumed five seconds at every boundary.
- Preserve native ambient/action sound in quiet segments; these are generated sounds, not microphone recordings.
- Maintain the same answer and voice instructions across speech segments, with phrase-aware text budgets and no forced conversational ending at every five-second boundary.
- Treat singing as a distinct delivery mode with melodic instructions and native accompaniment only when requested.

The backend can still emit an AAC-bearing MP4 for request compatibility and preview. The relay's fidelity improvement is that it consumes verified original PCM, not that no AAC file exists anywhere in the system. Audibility checks detect missing/near-silent speech; they do not prove intelligibility, correct lyrics, singing quality or lip sync.

## 12. Continuity and the quality/reliability cost of the demo

Motion context is raw generated audiovisual state, not repeated MP4 decoding and re-encoding. The controller saves accepted video/audio tails, checks their shape, clock and hashes, and restores a new backend branch from the accepted tail when needed. The next video prefix is checked for exact equality with the expected retained latent prefix. A decoded-frame seam check catches gross boundary discontinuity.

Window-relative position encoding bounds the local content-time coordinate. The absolute media clock still advances to support correct playback and audio alignment. Rotary position is not a frequency that must numerically explode with the number of broadcast hours; however, bounded coordinates do not solve error accumulation in the generated content.

The live screen checks full decode validity and samples first/middle/last-frame luminance, chroma, saturation, texture and growing black bands against a reviewed anchor. It permits at most one same-parent retry. Rejected output is not published or adopted as accepted context. If both attempts fail, production stops. The old session's clip-872 brightness rejection demonstrates this policy in actual operation.

This is a conservative screening layer, not a face or world-state verifier. It can miss incorrect subtitles, repeated actions, impossible mirror reflections or a disappearing prop. It can also reject an intentional change in clothing/background because global color statistics changed. Private scene diagnostics retain those warnings and are not a license to weaken the production gate.

Two identical clean reference blocks improved one sampled long chain's appearance/continuity but increased compute. Exact caching was selected to reduce that cost while retaining the conditioning. Periodic hard resets were not used as a hidden cure because the operator explicitly required one camera and continuous movement.

A one-clip buffer also makes quality retries more visible. A historical 7.352 s retried request caused a 2.458 s waiting interval. The current zero-underrun snapshot does not erase that failure mode.

## 13. Recorded optimization-history ledger

The ledger groups related work so that repeated campaigns are not counted as independent additive improvements. Dates identify the retained experiment families. The [full historical appendix](history.md) includes the early 653.838-second dense-H3 baseline, every stage of that historical trajectory, and all indexed native experiment families and avatar documents. A separate machine-readable inventory records documentation hashes and evidence status. This is all locatable history in the stated workspace scope, not a claim to recover deleted runs. “Retained” means the relevant current mechanism is present; VSA-specific details may not transfer to Ref2VA.

| Period / family | Change and measured outcome | Final interpretation for the current demo |
|---|---|---|
| Early dense H3 serving | AdaLN reuse, sequence parallelism, explicit RDMA, direct layout, GPU byte packing and chunked output | Architectural foundation; the old 49-forward schedule is not the current four-step task |
| FastH3 model change | Four-step student and VSA replace the base trajectory/attention work | Historical long-clip route; model/attention changes are not lossless serving gains |
| Sep 10–11 communication overlap | QK/V overlap, O pipeline, gate overlap, view cleanup, early-Q/coarse work | Matched means 25.761→25.135; 25.118→24.340; 24.365→24.012 s in separate rounds; no single newly measured combined curve |
| Historical E4M3 wire + TAEH3 branch | Reached a historical 14.561 s endpoint | Excluded from the accepted baseline after quality rejection; BF16 wire and native video VAE retained later |
| Sep 14 MXFP8 projection path | Compare blockwise FP8 and MXFP8; preserve original non-target precision and BF16 wire | Approximate arithmetic choice; do not attribute changed surrounding schedules to quantization alone |
| Persistent RDMA | 17.9214→17.5780 s mean, five requests/arm | Exact relative to the selected MXFP8 baseline; retained registration trades memory for lower setup work |
| SwiGLU/FC2 packing fusion | About 130 ms complete-request reduction; payload/scale/output byte checks | Retained for current MXFP8 FC2; LUT variant gave no extra benefit |
| VAE gather overlap | 17.4510→17.4168 s historical means | Small observed change; later isolated early-gather comparisons did not establish a universal gain |
| VAE INT8 / MXFP8 experiments | INT8 ConvRot candidate 17.4168→16.0726 s; later MXFP8 VAE base 15.8322 s | Historical quality/performance candidates; current video decoder uses NVFP4 |
| CAKE descriptor specialization | 15.8322→15.6370 s in the audited long-clip progression | Grid-constant/TMA optimization; exactness relative to its parent checked |
| Combined QK/V split + VAE batching + AV overlap | Qualified complete-request mean 15.1336 s | Combined gain, not separately attributable; VSA split flags are off in current Ref2VA |
| O producer lookahead | 15.1336→14.9692 s, five formal requests all below 15 s | Qualified real-time long-clip milestone; retains output precision |
| NVFP4 video VAE | Single-request 14.973→14.444 s combined result; native audio PCM unchanged | Deliberate video approximation; early gather alone contributed only a small observed difference |
| Eager DiT scaling | 1/2/4/8 GPU means 89.629/58.249/28.321/16.072 s; matching final latents | DiT-only synthetic conditioning; communication routes also differ; not the fully optimized serving path |
| Tile-aware lowp QKV communication | Real-layer module improvement about 11%, full request 14.442→14.869 s | Rejected: slower complete request and changed video/audio; keep BF16 wire |
| V/QKV host-gap experiments | V-barrier-to-pack median 177.185→19.584 µs; full request 13.688→13.710 s | Local scheduling improvement did not establish E2E benefit; not credited as a speedup |
| Native O consumer Graph microbenchmark | Four O chunks 4.801→4.710 ms in a synthetic eight-rank test | Mechanism evidence; full-model qualification handled separately |
| First-step quality protection | Long clip 13.545 s fast vs 15.255 s BF16-first; O-only BF16 13.934 s | Quality frontier, not a free optimization; full first-step protection crossed the 15 s deadline |
| Exact VAE constant cache + early AAC | Config A 15.240→15.044 s; fast path 13.551→13.304 s | Byte-equivalent output in tested pairs; overlapping output costs became worth optimizing |
| GPU blend/RGB fusion | 13.334→13.261 s median; matched MP4/PCM | Retained; component speedup much larger than request speedup |
| Encoder GPU-output residency | 13.334→13.358 s median in the same campaign | Removed redundant transfers but no demonstrated request acceleration |
| Tiled O producer / full consumer Graph | 13.253→13.120→13.094 s long-clip medians | Producer gain supported; 26 ms extra Graph difference not independently stable; VSA-dependent integration |
| Chunked post-attention residual/norm | 13.117→13.103/13.102 s medians | Kept off: overlapping ranges, about 0.11% difference, no robust gain |
| Prepared norm submission | 13.115→13.079/13.090 s medians; CPU enqueue also reduced | Disabled: three-sample ranges overlap; no stable full-request gain established |
| Split Q/K with original first step | 15.255→15.303 s with copy, 15.253 s without copy; byte-equivalent | 57.04% K-GEMM overlap did not establish E2E acceleration; candidate only |
| Offline 2× reconstruction research | H3 latent-only 18.20–20.61 s; refine about 1850 s; SeedVR2 INT8 233.56–242.23 s | Two scenes and different GPU counts/boundaries; none qualifies as current live postprocessing |
| OpenVDN eight-step continuation | 5/22-frame context medians 23.219/25.208 s for roughly 15 s windows | Did not meet real time; not the selected demo model |
| VDN gather/scan/factor work | Exact window gather and scan Graph microbenchmarks; one faster solve failed numerical limits | Research evidence only; cannot be counted as current H3 speedup |
| Ref2VA five-second motion-context comparison | Five-frame context reduced work relative to 22-frame trials; warm finite measurements retained | Selected shorter context; stability must be checked separately from latency |
| First-step BF16 long-run test | 3.888→4.492 s medians; degradation appeared later but returned | Useful mitigation, not a long-run quality solution |
| 1088 weight residency and BF16 rollback | Transfers removed but 5.611→5.704 s; all-BF16 variants slower | Reverted to resident MXFP8; larger-resolution target not achieved by this change |
| Raw video prefix and accepted-tail rollback | Exact prefix/seam checks and same-parent retry | Retained; prevents bad candidate adoption and discontinuous retry resets |
| Original PCM relay | Avoid AAC-generation loss on the relay input | Retained; does not cure extra words already generated in PCM |
| Shared output validation decode | 5.044→4.774 s matched short-clip median | Exact application work reduction; retained |
| Exact reference/conditioning reuse | Four pairs and three-action OFF/ON/OFF equality; lower preparation scopes | Retained with bounded cache and distributed agreement |
| Local small text model / direct actions | Subsecond planning in small tests; supported commands bypass model | Retained; generation and queueing still dominate response onset |
| All-step Sage/MXFP8 live switch | Historical idle 4.575→3.857 s | Explicit temporary lossy choice to regain headroom |
| Prompt-only voice | Remove per-line reference synthesis and reference blocks | Selected after demo review; voice consistency still needs continued validation |
| One-clip lookahead | Historical ready-to-local-playback median 1.242 s | Retained; lower delay with less retry reserve |
| 56/73-frame trials | Mixed speech deadlines and quality issues | Implemented experimentally; 124-frame production default retained |
| Action-specific prompts and shared world harness | One-cycle gestures; book/food/cloth/water/mirror/window/wardrobe state research | Simple public gestures enabled; complex recipes remain selectively unqualified |
| Fresh-session recovery | New opening after terminal drift failure | Explicit operator restart; not evidence of seamless autonomous recovery |

These rows should not be summed. Some are overlapping implementations, some change arithmetic, some are different prompts/resolutions, and some deliberately record regressions. The historical 653.838 s dense-H3 baseline and rejected approximate branches are preserved in the source inventory rather than presented as a “current 160× speedup” claim.

## 14. Nsight analysis: reproducible scope and interpretation

This report **re-analyzes eight preserved SQLite trace exports on CPU**. It did not interrupt the restarted broadcast to claim a new same-configuration CUDA capture. The current production timing table comes from current receipts; the detailed internal GPU/CPU mechanisms below come from explicitly identified historical captures.

| Capture family | Captures | Configuration distinction | Main question answered |
|---|---:|---|---|
| 15-second Config A | 1 | First BF16 step, later MXFP8/Sage; VSA; 1280×704 | Where the long-clip pipeline spent time; decoder/output overlap |
| 1088 projection residency | 2 | BF16-first, old staging versus resident BF16 plus online packing | Did removing weight H2D shorten the critical path? |
| 1088 all-BF16 projection diagnostic | 1 | Resident BF16 projections | What compute cost replaces the eliminated quantized path? |
| Five-second reference-cache comparison | 4 | One/two reference blocks, hit/miss; BF16-first | Which preparation work can be reused for idle and fresh dialogue? |

The included analyzer extracts CPU NVTX range distributions, per-device kernel spans, merged kernel interval coverage, top kernel work and copy bytes/counts. Each SQLite export has a file hash. It uses only read-only SQLite connections and can be rerun while the service stays live. [Analyzer](analyze_traces.py) · [JSON](evidence/nsys-reanalysis.json) · [CSV](evidence/nsys-reanalysis.csv)

Important interpretation rules:

1. **Do not sum the eight GPUs' simultaneous work as request latency.** Kernel sums and memcpy sums are useful workload measurements, not a serial wall clock.
2. **Do not add parent and child NVTX ranges.** DiT contains its four steps; decode contains video/audio/output work that can overlap.
3. **A long NCCL or RDMA barrier can be a wait for a late rank.** It is not necessarily slow wire transmission. The historical video-VAE entry showed other ranks waiting roughly 59 ms while rank zero submitted audio decode.
4. **A GPU timeline gap is not proof that the NIC is idle.** RDMA progress can occur outside CUDA kernel intervals.
5. **Kernel-interval union is not SM utilization.** A communication kernel may occupy a timeline while waiting rather than performing tensor work.
6. **Profiled request duration is not the unprofiled benchmark.** Profiler startup, event recording and instrumentation can alter scheduling.
7. **Short microbenchmarks do not certify end-to-end benefit.** The V-pack host-gap example is a concrete negative result.
8. **Check capture completeness and configuration.** A VSA trace cannot be relabeled as current dense Ref2VA just because some kernel names match.

A prior cache capture also left profiler injection variables in subprocess environments; a later ffprobe failed with a missing profiler session despite valid audio. That failure was resolved with an unprofiled backend and preserved as a failed experiment, not discarded as a slow sample.

### Reading the current bottleneck responsibly

Current receipts place about 97% of mean producer time inside the backend request. Historical five-second traces identify DiT as the dominant range, with preparation significant on new text/reference conditions and video reconstruction the next material stage. Audio VAE is about 24 ms in the cited cache capture. This supports prioritizing current dense DiT and conditioning work over an audio-decoder precision rollback for speed.

It does **not** provide an exact current all-step-fast DiT/attention/GEMM/communication percentage. A fresh current-configuration profile is the next measurement needed before promising a further kernel-level gain. The report avoids repurposing a BF16-first 1088 profile as the current 960 all-fast breakdown.

## 15. Remaining work, ordered by value to this demo

| Priority | Work | Why this follows from evidence | Acceptance requirement |
|---|---|---|---|
| 1 | Long-run quality and explicit recovery policy | The last stream stopped on brightness drift despite sufficient throughput | Longer continuous reading/dialogue runs; retain failures; review face, lighting, props and raw AV; no hidden hard reset |
| 2 | Measure actual first-response onset | Ready time is not viewer experience; queueing and in-clip gaze/speech onset remain material | Correlate received messages with local and viewer-recorded first meaningful response |
| 3 | Current dense Ref2VA DiT profile | Historical VSA opportunities do not automatically apply; backend dominates | Fresh isolated capture and matching unprofiled requests; same prompt, tail, precision and geometry |
| 4 | Keep reducing exact conditioning and data handling work | New speech still misses text conditioning; unchanged image work can be reused | Exact cache keys/invalidation, distributed agreement, latent and full-output comparisons |
| 5 | Adaptive buffering with an explicit latency budget | One clip lowers delay but retries can exhaust it | Measure underruns and response distribution together; no hidden growing queue |
| 6 | Better speech continuity evaluation | ASR can miss unnatural prosody and clipped phonemes | Listening, boundary alignment, timbre, extra/missing speech and lip-sync review |
| 7 | Capability-specific visual qualification | State prediction does not guarantee action execution | Repeated requests, cancellation, interruption and persistent object state; retain negative examples |
| 8 | Revisit 73/56 frames only after backend headroom improves | 56-frame mixed workload currently loses throughput; 73-frame speech has little reserve | Sustained mixed speech/action run with comfortable P95 margin and no phrase fragmentation |
| 9 | Higher resolution after deadline/quality stability | 1088 residency alone failed; frame count and image area both cost compute | Unchanged-policy resolution comparison, paired quality review and full producer deadline statistics |

The credible demo claim is that an open H3-based audiovisual generator can sustain interactive, bounded-window live production on this eight-GPU system under the measured conditions, with a stateful application and explicit quality gates. The evidence does not yet support uninterrupted 24-hour coherence, unrestricted reliable object manipulation, or a universal low-latency viewer SLO.

## 16. Evidence package and reproducibility

- [Current live production observation](evidence/current-live-observation.json): aggregate timing only; no viewer identities or message text.
- [Extended production observation](evidence/extended-live-observation.json): 456 accepted continuations, including later deadline misses and relay waits.
- [Measurements and selected runtime configuration](evidence/measurements.json): unprofiled samples, residency comparisons, cache profile ranges, short-window results.
- [Nsight SQLite reanalysis](evidence/nsys-reanalysis.json) and [CSV ranges](evidence/nsys-reanalysis.csv): eight retained captures; exact hashes and per-device detail.
- [Source manifest](evidence/source-manifest.json): local experiment documents and measurement files, with hashes and portable workspace placeholders.
- [Optimization history inventory](evidence/history-inventory.json): discovered experiment families and their retained records.
- [Backend execution counters](evidence/backend-dispatch-snapshot.json): accumulated eight-rank projection/fusion dispatch, explicitly not an isolated request benchmark.
- [Historical appendix](history.md): early stages, rollbacks, negative results and every indexed experiment family.
- [Software versions](evidence/software-versions.json): environment metadata, with source-overlay caveats.
- [Selected implementation hashes](evidence/implementation-manifest.json): application control, model fusion and projection paths.
- [Trace analyzer](analyze_traces.py), [report builder](build_report.py), and [history indexer](build_history.py): CPU-only reproducibility tools.

```bash
# Read preserved SQLite exports; does not start a GPU request.
python analyze_traces.py --workspace /path/to/workspace \
  --temporary-root /path/to/retained-tmp --output evidence/nsys-reanalysis.json

# Rebuild charts and self-contained HTML from the bundled evidence.
python build_report.py

# When a new offline capture is explicitly scheduled, keep its timing separate.
# Export a recorded report (command shown for reproduction, not run on live):
nsys export --type sqlite --output request.sqlite request.nsys-rep
nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum request.nsys-rep
```

The local analyzer uses the retained paths recorded in its trace catalog. To reproduce on another machine, supply the corresponding artifacts at updated paths and verify their hashes. Raw traces and model weights are not embedded in this lightweight report bundle. A directory named “prepared,” a configuration flag, or a successful HTTP response alone is never used as proof that a proposed optimization ran or that an action succeeded.
