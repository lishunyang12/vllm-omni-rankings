# Optimization history and experiment index

[Return to the current system report](report.md)

This appendix preserves the path that led to the current demo, including quality rollbacks, negative results, diagnostic-only traces and unqualified plans. The current system is a four-step, dense-Sage Ref2VA continuation workload. Historical VSA and VDN results must not be represented as current configuration measurements.

The retained workspace contains **55 native experiment families**, **49 avatar documentation files**, and **10 avatar experiment directories** in the indexed scope. File hashes and portable source paths are in the [machine-readable inventory](evidence/history-inventory.json). Deleted records and unindexed nested runs are not claimed to have been recovered.

## Early trajectory, including rejected branches

These stages are imported from the September 11 historical audit. They change the step schedule, attention graph, arithmetic and decoder at different points. Values are historical complete-request seconds under the source campaign; they are not interchangeable with current five-second publication time. “Exact” applies to the optimization relative to its parent contract, not to the entire accumulated model. Do not sum adjacent changes or report a universal speedup to the present avatar.

| Stage | Change | Historical seconds | Fidelity of change | Later status |
|---|---|---:|---|---|
| S0 | Original H3 control | 653.838 | Original | Historical; not requalified as current live |
| S1 | AdaLN TP1 / SP8 | 605.348 | Exact | Historical; not requalified as current live |
| S2 | Explicit RDMA Ulysses exchange | 592.079 | Exact | Historical; not requalified as current live |
| S3 | Direct Q/K layout | 590.562 | Exact | Historical; not requalified as current live |
| F0 | Dense FastH3, four forwards | 66.785 | Lossy | Historical; not requalified as current live |
| P0 | FastH3 + VSA + FlashInfer | 43.420 | Lossy | Historical; not requalified as current live |
| P1 | GPU FP32-to-uint8 video pack | 36.568 | Exact | Historical; not requalified as current live |
| P2 | Chunked pinned D2H + background mux | 31.399 | Exact | Historical; not requalified as current live |
| P3 | Persistent RDMA communication | 30.534 | Exact | Historical; not requalified as current live |
| P4a | Fused VSA tile packing | 30.013 | Exact | Historical; not requalified as current live |
| P4c | Compact VSA untile | 29.880 | Exact | Historical; not requalified as current live |
| P4d | Direct attention O path | 29.873 | Exact / noise-sized | Historical; not requalified as current live |
| P5 | Exact VAE/direct Q-to-K cleanup | 28.236 | Exact | Historical; not requalified as current live |
| Q1 | Online FP8 MLP | 25.172 | Lossy | Historical; not requalified as current live |
| Q2 | MLP + output FP8 | 24.887 | Lossy | Historical; not requalified as current live |
| Q5 | QKV E4M3 wire + O bundle | 21.649 | Lossy | Rejected E4M3/TAEH3 descendant; excluded |
| D1 | TAEH3 decoder, FP32 | 16.738 | Lossy | Rejected E4M3/TAEH3 descendant; excluded |
| D2 | TAEH3 decoder, FP16 | 16.289 | Lossy | Rejected E4M3/TAEH3 descendant; excluded |
| D3 | All-main FP8 coverage | 15.634 | Lossy | Rejected E4M3/TAEH3 descendant; excluded |
| D4 | Persistent RDMA steady state | 15.165 | Exact | Rejected E4M3/TAEH3 descendant; excluded |
| D5 | OMP=28 CPU orchestration | 14.671 | Exact | Rejected E4M3/TAEH3 descendant; excluded |
| D6 | Cross-rank Audio VAE | 14.561 | Exact | Rejected E4M3/TAEH3 descendant; excluded |

The fastest rejected descendant is retained here to explain the rollback, not to advertise an accepted endpoint. The later native-VAE/BF16-wire work follows a different accepted branch. [Original numerical audit extract](evidence/early-history.json)

## Exact overlap branch and negative retests

| Round | Mechanism | Before mean | After mean | Evidence |
|---|---|---:|---:|---|
| v1 | QK/V overlap + four-chunk reverse O | 25.7613 s | 25.1353 s | Matched complete-request pairs; n=3 per arm |
| v2 | Gate GEMM during QKV exchanges | 25.1177 s | 24.3403 s | Matched complete-request pairs; n=3 per arm |
| v3 | Layout views + early Q + coarse overlap | 24.3647 s | 24.0117 s | Matched complete-request pairs; n=3 per arm |
| v4 | Early coarse softmax (exploratory) | 24.0528 s | 23.9402 s | Observed only; no stable gain established; n=5 per arm |

The later same-feature baseline retest was 25.1176 s, rather than the historical 24.0117 s. Its output-fusion and boundary candidates took 25.2708 and 25.2792 s. This variability is why a compounded percentage from different campaigns is not a freshly measured cumulative improvement. The v9 V-pack microbenchmark improved 0.5227 to 0.3484 ms, but its simulated single-GPU chain improved only about 0.30%; it was not an SP8 MP4 qualification.

## Every indexed native experiment family

A description of a plan remains a plan. Where a completed result was reviewed, its limits are included. Exact documentation hashes are in the inventory; related families are cross-referenced to avoid counting one experiment twice.

### h3-ulysses-overlap-20260910

Three matched scheduling rounds support exact improvements relative to their fixed precision baseline. Later v7 output/boundary fusion regressed; v9 remained a single-GPU prototype. See the branch tables below.

Retained top-level documentation: `LATEST_LOSSLESS_STATUS.md`, `LOSSLESS_GOAL.md`, `RECOVERY_CONTEXT.md`, `WORK_STATE.md`, `report-methodology.md`.

### h3-vsa-preprocessing-investigation-20260910

Analyzed 4,800 barrier-to-fine-attention intervals. On GPU 7, 5.574 of 5.877 ms was GPU activity. Local candidates saved 1.332 ms; the projected whole-model benefit was not a measured E2E result.

Retained top-level documentation: `README.md`.

### h3-vsa-sage-pr4691-20260910

Sage attention integration, shape/dispatch qualification and BF16 comparison. This changes attention arithmetic; it is not lossless relative to BF16.

Retained top-level documentation: `WORK_STATE.md`, `pr4951-assessment.md`, `report-methodology.md`.

### h3-vsa-sage-teahouse15s-20260910

Three separate long-clip arms: BF16 VSA, Sage PR4691, and locally repaired CAKE PR4951. Their means were 27.494, 25.902 and 25.706 seconds; no composition with a different benchmark baseline.

Retained top-level documentation: `WORK_STATE.md`, `pr4951-assessment.md`, `report-methodology.md`.

### h3-cumulative-optimization-20260911

Audits the early dense-to-FastH3 path, three exact overlap rounds, later negative retests, and rejected E4M3/TAEH3 descendants. Historical stages are reproduced below with their original evidence scope.

Retained top-level documentation: `README.md`.

### h3-quality-comparison-20260913

Quality comparison controllers, geometry/source guards and evolving campaign scope. Shared tooling used by later admitted measurements.

Retained top-level documentation: `CAMPAIGN_STATUS.md`.

### fasth3-mxfp8-fusion-20260914

Fuse SwiGLU, BF16 rounding and MXFP8 FC2 input production: 17.5780 to 17.4478 s mean in the retained campaign. LUT alternative did not add a qualified gain.

Retained top-level documentation: `LUT.README.zh.md`, `README.zh.md`.

### fasth3-mxfp8-next-20260914

Large optimization family: VAE precision/batching, CAKE descriptors, QKV split, AV overlap, O lookahead and shipping worktrees. Audited inventory reaches 14.9692 s mean for the qualified long-clip combination; individual effects are not all isolated.

Retained top-level documentation: `README.zh.md`, `RESUME.zh.md`.

### fasth3-mxfp8-profile-20260914

Prepared profiling/admission tools and CPU checks. Its README explicitly says the proposed GPU capture was not yet qualified; existence of the plan is not a successful trace.

Retained top-level documentation: `README.zh.md`.

### fasth3-mxfp8-retention-20260914

Retain RDMA registrations across denoising requests: 17.9214 to 17.5780 s mean, five requests per arm, matching media bytes. More memory remains resident.

Retained top-level documentation: `README.zh.md`.

### fasth3-mxfp8-vs-blockwise-20260914

Separate projection-precision comparison family. Keep its model/dispatch qualification separate from later resident-buffer and fusion improvements.

Retained top-level documentation: `README.zh.md`.

### h3-best-paths-20260914

Catalog and selection of candidate paths; use linked result campaigns for measurements.

Retained top-level documentation: `README.zh.md`, `fast-version-index.zh.md`.

### h3-dit-vsa-scaling-20260914

Early one/two/four-GPU bring-up, single DiT-only samples with synthetic conditioning. Not the final scaling result and not complete video generation.

Retained top-level documentation: `README.zh.md`, `RESULTS.zh.md`.

### h3-e2e-cumulative-rerun-20260914

Rerun plan indexed. No new successful E2E result inferred from the plan.

Retained top-level documentation: `PLAN.zh.md`.

### h3-dit-mxfp8-swiglu-nsys-20260915

Profiling plan/specs indexed. Read result admission before treating any nested capture as completed.

Retained top-level documentation: No top-level Markdown; JSON/spec files indexed..

### h3-dit-vsa-usp-scaling-20260915

Qualified eager DiT-only scaling: 89.6286 / 58.2486 / 28.3212 / 16.0722 s on 1/2/4/8 GPUs. Final latents match. Communication routes also change; excludes encoder/VAE/MP4 and is not the optimized live DiT.

Retained top-level documentation: `README.zh.md`, `RESULTS.zh.md`.

### h3-fasth3-vs-best-video-20260915

Published synchronized original/FastH3/best-path sample comparison and media audits. Different trajectories cannot be labeled one lossless speedup.

Retained top-level documentation: `README.zh.md`, `RESULTS.zh.md`, `RESUME.zh.md`.

### h3-original-history-best-video-20260915

Historical-path media replays and side-by-side delivery. Preserve their original shapes, frame counts and trajectory differences.

Retained top-level documentation: `RESULTS.zh.md`, `RESUME.zh.md`.

### h3-postrollback-nsys-20260915

Published before/after diagnostic campaign combining multiple precision and scheduling changes. Preserve the disclosed teardown exception; not an unprofiled single-factor latency benchmark.

Retained top-level documentation: `README.md`.

### h3-best-latency-nsys-20260916

Controller status is stopped after a failed prerequisite run. Do not count this directory as a successful final Nsight capture.

Retained top-level documentation: `README.md`.

### h3-infinite-cartoon-20260916

Early long-form continuity research branch. Not the deployed Zhiwei profile and not an indefinite-stability result.

Retained top-level documentation: `README.zh.md`, `RESUME.zh.md`.

### h3-ulysses-lowp-pr4924-review-20260916

Tile-aware low-precision QKV communication saved about 11% in isolated real-layer modules, but the full matched request regressed 14.442 to 14.869 s and changed both picture and audio. Not adopted.

Retained top-level documentation: `README.zh.md`, `RESUME.zh.md`.

### h3-vae-nvfp4-pipeline-20260916

Native NVFP4 video decode plus early gather: matched family 14.973 to 14.444 s in a single-request comparison. Video approximation is explicit; generated native audio PCM matched.

Retained top-level documentation: `RESULTS.zh.md`, `RESUME.zh.md`.

### h3-o-consumer-full-20260917

Full output equality and contended scheduling traces validated dependencies. Local gaps shrank, but timing_valid=false because of external GPU load. Never use these timings as normal performance.

Retained top-level documentation: `BUBBLE_AUDIT.zh.md`, `DIAGNOSTIC_RESULTS.zh.md`, `FA4_AND_PRODUCER.zh.md`, `GREEN_CONTEXT_ASSESSMENT.zh.md`, `SCHEDULING_RESULTS.zh.md`.

### h3-o-mxfp8-pre-a2a-lossless-20260917

Eight-rank O producer/consumer prototypes, synthetic byte checks and Graph experiments. Four-chunk microbenchmark 4.801 to 4.710 ms; full-model evidence is a separate campaign.

Retained top-level documentation: `CONSUMER_GRAPH_RESULTS.zh.md`, `O_BF16_EVALUATION.zh.md`, `O_STARTUP_PLAN.zh.md`, `README.zh.md`.

### h3-original-single-lossy-ablation-20260917

Original H3 isolated lossy ablations with source/dispatch/media qualification. Quality evidence, not warmed production throughput.

Retained top-level documentation: `README.zh.md`.

### h3-qkv-transfer-gap-20260917

V-barrier-to-pack median reduced 177.185 to 19.584 microseconds, but full request changed 13.688 to 13.710 s. No established E2E win.

Retained top-level documentation: `MICROBENCHMARK.zh.md`, `RESULTS.zh.md`.

### vdn-original-single-lossy-ablation-20260917

VDN precision quality ablations and repeatability diagnostics. Original repetitions also differed, preventing naive attribution of all pixel differences to precision.

Retained top-level documentation: `README.zh.md`, `results-summary.md`.

### fasth3-combined-first-step-original-20260918

Combined first-step-quality experiment family. Later precision-schedule campaign supplies the qualified timing comparison.

Retained top-level documentation: `METRICS.zh.md`, `README.zh.md`, `RESULTS.md`.

### fasth3-cpu-pipeline-20260918

CPU/output pipeline experiment family indexed; no independently requalified gain assigned in this report.

Retained top-level documentation: `README.zh.md`.

### fasth3-dit-bubbles-20260918

Initial DiT bubble experiments. Use the separately qualified live family for complete-request timing.

Retained top-level documentation: `REVIEW.zh.md`.

### fasth3-dit-bubbles-live-20260918

Full request medians: baseline 13.253, tiled O producer 13.120, plus consumer Graph 13.094 s. Producer result supported; marginal Graph delta overlaps variability. VSA-dependent integration.

Retained top-level documentation: `RESULTS.zh.md`.

### fasth3-dit-chunknorm-live-20260918

Chunked residual/norm: 13.117 to 13.103/13.102 s medians; no robust improvement. Disabled in the current runtime.

Retained top-level documentation: `RESULTS.zh.md`, `WORK_STATE.md`.

### fasth3-dit-comm-review-20260918

Encoder GPU-output candidate 13.334 to 13.358 s: no gain. GPU blend/RGB candidate 13.334 to 13.261 s: matched media and smaller complete-request benefit than its kernel microbenchmark.

Retained top-level documentation: `LIVE-RESULTS.zh.md`, `REVIEW.zh.md`.

### fasth3-dit-norm-submit-20260918

Prepared CPU submission: 13.115 to 13.079/13.090 s medians. Three-sample ranges overlap; remains off. CPU enqueue reduction is not whole-model certification.

Retained top-level documentation: `RESULTS.zh.md`, `WORK_STATE.md`.

### fasth3-early-step-original-20260918

Quality study of original first-step projection/attention restoration. Keep this multi-scene study distinct from the subsequent warmed latency campaign.

Retained top-level documentation: `METRICS.zh.md`, `README.zh.md`.

### fasth3-encoder-resident-20260918

Preparation directory. Completed encoder-residency comparison is recorded in fasth3-dit-comm-review; do not double-count.

Retained top-level documentation: No top-level Markdown; JSON/spec files indexed..

### fasth3-fastest-plus-aac-20260918

Fast path plus exact constant reuse/early AAC: 13.551 to 13.304 s median; tested complete MP4 bytes match.

Retained top-level documentation: `README.zh.md`, `RESULTS.zh.md`.

### fasth3-nvenc-vs-cpu-20260918

Encoder tuning plans retained. No qualified adoption inferred from plans; current request packaging remains CPU/PyAV.

Retained top-level documentation: No top-level Markdown; JSON/spec files indexed..

### fasth3-oproj-bf16-20260918

Seven-scene quality ablation restoring original BF16 O weights only. All 42 accumulated videos decoded. Includes first compilation; not a live throughput benchmark.

Retained top-level documentation: `METRICS.zh.md`, `README.zh.md`.

### fasth3-output-fusion-20260918

Implementation/audit family for blend and RGB output fusion. Full-request results are in fasth3-dit-comm-review.

Retained top-level documentation: `README.zh.md`.

### fasth3-output-fusion-live-20260918

Launch preparation for output fusion. Completed measurement is cross-referenced to fasth3-dit-comm-review rather than counted twice.

Retained top-level documentation: No top-level Markdown; JSON/spec files indexed..

### fasth3-precision-schedule-realtime-20260918

Fast baseline 13.545 s median; first-step original 15.255 s; O-only BF16 13.934 s. Exposes a quality/latency frontier rather than a free improvement.

Retained top-level documentation: `README.md`, `RESULTS.zh.md`.

### fasth3-qk-split-overlap-20260918

First-step-original Q/K split: 15.255 s baseline, 15.303 s with a Q copy, 15.253 s without it. Bytes match; 57.04% K-GEMM overlap does not prove request speedup. Experimental.

Retained top-level documentation: `README.md`, `REMAINING-LOSSLESS-OPPORTUNITIES.zh.md`, `RESULTS.zh.md`.

### fasth3-qkvo-bf16-20260918

Seven-scene quality ablation restoring original Q/K/V/O, leaving FFN MXFP8. All 49 accumulated videos decoded; pixel differences are not subjective quality scores.

Retained top-level documentation: `METRICS.zh.md`, `README.zh.md`.

### fasth3-realtime-resume-20260918

Three warm long-clip complete requests: 13.525 / 13.549 / 13.624 s, with the existing accepted native-decoder stack. Finite real-time demonstration.

Retained top-level documentation: `README.zh.md`, `RESULTS.zh.md`.

### fasth3-vae-lossless-overlap-20260918

Exact decoder constants, first-window scheduling, early AAC and hardware traces. Config A baseline 15.240 s median, cached constants plus AAC 15.044 s; compatible overlap effects are not additive.

Retained top-level documentation: `README.md`, `RESULTS.zh.md`.

### h3-interactive-infinite-live-20260918

Early live harness: director, story, serving and observability. Later avatar application replaces the selected production integration.

Retained top-level documentation: `DIRECTOR.zh.md`, `DRY_RUN.zh.md`, `EPISODE-02.zh.md`, `EPISODE-03.zh.md`, `HARNESS-REVIEW.zh.md`, `OBSERVABILITY.zh.md`, `PLAN.zh.md`, `SERVE.zh.md`, `STORY.zh.md`.

### h3-upscaler-vs-seedvr2-20260918

Two-scene offline 2x study: H3 latent-only 18.20–20.61 s on four GPUs, refine about 1,850 s, SeedVR2 INT8 233.56–242.23 s on one GPU. Different boundaries/hardware; none is a qualified live postprocess.

Retained top-level documentation: `COMPUTE.zh.md`, `METHODOLOGY.zh.md`, `REPORT.zh.md`, `VAE-COMPUTE.zh.md`.

### vdn-combined-realtime-20260918

Eight-step VDN full requests: 39.875 s median at 1280x704, 45.408 s at 1344x768. Both fail the 15-second budget; not the production model.

Retained top-level documentation: `README.zh.md`, `RESULTS.zh.md`.

### vdn-factor-fusion-20260920

Exact gather microbenchmark 17.311 to 14.008 ms; scan Graph 2.166 to 1.831 ms. Faster Gauss-Jordan solve failed numerical limits. No current-H3 speedup credit.

Retained top-level documentation: `RESULTS.zh.md`.

### vdn-ref2va-allopt-20260920

Initial combined VDN continuation branch; context-cost and CPU/source checks retained. Do not infer a completed warm generation from those checks.

Retained top-level documentation: `INNOVATION.zh.md`.

### vdn-ref2va-allopt-v2-20260920

Second VDN integration, Sage batch probe and context checks. Superseded experiment, not the selected four-step Ref2VA runtime.

Retained top-level documentation: No top-level Markdown; JSON/spec files indexed..

### vdn-ref2va-allopt-v3-20260920

Completed VDN 960x544 continuation: five-frame context 23.219 s median, 22-frame context 25.208 s for roughly 15-second windows. Not real time; not a visual-quality acceptance.

Retained top-level documentation: No top-level Markdown; JSON/spec files indexed..

### vdn-ref2va-gather-20260920

Exact gather integration and standalone checks. Follow-up complete-model continuation measurements belong to the v3 family.

Retained top-level documentation: `RESUME.zh.md`.

## Avatar-serving evolution

| Area | Evolution and decision | Source document |
|---|---|---|
| Five-second serving | Long-clip throughput becomes bounded continuation, a durable chat/reply queue, and continuous broadcasting | H3_FIVE_SECOND_LIVE_SYSTEM.md |
| Scene continuity | Raw AV tails, exact video prefix, same-parent retry, fixed camera and drift screens; hard resets are not concealed | ZHIWEI_CONTINUITY.md; MULTIREF_DEGRADATION_REVIEW.md |
| Resolution and weights | 1088 target, BF16 residency and all-BF16 controls fail to improve the selected live budget | INFRA_1088.md |
| Precision | First-step BF16 protects quality in a bounded sample; all-step Sage/MXFP8 later selected explicitly for headroom | ZHIWEI_RUNTIME_LOCK.md; LIVE_OVERHEAD_20260922.md |
| Exact application reuse | Shared validation decode and exact reference caches remove repeat work without deleting conditioning | ZHIWEI_LATENCY_20260921.md; LIVE_OVERHEAD_20260922.md |
| Audio diagnosis | Extra words already existed in original PCM; projection/attention rollbacks did not remove the larger failures | AUDIO_LOSSY_ABLATION_20260921.md |
| Audio conditioning | Reference trials and prompt leakage fixes, then user-selected prompt-only native voice | SPEECH_FIDELITY_20260921.md; PROMPT_ONLY_VOICE_20260922.md |
| Interaction delay | Local small planner, direct actions and one-clip lookahead; actual viewer onset remains a separate measurement | ZHIWEI_VIEWER_QUEUE.md; ZHIWEI_LATENCY_20260921.md |
| Shorter windows | 56/73-frame backend/application support tested; keep 124 because mixed speech and boundary quality are not yet qualified | H3_TWO_SECOND_FEASIBILITY_20260922.md |
| Action semantics | Ground answers in known scene state; preserve whole-answer attention; qualify singing/nod and avoid repeated action phases | ZHIWEI_DIALOGUE_GROUNDING.md; ZHIWEI_SINGING_NOD_FIX_20260922.md |
| Stateful world research | Object interactions, prerequisites and cancellation prototypes; visual success is not guaranteed by a planner state change | ZHIWEI_WORLD_INTERACTION.md; ZHIWEI_INTERACTION_LAB_20260922.md |
| Monitoring and restart | Session-scoped English dashboard, playback/producer clocks, checkpoints and verified fresh stream | OBSERVABILITY.md; current report receipts |

## Retained avatar documentation

The following filenames identify the complete current docs-directory inventory; inclusion does not mean every proposed feature is deployed. The current report distinguishes production behavior from research profiles.

- `AUDIO_LOSSY_ABLATION_20260921.md`
- `AUDIO_QUALITY.md`
- `H3_FIVE_SECOND_LIVE_SYSTEM.md`
- `H3_TWO_SECOND_FEASIBILITY_20260922.md`
- `HARNESS_REVIEW_20260921.md`
- `INFRA_1088.md`
- `LINA_CAMERA.md`
- `LINA_DESIGN.md`
- `LINA_HARNESS.md`
- `LIVE_OVERHEAD_20260922.md`
- `LOCAL_INTERACTION.md`
- `MULTIREF_DEGRADATION_REVIEW.md`
- `OBSERVABILITY.md`
- `PERSONA.md`
- `PROMPT_ONLY_VOICE_20260922.md`
- `QUALITY.md`
- `SETUP.md`
- `SPEECH_FIDELITY_20260921.md`
- `STATUS.md`
- `STREAMING_PROJECT_REVIEW.md`
- `THIRD_PARTY.md`
- `TIMELINE_DIRECTOR_REVIEW.md`
- `ZHIWEI_ASSET_AUDIT.md`
- `ZHIWEI_CAUSAL_INTERACTION.md`
- `ZHIWEI_CLARITY.md`
- `ZHIWEI_CONTINUITY.md`
- `ZHIWEI_CREATIVE_BRIEF.md`
- `ZHIWEI_DIALOGUE_GROUNDING.md`
- `ZHIWEI_FACE_REVIEW.md`
- `ZHIWEI_GAMEPLAY.md`
- `ZHIWEI_HARNESS.md`
- `ZHIWEI_HARNESS_EVAL.md`
- `ZHIWEI_HARNESS_EVAL_V2.md`
- `ZHIWEI_INTERACTIONS.md`
- `ZHIWEI_INTERACTION_LAB_20260922.md`
- `ZHIWEI_LATENCY_20260921.md`
- `ZHIWEI_LIVE_20260921.md`
- `ZHIWEI_LOTUS_SCENE.md`
- `ZHIWEI_OPEN_INTERACTION.md`
- `ZHIWEI_RUNTIME.md`
- `ZHIWEI_RUNTIME_LOCK.md`
- `ZHIWEI_SCENE_ACTION_CATALOG.md`
- `ZHIWEI_SHOW_BIBLE_V1.md`
- `ZHIWEI_SINGING_NOD_FIX_20260922.md`
- `ZHIWEI_TEXT_REVIEW_20260921.md`
- `ZHIWEI_VIEWER_QUEUE.md`
- `ZHIWEI_VOICE_PERSONA.md`
- `ZHIWEI_WINDOW_POC.md`
- `ZHIWEI_WORLD_INTERACTION.md`

## Retained avatar experiment directories

- `bf16-resident-comparison`
- `candy-bite-plan-20260922`
- `continuity-prefix`
- `infra-1088-20260921`
- `live-restart-report-20260922`
- `lotus-continuity`
- `rolling-quality`
- `short-window-20260922`
- `tts-reference`
- `zhiwei-singing-nod-20260922`

All figures and conclusions in the current report use the explicitly referenced evidence subset. Historical inventories are useful for traceability, but cannot replace admission checks, paired media review or a matching unprofiled performance test.
