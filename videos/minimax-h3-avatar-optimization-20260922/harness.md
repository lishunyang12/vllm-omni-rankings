### 3.7 Harness architecture and evidence contracts

The harness translates conversation into a bounded audiovisual workload and records what happened at each boundary. It does not replace the video model's physical reasoning. The implemented public path uses a durable chat inbox, reply queue, segment scheduler, continuity controller and playback receipts. A separate research path adds explicit object state, multi-action planning and review-gated state transitions. These paths share the rendering stack but must not be described as one fully deployed autonomous world simulator.

![Harness architecture and evidence contracts. Solid connections identify an implemented control or receipt dependency; the private action path and its additional semantic review are distinguished from public queue admission. Neither a prompt nor a generated file is sufficient evidence of a completed physical action.](figures/harness-contracts.svg)

**TABLECAPTION: Harness components, responsibilities and deployment scope.** Source hashes and the actual local routing probe are retained in the [harness manifest](evidence/harness-system.json).

| Component | Responsibility | State or output | Scope |
|---|---|---|---|
| Chat inbox and text worker | Deduplicate real messages, bound waiting work and retain viewer context | Message identity, arrival time, expiry, same-viewer dialogue | Public control path |
| Public dialogue policy | Produce a relevant complete answer; obtain only relevant scene facts | Natural utterance text, language, attention and delivery mode | Public control path; not a fixed total-answer character limit |
| Reply queue and segment scheduler | Preserve turn identity and consecutive answer delivery | Queued, selected, generated, relay-started and relay-finished records | Public control path |
| Action registry and research planner | Extract affirmative requests, map only available capabilities, refuse unsupported substitutions | Validated decision with at most three action identifiers | Separate research path |
| Pure action compiler | Arrange dependencies, object contact, one action cycle and persistent exit state | A sequence of before/after states and timed prompt beats | Separate research path; desired effects remain predictions |
| World ledger and grounding | Record state revisions, provenance, pending work and viewer-specific dialogue | SQLite transactions and a digest of the planning context | Separate research path |
| Renderer and continuity controller | Attach clean references and the accepted raw AV suffix; screen candidates | Video, original PCM, hashes, raw-tail checkpoint and timing | Shared generation machinery; private override requires a non-broadcast run |
| Review and playback adapter | Compare an explicit observation with the expected state; commit only after confirmed playback | Verified/played receipts or rejection/uncertainty | Research contract; no general automatic visual verifier has been established |

The research state includes book location, hand ownership, mouth state, gaze, tea availability, coarse cup fill, candy history and the near window leaf. Outfit, room and exterior variants require exact prepared references. The ledger deliberately does not infer an exact candy count, tea temperature, invisible scene geometry or a viewer's current screen. Window reachability and hinge type must be supplied and verified; a private geometry hypothesis is not public action qualification.

Planning uses a state snapshot and context digest. Admission checks the digest again, preventing an answer planned against an old scene from silently entering a changed world. Effect records retain the source action and viewer relation. Replaying the same playback receipt is idempotent. Cancellation preserves already completed effects and recovers held objects at a segment boundary; it does not reverse time or rewrite an in-flight diffusion request. An uncertain playback result requires reconciliation before further dependent work.

**TABLECAPTION: Research receipt lifecycle and the claim each stage permits.** The private GPU probe follows predicted states to collect footage; it does not automatically invoke the observed-world commit path.

| Receipt stage | What is known | Permitted statement | State advancement |
|---|---|---|---|
| Prepared | A validated phase and expected exit state exist | The action is planned | None |
| Generated | A candidate media artifact exists | The candidate was generated | None |
| Verified | Explicit review passes and its observed state matches the phase contract | The candidate meets the supplied review | None until playback |
| Played | The verified segment has a confirmed local playback receipt | The reviewed local effect can enter memory | Transactional state, provenance and memory update |
| Rejected / uncertain | Review failed or playback outcome is unresolved | Completion is not established | Rollback or reconciliation required |

### 3.8 Prompt construction and action execution

There are two distinct prompt interfaces. The language planner receives character policy, selected facts, recent dialogue and the viewer's message. The audiovisual renderer receives the current phase, exact speech where applicable, clean visual conditioning and the accepted motion/audio context. The language-model answer is not itself a complete video prompt.

The public dialogue policy allows an answer to expand when the question needs it. Speech is then divided into speakable windows, preserving turn identity and viewer attention across boundaries. The older research writer remains more restrictive: its current contract allows at most two lines, with twenty Chinese characters or twelve English words per line. That research constraint must not be presented as the public conversation policy or as a desired universal limit.

For object interaction, the compiler first establishes the necessary hand state. In the current private tea/food recipes, a held book is placed on the table once; a subsequent bite, sip or pour follows in one generation window. Rest keeps the book on the table, and picking it up is an explicit later instruction. This avoids repeatedly inserting a fresh pickup phase or forcing the reference image's reading pose at every boundary. The prompt names the existing object, contact sequence, single operation, release point and persistent exit state. Only actually present props are included.

**TABLECAPTION: Renderer prompt fields and their intended function.** Exact transmitted examples appear in Appendix B.1.

| Field | Example function | Evidence boundary |
|---|---|---|
| Scene and identity | Same adult character, side camera, book, lotus study and present props | A conditioning instruction, not identity verification |
| Entrance state | Book on the table, empty hand, current window angle | Must correspond to the accepted incoming scene |
| Timed beats | Continue overlap; reach and grip; sip; set down and release | Timing directives are approximate model control |
| Exit state | Same cup back on the table; no second reach | Predicted until the generated motion is reviewed |
| Voice and soundscape | Exact words if speaking; otherwise quiet room tone and action sounds | No per-answer external voice waveform in this profile |
| Reference suffix | Picture 2 repeats Picture 1 | Two blocks of one image, not two independent views |

A request from the held-book state typically plans one approximately five-second placement window followed by one approximately five-second action window. If the book is already on the table, the placement phase is omitted. These are planned media durations, not chat-response latency measurements: queue occupancy, current generation, local playback and Twitch buffering remain additional clocks.

Finite private samples show recognizable eating, sipping and pouring cycles with subsequent hold phases. Window work remains experimental: a state-specific reference can preserve a closure in one sample, yet other shutters may move without the intended contact; a more explicit spatial prompt also produced an uncommanded reopening. These outcomes belong in limitations and diagnostic evidence. A positive coarse motion example does not establish arbitrary action control, precise prop conservation or a long-run success rate.
