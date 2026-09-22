### 3.4 Live program and interaction policy

The selected program is a continuous reading session in a classical study. Zhiwei reads *Liaozhai Zhiyi* beside a lotus pond; the book, incense, bamboo slips, sweets and tea provide a stable setting for conversation. The default behavior is quiet reading with natural breathing and blinking. Page turning is enabled through a low-frequency schedule, approximately once per ten five-second windows when no queued interaction takes precedence. Eating, drinking and repeated head turns are not automatic idle routines.

The show bible defines possible discussion units and original story prompts, but it is a director's plan rather than an autonomous deployed program scheduler. Viewer dialogue, supported gestures and explicitly queued host contributions use the same generation queue. The application does not fabricate audience messages to keep the character talking.

**TABLECAPTION: Live scheduling decisions and their visible consequences.** These are control decisions, not claims that every requested movement is visually correct.

| Decision point | Selected behavior | Continuity or latency implication |
|---|---|---|
| No prepared interaction | Generate quiet reading; select a page turn when eligible | The stream remains active without an obligatory spoken monologue |
| A multi-segment answer is already in progress | Claim its next segment before a new queued answer | Preserve reply identity and consecutive delivery; later messages may wait |
| A fresh supported gesture is recognized | Compile the action directly | Avoid an unnecessary text-model call or a separate spoken acknowledgment |
| A new conversational question needs a reply | Use local Qwen3-4B with persona, scene facts and recent conversation | Text planning can overlap the current video request |
| The reply spans several windows | Segment by speakable content and retain viewer attention | A five-second boundary does not end the turn or send her back to the book |
| The complete spoken turn ends and no next reply is ready | Select a silent return-to-reading movement | Return happens after the answer, rather than between its clauses |
| An action candidate or continuation fails screening | Apply the accepted-parent retry policy in Section 4.6 | Withhold rejected output; insufficient buffer can become an explicit wait |
| The allowed future-media slot is occupied | Wait for relay progress before generating another clip | Bound prepared footage instead of accumulating a long response backlog |

### 3.5 From an audience message to visible behavior

The listener timestamps a real message and places it in the durable inbox with deduplication and expiry. The text worker selects an eligible message, obtains intended scene state, and prepares a supported action or a natural reply. Its conversation context includes up to two completed turns from the same viewer within ten minutes and recent host lines for repetition control. A follow-up from a viewer whose preceding turn is still being generated or aired waits for that turn; this avoids planning against an answer the viewer has not yet received.

The worker allows only a small amount of prepared reply work: an already queued response blocks another planning pass, as does an outstanding physical action. Once queued, the producer selects an active continuation first, otherwise the oldest queued reply, otherwise idle reading. The prompt compiler combines the selected reply span or action phase with the established persona, scene facts and accepted audiovisual context. It does not interrupt or rewrite a diffusion request that is already executing.

For an illustrative question such as “What are you reading?”, the logical sequence is message arrival → reply preparation → next available generation slot → a directed spoken segment → validation → publication → relay playback. A supported “nod once” request can bypass conversational generation and go directly to the action compiler. These are protocol examples, not additional measured response-time experiments.

The current public controls include conversation and a limited set of gestures, with selected additional direct actions described in the scene-action catalog. Extended book handling, pouring, window operation, mirror interaction, wardrobe changes and background transitions have separate research status. The general director interface can express those goals, but a compilable instruction or predicted scene change is not visual qualification. In particular, the existing scene ledger records intended and generated/playback state; it does not reliably recognize all objects in every output frame.

### 3.6 Publication, playback and observability

The controller admits a segment only after the continuation checks in Section 4. An accepted media file, native PCM sidecar and receipt become available to the relay, while the accepted latent suffix becomes the next model context. The relay emits fresh accepted clips in sequence using accumulated frame and sample positions. Video is encoded through the persistent H.264 path; original PCM is resampled and encoded once to broadcast AAC. FFmpeg transports the continuous stream to Twitch. Ordinary reading and dialogue have no added background-music bed; explicitly requested singing can use generated accompaniment.

Generation and playback advance independently. With one-clip lookahead, the producer waits when the next allowed slot is already occupied. Relay start/finish events update the dialogue receipts so that “generated,” “being shown,” and “finished” remain distinct. A labelled waiting interval is counted if accepted media arrives too late. This design reduces stale prepared footage but leaves less reserve for retries.

Prometheus and Grafana expose generation latency, validation/publication overhead, playback budget, queue depth, reply milestones, relay waiting, transport output and GPU activity. The useful latency chain is **arrival → text ready → selected → media ready → local playback → first meaningful response within the clip → viewer presentation**. The current instrumentation covers local milestones; screenshot values and local relay timestamps do not measure Twitch player buffering or perceptual response onset. Method-source hashes are included in the [live-logic manifest](evidence/live-logic.json).
