# Cosmos3-Edge promo — v3 (current)

`cosmos_edge_promo_v3.mp4` — 1280×720, ~41.5s, English VO, no presenter.
**Every demo clip is real Cosmos3-Edge output generated locally via vLLM-Omni**
(B300, `cosmos-venv` = vllm 0.26.0 + vllm-omni). Style follows the official
NVIDIA Cosmos launch: full-screen demo + bottom caption/provenance bar, an
FP8 side-by-side with speed/LPIPS, and Edge spec cards.

> v1/v2 (below) are superseded: they used official NVIDIA demo assets (Nano-era)
> and generic footage. v3 replaces ALL footage with real Edge-checkpoint output.

## Segments

| # | segment | source clip | on-screen |
|---|---|---|---|
| 1 | Title card | — | Cosmos3-Edge · Physical AI |
| 2 | **Hero — I2V** | `v3_i2v_driving.mp4` | one dashcam frame → full driving rollout |
| 3 | T2V | `v3_t2v_sort.mp4` | robotic sorting from a prompt |
| 4 | Physical-AI action | `v3_i2v_umi.mp4` | first-person manipulation rollout |
| 5 | **FP8 side-by-side** | `edge_t2v_official.mp4` \| `_fp8.mp4` | dense vs fp8 |
| 6 | Spec / close card | — | Edge vs Nano/Super + where it runs |

## How each clip was generated (official Edge params: 480×832 · gs 5.0 · flow_shift 3.0)

All via the offline examples on `nvidia/Cosmos3-Edge`, `--extra-body '{"flow_shift": 3.0, "max_sequence_length": 4096, "guardrails": false}'`.

- **I2V driving (hero)** — `image_to_video.py`, input = official coastal-highway dashcam frame, 189 frames, prompt: *"The car drives forward along the winding coastal highway, following the road as it curves to the left past the cliff, guardrail and ocean passing by on the right, smooth continuous motion."*
- **T2V sort** — `text_to_video.py`, 93 frames, prompt: *"A robotic arm slowly sorts colorful plastic blocks into small bins on a clean white table, bright even studio lighting, steady controlled motion."*
- **Action UMI** — `image_to_video.py`, input = official UMI first-frame, 93 frames.
- **FP8 pair** — `text_to_video.py`, 49 frames, *"A robot arm is cleaning a plate in the kitchen."*, dense vs `--quantization fp8`.

> ⚠️ Cosmos3-Edge MUST use 480×832 / gs 5.0 / flow_shift 3.0. The default
> `cosmos` preset (720×1280 / gs 6.0 / flow_shift 10.0 = Nano/Super) produces
> pure-noise output on Edge — fixed upstream in
> [vllm-omni#5596](https://github.com/vllm-project/vllm-omni/pull/5596).

## FP8 numbers (Edge, 832×480, 49f, B300)

| | dense (BF16) | FP8 |
|---|---|---|
| generation | 7.74 s | **6.71 s** (1.15×) |
| dense-vs-fp8 LPIPS(alex) | — | **0.169** |

## Rebuild

```bash
# footage: scratchpad/cosmos_edge_gen/gen_v3_batch.sh + gen_v3_batch2.sh
# cards/overlays: make_cards.py   VO: piper en_US-lessac-medium
bash scratchpad/cosmos_edge_gen/assemble_v3.sh
```

---

## v1 / v2 (superseded — reference/official-asset cuts)

`cosmos_edge_promo_v1.mp4` — 1920×1080, ~19.5s, silent. Official NVIDIA
Cosmos3-Edge demo assets + family spec cards.

`cosmos_edge_promo_v2.mp4` — 25.5s, English narration (piper `en_US-lessac-medium`;
Qwen3-TTS was attempted but its engine crashed with `OmniEngineDeadError`, so
piper was the fallback). `vo.wav` is the raw v2 narration.
