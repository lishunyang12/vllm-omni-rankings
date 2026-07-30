# Cosmos3-Edge on vLLM-Omni — promo (v1)

`cosmos_edge_promo_v1.mp4` — 1920×1080, 24fps, ~19.5s, silent (VO to be added; no music).

Timeline: title → robot UMI (real-time policy) → AV driving (world model) → AV night
(text/image/video/action) → Cosmos3 family comparison card → closing.

Footage: official NVIDIA Cosmos3-Edge demo assets (HF `nvidia/Cosmos3-Edge/assets`).
Cards rendered from the official model-family spec; "FP8 + SAGE accelerated on vLLM-Omni today"
footer reflects vLLM-Omni's shipped `--quantization fp8` + SAGE attention support.

VO script (~20s, for Qwen3-TTS):
"Meet Cosmos3-Edge on vLLM-Omni — NVIDIA's 4-billion-parameter world model for Physical AI.
Understand, simulate, and interact with the physical world — from text, image, and video to
robot action, in one compact checkpoint. The smallest, fastest Cosmos — built for the edge,
from Jetson to RTX. Accelerated today with FP8 and SAGE. Online and offline. Available now."

## v2 (with VO)
`cosmos_edge_promo_v2.mp4` — 25.5s, English narration (piper `en_US-lessac-medium`).
Qwen3-TTS was attempted first but its vLLM-Omni engine core crashed in this env
(`OmniEngineDeadError`); fell back to piper (offline neural TTS) to deliver the VO.
Re-timed to 25.5s so the video matches the narration. `vo.wav` is the raw narration.
