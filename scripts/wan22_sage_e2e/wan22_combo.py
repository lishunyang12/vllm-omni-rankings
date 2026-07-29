"""Wan2.2-T2V-A14B: int8 SAGE x skip-softmax combo for sweet-spot search.
Per-expert skip-softmax calibration is hardcoded from the (deleted) checkpoint sparse_attention_config.
modes: dense | int8 | skip | combo ; --sparsity sets skip-softmax target_sparsity."""

import argparse, time, numpy as np, torch
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

NEG = "blurry, low quality, distorted, deformed, watermark, text, static, jpeg artifacts"

# skip-softmax applies only to self-attn (attn1) of blocks 2..37; everything else is ignored.
IGNORE = (
    ["blocks.0.attn1", "blocks.0.attn2", "blocks.1.attn1", "blocks.1.attn2"]
    + [f"blocks.{i}.attn2" for i in range(2, 38)]
    + ["blocks.38.attn1", "blocks.38.attn2", "blocks.39.attn1", "blocks.39.attn2"]
)
CALIB = {"by_expert": {
    "transformer":   {"a": 2142.7334009837796, "b": 4.282667871834358, "ignore": IGNORE},
    "transformer_2": {"a": 314.68242763240454, "b": 6.166472427782809, "ignore": IGNORE},
}}


def extract_frames(out):
    o = out[0] if isinstance(out, list) and out else out
    if isinstance(o, OmniRequestOutput) and getattr(o, "request_output", None) is not None:
        o = o.request_output
    frames = getattr(o, "images", None) or o
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        frames = frames[0]
    arrs = []
    for f in frames:
        if hasattr(f, "convert"):
            arrs.append(np.asarray(f.convert("RGB")))
        elif isinstance(f, np.ndarray):
            arrs.append(f)
        elif torch.is_tensor(f):
            arrs.append(f.detach().cpu().numpy())
    return np.stack(arrs) if arrs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dense", "int8", "skip", "combo"], required=True)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--guidance2", type=float, default=4.0)
    p.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.")
    p.add_argument("--save", default=None)
    a = p.parse_args()

    attn = {"default": {"backend": "TRTLLM_ATTN"}}
    if a.mode in ("int8", "combo"):
        attn["default"]["quant"] = {"dtype_qk": "int8"}
    if a.mode in ("skip", "combo"):
        attn["default"]["skip_softmax"] = {"target_sparsity": a.sparsity}
        attn["default"]["skip_calibration"] = CALIB

    tag = f"{a.mode}" + (f"@s{a.sparsity}" if a.mode in ("skip", "combo") else "")
    print(f"[combo] {tag} {a.w}x{a.h} {a.frames}f {a.steps}steps compile=ON CFG={a.guidance}/{a.guidance2}", flush=True)

    t_load = time.perf_counter()
    omni = Omni(model=a.model, enforce_eager=False, diffusion_attention_config=attn)
    print(f"[combo] model loaded in {time.perf_counter()-t_load:.1f}s", flush=True)

    sp = OmniDiffusionSamplingParams(
        height=a.h, width=a.w, num_frames=a.frames, num_inference_steps=a.steps,
        guidance_scale=a.guidance, guidance_scale_2=a.guidance2,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    t0 = time.perf_counter()
    out = omni.generate({"prompt": a.prompt, "negative_prompt": NEG}, sp)
    dt = time.perf_counter() - t0
    o = out[0] if isinstance(out, list) and out else out
    print(f"\n[combo] === RESULT {tag} ===", flush=True)
    print(f"[combo] generation : {dt:.2f} s  ({dt/a.steps:.3f} s/step)", flush=True)
    print(f"[combo] finished   : {getattr(o, 'finished', None)}", flush=True)
    if a.save:
        arr = extract_frames(out)
        if arr is not None:
            np.save(a.save, arr); print(f"[combo] frames {arr.shape} -> {a.save}", flush=True)
    print("[combo] DONE", flush=True)


if __name__ == "__main__":
    main()
