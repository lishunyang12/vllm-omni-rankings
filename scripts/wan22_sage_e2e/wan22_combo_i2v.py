"""Wan2.2-I2V-A14B: SDPA/dense/int8-SAGE/skip/combo for the locked combo experiment (image-anchored).
Mirrors wan22_combo.py but for I2V (needs --image, builds an I2V prompt).
Per-expert skip-softmax calibration is injected via CALIB; set it to the I2V checkpoint's own
sparse_attention_config coefficients (the T2V values below are a PLACEHOLDER and are not exact for I2V)."""

import argparse, time, numpy as np, torch
from PIL import Image
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_image_to_video_prompt, get_model_class_name
from vllm_omni.outputs import OmniRequestOutput

NEG = "blurry, low quality, distorted, deformed, watermark, text, static, jpeg artifacts"

IGNORE = (
    ["blocks.0.attn1", "blocks.0.attn2", "blocks.1.attn1", "blocks.1.attn2"]
    + [f"blocks.{i}.attn2" for i in range(2, 38)]
    + ["blocks.38.attn1", "blocks.38.attn2", "blocks.39.attn1", "blocks.39.attn2"]
)
# PLACEHOLDER: these a/b are from the T2V checkpoint. Replace with the I2V checkpoint's
# sparse_attention_config coefficients for a faithful I2V skip-softmax run.
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
    p.add_argument("--mode", choices=["dense", "sdpa", "int8", "skip", "combo"], required=True)
    p.add_argument("--sparsity", type=float, default=0.7)
    p.add_argument("--until", type=float, default=0.94)
    p.add_argument("--threshold", type=float, default=None,
                   help="calibration-free skip: use skip_softmax_threshold instead of target_sparsity (no a/b needed)")
    p.add_argument("--model", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="The scene comes to life with smooth, natural motion.")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--guidance2", type=float, default=4.0)
    p.add_argument("--save", default=None)
    a = p.parse_args()

    attn = {"default": {"backend": "TORCH_SDPA" if a.mode == "sdpa" else "TRTLLM_ATTN"}}
    if a.mode in ("int8", "combo"):
        attn["default"]["quant"] = {"dtype_qk": "int8"}
    if a.mode in ("skip", "combo"):
        if a.threshold is not None:
            attn["default"]["skip_softmax"] = {"threshold": a.threshold, "disabled_until_timestep": a.until}
        else:
            attn["default"]["skip_softmax"] = {"target_sparsity": a.sparsity, "disabled_until_timestep": a.until}
            attn["default"]["skip_calibration"] = CALIB

    if a.mode in ("skip", "combo"):
        skip_tag = f"t{a.threshold}" if a.threshold is not None else f"s{a.sparsity}"
        tag = f"{a.mode}@{skip_tag}u{a.until}"
    else:
        tag = a.mode
    print(f"[i2v-combo] {tag} {a.w}x{a.h} {a.frames}f {a.steps}steps compile=ON CFG={a.guidance}/{a.guidance2}", flush=True)

    img = Image.open(a.image).convert("RGB").resize((a.w, a.h), Image.Resampling.LANCZOS)
    prompt_dict = build_image_to_video_prompt(
        get_model_class_name(a.model), a.prompt, NEG, {"image": img}, a.h, a.w, a.frames
    )

    t_load = time.perf_counter()
    omni = Omni(model=a.model, enforce_eager=False, diffusion_attention_config=attn)
    print(f"[i2v-combo] model loaded in {time.perf_counter()-t_load:.1f}s", flush=True)

    sp = OmniDiffusionSamplingParams(
        height=a.h, width=a.w, num_frames=a.frames, num_inference_steps=a.steps,
        guidance_scale=a.guidance, guidance_scale_2=a.guidance2,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    t0 = time.perf_counter()
    out = omni.generate(prompt_dict, sp)
    dt = time.perf_counter() - t0
    o = out[0] if isinstance(out, list) and out else out
    print(f"\n[i2v-combo] === RESULT {tag} ===", flush=True)
    print(f"[i2v-combo] generation : {dt:.2f} s  ({dt/a.steps:.3f} s/step)", flush=True)
    print(f"[i2v-combo] finished   : {getattr(o, 'finished', None)}", flush=True)
    if a.save:
        arr = extract_frames(out)
        if arr is not None:
            np.save(a.save, arr); print(f"[i2v-combo] frames {arr.shape} -> {a.save}", flush=True)
    print("[i2v-combo] DONE", flush=True)


if __name__ == "__main__":
    main()
