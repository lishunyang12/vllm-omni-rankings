"""Wan2.2-I2V-A14B image-to-video SAGE benchmark.
Standard load: 1280x720 / 81f / 50 steps, torch.compile, CFG on both experts.
Needs an --image (first-frame condition)."""

import argparse, time, numpy as np, torch
from PIL import Image
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_image_to_video_prompt, get_model_class_name
from vllm_omni.outputs import OmniRequestOutput

NEG = "blurry, low quality, distorted, deformed, watermark, text, static, jpeg artifacts"


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
    p.add_argument("--mode", choices=["dense", "fp8", "int8"], required=True)
    p.add_argument("--model", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    p.add_argument("--image", required=True, help="first-frame condition image")
    p.add_argument("--prompt", default="The scene comes to life with smooth, natural motion.")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--guidance2", type=float, default=4.0)
    p.add_argument("--save", default=None)
    a = p.parse_args()

    attn = {"default": {"backend": "TRTLLM_ATTN"}}
    if a.mode != "dense":
        dtype_qk = "fp8_e4m3" if a.mode == "fp8" else "int8"
        attn["default"]["quant"] = {"dtype_qk": dtype_qk, "q_block_size": 1, "k_block_size": 16}

    print(f"[i2v] mode={a.mode} {a.w}x{a.h} {a.frames}f {a.steps}steps compile=ON CFG={a.guidance}/{a.guidance2}", flush=True)
    img = Image.open(a.image).convert("RGB").resize((a.w, a.h), Image.Resampling.LANCZOS)
    prompt_dict = build_image_to_video_prompt(
        get_model_class_name(a.model), a.prompt, NEG, {"image": img}, a.h, a.w, a.frames
    )

    t_load = time.perf_counter()
    omni = Omni(model=a.model, enforce_eager=False, diffusion_attention_config=attn)
    print(f"[i2v] model loaded in {time.perf_counter()-t_load:.1f}s", flush=True)

    sp = OmniDiffusionSamplingParams(
        height=a.h, width=a.w, num_frames=a.frames, num_inference_steps=a.steps,
        guidance_scale=a.guidance, guidance_scale_2=a.guidance2,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    t0 = time.perf_counter()
    out = omni.generate(prompt_dict, sp)
    dt = time.perf_counter() - t0
    o = out[0] if isinstance(out, list) and out else out
    print(f"\n[i2v] === RESULT mode={a.mode} ===", flush=True)
    print(f"[i2v] generation : {dt:.2f} s  ({dt/a.steps:.3f} s/step)", flush=True)
    print(f"[i2v] finished   : {getattr(o, 'finished', None)}", flush=True)
    if a.save:
        arr = extract_frames(out)
        if arr is not None:
            np.save(a.save, arr); print(f"[i2v] frames {arr.shape} -> {a.save}", flush=True)
    print("[i2v] DONE", flush=True)


if __name__ == "__main__":
    main()
