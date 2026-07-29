"""Standard Wan2.2-A14B benchmark: 1280x720 / 81f / 50 steps, torch.compile, CFG on
both high-noise and low-noise experts (guidance>1 on both + negative prompt)."""

import argparse, time, numpy as np, torch
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
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
    p.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--guidance2", type=float, default=4.0)
    p.add_argument("--skip-threshold", type=float, default=None, help="Skip-Softmax, calibration-free threshold path")
    p.add_argument("--target-sparsity", type=float, default=None, help="Skip-Softmax via a*exp(b*sparsity) curve")
    p.add_argument("--calib-a", type=float, default=None, help="manual calibration a (needs sage-skip-calib branch)")
    p.add_argument("--calib-b", type=float, default=None, help="manual calibration b")
    p.add_argument("--skip-until", type=float, default=0.0, help="disabled_until_timestep (0 = always on)")
    p.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.")
    p.add_argument("--save", default=None)
    a = p.parse_args()

    attn = {"default": {"backend": "TRTLLM_ATTN"}}
    if a.mode != "dense":
        dtype_qk = "fp8_e4m3" if a.mode == "fp8" else "int8"
        attn["default"]["quant"] = {"dtype_qk": dtype_qk, "q_block_size": 1, "k_block_size": 16}
    if a.skip_threshold is not None:
        attn["default"]["skip_softmax"] = {"threshold": a.skip_threshold, "disabled_until_timestep": a.skip_until}
        skip_str = f"skip@thr={a.skip_threshold}(until={a.skip_until})"
    elif a.target_sparsity is not None:
        attn["default"]["skip_softmax"] = {"target_sparsity": a.target_sparsity, "disabled_until_timestep": a.skip_until}
        if a.calib_a is not None and a.calib_b is not None:
            attn["default"]["skip_calibration"] = {"a": a.calib_a, "b": a.calib_b}
        skip_str = f"skip@sparsity={a.target_sparsity}(a={a.calib_a},b={a.calib_b},until={a.skip_until})"
    else:
        skip_str = "no-skip"
    print(f"[bench] mode={a.mode} {skip_str} {a.w}x{a.h} {a.frames}f {a.steps}steps compile=ON "
          f"CFG guidance={a.guidance}/{a.guidance2} (both experts)", flush=True)
    t_load = time.perf_counter()
    omni = Omni(model=a.model, enforce_eager=False, diffusion_attention_config=attn)
    print(f"[bench] model loaded in {time.perf_counter()-t_load:.1f}s", flush=True)

    sp = OmniDiffusionSamplingParams(
        height=a.h, width=a.w, num_frames=a.frames, num_inference_steps=a.steps,
        guidance_scale=a.guidance, guidance_scale_2=a.guidance2,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    prompt = {"prompt": a.prompt, "negative_prompt": NEG}

    t0 = time.perf_counter()
    out = omni.generate(prompt, sp)
    dt = time.perf_counter() - t0
    o = out[0] if isinstance(out, list) and out else out
    print(f"\n[bench] === RESULT mode={a.mode} (CFG on) ===", flush=True)
    print(f"[bench] generation : {dt:.2f} s  ({dt/a.steps:.3f} s/step)", flush=True)
    print(f"[bench] finished   : {getattr(o, 'finished', None)}", flush=True)
    if a.save:
        arr = extract_frames(out)
        if arr is not None:
            np.save(a.save, arr); print(f"[bench] frames {arr.shape} -> {a.save}", flush=True)
    print("[bench] DONE", flush=True)


if __name__ == "__main__":
    main()
