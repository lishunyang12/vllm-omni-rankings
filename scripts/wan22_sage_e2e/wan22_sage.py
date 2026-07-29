"""Wan2.2-A14B e2e with TRTLLM_ATTN SAGE quantization.

Usage:
    python wan22_sage.py --mode fp8  --steps 8      # smoke test (pipeline + SAGE kernel)
    python wan22_sage.py --mode int8 --steps 8      # int8 SAGE (SM100/B200 only)
    python wan22_sage.py --mode dense --steps 50     # BF16 baseline
    python wan22_sage.py --mode fp8  --steps 50 --save frames.npy

Requires: vllm-omni (trtllm-sage-quant branch) + flashinfer >= 0.6.16rc1, on datacenter
Blackwell (SM100 B200/GB200 or SM103 B300). int8 QK kernels are SM100-only.
"""

import argparse
import time

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


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
    p.add_argument("--mode", choices=["dense", "fp8", "int8"], default="fp8")
    p.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--save", default=None)
    p.add_argument("--compile", action="store_true", help="enable torch.compile (default: eager)")
    a = p.parse_args()

    attn = {"default": {"backend": "TRTLLM_ATTN"}}
    if a.mode != "dense":
        dtype_qk = "fp8_e4m3" if a.mode == "fp8" else "int8"
        attn["default"]["quant"] = {"dtype_qk": dtype_qk, "q_block_size": 1, "k_block_size": 16}

    print(f"[e2e] mode={a.mode}  {a.w}x{a.h}  {a.frames}f  {a.steps}steps  attn={attn}", flush=True)
    t_load = time.perf_counter()
    omni = Omni(model=a.model, enforce_eager=not a.compile, diffusion_attention_config=attn)
    print(f"[e2e] model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    sp = OmniDiffusionSamplingParams(
        height=a.h,
        width=a.w,
        num_frames=a.frames,
        num_inference_steps=a.steps,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    t0 = time.perf_counter()
    out = omni.generate({"prompt": "A serene lakeside sunrise with mist over the water."}, sp)
    dt = time.perf_counter() - t0

    o = out[0] if isinstance(out, list) and out else out
    print(f"\n[e2e] === RESULT mode={a.mode} ===", flush=True)
    print(f"[e2e] generation : {dt:.2f} s  ({dt / a.steps:.3f} s/step)", flush=True)
    print(f"[e2e] finished   : {getattr(o, 'finished', None)}", flush=True)

    if a.save:
        arr = extract_frames(out)
        if arr is not None:
            np.save(a.save, arr)
            print(f"[e2e] frames {arr.shape} -> {a.save}", flush=True)
    print("[e2e] DONE", flush=True)


if __name__ == "__main__":
    main()
