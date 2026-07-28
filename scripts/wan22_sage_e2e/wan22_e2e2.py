import argparse, time, numpy as np, torch
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


def _extract_frames(out):
    o = out[0] if isinstance(out, list) and out else out
    # unwrap pipeline output
    if isinstance(o, OmniRequestOutput) and getattr(o, "request_output", None) is not None:
        o = o.request_output
    frames = getattr(o, "images", None) or o
    # frames: list of PIL images (video) or nested
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        frames = frames[0]
    arrs = []
    for f in frames:
        if hasattr(f, "convert"):  # PIL
            arrs.append(np.asarray(f.convert("RGB")))
        elif isinstance(f, np.ndarray):
            arrs.append(f)
        elif torch.is_tensor(f):
            arrs.append(f.detach().cpu().numpy())
    return np.stack(arrs) if arrs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dense", "sage"], required=True)
    p.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--save", default=None)
    p.add_argument("--eager", action="store_true", help="disable torch.compile")
    args = p.parse_args()

    if args.mode == "dense":
        attn = {"default": {"backend": "TRTLLM_ATTN"}}
    else:
        attn = {"default": {"backend": "TRTLLM_ATTN",
                            "quant": {"dtype_qk": "fp8_e4m3", "q_block_size": 1, "k_block_size": 16}}}

    print(f"[e2e] mode={args.mode} eager={args.eager} {args.width}x{args.height} {args.frames}f {args.steps}steps", flush=True)
    print(f"[e2e] attn={attn}", flush=True)
    t_load = time.perf_counter()
    omni = Omni(model=args.model, enforce_eager=args.eager, diffusion_attention_config=attn)
    print(f"[e2e] model loaded in {time.perf_counter()-t_load:.1f}s", flush=True)

    sp = OmniDiffusionSamplingParams(
        height=args.height, width=args.width, num_frames=args.frames,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    prompt = {"prompt": "A serene lakeside sunrise with mist over the water."}

    t0 = time.perf_counter()
    out = omni.generate(prompt, sp)
    dt = time.perf_counter() - t0
    print(f"\n[e2e] === RESULT mode={args.mode} ===", flush=True)
    print(f"[e2e] generation time : {dt:.2f} s  ({dt/args.steps:.3f} s/step)", flush=True)

    arr = _extract_frames(out)
    print(f"[e2e] frames shape     : {None if arr is None else arr.shape}", flush=True)
    if arr is not None and args.save:
        np.save(args.save, arr)
        print(f"[e2e] saved frames -> {args.save}", flush=True)
    print("[e2e] DONE", flush=True)


if __name__ == "__main__":
    main()
