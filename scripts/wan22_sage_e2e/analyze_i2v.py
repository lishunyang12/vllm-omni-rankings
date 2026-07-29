"""Standalone I2V analysis: reads i2v_{dense,fp8,int8}.npy in the cwd, emits LPIPS
(all frames + first 16), compare frames, mp4 videos, and results.md into b200_i2v_out/.
Timings are read from b200_i2v_out/gen_{mode}.log if present.
Run: $HOME/omni-env/bin/python analyze_i2v.py"""

import os
import re
import numpy as np
import torch
import lpips
from PIL import Image, ImageDraw

OUT = os.environ.get("OUT", "b200_i2v_out")
PROMPT = os.environ.get(
    "PROMPT",
    "The puppy suddenly leaps up and shakes its whole body, ears flapping and fur flying, "
    "then bounds toward the camera - fast, dynamic motion.",
)
os.makedirs(OUT, exist_ok=True)


def load(p):
    a = np.load(p).squeeze()
    if a.dtype != np.uint8 and float(a.max()) <= 1.0 + 1e-3:
        a = a * 255.0
    return a.clip(0, 255).astype("uint8")


def to_t(u8):
    return (torch.from_numpy(np.ascontiguousarray(u8)).float() / 127.5 - 1.0).permute(0, 3, 1, 2)


def gen_time(mode):
    for path in (f"{OUT}/gen_{mode}.log", f"gen_{mode}.log"):
        try:
            m = re.search(r"generation\s*:\s*([\d.]+)\s*s", open(path).read())
            if m:
                return float(m.group(1))
        except Exception:
            pass
    return None


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D, F, I = load("i2v_dense.npy"), load("i2v_fp8.npy"), load("i2v_int8.npy")
    Dt, Ft, It = to_t(D), to_t(F), to_t(I)
    n = Dt.shape[0]
    loss = lpips.LPIPS(net="alex").to(dev).eval()

    def lp(A, B, lo, hi):
        with torch.no_grad():
            return float(np.mean([loss(A[i:i+1].to(dev), B[i:i+1].to(dev)).item() for i in range(lo, hi)]))

    res = {nm: {"all": lp(Dt, B, 0, n), "f16": lp(Dt, B, 0, min(16, n))}
           for nm, B in [("fp8", Ft), ("int8", It)]}
    print("\n==== I2V LPIPS vs dense (lower = closer to ref) ====")
    for nm in ("fp8", "int8"):
        print(f"{nm:4s}: all {n}f = {res[nm]['all']:.4f}   first16 = {res[nm]['f16']:.4f}")

    def label(im, t):
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 8 * len(t) + 10, 26], fill=(0, 0, 0))
        d.text((5, 6), t, fill=(255, 255, 255))
        return im

    h, w = D.shape[1:3]
    sw, sh = w // 2, h // 2
    for fi in [0, 4, 8, 15]:
        if fi >= n:
            continue
        tiles = [label(Image.fromarray(a[fi]).resize((sw, sh), Image.Resampling.LANCZOS), f"{k} f{fi}")
                 for k, a in [("ref", D), ("fp8", F), ("int8", I)]]
        g = Image.new("RGB", (sw * 3, sh))
        for i, t in enumerate(tiles):
            g.paste(t, (i * sw, 0))
        g.save(f"{OUT}/i2v_cmp_f{fi:02d}.png")
        print("wrote", f"{OUT}/i2v_cmp_f{fi:02d}.png")

    try:
        import imageio.v2 as imageio
        for nm, a in [("dense", D), ("fp8", F), ("int8", I)]:
            wtr = imageio.get_writer(f"{OUT}/b200_i2v_{nm}.mp4", fps=16, codec="libx264",
                                     quality=6, macro_block_size=8,
                                     ffmpeg_params=["-crf", "26", "-pix_fmt", "yuv420p"])
            for fr in a:
                wtr.append_data(fr)
            wtr.close()
            print("wrote", f"{OUT}/b200_i2v_{nm}.mp4")
    except Exception as e:
        print("VIDEO SKIPPED (pip install imageio-ffmpeg):", e)

    td, tf, ti = gen_time("dense"), gen_time("fp8"), gen_time("int8")
    sec = lambda t: f"{t:.1f} s" if t else "—"
    spd = lambda t: f"{td/t:.2f}×" if (td and t) else "—"
    L = ["# Wan2.2-I2V-A14B SAGE — B200/SM100, high-motion I2V (Ruqing protocol)\n",
         "Model Wan2.2-I2V-A14B, 1280x720/81f/50steps, torch.compile, CFG 4/4, seed 0. "
         "Cond img: picsum id237 (puppy). LPIPS(alex) vs same-machine dense.\n",
         f"Prompt: _{PROMPT}_\n",
         "| mode | generation | vs dense | LPIPS all 81f ↓ | LPIPS first 16f ↓ |",
         "|---|---|---|---|---|",
         f"| dense (BF16) | {sec(td)} | — | — | — |",
         f"| **fp8 SAGE** | {sec(tf)} | {spd(tf)} | {res['fp8']['all']:.4f} | **{res['fp8']['f16']:.4f}** |",
         f"| int8 SAGE | {sec(ti)} | {spd(ti)} | {res['int8']['all']:.4f} | **{res['int8']['f16']:.4f}** |\n"]
    b = "int8" if res["int8"]["f16"] <= res["fp8"]["f16"] else "fp8"
    L.append(f"First-16-frame LPIPS: **{b} is closer to dense** "
             f"(int8 {res['int8']['f16']:.4f} vs fp8 {res['fp8']['f16']:.4f}). Image-anchored I2V removes the "
             f"T2V content-divergence confound, so this reflects fidelity, not sampling drift. "
             f"See i2v_cmp_f*.png for fp8-blur vs int8-sharpness.\n")
    open(f"{OUT}/results.md", "w").write("\n".join(L))
    print("\n" + "\n".join(L))
    print("outputs in", OUT + "/")


if __name__ == "__main__":
    main()
