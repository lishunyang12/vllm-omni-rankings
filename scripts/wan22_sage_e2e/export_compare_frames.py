"""Side-by-side frame stills to show the qualitative fp8-vs-int8 difference vs a dense ref.
Usage: python export_compare_frames.py dense.npy fp8.npy int8.npy --frames 0 4 8 --out cmp
Emits cmp_fXX.png (ref | fp8 | int8, labeled) so fp8 blur/distortion vs int8 content-drift is visible."""

import argparse
import numpy as np
from PIL import Image, ImageDraw


def load(path):
    a = np.load(path).squeeze()
    if a.dtype != np.uint8 and float(a.max()) <= 1.0 + 1e-3:
        a = a * 255.0
    return a.clip(0, 255).astype("uint8")


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 * len(text) + 10, 26], fill=(0, 0, 0))
    d.text((5, 6), text, fill=(255, 255, 255))
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dense")
    p.add_argument("fp8")
    p.add_argument("int8", nargs="?", default=None)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 4, 8, 15])
    p.add_argument("--out", default="cmp")
    p.add_argument("--scale", type=float, default=0.5, help="downscale for smaller files")
    args = p.parse_args()

    cols = [("ref (dense)", load(args.dense)), ("fp8 SAGE", load(args.fp8))]
    if args.int8:
        cols.append(("int8 SAGE", load(args.int8)))

    n = min(c[1].shape[0] for c in cols)
    h, w = cols[0][1].shape[1:3]
    sw, sh = int(w * args.scale), int(h * args.scale)
    for fi in args.frames:
        if fi >= n:
            continue
        tiles = []
        for name, arr in cols:
            im = Image.fromarray(arr[fi]).resize((sw, sh), Image.Resampling.LANCZOS)
            tiles.append(label(im, f"{name} f{fi}"))
        grid = Image.new("RGB", (sw * len(tiles), sh))
        for i, t in enumerate(tiles):
            grid.paste(t, (i * sw, 0))
        out = f"{args.out}_f{fi:02d}.png"
        grid.save(out)
        print("wrote", out, grid.size)


if __name__ == "__main__":
    main()
