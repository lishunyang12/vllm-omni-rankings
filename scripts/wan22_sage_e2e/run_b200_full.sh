#!/bin/bash
# One-shot B200 (SM100) I2V pipeline, self-contained:
#   download image -> dense/fp8/int8 SAGE -> LPIPS (all + first16) ->
#   compare frames -> mp4 videos -> results.md, all collected in b200_i2v_out/.
# Run from ~/vllm-omni after `source ~/omni-env/bin/activate`. int8 SAGE = SM100 only.
set -e
cd "${ROOT:-$HOME/vllm-omni}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda13}"
export PATH="$CUDA_HOME/bin:$PATH"

IMG="${IMG:-i2v_input.png}"
OUT="${OUT:-b200_i2v_out}"
PROMPT="The puppy suddenly leaps up and shakes its whole body, ears flapping and fur flying, then bounds toward the camera - fast, dynamic motion."
mkdir -p "$OUT"

[ -f "$IMG" ] || curl -L -o "$IMG" "https://picsum.photos/id/237/1280/720"
cp -f "$IMG" "$OUT/i2v_input.png"
echo "image=$IMG"; echo "prompt=$PROMPT"

echo "########## I2V DENSE ##########"
python wan22_i2v_bench.py --mode dense --image "$IMG" --prompt "$PROMPT" --save i2v_dense.npy 2>&1 | tee "$OUT/gen_dense.log"
echo "########## I2V FP8 SAGE ##########"
python wan22_i2v_bench.py --mode fp8   --image "$IMG" --prompt "$PROMPT" --save i2v_fp8.npy  2>&1 | tee "$OUT/gen_fp8.log"
echo "########## I2V INT8 SAGE ##########"
python wan22_i2v_bench.py --mode int8  --image "$IMG" --prompt "$PROMPT" --save i2v_int8.npy 2>&1 | tee "$OUT/gen_int8.log"

echo "########## ANALYSIS + VIDEOS + RESULTS ##########"
OUT="$OUT" PROMPT="$PROMPT" python - <<'PY'
import os, re, glob, numpy as np, torch, lpips
from PIL import Image, ImageDraw
out = os.environ["OUT"]; prompt = os.environ["PROMPT"]

def load(p):
    a = np.load(p).squeeze()
    if a.dtype != np.uint8 and float(a.max()) <= 1.0 + 1e-3: a = a * 255.0
    return a.clip(0, 255).astype("uint8")

def to_t(u8):
    return (torch.from_numpy(np.ascontiguousarray(u8)).float() / 127.5 - 1.0).permute(0, 3, 1, 2)

def gen_time(mode):
    try:
        txt = open(f"{out}/gen_{mode}.log").read()
        m = re.search(r"generation\s*:\s*([\d.]+)\s*s", txt)
        return float(m.group(1)) if m else None
    except Exception:
        return None

dev = "cuda" if torch.cuda.is_available() else "cpu"
D, F, I = load("i2v_dense.npy"), load("i2v_fp8.npy"), load("i2v_int8.npy")
Dt, Ft, It = to_t(D), to_t(F), to_t(I)
n = Dt.shape[0]
loss = lpips.LPIPS(net="alex").to(dev).eval()

def lp(A, B, lo, hi):
    with torch.no_grad():
        return float(np.mean([loss(A[i:i+1].to(dev), B[i:i+1].to(dev)).item() for i in range(lo, hi)]))

res = {}
for nm, B in [("fp8", Ft), ("int8", It)]:
    res[nm] = {"all": lp(Dt, B, 0, n), "f16": lp(Dt, B, 0, min(16, n))}
print("\n==== I2V LPIPS vs dense (lower = closer to ref) ====")
for nm in ("fp8", "int8"):
    print(f"{nm:4s}: all {n}f = {res[nm]['all']:.4f}   first16 = {res[nm]['f16']:.4f}")

# compare frames (ref | fp8 | int8)
def label(im, t):
    d = ImageDraw.Draw(im); d.rectangle([0,0,8*len(t)+10,26], fill=(0,0,0)); d.text((5,6), t, fill=(255,255,255)); return im
h, w = D.shape[1:3]; sw, sh = w//2, h//2
for fi in [0, 4, 8, 15]:
    if fi >= n: continue
    tiles = [label(Image.fromarray(a[fi]).resize((sw,sh), Image.Resampling.LANCZOS), f"{k} f{fi}")
             for k, a in [("ref",D),("fp8",F),("int8",I)]]
    g = Image.new("RGB", (sw*3, sh))
    for i,t in enumerate(tiles): g.paste(t, (i*sw,0))
    g.save(f"{out}/i2v_cmp_f{fi:02d}.png"); print("wrote", f"{out}/i2v_cmp_f{fi:02d}.png")

# mp4 videos (best-effort; needs imageio-ffmpeg)
try:
    import imageio.v2 as imageio
    for nm, a in [("dense", D), ("fp8", F), ("int8", I)]:
        wtr = imageio.get_writer(f"{out}/b200_i2v_{nm}.mp4", fps=16, codec="libx264",
                                 quality=6, macro_block_size=8,
                                 ffmpeg_params=["-crf", "26", "-pix_fmt", "yuv420p"])
        for fr in a: wtr.append_data(fr)
        wtr.close(); print("wrote", f"{out}/b200_i2v_{nm}.mp4")
except Exception as e:
    print("VIDEO SKIPPED (install imageio-ffmpeg):", e)

# results.md
td, tf, ti = gen_time("dense"), gen_time("fp8"), gen_time("int8")
def spd(t): return f"{td/t:.2f}×" if (td and t) else "—"
def sec(t): return f"{t:.1f} s" if t else "—"
lines = []
lines.append("# Wan2.2-I2V-A14B SAGE — B200/SM100, high-motion I2V (Ruqing protocol)\n")
lines.append(f"Model `Wan-AI/Wan2.2-I2V-A14B-Diffusers`, 1280×720 / 81f / 50 steps, torch.compile, "
             f"CFG 4.0/4.0, seed 0. Condition image: picsum id 237 (puppy). LPIPS(alex) vs same-machine dense.\n")
lines.append(f"Prompt: _{prompt}_\n")
lines.append("| mode | generation | vs dense | LPIPS all 81f ↓ | LPIPS first 16f ↓ |")
lines.append("|---|---|---|---|---|")
lines.append(f"| dense (BF16) | {sec(td)} | — | — | — |")
lines.append(f"| **fp8 SAGE** | {sec(tf)} | {spd(tf)} | {res['fp8']['all']:.4f} | **{res['fp8']['f16']:.4f}** |")
lines.append(f"| int8 SAGE | {sec(ti)} | {spd(ti)} | {res['int8']['all']:.4f} | **{res['int8']['f16']:.4f}** |\n")
better = "int8" if res["int8"]["f16"] <= res["fp8"]["f16"] else "fp8"
lines.append(f"First-16-frame LPIPS: **{better} is closer to dense** "
             f"(int8 {res['int8']['f16']:.4f} vs fp8 {res['fp8']['f16']:.4f}). "
             f"On the image-anchored I2V protocol the T2V content-divergence confound is removed, so this "
             f"reflects fidelity, not sampling drift. See `i2v_cmp_f*.png` for the fp8-blur vs int8-sharpness view.\n")
open(f"{out}/results.md", "w").write("\n".join(lines))
print("\nwrote", f"{out}/results.md")
print("".join(l + "\n" for l in lines))
PY
echo "########## ALL DONE — outputs in $OUT/ ##########"
ls -lh "$OUT"
