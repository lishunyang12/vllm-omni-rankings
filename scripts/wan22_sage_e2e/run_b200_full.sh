#!/bin/bash
# Single-command B200 (SM100) I2V pipeline. After `git pull`, just:
#   bash scripts/wan22_sage_e2e/run_b200_full.sh
# Runs dense/fp8/int8 SAGE, LPIPS (all + first16), compare frames, mp4 videos,
# results.md — all collected in b200_i2v_out/ next to this script. int8 = SM100 only.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda13}"
export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-$HOME/omni-env/bin/python}"

IMG="${IMG:-i2v_input.png}"
OUT="${OUT:-b200_i2v_out}"
PROMPT="The puppy suddenly leaps up and shakes its whole body, ears flapping and fur flying, then bounds toward the camera - fast, dynamic motion."
mkdir -p "$OUT"

[ -f "$IMG" ] || curl -L -o "$IMG" "https://picsum.photos/id/237/1280/720"
cp -f "$IMG" "$OUT/i2v_input.png"
echo "prompt=$PROMPT"

echo "########## I2V DENSE ##########"
$PY wan22_i2v_bench.py --mode dense --image "$IMG" --prompt "$PROMPT" --save i2v_dense.npy 2>&1 | tee "$OUT/gen_dense.log"
echo "########## I2V FP8 SAGE ##########"
$PY wan22_i2v_bench.py --mode fp8   --image "$IMG" --prompt "$PROMPT" --save i2v_fp8.npy  2>&1 | tee "$OUT/gen_fp8.log"
echo "########## I2V INT8 SAGE ##########"
$PY wan22_i2v_bench.py --mode int8  --image "$IMG" --prompt "$PROMPT" --save i2v_int8.npy 2>&1 | tee "$OUT/gen_int8.log"

echo "########## ANALYSIS + VIDEOS + RESULTS ##########"
OUT="$OUT" PROMPT="$PROMPT" $PY - <<'PY'
import os, re, numpy as np, torch, lpips
from PIL import Image, ImageDraw
out = os.environ["OUT"]; prompt = os.environ["PROMPT"]

def load(p):
    a = np.load(p).squeeze()
    if a.dtype != np.uint8 and float(a.max()) <= 1.0 + 1e-3: a = a * 255.0
    return a.clip(0, 255).astype("uint8")

def to_t(u8):
    return (torch.from_numpy(np.ascontiguousarray(u8)).float() / 127.5 - 1.0).permute(0, 3, 1, 2)

def gt(m):
    try:
        return float(re.search(r"generation\s*:\s*([\d.]+)", open(f"{out}/gen_{m}.log").read()).group(1))
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

res = {nm: {"all": lp(Dt, B, 0, n), "f16": lp(Dt, B, 0, min(16, n))} for nm, B in [("fp8", Ft), ("int8", It)]}
print("\n==== I2V LPIPS vs dense (lower = closer to ref) ====")
for nm in ("fp8", "int8"):
    print(f"{nm:4s}: all {n}f = {res[nm]['all']:.4f}   first16 = {res[nm]['f16']:.4f}")

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

try:
    import imageio.v2 as imageio
    for nm, a in [("dense", D), ("fp8", F), ("int8", I)]:
        wtr = imageio.get_writer(f"{out}/b200_i2v_{nm}.mp4", fps=16, codec="libx264",
                                 quality=6, macro_block_size=8, ffmpeg_params=["-crf","26","-pix_fmt","yuv420p"])
        for fr in a: wtr.append_data(fr)
        wtr.close(); print("wrote", f"{out}/b200_i2v_{nm}.mp4")
except Exception as e:
    print("VIDEO SKIPPED (pip install imageio-ffmpeg):", e)

td, tf, ti = gt("dense"), gt("fp8"), gt("int8")
sec = lambda t: f"{t:.1f} s" if t else "—"
spd = lambda t: f"{td/t:.2f}×" if (td and t) else "—"
L = ["# Wan2.2-I2V-A14B SAGE — B200/SM100, high-motion I2V (Ruqing protocol)\n",
     "Model Wan2.2-I2V-A14B, 1280x720/81f/50steps, torch.compile, CFG 4/4, seed 0. "
     "Cond img: picsum id237 (puppy). LPIPS(alex) vs same-machine dense.\n",
     f"Prompt: _{prompt}_\n",
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
open(f"{out}/results.md", "w").write("\n".join(L))
print("\n".join(L))
PY
echo "########## ALL DONE -> $OUT/ ##########"
ls -lh "$OUT"

echo "########## PUSH TO vllm-omni-rankings ##########"
REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$REPO" ]; then
  cd "$REPO"
  git add "scripts/wan22_sage_e2e/$OUT" scripts/wan22_sage_e2e/.gitignore
  if git -c user.name="lishunyang12" -c user.email="lishunyang12@163.com" \
        commit -m "B200 I2V high-motion SAGE results (dense/fp8/int8) + videos + frames"; then
    git push origin HEAD && echo "PUSHED to $(git remote get-url origin)" \
      || echo "PUSH FAILED — run: gh auth login  (or set a git credential), then: git push origin HEAD"
  else
    echo "nothing new to commit"
  fi
else
  echo "not inside a git repo — outputs are in $OUT/ ; copy them into the rankings repo and push manually"
fi
