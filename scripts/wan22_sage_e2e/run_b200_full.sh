#!/bin/bash
# One-shot B200 (SM100) FULL pipeline: T2V + I2V, each dense/fp8/int8 SAGE, then
# LPIPS (all frames + first 16, Ruqing protocol) and side-by-side compare frames.
# Self-contained analysis (no dependency on any lpips_bench.py version).
# Run from ~/vllm-omni after `source ~/omni-env/bin/activate`. int8 SAGE = SM100 only.
set -e
cd "${ROOT:-$HOME/vllm-omni}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda13}"
export PATH="$CUDA_HOME/bin:$PATH"

IMG="${IMG:-i2v_input.png}"
PROMPT="The puppy suddenly leaps up and shakes its whole body, ears flapping and fur flying, then bounds toward the camera - fast, dynamic motion."

analyze() {   # $1 = prefix (t2v | i2v)
  PREFIX="$1" python - <<'PY'
import os, numpy as np, torch, lpips
from PIL import Image, ImageDraw
pre = os.environ["PREFIX"]

def load(p):
    a = np.load(p).squeeze()
    if a.dtype != np.uint8 and float(a.max()) <= 1.0 + 1e-3:
        a = a * 255.0
    return a.clip(0, 255).astype("uint8")

def to_t(u8):
    return (torch.from_numpy(np.ascontiguousarray(u8)).float() / 127.5 - 1.0).permute(0, 3, 1, 2)

dev = "cuda" if torch.cuda.is_available() else "cpu"
D, F, I = load(f"{pre}_dense.npy"), load(f"{pre}_fp8.npy"), load(f"{pre}_int8.npy")
Dt, Ft, It = to_t(D), to_t(F), to_t(I)
n = Dt.shape[0]
loss = lpips.LPIPS(net="alex").to(dev).eval()

def lp(A, B, lo, hi):
    with torch.no_grad():
        return float(np.mean([loss(A[i:i+1].to(dev), B[i:i+1].to(dev)).item() for i in range(lo, hi)]))

print(f"\n==== {pre.upper()} LPIPS vs dense (lower = closer to ref) ====")
for nm, B in [("fp8 ", Ft), ("int8", It)]:
    print(f"{nm}: all {n}f = {lp(Dt,B,0,n):.4f}   first16 = {lp(Dt,B,0,min(16,n)):.4f}")

def label(im, t):
    d = ImageDraw.Draw(im); d.rectangle([0,0,8*len(t)+10,26], fill=(0,0,0)); d.text((5,6), t, fill=(255,255,255)); return im

h, w = D.shape[1:3]; sw, sh = w//2, h//2
for fi in [0, 4, 8, 15]:
    if fi >= n: continue
    tiles = [label(Image.fromarray(a[fi]).resize((sw,sh), Image.Resampling.LANCZOS), f"{k} f{fi}")
             for k, a in [("ref",D),("fp8",F),("int8",I)]]
    g = Image.new("RGB", (sw*3, sh))
    for i,t in enumerate(tiles): g.paste(t, (i*sw,0))
    g.save(f"{pre}_cmp_f{fi:02d}.png"); print("wrote", f"{pre}_cmp_f{fi:02d}.png")
PY
}

echo "##################### T2V #####################"
echo "########## T2V DENSE ##########"
python wan22_bench.py --mode dense --save t2v_dense.npy
echo "########## T2V FP8 SAGE ##########"
python wan22_bench.py --mode fp8   --save t2v_fp8.npy
echo "########## T2V INT8 SAGE ##########"
python wan22_bench.py --mode int8  --save t2v_int8.npy
analyze t2v

echo "##################### I2V #####################"
[ -f "$IMG" ] || curl -L -o "$IMG" "https://picsum.photos/id/237/1280/720"
echo "image=$IMG"; echo "prompt=$PROMPT"
echo "########## I2V DENSE ##########"
python wan22_i2v_bench.py --mode dense --image "$IMG" --prompt "$PROMPT" --save i2v_dense.npy
echo "########## I2V FP8 SAGE ##########"
python wan22_i2v_bench.py --mode fp8   --image "$IMG" --prompt "$PROMPT" --save i2v_fp8.npy
echo "########## I2V INT8 SAGE ##########"
python wan22_i2v_bench.py --mode int8  --image "$IMG" --prompt "$PROMPT" --save i2v_int8.npy
analyze i2v

echo "########## ALL DONE ##########"
