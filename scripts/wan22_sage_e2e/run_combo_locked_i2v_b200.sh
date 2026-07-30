#!/bin/bash
# B200 I2V locked combo experiment at skip sweet spot (target_sparsity=0.7, until=0.94).
# Five configs vs dense: SDPA | dense(trtllm) | skip-only | int8 SAGE-only | int8 SAGE + skip.
# Emits per-config mp4 videos + LPIPS(all/first16) vs dense. int8 = SM100 only. Run from ~/vllm-omni.
# NOTE: skip calibration in wan22_combo_i2v.py is a T2V-derived PLACEHOLDER; for a faithful I2V
# skip run, replace CALIB with the I2V checkpoint's sparse_attention_config coefficients.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda13}"; export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-$HOME/omni-env/bin/python}"
STEPS="${STEPS:-50}"; H="${H:-720}"; W="${W:-1280}"; FRAMES="${FRAMES:-81}"
# I2V has no per-expert skip calibration, so use the calibration-free threshold path.
# THR/U come from the B300 I2V threshold sweep: thr=0.5/u=0.94 is near-lossless (LPIPS first16 0.036,
# 1.09x); set THR=1.0 for a bit more speed (1.12x, LPIPS 0.070).
THR="${THR:-0.5}"; U="${U:-0.94}"
IMG="${IMG:-i2v_input.png}"
IMG_URL="${IMG_URL:-https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg}"
PROMPT="${PROMPT:-Cherry blossoms swaying gently in the breeze, petals falling, smooth motion}"
OUT="${OUT:-combo_locked_i2v_out}"; mkdir -p "$OUT"

RAW=https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/wan22_sage_e2e
[ -f wan22_combo_i2v.py ] || curl -sL "$RAW/wan22_combo_i2v.py" -o wan22_combo_i2v.py
[ -f "$IMG" ] || curl -L -o "$IMG" "$IMG_URL"
cp -f "$IMG" "$OUT/i2v_input.png"
echo "image=$IMG  prompt=$PROMPT"

run() { local label="$1"; shift; echo "########## $label ##########"
  $PY wan22_combo_i2v.py "$@" --image "$IMG" --prompt "$PROMPT" --steps "$STEPS" --h "$H" --w "$W" --frames "$FRAMES" \
      --save "${label}.npy" 2>&1 | tee "$OUT/gen_${label}.log"; }

run sdpa  --mode sdpa
run dense --mode dense
run skip  --mode skip  --threshold "$THR" --until "$U"
run int8  --mode int8
run combo --mode combo --threshold "$THR" --until "$U"

echo "########## RESULTS (vs dense) ##########"
OUT="$OUT" STEPS="$STEPS" THR="$THR" U="$U" $PY - <<'PY'
import os, re, numpy as np, torch, lpips
out=os.environ["OUT"]; steps=int(os.environ["STEPS"]); S=os.environ["THR"]; U=os.environ["U"]
def load(p):
    a=np.load(p).squeeze()
    if a.dtype!=np.uint8 and float(a.max())<=1.0+1e-3: a=a*255.0
    return a.clip(0,255).astype("uint8")
def to_t(u8): return (torch.from_numpy(np.ascontiguousarray(u8)).float()/127.5-1.0).permute(0,3,1,2)
def gt(l):
    try: return float(re.search(r"generation\s*:\s*([\d.]+)",open(f"{out}/gen_{l}.log").read()).group(1))
    except: return None
dev="cuda" if torch.cuda.is_available() else "cpu"
loss=lpips.LPIPS(net="alex").to(dev).eval()
D=to_t(load("dense.npy")); n=D.shape[0]; td=gt("dense")
def lp(B,lo,hi):
    with torch.no_grad(): return float(np.mean([loss(D[i:i+1].to(dev),B[i:i+1].to(dev)).item() for i in range(lo,hi)]))
rows=[("SDPA (baseline)","sdpa"),("dense (trtllm)","dense"),(f"skip @thr{S}/u{U}","skip"),
      ("int8 SAGE","int8"),(f"int8+skip @thr{S}/u{U}","combo")]
ts=gt("sdpa")
print(f"\n| config | s/step | vs dense | vs SDPA | LPIPS |")
print("|---|---|---|---|---|")
for name,lab in rows:
    t=gt(lab)
    vd = "—" if lab=="dense" else f"{td/t:.3f}×"
    vs = "—" if lab=="sdpa" else (f"{ts/t:.3f}×" if ts else "—")
    lp_s = "—" if lab=="dense" else f"{lp(to_t(load(f'{lab}.npy')),0,n):.4f}"
    print(f"| {name} | {t/steps:.3f} | {vd} | {vs} | {lp_s} |")
try:
    import imageio.v2 as imageio
    for _,lab in rows:
        a=load(f"{lab}.npy")
        w=imageio.get_writer(f"{out}/{lab}.mp4", fps=16, codec="libx264", quality=6,
                             macro_block_size=8, ffmpeg_params=["-crf","26","-pix_fmt","yuv420p"])
        for fr in a: w.append_data(fr)
        w.close(); print("wrote", f"{out}/{lab}.mp4")
except Exception as e:
    print("VIDEO SKIPPED (pip install imageio-ffmpeg):", e)
PY
echo "########## DONE -> $OUT/ ##########"
