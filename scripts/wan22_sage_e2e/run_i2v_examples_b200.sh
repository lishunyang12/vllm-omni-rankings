#!/bin/bash
# Three Wan2.2 I2V examples (fixed condition image + motion prompt) x {dense, fp8, int8} on B200.
# Deliberately spans motion levels (high / medium / gentle) to show fp8's quality gap growing
# with motion while int8 stays faithful. Per example: LPIPS (all + first16), compare frames, mp4s.
# Standard load 1280x720 / 81f / 50 steps / compile / CFG 4.0-4.0. int8 SAGE = SM100 only.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda13}"; export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-$HOME/omni-env/bin/python}"
H="${H:-720}"; W="${W:-1280}"; STEPS="${STEPS:-50}"; FRAMES="${FRAMES:-81}"
OUT="${OUT:-i2v_examples_out}"; mkdir -p "$OUT"

# name | picsum-id | motion prompt        (motion: high / medium / gentle)
EXAMPLES=(
  "dog|237|The dog suddenly shakes its whole body and bounds toward the camera, fur flying, fast dynamic motion."
  "street|1076|The scene comes alive: people walk and traffic flows past, lights flicker, lively handheld camera motion."
  "nature|1018|Clouds drift slowly across the sky and water ripples gently, soft breeze, slow cinematic camera push-in."
)

for entry in "${EXAMPLES[@]}"; do
  IFS='|' read -r NAME PID PROMPT <<< "$entry"
  IMG="img_${NAME}.png"
  [ -f "$IMG" ] || curl -L -o "$IMG" "https://picsum.photos/id/${PID}/${W}/${H}"
  cp -f "$IMG" "$OUT/$IMG"
  echo "############################## $NAME (picsum id$PID) ##############################"
  echo "prompt=$PROMPT"
  for MODE in dense fp8 int8; do
    echo "########## $NAME / $MODE ##########"
    $PY wan22_i2v_bench.py --mode "$MODE" --image "$IMG" --prompt "$PROMPT" \
        --h "$H" --w "$W" --steps "$STEPS" --frames "$FRAMES" \
        --save "${NAME}_${MODE}.npy" 2>&1 | tee "$OUT/gen_${NAME}_${MODE}.log"
  done
  NAME="$NAME" OUT="$OUT" PROMPT="$PROMPT" $PY - <<'PY'
import os, re, numpy as np, torch, lpips
from PIL import Image, ImageDraw
name=os.environ["NAME"]; out=os.environ["OUT"]; prompt=os.environ["PROMPT"]
def load(p):
    a=np.load(p).squeeze()
    if a.dtype!=np.uint8 and float(a.max())<=1.0+1e-3: a=a*255.0
    return a.clip(0,255).astype("uint8")
def to_t(u8): return (torch.from_numpy(np.ascontiguousarray(u8)).float()/127.5-1.0).permute(0,3,1,2)
def gt(m):
    try: return float(re.search(r"generation\s*:\s*([\d.]+)",open(f"{out}/gen_{name}_{m}.log").read()).group(1))
    except: return None
dev="cuda" if torch.cuda.is_available() else "cpu"
D,F,I=load(f"{name}_dense.npy"),load(f"{name}_fp8.npy"),load(f"{name}_int8.npy")
Dt,Ft,It=to_t(D),to_t(F),to_t(I); n=Dt.shape[0]
loss=lpips.LPIPS(net="alex").to(dev).eval()
def lp(A,B,lo,hi):
    with torch.no_grad(): return float(np.mean([loss(A[i:i+1].to(dev),B[i:i+1].to(dev)).item() for i in range(lo,hi)]))
res={nm:{"all":lp(Dt,B,0,n),"f16":lp(Dt,B,0,min(16,n))} for nm,B in[("fp8",Ft),("int8",It)]}
print(f"\n==== {name} I2V LPIPS vs dense ====")
for nm in("fp8","int8"): print(f"{nm:4s}: all {n}f = {res[nm]['all']:.4f}   first16 = {res[nm]['f16']:.4f}")
def label(im,t):
    d=ImageDraw.Draw(im); d.rectangle([0,0,8*len(t)+10,26],fill=(0,0,0)); d.text((5,6),t,fill=(255,255,255)); return im
h,w=D.shape[1:3]; sw,sh=w//2,h//2
for fi in[0,4,8,15]:
    if fi>=n: continue
    tiles=[label(Image.fromarray(a[fi]).resize((sw,sh),Image.Resampling.LANCZOS),f"{k} f{fi}") for k,a in[("ref",D),("fp8",F),("int8",I)]]
    g=Image.new("RGB",(sw*3,sh))
    for i,t in enumerate(tiles): g.paste(t,(i*sw,0))
    g.save(f"{out}/{name}_cmp_f{fi:02d}.png"); print("wrote",f"{out}/{name}_cmp_f{fi:02d}.png")
try:
    import imageio.v2 as imageio
    for nm,a in[("dense",D),("fp8",F),("int8",I)]:
        wtr=imageio.get_writer(f"{out}/{name}_{nm}.mp4",fps=16,codec="libx264",quality=6,macro_block_size=8,ffmpeg_params=["-crf","26","-pix_fmt","yuv420p"])
        for fr in a: wtr.append_data(fr)
        wtr.close(); print("wrote",f"{out}/{name}_{nm}.mp4")
except Exception as e: print("VIDEO SKIPPED:",e)
td,tf,ti=gt("dense"),gt("fp8"),gt("int8")
sec=lambda t:f"{t:.1f} s" if t else "—"; spd=lambda t:f"{td/t:.2f}×" if(td and t)else "—"
row=(f"\n### {name}\n_{prompt}_\n\n"
     f"| mode | generation | vs dense | LPIPS all ↓ | LPIPS first16 ↓ |\n|---|---|---|---|---|\n"
     f"| dense | {sec(td)} | — | — | — |\n"
     f"| fp8 | {sec(tf)} | {spd(tf)} | {res['fp8']['all']:.4f} | {res['fp8']['f16']:.4f} |\n"
     f"| int8 | {sec(ti)} | {spd(ti)} | {res['int8']['all']:.4f} | {res['int8']['f16']:.4f} |\n")
open(f"{out}/results.md","a").write(row); print(row)
PY
done
echo "########## ALL DONE -> $OUT/ ##########"; ls -lh "$OUT"
