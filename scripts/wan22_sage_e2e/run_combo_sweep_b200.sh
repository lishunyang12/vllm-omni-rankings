#!/bin/bash
# B200 QUICK sweet-spot sweep for int8 SAGE x skip-softmax over the two skip coefficients:
#   target_sparsity  x  disabled_until_timestep.
# Fast-by-default (STEPS=20, small grid) so you can find the knee in ~20 min, then confirm the
# winner at full STEPS=50. Runs dense + int8-only + combo@(sparsity:until) grid; LPIPS vs dense;
# auto-flags the sweet spot (max speedup within +0.02 first16-LPIPS of int8-only) and prints its coeffs.
# Curls wan22_combo.py if missing. Run from ~/vllm-omni. int8 = SM100 only.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda13}"; export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-$HOME/omni-env/bin/python}"
STEPS="${STEPS:-20}"; H="${H:-720}"; W="${W:-1280}"; FRAMES="${FRAMES:-81}"
# grid of "sparsity:until" points (both in [0,1]). until = disabled_until_timestep:
#   until=1.0 skips ALL steps (max speed, riskiest); lowering until keeps the first (1-until)
#   fraction of high-noise steps dense (safer). Speedup peaks near s=0.9 / until=1.0 (~1.18x).
# We already know the speedup landscape; this sweep adds the LPIPS (quality) axis on the fast corner.
PAIRS="${PAIRS:-0.5:0.94 0.7:1.0 0.7:0.94 0.9:1.0 0.9:0.94 0.9:0.88}"
OUT="${OUT:-combo_out}"; mkdir -p "$OUT"

RAW=https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/wan22_sage_e2e
[ -f wan22_combo.py ] || curl -sL "$RAW/wan22_combo.py" -o wan22_combo.py

run() {  # $1=label ; rest=args
  local label="$1"; shift
  echo "########## $label ##########"
  $PY wan22_combo.py "$@" --steps "$STEPS" --h "$H" --w "$W" --frames "$FRAMES" \
      --save "${label}.npy" 2>&1 | tee "$OUT/gen_${label}.log"
}

echo "QUICK sweep: STEPS=$STEPS  grid=[$PAIRS]"
run dense --mode dense
run int8  --mode int8
for p in $PAIRS; do
  s="${p%%:*}"; u="${p##*:}"
  run "combo_s${s}_u${u}" --mode combo --sparsity "$s" --until "$u"
done

echo "########## SWEET-SPOT ANALYSIS ##########"
OUT="$OUT" PAIRS="$PAIRS" STEPS="$STEPS" $PY - <<'PY'
import os, re, numpy as np, torch, lpips
out=os.environ["OUT"]; pairs=os.environ["PAIRS"].split(); steps=int(os.environ["STEPS"])
def load(p):
    a=np.load(p).squeeze()
    if a.dtype!=np.uint8 and float(a.max())<=1.0+1e-3: a=a*255.0
    return a.clip(0,255).astype("uint8")
def to_t(u8): return (torch.from_numpy(np.ascontiguousarray(u8)).float()/127.5-1.0).permute(0,3,1,2)
def gt(label):
    try: return float(re.search(r"generation\s*:\s*([\d.]+)",open(f"{out}/gen_{label}.log").read()).group(1))
    except: return None
dev="cuda" if torch.cuda.is_available() else "cpu"
loss=lpips.LPIPS(net="alex").to(dev).eval()
D=to_t(load("dense.npy")); n=D.shape[0]; td=gt("dense")
def lp(B,lo,hi):
    with torch.no_grad(): return float(np.mean([loss(D[i:i+1].to(dev),B[i:i+1].to(dev)).item() for i in range(lo,hi)]))
def row(label):
    B=to_t(load(f"{label}.npy")); t=gt(label)
    return {"lab":label,"t":t,"spd":(td/t if td and t else None),"all":lp(B,0,n),"f16":lp(B,0,min(16,n))}
int8=row("int8")
combos=[row(f"combo_s{p.split(':')[0]}_u{p.split(':')[1]}") for p in pairs]
print("\n| config | s/step | vs dense | LPIPS all | LPIPS first16 |")
print("|---|---|---|---|---|")
print(f"| dense | {td/steps:.3f} | — | — | — |")
print(f"| int8 SAGE | {int8['t']/steps:.3f} | {int8['spd']:.2f}× | {int8['all']:.4f} | {int8['f16']:.4f} |")
for r in combos:
    m=re.search(r"combo_s(.+)_u(.+)",r["lab"])
    print(f"| int8+skip s={m.group(1)} u={m.group(2)} | {r['t']/steps:.3f} | {r['spd']:.2f}× | {r['all']:.4f} | {r['f16']:.4f} |")
budget=int8["f16"]+0.02
ok=[r for r in combos if r["f16"]<=budget]
sweet=max(ok, key=lambda r:r["spd"]) if ok else None
print()
if sweet:
    m=re.search(r"combo_s(.+)_u(.+)",sweet["lab"])
    print(f"**Sweet spot: target_sparsity={m.group(1)}, disabled_until_timestep={m.group(2)}** — "
          f"{sweet['spd']:.2f}× vs dense (int8-only {int8['spd']:.2f}×), first16 LPIPS {sweet['f16']:.4f} "
          f"(budget {budget:.4f} = int8 + 0.02). Confirm at STEPS=50 before committing.")
else:
    print(f"No combo stayed within +0.02 of int8-only first16 ({int8['f16']:.4f}). "
          f"Lower target_sparsity or raise `until` (skip fewer/later steps), then re-run.")
PY
echo "########## DONE -> $OUT/  (confirm winner with STEPS=50) ##########"
