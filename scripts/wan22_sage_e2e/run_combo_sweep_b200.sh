#!/bin/bash
# B200 sweet-spot sweep: int8 SAGE x skip-softmax over target_sparsity.
# Runs dense + int8-only + combo@{sparsities}, then LPIPS (all + first16) vs dense,
# and prints a table with an auto-flagged sweet spot (highest sparsity that stays close to int8-only).
# Curls wan22_combo.py if missing. Run from ~/vllm-omni. int8 = SM100 only.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda13}"; export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-$HOME/omni-env/bin/python}"
STEPS="${STEPS:-50}"; H="${H:-720}"; W="${W:-1280}"; FRAMES="${FRAMES:-81}"
SPARS="${SPARS:-0.3 0.5 0.7}"
OUT="${OUT:-combo_out}"; mkdir -p "$OUT"

RAW=https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/wan22_sage_e2e
[ -f wan22_combo.py ] || curl -sL "$RAW/wan22_combo.py" -o wan22_combo.py

run() {  # $1=label $2..=extra args
  local label="$1"; shift
  echo "########## $label ##########"
  $PY wan22_combo.py "$@" --steps "$STEPS" --h "$H" --w "$W" --frames "$FRAMES" \
      --save "${label}.npy" 2>&1 | tee "$OUT/gen_${label}.log"
}

run dense --mode dense
run int8  --mode int8
for s in $SPARS; do
  run "combo_s${s}" --mode combo --sparsity "$s"
done

echo "########## SWEET-SPOT ANALYSIS ##########"
OUT="$OUT" SPARS="$SPARS" STEPS="$STEPS" $PY - <<'PY'
import os, re, numpy as np, torch, lpips
out=os.environ["OUT"]; spars=os.environ["SPARS"].split(); steps=int(os.environ["STEPS"])
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
D=to_t(load("dense.npy")); n=D.shape[0]
def lp(B,lo,hi):
    with torch.no_grad(): return float(np.mean([loss(D[i:i+1].to(dev),B[i:i+1].to(dev)).item() for i in range(lo,hi)]))
labels=["int8"]+[f"combo_s{s}" for s in spars]
rows=[]
td=gt("dense")
for lab in labels:
    B=to_t(load(f"{lab}.npy")); t=gt(lab)
    rows.append({"lab":lab,"t":t,"spd":(td/t if td and t else None),
                 "all":lp(B,0,n),"f16":lp(B,0,min(16,n))})
int8_f16=rows[0]["f16"]
print("\n| mode | s/step | vs dense | LPIPS all | LPIPS first16 |")
print("|---|---|---|---|---|")
print(f"| dense | {td/steps:.3f} | — | — | — |")
for r in rows:
    print(f"| {r['lab']} | {r['t']/steps:.3f} | {r['spd']:.2f}× | {r['all']:.4f} | {r['f16']:.4f} |")
# sweet spot: highest sparsity whose first16 LPIPS <= int8-only + 0.02 (skip adds little)
combo=[r for r in rows if r["lab"].startswith("combo_s")]
ok=[r for r in combo if r["f16"]<=int8_f16+0.02]
sweet=max(ok, key=lambda r:r["spd"]) if ok else None
print()
if sweet:
    s=sweet["lab"].split("_s")[1]
    print(f"**Sweet spot: combo @ sparsity {s}** — {sweet['spd']:.2f}× vs dense, "
          f"first16 LPIPS {sweet['f16']:.4f} (int8-only {int8_f16:.4f}). "
          f"Highest sparsity that keeps quality within +0.02 of int8-only.")
else:
    print(f"No combo stayed within +0.02 of int8-only (int8 first16 {int8_f16:.4f}); "
          f"try lower sparsities than {min(spars)}.")
PY
echo "########## DONE -> $OUT/ ##########"
