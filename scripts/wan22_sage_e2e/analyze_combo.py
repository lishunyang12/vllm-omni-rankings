"""LPIPS for the locked combo experiment over existing npy (dense/skip/int8/combo), skipping sdpa.
Run from the dir holding dense.npy/skip.npy/int8.npy/combo.npy (e.g. ~/vllm-omni)."""
import os, numpy as np, torch, lpips
def load(p):
    a=np.load(p).squeeze()
    if a.dtype!=np.uint8 and float(a.max())<=1.0+1e-3: a=a*255.0
    return a.clip(0,255).astype("uint8")
def to_t(u8): return (torch.from_numpy(np.ascontiguousarray(u8)).float()/127.5-1.0).permute(0,3,1,2)
dev="cuda" if torch.cuda.is_available() else "cpu"
loss=lpips.LPIPS(net="alex").to(dev).eval()
D=to_t(load("dense.npy")); n=D.shape[0]
def lp(B,lo,hi):
    with torch.no_grad(): return float(np.mean([loss(D[i:i+1].to(dev),B[i:i+1].to(dev)).item() for i in range(lo,hi)]))
spd={"dense":9.707,"skip":8.491,"int8":8.321,"combo":7.744}  # s/step from the run
print("\n| config | s/step | vs dense | LPIPS all | LPIPS first16 |")
print("|---|---|---|---|---|")
print(f"| dense | {spd['dense']:.3f} | — | — | — |")
for lab,name in [("skip","skip @thr0.5/u0.94"),("int8","int8 SAGE"),("combo","int8+skip @thr0.5/u0.94")]:
    p=f"{lab}.npy"
    if not os.path.exists(p): print(f"| {name} | {spd.get(lab,'?')} | — | (missing {p}) | |"); continue
    B=to_t(load(p))
    print(f"| {name} | {spd[lab]:.3f} | {spd['dense']/spd[lab]:.3f}× | {lp(B,0,n):.4f} | {lp(B,0,min(16,n)):.4f} |")
