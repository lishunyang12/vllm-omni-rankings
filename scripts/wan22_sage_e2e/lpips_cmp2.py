import sys, numpy as np, torch, lpips

SB = "/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad"
dev = "cuda"
loss_fn = lpips.LPIPS(net="alex").to(dev).eval()


def to_t(x):
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    x = x * 2.0 - 1.0 if x.max() <= 1.5 else x / 127.5 - 1.0
    return x.permute(0, 3, 1, 2)


def compare(pa, pb, label):
    a = np.load(pa).squeeze(); b = np.load(pb).squeeze()
    A, B = to_t(a), to_t(b)
    lps, mses = [], []
    with torch.no_grad():
        for i in range(A.shape[0]):
            fa, fb = A[i:i+1].to(dev), B[i:i+1].to(dev)
            lps.append(loss_fn(fa, fb).item())
            mses.append(torch.mean((fa - fb) ** 2).item())
    lps = np.array(lps); mses = np.array(mses)
    psnr = 10 * np.log10(1.0 / np.clip(mses / 4.0, 1e-12, None))
    print(f"{label:28s} LPIPS {lps.mean():.4f}  PSNR {psnr.mean():5.2f} dB  MSE {mses.mean():.5f}")
    return lps.mean(), psnr.mean()


print("=== Wan2.2-A14B 720p/81f/50step (torch.compile, B300) ===")
compare(f"{SB}/dense_frames.npy", f"{SB}/dense2_frames.npy", "dense#1 vs dense#2 (FLOOR)")
compare(f"{SB}/dense_frames.npy", f"{SB}/sage_frames.npy",   "dense#1 vs fp8-SAGE")
compare(f"{SB}/dense2_frames.npy", f"{SB}/sage_frames.npy",  "dense#2 vs fp8-SAGE")
