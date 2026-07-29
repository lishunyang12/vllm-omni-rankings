"""LPIPS/PSNR between two saved frame arrays. Usage: python lpips_bench.py a.npy b.npy"""

import sys
import numpy as np
import torch
import lpips


def to_t(x):
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    x = x * 2.0 - 1.0 if x.max() <= 1.5 else x / 127.5 - 1.0
    return x.permute(0, 3, 1, 2)  # (N,3,H,W)


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    a = np.load(pa).squeeze()
    b = np.load(pb).squeeze()
    print(f"{pa} {a.shape} | {pb} {b.shape}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A, B = to_t(a), to_t(b)
    loss_fn = lpips.LPIPS(net="alex").to(dev).eval()
    lps, mses = [], []
    with torch.no_grad():
        for i in range(A.shape[0]):
            fa, fb = A[i : i + 1].to(dev), B[i : i + 1].to(dev)
            lps.append(loss_fn(fa, fb).item())
            mses.append(torch.mean((fa - fb) ** 2).item())
    lps, mses = np.array(lps), np.array(mses)
    psnr = 10 * np.log10(1.0 / np.clip(mses / 4.0, 1e-12, None))
    print(f"LPIPS(alex): mean {lps.mean():.4f}  min {lps.min():.4f}  max {lps.max():.4f}")
    print(f"PSNR       : mean {psnr.mean():.2f} dB")
    print(f"MSE[-1,1]  : mean {mses.mean():.5f}")


if __name__ == "__main__":
    main()
