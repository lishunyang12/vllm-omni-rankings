"""LPIPS/PSNR between two saved frame arrays.
Usage: python lpips_bench.py a.npy b.npy [--first N]
--first N restricts the metric to the first N frames (Ruqing's I2V protocol: the early frames
are most anchored to the condition image, so the score reflects fidelity, not content drift)."""

import argparse
import numpy as np
import torch
import lpips


def to_t(x):
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    x = x * 2.0 - 1.0 if x.max() <= 1.5 else x / 127.5 - 1.0
    return x.permute(0, 3, 1, 2)  # (N,3,H,W)


def score(A, B, loss_fn, dev, lo, hi):
    lps, mses = [], []
    with torch.no_grad():
        for i in range(lo, hi):
            fa, fb = A[i : i + 1].to(dev), B[i : i + 1].to(dev)
            lps.append(loss_fn(fa, fb).item())
            mses.append(torch.mean((fa - fb) ** 2).item())
    lps, mses = np.array(lps), np.array(mses)
    psnr = 10 * np.log10(1.0 / np.clip(mses / 4.0, 1e-12, None))
    return lps, mses, psnr


def report(tag, lps, mses, psnr):
    print(f"[{tag}] LPIPS(alex): mean {lps.mean():.4f}  min {lps.min():.4f}  max {lps.max():.4f}")
    print(f"[{tag}] PSNR       : mean {psnr.mean():.2f} dB")
    print(f"[{tag}] MSE[-1,1]  : mean {mses.mean():.5f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--first", type=int, default=None, help="restrict metric to first N frames")
    args = p.parse_args()

    a = np.load(args.a).squeeze()
    b = np.load(args.b).squeeze()
    print(f"{args.a} {a.shape} | {args.b} {b.shape}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A, B = to_t(a), to_t(b)
    n = A.shape[0]
    loss_fn = lpips.LPIPS(net="alex").to(dev).eval()

    report(f"all {n}f", *score(A, B, loss_fn, dev, 0, n))
    k = args.first if args.first is not None else 16
    if k < n:
        report(f"first {k}f", *score(A, B, loss_fn, dev, 0, k))


if __name__ == "__main__":
    main()
