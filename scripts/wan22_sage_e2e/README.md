# Wan2.2-A14B — fp8 SAGE (trtllm-gen) end-to-end results

Real end-to-end text-to-video run of **Wan2.2-T2V-A14B** with the trtllm-gen diffusion
attention backend in **fp8 SAGE** mode, vs the BF16-dense baseline.

## Setup

| | |
|---|---|
| GPU | **NVIDIA B300 SXM6 (SM 10.3 / sm_103)**, single GPU |
| Model | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (28B MoE, dual expert) |
| Workload | **1280×720 · 81 frames · 50 steps** |
| Compile | `torch.compile` (regional, 40× `WanTransformerBlock` ×2 transformers, `dynamic=True`) |
| Prompt / seed | "A serene lakeside sunrise with mist over the water." / 0 |
| Backend | `TRTLLM_ATTN`, `quant={dtype_qk: fp8_e4m3, q_block_size: 1, k_block_size: 16}` (V per-channel fp8) |
| FlashInfer | **0.6.16rc3** (isolated install; provides `trtllm_sage_attention_quantize`, PR #3982) |
| Peak GPU | ~145 GiB (720p run) |

## 1. Latency — SAGE wins at production load

| run | generation time | per step | vs dense |
|---|---|---|---|
| dense (BF16) #1 | 340.42 s | 6.808 s | — |
| dense (BF16) #2 | 338.88 s | 6.778 s | (reproducibility check) |
| **fp8 SAGE** | **282.13 s** | **5.643 s** | **1.21× (−17%)** |

At long sequence length (~75k tokens: 21 latent frames × 3600 patch tokens) attention
dominates, so the fp8 matmul speedup outweighs the extra quantize kernel. At a tiny load
(480×832 / 17f / 6 steps) SAGE was ~8% *slower* — the quantize overhead is not amortized
there. The benefit is load-dependent.

## 2. Quality — honest read

LPIPS (AlexNet) and PSNR, per-frame mean over all 81 frames:

| pair | LPIPS ↓ | PSNR ↑ | MSE |
|---|---|---|---|
| dense#1 vs dense#2 (**nondeterminism floor**) | **0.0000** | 120 dB | 0.00000 |
| dense vs **fp8 SAGE** | **0.2839** | 20.9 dB | 0.03227 |

- The floor is **exactly 0** — dense is bit-reproducible under torch.compile, so the 0.284
  is **entirely real fp8-SAGE divergence**, not nondeterminism.
- fp8 SAGE does **not** reproduce the dense sample. Per-call attention error (~5% rel)
  compounds over 50 diffusion steps into a **different sampling trajectory**.
- **But both outputs are individually high-quality, artifact-free videos** — same scene,
  composition, and style; fp8 shows no blur/blocking/quantization artifacts (see frames).

**Bottom line:** fp8 SAGE = 1.21× faster and produces good video, but it is a *lossy*
approximation — it yields a different-but-equally-good sample, not a pixel match to dense.
If you need parity with a dense reference, fp8 SAGE at 50 steps diverges; if you want good
video faster, it delivers.

## Files

| file | what |
|---|---|
| `wan22_dense.mp4` | BF16 dense output |
| `wan22_sage.mp4` | fp8 SAGE output |
| `wan22_dense_vs_sage_sbs.mp4` | side-by-side (dense ∣ sage) |
| `f0_dense.png` / `f0_sage.png` | frame 0 stills |
| `f40_dense.png` / `f40_sage.png` | frame 40 stills |

## Reproduce

```bash
# isolated FlashInfer 0.6.16rc3 (does not touch the main venv):
#   uv pip install --target $TGT --no-deps flashinfer-python==0.6.16rc3
#   env FLASHINFER_DISABLE_VERSION_CHECK=1 FLASHINFER_CUBIN_DIR=<fresh dir> PYTHONPATH=$TGT \
#       python wan22_e2e2.py --mode {dense,sage} --save frames.npy
# attention config: {"default": {"backend": "TRTLLM_ATTN",
#   "quant": {"dtype_qk": "fp8_e4m3", "q_block_size": 1, "k_block_size": 16}}}
```
