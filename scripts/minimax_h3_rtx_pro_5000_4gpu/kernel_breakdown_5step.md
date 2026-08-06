# Preliminary 5-step Nsight result

This is a topology-screening profile, not the final latency result. It uses
five requested scheduler steps (four executed denoise updates), BF16 cuDNN
attention, resident weights, and physical GPU order `4,6,5,7` (Ulysses-PXB).

## T2V

GPU kernel aggregate-time share:

- NCCL AllGather: 0.93%
- NCCL SendRecv: 8.53%
- NCCL other: 14.63%
- NCCL total: 24.09%
- Dense FlashAttention/FMHA: 31.15%
- Other GEMM, norm, elementwise, and VAE: 44.76%

Load balance by aggregate kernel time:

- logical GPU 0: 27,163.56 ms (+0.56% vs mean)
- logical GPU 1: 27,145.53 ms (+0.49% vs mean)
- logical GPU 2: 26,862.97 ms (-0.55% vs mean)
- logical GPU 3: 26,876.26 ms (-0.50% vs mean)
- maximum deviation from mean: 0.56%
- max-min/mean: 1.11%

## First-frame I2V (FL2VA)

GPU kernel aggregate-time share:

- NCCL AllGather: 0.89%
- NCCL SendRecv: 7.26%
- NCCL other: 15.76%
- NCCL total: 23.91%
- Dense FlashAttention/FMHA: 32.53%
- Other GEMM, norm, elementwise, and VAE: 43.56%

Load balance by aggregate kernel time:

- logical GPU 0: 28,902.52 ms (-0.09% vs mean)
- logical GPU 1: 28,936.85 ms (+0.03% vs mean)
- logical GPU 2: 28,939.23 ms (+0.03% vs mean)
- logical GPU 3: 28,937.92 ms (+0.03% vs mean)
- maximum deviation from mean: 0.09%
- max-min/mean: 0.13%

Logical-to-physical mapping was `0->4, 1->6, 2->5, 3->7`.
