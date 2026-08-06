| Mode | Rank-0 projected diffuse | Attention GPU time (4-GPU total) | Attention share | Attention speedup vs dense | Primary FMHA average |
|---|---:|---:|---:|---:|---:|
| `trtllm_dense` | 75.383 s | 161.747 s | 55.0% | 1.000× | 15.879 ms |
| `sage_fp8` | 63.054 s | 121.498 s | 50.0% | 1.331× | 12.062 ms |
| `skip_softmax_05_gate099` | 65.797 s | 132.495 s | 52.2% | 1.221× | 13.117 ms |
| `sage_fp8_skip_05_gate099` | 58.072 s | 105.466 s | 47.4% | 1.534× | 10.220 ms |

Attention GPU time includes every FMHA kernel and SAGE quantization kernel in the capture. The total is summed across four GPUs; the projected diffuse time is the rank-0 wall-clock GPU projection.
