| Mode | Median diffuse | Speedup | CV | Span/median |
|---|---:|---:|---:|---:|
| TRTLLM dense | 109.347 s | 1.000× | 0.12% | 0.27% |
| FP8 SAGE | 90.014 s | 1.215× | 0.17% | 0.46% |
| Skip-Softmax 0.05/0.97 | 101.734 s | 1.075× | 0.10% | 0.25% |
| FP8 SAGE + Skip-Softmax 0.05/0.97 | 86.983 s | 1.257× | 0.13% | 0.30% |

Strict timing qualification: pass.
