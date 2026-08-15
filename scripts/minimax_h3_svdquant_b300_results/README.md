# MiniMax-H3 SVDQuant on NVIDIA B300

This directory contains the static academic results note and its machine-readable
summary for the MiniMax-H3 native fused SVDQuant experiment conducted on 15 August
2026.

- Live report: <https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_svdquant_b300_results/>
- Machine-readable data: [results.json](results.json)
- Visual quality comparison: [BF16 vs SVDQuant, five official cases](../minimax_h3_svdquant_official_comparison.html)

## Result

Under a matched T2VA protocol (1344 x 768, 5 s, 50 steps, seed 1101, one full
warmup plus three measured rounds), native fused rank-32 SVDQuant with bundled
B300 tactics reduced mean wall time from 134.599 s to 103.977 s: 1.2945x E2E.
Mean denoise time improved by 1.3129x and measured worker peak memory fell by
34.25%.

The sample count is deliberately reported: these are stable engineering
measurements, not a population-level performance claim. The result is validated
on B300 (SM103); it must not be generalized to other Blackwell SKUs without
per-device kernel validation and tactic tuning.
