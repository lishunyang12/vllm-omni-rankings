# v55 r2: allocator normal E2E result

Independent post-run audit **PASS**. Root drained the two-arm controller with exit 0 before this audit. No source, default or GPU state was changed by the audit.

| Allocator | Five formal seconds | Mean | Median | Sample SD |
|---|---|---:|---:|---:|
| expandable_segments:True | 22.463, 22.979, 22.341, 22.729, 22.450 | 22.5924 | 22.4630 | 0.2590 |
| expandable_segments:False | 21.880, 21.733, 21.724, 21.781, 21.738 | 21.7712 | 21.7380 | 0.0647 |

Fixed segments saved **0.8212 s (3.63485%)** in this sequential cohort. Warmups are excluded. All five fixed-arm samples were below all five expandable-arm samples, but the arm order was True then False; this is not a counterbalanced replication. The next isolated four-arm v58 trial will reverse the release-mode allocator order and test communicator retention separately and in combination.

The independent audit rehashed all 12 full MP4 files against the original references, checked all 96 rank/request exact-VAE records (1512 fast calls per main exact operator, 3024 residual fast calls, every fallback zero), mixed batching plans, all 16 actual worker environments, pinned model/FI/header/runtime sources and original 80 compiler choices. Both arms had only owned GPU activity and exited naturally without cleanup signals.

| Allocator | Original benchmark sampled GPU peak MiB | Five-second per-GPU sampled peaks MiB |
|---|---:|---|
| expandable_segments:True | 63467.0 | 0:63395.0, 1:63407.0, 2:63219.0, 3:63363.0, 4:63467.0, 5:63291.0, 6:63227.0, 7:63311.0 |
| expandable_segments:False | 64525.0 | 0:64433.0, 1:64465.0, 2:64277.0, 3:64421.0, 4:64525.0, 5:64349.0, 6:64285.0, 7:64429.0 |

Memory values are sampled NVML memory.used across loading and requests; they are not PyTorch allocation-counter peaks. Timing does not isolate a particular allocator API, and the prior v54 trace is not included in these normal measurements.

Full audit: `r2-audit-full-r1.json`, SHA256 `13fe0cc94b0a80c01769b15a732c263d68c383927040ce8521bed324d11a2cc9`. The failed r1 remains preserved; its rejected environment values were not recorded and its root cause remains unknown.
