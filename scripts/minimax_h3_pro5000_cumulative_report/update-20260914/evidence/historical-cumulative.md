# H3 cumulative 优化加成结果

2026-09-11 · 仅整理现有实测 · 未运行新基准

## 1. 近期无损主线：累计收益与最新复测

历史三轮 v1–v3 的实际端点为 25.761333 → 24.011667 秒，跨轮端点比较减少 6.7918%，加速 1.07287×。将各轮独立 A/B 比值连乘得到累计指数 1.07319×，对应约 6.8194% 的省时估算。两者都不能写成最新一轮已复现的稳定收益：最新 v7 campaign 的基线已经包含 v1–v3 功能，正常请求均值为 25.117600 秒，未复现 24.011667 秒端点。本报告整理已有证据，没有新增 GPU 测试。

| 累计配置 | 本轮基线 → 候选 / s | 本轮省时 | 累计指数：A/B 比值连乘 | 累计省时估算 |
| --- | --- | --- | --- | --- |
| 原调度 | 25.761333 | — | 1.00000× | 0% |
| v1：QK/V 重叠＋四段 O 流水 | 25.761333 → 25.135333 | 2.4300% | 1.02491× | 2.4300% |
| v2：加 gate GEMM 重叠 | 25.117667 → 24.340333 | 3.0948% | 1.05764× | 5.4496% |
| v3：加 view 去复制＋early-Q＋coarse 重叠 | 24.364667 → 24.011667 | 1.4488% | 1.07319× | 6.8194% |

![近期无损主线：累计收益与最新复测](cumulative-lossless.png)

## 2. 后续候选：没有追加到历史主线累计估算

v1–v3 每组 3 次正常请求；v4、v7 每组 5 次。均排除预热、stage profiler 与 Nsight。这些无损组的正式 MP4 与原 BF16 路径 SHA-256 aa55fa19…f596a 一致。此处无损是相对于既有四步/VSA/Cake 路径的输出合同；不等于原始 49 步 dense H3。

| 检查 | 实测 | 累计处理 |
| --- | --- | --- |
| v4：提前 coarse softmax | 24.0528 → 23.9402 s；观察值 +0.4681% | 稳定收益未确认；不计入 |
| v7：新一轮同功能基线 | 25.1176 s（5 次均值） | 已经含 v1–v3；不能误认作 v2 的 25.117667 s 基线 |
| v7：output producer＋side norm | 25.1176 → 25.2708 s；慢 0.610% | 无新增 E2E 收益 |
| v7：再加 block boundary fusion | 25.1176 → 25.2792 s；慢 0.643% | 无新增 E2E 收益；相对 output 仅慢 0.033% |
| v8：producer 变体 | 局部候选慢约 2.3–3.7% | 已筛掉；没有模型累计收益 |
| v9：V pack＋partial max | 0.522669 → 0.348384 ms；局部省 33.345% | 仅 GPU 微基准 |
| v9：early-K＋V pack 模型链 | 3850.6968 → 3839.1018 ms；中位数省 0.3011% | 单卡模拟 50-call 链；SP8/完整视频未测 |
| allocator 长尾诊断 | 旧 trace 指向 MLP 分配；新 capture 首次失败，CPU 修正完成 | 具体分配归属仍未实证；没有加速结果 |

## 3. 累计口径

单轮省时 = 1 − T候选/T基线；累计加速指数 = ∏(T本轮基线/T本轮候选)；累计省时估算 = 1 − 1/累计指数。连乘需要各轮效果能组合且环境可比，因此这里只标估算，不制造一个未测过的端到端耗时。本轮重新核验的 v1/v2/v3/v4/v7 缓存均包含同样 80 份 .best_config，内容一致；没有证据把最新 25.118 秒与历史 24.012 秒的差异直接归因于选核漂移。保留 80 份静态选择也不能单独证明运行时执行了相同旧图。v2 曾有一份另行冷调优、视频 SHA 改变的试跑，已排除。不累加不同 GPU 的同时段、不把已经被通信覆盖的 kernel 时间当净收益，也不把 33% 的微基准改善加到 E2E 百分比上。


## 4. 更早的完整 H3 历史

下表保留旧累计报告每个实测检查点。它混合工程优化与质量边界变化，也不是当前无损 campaign 的同一条受控曲线。原报告的 accepted 标记已被后续画质记录部分覆盖：Q5 的 E4M3 wire 及其 D1–D6 后继分支不再纳入当前累计结果。原报告称为 Exact 的工程项沿用其历史分类，不扩展成整条路径的逐字节证明；相邻差值不是每个 kernel 的独立因果贡献。

| 阶段 | 历史改动 | E2E / s | 相对 S0 的历史比值 | 性质/当前解释 |
| --- | --- | --- | --- | --- |
| S0 | Original H3 control | 653.838 | 1.000× | Original；历史值 |
| S1 | AdaLN TP1 / SP8 | 605.348 | 1.080× | Exact；历史值 |
| S2 | Explicit RDMA Ulysses exchange | 592.079 | 1.104× | Exact；历史值 |
| S3 | Direct Q/K layout | 590.562 | 1.107× | Exact；历史值 |
| F0 | Dense FastH3, four forwards | 66.785 | 9.790× | Lossy；历史值 |
| P0 | FastH3 + VSA + FlashInfer | 43.420 | 15.058× | Lossy；历史值 |
| P1 | GPU FP32-to-uint8 video pack | 36.568 | 17.880× | Exact；历史值 |
| P2 | Chunked pinned D2H + background mux | 31.399 | 20.824× | Exact；历史值 |
| P3 | Persistent RDMA communication | 30.534 | 21.413× | Exact；历史值 |
| P4a | Fused VSA tile packing | 30.013 | 21.785× | Exact；历史值 |
| P4c | Compact VSA untile | 29.880 | 21.882× | Exact；历史值 |
| P4d | Direct attention O path | 29.873 | 21.887× | Exact / noise-sized；历史值 |
| P5 | Exact VAE/direct Q-to-K cleanup | 28.236 | 23.156× | Exact；历史值 |
| Q1 | Online FP8 MLP | 25.172 | 25.975× | Lossy；历史值 |
| Q2 | MLP + output FP8 | 24.887 | 26.272× | Lossy；历史值 |
| Q5 | QKV E4M3 wire + O bundle | 21.649 | 不列入当前累计 | Lossy；已撤回分支 |
| D1 | TAEH3 decoder, FP32 | 16.738 | 不列入当前累计 | Lossy；已撤回分支 |
| D2 | TAEH3 decoder, FP16 | 16.289 | 不列入当前累计 | Lossy；已撤回分支 |
| D3 | All-main FP8 coverage | 15.634 | 不列入当前累计 | Lossy；已撤回分支 |
| D4 | Persistent RDMA steady state | 15.165 | 不列入当前累计 | Exact；已撤回分支 |
| D5 | OMP=28 CPU orchestration | 14.671 | 不列入当前累计 | Exact；已撤回分支 |
| D6 | Cross-rank Audio VAE | 14.561 | 不列入当前累计 | Exact；已撤回分支 |

![更早的完整 H3 历史](historical-overview.png)

## 5. Sage 与 FP8：独立比较分支

Sage 三组相对同轮 BF16 VSA：PR4691 省时约 5.79%，修复 ragged tail 的 PR4951/Cake 省时约 6.50%。三组正式 MP4 的 SHA 不同，属于数值近似比较。Sage campaign 的 Cake 视频 SHA af4edbfa…d921 与下一轮 Ulysses BF16 基线 aa55fa19…f596a 也不同，因此不把跨 campaign 的差值接成无损累计链。Ulysses 四组内，blockwise FP8 主干线性层＋重叠组合相对 BF16 原调度省时约 20.09%；QKV wire 仍是 BF16。同精度的调度变更视频逐字节相同，跨精度不相同；未测过“FP8＋v2＋v3＋v9”的完整组合，不合成其耗时。

| 同轮分组 | 配置 | 正常均值 / s | 相对本组基线 | 省时 |
| --- | --- | --- | --- | --- |
| Sage 三组 | BF16 VSA | 27.493667 | 1.00000× | 0.0000% |
| Sage 三组 | PR4691 Sage | 25.902333 | 1.06144× | 5.7880% |
| Sage 三组 | PR4951 Cake + ragged fix | 25.706333 | 1.06953× | 6.5009% |
| Ulysses 四组 | BF16 · 原调度 | 25.761333 | 1.00000× | 0.0000% |
| Ulysses 四组 | BF16 · 重叠调度 | 25.135333 | 1.02491× | 2.4300% |
| Ulysses 四组 | FP8 blockwise · 原调度 | 20.984667 | 1.22763× | 18.5420% |
| Ulysses 四组 | FP8 blockwise · 重叠调度 | 20.586333 | 1.25138× | 20.0882% |

## 6. 已撤回终点与尚缺的测量

历史 14.561 秒来自 [14.576, 14.555, 14.551] 三次 steady-state 请求，但它继承了后来被完整视频人工评审拒绝的 E4M3 QKV 传输，不能再称当前可接受终点，也不能据此延伸当前实时结论。后来 BF16-wire 的 FP8/TAEH3 四格质量对照属于顺序单次、无预热、共享缓存请求；39.366/28.055/28.983/23.494 秒只作对应视频的时间记录，不能相除或相减宣称 FP8 或解码器的单因素速度收益。当前 v9 的八卡副本仅完成 CPU 审阅准备，明确 PREPARED_NOT_RUN。下一次若需要一条严格的累计曲线，应在同一轮按固定配置逐级启用并保留每级重复测量；本次没有重跑 GPU。


## 7. 证据与复算

normal-samples.csv 保留 58 次唯一正常请求，每次都有 measurement_id；v1 与 Ulysses 四组报告引用同一批 6 次 BF16 请求，已去重并保留 analysis_aliases。cumulative-results.csv 保留各轮比值计算；data.json 保留原始路径、SHA-256 与解释边界。三个 *-history-audit.json 分别记录早期历史、近期无损配置和 Sage/FP8 分支的只读核验。源报告中的过时状态以完成的原始 JSON/TSV 与后续质量记录为准。



原始核心证据：

- [v1–v4 breakdown](/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-ulysses-overlap-20260910/optimization-breakdown/README.md)
- [v7 完整验证](/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-ulysses-overlap-20260910/bubble-review-20260911/continuation-results.md)
- [v9 验证](/lustre/raplab/client/sylarl/minimax-h3-native/experiments/h3-ulysses-overlap-20260910/lossless-v9-preprocess/verified-results-stage1.json)
- [旧累计报告数据](/lustre/raplab/client/sylarl/minimax-h3-native/vllm-omni-rankings/scripts/minimax_h3_pro5000_cumulative_report/report_data.json)
- [后续质量排除规则](/lustre/raplab/client/sylarl/minimax-h3-native/vllm-omni-rankings/scripts/minimax_h3_pro5000_cumulative_report/quality/moon-teahouse-seed1101/README.md)
