# MiniMax-H3：回退旧有损方案后，重新优化到 15 秒以内

本表补充原《MiniMax-H3: From 653.838 Seconds to 14.561 Seconds》报告。
叙述起点是撤回旧有损分支后，在保留 FastH3 四步、VSA 与既有无损工程优化的基础上，
重新建立的 BF16 VSA / BF16 主干线性层 / BF16 通信 / 完整 H3 VAE 对照。
这份新对照实测为 27.493667 秒，而不是历史 VSA 检查点的 43.420 秒，
也不是后期 MXFP8 DiT 基线的 17.4168 秒。

仅列最终采用的优化机制及其已验证演进；新增收益行须大于 0.100 秒。
各行是各自轮次的实测对照，保留重新测量的基线，不构造未执行的连续累计实验。
“无损”仅指相对该轮选定参考输出的字节一致性；FastH3、VSA、Sage/CAKE、MXFP8 仍包含近似计算。

| 阶段 | 采用的方案与流水变化 | 性质 | 本轮完整请求耗时 / 秒 | 本轮节省 / 秒 |
| --- | --- | --- | ---: | ---: |
| 回退后的起点 | BF16 VSA、BF16 主干与通信、完整 H3 VAE；保留既有无损工程优化 | 基线 | 27.4937 | — |
| Attention 计算 | Sage/CAKE attention，包含 ragged-tail 修正 | 有损计算 | 27.4937 → 25.7063 | 1.7873 |
| 第一轮流水 | Q/K 准备覆盖 V 通信；O 分四块回传，逐块 gate 与 O projection 覆盖后续通信 | 无损组合 | 25.7613 → 25.1353 | 0.6260 |
| Gate projection 掩盖 | VSA gate 的原 BF16 GEMM 从通信前移入 side stream，与 Q/K 通信重叠 | 无损调度 | 25.1177 → 24.3403 | 0.7773 |
| Attention 依赖重排 | 去掉 Q/K/V 布局复制；提前 Q 准备；coarse 与 fine 并行 | 无损组合 | 24.3647 → 24.0117 | 0.3530 |
| QKV 投影拆分 | 融合 QKV 改为 QK + V，让 V GEMM 覆盖 Q/K 通信；后续再调整 V-first 与 Q/K 准备顺序 | 无损调度 | 24.0268 → 23.6654（首次拆分对照） | 0.3614 |
| VAE 任务流水 | 两个时间窗口配对，均衡八卡空间 tile，并重叠结果组装 | 无损调度 | 23.6536 → 22.6670 | 0.9866 |
| VAE 合批 | 在配对基础上做精确 B2 合批，敏感 to_out 保持 B1 | 无损调度 | 22.6670 → 22.3662 | 0.3008 |
| 分配器优化 | 根据分配长尾诊断，使用 expandable_segments:False | 无损工程 | 22.5924 → 21.7712 | 0.8212 |
| 新 DiT 精度基线 | 采用 DiT MXFP8，保持 BF16 通信和完整 H3 VAE；适配量化调用边界 | 有损计算／新基线 | 17.9240 | 不做跨轮归因 |
| RDMA 生命周期 | 注册通信缓冲跨请求常驻 | 无损工程 | 17.9214 → 17.5780 | 0.3434 |
| MLP producer 融合 | SwiGLU 与 FC2 输入 MXFP8 量化融合，保留参考舍入 | 无损融合 | 17.5780 → 17.4478 | 0.1302 |
| 后期既有基线 | 冻结现有 MXFP8 DiT、完整 VAE 与工程配置后复测 | 基线 | 17.4168 | — |
| VAE 在线量化 | 完整 H3 VAE 原 FP32 权重在线 MXFP8，保持完整解码器架构 | 有损计算 | 17.4168 → 15.8322 | 1.5846 |
| CAKE 描述符 | TMA 描述符传值适配；本项不是 H3 specialization 的独立收益 | 无损工程 | 15.8322 → 15.6370 | 0.1952 |
| MXFP8 流水组合 | 重新适配 QK/V split 与 gate 调度，加 VAE B4、音视频解码并发 | 无损组合 | 15.6370 → 15.1336 | 0.5034 |
| O producer 提前提交 | 四块 O 打包提前排入独立流；逐块 ready event 接原 RDMA 与投影流水 | 无损调度 | 15.1336 → **14.9692** | **0.1644** |

## 表格解释

- 第一轮的“QK/V 重叠”是已接收 Q/K 的准备工作与 V 通信重叠；后来的“QKV 拆分”才是拆投影 GEMM。这是两项不同的流水改造。
- VSA gate projection 是产生 coarse gate 的 BF16 GEMM；O 回传后的逐元素 gate 运算以及 MLP 的 SwiGLU 是不同操作。
- QKV 拆分的 0.3614 秒来自首次完整对照。后续 V-first、gate-after-V 与 Q/K 准备调序属于该机制的继续迭代，不把各版相对融合 QKV 基线的收益重复累加。
- DiT MXFP8 的 17.9240 秒是独立精度评测点。该轮改用了量化调用边界、关闭旧 BF16 V-split，没有同配置 BF16 对照；不得把 21.7712 到 17.9240 的全部差值归给量化。随后按 MXFP8 接口重做 split，并通过完整输出检查。
- 小于等于 0.100 秒的改动不单列新增收益。17.4168 秒是包含既有 VAE gather 调度的实测基线；此处没有声称删去继承代码后重新测量。
- 表格包含顺序对照与历史检查点。早期 v1–v3 的历史 24.0117 秒在后续同功能复测中没有原样复现；每行保留真实基线以显示漂移，不把跨轮差值合成稳定加速比。
- 最终选择 model-oproducer-r2。五次正式请求为 14.973 / 14.975 / 14.968 / 14.954 / 14.976 秒，均低于 15 秒。没有启用后续 prefix 调度。
- VAE MXFP8 的 362 帧 post-codec SSIM 为 0.978920、PSNR 为 43.041443 dB，音频 PCM 一致。后续调度优化的六个完整 MP4 匹配各自参考字节。量化不因此被称为无损。

## 原始证据

以下路径相对于本机 minimax-h3-native 项目根目录；是实测证据来源，不是公开安装入口。

1. experiments/h3-cumulative-optimization-20260911/README.md：回退后 BF16 VSA、Sage 和早期无损 A/B。
2. experiments/h3-ulysses-overlap-20260910/optimization-breakdown/README.md：gate、QK/V、O 流水、复制和 fine/coarse 的 Nsight 证据。
3. experiments/h3-ulysses-overlap-20260910/lossless-v10-vsplit/r2-results.md：QKV 拆分完整模型 A/B 和真实依赖检查。
4. experiments/h3-ulysses-overlap-20260910/lossless-v12-defer-qk/WORK_STATE.md：最终继承的 V-first、gate-after-V、Q/K 准备顺序。
5. experiments/h3-ulysses-overlap-20260910/lossless-v26-postrun-audit/SELECTED_RESULTS.md：配对 VAE 与精确 B2。
6. experiments/h3-ulysses-overlap-20260910/lossless-v55-allocator/reports/RESULT_r2.md：分配器完整模型 A/B。
7. experiments/fasth3-mxfp8-vs-blockwise-20260914/reports/comparison.zh.md：DiT 量化独立检查点及归因边界。
8. experiments/fasth3-mxfp8-retention-20260914/README.zh.md：RDMA 跨请求常驻。
9. experiments/fasth3-mxfp8-fusion-20260914/README.zh.md：SwiGLU/FC2 输入量化融合。
10. experiments/fasth3-mxfp8-next-20260914/reports/lossless-inventory-20260914.zh.md：当前继承配置与完整检查。
11. experiments/fasth3-mxfp8-next-20260914/model-oproducer-r2/reports/full-qualification.json：最终正式请求及质量凭证。
12. experiments/fasth3-mxfp8-next-20260914/model-vae-mxfp8-r2/reports/comparison-qualified.json：VAE 量化完整视频对照。
