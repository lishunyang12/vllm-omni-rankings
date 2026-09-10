# H3 通信重叠与 FP8 blockwise：完整四组 A/B

[打开同步视频与交互结果页](index.html)

激活 1×128、权重 128×128 的在线 FP8 E4M3，覆盖每卡 50 个 QKV、50 个 O、50 个 FC1、50 个 FC2。固定 Cake attention、BF16 通信和 gate、完整 H3 VAE；同一茶屋 prompt、seed 1101、362 帧、4 denoise steps。

| 组别 | 无 profiler E2E / s | 独立阶段计时 DiT / s | E2E 三个样本 / s |
|---|---:|---:|---|
| BF16 · 原调度 | 25.761 | 18.949 | [25.684, 25.853, 25.747] |
| BF16 · 重叠调度 | 25.135 | 18.293 | [25.17, 25.047, 25.189] |
| FP8 blockwise · 原调度 | 20.985 | 14.305 | [21.027, 20.9, 21.027] |
| FP8 blockwise · 重叠调度 | 20.586 | 13.830 | [20.502, 20.561, 20.696] |

每组预热一次、测三次；Nsight 另行采集。依次运行、默认动态频率，3 次重复不足以确认微小差异。

已检查四组在 1、4.5、9、14 秒的实际对照帧：茶屋人物、接收器近景及月球机械结构的主要场景均保留，样本帧未见明显花屏或破碎块。FP8 改变了牌匾、灯具、人物脸部与动作、接收器形态及月球结构细节，因此不能称为无损。相同精度下，原调度与重叠调度的三次正式 MP4 均逐字节一致；差异来自本次跨精度对照。像素指标不能单独代表观感优劣；本次只覆盖一个 prompt 和 seed，完整视频可在上方同步检查。

重叠方案保持原 RDMA 完成协议：Q/K 预处理覆盖 V 传输；反向 O 分四段，gate 和 O 投影覆盖后续段传输。O 的远端发送量增加 6.61%，完整模型数据包含这些开销。

TMA 描述符使用独立内存池，避免原生 cuMemcpyHtoD 初始化与前一分配者的异步写入相撞；此修复应用于全部四组。该实验按 tensor binding 保留描述符直到 worker 结束，长期服务的缓存边界尚未在本次验证。

[测量口径与复现](report-methodology.md) · [全部 A/B 数据](summary/comparisons.csv) · [阶段占比](summary/stage-shares.csv) · [视频数值差异](summary/quality-summary.csv) · [Nsight 八卡阶段变化](summary/nsys-step03-phase-changes.csv) · [异步集成检查](model-integration-check.json) · [候选补丁](candidate.patch)

- BF16 · 原调度：[视频](bf16-base.mp4) · [原生 Nsight](bf16-base-warmed.nsys-rep)
- BF16 · 重叠调度：[视频](bf16-overlap.mp4) · [原生 Nsight](bf16-overlap-warmed.nsys-rep)
- FP8 blockwise · 原调度：[视频](fp8-base.mp4) · [原生 Nsight](fp8-base-warmed.nsys-rep)
- FP8 blockwise · 重叠调度：[视频](fp8-overlap.mp4) · [原生 Nsight](fp8-overlap-warmed.nsys-rep)

[原始数据 ZIP](raw-data.zip) · [清单](manifest.json) · [SHA256](SHA256SUMS.txt)
