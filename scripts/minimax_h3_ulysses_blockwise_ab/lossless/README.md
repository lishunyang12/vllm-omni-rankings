# H3 BF16 gate GEMM overlap

[同步视频、完整数据和 GPU 时间线](index.html)

在固定编译选核下，正常请求从 25.117667 s 降到 24.340333 s，
额外减少 3.095% / 0.777333 s。
相对前序 BF16 原调度累计减少 5.516%。
两个调度各预热一次、正式测三次，全部正式 MP4 与原 BF16 视频逐字节一致。
单个 prompt、seed 1101、362 帧、4 denoise steps；不推及任意输入或独立重新调优的编译缓存。

| 调度 | 三个正常请求 / s |
|---|---|
| v1，固定缓存 | [25.141, 25.082, 25.13] |
| v2，gate 重叠 | [24.408, 24.272, 24.341] |

小 GEMM 只计算 Gc = X Wgateᵀ，形状 M11992 N7168 K5376。
合并为 O = Oc ⊙ Gc + Of，即 Gf = 1。新的执行顺序让相同的 BF16 GEMM
在 Q/K RDMA 传输期间运行；完整 gate/O 数值、通信字节数及完成协议保持一致。

[阶段耗时](stages.csv)来自独立计时；前序 v1 阶段基线与本次 v2 阶段测量分开记录。
[GPU 时间线](gate-timelines.svg)包含 QKV/Gate GEMM、QK norm/RoPE、RDMA 窗口及原始计数器采样。
[原生 Nsight](v2-gate-warmed.nsys-rep)的耗时不用于正常请求加速结论。
这一层从 QKV GEMM 开始到最后的前向通信窗口结束，
由 29.115 ms 缩到 25.997 ms。
Gate 与通信窗口重叠 4.473 ms；窗口包含完成或调度等待，
不能将其全部视为 PCIe 持续搬运时间。

[增量补丁](gate-incremental.patch)应用在[前序候选补丁](../candidate.patch)之后。
源码和执行脚本在[原始记录](raw-data.zip)，正常请求、哈希、源码和硬件身份在[data.json](data.json)。
同一模型树下设 VLLM_OMNI_H3_LOSSLESS_GATE=0/1，配合
VLLM_OMNI_H3_EXPERIMENT_OVERLAP=1；run_model_case.sh 的 bf16-overlap 模式设置后者。
H3_LOSSLESS_COMPILE_CACHE 指向原始 Inductor/Triton 缓存的副本。
正式执行前后均检查原有的80个选核配置；新缓存控制产生的变化独立留作诊断证据。

[其他无损候选](communication-options.md)：QKV 拆分通过算子微测但未集成进此完整模型方案；
O8段未通过逐位一致性；coarse tail复用减少字节但耗时提升很小。
