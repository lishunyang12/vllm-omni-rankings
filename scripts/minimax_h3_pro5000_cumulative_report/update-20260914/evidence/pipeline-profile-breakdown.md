# H3 SP8 通算重叠优化 breakdown

已确认的完整模型收益分三轮：v1 的 QK/V 与 O 流水、v2 的 gate GEMM 重叠、v3 的去拷贝/early-Q/coarse 三项组合。v4 提前 softmax 的重叠真实存在，但尚未证实稳定 E2E 收益。本报告只整理已有实测，没有重新占用 GPU。

这里的通算融合主要通过数据依赖重排、跨 CUDA stream 重叠、O 分块流水以及删除冗余复制实现。

范围为同一 H3 配置：TP1×SP8、50 个主 block、4 步、top-k162、相同视频提示词/seed/权重。主干线性层和通信保持 BF16；各轮都沿用相同 Sage/Cake 的 Q/K INT8 与 P/V FP8 实现。这里没有把额外 FP8 线性层实验算入通算收益。

## 1. 完整模型：每行都是独立的同轮 A/B

| 轮次 | 本轮增加的改动 | 正常请求均值 / s | 少用时间 | 相对本轮基线 | 归因范围 |
|---|---|---:|---:|---:|---|
| v1 | QK 准备与 V 交换重叠；O 分四块流水 | 25.761 → 25.135 | 0.626 s | 2.43% | 两项组合，未做完整模型单项拆分 |
| v2 | 原 BF16 gate GEMM 移到 QKV 通信期间 | 25.118 → 24.340 | 0.777 s | 3.09% | 固定编译配置的单项 A/B |
| v3 | 去三次输入复制；提前 Q；coarse 与 fine 并行 | 24.365 → 24.012 | 0.353 s | 1.45% | 三项组合，未做完整模型单项拆分 |
| v4 | coarse softmax 提前到 V 交换期间 | 24.053 → 23.940 | 0.113 s | 0.47% | 仅观察值；未确认稳定收益，默认关闭 |

v1/v2/v3各组3次正式请求，v4各组5次；均排除预热和Nsight。跨轮重新启动、重新测基线，所以表格前后端点不完全相等，不能把这些差值相加当作一次受控实验的累计贡献。v2曾出现编译缓存漂移的试跑已排除，采用固定原编译选择且输出一致的正式对照。本次重新核验的42份正式MP4（28份normal、9份stage、5份Nsight）均与原BF16基线逐字节一致；后两类不计入正常请求E2E。

![正常A/B和独立阶段耗时](optimization-breakdown.png)

## 2. 单层：具体覆盖了什么

| 改动 | 原来暴露的顺序 | 新排布 | GPU证据与限制 |
|---|---|---|---|
| v1 QK/V | 等V交换完成后处理Q/K | K到达后，side做Q/K pack、pool、scores、top-k，comm继续交换V | step03/layer0：GPU0/7的QK kernel与V内部区间实际相交2.437/2.795ms；QK不需要V |
| v1 四段O | 整块O交换结束，再逐元素gate和O投影 | 每块2998行；第i块逐元素gate/O投影与后续块交换重叠 | step03/layer0：GPU0/7实际重叠3.227/3.546ms；GPU7 reverse路径8.626→6.754ms。最后一块投影仍暴露 |
| v2 gate GEMM | gate投影在QKV交换前串行执行 | gate只依赖原hidden input，移到side与交换并行 | GPU7 step03/layer0：gate约4.536ms，其中4.473ms与Q交换内部区间重叠；QKV GEMM开始至V closing barrier开始29.115→25.997ms，净缩短并不等于4.473ms |
| v3 去布局复制 | Q/K/V分别transpose.contiguous，再量化 | 原quantizer直接读BHSD strided view | GPU0/7每层三次复制合计1.141/1.226ms消失；减少567,017,472B本地复制载荷/卡/层，不是网络字节 |
| v3 early-Q | Q准备等K，Q量化还等V | Q到达后提前pack/pool/INT8量化，后续复用 | GPU0/7 Q-only约0.725/0.740ms；与K交换bracket重叠0.562/0.600ms，内部区间为0.492/0.477ms |
| v3 coarse/fine | fine后串行做V pool、softmax、PV、cast | V pack后记录ready event，side做原coarse，O0前join | 原fine后coarse约0.534/0.590ms；v3两卡各50层中coarse均先于fine完成。并发kernel跨度变长不等于额外算术时间 |
| v4 early-softmax | softmax位于已经重叠的coarse分支 | 同一softmax移到V交换期间 | GPU0/7约0.133/0.134ms完全被V内部区间覆盖，但coarse原本已先于fine完成，本次trace未观察到QKV→O就绪缩短 |

v1/v2例子取step03/layer0；v3/v4数字为step03、GPU0/SP0与GPU7/SP7各50层均值。输出端的逐元素gate与v2提前执行的Gc投影GEMM是不同操作：前者必须等对应O块到达，后者只依赖原hidden input。bracket包含opening开始到closing结束，内部区间为opening结束到closing开始；两者都不是独立测得的纯NIC payload时间。

## 3. v3关键路径的可核对变化

| 指标（ms/层） | GPU0：v2 → v3 | GPU7：v2 → v3 |
|---|---:|---:|
| QKV开始 → O投影输出就绪 | 54.832 → 53.065 | 54.082 → 52.445 |
| fine结束 → O0 opening开始 | 1.275 → 0.730 | 1.370 → 0.767 |
| 三次输入布局复制之和 | 1.141 → 0.000 | 1.226 → 0.000 |

GPU0/7每层自己的QKV起点到O输出就绪分别少1.767/1.637ms。这些是两份独立profile的区间证据，不能相加、跨卡求和或直接乘200替代完整请求的0.353秒实测。v3三项之间有资源竞争和依赖交互，不能按micro比例分摊其完整模型收益。

## 4. 请求阶段：收益主要落在DiT

| 独立阶段测量 | Encoder / s | DiT / s | Decode / s | Other / s | 阶段请求总计 / s |
|---|---:|---:|---:|---:|---:|
| Original | 0.122 | 18.949 | 6.588 | 0.087 | 25.745 |
| v1 | 0.123 | 18.293 | 6.662 | 0.087 | 25.165 |
| v2 | 0.122 | 17.514 | 6.604 | 0.087 | 24.327 |

初始→v1的DiT少0.656s；历史v1→后续v2的DiT少0.779s，和正常请求收益量级一致。v2没有重测固定缓存v1的stage-only组，因此后一个差值仅作跨轮阶段参照。Encoder约0.12s、Decode约6.6s没有针对性修改；Decode波动不归因于通算重叠。Video/Audio VAE都包含在Decode内，不再叠加。v3/v4未单独采集同口径三重复stage-only请求，因此此表留到v2；不会用单次Nsight阶段数据补成正常请求breakdown。

## 5. 通信量、barrier和仍暴露的尾部

- 原整块O：Q/K/V各一次交换、O一次，共8个Ulysses barrier/卡/层。四段O后共14个；v2/v3/v4均保持14个。
- QKV远端发送量保持451,282,944B/卡/层。O远端发送153,814,528→163,975,168B（+6.61%），原因是四块重复各自的270行coarse tail。接收同量，未把收发相加成单向带宽。
- 最后一块O的gate/投影/写回没有后一块传输来覆盖；v3最后O close到输出就绪约GPU0 1.350ms、GPU7 1.455ms。
- GPU7仍是v4全部50层的最后fine完成者，GPU0 O0 opening约2.258ms、GPU7约0.004ms；不能靠删除GPU0 barrier解决迟到rank。
- 同窗频率证据仍是GPU0约2264MHz、GPU7约1949MHz。频率差很可能是重要因素，但共同锁频因果对照尚未执行，管理员权限仍缺失。

## 6. Output projection 之后的后续机会

最新v4 trace中，GPU7的step03、49个层间区间平均为30.917ms；FC1/FC2占94.04%，未被六个kernel覆盖的部分仅0.017ms。O后的residual/norm/AdaLN、gate/up投影与SiLU×up均已有融合。优先候选是O分块后直接消费residual/norm并省掉组装复制，以及MLP residual与下一层norm融合；两者都应先保持原整块MLP GEMM，并保留各自不同的BF16舍入边界。这些候选尚未实现或获得新的E2E收益。详见[后续机会与优先级](post-output-opportunities.md)、[源码依赖](post-output-dependencies.md)和[尾部实际计时](post-output-timing.md)。

## 7. 查看原始证据

- [交互式逐层GPU时间线](timeline.html)：v2/v3/v4，GPU0/GPU7，step03任一层，实际kernel名称/stream/时间。
- [图SVG](optimization-breakdown.svg) · [机器可读breakdown与来源哈希](data.json)
- [正常请求与阶段计时复核](e2e-stage-evidence.md) · [v1/v2逐kernel机制与通信量](v1-v2-mechanism.md) · [v3/v4逐项归因边界](v3-v4-attribution.md)
- [原始v1四组报告数据](../summary/report-data.json) · [v2正常请求与gate时间线](../lossless-v2/report/data.json)
- [v3调度证据](../lossless-v3/analysis/candidate-schedule.md) · [v4调度证据](../lossless-v4/analysis/candidate-schedule.md)
- [v3完整模型A/B](../lossless-v3/normal-comparison.json) · [v4完整模型A/B](../lossless-v4/normal-comparison.json)
